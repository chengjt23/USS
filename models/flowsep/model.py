import os
import sys
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np
import pytorch_lightning as pl
from torch.optim.lr_scheduler import LambdaLR
from einops import rearrange, repeat
from contextlib import contextmanager
from functools import partial
from pytorch_lightning.utilities.rank_zero import rank_zero_only
import soundfile as sf

from utils.tools import (
    exists,
    default,
    count_params,
    rank_zero_print,
    instantiate_from_config,
)
from utils.diffusion import (
    extract_into_tensor,
    betas_for_alpha_bar,
    noise_like,
)
from models.flowsep.ema import LitEma
from models.flowsep.distributions import DiagonalGaussianDistribution
from models.flowsep.text_encoder import FlanT5HiddenState

__conditioning_keys__ = {"concat": "c_concat", "crossattn": "c_crossattn", "adm": "y"}


class PriorNet(nn.Module):
    def __init__(self, in_channels=8, base_channels=96):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        self.conv_mid1 = nn.Conv2d(base_channels, base_channels, 3, padding=1)
        self.conv_mid2 = nn.Conv2d(base_channels, base_channels, 3, padding=1)
        self.conv_mid3 = nn.Conv2d(base_channels, base_channels, 3, padding=1)
        self.conv_out = nn.Conv2d(base_channels, in_channels, 3, padding=1)

    def forward(self, x, text_emb=None):
        h = F.silu(self.conv_in(x))
        h = h + F.silu(self.conv_mid1(h))
        h = h + F.silu(self.conv_mid2(h))
        h = h + F.silu(self.conv_mid3(h))
        return self.conv_out(h)


class CondPriorNet(nn.Module):
    def __init__(self, in_channels=8, base_channels=96, text_dim=1024):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        self.conv_mid1 = nn.Conv2d(base_channels, base_channels, 3, padding=1)
        self.conv_mid2 = nn.Conv2d(base_channels, base_channels, 3, padding=1)
        self.conv_mid3 = nn.Conv2d(base_channels, base_channels, 3, padding=1)
        self.conv_out = nn.Conv2d(base_channels, in_channels, 3, padding=1)
        self.film1 = nn.Linear(text_dim, base_channels * 2)
        self.film2 = nn.Linear(text_dim, base_channels * 2)
        self.film3 = nn.Linear(text_dim, base_channels * 2)

    def _film(self, h, film_layer, text_vec):
        gb = film_layer(text_vec).unsqueeze(-1).unsqueeze(-1)
        gamma, beta = gb.chunk(2, dim=1)
        return (1 + gamma) * h + beta

    def forward(self, x, text_emb):
        text_vec = text_emb.mean(dim=1)
        h = F.silu(self.conv_in(x))
        h = h + self._film(F.silu(self.conv_mid1(h)), self.film1, text_vec)
        h = h + self._film(F.silu(self.conv_mid2(h)), self.film2, text_vec)
        h = h + self._film(F.silu(self.conv_mid3(h)), self.film3, text_vec)
        return self.conv_out(h)


class CrossAttnPriorNet(nn.Module):
    def __init__(self, in_channels=8, base_channels=96, text_dim=1024):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        self.conv_mid1 = nn.Conv2d(base_channels, base_channels, 3, padding=1)
        self.conv_mid2 = nn.Conv2d(base_channels, base_channels, 3, padding=1)
        self.norm = nn.LayerNorm(base_channels)
        self.to_q = nn.Linear(base_channels, base_channels)
        self.to_k = nn.Linear(text_dim, base_channels)
        self.to_v = nn.Linear(text_dim, base_channels)
        self.to_out = nn.Linear(base_channels, base_channels)
        self.scale = base_channels ** -0.5
        self.conv_mid3 = nn.Conv2d(base_channels, base_channels, 3, padding=1)
        self.conv_out = nn.Conv2d(base_channels, in_channels, 3, padding=1)

    def forward(self, x, text_emb):
        B, _, H, W = x.shape
        h = F.silu(self.conv_in(x))
        h = h + F.silu(self.conv_mid1(h))
        h = h + F.silu(self.conv_mid2(h))
        h_flat = h.permute(0, 2, 3, 1).reshape(B, H * W, -1)
        q = self.to_q(self.norm(h_flat))
        k = self.to_k(text_emb)
        v = self.to_v(text_emb)
        attn = F.softmax(torch.bmm(q, k.transpose(1, 2)) * self.scale, dim=-1)
        h = h + torch.bmm(attn, v).reshape(B, H, W, -1).permute(0, 3, 1, 2)
        h = h + F.silu(self.conv_mid3(h))
        return self.conv_out(h)


class ConcatFiLMPriorNet(nn.Module):
    def __init__(self, in_channels=8, base_channels=96, text_dim=1024, text_proj_channels=8):
        super().__init__()
        self.text_proj = nn.Linear(text_dim, text_proj_channels)
        self.conv_in = nn.Conv2d(in_channels + text_proj_channels, base_channels, 3, padding=1)
        self.conv_mid1 = nn.Conv2d(base_channels, base_channels, 3, padding=1)
        self.conv_mid2 = nn.Conv2d(base_channels, base_channels, 3, padding=1)
        self.conv_mid3 = nn.Conv2d(base_channels, base_channels, 3, padding=1)
        self.conv_out = nn.Conv2d(base_channels, in_channels, 3, padding=1)
        self.film1 = nn.Linear(text_dim, base_channels * 2)
        self.film2 = nn.Linear(text_dim, base_channels * 2)
        self.film3 = nn.Linear(text_dim, base_channels * 2)

    def _film(self, h, film_layer, text_vec):
        gb = film_layer(text_vec).unsqueeze(-1).unsqueeze(-1)
        gamma, beta = gb.chunk(2, dim=1)
        return (1 + gamma) * h + beta

    def forward(self, x, text_emb):
        text_vec = text_emb.mean(dim=1)
        t_sp = self.text_proj(text_vec).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, x.shape[2], x.shape[3])
        h = F.silu(self.conv_in(torch.cat([x, t_sp], dim=1)))
        h = h + self._film(F.silu(self.conv_mid1(h)), self.film1, text_vec)
        h = h + self._film(F.silu(self.conv_mid2(h)), self.film2, text_vec)
        h = h + self._film(F.silu(self.conv_mid3(h)), self.film3, text_vec)
        return self.conv_out(h)


class PriorUNetWrapper(nn.Module):
    def __init__(self, unet_config):
        super().__init__()
        self.unet = instantiate_from_config(unet_config)

    def forward(self, x, text_emb):
        B = x.shape[0]
        t = torch.zeros(B, device=x.device)
        context, mask = text_emb
        return self.unet(x, t, context_list=[context], context_attn_mask_list=[mask])


def disabled_train(self, mode=True):
    return self


class DDPM(pl.LightningModule):
    def __init__(
        self,
        unet_config,
        sampling_rate=None,
        timesteps=1000,
        beta_schedule="linear",
        loss_type="l2",
        ckpt_path=None,
        ignore_keys=[],
        load_only_unet=False,
        monitor="val/loss",
        use_ema=True,
        first_stage_key="image",
        latent_t_size=256,
        latent_f_size=16,
        channels=3,
        extra_channels=False,
        extra_channel_key="mixed_mel",
        log_every_t=100,
        clip_denoised=True,
        linear_start=1e-4,
        linear_end=2e-2,
        cosine_s=8e-3,
        given_betas=None,
        original_elbo_weight=0.0,
        v_posterior=0.0,
        l_simple_weight=1.0,
        conditioning_key=None,
        parameterization="eps",
        scheduler_config=None,
        use_positional_encodings=False,
        learn_logvar=False,
        logvar_init=0.0,
        evaluator=None,
    ):
        super().__init__()
        assert parameterization in ["eps", "x0", "v"], 'currently only supporting "eps" and "x0" and "v"'
        self.parameterization = parameterization
        self.state = None
        rank_zero_print(
            f"{self.__class__.__name__}: Running in {self.parameterization}-prediction mode"
        )
        assert sampling_rate is not None
        
        self.clip_denoised = clip_denoised
        self.log_every_t = log_every_t
        self.first_stage_key = first_stage_key
        self.extra_channel_key = extra_channel_key
        self.sampling_rate = sampling_rate

        if self.global_rank == 0:
            self.evaluator = evaluator

        self.latent_t_size = latent_t_size
        self.latent_f_size = latent_f_size

        self.channels = channels
        self.extra_channels = extra_channels
        if self.extra_channels:
            assert self.extra_channel_key is not None 
        self.use_positional_encodings = use_positional_encodings
        self.model = DiffusionWrapper(unet_config, conditioning_key)
        count_params(self.model, verbose=True)
        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self.model)

        self.use_scheduler = scheduler_config is not None
        if self.use_scheduler:
            self.scheduler_config = scheduler_config

        self.v_posterior = v_posterior
        self.original_elbo_weight = original_elbo_weight
        self.l_simple_weight = l_simple_weight

        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(
                ckpt_path, ignore_keys=ignore_keys, only_model=load_only_unet
            )

        self.register_schedule(
            given_betas=given_betas,
            beta_schedule=beta_schedule,
            timesteps=timesteps,
            linear_start=linear_start,
            linear_end=linear_end,
            cosine_s=cosine_s,
        )

        self.loss_type = loss_type

        self.learn_logvar = learn_logvar
        self.logvar = torch.full(fill_value=logvar_init, size=(self.num_timesteps,))

        if self.learn_logvar:
            self.logvar = nn.Parameter(self.logvar, requires_grad=True)
        else:
            self.logvar = nn.Parameter(self.logvar, requires_grad=False)

        self.logger_save_dir = None
        self.logger_exp_name = None
        self.logger_exp_group_name = None
        self.logger_version = None

        self.label_indices_total = None
        self.metrics_buffer = {}
        self.initial_learning_rate = None
        self.test_data_subset_path = None
    
    def get_log_dir(self):
        return os.path.join(self.logger_save_dir, self.logger_exp_group_name, self.logger_exp_name)

    def set_log_dir(self, save_dir, exp_group_name, exp_name):
        self.logger_save_dir = save_dir
        self.logger_exp_group_name = exp_group_name
        self.logger_exp_name = exp_name

    def register_schedule(
        self,
        given_betas=None,
        beta_schedule="linear",
        timesteps=1000,
        linear_start=1e-4,
        linear_end=2e-2,
        cosine_s=8e-3,
    ):
        if exists(given_betas):
            betas = given_betas
        else:
            betas = betas_for_alpha_bar(timesteps, alpha_transform_type="cosine")
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

        (timesteps,) = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert (
            alphas_cumprod.shape[0] == self.num_timesteps
        ), "alphas have to be defined for each timestep"
    
        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer("betas", to_torch(betas))
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))
        self.register_buffer("alphas_cumprod_prev", to_torch(alphas_cumprod_prev))

        self.register_buffer("sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", to_torch(np.sqrt(1.0 - alphas_cumprod))
        )
        self.register_buffer(
            "log_one_minus_alphas_cumprod", to_torch(np.log(1.0 - alphas_cumprod))
        )
        self.register_buffer(
            "sqrt_recip_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod))
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod - 1))
        )

        posterior_variance = (1 - self.v_posterior) * betas * (
            1.0 - alphas_cumprod_prev
        ) / (1.0 - alphas_cumprod) + self.v_posterior * betas
        self.register_buffer("posterior_variance", to_torch(posterior_variance))
        self.register_buffer(
            "posterior_log_variance_clipped",
            to_torch(np.log(np.maximum(posterior_variance, 1e-20))),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            to_torch(
                (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod)
            ),
        )

        if self.parameterization == "eps":
            lvlb_weights = self.betas**2 / (
                2
                * self.posterior_variance
                * to_torch(alphas)
                * (1 - self.alphas_cumprod)
            )
        elif self.parameterization == "x0":
            lvlb_weights = (
                0.5
                * np.sqrt(torch.Tensor(alphas_cumprod))
                / (2.0 * 1 - torch.Tensor(alphas_cumprod))
            )
        elif self.parameterization == "v":
            lvlb_weights = torch.ones_like(self.betas ** 2 / (
                    2 * self.posterior_variance * to_torch(alphas) * (1 - self.alphas_cumprod)))
        else:
            raise NotImplementedError("mu not supported")
        lvlb_weights[0] = lvlb_weights[1]
        self.register_buffer("lvlb_weights", lvlb_weights, persistent=False)
        assert not torch.isnan(self.lvlb_weights).all()

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.model.parameters())
            self.model_ema.copy_to(self.model)
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.model.parameters())

    def init_from_ckpt(self, path, ignore_keys=list(), only_model=False):
        sd = torch.load(path, map_location="cpu")
        if "state_dict" in list(sd.keys()):
            sd = sd["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    rank_zero_print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = (
            self.load_state_dict(sd, strict=False)
            if not only_model
            else self.model.load_state_dict(sd, strict=False)
        )
        rank_zero_print(
            f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys"
        )
        if len(missing) > 0:
            rank_zero_print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            rank_zero_print(f"Unexpected Keys: {unexpected}")

    def q_mean_variance(self, x_start, t):
        mean = extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract_into_tensor(
            self.log_one_minus_alphas_cumprod, t, x_start.shape
        )
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
            * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool):
        model_out = self.model(x, t)
        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out
        if clip_denoised:
            x_recon.clamp_(-1.0, 1.0)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, repeat_noise=False):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(
            x=x, t=t, clip_denoised=clip_denoised
        )
        noise = noise_like(x.shape, device, repeat_noise)
        nonzero_mask = (
            (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1))).contiguous()
        )
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    def get_loss(self, pred, target, mean=True):
        if self.loss_type == "l1":
            loss = (target - pred).abs()
            if mean:
                loss = loss.mean()
        elif self.loss_type == "l2":
            if mean:
                loss = torch.nn.functional.mse_loss(target, pred)
            else:
                loss = torch.nn.functional.mse_loss(target, pred, reduction="none")
        else:
            raise NotImplementedError("unknown loss type '{loss_type}'")

        return loss

    def get_input(self, batch, k):
        fname = batch["fname"]
        text = batch["text"]
        label_indices = batch["label_vector"]
        waveform = batch["waveform"]
        stft = batch.get("stft")
        fbank = batch.get("log_mel_spec")
        ret = {}

        if fbank is not None:
            ret["fbank"] = (
                fbank.unsqueeze(1).to(memory_format=torch.contiguous_format).float()
            )
        if stft is not None:
            ret["stft"] = stft.to(memory_format=torch.contiguous_format).float()
        ret["waveform"] = waveform.to(memory_format=torch.contiguous_format).float()
        ret["text"] = list(text)
        ret["fname"] = fname

        for key in batch.keys():
            if key not in ret.keys():
                ret[key] = batch[key]

        return ret[k]

    def warmup_step(self):
        if self.initial_learning_rate is None:
            self.initial_learning_rate = self.learning_rate

        step = self.global_step
        warmup_steps = self.lr_warmup_steps
        max_lr = self.lr_max
        min_lr = self.lr_min
        decay_until = self.lr_decay_until_step

        if decay_until is not None:
            import math
            if step < warmup_steps:
                lr = max_lr * step / warmup_steps
            elif step > decay_until:
                lr = min_lr
            else:
                ratio = (step - warmup_steps) / (decay_until - warmup_steps)
                lr = min_lr + 0.5 * (1.0 + math.cos(math.pi * ratio)) * (max_lr - min_lr)
        else:
            if step <= warmup_steps:
                lr = (step / warmup_steps) * self.initial_learning_rate
            else:
                lr = self.initial_learning_rate

        self.trainer.optimizers[0].param_groups[0]["lr"] = lr

    def on_validation_epoch_start(self) -> None:
        for key in self.cond_stage_model_metadata.keys():
            metadata = self.cond_stage_model_metadata[key]
            model_idx, cond_stage_key, conditioning_key = metadata["model_idx"], metadata["cond_stage_key"], metadata["conditioning_key"]
        return super().on_validation_epoch_start()
    
    def on_train_epoch_start(self, *args, **kwargs):
        pass

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self.model)

    @staticmethod
    def _si_sdr_batch(ref, est):
        ref = ref - ref.mean(dim=-1, keepdim=True)
        est = est - est.mean(dim=-1, keepdim=True)
        dot = (ref * est).sum(dim=-1, keepdim=True)
        s_target = dot * ref / (ref.pow(2).sum(dim=-1, keepdim=True) + 1e-8)
        e_noise = est - s_target
        return 10 * torch.log10(s_target.pow(2).sum(dim=-1) / (e_noise.pow(2).sum(dim=-1) + 1e-8) + 1e-8)

    def _load_panns_model(self):
        if self._panns_model is not None:
            return self._panns_model
        import importlib.util
        fad_dir = os.path.normpath(os.path.join(
            os.path.dirname(os.path.abspath(__file__)), '..', '..', 'metrics', 'fad'))
        pu_spec = importlib.util.spec_from_file_location(
            "_panns_pytorch_utils", os.path.join(fad_dir, "pytorch_utils.py"))
        pu_mod = importlib.util.module_from_spec(pu_spec)
        sys.modules["pytorch_utils"] = pu_mod
        pu_spec.loader.exec_module(pu_mod)
        m_spec = importlib.util.spec_from_file_location(
            "_panns_models", os.path.join(fad_dir, "models.py"))
        m_mod = importlib.util.module_from_spec(m_spec)
        m_spec.loader.exec_module(m_mod)
        model = m_mod.Cnn14_16k(
            sample_rate=16000, window_size=512, hop_size=160,
            mel_bins=64, fmin=50, fmax=8000, classes_num=527,
        )
        ckpt = torch.load(self.panns_ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        model.to(self.device)
        model.eval()
        self._panns_model = model
        return model

    def _panns_embedding(self, wav, sr=16000):
        model = self._load_panns_model()
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
        with torch.no_grad():
            out = model(wav.float().to(self.device), None)
        return out["embedding"].float().cpu()

    def _load_clap_model(self):
        if self._clap_model is not None:
            return self._clap_model
        clap_dir = os.path.normpath(os.path.join(
            os.path.dirname(os.path.abspath(__file__)), '..', '..', 'metrics', 'clapscore'))
        if clap_dir not in sys.path:
            sys.path.insert(0, clap_dir)
        from models.clap_encoder import CLAP_Encoder
        model = CLAP_Encoder(
            pretrained_path=self.clap_ckpt_path,
            sampling_rate=32000,
            device=self.device,
        ).eval()
        self._clap_model = model
        return model

    def _clap_audio_embedding(self, wav, sr=16000):
        model = self._load_clap_model()
        if sr != 32000:
            wav = torchaudio.functional.resample(wav, sr, 32000)
        with torch.no_grad():
            embed = model.get_query_embed(modality='audio', audio=wav.float().to(self.device), device=self.device)
        return embed.float().cpu()

    def _clap_text_embedding(self, text_list):
        model = self._load_clap_model()
        with torch.no_grad():
            embed = model.get_query_embed(modality='text', text=text_list, device=self.device)
        return embed.cpu()

    @staticmethod
    def _frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
        from scipy import linalg
        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)
        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)
        diff = mu1 - mu2
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        tr_covmean = np.trace(covmean)
        return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)

    def _gather_validation_tensor_list(self, tensor_list):
        local_tensors = [tensor.detach().cpu() for tensor in tensor_list]
        if dist.is_available() and dist.is_initialized():
            gathered_tensors = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(gathered_tensors, local_tensors)
            local_tensors = []
            for tensors in gathered_tensors:
                if tensors:
                    local_tensors.extend(tensors)
        if len(local_tensors) == 0:
            return None
        return torch.cat(local_tensors, dim=0)

    def on_validation_epoch_start(self):
        self._fad_ref_embs = []
        self._fad_pred_embs = []
        self._clap_text_embs = []
        self._clap_pred_audio_embs = []
        self._clap_ref_audio_embs = []

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        name = self.get_validation_folder_name()
        waveform = self.generate_sample(
            [batch],
            name=name,
            unconditional_guidance_scale=self.evaluation_params[
                "unconditional_guidance_scale"
            ],
            ddim_steps=self.evaluation_params["ddim_sampling_steps"],
            n_gen=self.evaluation_params["n_candidates_per_samples"],
            save=False,
            save_mixed=False,
        )
        loss_dict = {}

        if waveform is not None:
            try:
                pred = torch.from_numpy(waveform).float()
                if pred.ndim == 3:
                    pred = pred.squeeze(1)
                ref = batch["waveform"].cpu().float()
                if ref.ndim == 3:
                    ref = ref.squeeze(1)
                min_len = min(pred.shape[-1], ref.shape[-1])
                ref_t, pred_t = ref[..., :min_len], pred[..., :min_len]
                si_sdr_val = self._si_sdr_batch(ref_t, pred_t).mean()
                loss_dict["val/si_sdr"] = si_sdr_val.item()
                try:
                    from metrics.dnsmos.dnsmos_debug import compute_dnsmos, PRIMARY_MODEL, P808_MODEL
                    if not hasattr(self, '_dnsmos_primary_sess'):
                        import onnxruntime as ort
                        self._dnsmos_primary_sess = ort.InferenceSession(PRIMARY_MODEL, providers=["CPUExecutionProvider"])
                        self._dnsmos_p808_sess = ort.InferenceSession(P808_MODEL, providers=["CPUExecutionProvider"])
                    dnsmos_vals = {"sig": [], "bak": [], "ovr": [], "p808_mos": []}
                    for i in range(pred_t.shape[0]):
                        e = pred_t[i].numpy().astype(np.float32)
                        if self.sampling_rate != 16000:
                            e = torchaudio.functional.resample(torch.from_numpy(e), self.sampling_rate, 16000).numpy().astype(np.float32)
                        scores = compute_dnsmos(e, self._dnsmos_primary_sess, self._dnsmos_p808_sess)
                        for k in dnsmos_vals:
                            dnsmos_vals[k].append(scores[k.upper()])
                    for k, v in dnsmos_vals.items():
                        loss_dict[f"val/dnsmos_{k}"] = float(np.mean(v))
                except Exception as e:
                    raise RuntimeError(f"Failed to compute DNSMOS scores: {e}")
                if hasattr(self, "_fad_ref_embs") and self.panns_ckpt_path:
                    self._fad_ref_embs.append(self._panns_embedding(ref_t, sr=self.sampling_rate))
                    self._fad_pred_embs.append(self._panns_embedding(pred_t, sr=self.sampling_rate))
                if hasattr(self, "_clap_text_embs") and self.clap_ckpt_path:
                    self._clap_pred_audio_embs.append(self._clap_audio_embedding(pred_t, sr=self.sampling_rate))
                    self._clap_ref_audio_embs.append(self._clap_audio_embedding(ref_t, sr=self.sampling_rate))
                    captions = batch.get("caption", batch.get("text", []))
                    if captions and any(c != "" for c in captions):
                        self._clap_text_embs.append(self._clap_text_embedding(captions))
            except Exception as e:
                raise RuntimeError(f"Failed in validation metric computation: {e}")

        batch_size = batch["waveform"].shape[0] if "waveform" in batch else None
        self.log_dict(
        {k: float(v) for k, v in loss_dict.items()},
        prog_bar=False,
        logger=True,
        on_step=False,
        on_epoch=True,
        sync_dist=False,
        batch_size=batch_size,
        )

    def on_validation_epoch_end(self):
        ref_all_tensor = self._gather_validation_tensor_list(self._fad_ref_embs)
        pred_all_tensor = self._gather_validation_tensor_list(self._fad_pred_embs)
        if ref_all_tensor is not None and pred_all_tensor is not None:
            try:
                ref_all = ref_all_tensor.float().numpy()
                pred_all = pred_all_tensor.float().numpy()
                mu_r = np.mean(ref_all, axis=0)
                mu_p = np.mean(pred_all, axis=0)
                sigma_r = np.cov(ref_all, rowvar=False)
                sigma_p = np.cov(pred_all, rowvar=False)
                fad = self._frechet_distance(mu_r, sigma_r, mu_p, sigma_p)
                self.log("val/fad", fad, prog_bar=True, logger=True)
            except Exception as e:
                raise RuntimeError(f"Failed to compute FAD: {e}")
        self._fad_ref_embs = []
        self._fad_pred_embs = []

        pred_audio_all = self._gather_validation_tensor_list(self._clap_pred_audio_embs)
        ref_audio_all = self._gather_validation_tensor_list(self._clap_ref_audio_embs)
        text_all = self._gather_validation_tensor_list(self._clap_text_embs)
        if pred_audio_all is not None and ref_audio_all is not None:
            try:
                import torch.nn.functional as F
                clapscore_a = F.cosine_similarity(pred_audio_all, ref_audio_all, dim=1).mean().item()
                self.log("val/clapscore_a", clapscore_a, prog_bar=True, logger=True)
                if text_all is not None:
                    min_n = min(text_all.shape[0], pred_audio_all.shape[0])
                    clapscore = (text_all[:min_n] * pred_audio_all[:min_n]).sum(-1).mean().item()
                    self.log("val/clapscore", clapscore, prog_bar=True, logger=True)
            except Exception as e:
                raise RuntimeError(f"Failed to compute CLAPScore: {e}")
        self._clap_text_embs = []
        self._clap_pred_audio_embs = []
        self._clap_ref_audio_embs = []

    def get_validation_folder_name(self):
        return "val_%s_cfg_scale_%s_ddim_%s_n_cand_%s" % (self.global_step, self.evaluation_params["unconditional_guidance_scale"], self.evaluation_params["ddim_sampling_steps"], self.evaluation_params["n_candidates_per_samples"])


class LatentDiffusion(DDPM):
    def __init__(
        self,
        first_stage_config,
        cond_stage_config=None,
        num_timesteps_cond=None,
        cond_stage_key="image",
        unconditional_prob_cfg=0.1,
        cond_stage_trainable=False,
        concat_mode=True,
        cond_stage_forward=None,
        conditioning_key=None,
        scale_factor=1.0,
        batchsize=None,
        evaluation_params={},
        scale_by_std=False,
        base_learning_rate=None,
        clap_trainable=False,
        retrival_num=0,
        only_head=False,
        use_retrival=False,
        fbank_shift=None,
        data_mean=None,
        data_std=None,
        use_clap=False,
        sigma_min=1e-4,
        euler=False,
        bridge_mode=False,
        sb_schedule=False,
        sb_rho=1.0,
        chen_bridge=False,
        fg_schedule="vp_linear",
        fg_beta0=0.01,
        fg_beta1=20.0,
        chen_sampler_type="sde",
        chen_sampling_eps=1e-3,
        bridge_input_start_noise=False,
        bridge_input_start_noise_std=0.0,
        data_prediction=False,
        use_ei_solver=False,
        logit_normal_t=0.0,
        asym_noise=False,
        loss_t_weight=0.0,
        reparam_bridge=False,
        learnable_prior=False,
        prior_lambda=0.5,
        prior_use_text=False,
        prior_type=None,
        prior_unet_config=None,
        prior_mse_weight=0.0,
        panns_ckpt_path=None,
        clap_ckpt_path=None,
        lr_warmup_steps=1000,
        lr_max=None,
        lr_min=None,
        lr_decay_until_step=None,
        *args,
        **kwargs,
    ):
        self.use_retrival = use_retrival
        self.only_head = only_head
        self.clap_trainable = clap_trainable
        self.retrival_num = retrival_num
        self.learning_rate = base_learning_rate
        self.lr_warmup_steps = lr_warmup_steps
        self.lr_max = lr_max if lr_max is not None else base_learning_rate
        self.lr_min = lr_min if lr_min is not None else base_learning_rate
        self.lr_decay_until_step = lr_decay_until_step
        self.num_timesteps_cond = default(num_timesteps_cond, 1)
        self.scale_by_std = scale_by_std
        self.evaluation_params = evaluation_params
        self.sigma_min = sigma_min
        self.euler = euler
        self.bridge_mode = bridge_mode
        self.sb_schedule = sb_schedule
        self.sb_rho = sb_rho
        self.chen_bridge = chen_bridge
        self.fg_schedule = fg_schedule
        self.fg_beta0 = fg_beta0
        self.fg_beta1 = fg_beta1
        self.chen_sampler_type = chen_sampler_type
        self.chen_sampling_eps = chen_sampling_eps
        self.bridge_input_start_noise = bridge_input_start_noise
        self.bridge_input_start_noise_std = bridge_input_start_noise_std
        self.data_prediction = data_prediction
        self.use_ei_solver = use_ei_solver
        self.logit_normal_t = logit_normal_t
        self.asym_noise = asym_noise
        self.loss_t_weight = loss_t_weight
        self.reparam_bridge = reparam_bridge
        self.learnable_prior = learnable_prior
        self.prior_lambda = prior_lambda
        self.prior_use_text = prior_use_text
        self.prior_type = prior_type
        self.prior_unet_config = prior_unet_config
        self.prior_mse_weight = prior_mse_weight
        self.prior_needs_text = prior_type in ("crossattn", "concat_film", "unet", "unet_explicit") or prior_use_text
        self.panns_ckpt_path = panns_ckpt_path
        self._panns_model = None
        self.clap_ckpt_path = clap_ckpt_path
        self._clap_model = None
        assert self.num_timesteps_cond <= kwargs["timesteps"]

        if conditioning_key is None:
            conditioning_key = "concat" if concat_mode else "crossattn"
        if cond_stage_config == "__is_unconditional__":
            conditioning_key = None
        else:
            conditioning_key = list(cond_stage_config.keys())

        self.condition_key = conditioning_key
        if fbank_shift:
            self.fbank_shift = 5.5
        else:
            self.fbank_shift = None

        if data_mean:
            self.data_mean = data_mean
            self.data_std = data_std
        else:
            self.data_mean = None
            self.data_std = None

        ckpt_path = kwargs.pop("ckpt_path", None)
        ignore_keys = kwargs.pop("ignore_keys", [])
        super().__init__(conditioning_key=conditioning_key, *args, **kwargs)
        self.use_clap = use_clap

        self.concat_mode = concat_mode
        self.cond_stage_key = cond_stage_key
        self.cond_stage_key_orig = cond_stage_key
        try:
            ch_mult = first_stage_config.get("params", {}).get("ddconfig", {}).get("ch_mult", [1])
            self.num_downs = len(ch_mult) - 1
        except Exception:
            self.num_downs = 0
        if not scale_by_std:
            self.scale_factor = scale_factor
        else:
            self.register_buffer("scale_factor", torch.tensor(scale_factor))
        self.instantiate_first_stage(first_stage_config)
        if self.learnable_prior:
            assert self.extra_channels, "learnable_prior requires extra_channels"
            if self.prior_type in ("unet", "unet_explicit"):
                assert self.prior_unet_config is not None, "prior_unet_config required for unet prior"
                self.prior_net = PriorUNetWrapper(self.prior_unet_config)
            elif self.prior_type == "crossattn":
                self.prior_net = CrossAttnPriorNet(in_channels=self.channels)
            elif self.prior_type == "concat_film":
                self.prior_net = ConcatFiLMPriorNet(in_channels=self.channels)
            elif self.prior_use_text:
                self.prior_net = CondPriorNet(in_channels=self.channels)
            else:
                self.prior_net = PriorNet(in_channels=self.channels)
            count_params(self.prior_net, verbose=True)
        else:
            self.prior_net = None
        self.unconditional_prob_cfg = unconditional_prob_cfg
        self.cond_stage_models = nn.ModuleList([])
        self.cond_stage_model_metadata = {}
        if conditioning_key is not None:
            self.instantiate_cond_stage(cond_stage_config)
            self.cond_stage_forward = cond_stage_forward

        self.clip_denoised = False
        self.bbox_tokenizer = None
        self.conditional_dry_run_finished = False
        self.restarted_from_ckpt = False
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys)
            self.restarted_from_ckpt = True

    def on_load_checkpoint(self, checkpoint):
        state_dict = checkpoint.get("state_dict", {})
        keys_to_remove = [k for k in state_dict if k.startswith("_panns_model.") or k.startswith("_clap_model.")]
        for k in keys_to_remove:
            del state_dict[k]

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.model.parameters())

        if self.learnable_prior:
            params = params + list(self.prior_net.parameters())

        if self.clap_trainable:
            for each in self.cond_stage_models:
                params = params + list(each.parameters())

        if self.learn_logvar:
            params.append(self.logvar)
        opt = torch.optim.AdamW(params, lr=lr)
        if self.use_scheduler:
            assert "target" in self.scheduler_config
            scheduler = instantiate_from_config(self.scheduler_config)
            scheduler = [
                {
                    "scheduler": LambdaLR(opt, lr_lambda=scheduler.schedule),
                    "interval": "step",
                    "frequency": 1,
                }
            ]
            return [opt], scheduler
        return opt

    def make_cond_schedule(self):
        self.cond_ids = torch.full(
            size=(self.num_timesteps,),
            fill_value=self.num_timesteps - 1,
            dtype=torch.long,
        )
        ids = torch.round(
            torch.linspace(0, self.num_timesteps - 1, self.num_timesteps_cond)
        ).long()
        self.cond_ids[: self.num_timesteps_cond] = ids

    @rank_zero_only
    @torch.no_grad()
    def on_train_batch_start(self, batch, batch_idx):
        if (
            self.scale_factor == 1
            and self.scale_by_std
            and self.current_epoch == 0
            and self.global_step == 0
            and batch_idx == 0
            and not self.restarted_from_ckpt
        ):
            x = super().get_input(batch, self.first_stage_key)
            x = x.to(self.device)
            encoder_posterior = self.encode_first_stage(x)
            z = self.get_first_stage_encoding(encoder_posterior).detach()
            del self.scale_factor
            self.register_buffer("scale_factor", 1.0 / z.flatten().std())

    def register_schedule(
        self,
        given_betas=None,
        beta_schedule="linear",
        timesteps=1000,
        linear_start=1e-4,
        linear_end=2e-2,
        cosine_s=8e-3,
    ):
        super().register_schedule(
            given_betas, beta_schedule, timesteps, linear_start, linear_end, cosine_s
        )

        self.shorten_cond_schedule = self.num_timesteps_cond > 1
        if self.shorten_cond_schedule:
            self.make_cond_schedule()

    def instantiate_first_stage(self, config):
        model = instantiate_from_config(config)
        self.first_stage_model = model.eval()
        self.first_stage_model.train = disabled_train
        for param in self.first_stage_model.parameters():
            param.requires_grad = False

    def make_decision(self, probability):
        if float(torch.rand(1)) < probability:
            return True
        else:
            return False

    def instantiate_cond_stage(self, config):
        self.cond_stage_model_metadata = {}

        for i, cond_model_key in enumerate(config.keys()):
            model = instantiate_from_config(config[cond_model_key])

            self.cond_stage_models.append(model)
            self.cond_stage_model_metadata[cond_model_key] = {
                "model_idx": i,
                "cond_stage_key": config[cond_model_key]["cond_stage_key"],
                "conditioning_key": config[cond_model_key]["conditioning_key"],
            }

    def get_first_stage_encoding(self, encoder_posterior):
        if isinstance(encoder_posterior, DiagonalGaussianDistribution):
            z = encoder_posterior.sample()
        elif isinstance(encoder_posterior, torch.Tensor):
            z = encoder_posterior
        else:
            raise NotImplementedError(
                f"encoder_posterior of type '{type(encoder_posterior)}' not yet implemented"
            )
        return self.scale_factor * z

    def get_learned_conditioning(self, c, key, unconditional_cfg):
        assert key in self.cond_stage_model_metadata.keys()
        
        if not unconditional_cfg:
            c = self.cond_stage_models[self.cond_stage_model_metadata[key]["model_idx"]](c)
        else:
            if isinstance(c, dict):
                c = c[list(c.keys())[0]]
        
            if isinstance(c, torch.Tensor):
                batchsize = c.size(0)
            elif isinstance(c, list):
                if key == "crossattn_llama":
                    batchsize = len(c[0])
                else:
                    if "clap_retrival" in key:
                        batchsize = len(c[0])
                    else:
                        batchsize = len(c)
            else:
                raise NotImplementedError()
            c = self.cond_stage_models[self.cond_stage_model_metadata[key]["model_idx"]].get_unconditional_condition(batchsize)

        return c

    def get_input(
        self,
        batch,
        k,
        return_first_stage_encode=True,
        return_decoding_output=False,
        return_encoder_input=False,
        return_encoder_output=False,
        unconditional_prob_cfg=0.1,
    ):    
        x = super().get_input(batch, k)

        x = x.to(self.device, non_blocking=True)

        if return_first_stage_encode:
            encoder_posterior = self.encode_first_stage(x)

            if isinstance(encoder_posterior, torch.Tensor):
                z = encoder_posterior
            else:
                z = self.get_first_stage_encoding(encoder_posterior).detach()

            if self.extra_channels: 
                extra = super().get_input(batch, self.extra_channel_key).to(self.device, non_blocking=True)
                extra = extra.reshape(extra.shape[0], 1, extra.shape[1], -1)
                extra_posterior = self.encode_first_stage(extra)
                e = self.get_first_stage_encoding(extra_posterior).detach()
                z = torch.cat([z, e], dim=1)
        else:
            z = None

        cond_dict = {}

        if len(self.cond_stage_model_metadata.keys()) > 0:
            unconditional_cfg = False
            if self.conditional_dry_run_finished and self.make_decision(unconditional_prob_cfg):
                unconditional_cfg = True
            for cond_model_key in self.cond_stage_model_metadata.keys():
                cond_stage_key = self.cond_stage_model_metadata[cond_model_key]["cond_stage_key"]
                
                if cond_model_key in cond_dict.keys():
                    continue
                
                if cond_stage_key != "all":
                    xc = super().get_input(batch, cond_stage_key)
                    if type(xc) == torch.Tensor:
                        xc = xc.to(self.device)
                else:
                    xc = batch

                c = self.get_learned_conditioning(xc, key=cond_model_key, unconditional_cfg=unconditional_cfg)
                
                if isinstance(c, dict):
                    for k in c.keys():
                        cond_dict[k] = c[k]
                else:
                    cond_dict[cond_model_key] = c

        out = [z, cond_dict]

        if return_decoding_output:
            xrec = self.decode_first_stage(z)
            out += [xrec]
        
        if return_encoder_input:
            out += [x]

        if return_encoder_output:
            out += [encoder_posterior]

        if not self.conditional_dry_run_finished:
            self.conditional_dry_run_finished = True

        return out

    def decode_first_stage(self, z):
        with torch.no_grad():
            z = 1.0 / self.scale_factor * z
            decoding = self.first_stage_model.decode(z)
        return decoding

    def mel_spectrogram_to_waveform(
        self, mel, savepath=".", bs=None, name="outwav", save=True
    ):
        if len(mel.size()) == 4:
            mel = mel.squeeze(1)
        mel = mel.permute(0, 2, 1)
        waveform = self.first_stage_model.vocoder(mel)
        waveform = waveform.cpu().detach().float().numpy()
        if save:
            self.save_waveform(waveform, savepath, name)
        return waveform

    def encode_first_stage(self, x):
        with torch.no_grad():
            return self.first_stage_model.encode(x)

    def extract_possible_loss_in_cond_dict(self, cond_dict):
        assert isinstance(cond_dict, dict)
        losses = {}

        for cond_key in cond_dict.keys():
            if "loss" in cond_key and "noncond" in cond_key:
                assert cond_key not in losses.keys()
                losses[cond_key] = cond_dict[cond_key]

        return losses

    def filter_useful_cond_dict(self, cond_dict):
        new_cond_dict = {}
        for key in cond_dict.keys():
            if key in self.cond_stage_model_metadata.keys():
                new_cond_dict[key] = cond_dict[key]
        
        for key in self.cond_stage_model_metadata.keys():
            assert key in new_cond_dict.keys(), "%s, %s" % (key, str(new_cond_dict.keys()))

        return new_cond_dict

    def shared_step(self, batch, **kwargs):
        if self.training:
            unconditional_prob_cfg = self.unconditional_prob_cfg
        else:
            unconditional_prob_cfg = 0.0

        x, c = self.get_input(batch, self.first_stage_key, unconditional_prob_cfg=unconditional_prob_cfg)
        
        loss, loss_dict = self(x, self.filter_useful_cond_dict(c))
        
        additional_loss_for_cond_modules = self.extract_possible_loss_in_cond_dict(c)

        assert isinstance(additional_loss_for_cond_modules, dict)

        loss_dict.update(additional_loss_for_cond_modules)

        if len(additional_loss_for_cond_modules.keys()) > 0:
            for k in additional_loss_for_cond_modules.keys():
                loss = loss + additional_loss_for_cond_modules[k]

        return loss, loss_dict
    
    def training_step(self, batch, batch_idx):
        self.warmup_step()

        if (
            self.state is None
            and len(self.trainer.optimizers[0].state_dict()["state"].keys()) > 0
        ):
            self.state = (
                self.trainer.optimizers[0].state_dict()["state"][1]["exp_avg"].clone()
            )
        elif self.state is not None and batch_idx % 100 == 0:
            try:
                assert (
                torch.sum(
                    torch.abs(
                        self.state
                        - self.trainer.optimizers[0].state_dict()["state"][1]["exp_avg"]
                    )
                )
                > 1e-7
                ), "Optimizer is not working"
            except:
                pass

        if len(self.metrics_buffer.keys()) > 0:
            for k in self.metrics_buffer.keys():
                self.log(
                    k,
                    self.metrics_buffer[k],
                    prog_bar=False,
                    logger=True,
                    on_step=True,
                    on_epoch=False,
                )
            self.metrics_buffer = {}
        
        loss, loss_dict = self.shared_step(batch)

        # loss_value = float(loss.detach().cpu())
        # if loss_value > 100.0:
        #     sample_metadata = batch.get("sample_metadata")
        #     try:
        #         sample_metadata_str = json.dumps(sample_metadata, ensure_ascii=False)
        #     except Exception:
        #         sample_metadata_str = str(sample_metadata)
        #     print(
        #         f"[high-loss-batch] rank={self.global_rank}, global_step={self.global_step}, batch_idx={batch_idx}, loss={loss_value}, sample_metadata={sample_metadata_str}",
        #         flush=True,
        #     )

        primary_loss_keys = ("train/loss_simple", "train/loss_vlb", "train/loss")
        other_loss_dict = {
            k: float(v) for k, v in loss_dict.items() if k not in primary_loss_keys
        }
        if len(other_loss_dict) > 0:
            self.log_dict(
                other_loss_dict,
                prog_bar=False,
                logger=True,
                on_step=True,
                on_epoch=True,
            )

        for key in primary_loss_keys:
            if key in loss_dict:
                self.log(
                    key,
                    float(loss_dict[key]),
                    prog_bar=True,
                    logger=True,
                    on_step=True,
                    on_epoch=True,
                )

        self.log(
            "global_step",
            float(self.global_step),
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=False,
        )

        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log(
            "lr_abs",
            float(lr),
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=False
        )

        return loss

    def forward(self, x, c, *args, **kwargs):
        if self.logit_normal_t > 0:
            t = torch.sigmoid(torch.randn([x.shape[0]], device=self.device) * self.logit_normal_t)
        else:
            t = torch.rand([x.shape[0]], device=self.device)

        if self.reparam_bridge:
            t = t.clamp(min=1e-3, max=1 - 1e-3)

        loss, loss_dict = self.p_losses(x, c, t, *args, **kwargs)
        return loss, loss_dict

    def apply_model(self, x_noisy, t, cond, return_ids=False):
        x_recon = self.model(x_noisy, t, cond_dict=cond)

        if isinstance(x_recon, tuple) and not return_ids:
            return x_recon[0]
        else:
            return x_recon

    def _expand_to_input_dim(self, v, x):
        while v.ndim < x.ndim:
            v = v.unsqueeze(-1)
        return v

    def _chen_bridge_xt(self, x0, x1, t, eps):
        terms_t = self._chen_vp_terms(t)
        alpha_t = terms_t["alpha_t"]
        alpha_bar_t = terms_t["alpha_bar_t"]
        sigma_t2 = terms_t["sigma_t2"]
        sigma_bar_t2 = terms_t["sigma_bar_t2"]
        sigma_12 = terms_t["sigma_12"]
        sigma_t = terms_t["sigma_t"]
        sigma_bar_t = terms_t["sigma_bar_t"]
        sigma_1 = terms_t["sigma_1"]

        w_x0 = self._expand_to_input_dim(alpha_t * sigma_bar_t2 / (sigma_12 + 1e-12), x0)
        w_x1 = self._expand_to_input_dim(alpha_bar_t * sigma_t2 / (sigma_12 + 1e-12), x0)
        w_eps = self._expand_to_input_dim(alpha_t * sigma_bar_t * sigma_t / (sigma_1 + 1e-12), x0)

        return w_x0 * x0 + w_x1 * x1 + w_eps * eps

    def _chen_vp_terms(self, t):
        beta0 = t.new_tensor(self.fg_beta0)
        beta1 = t.new_tensor(self.fg_beta1)
        beta_delta = beta1 - beta0

        if self.fg_schedule == "vp_linear":
            integral_t = beta0 * t + 0.5 * beta_delta * t * t
            integral_1 = beta0 + 0.5 * beta_delta

            alpha_t = torch.exp(-0.5 * integral_t)
            alpha_bar_t = torch.exp(0.5 * (integral_1 - integral_t))

            sigma_t2 = torch.exp(integral_t) - 1.0
            sigma_bar_t2 = torch.exp(integral_1) - torch.exp(integral_t)
            sigma_12 = torch.exp(integral_1) - 1.0
        elif self.fg_schedule == "gmax_linear":
            sigma_t2 = beta0 * t + 0.5 * beta_delta * t * t
            sigma_12 = beta0 + 0.5 * beta_delta
            sigma_bar_t2 = sigma_12 - sigma_t2

            alpha_t = torch.ones_like(t)
            alpha_bar_t = torch.ones_like(t)
        else:
            raise NotImplementedError(f"Unsupported fg_schedule: {self.fg_schedule}")

        sigma_t = torch.sqrt(torch.clamp(sigma_t2, min=1e-12))
        sigma_bar_t = torch.sqrt(torch.clamp(sigma_bar_t2, min=1e-12))
        sigma_1 = torch.sqrt(torch.clamp(sigma_12, min=1e-12))

        return {
            "alpha_t": alpha_t,
            "alpha_bar_t": alpha_bar_t,
            "sigma_t2": sigma_t2,
            "sigma_bar_t2": sigma_bar_t2,
            "sigma_12": sigma_12,
            "sigma_t": sigma_t,
            "sigma_bar_t": sigma_bar_t,
            "sigma_1": sigma_1,
        }

    def _chen_predict_x0(self, model_out, x1):
        if self.data_prediction:
            return model_out
        if x1 is not None:
            return model_out + x1
        return model_out

    def _add_bridge_start_noise(self, x, spr_t):
        if not self.bridge_input_start_noise:
            return x
        if self.bridge_input_start_noise_std <= 0:
            return x
        weight = torch.clamp(1.0 - spr_t, min=0.0)
        return x + self.bridge_input_start_noise_std * weight * torch.randn_like(x)

    def p_losses(self, x_start, cond, t, noise=None):
        channel = x_start.shape[1]

        if channel != self.channels:
            x_extra = x_start[:, self.channels:, :, :]
            x_start = x_start[:, :self.channels, :, :]
        noise = default(noise, lambda: torch.randn_like(x_start))

        spr_t = t.view(-1, 1, 1, 1)

        chen_bridge_active = self.chen_bridge and channel != self.channels

        if chen_bridge_active:
            if self.learnable_prior and self.prior_type in ("unet", "unet_explicit"):
                prior_out = self.prior_net(x_extra, cond["crossattn_text"])
            elif self.learnable_prior:
                text_emb = cond["crossattn_text"][0] if self.prior_needs_text else None
                prior_out = self.prior_lambda * self.prior_net(x_extra, text_emb)
            x_noisy = self._chen_bridge_xt(x_start, x_extra, t, noise)
            if self.bridge_mode:
                x_noisy = self._add_bridge_start_noise(x_noisy, spr_t)
            target = x_start

        elif self.bridge_mode and channel != self.channels:
            if self.sb_schedule:
                if self.asym_noise:
                    sigma_bb = self.sb_rho * torch.sqrt(spr_t + 1e-8) * (1 - spr_t)
                else:
                    sigma_bb = self.sb_rho * torch.sqrt(spr_t * (1 - spr_t) + 1e-8)
                if self.reparam_bridge:
                    if self.learnable_prior:
                        if self.prior_type in ("unet", "unet_explicit"):
                            prior_out = self.prior_net(x_extra, cond["crossattn_text"])
                        else:
                            text_emb = cond["crossattn_text"][0] if self.prior_needs_text else None
                            prior_out = self.prior_lambda * self.prior_net(x_extra, text_emb)
                        prior_mse = F.mse_loss(prior_out, x_start)
                        x_noisy = (1 - spr_t) * prior_out + spr_t * x_start + sigma_bb * noise
                    else:
                        x_noisy = spr_t * x_start + sigma_bb * noise
                else:
                    x_noisy = (1 - spr_t) * x_extra + spr_t * x_start + sigma_bb * noise
            else:
                x_noisy = (1 - spr_t) * x_extra + spr_t * x_start + self.sigma_min * noise
            x_noisy = self._add_bridge_start_noise(x_noisy, spr_t)
            target = x_start if self.data_prediction else x_start - x_extra
        elif self.sb_schedule:
            if self.asym_noise:
                sigma_bb = self.sb_rho * torch.sqrt(spr_t + 1e-8) * (1 - spr_t)
            else:
                sigma_bb = self.sb_rho * torch.sqrt(spr_t * (1 - spr_t) + 1e-8)
            noise_extra = torch.randn_like(x_start)
            x_noisy = (1 - (1 - self.sigma_min) * spr_t) * noise + spr_t * x_start + sigma_bb * noise_extra
            target = x_start if self.data_prediction else x_start - (1 - self.sigma_min) * noise
        else:
            x_noisy = (1 - (1 - self.sigma_min) * spr_t) * noise + spr_t * x_start
            target = x_start if self.data_prediction else x_start - (1 - self.sigma_min) * noise

        if channel != self.channels:
            unet_cond = prior_out if (self.learnable_prior and self.prior_type == "unet_explicit") else x_extra
            x_noisy = torch.cat([x_noisy, unet_cond], dim=1)

        model_output = self.apply_model(x_noisy, t, cond)

        if channel != self.channels:
            model_output = model_output[:, :self.channels, :, :]

        loss_dict = {}
        prefix = "train" if self.training else "val"
        if len(model_output.shape) == 3:
            loss_simple = self.get_loss(model_output, target, mean=False).mean([1, 2])
        else:
            loss_simple = self.get_loss(model_output, target, mean=False).mean([1, 2, 3])

        if self.loss_t_weight > 0:
            t_w = t.pow(self.loss_t_weight) / (1 - t + 1e-4)
            t_w = t_w / (t_w.mean() + 1e-8)
            loss_simple = loss_simple * t_w

        loss_dict.update({f"{prefix}/loss_simple": loss_simple.mean()})

        t_int = (t * 1000).long()

        logvar_t = self.logvar[t_int].to(self.device)
        loss = loss_simple / torch.exp(logvar_t) + logvar_t
        if self.learn_logvar:
            loss_dict.update({f"{prefix}/loss_gamma": loss.mean()})
            loss_dict.update({"logvar": self.logvar.data.mean()})

        loss = self.l_simple_weight * loss.mean()

        if len(model_output.shape) == 3:
            loss_vlb = self.get_loss(model_output, target, mean=False).mean(dim=(1, 2))
        else:
            loss_vlb = self.get_loss(model_output, target, mean=False).mean(dim=(1, 2, 3))
        loss_vlb = (self.lvlb_weights[t_int] * loss_vlb).mean()
        loss_dict.update({f"{prefix}/loss_vlb": loss_vlb})
        loss += self.original_elbo_weight * loss_vlb
        loss_dict.update({f"{prefix}/loss": loss})

        if self.learnable_prior and self.bridge_mode and channel != self.channels:
            loss_dict.update({f"{prefix}/prior_mse": prior_mse.detach()})
            if self.prior_mse_weight > 0:
                loss = loss + self.prior_mse_weight * prior_mse

        return loss, loss_dict

    def save_waveform(self, waveform, savepath, name="outwav"):
        wave_num = waveform.shape[0]
        path = None
        for i in range(waveform.shape[0]):
            if type(name) is str:
                if wave_num == 1:
                    path = os.path.join(
                        savepath, "%s_%s_%s.wav" % (self.global_step, i, name)
                    )
                else:
                    path = os.path.join(
                        savepath, "%s_%s_%s.wav" % (self.global_step, i, name)
                    )
                    path = path[:-4] + "_" + str(i) + ".wav"

            elif type(name) is list:
                if wave_num != len(name):
                    if path:
                        path = path[:-5] + "" + str(i) + ".wav"
                    else:
                        path = os.path.join(
                            savepath,
                            "%s.wav"
                            % (
                                os.path.basename(name[i])
                                if (not ".wav" in name[i])
                                else os.path.basename(name[i]).split(".")[0]
                            ),
                        )
                        path = path[:-4] + "_" + str(i) + ".wav"
                else:
                    path = os.path.join(
                        savepath,
                        "%s.wav"
                        % (
                            os.path.basename(name[i])
                            if (not ".wav" in name[i])
                            else os.path.basename(name[i]).split(".")[0]
                        ),
                    )
            else:
                raise NotImplementedError
            try:
                sf.write(path, waveform[i, 0], samplerate=self.sampling_rate)
            except:
                sf.write(path, waveform[i], samplerate=self.sampling_rate)

    def _model_forward(self, x, t_batch, cond, x_T):
        if self.extra_channels and x_T is not None:
            out = self.apply_model(torch.cat([x, x_T], dim=1), t_batch, cond)
            return out[:, :self.channels]
        return self.apply_model(x, t_batch, cond)

    def _to_velocity(self, model_out, x, t, x_T):
        if not self.data_prediction:
            return model_out
        if self.bridge_mode and x_T is not None:
            return model_out - x_T
        spr_t = t.view(-1, 1, 1, 1)
        denom = 1 - (1 - self.sigma_min) * spr_t + 1e-8
        return (model_out - (1 - self.sigma_min) * x) / denom

    def solve_chen_sde(self, n_timesteps, batch_size, shape, cond=None, unconditional_guidance_scale=1.0, unconditional_conditioning=None, x_T=None, temperature=1.0, spks=None):
        if len(shape) == 3:
            C, H, W = shape
            size = (batch_size, C, H, W)
        else:
            C, L = shape
            size = (batch_size, C, L)

        x1 = x_T
        if x1 is not None:
            x = x1.clone()
        else:
            x = torch.randn(size, device=self.device) * temperature

        sampling_eps = self.chen_sampling_eps
        t_span = torch.linspace(1.0 - sampling_eps, sampling_eps, n_timesteps + 1, device=self.device)

        for step in range(1, len(t_span)):
            s = t_span[step - 1]
            t = t_span[step]
            s_batch = s.view(1).expand(batch_size)
            t_batch = t.view(1).expand(batch_size)

            model_out = self._model_forward(x, s_batch, cond, x1)
            x0_pred = self._chen_predict_x0(model_out, x1)

            terms_s = self._chen_vp_terms(s_batch)
            terms_t = self._chen_vp_terms(t_batch)

            ratio_sigma = terms_t["sigma_t2"] / (terms_s["sigma_t2"] + 1e-12)
            coef_x = self._expand_to_input_dim(
                terms_t["alpha_t"] * terms_t["sigma_t2"] / (terms_s["alpha_t"] * terms_s["sigma_t2"] + 1e-12), x
            )
            coef_theta = self._expand_to_input_dim(terms_t["alpha_t"] * (1.0 - ratio_sigma), x)
            coef_noise = self._expand_to_input_dim(
                terms_t["alpha_t"] * terms_t["sigma_t"] * torch.sqrt(torch.clamp(1.0 - ratio_sigma, min=0.0)), x
            )

            eps_step = torch.randn_like(x) * temperature
            x = coef_x * x + coef_theta * x0_pred + coef_noise * eps_step

        return x

    def solve_chen_ode(self, n_timesteps, batch_size, shape, cond=None, unconditional_guidance_scale=1.0, unconditional_conditioning=None, x_T=None, temperature=1.0, spks=None):
        if len(shape) == 3:
            C, H, W = shape
            size = (batch_size, C, H, W)
        else:
            C, L = shape
            size = (batch_size, C, L)

        x1 = x_T
        if x1 is not None:
            x = x1.clone()
        else:
            x = torch.randn(size, device=self.device) * temperature

        sampling_eps = self.chen_sampling_eps
        t_span = torch.linspace(1.0 - sampling_eps, sampling_eps, n_timesteps + 1, device=self.device)

        for step in range(1, len(t_span)):
            s = t_span[step - 1]
            t = t_span[step]
            s_batch = s.view(1).expand(batch_size)
            t_batch = t.view(1).expand(batch_size)

            model_out = self._model_forward(x, s_batch, cond, x1)
            x0_pred = self._chen_predict_x0(model_out, x1)

            terms_s = self._chen_vp_terms(s_batch)
            terms_t = self._chen_vp_terms(t_batch)

            coef_x = self._expand_to_input_dim(
                terms_t["alpha_t"] * terms_t["sigma_t"] * terms_t["sigma_bar_t"]
                / (terms_s["alpha_t"] * terms_s["sigma_t"] * terms_s["sigma_bar_t"] + 1e-12),
                x,
            )
            coef_theta = self._expand_to_input_dim(
                terms_t["alpha_t"]
                * (
                    terms_t["sigma_bar_t2"]
                    - (terms_s["sigma_bar_t"] * terms_t["sigma_t"] * terms_t["sigma_bar_t"] / (terms_s["sigma_t"] + 1e-12))
                )
                / (terms_t["sigma_12"] + 1e-12),
                x,
            )

            if x1 is None:
                coef_x1 = 0.0
                x1_term = 0.0
            else:
                coef_x1 = self._expand_to_input_dim(
                    terms_t["alpha_bar_t"]
                    * (
                        terms_t["sigma_t2"]
                        - (terms_s["sigma_t"] * terms_t["sigma_t"] * terms_t["sigma_bar_t"] / (terms_s["sigma_bar_t"] + 1e-12))
                    )
                    / (terms_t["sigma_12"] + 1e-12),
                    x,
                )
                x1_term = coef_x1 * x1

            x = coef_x * x + coef_theta * x0_pred + x1_term

        return x

    def solve_euler(self, n_timesteps, batch_size, shape, cond=None, unconditional_guidance_scale=1.0, unconditional_conditioning=None, x_T=None, temperature=1.0, spks=None):
        if len(shape) == 3:
            C, H, W = shape
            size = (batch_size, C, H, W)
        else:
            C, L = shape
            size = (batch_size, C, L)

        if self.bridge_mode and x_T is not None:
            x = x_T.clone()
        else:
            x = torch.randn(size, device=self.device) * temperature

        t_span = torch.linspace(0, 1, n_timesteps + 1, device=self.device)
        t, _, dt = t_span[0], t_span[-1], t_span[1] - t_span[0]

        for step in range(1, len(t_span)):
            t_batch = t.view(1).expand(batch_size)
            model_out = self._model_forward(x, t_batch, cond, x_T)
            dphi_dt = self._to_velocity(model_out, x, t_batch, x_T)
            x = x + dt * dphi_dt
            t = t + dt
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t

        return x

    def solve_ei(self, n_timesteps, batch_size, shape, cond=None, unconditional_guidance_scale=1.0, unconditional_conditioning=None, x_T=None, temperature=1.0, spks=None):
        if len(shape) == 3:
            C, H, W = shape
            size = (batch_size, C, H, W)
        else:
            C, L = shape
            size = (batch_size, C, L)

        if self.reparam_bridge:
            if self.learnable_prior:
                if self.prior_type in ("unet", "unet_explicit"):
                    cached_prior = self.prior_net(x_T, cond["crossattn_text"])
                else:
                    text_emb = cond["crossattn_text"][0] if self.prior_needs_text else None
                    cached_prior = self.prior_lambda * self.prior_net(x_T, text_emb)
                x = cached_prior.clone()
            else:
                x = torch.zeros(size, device=self.device)
        elif self.bridge_mode and x_T is not None:
            x = x_T.clone()
        else:
            x = torch.randn(size, device=self.device) * temperature

        sampling_eps = 1e-3
        t_span = torch.linspace(sampling_eps, 1 - sampling_eps, n_timesteps + 1, device=self.device)
        rho = self.sb_rho

        ei_extra = cached_prior if (self.learnable_prior and self.prior_type == "unet_explicit") else x_T

        for step in range(1, len(t_span)):
            t_prev = t_span[step - 1]
            t_curr = t_span[step]
            t_batch = t_prev.view(1).expand(batch_size)
            model_out = self._model_forward(x, t_batch, cond, ei_extra)

            if self.bridge_mode and self.sb_schedule and self.data_prediction and x_T is not None:
                if self.asym_noise:
                    sigma_p = torch.sqrt(t_prev + 1e-8) * (1 - t_prev)
                    sigma_c = torch.sqrt(t_curr + 1e-8) * (1 - t_curr)
                else:
                    sigma_p = torch.sqrt(t_prev * (1 - t_prev) + 1e-8)
                    sigma_c = torch.sqrt(t_curr * (1 - t_curr) + 1e-8)
                R = (sigma_c / (sigma_p + 1e-8)).view(1, 1, 1, 1)
                w_s = t_curr.view(1, 1, 1, 1) - R * t_prev.view(1, 1, 1, 1)
                if self.reparam_bridge:
                    if self.learnable_prior:
                        w_P = (1 - t_curr).view(1, 1, 1, 1) - R * (1 - t_prev).view(1, 1, 1, 1)
                        x = R * x + w_s * model_out + w_P * cached_prior
                    else:
                        x = R * x + w_s * model_out
                else:
                    w_y = (1 - t_curr).view(1, 1, 1, 1) - R * (1 - t_prev).view(1, 1, 1, 1)
                    x = R * x + w_s * model_out + w_y * x_T
            else:
                dphi_dt = self._to_velocity(model_out, x, t_batch, x_T)
                dt = t_curr - t_prev
                x = x + dt * dphi_dt

        return x

    @torch.no_grad()
    def sample_log(
        self,
        cond,
        batch_size,
        ddim,
        ddim_steps,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
        use_plms=False,
        mask=None,
        x_T=None,
        **kwargs,
    ):
        if mask is not None:
            shape = (self.channels, mask.size()[-2], mask.size()[-1])
        else:
            shape = (self.channels, self.latent_t_size, self.latent_f_size)

        intermediate = None
        if self.chen_bridge:
            if self.chen_sampler_type == "sde":
                solver_fn = self.solve_chen_sde
            elif self.chen_sampler_type == "ode":
                solver_fn = self.solve_chen_ode
            else:
                raise NotImplementedError(f"Unsupported chen_sampler_type: {self.chen_sampler_type}")
        else:
            solver_fn = self.solve_ei if self.use_ei_solver else self.solve_euler
        samples = solver_fn(
            ddim_steps,
            batch_size,
            shape,
            cond,
            unconditional_guidance_scale=unconditional_guidance_scale,
            unconditional_conditioning=unconditional_conditioning,
            x_T=x_T
        )

        return samples, intermediate

    @torch.no_grad()
    def generate_sample(
        self,
        batchs,
        ddim_steps=200,
        ddim_eta=1.0,
        x_T=None,
        n_gen=1,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
        name="waveform",
        use_plms=False,
        limit_num=None,
        save=True,
        save_mixed=True,
        **kwargs,
    ):
        assert x_T is None

        if use_plms:
            assert ddim_steps is not None

        use_ddim = ddim_steps is not None
        waveform_save_path = None
        try:
            waveform_save_path = os.path.join(self.get_log_dir(), name)
        except:
            waveform_save_path = name
        os.makedirs(waveform_save_path, exist_ok=True)

        if (
            "audiocaps" in waveform_save_path
            and len(os.listdir(waveform_save_path)) >= 964
        ):
            return waveform_save_path

        with self.ema_scope("Plotting"):
            for i, batch in enumerate(batchs):
                z, c = self.get_input(
                    batch,
                    self.first_stage_key, 
                    unconditional_prob_cfg=0.0
                )

                fnames = list(super().get_input(batch, "fname"))

                if self.extra_channels:
                    extra = super().get_input(batch, self.extra_channel_key).to(self.device)
                    extra = extra.reshape(extra.shape[0], 1, extra.shape[1], extra.shape[2])
                    try:
                        extra_posterior = self.encode_first_stage(extra)
                    except:
                        pass
                    x_T = self.get_first_stage_encoding(extra_posterior).detach()

                    if save_mixed:
                        mixed_save_path = os.path.join(waveform_save_path, "mixed")
                        os.makedirs(mixed_save_path, exist_ok=True)

                        mixed_waveform = batch["mixed_waveform"]

                        count = 0
                        for name in fnames:
                            sf.write(os.path.join(mixed_save_path, os.path.basename(name)), mixed_waveform[count].cpu().numpy().T, self.sampling_rate)
                            count += 1

                if limit_num is not None and i * z.size(0) > limit_num:
                    break

                if self.condition_key:
                    c = self.filter_useful_cond_dict(c)

                text = super().get_input(batch, "text")

                batch_size = z.shape[0] * n_gen

                if self.condition_key:
                    for cond_key in c.keys():   
                        if isinstance(c[cond_key], list):
                            for i in range(len(c[cond_key])):
                                c[cond_key][i] = torch.cat([c[cond_key][i]] * n_gen, dim=0)        
                        elif isinstance(c[cond_key], dict):
                            for k in c[cond_key].keys():
                                c[cond_key][k] = torch.cat([c[cond_key][k]] * n_gen, dim=0)   
                        else:
                            c[cond_key] = torch.cat([c[cond_key]] * n_gen, dim=0)
                
                    text = text * n_gen

                    if unconditional_guidance_scale != 1.0:
                        unconditional_conditioning = {}
                        for key in self.cond_stage_model_metadata:
                            model_idx = self.cond_stage_model_metadata[key]["model_idx"]
                            unconditional_conditioning[key] = self.cond_stage_models[model_idx].get_unconditional_condition(batch_size)

                samples, _ = self.sample_log(
                    cond=c,
                    batch_size=batch_size,
                    x_T=x_T,
                    ddim=use_ddim,
                    ddim_steps=ddim_steps,
                    eta=ddim_eta,
                    unconditional_guidance_scale=unconditional_guidance_scale,
                    unconditional_conditioning=unconditional_conditioning,
                    use_plms=use_plms,
                )
                if self.extra_channels:
                    samples = samples[:, :self.channels, :, :]

                mel = self.decode_first_stage(samples)

                if self.fbank_shift:
                    mel = mel - self.fbank_shift

                if self.data_std:
                    mel = (mel * self.data_std) + self.data_mean

                waveform = self.mel_spectrogram_to_waveform(mel, savepath=waveform_save_path, bs=None, name=fnames, save=False)
                if n_gen >= 3:
                    if self.use_clap: 
                        try: 
                            best_index = []
                            if self.use_retrival:
                                similarity = self.get_retrival_similarity(batch, waveform)
                            else:
                                similarity = self.clap.cos_similarity(torch.FloatTensor(waveform).squeeze(1), text)
                            for i in range(z.shape[0]):
                                candidates = similarity[i :: z.shape[0]]
                                max_index = torch.argmax(candidates).item()
                                best_index.append(i + max_index * z.shape[0])
                            waveform = waveform[best_index]
                        except Exception as e:
                            rank_zero_print("Warning: while calculating CLAP score (not fatal), ", e)
                else:
                    waveform = waveform[: z.shape[0]]

            if save:
                self.save_waveform(waveform, waveform_save_path, name=fnames)
                return waveform
            else:
                return waveform


class DiffusionWrapper(pl.LightningModule):
    def __init__(self, diff_model_config, conditioning_key):
        super().__init__()
        self.diffusion_model = instantiate_from_config(diff_model_config)

        self.conditioning_key = conditioning_key

        if self.conditioning_key is not None:
            for key in self.conditioning_key:
                if "concat" in key or "crossattn" in key or "hybrid" in key or "film" in key or "noncond" in key:
                    continue
                else:
                    raise ValueError("The conditioning key %s is illegal" % key)
        
        self.being_verbosed_once = False

    def forward(self, x, t, cond_dict: dict={}):
        x = x.contiguous()
        t = t.contiguous()

        xc = x

        y = None
        context_list, attn_mask_list = None, None

        for key in cond_dict.keys():
            if "crossattn" in key:
                context_list, attn_mask_list = [], []

        for key in cond_dict.keys():
            if "concat" in key:
                xc = torch.cat([x, cond_dict[key].unsqueeze(1)], dim=1)    
            elif "film" in key:
                if y is None:
                    y = cond_dict[key].squeeze(1)
                else:
                    if self.diffusion_model.concate_film:
                        y = [y, cond_dict[key].squeeze(1)]
                    else:
                        y = torch.cat([y, cond_dict[key].squeeze(1)], dim=-1)
            elif "crossattn" in key:
                if isinstance(cond_dict[key], dict):
                    for k in cond_dict[key].keys():
                        if "crossattn" in k:
                            context, attn_mask = cond_dict[key][k]
                else:
                    assert len(cond_dict[key]) == 2, "The context condition for %s you returned should have two element, one context one mask" % (key)
                    context, attn_mask = cond_dict[key]
                
                context_list.append(context)
                attn_mask_list.append(attn_mask)

            elif "noncond" in key:
                continue
            else:
                raise NotImplementedError()

        out = self.diffusion_model(xc, t, context_list=context_list, y=y, context_attn_mask_list=attn_mask_list)
        return out
