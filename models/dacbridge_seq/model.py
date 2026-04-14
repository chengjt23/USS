import os

import soundfile as sf
import torch
import torch.nn.functional as F

from models.dacbridge.model import LatentDiffusionDACVAE
from models.flowsep.model import LatentDiffusion
from utils.tools import default


class LatentDiffusionDACVAE1D(LatentDiffusionDACVAE):
    def __init__(self, backbone_config, latent_t_size=104, channels=128, *args, **kwargs):
        super().__init__(
            unet_config=backbone_config,
            latent_t_size=latent_t_size,
            latent_f_size=1,
            channels=channels,
            *args,
            **kwargs,
        )

    def p_losses(self, x_start, cond, t, noise=None):
        channel = x_start.shape[1]
        if channel != self.channels:
            x_extra = x_start[:, self.channels :, :]
            x_start = x_start[:, : self.channels, :]
        noise = default(noise, lambda: torch.randn_like(x_start))
        spr_t = t.view(-1, 1, 1)
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
            model_output = model_output[:, : self.channels, :]

        loss_dict = {}
        prefix = "train" if self.training else "val"
        loss_simple = self.get_loss(model_output, target, mean=False).mean([1, 2])

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
        loss_vlb = self.get_loss(model_output, target, mean=False).mean(dim=(1, 2))
        loss_vlb = (self.lvlb_weights[t_int] * loss_vlb).mean()
        loss_dict.update({f"{prefix}/loss_vlb": loss_vlb})
        loss += self.original_elbo_weight * loss_vlb
        loss_dict.update({f"{prefix}/loss": loss})

        if self.learnable_prior and self.bridge_mode and channel != self.channels:
            loss_dict.update({f"{prefix}/prior_mse": prior_mse.detach()})
            if self.prior_mse_weight > 0:
                loss = loss + self.prior_mse_weight * prior_mse

        return loss, loss_dict

    def _to_velocity(self, model_out, x, t, x_T):
        if not self.data_prediction:
            return model_out
        if self.bridge_mode and x_T is not None:
            return model_out - x_T
        spr_t = t.view(-1, 1, 1)
        denom = 1 - (1 - self.sigma_min) * spr_t + 1e-8
        return (model_out - (1 - self.sigma_min) * x) / denom

    def solve_ei(
        self,
        n_timesteps,
        batch_size,
        shape,
        cond=None,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
        x_T=None,
        temperature=1.0,
        spks=None,
    ):
        del unconditional_guidance_scale, unconditional_conditioning, spks
        if len(shape) == 2:
            channels, length = shape
            size = (batch_size, channels, length)
        else:
            raise RuntimeError(f"Unexpected latent shape spec {shape}")

        cached_prior = None
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
                r = (sigma_c / (sigma_p + 1e-8)).view(1, 1, 1)
                w_s = t_curr.view(1, 1, 1) - r * t_prev.view(1, 1, 1)
                if self.reparam_bridge:
                    if self.learnable_prior:
                        w_p = (1 - t_curr).view(1, 1, 1) - r * (1 - t_prev).view(1, 1, 1)
                        x = r * x + w_s * model_out + w_p * cached_prior
                    else:
                        x = r * x + w_s * model_out
                else:
                    w_y = (1 - t_curr).view(1, 1, 1) - r * (1 - t_prev).view(1, 1, 1)
                    x = r * x + w_s * model_out + w_y * x_T
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
        del ddim, use_plms, kwargs
        if mask is not None:
            shape = (self.channels, mask.size(-1))
        else:
            shape = (self.channels, self.latent_t_size)

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
            x_T=x_T,
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
        del ddim_eta, kwargs
        assert x_T is None
        use_ddim = ddim_steps is not None
        try:
            waveform_save_path = os.path.join(self.get_log_dir(), name)
        except Exception:
            waveform_save_path = name
        os.makedirs(waveform_save_path, exist_ok=True)

        if "audiocaps" in waveform_save_path and len(os.listdir(waveform_save_path)) >= 964:
            return waveform_save_path

        with self.ema_scope("Plotting"):
            for i, batch in enumerate(batchs):
                z, c = self.get_input(batch, self.first_stage_key, unconditional_prob_cfg=0.0)
                fnames = list(super(LatentDiffusion, self).get_input(batch, "fname"))

                if self.extra_channels:
                    extra = super(LatentDiffusion, self).get_input(batch, self.extra_channel_key)
                    extra = extra.to(self.device)
                    extra_z = self.encode_first_stage(extra)
                    x_T = self.get_first_stage_encoding(extra_z).detach()

                    if save_mixed:
                        mixed_save_path = os.path.join(waveform_save_path, "mixed")
                        os.makedirs(mixed_save_path, exist_ok=True)
                        mixed_waveform = batch["mixed_waveform"]
                        for j, fname in enumerate(fnames):
                            sf.write(
                                os.path.join(mixed_save_path, os.path.basename(fname)),
                                mixed_waveform[j].cpu().numpy().T,
                                self.sampling_rate,
                            )

                if limit_num is not None and i * z.size(0) > limit_num:
                    break

                if self.condition_key:
                    c = self.filter_useful_cond_dict(c)

                text = super(LatentDiffusion, self).get_input(batch, "text")
                batch_size = z.shape[0] * n_gen

                if self.condition_key:
                    for cond_key in c.keys():
                        if isinstance(c[cond_key], list):
                            for ci in range(len(c[cond_key])):
                                c[cond_key][ci] = torch.cat([c[cond_key][ci]] * n_gen, dim=0)
                        elif isinstance(c[cond_key], dict):
                            for dk in c[cond_key].keys():
                                c[cond_key][dk] = torch.cat([c[cond_key][dk]] * n_gen, dim=0)
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
                    unconditional_guidance_scale=unconditional_guidance_scale,
                    unconditional_conditioning=unconditional_conditioning,
                    use_plms=use_plms,
                )
                if self.extra_channels:
                    samples = samples[:, : self.channels]

                wav_out = self.decode_first_stage(samples)
                waveform = wav_out.cpu().detach().float().numpy()

                if n_gen >= 3 and self.use_clap:
                    try:
                        best_index = []
                        similarity = self.clap.cos_similarity(torch.FloatTensor(waveform).squeeze(1), text)
                        for idx in range(z.shape[0]):
                            candidates = similarity[idx :: z.shape[0]]
                            best_index.append(idx + torch.argmax(candidates).item() * z.shape[0])
                        waveform = waveform[best_index]
                    except Exception as e:
                        from utils.tools import rank_zero_print

                        rank_zero_print("Warning: while calculating CLAP score (not fatal), ", e)
                else:
                    waveform = waveform[: z.shape[0]]

            if save:
                self.save_waveform(waveform, waveform_save_path, name=fnames)
            return waveform
