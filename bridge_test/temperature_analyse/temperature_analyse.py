import sys
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))

import argparse
import yaml
import torch
import torchaudio
import numpy as np
import torch.nn.functional as torchF
import matplotlib.pyplot as plt
from pytorch_lightning import seed_everything

from utils.tools import instantiate_from_config
from utils.audio import TacotronSTFT, get_mel_from_wav
from data.wds_datamodule import WDSDataModule
from train import WrappedDataLoader, build_stft_tool


def load_model(config_path, ckpt_path):
    with open(config_path, "r") as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)
    model = instantiate_from_config(configs["model"]).cuda().eval()
    ckpt = torch.load(ckpt_path, map_location="cuda", weights_only=False)
    if "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    model.load_state_dict(ckpt, strict=False)
    return model, configs


def build_val_loader(configs):
    stft_tool = build_stft_tool(configs)
    datamodule = WDSDataModule(**configs["datamodule"]["data_config"])
    _, val_loader, _ = datamodule.make_loader
    return WrappedDataLoader(val_loader, configs, stft_tool)


def load_panns_model(device="cuda"):
    import importlib.util
    fad_dir = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../metrics/fad"))
    pu_spec = importlib.util.spec_from_file_location("_panns_pytorch_utils", os.path.join(fad_dir, "pytorch_utils.py"))
    pu_mod = importlib.util.module_from_spec(pu_spec)
    sys.modules["pytorch_utils"] = pu_mod
    pu_spec.loader.exec_module(pu_mod)
    m_spec = importlib.util.spec_from_file_location("_panns_models", os.path.join(fad_dir, "models.py"))
    m_mod = importlib.util.module_from_spec(m_spec)
    m_spec.loader.exec_module(m_mod)
    model = m_mod.Cnn14_16k(sample_rate=16000, window_size=512, hop_size=160,
                             mel_bins=64, fmin=50, fmax=8000, classes_num=527)
    ckpt = torch.load(os.path.join(fad_dir, "Cnn14_16k_mAP=0.438.pth"), map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval()


def load_clap_model(device="cuda"):
    clap_dir = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../metrics/clapscore"))
    if clap_dir not in sys.path:
        sys.path.insert(0, clap_dir)
    from models.clap_encoder import CLAP_Encoder
    ckpt_path = os.path.join(clap_dir, "music_speech_audioset_epoch_15_esc_89.98.pt")
    return CLAP_Encoder(pretrained_path=ckpt_path, sampling_rate=32000, device=device).eval()


def panns_embedding(model, wav, sr=16000, device="cuda"):
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    with torch.no_grad():
        out = model(wav.float().to(device), None)
    return out["embedding"].float().cpu()


def clap_audio_embedding(model, wav, sr=16000, device="cuda"):
    if sr != 32000:
        wav = torchaudio.functional.resample(wav, sr, 32000)
    with torch.no_grad():
        embed = model.get_query_embed(modality="audio", audio=wav.float().to(device), device=device)
    return embed.float().cpu()


def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    from scipy import linalg
    mu1, mu2 = np.atleast_1d(mu1), np.atleast_1d(mu2)
    sigma1, sigma2 = np.atleast_2d(sigma1), np.atleast_2d(sigma2)
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        covmean = linalg.sqrtm((sigma1 + np.eye(sigma1.shape[0]) * eps).dot(sigma2 + np.eye(sigma2.shape[0]) * eps))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean))


def si_sdr(pred, ref):
    ref = ref - ref.mean(dim=-1, keepdim=True)
    pred = pred - pred.mean(dim=-1, keepdim=True)
    alpha = (pred * ref).sum(dim=-1, keepdim=True) / (ref * ref).sum(dim=-1, keepdim=True).clamp(min=1e-8)
    proj = alpha * ref
    noise = pred - proj
    return 10 * torch.log10((proj ** 2).sum(dim=-1) / (noise ** 2).sum(dim=-1).clamp(min=1e-8))


def run_inference(model, batch, steps, temperature):
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.cuda()

    z, c = model.get_input(batch, model.first_stage_key, unconditional_prob_cfg=0.0)
    c = model.filter_useful_cond_dict(c)
    bs = z.shape[0]

    x_T = None
    if model.extra_channels:
        extra = model.__class__.__mro__[1].get_input(model, batch, model.extra_channel_key).to(model.device)
        extra = extra.reshape(extra.shape[0], 1, extra.shape[1], extra.shape[2])
        extra_posterior = model.encode_first_stage(extra)
        x_T = model.get_first_stage_encoding(extra_posterior).detach()

    shape = (model.channels, model.latent_t_size, model.latent_f_size)
    samples = model.solve_chen_sde(
        steps, bs, shape, c,
        unconditional_guidance_scale=1.0,
        x_T=x_T,
        temperature=temperature,
    )
    if model.extra_channels:
        samples = samples[:, :model.channels, :, :]

    mel = model.decode_first_stage(samples)
    if model.fbank_shift:
        mel = mel - model.fbank_shift
    if model.data_std:
        mel = (mel * model.data_std) + model.data_mean

    return model.mel_spectrogram_to_waveform(mel, save=False)


def sweep_temperature(model, configs, val_loader, panns_model, clap_model,
                      temperatures, infer_steps, max_samples, device="cuda"):
    sr = configs["preprocessing"]["audio"]["sampling_rate"]
    results = {t: {"fad_ref": [], "fad_pred": [], "clap_ref": [], "clap_pred": [], "si_sdr": []}
               for t in temperatures}

    batches, count = [], 0
    for batch in val_loader:
        if count >= max_samples:
            break
        batches.append(batch)
        count += min(batch["waveform"].shape[0], max_samples - count)

    for temp in temperatures:
        print(f"  temperature={temp:.1f} ...", end=" ", flush=True)
        sample_count = 0
        with torch.no_grad(), model.ema_scope():
            for batch in batches:
                bs = batch["waveform"].shape[0]
                n = min(bs, max_samples - sample_count)
                if n <= 0:
                    break

                pred_wav = run_inference(model, batch, infer_steps, temp)
                pred_t = torch.from_numpy(pred_wav).float()
                if pred_t.ndim == 3:
                    pred_t = pred_t.squeeze(1)

                ref_t = batch["waveform"].cpu().float()
                if ref_t.ndim == 3:
                    ref_t = ref_t.squeeze(1)

                min_len = min(pred_t.shape[-1], ref_t.shape[-1])
                ref_t = ref_t[..., :min_len][:n]
                pred_t = pred_t[..., :min_len][:n]

                results[temp]["fad_ref"].append(panns_embedding(panns_model, ref_t, sr, device))
                results[temp]["fad_pred"].append(panns_embedding(panns_model, pred_t, sr, device))
                results[temp]["clap_ref"].append(clap_audio_embedding(clap_model, ref_t, sr, device))
                results[temp]["clap_pred"].append(clap_audio_embedding(clap_model, pred_t, sr, device))
                results[temp]["si_sdr"].extend(si_sdr(pred_t, ref_t).tolist())

                sample_count += n

        ref_all = torch.cat(results[temp]["fad_ref"], dim=0).numpy()
        pred_all = torch.cat(results[temp]["fad_pred"], dim=0).numpy()
        fad = frechet_distance(np.mean(ref_all, 0), np.cov(ref_all, rowvar=False),
                               np.mean(pred_all, 0), np.cov(pred_all, rowvar=False))

        clap_ref = torch.cat(results[temp]["clap_ref"], dim=0)
        clap_pred = torch.cat(results[temp]["clap_pred"], dim=0)
        clapscore = torchF.cosine_similarity(clap_ref, clap_pred, dim=1).mean().item()
        mean_si_sdr = np.mean(results[temp]["si_sdr"])

        results[temp]["fad"] = fad
        results[temp]["clapscore_a"] = clapscore
        results[temp]["mean_si_sdr"] = mean_si_sdr
        print(f"SI-SDR={mean_si_sdr:.3f}  FAD={fad:.4f}  CLAP_A={clapscore:.4f}")

    return results


def print_table(results, temperatures):
    print("\n" + "=" * 70)
    print(f"{'Temp':<10} {'SI-SDR':>10} {'FAD':>10} {'CLAPScore_A':>14}")
    print("-" * 70)
    for t in temperatures:
        r = results[t]
        print(f"{t:<10.1f} {r['mean_si_sdr']:>10.3f} {r['fad']:>10.4f} {r['clapscore_a']:>14.4f}")
    print("=" * 70)


def save_plots(results, temperatures, out_dir):
    si_sdrs = [results[t]["mean_si_sdr"] for t in temperatures]
    fads = [results[t]["fad"] for t in temperatures]
    claps = [results[t]["clapscore_a"] for t in temperatures]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].plot(temperatures, si_sdrs, marker="o", color="#1565C0")
    axes[0].set_xlabel("Temperature")
    axes[0].set_ylabel("SI-SDR (dB)")
    axes[0].set_title("SI-SDR vs Temperature")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(temperatures, fads, marker="o", color="#C62828")
    axes[1].set_xlabel("Temperature")
    axes[1].set_ylabel("FAD")
    axes[1].set_title("FAD vs Temperature  (↓ better)")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(temperatures, claps, marker="o", color="#2E7D32")
    axes[2].set_xlabel("Temperature")
    axes[2].set_ylabel("CLAPScore_A")
    axes[2].set_title("CLAPScore_A vs Temperature  (↑ better)")
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    save_path = os.path.join(out_dir, "temperature_sweep.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved → {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True)
    parser.add_argument("-k", "--ckpt_path", type=str, required=True)
    parser.add_argument("--temperatures", nargs="+", type=float,
                        default=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    assert torch.cuda.is_available(), "CUDA not available"
    seed_everything(args.seed)
    torch.set_float32_matmul_precision("medium")

    out_dir = os.path.dirname(os.path.abspath(__file__))

    print("Loading model ...")
    model, configs = load_model(args.config, args.ckpt_path)
    assert getattr(model, "chen_bridge", False) and getattr(model, "chen_sampler_type", "") == "sde", \
        "This script is for chen_bridge SDE models only"

    print("Loading val data ...")
    val_loader = build_val_loader(configs)

    print("Loading metric models ...")
    panns_model = load_panns_model()
    clap_model = load_clap_model()

    print(f"\nSweeping temperature: {args.temperatures}  (steps={args.steps}, max_samples={args.max_samples})\n")
    results = sweep_temperature(model, configs, val_loader, panns_model, clap_model,
                                args.temperatures, args.steps, args.max_samples)

    print_table(results, args.temperatures)
    save_plots(results, args.temperatures, out_dir)

    result_path = os.path.join(out_dir, "temperature_result")
    with open(result_path, "w") as f:
        f.write(f"config: {args.config}\n")
        f.write(f"ckpt: {args.ckpt_path}\n")
        f.write(f"steps: {args.steps}  max_samples: {args.max_samples}\n\n")
        f.write(f"{'Temp':<10} {'SI-SDR':>10} {'FAD':>10} {'CLAPScore_A':>14}\n")
        f.write("-" * 48 + "\n")
        for t in args.temperatures:
            r = results[t]
            f.write(f"{t:<10.1f} {r['mean_si_sdr']:>10.3f} {r['fad']:>10.4f} {r['clapscore_a']:>14.4f}\n")
    print(f"Results saved → {result_path}")


if __name__ == "__main__":
    main()
