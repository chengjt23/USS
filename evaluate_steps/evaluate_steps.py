import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

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
    ckpt = torch.load(ckpt_path, map_location="cuda")
    if "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    model.load_state_dict(ckpt, strict=False)
    return model, configs


def build_val_loader(configs):
    stft_tool = build_stft_tool(configs)
    datamodule = WDSDataModule(**configs["datamodule"]["data_config"])
    _, val_loader, _ = datamodule.make_loader
    return WrappedDataLoader(val_loader, configs, stft_tool), stft_tool


def load_panns_model(device="cuda"):
    import importlib.util
    fad_dir = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'metrics', 'fad'))
    pu_spec = importlib.util.spec_from_file_location("_panns_pytorch_utils", os.path.join(fad_dir, "pytorch_utils.py"))
    pu_mod = importlib.util.module_from_spec(pu_spec)
    sys.modules["pytorch_utils"] = pu_mod
    pu_spec.loader.exec_module(pu_mod)
    m_spec = importlib.util.spec_from_file_location("_panns_models", os.path.join(fad_dir, "models.py"))
    m_mod = importlib.util.module_from_spec(m_spec)
    m_spec.loader.exec_module(m_mod)
    model = m_mod.Cnn14_16k(
        sample_rate=16000, window_size=512, hop_size=160,
        mel_bins=64, fmin=50, fmax=8000, classes_num=527,
    )
    panns_ckpt_path = os.path.join(fad_dir, "Cnn14_16k_mAP=0.438.pth")
    ckpt = torch.load(panns_ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model


def load_clap_model(device="cuda"):
    clap_dir = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'metrics', 'clapscore'))
    if clap_dir not in sys.path:
        sys.path.insert(0, clap_dir)
    from models.clap_encoder import CLAP_Encoder
    clap_ckpt_path = os.path.join(clap_dir, "music_speech_audioset_epoch_15_esc_89.98.pt")
    model = CLAP_Encoder(pretrained_path=clap_ckpt_path, sampling_rate=32000, device=device).eval()
    return model


def panns_embedding(panns_model, wav, sr=16000, device="cuda"):
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    with torch.no_grad():
        out = panns_model(wav.float().to(device), None)
    return out["embedding"].float().cpu()


def clap_audio_embedding(clap_model, wav, sr=16000, device="cuda"):
    if sr != 32000:
        wav = torchaudio.functional.resample(wav, sr, 32000)
    with torch.no_grad():
        embed = clap_model.get_query_embed(modality='audio', audio=wav.float().to(device), device=device)
    return embed.float().cpu()


def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    from scipy import linalg
    mu1, mu2 = np.atleast_1d(mu1), np.atleast_1d(mu2)
    sigma1, sigma2 = np.atleast_2d(sigma1), np.atleast_2d(sigma2)
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean))


def run_inference(model, batch, infer_steps):
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

    samples, _ = model.sample_log(
        cond=c, batch_size=bs, x_T=x_T,
        ddim=True, ddim_steps=infer_steps,
        unconditional_guidance_scale=1.0,
    )
    if model.extra_channels:
        samples = samples[:, :model.channels, :, :]

    mel = model.decode_first_stage(samples)
    if model.fbank_shift:
        mel = mel - model.fbank_shift
    if model.data_std:
        mel = (mel * model.data_std) + model.data_mean

    pred_wav = model.mel_spectrogram_to_waveform(mel, save=False)
    return pred_wav


def evaluate_one_model(model, configs, val_loader, panns_model, clap_model, steps_list, max_samples, device="cuda"):
    sr = configs["preprocessing"]["audio"]["sampling_rate"]
    results = {s: {"fad_ref_embs": [], "fad_pred_embs": [], "clap_ref_embs": [], "clap_pred_embs": []} for s in steps_list}

    batches = []
    count = 0
    for batch in val_loader:
        if count >= max_samples:
            break
        batches.append(batch)
        bs = batch["waveform"].shape[0]
        count += min(bs, max_samples - count)

    for step in steps_list:
        print(f"    steps={step} ...", end=" ", flush=True)
        sample_count = 0
        with torch.no_grad(), model.ema_scope():
            for batch in batches:
                bs = batch["waveform"].shape[0]
                n = min(bs, max_samples - sample_count)
                if n <= 0:
                    break

                pred_wav = run_inference(model, batch, step)
                pred_t = torch.from_numpy(pred_wav).float()
                if pred_t.ndim == 3:
                    pred_t = pred_t.squeeze(1)

                ref_t = batch["waveform"].cpu().float()
                if ref_t.ndim == 3:
                    ref_t = ref_t.squeeze(1)

                min_len = min(pred_t.shape[-1], ref_t.shape[-1])
                ref_t, pred_t = ref_t[..., :min_len][:n], pred_t[..., :min_len][:n]

                results[step]["fad_ref_embs"].append(panns_embedding(panns_model, ref_t, sr, device))
                results[step]["fad_pred_embs"].append(panns_embedding(panns_model, pred_t, sr, device))
                results[step]["clap_ref_embs"].append(clap_audio_embedding(clap_model, ref_t, sr, device))
                results[step]["clap_pred_embs"].append(clap_audio_embedding(clap_model, pred_t, sr, device))

                sample_count += n

        ref_all = torch.cat(results[step]["fad_ref_embs"], dim=0).numpy()
        pred_all = torch.cat(results[step]["fad_pred_embs"], dim=0).numpy()
        fad = frechet_distance(np.mean(ref_all, 0), np.cov(ref_all, rowvar=False),
                               np.mean(pred_all, 0), np.cov(pred_all, rowvar=False))

        clap_ref = torch.cat(results[step]["clap_ref_embs"], dim=0)
        clap_pred = torch.cat(results[step]["clap_pred_embs"], dim=0)
        clapscore_a = torchF.cosine_similarity(clap_ref, clap_pred, dim=1).mean().item()

        results[step]["fad"] = fad
        results[step]["clapscore_a"] = clapscore_a
        print(f"FAD={fad:.4f}, CLAPScore_A={clapscore_a:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", required=True, help="config yaml paths for each model")
    parser.add_argument("--ckpt_paths", nargs="+", required=True, help="checkpoint paths for each model")
    parser.add_argument("--names", nargs="+", default=None, help="display names for each model")
    parser.add_argument("--max_samples", type=int, default=200)
    args = parser.parse_args()

    assert len(args.configs) == len(args.ckpt_paths), "configs and ckpt_paths must have the same length"
    if args.names is None:
        args.names = [os.path.splitext(os.path.basename(c))[0] for c in args.configs]
    assert len(args.names) == len(args.configs)

    seed_everything(0)
    steps_list = [1, 5, 10, 15, 20]
    device = "cuda"

    print("Loading metric models ...")
    panns_model = load_panns_model(device)
    clap_model = load_clap_model(device)

    all_results = {}
    for idx, (cfg_path, ckpt_path, name) in enumerate(zip(args.configs, args.ckpt_paths, args.names)):
        print(f"\n[{idx+1}/{len(args.configs)}] Model: {name}")
        print(f"  config: {cfg_path}")
        print(f"  ckpt:   {ckpt_path}")

        model, configs = load_model(cfg_path, ckpt_path)
        val_loader, _ = build_val_loader(configs)
        results = evaluate_one_model(model, configs, val_loader, panns_model, clap_model, steps_list, args.max_samples, device)
        all_results[name] = results

        del model
        torch.cuda.empty_cache()

    print("\n" + "=" * 70)
    print("Results Summary")
    print("=" * 70)
    header = f"{'Model':<30}" + "".join(f"{'step='+str(s):>12}" for s in steps_list)
    print("\n--- FAD (lower is better) ---")
    print(header)
    for name in all_results:
        row = f"{name:<30}" + "".join(f"{all_results[name][s]['fad']:>12.4f}" for s in steps_list)
        print(row)

    print("\n--- CLAPScore_A (higher is better) ---")
    print(header)
    for name in all_results:
        row = f"{name:<30}" + "".join(f"{all_results[name][s]['clapscore_a']:>12.4f}" for s in steps_list)
        print(row)

    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    for name in all_results:
        vals = [all_results[name][s]["fad"] for s in steps_list]
        ax.plot(steps_list, vals, marker="o", label=name)
    ax.set_xlabel("Inference Steps")
    ax.set_ylabel("FAD")
    ax.set_title("FAD vs Inference Steps")
    ax.set_xticks(steps_list)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fad_path = os.path.join(save_dir, "fad_vs_steps.png")
    fig.savefig(fad_path, dpi=150)
    print(f"\nFAD plot saved to {fad_path}")

    fig, ax = plt.subplots(figsize=(8, 5))
    for name in all_results:
        vals = [all_results[name][s]["clapscore_a"] for s in steps_list]
        ax.plot(steps_list, vals, marker="o", label=name)
    ax.set_xlabel("Inference Steps")
    ax.set_ylabel("CLAPScore_A")
    ax.set_title("CLAPScore_A vs Inference Steps")
    ax.set_xticks(steps_list)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    clap_path = os.path.join(save_dir, "clapscore_a_vs_steps.png")
    fig.savefig(clap_path, dpi=150)
    print(f"CLAPScore_A plot saved to {clap_path}")


if __name__ == "__main__":
    main()
