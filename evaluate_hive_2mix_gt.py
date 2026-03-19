"""
Evaluate the upper-bound metrics (FAD, CLAPScore, CLAPScore_A) caused by
the compression / decompression pipeline alone (STFT -> mel -> VAE enc -> VAE dec -> BigVGAN),
without any separation model inference.

Usage:
    python evaluate_hive_2mix_gt.py \
        -c configs/bridgesep/hive_2mix/bridge_sb_ei_2mix.yaml \
        --panns_ckpt_path  metrics/fad/Cnn14_16k_mAP=0.438.pth \
        --clap_ckpt_path   metrics/clapscore/music_speech_audioset_epoch_15_esc_89.98.pt \
        -n 200
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import importlib.util
import yaml
import torch
import torch.nn.functional as torchF
import torchaudio
import numpy as np
from scipy import linalg
from pytorch_lightning import seed_everything

from utils.tools import instantiate_from_config
from utils.audio import TacotronSTFT, get_mel_from_wav
from data.wds_datamodule import WDSDataModule
from train import WrappedDataLoader, build_stft_tool


# ---------------------------------------------------------------------------
# Metric helpers  (same logic as in model.py / fad_debug.py / clapscore*)
# ---------------------------------------------------------------------------

def _load_panns_model(panns_ckpt_path, device):
    fad_dir = os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'metrics', 'fad'))
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
    ckpt = torch.load(panns_ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model


def _panns_embedding(panns_model, wav, sr, device):
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    with torch.no_grad():
        out = panns_model(wav.float().to(device), None)
    return out["embedding"].float().cpu()


def _load_clap_model(clap_ckpt_path, device):
    clap_dir = os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'metrics', 'clapscore'))
    if clap_dir not in sys.path:
        sys.path.insert(0, clap_dir)
    from models.clap_encoder import CLAP_Encoder
    model = CLAP_Encoder(
        pretrained_path=clap_ckpt_path,
        sampling_rate=32000,
        device=device,
    ).eval()
    return model


def _clap_audio_embedding(clap_model, wav, sr, device):
    if sr != 32000:
        wav = torchaudio.functional.resample(wav, sr, 32000)
    with torch.no_grad():
        embed = clap_model.get_query_embed(modality='audio', audio=wav.float().to(device), device=device)
    return embed.float().cpu()


def _clap_text_embedding(clap_model, text_list, device):
    with torch.no_grad():
        embed = clap_model.get_query_embed(modality='text', text=text_list, device=device)
    return embed.cpu()


def _frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate upper-bound metrics of the VAE+BigVGAN compression pipeline on GT audio.")
    parser.add_argument("-c", "--config_yaml", type=str, required=True,
                        help="Path to the model config yaml")
    parser.add_argument("-n", "--max_samples", type=int, default=200,
                        help="Max number of samples to evaluate")
    parser.add_argument("--panns_ckpt_path", type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                             "metrics", "fad", "Cnn14_16k_mAP=0.438.pth"))
    parser.add_argument("--clap_ckpt_path", type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                             "metrics", "clapscore",
                                             "music_speech_audioset_epoch_15_esc_89.98.pt"))
    args = parser.parse_args()

    seed_everything(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(args.config_yaml, "r") as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)

    # ---- build model (only VAE + vocoder are needed) ----
    print("Instantiating model (VAE + BigVGAN) ...")
    model = instantiate_from_config(configs["model"]).to(device).eval()

    # ---- build STFT tool & data loader ----
    stft_tool = build_stft_tool(configs)
    sampling_rate = configs["preprocessing"]["audio"]["sampling_rate"]
    hopsize = configs["preprocessing"]["stft"]["hop_length"]
    duration = configs["preprocessing"]["audio"]["duration"]
    target_length = int(duration * sampling_rate / hopsize)

    datamodule = WDSDataModule(**configs["datamodule"]["data_config"])
    _, val_loader, _ = datamodule.make_loader
    val_loader = WrappedDataLoader(val_loader, configs, stft_tool)

    # ---- load metric models ----
    print("Loading PANNs for FAD ...")
    panns_model = _load_panns_model(args.panns_ckpt_path, device)
    print("Loading CLAP for CLAPScore ...")
    clap_model = _load_clap_model(args.clap_ckpt_path, device)

    # ---- accumulators ----
    fad_ref_embs, fad_pred_embs = [], []
    clap_ref_audio_embs, clap_pred_audio_embs = [], []
    clap_text_embs = []
    count = 0

    print(f"Starting evaluation (max {args.max_samples} samples) ...")
    with torch.no_grad():
        for batch in val_loader:
            if count >= args.max_samples:
                break

            # move tensors to device
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            # --- GT target mel (same as model.get_input for first_stage_key="fbank") ---
            fbank = batch["log_mel_spec"]                          # (B, T, n_mel)
            fbank = fbank.unsqueeze(1).contiguous().float()        # (B, 1, T, n_mel)

            # --- VAE encode ---
            encoder_posterior = model.encode_first_stage(fbank)
            z = model.get_first_stage_encoding(encoder_posterior).detach()
            # z shape: (B, C, latent_T, latent_F)

            # --- VAE decode ---
            mel_recon = model.decode_first_stage(z)                # (B, 1, T, n_mel)

            # --- undo fbank_shift / data_std if they were applied ---
            if model.fbank_shift:
                mel_recon = mel_recon - model.fbank_shift
            if model.data_std:
                mel_recon = (mel_recon * model.data_std) + model.data_mean

            # --- BigVGAN vocoder ---
            pred_wav = model.mel_spectrogram_to_waveform(mel_recon, save=False)
            # pred_wav: numpy (B, samples) or (B, 1, samples)

            # --- reference waveform ---
            ref_wav = batch["waveform"].cpu().float()
            if ref_wav.ndim == 3:
                ref_wav = ref_wav.squeeze(1)                       # (B, samples)

            bs = fbank.shape[0]
            actual = min(bs, args.max_samples - count)

            # --- metric embeddings ---
            pred_tensor = torch.from_numpy(pred_wav[:actual]).float()
            if pred_tensor.ndim == 3:
                pred_tensor = pred_tensor.squeeze(1)
            ref_tensor = ref_wav[:actual]

            min_len = min(pred_tensor.shape[-1], ref_tensor.shape[-1])
            pred_tensor = pred_tensor[..., :min_len]
            ref_tensor = ref_tensor[..., :min_len]

            # FAD embeddings
            fad_ref_embs.append(_panns_embedding(panns_model, ref_tensor, sampling_rate, device))
            fad_pred_embs.append(_panns_embedding(panns_model, pred_tensor, sampling_rate, device))

            # CLAP audio embeddings
            clap_ref_audio_embs.append(_clap_audio_embedding(clap_model, ref_tensor, sampling_rate, device))
            clap_pred_audio_embs.append(_clap_audio_embedding(clap_model, pred_tensor, sampling_rate, device))

            # CLAP text embeddings
            captions = batch.get("caption", batch.get("text", []))
            if captions and any(c != "" for c in captions):
                clap_text_embs.append(_clap_text_embedding(clap_model, list(captions[:actual]), device))

            count += actual
            print(f"  [{count}/{args.max_samples}] processed")

    # ---- compute FAD ----
    ref_all = torch.cat(fad_ref_embs, dim=0).float().numpy()
    pred_all = torch.cat(fad_pred_embs, dim=0).float().numpy()
    mu_r, sigma_r = np.mean(ref_all, axis=0), np.cov(ref_all, rowvar=False)
    mu_p, sigma_p = np.mean(pred_all, axis=0), np.cov(pred_all, rowvar=False)
    fad = _frechet_distance(mu_r, sigma_r, mu_p, sigma_p)

    # ---- compute CLAPScore_A  (cosine sim between audio embeddings) ----
    pred_audio_all = torch.cat(clap_pred_audio_embs, dim=0)
    ref_audio_all = torch.cat(clap_ref_audio_embs, dim=0)
    clapscore_a = torchF.cosine_similarity(pred_audio_all, ref_audio_all, dim=1).mean().item()

    # ---- compute CLAPScore  (text . pred_audio) ----
    clapscore = float("nan")
    if len(clap_text_embs) > 0:
        text_all = torch.cat(clap_text_embs, dim=0)
        min_n = min(text_all.shape[0], pred_audio_all.shape[0])
        clapscore = (text_all[:min_n] * pred_audio_all[:min_n]).sum(-1).mean().item()

    # ---- print results ----
    print("\n" + "=" * 50)
    print("  GT -> STFT/mel -> VAE enc -> VAE dec -> BigVGAN")
    print("  (Upper-bound metrics of the compression pipeline)")
    print("=" * 50)
    print(f"  Samples evaluated : {count}")
    print(f"  FAD               : {fad:.4f}")
    print(f"  CLAPScore         : {clapscore:.4f}")
    print(f"  CLAPScore_A       : {clapscore_a:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()