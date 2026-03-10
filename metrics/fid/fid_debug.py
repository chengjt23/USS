import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.linalg
from scipy.signal import resample_poly
from math import gcd
from torch.utils.data import Dataset, DataLoader
import soundfile as sf


class WaveDataset(Dataset):
    def __init__(self, audio_dir, sr=16000):
        self.sr = sr
        self.files = sorted([
            os.path.join(audio_dir, f)
            for f in os.listdir(audio_dir)
            if f.endswith((".wav", ".flac", ".mp3"))
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        wav, orig_sr = sf.read(self.files[idx], dtype="float32")
        if wav.ndim > 1:
            wav = wav[:, 0]
        if orig_sr != self.sr:
            g = gcd(orig_sr, self.sr)
            wav = resample_poly(wav, self.sr // g, orig_sr // g).astype(np.float32)
        return torch.from_numpy(wav)


def load_cnn14(sr=16000):
    from audioldm_eval.feature_extractors.panns import Cnn14
    if sr == 16000:
        model = Cnn14(
            features_list=["2048", "logits"],
            sample_rate=16000, window_size=512, hop_size=160,
            mel_bins=64, fmin=50, fmax=8000, classes_num=527,
        )
    elif sr == 32000:
        model = Cnn14(
            features_list=["2048", "logits"],
            sample_rate=32000, window_size=1024, hop_size=320,
            mel_bins=64, fmin=50, fmax=14000, classes_num=527,
        )
    else:
        raise ValueError(f"Cnn14 only supports 16000/32000 Hz, got {sr}")
    model.eval()
    return model


def get_embeddings_cnn14(model, audio_dir, sr=16000, device="cpu"):
    dataset = WaveDataset(audio_dir, sr)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    model = model.to(device)
    embd_list = []
    with torch.no_grad():
        for wav in loader:
            wav = wav.float().to(device)
            out = model(wav)
            embd_list.append(out["2048"].cpu())
    return torch.cat(embd_list, dim=0)


def get_embeddings_mel(audio_dir, sr=16000, n_mels=128, n_fft=1024, hop=160):
    import torchaudio
    dataset = WaveDataset(audio_dir, sr)
    mel_fn = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr, n_fft=n_fft, hop_length=hop, n_mels=n_mels
    )
    embd_list = []
    for i in range(len(dataset)):
        wav = dataset[i].unsqueeze(0)
        mel = mel_fn(wav)
        log_mel = torch.log(mel.clamp(min=1e-7))
        embd_list.append(log_mel.mean(dim=-1).squeeze(0))
    return torch.stack(embd_list, dim=0)


def calculate_fid(features_1, features_2, eps=1e-6):
    assert features_1.dim() == 2 and features_2.dim() == 2

    f1 = features_1.numpy()
    f2 = features_2.numpy()

    mu1 = np.mean(f1, axis=0)
    sigma1 = np.cov(f1, rowvar=False)
    mu2 = np.mean(f2, axis=0)
    sigma2 = np.cov(f2, rowvar=False)

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    diff = mu1 - mu2
    covmean, _ = scipy.linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        print(f"WARNING: FID produces singular product; adding {eps} to diagonal")
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = scipy.linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f"Imaginary component {m}")
        covmean = covmean.real
    tr_covmean = np.trace(covmean)
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)


def compute_fid(ref_dir, gen_dir, use_cnn14=False, sr=16000, device="cpu"):
    if use_cnn14:
        model = load_cnn14(sr)
        feat_ref = get_embeddings_cnn14(model, ref_dir, sr, device)
        feat_gen = get_embeddings_cnn14(model, gen_dir, sr, device)
    else:
        feat_ref = get_embeddings_mel(ref_dir, sr)
        feat_gen = get_embeddings_mel(gen_dir, sr)
    return calculate_fid(feat_ref, feat_gen)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ref_dir", type=str, default=None)
    parser.add_argument("--gen_dir", type=str, default=None)
    parser.add_argument("--use_cnn14", action="store_true")
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    if args.ref_dir and args.gen_dir:
        fid = compute_fid(args.ref_dir, args.gen_dir, args.use_cnn14, args.sr, args.device)
        print(f"FID = {fid:.4f}")
    else:
        print("=== FID demo with synthetic audio ===")
        tmp_ref = os.path.join(os.path.dirname(__file__), "_tmp_ref")
        tmp_gen = os.path.join(os.path.dirname(__file__), "_tmp_gen")
        os.makedirs(tmp_ref, exist_ok=True)
        os.makedirs(tmp_gen, exist_ok=True)

        rng = np.random.default_rng(42)
        sr = 16000
        duration = 3
        n_samples = 10

        for i in range(n_samples):
            t = np.linspace(0, duration, sr * duration, endpoint=False)
            freq = 200 + i * 50
            ref = (np.sin(2 * np.pi * freq * t) * 0.5 + rng.standard_normal(sr * duration).astype(np.float32) * 0.05).astype(np.float32)
            gen = (np.sin(2 * np.pi * freq * t) * 0.5 + rng.standard_normal(sr * duration).astype(np.float32) * 0.3).astype(np.float32)
            sf.write(os.path.join(tmp_ref, f"{i}.wav"), ref, sr)
            sf.write(os.path.join(tmp_gen, f"{i}.wav"), gen, sr)

        fid = compute_fid(tmp_ref, tmp_gen, use_cnn14=False, sr=sr)
        print(f"FID (mel-based) = {fid:.4f}")

        import shutil
        shutil.rmtree(tmp_ref)
        shutil.rmtree(tmp_gen)
        print("Temp dirs cleaned.")
