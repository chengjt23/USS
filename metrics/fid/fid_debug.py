import os
import numpy as np
import torch
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


def get_embeddings(model, audio_dir, sr=16000, device="cpu"):
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


def calculate_fid(features_1, features_2, eps=1e-6):
    f1 = features_1.numpy() if torch.is_tensor(features_1) else features_1
    f2 = features_2.numpy() if torch.is_tensor(features_2) else features_2

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


def compute_fid(dir_a, dir_b, sr=16000, device="cpu"):
    model = load_cnn14(sr)
    feat_a = get_embeddings(model, dir_a, sr, device)
    feat_b = get_embeddings(model, dir_b, sr, device)
    return calculate_fid(feat_a, feat_b)


if __name__ == "__main__":
    import time
    import shutil

    print("=== FID demo with synthetic audio ===")
    tmp_a = os.path.join(os.path.dirname(__file__), "_tmp_a")
    tmp_b = os.path.join(os.path.dirname(__file__), "_tmp_b")
    os.makedirs(tmp_a, exist_ok=True)
    os.makedirs(tmp_b, exist_ok=True)

    rng = np.random.default_rng(42)
    sr = 16000
    duration = 3
    n_samples = 10

    for i in range(n_samples):
        t = np.linspace(0, duration, sr * duration, endpoint=False)
        freq = 200 + i * 50
        audio_a = (np.sin(2 * np.pi * freq * t) * 0.5 + rng.standard_normal(sr * duration).astype(np.float32) * 0.05).astype(np.float32)
        audio_b = (np.sin(2 * np.pi * freq * t) * 0.5 + rng.standard_normal(sr * duration).astype(np.float32) * 0.3).astype(np.float32)
        sf.write(os.path.join(tmp_a, f"{i}.wav"), audio_a, sr)
        sf.write(os.path.join(tmp_b, f"{i}.wav"), audio_b, sr)

    n_runs = 5
    times = []
    fid_values = []
    model = load_cnn14(sr)
    feat_a = get_embeddings(model, tmp_a, sr)
    feat_b = get_embeddings(model, tmp_b, sr)

    for r in range(n_runs):
        t0 = time.time()
        fid = calculate_fid(feat_a, feat_b)
        elapsed = time.time() - t0
        times.append(elapsed)
        fid_values.append(fid)
        print(f"  run {r+1}/{n_runs}: FID = {fid:.4f}, time = {elapsed:.4f}s")

    print(f"\nFID mean = {np.mean(fid_values):.4f}")
    print(f"Time mean = {np.mean(times):.4f}s (over {n_runs} runs)")

    shutil.rmtree(tmp_a)
    shutil.rmtree(tmp_b)
    print("Temp dirs cleaned.")
