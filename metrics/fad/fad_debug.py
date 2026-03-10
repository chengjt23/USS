import os
import sys
import numpy as np
import torch
from scipy import linalg
from scipy.signal import resample_poly
from math import gcd
from torch.utils.data import Dataset, DataLoader
import soundfile as sf

sys.path.insert(0, os.path.dirname(__file__))


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


def load_cnn14(checkpoint_path, sr=16000, device="cpu"):
    from models import Cnn14, Cnn14_16k

    if sr == 16000:
        model = Cnn14_16k(
            sample_rate=16000, window_size=512, hop_size=160,
            mel_bins=64, fmin=50, fmax=8000, classes_num=527,
        )
    elif sr == 32000:
        model = Cnn14(
            sample_rate=32000, window_size=1024, hop_size=320,
            mel_bins=64, fmin=50, fmax=14000, classes_num=527,
        )
    else:
        raise ValueError(f"Cnn14 only supports 16000/32000 Hz, got {sr}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    return model


def get_embeddings(model, audio_dir, sr=16000, device="cpu"):
    dataset = WaveDataset(audio_dir, sr)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    embd_list = []
    with torch.no_grad():
        for wav in loader:
            wav = wav.float().to(device)
            out = model(wav, None)
            embd_list.append(out["embedding"].cpu())
    return torch.cat(embd_list, dim=0)


def calculate_statistics(embd):
    if torch.is_tensor(embd):
        embd = embd.numpy()
    mu = np.mean(embd, axis=0)
    sigma = np.cov(embd, rowvar=False)
    return mu, sigma


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
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


def compute_fad(dir_a, dir_b, checkpoint_path, sr=16000, device="cpu"):
    model = load_cnn14(checkpoint_path, sr, device)
    embd_a = get_embeddings(model, dir_a, sr, device)
    embd_b = get_embeddings(model, dir_b, sr, device)
    mu_a, sigma_a = calculate_statistics(embd_a)
    mu_b, sigma_b = calculate_statistics(embd_b)
    return calculate_frechet_distance(mu_a, sigma_a, mu_b, sigma_b)


if __name__ == "__main__":
    import time
    import shutil
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    print(f"=== FAD demo (PANNs Cnn14, 2048-dim, sr={args.sr}) ===")
    tmp_a = os.path.join(os.path.dirname(__file__), "_tmp_a")
    tmp_b = os.path.join(os.path.dirname(__file__), "_tmp_b")
    os.makedirs(tmp_a, exist_ok=True)
    os.makedirs(tmp_b, exist_ok=True)

    rng = np.random.default_rng(42)
    sr = args.sr
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
    fad_values = []
    model = load_cnn14(args.checkpoint_path, sr, args.device)
    embd_a = get_embeddings(model, tmp_a, sr, args.device)
    embd_b = get_embeddings(model, tmp_b, sr, args.device)
    print(f"Embedding shape: {embd_a.shape}")

    for r in range(n_runs):
        t0 = time.time()
        mu_a, sigma_a = calculate_statistics(embd_a)
        mu_b, sigma_b = calculate_statistics(embd_b)
        fad = calculate_frechet_distance(mu_a, sigma_a, mu_b, sigma_b)
        elapsed = time.time() - t0
        times.append(elapsed)
        fad_values.append(fad)
        print(f"  run {r+1}/{n_runs}: FAD = {fad:.4f}, time = {elapsed:.4f}s")

    print(f"\nFAD mean = {np.mean(fad_values):.4f}")
    print(f"Time mean = {np.mean(times):.4f}s (over {n_runs} runs)")

    shutil.rmtree(tmp_a)
    shutil.rmtree(tmp_b)
    print("Temp dirs cleaned.")
