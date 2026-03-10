import os
import numpy as np
import torch
from torch import nn
from scipy import linalg
from scipy.signal import resample_poly
from math import gcd
from torch.utils.data import Dataset
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


VGGISH_URLS = {
    'vggish': 'https://github.com/harritaylor/torchvggish/releases/download/v0.1/vggish-10086976.pth',
    'pca': 'https://github.com/harritaylor/torchvggish/releases/download/v0.1/vggish_pca_params-970ea276.pth',
}


def load_vggish():
    from torchvggish.vggish import VGGish
    model = VGGish(urls=VGGISH_URLS, pretrained=True, postprocess=False)
    model.embeddings = nn.Sequential(*list(model.embeddings.children())[:-1])
    model.eval()
    return model


def get_embeddings(model, audio_dir, sr=16000):
    dataset = WaveDataset(audio_dir, sr)
    embd_list = []
    for i in range(len(dataset)):
        audio = dataset[i].numpy()
        embd = model.forward(audio, sr)
        if embd.device.type == "cuda":
            embd = embd.cpu()
        embd_list.append(embd.detach().numpy())
    return np.concatenate(embd_list, axis=0)


def calculate_statistics(embd):
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


def compute_fad(dir_a, dir_b, sr=16000):
    model = load_vggish()
    embd_a = get_embeddings(model, dir_a, sr)
    embd_b = get_embeddings(model, dir_b, sr)
    mu_a, sigma_a = calculate_statistics(embd_a)
    mu_b, sigma_b = calculate_statistics(embd_b)
    return calculate_frechet_distance(mu_a, sigma_a, mu_b, sigma_b)


if __name__ == "__main__":
    import time
    import shutil

    print("=== FAD demo with synthetic audio ===")
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
    fad_values = []
    model = load_vggish()
    embd_a = get_embeddings(model, tmp_a, sr)
    embd_b = get_embeddings(model, tmp_b, sr)

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
