import os
import numpy as np
import torch
from torch import nn
from scipy import linalg
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


def load_vggish():
    model = torch.hub.load("harritaylor/torchvggish", "vggish")
    model.postprocess = False
    model.embeddings = nn.Sequential(*list(model.embeddings.children())[:-1])
    model.eval()
    return model


def get_embeddings_vggish(model, audio_dir, sr=16000):
    dataset = WaveDataset(audio_dir, sr)
    embd_list = []
    for i in range(len(dataset)):
        audio = dataset[i].numpy()
        embd = model.forward(audio, sr)
        if embd.device.type == "cuda":
            embd = embd.cpu()
        embd_list.append(embd.detach().numpy())
    return np.concatenate(embd_list, axis=0)


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
        embd_list.append(log_mel.mean(dim=-1).squeeze(0).numpy())
    return np.stack(embd_list, axis=0)


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


def compute_fad(ref_dir, gen_dir, use_vggish=False, sr=16000):
    if use_vggish:
        model = load_vggish()
        embd_ref = get_embeddings_vggish(model, ref_dir, sr)
        embd_gen = get_embeddings_vggish(model, gen_dir, sr)
    else:
        embd_ref = get_embeddings_mel(ref_dir, sr)
        embd_gen = get_embeddings_mel(gen_dir, sr)

    mu_ref, sigma_ref = calculate_statistics(embd_ref)
    mu_gen, sigma_gen = calculate_statistics(embd_gen)
    return calculate_frechet_distance(mu_ref, sigma_ref, mu_gen, sigma_gen)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ref_dir", type=str, default=None)
    parser.add_argument("--gen_dir", type=str, default=None)
    parser.add_argument("--use_vggish", action="store_true")
    parser.add_argument("--sr", type=int, default=16000)
    args = parser.parse_args()

    if args.ref_dir and args.gen_dir:
        fad = compute_fad(args.ref_dir, args.gen_dir, args.use_vggish, args.sr)
        print(f"FAD = {fad:.4f}")
    else:
        print("=== FAD demo with synthetic audio ===")
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

        fad = compute_fad(tmp_ref, tmp_gen, use_vggish=False, sr=sr)
        print(f"FAD (mel-based) = {fad:.4f}")

        import shutil
        shutil.rmtree(tmp_ref)
        shutil.rmtree(tmp_gen)
        print("Temp dirs cleaned.")
