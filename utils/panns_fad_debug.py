import torch
import numpy as np
from scipy.linalg import sqrtm
from panns_inference import AudioTagging


def extract_panns_embeddings(waveforms, sr=16000, device="cuda"):
    at = AudioTagging(checkpoint_path=None, device=device)
    if sr != 32000:
        waveforms = torch.nn.functional.interpolate(
            waveforms.unsqueeze(1), scale_factor=32000 / sr, mode="linear"
        ).squeeze(1)
    embeddings = []
    for i in range(waveforms.shape[0]):
        wav = waveforms[i].unsqueeze(0).numpy()
        _, emb = at.inference(wav)
        embeddings.append(torch.from_numpy(emb).squeeze(0))
    return torch.stack(embeddings, dim=0)


def frechet_distance(emb1, emb2, eps=1e-6):
    mu1, mu2 = emb1.mean(0).numpy(), emb2.mean(0).numpy()
    sigma1 = np.cov(emb1.numpy(), rowvar=False) + eps * np.eye(emb1.shape[1])
    sigma2 = np.cov(emb2.numpy(), rowvar=False) + eps * np.eye(emb2.shape[1])
    diff = mu1 - mu2
    covmean, _ = sqrtm(sigma1 @ sigma2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean))


if __name__ == "__main__":
    n_samples = 20
    sr = 16000
    duration = 3.0
    length = int(sr * duration)

    ref_wavs = torch.randn(n_samples, length) * 0.1
    pred_wavs = torch.randn(n_samples, length) * 0.1

    print(f"Extracting PANNs embeddings for {n_samples} ref samples...")
    ref_emb = extract_panns_embeddings(ref_wavs, sr=sr)
    print(f"  ref embedding shape: {ref_emb.shape}")

    print(f"Extracting PANNs embeddings for {n_samples} pred samples...")
    pred_emb = extract_panns_embeddings(pred_wavs, sr=sr)
    print(f"  pred embedding shape: {pred_emb.shape}")

    fad = frechet_distance(ref_emb, pred_emb)
    print(f"\nFAD (random vs random): {fad:.4f}")

    pred_similar = ref_wavs + torch.randn_like(ref_wavs) * 0.01
    print(f"\nExtracting PANNs embeddings for similar samples...")
    similar_emb = extract_panns_embeddings(pred_similar, sr=sr)
    fad_similar = frechet_distance(ref_emb, similar_emb)
    print(f"FAD (ref vs similar):   {fad_similar:.4f}")
    print(f"FAD (ref vs random):    {fad:.4f}")
