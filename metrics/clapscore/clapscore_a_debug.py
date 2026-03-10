import os
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))

from models.clap_encoder import CLAP_Encoder


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    sr = 32000
    duration = 3
    n_samples = 10

    print(f"=== CLAPScore_A demo (CLAP, sr={sr}) ===")

    model = CLAP_Encoder(
        pretrained_path=args.pretrained_path,
        sampling_rate=sr,
        device=args.device,
    ).eval()

    rng = np.random.default_rng(42)
    audios_a, audios_b = [], []
    for i in range(n_samples):
        t = np.linspace(0, duration, sr * duration, endpoint=False)
        freq = 200 + i * 50
        audio_a = (np.sin(2 * np.pi * freq * t) * 0.5 + rng.standard_normal(sr * duration).astype(np.float32) * 0.05).astype(np.float32)
        audio_b = (np.sin(2 * np.pi * freq * t) * 0.5 + rng.standard_normal(sr * duration).astype(np.float32) * 0.3).astype(np.float32)
        audios_a.append(audio_a)
        audios_b.append(audio_b)
    batch_a = torch.from_numpy(np.stack(audios_a)).to(args.device)
    batch_b = torch.from_numpy(np.stack(audios_b)).to(args.device)

    n_runs = 5
    times = []
    score_values = []

    for r in range(n_runs):
        t0 = time.time()
        with torch.no_grad():
            embed_a = model.get_query_embed(modality='audio', audio=batch_a, device=args.device)
            embed_b = model.get_query_embed(modality='audio', audio=batch_b, device=args.device)
            scores = F.cosine_similarity(embed_a, embed_b, dim=1)
        elapsed = time.time() - t0
        times.append(elapsed)
        mean_score = scores.mean().item()
        score_values.append(mean_score)
        print(f"  run {r+1}/{n_runs}: CLAPScore_A = {mean_score:.4f}, time = {elapsed:.4f}s")

    print(f"\nCLAPScore_A mean = {np.mean(score_values):.4f}")
    print(f"Time mean = {np.mean(times):.4f}s (over {n_runs} runs)")
