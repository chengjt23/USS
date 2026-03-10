import os
import sys
import time
import numpy as np
import torch

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

    print(f"=== CLAPScore demo (CLAP, sr={sr}) ===")

    model = CLAP_Encoder(
        pretrained_path=args.pretrained_path,
        sampling_rate=sr,
        device=args.device,
    ).eval()

    rng = np.random.default_rng(42)
    captions = [f"a sound of frequency {200 + i * 50} hertz" for i in range(n_samples)]
    audios = []
    for i in range(n_samples):
        t = np.linspace(0, duration, sr * duration, endpoint=False)
        freq = 200 + i * 50
        audio = (np.sin(2 * np.pi * freq * t) * 0.5 + rng.standard_normal(sr * duration).astype(np.float32) * 0.05).astype(np.float32)
        audios.append(audio)
    audio_batch = torch.from_numpy(np.stack(audios)).to(args.device)

    n_runs = 5
    times = []
    score_values = []

    for r in range(n_runs):
        t0 = time.time()
        with torch.no_grad():
            text_embed = model.get_query_embed(modality='text', text=captions, device=args.device)
            audio_embed = model.get_query_embed(modality='audio', audio=audio_batch, device=args.device)
            scores = (text_embed * audio_embed).sum(-1)
        elapsed = time.time() - t0
        times.append(elapsed)
        mean_score = scores.mean().item()
        score_values.append(mean_score)
        print(f"  run {r+1}/{n_runs}: CLAPScore = {mean_score:.4f}, time = {elapsed:.4f}s")

    print(f"\nCLAPScore mean = {np.mean(score_values):.4f}")
    print(f"Time mean = {np.mean(times):.4f}s (over {n_runs} runs)")
