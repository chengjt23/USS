import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "9"
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../"))

import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from data.wds_datamodule import create_wds_dataloader
from utils.audio import TacotronSTFT, get_mel_from_wav
from utils.tools import instantiate_from_config

CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../../configs/bridgesep/hive_2mix/chen_bridge_sde_2mix.yaml",
)
MAX_BATCHES = 50
BATCH_SIZE = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def pad_spec(spec, target_length):
    p = target_length - spec.shape[1]
    if p > 0:
        spec = torch.nn.ZeroPad2d((0, 0, 0, p))(spec)
    elif p < 0:
        spec = spec[:, :target_length, :]
    if spec.size(-1) % 2 != 0:
        spec = spec[..., :-1]
    return spec


def wav_feature_extraction(waveform, stft_tool):
    if waveform.dim() == 3:
        waveform = waveform.squeeze(1)
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    log_mel_specs = []
    for i in range(waveform.shape[0]):
        wav_np = waveform[i].numpy() if isinstance(waveform[i], torch.Tensor) else waveform[i]
        log_mel_spec, _, _ = get_mel_from_wav(torch.FloatTensor(wav_np), stft_tool)
        log_mel_specs.append(torch.FloatTensor(log_mel_spec.T))
    return torch.stack(log_mel_specs, dim=0)


def encode(vae, mel, device):
    x = mel.unsqueeze(1).to(device)
    with torch.no_grad():
        posterior = vae.encode(x)
    return posterior.mode()


def main():
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    stft_tool = TacotronSTFT(
        config["preprocessing"]["stft"]["filter_length"],
        config["preprocessing"]["stft"]["hop_length"],
        config["preprocessing"]["stft"]["win_length"],
        config["preprocessing"]["mel"]["n_mel_channels"],
        config["preprocessing"]["audio"]["sampling_rate"],
        config["preprocessing"]["mel"]["mel_fmin"],
        config["preprocessing"]["mel"]["mel_fmax"],
    )

    vae_cfg = config["model"]["params"]["first_stage_config"]
    vae = instantiate_from_config(vae_cfg).eval().to(DEVICE)

    dc = config["datamodule"]["data_config"]
    loader = create_wds_dataloader(
        root_dir=dc["val_dir"],
        sample_rate=dc["sample_rate"],
        batch_size=BATCH_SIZE,
        num_workers=2,
        shuffle_buffer=0,
        drop_last=False,
        persistent_workers=False,
        is_val=True,
        mix_selected=dc.get("mix_selected"),
    )

    sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
    hop_length = config["preprocessing"]["stft"]["hop_length"]
    duration = config["preprocessing"]["audio"]["duration"]
    target_length = int(duration * sampling_rate / hop_length)

    dist_mix_clean = []
    dist_noise_clean = []
    all_z_clean, all_z_mix, all_z_noise = [], [], []

    for batch_idx, batch in enumerate(loader):
        if batch_idx >= MAX_BATCHES:
            break

        mix_audio = batch["mix"]
        sources_audio = batch["sources"]

        if sources_audio.ndim == 4 and sources_audio.shape[1] > 0:
            target_audio = sources_audio[:, 0, :, :]
        else:
            continue

        if target_audio.shape[1] > 1:
            target_audio = target_audio.mean(dim=1, keepdim=True)
        mix_mono = mix_audio.mean(dim=1, keepdim=True) if mix_audio.shape[1] > 1 else mix_audio

        log_mel = wav_feature_extraction(target_audio, stft_tool)
        log_mel = pad_spec(log_mel, target_length)

        mixed_mel = wav_feature_extraction(mix_mono, stft_tool)
        mixed_mel = pad_spec(mixed_mel, target_length)

        z_clean = encode(vae, log_mel, DEVICE)
        z_mix = encode(vae, mixed_mel, DEVICE)
        z_noise = torch.randn_like(z_clean)

        B = z_clean.shape[0]
        zc_flat = z_clean.view(B, -1).cpu().float()
        zm_flat = z_mix.view(B, -1).cpu().float()
        zn_flat = z_noise.view(B, -1).cpu().float()

        dist_mix_clean.extend((zm_flat - zc_flat).norm(dim=1).tolist())
        dist_noise_clean.extend((zn_flat - zc_flat).norm(dim=1).tolist())

        all_z_clean.append(zc_flat)
        all_z_mix.append(zm_flat)
        all_z_noise.append(zn_flat)

        print(f"batch {batch_idx+1}/{MAX_BATCHES}  n={len(dist_mix_clean)}", end="\r")

    print()
    dist_mix_clean = np.array(dist_mix_clean)
    dist_noise_clean = np.array(dist_noise_clean)

    print("=" * 55)
    print(f"{'Metric':<30} {'mix→clean':>10} {'noise→clean':>12}")
    print("-" * 55)
    print(f"{'Mean L2':<30} {dist_mix_clean.mean():>10.3f} {dist_noise_clean.mean():>12.3f}")
    print(f"{'Median L2':<30} {np.median(dist_mix_clean):>10.3f} {np.median(dist_noise_clean):>12.3f}")
    print(f"{'Std L2':<30} {dist_mix_clean.std():>10.3f} {dist_noise_clean.std():>12.3f}")
    print(f"{'Min L2':<30} {dist_mix_clean.min():>10.3f} {dist_noise_clean.min():>12.3f}")
    print(f"{'Max L2':<30} {dist_mix_clean.max():>10.3f} {dist_noise_clean.max():>12.3f}")
    ratio = dist_mix_clean.mean() / dist_noise_clean.mean()
    print(f"{'Ratio (mix/noise)':<30} {ratio:>10.4f}")
    print("=" * 55)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].hist(dist_mix_clean, bins=40, alpha=0.7, label=f"mix→clean  μ={dist_mix_clean.mean():.2f}", color="#2196F3")
    axes[0].hist(dist_noise_clean, bins=40, alpha=0.7, label=f"noise→clean  μ={dist_noise_clean.mean():.2f}", color="#FF5722")
    axes[0].set_xlabel("L2 Distance in Latent Space")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Distribution of L2 Distances")
    axes[0].legend()

    axes[1].boxplot([dist_mix_clean, dist_noise_clean], labels=["mix → clean", "noise → clean"], patch_artist=True,
                    boxprops=dict(facecolor="#90CAF9"), medianprops=dict(color="black", linewidth=2))
    axes[1].set_ylabel("L2 Distance")
    axes[1].set_title("Box Plot: Latent Distances")

    z_clean_all = torch.cat(all_z_clean, dim=0).numpy()
    z_mix_all = torch.cat(all_z_mix, dim=0).numpy()
    z_noise_all = torch.cat(all_z_noise, dim=0).numpy()

    n_pca = min(500, len(z_clean_all))
    idx = np.random.choice(len(z_clean_all), n_pca, replace=False)
    all_z = np.concatenate([z_clean_all[idx], z_mix_all[idx], z_noise_all[idx]], axis=0)
    pca = PCA(n_components=2)
    z2d = pca.fit_transform(all_z)
    zc2d = z2d[:n_pca]
    zm2d = z2d[n_pca:2*n_pca]
    zn2d = z2d[2*n_pca:]

    axes[2].scatter(zn2d[:, 0], zn2d[:, 1], s=8, alpha=0.3, color="#BDBDBD", label="noise", zorder=1)
    axes[2].scatter(zm2d[:, 0], zm2d[:, 1], s=8, alpha=0.5, color="#FF9800", label="mix", zorder=2)
    axes[2].scatter(zc2d[:, 0], zc2d[:, 1], s=8, alpha=0.7, color="#1565C0", label="clean", zorder=3)
    for i in range(min(100, n_pca)):
        axes[2].plot([zc2d[i, 0], zm2d[i, 0]], [zc2d[i, 1], zm2d[i, 1]], "orange", alpha=0.15, lw=0.5)
    axes[2].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    axes[2].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    axes[2].set_title("PCA: Latent Space (clean / mix / noise)")
    axes[2].legend(markerscale=2)

    fig.suptitle(f"Latent Distance Analysis  (n={len(dist_mix_clean)}, ratio={ratio:.3f})", fontsize=13)
    plt.tight_layout()
    save_path = os.path.join(OUT_DIR, "latent_distance.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {save_path}")


if __name__ == "__main__":
    main()
