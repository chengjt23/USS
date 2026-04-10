import argparse
import importlib.util
import io
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
import torchaudio
import webdataset as wds
import yaml
from scipy import linalg
from tqdm import tqdm

SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
USS_ROOT = SCRIPT_PATH.parents[2]
WORKSPACE_ROOT = USS_ROOT.parent
DEFAULT_CONFIG = USS_ROOT / "configs" / "bridgesep" / "hive_2mix" / "flowsep_2mix.yaml"
DEFAULT_PANNS_CKPT = WORKSPACE_ROOT / "Cnn14_16k_mAP=0.438.pth"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs"
DEFAULT_LOAD_SRS = [8000, 12000, 16000, 20000, 24000, 28000, 32000, 36000, 44100, 48000]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_yaml", type=str, default=str(DEFAULT_CONFIG))
    parser.add_argument("--tar_path", type=str, default=None)
    parser.add_argument("--panns_ckpt_path", type=str, default=str(DEFAULT_PANNS_CKPT))
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--load_srs", type=str, default=",".join(str(v) for v in DEFAULT_LOAD_SRS))
    return parser.parse_args()


def parse_load_srs(text: str):
    values = sorted({int(x.strip()) for x in text.split(",") if x.strip()})
    if 16000 not in values or 44100 not in values:
        raise ValueError("load_srs must include 16000 and 44100")
    return values


def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def collect_val_tars(configs):
    data_config = configs["datamodule"]["data_config"]
    val_dir = Path(data_config["val_dir"])
    mix_selected = data_config.get("mix_selected") or []
    tar_paths = []
    if mix_selected:
        for mix_name in mix_selected:
            mix_dir = val_dir / mix_name
            if mix_dir.exists():
                tar_paths.extend(sorted(mix_dir.glob("*.tar")))
    if not tar_paths and val_dir.exists():
        tar_paths.extend(sorted(val_dir.rglob("*.tar")))
    if not tar_paths:
        raise FileNotFoundError(f"No val tar files found from config path: {val_dir}")
    return tar_paths


def choose_tar(configs, tar_path: str | None, seed: int):
    if tar_path is not None:
        path = Path(tar_path)
        if not path.exists():
            raise FileNotFoundError(f"tar_path does not exist: {path}")
        return path
    tar_paths = collect_val_tars(configs)
    rng = random.Random(seed)
    return rng.choice(tar_paths)


def load_audio_bytes(audio_bytes: bytes):
    wav, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    return torch.from_numpy(np.asarray(wav, dtype=np.float32)), int(sr)


def load_mix_and_s1_from_tar(tar_path: Path):
    dataset = wds.DataPipeline(
        wds.SimpleShardList([str(tar_path)]),
        wds.tarfile_to_samples(handler=wds.warn_and_continue),
    )
    mix_waveforms = []
    s1_waveforms = []
    mix_srs = []
    s1_srs = []
    sample_keys = []
    skipped = 0
    for sample in dataset:
        mix_bytes = sample.get("mix.wav") or sample.get("mix")
        source_keys = sorted(
            k for k in sample.keys()
            if isinstance(k, str) and k.startswith("s") and k.endswith(".wav")
        )
        s1_key = "s1.wav" if "s1.wav" in sample else (source_keys[1] if len(source_keys) > 1 else None)
        if mix_bytes is None or s1_key is None:
            skipped += 1
            continue
        mix_wav, mix_sr = load_audio_bytes(mix_bytes)
        s1_wav, s1_sr = load_audio_bytes(sample[s1_key])
        mix_waveforms.append(mix_wav)
        s1_waveforms.append(s1_wav)
        mix_srs.append(mix_sr)
        s1_srs.append(s1_sr)
        sample_keys.append(sample.get("__key__", str(len(sample_keys))))
    if not mix_waveforms:
        raise RuntimeError(f"No valid mix/s1 pairs found in tar: {tar_path}")
    return mix_waveforms, s1_waveforms, mix_srs, s1_srs, sample_keys, skipped


def summarize_dataset(mix_waveforms, s1_waveforms, mix_srs, s1_srs):
    mix_lengths = np.array([wav.numel() for wav in mix_waveforms], dtype=np.int64)
    s1_lengths = np.array([wav.numel() for wav in s1_waveforms], dtype=np.int64)
    mix_srs_arr = np.array(mix_srs, dtype=np.int64)
    s1_srs_arr = np.array(s1_srs, dtype=np.int64)
    mix_durations = mix_lengths / mix_srs_arr
    s1_durations = s1_lengths / s1_srs_arr
    return {
        "num_samples": len(mix_waveforms),
        "mix_sr_unique": sorted(set(mix_srs)),
        "s1_sr_unique": sorted(set(s1_srs)),
        "mix_len_unique": sorted(set(mix_lengths.tolist())),
        "s1_len_unique": sorted(set(s1_lengths.tolist())),
        "mix_duration_mean": float(mix_durations.mean()),
        "s1_duration_mean": float(s1_durations.mean()),
        "mix_duration_min": float(mix_durations.min()),
        "mix_duration_max": float(mix_durations.max()),
        "s1_duration_min": float(s1_durations.min()),
        "s1_duration_max": float(s1_durations.max()),
    }


def load_module(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {module_name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_panns_model(checkpoint_path: Path, device: str):
    fad_dir = USS_ROOT / "metrics" / "fad"
    load_module("pytorch_utils", fad_dir / "pytorch_utils.py")
    models_module = load_module("_panns_models_sr_sweep", fad_dir / "models.py")
    model = models_module.Cnn14_16k(
        sample_rate=16000,
        window_size=512,
        hop_size=160,
        mel_bins=64,
        fmin=50,
        fmax=8000,
        classes_num=527,
    )
    try:
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    return model


def embed_waveforms(model, waveforms, assumed_sr: int, batch_size: int, device: str, desc: str):
    embeddings = []
    for start in tqdm(range(0, len(waveforms), batch_size), desc=desc, leave=False, dynamic_ncols=True):
        batch_waveforms = waveforms[start:start + batch_size]
        if assumed_sr != 16000:
            batch_waveforms = [
                torchaudio.functional.resample(wav.unsqueeze(0), assumed_sr, 16000).squeeze(0)
                for wav in batch_waveforms
            ]
        batch = torch.nn.utils.rnn.pad_sequence(batch_waveforms, batch_first=True)
        with torch.no_grad():
            out = model(batch.float().to(device), None)
        embeddings.append(out["embedding"].float().cpu())
    return torch.cat(embeddings, dim=0)


def frechet_distance(emb_a: torch.Tensor, emb_b: torch.Tensor, eps: float = 1e-6):
    a = emb_a.float().numpy()
    b = emb_b.float().numpy()
    mu_a = np.mean(a, axis=0)
    mu_b = np.mean(b, axis=0)
    sigma_a = np.cov(a, rowvar=False)
    sigma_b = np.cov(b, rowvar=False)
    diff = mu_a - mu_b
    covmean, _ = linalg.sqrtm(sigma_a.dot(sigma_b), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma_a.shape[0]) * eps
        covmean = linalg.sqrtm((sigma_a + offset).dot(sigma_b + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(sigma_a) + np.trace(sigma_b) - 2 * np.trace(covmean))


def plot_results(load_srs, fad_values, tar_name: str, output_path: Path):
    plt.figure(figsize=(9, 5))
    plt.plot(load_srs, fad_values, marker="o", linewidth=2)
    plt.xticks(load_srs, rotation=45)
    plt.xlabel("assumed load sample rate (Hz)")
    plt.ylabel("FAD(mix, s1)")
    plt.title(f"FAD vs assumed load sample rate\n{tar_name}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main():
    args = parse_args()
    load_srs = parse_load_srs(args.load_srs)
    configs = load_config(args.config_yaml)
    tar_path = choose_tar(configs, args.tar_path, args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    panns_ckpt_path = Path(args.panns_ckpt_path)
    if not panns_ckpt_path.exists():
        raise FileNotFoundError(f"PANNs checkpoint not found: {panns_ckpt_path}")

    print(f"Selected tar: {tar_path}")
    mix_waveforms, s1_waveforms, mix_srs, s1_srs, sample_keys, skipped = load_mix_and_s1_from_tar(tar_path)
    summary = summarize_dataset(mix_waveforms, s1_waveforms, mix_srs, s1_srs)
    print(f"Samples loaded: {summary['num_samples']}")
    print(f"Skipped samples: {skipped}")
    print(f"mix sr unique: {summary['mix_sr_unique']}")
    print(f"s1 sr unique: {summary['s1_sr_unique']}")
    print(f"mix length unique: {summary['mix_len_unique'][:5]}")
    print(f"s1 length unique: {summary['s1_len_unique'][:5]}")
    print(
        f"mix duration sec mean/min/max: "
        f"{summary['mix_duration_mean']:.4f}/{summary['mix_duration_min']:.4f}/{summary['mix_duration_max']:.4f}"
    )
    print(
        f"s1 duration sec mean/min/max: "
        f"{summary['s1_duration_mean']:.4f}/{summary['s1_duration_min']:.4f}/{summary['s1_duration_max']:.4f}"
    )
    if sample_keys:
        print(f"First sample key: {sample_keys[0]}")

    device = args.device
    print(f"Loading PANNs model on {device} ...")
    panns_model = load_panns_model(panns_ckpt_path, device)

    fad_values = []
    for assumed_sr in tqdm(load_srs, desc="SR sweep", dynamic_ncols=True):
        mix_emb = embed_waveforms(
            panns_model,
            mix_waveforms,
            assumed_sr,
            args.batch_size,
            device,
            f"mix@{assumed_sr}",
        )
        s1_emb = embed_waveforms(
            panns_model,
            s1_waveforms,
            assumed_sr,
            args.batch_size,
            device,
            f"s1@{assumed_sr}",
        )
        fad = frechet_distance(mix_emb, s1_emb)
        fad_values.append(fad)
        print(f"assumed_sr={assumed_sr}: FAD={fad:.6f}")

    tar_name = tar_path.stem
    fig_path = output_dir / f"{tar_name}_fad_vs_assumed_sr.png"
    txt_path = output_dir / f"{tar_name}_fad_vs_assumed_sr.txt"
    plot_results(load_srs, fad_values, tar_name, fig_path)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"tar_path={tar_path}\n")
        f.write(f"num_samples={summary['num_samples']}\n")
        f.write(f"skipped_samples={skipped}\n")
        f.write(f"mix_sr_unique={summary['mix_sr_unique']}\n")
        f.write(f"s1_sr_unique={summary['s1_sr_unique']}\n")
        f.write(f"mix_duration_mean={summary['mix_duration_mean']:.6f}\n")
        f.write(f"s1_duration_mean={summary['s1_duration_mean']:.6f}\n")
        for sr, fad in zip(load_srs, fad_values):
            f.write(f"{sr}\t{fad:.10f}\n")

    print("Finished.")
    print(f"Figure saved to: {fig_path}")
    print(f"Results saved to: {txt_path}")


if __name__ == "__main__":
    main()
