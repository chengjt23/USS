import argparse
import importlib.util
import io
import json
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
import torch.nn.functional as F
import webdataset as wds
import yaml
from scipy import linalg

SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
USS_ROOT = SCRIPT_PATH.parents[2]
WORKSPACE_ROOT = USS_ROOT.parent
DEFAULT_CONFIG = USS_ROOT / "configs" / "dacvae-bridge" / "hive-2mix-44100hz-4s" / "chen_bridge_sde_2mix_dacvae.yaml"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs"
DEFAULT_PANNS_CKPT = WORKSPACE_ROOT / "Cnn14_16k_mAP=0.438.pth"
DEFAULT_CLAP_CKPT = USS_ROOT / "metrics" / "clapscore" / "music_speech_audioset_epoch_15_esc_89.98.pt"
DEFAULT_NUM_TARS = 5
DEFAULT_METRIC_DURATION = 4.0

if str(USS_ROOT) not in sys.path:
    sys.path.insert(0, str(USS_ROOT))

from utils.tools import instantiate_from_config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_yaml", type=str, default=str(DEFAULT_CONFIG))
    parser.add_argument("--tar_path", type=str, default=None)
    parser.add_argument("--num_tars", type=int, default=DEFAULT_NUM_TARS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--panns_ckpt_path", type=str, default=str(DEFAULT_PANNS_CKPT))
    parser.add_argument("--clap_ckpt_path", type=str, default=str(DEFAULT_CLAP_CKPT))
    return parser.parse_args()


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


def choose_tars(configs, tar_path: str | None, seed: int, num_tars: int):
    if tar_path is not None:
        paths = [Path(p.strip()) for p in tar_path.split(",") if p.strip()]
        if not paths:
            raise ValueError("tar_path is empty")
        for path in paths:
            if not path.exists():
                raise FileNotFoundError(f"tar_path does not exist: {path}")
        return paths
    tar_paths = collect_val_tars(configs)
    rng = random.Random(seed)
    return rng.sample(tar_paths, min(num_tars, len(tar_paths)))


def load_audio_bytes(audio_bytes: bytes):
    wav, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    wav = np.asarray(wav, dtype=np.float32)
    tensor = torch.from_numpy(wav)
    if tensor.ndim == 2:
        tensor = tensor.mean(dim=1)
    return tensor, int(sr)


def load_s1_from_tar(tar_path: Path):
    dataset = wds.DataPipeline(
        wds.SimpleShardList([str(tar_path)]),
        wds.tarfile_to_samples(handler=wds.warn_and_continue),
    )
    waveforms = []
    sample_rates = []
    sample_keys = []
    skipped = 0
    for sample in dataset:
        source_keys = sorted(
            k for k in sample.keys()
            if isinstance(k, str) and k.startswith("s") and k.endswith(".wav")
        )
        s1_key = "s1.wav" if "s1.wav" in sample else (source_keys[1] if len(source_keys) > 1 else None)
        if s1_key is None:
            skipped += 1
            continue
        wav, sr = load_audio_bytes(sample[s1_key])
        waveforms.append(wav)
        sample_rates.append(sr)
        sample_keys.append(sample.get("__key__", str(len(sample_keys))))
    if not waveforms:
        raise RuntimeError(f"No valid s1 samples found in tar: {tar_path}")
    return waveforms, sample_rates, sample_keys, skipped


def load_s1_from_tars(tar_paths):
    all_waveforms = []
    all_sample_rates = []
    all_sample_keys = []
    per_tar = []
    skipped_total = 0
    for tar_path in tar_paths:
        waveforms, sample_rates, sample_keys, skipped = load_s1_from_tar(tar_path)
        all_waveforms.extend(waveforms)
        all_sample_rates.extend(sample_rates)
        all_sample_keys.extend(sample_keys)
        skipped_total += skipped
        per_tar.append({
            "tar_path": str(tar_path.resolve()),
            "sample_count": len(waveforms),
            "skipped_count": int(skipped),
        })
    return all_waveforms, all_sample_rates, all_sample_keys, skipped_total, per_tar


def summarize_dataset(waveforms, sample_rates):
    lengths = np.array([wav.numel() for wav in waveforms], dtype=np.int64)
    srs = np.array(sample_rates, dtype=np.int64)
    durations = lengths / srs
    return {
        "num_samples": len(waveforms),
        "sr_unique": sorted(set(sample_rates)),
        "len_unique": sorted(set(lengths.tolist())),
        "duration_mean": float(durations.mean()),
        "duration_min": float(durations.min()),
        "duration_max": float(durations.max()),
    }


def preprocess_waveform(wav: torch.Tensor, src_sr: int, dst_sr: int, target_len: int):
    wav = wav.reshape(-1)
    if src_sr != dst_sr:
        wav = torchaudio.functional.resample(wav.unsqueeze(0), src_sr, dst_sr).squeeze(0)
    if wav.numel() > target_len:
        wav = wav[:target_len]
    elif wav.numel() < target_len:
        wav = F.pad(wav, (0, target_len - wav.numel()))
    return wav.unsqueeze(0)


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
    models_module = load_module("_panns_models_dacvae_codec", fad_dir / "models.py")
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


def load_clap_model(checkpoint_path: Path, device: str):
    clap_dir = USS_ROOT / "metrics" / "clapscore"
    if str(clap_dir) not in sys.path:
        sys.path.insert(0, str(clap_dir))
    from models.clap_encoder import CLAP_Encoder
    return CLAP_Encoder(pretrained_path=str(checkpoint_path), sampling_rate=32000, device=device).eval()


def panns_embedding(model, wav, sr, device):
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    with torch.no_grad():
        out = model(wav.float().to(device), None)
    return out["embedding"].float().cpu()


def clap_audio_embedding(model, wav, sr, device):
    if sr != 32000:
        wav = torchaudio.functional.resample(wav, sr, 32000)
    with torch.no_grad():
        embed = model.get_query_embed(modality="audio", audio=wav.float().to(device), device=device)
    return embed.float().cpu()


def si_sdr(pred, ref):
    ref = ref - ref.mean(dim=-1, keepdim=True)
    pred = pred - pred.mean(dim=-1, keepdim=True)
    alpha = (pred * ref).sum(dim=-1, keepdim=True) / (ref * ref).sum(dim=-1, keepdim=True).clamp(min=1e-8)
    proj = alpha * ref
    noise = pred - proj
    return 10 * torch.log10((proj ** 2).sum(dim=-1) / (noise ** 2).sum(dim=-1).clamp(min=1e-8))


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


def summarize_values(values):
    if len(values) == 0:
        return None
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "count": int(arr.size),
    }


def try_import_pesq():
    try:
        from pesq import pesq as pesq_fn
        return pesq_fn, None
    except Exception as e:
        return None, str(e)


def try_create_dnsmos_sessions():
    try:
        from metrics.dnsmos.dnsmos_debug import PRIMARY_MODEL, P808_MODEL
        import onnxruntime as ort
        primary = ort.InferenceSession(PRIMARY_MODEL, providers=["CPUExecutionProvider"])
        p808 = ort.InferenceSession(P808_MODEL, providers=["CPUExecutionProvider"])
        return primary, p808, None
    except Exception as e:
        return None, None, str(e)


def compute_dnsmos_batch(batch_wav: torch.Tensor, sr: int, primary_sess, p808_sess):
    from metrics.dnsmos.dnsmos_debug import compute_dnsmos
    metrics = {"sig": [], "bak": [], "ovr": [], "p808_mos": []}
    for i in range(batch_wav.shape[0]):
        wav = batch_wav[i]
        if sr != 16000:
            wav = torchaudio.functional.resample(wav.unsqueeze(0), sr, 16000).squeeze(0)
        scores = compute_dnsmos(wav.numpy().astype(np.float32), primary_sess, p808_sess)
        metrics["sig"].append(float(scores["SIG"]))
        metrics["bak"].append(float(scores["BAK"]))
        metrics["ovr"].append(float(scores["OVR"]))
        metrics["p808_mos"].append(float(scores["P808_MOS"]))
    return metrics


def aggregate_dnsmos(storage, batch_metrics):
    for key, values in batch_metrics.items():
        storage[key].extend(values)


def compute_pesq_batch(ref_wav: torch.Tensor, pred_wav: torch.Tensor, sr: int, pesq_fn):
    values = []
    for i in range(ref_wav.shape[0]):
        ref = ref_wav[i]
        pred = pred_wav[i]
        if sr != 16000:
            ref = torchaudio.functional.resample(ref.unsqueeze(0), sr, 16000).squeeze(0)
            pred = torchaudio.functional.resample(pred.unsqueeze(0), sr, 16000).squeeze(0)
        try:
            values.append(float(pesq_fn(16000, ref.numpy().astype(np.float32), pred.numpy().astype(np.float32), "wb")))
        except Exception:
            values.append(float("nan"))
    return values


def crop_for_metrics(ref_wav: torch.Tensor, pred_wav: torch.Tensor, sr: int, metric_duration: float):
    metric_len = int(sr * metric_duration)
    target_len = min(ref_wav.shape[-1], pred_wav.shape[-1], metric_len)
    return ref_wav[..., :target_len], pred_wav[..., :target_len]


def instantiate_first_stage(configs, device: str):
    first_stage_config = configs["model"]["params"]["first_stage_config"]
    model = instantiate_from_config(first_stage_config).to(device).eval()
    return model


def maybe_int_attr(obj, name: str):
    value = getattr(obj, name, None)
    if value is None:
        return None
    return int(value)


def main():
    args = parse_args()
    configs = load_config(args.config_yaml)
    tar_paths = choose_tars(configs, args.tar_path, args.seed, args.num_tars)
    waveforms, sample_rates, sample_keys, skipped, per_tar = load_s1_from_tars(tar_paths)
    dataset_summary = summarize_dataset(waveforms, sample_rates)

    codec = instantiate_first_stage(configs, args.device)
    target_sr = int(codec.data_sr)
    target_len = int(codec.max_samples)
    duration = target_len / target_sr
    metric_duration = DEFAULT_METRIC_DURATION
    metric_target_len = int(metric_duration * target_sr)

    panns_model = None
    panns_error = None
    panns_ckpt = Path(args.panns_ckpt_path)
    if panns_ckpt.exists():
        try:
            panns_model = load_panns_model(panns_ckpt, args.device)
        except Exception as e:
            panns_error = str(e)
    else:
        panns_error = f"missing checkpoint: {panns_ckpt}"

    clap_model = None
    clap_error = None
    clap_ckpt = Path(args.clap_ckpt_path)
    if clap_ckpt.exists():
        try:
            clap_model = load_clap_model(clap_ckpt, args.device)
        except Exception as e:
            clap_error = str(e)
    else:
        clap_error = f"missing checkpoint: {clap_ckpt}"

    pesq_fn, pesq_error = try_import_pesq()
    dnsmos_primary, dnsmos_p808, dnsmos_error = try_create_dnsmos_sessions()

    si_sdr_values = []
    pesq_values = []
    clap_audio_cosine = []
    fad_ref_embs = []
    fad_recon_embs = []
    dnsmos_ref = {"sig": [], "bak": [], "ovr": [], "p808_mos": []}
    dnsmos_recon = {"sig": [], "bak": [], "ovr": [], "p808_mos": []}

    for start in range(0, len(waveforms), args.batch_size):
        batch_waveforms = waveforms[start:start + args.batch_size]
        batch_srs = sample_rates[start:start + args.batch_size]
        ref = torch.stack(
            [preprocess_waveform(wav, sr, target_sr, target_len) for wav, sr in zip(batch_waveforms, batch_srs)],
            dim=0,
        ).to(args.device)
        with torch.no_grad():
            recon = codec.decode(codec.encode(ref))
        ref_t = ref.detach().cpu().float().squeeze(1)
        recon_t = recon.detach().cpu().float().squeeze(1)
        ref_t, recon_t = crop_for_metrics(ref_t, recon_t, target_sr, metric_duration)

        si_sdr_values.extend(si_sdr(recon_t, ref_t).tolist())
        if pesq_fn is not None:
            pesq_values.extend(compute_pesq_batch(ref_t, recon_t, target_sr, pesq_fn))

        if panns_model is not None:
            fad_ref_embs.append(panns_embedding(panns_model, ref_t, target_sr, args.device))
            fad_recon_embs.append(panns_embedding(panns_model, recon_t, target_sr, args.device))

        if clap_model is not None:
            ref_emb = clap_audio_embedding(clap_model, ref_t, target_sr, args.device)
            recon_emb = clap_audio_embedding(clap_model, recon_t, target_sr, args.device)
            clap_audio_cosine.extend(torch.sum(F.normalize(ref_emb, dim=1) * F.normalize(recon_emb, dim=1), dim=1).tolist())

        if dnsmos_primary is not None and dnsmos_p808 is not None:
            aggregate_dnsmos(dnsmos_ref, compute_dnsmos_batch(ref_t, target_sr, dnsmos_primary, dnsmos_p808))
            aggregate_dnsmos(dnsmos_recon, compute_dnsmos_batch(recon_t, target_sr, dnsmos_primary, dnsmos_p808))

    fad_value = None
    if fad_ref_embs and fad_recon_embs:
        fad_value = frechet_distance(torch.cat(fad_ref_embs, dim=0), torch.cat(fad_recon_embs, dim=0))

    dnsmos_delta = None
    if len(dnsmos_ref["sig"]) > 0:
        dnsmos_delta = {
            key: summarize_values(np.asarray(dnsmos_recon[key]) - np.asarray(dnsmos_ref[key]))
            for key in dnsmos_ref
        }

    summary = {
        "config_yaml": str(Path(args.config_yaml).resolve()),
        "tar_paths": [str(path.resolve()) for path in tar_paths],
        "tar_count": len(tar_paths),
        "seed": int(args.seed),
        "sample_count": len(waveforms),
        "skipped_count": int(skipped),
        "per_tar": per_tar,
        "dataset_summary": dataset_summary,
        "codec": {
            "data_sr": target_sr,
            "codec_sr": int(codec.codec_sr),
            "target_len": target_len,
            "duration": float(duration),
            "metric_target_len": metric_target_len,
            "metric_duration": float(metric_duration),
            "reshape_channels": maybe_int_attr(codec, "reshape_channels"),
            "freq_dim": maybe_int_attr(codec, "freq_dim"),
            "feature_dim": int(codec.feature_dim),
        },
        "metrics": {
            "si_sdr": summarize_values(si_sdr_values),
            "pesq_wb_16k": summarize_values(pesq_values),
            "fad": fad_value,
            "clap_audio_cosine": summarize_values(clap_audio_cosine),
            "dnsmos_ref": {key: summarize_values(values) for key, values in dnsmos_ref.items()} if len(dnsmos_ref["sig"]) > 0 else None,
            "dnsmos_recon": {key: summarize_values(values) for key, values in dnsmos_recon.items()} if len(dnsmos_recon["sig"]) > 0 else None,
            "dnsmos_delta_recon_minus_ref": dnsmos_delta,
        },
        "metric_availability": {
            "pesq_wb_16k": pesq_fn is not None,
            "panns_fad": panns_model is not None,
            "clap_audio": clap_model is not None,
            "dnsmos": dnsmos_primary is not None and dnsmos_p808 is not None,
        },
        "metric_errors": {
            "pesq_wb_16k": pesq_error,
            "panns_fad": panns_error,
            "clap_audio": clap_error,
            "dnsmos": dnsmos_error,
        },
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"codec_metrics_{len(tar_paths)}tars_seed{args.seed}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"tar_count: {len(tar_paths)}")
    print(f"sample_count: {len(waveforms)}")
    print(f"codec_target_sr: {target_sr}")
    print(f"codec_duration: {duration:.4f}")
    print(f"metric_duration: {metric_duration:.4f}")
    if summary["metrics"]["si_sdr"] is not None:
        print(f"si_sdr_mean: {summary['metrics']['si_sdr']['mean']:.4f}")
    if summary["metrics"]["pesq_wb_16k"] is not None:
        print(f"pesq_wb_16k_mean: {summary['metrics']['pesq_wb_16k']['mean']:.4f}")
    if fad_value is not None:
        print(f"fad: {fad_value:.6f}")
    if summary["metrics"]["clap_audio_cosine"] is not None:
        print(f"clap_audio_cosine_mean: {summary['metrics']['clap_audio_cosine']['mean']:.6f}")
    if summary["metrics"]["dnsmos_recon"] is not None:
        print(f"dnsmos_ovr_mean: {summary['metrics']['dnsmos_recon']['ovr']['mean']:.4f}")
    print(f"output_json: {output_path}")


if __name__ == "__main__":
    main()
