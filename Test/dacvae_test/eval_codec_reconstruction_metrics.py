import argparse
from concurrent.futures import ThreadPoolExecutor
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
from tqdm import tqdm

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
DEFAULT_PESQ_WORKERS = 4

_RESAMPLER_CACHE = {}

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
    parser.add_argument("--pesq_workers", type=int, default=DEFAULT_PESQ_WORKERS)
    parser.add_argument("--skip_dnsmos", action="store_true")
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


def load_target_and_mix_from_tar(tar_path: Path):
    dataset = wds.DataPipeline(
        wds.SimpleShardList([str(tar_path)]),
        wds.tarfile_to_samples(handler=wds.warn_and_continue),
    )
    target_waveforms = []
    target_sample_rates = []
    mix_waveforms = []
    mix_sample_rates = []
    sample_keys = []
    skipped = 0
    for sample in dataset:
        source_keys = sorted(
            k for k in sample.keys()
            if isinstance(k, str) and k.startswith("s") and k.endswith(".wav")
        )
        target_key = "s1.wav" if "s1.wav" in sample else (source_keys[1] if len(source_keys) > 1 else None)
        mix_key = "mix.wav" if "mix.wav" in sample else None
        if target_key is None or mix_key is None:
            skipped += 1
            continue
        target_wav, target_sr = load_audio_bytes(sample[target_key])
        mix_wav, mix_sr = load_audio_bytes(sample[mix_key])
        target_waveforms.append(target_wav)
        target_sample_rates.append(target_sr)
        mix_waveforms.append(mix_wav)
        mix_sample_rates.append(mix_sr)
        sample_keys.append(sample.get("__key__", str(len(sample_keys))))
    if not target_waveforms:
        raise RuntimeError(f"No valid target and mix samples found in tar: {tar_path}")
    return target_waveforms, target_sample_rates, mix_waveforms, mix_sample_rates, sample_keys, skipped


def load_target_and_mix_from_tars(tar_paths):
    all_target_waveforms = []
    all_target_sample_rates = []
    all_mix_waveforms = []
    all_mix_sample_rates = []
    all_sample_keys = []
    per_tar = []
    skipped_total = 0
    for tar_path in tqdm(tar_paths, desc="Loading tars", unit="tar", dynamic_ncols=True):
        target_waveforms, target_sample_rates, mix_waveforms, mix_sample_rates, sample_keys, skipped = load_target_and_mix_from_tar(tar_path)
        all_target_waveforms.extend(target_waveforms)
        all_target_sample_rates.extend(target_sample_rates)
        all_mix_waveforms.extend(mix_waveforms)
        all_mix_sample_rates.extend(mix_sample_rates)
        all_sample_keys.extend(sample_keys)
        skipped_total += skipped
        per_tar.append({
            "tar_path": str(tar_path.resolve()),
            "sample_count": len(target_waveforms),
            "skipped_count": int(skipped),
        })
    return all_target_waveforms, all_target_sample_rates, all_mix_waveforms, all_mix_sample_rates, all_sample_keys, skipped_total, per_tar


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


def resample_audio(wav: torch.Tensor, src_sr: int, dst_sr: int):
    if src_sr == dst_sr:
        return wav
    key = (int(src_sr), int(dst_sr), str(wav.device))
    resampler = _RESAMPLER_CACHE.get(key)
    if resampler is None:
        resampler = torchaudio.transforms.Resample(orig_freq=src_sr, new_freq=dst_sr).to(wav.device)
        _RESAMPLER_CACHE[key] = resampler
    return resampler(wav.float())


def preprocess_waveform(wav: torch.Tensor, src_sr: int, dst_sr: int, target_len: int):
    wav = wav.reshape(-1)
    if src_sr != dst_sr:
        wav = resample_audio(wav.unsqueeze(0), src_sr, dst_sr).squeeze(0)
    if wav.numel() > target_len:
        wav = wav[:target_len]
    elif wav.numel() < target_len:
        wav = F.pad(wav, (0, target_len - wav.numel()))
    return wav.unsqueeze(0)


def preprocess_waveforms_batch(waveforms, sample_rates, dst_sr: int, target_len: int):
    if not waveforms:
        return torch.empty(0, 1, target_len)
    if len(set(sample_rates)) == 1 and len({wav.numel() for wav in waveforms}) == 1:
        batch = torch.stack([wav.reshape(-1) for wav in waveforms], dim=0)
        batch = resample_audio(batch, sample_rates[0], dst_sr)
        if batch.shape[-1] > target_len:
            batch = batch[..., :target_len]
        elif batch.shape[-1] < target_len:
            batch = F.pad(batch, (0, target_len - batch.shape[-1]))
        return batch.unsqueeze(1)
    return torch.stack(
        [preprocess_waveform(wav, sr, dst_sr, target_len) for wav, sr in zip(waveforms, sample_rates)],
        dim=0,
    )


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
        wav = resample_audio(wav, sr, 16000)
    with torch.no_grad():
        out = model(wav.float().to(device), None)
    return out["embedding"].float().cpu()


def clap_audio_embedding(model, wav, sr, device):
    if sr != 32000:
        wav = resample_audio(wav, sr, 32000)
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


def _repeat_to_length(batch_np: np.ndarray, target_len: int):
    if batch_np.shape[1] >= target_len:
        return batch_np[:, :target_len]
    repeat_count = (target_len + batch_np.shape[1] - 1) // batch_np.shape[1]
    return np.tile(batch_np, (1, repeat_count))[:, :target_len]


def compute_dnsmos_batch(batch_wav: torch.Tensor, sr: int, primary_sess, p808_sess):
    from metrics.dnsmos.dnsmos_debug import INPUT_LENGTH, SR, audio_melspec, compute_dnsmos, get_polyfit_val
    if sr != SR:
        batch_wav = resample_audio(batch_wav, sr, SR)
        sr = SR
    batch_np = batch_wav.cpu().numpy().astype(np.float32)
    batch_np = _repeat_to_length(batch_np, int(INPUT_LENGTH * sr))
    metrics = {"sig": [], "bak": [], "ovr": [], "p808_mos": []}
    primary_out = None
    primary_input_name = primary_sess.get_inputs()[0].name
    try:
        primary_out = primary_sess.run(None, {primary_input_name: batch_np})[0]
        if not isinstance(primary_out, np.ndarray) or primary_out.shape[0] != batch_np.shape[0]:
            primary_out = None
    except Exception:
        primary_out = None
    if primary_out is not None:
        primary_scores = np.asarray([get_polyfit_val(row[0], row[1], row[2]) for row in primary_out], dtype=np.float32)
        metrics["sig"].extend(primary_scores[:, 0].astype(np.float64).tolist())
        metrics["bak"].extend(primary_scores[:, 1].astype(np.float64).tolist())
        metrics["ovr"].extend(primary_scores[:, 2].astype(np.float64).tolist())
    p808_input = np.stack([np.asarray(audio_melspec(audio=audio[:-160]), dtype=np.float32) for audio in batch_np], axis=0)
    p808_out = None
    p808_input_name = p808_sess.get_inputs()[0].name
    try:
        p808_out = p808_sess.run(None, {p808_input_name: p808_input})[0]
        if not isinstance(p808_out, np.ndarray) or p808_out.shape[0] != batch_np.shape[0]:
            p808_out = None
    except Exception:
        p808_out = None
    if p808_out is not None:
        metrics["p808_mos"].extend(np.asarray(p808_out).reshape(batch_np.shape[0], -1)[:, 0].astype(np.float64).tolist())
    if primary_out is None or p808_out is None:
        metrics = {"sig": [], "bak": [], "ovr": [], "p808_mos": []}
        for i in range(batch_np.shape[0]):
            scores = compute_dnsmos(batch_np[i], primary_sess, p808_sess, sr=sr)
            metrics["sig"].append(float(scores["SIG"]))
            metrics["bak"].append(float(scores["BAK"]))
            metrics["ovr"].append(float(scores["OVR"]))
            metrics["p808_mos"].append(float(scores["P808_MOS"]))
    return metrics


def aggregate_dnsmos(storage, batch_metrics):
    for key, values in batch_metrics.items():
        storage[key].extend(values)


def compute_single_pesq(item):
    ref_np, pred_np, pesq_fn = item
    try:
        return float(pesq_fn(16000, ref_np, pred_np, "wb"))
    except Exception:
        return float("nan")


def compute_pesq_batch(ref_wav: torch.Tensor, pred_wav: torch.Tensor, sr: int, pesq_fn, executor=None):
    if sr != 16000:
        ref_wav = resample_audio(ref_wav, sr, 16000)
        pred_wav = resample_audio(pred_wav, sr, 16000)
    ref_np = ref_wav.cpu().numpy().astype(np.float32)
    pred_np = pred_wav.cpu().numpy().astype(np.float32)
    items = [(ref_np[i], pred_np[i], pesq_fn) for i in range(ref_np.shape[0])]
    if executor is None:
        return [compute_single_pesq(item) for item in items]
    return list(executor.map(compute_single_pesq, items))


def crop_for_metrics(ref_wav: torch.Tensor, pred_wav: torch.Tensor, sr: int, metric_duration: float):
    metric_len = int(sr * metric_duration)
    target_len = min(ref_wav.shape[-1], pred_wav.shape[-1], metric_len)
    return ref_wav[..., :target_len], pred_wav[..., :target_len]


def init_metric_storage():
    return {
        "si_sdr": [],
        "pesq_wb_16k": [],
        "clap_audio_cosine": [],
        "fad_ref_embs": [],
        "fad_eval_embs": [],
        "dnsmos_ref": {"sig": [], "bak": [], "ovr": [], "p808_mos": []},
        "dnsmos_eval": {"sig": [], "bak": [], "ovr": [], "p808_mos": []},
    }


def finalize_metric_storage(storage):
    fad_value = None
    if storage["fad_ref_embs"] and storage["fad_eval_embs"]:
        fad_value = frechet_distance(torch.cat(storage["fad_ref_embs"], dim=0), torch.cat(storage["fad_eval_embs"], dim=0))

    dnsmos_delta = None
    if len(storage["dnsmos_ref"]["sig"]) > 0:
        dnsmos_delta = {
            key: summarize_values(np.asarray(storage["dnsmos_eval"][key]) - np.asarray(storage["dnsmos_ref"][key]))
            for key in storage["dnsmos_ref"]
        }

    return {
        "si_sdr": summarize_values(storage["si_sdr"]),
        "pesq_wb_16k": summarize_values(storage["pesq_wb_16k"]),
        "fad": fad_value,
        "clap_audio_cosine": summarize_values(storage["clap_audio_cosine"]),
        "dnsmos_ref": {key: summarize_values(values) for key, values in storage["dnsmos_ref"].items()} if len(storage["dnsmos_ref"]["sig"]) > 0 else None,
        "dnsmos_eval": {key: summarize_values(values) for key, values in storage["dnsmos_eval"].items()} if len(storage["dnsmos_eval"]["sig"]) > 0 else None,
        "dnsmos_delta_eval_minus_ref": dnsmos_delta,
    }


def print_metric_summary(prefix: str, metrics):
    if metrics["si_sdr"] is not None:
        print(f"{prefix}_si_sdr_mean: {metrics['si_sdr']['mean']:.4f}")
    if metrics["pesq_wb_16k"] is not None:
        print(f"{prefix}_pesq_wb_16k_mean: {metrics['pesq_wb_16k']['mean']:.4f}")
    if metrics["fad"] is not None:
        print(f"{prefix}_fad: {metrics['fad']:.6f}")
    if metrics["clap_audio_cosine"] is not None:
        print(f"{prefix}_clap_audio_cosine_mean: {metrics['clap_audio_cosine']['mean']:.6f}")
    if metrics["dnsmos_eval"] is not None:
        print(f"{prefix}_dnsmos_ovr_mean: {metrics['dnsmos_eval']['ovr']['mean']:.4f}")


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
    target_waveforms, target_sample_rates, mix_waveforms, mix_sample_rates, sample_keys, skipped, per_tar = load_target_and_mix_from_tars(tar_paths)
    dataset_summary = summarize_dataset(target_waveforms, target_sample_rates)
    mix_dataset_summary = summarize_dataset(mix_waveforms, mix_sample_rates)

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
    pesq_executor = ThreadPoolExecutor(max_workers=max(1, args.pesq_workers)) if pesq_fn is not None and args.pesq_workers > 1 else None
    dnsmos_primary = None
    dnsmos_p808 = None
    dnsmos_error = None
    if args.skip_dnsmos:
        dnsmos_error = "skipped by --skip_dnsmos"
    else:
        dnsmos_primary, dnsmos_p808, dnsmos_error = try_create_dnsmos_sessions()

    target_codec_metrics = init_metric_storage()
    mix_codec_metrics = init_metric_storage()

    try:
        for start in tqdm(
            range(0, len(target_waveforms), args.batch_size),
            desc="Evaluating codec",
            unit="batch",
            dynamic_ncols=True,
        ):
            batch_target_waveforms = target_waveforms[start:start + args.batch_size]
            batch_target_srs = target_sample_rates[start:start + args.batch_size]
            batch_mix_waveforms = mix_waveforms[start:start + args.batch_size]
            batch_mix_srs = mix_sample_rates[start:start + args.batch_size]

            ref = preprocess_waveforms_batch(batch_target_waveforms, batch_target_srs, target_sr, target_len).to(args.device)
            mix = preprocess_waveforms_batch(batch_mix_waveforms, batch_mix_srs, target_sr, target_len).to(args.device)

            with torch.no_grad():
                codec_input = torch.cat([ref, mix], dim=0)
                codec_output = codec.decode(codec.encode(codec_input))
                recon = codec_output[:ref.shape[0]]
                mix_recon = codec_output[ref.shape[0]:]

            ref_t = ref.detach().cpu().float().squeeze(1)
            recon_t = recon.detach().cpu().float().squeeze(1)
            mix_recon_t = mix_recon.detach().cpu().float().squeeze(1)

            ref_t_target, recon_t = crop_for_metrics(ref_t, recon_t, target_sr, metric_duration)
            ref_t_mix, mix_recon_t = crop_for_metrics(ref_t, mix_recon_t, target_sr, metric_duration)

            target_codec_metrics["si_sdr"].extend(si_sdr(recon_t, ref_t_target).tolist())
            mix_codec_metrics["si_sdr"].extend(si_sdr(mix_recon_t, ref_t_mix).tolist())

            ref_16k = None
            recon_16k = None
            mix_recon_16k = None
            if pesq_fn is not None or panns_model is not None or (dnsmos_primary is not None and dnsmos_p808 is not None):
                ref_16k = resample_audio(ref_t_target, target_sr, 16000)
                recon_16k = resample_audio(recon_t, target_sr, 16000)
                mix_recon_16k = resample_audio(mix_recon_t, target_sr, 16000)

            if pesq_fn is not None:
                target_codec_metrics["pesq_wb_16k"].extend(compute_pesq_batch(ref_16k, recon_16k, 16000, pesq_fn, executor=pesq_executor))
                mix_codec_metrics["pesq_wb_16k"].extend(compute_pesq_batch(ref_16k, mix_recon_16k, 16000, pesq_fn, executor=pesq_executor))

            if panns_model is not None:
                ref_emb = panns_embedding(panns_model, ref_16k, 16000, args.device)
                recon_emb = panns_embedding(panns_model, recon_16k, 16000, args.device)
                mix_recon_emb = panns_embedding(panns_model, mix_recon_16k, 16000, args.device)
                target_codec_metrics["fad_ref_embs"].append(ref_emb)
                mix_codec_metrics["fad_ref_embs"].append(ref_emb)
                target_codec_metrics["fad_eval_embs"].append(recon_emb)
                mix_codec_metrics["fad_eval_embs"].append(mix_recon_emb)

            if clap_model is not None:
                ref_32k = resample_audio(ref_t_target, target_sr, 32000)
                recon_32k = resample_audio(recon_t, target_sr, 32000)
                mix_recon_32k = resample_audio(mix_recon_t, target_sr, 32000)
                ref_emb = clap_audio_embedding(clap_model, ref_32k, 32000, args.device)
                recon_emb = clap_audio_embedding(clap_model, recon_32k, 32000, args.device)
                mix_recon_emb = clap_audio_embedding(clap_model, mix_recon_32k, 32000, args.device)
                target_codec_metrics["clap_audio_cosine"].extend(torch.sum(F.normalize(ref_emb, dim=1) * F.normalize(recon_emb, dim=1), dim=1).tolist())
                mix_codec_metrics["clap_audio_cosine"].extend(torch.sum(F.normalize(ref_emb, dim=1) * F.normalize(mix_recon_emb, dim=1), dim=1).tolist())

            if dnsmos_primary is not None and dnsmos_p808 is not None:
                ref_dnsmos = compute_dnsmos_batch(ref_16k, 16000, dnsmos_primary, dnsmos_p808)
                recon_dnsmos = compute_dnsmos_batch(recon_16k, 16000, dnsmos_primary, dnsmos_p808)
                mix_recon_dnsmos = compute_dnsmos_batch(mix_recon_16k, 16000, dnsmos_primary, dnsmos_p808)
                aggregate_dnsmos(target_codec_metrics["dnsmos_ref"], ref_dnsmos)
                aggregate_dnsmos(mix_codec_metrics["dnsmos_ref"], ref_dnsmos)
                aggregate_dnsmos(target_codec_metrics["dnsmos_eval"], recon_dnsmos)
                aggregate_dnsmos(mix_codec_metrics["dnsmos_eval"], mix_recon_dnsmos)
    finally:
        if pesq_executor is not None:
            pesq_executor.shutdown(wait=True)

    target_codec_summary = finalize_metric_storage(target_codec_metrics)
    mix_codec_summary = finalize_metric_storage(mix_codec_metrics)

    summary = {
        "config_yaml": str(Path(args.config_yaml).resolve()),
        "tar_paths": [str(path.resolve()) for path in tar_paths],
        "tar_count": len(tar_paths),
        "seed": int(args.seed),
        "sample_count": len(target_waveforms),
        "skipped_count": int(skipped),
        "per_tar": per_tar,
        "dataset_summary": dataset_summary,
        "mix_dataset_summary": mix_dataset_summary,
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
            "si_sdr": target_codec_summary["si_sdr"],
            "pesq_wb_16k": target_codec_summary["pesq_wb_16k"],
            "fad": target_codec_summary["fad"],
            "clap_audio_cosine": target_codec_summary["clap_audio_cosine"],
            "dnsmos_ref": target_codec_summary["dnsmos_ref"],
            "dnsmos_recon": target_codec_summary["dnsmos_eval"],
            "dnsmos_delta_recon_minus_ref": target_codec_summary["dnsmos_delta_eval_minus_ref"],
            "target_codec_vs_target": target_codec_summary,
            "mix_codec_vs_target": mix_codec_summary,
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
    print(f"sample_count: {len(target_waveforms)}")
    print(f"codec_target_sr: {target_sr}")
    print(f"codec_duration: {duration:.4f}")
    print(f"metric_duration: {metric_duration:.4f}")
    if summary["metrics"]["si_sdr"] is not None:
        print(f"si_sdr_mean: {summary['metrics']['si_sdr']['mean']:.4f}")
    if summary["metrics"]["pesq_wb_16k"] is not None:
        print(f"pesq_wb_16k_mean: {summary['metrics']['pesq_wb_16k']['mean']:.4f}")
    if summary["metrics"]["fad"] is not None:
        print(f"fad: {summary['metrics']['fad']:.6f}")
    if summary["metrics"]["clap_audio_cosine"] is not None:
        print(f"clap_audio_cosine_mean: {summary['metrics']['clap_audio_cosine']['mean']:.6f}")
    if summary["metrics"]["dnsmos_recon"] is not None:
        print(f"dnsmos_ovr_mean: {summary['metrics']['dnsmos_recon']['ovr']['mean']:.4f}")
    print_metric_summary("mix_codec_vs_target", summary["metrics"]["mix_codec_vs_target"])
    print(f"output_json: {output_path}")


if __name__ == "__main__":
    main()
