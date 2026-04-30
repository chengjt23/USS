import argparse
import importlib.util
import io
import json
import os
import random
import sys
import tempfile
import types
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
import torchaudio
import webdataset as wds
import yaml
from scipy import linalg
from tqdm import tqdm

SCRIPT_PATH = Path(__file__).resolve()
USS_ROOT = SCRIPT_PATH.parents[3]
WORKSPACE_ROOT = SCRIPT_PATH.parents[4]
FLOWSEP_ROOT = WORKSPACE_ROOT / "FlowSep"
SAM_AUDIO_ROOT = WORKSPACE_ROOT / "sam-audio"
DACVAE_ROOT = WORKSPACE_ROOT / "dacvae"
DEFAULT_USS_CONFIG = USS_ROOT / "configs" / "dacvae-seq-dit" / "hive-2mix-44100hz-4s" / "chen_bridge_sde_2mix_dacvae_seq_dit.yaml"
DEFAULT_FLOWSEP_CONFIG = FLOWSEP_ROOT / "lass_config" / "2channel_flow.yaml"
DEFAULT_OUTPUT_DIR = SCRIPT_PATH.parent / "outputs"
DEFAULT_PANNS_CKPT = WORKSPACE_ROOT / "Cnn14_16k_mAP=0.438.pth"
DEFAULT_CLAP_CKPT = USS_ROOT / "metrics" / "clapscore" / "music_speech_audioset_epoch_15_esc_89.98.pt"
DEFAULT_FLOWSEP_CKPT_URL = "https://zenodo.org/records/13869712/files/v2_100k.ckpt?download=1"
for path in [USS_ROOT, FLOWSEP_ROOT, FLOWSEP_ROOT / "src", SAM_AUDIO_ROOT, DACVAE_ROOT]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
if "ipdb" not in sys.modules:
    ipdb_module = types.ModuleType("ipdb")
    ipdb_module.set_trace = lambda *args, **kwargs: None
    sys.modules["ipdb"] = ipdb_module
if "wandb" not in sys.modules:
    wandb_module = types.ModuleType("wandb")
    wandb_module.init = lambda *args, **kwargs: None
    wandb_module.log = lambda *args, **kwargs: None
    wandb_module.Audio = lambda *args, **kwargs: None
    sys.modules["wandb"] = wandb_module
utilities_module = sys.modules.setdefault("utilities", types.ModuleType("utilities"))
utilities_module.__path__ = [str(FLOWSEP_ROOT / "src" / "utilities")]
utilities_audio_module = sys.modules.setdefault("utilities.audio", types.ModuleType("utilities.audio"))
utilities_audio_module.__path__ = [str(FLOWSEP_ROOT / "src" / "utilities" / "audio")]
from latent_diffusion.util import instantiate_from_config

_RESAMPLER_CACHE = {}


def load_py_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


load_py_module("utilities.audio.audio_processing", FLOWSEP_ROOT / "src" / "utilities" / "audio" / "audio_processing.py")
FLOWSEP_STFT_MODULE = load_py_module("_flowsep_audio_stft", FLOWSEP_ROOT / "src" / "utilities" / "audio" / "stft.py")
TacotronSTFT = FLOWSEP_STFT_MODULE.TacotronSTFT


def get_mel_from_wav(audio, stft_tool):
    audio = torch.clip(torch.FloatTensor(audio).unsqueeze(0), -1, 1)
    audio = torch.autograd.Variable(audio, requires_grad=False)
    melspec, magnitudes, _, energy = stft_tool.mel_spectrogram(audio)
    return torch.squeeze(melspec, 0).numpy().astype(np.float32), torch.squeeze(magnitudes, 0).numpy().astype(np.float32), torch.squeeze(energy, 0).numpy().astype(np.float32)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--uss_config", type=str, default=str(DEFAULT_USS_CONFIG))
    parser.add_argument("--flowsep_config", type=str, default=str(DEFAULT_FLOWSEP_CONFIG))
    parser.add_argument("--ckpt_path", type=str, default="")
    parser.add_argument("--hf_repo_id", type=str, default="")
    parser.add_argument("--hf_filename", type=str, default="v2_100k.ckpt")
    parser.add_argument("--ckpt_url", type=str, default=DEFAULT_FLOWSEP_CKPT_URL)
    parser.add_argument("--gpus", type=str, default="0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_tars", type=int, default=0)
    parser.add_argument("--tar_path", type=str, default="")
    parser.add_argument("--max_samples_per_tar", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--infer_step", type=int, default=0)
    parser.add_argument("--metric_duration", type=float, default=4.0)
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--panns_ckpt_path", type=str, default=str(DEFAULT_PANNS_CKPT))
    parser.add_argument("--clap_ckpt_path", type=str, default=str(DEFAULT_CLAP_CKPT))
    parser.add_argument("--sam_judge_model_path", type=str, default="facebook/sam-audio-judge")
    parser.add_argument("--pesq_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def load_yaml(path):
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


def choose_tars(configs, tar_path, seed, num_tars):
    if tar_path is not None:
        paths = [Path(p.strip()) for p in tar_path.split(",") if p.strip()]
        if not paths:
            raise ValueError("tar_path is empty")
        for path in paths:
            if not path.exists():
                raise FileNotFoundError(f"tar_path does not exist: {path}")
        return paths
    tar_paths = collect_val_tars(configs)
    if num_tars <= 0:
        return tar_paths
    rng = random.Random(seed)
    return rng.sample(tar_paths, min(num_tars, len(tar_paths)))


def resample_audio(wav, src_sr, dst_sr):
    if src_sr == dst_sr:
        return wav
    key = (int(src_sr), int(dst_sr), str(wav.device))
    resampler = _RESAMPLER_CACHE.get(key)
    if resampler is None:
        resampler = torchaudio.transforms.Resample(orig_freq=src_sr, new_freq=dst_sr).to(wav.device)
        _RESAMPLER_CACHE[key] = resampler
    return resampler(wav.float())


def load_panns_model(checkpoint_path, device):
    fad_dir = USS_ROOT / "metrics" / "fad"
    load_py_module("pytorch_utils", fad_dir / "pytorch_utils.py")
    models_module = load_py_module(f"_panns_models_flowsep_{os.getpid()}", fad_dir / "models.py")
    model = models_module.Cnn14_16k(sample_rate=16000, window_size=512, hop_size=160, mel_bins=64, fmin=50, fmax=8000, classes_num=527)
    try:
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    return model


def load_clap_model(checkpoint_path, device):
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


def frechet_distance(emb_a, emb_b, eps=1e-6):
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
    return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max()), "count": int(arr.size)}


def try_import_pesq():
    try:
        from pesq import pesq as pesq_fn
        return pesq_fn, None
    except Exception as e:
        return None, str(e)


def compute_single_pesq(item):
    ref_np, pred_np, pesq_fn = item
    try:
        return float(pesq_fn(16000, ref_np, pred_np, "wb"))
    except Exception:
        return float("nan")


def compute_pesq_batch(ref_wav, pred_wav, sr, pesq_fn, executor=None):
    if sr != 16000:
        ref_wav = resample_audio(ref_wav, sr, 16000)
        pred_wav = resample_audio(pred_wav, sr, 16000)
    ref_np = ref_wav.cpu().numpy().astype(np.float32)
    pred_np = pred_wav.cpu().numpy().astype(np.float32)
    items = [(ref_np[i], pred_np[i], pesq_fn) for i in range(ref_np.shape[0])]
    if executor is None:
        return [compute_single_pesq(item) for item in items]
    return list(executor.map(compute_single_pesq, items))


def parse_gpu_ids(gpus):
    gpus = (gpus or "").strip().lower()
    if gpus in {"", "cpu", "none"}:
        return []
    return [int(x.strip()) for x in gpus.split(",") if x.strip()]


def download_file(url, target_path):
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as src, open(target_path, "wb") as dst, tqdm(total=int(src.headers.get("Content-Length", 0)), desc=f"Downloading {target_path.name}", unit="B", unit_scale=True, dynamic_ncols=True) as pbar:
        while True:
            chunk = src.read(1024 * 1024)
            if not chunk:
                break
            dst.write(chunk)
            pbar.update(len(chunk))
    return target_path


def resolve_flowsep_ckpt(args, output_dir):
    if args.ckpt_path:
        ckpt_path = Path(args.ckpt_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        return ckpt_path.resolve()
    cache_dir = output_dir / "pretrained"
    cache_dir.mkdir(parents=True, exist_ok=True)
    if args.hf_repo_id:
        try:
            from huggingface_hub import hf_hub_download
            return Path(hf_hub_download(repo_id=args.hf_repo_id, filename=args.hf_filename, local_dir=str(cache_dir), local_dir_use_symlinks=False))
        except Exception as e:
            print(f"HuggingFace download failed, fallback to url: {e}")
    target_path = cache_dir / args.hf_filename
    if target_path.exists():
        return target_path.resolve()
    return download_file(args.ckpt_url, target_path).resolve()


@contextmanager
def pushd(path):
    old = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def normalize_flowsep_wav(wav):
    wav = wav.reshape(-1).float()
    wav = wav - wav.mean()
    return wav / wav.abs().max().clamp(min=1e-8) * 0.5


def crop_or_pad(wav, target_len):
    wav = wav.reshape(-1).float()
    if wav.numel() > target_len:
        return wav[:target_len]
    if wav.numel() < target_len:
        return F.pad(wav, (0, target_len - wav.numel()))
    return wav


def load_audio_bytes(audio_bytes):
    try:
        wav, sr = torchaudio.load(io.BytesIO(audio_bytes), format="wav")
    except Exception:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_bytes)
            temp_path = f.name
        try:
            wav, sr = torchaudio.load(temp_path)
        finally:
            try:
                os.remove(temp_path)
            except OSError:
                pass
    wav = wav.float()
    if wav.ndim == 2 and wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=False)
    elif wav.ndim == 2:
        wav = wav.squeeze(0)
    return wav, int(sr)


def parse_caption(sample):
    meta = sample.get("json") or sample.get("__json__") or sample.get("txt") or sample.get("meta")
    if meta is None:
        return ""
    if isinstance(meta, (bytes, bytearray)):
        meta = meta.decode("utf-8")
    try:
        meta = json.loads(meta) if isinstance(meta, str) else meta
    except Exception:
        return ""
    labels = meta.get("labels", meta.get("label", []))
    if isinstance(labels, str):
        return labels
    if isinstance(labels, list) and len(labels) > 0 and isinstance(labels[0], str):
        return labels[0]
    return ""


def iter_tar_samples(tar_path, max_samples_per_tar):
    count = 0
    dataset = wds.DataPipeline(wds.SimpleShardList([str(tar_path)]), wds.tarfile_to_samples(handler=wds.warn_and_continue))
    for sample in dataset:
        if max_samples_per_tar > 0 and count >= max_samples_per_tar:
            break
        mix_bytes = sample.get("mix.wav") or sample.get("mix")
        source_keys = sorted(k for k in sample.keys() if isinstance(k, str) and k.startswith("s") and k.endswith(".wav"))
        if mix_bytes is None or not source_keys:
            continue
        target_bytes = sample[source_keys[0]]
        mix_wav, mix_sr = load_audio_bytes(mix_bytes)
        target_wav, target_sr = load_audio_bytes(target_bytes)
        yield {"tar_path": str(Path(tar_path).resolve()), "sample_key": str(sample.get("__key__", count)), "caption": parse_caption(sample), "mix_wav": mix_wav, "mix_sr": mix_sr, "target_wav": target_wav, "target_sr": target_sr}
        count += 1


def pad_spec(spec, target_length):
    p = target_length - spec.shape[0]
    if p > 0:
        spec = F.pad(spec, (0, 0, 0, p))
    elif p < 0:
        spec = spec[:target_length, :]
    return spec[..., :-1] if spec.size(-1) % 2 != 0 else spec


def wav_feature_extraction(wav, stft_tool, target_length):
    log_mel_spec, stft, _ = get_mel_from_wav(wav, stft_tool)
    return pad_spec(torch.from_numpy(log_mel_spec.T).float(), target_length), pad_spec(torch.from_numpy(stft.T).float(), target_length)


def clap_text_embedding(model, texts, device):
    with torch.no_grad():
        return model.get_query_embed(modality="text", text=list(texts), device=device).float().cpu()


def sam_judge_batch(model, processor, mix_wav, pred_wav, texts, device):
    mix_wav = resample_audio(mix_wav, 16000, 48000)
    pred_wav = resample_audio(pred_wav, 16000, 48000)
    batch = processor(text=list(texts), input_audio=[x.cpu() for x in mix_wav], separated_audio=[x.cpu() for x in pred_wav], sampling_rate=48000).to(device)
    with torch.inference_mode():
        output = model(**batch)
    return {"overall": output.overall.squeeze(-1).cpu().tolist(), "faithfulness": output.faithfulness.squeeze(-1).cpu().tolist(), "recall": output.recall.squeeze(-1).cpu().tolist(), "precision": output.precision.squeeze(-1).cpu().tolist()}


def load_sam_judge(model_path, device):
    try:
        from sam_audio import SAMAudioJudgeModel, SAMAudioJudgeProcessor
        model = SAMAudioJudgeModel.from_pretrained(model_path).to(device).eval()
        processor = SAMAudioJudgeProcessor.from_pretrained(model_path)
        return model, processor, None
    except Exception as e:
        return None, None, str(e)


def build_flowsep_stft(config):
    return TacotronSTFT(config["preprocessing"]["stft"]["filter_length"], config["preprocessing"]["stft"]["hop_length"], config["preprocessing"]["stft"]["win_length"], config["preprocessing"]["mel"]["n_mel_channels"], config["preprocessing"]["audio"]["sampling_rate"], config["preprocessing"]["mel"]["mel_fmin"], config["preprocessing"]["mel"]["mel_fmax"])


def load_flowsep_model(flowsep_config_path, ckpt_path, device, infer_step):
    config = load_yaml(flowsep_config_path)
    config["model"]["params"]["cond_stage_config"]["crossattn_text"]["target"] = "models.flowsep.text_encoder.FlanT5HiddenState"
    config["model"]["params"]["first_stage_config"]["params"]["reload_from_ckpt"] = None
    config["model"]["params"]["evaluation_params"]["n_candidates_per_samples"] = 1
    if infer_step > 0:
        config["model"]["params"]["evaluation_params"]["ddim_sampling_steps"] = infer_step
    with pushd(FLOWSEP_ROOT):
        model = instantiate_from_config(config["model"]).to(device).eval()
    try:
        checkpoint = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(str(ckpt_path), map_location="cpu")
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=True)
    return model, config, build_flowsep_stft(config)


def build_inference_batch(samples, flowsep_config, stft_tool, metric_duration):
    sr = flowsep_config["preprocessing"]["audio"]["sampling_rate"]
    infer_len = int(flowsep_config["preprocessing"]["audio"]["duration"] * sr)
    metric_len = int(metric_duration * sr)
    target_frames = int(flowsep_config["preprocessing"]["audio"]["duration"] * sr / flowsep_config["preprocessing"]["stft"]["hop_length"])
    batch = {"fname": [], "text": [], "caption": [], "label_vector": [], "waveform": [], "stft": [], "log_mel_spec": [], "mixed_waveform": [], "mixed_mel": [], "duration": flowsep_config["preprocessing"]["audio"]["duration"], "sampling_rate": sr, "random_start_sample_in_original_audio_file": 0}
    refs, mixes, ids = [], [], []
    for sample in samples:
        mix_16k = crop_or_pad(resample_audio(sample["mix_wav"].unsqueeze(0), sample["mix_sr"], sr).squeeze(0), metric_len)
        ref_16k = crop_or_pad(resample_audio(sample["target_wav"].unsqueeze(0), sample["target_sr"], sr).squeeze(0), metric_len)
        mix_in = crop_or_pad(F.pad(normalize_flowsep_wav(mix_16k), (0, max(0, infer_len - metric_len))), infer_len)
        ref_in = crop_or_pad(F.pad(normalize_flowsep_wav(ref_16k), (0, max(0, infer_len - metric_len))), infer_len)
        ref_mel, ref_stft = wav_feature_extraction(ref_in, stft_tool, target_frames)
        mix_mel, _ = wav_feature_extraction(mix_in, stft_tool, target_frames)
        sample_id = f"{Path(sample['tar_path']).stem}:{sample['sample_key']}"
        batch["fname"].append(sample_id)
        batch["text"].append(sample["caption"])
        batch["caption"].append(sample["caption"])
        batch["label_vector"].append(torch.zeros(1))
        batch["waveform"].append(ref_in.unsqueeze(0))
        batch["stft"].append(ref_stft)
        batch["log_mel_spec"].append(ref_mel)
        batch["mixed_waveform"].append(mix_in.unsqueeze(0))
        batch["mixed_mel"].append(mix_mel)
        refs.append(ref_16k)
        mixes.append(mix_16k)
        ids.append(sample_id)
    batch["label_vector"] = torch.stack(batch["label_vector"], dim=0)
    batch["waveform"] = torch.stack(batch["waveform"], dim=0)
    batch["stft"] = torch.stack(batch["stft"], dim=0)
    batch["log_mel_spec"] = torch.stack(batch["log_mel_spec"], dim=0)
    batch["mixed_waveform"] = torch.stack(batch["mixed_waveform"], dim=0)
    batch["mixed_mel"] = torch.stack(batch["mixed_mel"], dim=0)
    return batch, torch.stack(refs, dim=0), torch.stack(mixes, dim=0), ids


def run_flowsep_inference(model, batch, ddim_steps, guidance_scale):
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.to(model.device)
    z, cond = model.get_input(batch, model.first_stage_key, unconditional_prob_cfg=0.0)
    cond = model.filter_useful_cond_dict(cond)
    unconditional_conditioning = None
    if guidance_scale != 1.0:
        unconditional_conditioning = {}
        for key in model.cond_stage_model_metadata:
            model_idx = model.cond_stage_model_metadata[key]["model_idx"]
            unconditional_conditioning[key] = model.cond_stage_models[model_idx].get_unconditional_condition(z.shape[0])
    extra = model.__class__.__mro__[1].get_input(model, batch, model.extra_channel_key).to(model.device)
    extra = extra.reshape(extra.shape[0], 1, extra.shape[1], extra.shape[2])
    x_t = model.get_first_stage_encoding(model.encode_first_stage(extra)).detach()
    samples, _ = model.sample_log(cond=cond, batch_size=z.shape[0], x_T=x_t, ddim=True, ddim_steps=ddim_steps, unconditional_guidance_scale=guidance_scale, unconditional_conditioning=unconditional_conditioning)
    samples = samples[:, :model.channels, :, :] if model.extra_channels else samples
    mel = model.decode_first_stage(samples)
    if model.fbank_shift:
        mel = mel - model.fbank_shift
    if model.data_std:
        mel = (mel * model.data_std) + model.data_mean
    pred = model.mel_spectrogram_to_waveform(mel, save=False)
    pred = pred.astype(np.float32) / 32767.0 if np.issubdtype(pred.dtype, np.integer) else pred.astype(np.float32)
    pred = torch.from_numpy(pred)
    return pred.squeeze(1) if pred.ndim == 3 else pred


def worker_main(rank, state):
    args = argparse.Namespace(**state["args"])
    tar_paths = [Path(p) for p in state["tar_paths"]]
    gpu_ids = state["gpu_ids"]
    output_dir = Path(args.output_dir)
    partial_dir = output_dir / "partials"
    partial_dir.mkdir(parents=True, exist_ok=True)
    local_tars = tar_paths if len(gpu_ids) <= 1 else tar_paths[rank::len(gpu_ids)]
    device = f"cuda:{gpu_ids[rank]}" if gpu_ids else "cpu"
    torch.cuda.set_device(gpu_ids[rank]) if gpu_ids else None
    torch.manual_seed(args.seed + rank)
    model, flowsep_config, stft_tool = load_flowsep_model(args.flowsep_config, args.ckpt_path, device, args.infer_step)
    ddim_steps = flowsep_config["model"]["params"]["evaluation_params"]["ddim_sampling_steps"]
    guidance_scale = flowsep_config["model"]["params"]["evaluation_params"]["unconditional_guidance_scale"]
    panns_model = load_panns_model(Path(args.panns_ckpt_path), device)
    clap_model = load_clap_model(Path(args.clap_ckpt_path), device)
    pesq_fn, pesq_error = try_import_pesq()
    pesq_executor = ThreadPoolExecutor(max_workers=max(1, args.pesq_workers)) if pesq_fn is not None and args.pesq_workers > 1 else None
    sam_judge_model, sam_judge_processor, sam_judge_error = load_sam_judge(args.sam_judge_model_path, device)
    sample_results, fad_ref_embs, fad_pred_embs, clap_ref_embs, clap_pred_embs, clap_text_embs = [], [], [], [], [], []
    batch_buffer = []
    total = len(local_tars) * args.max_samples_per_tar if args.max_samples_per_tar > 0 else None
    pbar = tqdm(total=total, desc=f"gpu{gpu_ids[rank] if gpu_ids else 'cpu'}", unit="sample", dynamic_ncols=True)
    def flush_batch():
        nonlocal batch_buffer
        if not batch_buffer:
            return
        batch, ref_wav, mix_wav, sample_ids = build_inference_batch(batch_buffer, flowsep_config, stft_tool, args.metric_duration)
        with torch.no_grad(), model.ema_scope():
            pred_wav = run_flowsep_inference(model, batch, ddim_steps, guidance_scale).cpu()
        metric_len = int(args.metric_duration * flowsep_config["preprocessing"]["audio"]["sampling_rate"])
        ref_wav = ref_wav[:, :metric_len]
        mix_wav = mix_wav[:, :metric_len]
        pred_wav = pred_wav[:, :metric_len]
        si_sdr_vals = si_sdr(pred_wav, ref_wav).cpu().tolist()
        pesq_vals = compute_pesq_batch(ref_wav, pred_wav, 16000, pesq_fn, executor=pesq_executor) if pesq_fn is not None else [float("nan")] * pred_wav.size(0)
        fad_ref_embs.append(panns_embedding(panns_model, ref_wav, 16000, device))
        fad_pred_embs.append(panns_embedding(panns_model, pred_wav, 16000, device))
        clap_ref_embs.append(clap_audio_embedding(clap_model, ref_wav, 16000, device))
        clap_pred_embs.append(clap_audio_embedding(clap_model, pred_wav, 16000, device))
        clap_text_embs.append(clap_text_embedding(clap_model, batch["caption"], device))
        judge_vals = sam_judge_batch(sam_judge_model, sam_judge_processor, mix_wav, pred_wav, batch["caption"], device) if sam_judge_model is not None else {"overall": [None] * len(batch_buffer), "faithfulness": [None] * len(batch_buffer), "recall": [None] * len(batch_buffer), "precision": [None] * len(batch_buffer)}
        for idx, sample in enumerate(batch_buffer):
            sample_results.append({"sample_id": sample_ids[idx], "tar_path": sample["tar_path"], "sample_key": sample["sample_key"], "caption": sample["caption"], "si_sdr": float(si_sdr_vals[idx]), "pesq_wb_16k": float(pesq_vals[idx]) if np.isfinite(pesq_vals[idx]) else None, "sam_audio_judge_overall": float(judge_vals["overall"][idx]) if judge_vals["overall"][idx] is not None else None, "sam_audio_judge_faithfulness": float(judge_vals["faithfulness"][idx]) if judge_vals["faithfulness"][idx] is not None else None, "sam_audio_judge_recall": float(judge_vals["recall"][idx]) if judge_vals["recall"][idx] is not None else None, "sam_audio_judge_precision": float(judge_vals["precision"][idx]) if judge_vals["precision"][idx] is not None else None})
        batch_buffer = []
    try:
        for tar_path in local_tars:
            for sample in iter_tar_samples(tar_path, args.max_samples_per_tar):
                batch_buffer.append(sample)
                pbar.update(1)
                if len(batch_buffer) >= args.batch_size:
                    flush_batch()
            flush_batch()
    finally:
        flush_batch()
        pbar.close()
        if pesq_executor is not None:
            pesq_executor.shutdown(wait=True)
    torch.save({"sample_results": sample_results, "fad_ref_embs": torch.cat(fad_ref_embs, dim=0) if fad_ref_embs else None, "fad_pred_embs": torch.cat(fad_pred_embs, dim=0) if fad_pred_embs else None, "clap_ref_embs": torch.cat(clap_ref_embs, dim=0) if clap_ref_embs else None, "clap_pred_embs": torch.cat(clap_pred_embs, dim=0) if clap_pred_embs else None, "clap_text_embs": torch.cat(clap_text_embs, dim=0) if clap_text_embs else None, "pesq_error": pesq_error, "sam_judge_error": sam_judge_error}, partial_dir / f"worker_{rank}.pt")


def merge_results(args, tar_paths, ckpt_path, gpu_ids):
    output_dir = Path(args.output_dir)
    partials = sorted((output_dir / "partials").glob("worker_*.pt"))
    sample_results, fad_ref, fad_pred, clap_ref, clap_pred, clap_text = [], [], [], [], [], []
    pesq_error = None
    sam_judge_error = None
    for partial in partials:
        data = torch.load(partial, map_location="cpu", weights_only=False)
        sample_results.extend(data["sample_results"])
        if data["fad_ref_embs"] is not None:
            fad_ref.append(data["fad_ref_embs"])
            fad_pred.append(data["fad_pred_embs"])
            clap_ref.append(data["clap_ref_embs"])
            clap_pred.append(data["clap_pred_embs"])
            clap_text.append(data["clap_text_embs"])
        pesq_error = pesq_error or data.get("pesq_error")
        sam_judge_error = sam_judge_error or data.get("sam_judge_error")
    sample_results.sort(key=lambda x: x["sample_id"])
    summary = {"uss_config": str(Path(args.uss_config).resolve()), "flowsep_config": str(Path(args.flowsep_config).resolve()), "flowsep_ckpt": str(ckpt_path), "num_tars": len(tar_paths), "num_workers": max(1, len(gpu_ids)), "num_samples": len(sample_results), "metrics": {"si_sdr": summarize_values([x["si_sdr"] for x in sample_results if x["si_sdr"] is not None]), "pesq_wb_16k": summarize_values([x["pesq_wb_16k"] for x in sample_results if x["pesq_wb_16k"] is not None]), "sam_audio_judge_overall": summarize_values([x["sam_audio_judge_overall"] for x in sample_results if x["sam_audio_judge_overall"] is not None]), "sam_audio_judge_faithfulness": summarize_values([x["sam_audio_judge_faithfulness"] for x in sample_results if x["sam_audio_judge_faithfulness"] is not None]), "sam_audio_judge_recall": summarize_values([x["sam_audio_judge_recall"] for x in sample_results if x["sam_audio_judge_recall"] is not None]), "sam_audio_judge_precision": summarize_values([x["sam_audio_judge_precision"] for x in sample_results if x["sam_audio_judge_precision"] is not None])}, "metric_errors": {"pesq_wb_16k": pesq_error, "sam_audio_judge": sam_judge_error}}
    if fad_ref:
        fad_ref_all = torch.cat(fad_ref, dim=0)
        fad_pred_all = torch.cat(fad_pred, dim=0)
        clap_ref_all = torch.cat(clap_ref, dim=0)
        clap_pred_all = torch.cat(clap_pred, dim=0)
        clap_text_all = torch.cat(clap_text, dim=0)
        summary["metrics"]["fad"] = frechet_distance(fad_ref_all, fad_pred_all)
        summary["metrics"]["clapscore_a"] = float(torch.nn.functional.cosine_similarity(clap_ref_all, clap_pred_all, dim=1).mean().item())
        summary["metrics"]["clapscore"] = float((clap_text_all * clap_pred_all).sum(dim=1).mean().item())
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(output_dir / "samples.jsonl", "w", encoding="utf-8") as f:
        for item in sample_results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    partial_dir = output_dir / "partials"
    partial_dir.mkdir(parents=True, exist_ok=True)
    for partial in partial_dir.glob("worker_*.pt"):
        partial.unlink()
    uss_config = load_yaml(args.uss_config)
    if args.num_tars <= 0:
        args.num_tars = int(uss_config["datamodule"]["data_config"].get("val_tar_count", 0))
    if args.max_samples_per_tar <= 0:
        args.max_samples_per_tar = int(uss_config["datamodule"]["data_config"].get("val_samples_per_tar", 0) or 0)
    tar_paths = choose_tars(uss_config, args.tar_path or None, args.seed, args.num_tars)
    ckpt_path = resolve_flowsep_ckpt(args, output_dir)
    args.ckpt_path = str(ckpt_path)
    gpu_ids = parse_gpu_ids(args.gpus)
    state = {"args": vars(args), "tar_paths": [str(p) for p in tar_paths], "gpu_ids": gpu_ids}
    if gpu_ids and len(gpu_ids) > 1:
        mp.spawn(worker_main, args=(state,), nprocs=len(gpu_ids), join=True)
    else:
        worker_main(0, state)
    merge_results(args, tar_paths, ckpt_path, gpu_ids)


if __name__ == "__main__":
    main()
