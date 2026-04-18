import os
os.environ["CUDA_VISIBLE_DEVICES"] = "9"
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
import yaml
from tqdm import tqdm

SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
USS_ROOT = SCRIPT_PATH.parents[2]
DEFAULT_TEST_CASES_DIR = SCRIPT_DIR / "test_cases"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "inference_result"
AUDIO_SUFFIXES = (".wav", ".flac", ".mp3", ".ogg", ".m4a")

if str(USS_ROOT) not in sys.path:
    sys.path.insert(0, str(USS_ROOT))

from data.wds_datamodule import wds_collate_fn
from train import build_stft_tool, convert_wds_batch_to_model_format, get_required_audio_feature_keys
from utils.tools import instantiate_from_config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_yaml", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--test_cases_dir", type=str, default=str(DEFAULT_TEST_CASES_DIR))
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--ddim_steps", type=int, default=None)
    parser.add_argument("--guidance_scale", type=float, default=None)
    return parser.parse_args()


def load_config(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def load_model(configs, ckpt_path: Path, device: str):
    if "precision" in configs:
        torch.set_float32_matmul_precision(configs["precision"])
    model = instantiate_from_config(configs["model"]).to(device).eval()
    try:
        ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(str(ckpt_path), map_location=device)
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=False)
    return model


def list_sample_dirs(test_cases_dir: Path):
    sample_dirs = sorted(path for path in test_cases_dir.iterdir() if path.is_dir())
    if not sample_dirs:
        raise FileNotFoundError(f"No sample directories found in {test_cases_dir}")
    return sample_dirs


def choose_audio_file(sample_dir: Path, stem: str):
    candidates = [
        path for path in sample_dir.iterdir()
        if path.is_file() and path.stem.lower() == stem.lower() and path.suffix.lower() in AUDIO_SUFFIXES
    ]
    if not candidates:
        return None
    return sorted(candidates, key=lambda path: (AUDIO_SUFFIXES.index(path.suffix.lower()), path.name.lower()))[0]


def list_source_audio_files(sample_dir: Path):
    candidates = [
        path for path in sample_dir.iterdir()
        if path.is_file() and path.stem.lower().startswith("s") and path.suffix.lower() in AUDIO_SUFFIXES
    ]
    return sorted(candidates, key=source_sort_key)


def source_sort_key(path: Path):
    stem = path.stem.lower()
    suffix = stem[1:]
    if suffix.isdigit():
        return (0, int(suffix), stem)
    return (1, stem)


def choose_metadata_file(sample_dir: Path):
    candidates = [
        path for path in sample_dir.iterdir()
        if path.is_file() and (path.name.lower() in {"json", "meta", "txt"} or path.suffix.lower() in {".json", ".txt"})
    ]
    if not candidates:
        return None
    return sorted(candidates, key=metadata_sort_key)[0]


def metadata_sort_key(path: Path):
    name = path.name.lower()
    if name == "json":
        return (0, name)
    if name.endswith(".json"):
        return (1, name)
    if name == "meta":
        return (2, name)
    if name == "txt":
        return (3, name)
    if name.endswith(".txt"):
        return (4, name)
    return (5, name)


def load_metadata(sample_dir: Path):
    metadata_path = choose_metadata_file(sample_dir)
    if metadata_path is None:
        return {}
    text = metadata_path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except Exception:
        return {}


def extract_labels(meta: dict):
    labels = meta.get("labels")
    if labels is None:
        labels = meta.get("label")
    if labels is None:
        labels = meta.get("caption")
    if labels is None:
        labels = meta.get("text")
    if labels is None:
        return []
    if isinstance(labels, str):
        return [labels]
    if isinstance(labels, list):
        return [item for item in labels if isinstance(item, str)]
    return []


def load_audio_tensor(path: Path, target_sr: int):
    audio, sr = torchaudio.load(str(path))
    audio = audio.float()
    if sr != target_sr:
        audio = torchaudio.functional.resample(audio, sr, target_sr)
    return audio


def pad_audio_length(audio: torch.Tensor, target_length: int):
    if audio.shape[-1] >= target_length:
        return audio[..., :target_length]
    return F.pad(audio, (0, target_length - audio.shape[-1]))


def align_source_channels(audio: torch.Tensor, max_channels: int):
    channels = audio.shape[0]
    if channels == max_channels:
        return audio
    if channels == 1 and max_channels > 1:
        return audio.expand(max_channels, -1)
    if channels > max_channels:
        return audio[:max_channels, :]
    return audio


def build_sources_tensor(source_paths, target_sr: int, mix_channels: int):
    if not source_paths:
        return torch.empty(0)
    source_tensors = [load_audio_tensor(path, target_sr) for path in source_paths]
    max_channels = max([mix_channels] + [tensor.shape[0] for tensor in source_tensors])
    max_length = max(tensor.shape[-1] for tensor in source_tensors)
    source_tensors = [align_source_channels(tensor, max_channels) for tensor in source_tensors]
    source_tensors = [pad_audio_length(tensor, max_length) for tensor in source_tensors]
    return torch.stack(source_tensors, dim=0)


def build_model_batch(sample_dir: Path, configs, stft_tool, target_sr: int):
    mix_path = choose_audio_file(sample_dir, "mix")
    if mix_path is None:
        raise FileNotFoundError(f"Sample {sample_dir.name} is missing mix audio")
    mix_tensor = load_audio_tensor(mix_path, target_sr)
    source_paths = list_source_audio_files(sample_dir)
    sources_tensor = build_sources_tensor(source_paths, target_sr, mix_tensor.shape[0])
    meta = load_metadata(sample_dir)
    labels = extract_labels(meta)
    raw_batch = wds_collate_fn([(mix_tensor, sources_tensor, labels, {"sample_dir": sample_dir.name, "meta": meta})])
    model_batch = convert_wds_batch_to_model_format(raw_batch, configs, stft_tool)
    model_batch["fname"][0] = f"{sample_dir.name}.wav"
    return model_batch


def infer_single_sample(model, batch, sample_name: str, ddim_steps, guidance_scale: float):
    return model.generate_sample(
        [batch],
        ddim_steps=ddim_steps,
        unconditional_guidance_scale=guidance_scale,
        n_gen=1,
        name=sample_name,
        save=False,
        save_mixed=False,
    )


def format_waveform_for_save(waveform):
    array = np.asarray(waveform, dtype=np.float32)
    if array.ndim == 3:
        array = array[0]
    if array.ndim == 2 and array.shape[0] <= 8 and array.shape[0] < array.shape[1]:
        array = array.T
    if array.ndim == 2 and array.shape[1] == 1:
        array = array[:, 0]
    return array


def get_target_sample_rate(configs):
    data_config = configs.get("datamodule", {}).get("data_config", {})
    if "sample_rate" in data_config:
        return int(data_config["sample_rate"])
    audio_config = configs.get("preprocessing", {}).get("audio", {})
    if "sampling_rate" in audio_config:
        return int(audio_config["sampling_rate"])
    return int(configs.get("model", {}).get("params", {}).get("sampling_rate", 16000))


def main():
    args = parse_args()
    config_path = Path(args.config_yaml).resolve()
    ckpt_path = Path(args.ckpt_path).resolve()
    test_cases_dir = Path(args.test_cases_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")
    if not test_cases_dir.exists():
        raise FileNotFoundError(f"Test cases directory not found: {test_cases_dir}")

    configs = load_config(config_path)
    target_sr = get_target_sample_rate(configs)
    required_audio_feature_keys = get_required_audio_feature_keys(configs)
    stft_tool = build_stft_tool(configs) if required_audio_feature_keys else None
    model = load_model(configs, ckpt_path, args.device)

    eval_params = configs.get("model", {}).get("params", {}).get("evaluation_params", {})
    ddim_steps = args.ddim_steps if args.ddim_steps is not None else eval_params.get("ddim_sampling_steps", 10)
    guidance_scale = args.guidance_scale if args.guidance_scale is not None else eval_params.get("unconditional_guidance_scale", 1.0)
    pred_sr = int(getattr(model, "sampling_rate", configs.get("model", {}).get("params", {}).get("sampling_rate", target_sr)))

    sample_dirs = list_sample_dirs(test_cases_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for sample_dir in tqdm(sample_dirs, desc="Inference", unit="sample"):
        try:
            batch = build_model_batch(sample_dir, configs, stft_tool, target_sr)
            sample_output_dir = output_dir / sample_dir.name
            shutil.copytree(sample_dir, sample_output_dir, dirs_exist_ok=True)
            pred_waveform = infer_single_sample(model, batch, sample_dir.name, ddim_steps, guidance_scale)
            sf.write(str(sample_output_dir / "infer.wav"), format_waveform_for_save(pred_waveform), pred_sr)
        except Exception as e:
            raise RuntimeError(f"Failed on sample '{sample_dir.name}': {e}") from e

    print(f"Processed samples: {len(sample_dirs)}")
    print(f"Output dir: {output_dir}")
    print(f"Inference sample rate: {pred_sr}")


if __name__ == "__main__":
    main()
