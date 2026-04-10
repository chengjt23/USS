import argparse
import io
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
import webdataset as wds
import yaml

SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
USS_ROOT = SCRIPT_PATH.parents[2]
DEFAULT_CONFIG = USS_ROOT / "configs" / "bridgesep" / "hive_2mix" / "flowsep_2mix.yaml"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "bad_infer_outputs"

if str(USS_ROOT) not in sys.path:
    sys.path.insert(0, str(USS_ROOT))

from data.wds_datamodule import decode_sample_to_tensors, wds_collate_fn
from models.flowsep.model import DDPM
from train import build_stft_tool, convert_wds_batch_to_model_format
from utils.tools import instantiate_from_config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("-c", "--config_yaml", type=str, default=str(DEFAULT_CONFIG))
    parser.add_argument("--tar_path", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_root", type=str, default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--save_sr", type=int, default=44100)
    parser.add_argument("--save_duration", type=float, default=4.0)
    parser.add_argument("--ddim_steps", type=int, default=None)
    parser.add_argument("--guidance_scale", type=float, default=None)
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


def choose_tar(configs, tar_path, seed: int):
    if tar_path:
        path = Path(tar_path)
        if not path.exists():
            raise FileNotFoundError(f"tar_path does not exist: {path}")
        return path
    rng = random.Random(seed)
    return rng.choice(collect_val_tars(configs))


def load_audio_bytes(audio_bytes: bytes):
    wav, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    return torch.from_numpy(wav), int(sr)


def sanitize_name(name: str):
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name)


def load_valid_samples_from_tar(tar_path: Path):
    dataset = wds.DataPipeline(
        wds.SimpleShardList([str(tar_path)]),
        wds.tarfile_to_samples(handler=wds.warn_and_continue),
    )
    samples = []
    for sample in dataset:
        mix_bytes = sample.get("mix.wav") or sample.get("mix")
        source_keys = sorted(
            k for k in sample.keys()
            if isinstance(k, str) and k.startswith("s") and k.endswith(".wav")
        )
        if mix_bytes is None or len(source_keys) < 2:
            continue
        samples.append(sample)
    if not samples:
        raise RuntimeError(f"No valid 2-source samples found in tar: {tar_path}")
    return samples


def select_samples(samples, num_samples: int, seed: int):
    rng = random.Random(seed)
    if len(samples) <= num_samples:
        chosen = list(samples)
        rng.shuffle(chosen)
        return chosen
    return rng.sample(samples, num_samples)


def build_wrong_infer_batch(selected_samples, configs):
    target_sr = int(configs["datamodule"]["data_config"]["sample_rate"])
    raw_items = []
    infos = []
    for sample in selected_samples:
        mix_bytes = sample.get("mix.wav") or sample.get("mix")
        source_keys = sorted(
            k for k in sample.keys()
            if isinstance(k, str) and k.startswith("s") and k.endswith(".wav")
        )
        mix_wav, mix_sr = load_audio_bytes(mix_bytes)
        s1_wav, s1_sr = load_audio_bytes(sample[source_keys[0]])
        s2_wav, s2_sr = load_audio_bytes(sample[source_keys[1]])
        mix_tensor, sources_tensor, labels = decode_sample_to_tensors(sample, target_sr)
        raw_items.append((mix_tensor, sources_tensor, labels, {"sample_key": sample.get("__key__", "sample")}))
        infos.append({
            "sample_key": sample.get("__key__", f"sample_{len(infos)}"),
            "mix_wav": mix_wav,
            "mix_sr": mix_sr,
            "s1_wav": s1_wav,
            "s1_sr": s1_sr,
            "s2_wav": s2_wav,
            "s2_sr": s2_sr,
        })
    raw_batch = wds_collate_fn(raw_items)
    stft_tool = build_stft_tool(configs)
    model_batch = convert_wds_batch_to_model_format(raw_batch, configs, stft_tool)
    for i, info in enumerate(infos):
        model_batch["fname"][i] = f"{sanitize_name(info['sample_key'])}.wav"
    return model_batch, infos


def load_model(configs, ckpt_path: str, device: str):
    if "precision" in configs:
        torch.set_float32_matmul_precision(configs["precision"])
    model = instantiate_from_config(configs["model"]).to(device).eval()
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=False)
    return model


def infer_waveforms(model, batch, ddim_steps: int, guidance_scale: float):
    with torch.no_grad(), model.ema_scope("sr_test"):
        z, c = model.get_input(batch, model.first_stage_key, unconditional_prob_cfg=0.0)
        if model.condition_key:
            c = model.filter_useful_cond_dict(c)
        batch_size = z.shape[0]
        x_T = None
        if model.extra_channels:
            extra = DDPM.get_input(model, batch, model.extra_channel_key).to(model.device)
            extra = extra.reshape(extra.shape[0], 1, extra.shape[1], extra.shape[2])
            extra_posterior = model.encode_first_stage(extra)
            x_T = model.get_first_stage_encoding(extra_posterior).detach()
        unconditional_conditioning = None
        if model.condition_key and guidance_scale != 1.0:
            unconditional_conditioning = {}
            for key in model.cond_stage_model_metadata:
                model_idx = model.cond_stage_model_metadata[key]["model_idx"]
                unconditional_conditioning[key] = model.cond_stage_models[model_idx].get_unconditional_condition(batch_size)
        samples, _ = model.sample_log(
            cond=c,
            batch_size=batch_size,
            x_T=x_T,
            ddim=ddim_steps is not None,
            ddim_steps=ddim_steps,
            eta=1.0,
            unconditional_guidance_scale=guidance_scale,
            unconditional_conditioning=unconditional_conditioning,
            use_plms=False,
        )
        if model.extra_channels:
            samples = samples[:, :model.channels, :, :]
        mel = model.decode_first_stage(samples)
        if model.fbank_shift:
            mel = mel - model.fbank_shift
        if model.data_std:
            mel = (mel * model.data_std) + model.data_mean
        waveform = model.mel_spectrogram_to_waveform(mel, save=False)
    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.ndim == 3:
        waveform = waveform.squeeze(1)
    return waveform


def resample_trim_pad(wav: torch.Tensor, src_sr: int, dst_sr: int, duration: float):
    if wav.dim() != 1:
        wav = wav.reshape(-1)
    if src_sr != dst_sr:
        wav = torchaudio.functional.resample(wav.unsqueeze(0), src_sr, dst_sr).squeeze(0)
    target_len = int(round(dst_sr * duration))
    if wav.numel() >= target_len:
        wav = wav[:target_len]
    else:
        wav = torch.nn.functional.pad(wav, (0, target_len - wav.numel()))
    return wav.cpu().numpy().astype(np.float32)


def save_audio(path: Path, wav: torch.Tensor, src_sr: int, dst_sr: int, duration: float):
    audio = resample_trim_pad(wav, src_sr, dst_sr, duration)
    sf.write(str(path), audio, dst_sr)


def save_predictions(output_dir: Path, infos, pred_waveforms, pred_sr: int, save_sr: int, save_duration: float):
    for idx, info in enumerate(infos):
        sample_dir = output_dir / f"{idx:02d}_{sanitize_name(info['sample_key'])}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        save_audio(sample_dir / "mix.wav", info["mix_wav"], info["mix_sr"], save_sr, save_duration)
        save_audio(sample_dir / "s1.wav", info["s1_wav"], info["s1_sr"], save_sr, save_duration)
        save_audio(sample_dir / "s2.wav", info["s2_wav"], info["s2_sr"], save_sr, save_duration)
        pred_tensor = torch.from_numpy(pred_waveforms[idx]).float()
        save_audio(sample_dir / "infer.wav", pred_tensor, pred_sr, save_sr, save_duration)


def main():
    args = parse_args()
    configs = load_config(args.config_yaml)
    tar_path = choose_tar(configs, args.tar_path, args.seed)
    all_samples = load_valid_samples_from_tar(tar_path)
    selected_samples = select_samples(all_samples, args.num_samples, args.seed)
    model_batch, infos = build_wrong_infer_batch(selected_samples, configs)
    model = load_model(configs, args.ckpt_path, args.device)

    eval_params = configs.get("model", {}).get("params", {}).get("evaluation_params", {})
    ddim_steps = args.ddim_steps if args.ddim_steps is not None else eval_params.get("ddim_sampling_steps", 10)
    guidance_scale = args.guidance_scale if args.guidance_scale is not None else eval_params.get("unconditional_guidance_scale", 1.0)

    pred_waveforms = infer_waveforms(model, model_batch, ddim_steps, guidance_scale)
    pred_sr = int(getattr(model, "sampling_rate", configs["model"]["params"].get("sampling_rate", 16000)))

    ckpt_stem = sanitize_name(Path(args.ckpt_path).stem)
    output_dir = Path(args.output_root) / f"{ckpt_stem}_{sanitize_name(tar_path.stem)}"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_predictions(output_dir, infos, pred_waveforms, pred_sr, args.save_sr, args.save_duration)

    print(f"Selected tar: {tar_path}")
    print(f"Available valid samples in tar: {len(all_samples)}")
    print(f"Saved sample count: {len(infos)}")
    print(f"Inference sr: {pred_sr}")
    print(f"Save sr: {args.save_sr}")
    print(f"Save duration: {args.save_duration}")
    print(f"Output dir: {output_dir}")


if __name__ == "__main__":
    main()
