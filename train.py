import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,6,7"
import argparse
import yaml
import torch
import numpy as np
import math
import warnings
import logging
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from data import build_datamodule
from utils.audio import TacotronSTFT, get_mel_from_wav
from utils.tools import get_restore_step, instantiate_from_config

logging.getLogger('fsspec').setLevel(logging.ERROR)
logging.getLogger('numba').setLevel(logging.WARNING)


def configure_runtime_warnings():
    warnings.filterwarnings(
        "ignore",
        message=r"Grad strides do not match bucket view strides\.",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"The AccumulateGrad node's stream does not match the stream of the node that produced the incoming gradient\.",
        category=UserWarning,
    )
    if hasattr(torch.autograd, "graph") and hasattr(torch.autograd.graph, "set_warn_on_accumulate_grad_stream_mismatch"):
        torch.autograd.graph.set_warn_on_accumulate_grad_stream_mismatch(False)


def build_stft_tool(config):
    return TacotronSTFT(
        config["preprocessing"]["stft"]["filter_length"],
        config["preprocessing"]["stft"]["hop_length"],
        config["preprocessing"]["stft"]["win_length"],
        config["preprocessing"]["mel"]["n_mel_channels"],
        config["preprocessing"]["audio"]["sampling_rate"],
        config["preprocessing"]["mel"]["mel_fmin"],
        config["preprocessing"]["mel"]["mel_fmax"],
    )


def get_required_audio_feature_keys(config):
    model_params = config.get("model", {}).get("params", {})
    keys = {model_params.get("first_stage_key"), model_params.get("extra_channel_key")}
    return {k for k in keys if k in {"fbank", "stft", "mixed_mel"}}


def wav_feature_extraction(waveform, stft_tool):
    if waveform.dim() == 3:
        waveform = waveform.squeeze(1)
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    log_mel_spec, stft, energy = get_mel_from_wav(waveform, stft_tool, return_numpy=False)
    log_mel_spec = log_mel_spec.transpose(1, 2).contiguous().float()
    stft = stft.transpose(1, 2).contiguous().float()
    return log_mel_spec, stft


def pad_spec(spec, target_length):
    n_frames = spec.shape[1]
    p = target_length - n_frames
    if p > 0:
        spec = torch.nn.ZeroPad2d((0, 0, 0, p))(spec)
    elif p < 0:
        spec = spec[:, :target_length, :]
    if spec.size(-1) % 2 != 0:
        spec = spec[..., :-1]
    return spec


def convert_wds_batch_to_model_format(batch, config, stft_tool):
    mix_audio = batch["mix"]
    sources_audio = batch["sources"]
    labels = batch["labels"]
    batch_size = mix_audio.shape[0]
    sample_metadata = batch.get("metadata", [{} for _ in range(batch_size)])
    required_audio_feature_keys = get_required_audio_feature_keys(config)

    if sources_audio.ndim == 4 and sources_audio.shape[1] > 0:
        target_audio = sources_audio[:, 0, :, :]
    else:
        target_audio = mix_audio

    if target_audio.shape[1] > 1:
        target_audio = target_audio.mean(dim=1, keepdim=True)
    if mix_audio.shape[1] > 1:
        mix_audio_mono = mix_audio.mean(dim=1, keepdim=True)
    else:
        mix_audio_mono = mix_audio

    sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
    hopsize = config["preprocessing"]["stft"]["hop_length"]
    duration = config["preprocessing"]["audio"]["duration"]
    target_length = int(duration * sampling_rate / hopsize)

    log_mel_spec = None
    stft = None
    mixed_mel = None
    if required_audio_feature_keys:
        target_log_mel_spec, target_stft = wav_feature_extraction(target_audio, stft_tool)
        if "fbank" in required_audio_feature_keys:
            log_mel_spec = pad_spec(target_log_mel_spec, target_length)
        if "stft" in required_audio_feature_keys:
            stft = pad_spec(target_stft, target_length)
        if "mixed_mel" in required_audio_feature_keys:
            mixed_mel, _ = wav_feature_extraction(mix_audio_mono, stft_tool)
            mixed_mel = pad_spec(mixed_mel, target_length)

    text_list = []
    for i in range(batch_size):
        if len(labels[i]) > 0:
            text = labels[i][0] if isinstance(labels[i][0], str) else ""
        else:
            text = ""
        text_list.append(text)

    return {
        "fname": [f"batch_item_{i}.wav" for i in range(batch_size)],
        "text": text_list,
        "label_vector": torch.zeros(batch_size, 0),
        "waveform": target_audio,
        "stft": stft,
        "log_mel_spec": log_mel_spec,
        "duration": duration,
        "sampling_rate": sampling_rate,
        "random_start_sample_in_original_audio_file": 0,
        "mixed_waveform": mix_audio_mono,
        "mixed_mel": mixed_mel,
        "caption": text_list,
        "sample_metadata": sample_metadata,
    }


class WrappedDataLoader:
    def __init__(self, base_loader, config, stft_tool, prefetch_batches=8, explicit_length=None):
        self.base_loader = base_loader
        self.config = config
        self.stft_tool = stft_tool
        self.prefetch_batches = prefetch_batches
        self.explicit_length = explicit_length

    def _pin_memory(self, item):
        if not torch.cuda.is_available():
            return item
        if torch.is_tensor(item):
            if item.device.type == "cpu":
                return item.pin_memory()
            return item
        if isinstance(item, dict):
            return {k: self._pin_memory(v) for k, v in item.items()}
        if isinstance(item, list):
            return [self._pin_memory(v) for v in item]
        if isinstance(item, tuple):
            return tuple(self._pin_memory(v) for v in item)
        return item

    def __iter__(self):
        import threading, queue
        q = queue.Queue(maxsize=self.prefetch_batches)
        sentinel = object()

        def _producer():
            try:
                for raw in self.base_loader:
                    batch = convert_wds_batch_to_model_format(raw, self.config, self.stft_tool)
                    q.put(self._pin_memory(batch))
            except Exception as e:
                print(f"[WrappedDataLoader] producer error: {e}")
            finally:
                q.put(sentinel)

        t = threading.Thread(target=_producer, daemon=True)
        t.start()
        while True:
            item = q.get()
            if item is sentinel:
                break
            yield item
        t.join()

    def __len__(self):
        if self.explicit_length is not None:
            return self.explicit_length
        return len(self.base_loader)


def main(configs, config_yaml_path, exp_group_name, exp_name):
    configure_runtime_warnings()
    if "seed" in configs.keys():
        seed_everything(configs["seed"])
    else:
        seed_everything(0)
    if "precision" in configs.keys():
        torch.set_float32_matmul_precision(configs["precision"])
    else:
        torch.set_float32_matmul_precision("medium")

    log_path = configs["log_directory"]
    exp_group_name = configs["exp_group"]
    exp_name = configs["exp_name"]
    batch_size = configs["model"]["params"]["batchsize"]

    datamodule = build_datamodule(configs)
    train_loader, val_loader, test_loader = datamodule.make_loader
    device_count = torch.cuda.device_count()

    data_config = configs.get("datamodule", {}).get("data_config", {})
    val_progress_total = None
    val_tar_count = data_config.get("val_tar_count")
    val_samples_per_tar = data_config.get("val_samples_per_tar")
    if val_loader is not None and val_tar_count is not None and val_samples_per_tar is not None:
        mix_selected = data_config.get("mix_selected")
        selected_dir_count = len(mix_selected) if mix_selected is not None else 1
        total_selected_tars = val_tar_count * selected_dir_count
        local_tar_count = math.ceil(total_selected_tars / max(device_count, 1))
        val_batch_size = data_config.get("val_batch_size", data_config.get("batch_size", 1))
        val_progress_total = math.ceil(local_tar_count * val_samples_per_tar / val_batch_size)

    required_audio_feature_keys = get_required_audio_feature_keys(configs)
    stft_tool = build_stft_tool(configs) if required_audio_feature_keys else None
    loader = WrappedDataLoader(train_loader, configs, stft_tool)
    val_loader = WrappedDataLoader(val_loader, configs, stft_tool, explicit_length=val_progress_total)
    config_reload_from_ckpt = configs.get("reload_from_ckpt")
    limit_val_batches = configs.get("step", {}).get("limit_val_batches")
    limit_train_batches = configs.get("step", {}).get("limit_train_batches", 10000)

    save_checkpoint_every_n_steps = configs["step"]["save_checkpoint_every_n_steps"]
    max_steps = configs["step"]["max_steps"]
    save_top_k = configs["step"]["save_top_k"]

    checkpoint_path = os.path.join(log_path, exp_group_name, exp_name, "checkpoints")
    wandb_path = os.path.join(log_path, exp_group_name, exp_name)

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_path,
        monitor="global_step",
        mode="max",
        filename="checkpoint-global_step={global_step:.0f}",
        every_n_train_steps=save_checkpoint_every_n_steps,
        save_top_k=save_top_k,
        auto_insert_metric_name=False,
        save_last=True,
    )

    os.makedirs(checkpoint_path, exist_ok=True)

    if len(os.listdir(checkpoint_path)) > 0:
        restore_step, _ = get_restore_step(checkpoint_path)
        resume_from_checkpoint = os.path.join(checkpoint_path, restore_step)
    elif config_reload_from_ckpt is not None:
        resume_from_checkpoint = config_reload_from_ckpt
    else:
        resume_from_checkpoint = None

    latent_diffusion = instantiate_from_config(configs["model"])
    if hasattr(latent_diffusion, 'set_log_dir'):
        latent_diffusion.set_log_dir(log_path, exp_group_name, exp_name)

    wandb_logger = WandbLogger(project=exp_group_name, name=exp_name, save_dir=wandb_path, config=configs)

    trainer = Trainer(
        accelerator="gpu",
        devices=device_count,
        logger=wandb_logger,
        max_steps=max_steps,
        num_sanity_val_steps=2,
        limit_val_batches=limit_val_batches,
        limit_train_batches=limit_train_batches,
        precision="bf16-mixed",
        strategy=DDPStrategy(find_unused_parameters=True),
        callbacks=[checkpoint_callback],
    )

    trainer.fit(latent_diffusion, loader, val_loader, ckpt_path=resume_from_checkpoint)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_yaml", type=str, default="configs/flowsep/flowsep.yaml", help="path to config")
    parser.add_argument("--panns_ckpt_path", type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "metrics", "fad", "Cnn14_16k_mAP=0.438.pth"))
    parser.add_argument("--clap_ckpt_path", type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "metrics", "clapscore", "music_speech_audioset_epoch_15_esc_89.98.pt"))
    args = parser.parse_args()

    assert torch.cuda.is_available(), "CUDA is not available"

    config_yaml_path = args.config_yaml
    exp_name = os.path.basename(config_yaml_path.split(".")[0])
    exp_group_name = os.path.basename(os.path.dirname(config_yaml_path))

    with open(config_yaml_path, "r") as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)

    configs["model"]["params"]["panns_ckpt_path"] = args.panns_ckpt_path
    configs["model"]["params"]["clap_ckpt_path"] = args.clap_ckpt_path
    main(configs, config_yaml_path, exp_group_name, exp_name)
