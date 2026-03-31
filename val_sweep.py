import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
import yaml
import torch
import logging
from itertools import product
from pytorch_lightning import Trainer, seed_everything

from data.wds_datamodule import WDSDataModule
from utils.audio import TacotronSTFT, get_mel_from_wav
from utils.tools import instantiate_from_config

logging.getLogger('fsspec').setLevel(logging.ERROR)
logging.getLogger('numba').setLevel(logging.WARNING)


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


def wav_feature_extraction(waveform, stft_tool):
    if waveform.dim() == 3:
        waveform = waveform.squeeze(1)
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    log_mel_specs, stfts = [], []
    for i in range(waveform.shape[0]):
        wav_tensor = torch.FloatTensor(waveform[i].numpy() if isinstance(waveform[i], torch.Tensor) else waveform[i])
        log_mel_spec, stft, energy = get_mel_from_wav(wav_tensor, stft_tool)
        log_mel_specs.append(torch.FloatTensor(log_mel_spec.T))
        stfts.append(torch.FloatTensor(stft.T))
    return torch.stack(log_mel_specs, dim=0), torch.stack(stfts, dim=0)


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

    log_mel_spec, stft = wav_feature_extraction(target_audio, stft_tool)
    log_mel_spec = pad_spec(log_mel_spec, target_length)
    stft = pad_spec(stft, target_length)

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
    }


class WrappedDataLoader:
    def __init__(self, base_loader, config, stft_tool):
        self.base_loader = base_loader
        self.config = config
        self.stft_tool = stft_tool

    def __iter__(self):
        import threading, queue
        q = queue.Queue(maxsize=2)
        sentinel = object()

        def _producer():
            try:
                for raw in self.base_loader:
                    q.put(convert_wds_batch_to_model_format(raw, self.config, self.stft_tool))
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
        return len(self.base_loader)


def parse_float_list(s):
    return [float(x) for x in s.split(",")]


def parse_int_list(s):
    return [int(x) for x in s.split(",")]


def main(configs, ckpt_path, guidance_scales, ddim_steps_list, limit_val_batches):
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

    datamodule = WDSDataModule(**configs["datamodule"]["data_config"])
    _, val_loader, _ = datamodule.make_loader

    stft_tool = build_stft_tool(configs)
    val_loader = WrappedDataLoader(val_loader, configs, stft_tool)

    if limit_val_batches is None:
        limit_val_batches = configs.get("step", {}).get("limit_val_batches")

    model = instantiate_from_config(configs["model"])
    if hasattr(model, 'set_log_dir'):
        model.set_log_dir(log_path, exp_group_name, exp_name)

    combos = list(product(guidance_scales, ddim_steps_list))
    all_results = []
    first_run = True

    for guidance_scale, ddim_steps in combos:
        model.evaluation_params["unconditional_guidance_scale"] = guidance_scale
        model.evaluation_params["ddim_sampling_steps"] = ddim_steps

        print(f"\n>>> guidance_scale={guidance_scale}, ddim_steps={ddim_steps}")

        trainer = Trainer(
            accelerator="gpu",
            devices=1,
            limit_val_batches=limit_val_batches,
            precision="bf16-mixed",
            enable_progress_bar=True,
        )

        if first_run:
            results = trainer.validate(model, val_loader, ckpt_path=ckpt_path)
            first_run = False
        else:
            results = trainer.validate(model, val_loader)

        metrics = results[0] if results else {}
        all_results.append({
            "guidance_scale": guidance_scale,
            "ddim_steps": ddim_steps,
            **metrics,
        })

    print("\n" + "=" * 80)
    print("SWEEP RESULTS SUMMARY")
    print("=" * 80)

    if all_results:
        metric_keys = [k for k in all_results[0].keys() if k not in ("guidance_scale", "ddim_steps")]
        header = f"{'guidance':>10} {'ddim_steps':>12}" + "".join(f"  {k:>20}" for k in metric_keys)
        print(header)
        print("-" * len(header))
        for row in all_results:
            line = f"{row['guidance_scale']:>10.2f} {row['ddim_steps']:>12d}"
            for k in metric_keys:
                v = row.get(k, float("nan"))
                line += f"  {v:>20.4f}"
            print(line)

    print("=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_yaml", type=str, required=True)
    parser.add_argument("-k", "--ckpt_path", type=str, required=True)
    parser.add_argument("--guidance_scales", type=str, default="1.0,1.5,2.0,3.0,5.0",
                        help="comma-separated list of unconditional_guidance_scale values")
    parser.add_argument("--ddim_steps", type=str, default="5,10,20,50",
                        help="comma-separated list of ddim_sampling_steps values")
    parser.add_argument("--limit_val_batches", type=int, default=None)
    parser.add_argument("--panns_ckpt_path", type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "metrics", "fad", "Cnn14_16k_mAP=0.438.pth"))
    parser.add_argument("--clap_ckpt_path", type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "metrics", "clapscore", "music_speech_audioset_epoch_15_esc_89.98.pt"))
    args = parser.parse_args()

    assert torch.cuda.is_available(), "CUDA is not available"

    with open(args.config_yaml, "r") as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)

    configs["model"]["params"]["panns_ckpt_path"] = args.panns_ckpt_path
    configs["model"]["params"]["clap_ckpt_path"] = args.clap_ckpt_path

    main(
        configs,
        args.ckpt_path,
        parse_float_list(args.guidance_scales),
        parse_int_list(args.ddim_steps),
        args.limit_val_batches,
    )
