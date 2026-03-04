import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import yaml
import torch
import torchaudio
import numpy as np
from utils.tools import instantiate_from_config
from utils.audio import TacotronSTFT, get_mel_from_wav
from pytorch_lightning import seed_everything


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


def main(configs, exp_group_name, exp_name, text, wav, load_checkpoint, infer_step, save_mixed):
    seed_everything(0)

    latent_diffusion = instantiate_from_config(configs["model"]).to("cuda")

    ckpt = torch.load(load_checkpoint, map_location="cuda")["state_dict"]
    latent_diffusion.load_state_dict(ckpt, strict=True)

    stft_tool = build_stft_tool(configs)
    sampling_rate = configs["preprocessing"]["audio"]["sampling_rate"]
    hopsize = configs["preprocessing"]["stft"]["hop_length"]
    duration = configs["preprocessing"]["audio"]["duration"]
    target_length = int(duration * sampling_rate / hopsize)

    count = 0
    for cur_text in text:
        cur_wav = wav[count]
        
        waveform, sr = torchaudio.load(cur_wav)
        if sr != sampling_rate:
            waveform = torchaudio.functional.resample(waveform, sr, sampling_rate)
        
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        max_samples = int(duration * sampling_rate)
        if waveform.shape[1] > max_samples:
            waveform = waveform[:, :max_samples]
        elif waveform.shape[1] < max_samples:
            waveform = torch.nn.functional.pad(waveform, (0, max_samples - waveform.shape[1]))

        mixed_mel, stft = wav_feature_extraction(waveform.unsqueeze(0), stft_tool)
        mixed_mel = pad_spec(mixed_mel, target_length)
        stft = pad_spec(stft, target_length)

        batch = {}
        batch["fname"] = [cur_wav]
        batch["text"] = [cur_text]
        batch["caption"] = [cur_text]
        batch["waveform"] = waveform.unsqueeze(0).cuda()
        batch["log_mel_spec"] = torch.rand(1, target_length, configs["preprocessing"]["mel"]["n_mel_channels"]).cuda()
        batch["sampling_rate"] = torch.tensor([sampling_rate]).cuda()
        batch["label_vector"] = torch.zeros(1, 0).cuda()
        batch["stft"] = stft.cuda()
        batch["mixed_waveform"] = waveform.unsqueeze(0).cuda()
        batch["mixed_mel"] = mixed_mel.unsqueeze(1).cuda()

        latent_diffusion.generate_sample(
            [batch],
            name="result",
            unconditional_guidance_scale=1.0,
            ddim_steps=infer_step,
            n_gen=1,
            save_mixed=save_mixed
        )
        count += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_yaml", type=str, default="configs/flowsep/flowsep.yaml")
    parser.add_argument("-t", "--text", type=str, default="A rocket flies by followed by a loud explosion")
    parser.add_argument("-a", "--audio", type=str, default="metadata/mixed/sample.wav")
    parser.add_argument("-l", "--load_checkpoint", type=str, default="model_logs/pretrained/flowsep.ckpt")
    parser.add_argument("-s", "--infer_step", type=int, default=20)
    parser.add_argument("--no_mixed", action="store_true", help="do not save mixed waveform")
    args = parser.parse_args()

    assert torch.cuda.is_available(), "CUDA is not available"

    exp_name = os.path.basename(args.config_yaml.split(".")[0])
    exp_group_name = os.path.basename(os.path.dirname(args.config_yaml))

    with open(args.config_yaml, "r") as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)

    main(
        configs,
        exp_group_name,
        exp_name,
        [args.text],
        [args.audio],
        args.load_checkpoint,
        args.infer_step,
        not args.no_mixed,
    )
