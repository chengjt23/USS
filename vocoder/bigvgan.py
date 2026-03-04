import os
import json
import torch
import bigvgan


def get_vocoder(config, device, mel_bins):
    root = os.getenv("BIGVGAN_PATH", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bigvgan"))
    with open(os.path.join(root, "config.json"), "r") as f:
        config = json.load(f)
    config = bigvgan.AttrDict(config)
    vocoder = bigvgan.BigVGAN(config)
    ckpt = torch.load(os.path.join(root, "g_01000000"), map_location="cpu")
    vocoder.load_state_dict(ckpt["generator"])
    vocoder.eval()
    vocoder.remove_weight_norm()
    vocoder.to(device)
    return vocoder


def vocoder_infer(mels, vocoder, lengths=None):
    with torch.no_grad():
        wavs = vocoder(mels).squeeze(1)
    wavs = (wavs.cpu().numpy() * 32768).astype("int16")
    if lengths is not None:
        wavs = wavs[:, :lengths]
    return wavs


def synth_one_sample(mel_input, mel_prediction, labels, vocoder):
    if vocoder is not None:
        wav_reconstruction = vocoder_infer(mel_input.permute(0, 2, 1), vocoder)
        wav_prediction = vocoder_infer(mel_prediction.permute(0, 2, 1), vocoder)
    else:
        wav_reconstruction = wav_prediction = None
    return wav_reconstruction, wav_prediction
