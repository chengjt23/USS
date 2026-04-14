import json
import os

import dacvae
import torch
import torch.nn as nn
import torchaudio
from huggingface_hub import snapshot_download


class DACVAEFirstStage1D(nn.Module):
    def __init__(
        self,
        encoder_dim=64,
        encoder_rates=[2, 8, 10, 12],
        latent_dim=1024,
        decoder_dim=1536,
        decoder_rates=[12, 10, 8, 2],
        n_codebooks=16,
        codebook_size=1024,
        codebook_dim=128,
        quantizer_dropout=False,
        sample_rate=48000,
        data_sample_rate=44100,
        duration=4.0,
        ckpt_path=None,
        pretrained_model_name_or_path=None,
        cache_dir=None,
    ):
        super().__init__()
        pretrained_dir = None
        if pretrained_model_name_or_path is not None:
            pretrained_dir = self._resolve_pretrained_dir(pretrained_model_name_or_path, cache_dir)
            codec_config = self._load_pretrained_config(pretrained_dir)
            encoder_dim = codec_config.get("encoder_dim", encoder_dim)
            encoder_rates = codec_config.get("encoder_rates", encoder_rates)
            latent_dim = codec_config.get("latent_dim", latent_dim)
            decoder_dim = codec_config.get("decoder_dim", decoder_dim)
            decoder_rates = codec_config.get("decoder_rates", decoder_rates)
            n_codebooks = codec_config.get("n_codebooks", n_codebooks)
            codebook_size = codec_config.get("codebook_size", codebook_size)
            codebook_dim = codec_config.get("codebook_dim", codebook_dim)
            quantizer_dropout = codec_config.get("quantizer_dropout", quantizer_dropout)
            sample_rate = codec_config.get("sample_rate", sample_rate)
        model = dacvae.DACVAE(
            encoder_dim=encoder_dim,
            encoder_rates=encoder_rates,
            latent_dim=latent_dim,
            decoder_dim=decoder_dim,
            decoder_rates=decoder_rates,
            n_codebooks=n_codebooks,
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
            quantizer_dropout=quantizer_dropout,
            sample_rate=sample_rate,
        ).eval()
        self.encoder = model.encoder
        self.quantizer = model.quantizer
        self.decoder = model.decoder
        self.hop_length = 1
        for rate in encoder_rates:
            self.hop_length *= rate
        self.codec_sr = sample_rate
        self.data_sr = data_sample_rate
        self.feature_dim = codebook_dim
        self.max_samples = int(duration * data_sample_rate)
        if pretrained_dir is not None:
            self._load_ckpt(os.path.join(pretrained_dir, "checkpoint.pt"))
        elif ckpt_path is not None:
            self._load_ckpt(ckpt_path)

    def _resolve_pretrained_dir(self, model_name_or_path, cache_dir=None):
        if os.path.isdir(model_name_or_path):
            return model_name_or_path
        return snapshot_download(repo_id=model_name_or_path, cache_dir=cache_dir)

    def _load_pretrained_config(self, model_dir):
        with open(os.path.join(model_dir, "config.json"), "r", encoding="utf-8") as f:
            config = json.load(f)
        return config.get("audio_codec", {})

    def _load_ckpt(self, path):
        state_dict = torch.load(path, map_location="cpu")
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        prefix = "audio_codec."
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith(prefix):
                new_state_dict[key[len(prefix):]] = value
        if new_state_dict:
            self.load_state_dict(new_state_dict, strict=False)
        else:
            self.load_state_dict(state_dict, strict=False)

    def _pad(self, wavs):
        length = wavs.size(-1)
        if length % self.hop_length:
            pad = self.hop_length - (length % self.hop_length)
            return torch.nn.functional.pad(wavs, (0, pad), "reflect")
        return wavs

    def _resample(self, wav, from_sr, to_sr):
        if from_sr == to_sr:
            return wav
        return torchaudio.functional.resample(wav, from_sr, to_sr)

    def _truncate_pad(self, wav):
        length = wav.size(-1)
        if length > self.max_samples:
            wav = wav[..., : self.max_samples]
        elif length < self.max_samples:
            wav = torch.nn.functional.pad(wav, (0, self.max_samples - length))
        return wav

    def encode(self, waveform):
        waveform = self._truncate_pad(waveform)
        with torch.no_grad(), torch.backends.cudnn.flags(enabled=False):
            wav = self._resample(waveform, self.data_sr, self.codec_sr)
            z = self.encoder(self._pad(wav))
            mean, _ = self.quantizer.in_proj(z).chunk(2, dim=1)
        if mean.shape[1] != self.feature_dim:
            self.feature_dim = mean.shape[1]
        return mean

    def decode(self, z):
        if z.dim() != 3:
            raise RuntimeError(f"Unexpected latent rank {z.dim()}; expected 3")
        if z.shape[1] != self.feature_dim:
            raise RuntimeError(
                f"Unexpected latent channel count {z.shape[1]}; expected {self.feature_dim}"
            )
        with torch.no_grad(), torch.backends.cudnn.flags(enabled=False):
            emb = self.quantizer.out_proj(z)
            wav = self.decoder(emb)
        return self._resample(wav, self.codec_sr, self.data_sr)
