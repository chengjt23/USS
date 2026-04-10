import torch
import torch.nn as nn
import torchaudio
import dacvae


class DACVAEFirstStage(nn.Module):
    def __init__(self, encoder_dim=64, encoder_rates=[2, 8, 10, 12],
                 latent_dim=1024, decoder_dim=1536, decoder_rates=[12, 10, 8, 2],
                 n_codebooks=16, codebook_size=1024, codebook_dim=128,
                 quantizer_dropout=False, sample_rate=48000,
                 reshape_channels=8, data_sample_rate=44100, ckpt_path=None):
        super().__init__()
        model = dacvae.DACVAE(
            encoder_dim=encoder_dim, encoder_rates=encoder_rates,
            latent_dim=latent_dim, decoder_dim=decoder_dim,
            decoder_rates=decoder_rates, n_codebooks=n_codebooks,
            codebook_size=codebook_size, codebook_dim=codebook_dim,
            quantizer_dropout=quantizer_dropout, sample_rate=sample_rate,
        ).eval()
        self.encoder = model.encoder
        self.quantizer = model.quantizer
        self.decoder = model.decoder
        self.hop_length = 1
        for r in encoder_rates:
            self.hop_length *= r
        self.codec_sr = sample_rate
        self.data_sr = data_sample_rate
        self.reshape_channels = reshape_channels
        self.freq_dim = latent_dim // reshape_channels
        if ckpt_path is not None:
            self._load_ckpt(ckpt_path)

    def _load_ckpt(self, path):
        sd = torch.load(path, map_location="cpu")
        if "state_dict" in sd:
            sd = sd["state_dict"]
        prefix = "audio_codec."
        new_sd = {}
        for k, v in sd.items():
            if k.startswith(prefix):
                new_sd[k[len(prefix):]] = v
        if new_sd:
            self.load_state_dict(new_sd, strict=False)
        else:
            self.load_state_dict(sd, strict=False)

    def _pad(self, wavs):
        length = wavs.size(-1)
        if length % self.hop_length:
            p = self.hop_length - (length % self.hop_length)
            return torch.nn.functional.pad(wavs, (0, p), "reflect")
        return wavs

    def _resample(self, wav, from_sr, to_sr):
        if from_sr == to_sr:
            return wav
        return torchaudio.functional.resample(wav, from_sr, to_sr)

    def encode(self, waveform):
        with torch.no_grad(), torch.backends.cudnn.flags(enabled=False):
            wav = self._resample(waveform, self.data_sr, self.codec_sr)
            z = self.encoder(self._pad(wav))
            mean, _ = self.quantizer.in_proj(z).chunk(2, dim=1)
        B, D, T = mean.shape
        return mean.reshape(B, self.reshape_channels, self.freq_dim, T).permute(0, 1, 3, 2)

    def decode(self, z):
        B, C, T, F = z.shape
        latent = z.permute(0, 1, 3, 2).reshape(B, C * F, T)
        with torch.no_grad(), torch.backends.cudnn.flags(enabled=False):
            emb = self.quantizer.out_proj(latent)
            wav = self.decoder(emb)
        return self._resample(wav, self.codec_sr, self.data_sr)
