import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import yaml
import torch
import torchaudio
import numpy as np
from pytorch_lightning import seed_everything

from utils.tools import instantiate_from_config
from utils.audio import TacotronSTFT, get_mel_from_wav
from data.wds_datamodule import WDSDataModule
from train import WrappedDataLoader, build_stft_tool


def si_sdr(ref, est):
    ref = ref - np.mean(ref)
    est = est - np.mean(est)
    dot = np.sum(ref * est)
    s_target = dot * ref / (np.sum(ref ** 2) + 1e-8)
    e_noise = est - s_target
    return 10 * np.log10(np.sum(s_target ** 2) / (np.sum(e_noise ** 2) + 1e-8) + 1e-8)


def sdr(ref, est):
    noise = est - ref
    return 10 * np.log10(np.sum(ref ** 2) / (np.sum(noise ** 2) + 1e-8) + 1e-8)


def compute_pesq(ref, est, sr):
    try:
        from pesq import pesq as pesq_fn
        if sr != 16000:
            ref_16k = torchaudio.functional.resample(torch.from_numpy(ref).float(), sr, 16000).numpy()
            est_16k = torchaudio.functional.resample(torch.from_numpy(est).float(), sr, 16000).numpy()
        else:
            ref_16k, est_16k = ref, est
        return pesq_fn(16000, ref_16k, est_16k, "wb")
    except Exception:
        return float("nan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_yaml", type=str, required=True)
    parser.add_argument("-l", "--checkpoint", type=str, required=True)
    parser.add_argument("-s", "--infer_steps", type=int, default=20)
    parser.add_argument("-n", "--max_samples", type=int, default=200)
    parser.add_argument("--no_pesq", action="store_true")
    args = parser.parse_args()

    seed_everything(0)
    with open(args.config_yaml, "r") as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)

    model = instantiate_from_config(configs["model"]).cuda().eval()
    ckpt = torch.load(args.checkpoint, map_location="cuda")
    if "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    model.load_state_dict(ckpt, strict=False)

    stft_tool = build_stft_tool(configs)
    sampling_rate = configs["preprocessing"]["audio"]["sampling_rate"]
    hopsize = configs["preprocessing"]["stft"]["hop_length"]
    duration = configs["preprocessing"]["audio"]["duration"]
    target_length = int(duration * sampling_rate / hopsize)

    datamodule = WDSDataModule(**configs["datamodule"]["data_config"])
    _, val_loader, _ = datamodule.make_loader
    val_loader = WrappedDataLoader(val_loader, configs, stft_tool)

    si_sdr_list, sdr_list, pesq_list = [], [], []
    count = 0

    with torch.no_grad(), model.ema_scope():
        for batch in val_loader:
            if count >= args.max_samples:
                break

            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.cuda()

            z, c = model.get_input(batch, model.first_stage_key, unconditional_prob_cfg=0.0)
            c = model.filter_useful_cond_dict(c)
            bs = z.shape[0]

            x_T = None
            if model.extra_channels:
                extra = model.__class__.__mro__[1].get_input(model, batch, model.extra_channel_key).to(model.device)
                extra = extra.reshape(extra.shape[0], 1, extra.shape[1], extra.shape[2])
                extra_posterior = model.encode_first_stage(extra)
                x_T = model.get_first_stage_encoding(extra_posterior).detach()

            samples, _ = model.sample_log(
                cond=c, batch_size=bs, x_T=x_T,
                ddim=True, ddim_steps=args.infer_steps,
                unconditional_guidance_scale=1.0,
            )
            if model.extra_channels:
                samples = samples[:, :model.channels, :, :]

            mel = model.decode_first_stage(samples)
            if model.fbank_shift:
                mel = mel - model.fbank_shift
            if model.data_std:
                mel = (mel * model.data_std) + model.data_mean

            pred_wav = model.mel_spectrogram_to_waveform(mel, save=False)

            target_wav = batch["waveform"].cpu().numpy()
            if target_wav.ndim == 3:
                target_wav = target_wav.squeeze(1)

            for i in range(min(bs, args.max_samples - count)):
                ref = target_wav[i]
                est = pred_wav[i].flatten()
                min_len = min(len(ref), len(est))
                ref, est = ref[:min_len].astype(np.float64), est[:min_len].astype(np.float64)

                si_sdr_list.append(si_sdr(ref, est))
                sdr_list.append(sdr(ref, est))
                if not args.no_pesq:
                    pesq_list.append(compute_pesq(ref.astype(np.float32), est.astype(np.float32), sampling_rate))
                count += 1

            print(f"[{count}/{args.max_samples}] SI-SDR={np.mean(si_sdr_list):.2f} SDR={np.mean(sdr_list):.2f}", end="")
            if pesq_list:
                print(f" PESQ={np.nanmean(pesq_list):.3f}", end="")
            print()

    print("\n===== Results =====")
    print(f"Samples:  {count}")
    print(f"SI-SDR:   {np.mean(si_sdr_list):.3f} dB")
    print(f"SDR:      {np.mean(sdr_list):.3f} dB")
    if pesq_list:
        print(f"PESQ:     {np.nanmean(pesq_list):.3f}")


if __name__ == "__main__":
    main()
