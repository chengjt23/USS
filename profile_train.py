import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import argparse
import yaml
import torch
import time
import numpy as np

from data.wds_datamodule import WDSDataModule
from utils.audio import TacotronSTFT, get_mel_from_wav
from utils.tools import instantiate_from_config
from train import build_stft_tool, convert_wds_batch_to_model_format


def profile_step(model, batch, device):
    timings = {}

    # ---- 1. mel 特征提取 (CPU) ----
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    x = model.get_input(batch, model.first_stage_key)
    x = x.to(device)
    torch.cuda.synchronize()
    timings["1_get_input_mel_to_gpu"] = time.perf_counter() - t0

    # ---- 2. VAE encode (target) ----
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    encoder_posterior = model.encode_first_stage(x)
    if isinstance(encoder_posterior, torch.Tensor):
        z = encoder_posterior
    else:
        z = model.get_first_stage_encoding(encoder_posterior).detach()
    torch.cuda.synchronize()
    timings["2_vae_encode_target"] = time.perf_counter() - t0

    # ---- 3. VAE encode (mix, extra_channels) ----
    if model.extra_channels:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        extra = model.get_input(batch, model.extra_channel_key)
        extra = extra.to(device).reshape(extra.shape[0], 1, extra.shape[1], -1) if extra.ndim == 3 else extra.to(device)
        if extra.ndim == 3:
            extra = extra.reshape(extra.shape[0], 1, extra.shape[1], -1)
        extra_posterior = model.encode_first_stage(extra)
        e = model.get_first_stage_encoding(extra_posterior).detach()
        z = torch.cat([z, e], dim=1)
        torch.cuda.synchronize()
        timings["3_vae_encode_mix"] = time.perf_counter() - t0

    # ---- 4. FlanT5 text encode ----
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    cond_dict = {}
    for cond_model_key in model.cond_stage_model_metadata.keys():
        cond_stage_key = model.cond_stage_model_metadata[cond_model_key]["cond_stage_key"]
        xc = model.get_input(batch, cond_stage_key)
        if isinstance(xc, torch.Tensor):
            xc = xc.to(device)
        c = model.get_learned_conditioning(xc, key=cond_model_key, unconditional_cfg=False)
        if isinstance(c, dict):
            for k in c.keys():
                cond_dict[k] = c[k]
        else:
            cond_dict[cond_model_key] = c
    torch.cuda.synchronize()
    timings["4_flant5_encode"] = time.perf_counter() - t0

    cond_dict = model.filter_useful_cond_dict(cond_dict)

    # ---- 5. UNet forward ----
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    t_rand = torch.rand([z.shape[0]], device=device)
    loss, loss_dict = model.p_losses(z, cond_dict, t_rand)
    torch.cuda.synchronize()
    timings["5_unet_forward"] = time.perf_counter() - t0

    # ---- 6. Backward ----
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    loss.backward()
    torch.cuda.synchronize()
    timings["6_backward"] = time.perf_counter() - t0

    return timings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_yaml", type=str, default="configs/flowsep/bridge.yaml")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()

    with open(args.config_yaml, "r") as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)

    device = torch.device("cuda:0")

    datamodule = WDSDataModule(**configs["datamodule"]["data_config"])
    train_loader, _, _ = datamodule.make_loader
    stft_tool = build_stft_tool(configs)

    model = instantiate_from_config(configs["model"])
    model = model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(configs["model"]["params"]["base_learning_rate"]))

    all_timings = []

    step = 0
    for raw_batch in train_loader:
        batch = convert_wds_batch_to_model_format(raw_batch, configs, stft_tool)

        # data loading 计时
        torch.cuda.synchronize()
        t_data_start = time.perf_counter()
        batch_next_ready = True
        torch.cuda.synchronize()
        t_data = time.perf_counter() - t_data_start

        optimizer.zero_grad()
        timings = profile_step(model, batch, device)
        timings["0_data_preprocess"] = t_data

        # optimizer step
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        optimizer.step()
        torch.cuda.synchronize()
        timings["7_optimizer_step"] = time.perf_counter() - t0

        step += 1
        if step <= args.warmup:
            print(f"[warmup {step}/{args.warmup}] skipped")
            continue

        all_timings.append(timings)
        total = sum(timings.values())
        print(f"\n--- Step {step} (total: {total:.3f}s) ---")
        for k in sorted(timings.keys()):
            pct = timings[k] / total * 100
            print(f"  {k:30s}: {timings[k]:.4f}s  ({pct:5.1f}%)")

        if step >= args.warmup + args.steps:
            break

    print("\n" + "=" * 60)
    print(f"Average over {len(all_timings)} steps:")
    print("=" * 60)
    keys = sorted(all_timings[0].keys())
    avg = {k: np.mean([t[k] for t in all_timings]) for k in keys}
    total_avg = sum(avg.values())
    for k in keys:
        pct = avg[k] / total_avg * 100
        bar = "█" * int(pct / 2)
        print(f"  {k:30s}: {avg[k]:.4f}s  ({pct:5.1f}%)  {bar}")
    print(f"  {'TOTAL':30s}: {total_avg:.4f}s")


if __name__ == "__main__":
    main()
