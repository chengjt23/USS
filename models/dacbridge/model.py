import os
import torch
import numpy as np
import soundfile as sf

from models.flowsep.model import LatentDiffusion
from pytorch_lightning.utilities.rank_zero import rank_zero_only


class LatentDiffusionDACVAE(LatentDiffusion):

    def encode_first_stage(self, x):
        with torch.no_grad():
            return self.first_stage_model.encode(x)

    def decode_first_stage(self, z):
        with torch.no_grad():
            z = 1.0 / self.scale_factor * z
            return self.first_stage_model.decode(z)

    def get_input(self, batch, k, return_first_stage_encode=True,
                  return_decoding_output=False, return_encoder_input=False,
                  return_encoder_output=False, unconditional_prob_cfg=0.1):
        x = super(LatentDiffusion, self).get_input(batch, k)
        x = x.to(self.device, non_blocking=True)

        if return_first_stage_encode:
            encoder_posterior = self.encode_first_stage(x)
            z = self.get_first_stage_encoding(encoder_posterior).detach()

            if self.extra_channels:
                extra = super(LatentDiffusion, self).get_input(batch, self.extra_channel_key)
                extra = extra.to(self.device, non_blocking=True)
                extra_posterior = self.encode_first_stage(extra)
                e = self.get_first_stage_encoding(extra_posterior).detach()
                z = torch.cat([z, e], dim=1)
        else:
            z = None

        cond_dict = {}
        if len(self.cond_stage_model_metadata.keys()) > 0:
            unconditional_cfg = False
            if self.conditional_dry_run_finished and self.make_decision(unconditional_prob_cfg):
                unconditional_cfg = True
            for cond_model_key in self.cond_stage_model_metadata.keys():
                cond_stage_key = self.cond_stage_model_metadata[cond_model_key]["cond_stage_key"]
                if cond_model_key in cond_dict:
                    continue
                if cond_stage_key != "all":
                    xc = super(LatentDiffusion, self).get_input(batch, cond_stage_key)
                    if isinstance(xc, torch.Tensor):
                        xc = xc.to(self.device)
                else:
                    xc = batch
                c = self.get_learned_conditioning(xc, key=cond_model_key, unconditional_cfg=unconditional_cfg)
                if isinstance(c, dict):
                    for ck in c.keys():
                        cond_dict[ck] = c[ck]
                else:
                    cond_dict[cond_model_key] = c

        out = [z, cond_dict]
        if return_decoding_output:
            out += [self.decode_first_stage(z)]
        if return_encoder_input:
            out += [x]
        if return_encoder_output:
            out += [encoder_posterior]
        if not self.conditional_dry_run_finished:
            self.conditional_dry_run_finished = True
        return out

    @torch.no_grad()
    def generate_sample(self, batchs, ddim_steps=200, ddim_eta=1.0, x_T=None,
                        n_gen=1, unconditional_guidance_scale=1.0,
                        unconditional_conditioning=None, name="waveform",
                        use_plms=False, limit_num=None, save=True,
                        save_mixed=True, **kwargs):
        assert x_T is None
        use_ddim = ddim_steps is not None
        waveform_save_path = None
        try:
            waveform_save_path = os.path.join(self.get_log_dir(), name)
        except:
            waveform_save_path = name
        os.makedirs(waveform_save_path, exist_ok=True)

        if "audiocaps" in waveform_save_path and len(os.listdir(waveform_save_path)) >= 964:
            return waveform_save_path

        with self.ema_scope("Plotting"):
            for i, batch in enumerate(batchs):
                z, c = self.get_input(batch, self.first_stage_key, unconditional_prob_cfg=0.0)
                fnames = list(super(LatentDiffusion, self).get_input(batch, "fname"))

                if self.extra_channels:
                    extra = super(LatentDiffusion, self).get_input(batch, self.extra_channel_key)
                    extra = extra.to(self.device)
                    extra_z = self.encode_first_stage(extra)
                    x_T = self.get_first_stage_encoding(extra_z).detach()

                    if save_mixed:
                        mixed_save_path = os.path.join(waveform_save_path, "mixed")
                        os.makedirs(mixed_save_path, exist_ok=True)
                        mixed_waveform = batch["mixed_waveform"]
                        for j, fname in enumerate(fnames):
                            sf.write(os.path.join(mixed_save_path, os.path.basename(fname)),
                                     mixed_waveform[j].cpu().numpy().T, self.sampling_rate)

                if limit_num is not None and i * z.size(0) > limit_num:
                    break

                if self.condition_key:
                    c = self.filter_useful_cond_dict(c)

                text = super(LatentDiffusion, self).get_input(batch, "text")
                batch_size = z.shape[0] * n_gen

                if self.condition_key:
                    for cond_key in c.keys():
                        if isinstance(c[cond_key], list):
                            for ci in range(len(c[cond_key])):
                                c[cond_key][ci] = torch.cat([c[cond_key][ci]] * n_gen, dim=0)
                        elif isinstance(c[cond_key], dict):
                            for dk in c[cond_key].keys():
                                c[cond_key][dk] = torch.cat([c[cond_key][dk]] * n_gen, dim=0)
                        else:
                            c[cond_key] = torch.cat([c[cond_key]] * n_gen, dim=0)
                    text = text * n_gen
                    if unconditional_guidance_scale != 1.0:
                        unconditional_conditioning = {}
                        for key in self.cond_stage_model_metadata:
                            model_idx = self.cond_stage_model_metadata[key]["model_idx"]
                            unconditional_conditioning[key] = self.cond_stage_models[model_idx].get_unconditional_condition(batch_size)

                samples, _ = self.sample_log(
                    cond=c, batch_size=batch_size, x_T=x_T,
                    ddim=use_ddim, ddim_steps=ddim_steps, eta=ddim_eta,
                    unconditional_guidance_scale=unconditional_guidance_scale,
                    unconditional_conditioning=unconditional_conditioning,
                    use_plms=use_plms,
                )
                if self.extra_channels:
                    samples = samples[:, :self.channels, :, :]

                wav_out = self.decode_first_stage(samples)
                waveform = wav_out.cpu().detach().float().numpy()

                if n_gen >= 3 and self.use_clap:
                    try:
                        best_index = []
                        similarity = self.clap.cos_similarity(
                            torch.FloatTensor(waveform).squeeze(1), text)
                        for idx in range(z.shape[0]):
                            candidates = similarity[idx :: z.shape[0]]
                            best_index.append(idx + torch.argmax(candidates).item() * z.shape[0])
                        waveform = waveform[best_index]
                    except Exception as e:
                        from utils.tools import rank_zero_print
                        rank_zero_print("Warning: while calculating CLAP score (not fatal), ", e)
                else:
                    waveform = waveform[:z.shape[0]]

            if save:
                self.save_waveform(waveform, waveform_save_path, name=fnames)
            return waveform
