import os
import io
import json
import glob
import random
from typing import List, Tuple

import torch
import torchaudio
torchaudio.set_audio_backend("soundfile")
import webdataset as wds
import pytorch_lightning as pl


def decode_sample_to_tensors(sample: dict, target_sr: int):
    mix_bytes = sample.get("mix.wav") or sample.get("mix")
    if mix_bytes is None:
        raise KeyError("sample missing 'mix.wav' key")
    mix_tensor, sr = torchaudio.load(io.BytesIO(mix_bytes))
    if sr != target_sr:
        mix_tensor = torchaudio.functional.resample(mix_tensor, sr, target_sr)

    source_keys = sorted([k for k in sample.keys() if isinstance(k, str) and k.startswith("s") and k.endswith(".wav")])

    src_tensors = []
    for k in source_keys:
        src_tensor, _ = torchaudio.load(io.BytesIO(sample[k]))
        if sr != target_sr:
            src_tensor = torchaudio.functional.resample(src_tensor, sr, target_sr)
        src_tensors.append(src_tensor)

    if len(src_tensors) > 0:
        def to_mono_if_needed(t):
            if t.ndim == 1:
                return t.unsqueeze(0)
            return t
        src_tensors = [to_mono_if_needed(t) for t in src_tensors]
        ch_counts = [t.shape[0] for t in src_tensors]
        max_ch = max(ch_counts + [mix_tensor.shape[0]])
        def ensure_chans(t):
            c = t.shape[0]
            if c == max_ch:
                return t
            if c == 1 and max_ch > 1:
                return t.expand(max_ch, -1)
            if c > max_ch:
                return t[:max_ch, :]
            return t
        src_tensors = [ensure_chans(t) for t in src_tensors]
        sources_tensor = torch.stack(src_tensors, dim=0)
    else:
        sources_tensor = torch.empty(0)

    json_b = sample.get("json") or sample.get("__json__") or sample.get("txt") or sample.get("meta")
    if json_b is None:
        labels = []
    else:
        if isinstance(json_b, (bytes, bytearray)):
            json_str = json_b.decode("utf-8")
        elif isinstance(json_b, str):
            json_str = json_b
        else:
            json_str = json_b
        try:
            meta = json.loads(json_str) if isinstance(json_str, str) else json_str
        except Exception:
            meta = {}
        labels = meta.get("labels", meta.get("label", []))

    return mix_tensor, sources_tensor, labels


def wds_collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor, List]]):
    mixes = [item[0] for item in batch]
    sources = [item[1] for item in batch]
    labels = [item[2] for item in batch]

    max_T = max(m.shape[-1] for m in mixes)
    n_src_list = [s.shape[0] if s.ndim == 3 else 0 for s in sources]
    max_N = max(n_src_list) if len(n_src_list) > 0 else 0
    max_C = max(m.shape[0] for m in mixes)

    padded_mixes = []
    padded_sources = []

    for m, s in zip(mixes, sources):
        if m.shape[0] < max_C:
            m = m.expand(max_C, -1)
        elif m.shape[0] > max_C:
            m = m[:max_C, :]
        if m.shape[-1] < max_T:
            m = torch.nn.functional.pad(m, (0, max_T - m.shape[-1]))
        padded_mixes.append(m)

        if s.ndim != 3 or s.shape[0] == 0:
            if max_N == 0:
                padded_sources.append(torch.empty(0))
                continue
            padded_sources.append(torch.zeros((max_N, max_C, max_T)))
        else:
            if s.shape[1] < max_C:
                s = s.expand(s.shape[0], max_C, -1)
            elif s.shape[1] > max_C:
                s = s[:, :max_C, :]
            if s.shape[-1] < max_T:
                s = torch.nn.functional.pad(s, (0, max_T - s.shape[-1]))
            if s.shape[0] < max_N:
                s = torch.cat([s, torch.zeros((max_N - s.shape[0], max_C, max_T))], dim=0)
            padded_sources.append(s)

    mix_batch = torch.stack(padded_mixes, dim=0)
    if max_N == 0:
        sources_batch = torch.empty((mix_batch.shape[0], 0))
    else:
        sources_batch = torch.stack(padded_sources, dim=0)

    return {"mix": mix_batch, "sources": sources_batch, "labels": labels}


def create_wds_dataloader(
    root_dir: str,
    sample_rate: int = 32000,
    batch_size: int = 4,
    num_workers: int = 4,
    shuffle_buffer: int = 1000,
    drop_last: bool = False,
    persistent_workers: bool = True,
    is_val: bool = False,
):
    if is_val:
        subdirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        if len(subdirs) != 4:
            raise ValueError(f"Validation root_dir must contain exactly 4 subdirectories, found {len(subdirs)}")
        tar_paths = []
        rng = random.Random(42)
        for subdir in sorted(subdirs):
            subdir_path = os.path.join(root_dir, subdir)
            tar_files = sorted(glob.glob(os.path.join(subdir_path, "*.tar")))
            if len(tar_files) == 0:
                raise FileNotFoundError(f"No .tar files found in subdirectory: {subdir_path}")
            selected_tars = rng.sample(tar_files, 2)
            tar_paths.extend(selected_tars)
    else:
        tar_paths = sorted(glob.glob(os.path.join(root_dir, "**", "*.tar"), recursive=True))
        random.shuffle(tar_paths)

    if len(tar_paths) == 0:
        raise FileNotFoundError("No .tar files found in given root_dir")

    dataset = wds.DataPipeline(
        wds.SimpleShardList(tar_paths),
        wds.shuffle(100 if shuffle_buffer > 0 else 0),
        wds.split_by_node,
        wds.split_by_worker,
        wds.tarfile_to_samples(handler=wds.warn_and_continue),
        wds.shuffle(shuffle_buffer),
        wds.map(lambda sample: decode_sample_to_tensors(sample, sample_rate))
    )

    loader = wds.WebLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=wds_collate_fn,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=persistent_workers,
        drop_last=drop_last,
    )

    return loader


class WDSDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_dir: str,
        val_dir: str = None,
        test_dir: str = None,
        sample_rate: int = 32000,
        batch_size: int = 4,
        num_workers: int = 8,
        shuffle_buffer: int = 1000,
        drop_last: bool = False,
        persistent_workers: bool = True,
    ):
        super().__init__()
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        self.sample_rate = sample_rate
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle_buffer = shuffle_buffer
        self.drop_last = drop_last
        self.persistent_workers = persistent_workers

    def train_dataloader(self):
        return create_wds_dataloader(
            root_dir=self.train_dir,
            sample_rate=self.sample_rate,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle_buffer=self.shuffle_buffer,
            drop_last=self.drop_last,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self):
        if self.val_dir is None:
            return None
        return create_wds_dataloader(
            root_dir=self.val_dir,
            sample_rate=self.sample_rate,
            batch_size=self.batch_size,
            num_workers=1,
            shuffle_buffer=0,
            drop_last=False,
            persistent_workers=self.persistent_workers,
            is_val=True,
        )

    def test_dataloader(self):
        if self.test_dir is None:
            return None
        return create_wds_dataloader(
            root_dir=self.test_dir,
            sample_rate=self.sample_rate,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle_buffer=0,
            drop_last=False,
            persistent_workers=self.persistent_workers,
        )

    @property
    def make_loader(self):
        return self.train_dataloader(), self.val_dataloader(), self.test_dataloader()
