import os
import io
import json
import glob
import random
from typing import Tuple

import torch
import soundfile as sf
import webdataset as wds
import pytorch_lightning as pl

from .wds_datamodule import wds_collate_fn


def _load_wav_bytes(buf: bytes):
    data, sr = sf.read(io.BytesIO(buf), dtype="float32")
    tensor = torch.from_numpy(data)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    else:
        tensor = tensor.T
    return tensor, sr


def decode_sample_to_tensors(sample: dict, target_sr: int):
    mix_bytes = sample.get("mix.wav") or sample.get("mix")
    if mix_bytes is None:
        raise KeyError("sample missing 'mix.wav' key")
    mix_tensor, mix_sr = _load_wav_bytes(mix_bytes)
    if mix_sr != target_sr:
        raise ValueError(f"mix sample rate {mix_sr} != target_sr {target_sr}")

    source_keys = sorted(
        [k for k in sample.keys() if isinstance(k, str) and k.startswith("s") and k.endswith(".wav")]
    )

    src_tensors = []
    for key in source_keys:
        src_tensor, src_sr = _load_wav_bytes(sample[key])
        if src_sr != target_sr:
            raise ValueError(f"source sample rate {src_sr} != target_sr {target_sr}")
        src_tensors.append(src_tensor)

    if len(src_tensors) > 0:
        ch_counts = [t.shape[0] for t in src_tensors]
        max_ch = max(ch_counts + [mix_tensor.shape[0]])

        def ensure_chans(tensor: torch.Tensor):
            c = tensor.shape[0]
            if c == max_ch:
                return tensor
            if c == 1 and max_ch > 1:
                return tensor.expand(max_ch, -1)
            if c > max_ch:
                return tensor[:max_ch, :]
            return tensor

        src_tensors = [ensure_chans(tensor) for tensor in src_tensors]
        sources_tensor = torch.stack(src_tensors, dim=0)
    else:
        sources_tensor = torch.empty(0)

    json_b = sample.get("json") or sample.get("__json__") or sample.get("txt") or sample.get("meta")
    if json_b is None:
        labels = []
        meta = {}
    else:
        if isinstance(json_b, (bytes, bytearray)):
            json_str = json_b.decode("utf-8")
        else:
            json_str = json_b
        try:
            meta = json.loads(json_str) if isinstance(json_str, str) else json_str
        except Exception:
            meta = {}
        labels = meta.get("labels", meta.get("label", []))

    return mix_tensor, sources_tensor, labels, meta


def create_vggsound_dataloader(
    root_dir: str,
    sample_rate: int = 16000,
    batch_size: int = 4,
    num_workers: int = 4,
    shuffle_buffer: int = 1000,
    drop_last: bool = False,
    persistent_workers: bool = True,
    is_val: bool = False,
    mix_selected: list = None,
    data_ratio: float = 1.0,
    val_tar_count: int = 2,
):
    if mix_selected is not None:
        selected_dirs = [os.path.join(root_dir, mix_name) for mix_name in mix_selected]
    else:
        selected_dirs = [root_dir]

    if is_val:
        tar_paths = []
        rng = random.Random(42)
        for directory in selected_dirs:
            subdirs = sorted([s for s in os.listdir(directory) if os.path.isdir(os.path.join(directory, s))]) if mix_selected is None else [directory]
            for subdir_path in (subdirs if mix_selected is None else [directory]):
                if mix_selected is None:
                    subdir_path = os.path.join(directory, subdir_path)
                tar_files = sorted(glob.glob(os.path.join(subdir_path, "*.tar")))
                if len(tar_files) == 0:
                    continue
                tar_paths.extend(rng.sample(tar_files, min(val_tar_count, len(tar_files))))
    else:
        tar_paths = []
        for directory in selected_dirs:
            tar_paths.extend(glob.glob(os.path.join(directory, "**", "*.tar"), recursive=True))
        tar_paths.sort()
        if data_ratio < 1.0:
            rng = random.Random(42)
            tar_paths = rng.sample(tar_paths, max(1, int(len(tar_paths) * data_ratio)))
        random.shuffle(tar_paths)

    if len(tar_paths) == 0:
        raise FileNotFoundError("No .tar files found in given root_dir")

    if not is_val:
        dataset = wds.DataPipeline(
            wds.ResampledShards(tar_paths),
            wds.split_by_node,
            wds.split_by_worker,
            wds.tarfile_to_samples(handler=wds.warn_and_continue),
            wds.shuffle(shuffle_buffer),
            wds.map(lambda sample: decode_sample_to_tensors(sample, sample_rate)),
        )
    else:
        dataset = wds.DataPipeline(
            wds.SimpleShardList(tar_paths),
            wds.tarfile_to_samples(handler=wds.warn_and_continue),
            wds.shuffle(shuffle_buffer),
            wds.map(lambda sample: decode_sample_to_tensors(sample, sample_rate)),
        )

    return wds.WebLoader(
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


class VGGSoundDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_dir: str,
        val_dir: str = None,
        test_dir: str = None,
        sample_rate: int = 16000,
        batch_size: int = 4,
        num_workers: int = 8,
        shuffle_buffer: int = 1000,
        drop_last: bool = False,
        persistent_workers: bool = True,
        mix_selected: list = None,
        data_ratio: float = 1.0,
        val_tar_count: int = 2,
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
        self.mix_selected = mix_selected
        self.data_ratio = data_ratio
        self.val_tar_count = val_tar_count

    def train_dataloader(self):
        return create_vggsound_dataloader(
            root_dir=self.train_dir,
            sample_rate=self.sample_rate,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle_buffer=self.shuffle_buffer,
            drop_last=self.drop_last,
            persistent_workers=self.persistent_workers,
            mix_selected=self.mix_selected,
            data_ratio=self.data_ratio,
        )

    def val_dataloader(self):
        if self.val_dir is None:
            return None
        return create_vggsound_dataloader(
            root_dir=self.val_dir,
            sample_rate=self.sample_rate,
            batch_size=self.batch_size,
            num_workers=1,
            shuffle_buffer=0,
            drop_last=False,
            persistent_workers=self.persistent_workers,
            is_val=True,
            mix_selected=self.mix_selected,
            val_tar_count=self.val_tar_count,
        )

    def test_dataloader(self):
        if self.test_dir is None:
            return None
        return create_vggsound_dataloader(
            root_dir=self.test_dir,
            sample_rate=self.sample_rate,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle_buffer=0,
            drop_last=False,
            persistent_workers=self.persistent_workers,
            mix_selected=self.mix_selected,
        )

    @property
    def make_loader(self) -> Tuple:
        return self.train_dataloader(), self.val_dataloader(), self.test_dataloader()
