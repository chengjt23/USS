import argparse
import json
from pathlib import Path

import webdataset as wds

SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
DEFAULT_TAR_PATH = SCRIPT_DIR / "sample.tar"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "test_cases"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tar_path", type=str, default=str(DEFAULT_TAR_PATH))
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--num_samples", type=int, default=20)
    return parser.parse_args()


def sanitize_sample_name(name: str):
    text = str(name).strip()
    if not text:
        return "sample"
    return text.replace("\\", "_").replace("/", "_").replace(":", "_")


def value_to_bytes(value):
    if isinstance(value, bytes):
        return value
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, memoryview):
        return value.tobytes()
    if isinstance(value, str):
        return value.encode("utf-8")
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False).encode("utf-8")
    return str(value).encode("utf-8")


def save_sample(sample: dict, sample_index: int, output_dir: Path):
    raw_key = sample.get("__key__", f"sample_{sample_index:05d}")
    sample_name = f"{sample_index:05d}_{sanitize_sample_name(raw_key)}"
    sample_dir = output_dir / sample_name
    if sample_dir.exists():
        raise FileExistsError(f"Target sample directory already exists: {sample_dir}")
    sample_dir.mkdir(parents=True, exist_ok=False)
    for key, value in sample.items():
        if not isinstance(key, str) or key.startswith("__"):
            continue
        file_path = sample_dir / key
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_bytes(value_to_bytes(value))
    return sample_dir


def main():
    args = parse_args()
    tar_path = Path(args.tar_path).resolve()
    output_dir = Path(args.output_dir).resolve()

    if args.num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if not tar_path.exists():
        raise FileNotFoundError(f"Tar file not found: {tar_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = wds.WebDataset(str(tar_path), shardshuffle=False)

    extracted = 0
    for sample in dataset:
        save_sample(sample, extracted, output_dir)
        extracted += 1
        if extracted >= args.num_samples:
            break

    print(f"Extracted {extracted} samples to {output_dir}")


if __name__ == "__main__":
    main()
