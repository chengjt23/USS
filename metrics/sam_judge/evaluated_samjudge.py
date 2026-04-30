import warnings
warnings.filterwarnings("ignore")
import os
import json
import torch
import torchaudio
from pathlib import Path
from tqdm import tqdm
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import argparse
from sam_audio import SAMAudioJudgeModel, SAMAudioJudgeProcessor

# 设置 Hugging Face 镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


class AudioSeparationDataset(Dataset):
    """Dataset for audio separation evaluation."""
    
    def __init__(self, base_path, mix_type):
        self.base_path = Path(base_path)
        self.mix_path = self.base_path / mix_type
        self.samples = []
        
        if not self.mix_path.exists():
            print(f"Warning: {self.mix_path} does not exist")
            return
            
        # 遍历所有子文件夹
        for folder in sorted(self.mix_path.iterdir()):
            if folder.is_dir():
                labels_path = folder / "labels.json"
                mix_path = folder / "mix.wav"
                
                if labels_path.exists() and mix_path.exists():
                    with open(labels_path, 'r') as f:
                        labels = json.load(f)
                    
                    # 为每个分离的音频创建一个样本
                    for i, label in enumerate(labels, 1):
                        est_path = folder / f"est{i}.wav"
                        ref_path = folder / f"ref{i}.wav"
                        
                        if est_path.exists():
                            self.samples.append({
                                'folder': folder.name,
                                'mix_path': str(mix_path),
                                'est_path': str(est_path),
                                'ref_path': str(ref_path) if ref_path.exists() else None,
                                'label': label,
                                'est_idx': i
                            })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch):
    """Custom collate function to handle audio loading."""
    return batch


def evaluate_sample(model, processor, sample, device, resamplers, target_sr=48000):
    """Evaluate a single sample."""
    try:
        # Load audio files
        input_audio, sr = torchaudio.load(sample['mix_path'])
        separated_audio, sr_sep = torchaudio.load(sample['est_path'])
        
        # Resample to target sample rate if necessary
        if sr != target_sr:
            if sr not in resamplers:
                resamplers[sr] = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
            input_audio = resamplers[sr](input_audio)
        
        if sr_sep != target_sr:
            if sr_sep not in resamplers:
                resamplers[sr_sep] = torchaudio.transforms.Resample(orig_freq=sr_sep, new_freq=target_sr)
            separated_audio = resamplers[sr_sep](separated_audio)
        
        # Process inputs
        inputs = processor(
            text=[sample['label']],
            input_audio=[input_audio],
            separated_audio=[separated_audio],
            sampling_rate=target_sr,
        ).to(device)
        
        # Get quality scores
        with torch.inference_mode():
            result = model(**inputs)
        
        return {
            'folder': sample['folder'],
            'est_idx': sample['est_idx'],
            'label': sample['label'],
            'overall': result.overall.item(),
            'recall': result.recall.item(),
            'precision': result.precision.item(),
            'faithfulness': result.faithfulness.item(),
        }
    except Exception as e:
        print(f"Error processing {sample['folder']}/est{sample['est_idx']}: {e}")
        return {
            'folder': sample['folder'],
            'est_idx': sample['est_idx'],
            'label': sample['label'],
            'error': str(e)
        }


def setup_distributed():
    """Setup distributed training."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
        return local_rank, rank, world_size
    else:
        return 0, 0, 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_path', type=str, 
                        default='/gpfs-flash/hulab/public_datasets/audio_datasets/USS-Qwen/code/model_review/experiments_curationed/CLIPSep')
    parser.add_argument('--model_path', type=str, default='pretrained_zoos/sam-audio-judge')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results')
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()
    
    # Setup distributed
    local_rank, rank, world_size = setup_distributed()
    is_main_process = rank == 0
    
    # Set device
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{local_rank}')
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')
    
    if is_main_process:
        print(f"{'='*60}")
        print(f"Using {world_size} GPU(s)")
        print(f"Output directory: {args.output_dir}")
        print(f"{'='*60}")
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model and processor (only once)
    if is_main_process:
        print("\n[1/2] Loading model and processor...")
    
    model = SAMAudioJudgeModel.from_pretrained(args.model_path).to(device).eval()
    processor = SAMAudioJudgeProcessor.from_pretrained(args.model_path)
    
    if is_main_process:
        print("✓ Model loaded successfully!\n")
    
    # Mix types to evaluate
    mix_types = ['2mix', '3mix', '4mix', '5mix']
    all_mix_stats = {}
    
    if is_main_process:
        print("[2/2] Starting evaluation...")
    
    for mix_idx, mix_type in enumerate(mix_types):
        if is_main_process:
            print(f"\n{'─'*60}")
            print(f"Processing {mix_type} ({mix_idx+1}/{len(mix_types)})")
            print(f"{'─'*60}")
        
        # Create dataset
        dataset = AudioSeparationDataset(args.base_path, mix_type)
        
        if len(dataset) == 0:
            if is_main_process:
                print(f"⚠ No samples found for {mix_type}, skipping...")
            continue
        
        if is_main_process:
            print(f"Found {len(dataset)} samples")
        
        # Create sampler and dataloader
        if world_size > 1:
            sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
        else:
            sampler = None
        
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            sampler=sampler,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            pin_memory=True
        )
        
        # Calculate total for this process
        local_total = len(dataloader)
        
        # Evaluate with progress bar
        all_results = []
        
        if is_main_process:
            # 主进程显示进度条
            pbar = tqdm(
                total=local_total,
                desc=f"GPU {rank}",
                unit="sample",
                ncols=100,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
            )
        # 在评估循环之前创建 resamplers 字典
        resamplers = {}  # 缓存不同采样率的 resampler
        for batch_idx, batch in enumerate(dataloader):
            for sample in batch:
                result = evaluate_sample(model, processor, sample, device, resamplers)
                all_results.append(result)
            
            if is_main_process:
                pbar.update(1)
                # 显示当前处理的文件
                pbar.set_postfix_str(f"folder={batch[0]['folder']}")
        
        if is_main_process:
            pbar.close()
        
        # Gather results from all processes
        if world_size > 1:
            gathered_results = [None] * world_size
            dist.all_gather_object(gathered_results, all_results)
            if is_main_process:
                all_results = []
                for results in gathered_results:
                    all_results.extend(results)
        
        if is_main_process:
            # Organize results by folder
            folder_results = {}
            for result in all_results:
                folder = result['folder']
                if folder not in folder_results:
                    folder_results[folder] = []
                folder_results[folder].append(result)
            
            # Sort results within each folder by est_idx
            for folder in folder_results:
                folder_results[folder] = sorted(folder_results[folder], key=lambda x: x['est_idx'])
            
            # Save results to JSON
            output_path = os.path.join(args.output_dir, f"{mix_type}.json")
            with open(output_path, 'w') as f:
                json.dump(folder_results, f, indent=2)
            print(f"\n✓ Results saved to {output_path}")
            
            # Calculate statistics
            valid_results = [r for r in all_results if 'error' not in r]
            if valid_results:
                stats = {
                    'num_samples': len(valid_results),
                    'num_errors': len(all_results) - len(valid_results),
                    'overall_mean': sum(r['overall'] for r in valid_results) / len(valid_results),
                    'recall_mean': sum(r['recall'] for r in valid_results) / len(valid_results),
                    'precision_mean': sum(r['precision'] for r in valid_results) / len(valid_results),
                    'faithfulness_mean': sum(r['faithfulness'] for r in valid_results) / len(valid_results),
                }
                all_mix_stats[mix_type] = stats
                
                print(f"\n📊 {mix_type} Statistics:")
                print(f"   ├── Total samples: {stats['num_samples']}")
                print(f"   ├── Errors: {stats['num_errors']}")
                print(f"   ├── Overall Mean: {stats['overall_mean']:.4f}")
                print(f"   ├── Recall Mean: {stats['recall_mean']:.4f}")
                print(f"   ├── Precision Mean: {stats['precision_mean']:.4f}")
                print(f"   └── Faithfulness Mean: {stats['faithfulness_mean']:.4f}")
        
        # Synchronize before next mix type
        if world_size > 1:
            dist.barrier()
    
    # Print final summary
    if is_main_process:
        print(f"\n{'='*80}")
        print("📈 FINAL SUMMARY")
        print(f"{'='*80}")
        print(f"{'Mix Type':<10} {'Samples':<10} {'Overall':<12} {'Recall':<12} {'Precision':<12} {'Faithfulness':<12}")
        print("─" * 80)
        
        total_samples = 0
        total_overall = 0
        total_recall = 0
        total_precision = 0
        total_faithfulness = 0
        
        for mix_type in mix_types:
            if mix_type in all_mix_stats:
                stats = all_mix_stats[mix_type]
                print(f"{mix_type:<10} {stats['num_samples']:<10} {stats['overall_mean']:<12.4f} "
                      f"{stats['recall_mean']:<12.4f} {stats['precision_mean']:<12.4f} "
                      f"{stats['faithfulness_mean']:<12.4f}")
                total_samples += stats['num_samples']
                total_overall += stats['overall_mean'] * stats['num_samples']
                total_recall += stats['recall_mean'] * stats['num_samples']
                total_precision += stats['precision_mean'] * stats['num_samples']
                total_faithfulness += stats['faithfulness_mean'] * stats['num_samples']
        
        if total_samples > 0:
            print("─" * 80)
            print(f"{'Weighted':<10} {total_samples:<10} {total_overall/total_samples:<12.4f} "
                  f"{total_recall/total_samples:<12.4f} {total_precision/total_samples:<12.4f} "
                  f"{total_faithfulness/total_samples:<12.4f}")
        
        # Save summary statistics
        summary_path = os.path.join(args.output_dir, "summary.json")
        with open(summary_path, 'w') as f:
            json.dump(all_mix_stats, f, indent=2)
        print(f"\n✓ Summary saved to {summary_path}")
        print(f"\n{'='*80}")
        print("🎉 Evaluation completed!")
        print(f"{'='*80}")
    
    # Cleanup
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
