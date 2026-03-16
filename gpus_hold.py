import torch
import sys
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"

def hold_gpus(ratio=0.9):
    if not torch.cuda.is_available():
        print("No GPU available.")
        return
    
    tensors = []
    for i in range(torch.cuda.device_count()):
        torch.cuda.set_device(i)
        total = torch.cuda.get_device_properties(i).total_memory
        alloc = int(total * ratio)
        try:
            t = torch.empty(alloc // 4, dtype=torch.float32, device=f"cuda:{i}")
            tensors.append(t)
            print(f"GPU {i}: allocated {alloc / 1024**3:.2f} GB / {total / 1024**3:.2f} GB")
        except RuntimeError as e:
            print(f"GPU {i}: {e}")
    
    print("Holding GPU memory. Press Ctrl+C to release.")
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        del tensors
        torch.cuda.empty_cache()
        print("Released.")

if __name__ == "__main__":
    ratio = float(sys.argv[1]) if len(sys.argv) > 1 else 0.9
    hold_gpus(ratio)
