import torch

def check_cuda_devices():
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"總共有 {num_gpus} 個 GPU 可用：")
        
        for i in range(num_gpus):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  設備狀態: {'can' if torch.cuda.get_device_capability(i) else 'can not'}")
            print(f"  CUDA 記憶體分配: {torch.cuda.memory_allocated(i) / 1024 ** 2:.2f} MB")
            print(f"  CUDA 記憶體快取: {torch.cuda.memory_reserved(i) / 1024 ** 2:.2f} MB")
    else:
        print("CUDA 設備不可用，將使用 CPU")

# 執行檢查
check_cuda_devices()

import torch
print("PyTorch CUDA 版本:", torch.version.cuda)
