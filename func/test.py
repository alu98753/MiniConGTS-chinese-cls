import torch

# Check if CUDA is available
cuda_available = torch.cuda.is_available()
print(f"Is CUDA available? {cuda_available}")

# Check the CUDA version
if cuda_available:
    print(f"CUDA version: {torch.version.cuda}")
    print(f"PyTorch is using CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"Number of CUDA devices: {torch.cuda.device_count()}")

# Check PyTorch version
print(f"PyTorch version: {torch.__version__}")
