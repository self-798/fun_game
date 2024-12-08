import torch

# 检查是否可以使用 GPU（ROCm 环境下，GPU 会显示为 CUDA）
if torch.cuda.is_available():
    device = torch.device("cuda")  # 选择第一个 GPU
    print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("CUDA is not available. Using CPU.")

print(device)
