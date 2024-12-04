import torch
import os 

print("CUDA available:", torch.cuda.is_available())
torch.cuda.set_device(0)

print(    torch.cuda.set_device(0))
print("CUDA device count:", torch.cuda.device_count())
print("Current device:", torch.cuda.current_device())
print("Device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No CUDA Device")
saved_model_path = r"E:\NYCU-Project\Class\NLP\MiniConGTS\modules\models\saved_models\best_model_ch.pt"
if not os.path.exists(saved_model_path):
    raise FileNotFoundError(f"模型文件 {saved_model_path} 未找到。")

try:
    with open(r"E:\NYCU-Project\Class\NLP\MiniConGTS\modules\models\saved_models\best_model_ch.pt", 'rb') as f:
        f.read()
except Exception as e:
    print(f"文件读取错误：{e}")
