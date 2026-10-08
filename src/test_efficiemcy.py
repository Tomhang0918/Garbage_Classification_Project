import torch
import time
from torchvision import models
import torch.nn as nn

print("⚙️ [备用方案] 正在计算模型指标...\n")

def get_model(name):
    if name == 'ResNet18':
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 6)
    elif name == 'MobileNetV3':
        model = models.mobilenet_v3_large(weights=None)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, 6)
    return model

device = torch.device("cpu")
dummy_input = torch.randn(1, 3, 224, 224).to(device)

for model_name in ['ResNet18', 'MobileNetV3']:
    model = get_model(model_name).to(device)
    model.eval()
    
    # 1. 计算参数量 (纯 PyTorch 实现)
    total_params = sum(p.numel() for p in model.parameters())
    
    # 2. 计算 CPU 推理时间
    with torch.no_grad():
        for _ in range(10): model(dummy_input) # 预热
        start_time = time.time()
        for _ in range(100): model(dummy_input)
        end_time = time.time()
    avg_time_ms = ((end_time - start_time) / 100) * 1000
    
    print(f"--- {model_name} ---")
    print(f"参数量 (Params): {total_params / 1e6:.2f} M")
    print(f"CPU推理时间:     {avg_time_ms:.2f} ms")
    print(f"理论 FLOPs (参考值): {'1.82 G' if model_name=='ResNet18' else '0.23 G'}\n")