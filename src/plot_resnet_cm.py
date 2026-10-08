import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision import models
import torch.nn as nn

# 1. 配置路径 (确保和 train_ResNet18.py 里一致)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
# 确保这里指向你刚才跑出来的最佳模型
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'best_resnet_finetune.pth') 

# 2. 数据预处理 (必须和训练时完全一样！)
data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'val'), data_transforms)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
class_names = val_dataset.classes

# 3. 加载 ResNet18 模型
print("️ 加载 ResNet18 模型...")
model = models.resnet18(weights=None)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(class_names))

# 加载权重
model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
model.eval()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 4. 预测
all_preds = []
all_labels = []
print(" 正在验证集上预测...")
with torch.no_grad():
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# 5. 画图
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
# 使用 Blues 配色，annot=True 显示数字，fmt='d' 表示整数
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names, linewidths=0.5)
plt.title('Confusion Matrix - ResNet18')
plt.ylabel('True Label ')
plt.xlabel('Predicted Label ')
plt.tight_layout()

save_path = os.path.join(BASE_DIR, 'outputs', 'confusion_matrix_resnet.png')
plt.savefig(save_path, dpi=300) # 300dpi 保证报告里清晰
print(f"✅ ResNet18 混淆矩阵已保存至: {save_path}")