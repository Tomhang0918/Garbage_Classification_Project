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

# 1. 配置路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'best_mobilenet_finetune.pth') 

# 2. 数据预处理
data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'val'), data_transforms)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
class_names = val_dataset.classes

# 3. 加载 MobileNetV3 模型 (注意结构不同！)
print("🏗️ 加载 MobileNetV3 模型...")
model = models.mobilenet_v3_large(weights=None)
num_ftrs = model.classifier[3].in_features 
model.classifier[3] = nn.Linear(num_ftrs, len(class_names)) 

model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
model.eval()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 4. 预测
all_preds = []
all_labels = []
print("🔍 正在验证集上预测...")
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
# 使用 Greens 配色，和 ResNet 区分开
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
            xticklabels=class_names, yticklabels=class_names, linewidths=0.5)
plt.title('Confusion Matrix - MobileNetV3')
plt.ylabel('True Label ')
plt.xlabel('Predicted Label ')
plt.tight_layout()

save_path = os.path.join(BASE_DIR, 'outputs', 'confusion_matrix_mobilenet.png')
plt.savefig(save_path, dpi=300)
print(f"✅ MobileNetV3 混淆矩阵已保存至: {save_path}")