import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# ================= 1. 配置 =================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_SAVE_PATH = os.path.join(BASE_DIR, 'models', 'best_mobilenet_finetune.pth')
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

BATCH_SIZE = 32  # 5060 显存大，用 32 更稳
NUM_EPOCHS = 30  
LEARNING_RATE = 0.00005  #  关键：极小学习率，保护预训练权重！

# ================= 2. 数据 (极简模式，无增强) =================
print("🔄 加载数据...")
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

image_datasets = {x: datasets.ImageFolder(os.path.join(DATA_DIR, x), data_transforms[x]) for x in ['train', 'val']}
dataloaders = {x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=True, num_workers=0) for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

print(f"✅ 类别: {class_names}")
print(f"✅ 训练集: {dataset_sizes['train']}, 验证集: {dataset_sizes['val']}")

# ================= 3. 模型 (MobileNetV3 全参数微调) =================
print("🏗️ 加载 MobileNetV3 Large...")
# 注意这里用的是 mobilenet_v3_large
model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)

# ⚠️ 关键修改：MobileNetV3 的最后一层叫 classifier，不是 fc
num_ftrs = model.classifier[3].in_features 
model.classifier[3] = nn.Linear(num_ftrs, len(class_names)) 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"🚀 使用设备: {device}")
model = model.to(device)

# ================= 4. 优化器 (Adam + 极小学习率) =================
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5) 

# ================= 5. 训练 =================
print(" 开始训练...")
best_acc = 0.0
train_loss_history = []
val_acc_history = []

for epoch in range(NUM_EPOCHS):
    print(f'\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---')
    
    # 训练
    model.train()
    running_loss = 0.0
    for inputs, labels in dataloaders['train']:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    
    epoch_loss = running_loss / dataset_sizes['train']
    train_loss_history.append(epoch_loss)
    print(f' Train Loss: {epoch_loss:.4f}')

    # 验证
    model.eval()
    corrects = 0
    with torch.no_grad():
        for inputs, labels in dataloaders['val']:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            corrects += torch.sum(preds == labels.data)
    
    epoch_acc = corrects.double() / dataset_sizes['val']
    val_acc_history.append(epoch_acc.item())
    print(f'🎯 Val Accuracy: {epoch_acc:.4f}')

    if epoch_acc > best_acc:
        best_acc = epoch_acc
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f'💾 保存最佳模型! Acc: {best_acc:.4f}')

    scheduler.step()

# 保存曲线
plt.figure(figsize=(10, 5))
plt.plot(train_loss_history, label='Train Loss')
plt.plot(val_acc_history, label='Val Acc')
plt.legend()
plt.savefig(os.path.join(BASE_DIR, 'outputs', 'mobilenet_final_curve.png'))
print(f"\n🎉 训练结束! 最佳 Val Acc: {best_acc:.4f}")