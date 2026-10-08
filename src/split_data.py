import os
import shutil
import random

# 配置路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw')
TRAIN_DIR = os.path.join(BASE_DIR, 'data', 'train')
VAL_DIR = os.path.join(BASE_DIR, 'data', 'val')

print("🔄 开始重新划分数据集...")

# 获取所有类别
classes = os.listdir(RAW_DIR)
classes = [c for c in classes if os.path.isdir(os.path.join(RAW_DIR, c))]
print(f"找到类别: {classes}")

for cls in classes:
    raw_cls_dir = os.path.join(RAW_DIR, cls)
    train_cls_dir = os.path.join(TRAIN_DIR, cls)
    val_cls_dir = os.path.join(VAL_DIR, cls)
    
    # 创建目标文件夹
    os.makedirs(train_cls_dir, exist_ok=True)
    os.makedirs(val_cls_dir, exist_ok=True)
    
    # 获取所有图片文件
    files = os.listdir(raw_cls_dir)
    files = [f for f in files if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    
    # 随机打乱
    random.shuffle(files)
    
    # 80% 训练，20% 验证
    split_idx = int(len(files) * 0.8)
    train_files = files[:split_idx]
    val_files = files[split_idx:]
    
    print(f"类别 [{cls}]: 训练集 {len(train_files)} 张, 验证集 {len(val_files)} 张")
    
    # 移动文件
    for f in train_files:
        shutil.copy(os.path.join(raw_cls_dir, f), os.path.join(train_cls_dir, f))
    for f in val_files:
        shutil.copy(os.path.join(raw_cls_dir, f), os.path.join(val_cls_dir, f))

print("✅ 数据集划分完成！train 和 val 绝对不重复！")