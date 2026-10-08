# 面向边缘设备的轻量化垃圾分类算法

## 项目简介

针对边缘设备中计算资源有限与分类精度不足之间的矛盾，本项目提出一种面向低资源场景的轻量化视觉分类优化框架。该框架通过深度可分离卷积降低计算复杂度，引入SE注意力机制增强关键视觉特征表达，并结合极小学习率迁移微调策略提高小样本条件下的模型稳定性。

在垃圾分类任务中，相比ResNet18基线，本方案使模型参数量降低53%，验证准确率反超2.56%（达93.7%），实现了低资源约束下精度与效率的双重突破。

**作者**：Tom

---

## 环境配置与依赖

为了保证实验的严谨性与可复现性，本项目运行环境如下：

**硬件环境**：NVIDIA RTX 5060 GPU（用于训练加速）/ Intel Core Ultra 7 255HX（用于CPU推理测试）

**软件环境**：Windows 11，Python 3.14.6，CUDA 12.8，PyTorch 2.12.0.dev20260408+cu128

### 1. 安装基础依赖

```bash
pip install -r requirements.txt
```

### 2. 安装PyTorch（GPU版本）

为了利用NVIDIA显卡（如RTX5060）加速训练，需前往[PyTorch官网](https://pytorch.org/get-started/locally/?spm=a2ty_o01.29997173.0.0.2f64c921wavvsz)获取适合您的CUDA版本的安装命令。例如：

```bash
#示例：CUDA 12.8版本
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

---

## 数据集准备与严谨划分

为了保证实验的严谨性，杜绝数据泄露，本项目采用**8:2随机不重复划分**策略。

1. 请将下载的原始垃圾分类数据集（包含 6 个类别文件夹）放入 `data/raw/` 目录下。请前往获取适合您 CUDA 版本的安装命令。例如
   结构应如下：

```textile
data/
└── raw/
    ├── cardboard/
    ├── glass/
    ├── metal/
    ├── paper/
    ├── plastic/
    └── trash/
```

2. 运行数据划分脚本，自动将数据分配至`train`和`val`文件夹：

```bash
python src/split_data.py
```

**注： 如需再次运行该脚本，要将`train`和`val`文件夹中的数据删除以防数据泄露。**

---

## 运行指南

所有核心代码均位于`src/`目录。

### 1. 训练基线模型（ResNet18)

```bash
python src/train_ResNet18.py
```

*训练完成后，最佳模型权重将保存至 `models/best_resnet_finetune.pth`，Loss/Acc曲线保存至 `outputs/`。*

### 2. 训练轻量化创新模型（MobileNetV3)

```bash
python src/train_MobileNetV3.py
```

*训练完成后，最佳模型权重将保存至`model/best_mobilenet_finetune.pth`，Loss/Acc曲线保存至 `outputs/`。*

## 3. 复现报告中的“边缘部署效率指标”

为了验证模型在低资源设备上的部署优势，我们提供了专门的评估脚本，用于计算 **FLOPs、模型体积与CPU推理延迟**。

```bash
python src/test_efficiency.py
```

*运行完成后，终端将输出与报告一致的参数量、FLOPs和CPU推理时间（ms）对比数据。*

### 4. 生成混淆矩阵（细粒度误差分析）

为了深入分析模型的细粒度分类能力，运行以下脚本生成混淆矩阵图：

```bash
python src/plot_resnet_cm.py
python src/plot_mobilenet_cm.py
```

*运行完成后，两张混淆矩阵图将保存至`outputs/`，并以不同颜色进行区分。*

---

## 项目结构

```textile
Garbage_Classification_Project/
│
│── data/                   # 数据集目录
│   |── raw/                # 原始数据集（需手动放入）
│   |── train/              # 训练（由split_data.py自动生成）
│   └── val/                # 验证集 (由split_data.py自动生成）
│
│── src/                    # 核心源代码
│   |── split_data.py       # 数据集8:2随机划分脚本
│   |—— train_ResNet18.py   # ResNet18训练脚本
│   |—— train_MobileNetV3.py # MobileNetV3训练脚本
│   |—— test_efficiency.py  # 边缘部署效率指标(FLOPs/推理时间)测试脚本 
│   |—— plot_resnet_cm.py   # ResNet18混淆矩阵生成脚本
│   └── plot_mobilenet_cm.py # MobileNetV3混淆矩阵生成脚本
│    
│── models/                 # 训练好的模型权重（.pth）
│── outputs/                # 实验结果图表（Loss曲线、混淆矩阵）
|── requirements.txt        # Python依赖清单
└── README.md               # 项目说明文档
```
