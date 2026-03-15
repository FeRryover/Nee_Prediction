# 🌾 农业生态系统碳通量预测

基于 **TCN-Informer** 模型的农业生态系统净生态系统交换（NEE）预测研究。

## 🚀 项目概述

本项目旨在利用深度学习模型预测农业生态系统的碳通量，以支持可持续农业和气候变化研究。通过结合 **TCN（时间卷积网络）** 的局部特征提取能力和 **Informer** 的长序列建模能力，构建高效的碳通量预测模型。

## 🛠️ 技术栈

- **Python 3.x**
- **PyTorch** - 深度学习框架
- **TCN** - 时间卷积网络
- **Informer** - 长序列预测模型
- **Pandas** - 数据处理
- **Scikit-learn** - 模型评估

## 📂 项目结构

```
Nee_Prediction/
├── data/                     # 数据集目录
│   ├── nee_data.csv          # 碳通量数据
│   └── ...
├── models/                   # 模型定义
│   ├── tcn_informer.py       # TCN-Informer 模型
│   └── ...
├── scripts/                  # 训练与评估脚本
│   ├── train.py              # 模型训练
│   ├── evaluate.py           # 模型评估
│   └── ...
├── results/                  # 实验结果
│   ├── predictions.csv       # 预测结果
│   └── metrics.json          # 评估指标
├── utils/                    # 工具函数
│   ├── data_loader.py        # 数据加载与预处理
│   └── metrics.py            # 评估指标计算
├── README.md                 # 项目说明
└── requirements.txt          # 依赖库
```

## 📋 安装与配置

### 1. 克隆仓库

```bash
git clone <repository-url>
cd Nee_Prediction
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 数据准备

将您的碳通量数据集放入 `data/` 目录，并确保格式正确：

```csv
# 示例数据格式
Date,NEE,PAR,Temp,Humidity,WindSpeed,Precipitation
2023-01-01,1.2,450,25.5,60.2,3.1,0.0
2023-01-02,0.8,420,24.8,62.5,2.8,0.5
...
```

## 🏃 运行模型

### 训练模型

```bash
python scripts/train.py --config configs/default.yaml
```

**配置参数：**

- `--config`: 配置文件路径
- `--epochs`: 训练轮数
- `--batch_size`: 批大小
- `--learning_rate`: 学习率
- `--window_size`: 输入序列长度
- `--horizon`: 预测 horizon

### 评估模型

```bash
python scripts/evaluate.py --model_path results/best_model.pth --data_path data/nee_data.csv
```

## 📊 评估指标

模型评估使用以下指标：

- **MAE** (Mean Absolute Error)
- **MSE** (Mean Squared Error)
- **RMSE** (Root Mean Squared Error)
- **R²** (R-squared)
- **MAPE** (Mean Absolute Percentage Error)

## 📈 模型架构

```
Input Sequence → TCN Blocks → Attention → FeedForward → Output
```

- **TCN Blocks**: 提取局部特征和时间依赖
- **Attention**: 捕捉长距离依赖
- **FeedForward**: 非线性变换

## 🧪 实验结果

训练完成后，结果将保存在 `results/` 目录：

- `predictions.csv`: 预测的碳通量值
- `metrics.json`: 评估指标
- `training_log.txt`: 训练日志

## 🔧 扩展与定制

### 添加新的特征

1. 修改 `utils/data_loader.py` 中的特征工程部分
2. 更新 `configs/default.yaml` 中的特征配置
3. 重新训练模型

### 调整模型参数

修改 `configs/default.yaml` 文件：

```yaml
tcn:
  num_layers: 3
  kernel_size: 3
  dropout: 0.1

informer:
  num_heads: 8
  d_model: 256
  dropout: 0.1
```

## 🤝 贡献

欢迎贡献！请遵循以下步骤：

1. Fork 本项目
2. 创建新分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开 Pull Request

## 📄 许可证

本项目采用 MIT 许可证。

## 📧 联系

如果您有任何问题或建议，请联系：

- 邮箱: [EMAIL_ADDRESS]
- GitHub: [Your GitHub Profile]

## 🙏 致谢

- 感谢 TCN 和 Informer 模型的原始作者
- 感谢提供高质量农业生态系统数据的研究者

---

**祝您研究顺利！** 🌾✨
