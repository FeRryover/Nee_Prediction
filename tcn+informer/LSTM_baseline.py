import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # <--- 加上这一行，强制使用独立的弹窗显示图片
import matplotlib.pyplot as plt
import os
from datetime import datetime

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

plt.rc('font', family='sans-serif')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'SimHei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use("ggplot")


# ==========================================
# 1. 核心评估与数据清洗函数 (完美复用)
# ==========================================
def cal_eval(y_real, y_pred):
    y_real, y_pred = np.array(y_real).ravel(), np.array(y_pred).ravel()
    r2 = r2_score(y_real, y_pred)
    mse = mean_squared_error(y_real, y_pred)
    rmse = np.sqrt(mean_squared_error(y_real, y_pred))
    mae = mean_absolute_error(y_real, y_pred)
    mape = mean_absolute_percentage_error(y_real, y_pred) * 100
    df_eval = pd.DataFrame({'R2': r2, 'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'MAPE': mape}, index=['Eval'])
    return df_eval


def data_cleansing(df):
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(by='date').reset_index(drop=True)
    cols = [c for c in df.columns if c != 'date']
    df[cols] = df[cols].interpolate(method='linear', limit=6)
    if 'date' in df.columns:
        df.set_index('date', inplace=True)
        df[cols] = df[cols].interpolate(method='time', limit=48)
        df.reset_index(inplace=True)
    df[cols] = df[cols].ffill().bfill()
    return df


# ==========================================
# 2. LSTM 专用的滑动窗口数据加载器
# ==========================================
def create_lstm_dataset(data, window, length_size):
    """
    将序列数据转换为 LSTM 需要的 3D 张量 (Samples, Seq_len, Features)
    """
    sequence_length = window + length_size
    num_samples = len(data) - sequence_length + 1

    result = np.empty((num_samples, sequence_length, data.shape[1]), dtype=np.float32)
    for i in range(num_samples):
        result[i] = data[i: i + sequence_length]

    # X: 截取前 window 步 (保留所有特征维度)
    X = result[:, :window, :]
    # Y: 截取后 length_size 步的目标变量 (最后一列)
    Y = result[:, -length_size:, -1:]

    return torch.tensor(X), torch.tensor(Y)


# ==========================================
# 3. 读取数据与特征工程 (严格对齐你的预处理)
# ==========================================
#data_path = 'data/Yangtze River Delta of China/DT_NEE(20141201-20171130).csv'
data_path = 'data/Yangtze River Delta of China/SX_NEE(20150715-20190424).csv'
dataset_name = os.path.splitext(os.path.basename(data_path))[0]

print(f"开始读取数据集: {data_path} ...")
df_raw = pd.read_csv(data_path)
df = data_cleansing(df_raw)

if 'Target' in df.columns:
    df.rename(columns={'Target': 'target'}, inplace=True)

for col in ['K↓', 'Tair', 'VPD']:
    for lag in range(1, 4):
        df[f'{col}_lag{lag}'] = df[col].shift(lag)

for col in ['K↓', 'Tair']:
    df[f'{col}_diff'] = df[col].diff()

df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

feature_cols = [c for c in df.columns if c not in ['date', 'target']]
data_target = df[['target']].values
features = df[feature_cols].values

# 严格的时间序列切分 (80% 训练, 20% 测试)
data_length = len(df)
train_size = int(0.8 * data_length)

features_train = features[:train_size, :]
features_test = features[train_size:, :]
target_train = data_target[:train_size, :]
target_test = data_target[train_size:, :]

# 标准化与 PCA 降维
scaler_pca = StandardScaler()
features_train_scaled = scaler_pca.fit_transform(features_train)
features_test_scaled = scaler_pca.transform(features_test)

pca = PCA(n_components=0.95)
features_train_pca = pca.fit_transform(features_train_scaled)
features_test_pca = pca.transform(features_test_scaled)

# 拼接特征与目标
data_train_raw = np.concatenate((features_train_pca, target_train), axis=1)
data_test_raw = np.concatenate((features_test_pca, target_test), axis=1)

# 全局输入归一化
scaler = MinMaxScaler()
data_train = scaler.fit_transform(data_train_raw)
data_test = scaler.transform(data_test_raw)

# ==========================================
# 4. 构建 PyTorch DataLoader
# ==========================================
window = 96  # 输入过去 96 步
length_size = 48  # 预测未来 48 步
batch_size = 128
input_size = data_train.shape[1]  # 特征维度

print("正在构建 LSTM 数据集张量...")
x_train, y_train = create_lstm_dataset(data_train, window, length_size)
x_test, y_test = create_lstm_dataset(data_test, window, length_size)

train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)


# ==========================================
# 5. 定义 LSTM 模型架构
# ==========================================
class LSTMForecastModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMForecastModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM 核心层
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.1)
        # 全连接层：将最后一步的隐藏状态映射为未来 48 步的预测
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        out, _ = self.lstm(x)
        # 提取序列中最后一个时间步的输出: out shape 变为 (batch, hidden_size)
        out = out[:, -1, :]
        # 全连接层映射: out shape 变为 (batch, output_size)
        out = self.fc(out)
        # 增加最后一个维度以匹配 y 的 shape: (batch, output_size, 1)
        return out.unsqueeze(-1)


# ==========================================
# 6. 模型训练配置
# ==========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
hidden_size = 64
num_layers = 2
num_epochs = 100
learning_rate = 0.001

model = LSTMForecastModel(input_size, hidden_size, num_layers, length_size).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 早停机制 (Early Stopping) 简易实现
best_loss = float('inf')
patience = 15
patience_counter = 0

print(f"\n开始在 {device} 上训练 LSTM 模型...")
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    loop = tqdm(train_loader, total=len(train_loader), leave=False, desc=f"Epoch [{epoch + 1}/{num_epochs}]")

    for batch_x, batch_y in loop:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch: {epoch + 1:03d} | Train Loss: {avg_loss:.6f}")

    # 早停判断
    if avg_loss < best_loss:
        best_loss = avg_loss
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"连续 {patience} 轮 Loss 未下降，触发 Early Stopping!")
            break

# ==========================================
# 7. 模型预测与反归一化
# ==========================================
model.eval()
preds, trues = [], []

with torch.no_grad():
    for batch_x, batch_y in test_loader:
        batch_x = batch_x.to(device)
        outputs = model(batch_x)
        preds.append(outputs.cpu().numpy())
        trues.append(batch_y.numpy())

# 拼接所有批次
pred_array = np.concatenate(preds, axis=0)
true_array = np.concatenate(trues, axis=0)

# 降维到 2D 以进行反归一化: (Samples, 48)
pred_2d = pred_array.squeeze(-1)
true_2d = true_array.squeeze(-1)

print("\n训练完成，正在反归一化...")
# 重新建立仅针对 Target 的 Scaler 以进行逆变换
# --- 改进：在训练集范围上 fit，确保反归一化准确 ---
scaler_target = MinMaxScaler()
raw_target_train = data_full[:train_size, -1:]
scaler_target.fit(raw_target_train)

# 获取最后一步(最远预测)作为评估标准
pred_final = scaler_target.inverse_transform(pred_2d[:, -1:])
true_final = scaler_target.inverse_transform(true_2d[:, -1:])

df_eval = cal_eval(true_final, pred_final)
print("\n====== LSTM 模型评估结果 ======")
print(df_eval)

# ==========================================
# 8. 统一且规范的文件保存逻辑 (专属文件夹版)
# ==========================================
now = datetime.now().strftime("%Y%m%d_%H%M%S")
run_folder_name = f"LSTM_{now}_{dataset_name}"
output_dir = os.path.join('result', run_folder_name)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print(f"\n==========================================")
print(f"[INFO] 本次运行的所有结果将保存在: {output_dir}")
print(f"==========================================")

# 保存评估指标 (Metrics)
metrics_filename = f'{run_folder_name}_metrics.csv'
metrics_path = os.path.join(output_dir, metrics_filename)
df_eval.to_csv(metrics_path, index=False, encoding='utf-8-sig')

# 保存真实值和预测值 (Data)
# --- 改进：严谨的时间戳对齐 ---
test_dates = df['date'].iloc[val_size + window + length_size - 1 : val_size + window + length_size - 1 + len(true_final)].reset_index(drop=True)
data_filename = f'{run_folder_name}_data.csv'
data_path = os.path.join(output_dir, data_filename)
result_df = pd.DataFrame({'时间': test_dates, '真实值': true_final.ravel(), '预测值': pred_final.ravel()})
result_df.to_csv(data_path, index=False, encoding='utf-8-sig')

# 绘制并保存结果图 (Image)
plt.figure(figsize=(12, 4))
plt.plot(result_df['预测值'], label='Predict', color='red', alpha=0.8)
plt.plot(result_df['真实值'], label='Real', color='blue', alpha=0.5)
plt.title(f'LSTM Result ({dataset_name})')
plt.legend()

img_filename = f'{run_folder_name}.png'
img_save_path = os.path.join(output_dir, img_filename)
plt.savefig(img_save_path, bbox_inches='tight')
print(f'[SUCCESS] 实验文件已全部保存完毕！\n')

plt.show()