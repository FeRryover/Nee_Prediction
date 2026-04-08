import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from tqdm import tqdm

import torch
import sys

print(f"PyTorch Version: {torch.__version__}")
sys.stdout.flush()
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

from models import ExoTST
from utils.timefeatures import time_features

# 解决画图中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def tslib_data_loader(window, length_size, batch_size, data, data_mark, shuffle=True):
    seq_len = window
    sequence_length = seq_len + length_size
    num_samples = len(data) - sequence_length + 1
    print(f"TSLIB data generator start, seq_len: {seq_len}, length_size: {length_size}, total samples: {num_samples}")

    result = np.empty((num_samples, sequence_length, data.shape[1]), dtype=np.float32)
    result_mark = np.empty((num_samples, sequence_length, data_mark.shape[1]), dtype=np.float32)

    for i in range(num_samples):
        result[i] = data[i: i + sequence_length]
        result_mark[i] = data_mark[i: i + sequence_length]

    print("TSLIB window splicing finished...")

    x_temp = result[:, :-length_size]
    y_temp = result[:, -(length_size + int(window / 2)):]

    x_temp_mark = result_mark[:, :-length_size]
    y_temp_mark = result_mark[:, -(length_size + int(window / 2)):]

    x_temp = torch.tensor(x_temp).type(torch.float32)
    x_temp_mark = torch.tensor(x_temp_mark).type(torch.float32)
    y_temp = torch.tensor(y_temp).type(torch.float32)
    y_temp_mark = torch.tensor(y_temp_mark).type(torch.float32)

    ds = TensorDataset(x_temp, y_temp, x_temp_mark, y_temp_mark)
    dataloader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    return dataloader, x_temp, y_temp, x_temp_mark, y_temp_mark


def model_train_val(net, train_loader, val_loader, length_size, optimizer, criterion, scheduler, num_epochs, device,
                    early_patience=0.15, print_train=False):
    train_loss = []
    val_loss = []
    early_patience_epochs = int(early_patience * num_epochs)
    best_val_loss = float('inf')
    early_stop_counter = 0

    for epoch in range(num_epochs):
        total_train_loss = 0
        net.train()
        loop = tqdm(train_loader, total=len(train_loader), leave=True, desc=f"Epoch [{epoch+1}/{num_epochs}]")
        for i, (datapoints, labels, datapoints_mark, labels_mark) in enumerate(loop):
            datapoints, labels, datapoints_mark, labels_mark = datapoints.to(device), labels.to(
                device), datapoints_mark.to(device), labels_mark.to(device)
            optimizer.zero_grad()
            
            # ExoTST 保持原样 (因为它内部会切片)
            preds = net(datapoints, datapoints_mark, labels, labels_mark, None)
            preds = preds[:, -length_size:, -1:]
            labels = labels[:, -length_size:, -1:]
            
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_train_loss = total_train_loss / len(train_loader)
        train_loss.append(avg_train_loss)

        # 验证集
        net.eval()
        with torch.no_grad():
            total_val_loss = 0
            for val_x, val_y, val_x_mark, val_y_mark in val_loader:
                val_x, val_y, val_x_mark, val_y_mark = val_x.to(device), val_y.to(device), val_x_mark.to(
                    device), val_y_mark.to(device)
                
                pred_val_y = net(val_x, val_x_mark, val_y, val_y_mark, None)
                pred_val_y = pred_val_y[:, -length_size:, -1:]
                val_y_true = val_y[:, -length_size:, -1:]
                
                val_loss_batch = criterion(pred_val_y, val_y_true)
                total_val_loss += val_loss_batch.item()

            avg_val_loss = total_val_loss / len(val_loader)
            val_loss.append(avg_val_loss)
            scheduler.step(avg_val_loss)

        if print_train:
            loop.write(f"Epoch: {epoch + 1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stop_counter = 0
            torch.save(net.state_dict(), 'exotst_checkpoint.pth')
        else:
            early_stop_counter += 1
            if early_stop_counter >= early_patience_epochs:
                loop.write(f'Early stopping triggered at epoch {epoch + 1}.')
                break

    net.load_state_dict(torch.load('exotst_checkpoint.pth'))
    return net, train_loss, val_loss, epoch + 1


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
# 数据读取与预处理
# ==========================================
data_path = 'data/Yangtze River Delta of China/DT_NEE(20141201-20171130).csv'
#data_path = 'data/Yangtze River Delta of China/SX_NEE(20150715-20190424).csv'

dataset_name = os.path.splitext(os.path.basename(data_path))[0]

print(f"开始读取数据集: {data_path} ...")
df_raw = pd.read_csv(data_path)
print(f"数据读取完成, 原始形状: {df_raw.shape}")

df = data_cleansing(df_raw)
print(f"数据清洗完成, 清洗后形状: {df.shape}")

if 'Target' in df.columns:
    df.rename(columns={'Target': 'target'}, inplace=True)

# 特征工程 (保持简单，依赖模型自身融合)
for col in ['K↓', 'Tair', 'VPD']:
    for lag in range(1, 4):
        df[f'{col}_lag{lag}'] = df[col].shift(lag)

for col in ['K↓', 'Tair']:
    df[f'{col}_diff'] = df[col].diff()
print("特征工程完成(滞后+差分)...")

df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

# 准备时间标签
df_stamp = df[['date']].copy()
df_stamp['date'] = pd.to_datetime(df_stamp['date'])
data_stamp = time_features(df_stamp, timeenc=1, freq='h')

# 特征提取
feature_cols = [c for c in df.columns if c not in ['date', 'target']]
data_target = df[['target']].values
features = df[feature_cols].values

# 数据合并与归一化
data_full = np.concatenate((features, data_target), axis=1) # 18特征+1目标
data_length = len(data_full)
train_ratio, val_ratio = 0.6, 0.8
train_size = int(train_ratio * data_length)
val_size = int(val_ratio * data_length)

scaler = MinMaxScaler()
data_inverse = scaler.fit_transform(data_full)

data_train = data_inverse[:train_size, :]
data_train_mark = data_stamp[:train_size, :]
data_val = data_inverse[train_size: val_size, :]
data_val_mark = data_stamp[train_size: val_size, :]
data_test = data_inverse[val_size:, :]
data_test_mark = data_stamp[val_size:, :]

window = 96
length_size = 48
batch_size = 128
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_epochs = 80
learning_rate = 0.0001

# 准备 DataLoader
train_loader, _, _, _, _ = tslib_data_loader(window, length_size, batch_size, data_train, data_train_mark)
val_loader, _, _, _, _ = tslib_data_loader(window, length_size, batch_size, data_val, data_val_mark)
test_loader, _, _, _, _ = tslib_data_loader(window, length_size, batch_size, data_test, data_test_mark)




# 配置参数
data_dim = data_inverse.shape[1]

class Config:
    def __init__(self):
        self.seq_len = window
        self.pred_len = length_size
        self.enc_in = data_train.shape[1]
        self.d_model = 128
        self.n_heads = 8
        self.e_layers = 3
        self.d_ff = 256
        self.dropout = 0.1
        self.factor = 5
        self.activation = 'gelu'
        self.output_attention = False
        self.task_name = 'long_term_forecast'
        self.embed = 'timeF'
        self.freq = 'h'


config = Config()
model_type = 'SOTA_ExoTST'
net = ExoTST.Model(config).to(device)

criterion = nn.MSELoss().to(device)
optimizer = optim.Adam(net.parameters(), lr=learning_rate)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

# 模型训练 (采用统一的 model_train_val)
trained_model, train_loss, val_loss, final_epoch = model_train_val(net, train_loader, val_loader, length_size, 
                                                                    optimizer, criterion, scheduler, num_epochs,
                                                                    device, print_train=True)

# 评估
trained_model.eval()
preds = []
trues = []
with torch.no_grad():
    for x, y, x_mark, y_mark in test_loader:
        x, y, x_mark, y_mark = x.to(device), y.to(device), x_mark.to(device), y_mark.to(device)
        outputs = trained_model(x, x_mark, y, y_mark)
        preds.append(outputs.detach().cpu().numpy())
        trues.append(y[:, -length_size:, -1:].detach().cpu().numpy())

pred = np.concatenate(preds, axis=0)
true = np.concatenate(trues, axis=0)

# 取出预测和真实的目标列 (最后一列)
true = true[:, :, -1]
pred = pred[:, :, -1]

# 反归一化
# --- 改进：必须在原始单位的训练集上 fit，才能正确还原量纲 ---
y_scaler = MinMaxScaler()
raw_target_train = data_full[:train_size, -1:] 
y_scaler.fit(raw_target_train)

pred_uninverse = y_scaler.inverse_transform(pred.reshape(-1, 1)).reshape(pred.shape)
true_uninverse = y_scaler.inverse_transform(true.reshape(-1, 1)).reshape(true.shape)

# 我们只看最后一个步长的预测结果进行 R2 评估 (或者全序列)
true_final = true_uninverse[:, -1]
pred_final = pred_uninverse[:, -1]

df_eval = cal_eval(true_final, pred_final)
print(df_eval)

# ==========================================
# 结果保存
# ==========================================
now = datetime.now().strftime("%Y%m%d_%H%M%S")
run_folder_name = f"{model_type}_{now}_{dataset_name}"
output_dir = os.path.join('result', run_folder_name)
if not os.path.exists(output_dir): os.makedirs(output_dir)

print(f"\n[INFO] 结果将保存在: {output_dir}")
df_eval.to_csv(os.path.join(output_dir, f'{run_folder_name}_metrics.csv'), index=False, encoding='utf-8-sig')

# --- 改进：严谨的时间戳对齐 ---
test_dates = df['date'].iloc[val_size + window + length_size - 1 : val_size + window + length_size - 1 + len(true_final)].reset_index(drop=True)
result_df = pd.DataFrame({'时间': test_dates, '真实值': true_final, '预测值': pred_final})
result_df.to_csv(os.path.join(output_dir, f'{run_folder_name}_data.csv'), index=False, encoding='utf-8-sig')

plt.figure(figsize=(12, 4))
plt.plot(true_final, label='Real', color='blue', alpha=0.5)
plt.plot(pred_final, label='Predict', color='red', alpha=0.8)
plt.title(f'{model_type} Result ({dataset_name})')
plt.legend()
plt.savefig(os.path.join(output_dir, f'{run_folder_name}.png'), bbox_inches='tight')
plt.show()
