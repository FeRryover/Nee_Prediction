import pandas as pd
import numpy as np
import copy
import matplotlib

# 本地运行，保留这行以防弹窗报错
matplotlib.use('Agg')
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

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(CURRENT_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

# ==========================================
# 核心导入：PatchTST 模型
# ==========================================
from models import PatchTST
from utils.timefeatures import time_features

# 解决画图中文显示问题
plt.rc('font', family='sans-serif')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'SimHei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use("ggplot")


def env_int(name, default):
    return int(os.getenv(name, str(default)))


def env_float(name, default):
    return float(os.getenv(name, str(default)))


def tslib_data_loader(window, length_size, batch_size, data, data_mark, shuffle=True):
    seq_len = window
    sequence_length = seq_len + length_size
    num_samples = len(data) - sequence_length + 1

    result = np.empty((num_samples, sequence_length, data.shape[1]), dtype=np.float32)
    result_mark = np.empty((num_samples, sequence_length, data_mark.shape[1]), dtype=np.float32)

    for i in range(num_samples):
        result[i] = data[i: i + sequence_length]
        result_mark[i] = data_mark[i: i + sequence_length]

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
    best_state_dict = None
    early_stop_counter = 0

    for epoch in range(num_epochs):
        total_train_loss = 0
        net.train()
        loop = tqdm(train_loader, total=len(train_loader), leave=True, desc=f"Epoch [{epoch+1}/{num_epochs}]")
        for i, (datapoints, labels, datapoints_mark, labels_mark) in enumerate(loop):
            datapoints, labels = datapoints.to(device), labels.to(device)
            optimizer.zero_grad()
            
            # PatchTST forward (encoder-only)
            preds = net(datapoints, None, None, None)
            loss = criterion(preds, labels[:, -length_size:, -1:])
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
                val_x, val_y = val_x.to(device), val_y.to(device)
                pred_val_y = net(val_x, None, None, None)
                val_loss_batch = criterion(pred_val_y, val_y[:, -length_size:, -1:])
                total_val_loss += val_loss_batch.item()

            avg_val_loss = total_val_loss / len(val_loader)
            val_loss.append(avg_val_loss)
            scheduler.step(avg_val_loss)

        if print_train:
            loop.write(f"Epoch: {epoch + 1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stop_counter = 0
            best_state_dict = copy.deepcopy(net.state_dict())
        else:
            early_stop_counter += 1
            if early_stop_counter >= early_patience_epochs:
                loop.write(f'Early stopping triggered at epoch {epoch + 1}.')
                break

    if best_state_dict is not None:
        net.load_state_dict(best_state_dict)
    return net, train_loss, val_loss, epoch + 1


def cal_eval(y_real, y_pred):
    y_real, y_pred = np.array(y_real).ravel(), np.array(y_pred).ravel()

    r2 = r2_score(y_real, y_pred)
    mse = mean_squared_error(y_real, y_pred)
    rmse = np.sqrt(mse)
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
#data_path = 'data/Yangtze River Delta of China/DT_NEE(20141201-20171130).csv'
data_path_input = os.getenv('DATA_PATH', 'data/Yangtze River Delta of China/SX_NEE(20150715-20190424).csv')
data_path = data_path_input if os.path.isabs(data_path_input) else os.path.join(PROJECT_DIR, data_path_input)
dataset_name = os.path.splitext(os.path.basename(data_path))[0]

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

data_length = len(features)
train_ratio, val_ratio = 0.6, 0.8
train_size = int(train_ratio * data_length)
val_size = int(val_ratio * data_length)

use_pca = bool(env_int('HP_USE_PCA', 0))
if use_pca:
    print("[INFO] PCA mode enabled (n_components=0.95)")
    scaler_features = StandardScaler()
    train_features = features[:train_size, :]
    scaler_features.fit(train_features)
    features_scaled = scaler_features.transform(features)

    pca = PCA(n_components=0.95)
    pca.fit(features_scaled[:train_size, :])
    features_used = pca.transform(features_scaled)
else:
    features_used = features

df_stamp = df[['date']].copy()
df_stamp['date'] = pd.to_datetime(df_stamp['date'])
data_stamp = time_features(df_stamp, timeenc=1, freq='h')

# 数据合并与归一化
data_full = np.concatenate((features_used, data_target), axis=1)

scaler = MinMaxScaler()
data_train_raw = data_full[:train_size, :]
scaler.fit(data_train_raw)
data_scaled = scaler.transform(data_full)

data_train = data_scaled[:train_size, :]
data_train_mark = data_stamp[:train_size, :]
data_val = data_scaled[train_size: val_size, :]
data_val_mark = data_stamp[train_size: val_size, :]
data_test = data_scaled[val_size:, :]
data_test_mark = data_stamp[val_size:, :]

window = env_int('HP_WINDOW', 96)
length_size = env_int('HP_LENGTH', 48)
batch_size = env_int('HP_BATCH_SIZE', 64)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_epochs = env_int('HP_EPOCHS', 120)
learning_rate = env_float('HP_LR', 0.0003)

# 准备 DataLoader
train_loader, _, _, _, _ = tslib_data_loader(window, length_size, batch_size, data_train, data_train_mark, shuffle=True)
val_loader, _, _, _, _ = tslib_data_loader(window, length_size, batch_size, data_val, data_val_mark, shuffle=False)
test_loader, _, _, _, _ = tslib_data_loader(window, length_size, batch_size, data_test, data_test_mark, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ==========================================
# 配置参数：严格对齐消融实验
# ==========================================
# 配置参数
data_dim = data_scaled.shape[1]

class Config:
    def __init__(self):
        self.enc_in = data_dim
        self.dec_in = data_dim
        self.c_out = 1
        self.pred_len = length_size

        # 核心对齐参数
        self.d_model = env_int('HP_D_MODEL', 128)
        self.e_layers = env_int('HP_E_LAYERS', 4)
        self.dropout = env_float('HP_DROPOUT', 0.2)


config = Config()
model_type = 'PatchTST_Best'
net = PatchTST.Model(config).to(device)

criterion = nn.MSELoss().to(device)
optimizer = optim.Adam(net.parameters(), lr=learning_rate)

# 学习率调度器
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=env_int('HP_SCHED_PATIENCE', 8))

# 训练与预测 (采用统一模式)
trained_model, train_loss, val_loss, final_epoch = model_train_val(net, train_loader, val_loader, length_size, 
                                                                    optimizer, criterion, scheduler, num_epochs, 
                                                                    device, print_train=True)

trained_model.eval()
preds = []
trues = []
with torch.no_grad():
    for x, y, x_mark, y_mark in test_loader:
        x, y, x_mark, y_mark = x.to(device), y.to(device), x_mark.to(device), y_mark.to(device)
        outputs = trained_model(x, None, None, None)
        preds.append(outputs.detach().cpu().numpy())
        trues.append(y[:, -length_size:, -1:].detach().cpu().numpy())

# 合并所有 Batch -> 形状 [样本总数, 48, 1]
full_pred = np.concatenate(preds, axis=0)
full_true = np.concatenate(trues, axis=0)

# 【关键点】提取预测序列的最后一个时间步进行点对点对比
# 形状从 [N, 48, 1] 变为 [N, 1]
final_pred_point = full_pred[:, -1, :].reshape(-1, 1)
final_true_point = full_true[:, -1, :].reshape(-1, 1)

# 重新初始化并拟合针对目标列的 Scaler
target_scaler = MinMaxScaler()
target_scaler.fit(data_target[:train_size])

# 执行反归一化
pred_uninverse = target_scaler.inverse_transform(final_pred_point)
true_uninverse = target_scaler.inverse_transform(final_true_point)

true, pred = true_uninverse, pred_uninverse

df_eval = cal_eval(true, pred)
print(df_eval)

# ==========================================
# 结果保存
# ==========================================
now = datetime.now().strftime("%Y%m%d_%H%M%S")
run_folder_name = f"{model_type}_{now}_{dataset_name}"
base_output_dir = os.getenv('MODEL_OUTPUT_DIR', 'result_best')
output_dir = os.path.join(base_output_dir, run_folder_name)
os.makedirs(output_dir, exist_ok=True)

print(f"\n==========================================")
print(f"[INFO] 结果将保存在: {output_dir}")
print(f"==========================================")

metrics_filename = f'{run_folder_name}_metrics.csv'
metrics_path = os.path.join(output_dir, metrics_filename)
df_eval.to_csv(metrics_path, index=False, encoding='utf-8-sig')

test_dates = df['date'].iloc[val_size + window + length_size - 1 : val_size + window + length_size - 1 + len(true)].reset_index(drop=True)
data_filename = f'{run_folder_name}_data.csv'
data_path = os.path.join(output_dir, data_filename)
result_df = pd.DataFrame({'时间': test_dates, '真实值': true.flatten(), '预测值': pred.flatten()})
result_df.to_csv(data_path, index=False, encoding='utf-8-sig')

df_pred_true = pd.DataFrame({'Predict': pred.flatten(), 'Real': true.flatten()})
plt.figure(figsize=(12, 4))
plt.plot(df_pred_true['Predict'], label='Predict', color='red', alpha=0.8)
plt.plot(df_pred_true['Real'], label='Real', color='blue', alpha=0.5)
plt.title(f'{model_type} Result ({dataset_name})')
plt.legend()

img_filename = f'{run_folder_name}.png'
img_save_path = os.path.join(output_dir, img_filename)
plt.savefig(img_save_path, bbox_inches='tight')

# 恢复本地跑图时的弹窗功能
plt.show()