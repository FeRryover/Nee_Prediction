import pandas as pd
import numpy as np
import matplotlib
# 如果画图报错或者不弹窗，请取消注释下面这行代码
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
from torch.utils.data import DataLoader, TensorDataset

# ==========================================
# 核心修改：导入纯 TCN 模型
# ==========================================
from models import TCN
from utils.timefeatures import time_features

# 解决画图中文显示问题
plt.rc('font', family='sans-serif')
plt.style.use("ggplot")
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def tslib_data_loader(window, length_size, batch_size, data, data_mark):
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
    dataloader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    return dataloader, x_temp, y_temp, x_temp_mark, y_temp_mark


def model_train(net, train_loader, length_size, optimizer, criterion, num_epochs, device):
    train_loss = []

    for epoch in range(num_epochs):
        total_train_loss = 0
        net.train()
        loop = tqdm(train_loader, total=len(train_loader), leave=True, desc=f"Epoch [{epoch + 1}/{num_epochs}]")
        for i, (datapoints, labels, datapoints_mark, labels_mark) in enumerate(loop):
            datapoints, labels, datapoints_mark, labels_mark = datapoints.to(device), labels.to(
                device), datapoints_mark.to(device), labels_mark.to(device)

            optimizer.zero_grad()
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

        # 优化输出避免错行
        loop.set_description(f"Epoch [{epoch + 1}/{num_epochs}] Train Loss: {avg_train_loss:.4f}")

    return net, train_loss


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
# 数据读取与预处理 (与基线完全一致)
# ==========================================
#data_path = 'data/Yangtze River Delta of China/DT_NEE(20141201-20171130).csv'
data_path = 'data/Yangtze River Delta of China/SX_NEE(20150715-20190424).csv'
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

df_stamp = df[['date']].copy()
df_stamp['date'] = pd.to_datetime(df_stamp['date'])
data_stamp = time_features(df_stamp, timeenc=1, freq='h')

data_length = len(df)
train_set = 0.8
train_size = int(train_set * data_length)

features_train = features[:train_size, :]
features_test = features[train_size:, :]
target_train = data_target[:train_size, :]
target_test = data_target[train_size:, :]
data_stamp_train = data_stamp[:train_size, :]
data_stamp_test = data_stamp[train_size:, :]

scaler_pca = StandardScaler()
features_train_scaled = scaler_pca.fit_transform(features_train)
features_test_scaled = scaler_pca.transform(features_test)

pca = PCA(n_components=0.95)
features_train_pca = pca.fit_transform(features_train_scaled)
features_test_pca = pca.transform(features_test_scaled)

data_train_raw = np.concatenate((features_train_pca, target_train), axis=1)
data_test_raw = np.concatenate((features_test_pca, target_test), axis=1)

scaler = MinMaxScaler()
data_train = scaler.fit_transform(data_train_raw)
data_test = scaler.transform(data_test_raw)

data_train_mark = data_stamp_train
data_test_mark = data_stamp_test
data_dim = data_train.shape[1]

window = 96
length_size = 48
batch_size = 128
num_epochs = 100
learning_rate = 0.0001

train_loader, x_train, y_train, x_train_mark, y_train_mark = tslib_data_loader(window, length_size, batch_size,
                                                                               data_train, data_train_mark)
test_loader, x_test, y_test, x_test_mark, y_test_mark = tslib_data_loader(window, length_size, batch_size, data_test,
                                                                          data_test_mark)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ==========================================
# 配置参数：严格对齐消融实验
# ==========================================
class Config:
    def __init__(self):
        self.enc_in = data_dim
        self.dec_in = data_dim
        self.c_out = 1
        self.pred_len = length_size

        # 核心对齐参数
        self.d_model = 64
        #self.e_layers = 3   3不能让TCN看完感受野
        self.e_layers = 5
        self.dropout = 0.1

        # 兼容项 (TCN 实际不使用, 但防止代码冲突保留)
        self.task_name = 'short_term_forecast'


config = Config()
model_type = 'Ablation_TCN'
net = TCN.Model(config).to(device)

criterion = nn.MSELoss().to(device)
optimizer = optim.Adam(net.parameters(), lr=learning_rate)

# ==========================================
# 训练与预测
# ==========================================
trained_model, train_loss = model_train(net, train_loader, length_size, optimizer, criterion, num_epochs, device)

trained_model.eval()
preds = []
trues = []
with torch.no_grad():
    for x, y, x_mark, y_mark in test_loader:
        x, y, x_mark, y_mark = x.to(device), y.to(device), x_mark.to(device), y_mark.to(device)
        outputs = trained_model(x, x_mark, y, y_mark)
        preds.append(outputs.detach().cpu().numpy())
        trues.append(y[:, -length_size:, -1:].detach().cpu().numpy())


# pred = np.concatenate(preds, axis=0)[:, :, -1]
# true = np.concatenate(trues, axis=0)[:, :, -1]
#
# # 使用目标列的 Scaler 进行精确反归一化
# target_scaler = MinMaxScaler()
# target_scaler.fit(data_target)
# pred_uninverse = target_scaler.inverse_transform(pred[:, -1:])
# true_uninverse = target_scaler.inverse_transform(true[:, -1:])

# 合并所有 Batch -> 形状 [样本总数, 48, 1]
full_pred = np.concatenate(preds, axis=0)
full_true = np.concatenate(trues, axis=0)

# 【关键点】提取预测序列的最后一个时间步进行点对点对比
# 形状从 [N, 48, 1] 变为 [N, 1]
final_pred_point = full_pred[:, -1, :].reshape(-1, 1)
final_true_point = full_true[:, -1, :].reshape(-1, 1)

# 重新初始化并拟合针对目标列的 Scaler
# --- 改进：在原始单位的训练集上 fit，还原真实物理量纲 ---
target_scaler = MinMaxScaler()
raw_target_train = data_target[:train_size, :]
target_scaler.fit(raw_target_train)

# 执行反归一化
pred_uninverse = target_scaler.inverse_transform(final_pred_point)
true_uninverse = target_scaler.inverse_transform(final_true_point)

true, pred = true_uninverse, pred_uninverse

df_eval = cal_eval(true, pred)
print(df_eval)

# ==========================================
# 结果保存
# ==========================================
output_dir = 'result'
now = datetime.now().strftime("%Y%m%d_%H%M%S")
run_folder_name = f"{model_type}_{now}_{dataset_name}"
output_dir = os.path.join('result', run_folder_name)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print(f"\n==========================================")
print(f"[INFO] 纯 TCN 消融实验结果将保存在: {output_dir}")
print(f"==========================================")

metrics_filename = f'{run_folder_name}_metrics.csv'
metrics_path = os.path.join(output_dir, metrics_filename)
df_eval.to_csv(metrics_path, index=False, encoding='utf-8-sig')

# --- 改进：严谨的时间戳对齐 ---
test_dates = df['date'].iloc[val_size + window + length_size - 1 : val_size + window + length_size - 1 + len(true_final)].reset_index(drop=True)
data_filename = f'{run_folder_name}_data.csv'
data_path = os.path.join(output_dir, data_filename)
result_df = pd.DataFrame({'时间': test_dates, '真实值': true_final.ravel(), '预测值': pred_final.ravel()})
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
plt.show()