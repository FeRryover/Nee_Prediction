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

plt.rc('font', family='sans-serif')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'SimHei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False # 解决负号显示
plt.style.use("ggplot")

# ==========================================
# 核心修改 1：导入标准 Informer 进行消融实验
# ==========================================
from models import Informer
from utils.timefeatures import time_features

# 解决画图中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def tslib_data_loader(window, length_size, batch_size, data, data_mark):
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
    dataloader = DataLoader(ds, batch_size=batch_size, shuffle=True)

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
            
            # --- 核心改进：同步训练掩码 ---
            labels_masked = labels.clone()
            labels_masked[:, -length_size:, -1] = 0
            
            preds = net(datapoints, datapoints_mark, labels_masked, labels_mark, None)
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
                
                val_y_masked = val_y.clone()
                val_y_masked[:, -length_size:, -1] = 0
                
                pred_val_y = net(val_x, val_x_mark, val_y_masked, val_y_mark, None)
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
            torch.save(net.state_dict(), 'informer_checkpoint.pth')
        else:
            early_stop_counter += 1
            if early_stop_counter >= early_patience_epochs:
                loop.write(f'Early stopping triggered at epoch {epoch + 1}.')
                break

    net.load_state_dict(torch.load('informer_checkpoint.pth'))
    return net, train_loss, val_loss, epoch + 1


def cal_eval(y_real, y_pred):
    y_real, y_pred = np.array(y_real).ravel(), np.array(y_pred).ravel()

    r2 = r2_score(y_real, y_pred)
    mse = mean_squared_error(y_real, y_pred)
    rmse = np.sqrt(mean_squared_error(y_real, y_pred))
    mae = mean_absolute_error(y_real, y_pred)
    # Note: If target contains 0 (e.g., carbon flux equilibrium), MAPE might be extremely large.
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
data_path = 'data/Yangtze River Delta of China/SX_NEE(20150715-20190424).csv'
dataset_name = os.path.splitext(os.path.basename(data_path))[0]

print(f"开始读取数据集: {data_path} ...")
df_raw = pd.read_csv(data_path)
print(f"数据读取完成, 原始形状: {df_raw.shape}")

df = data_cleansing(df_raw)
print(f"数据清洗完成, 清洗后形状: {df.shape}")

if 'Target' in df.columns:
    df.rename(columns={'Target': 'target'}, inplace=True)

# 特征工程
for col in ['K↓', 'Tair', 'VPD']:
    for lag in range(1, 4):
        df[f'{col}_lag{lag}'] = df[col].shift(lag)

for col in ['K↓', 'Tair']:
    df[f'{col}_diff'] = df[col].diff()
print("特征工程完成(滞后+差分)...")

df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

feature_cols = [c for c in df.columns if c not in ['date', 'target']]
data_target = df[['target']].values
features = df[feature_cols].values

print("特征和目标变量提取完成，准备时间特征编码...")
df_stamp = df[['date']].copy()
df_stamp['date'] = pd.to_datetime(df_stamp['date'])
data_stamp = time_features(df_stamp, timeenc=1, freq='h')
print("时间特征编码完成...")

# 严格切分训练集与测试集防止泄露
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


# ==========================================
# 核心修改 2：适配 Informer 的超参数配置
# ==========================================
# 配置参数
data_dim = data_inverse.shape[1]

class Config:
    def __init__(self):
        self.seq_len = window
        self.label_len = int(window / 2)
        self.pred_len = length_size
        self.freq = 'h'

        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.stop_ratio = 0.2

        self.dec_in = data_dim
        self.enc_in = data_dim
        self.c_out = 1

        # Informer 专属架构参数调整
        # self.d_model = 128  # 增加模型维度
        # self.n_heads = 8  # 增加注意力头数
        # self.dropout = 0.05
        # self.e_layers = 2  # 编码器层数
        # self.d_layers = 1  # 解码器层数
        # self.d_ff = 256  # 前馈网络维度
        # self.factor = 5  # ProbSparse 注意力因子

        #与TCN+informer参数对齐
        self.d_model = 64  # 模型维度
        self.n_heads = 4  # 多头注意力头数
        self.dropout = 0.1  # 丢弃率
        self.e_layers = 3  # 编码器块的数量
        self.d_layers = 3  # 解码器块的数量
        self.d_ff = 128  # 全连接网络维度
        self.factor = 5  # 注意力因子

        self.activation = 'gelu'
        self.distil = True  # 开启 Informer 的特征蒸馏机制

        self.embed = 'timeF'
        self.output_attention = False
        self.task_name = 'short_term_forecast'


config = Config()

# ==========================================
# 核心修改 3：使用标准 Informer 初始化
# ==========================================
model_type = 'Ablation_Informer'
net = Informer.Model(config).to(device)

# 优化器与调度器
criterion = nn.MSELoss().to(device)
optimizer = optim.Adam(net.parameters(), lr=learning_rate)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

# 模型训练 (统一使用 model_train_val)
trained_model, train_loss, val_loss, final_epoch = model_train_val(net, train_loader, val_loader, length_size, 
                                                                    optimizer, criterion, scheduler, num_epochs,
                                                                    device, print_train=True)

trained_model.eval()
preds = []
trues = []
with torch.no_grad():
    for x, y, x_mark, y_mark in test_loader:
        x, y, x_mark, y_mark = x.to(device), y.to(device), x_mark.to(device), y_mark.to(device)
        
        # --- 核心改进：解决数据泄露问题 ---
        # 构造公平的解码器输入：将 y 中的预测部分 (后 length_size 步) 的目标列 (最后一列) 清零
        # 这样模型在预测时只能看到未来的外生变量（特征），无法通过输入直接看到 NEE 答案
        y_masked = y.clone()
        y_masked[:, -length_size:, -1] = 0  
        
        outputs = trained_model(x, x_mark, y_masked, y_mark)
        preds.append(outputs.detach().cpu().numpy())
        trues.append(y[:, -length_size:, -1:].detach().cpu().numpy())

pred = np.concatenate(preds, axis=0)
true = np.concatenate(trues, axis=0)

true = true[:, :, -1]
pred = pred[:, :, -1]

# --- 改进：Scaler 应该在原始单位上 fit ---
# 重新从原始数据中定义一个针对目标列的 Scaler，确保反归一化回到原始单位
target_scaler = MinMaxScaler()
target_scaler.fit(data_target) 

pred_uninverse = target_scaler.inverse_transform(pred[:, -1:])
true_uninverse = target_scaler.inverse_transform(true[:, -1:])

true, pred = true_uninverse, pred_uninverse

df_eval = cal_eval(true, pred)
print(df_eval)

# ==========================================
# 结果保存：将记录在 Ablation_Informer 的文件夹中
# ==========================================
output_dir = 'result'
now = datetime.now().strftime("%Y%m%d_%H%M%S")
run_folder_name = f"{model_type}_{now}_{dataset_name}"
output_dir = os.path.join('result', run_folder_name)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print(f"\n==========================================")
print(f"[INFO] Informer消融实验的所有结果将保存在: {output_dir}")
print(f"==========================================")

metrics_filename = f'{run_folder_name}_metrics.csv'
metrics_path = os.path.join(output_dir, metrics_filename)
df_eval.to_csv(metrics_path, index=False, encoding='utf-8-sig')
print(f'[SUCCESS] 评估指标已保存: {metrics_filename}')

# --- 改进：严谨的时间戳对齐 ---
test_dates = df['date'].iloc[val_size + window + length_size - 1 : val_size + window + length_size - 1 + len(true_final)].reset_index(drop=True)
data_filename = f'{run_folder_name}_data.csv'
data_path = os.path.join(output_dir, data_filename)
result_df = pd.DataFrame({'时间': test_dates, '真实值': true_final, '预测值': pred_final})
result_df.to_csv(data_path, index=False, encoding='utf-8-sig')
print(f'[SUCCESS] 预测数据已保存: {data_filename}')

df_pred_true = pd.DataFrame({'Predict': pred_final, 'Real': true_final})
plt.figure(figsize=(12, 4))
plt.plot(df_pred_true['Predict'], label='Predict', color='red', alpha=0.8)
plt.plot(df_pred_true['Real'], label='Real', color='blue', alpha=0.5)

plt.title(f'{model_type} Result ({dataset_name})')
plt.legend()

img_filename = f'{run_folder_name}.png'
img_save_path = os.path.join(output_dir, img_filename)
plt.savefig(img_save_path, bbox_inches='tight')
print(f'[SUCCESS] 预测结果图已保存: {img_filename}\n')

plt.show()