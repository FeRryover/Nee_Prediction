import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
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
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

plt.rc('font', family='Arial')
plt.style.use("ggplot")
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


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


def create_gru_dataset(data, window, length_size):
    sequence_length = window + length_size
    num_samples = len(data) - sequence_length + 1

    result = np.empty((num_samples, sequence_length, data.shape[1]), dtype=np.float32)
    for i in range(num_samples):
        result[i] = data[i: i + sequence_length]

    X = result[:, :window, :]
    Y = result[:, -length_size:, -1:]

    return torch.tensor(X), torch.tensor(Y)


#data_path = 'data/Yangtze River Delta of China/DT_NEE(20141201-20171130).csv'
data_path = 'data/Yangtze River Delta of China/SX_NEE(20150715-20190424).csv'

dataset_name = os.path.splitext(os.path.basename(data_path))[0]

print(f"开始读取数据集: {data_path} ...")
df_raw = pd.read_csv(data_path)
df = data_cleansing(df_raw)

if 'Target' in df.columns:
    df.rename(columns={'Target': 'target'}, inplace=True)

df['date'] = pd.to_datetime(df['date'])
df['hour'] = df['date'].dt.hour
df['dayofweek'] = df['date'].dt.dayofweek
df['month'] = df['date'].dt.month

for col in ['K↓', 'Tair', 'VPD']:
    for lag in range(1, 5):
        df[f'{col}_lag{lag}'] = df[col].shift(lag)

for col in ['K↓', 'Tair']:
    df[f'{col}_diff'] = df[col].diff()

df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

feature_cols = [c for c in df.columns if c not in ['date', 'target']]
data_target = df[['target']].values
features = df[feature_cols].values

data_length = len(df)
train_size = int(0.7 * data_length)
val_size = int(0.1 * data_length)

features_train = features[:train_size, :]
features_val = features[train_size:train_size+val_size, :]
features_test = features[train_size+val_size:, :]
target_train = data_target[:train_size, :]
target_val = data_target[train_size:train_size+val_size, :]
target_test = data_target[train_size+val_size:, :]

use_pca = True

if use_pca:
    scaler_pca = StandardScaler()
    features_train_scaled = scaler_pca.fit_transform(features_train)
    features_val_scaled = scaler_pca.transform(features_val)
    features_test_scaled = scaler_pca.transform(features_test)

    pca = PCA(n_components=0.95)
    features_train_pca = pca.fit_transform(features_train_scaled)
    features_val_pca = pca.transform(features_val_scaled)
    features_test_pca = pca.transform(features_test_scaled)

    data_train_raw = np.concatenate((features_train_pca, target_train), axis=1)
    data_val_raw = np.concatenate((features_val_pca, target_val), axis=1)
    data_test_raw = np.concatenate((features_test_pca, target_test), axis=1)
else:
    data_train_raw = np.concatenate((features_train, target_train), axis=1)
    data_val_raw = np.concatenate((features_val, target_val), axis=1)
    data_test_raw = np.concatenate((features_test, target_test), axis=1)

scaler = MinMaxScaler()
data_train = scaler.fit_transform(data_train_raw)
data_val = scaler.transform(data_val_raw)
data_test = scaler.transform(data_test_raw)

window = 96
length_size = 48
batch_size = 96
input_size = data_train.shape[1]

print("正在构建 GRU 数据集张量...")
x_train, y_train = create_gru_dataset(data_train, window, length_size)
x_val, y_val = create_gru_dataset(data_val, window, length_size)
x_test, y_test = create_gru_dataset(data_test, window, length_size)

train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=batch_size, shuffle=False)
test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)


class GRUForecastModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(GRUForecastModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out.unsqueeze(-1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
hidden_size = 64
num_layers = 2
num_epochs = 120
learning_rate = 0.0008

print(f"\n模型配置（保持原版架构）: hidden_size={hidden_size}, num_layers={num_layers}, window={window}")
print(f"开始在 {device} 上训练优化版 GRU 模型...")

model = GRUForecastModel(input_size, hidden_size, num_layers, length_size).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.6, patience=10)

best_val_loss = float('inf')
patience = 18
patience_counter = 0
train_losses = []
val_losses = []

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

    avg_train_loss = total_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            val_loss += loss.item()
    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    print(f"Epoch: {epoch + 1:03d} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

    scheduler.step(avg_val_loss)

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"连续 {patience} 轮 Val Loss 未下降，触发 Early Stopping!")
            break

model.eval()
preds, trues = [], []

with torch.no_grad():
    for batch_x, batch_y in test_loader:
        batch_x = batch_x.to(device)
        outputs = model(batch_x)
        preds.append(outputs.cpu().numpy())
        trues.append(batch_y.numpy())

pred_array = np.concatenate(preds, axis=0)
true_array = np.concatenate(trues, axis=0)

pred_2d = pred_array.squeeze(-1)
true_2d = true_array.squeeze(-1)

print("\n训练完成，正在反归一化...")
scaler_target = MinMaxScaler()
scaler_target.fit(np.array(data_target).reshape(-1, 1))

pred_final = scaler_target.inverse_transform(pred_2d[:, -1:])
true_final = scaler_target.inverse_transform(true_2d[:, -1:])

df_eval = cal_eval(true_final, pred_final)
print("\n====== 优化版 GRU 模型评估结果（保持原版架构）======")
print(df_eval)

now = datetime.now().strftime("%Y%m%d_%H%M%S")
run_folder_name = f"GRU_v2_{now}_{dataset_name}"
output_dir = os.path.join('result', run_folder_name)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print(f"\n==========================================")
print(f"[INFO] 本次运行的所有结果将保存在: {output_dir}")
print(f"==========================================")

metrics_filename = f'{run_folder_name}_metrics.csv'
metrics_path = os.path.join(output_dir, metrics_filename)
df_eval.to_csv(metrics_path, index=False, encoding='utf-8-sig')

test_dates = df['date'].iloc[-len(true_final.flatten()):].reset_index(drop=True)
data_filename = f'{run_folder_name}_data.csv'
data_path = os.path.join(output_dir, data_filename)
result_df = pd.DataFrame({
    '时间': test_dates,
    '真实值': true_final.flatten(),
    '预测值': pred_final.flatten()
})
result_df.to_csv(data_path, index=False, encoding='utf-8-sig')

plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss', alpha=0.8)
plt.plot(val_losses, label='Val Loss', alpha=0.8)
plt.title('Training History')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(result_df['预测值'][:200], label='Predict', color='red', alpha=0.8)
plt.plot(result_df['真实值'][:200], label='Real', color='blue', alpha=0.5)
plt.title(f'GRU v2 Result ({dataset_name})')
plt.legend()
plt.tight_layout()

img_filename = f'{run_folder_name}.png'
img_save_path = os.path.join(output_dir, img_filename)
plt.savefig(img_save_path, bbox_inches='tight', dpi=150)
print(f'[SUCCESS] 实验文件已全部保存完毕！\n')

plt.close()
