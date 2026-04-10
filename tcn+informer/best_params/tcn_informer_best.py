import copy
import os
import sys
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(CURRENT_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from models import TCNInformer
from utils.timefeatures import time_features


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
    return dataloader


def model_train_val(net, train_loader, val_loader, length_size, optimizer, criterion, scheduler, num_epochs, device,
                    early_patience=0.15):
    train_loss = []
    val_loss = []

    early_patience_epochs = max(1, int(early_patience * num_epochs))
    best_val_loss = float('inf')
    best_state_dict = None
    early_stop_counter = 0

    for epoch in range(num_epochs):
        total_train_loss = 0
        net.train()
        loop = tqdm(train_loader, total=len(train_loader), leave=True, desc=f"Epoch [{epoch + 1}/{num_epochs}]")
        for datapoints, labels, datapoints_mark, labels_mark in loop:
            datapoints, labels = datapoints.to(device), labels.to(device)
            datapoints_mark, labels_mark = datapoints_mark.to(device), labels_mark.to(device)
            optimizer.zero_grad()

            labels_masked = labels.clone()
            labels_masked[:, -length_size:, -1] = 0

            preds = net(datapoints, datapoints_mark, labels_masked, labels_mark, None)
            preds = preds[:, -length_size:, -1:]
            labels_y = labels[:, -length_size:, -1:]
            loss = criterion(preds, labels_y)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        train_loss.append(total_train_loss / len(train_loader))

        net.eval()
        with torch.no_grad():
            total_val_loss = 0
            for val_x, val_y, val_x_mark, val_y_mark in val_loader:
                val_x, val_y = val_x.to(device), val_y.to(device)
                val_x_mark, val_y_mark = val_x_mark.to(device), val_y_mark.to(device)

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

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stop_counter = 0
            best_state_dict = copy.deepcopy(net.state_dict())
        else:
            early_stop_counter += 1
            if early_stop_counter >= early_patience_epochs:
                break

    if best_state_dict is not None:
        net.load_state_dict(best_state_dict)
    return net, train_loss, val_loss


def cal_eval(y_real, y_pred):
    y_real, y_pred = np.array(y_real).ravel(), np.array(y_pred).ravel()
    r2 = r2_score(y_real, y_pred)
    mse = mean_squared_error(y_real, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_real, y_pred)
    mape = mean_absolute_percentage_error(y_real, y_pred) * 100
    return pd.DataFrame({'R2': r2, 'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'MAPE': mape}, index=['Eval'])


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


def main():
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
    train_size = int(data_length * 0.6)
    val_size = int(data_length * 0.8)

    use_pca = bool(env_int('HP_USE_PCA', 0))
    if use_pca:
        scaler_features = StandardScaler()
        scaler_features.fit(features[:train_size, :])
        features_scaled = scaler_features.transform(features)

        pca = PCA(n_components=0.95)
        pca.fit(features_scaled[:train_size, :])
        features_used = pca.transform(features_scaled)
    else:
        features_used = features

    df_stamp = df[['date']].copy()
    df_stamp['date'] = pd.to_datetime(df_stamp['date'])
    data_stamp = time_features(df_stamp, timeenc=1, freq='h')

    data_full = np.concatenate((features_used, data_target), axis=1)
    scaler = MinMaxScaler()
    scaler.fit(data_full[:train_size, :])
    data_scaled = scaler.transform(data_full)

    data_train = data_scaled[:train_size, :]
    data_train_mark = data_stamp[:train_size, :]
    data_val = data_scaled[train_size:val_size, :]
    data_val_mark = data_stamp[train_size:val_size, :]
    data_test = data_scaled[val_size:, :]
    data_test_mark = data_stamp[val_size:, :]

    window = env_int('HP_WINDOW', 96)
    length_size = env_int('HP_LENGTH', 48)
    batch_size = env_int('HP_BATCH_SIZE', 64)
    num_epochs = env_int('HP_EPOCHS', 120)
    learning_rate = env_float('HP_LR', 0.0002)
    scheduler_patience = env_int('HP_SCHED_PATIENCE', 8)
    early_patience = env_float('HP_EARLY_PATIENCE', 0.15)

    train_loader = tslib_data_loader(window, length_size, batch_size, data_train, data_train_mark, shuffle=True)
    val_loader = tslib_data_loader(window, length_size, batch_size, data_val, data_val_mark, shuffle=False)
    test_loader = tslib_data_loader(window, length_size, batch_size, data_test, data_test_mark, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    class Config:
        def __init__(self):
            self.seq_len = window
            self.label_len = int(window / 2)
            self.pred_len = length_size
            self.freq = 'h'

            # 与原TCNInformer接口对齐，避免缺失配置属性
            self.batch_size = batch_size
            self.num_epochs = num_epochs
            self.learning_rate = learning_rate
            self.stop_ratio = early_patience

            self.dec_in = data_scaled.shape[1]
            self.enc_in = data_scaled.shape[1]
            self.c_out = 1

            self.d_model = env_int('HP_D_MODEL', 64)
            self.n_heads = env_int('HP_N_HEADS', 4)
            self.dropout = env_float('HP_DROPOUT', 0.1)
            self.e_layers = env_int('HP_E_LAYERS', 3)
            self.d_layers = env_int('HP_D_LAYERS', 3)
            self.d_ff = env_int('HP_D_FF', 128)
            self.factor = env_int('HP_FACTOR', 5)
            self.activation = 'gelu'
            self.channel_independence = 0
            self.time_dims = 4

            self.top_k = 6
            self.num_kernels = 6
            self.distil = 1
            self.embed = 'timeF'
            self.output_attention = 0
            self.task_name = 'short_term_forecast'

    config = Config()
    model_type = 'TCNInformer_Best'
    net = TCNInformer.Model(config).to(device)

    criterion = nn.MSELoss().to(device)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=scheduler_patience)

    trained_model, train_loss, val_loss = model_train_val(
        net, train_loader, val_loader, length_size, optimizer, criterion, scheduler, num_epochs, device,
        early_patience=early_patience
    )

    trained_model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for x, y, x_mark, y_mark in test_loader:
            x, y = x.to(device), y.to(device)
            x_mark, y_mark = x_mark.to(device), y_mark.to(device)
            y_masked = y.clone()
            y_masked[:, -length_size:, -1] = 0

            outputs = trained_model(x, x_mark, y_masked, y_mark)
            preds.append(outputs.detach().cpu().numpy())
            trues.append(y[:, -length_size:, -1:].detach().cpu().numpy())

    pred = np.concatenate(preds, axis=0)
    true = np.concatenate(trues, axis=0)

    pred = pred[:, -length_size:, -1]
    true = true[:, :, -1]

    target_scaler = MinMaxScaler()
    target_scaler.fit(data_target[:train_size])
    pred_uninverse = target_scaler.inverse_transform(pred.reshape(-1, 1)).reshape(pred.shape)
    true_uninverse = target_scaler.inverse_transform(true.reshape(-1, 1)).reshape(true.shape)

    true_all, pred_all = true_uninverse, pred_uninverse
    df_eval = cal_eval(true_all, pred_all)

    true_plot = true_all[:, -1]
    pred_plot = pred_all[:, -1]

    base_output_dir = os.getenv('MODEL_OUTPUT_DIR', 'result_best')
    os.makedirs(base_output_dir, exist_ok=True)

    now = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_folder_name = f'{model_type}_{now}_{dataset_name}'
    output_dir = os.path.join(base_output_dir, run_folder_name)
    os.makedirs(output_dir, exist_ok=True)

    metrics_filename = f'{run_folder_name}_metrics.csv'
    df_eval.to_csv(os.path.join(output_dir, metrics_filename), index=False, encoding='utf-8-sig')

    test_dates = df['date'].iloc[
        val_size + window + length_size - 1: val_size + window + length_size - 1 + len(true_plot)
    ].reset_index(drop=True)
    data_filename = f'{run_folder_name}_data.csv'
    result_df = pd.DataFrame({'时间': test_dates, '真实值': true_plot.flatten(), '预测值': pred_plot.flatten()})
    result_df.to_csv(os.path.join(output_dir, data_filename), index=False, encoding='utf-8-sig')

    plt.figure(figsize=(12, 4))
    plt.plot(pred_plot.flatten(), label='Predict', color='red', alpha=0.8)
    plt.plot(true_plot.flatten(), label='Real', color='blue', alpha=0.5)
    plt.title(f'{model_type} Result ({dataset_name})')
    plt.legend()

    img_filename = f'{run_folder_name}.png'
    plt.savefig(os.path.join(output_dir, img_filename), bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    main()
