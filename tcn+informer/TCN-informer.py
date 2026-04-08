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
print(torch.__version__)
sys.stdout.flush()
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

plt.rc('font', family='sans-serif')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'SimHei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use("ggplot")
# 自己写的函数文件functionfile.py
# 如果需要调整TSlib-test.ipynb文件的路径位置 注意同时调整导入的路径
from models import TCNInformer
from utils.timefeatures import time_features

# 解决画图中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def tslib_data_loader(window, length_size, batch_size, data, data_mark):
    """
    数据加载器函数，用于加载和预处理时间序列数据，以用于训练模型。

    仅仅适用于 多变量预测多变量（可以单独取单变量的输出），或者单变量预测单变量
    也就是y里也会有外生变量？？

    参数:
    - window: 窗口大小，用于截取输入序列的长度。
    - length_size: 目标序列的长度。
    - batch_size: 批量大小，决定每个训练批次包含的数据样本数量。
    - data: 输入时间序列数据。
    - data_mark: 输入时间序列的数据标记，用于辅助模型训练或增加模型的多样性。

    返回值:
    - dataloader: 数据加载器，用于批量加载处理后的训练数据。
    - x_temp: 处理后的输入数据。
    - y_temp: 处理后的目标数据。
    - x_temp_mark: 处理后的输入数据的标记。
    - y_temp_mark: 处理后的目标数据的标记。
    """

    # 构建模型的输入
    seq_len = window
    sequence_length = seq_len + length_size
    num_samples = len(data) - sequence_length + 1
    print(f"TSLIB data generator start, seq_len: {seq_len}, length_size: {length_size}, total samples: {num_samples}")
    
    # 使用预分配空间代替列表推导式，极大提升大切片场景下的性能
    result = np.empty((num_samples, sequence_length, data.shape[1]), dtype=np.float32)
    result_mark = np.empty((num_samples, sequence_length, data_mark.shape[1]), dtype=np.float32)
    
    for i in range(num_samples):
        result[i] = data[i : i + sequence_length]
        result_mark[i] = data_mark[i : i + sequence_length]
        
    print("TSLIB window splicing finished...")

    # 划分x与y
    x_temp = result[:, :-length_size]
    y_temp = result[:, -(length_size + int(window / 2)):]

    x_temp_mark = result_mark[:, :-length_size]
    y_temp_mark = result_mark[:, -(length_size + int(window / 2)):]

    # 转换为Tensor和数据类型
    x_temp = torch.tensor(x_temp).type(torch.float32)
    x_temp_mark = torch.tensor(x_temp_mark).type(torch.float32)
    y_temp = torch.tensor(y_temp).type(torch.float32)
    y_temp_mark = torch.tensor(y_temp_mark).type(torch.float32)

    ds = TensorDataset(x_temp, y_temp, x_temp_mark, y_temp_mark)
    dataloader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    return dataloader, x_temp, y_temp, x_temp_mark, y_temp_mark


def model_train(net, train_loader, length_size, optimizer, criterion, num_epochs, device, print_train=False):
    """
    训练模型并应用早停机制。

    参数:
        net (torch.nn.Module): 待训练的模型。
        train_loader (torch.utils.data.DataLoader): 训练数据加载器。
        length_size (int): 输出序列的长度。
        optimizer (torch.optim.Optimizer): 优化器。
        criterion (torch.nn.Module): 损失函数。
        num_epochs (int): 总训练轮数。
        device (torch.device): 设备（CPU或GPU）。
        print_train (bool, optional): 是否在训练中打印进度，默认为False。
    返回:
        net (torch.nn.Module): 训练好的模型。
        train_loss (list): 训练过程中每个epoch的平均训练损失列表。
        best_epoch (int): 达到最佳验证损失的epoch数。
    """

    train_loss = []  # 用于记录每个epoch的平均训练损失
    print_frequency = 1 #num_epochs / 20  # 计算打印训练状态的频率

    for epoch in range(num_epochs):
        total_train_loss = 0  # 初始化一个epoch的总损失

        net.train()  # 将模型设置为训练模式
        loop = tqdm(train_loader, total=len(train_loader), leave=True, desc=f"Epoch [{epoch+1}/{num_epochs}]")
        for i, (datapoints, labels, datapoints_mark, labels_mark) in enumerate(loop):
            datapoints, labels, datapoints_mark, labels_mark = datapoints.to(device), labels.to(
                device), datapoints_mark.to(device), labels_mark.to(device)
            optimizer.zero_grad()
            
            # --- 核心改进：同步训练掩码 ---
            # 训练阶段也必须遮蔽未来的目标变量，防止模型学会“复制”逻辑
            labels_masked = labels.clone()
            labels_masked[:, -length_size:, -1] = 0
            
            preds = net(datapoints, datapoints_mark, labels_masked, labels_mark, None)
            preds = preds[:, -length_size:, -1:]
            labels = labels[:, -length_size:, -1:] # 仅取目标变量通道
            loss = criterion(preds, labels)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新模型参数
            total_train_loss += loss.item()  # 累加损失值
            loop.set_postfix(loss=loss.item())

        avg_train_loss = total_train_loss / len(train_loader)  # 计算该epoch的平均损失
        train_loss.append(avg_train_loss)  # 将平均损失添加到列表中

        # 如果设置为打印训练状态，则按频率打印
        if print_train:
            if (epoch + 1) % print_frequency == 0:
                print(f"Epoch: {epoch + 1}, Train Loss: {avg_train_loss:.4f}")

    return net, train_loss, epoch + 1


def model_train_val(net, train_loader, val_loader, length_size, optimizer, criterion, scheduler, num_epochs, device,
                    early_patience=0.15, print_train=False):
    train_loss = []
    val_loss = []
    print_frequency = 1

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

        # 验证环节
        net.eval()
        with torch.no_grad():
            total_val_loss = 0
            for val_x, val_y, val_x_mark, val_y_mark in val_loader:
                val_x, val_y, val_x_mark, val_y_mark = val_x.to(device), val_y.to(device), val_x_mark.to(
                    device), val_y_mark.to(device)
                
                # --- 核心改进：验证集也要掩码 ---
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
            torch.save(net.state_dict(), 'checkpoint.pth') # 保存最佳模型
        else:
            early_stop_counter += 1
            if early_stop_counter >= early_patience_epochs:
                loop.write(f'Early stopping triggered at epoch {epoch + 1}.')
                break

    net.load_state_dict(torch.load('checkpoint.pth')) # 加载最佳模型
    return net, train_loss, val_loss, epoch + 1


# 计算点预测的评估指标
def cal_eval(y_real, y_pred):
    """
    输入参数:
    y_real - numpy数组，表示测试集的真实目标值。
    y_pred - numpy数组，表示预测的结果。

    输出:
    df_eval - pandas DataFrame对象
    """

    y_real, y_pred = np.array(y_real).ravel(), np.array(y_pred).ravel()

    #mse和rmse的代码更改过，注意分辨
    r2 = r2_score(y_real, y_pred)
    mse = mean_squared_error(y_real, y_pred) #, squared=True)
    #rmse = mean_squared_error(y_real, y_pred) #, squared=False)  # RMSE and MAE are various on different scales
    #在scikit-learn的1.6.1版本中，计算均方根误差要用root_mean_squared_error函数
    #即rmse = root_mean_squared_error(y_real, y_pred)
    rmse = np.sqrt(mean_squared_error(y_real, y_pred))
    mae = mean_absolute_error(y_real, y_pred)
    mape = mean_absolute_percentage_error(y_real, y_pred) * 100  # Note that dataset cannot have any 0 value.

    df_eval = pd.DataFrame({'R2': r2,
                            'MSE': mse, 'RMSE': rmse,
                            'MAE': mae, 'MAPE': mape},
                           index=['Eval'])

    return df_eval


def data_cleansing(df):
    """集成自 data_preprocessing.py 的清洗逻辑，兼容多个数据集"""
    # 确保日期格式正确并排序
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(by='date').reset_index(drop=True)
    
    # 记录原始列，排除日期列进行插值
    cols = [c for c in df.columns if c != 'date']
    
    # 1. 线性插值处理短时间缺失 (Limit = 6, 即 3 小时)
    df[cols] = df[cols].interpolate(method='linear', limit=6)
    
    # 2. 基于时间的插值处理中等时间缺失 (Limit = 48, 即 24 小时)
    if 'date' in df.columns:
        df.set_index('date', inplace=True)
        df[cols] = df[cols].interpolate(method='time', limit=48)
        df.reset_index(inplace=True)
    
    # 3. 前后填充处理剩余的长期缺失
    df[cols] = df[cols].ffill().bfill()
    
    return df


# 读取数据 - 切换到长江三角洲 DT 农田数据集（30分钟级）
#data_path = 'data/Yangtze River Delta of China/DT_NEE(20141201-20171130).csv'
data_path = 'data/Yangtze River Delta of China/SX_NEE(20150715-20190424).csv'
dataset_name = os.path.splitext(os.path.basename(data_path))[0]

print(f"开始读取数据集: {data_path} ...")
df_raw = pd.read_csv(data_path)
print(f"数据读取完成, 原始形状: {df_raw.shape}")

# 执行统一清洗
df = data_cleansing(df_raw)
print(f"数据清洗完成, 清洗后形状: {df.shape}")

# 统一目标列名，适应原代码逻辑
if 'Target' in df.columns:
    df.rename(columns={'Target': 'target'}, inplace=True)

# --- 方向 A 优化：特征工程 (Feature Engineering) ---
# 1. 滞后项 (Lagged Features): 捕捉生态滞后响应 (根据新数据集字段 K↓(辐射), Tair(温度), VPD(湿度差))
for col in ['K↓', 'Tair', 'VPD']:
    for lag in range(1, 4):
        df[f'{col}_lag{lag}'] = df[col].shift(lag)

# 2. 差分项 (Differential Features): 捕捉环境突变速度
for col in ['K↓', 'Tair']:
    df[f'{col}_diff'] = df[col].diff()
print("特征工程完成(滞后+差分)...")

# 3. 清理 NaN (由于 shift 和 diff 产生)
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True) # 必须重置索引以保证后续矩阵拼接的对齐

# 提取特征和目标变量
feature_cols = [c for c in df.columns if c not in ['date', 'target']]
data_target = df[['target']].values  # shape: (N, 1)
features = df[feature_cols].values   # shape: (N, num_features)

print("特征和目标变量提取完成，准备时间特征编码...")
# 时间戳特征提取 (30分钟数据，频率必须是 'h' 或 't' 以捕捉昼夜节律)
df_stamp = df[['date']].copy()
df_stamp['date'] = pd.to_datetime(df_stamp['date'])
data_stamp = time_features(df_stamp, timeenc=1, freq='h')

# --- V3: 增加正余弦周期编码 (Sin/Cos Encoding) ---
# data_stamp 原有: [HourOfDay, DayOfWeek, DayOfMonth, DayOfYear]
# 我们为核心周期 (Hour, Day) 增加周期性表达
hour_rad = (df_stamp['date'].dt.hour / 23.0) * 2 * np.pi
day_rad = (df_stamp['date'].dt.dayofyear / 365.0) * 2 * np.pi

sin_cos_features = np.stack([
    np.sin(hour_rad), np.cos(hour_rad),
    np.sin(day_rad), np.cos(day_rad)
], axis=1)

data_stamp = np.concatenate([data_stamp, sin_cos_features], axis=1)
print(f"时间特征扩充完成 (Sin/Cos): {data_stamp.shape}")
print("时间特征编码完成...")

# ==========================================
# --- 核心修复：防止数据泄露 (Data Leakage) ---
# **绝对要求：必须在调用任何 .fit() 之前切分训练集与测试集**
# ==========================================
data_length = len(df)
train_set = 0.8  # 保留 80% 训练集, 20% 测试集
train_size = int(train_set * data_length)

features_train = features[:train_size, :]
features_test = features[train_size:, :]
target_train = data_target[:train_size, :]
target_test = data_target[train_size:, :]
data_stamp_train = data_stamp[:train_size, :]
data_stamp_test = data_stamp[train_size:, :]
print("训练/测试集切分完成...")

# --- 方向 B 优化 (隔离模式)：PCA 降维去噪 ---
# 1. StandardScaler：【只对训练集 fitting，不对测试集学习参数】
print("开始进行数据标准化与 PCA 降维...")
scaler_pca = StandardScaler()
features_train_scaled = scaler_pca.fit_transform(features_train)
features_test_scaled = scaler_pca.transform(features_test) 

# 数据标准化完成
print(f"数据标准化完成：使用全部 {features_train_scaled.shape[1]} 维特征。")

# 3. 将 PCA 过滤后的特征和原本的目标变量重新拼接成网络期待的输入矩阵
print("进行矩阵拼接与 MinMaxScaler 归一化...")
data_train_raw = np.concatenate((features_train_scaled, target_train), axis=1)
data_test_raw = np.concatenate((features_test_scaled, target_test), axis=1)

# --- 最终层隔离优化 ---
# 4. MinMaxScaler 归一化网络整体输入：【全局绝对极值必须局限于训练集】
scaler = MinMaxScaler()
data_train = scaler.fit_transform(data_train_raw)
data_test = scaler.transform(data_test_raw)           

data_train_mark = data_stamp_train
data_test_mark = data_stamp_test
data_dim = data_train.shape[1]  # 特征维度 (PCA维数) + 1 个目标维度

n_feature = data_dim
window = 96  # 输入序列: 过去 2 天的半小时数据 (48 * 2)
length_size = 48  # 预测未来的序列长度: 未来 1 天 (48 * 1)
batch_size = 128

# --- V3: 公平归一化 (StandardScaler, Fit 仅在训练集) ---
data_full = np.concatenate((features, data_target), axis=1) # 18特征+1目标
data_length = len(data_full)
train_ratio, val_ratio = 0.6, 0.8
train_size = int(data_length * train_ratio)
val_size = int(data_length * val_ratio)

scaler = StandardScaler()
data_train_raw = data_full[:train_size, :]
scaler.fit(data_train_raw)
data_scaled = scaler.transform(data_full)
data_dim = data_scaled.shape[1]

# 切分数据集
data_train = data_scaled[:train_size, :]
data_train_mark = data_stamp[:train_size, :]
data_val = data_scaled[train_size: val_size, :]
data_val_mark = data_stamp[train_size: val_size, :]
data_test = data_scaled[val_size:, :]
data_test_mark = data_stamp[val_size:, :]

# 封装 DataLoader
train_loader, x_train, y_train, x_train_mark, y_train_mark = tslib_data_loader(window, length_size, batch_size,
                                                                               data_train, data_train_mark)
val_loader, x_val, y_val, x_val_mark, y_val_mark = tslib_data_loader(window, length_size, batch_size, data_val,
                                                                     data_val_mark)
test_loader, x_test, y_test, x_test_mark, y_test_mark = tslib_data_loader(window, length_size, batch_size, data_test,
                                                                          data_test_mark)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_epochs = 80  # 训练迭代次数
learning_rate = 0.0001  # V6: 稳定学习率
scheduler_patience = int(0.25 * num_epochs)  # 转换为整数  学习率调整的patience
early_patience = 0.1  # 训练迭代的早停比例 即patience=0.1*num_epochs


class Config:
    def __init__(self):
        # basic
        self.seq_len = window  # 输入序列长度
        self.label_len = int(window / 2)  # 标签序列长度
        self.pred_len = length_size  # 预测序列长度
        #self.freq = 'b'  # 时间的频率，
        self.freq = 'h'  # 切换为小时级别频率，适配半小时高频数据
        # 模型训练
        self.batch_size = batch_size  # 批次大小
        self.num_epochs = num_epochs  # 训练的轮数
        self.learning_rate = learning_rate  # 学习率
        self.stop_ratio = early_patience  # 早停的比例
        # 模型 define
        self.dec_in = data_dim  # 解码器输入特征数量, 输入几个变量就是几
        self.enc_in = data_dim  # 编码器输入特征数量
        self.c_out = 1  # 输出维度##########这个很重要
        # 模型超参数 (V6: 开启 FullAttention)
        self.d_model = 64  # 稳定维度
        self.n_heads = 4   # 稳定头数
        self.dropout = 0.1  # 丢弃率
        self.e_layers = 3   # 稳定层数
        self.d_layers = 3   # 稳定层数
        self.d_ff = 128     # 稳定维度
        self.factor = 5     # 注意力因子
        self.activation = 'gelu'  # 激活函数
        self.channel_independence = 0  # 频道独立性，0:频道依赖，1:频道独立
        self.time_dims = 8  # V3: 8 维时间特征 (4 线性 + 4 Sin/Cos)

        self.top_k = 6  # TimesBlock中的参数
        self.num_kernels = 6  # Inception中的参数
        self.distil = 1  # 是否使用蒸馏，1为True
        # 一般不需要动的参数
        self.embed = 'timeF'  # 时间特征编码方式
        self.output_attention = 0  # 是否输出注意力
        self.task_name = 'short_term_forecast'  # 模型的任务，一般不动但是必须这个参数


config = Config()

model_type = 'DCATCN-TCNInformer'
net = TCNInformer.Model(config).to(device)

criterion = nn.MSELoss().to(device)  # 损失函数 (回归 MSE 以保证波峰捕捉力度)
optimizer = optim.Adam(net.parameters(), lr=learning_rate)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

# 模型训练 (采用统一的 model_train_val)
trained_model, train_loss, val_loss, final_epoch = model_train_val(net, train_loader, val_loader, length_size, 
                                                                    optimizer, criterion, scheduler, num_epochs,
                                                                    device, print_train=True)

"""
trained_model, train_loss, val_loss, final_epoch = model_train_val(
    net=net,
    train_loader=train_loader,
    val_loader=val_loader,
    length_size=length_size,
    optimizer=optimizer,
    criterion=criterion,
    scheduler=scheduler,
    num_epochs=num_epochs,
    device=device,
    early_patience=early_patience,
    print_train=False       
)
"""
    #print_train改为True就会显示轮次，而不是早停法，用False就是早停法

trained_model.eval()  # 模型转换为验证模式
# 预测并调整维度
preds = []
trues = []
with torch.no_grad():
    for x, y, x_mark, y_mark in test_loader:
        x, y, x_mark, y_mark = x.to(device), y.to(device), x_mark.to(device), y_mark.to(device)
        
        # --- 核心改进：解决数据泄露问题 ---
        # 构造公平的解码器输入：将 y 中的预测部分 (后 length_size 步) 的目标列 (最后一列) 置 0
        # 这样模型在预测时只能看到未来的外生变量（特征），无法通过输入直接看到 NEE 答案
        y_masked = y.clone()
        y_masked[:, -length_size:, -1] = 0  
        
        outputs = trained_model(x, x_mark, y_masked, y_mark)
        preds.append(outputs.detach().cpu().numpy())
        trues.append(y[:, -length_size:, -1:].detach().cpu().numpy())

pred = np.concatenate(preds, axis=0)
true = np.concatenate(trues, axis=0)

# --- V6: 全时段评估 (Full Horizon Evaluation) ---
# outputs shape [N, 144, 19], trues shape [N, 48, 1]
# 我们只关注最后 length_size (48步) 的预测质量
true = true[:, :, -1] # [N, 48, 1] -> [N, 48] (已经是 NEE 这一列)
pred = pred[:, -length_size:, -1] # [N, 144, 19] -> [N, 48] (NEE 在最后一列)

print("Shape of pred after horizon adjustment:", pred.shape)
print("Shape of true after horizon adjustment:", true.shape)

# --- Scaler 只在训练集上 fit (V6 修复 NameError) ---
target_scaler = StandardScaler()
target_train = data_train_raw[:, -1:] # 采样原始数据最后一列 NEE
target_scaler.fit(target_train) 

# 反归一化：将整体 [N, 48] 拉平为 [N*48, 1] 进行变换，再恢复形状
pred_uninverse = target_scaler.inverse_transform(pred.reshape(-1, 1)).reshape(pred.shape)
true_uninverse = target_scaler.inverse_transform(true.reshape(-1, 1)).reshape(true.shape)

true_all, pred_all = true_uninverse, pred_uninverse

# 计算评价指标 (使用全时段数据，更严谨)
df_eval = cal_eval(true_all, pred_all)  
print(df_eval)

# --- 为保存和绘图准备数据 (仅取预测窗的最后一步，以保持时间轴连续) ---
true_plot = true_all[:, -1]
pred_plot = pred_all[:, -1]

# --- 保存结果 ---
# 创建 result 和 img 文件夹
output_dir = 'result'
img_dir = os.path.join(output_dir, 'img')
if not os.path.exists(img_dir):
    os.makedirs(img_dir)

# 1. 提取数据集名称和当前时间戳
dataset_name = os.path.splitext(os.path.basename(data_path))[0]
now = datetime.now().strftime("%Y%m%d_%H%M%S")

# 2. 定义本次运行的专属文件夹名称 (例如: TCNInformer_20231024_153000_DT_NEE)
run_folder_name = f"TCNInformer_{now}_{dataset_name}"

# 3. 创建专属文件夹路径 (result/TCNInformer_20231024_153000_DT_NEE)
output_dir = os.path.join('result', run_folder_name)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print(f"\n==========================================")
print(f"[INFO] 本次运行的所有结果将保存在: {output_dir}")
print(f"==========================================")

# 4. 保存评估指标 (Metrics) 到专属文件夹
metrics_filename = f'{run_folder_name}_metrics.csv'
metrics_path = os.path.join(output_dir, metrics_filename)
df_eval.to_csv(metrics_path, index=False, encoding='utf-8-sig')
print(f'[SUCCESS] 评估指标已保存: {metrics_filename}')

# 5. 保存真实值和预测值 (Data) 到专属文件夹
# 确保时间轴对齐：取测试集中每一个 window 对应的最后一个预测点的时间
test_dates = df['date'].iloc[-len(true_plot):].reset_index(drop=True)
data_filename = f'{run_folder_name}_data.csv'
data_path = os.path.join(output_dir, data_filename)
result_df = pd.DataFrame({'时间': test_dates,'真实值': true_plot.flatten(), '预测值': pred_plot.flatten()})
result_df.to_csv(data_path, index=False, encoding='utf-8-sig')
print(f'[SUCCESS] 预测数据已保存: {data_filename}')

# 6. 绘制并保存结果图 (Image) 到专属文件夹
df_pred_true = pd.DataFrame({'Predict': pred_plot.flatten(), 'Real': true_plot.flatten()})
plt.figure(figsize=(12, 4))
plt.plot(df_pred_true['Predict'], label='Predict', color='red', alpha=0.8)
plt.plot(df_pred_true['Real'], label='Real', color='blue', alpha=0.5)

# 标题带上数据集名称
plt.title(f'{model_type} Result ({dataset_name})')
plt.legend()

# 保存图片
img_filename = f'{run_folder_name}.png'
img_save_path = os.path.join(output_dir, img_filename)
plt.savefig(img_save_path, bbox_inches='tight')
print(f'[SUCCESS] 预测结果图已保存: {img_filename}\n')

# 最后展示图片
plt.show()
