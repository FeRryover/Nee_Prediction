import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

import torch

print(torch.__version__)
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

plt.rc('font', family='Arial')
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
    result = np.array([data[i: i + sequence_length] for i in range(len(data) - sequence_length + 1)])
    result_mark = np.array([data_mark[i: i + sequence_length] for i in range(len(data) - sequence_length + 1)])

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
    print_frequency = num_epochs / 20  # 计算打印训练状态的频率

    for epoch in range(num_epochs):
        total_train_loss = 0  # 初始化一个epoch的总损失

        net.train()  # 将模型设置为训练模式
        for i, (datapoints, labels, datapoints_mark, labels_mark) in enumerate(train_loader):
            datapoints, labels, datapoints_mark, labels_mark = datapoints.to(device), labels.to(
                device), datapoints_mark.to(device), labels_mark.to(device)
            optimizer.zero_grad()  # 清空梯度
            preds = net(datapoints, datapoints_mark, labels, labels_mark, None)  # 前向传播
            preds = preds[:, -length_size:, -1:] # 仅取目标变量通道
            labels = labels[:, -length_size:, -1:] # 仅取目标变量通道
            loss = criterion(preds, labels)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新模型参数
            total_train_loss += loss.item()  # 累加损失值

        avg_train_loss = total_train_loss / len(train_loader)  # 计算该epoch的平均损失
        train_loss.append(avg_train_loss)  # 将平均损失添加到列表中

        # 如果设置为打印训练状态，则按频率打印
        if print_train:
            if (epoch + 1) % print_frequency == 0:
                print(f"Epoch: {epoch + 1}, Train Loss: {avg_train_loss:.4f}")

    return net, train_loss, epoch + 1


def model_train_val(net, train_loader, val_loader, length_size, optimizer, criterion, scheduler, num_epochs, device,
                    early_patience=0.15, print_train=False):
    """
    训练模型并应用早停机制。

    参数:
        model (torch.nn.Module): 待训练的模型。
        train_loader (torch.utils.data.DataLoader): 训练数据加载器。
        val_loader (torch.utils.data.DataLoader): 验证数据加载器。
        optimizer (torch.optim.Optimizer): 优化器。
        criterion (torch.nn.Module): 损失函数。
        scheduler (torch.optim.lr_scheduler._LRScheduler): 学习率调度器。
        num_epochs (int): 总训练轮数。
        device (torch.device): 设备（CPU或GPU）。
        early_patience (float, optional): 早停耐心值，默认为0.15 * num_epochs。
        print_train: 是否打印训练信息。
    返回:
        torch.nn.Module: 训练好的模型。
        list: 训练过程中每个epoch的平均训练损失列表。
        list: 训练过程中每个epoch的平均验证损失列表。
        int: 早停触发时的epoch数。
    """

    train_loss = []  # 用于记录每个epoch的平均损失
    val_loss = []  # 用于记录验证集上的损失，用于早停判断
    print_frequency = num_epochs / 20  # 计算打印频率

    early_patience_epochs = int(early_patience * num_epochs)  # 早停耐心值（转换为epoch数）
    best_val_loss = float('inf')  # 初始化最佳验证集损失
    early_stop_counter = 0  # 早停计数器

    for epoch in range(num_epochs):
        total_train_loss = 0  # 初始化一个epoch的总损失

        net.train()  # 将模型设置为训练模式
        for i, (datapoints, labels, datapoints_mark, labels_mark) in enumerate(train_loader):
            datapoints, labels, datapoints_mark, labels_mark = datapoints.to(
                device), labels.to(device), datapoints_mark.to(device), labels_mark.to(device)
            optimizer.zero_grad()  # 清空梯度
            preds = net(datapoints, datapoints_mark, labels, labels_mark, None)
            preds = preds[:, -length_size:, -1:] # 仅取目标变量通道
            labels = labels[:, -length_size:, -1:] # 仅取目标变量通道
            loss = criterion(preds, labels)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新模型参数
            total_train_loss += loss.item()  # 累加损失值

        avg_train_loss = total_train_loss / len(train_loader)  # 计算本epoch的平均损失
        train_loss.append(avg_train_loss)  # 记录平均损失

        with torch.no_grad():  # 关闭自动求导以节省内存和提高效率
            total_val_loss = 0
            for val_x, val_y, val_x_mark, val_y_mark in val_loader:
                val_x, val_y, val_x_mark, val_y_mark = val_x.to(device), val_y.to(device), val_x_mark.to(
                    device), val_y_mark.to(device)  # 将数据移到GPU
                pred_val_y = net(val_x, val_x_mark, val_y, val_y_mark, None).squeeze()  # 前向传播
                val_y = val_y[:, -length_size:].squeeze()  # 注意这一步
                val_loss_batch = criterion(pred_val_y, val_y)  # 计算损失
                total_val_loss += val_loss_batch.item()

            avg_val_loss = total_val_loss / len(val_loader)  # 计算本epoch的平均验证损失
            val_loss.append(avg_val_loss)  # 记录平均验证损失

            scheduler.step(avg_val_loss)  # 更新学习率（基于当前验证损失）

        # 打印训练信息
        if print_train == True:
            if (epoch + 1) % print_frequency == 0:
                print(f"Epoch: {epoch + 1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # 早停判断
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stop_counter = 0  # 重置早停计数器
        else:
            early_stop_counter += 1
            if early_stop_counter >= early_patience_epochs:
                print(f'Early stopping triggered at epoch {epoch + 1}.')
                break  # 早停

    net.train()  # 恢复训练模式
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


#df = pd.read_csv('data\\禹城数据（05年-10年）.csv')
df = pd.read_csv('data\\change_data\\smear_original_freq_processed.csv')

# --- 方向 A 优化：特征工程 (Feature Engineering) ---
# 1. 滞后项 (Lagged Features): 捕捉生态滞后响应
#for col in ['PAR', 'DA_TAC', 'L1-VWC']:
for col in ['PAR', 'AirTemp', 'SoilWatCont']:
    for lag in range(1, 4):
        df[f'{col}_lag{lag}'] = df[col].shift(lag)

# 2. 差分项 (Differential Features): 捕捉环境突变速度
#for col in ['PAR', 'DA_TAC']:
for col in ['PAR', 'AirTemp']:
    df[f'{col}_diff'] = df[col].diff()

# 3. 清理 NaN (由于 shift 和 diff 产生)
df.dropna(inplace=True)

# 4. 重新排列列顺序：确保目标变量 'target' 始终在最后一列
feature_cols = [c for c in df.columns if c not in ['date', 'target']]
df = df[['date'] + feature_cols + ['target']]

# 注意多变量情况下，目标变量必须为最后一列
data_target = df['target']  # 预测的目标变量
features = df[feature_cols] # 提取特征列

# --- 方向 B 优化：PCA 降维去噪 (PCA Denoising) ---
# 1. 标准化 (Standardization): PCA 前必须进行标准化
scaler_pca = StandardScaler()
features_scaled = scaler_pca.fit_transform(features)

# 2. 应用 PCA: 保留 95% 的方差，消除特征间的共线性
pca = PCA(n_components=0.95) 
features_pca = pca.fit_transform(features_scaled)
print(f"PCA 降维完成：特征维度从 {features.shape[1]} 降至 {features_pca.shape[1]}")

# 3. 合并 PCA 特征与目标变量
data_pca = np.hstack((features_pca, data_target.values.reshape(-1, 1)))
data = pd.DataFrame(data_pca)

data_dim = data.shape[1]  # 动态更新变量总数 (主成分数 + 1个目标变量)
data_target = data.iloc[:, -1]  # 更新目标变量引用

# 时间戳
df_stamp = df[['date']]
df_stamp['date'] = pd.to_datetime(df_stamp.date)
#data_stamp = time_features(df_stamp, timeenc=1, freq='B')  # 这一步很关键，注意数据的freq
data_stamp = time_features(df_stamp, timeenc=1, freq='t')  # 这一步很关键，注意数据的freq
"""
The following frequencies are supported:
    Y   - yearly
        alias: A
    M   - monthly
    W   - weekly
    D   - daily
    B   - business days
    H   - hourly
    T   - minutely
        alias: min
    S   - secondly
"""
# # 无验证集

# # 数据归一化
scaler = MinMaxScaler()
data_inverse = scaler.fit_transform(np.array(data))

data_length = len(data_inverse)
train_set = 0.8

data_train = data_inverse[:int(train_set * data_length), :]  # 读取目标数据，第一列记为0：1，后面以此类推, 训练集和验证集，如果是多维输入的话最后一列为目标列数据
data_train_mark = data_stamp[:int(train_set * data_length), :]
data_test = data_inverse[int(train_set * data_length):, :]  # 这里把训练集和测试集分开了，也可以换成两个csv文件
data_test_mark = data_stamp[int(train_set * data_length):, :]

n_feature = data_dim
#window = 20  # 模型输入序列长度 (从 10 增加到 20 以增强历史背景感知)
window = 60
length_size = 1  # 预测结果的序列长度
batch_size = 32

train_loader, x_train, y_train, x_train_mark, y_train_mark = tslib_data_loader(window, length_size, batch_size,
                                                                               data_train, data_train_mark)
test_loader, x_test, y_test, x_test_mark, y_test_mark = tslib_data_loader(window, length_size, batch_size, data_test,
                                                                          data_test_mark)
"""
# 有验证集

# 数据归一化
scaler = MinMaxScaler()
data_inverse = scaler.fit_transform(np.array(data))
data_length = len(data_inverse)
train_ratio = 0.6
val_ratio = 0.8
# 6：2：2
window = 30  # 模型输入序列长度 过去30天的数据
length_size = 1  # 预测结果的序列长度  预测未来1天
train_size = int(data_length * train_ratio)
val_size = int(data_length * val_ratio)
data_train = data_inverse[:train_size, :]
data_train_mark = data_stamp[:train_size, :]
data_val = data_inverse[train_size: val_size, :]
data_val_mark = data_stamp[train_size: val_size, :]
data_test = data_inverse[val_size:, :]
data_test_mark = data_stamp[val_size:, :]
batch_size = 32
train_loader, x_train, y_train, x_train_mark, y_train_mark = tslib_data_loader(window, length_size, batch_size,
                                                                               data_train, data_train_mark)
val_loader, x_val, y_val, x_val_mark, y_val_mark = tslib_data_loader(window, length_size, batch_size, data_val,
                                                                     data_val_mark)
test_loader, x_test, y_test, x_test_mark, y_test_mark = tslib_data_loader(window, length_size, batch_size, data_test,
                                                                          data_test_mark)

                                                                          """
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_epochs = 100  # 训练迭代次数
learning_rate = 0.0001  # 学习率
scheduler_patience = int(0.25 * num_epochs)  # 转换为整数  学习率调整的patience
early_patience = 0.2  # 训练迭代的早停比例 即patience=0.25*num_epochs


class Config:
    def __init__(self):
        # basic
        self.seq_len = window  # 输入序列长度
        self.label_len = int(window / 2)  # 标签序列长度
        self.pred_len = length_size  # 预测序列长度
        #self.freq = 'b'  # 时间的频率，
        self.freq = 't'  # 时间的频率，
        # 模型训练
        self.batch_size = batch_size  # 批次大小
        self.num_epochs = num_epochs  # 训练的轮数
        self.learning_rate = learning_rate  # 学习率
        self.stop_ratio = early_patience  # 早停的比例
        # 模型 define
        self.dec_in = data_dim  # 解码器输入特征数量, 输入几个变量就是几
        self.enc_in = data_dim  # 编码器输入特征数量
        self.c_out = 1  # 输出维度##########这个很重要
        # 模型超参数
        self.d_model = 64  # 模型维度
        self.n_heads = 4  # 多头注意力头数
        self.dropout = 0.1  # 丢弃率
        self.e_layers = 3  # 编码器块的数量
        self.d_layers = 3  # 解码器块的数量
        self.d_ff = 128  # 全连接网络维度
        self.factor = 5  # 注意力因子
        self.activation = 'gelu'  # 激活函数
        self.channel_independence = 0  # 频道独立性，0:频道依赖，1:频道独立

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
optimizer = optim.Adam(net.parameters(), lr=learning_rate)  # 优化器
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=scheduler_patience, verbose=True)  # 学习率调整策略

trained_model, train_loss, final_epoch = model_train(net, train_loader, length_size, optimizer, criterion, num_epochs,
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
pred = trained_model(x_test.to(device), x_test_mark.to(device), y_test.to(device), y_test_mark.to(device))
true = y_test[:, -length_size:, -1:].detach().cpu()
pred = pred.detach().cpu()
# 检查pred和true的维度并调整
print("Shape of true before adjustment:", true.shape)
print("Shape of pred before adjustment:", pred.shape)

# 可能需要调整pred和true的维度，使其变为二维数组
true = true[:, :, -1]
pred = pred[:, :, -1]  # 假设需要将pred调整为二维数组，去掉最后一维
# true =np.array(true)
# 假设需要将true调整为二维数组

print("Shape of pred after adjustment:", pred.shape)
print("Shape of true after adjustment:", true.shape)

# 这段代码是为了重新更新scaler，因为之前定义的scaler是是十六维，这里重新根据目标数据定义一下scaler
y_data_test_inverse = scaler.fit_transform(np.array(data_target).reshape(-1, 1))
pred_uninverse = scaler.inverse_transform(pred[:, -1:])  # 如果是多步预测， 选取最后一列
true_uninverse = scaler.inverse_transform(true[:, -1:])

true, pred = true_uninverse, pred_uninverse

df_eval = cal_eval(true, pred)  # 评估指标dataframe
print(df_eval)

# --- 保存结果 ---
# 创建 result 和 img 文件夹
output_dir = 'result'
img_dir = os.path.join(output_dir, 'img')
if not os.path.exists(img_dir):
    os.makedirs(img_dir)

# 生成唯一时间戳
now = datetime.now().strftime("%Y%m%d_%H%M%S")

# 绘制并保存结果图
df_pred_true = pd.DataFrame({'Predict': pred.flatten(), 'Real': true.flatten()})
plt.figure(figsize=(12, 4))
plt.plot(df_pred_true['Predict'], label='Predict', color='red')
plt.plot(df_pred_true['Real'], label='Real', color='blue')
plt.title(model_type + ' Result')
plt.legend()
plt.savefig(os.path.join(img_dir, f'{now}.png'))
print(f'预测结果图已保存到 {os.path.join(img_dir, f"{now}.png")} 中。')
plt.show()

# 保存真实值和预测值到 CSV 文件
output_filename = f'{now}.csv'
output_path = os.path.join(output_dir, output_filename)
result_df = pd.DataFrame({'真实值': true.flatten(), '预测值': pred.flatten()})
result_df.to_csv(output_path, index=False, encoding='utf-8')
print(f'真实值和预测值已保存到 {output_path} 文件中。')
