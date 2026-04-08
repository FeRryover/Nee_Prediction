import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # <--- 加上这一行，强制使用独立的弹窗显示图片
import matplotlib.pyplot as plt
import os
from datetime import datetime
import lightgbm as lgb

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from tqdm import tqdm

plt.rc('font', family='sans-serif')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'SimHei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use("ggplot")


# ==========================================
# 1. 核心评估与数据清洗函数 (完全复用你的代码)
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
# 2. LightGBM 专用的窗口展平函数 (核心改造)
# ==========================================
def create_lgbm_dataset(data, window, length_size):
    """
    将时间序列数据转换为 LightGBM 需要的 2D 表格结构
    输入: data shape 为 (N, num_features)
    输出:
      X shape 为 (Samples, window * num_features)
      Y shape 为 (Samples, length_size) -> 代表未来 48 步的目标值
    """
    sequence_length = window + length_size
    num_samples = len(data) - sequence_length + 1

    result = np.empty((num_samples, sequence_length, data.shape[1]), dtype=np.float32)
    for i in range(num_samples):
        result[i] = data[i: i + sequence_length]

    # X: 截取前 window 步，并将其全部展平成 1D 特征向量
    X = result[:, :window, :].reshape(num_samples, -1)
    # Y: 截取后 length_size 步的目标变量 (你的代码中 Target 在拼接后的最后一列)
    Y = result[:, -length_size:, -1]

    return X, Y


# ==========================================
# 3. 读取数据与特征工程 (严格对齐你的预处理)
# ==========================================
data_path = 'data/Yangtze River Delta of China/DT_NEE(20141201-20171130).csv'
#data_path = 'data/Yangtze River Delta of China/SX_NEE(20150715-20190424).csv'
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

# 严格的时间序列切分
data_length = len(df)
train_size = int(0.8 * data_length)

features_train = features[:train_size, :]
features_test = features[train_size:, :]
target_train = data_target[:train_size, :]
target_test = data_target[train_size:, :]

# PCA 降维过滤
scaler_pca = StandardScaler()
features_train_scaled = scaler_pca.fit_transform(features_train)
features_test_scaled = scaler_pca.transform(features_test)

pca = PCA(n_components=0.95)
features_train_pca = pca.fit_transform(features_train_scaled)
features_test_pca = pca.transform(features_test_scaled)

# 拼接特征与目标
data_train_raw = np.concatenate((features_train_pca, target_train), axis=1)
data_test_raw = np.concatenate((features_test_pca, target_test), axis=1)

# 整体输入归一化
scaler = MinMaxScaler()
data_train = scaler.fit_transform(data_train_raw)
data_test = scaler.transform(data_test_raw)

# ==========================================
# 4. 构建 LightGBM 训练与测试集
# ==========================================
window = 96  # 过去 96 步
length_size = 48  # 预测未来 48 步

print("正在构建 LightGBM 数据集矩阵...")
X_train, Y_train = create_lgbm_dataset(data_train, window, length_size)
X_test, Y_test = create_lgbm_dataset(data_test, window, length_size)

# ==========================================
# 5. 模型训练: 直接多步预测策略 (Direct Strategy)
# ==========================================
# 我们将为未来的 48 个半小时，分别训练 48 个专属的 LightGBM 模型
models = []
Y_pred_scaled = np.zeros_like(Y_test)  # 用于存放预测结果

# LightGBM 参数 (映射了你深度学习的部分设定)
lgb_params = {
    'n_estimators': 300,  # 对应 DL 中的迭代次数
    'learning_rate': 0.05,  # 树模型学习率通常比 DL 稍大
    'max_depth': 6,  # 控制过拟合
    'num_leaves': 31,
    'objective': 'regression',
    'metric': 'mse',  # 对应你的 nn.MSELoss()
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'n_jobs': -1  # 开启多线程加速
}

print(f"开始训练 {length_size} 个 LightGBM 模型进行多步预测...")
loop = tqdm(range(length_size), desc="Training Multiple LGBM Models")

for step in loop:
    # 针对第 step 步训练模型
    model = lgb.LGBMRegressor(**lgb_params)

    # 从 Y_train 中取出当前步作为 Target 进行拟合
    model.fit(
        X_train, Y_train[:, step],
        eval_set=[(X_test, Y_test[:, step])],
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]  # 对应你的 early_patience
    )

    # 存入列表，并进行预测
    models.append(model)
    Y_pred_scaled[:, step] = model.predict(X_test)

# ==========================================
# 6. 反归一化与评估 (与你文件中的逻辑严格一致)
# ==========================================
# 由于你的评估是只对最后一步 (或者你需要的所有步) 进行评估
# 这里我们将对所有测试样本的最后一个预测步 (即未来第 48 步) 进行逆变换和对比
print("\n训练完成，正在反归一化...")

# 重新建立仅针对 Target 的 Scaler
# --- 改进：使用原始单位的训练集进行 fit ---
target_scaler_inv = MinMaxScaler()
raw_target_train = target_train
target_scaler_inv.fit(raw_target_train)

# 获取最后一步(最远预测)作为评估标准
pred_uninverse = target_scaler_inv.inverse_transform(Y_pred_scaled[:, -1:])
true_uninverse = target_scaler_inv.inverse_transform(Y_test[:, -1:])

# 为了与深度学习模型的时间轴完全对齐（跳过开头的 window + length_size - 1 个由于滑动窗口不可达的点）
shift = window + length_size - 1
true_final = true_uninverse[shift:]
pred_final = pred_uninverse[shift:]

df_eval = cal_eval(true_final, pred_final)
print("\n====== LightGBM 模型评估结果 ======")
print(df_eval)

# ==========================================
# 7. 保存结果并绘图 (自动提取数据集名称并加入时间戳)
# ==========================================
import os
from datetime import datetime

# 1. 提取数据集名称和当前时间戳
dataset_name = os.path.splitext(os.path.basename(data_path))[0]
now = datetime.now().strftime("%Y%m%d_%H%M%S")

# 2. 定义本次运行的专属文件夹名称 (例如: TCNInformer_20231024_153000_DT_NEE)
run_folder_name = f"LightGBM_{now}_{dataset_name}"

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
data_filename = f'{run_folder_name}_data.csv'
data_path = os.path.join(output_dir, data_filename)
# --- 改进：严谨的时间戳对齐 (与深度学习模型对齐逻辑) ---
test_dates = df['date'].iloc[train_size + window + length_size - 1 : train_size + window + length_size - 1 + len(true_final)].reset_index(drop=True)
result_df = pd.DataFrame({'时间': test_dates, '真实值': true_final.ravel(), '预测值': pred_final.ravel()})
result_df.to_csv(data_path, index=False, encoding='utf-8-sig')
print(f'[SUCCESS] 预测数据已保存: {data_filename}')

# 6. 绘制并保存结果图 (Image) 到专属文件夹
df_pred_true = pd.DataFrame({'Predict': pred_final.flatten(), 'Real': true_final.flatten()})
plt.figure(figsize=(12, 4))
plt.plot(df_pred_true['Predict'], label='Predict', color='red', alpha=0.8)
plt.plot(df_pred_true['Real'], label='Real', color='blue', alpha=0.5)

# 标题带上数据集名称
plt.title(f'LightGBM Result ({dataset_name})')
plt.legend()

# 保存图片
img_filename = f'{run_folder_name}.png'
img_save_path = os.path.join(output_dir, img_filename)
plt.savefig(img_save_path, bbox_inches='tight')
print(f'[SUCCESS] 预测结果图已保存: {img_filename}\n')

# 最后展示图片
plt.show()