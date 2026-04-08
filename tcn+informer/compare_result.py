import os
import pandas as pd
import numpy as np
import matplotlib
# 如果画图报错或者不弹窗，请取消注释下面这行代码
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime

# 设置全局绘图风格
plt.style.use('seaborn-v0_8-muted')
plt.rc('font', family='sans-serif')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'SimHei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12

# ==========================================
# 1. 自动搜索并按数据集分类实验结果
# ==========================================
result_root = 'result'
# 定义你想对比的模型关键词
target_models = ['Informer', 'Ablation_TCN', 'Baseline_PatchTST', 'LightGBM', 'LSTM', 'TCN_Informer', 'SOTA_ExoTST']
target_datasets = ['DT', 'SX']  # 明确区分两个数据集

summary_data = []
# 用于存储不同数据集的预测数据: plot_data_dict['DT']['Informer'] = df
plot_data_dict = {ds: {} for ds in target_datasets}

print("正在扫描并分类实验结果...")

for folder in os.listdir(result_root):
    folder_path = os.path.join(result_root, folder)
    if not os.path.isdir(folder_path): continue

    # 匹配模型类型
    matched_model = None
    for m in target_models:
        if m.lower() in folder.lower():
            matched_model = m
            break

    # 匹配数据集类型
    matched_dataset = None
    for ds in target_datasets:
        if ds in folder:
            matched_dataset = ds
            break

    # 如果模型和数据集都匹配到了，读取数据
    if matched_model and matched_dataset:
        metrics_files = [f for f in os.listdir(folder_path) if 'metrics.csv' in f]
        data_files = [f for f in os.listdir(folder_path) if 'data.csv' in f]

        if metrics_files and data_files:
            # 读取指标，并打上数据集和模型的标签
            df_m = pd.read_csv(os.path.join(folder_path, metrics_files[0]))
            df_m['Model'] = matched_model
            df_m['Dataset'] = matched_dataset
            summary_data.append(df_m)

            # 读取预测数据
            df_d = pd.read_csv(os.path.join(folder_path, data_files[0]))
            plot_data_dict[matched_dataset][matched_model] = df_d

# 合并所有指标数据
if not summary_data:
    print("未找到任何匹配的实验结果，请检查 result 文件夹！")
    exit()

df_summary_all = pd.concat(summary_data).reset_index(drop=True)

# ==========================================
# 2. 创建总输出目录 (result/Compare/[时间戳])
# ==========================================
now = datetime.now().strftime("%Y%m%d_%H%M%S")
base_output_dir = os.path.join(result_root, 'Compare', now)
os.makedirs(base_output_dir, exist_ok=True)
print(f"\n[INFO] 对比结果将保存在主目录: {base_output_dir}")

# ==========================================
# 3. 遍历数据集，分别生成表格和图片并保存到子目录
# ==========================================
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

for dataset in target_datasets:
    print(f"\n" + "=" * 40)
    print(f"开始处理数据集: {dataset}")
    print("=" * 40)

    # 筛选当前数据集的指标，并按 R2 降序排列
    df_ds = df_summary_all[df_summary_all['Dataset'] == dataset].copy()
    if df_ds.empty:
        print(f"数据集 {dataset} 暂无数据，跳过。")
        continue

    df_ds = df_ds.sort_values(by='R2', ascending=False).reset_index(drop=True)
    print(df_ds[['Model', 'R2', 'MSE', 'MAE', 'MAPE']])

    # 创建该数据集的专属子文件夹 (例如: result/Compare/20260404_180000/DT)
    dataset_out_dir = os.path.join(base_output_dir, dataset)
    os.makedirs(dataset_out_dir, exist_ok=True)

    # 保存该数据集的综合指标 CSV (可选，方便你后续查阅具体数字)
    df_ds.to_csv(os.path.join(dataset_out_dir, f'summary_metrics_{dataset}.csv'), index=False, encoding='utf-8-sig')

    # ------------------------------------------
    # 绘制当前数据集的指标柱状图
    # ------------------------------------------
    fig, ax1 = plt.subplots(figsize=(10, 6))
    x = np.arange(len(df_ds))
    width = 0.35

    bars1 = ax1.bar(x - width / 2, df_ds['R2'], width, label='R2 Score', color='#3498db')
    ax1.set_ylabel('R2 Score (越高越好)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(df_ds['Model'], rotation=15)
    ax1.set_ylim(0, max(1.1, df_ds['R2'].max() * 1.2))  # 动态调整 Y 轴

    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width / 2, df_ds['MSE'], width, label='MSE Loss', color='#e74c3c', alpha=0.7)
    ax2.set_ylabel('MSE Loss (越低越好)')

    plt.title(f'不同模型在 {dataset} 数据集上的性能对比')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.tight_layout()

    # 保存到专属子文件夹
    metrics_img_path = os.path.join(dataset_out_dir, 'metrics_comparison.png')
    plt.savefig(metrics_img_path, dpi=300)

    # ------------------------------------------
    # 绘制当前数据集的预测曲线图
    # ------------------------------------------
    current_plot_data = plot_data_dict[dataset]
    if not current_plot_data: continue

    plt.figure(figsize=(15, 6))
    start_idx, end_idx = 100, 300  # 选取 200 个点展示

    # 画真实值 (取该数据集下任意一个模型的真实值作为基准)
    first_model = list(current_plot_data.keys())[0]
    plt.plot(current_plot_data[first_model]['真实值'].iloc[start_idx:end_idx].values,
             label='真实观测值 (Real)', color='black', linewidth=2.5, linestyle='--')

    # 循环画各模型预测值
    for i, model_name in enumerate(df_ds['Model']):
        if model_name in current_plot_data:
            data = current_plot_data[model_name]['预测值'].iloc[start_idx:end_idx].values
            plt.plot(data, label=f'{model_name} 预测', alpha=0.8, linewidth=1.5, color=colors[i % len(colors)])

    plt.title(f'各模型预测碳通量(NEE)拟合效果对比 ({dataset} 数据集)')
    plt.xlabel('时间步 (30min)')
    plt.ylabel('NEE')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # 保存到专属子文件夹
    curve_img_path = os.path.join(dataset_out_dir, 'prediction_curve_comparison.png')
    plt.savefig(curve_img_path, dpi=300)

    print(f"[{dataset}] 图表已保存至: {dataset_out_dir}")

plt.show()