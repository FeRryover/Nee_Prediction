#用于绘制beseline的结果对比图，自动搜索result文件夹下的Baseline_DT和Baseline_SX两个子文件夹，分别读取其中的指标数据和预测数据，并生成对比图表。
#最终结果保存在代码同级目录的 result_compare/[时间戳] 目录下，每个数据集一个子文件夹，包含综合指标表格和对比图像。

import os
import pandas as pd
import numpy as np
import matplotlib
# 如果画图报错或者不弹窗，请取消注释下面这行代码
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from datetime import datetime

# 设置全局绘图风格
plt.style.use('seaborn-v0_8-muted')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文显示
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12

# ==========================================
# 1. 自动搜索并按数据集分类实验结果
# ==========================================
script_dir = os.path.dirname(os.path.abspath(__file__))
result_root = os.path.join(script_dir, 'result')
# 定义你想对比的模型关键词
target_models = ['GRU', 'ExoTST', 'iTransformer', 'LightGBM', 'LSTM', 'PatchTST', 'Informer', 'TCNInformer', 'TCN']
target_datasets = ['DT', 'SX']  # 明确区分两个数据集
dataset_folder_map = {
    'DT': 'Baseline_DT',
    'SX': 'Baseline_SX'
}

summary_data = []
# 用于存储不同数据集的预测数据: plot_data_dict['DT']['Informer'] = df
plot_data_dict = {ds: {} for ds in target_datasets}

print("正在扫描并分类实验结果...")

for dataset in target_datasets:
    dataset_dir = os.path.join(result_root, dataset_folder_map[dataset])
    if not os.path.isdir(dataset_dir):
        print(f"[WARN] 未找到目录: {dataset_dir}")
        continue

    for folder in os.listdir(dataset_dir):
        folder_path = os.path.join(dataset_dir, folder)
        if not os.path.isdir(folder_path):
            continue

        # 匹配模型类型（优先匹配更长名称，避免 TCNInformer 被 TCN 误匹配）
        matched_model = None
        for m in sorted(target_models, key=len, reverse=True):
            if m.lower() in folder.lower():
                matched_model = m
                break

        # 如果模型匹配到了，读取数据
        if matched_model:
            metrics_files = [f for f in os.listdir(folder_path) if 'metrics.csv' in f]
            data_files = [f for f in os.listdir(folder_path) if 'data.csv' in f]

            if metrics_files and data_files:
                # 读取指标，并打上数据集和模型的标签
                df_m = pd.read_csv(os.path.join(folder_path, metrics_files[0]))
                df_m['Model'] = matched_model
                df_m['Dataset'] = dataset
                summary_data.append(df_m)

                # 读取预测数据
                df_d = pd.read_csv(os.path.join(folder_path, data_files[0]))
                plot_data_dict[dataset][matched_model] = df_d

# 合并所有指标数据
if not summary_data:
    print("未找到任何匹配的实验结果，请检查 result 文件夹！")
    exit()

df_summary_all = pd.concat(summary_data).reset_index(drop=True)

# ==========================================
# 2. 创建总输出目录 (result_compare/[时间戳])
# ==========================================
out_dir = os.path.join(script_dir, 'result_compare', 'Baseline_' + datetime.now().strftime("%Y%m%d_%H%M%S"))
base_output_dir = os.path.join(script_dir, 'result_compare', out_dir)
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

    # 创建该数据集的专属子文件夹 (例如: result_compare/20260404_180000/DT)
    dataset_out_dir = os.path.join(base_output_dir, dataset)
    os.makedirs(dataset_out_dir, exist_ok=True)

    # 保存该数据集的综合指标 CSV (可选，方便你后续查阅具体数字)
    df_ds.to_csv(os.path.join(dataset_out_dir, f'summary_metrics_{dataset}.csv'), index=False, encoding='utf-8-sig')

    # ------------------------------------------
    # 绘制当前数据集的指标柱状图
    # ------------------------------------------
    fig, ax1 = plt.subplots(figsize=(10, 6))
    x = np.arange(len(df_ds))
    width = 0.6

    bars1 = ax1.bar(x, df_ds['R2'], width, label='R2 Score', color='#3498db')
    ax1.bar_label(bars1, fmt='%.3f', padding=3)
    ax1.set_ylabel('R2 Score (越高越好)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(df_ds['Model'], rotation=15)
    ax1.set_ylim(0, max(1.1, df_ds['R2'].max() * 1.2))  # 动态调整 Y 轴

    plt.title(f'不同模型在 {dataset} 数据集上的性能对比')
    ax1.legend(loc='upper right')
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
    
    # 学术高对比度配色 (Top 1 亮红, Top 2 藏青蓝)
    academic_colors = ['#e74c3c', '#2980b9'] 

    # 保留三条线: 真实值 + R2排名前二模型
    sorted_models = [m for m in df_ds['Model'].tolist() if m in current_plot_data]
    selected_models = sorted_models[:2]

    # 1. 画真实值 (作为底层基准，使用黑色虚线)
    ref_model_for_real = selected_models[0] if selected_models else list(current_plot_data.keys())[0]
    plt.plot(current_plot_data[ref_model_for_real]['真实值'].iloc[start_idx:end_idx].values,
             label='真实观测值 (Ground Truth)', color='#2c3e50', linewidth=1.5, linestyle='--', zorder=1)

    # 2. 画预测值 (突出显示你的模型)
    for i, model_name in enumerate(selected_models):
        data = current_plot_data[model_name]['预测值'].iloc[start_idx:end_idx].values
        
        # 视觉层级控制：Top 1 (你的模型) 更粗、更不透明且层级最高
        line_weight = 2.5 if i == 0 else 1.8
        line_alpha = 1.0 if i == 0 else 0.85
        z_order = 3 if i == 0 else 2  # 确保 Top 1 的线画在最上层，不被基线遮挡
        
        plt.plot(data, label=f'{model_name} 预测', 
                 alpha=line_alpha, 
                 linewidth=line_weight, 
                 color=academic_colors[i % len(academic_colors)],
                 zorder=z_order)

    plt.title(f'各模型预测拟合效果微观对比 ({dataset} 数据集)', fontweight='bold')
    plt.xlabel('时间步', fontweight='bold')
    plt.ylabel('预测数值', fontweight='bold')
    
    # 优化图例：去掉边框，放到右上角外部
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False)
    plt.grid(True, linestyle=':', alpha=0.6) # 网格线改用柔和的点线
    plt.tight_layout()

    # 保存到专属子文件夹
    curve_img_path = os.path.join(dataset_out_dir, 'prediction_curve_comparison.png')
    plt.savefig(curve_img_path, dpi=300, bbox_inches='tight')  # bbox_inches 防止图例被裁减

    print(f"[{dataset}] 图表已保存至: {dataset_out_dir}")
    
plt.show()