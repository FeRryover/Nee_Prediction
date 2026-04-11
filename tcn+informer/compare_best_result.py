# 这个脚本专门用来读取 analyze_best_results.py 生成的排名表（ranking.csv），并严格按照其中提供的绝对路径提取最优结果数据，确保不会因为文件移动或命名不一致而抓错数据。

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from datetime import datetime

# ==========================================
# 0. 全局样式设置
# ==========================================
plt.style.use('seaborn-v0_8-muted')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文显示问题
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12

COLOR_OURS = '#e74c3c'      # 我们的模型 (红色)
COLOR_BASELINE = '#3498db'  # 其他基线模型 (蓝色)
COLOR_BEST_BASE = '#2980b9' # 最强基线 (深蓝色，用于折线图)

# ==========================================
# 1. 精准读取：根据 ranking.csv 提供的路径提取最优结果
# ==========================================
script_dir = os.path.dirname(os.path.abspath(__file__))

# ★ 请确认这里是你存放 DT_ranking.csv 和 SX_ranking.csv 的文件夹名
# 如果你有更新的文件夹，请修改 'Best_summary_20260411_164242'
analysis_dir_name = 'Best_summary_20260411_164242'
ranking_dir = os.path.join(script_dir, 'result_compare', analysis_dir_name)

target_datasets = ['DT', 'SX']
summary_data = []
plot_data_dict = {ds: {} for ds in target_datasets}

print(f"正在读取排行榜指引文件，目录: {ranking_dir} ...")

for dataset in target_datasets:
    ranking_csv_path = os.path.join(ranking_dir, f'{dataset}_ranking.csv')
    
    if not os.path.exists(ranking_csv_path):
        print(f"[警告] 找不到排行榜文件: {ranking_csv_path}")
        continue
        
    # 读取该数据集下的排行榜
    df_rank = pd.read_csv(ranking_csv_path)
    
    # 遍历排行榜里的每一个模型
    for index, row in df_rank.iterrows():
        model_name = row['Model']
        metrics_file = row['Metrics_File']
        data_file = row['Data_File']
        
        try:
            # 1. 严格按照榜单中的绝对路径读取指标
            df_m = pd.read_csv(metrics_file)
            df_m['Model'] = model_name
            df_m['Dataset'] = dataset
            summary_data.append(df_m)
            
            # 2. 严格按照榜单中的绝对路径读取预测曲线数据
            df_d = pd.read_csv(data_file)
            plot_data_dict[dataset][model_name] = df_d
            
            print(f"  ✅ 成功加载最优记录: [{dataset}] {model_name}")
            
        except Exception as e:
            print(f"  ❌ 读取 [{dataset}] {model_name} 失败: 请检查文件是否被移动或删除。")
            print(f"     错误路径: {metrics_file}")

if not summary_data:
    print("\n🚨 致命错误: 未能成功加载任何数据，请检查 ranking_csv 里的路径是否正确！")
    exit()

df_summary_all = pd.concat(summary_data).reset_index(drop=True)

# ==========================================
# 2. 创建输出目录
# ==========================================
out_dir = os.path.join(script_dir, 'result_compare', 'Best_' + datetime.now().strftime("%Y%m%d_%H%M%S"))
os.makedirs(out_dir, exist_ok=True)
print(f"\n✅ 数据提取完毕！将保存绘图至: {out_dir}\n")

# ==========================================
# 3. 辅助画图函数 (修复X轴顺序 + 替换为RMSE)
# ==========================================
def plot_1x3_bar_charts(df, title, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # ★ 使用最稳妥的学术指标组合：R2, MAE, RMSE
    metrics = [('R2', '决定系数 R² (↑)', True), 
               ('MAE', '平均绝对误差 MAE (↓)', False), 
               ('RMSE', '均方根误差 RMSE (↓)', False)]

    # ★ 核心修复：永远按照 R2 降序排列并固定 X 轴，防止各个图的模型顺序乱跳！
    df_sorted = df.sort_values(by='R2', ascending=False).reset_index(drop=True)
    x = np.arange(len(df_sorted))
    colors = [COLOR_OURS if m == 'TCNInformer' else COLOR_BASELINE for m in df_sorted['Model']]

    for ax, (metric, ylabel, is_higher_better) in zip(axes, metrics):
        bars = ax.bar(x, df_sorted[metric], width=0.6, color=colors, alpha=0.85)
        
        fmt = '%.3f' if metric == 'R2' else '%.2f'
        ax.bar_label(bars, fmt=fmt, padding=3, fontsize=10)
        
        ax.set_title(ylabel, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(df_sorted['Model'], rotation=30, ha='right')
        
        max_val = df_sorted[metric].max()
        if is_higher_better and metric == 'R2':
            ax.set_ylim(0, max(1.05, max_val * 1.15))
        else:
            ax.set_ylim(0, max_val * 1.2)
            
        ax.grid(axis='y', linestyle='--', alpha=0.5)

    fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# ==========================================
# 4. 开始绘图
# ==========================================
ablation_list = ['TCN', 'Informer', 'TCNInformer']
compare_list = ['GRU', 'ExoTST', 'iTransformer', 'LightGBM', 'LSTM', 'PatchTST', 'TCNInformer']

for dataset in target_datasets:
    print(f"正在生成 {dataset} 数据集的图表...")
    df_ds = df_summary_all[df_summary_all['Dataset'] == dataset]
    if df_ds.empty: continue
    
    ds_out_dir = os.path.join(out_dir, dataset)
    os.makedirs(ds_out_dir, exist_ok=True)

    # 一、 消融实验图
    df_ablation = df_ds[df_ds['Model'].isin(ablation_list)].copy()
    if not df_ablation.empty:
        plot_1x3_bar_charts(df_ablation, f"消融实验对比 ({dataset} 数据集) - 最优参数", 
                            os.path.join(ds_out_dir, f'1_Ablation_Metrics_{dataset}.png'))

    # 二、 对比实验图
    df_compare = df_ds[df_ds['Model'].isin(compare_list)].copy()
    if not df_compare.empty:
        plot_1x3_bar_charts(df_compare, f"基准模型对比 ({dataset} 数据集) - 最优参数", 
                            os.path.join(ds_out_dir, f'2_Compare_Metrics_{dataset}.png'))

    # 三、 拟合曲线图
    current_data = plot_data_dict.get(dataset, {})
    if 'TCNInformer' in current_data:
        df_baselines = df_compare[df_compare['Model'] != 'TCNInformer']
        # 严格根据排行榜找到本数据集下真正的最强基线
        best_baseline = df_baselines.sort_values(by='R2', ascending=False).iloc[0]['Model'] if not df_baselines.empty else None
        
        plt.figure(figsize=(15, 6))
        start_idx, end_idx = 100, 300  
        
        real_values = current_data['TCNInformer']['真实值'].iloc[start_idx:end_idx].values
        plt.plot(real_values, label='真实观测值 (Ground Truth)', color='black', linewidth=1.5, linestyle='--', zorder=1)
        
        if best_baseline and best_baseline in current_data:
            base_preds = current_data[best_baseline]['预测值'].iloc[start_idx:end_idx].values
            plt.plot(base_preds, label=f'{best_baseline} (最强基线)', color=COLOR_BEST_BASE, linewidth=1.8, alpha=0.85, zorder=2)
            
        our_preds = current_data['TCNInformer']['预测值'].iloc[start_idx:end_idx].values
        plt.plot(our_preds, label='TCNInformer (本项目)', color=COLOR_OURS, linewidth=2.5, alpha=1.0, zorder=3)
        
        plt.title(f'模型局部微观拟合效果对比 ({dataset} 数据集)', fontweight='bold', fontsize=14)
        plt.xlabel('时间步', fontweight='bold')
        plt.ylabel('NEE 预测数值', fontweight='bold')
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False)
        plt.grid(True, linestyle=':', alpha=0.6)
        
        plt.figtext(0.5, -0.05, "注：本图展示为各模型在当前数据集下处于最优超参数配置时的局部拟合效果对比。", 
                    ha="center", fontsize=11, fontweight='bold', color='#555555')
        
        plt.tight_layout()
        plt.savefig(os.path.join(ds_out_dir, f'3_Fitting_Curve_{dataset}.png'), dpi=300, bbox_inches='tight')
        plt.close()

print(f"[{dataset}] 图表已保存至: {ds_out_dir}")