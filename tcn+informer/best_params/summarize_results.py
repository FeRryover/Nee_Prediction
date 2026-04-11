# 在 result_best_DT / result_best_SX文件夹中，汇总每个模型的最佳结果（R²最高的 trial），生成一个 CSV 文件和一个 JSON 文件。
# 即针对单个模型的训练结果重新生成 tuning_summary_*.csv 和 tuning_history_*.json，供后续分析使用。

import json
import time
from pathlib import Path
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
RESULT_DIR = SCRIPT_DIR.parent / "result_best_DT"
if not RESULT_DIR.exists():
    raise FileNotFoundError(f"Result directory not found: {RESULT_DIR}")

all_summary = []                 # 用于存储每个模型的最佳结果，列表中的每个元素是一个字典
all_histories = {}               # 用于存储每个模型的完整历史数据，键为模型名称，值为历史列表

for model_dir in RESULT_DIR.iterdir():
    if not model_dir.is_dir():
        continue
    # 找最新的 history json
    history_files = sorted(model_dir.glob("*_history_*.json"),
                           key=lambda p: p.stat().st_mtime, reverse=True)
    if not history_files:
        continue
    with open(history_files[0], encoding="utf-8") as f:
        history = json.load(f)
    all_histories[model_dir.name] = history      # 存入汇总字典

    # 找 R² 最高的 trial
    best_trial = max(history, key=lambda t: t.get("metrics", {}).get("R2", -1e9))
    all_summary.append({
        "Model": model_dir.name,
        "Best_R2": best_trial["metrics"]["R2"],
        "Best_HP": json.dumps(best_trial["hp"], ensure_ascii=False),
        "HP_USE_PCA": best_trial["hp"].get("HP_USE_PCA"),
        "Metrics_File": best_trial["metrics"]["metrics_file"],
    })

# 写入文件
ts = time.strftime("%Y%m%d_%H%M%S")
pd.DataFrame(all_summary).to_csv(RESULT_DIR / f"tuning_summary_{ts}.csv",
                                 index=False, encoding="utf-8-sig")
with open(RESULT_DIR / f"tuning_history_{ts}.json", "w", encoding="utf-8") as f:
    json.dump(all_histories, f, ensure_ascii=False, indent=2)

print(f"生成完成: tuning_summary_{ts}.csv 和 tuning_history_{ts}.json")