# 这个脚本专门用来针对单个模型进行超参数调优，独立于其他模型的调优过程。它会在指定的搜索空间内随机采样超参数组合，运行对应的训练脚本，并记录每次试验的结果。
# 最终会生成一个包含最佳超参数和对应指标的总结文件，以及一个包含所有试验历史的详细记录文件。
import json
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm


ROOT = Path(__file__).resolve().parent.parent
BEST_DIR = ROOT / "best_params"
RESULT_DIR = ROOT / "result_best"

DEFAULT_DATA_PATH = "data/Yangtze River Delta of China/SX_NEE(20150715-20190424).csv"
DEFAULT_TRIALS = int(os.getenv("TUNE_TRIALS", "8"))

# 在这里直接修改单模型调参配置
SELECTED_MODEL = "TCNInformer"
SELECTED_TRIALS = DEFAULT_TRIALS

MODELS = {
    "PatchTST": {
        "script": "PatchTST_best.py",
        "prefix": "PatchTST_Best",
        "space": {
            "HP_BATCH_SIZE": [32, 64, 96],
            "HP_EPOCHS": [80, 120, 160],
            "HP_LR": [0.0001, 0.0002, 0.0003],
            "HP_D_MODEL": [64, 128, 192],
            "HP_E_LAYERS": [2, 3, 4],
            "HP_DROPOUT": [0.1, 0.15, 0.2],
            "HP_SCHED_PATIENCE": [6, 8, 12],
            "HP_USE_PCA": [0, 1],
        },
    },
    "GRU": {
        "script": "GRU_best.py",
        "prefix": "GRU_Best",
        "space": {
            "HP_BATCH_SIZE": [32, 64, 96],
            "HP_EPOCHS": [120, 150, 180],
            "HP_LR": [0.0003, 0.0005, 0.0008],
            "HP_HIDDEN_SIZE": [64, 128, 192],
            "HP_NUM_LAYERS": [1, 2, 3],
            "HP_SCHED_PATIENCE": [8, 12, 16],
            "HP_EARLY_PATIENCE": [16, 24, 32],
            "HP_USE_PCA": [0, 1],
        },
    },
    "LSTM": {
        "script": "LSTM_best.py",
        "prefix": "LSTM_Best",
        "space": {
            "HP_BATCH_SIZE": [32, 64, 96],
            "HP_EPOCHS": [120, 150, 180],
            "HP_LR": [0.0003, 0.0005, 0.0008],
            "HP_HIDDEN_SIZE": [64, 128, 192],
            "HP_NUM_LAYERS": [1, 2, 3],
            "HP_EARLY_PATIENCE": [12, 20, 28],
            "HP_USE_PCA": [0, 1],
        },
    },
    "LightGBM": {
        "script": "LightGBM_best.py",
        "prefix": "LightGBM_Best",
        "space": {
            "HP_N_ESTIMATORS": [400, 700, 1000],
            "HP_LR": [0.02, 0.03, 0.05],
            "HP_MAX_DEPTH": [-1, 8, 12],
            "HP_NUM_LEAVES": [31, 63, 127],
            "HP_MIN_CHILD_SAMPLES": [10, 20, 30],
            "HP_REG_LAMBDA": [0.5, 1.0, 3.0],
            "HP_SUBSAMPLE": [0.8, 0.9, 1.0],
            "HP_COLSAMPLE": [0.8, 0.9, 1.0],
            "HP_EARLY_STOP_ROUNDS": [30, 50, 80],
            "HP_USE_PCA": [0, 1],
        },
    },
    "iTransformer": {
        "script": "iTransformer_best.py",
        "prefix": "iTransformer_Best",
        "space": {
            "HP_BATCH_SIZE": [32, 64, 96],
            "HP_EPOCHS": [80, 120, 160],
            "HP_LR": [0.0001, 0.0002, 0.0003],
            "HP_D_MODEL": [64, 128, 192],
            "HP_N_HEADS": [4, 8],
            "HP_E_LAYERS": [2, 3, 4],
            "HP_D_FF": [128, 256, 384],
            "HP_DROPOUT": [0.05, 0.1, 0.2],
            "HP_SCHED_PATIENCE": [6, 8, 12],
            "HP_USE_PCA": [0, 1],
        },
    },
    "Informer": {
        "script": "Informer_best.py",
        "prefix": "Informer_Best",
        "space": {
            "HP_BATCH_SIZE": [32, 64, 96],
            "HP_EPOCHS": [80, 120, 160],
            "HP_LR": [0.0001, 0.0002, 0.0003],
            "HP_D_MODEL": [64, 128, 192],
            "HP_N_HEADS": [4, 8],
            "HP_E_LAYERS": [2, 3, 4],
            "HP_D_LAYERS": [1, 2, 3],
            "HP_D_FF": [128, 256, 384],
            "HP_DROPOUT": [0.03, 0.05, 0.1],
            "HP_SCHED_PATIENCE": [6, 8, 12],
            "HP_USE_PCA": [0, 1],
        },
    },
    "ExoTST": {
        "script": "ExoTST_best.py",
        "prefix": "ExoTST_Best",
        "space": {
            "HP_BATCH_SIZE": [32, 64, 96],
            "HP_EPOCHS": [80, 120, 160],
            "HP_LR": [0.0001, 0.0002, 0.0003],
            "HP_D_MODEL": [128, 192, 256],
            "HP_N_HEADS": [4, 8],
            "HP_E_LAYERS": [2, 3, 4],
            "HP_D_FF": [256, 384, 512],
            "HP_DROPOUT": [0.1, 0.15, 0.2],
            "HP_SCHED_PATIENCE": [6, 8, 12],
            "HP_USE_PCA": [0, 1],
        },
    },
    "TCN": {
        "script": "TCN_best.py",
        "prefix": "TCN_Best",
        "space": {
            "HP_BATCH_SIZE": [32, 64, 96],
            "HP_EPOCHS": [100, 150, 200],
            "HP_LR": [0.0001, 0.0003, 0.0005],
            "HP_D_MODEL": [64, 96, 128],
            "HP_E_LAYERS": [4, 5, 6],
            "HP_DROPOUT": [0.1, 0.15, 0.2],
            "HP_SCHED_PATIENCE": [6, 8, 12],
            "HP_USE_PCA": [0, 1],
        },
    },
    "TCNInformer": {
        "script": "tcn_informer_best.py",
        "prefix": "TCNInformer",
        "space": {
            "HP_BATCH_SIZE": [32, 64, 96],
            "HP_EPOCHS": [80, 120, 160],
            "HP_LR": [0.0001, 0.0002, 0.0003],
            "HP_D_MODEL": [64, 96, 128],
            "HP_N_HEADS": [4, 8],
            "HP_E_LAYERS": [2, 3, 4],
            "HP_D_LAYERS": [2, 3, 4],
            "HP_D_FF": [128, 256, 384],
            "HP_DROPOUT": [0.05, 0.1, 0.15],
            "HP_FACTOR": [3, 5, 7],
            "HP_SCHED_PATIENCE": [6, 8, 12],
            "HP_EARLY_PATIENCE": [0.1, 0.15, 0.2],
            "HP_USE_PCA": [0, 1],
        },
    },
}


def format_seconds(seconds):
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    if minutes > 0:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def sample_space(space):
    sampled = {}
    for k, v in space.items():
        if isinstance(v, tuple) and len(v) == 2 and all(isinstance(x, int) for x in v):
            sampled[k] = random.randint(v[0], v[1])
        elif isinstance(v, tuple) and len(v) == 2:
            sampled[k] = random.uniform(v[0], v[1])
        elif isinstance(v, list):
            sampled[k] = random.choice(v)
        else:
            sampled[k] = v
    return sampled


def list_trial_paths(prefix, base_dir):
    if not base_dir.exists():
        return set()
    return {p for p in base_dir.glob(f"{prefix}_*")}


def move_new_trial_paths(prefix, model_dir, before_paths):
    after_paths = list_trial_paths(prefix, RESULT_DIR)
    new_paths = [p for p in after_paths if p not in before_paths]
    moved = []
    for src in new_paths:
        dst = model_dir / src.name
        if dst.exists():
            dst = model_dir / f"{src.name}_{int(time.time())}"
        shutil.move(str(src), str(dst))
        moved.append(dst)
    return moved


def newest_metric_file_in_dir(prefix, base_dir):
    if not base_dir.exists():
        return None
    candidates = sorted(base_dir.glob(f"{prefix}_*/*_metrics.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def run_trial(model_cfg, hp_env, trial_id, total_trials, model_dir):
    env = os.environ.copy()
    env["DATA_PATH"] = env.get("DATA_PATH", DEFAULT_DATA_PATH)
    env["MODEL_OUTPUT_DIR"] = str(model_dir)
    env["PYTHONUNBUFFERED"] = "1"
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(ROOT) if not existing_pythonpath else str(ROOT) + os.pathsep + existing_pythonpath
    env.update({k: str(v) for k, v in hp_env.items()})

    script_path = BEST_DIR / model_cfg["script"]
    print(f"\n[{model_cfg['prefix']}] Trial {trial_id}/{total_trials} -> {hp_env}")
    before_paths = list_trial_paths(model_cfg["prefix"], RESULT_DIR)

    proc = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    move_new_trial_paths(model_cfg["prefix"], model_dir, before_paths)

    metric_file = newest_metric_file_in_dir(model_cfg["prefix"], model_dir)
    if metric_file is None:
        if proc.returncode != 0:
            print(proc.stdout[-1500:])
        return None, proc.stdout

    df = pd.read_csv(metric_file)
    row = df.iloc[0].to_dict()
    row["metrics_file"] = str(metric_file)

    if proc.returncode != 0:
        # 允许绘图等后处理失败，但指标已落盘时仍计入本轮结果。
        print("[WARN] Trial exited with non-zero code, but metrics file exists. This trial is kept.")
        print(proc.stdout[-1200:])

    return row, proc.stdout


def tune_one_model(model_name, trials):
    model_cfg = {
        "name": model_name,
        **MODELS[model_name],
    }
    model_dir = RESULT_DIR / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    best = None
    history = []
    model_start = time.time()

    trial_bar = tqdm(range(1, trials + 1), desc=f"{model_name} tuning", unit="trial", leave=True)
    for i in trial_bar:
        hp = sample_space(model_cfg["space"])
        trial_start = time.time()
        metrics, _ = run_trial(model_cfg, hp, i, trials, model_dir)
        trial_seconds = time.time() - trial_start

        record = {"trial": i, "hp": hp, "metrics": metrics}
        history.append(record)

        model_elapsed = time.time() - model_start
        avg_trial_seconds = model_elapsed / i
        model_eta_seconds = avg_trial_seconds * (trials - i)

        if metrics is None:
            trial_bar.set_postfix_str(
                f"last={format_seconds(trial_seconds)} eta={format_seconds(model_eta_seconds)} status=failed"
            )
            continue

        score = float(metrics.get("R2", -1e9))
        if (best is None) or (score > best["score"]):
            best = {
                "score": score,
                "hp": hp,
                "metrics": metrics,
            }

        trial_bar.set_postfix_str(
            f"last={format_seconds(trial_seconds)} eta={format_seconds(model_eta_seconds)} best_r2={best['score']:.4f}"
        )

    model_minutes = (time.time() - model_start) / 60
    print(f"[{model_name}] completed in {model_minutes:.1f} min")

    summary_row = {
        "Model": model_name,
        "Best_R2": None if best is None else best["score"],
        "Best_HP": None if best is None else json.dumps(best["hp"], ensure_ascii=False),
        "HP_USE_PCA": None if best is None else best["hp"].get("HP_USE_PCA"),
        "Metrics_File": None if best is None else best["metrics"].get("metrics_file"),
    }

    ts = time.strftime("%Y%m%d_%H%M%S")
    summary_path = model_dir / f"{model_name}_summary_{ts}.csv"
    history_path = model_dir / f"{model_name}_history_{ts}.json"
    pd.DataFrame([summary_row]).to_csv(summary_path, index=False, encoding="utf-8-sig")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    print(f"[{model_name}] summary: {summary_path}")
    print(f"[{model_name}] details: {history_path}")


def main():
    if SELECTED_MODEL not in MODELS:
        print(
            f"Invalid SELECTED_MODEL={SELECTED_MODEL}. "
            f"Available: {', '.join(sorted(MODELS.keys()))}"
        )
        return
    if SELECTED_TRIALS <= 0:
        print(f"Invalid SELECTED_TRIALS={SELECTED_TRIALS}. It must be > 0.")
        return

    random.seed(42)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Start single-model tuning: model={SELECTED_MODEL}, trials={SELECTED_TRIALS}")
    print(f"DATA_PATH={os.getenv('DATA_PATH', DEFAULT_DATA_PATH)}")
    tune_one_model(SELECTED_MODEL, SELECTED_TRIALS)


if __name__ == "__main__":
    main()
