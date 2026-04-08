import itertools
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

# 默认调参数据集，可通过环境变量 DATA_PATH 覆盖
DEFAULT_DATA_PATH = "data/Yangtze River Delta of China/SX_NEE(20150715-20190424).csv"
#DEFAULT_DATA_PATH = "data/Yangtze River Delta of China/DT_NEE(20141201-20171130).csv"

# 为了先跑通流程，默认每模型3组；后续可提高到8-20组 
DEFAULT_TRIALS = int(os.getenv("TUNE_TRIALS", "3"))


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


def newest_metric_file(prefix):
    if not RESULT_DIR.exists():
        return None
    candidates = sorted(RESULT_DIR.glob(f"{prefix}_*/*_metrics.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def newest_metric_file_in_dir(prefix, base_dir):
    if not base_dir.exists():
        return None
    candidates = sorted(base_dir.glob(f"{prefix}_*/*_metrics.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


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


def run_trial(script_name, model_prefix, hp_env, trial_id, total_trials, model_dir):
    env = os.environ.copy()
    env["DATA_PATH"] = env.get("DATA_PATH", DEFAULT_DATA_PATH)
    env["MODEL_OUTPUT_DIR"] = str(model_dir)
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(ROOT) if not existing_pythonpath else str(ROOT) + os.pathsep + existing_pythonpath
    env.update({k: str(v) for k, v in hp_env.items()})

    script_path = BEST_DIR / script_name
    print(f"\n[{model_prefix}] Trial {trial_id}/{total_trials} -> {hp_env}")
    before_paths = list_trial_paths(model_prefix, RESULT_DIR)

    proc = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    move_new_trial_paths(model_prefix, model_dir, before_paths)

    if proc.returncode != 0:
        print(proc.stdout[-1500:])
        return None, proc.stdout

    metric_file = newest_metric_file_in_dir(model_prefix, model_dir)
    if metric_file is None:
        metric_file = newest_metric_file(model_prefix)
    if metric_file is None:
        return None, proc.stdout

    df = pd.read_csv(metric_file)
    row = df.iloc[0].to_dict()
    row["metrics_file"] = str(metric_file)
    return row, proc.stdout


def tune_model(model_cfg, trials, model_dir):
    best = None
    history = []
    model_start = time.time()

    trial_bar = tqdm(range(1, trials + 1), desc=f"{model_cfg['name']} trials", unit="trial", leave=True)

    for i in trial_bar:
        hp = sample_space(model_cfg["space"])
        trial_start = time.time()
        metrics, _ = run_trial(model_cfg["script"], model_cfg["prefix"], hp, i, trials, model_dir)
        trial_seconds = time.time() - trial_start
        model_elapsed = time.time() - model_start
        avg_trial_seconds = model_elapsed / i
        model_eta_seconds = avg_trial_seconds * (trials - i)

        record = {"trial": i, "hp": hp, "metrics": metrics}
        history.append(record)

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

        best_score = best["score"] if best is not None else None
        if best_score is None:
            trial_bar.set_postfix_str(
                f"last={format_seconds(trial_seconds)} eta={format_seconds(model_eta_seconds)}"
            )
        else:
            trial_bar.set_postfix_str(
                f"last={format_seconds(trial_seconds)} eta={format_seconds(model_eta_seconds)} best_r2={best_score:.4f}"
            )

    model_minutes = (time.time() - model_start) / 60
    print(f"[{model_cfg['name']}] completed in {model_minutes:.1f} min")

    return best, history


def main():
    random.seed(42)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    total_start = time.time()

    models = [
        {
            "name": "PatchTST",
            "script": "PatchTST_best.py",
            "prefix": "PatchTST_Best",
            "space": {
                "HP_BATCH_SIZE": [32, 64, 96],
                "HP_EPOCHS": [80, 120, 160],
                "HP_LR": [0.0001, 0.0002, 0.0003],
                "HP_D_MODEL": [64, 128, 192],
                "HP_E_LAYERS": [2, 3, 4],
                "HP_DROPOUT": [0.1, 0.15, 0.2],
            },
        },
        {
            "name": "GRU",
            "script": "GRU_best.py",
            "prefix": "GRU_Best",
            "space": {
                "HP_BATCH_SIZE": [32, 64, 96],
                "HP_EPOCHS": [120, 150, 180],
                "HP_LR": [0.0003, 0.0005, 0.0008],
                "HP_HIDDEN_SIZE": [64, 128, 192],
                "HP_SCHED_PATIENCE": [8, 12, 16],
                "HP_USE_PCA": [0, 1],
            },
        },
        {
            "name": "LSTM",
            "script": "LSTM_best.py",
            "prefix": "LSTM_Best",
            "space": {
                "HP_BATCH_SIZE": [32, 64, 96],
                "HP_EPOCHS": [120, 150, 180],
                "HP_LR": [0.0003, 0.0005, 0.0008],
                "HP_HIDDEN_SIZE": [64, 128, 192],
                "HP_USE_PCA": [0, 1],
            },
        },
        {
            "name": "LightGBM",
            "script": "LightGBM_best.py",
            "prefix": "LightGBM_Best",
            "space": {
                "HP_N_ESTIMATORS": [400, 700, 1000],
                "HP_LR": [0.02, 0.03, 0.05],
                "HP_NUM_LEAVES": [31, 63, 127],
                "HP_MIN_CHILD_SAMPLES": [10, 20, 30],
                "HP_EARLY_STOP_ROUNDS": [30, 50, 80],
                "HP_USE_PCA": [0, 1],
            },
        },
        {
            "name": "iTransformer",
            "script": "iTransformer_best.py",
            "prefix": "iTransformer_Best",
            "space": {
                "HP_BATCH_SIZE": [32, 64, 96],
                "HP_EPOCHS": [80, 120, 160],
                "HP_LR": [0.0001, 0.0002, 0.0003],
                "HP_D_MODEL": [64, 128, 192],
                "HP_E_LAYERS": [2, 3, 4],
                "HP_D_FF": [128, 256, 384],
            },
        },
        {
            "name": "Informer",
            "script": "Informer_best.py",
            "prefix": "Informer_Best",
            "space": {
                "HP_BATCH_SIZE": [32, 64, 96],
                "HP_EPOCHS": [80, 120, 160],
                "HP_LR": [0.0001, 0.0002, 0.0003],
                "HP_D_MODEL": [64, 128, 192],
                "HP_E_LAYERS": [2, 3, 4],
                "HP_D_LAYERS": [1, 2, 3],
                "HP_D_FF": [128, 256, 384],
            },
        },
        {
            "name": "ExoTST",
            "script": "ExoTST_best.py",
            "prefix": "ExoTST_Best",
            "space": {
                "HP_BATCH_SIZE": [32, 64, 96],
                "HP_EPOCHS": [80, 120, 160],
                "HP_LR": [0.0001, 0.0002, 0.0003],
                "HP_D_MODEL": [128, 192, 256],
                "HP_E_LAYERS": [2, 3, 4],
                "HP_D_FF": [256, 384, 512],
            },
        },
        {
            "name": "TCN",
            "script": "TCN_best.py",
            "prefix": "TCN_Best",
            "space": {
                "HP_BATCH_SIZE": [32, 64, 96],
                "HP_EPOCHS": [100, 150, 200],
                "HP_LR": [0.0001, 0.0003, 0.0005],
                "HP_D_MODEL": [64, 96, 128],
                "HP_E_LAYERS": [4, 5, 6],
                "HP_DROPOUT": [0.1, 0.15, 0.2],
                "HP_USE_PCA": [0, 1],
            },
        },
    ]

    all_summary = []
    all_histories = {}

    print(f"Start tuning {len(models)} models, trials/model={DEFAULT_TRIALS}")
    model_bar = tqdm(models, desc="All models", unit="model", leave=True)
    total_models = len(models)
    for model_index, m in enumerate(model_bar, start=1):
        print(f"\n===== Tuning {m['name']} =====")
        model_loop_start = time.time()
        model_dir = RESULT_DIR / m["name"]
        model_dir.mkdir(parents=True, exist_ok=True)
        best, history = tune_model(m, DEFAULT_TRIALS, model_dir)
        all_histories[m["name"]] = history

        if best is None:
            summary_row = {
                "Model": m["name"],
                "Best_R2": None,
                "Best_HP": None,
                "HP_USE_PCA": None,
                "Metrics_File": None,
            }
        else:
            hp_use_pca = best["hp"].get("HP_USE_PCA")
            summary_row = {
                "Model": m["name"],
                "Best_R2": best["score"],
                "Best_HP": json.dumps(best["hp"], ensure_ascii=False),
                "HP_USE_PCA": hp_use_pca,
                "Metrics_File": best["metrics"].get("metrics_file"),
            }

        all_summary.append(summary_row)

        model_ts = time.strftime("%Y%m%d_%H%M%S")
        model_summary_path = model_dir / f"{m['name']}_summary_{model_ts}.csv"
        pd.DataFrame([summary_row]).to_csv(model_summary_path, index=False, encoding="utf-8-sig")

        model_history_path = model_dir / f"{m['name']}_history_{model_ts}.json"
        with open(model_history_path, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

        model_minutes = (time.time() - model_loop_start) / 60
        best_r2 = "None" if best is None else f"{best['score']:.4f}"
        total_elapsed = time.time() - total_start
        avg_model_seconds = total_elapsed / model_index
        total_eta_seconds = avg_model_seconds * (total_models - model_index)
        model_bar.set_postfix_str(
            f"last={m['name']} {model_minutes:.1f}m eta={format_seconds(total_eta_seconds)} best_r2={best_r2}"
        )
        print(f"[{m['name']}] summary: {model_summary_path}")
        print(f"[{m['name']}] details: {model_history_path}")

    ts = time.strftime("%Y%m%d_%H%M%S")
    summary_path = RESULT_DIR / f"tuning_summary_{ts}.csv"
    pd.DataFrame(all_summary).to_csv(summary_path, index=False, encoding="utf-8-sig")

    detail_path = RESULT_DIR / f"tuning_history_{ts}.json"
    with open(detail_path, "w", encoding="utf-8") as f:
        json.dump(all_histories, f, ensure_ascii=False, indent=2)

    total_minutes = (time.time() - total_start) / 60
    print(f"Total elapsed: {total_minutes:.1f} min")
    print(f"\nDone. Summary: {summary_path}")
    print(f"Details: {detail_path}")


if __name__ == "__main__":
    main()
