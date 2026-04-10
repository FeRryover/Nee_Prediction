import json
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


DATASET_DIRS = {
    "DT": "result_best_DT",
    "SX": "result_best_SX",
}


REQUIRED_SUMMARY_COLUMNS = ["Model", "Best_R2", "Best_HP", "HP_USE_PCA", "Metrics_File"]
REQUIRED_METRICS_COLUMNS = ["R2", "MSE", "RMSE", "MAE", "MAPE"]


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def dm_test(e1: np.ndarray, e2: np.ndarray) -> Tuple[float, float]:
    """Diebold-Mariano (loss differential = e1^2 - e2^2)."""
    n = min(len(e1), len(e2))
    if n < 10:
        return float("nan"), float("nan")

    d = (e1[:n] ** 2) - (e2[:n] ** 2)
    d_mean = float(np.mean(d))
    d_var = float(np.var(d, ddof=1))
    if d_var <= 0:
        return float("nan"), float("nan")

    stat = d_mean / math.sqrt(d_var / n)
    p_value = 2.0 * (1.0 - normal_cdf(abs(stat)))
    return stat, p_value


def pick_latest_file(root: Path, pattern: str) -> Optional[Path]:
    files = list(root.glob(pattern))
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_mtime)


def resolve_metrics_file(dataset_root: Path, metrics_hint: str) -> Optional[Path]:
    hint_path = Path(metrics_hint)
    hint_name = hint_path.name

    candidates = list(dataset_root.rglob(hint_name))
    if candidates:
        return candidates[0]

    return None


def read_data_file_from_metrics(metrics_file: Path) -> Optional[Path]:
    data_file_name = metrics_file.name.replace("_metrics.csv", "_data.csv")
    data_file = metrics_file.with_name(data_file_name)
    if data_file.exists():
        return data_file
    return None


def parse_hp_json(hp_text: str) -> Dict:
    try:
        return json.loads(hp_text)
    except Exception:
        return {"raw": hp_text}


def analyze_dataset(dataset: str, dataset_root: Path) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, np.ndarray]]:
    summary_file = pick_latest_file(dataset_root, "tuning_summary_*.csv")
    if summary_file is None:
        raise FileNotFoundError(f"未找到 {dataset} 的 tuning_summary_*.csv")

    summary_df = pd.read_csv(summary_file)
    missing_cols = [c for c in REQUIRED_SUMMARY_COLUMNS if c not in summary_df.columns]
    if missing_cols:
        raise ValueError(f"{summary_file} 缺少字段: {missing_cols}")

    rows: List[Dict] = []
    errors_by_model: Dict[str, np.ndarray] = {}

    for _, r in summary_df.iterrows():
        model = str(r["Model"])
        best_r2 = float(r["Best_R2"])
        hp_text = str(r["Best_HP"])
        hp_use_pca = int(r["HP_USE_PCA"])
        metrics_hint = str(r["Metrics_File"])

        metrics_file = resolve_metrics_file(dataset_root, metrics_hint)
        if metrics_file is None:
            print(f"[WARN] {dataset}-{model} 未定位到 metrics 文件，跳过。hint={metrics_hint}")
            continue

        metrics_df = pd.read_csv(metrics_file)
        missing_metric_cols = [c for c in REQUIRED_METRICS_COLUMNS if c not in metrics_df.columns]
        if missing_metric_cols:
            print(f"[WARN] {metrics_file} 缺少指标字段: {missing_metric_cols}，跳过。")
            continue

        m0 = metrics_df.iloc[0]

        data_file = read_data_file_from_metrics(metrics_file)
        if data_file is None:
            print(f"[WARN] {dataset}-{model} 未找到 data 文件，跳过 DM。")
            data_file_str = ""
        else:
            data_file_str = str(data_file)
            df_data = pd.read_csv(data_file)
            if "真实值" in df_data.columns and "预测值" in df_data.columns:
                err = (df_data["真实值"].to_numpy(dtype=float) - df_data["预测值"].to_numpy(dtype=float))
                errors_by_model[model] = err
            else:
                print(f"[WARN] {data_file} 缺少 '真实值'/'预测值' 列，跳过 DM。")

        rows.append(
            {
                "Dataset": dataset,
                "Model": model,
                "R2": float(m0["R2"]),
                "MSE": float(m0["MSE"]),
                "RMSE": float(m0["RMSE"]),
                "MAE": float(m0["MAE"]),
                "MAPE": float(m0["MAPE"]),
                "Best_R2_Search": best_r2,
                "HP_USE_PCA": hp_use_pca,
                "Best_HP": hp_text,
                "Best_HP_Parsed": json.dumps(parse_hp_json(hp_text), ensure_ascii=False),
                "Metrics_File": str(metrics_file),
                "Data_File": data_file_str,
            }
        )

    result_df = pd.DataFrame(rows)
    if result_df.empty:
        raise ValueError(f"{dataset} 未解析到有效模型结果。")

    result_df = result_df.sort_values("R2", ascending=False).reset_index(drop=True)
    result_df["Rank_R2"] = np.arange(1, len(result_df) + 1)

    dm_rows = []
    if len(result_df) >= 2:
        champion = str(result_df.iloc[0]["Model"])
        runner_up = str(result_df.iloc[1]["Model"])
        if champion in errors_by_model and runner_up in errors_by_model:
            stat, p = dm_test(errors_by_model[champion], errors_by_model[runner_up])
            dm_rows.append(
                {
                    "Dataset": dataset,
                    "Model_1": champion,
                    "Model_2": runner_up,
                    "DM_Statistic": stat,
                    "p_value": p,
                    "Significant_0.05": bool(p < 0.05) if not np.isnan(p) else False,
                    "Interpretation": "Model_1损失显著更低" if (not np.isnan(stat) and not np.isnan(p) and stat < 0 and p < 0.05) else "差异不显著或Model_1不更优",
                }
            )
        else:
            dm_rows.append(
                {
                    "Dataset": dataset,
                    "Model_1": champion,
                    "Model_2": runner_up,
                    "DM_Statistic": np.nan,
                    "p_value": np.nan,
                    "Significant_0.05": False,
                    "Interpretation": "缺少 data 文件，无法做 DM 检验",
                }
            )

    dm_df = pd.DataFrame(dm_rows)
    return result_df, dm_df, errors_by_model


def plot_r2_comparison(all_df: pd.DataFrame, out_file: Path) -> None:
    pivot = all_df.pivot(index="Model", columns="Dataset", values="R2")
    pivot = pivot.sort_index()

    fig, ax = plt.subplots(figsize=(11, 6))
    width = 0.35
    x = np.arange(len(pivot.index))

    dt_vals = pivot["DT"].to_numpy() if "DT" in pivot.columns else np.zeros(len(pivot.index))
    sx_vals = pivot["SX"].to_numpy() if "SX" in pivot.columns else np.zeros(len(pivot.index))

    ax.bar(x - width / 2, dt_vals, width=width, label="DT", color="#1f77b4")
    ax.bar(x + width / 2, sx_vals, width=width, label="SX", color="#ff7f0e")

    ax.set_title("DT/SX 各模型 R2 对比")
    ax.set_ylabel("R2")
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=20)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_file, dpi=200)
    plt.close(fig)


def build_cross_dataset_rank(all_df: pd.DataFrame) -> pd.DataFrame:
    grouped = all_df.groupby("Model", as_index=False).agg(
        Mean_R2=("R2", "mean"),
        Std_R2=("R2", "std"),
        Mean_MAE=("MAE", "mean"),
        Mean_RMSE=("RMSE", "mean"),
        Mean_MSE=("MSE", "mean"),
        Mean_MAPE=("MAPE", "mean"),
    )
    grouped["Std_R2"] = grouped["Std_R2"].fillna(0.0)
    grouped = grouped.sort_values(["Mean_R2", "Std_R2"], ascending=[False, True]).reset_index(drop=True)
    grouped["CrossDataset_Rank"] = np.arange(1, len(grouped) + 1)

    # 优先展示 R2/MAE/RMSE，MSE/MAPE 放在后面
    ordered_cols = [
        "Model",
        "Mean_R2",
        "Mean_MAE",
        "Mean_RMSE",
        "Mean_MSE",
        "Mean_MAPE",
        "Std_R2",
        "CrossDataset_Rank",
    ]
    grouped = grouped[ordered_cols]
    return grouped


def main() -> None:
    root = Path(__file__).resolve().parent

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = root / "result" / "Compare" / f"final_best_analysis_{run_ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_dataset_rows = []
    all_dm_rows = []

    for ds, rel_dir in DATASET_DIRS.items():
        dataset_root = root / rel_dir
        if not dataset_root.exists():
            print(f"[WARN] 缺少目录: {dataset_root}")
            continue

        ds_df, dm_df, _ = analyze_dataset(ds, dataset_root)
        all_dataset_rows.append(ds_df)
        if not dm_df.empty:
            all_dm_rows.append(dm_df)

        ds_ordered_cols = [
            "Dataset",
            "Model",
            "R2",
            "MAE",
            "RMSE",
            "MSE",
            "MAPE",
            "Rank_R2",
            "Best_R2_Search",
            "HP_USE_PCA",
            "Best_HP",
            "Best_HP_Parsed",
            "Metrics_File",
            "Data_File",
        ]
        ds_df = ds_df[ds_ordered_cols]
        ds_df.to_csv(out_dir / f"{ds}_ranking.csv", index=False, encoding="utf-8-sig")

    if not all_dataset_rows:
        raise RuntimeError("没有可用数据，未生成输出。")

    all_df = pd.concat(all_dataset_rows, ignore_index=True)
    all_df = all_df.sort_values(["Dataset", "Rank_R2"]).reset_index(drop=True)

    all_ordered_cols = [
        "Dataset",
        "Model",
        "R2",
        "MAE",
        "RMSE",
        "MSE",
        "MAPE",
        "Rank_R2",
        "Best_R2_Search",
        "HP_USE_PCA",
        "Best_HP",
        "Best_HP_Parsed",
        "Metrics_File",
        "Data_File",
    ]
    all_df = all_df[all_ordered_cols]

    cross_rank_df = build_cross_dataset_rank(all_df)
    dm_all_df = pd.concat(all_dm_rows, ignore_index=True) if all_dm_rows else pd.DataFrame()

    all_df.to_csv(out_dir / "all_model_summary.csv", index=False, encoding="utf-8-sig")
    cross_rank_df.to_csv(out_dir / "cross_dataset_rank.csv", index=False, encoding="utf-8-sig")
    if not dm_all_df.empty:
        dm_all_df.to_csv(out_dir / "dm_test_summary.csv", index=False, encoding="utf-8-sig")

    plot_r2_comparison(all_df, out_dir / "r2_comparison.png")

    print("\n分析完成，输出目录:")
    print(out_dir)
    print("\n文件列表:")
    print("- all_model_summary.csv")
    print("- DT_ranking.csv / SX_ranking.csv")
    print("- cross_dataset_rank.csv")
    print("- dm_test_summary.csv (若可计算)")
    print("- r2_comparison.png")


if __name__ == "__main__":
    main()
