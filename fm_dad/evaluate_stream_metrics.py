"""
evaluate_stream_metrics.py — Compute stream pipeline ablation metrics & export to CSV.

Reads streaming outputs from data/stream_ablation/ (or user specified dir):
  - pipeline_penalties_baseline_baseline.csv / pipeline_penalties_binary_c2.csv
  - live_blacklist_baseline_baseline.csv / live_blacklist_binary_c2.csv
  - live_trust_history_baseline_baseline.csv / live_trust_history_binary_c2.csv

Computes:
  1. Gate-fire parity per agent
  2. Per-attack-type & Macro MCC (Equation 4.1)
  3. Confusion matrix metrics (TP, FP, FN, TN)
  4. Isolation latency T_isolate (mean, median)
  
Saves all computed metrics cleanly into a structured CSV file.

Usage:
    python3 evaluate_stream_metrics.py
    python3 evaluate_stream_metrics.py --results-dir data/stream_ablation --out-csv data/stream_ablation/stream_metrics_summary.csv
"""

from __future__ import annotations

import argparse
import glob
import math
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

_FM_DAD_DIR = Path(__file__).parent
if str(_FM_DAD_DIR) not in sys.path:
    sys.path.insert(0, str(_FM_DAD_DIR))


def _mcc(tp: int, fp: int, fn: int, tn: int) -> float:
    """Equation 4.1 MCC formula."""
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return (tp * tn - fp * fn) / denom if denom > 0 else 0.0


def _load_ground_truth(gt_dir: Path) -> pd.DataFrame:
    """Load and resolve ground truth per node_id across all cycle files."""
    gt_re = re.compile(r"node_attack_ground_truth_(\d+)\.csv$")
    files = sorted(f for f in gt_dir.iterdir() if gt_re.match(f.name))
    if not files:
        raise FileNotFoundError(f"No ground truth files found in {gt_dir}")

    frames = [pd.read_csv(f) for f in files]
    for df in frames:
        df.columns = df.columns.str.strip()

    gt_all = pd.concat(frames, ignore_index=True)

    def _resolve(g):
        if (g["is_attacker"] == 1).any():
            row = g[g["is_attacker"] == 1].iloc[0]
            return pd.Series({"is_attacker": 1, "attack_type": row["attack_type"]})
        return pd.Series({"is_attacker": 0, "attack_type": "NONE"})

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gt_base = gt_all.groupby("node_id", group_keys=False).apply(_resolve).reset_index()

    return gt_base


def compute_mcc_metrics(bl_df: pd.DataFrame, gt: pd.DataFrame, tau_min: float) -> Tuple[Dict[str, dict], float, int]:
    """Compute per-attack-type and macro MCC for a blacklist dataframe."""
    bl = bl_df.copy()
    bl["blacklisted"] = bl["current_trust"] < tau_min

    merged = bl.merge(gt, on="node_id", how="left")
    merged["is_attacker"] = merged["is_attacker"].fillna(0).astype(int)
    merged["attack_type"]  = merged["attack_type"].fillna("NONE")

    total_honest = (merged["is_attacker"] == 0).sum()
    fp_count     = int(((merged["is_attacker"] == 0) & merged["blacklisted"]).sum())
    tn_count     = total_honest - fp_count

    attack_types = sorted(merged[merged["is_attacker"] == 1]["attack_type"].unique())
    results = {}
    for at in attack_types:
        at_ids = merged[merged["attack_type"] == at]["node_id"].values
        tp = int(merged[merged["node_id"].isin(at_ids) &  merged["blacklisted"]].shape[0])
        fn = int(merged[merged["node_id"].isin(at_ids) & ~merged["blacklisted"]].shape[0])
        mcc_val = _mcc(tp, fp_count, fn, tn_count)
        results[at] = {"tp": tp, "fp": fp_count, "fn": fn, "tn": tn_count, "mcc": mcc_val}

    macro_mcc = float(np.mean([v["mcc"] for v in results.values()])) if results else 0.0
    return results, macro_mcc, fp_count


def compute_tisolate(hist_df: pd.DataFrame, gt: pd.DataFrame, tau_min: float) -> Tuple[float, float]:
    """Compute mean and median T_isolate latency."""
    attacker_ids = set(gt[gt["is_attacker"] == 1]["node_id"].tolist())
    att_h = hist_df[hist_df["node_id"].isin(attacker_ids)].copy()

    bl_events = att_h[att_h["trust_after"] < tau_min].groupby("node_id")["cycle_id"].min()
    first_pen = att_h[att_h["delta_applied"] > 0].groupby("node_id")["cycle_id"].min()

    latencies = []
    for nid, bl_cycle in bl_events.items():
        fp_cycle = first_pen.get(nid, bl_cycle)
        latencies.append(int(bl_cycle - fp_cycle))

    if not latencies:
        return 0.0, 0.0

    arr = np.array(latencies)
    return float(arr.mean()), float(np.median(arr))


def evaluate_stream(results_dir: Path, gt_dir: Path, tau_min: float, out_csv: Path) -> pd.DataFrame:
    """Run stream evaluation, print report, and save summary metrics to CSV."""
    pen_b_file = results_dir / "pipeline_penalties_baseline_baseline.csv"
    pen_c_file = results_dir / "pipeline_penalties_binary_c2.csv"

    bl_b_file  = results_dir / "live_blacklist_baseline_baseline.csv"
    bl_c_file  = results_dir / "live_blacklist_binary_c2.csv"

    hist_b_file = results_dir / "live_trust_history_baseline_baseline.csv"
    hist_c_file = results_dir / "live_trust_history_binary_c2.csv"

    for f in [pen_b_file, pen_c_file, bl_b_file, bl_c_file]:
        if not f.exists():
            raise FileNotFoundError(f"Required result file not found: {f}")

    # Load data
    gt    = _load_ground_truth(gt_dir)
    pen_b = pd.read_csv(pen_b_file)
    pen_c = pd.read_csv(pen_c_file)
    bl_b  = pd.read_csv(bl_b_file)
    bl_c  = pd.read_csv(bl_c_file)
    hist_b = pd.read_csv(hist_b_file) if hist_b_file.exists() else None
    hist_c = pd.read_csv(hist_c_file) if hist_c_file.exists() else None

    # 1. Gate-Fire Parity
    gf_b = pen_b[pen_b["gate_fired"] == True].groupby("agent").size().to_dict()
    gf_c = pen_c[pen_c["gate_fired"] == True].groupby("agent").size().to_dict()

    parity_ok = all(gf_b.get(a, -1) == gf_c.get(a, -2) for a in ["sp", "als", "fs", "igh"])

    # 2. MCC Calculation
    b_mcc_dict, b_macro, b_fp = compute_mcc_metrics(bl_b, gt, tau_min)
    c_mcc_dict, c_macro, c_fp = compute_mcc_metrics(bl_c, gt, tau_min)

    # 3. T_isolate Latency
    b_t_mean, b_t_med = compute_tisolate(hist_b, gt, tau_min) if hist_b is not None else (0.0, 0.0)
    c_t_mean, c_t_med = compute_tisolate(hist_c, gt, tau_min) if hist_c is not None else (0.0, 0.0)

    # Build Summary Rows for CSV Export
    records = []
    
    # Add Gate Parity rows
    for agent in ["sp", "als", "fs", "igh"]:
        records.append({
            "metric_category": "gate_fire_count",
            "item": agent.upper(),
            "baseline_value": gf_b.get(agent, 0),
            "config_a_binary_value": gf_c.get(agent, 0),
            "match": gf_b.get(agent, 0) == gf_c.get(agent, 0),
            "notes": "Gate fire parity check"
        })

    # Add MCC rows per attack
    for at in sorted(b_mcc_dict.keys()):
        bv = b_mcc_dict[at]
        cv = c_mcc_dict[at]
        records.append({
            "metric_category": "mcc_score",
            "item": f"{at}_mcc",
            "baseline_value": round(bv["mcc"], 4),
            "config_a_binary_value": round(cv["mcc"], 4),
            "match": False,
            "notes": f"TP_b={bv['tp']} FP_b={bv['fp']} FN_b={bv['fn']} | TP_c={cv['tp']} FP_c={cv['fp']} FN_c={cv['fn']}"
        })

    # Macro MCC
    records.append({
        "metric_category": "mcc_score",
        "item": "macro_mcc",
        "baseline_value": round(b_macro, 4),
        "config_a_binary_value": round(c_macro, 4),
        "match": False,
        "notes": f"Baseline FP={b_fp} | Config A FP={c_fp}"
    })

    # Blacklisted Node Count
    bl_b_count = int(bl_b["current_trust"].lt(tau_min).sum())
    bl_c_count = int(bl_c["current_trust"].lt(tau_min).sum())
    records.append({
        "metric_category": "blacklist_summary",
        "item": "blacklisted_node_count",
        "baseline_value": bl_b_count,
        "config_a_binary_value": bl_c_count,
        "match": False,
        "notes": f"Total honest nodes blacklisted (FP): Baseline={b_fp}, Config A={c_fp}"
    })

    # T_isolate Latency
    records.append({
        "metric_category": "responsiveness",
        "item": "tisolate_mean_cycles",
        "baseline_value": round(b_t_mean, 2),
        "config_a_binary_value": round(c_t_mean, 2),
        "match": False,
        "notes": "Mean cycles from initial penalty to blacklist threshold"
    })

    summary_df = pd.DataFrame(records)
    
    # Save CSV
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(out_csv, index=False)
    
    # Print Console Summary
    w = 78
    print("=" * w)
    print(f"FM-DAD STREAM PIPELINE ABLATION REPORT — Baseline vs. Config A (Binary)")
    print(f"  tau_min = {tau_min} | Output CSV: {out_csv}")
    print("=" * w)
    print(f"\n1. GATE-FIRE PARITY CHECK: {'✅ PASS' if parity_ok else '❌ FAIL'}")
    for a in ["sp", "als", "fs", "igh"]:
        b_val, c_val = gf_b.get(a, 0), gf_c.get(a, 0)
        print(f"  {a.upper():<6} Baseline={b_val:>5} | Config A={c_val:>5} | {'✅ Match' if b_val==c_val else '❌ Mismatch'}")

    print(f"\n2. MCC & CONFUSION MATRIX COMPARISON:")
    print(f"  {'Attack':<8} {'Baseline MCC':>14} {'Config A MCC':>14} {'Baseline TP/FP/FN':>22} {'Config A TP/FP/FN':>20}")
    print(f"  {'-'*76}")
    for at in sorted(b_mcc_dict.keys()):
        bv = b_mcc_dict[at]; cv = c_mcc_dict[at]
        b_tpfn = f"TP={bv['tp']} FP={bv['fp']} FN={bv['fn']}"
        c_tpfn = f"TP={cv['tp']} FP={cv['fp']} FN={cv['fn']}"
        print(f"  {at:<8} {bv['mcc']:>+14.4f} {cv['mcc']:>+14.4f}   {b_tpfn:<20} {c_tpfn}")
    print(f"  {'-'*76}")
    print(f"  {'Macro':<8} {b_macro:>+14.4f} {c_macro:>+14.4f}   FP={b_fp} (Baseline)       FP={c_fp} (Config A)")

    print(f"\n3. ISOLATION RESPONSIVENESS (T_isolate):")
    print(f"  Baseline (graded): {b_t_mean:.2f} cycles mean ({b_t_med:.1f} median)")
    print(f"  Config A (binary): {c_t_mean:.2f} cycles mean ({c_t_med:.1f} median)")

    print(f"\n[SUCCESS] Summary saved to: {out_csv}")
    print("=" * w)

    return summary_df


def main():
    p = argparse.ArgumentParser(description="Evaluate Stream Pipeline Ablation Metrics & Export CSV")
    p.add_argument("--results-dir", default=str(_FM_DAD_DIR / "data" / "stream_ablation"), help="Directory containing stream output CSVs")
    p.add_argument("--gt-dir",      default=str(_FM_DAD_DIR / "data" / "raw_csvs"), help="Directory containing node_attack_ground_truth_*.csv files")
    p.add_argument("--tau-min",     type=float, default=0.3, help="Blacklist trust threshold")
    p.add_argument("--out-csv",     default=str(_FM_DAD_DIR / "data" / "stream_ablation" / "stream_metrics_summary.csv"), help="Output path for metric summary CSV")
    args = p.parse_args()

    evaluate_stream(
        results_dir=Path(args.results_dir),
        gt_dir=Path(args.gt_dir),
        tau_min=args.tau_min,
        out_csv=Path(args.out_csv),
    )


if __name__ == "__main__":
    main()
