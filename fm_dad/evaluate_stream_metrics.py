"""
evaluate_stream_metrics.py — Compute stream pipeline ablation metrics & export to CSV.

Reads streaming outputs from data/stream_ablation/ (or user specified dir):
  - pipeline_penalties_<mode>_<run_id>.csv
  - live_blacklist_<mode>_<run_id>.csv
  - live_trust_history_<mode>_<run_id>.csv

Supports C1 (rule_based vs graded) and C2 (binary vs graded) ablation evaluations.
FS rule-based detection is marked N/A (Not Evaluable) per Section 1 (missing h_bc/h_obs metrics).

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
from typing import Dict, List, Tuple, Optional

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
    if hist_df is None or hist_df.empty:
        return 0.0, 0.0
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


def _find_file(results_dir: Path, prefix: str, mode: str) -> Optional[Path]:
    """Locate a result file for a given prefix and mode."""
    candidates = sorted(results_dir.glob(f"{prefix}_{mode}_*.csv"))
    if not candidates:
        # Try exact fallback
        fallback = results_dir / f"{prefix}_{mode}.csv"
        return fallback if fallback.exists() else None
    return candidates[0]


def evaluate_stream(results_dir: Path, gt_dir: Path, tau_min: float, out_csv: Path) -> pd.DataFrame:
    """Run stream evaluation for C1/C2, print report, and save summary metrics to CSV."""
    gt = _load_ground_truth(gt_dir)

    # Baseline files
    pen_b_file = _find_file(results_dir, "pipeline_penalties", "baseline")
    bl_b_file  = _find_file(results_dir, "live_blacklist", "baseline")
    hist_b_file = _find_file(results_dir, "live_trust_history", "baseline")

    # C2 (binary) files
    pen_c2_file = _find_file(results_dir, "pipeline_penalties", "binary")
    bl_c2_file  = _find_file(results_dir, "live_blacklist", "binary")
    hist_c2_file = _find_file(results_dir, "live_trust_history", "binary")

    # C1 (rule_based) files
    pen_c1_file = _find_file(results_dir, "pipeline_penalties", "rule_based")
    bl_c1_file  = _find_file(results_dir, "live_blacklist", "rule_based")
    hist_c1_file = _find_file(results_dir, "live_trust_history", "rule_based")

    if not pen_b_file or not bl_b_file:
        raise FileNotFoundError(f"Baseline files missing in {results_dir}")

    bl_b  = pd.read_csv(bl_b_file)
    pen_b = pd.read_csv(pen_b_file)
    hist_b = pd.read_csv(hist_b_file) if hist_b_file and hist_b_file.exists() else None

    gf_b = pen_b[pen_b["gate_fired"] == True].groupby("agent").size().to_dict()
    b_mcc_dict, b_macro, b_fp = compute_mcc_metrics(bl_b, gt, tau_min)
    b_t_mean, b_t_med = compute_tisolate(hist_b, gt, tau_min)

    records = []

    # Process C2 if available
    c2_available = pen_c2_file and bl_c2_file and pen_c2_file.exists() and bl_c2_file.exists()
    if c2_available:
        pen_c2  = pd.read_csv(pen_c2_file)
        bl_c2   = pd.read_csv(bl_c2_file)
        hist_c2 = pd.read_csv(hist_c2_file) if hist_c2_file and hist_c2_file.exists() else None
        gf_c2   = pen_c2[pen_c2["gate_fired"] == True].groupby("agent").size().to_dict()
        c2_mcc_dict, c2_macro, c2_fp = compute_mcc_metrics(bl_c2, gt, tau_min)
        c2_t_mean, c2_t_med = compute_tisolate(hist_c2, gt, tau_min)

        for agent in ["sp", "als", "fs", "igh"]:
            records.append({
                "ablation_study": "C2_binary",
                "metric_category": "gate_fire_count",
                "item": agent.upper(),
                "baseline_value": gf_b.get(agent, 0),
                "ablation_value": gf_c2.get(agent, 0),
                "match": gf_b.get(agent, 0) == gf_c2.get(agent, 0),
                "notes": "Gate fire parity check (C2 binary)"
            })
        for at in sorted(b_mcc_dict.keys()):
            bv = b_mcc_dict[at]; cv = c2_mcc_dict[at]
            records.append({
                "ablation_study": "C2_binary",
                "metric_category": "mcc_score",
                "item": f"{at}_mcc",
                "baseline_value": round(bv["mcc"], 4),
                "ablation_value": round(cv["mcc"], 4),
                "match": False,
                "notes": f"TP_b={bv['tp']} FP_b={bv['fp']} FN_b={bv['fn']} | TP_c2={cv['tp']} FP_c2={cv['fp']} FN_c2={cv['fn']}"
            })

    # Process C1 if available
    c1_available = pen_c1_file and bl_c1_file and pen_c1_file.exists() and bl_c1_file.exists()
    if c1_available:
        pen_c1  = pd.read_csv(pen_c1_file)
        bl_c1   = pd.read_csv(bl_c1_file)
        hist_c1 = pd.read_csv(hist_c1_file) if hist_c1_file and hist_c1_file.exists() else None
        gf_c1   = pen_c1[pen_c1["gate_fired"] == True].groupby("agent").size().to_dict()
        c1_mcc_dict, c1_macro, c1_fp = compute_mcc_metrics(bl_c1, gt, tau_min)
        c1_t_mean, c1_t_med = compute_tisolate(hist_c1, gt, tau_min)

        for at in sorted(b_mcc_dict.keys()):
            bv = b_mcc_dict[at]; cv = c1_mcc_dict.get(at, {"mcc": 0.0, "tp": 0, "fp": c1_fp, "fn": 0})
            is_evaluable = (at != "FS")
            records.append({
                "ablation_study": "C1_rule_based",
                "metric_category": "mcc_score",
                "item": f"{at}_mcc",
                "baseline_value": round(bv["mcc"], 4),
                "ablation_value": round(cv["mcc"], 4) if is_evaluable else "N/A",
                "match": False,
                "notes": f"LW-MAD rule detection. FS is N/A (Section 1: missing h_bc/h_obs hop data)." if not is_evaluable else f"TP_b={bv['tp']} FP_b={bv['fp']} | TP_c1={cv['tp']} FP_c1={cv['fp']}"
            })
        records.append({
            "ablation_study": "C1_rule_based",
            "metric_category": "mcc_score",
            "item": "macro_mcc",
            "baseline_value": round(b_macro, 4),
            "ablation_value": round(c1_macro, 4),
            "match": False,
            "notes": f"LW-MAD Rule-based Macro MCC. Baseline FP={b_fp} | Rule-based FP={c1_fp}"
        })

    summary_df = pd.DataFrame(records)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(out_csv, index=False)

    w = 78
    print("=" * w)
    print(f"FM-DAD STREAM PIPELINE ABLATION REPORT")
    print(f"  tau_min = {tau_min} | Output CSV: {out_csv}")
    print("=" * w)

    if c2_available:
        print(f"\n--- C2 ABLATION SUMMARY: Baseline (Graded) vs. Config A (Binary) ---")
        print(f"  Gate Parity: {'✅ PASS' if all(gf_b.get(a, -1) == gf_c2.get(a, -2) for a in ['sp','als','fs','igh']) else '❌ FAIL'}")
        print(f"  Macro MCC  : Baseline={b_macro:+.4f} | Binary={c2_macro:+.4f} | FP: Baseline={b_fp}, Binary={c2_fp}")

    if c1_available:
        print(f"\n--- C1 ABLATION SUMMARY: Baseline (DRL-Graded) vs. Rule-Based (LW-MAD) ---")
        print(f"  {'Attack':<8} {'Baseline (DRL) MCC':>20} {'Rule-Based (LW-MAD) MCC':>25} {'Evaluability':>15}")
        print(f"  {'-'*74}")
        for at in sorted(b_mcc_dict.keys()):
            bv = b_mcc_dict[at]
            if at == "FS":
                print(f"  {at:<8} {bv['mcc']:>+20.4f} {'N/A':>25} {'NOT EVALUABLE (missing h_bc/h_obs)':>35}")
            else:
                cv = c1_mcc_dict.get(at, {"mcc": 0.0})
                print(f"  {at:<8} {bv['mcc']:>+20.4f} {cv['mcc']:>+25.4f} {'Evaluable ✅':>25}")
        print(f"  {'-'*74}")
        print(f"  {'Macro':<8} {b_macro:>+20.4f} {c1_macro:>+25.4f}   FP: Baseline={b_fp}, Rule-Based={c1_fp}")
        print(f"  T_isolate : Baseline={b_t_mean:.2f} cycles mean | Rule-Based={c1_t_mean:.2f} cycles mean")

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
