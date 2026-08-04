"""
evaluate_stream_metrics.py — Compute stream pipeline ablation metrics & export to CSV.

Reads streaming outputs from data/stream_ablation/ (or user specified dir):
  - pipeline_penalties_<mode>_<run_id>.csv
  - live_blacklist_<mode>_<run_id>.csv
  - live_trust_history_<mode>_<run_id>.csv

Supports C1 (rule_based vs graded) and C2 (binary vs graded) ablation evaluations.

Use --mode c1, --mode c2, or --mode all to select which report sections to display.
Use --run-id-baseline, --run-id-c1, --run-id-c2 to explicitly select run IDs and
avoid ambiguity when multiple result files exist for the same mode.

Calculates:
  - Nodes passed each gate (Gate Fired Count)
  - Nodes removed for each attack (TP: attackers blacklisted, FP: honest blacklisted)
  - Per-attack-type & Macro MCC (Equation 4.1)
  - Isolation latency T_isolate (mean, median)
  - Gate-fired → blacklisted consistency check (for binary/rule_based modes)

Usage:
    python3 evaluate_stream_metrics.py --mode c1 --run-id-baseline baseline --run-id-c1 c1
    python3 evaluate_stream_metrics.py --mode c2 --run-id-baseline baseline --run-id-c2 c2
    python3 evaluate_stream_metrics.py --mode all
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_FM_DAD_DIR = Path(__file__).parent
if str(_FM_DAD_DIR) not in sys.path:
    sys.path.insert(0, str(_FM_DAD_DIR))


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------

def _mcc(tp: int, fp: int, fn: int, tn: int) -> float:
    """Equation 4.1 MCC formula."""
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return (tp * tn - fp * fn) / denom if denom > 0 else 0.0


# ---------------------------------------------------------------------------
# Ground truth loader
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# FIX 1 — Unambiguous file finder
# ---------------------------------------------------------------------------

def _find_file(results_dir: Path, prefix: str, mode: str,
               run_id: Optional[str] = None) -> Optional[Path]:
    """Locate a result file for a given prefix and mode.

    If run_id is given, match it exactly (prefix_mode_runid.csv) — this is
    the only way to guarantee you're reading the run you think you're
    reading when multiple result files exist for the same mode.

    If run_id is None, require there to be exactly one candidate; if there
    are multiple, refuse to silently guess — list them and tell the user to
    disambiguate with --run-id-* or by cleaning up stale result files.
    """
    if run_id:
        exact = results_dir / f"{prefix}_{mode}_{run_id}.csv"
        return exact if exact.exists() else None

    candidates = sorted(results_dir.glob(f"{prefix}_{mode}_*.csv"))
    if not candidates:
        fallback = results_dir / f"{prefix}_{mode}.csv"
        return fallback if fallback.exists() else None
    if len(candidates) > 1:
        names = ", ".join(c.name for c in candidates)
        raise RuntimeError(
            f"\n[AMBIGUOUS] Multiple '{prefix}_{mode}_*.csv' files found in "
            f"{results_dir}:\n  {names}\n"
            f"Refusing to silently pick one — pass --run-id-baseline/--run-id-c1/--run-id-c2 "
            f"to select explicitly, or delete stale result files."
        )
    return candidates[0]


# ---------------------------------------------------------------------------
# FIX 2 — Gate-fired → blacklisted consistency check
# ---------------------------------------------------------------------------

def _normalize_gate_fired(pen_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize gate_fired column to Python bool, handling str 'True'/'False'
    and stripping duplicate header rows introduced by stale appended CSVs."""
    pen = pen_df[pen_df["cycle_id"].astype(str) != "cycle_id"].copy()
    pen["gate_fired"] = pen["gate_fired"].astype(str).str.strip().map(
        {"True": True, "False": False, "1": True, "0": False}
    ).fillna(False)
    return pen


def check_gate_blacklist_consistency(pen_df: pd.DataFrame, bl_df: pd.DataFrame,
                                     agent: str, tau_min: float, label: str) -> None:
    """For binary/rule_based modes: every node whose gate fired at least
    once should be blacklisted by the end of the run (delta=1.0 forces
    trust to 0 on first fire). Warn loudly if this doesn't hold — it means
    either the wrong file was loaded, or there's a real logic bug in how
    the forced delta is applied."""
    agent_rows = pen_df[pen_df["agent"].str.lower() == agent.lower()]
    ever_gated = set(agent_rows.loc[agent_rows["gate_fired"] == True, "node_id"].unique())

    bl = bl_df.copy()
    bl["blacklisted"] = bl["current_trust"] < tau_min
    ever_blacklisted = set(bl.loc[bl["blacklisted"], "node_id"].unique())

    missing = ever_gated - ever_blacklisted
    if missing:
        print(f"  [WARNING] {label}/{agent.upper()}: {len(missing)} node(s) had the gate fire "
              f"at least once but are NOT blacklisted by end of run: {sorted(missing)[:10]}"
              f"{'...' if len(missing) > 10 else ''}.\n"
              f"    For binary/rule_based modes this should NOT happen — check for a "
              f"stale/wrong file (use --run-id-*) or a logic bug in how delta=1.0 is applied.")
    else:
        print(f"  [OK] {label}/{agent.upper()}: all {len(ever_gated)} gated node(s) are blacklisted ✅")


# ---------------------------------------------------------------------------
# Metric computations
# ---------------------------------------------------------------------------

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
        results[at] = {"tp": tp, "fp": fp_count, "fn": fn, "tn": tn_count,
                       "mcc": mcc_val, "total_attackers": len(at_ids)}

    macro_mcc = float(np.mean([v["mcc"] for v in results.values()])) if results else 0.0
    return results, macro_mcc, fp_count


def compute_tisolate(hist_df: pd.DataFrame, gt: pd.DataFrame, tau_min: float) -> Tuple[float, float]:
    """Compute mean and median T_isolate latency."""
    if hist_df is None or hist_df.empty:
        return 0.0, 0.0
    attacker_ids = set(gt[gt["is_attacker"] == 1]["node_id"].tolist())
    att_h = hist_df[hist_df["node_id"].isin(attacker_ids)].copy()

    bl_events = att_h[att_h["trust_after"] < tau_min].groupby("node_id")["cycle_id"].min()
    first_pen  = att_h[att_h["delta_applied"] > 0].groupby("node_id")["cycle_id"].min()

    latencies = []
    for nid, bl_cycle in bl_events.items():
        fp_cycle = first_pen.get(nid, bl_cycle)
        latencies.append(int(bl_cycle - fp_cycle))

    if not latencies:
        return 0.0, 0.0

    arr = np.array(latencies)
    return float(arr.mean()), float(np.median(arr))


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------

def evaluate_stream(
    results_dir: Path,
    gt_dir: Path,
    tau_min: float,
    out_csv: Path,
    mode_filter: str = "all",
    run_id_baseline: Optional[str] = None,
    run_id_c1: Optional[str] = None,
    run_id_c2: Optional[str] = None,
) -> pd.DataFrame:
    """Run stream evaluation, print expanded report, and save summary metrics to CSV."""
    gt = _load_ground_truth(gt_dir)

    # Locate result CSVs using FIX 1 — unambiguous file finder.
    # Only resolve files for the mode(s) being evaluated — this prevents
    # ambiguity errors for modes that aren't being requested.
    pen_b_file  = _find_file(results_dir, "pipeline_penalties", "baseline", run_id_baseline)
    bl_b_file   = _find_file(results_dir, "live_blacklist",      "baseline", run_id_baseline)
    hist_b_file = _find_file(results_dir, "live_trust_history",  "baseline", run_id_baseline)

    if mode_filter in ["all", "c2"]:
        pen_c2_file  = _find_file(results_dir, "pipeline_penalties", "binary", run_id_c2)
        bl_c2_file   = _find_file(results_dir, "live_blacklist",      "binary", run_id_c2)
        hist_c2_file = _find_file(results_dir, "live_trust_history",  "binary", run_id_c2)
    else:
        pen_c2_file = bl_c2_file = hist_c2_file = None

    if mode_filter in ["all", "c1"]:
        pen_c1_file  = _find_file(results_dir, "pipeline_penalties", "rule_based", run_id_c1)
        bl_c1_file   = _find_file(results_dir, "live_blacklist",      "rule_based", run_id_c1)
        hist_c1_file = _find_file(results_dir, "live_trust_history",  "rule_based", run_id_c1)
    else:
        pen_c1_file = bl_c1_file = hist_c1_file = None

    # Baseline files (optional if evaluating single ablation runs)
    b_available = (pen_b_file and bl_b_file and pen_b_file.exists() and bl_b_file.exists())

    if b_available:
        bl_b   = pd.read_csv(bl_b_file)
        pen_b  = pd.read_csv(pen_b_file)
        hist_b = pd.read_csv(hist_b_file) if hist_b_file and hist_b_file.exists() else None
        gf_b = pen_b[pen_b["gate_fired"] == True].groupby("agent").size().to_dict()
        b_mcc_dict, b_macro, b_fp = compute_mcc_metrics(bl_b, gt, tau_min)
        b_t_mean, b_t_med = compute_tisolate(hist_b, gt, tau_min)
    else:
        print("[NOTE] Baseline files not found — running standalone ablation evaluation.")
        gf_b = {}
        b_macro = 0.0
        b_fp = "N/A"
        b_t_mean = b_t_med = 0.0
        attack_types = sorted(gt[gt["is_attacker"] == 1]["attack_type"].unique())
        b_mcc_dict = {
            at: {
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "tn": 0,
                "mcc": 0.0,
                "total_attackers": len(gt[gt["attack_type"] == at]["node_id"].unique()),
            }
            for at in attack_types
        }

    records = []

    # C2 Binary
    c2_available = (pen_c2_file and bl_c2_file and
                    pen_c2_file.exists() and bl_c2_file.exists() and
                    mode_filter in ["all", "c2"])
    if c2_available:
        pen_c2  = _normalize_gate_fired(pd.read_csv(pen_c2_file, low_memory=False))
        bl_c2   = pd.read_csv(bl_c2_file)
        hist_c2 = pd.read_csv(hist_c2_file) if hist_c2_file and hist_c2_file.exists() else None
        gf_c2   = pen_c2[pen_c2["gate_fired"] == True].groupby("agent").size().to_dict()
        c2_mcc_dict, c2_macro, c2_fp = compute_mcc_metrics(bl_c2, gt, tau_min)
        c2_t_mean, c2_t_med = compute_tisolate(hist_c2, gt, tau_min)

        print(f"\n[CONSISTENCY CHECK] C2/binary (file: {pen_c2_file.name})")
        for agent in ["sp", "als", "igh", "fs"]:
            check_gate_blacklist_consistency(pen_c2, bl_c2, agent, tau_min, "C2/binary")

        for at in sorted(c2_mcc_dict.keys()):
            ag = at.lower()
            bv = b_mcc_dict.get(at, {"tp": 0, "mcc": 0.0, "total_attackers": 0})
            cv = c2_mcc_dict[at]
            records.append({
                "ablation_study": "C2_binary",
                "attack_type": at,
                "gate_fired_baseline": gf_b.get(ag, "N/A"),
                "gate_fired_ablation": gf_c2.get(ag, 0),
                "removed_attackers_tp_baseline": bv["tp"] if b_available else "N/A",
                "removed_attackers_tp_ablation": cv["tp"],
                "total_attackers": cv["total_attackers"],
                "removed_honest_fp_baseline": b_fp,
                "removed_honest_fp_ablation": c2_fp,
                "mcc_baseline": round(bv["mcc"], 4) if b_available else "N/A",
                "mcc_ablation": round(cv["mcc"], 4),
            })

    # C1 Rule-Based
    c1_available = (pen_c1_file and bl_c1_file and
                    pen_c1_file.exists() and bl_c1_file.exists() and
                    mode_filter in ["all", "c1"])
    if c1_available:
        pen_c1  = _normalize_gate_fired(pd.read_csv(pen_c1_file, low_memory=False))
        bl_c1   = pd.read_csv(bl_c1_file)
        hist_c1 = pd.read_csv(hist_c1_file) if hist_c1_file and hist_c1_file.exists() else None
        gf_c1   = pen_c1[pen_c1["gate_fired"] == True].groupby("agent").size().to_dict()
        c1_mcc_dict, c1_macro, c1_fp = compute_mcc_metrics(bl_c1, gt, tau_min)
        c1_t_mean, c1_t_med = compute_tisolate(hist_c1, gt, tau_min)

        print(f"\n[CONSISTENCY CHECK] C1/rule_based (file: {pen_c1_file.name})")
        for agent in ["sp", "als", "igh", "fs"]:
            check_gate_blacklist_consistency(pen_c1, bl_c1, agent, tau_min, "C1/rule_based")

        for at in sorted(c1_mcc_dict.keys()):
            ag = at.lower()
            bv = b_mcc_dict.get(at, {"tp": 0, "mcc": 0.0, "total_attackers": 0})
            cv = c1_mcc_dict[at]
            records.append({
                "ablation_study": "C1_rule_based",
                "attack_type": at,
                "gate_fired_baseline": gf_b.get(ag, "N/A"),
                "gate_fired_ablation": gf_c1.get(ag, 0),
                "removed_attackers_tp_baseline": bv["tp"] if b_available else "N/A",
                "removed_attackers_tp_ablation": cv["tp"],
                "total_attackers": cv["total_attackers"],
                "removed_honest_fp_baseline": b_fp,
                "removed_honest_fp_ablation": c1_fp,
                "mcc_baseline": round(bv["mcc"], 4) if b_available else "N/A",
                "mcc_ablation": round(cv["mcc"], 4),
            })

    summary_df = pd.DataFrame(records)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(out_csv, index=False)

    w = 96
    print("\n" + "=" * w)
    print(f"FM-DAD STREAM PIPELINE ABLATION REPORT")
    print(f"  tau_min = {tau_min} | Mode Filter = {mode_filter.upper()} | Output CSV: {out_csv}")
    print("=" * w)

    if c2_available:
        print(f"\n--- C2 ABLATION SUMMARY: Baseline (Graded) vs. Config A (Binary) ---")
        print(f"  Source: {pen_c2_file.name}")
        if b_available:
            print(f"  {'Attack':<8} {'Passed Gate (b/c2)':>20} {'Removed Attackers (b/c2)':>26} {'Removed Honest (b/c2)':>22} {'Baseline MCC':>14} {'Binary MCC':>12}")
            print(f"  {'-'*104}")
            for at in sorted(c2_mcc_dict.keys()):
                ag = at.lower(); bv = b_mcc_dict.get(at, {"tp":0, "mcc":0.0}); cv = c2_mcc_dict[at]
                gf_str  = f"{gf_b.get(ag,0)} / {gf_c2.get(ag,0)}"
                att_str = f"{bv['tp']} / {cv['tp']} (of {bv['total_attackers']})"
                fp_str  = f"{b_fp} / {c2_fp}"
                print(f"  {at:<8} {gf_str:>20} {att_str:>26} {fp_str:>22} {bv['mcc']:>+14.4f} {cv['mcc']:>+12.4f}")
            print(f"  {'-'*104}")
            print(f"  {'Macro':<8} {'—':>20} {'—':>26} {b_fp} / {c2_fp:>20} {b_macro:>+14.4f} {c2_macro:>+12.4f}")
        else:
            print(f"  {'Attack':<8} {'Passed Gate':>15} {'Removed Attackers (TP)':>26} {'Removed Honest (FP)':>22} {'Binary MCC':>14}")
            print(f"  {'-'*88}")
            for at in sorted(c2_mcc_dict.keys()):
                ag = at.lower(); cv = c2_mcc_dict[at]
                att_str = f"{cv['tp']} (of {cv['total_attackers']})"
                print(f"  {at:<8} {gf_c2.get(ag,0):>15} {att_str:>26} {c2_fp:>22} {cv['mcc']:>+14.4f}")
            print(f"  {'-'*88}")
            print(f"  {'Macro':<8} {'—':>15} {'—':>26} {c2_fp:>22} {c2_macro:>+14.4f}")

    if c1_available:
        print(f"\n--- C1 ABLATION SUMMARY: Baseline (DRL-Graded) vs. Rule-Based (LW-MAD) ---")
        print(f"  Source: {pen_c1_file.name}")
        if b_available:
            print(f"  {'Attack':<8} {'Passed Gate (b/c1)':>20} {'Removed Attackers (b/c1)':>26} {'Removed Honest (b/c1)':>22} {'Baseline MCC':>14} {'Rule-Based MCC':>16}")
            print(f"  {'-'*108}")
            for at in sorted(c1_mcc_dict.keys()):
                ag = at.lower(); bv = b_mcc_dict.get(at, {"tp":0, "mcc":0.0}); cv = c1_mcc_dict[at]
                gf_str  = f"{gf_b.get(ag,0)} / {gf_c1.get(ag,0)}"
                att_str = f"{bv['tp']} / {cv['tp']} (of {bv['total_attackers']})"
                fp_str  = f"{b_fp} / {c1_fp}"
                print(f"  {at:<8} {gf_str:>20} {att_str:>26} {fp_str:>22} {bv['mcc']:>+14.4f} {cv['mcc']:>+16.4f}")
            print(f"  {'-'*108}")
            print(f"  {'Macro':<8} {'—':>20} {'—':>26} {b_fp} / {c1_fp:>20} {b_macro:>+14.4f} {c1_macro:>+16.4f}")
            print(f"  T_isolate : Baseline={b_t_mean:.2f} cycles mean | Rule-Based={c1_t_mean:.2f} cycles mean")
        else:
            print(f"  {'Attack':<8} {'Passed Gate':>15} {'Removed Attackers (TP)':>26} {'Removed Honest (FP)':>22} {'Rule-Based MCC':>16}")
            print(f"  {'-'*90}")
            for at in sorted(c1_mcc_dict.keys()):
                ag = at.lower(); cv = c1_mcc_dict[at]
                att_str = f"{cv['tp']} (of {cv['total_attackers']})"
                print(f"  {at:<8} {gf_c1.get(ag,0):>15} {att_str:>26} {c1_fp:>22} {cv['mcc']:>+16.4f}")
            print(f"  {'-'*90}")
            print(f"  {'Macro':<8} {'—':>15} {'—':>26} {c1_fp:>22} {c1_macro:>+16.4f}")
            print(f"  T_isolate : Rule-Based={c1_t_mean:.2f} cycles mean")

    print(f"\n[SUCCESS] Detailed summary saved to: {out_csv}")
    print("=" * w)
    return summary_df

    print(f"\n[SUCCESS] Detailed summary saved to: {out_csv}")
    print("=" * w)
    return summary_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Evaluate Stream Pipeline Ablation Metrics & Export Detailed CSV")
    p.add_argument("--results-dir",       default=str(_FM_DAD_DIR / "data" / "stream_ablation"),
                   help="Directory containing stream output CSVs")
    p.add_argument("--gt-dir",            default=str(_FM_DAD_DIR / "data" / "raw_csvs"),
                   help="Directory containing node_attack_ground_truth_*.csv files")
    p.add_argument("--tau-min",           type=float, default=0.3,
                   help="Blacklist trust threshold")
    p.add_argument("--mode",              choices=["c1", "c2", "all"], default="all",
                   help="Ablation study to report: 'c1', 'c2', or 'all'")
    p.add_argument("--run-id-baseline",   default=None,
                   help="Exact run_id for baseline files (e.g. 'baseline'). "
                        "Required when multiple baseline files exist.")
    p.add_argument("--run-id-c1",         default=None,
                   help="Exact run_id for C1/rule_based files (e.g. 'c1'). "
                        "Required when multiple rule_based files exist.")
    p.add_argument("--run-id-c2",         default=None,
                   help="Exact run_id for C2/binary files (e.g. 'c2'). "
                        "Required when multiple binary files exist.")
    p.add_argument("--out-csv",           default=str(_FM_DAD_DIR / "data" / "stream_ablation" / "stream_metrics_summary.csv"),
                   help="Output path for metric summary CSV")
    args = p.parse_args()

    evaluate_stream(
        results_dir=Path(args.results_dir),
        gt_dir=Path(args.gt_dir),
        tau_min=args.tau_min,
        out_csv=Path(args.out_csv),
        mode_filter=args.mode,
        run_id_baseline=args.run_id_baseline,
        run_id_c1=args.run_id_c1,
        run_id_c2=args.run_id_c2,
    )


if __name__ == "__main__":
    main()
