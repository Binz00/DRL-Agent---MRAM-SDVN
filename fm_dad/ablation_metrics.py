"""
ablation_metrics.py — Compute and report C2 ablation metrics from run_ablation.py output.

Reads:
  data/results/baseline_taumin<X>_<run_id>.csv  + .manifest.json
  data/results/c2_taumin<X>_<run_id>.csv        + .manifest.json

Computes and reports side-by-side:
  1. MCC (macro + per-type) — reuses validate_pipeline.py ground-truth logic
  2. Detection latency — cycles from attack activation to first blacklist
  3. False-positive rate for high-mobility honest nodes (segmented by lambda_t_norm)
  4. Trust calibration responsiveness — cycles-to-blacklist distribution
  5. Gate-fire count parity check (Section 4, Check 1 of ablation spec)

Usage:
    python3 ablation_metrics.py \\
        --baseline data/results/baseline_taumin03_<run_id>.csv \\
        --c2       data/results/c2_taumin03_<run_id>.csv \\
        --gt-dir   data/raw_csvs \\
        --tau-min  0.3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_FM_DAD_DIR = Path(__file__).parent
if str(_FM_DAD_DIR) not in sys.path:
    sys.path.insert(0, str(_FM_DAD_DIR))


# ---------------------------------------------------------------------------
# Ground-truth loading (same logic as validate_pipeline.py)
# ---------------------------------------------------------------------------

def _load_gt(gt_dir: Path) -> pd.DataFrame:
    """Union attackers across ALL cycle GT files (IGH rotates per cycle)."""
    import re
    gt_re = re.compile(r"node_attack_ground_truth_(\d+)\.csv$")
    files = sorted(f for f in gt_dir.iterdir() if gt_re.match(f.name))
    if not files:
        print(f"[ERROR] No ground truth files found in {gt_dir}")
        sys.exit(1)

    frames = []
    for f in files:
        df = pd.read_csv(f)
        df.columns = df.columns.str.strip()
        frames.append(df)

    gt_all = pd.concat(frames, ignore_index=True)

    def _resolve(g):
        """For a given node_id group, attacker wins over honest."""
        if (g["is_attacker"] == 1).any():
            row = g[g["is_attacker"] == 1].iloc[0]
            return pd.Series({"is_attacker": 1, "attack_type": row["attack_type"]})
        return pd.Series({"is_attacker": 0, "attack_type": "NONE"})

    gt_base = (
        gt_all.groupby("node_id", group_keys=False)
        .apply(_resolve)
        .reset_index()
    )
    return gt_base


# ---------------------------------------------------------------------------
# MCC computation (from episode_eval.py pattern)
# ---------------------------------------------------------------------------

def _mcc(tp: int, fp: int, fn: int, tn: int) -> float:
    denom = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    if denom == 0:
        return 0.0
    return (tp * tn - fp * fn) / denom


def compute_mcc_table(
    results_df: pd.DataFrame,
    gt: pd.DataFrame,
    tau_min: float,
) -> Tuple[dict, float, int]:
    """
    Compute per-attack-type MCC and macro MCC.
    Returns (per_type_mcc, macro_mcc, fp_count).
    """
    # Final trust per node: last trust_after seen
    final_trust = (
        results_df.sort_values("cycle_id")
        .groupby("node_id")["trust_after"]
        .last()
        .reset_index()
    )
    final_trust["blacklisted"] = final_trust["trust_after"] < tau_min

    merged = final_trust.merge(gt, on="node_id", how="left")
    merged["is_attacker"] = merged["is_attacker"].fillna(0).astype(int)
    merged["attack_type"]  = merged["attack_type"].fillna("NONE")

    total_honest = (merged["is_attacker"] == 0).sum()
    fp_count     = int(((merged["is_attacker"] == 0) & merged["blacklisted"]).sum())
    tn_count     = total_honest - fp_count

    attack_types = sorted(merged[merged["is_attacker"] == 1]["attack_type"].unique())
    per_type_mcc = {}

    for at in attack_types:
        at_nodes  = merged[merged["attack_type"] == at]["node_id"].values
        tp = int(merged[merged["node_id"].isin(at_nodes) &  merged["blacklisted"]].shape[0])
        fn = int(merged[merged["node_id"].isin(at_nodes) & ~merged["blacklisted"]].shape[0])
        mcc = _mcc(tp, fp_count, fn, tn_count)
        per_type_mcc[at] = {"tp": tp, "fp": fp_count, "fn": fn, "tn": tn_count, "mcc": mcc}

    macro = float(np.mean([v["mcc"] for v in per_type_mcc.values()])) if per_type_mcc else 0.0
    return per_type_mcc, macro, fp_count


# ---------------------------------------------------------------------------
# Detection latency
# ---------------------------------------------------------------------------

def compute_detection_latency(
    results_df: pd.DataFrame,
    gt: pd.DataFrame,
    tau_min: float,
) -> Dict[str, list]:
    """
    For each attacker node, compute the first cycle it was blacklisted (trust < tau_min).
    Returns {node_id: first_blacklist_cycle} for attackers, plus summary stats.
    """
    attackers = set(gt[gt["is_attacker"] == 1]["node_id"].tolist())
    attacker_rows = results_df[
        (results_df["node_id"].isin(attackers)) & results_df["blacklisted"]
    ]
    first_bl = (
        attacker_rows.groupby("node_id")["cycle_id"].min().reset_index()
        .rename(columns={"cycle_id": "first_blacklist_cycle"})
    )
    return first_bl


# ---------------------------------------------------------------------------
# False-positive rate by mobility tier
# ---------------------------------------------------------------------------

def compute_fp_by_mobility(
    results_df: pd.DataFrame,
    gt: pd.DataFrame,
    tau_min: float,
    high_mobility_threshold: float = 0.5,
) -> dict:
    """
    Split honest nodes by lambda_t_norm (mobility proxy) into high/low mobility tiers.
    Compare false-positive blacklisting rates between tiers.
    lambda_t_norm is read from the ALS state table (available for all nodes).
    """
    honest_ids = set(gt[gt["is_attacker"] == 0]["node_id"].tolist())

    # Use last lambda_t_norm seen per node from results
    lambda_vals = (
        results_df[results_df["agent"] == "als"]
        .sort_values("cycle_id")
        .groupby("node_id")[["trust_after", "blacklisted"]]
        .last()
        .reset_index()
    )

    # Get lambda_t_norm from the als_state.csv directly if available
    als_state_path = _FM_DAD_DIR / "data" / "agent_inputs" / "als_state.csv"
    if als_state_path.exists():
        als_df = pd.read_csv(als_state_path)
        if "lambda_t_norm" in als_df.columns:
            median_lambda = als_df.groupby("node_id")["lambda_t_norm"].median().reset_index()
            lambda_vals = lambda_vals.merge(median_lambda, on="node_id", how="left")
        else:
            lambda_vals["lambda_t_norm"] = np.nan
    else:
        lambda_vals["lambda_t_norm"] = np.nan

    # Filter to honest nodes only
    honest_rows = lambda_vals[lambda_vals["node_id"].isin(honest_ids)].copy()

    high_mob = honest_rows[honest_rows["lambda_t_norm"] >= high_mobility_threshold]
    low_mob  = honest_rows[honest_rows["lambda_t_norm"] <  high_mobility_threshold]

    def _fp_rate(subset):
        if len(subset) == 0:
            return 0.0, 0, 0
        bl   = subset["blacklisted"].sum()
        rate = bl / len(subset)
        return float(rate), int(bl), len(subset)

    high_rate, high_bl, high_n = _fp_rate(high_mob)
    low_rate,  low_bl,  low_n  = _fp_rate(low_mob)

    return {
        "high_mobility": {
            "threshold":   high_mobility_threshold,
            "n_nodes":     high_n,
            "fp_count":    high_bl,
            "fp_rate":     round(high_rate, 4),
        },
        "low_medium_mobility": {
            "threshold":   high_mobility_threshold,
            "n_nodes":     low_n,
            "fp_count":    low_bl,
            "fp_rate":     round(low_rate, 4),
        },
    }


# ---------------------------------------------------------------------------
# Trust calibration responsiveness
# ---------------------------------------------------------------------------

def compute_trust_responsiveness(
    results_df: pd.DataFrame,
    gt: pd.DataFrame,
    tau_min: float,
) -> dict:
    """
    For attacking nodes, measure how many cycles they spent at intermediate trust
    (tau_min < trust < 1.0) before being blacklisted.
    Binary mode -> cliff-edge drop (0 or 1 cycle).
    Graded mode -> gradual descent (multiple cycles).
    """
    attackers = set(gt[gt["is_attacker"] == 1]["node_id"].tolist())
    att_rows  = results_df[results_df["node_id"].isin(attackers)].copy()

    # Unique attacker rows (just trust_after per cycle, one row per node per cycle via final_delta)
    trust_traj = (
        att_rows[att_rows["agent"] == "sp"]  # any agent gives the same trust_after
        .groupby(["node_id", "cycle_id"])["trust_after"]
        .first()
        .reset_index()
    )

    intermediate_counts = []
    for nid, grp in trust_traj.groupby("node_id"):
        grp = grp.sort_values("cycle_id")
        intermediate = ((grp["trust_after"] < 1.0) & (grp["trust_after"] >= tau_min)).sum()
        intermediate_counts.append(int(intermediate))

    if not intermediate_counts:
        return {"mean": None, "median": None, "p25": None, "p75": None}

    arr = np.array(intermediate_counts)
    return {
        "mean":   round(float(arr.mean()), 2),
        "median": round(float(np.median(arr)), 2),
        "p25":    round(float(np.percentile(arr, 25)), 2),
        "p75":    round(float(np.percentile(arr, 75)), 2),
        "n_attackers_evaluated": len(arr),
    }


# ---------------------------------------------------------------------------
# Gate-fire parity check
# ---------------------------------------------------------------------------

def check_gate_parity(baseline_mfst: dict, c2_mfst: dict) -> bool:
    """
    Section 4, Check 1: gate_fire_counts must be identical between graded and binary runs.
    Returns True if they match, False otherwise.
    """
    b_gc = baseline_mfst.get("gate_fire_counts", {})
    c_gc = c2_mfst.get("gate_fire_counts", {})
    match = all(b_gc.get(a, -1) == c_gc.get(a, -2) for a in ["sp", "als", "fs", "igh"])
    return match


# ---------------------------------------------------------------------------
# Report printer
# ---------------------------------------------------------------------------

def print_report(
    baseline_df:  pd.DataFrame,
    c2_df:        pd.DataFrame,
    baseline_mfst: dict,
    c2_mfst:      dict,
    gt:           pd.DataFrame,
    tau_min:      float,
) -> None:
    w = 72

    print("=" * w)
    print("FM-DAD C2 ABLATION REPORT — Graded Trust vs. Binary Blacklist")
    print(f"  tau_min       = {tau_min}")
    print(f"  n_cycles      = {baseline_mfst['n_cycles_found']}")
    print(f"  Dataset note  : {baseline_mfst.get('attacker_density_note', 'N/A')}")
    print(f"  git_commit    : {baseline_mfst.get('git_commit','?')} (baseline) "
          f"| {c2_mfst.get('git_commit','?')} (c2)")
    print("=" * w)

    # --- Check 1: Gate parity ---
    parity_ok = check_gate_parity(baseline_mfst, c2_mfst)
    b_gc  = baseline_mfst.get("gate_fire_counts", {})
    c2_gc = c2_mfst.get("gate_fire_counts", {})
    print("\nSECTION 4 CHECK 1 — Gate-fire count parity (MUST MATCH before trusting metrics)")
    print(f"  {'Agent':<8} {'Baseline':>10} {'C2':>10} {'Match?':>8}")
    print(f"  {'-'*40}")
    for a in ["sp", "als", "fs", "igh"]:
        bv, cv = b_gc.get(a, "?"), c2_gc.get(a, "?")
        ok = "✅" if bv == cv else "❌"
        print(f"  {a.upper():<8} {str(bv):>10} {str(cv):>10} {ok:>8}")
    print(f"\n  → Parity: {'✅ PASS — safe to trust downstream metrics' if parity_ok else '❌ FAIL — STOP: investigate before interpreting metrics'}")

    if not parity_ok:
        print("\n[ABORT] Gate-fire counts differ. Ablation is contaminated. Stopping report.")
        return

    # --- MCC ---
    print("\n" + "=" * w)
    print("METRIC 1 — MCC (macro + per-attack-type)")
    print(f"  {'Attack':<10} {'Baseline MCC':>14} {'C2 MCC':>10} {'Baseline TP/FP/FN':>22} {'C2 TP/FP/FN':>16}")
    print(f"  {'-'*72}")

    b_types, b_macro, b_fp = compute_mcc_table(baseline_df, gt, tau_min)
    c_types, c_macro, c_fp = compute_mcc_table(c2_df,       gt, tau_min)

    all_types = sorted(set(list(b_types.keys()) + list(c_types.keys())))
    for at in all_types:
        bv = b_types.get(at, {})
        cv = c_types.get(at, {})
        b_mcc  = f"{bv.get('mcc', 0):+.4f}" if bv else "  N/A  "
        c_mcc  = f"{cv.get('mcc', 0):+.4f}" if cv else "  N/A  "
        b_tpfn = f"TP={bv.get('tp','?')} FP={bv.get('fp','?')} FN={bv.get('fn','?')}" if bv else ""
        c_tpfn = f"TP={cv.get('tp','?')} FP={cv.get('fp','?')} FN={cv.get('fn','?')}" if cv else ""
        print(f"  {at:<10} {b_mcc:>14} {c_mcc:>10}   {b_tpfn:<20} {c_tpfn}")

    print(f"  {'':->72}")
    print(f"  {'Macro MCC':<10} {b_macro:>+14.4f} {c_macro:>+10.4f}   FP={b_fp} (baseline) | FP={c_fp} (c2)")

    # --- Detection latency ---
    print("\n" + "=" * w)
    print("METRIC 3 — Detection latency (first blacklist cycle for attackers)")
    b_lat = compute_detection_latency(baseline_df, gt, tau_min)
    c_lat = compute_detection_latency(c2_df,       gt, tau_min)

    for label, lat in [("Baseline", b_lat), ("C2 Binary", c_lat)]:
        if lat.empty:
            print(f"  {label}: no attackers blacklisted")
        else:
            print(f"  {label}: mean={lat['first_blacklist_cycle'].mean():.1f}  "
                  f"median={lat['first_blacklist_cycle'].median():.1f}  "
                  f"min={lat['first_blacklist_cycle'].min()}  "
                  f"n_blacklisted={len(lat)}")

    # --- FP by mobility ---
    print("\n" + "=" * w)
    print("METRIC 4 — False-positive blacklisting of honest nodes (by mobility tier)")
    for label, df_ in [("Baseline", baseline_df), ("C2 Binary", c2_df)]:
        mob = compute_fp_by_mobility(df_, gt, tau_min)
        print(f"  {label}:")
        print(f"    High-mobility   (λ≥0.5): "
              f"FP={mob['high_mobility']['fp_count']} / {mob['high_mobility']['n_nodes']} "
              f"nodes  rate={mob['high_mobility']['fp_rate']:.1%}")
        print(f"    Low/med-mobility(λ<0.5): "
              f"FP={mob['low_medium_mobility']['fp_count']} / {mob['low_medium_mobility']['n_nodes']} "
              f"nodes  rate={mob['low_medium_mobility']['fp_rate']:.1%}")

    # --- Trust responsiveness ---
    print("\n" + "=" * w)
    print("METRIC 6 — Trust calibration responsiveness (cycles at intermediate trust before blacklist)")
    for label, df_ in [("Baseline (graded)", baseline_df), ("C2 (binary)", c2_df)]:
        resp = compute_trust_responsiveness(df_, gt, tau_min)
        print(f"  {label}:")
        print(f"    mean={resp.get('mean','N/A')}  median={resp.get('median','N/A')}  "
              f"p25={resp.get('p25','N/A')}  p75={resp.get('p75','N/A')}  "
              f"(n={resp.get('n_attackers_evaluated','N/A')} attackers)")

    # --- System overhead (inference time saved) ---
    print("\n" + "=" * w)
    print("METRIC 5 — System overhead (wall-clock time)")
    b_t = baseline_mfst.get("elapsed_seconds", "?")
    c_t = c2_mfst.get("elapsed_seconds", "?")
    print(f"  Baseline (graded): {b_t} s")
    print(f"  C2 (binary):       {c_t} s")
    if isinstance(b_t, (int, float)) and isinstance(c_t, (int, float)) and b_t > 0:
        saving = b_t - c_t
        pct    = saving / b_t * 100
        print(f"  Time saved by skipping DQN inference: {saving:.2f} s ({pct:.1f}%)")

    print("\n" + "=" * w)
    print("END OF REPORT")
    print("=" * w)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="FM-DAD C2 Ablation Metrics",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--baseline", required=True, metavar="CSV",
                   help="Baseline (graded) results CSV from run_ablation.py.")
    p.add_argument("--c2",       required=True, metavar="CSV",
                   help="C2 (binary) results CSV from run_ablation.py.")
    p.add_argument("--gt-dir",   default="data/raw_csvs", metavar="DIR",
                   help="Directory containing node_attack_ground_truth_*.csv files.")
    p.add_argument("--tau-min",  type=float, default=0.3, metavar="FLOAT",
                   help="Blacklisting trust threshold used in both runs.")
    return p


def main() -> None:
    args = _build_parser().parse_args()

    baseline_csv  = Path(args.baseline)
    c2_csv        = Path(args.c2)
    baseline_mfst = Path(str(baseline_csv).replace(".csv", ".manifest.json"))
    c2_mfst_path  = Path(str(c2_csv).replace(".csv", ".manifest.json"))

    for p in [baseline_csv, c2_csv]:
        if not p.exists():
            print(f"[ERROR] File not found: {p}")
            sys.exit(1)

    baseline_df = pd.read_csv(baseline_csv)
    c2_df       = pd.read_csv(c2_csv)

    b_mfst = json.loads(baseline_mfst.read_text()) if baseline_mfst.exists() else {}
    c_mfst = json.loads(c2_mfst_path.read_text()) if c2_mfst_path.exists() else {}

    gt = _load_gt(Path(args.gt_dir))

    print_report(baseline_df, c2_df, b_mfst, c_mfst, gt, args.tau_min)


if __name__ == "__main__":
    main()
