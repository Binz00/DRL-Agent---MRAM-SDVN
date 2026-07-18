"""
validate_pipeline.py — FM-DAD pipeline validation.

Primary metric: MCC^X per attack variant (Equation 4.1).
Blacklist threshold: τ_min (Eq. 3.80) — selected by grid search over Table 4.1 candidates.

Usage:
    python3 validate_pipeline.py
"""

import glob
import math
import re
from pathlib import Path
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
FM_DAD_DIR    = Path(__file__).parent
PENALTIES_CSV = FM_DAD_DIR / "data" / "pipeline_penalties.csv"
RAW_CSV_DIR   = FM_DAD_DIR / "data" / "raw_csvs"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TAU_MIN = 0.5  # default — overridden by grid_search_tau() at runtime

# Grid search candidates — report Table 4.1 / Eq. 3.80
TAU_CANDIDATES = [0.30, 0.40, 0.50]


# ---------------------------------------------------------------------------
# MCC
# ---------------------------------------------------------------------------

def compute_mcc(tp: int, fp: int, fn: int, tn: int) -> float:
    """Implements Equation 4.1 from the report."""
    numerator   = tp * tn - fp * fn
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return numerator / denominator if denominator > 0 else 0.0


# ---------------------------------------------------------------------------
# Core evaluation helper — parameterised on τ_min
# ---------------------------------------------------------------------------

def _evaluate_at_tau(tau_min: float, df: pd.DataFrame, gt: pd.DataFrame) -> dict:
    """
    Run the full MCC^X evaluation at a given τ_min threshold.

    Returns a dict with keys: tau_min, mcc_results, macro_mcc, gt.
    Does NOT print anything — caller decides what to print.

    Args:
        tau_min : Blacklist threshold to evaluate.
        df      : pipeline_penalties.csv DataFrame (for cycle-level trust).
        gt      : Ground truth DataFrame with min_trust_reached already merged.
                  Must NOT have 'detected' pre-set — this function sets it.
    """
    gt = gt.copy()

    # Step 3 — classify using this tau_min
    gt["detected"] = gt["min_trust_reached"] < tau_min

    # Step 4 — MCC^X per attack type (correct definition: exclude other-type attackers)
    attack_types = sorted(gt.loc[gt["is_attacker"] == 1, "attack_type"].unique())
    honest_mask  = gt["is_attacker"] == 0

    mcc_results = []
    for atype in attack_types:
        # Target class: attackers of this specific type only
        is_target = (gt["is_attacker"] == 1) & (gt["attack_type"] == atype)

        # Restrict to: type-X attackers (positive class) + honest nodes (negative class).
        # Attackers of ALL other types are excluded — irrelevant to MCC^X (Eq. 4.1).
        relevant      = is_target | honest_mask
        rel_df        = gt[relevant]
        rel_is_target = is_target[relevant]
        rel_detected  = rel_df["detected"]

        tp = int(( rel_is_target &  rel_detected).sum())
        fp = int((~rel_is_target &  rel_detected).sum())  # honest nodes wrongly blacklisted
        fn = int(( rel_is_target & ~rel_detected).sum())
        tn = int((~rel_is_target & ~rel_detected).sum())  # honest nodes correctly safe

        mcc = compute_mcc(tp, fp, fn, tn)
        mcc_results.append((atype, tp, fp, fn, tn, mcc))

    macro_mcc = sum(r[5] for r in mcc_results) / len(mcc_results) if mcc_results else 0.0

    return {
        "tau_min":     tau_min,
        "mcc_results": mcc_results,
        "macro_mcc":   macro_mcc,
        "gt":          gt,
    }


# ---------------------------------------------------------------------------
# Grid search
# ---------------------------------------------------------------------------

def grid_search_tau(df: pd.DataFrame, gt_base: pd.DataFrame,
                    cycle_trust: pd.DataFrame) -> tuple:
    """
    Grid search over TAU_CANDIDATES (report Table 4.1, Eq. 3.80).

    Selection criterion: highest macro-averaged MCC^X across all attack variants.
    Tie-break: prefer higher τ_min (faster blacklisting = stronger security posture).

    Returns:
        (best_tau_min: float, best_result: dict)
    """
    W = 64
    print("=" * W)
    print("τ_min GRID SEARCH  (report Table 4.1, Eq. 3.80)")
    print(f"Candidates: {TAU_CANDIDATES}")
    print("=" * W)

    # Build header — attack types from data (sorted alphabetically)
    attack_types = sorted(gt_base.loc[gt_base["is_attacker"] == 1, "attack_type"].unique())
    header_atypes = "  ".join(f"{a:>8}" for a in attack_types)
    print(f"{'τ_min':<8} {header_atypes}  {'Macro':>8}  {'FP':>5}  Note")
    print("-" * W)

    all_results = []
    for tau in TAU_CANDIDATES:
        res = _evaluate_at_tau(tau, df, gt_base)
        all_results.append(res)
        mcc_by_type = {r[0]: r[5] for r in res["mcc_results"]}
        # FP is constant across attack types (same honest pool), read from first entry
        fp_total = res["mcc_results"][0][2] if res["mcc_results"] else 0
        note     = "<-- current default" if tau == 0.50 else ""
        atypes_str = "  ".join(f"{mcc_by_type.get(a, 0.0):>+8.4f}" for a in attack_types)
        print(f"  {tau:<6.2f} {atypes_str}  {res['macro_mcc']:>+8.4f}  {fp_total:>5}  {note}")

    # Select best: highest macro MCC, tie-break by higher τ_min
    best = max(all_results, key=lambda r: (round(r["macro_mcc"], 6), r["tau_min"]))

    print()
    print(f"  SELECTED: τ_min = {best['tau_min']}  "
          f"(macro MCC = {best['macro_mcc']:+.4f})")
    print("=" * W)
    return best["tau_min"], best


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def validate_results() -> None:

    # ------------------------------------------------------------------
    # Step 1 — Read pipeline_penalties.csv, compute min trust per node
    # ------------------------------------------------------------------
    if not PENALTIES_CSV.exists():
        print(f"[ERROR] {PENALTIES_CSV} not found.")
        print("Run  python3 bridge/run_pipeline.py  then re-run validate_pipeline.py.")
        return

    df = pd.read_csv(PENALTIES_CSV)

    min_trust = (
        df.groupby("node_id")["trust_score_after"]
        .min()
        .rename("min_trust_reached")
    )

    # Per-cycle trust data (for trust distribution table)
    cycle_trust = df[["cycle_id", "node_id", "trust_score_after"]].copy()

    # ------------------------------------------------------------------
    # Step 2 — Build ground truth as the UNION across all cycle GT files.
    # SP/ALS/FS sets are static; IGH rotates per cycle (ON/OFF scheduling),
    # so a node is an attacker if it was an attacker in ANY cycle.
    # ------------------------------------------------------------------
    gt_files = sorted(
        glob.glob(str(RAW_CSV_DIR / "node_attack_ground_truth_*.csv")),
        key=lambda p: int(re.search(r"_(\d+)\.csv$", p).group(1)),
    )
    if not gt_files:
        print(f"[ERROR] No ground truth files found in {RAW_CSV_DIR}")
        return

    gt_all = pd.concat([pd.read_csv(f) for f in gt_files], ignore_index=True)
    gt_all.columns = gt_all.columns.str.strip()

    # Union: attacker in any cycle → attacker overall.
    # attack_type: take the (unique) non-NONE type the node ever held.
    def _resolve(group):
        is_att = int(group["is_attacker"].max())
        if is_att:
            types = group.loc[group["is_attacker"] == 1, "attack_type"].unique()
            atype = types[0]  # nodes hold exactly one attack type across cycles
        else:
            atype = "NONE"
        return pd.Series({"is_attacker": is_att, "attack_type": atype})

    gt_base = gt_all.groupby("node_id").apply(_resolve).reset_index()

    # Sanity check: no node should have more than one distinct non-NONE attack_type
    att_nodes = gt_all[gt_all["is_attacker"] == 1]
    if not att_nodes.empty:
        multi_type = (
            att_nodes.groupby("node_id")["attack_type"]
            .nunique()
            .loc[lambda x: x > 1]
        )
        if not multi_type.empty:
            print(f"[WARNING] Nodes with conflicting attack types across cycles: "
                  f"{multi_type.index.tolist()}")

    print(f"[GT] Union across {len(gt_files)} cycles | "
          f"attackers={int(gt_base['is_attacker'].sum())} "
          f"({gt_base[gt_base['is_attacker']==1]['attack_type'].value_counts().to_dict()})")

    # Merge min_trust ONCE before grid search.
    # NOTE: do NOT set gt["detected"] here — _evaluate_at_tau does it per tau.
    gt_base = gt_base.merge(min_trust, on="node_id", how="left")
    gt_base["min_trust_reached"] = gt_base["min_trust_reached"].fillna(1.0)

    # ------------------------------------------------------------------
    # Grid search — select best τ_min
    # ------------------------------------------------------------------
    best_tau, best_result = grid_search_tau(df, gt_base, cycle_trust)

    # Use the best result's gt (already has "detected" set at best_tau)
    gt          = best_result["gt"]
    mcc_results = best_result["mcc_results"]
    macro_mcc   = best_result["macro_mcc"]

    # ------------------------------------------------------------------
    # Step 5 — Print full detailed output for best τ_min
    # ------------------------------------------------------------------
    W = 64
    print()
    print("=" * W)
    print("FM-DAD VALIDATION — MCC per Attack Variant (Eq. 4.1)")
    print("=" * W)
    print(f"τ_min = {best_tau}  (SC-3 blacklist threshold, Eq. 3.80 — grid-search selected)")
    print()
    print(f"{'Attack':<10} {'TP':>5} {'FP':>5} {'FN':>5} {'TN':>5}  {'MCC':>8}")
    print("-" * W)
    for atype, tp, fp, fn, tn, mcc in mcc_results:
        print(f"{atype:<10} {tp:>5} {fp:>5} {fn:>5} {tn:>5}  {mcc:>+8.4f}")
    print()
    print(f"Macro-averaged MCC (all attacks): {macro_mcc:+.4f}")
    print(f"MCC ∈ [−1,+1]: 0=random guessing, 1=perfect, −1=perfectly wrong")
    print("=" * W)

    # Trust distribution per cycle
    print()
    print("TRUST DISTRIBUTION PER CYCLE (nodes below τ_min):")
    print(f"{'Cycle':<7} {'Attackers below τ_min':>23} {'Honest below τ_min':>20}")

    attacker_ids = set(gt.loc[gt["is_attacker"] == 1, "node_id"])
    honest_ids   = set(gt.loc[gt["is_attacker"] == 0, "node_id"])
    total_att    = len(attacker_ids)
    total_hon    = len(honest_ids)

    for cycle_id in sorted(cycle_trust["cycle_id"].unique()):
        cyc       = cycle_trust[cycle_trust["cycle_id"] == cycle_id]
        att_below = int(cyc[cyc["node_id"].isin(attacker_ids)]["trust_score_after"].lt(best_tau).sum())
        hon_below = int(cyc[cyc["node_id"].isin(honest_ids )]["trust_score_after"].lt(best_tau).sum())
        print(f"  {cycle_id:<5}  {att_below:>10} / {total_att:<10}  {hon_below:>8} / {total_hon:<8}")

    print("=" * W)

    # ------------------------------------------------------------------
    # Step 6 — Export pipeline_mcc_validation.csv
    # ------------------------------------------------------------------
    def _label(row):
        is_att = row["is_attacker"] == 1
        det    = row["detected"]
        if is_att  and det:      return "TP"
        if not is_att and det:   return "FP"
        if is_att  and not det:  return "FN"
        return "TN"

    gt["classification"] = gt.apply(_label, axis=1)

    out = gt[["node_id", "is_attacker", "attack_type",
              "min_trust_reached", "detected", "classification"]].copy()
    out["min_trust_reached"] = out["min_trust_reached"].round(6)

    out_path = FM_DAD_DIR / "data" / "pipeline_mcc_validation.csv"
    out.to_csv(out_path, index=False)
    print(f"\n[SUCCESS] Exported → {out_path}")

    # ------------------------------------------------------------------
    # Grid search summary — report-ready one-liner
    # ------------------------------------------------------------------
    print()
    print(f"[GRID SEARCH] Best τ_min = {best_tau} selected "
          f"(macro MCC = {macro_mcc:+.4f}).")
    print(f"Evaluated candidates: {set(TAU_CANDIDATES)} per report Table 4.1 / Eq. 3.80.")
    print("Selection criterion: highest macro-averaged MCC^X across all attack variants.")
    print("Tie-break: higher τ_min preferred (stronger blacklist security posture).")


if __name__ == "__main__":
    validate_results()
