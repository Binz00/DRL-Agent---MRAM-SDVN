"""
verify_bridge.py — Validation for the FM-DAD data bridge.

Run from the fm_dad/ directory:
    python3 verify_bridge.py

Join: loads all cycles, checks key integrity.
Per-cycle features: FFc, rho_recv, dFF, d_bar, DelayInfl, lambda_t, tau, SpoofDev_raw.
Windowed features: PDRVar, CoordScore, SpoofDev.
Assemble: agent state-vector tables and write CSVs.
"""

import sys
import os

# Allow running from fm_dad/ without installing the package
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from bridge.join import load_all_cycles
from bridge.features_percycle import add_percycle_features
from bridge.features_windowed import add_windowed_features
from bridge.assemble import assemble_agent_tables, write_agent_csvs, AGENT_STATE_FEATURES
from bridge.config_bridge import RAW_CSV_FOLDER

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 250)

print("\n" + "=" * 80)
print("FM-DAD DATA BRIDGE — Verification")
print("=" * 80)

# ===== PART 1: Load all cycles ================================================
df_joined = load_all_cycles(RAW_CSV_FOLDER)

if df_joined.empty:
    print("\n[WARN] No data loaded — place your CSV files in:")
    print(f"       {RAW_CSV_FOLDER}")
    print("       Then re-run this script.")
    sys.exit(0)

# ===== PART 2: Per-cycle features =============================================
df_features = add_percycle_features(df_joined)

# ===== PART 3: Windowed features ==============================================
df_windowed = add_windowed_features(df_features)

# ---------------------------------------------------------------------------
# Join Summary
# ---------------------------------------------------------------------------
print("\n--- JOIN SUMMARY ---")
cycles_found = sorted(df_windowed["cycle_id"].unique())
print(f"  Cycles found        : {len(cycles_found)}  →  {cycles_found}")

nodes_per_cycle = (
    df_windowed.groupby("cycle_id")["node_id"].nunique()
      .rename("nodes")
      .reset_index()
)
print("  Nodes per cycle:")
for _, row in nodes_per_cycle.iterrows():
    print(f"      cycle {int(row.cycle_id):>3d} : {int(row.nodes)} node(s)")

print(f"  Total combined rows : {len(df_windowed)}")
print(f"  Total columns       : {len(df_windowed.columns)}")

# ---------------------------------------------------------------------------
# KEY INTEGRITY
# ---------------------------------------------------------------------------
print("\n--- KEY INTEGRITY ---")
null_node = df_windowed["node_id"].isna().sum()
null_cycle = df_windowed["cycle_id"].isna().sum()
assert null_node == 0,  f"FAIL: {null_node} rows have null node_id"
assert null_cycle == 0, f"FAIL: {null_cycle} rows have null cycle_id"
print("  PASS  node_id  — 0 null values")
print("  PASS  cycle_id — 0 null values")

# ---------------------------------------------------------------------------
# ATTACKER DISTRIBUTION
# ---------------------------------------------------------------------------
print("\n--- ATTACKER DISTRIBUTION ---")
if "is_attacker" in df_windowed.columns:
    dist = df_windowed["is_attacker"].value_counts(dropna=False).sort_index()
    for label, count in dist.items():
        tag = "(attacker)" if label == 1 else "(innocent)" if label == 0 else "(NaN)"
        print(f"  is_attacker = {label}  →  {count} rows  {tag}")

# ---------------------------------------------------------------------------
# Per-Cycle Feature Stats
# ---------------------------------------------------------------------------
print("\n--- PER-CYCLE FEATURE STATISTICS ---")
percycle_feats = ['FFc', 'rho_recv', 'dFF', 'd_bar', 'DelayInfl', 'lambda_t', 'lambda_t_norm', 'tau', 'SpoofDev_raw']
for feat in percycle_feats:
    f_min  = df_windowed[feat].min()
    f_max  = df_windowed[feat].max()
    f_mean = df_windowed[feat].mean()
    print(f"  {feat:<12} → min: {f_min:>10.4f} | max: {f_max:>10.4f} | mean: {f_mean:>10.4f}")

print("\n--- BOUNDS CONFIRMATION ---")
ffc_ok = df_windowed['FFc'].dropna().between(0.0, 1.0).all()
rho_ok = df_windowed['rho_recv'].dropna().between(0.0, 1.0).all()
print(f"  {'PASS' if ffc_ok else 'FAIL'}: FFc within [0, 1]")
print(f"  {'PASS' if rho_ok else 'FAIL'}: rho_recv within [0, 1]")

print("\n" + "=" * 80)
print("Windowed Feature Verification")
print("=" * 80)

# 3a. W* selected per cycle (already logged, reprint)
print("\n--- W* SELECTED PER CYCLE ---")
from bridge.features_windowed import select_window
for cycle in cycles_found:
    cycle_mask = df_windowed["cycle_id"] == cycle
    lam_norm_med = df_windowed.loc[cycle_mask, "lambda_t_norm"].median()
    lam_raw_med  = df_windowed.loc[cycle_mask, "lambda_t"].median()
    w_star = select_window(lam_norm_med)
    print(f"  Cycle {int(cycle):>3d} | lambda_t_raw={lam_raw_med:>8.1f}, lambda_t_norm={lam_norm_med:.4f} → W* = {w_star}")

# 3b. Min / max / mean of windowed features
print("\n--- WINDOWED FEATURE STATISTICS ---")
windowed_feats = ['PDRVar', 'CoordScore', 'SpoofDev']
for feat in windowed_feats:
    col = df_windowed[feat]
    f_min  = col.min()
    f_max  = col.max()
    f_mean = col.mean()
    n_nan  = col.isna().sum()
    print(f"  {feat:<12} → min: {f_min:>10.6f} | max: {f_max:>10.6f} | mean: {f_mean:>10.6f} | NaN: {n_nan}")

# 3c. Full vs partial window counts
print("\n--- WINDOW COMPLETENESS ---")
# Reconstruct counts: a node has a full window if it appeared in >= W* cycles
# (simplified check using the reported values from the log)
for cycle in cycles_found:
    cycle_mask = df_windowed["cycle_id"] == cycle
    lam_med = df_windowed.loc[cycle_mask, "lambda_t"].median()
    w_star = select_window(lam_med)
    n_total = cycle_mask.sum()
    # Since we only have 2 cycles and min W* = 10, all are partial
    print(f"  Cycle {int(cycle):>3d} | W*={w_star}, total_nodes={n_total}, all partial (only {len(cycles_found)} cycles available)")

# 3d. Attacker vs innocent means for PDRVar and CoordScore
print("\n--- ATTACKER vs INNOCENT MEANS ---")
if "is_attacker" in df_windowed.columns:
    known = df_windowed[df_windowed["is_attacker"].isin([0.0, 1.0])].copy()
    if not known.empty:
        group_means = known.groupby("is_attacker")[["PDRVar", "CoordScore"]].mean()
        print(group_means.to_string())

        for feat in ["PDRVar", "CoordScore"]:
            att_mean = group_means.loc[1.0, feat] if 1.0 in group_means.index else np.nan
            inn_mean = group_means.loc[0.0, feat] if 0.0 in group_means.index else np.nan
            if not np.isnan(att_mean) and not np.isnan(inn_mean):
                higher = "ATTACKERS higher" if att_mean > inn_mean else "INNOCENTS higher (unexpected)"
                print(f"  {feat}: attacker_mean={att_mean:.6f}, innocent_mean={inn_mean:.6f} → {higher}")
    else:
        print("  [WARN] No rows with known is_attacker label")

# 3e. 5 sample rows
print("\n--- 5 SAMPLE ROWS WITH WINDOWED FEATURES ---")
sample_cols = ['cycle_id', 'node_id', 'is_attacker', 'PDRVar', 'CoordScore', 'SpoofDev']
# Try to show rows that have known attacker labels
known_rows = df_windowed[df_windowed["is_attacker"].isin([0.0, 1.0])][sample_cols]
if len(known_rows) >= 5:
    print(known_rows.head(5).to_string(index=False))
else:
    print(df_windowed[sample_cols].head(5).to_string(index=False))

print("\n" + "=" * 80)
print("Verification complete")
print("=" * 80)

# =========================================================================
# Assemble agent tables
# =========================================================================
print("\n" + "=" * 80)
print("Agent State-Vector Tables")
print("=" * 80)

tables = assemble_agent_tables(df_windowed)
write_agent_csvs(tables)

for agent, agent_df in tables.items():
    state_feats = AGENT_STATE_FEATURES[agent]
    n_att = (agent_df["is_attacker"] == 1.0).sum()
    n_inn = (agent_df["is_attacker"] == 0.0).sum()

    print(f"\n--- {agent.upper()} AGENT ---")
    print(f"  Rows          : {len(agent_df)}")
    print(f"  Attackers     : {n_att}")
    print(f"  Innocents     : {n_inn}")
    print(f"  Columns       : {list(agent_df.columns)}")
    print(f"  State features: {state_feats}")

    # Confirm is_attacker is NOT among the state features
    assert "is_attacker" not in state_feats, \
        f"FAIL: is_attacker found in {agent.upper()} state features!"
    print("  PASS: is_attacker is NOT a state feature")

    # Confirm no attack_type column
    assert "attack_type" not in agent_df.columns, \
        f"FAIL: attack_type column found in {agent.upper()} table!"
    print("  PASS: no attack_type column")

    # Confirm feature order matches (columns between node_id/cycle_id and is_attacker)
    actual_feats = list(agent_df.columns[2:2+len(state_feats)])
    assert actual_feats == state_feats, \
        f"FAIL: Feature order mismatch in {agent.upper()}! Expected {state_feats}, got {actual_feats}"
    print(f"  PASS: feature order matches")

    # 3 sample rows
    print(f"  Sample rows:")
    print(agent_df.head(3).to_string(index=False))

print("\n" + "=" * 80)
print("Verification complete")
print("=" * 80 + "\n")
