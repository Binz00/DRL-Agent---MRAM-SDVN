"""
verify_bridge.py — Part 1 & Part 2 verification for the FM-DAD data bridge.

Run from the fm_dad/ directory:
    python3 verify_bridge.py

Checks:
    1. Detects cycle numbers and loads all cycles (Part 1).
    2. Computes per-cycle features (Part 2).
    3. Prints summary stats, key integrity verification, and attacker label count.
    4. Prints min/max/mean of each new feature.
    5. Confirms FFc and rho_recv are within [0, 1] bounds.
    6. Reports counts of NaN values in d_bar and DelayInfl.
    7. Shows 5 sample rows with node_id, is_attacker, and the 8 new features.
"""

import sys
import os

# Allow running from fm_dad/ without installing the package
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
from bridge.join import load_all_cycles
from bridge.features_percycle import add_percycle_features
from bridge.config_bridge import RAW_CSV_FOLDER

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 250)

print("\n" + "=" * 80)
print("FM-DAD DATA BRIDGE — Part 1 & Part 2 Verification")
print("=" * 80)

# --- 1. Load all cycles (Part 1) ----------------------------------------------
df_joined = load_all_cycles(RAW_CSV_FOLDER)

if df_joined.empty:
    print("\n[WARN] No data loaded — place your CSV files in:")
    print(f"       {RAW_CSV_FOLDER}")
    print("       Then re-run this script.")
    sys.exit(0)

# --- 2. Add per-cycle features (Part 2) --------------------------------------
df_features = add_percycle_features(df_joined)

# --- 3. Part 1 Summary stats --------------------------------------------------
print("\n--- PART 1 SUMMARY ---")
cycles_found = sorted(df_features["cycle_id"].unique())
print(f"  Cycles found        : {len(cycles_found)}  →  {cycles_found}")

nodes_per_cycle = (
    df_features.groupby("cycle_id")["node_id"].nunique()
      .rename("nodes")
      .reset_index()
)
print(f"  Nodes per cycle:")
for _, row in nodes_per_cycle.iterrows():
    print(f"      cycle {int(row.cycle_id):>3d} : {int(row.nodes)} node(s)")

print(f"  Total combined rows : {len(df_features)}")
print(f"  Total columns       : {len(df_features.columns)}")

# --- 4. Key integrity assertion -----------------------------------------------
print("\n--- KEY INTEGRITY ---")
null_node = df_features["node_id"].isna().sum()
null_cycle = df_features["cycle_id"].isna().sum()

assert null_node == 0,  f"FAIL: {null_node} rows have null node_id"
assert null_cycle == 0, f"FAIL: {null_cycle} rows have null cycle_id"
print("  PASS  node_id  — 0 null values")
print("  PASS  cycle_id — 0 null values")

# --- 5. Attacker distribution -------------------------------------------------
print("\n--- ATTACKER DISTRIBUTION ---")
if "is_attacker" in df_features.columns:
    dist = df_features["is_attacker"].value_counts(dropna=False).sort_index()
    for label, count in dist.items():
        tag = "(attacker)" if label == 1 else "(innocent)" if label == 0 else "(NaN)"
        print(f"  is_attacker = {label}  →  {count} rows  {tag}")
else:
    print("  [WARN] 'is_attacker' column not present")

# --- 6. Part 2 Verification: Feature Statistics --------------------------------
print("\n--- PART 2 FEATURE STATISTICS ---")
new_features = ['FFc', 'rho_recv', 'dFF', 'd_bar', 'DelayInfl', 'lambda_t', 'tau', 'SpoofDev_raw']
for feat in new_features:
    f_min = df_features[feat].min()
    f_max = df_features[feat].max()
    f_mean = df_features[feat].mean()
    print(f"  Feature '{feat:<12}' → min: {f_min:>8.4f} | max: {f_max:>8.4f} | mean: {f_mean:>8.4f}")

# --- 7. Part 2 Bounds Confirmation -------------------------------------------
print("\n--- BOUNDS CONFIRMATION ---")
# Check if FFc and rho_recv are in [0, 1] range (ignoring NaNs)
ffc_in_bounds = df_features['FFc'].dropna().between(0.0, 1.0).all()
rho_in_bounds = df_features['rho_recv'].dropna().between(0.0, 1.0).all()

if ffc_in_bounds:
    print("  PASS: FFc values are strictly within [0, 1]")
else:
    print("  FAIL: FFc values are outside [0, 1]")

if rho_in_bounds:
    print("  PASS: rho_recv values are strictly within [0, 1]")
else:
    print("  FAIL: rho_recv values are outside [0, 1]")

# --- 8. Part 2 NaN counts in d_bar and DelayInfl -----------------------------
print("\n--- NaN COUNTS ---")
nan_d_bar = df_features['d_bar'].isna().sum()
nan_delay_infl = df_features['DelayInfl'].isna().sum()
print(f"  d_bar NaN count    : {nan_d_bar}")
print(f"  DelayInfl NaN count: {nan_delay_infl}")

# --- 9. Show 5 sample rows with target columns --------------------------------
print("\n--- 5 SAMPLE ROWS WITH NEW FEATURES ---")
sample_cols = ['cycle_id', 'node_id', 'is_attacker'] + new_features
# Drop rows where everything in new_features is NaN just to show a meaningful sample if possible
sample_df = df_features[sample_cols].dropna(subset=['FFc', 'rho_recv', 'tau'], how='all')
if len(sample_df) >= 5:
    print(sample_df.head(5).to_string(index=False))
else:
    print(df_features[sample_cols].head(5).to_string(index=False))

print("\n" + "=" * 80)
print("Verification complete — Part 2 DONE")
print("=" * 80 + "\n")
