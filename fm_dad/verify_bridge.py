"""
verify_bridge.py — Part 1 verification for the FM-DAD data bridge.

Run from the fm_dad/ directory:
    python3 verify_bridge.py

Checks:
    1. Detects cycle numbers and loads all cycles.
    2. Prints: cycles found, nodes per cycle, total rows, column list.
    3. Asserts every row has a non-null node_id and cycle_id.
    4. Prints is_attacker distribution (0 vs 1).
    5. Shows the first 5 rows of the combined table.
"""

import sys
import os

# Allow running from fm_dad/ without installing the package
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
from bridge.join import load_all_cycles
from bridge.config_bridge import RAW_CSV_FOLDER

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

print("\n" + "=" * 70)
print("FM-DAD DATA BRIDGE — Part 1 Verification")
print("=" * 70)

# --- 1. Load all cycles -------------------------------------------------------
df = load_all_cycles(RAW_CSV_FOLDER)

if df.empty:
    print("\n[WARN] No data loaded — place your CSV files in:")
    print(f"       {RAW_CSV_FOLDER}")
    print("       Then re-run this script.")
    sys.exit(0)

# --- 2. Summary stats ---------------------------------------------------------
print("\n--- SUMMARY ---")
cycles_found = sorted(df["cycle_id"].unique())
print(f"  Cycles found        : {len(cycles_found)}  →  {cycles_found}")

nodes_per_cycle = (
    df.groupby("cycle_id")["node_id"].nunique()
      .rename("nodes")
      .reset_index()
)
print(f"  Nodes per cycle:")
for _, row in nodes_per_cycle.iterrows():
    print(f"      cycle {int(row.cycle_id):>3d} : {int(row.nodes)} node(s)")

print(f"  Total combined rows : {len(df)}")
print(f"  Total columns       : {len(df.columns)}")
print(f"\n  Column list:")
for i, col in enumerate(df.columns, 1):
    print(f"      {i:>3}. {col}")

# --- 3. Key integrity assertion -----------------------------------------------
print("\n--- KEY INTEGRITY ---")
null_node = df["node_id"].isna().sum()
null_cycle = df["cycle_id"].isna().sum()

assert null_node == 0,  f"FAIL: {null_node} rows have null node_id"
assert null_cycle == 0, f"FAIL: {null_cycle} rows have null cycle_id"
print("  PASS  node_id  — 0 null values")
print("  PASS  cycle_id — 0 null values")

# --- 4. Attacker distribution -------------------------------------------------
print("\n--- ATTACKER DISTRIBUTION ---")
if "is_attacker" in df.columns:
    dist = df["is_attacker"].value_counts(dropna=False).sort_index()
    for label, count in dist.items():
        tag = "(attacker)" if label == 1 else "(innocent)" if label == 0 else "(NaN)"
        print(f"  is_attacker = {label}  →  {count} rows  {tag}")
else:
    print("  [WARN] 'is_attacker' column not present (ground_truth file missing?)")

# --- 5. First 5 rows ----------------------------------------------------------
print("\n--- FIRST 5 ROWS ---")
print(df.head(5).to_string(index=False))

print("\n" + "=" * 70)
print("Verification complete — Part 1 DONE")
print("=" * 70 + "\n")
