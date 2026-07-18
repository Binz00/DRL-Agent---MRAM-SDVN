"""
assemble.py — Build four agent-specific state-vector tables (Part 4).

Public API:
    assemble_agent_tables(df) -> dict[str, pd.DataFrame]
        Returns {"sp": df_sp, "als": df_als, "fs": df_fs, "igh": df_igh}.

    write_agent_csvs(tables, output_dir)
        Writes each table as {agent}_state.csv.

Each agent receives ONLY its own state features in the EXACT order the
trained DRL agents expect.  is_attacker is a separate reward-label column
and is NEVER part of the state vector.
"""

import logging
from pathlib import Path
from typing import Dict

import pandas as pd

logger = logging.getLogger("bridge")


# ---------------------------------------------------------------------------
# Agent state-feature definitions — ORDER MATTERS
# ---------------------------------------------------------------------------

AGENT_STATE_FEATURES: Dict[str, list] = {
    "sp":  ["FFc", "dFF", "rho_recv", "tau", "lambda_t_norm"],
    "als": ["SpoofDev", "dFF", "tau", "lambda_t_norm"],
    "fs":  ["FFc", "dFF", "DelayInfl", "tau", "lambda_t_norm"],
    "igh": ["FFc", "dFF", "rho_recv", "d_bar", "tau",
            "PDRVar", "CoordScore", "lambda_t_norm"],
}

# Metadata columns kept alongside the state features (NOT part of the state)
META_COLS = ["node_id", "cycle_id", "is_attacker"]

# Extra non-state columns needed by specific agents' reward functions
EXTRA_COLS: Dict[str, list] = {
    "sp":  [],
    "als": [],
    "fs":  ["rho_recv", "hop_excess"],   # hop_excess: gate-only, NOT part of Eq. 3.44 state vector
    "igh": [],
}

# Default output directory
_BRIDGE_DIR = Path(__file__).parent
_FM_DAD_DIR = _BRIDGE_DIR.parent
AGENT_INPUT_DIR = str(_FM_DAD_DIR / "data" / "agent_inputs")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def assemble_agent_tables(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Build four agent-specific state-vector tables from the full feature DataFrame.

    Processing steps:
        1. Drop all rows where is_attacker is NaN (unlabeled nodes).
        2. For each agent, select [META_COLS + state_features + extra_cols].
        3. Drop rows where any required STATE feature is NaN.
        4. Log per-agent row counts, NaN drops, and attacker/innocent splits.

    The returned DataFrames have columns in this order:
        node_id, cycle_id, <state features in exact order>, is_attacker
        (plus any extra columns like rho_recv for FS, placed after is_attacker).

    Args:
        df: Full DataFrame from Parts 1–3 with all features computed.

    Returns:
        Dict mapping agent name ("sp", "als", "fs", "igh") to its table.
    """
    logger.info("=== assemble_agent_tables | input rows=%d ===", len(df))

    # ---- Step 1: Drop unlabeled rows ----------------------------------------
    n_before = len(df)
    df_labeled = df[df["is_attacker"].notna()].copy()
    n_dropped_label = n_before - len(df_labeled)
    logger.info(
        "Step 1: Dropped %d unlabeled rows (is_attacker=NaN). "
        "Remaining: %d rows.", n_dropped_label, len(df_labeled),
    )

    # ---- Step 2–4: Build each agent's table ---------------------------------
    tables: Dict[str, pd.DataFrame] = {}

    for agent, state_feats in AGENT_STATE_FEATURES.items():
        extras = [c for c in EXTRA_COLS.get(agent, []) if c in df_labeled.columns]

        # Column order: meta → state features → is_attacker → extras
        cols = ["node_id", "cycle_id"] + state_feats + ["is_attacker"] + extras
        agent_df = df_labeled[cols].copy()

        # Drop rows where any STATE feature is NaN
        n_before_nan = len(agent_df)
        agent_df = agent_df.dropna(subset=state_feats).reset_index(drop=True)
        n_dropped_nan = n_before_nan - len(agent_df)

        # Attacker / innocent split
        n_att = (agent_df["is_attacker"] == 1.0).sum()
        n_inn = (agent_df["is_attacker"] == 0.0).sum()

        logger.info(
            "  [%s] %d rows | dropped %d NaN-state rows | "
            "attackers=%d, innocents=%d | features=%s",
            agent.upper(), len(agent_df), n_dropped_nan,
            n_att, n_inn, state_feats,
        )

        tables[agent] = agent_df

    return tables


def write_agent_csvs(
    tables: Dict[str, pd.DataFrame],
    output_dir: str = AGENT_INPUT_DIR,
) -> None:
    """
    Write each agent table to a CSV file in the output directory.

    File names: sp_state.csv, als_state.csv, fs_state.csv, igh_state.csv.

    Args:
        tables    : Dict from assemble_agent_tables().
        output_dir: Directory to write CSVs into (created if missing).
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for agent, df in tables.items():
        path = out / f"{agent}_state.csv"
        df.to_csv(path, index=False)
        logger.info("  Written %s (%d rows, %d cols)", path.name, len(df), len(df.columns))
