"""
data_loader.py — CSV loader and transition builder for offline FM-DAD training.

Reads agent-specific CSV files (one per attack type) and constructs
(s_t, a placeholder, r placeholder, s_{t+1}, done) transition tuples
grouped by node_id and ordered by cycle_id.

Specification: Section 9 of the report.

Key rules:
    - NO attack_type column is ever read or used (R2).
    - The state vector columns are exactly those listed in AGENT_CONFIGS[name]['features'].
    - Consecutive rows for the SAME node_id form (s_t, s_{t+1}) pairs.
    - The LAST cycle of each node is a terminal transition (done=True).
    - Rewards and actions are NOT pre-recorded; they are generated on the fly
      during training.  This loader ONLY returns state pairs + ground-truth
      auxiliary columns needed by the reward function.
"""

from __future__ import annotations
from typing import List
import numpy as np
import pandas as pd
from config import get_logger, AGENT_CONFIGS

logger = get_logger("data_loader")

# Ground-truth / auxiliary columns needed by the reward functions.
# These are present in every CSV but are NEVER part of the state vector.
REWARD_AUX_COLUMNS = [
    "is_attacker",        # 0/1, ground-truth label for reward computation
    "blockchain_reject",  # 0/1, blockchain flag for r_end (Eq. 3.47)
    "PDR_t",              # current PDR  (for r_qos, Eq. 3.56)
    "d_bar_t",            # current mean delay (for r_qos, Eq. 3.56)
    # Extra features needed by some reward functions but NOT in every state:
    "rho_recv",           # used by IGH/SP/FS reward fp detection
    "lambda_t",           # used by ALS/FS reward fp detection
]


def load_transitions(csv_path: str, agent_name: str) -> List[dict]:
    """
    Load a training CSV and build a list of transition dicts for one agent.

    Each dict has:
        's'                : np.ndarray, state vector at t       (input_dim,)
        's_next'           : np.ndarray, state vector at t+1     (input_dim,)
        'done'             : bool, True only on last cycle of a node
        'is_attacker'      : int  (0 or 1)
        'blockchain_reject': int  (0 or 1)
        'PDR_t'            : float
        'd_bar_t'          : float
        'rho_recv'         : float  (0.0 if column absent)
        'lambda_t'         : float  (0.0 if column absent)

    NOTE: 'action' and 'reward' are NOT stored — they are computed on the fly
    during training (Section 9 of the report).

    Args:
        csv_path   : Path to the agent's CSV file.
        agent_name : One of 'sp', 'als', 'igh', 'fs'.

    Returns:
        List of transition dicts. Order within each node is temporal.

    Implements: Section 9 (offline training data format).
    """
    logger.info("[%s] Loading CSV: %s", agent_name, csv_path)

    cfg      = AGENT_CONFIGS[agent_name]
    features = cfg["features"]

    # ---- Read CSV ----------------------------------------------------------
    df = pd.read_csv(csv_path)
    logger.info("[%s] CSV loaded | rows=%d, columns=%s", agent_name, len(df), list(df.columns))

    # Validate required columns
    required = features + ["node_id", "cycle_id", "is_attacker", "blockchain_reject",
                           "PDR_t", "d_bar_t"]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"[{agent_name}] CSV missing columns: {missing}")

    # rho_recv / lambda_t may be absent from ALS state but still needed for reward
    for col in ("rho_recv", "lambda_t"):
        if col not in df.columns:
            df[col] = 0.0
            logger.info("[%s] Column '%s' not found in CSV, defaulting to 0.0.", agent_name, col)

    # Sort by node then time to guarantee temporal ordering
    df = df.sort_values(["node_id", "cycle_id"]).reset_index(drop=True)
    logger.info("[%s] Data sorted by (node_id, cycle_id).", agent_name)

    # ---- Build transitions per node ----------------------------------------
    transitions: List[dict] = []
    nodes = df["node_id"].unique()
    logger.info("[%s] Building transitions for %d unique nodes.", agent_name, len(nodes))

    for node_id in nodes:
        node_df = df[df["node_id"] == node_id].reset_index(drop=True)
        n_rows  = len(node_df)
        logger.debug("[%s] Node %s | %d cycles.", agent_name, node_id, n_rows)

        for t in range(n_rows - 1):
            row_t      = node_df.iloc[t]
            row_t_next = node_df.iloc[t + 1]

            s      = row_t[features].values.astype(np.float32)
            s_next = row_t_next[features].values.astype(np.float32)

            transition = {
                "s":                 s,
                "s_next":            s_next,
                "done":              False,
                "is_attacker":       int(row_t["is_attacker"]),
                "blockchain_reject": int(row_t["blockchain_reject"]),
                "PDR_t":             float(row_t["PDR_t"]),
                "d_bar_t":           float(row_t["d_bar_t"]),
                "rho_recv":          float(row_t["rho_recv"]),
                "lambda_t":          float(row_t["lambda_t"]),
            }
            transitions.append(transition)

        # Terminal transition: last cycle — use s_t = s_{T}, s_next = zeros
        last_row = node_df.iloc[-1]
        terminal = {
            "s":                 last_row[features].values.astype(np.float32),
            "s_next":            np.zeros(len(features), dtype=np.float32),
            "done":              True,
            "is_attacker":       int(last_row["is_attacker"]),
            "blockchain_reject": int(last_row["blockchain_reject"]),
            "PDR_t":             float(last_row["PDR_t"]),
            "d_bar_t":           float(last_row["d_bar_t"]),
            "rho_recv":          float(last_row["rho_recv"]),
            "lambda_t":          float(last_row["lambda_t"]),
        }
        transitions.append(terminal)

    logger.info(
        "[%s] Transitions built | total=%d (non-terminal=%d, terminal=%d)",
        agent_name,
        len(transitions),
        sum(1 for t in transitions if not t["done"]),
        sum(1 for t in transitions if t["done"]),
    )
    return transitions
