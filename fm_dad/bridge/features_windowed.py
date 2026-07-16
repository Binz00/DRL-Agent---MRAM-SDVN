"""
features_windowed.py — Windowed features requiring cross-cycle history (Part 3).

Computes three features that depend on a trailing window of past cycles:

    PDRVar     — Variance of node i's FFc series over the last W* cycles.
                 (Eq. 3.7 context: measures forwarding-fraction instability.)
    CoordScore — Double-maximum normalised cross-correlation over partner
                 nodes j ≠ i AND lag offsets τ ∈ [1, W*].  (Eq. 3.22.)
    SpoofDev   — Windowed mean of SpoofDev_raw over the last W* cycles.
                 (Smooths the per-cycle ALS deviation.)

Design for live compatibility:
    - Cycles are processed in ascending order.
    - Per-node history is stored in a dict (node_id → list of per-cycle records).
    - At each cycle, only the trailing W* entries are used.
    - The same code can later be called cycle-by-cycle in live mode.
"""

import logging
from collections import defaultdict
from typing import Dict, List, Set

import numpy as np
import pandas as pd

from bridge.config_bridge import (
    WINDOW_CANDIDATES,
    LAMBDA_W_HIGH,
    LAMBDA_W_MED,
)

logger = logging.getLogger("bridge")


# ---------------------------------------------------------------------------
# Window selection
# ---------------------------------------------------------------------------

def select_window(lambda_t_norm_cycle: float) -> int:
    """
    Map normalised network-wide lambda_t_norm (0–1) to W* ∈ {10, 15, 20}.

    Higher topology-change rate (higher lambda_t_norm) → shorter window because
    older observations become stale faster.  Lower lambda_t_norm → longer window
    because the network is stable and more history is useful.

    Thresholds are read from config_bridge.py (LAMBDA_W_HIGH, LAMBDA_W_MED),
    expressed on the normalised 0–1 scale.

    Args:
        lambda_t_norm_cycle: Median lambda_t_norm across all nodes in the cycle.

    Returns:
        int: Selected window size W*.

    Implements: Dynamic W* selection per Eq. 3.7 context.
    """
    if lambda_t_norm_cycle >= LAMBDA_W_HIGH:
        return WINDOW_CANDIDATES[0]   # 10
    elif lambda_t_norm_cycle >= LAMBDA_W_MED:
        return WINDOW_CANDIDATES[1]   # 15
    else:
        return WINDOW_CANDIDATES[2]   # 20


# ---------------------------------------------------------------------------
# CoordScore helper — Eq. 3.22 double-maximum cross-correlation
# ---------------------------------------------------------------------------

def _compute_coord_score(
    node_id: int,
    window_cycles: List[int],
    ffc_history: Dict[int, Dict[int, float]],
    active_nodes: Set[int],
) -> float:
    """
    Compute CoordScore for node i: the double maximum of normalised
    cross-correlation over all partner nodes j ≠ i AND lag offsets τ ∈ [1, W*].

    For each (j, τ) pair:
        - Align FFc series of i and j by cycle index within the window.
        - Shift by lag τ:  x = FFc_i[τ:]  vs  y = FFc_j[:-τ].
        - Drop positions where either value is NaN.
        - If fewer than 2 valid overlapping points, skip (correlation undefined).
        - Compute |Pearson r| between the valid segments.

    CoordScore_i = max over all (j, τ) of |r|.

    Args:
        node_id       : Target node i.
        window_cycles : Ordered list of cycle IDs in the current window.
        ffc_history   : node_id → {cycle_id: FFc} mapping for all nodes.
        active_nodes  : Set of node IDs active in the current cycle.

    Returns:
        float: CoordScore ∈ [0, 1].  Returns 0.0 if the window is too short
               or no valid partner-lag pair yields a correlation.

    Implements: Eq. 3.22 — double-maximum normalised cross-correlation.
    """
    W = len(window_cycles)
    if W < 3:
        # Need at least 3 time-points so that at lag=1 we still have ≥2 overlap
        return 0.0

    # Build node i's FFc vector over the window
    ffc_i = np.array([
        ffc_history.get(node_id, {}).get(c, np.nan) for c in window_cycles
    ])

    max_abs_corr = 0.0

    for j in active_nodes:
        if j == node_id:
            continue

        j_hist = ffc_history.get(j)
        if j_hist is None:
            continue

        ffc_j = np.array([j_hist.get(c, np.nan) for c in window_cycles])

        # Scan all lags τ ∈ [1, W-1]  (W-1 because at τ=W overlap is 0)
        for tau in range(1, W):
            x = ffc_i[tau:]       # length W - tau
            y = ffc_j[: W - tau]  # length W - tau

            valid = ~(np.isnan(x) | np.isnan(y))
            n_valid = valid.sum()
            if n_valid < 2:
                continue

            x_v = x[valid]
            y_v = y[valid]

            # Zero-variance guard — correlation is undefined
            if np.std(x_v) < 1e-12 or np.std(y_v) < 1e-12:
                continue

            r = np.corrcoef(x_v, y_v)[0, 1]
            if np.isnan(r):
                continue
            abs_r = abs(r)
            if abs_r > max_abs_corr:
                max_abs_corr = abs_r

    return float(max_abs_corr)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def add_windowed_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add PDRVar, CoordScore, and SpoofDev columns using a per-node history buffer.

    Processing order:
        1. Sort cycles ascending.
        2. For each cycle:
           a. Select W* from the cycle-wide median lambda_t.
           b. Ingest the current cycle's data into the per-node history buffer.
           c. For each node, extract the trailing W* entries from its buffer.
           d. Compute PDRVar (variance of FFc), SpoofDev (mean of SpoofDev_raw).
           e. Compute CoordScore (Eq. 3.22) using FFc histories of all active nodes.
        3. Log diagnostics: W* per cycle, full vs partial window counts,
           min/max/mean of each new feature.

    The history buffer is a plain dict and can be persisted between calls
    for future live-mode operation.

    Args:
        df: DataFrame from Part 2 with per-cycle features already computed.

    Returns:
        pd.DataFrame: Input DataFrame with PDRVar, CoordScore, SpoofDev added.

    Implements: PDRVar (Eq. 3.7 context), CoordScore (Eq. 3.22),
                SpoofDev (windowed mean of per-cycle ALS deviation).
    """
    logger.info("=== add_windowed_features | rows=%d ===", len(df))
    df = df.copy()

    # Initialise output columns
    df["PDRVar"]     = np.nan
    df["CoordScore"] = np.nan
    df["SpoofDev"]   = np.nan

    # Per-node history buffers
    # node_history: node_id → [(cycle_id, FFc, SpoofDev_raw), ...]
    node_history: Dict[int, List] = defaultdict(list)
    # ffc_history:  node_id → {cycle_id: FFc}  (fast lookup for cross-corr)
    ffc_history:  Dict[int, Dict[int, float]] = defaultdict(dict)

    cycles = sorted(df["cycle_id"].unique())
    min_w = min(WINDOW_CANDIDATES)

    if len(cycles) < min_w:
        logger.warning(
            "WINDOW WARNING: Only %d cycle(s) available, smallest W* = %d.  "
            "ALL windows will be partial — windowed features (PDRVar, CoordScore, "
            "SpoofDev) are statistically weak.  Collect more cycles for reliable "
            "detection.", len(cycles), min_w,
        )

    full_window_total  = 0
    partial_window_total = 0

    for cycle in cycles:
        cycle_mask = df["cycle_id"] == cycle
        cycle_idx  = df.index[cycle_mask]

        # ---- Network-wide lambda_t_norm → select W* ----
        lambda_t_norm_median = df.loc[cycle_mask, "lambda_t_norm"].median()
        W_star = select_window(lambda_t_norm_median)

        n_nodes_cycle = len(cycle_idx)
        logger.info(
            "  Cycle %d | lambda_t_norm_median=%.4f → W*=%d, nodes=%d",
            cycle, lambda_t_norm_median, W_star, n_nodes_cycle,
        )

        # ---- Step 1: Ingest current cycle into history ----
        for idx in cycle_idx:
            nid   = int(df.at[idx, "node_id"])
            ffc_v = float(df.at[idx, "FFc"]) if not pd.isna(df.at[idx, "FFc"]) else np.nan
            sp_v  = float(df.at[idx, "SpoofDev_raw"]) if not pd.isna(df.at[idx, "SpoofDev_raw"]) else np.nan

            node_history[nid].append((cycle, ffc_v, sp_v))
            ffc_history[nid][cycle] = ffc_v

        # ---- Step 2: Compute per-node windowed features ----
        active_nodes: Set[int] = set()
        for idx in cycle_idx:
            active_nodes.add(int(df.at[idx, "node_id"]))

        full_window_cycle  = 0
        partial_window_cycle = 0

        for idx in cycle_idx:
            nid  = int(df.at[idx, "node_id"])
            hist = node_history[nid]

            # Trailing window of at most W* entries
            window = hist[-W_star:]
            w_len  = len(window)

            if w_len >= W_star:
                full_window_cycle += 1
            else:
                partial_window_cycle += 1

            # -- PDRVar: variance of FFc over the window --
            ffc_vals = [h[1] for h in window if not np.isnan(h[1])]
            if len(ffc_vals) >= 2:
                df.at[idx, "PDRVar"] = float(np.var(ffc_vals, ddof=0))
            else:
                df.at[idx, "PDRVar"] = 0.0

            # -- SpoofDev: mean of SpoofDev_raw over the window --
            sp_vals = [h[2] for h in window if not np.isnan(h[2])]
            if sp_vals:
                df.at[idx, "SpoofDev"] = float(np.mean(sp_vals))
            # else: stays NaN (ALS columns missing)

            # -- CoordScore: Eq. 3.22 double maximum --
            window_cycles = [h[0] for h in window]
            df.at[idx, "CoordScore"] = _compute_coord_score(
                nid, window_cycles, ffc_history, active_nodes,
            )

        full_window_total   += full_window_cycle
        partial_window_total += partial_window_cycle
        logger.info(
            "  Cycle %d | full_window=%d, partial_window=%d",
            cycle, full_window_cycle, partial_window_cycle,
        )

    # ---- Summary statistics ----
    logger.info(
        "Window totals: full=%d, partial=%d", full_window_total, partial_window_total,
    )
    for feat in ("PDRVar", "CoordScore", "SpoofDev"):
        col = df[feat]
        logger.info(
            "Feature '%s' → min=%.6f, max=%.6f, mean=%.6f",
            feat, col.min(), col.max(), col.mean(),
        )

    return df
