"""
features_percycle.py — Computes per-cycle features without historical dependencies (Part 2).

Public API:
    add_percycle_features(df) -> pd.DataFrame
"""

import logging
import numpy as np
import pandas as pd

from bridge.config_bridge import DELAY_REF_FILE, LAMBDA_REF

logger = logging.getLogger("bridge")


def _load_d_ref() -> float:
    """
    Load the delay reference value d_ref_ms from delay_reference.csv.
    Falls back to a default value of 4.8767 if file reading fails.
    """
    try:
        d_ref_df = pd.read_csv(DELAY_REF_FILE)
        d_ref = float(d_ref_df['d_ref_ms'].iloc[0])
        logger.info("Loaded d_ref = %.4f ms from %s", d_ref, DELAY_REF_FILE)
        return d_ref
    except Exception as e:
        logger.error("Failed to load d_ref from %s, using fallback 4.8767: %s", DELAY_REF_FILE, e)
        return 4.8767


def add_percycle_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes and adds per-cycle state features to the joined DataFrame.

    Features computed:
        - FFc          : node_pdr normalized to [0, 1] range.
        - rho_recv     : inbound_ratio capped at 1.0, clipped to [0, 1].
        - dFF          : ff_deviation or sum_abs_ff_deviation, clipped at >= 0.
        - d_bar        : Mean hop delay sum / hop delay count.
        - DelayInfl    : d_bar normalized by baseline d_ref.
        - lambda_t     : Topology change rate (filled with cycle median if missing).
        - tau          : Trust score carried as-is.
        - SpoofDev_raw : Absolute reported-to-neighbor metric deviation.

    Args:
        df : Joined per-(cycle, node) DataFrame from Part 1.

    Returns:
        pd.DataFrame: Mutated copy containing the original columns plus the 8 new feature columns.
    """
    logger.info("=== add_percycle_features | rows=%d ===", len(df))
    df = df.copy()
    d_ref = _load_d_ref()

    # 1. FFc = node_pdr / 100.0 (converted to 0–1, clipped to [0,1])
    df['FFc'] = (df['node_pdr'] / 100.0).clip(0.0, 1.0)

    # 2. rho_recv = min(inbound_ratio, 1.0) (clipped to [0,1])
    df['rho_recv'] = df['inbound_ratio'].clip(0.0, 1.0)

    # 3. dFF = δFF per report Eq. 3.99 — maximum absolute per-flow deviation
    #         from the committed forwarding plan, aggregated across flows.
    #
    # abs_ff_deviation is computed per flow row in join.py (before _aggregate_to_node)
    # and MAX-aggregated via AGG_MAX in config_bridge.py.  At this point it is
    # already: (a) absolute-valued, (b) the worst-case flow for this node, and
    # (c) correctly non-negative — so no further .abs() or aggregation is needed.
    #
    # Fallback to sum_abs_ff_deviation for data loaded before this fix was applied.
    if 'abs_ff_deviation' in df.columns:
        df['dFF'] = df['abs_ff_deviation'].fillna(
            df['sum_abs_ff_deviation'].abs()
        ).clip(lower=0.0)
    else:
        # Fallback for older joined data that predates the abs_ff_deviation fix
        df['dFF'] = df['sum_abs_ff_deviation'].abs().clip(lower=0.0)

    # 4. d_bar = hop_delay_sum_ms / hop_delay_count (guard against count <= 0)
    invalid_count = df['hop_delay_count'].isna() | (df['hop_delay_count'] <= 0)
    df['d_bar'] = np.where(
        ~invalid_count,
        df['hop_delay_sum_ms'] / df['hop_delay_count'],
        np.nan
    )
    nan_d_bar_count = df['d_bar'].isna().sum()
    logger.info("d_bar calculation: %d values set to NaN (due to zero/missing hop_delay_count)", nan_d_bar_count)

    # 5. DelayInfl = d_bar / d_ref
    df['DelayInfl'] = df['d_bar'] / d_ref

    # 6. lambda_t = carry from joined table. Fill missing with cycle median
    lambda_t_missing_before = df['lambda_t'].isna().sum()
    if lambda_t_missing_before > 0:
        cycle_medians = df.groupby('cycle_id')['lambda_t'].transform('median')
        global_median = df['lambda_t'].median()
        if pd.isna(global_median):
            global_median = 0.0
        df['lambda_t'] = df['lambda_t'].fillna(cycle_medians).fillna(global_median)
    lambda_t_missing_after = df['lambda_t'].isna().sum()
    filled_count = lambda_t_missing_before - lambda_t_missing_after
    logger.info("lambda_t missing values: %d before -> %d after (%d filled using cycle median)",
                lambda_t_missing_before, lambda_t_missing_after, filled_count)

    # 6b. lambda_t_norm = min(lambda_t / LAMBDA_REF, 1.0)  (normalised to 0–1)
    df['lambda_t_norm'] = (df['lambda_t'] / LAMBDA_REF).clip(upper=1.0)

    # 7. tau = trust_score
    if 'trust_score' in df.columns:
        df['tau'] = df['trust_score']
    else:
        df['tau'] = 1.0

    # 8. SpoofDev_raw = |reported_metric - neighbor_metric|
    if 'spoof_dev' in df.columns:
        df['SpoofDev_raw'] = df['spoof_dev']
    elif 'reported_metric' in df.columns and 'neighbor_metric' in df.columns:
        df['SpoofDev_raw'] = (df['reported_metric'] - df['neighbor_metric']).abs()
    else:
        df['SpoofDev_raw'] = 0.0

    # Log summary statistics of each new feature
    new_features = ['FFc', 'rho_recv', 'dFF', 'd_bar', 'DelayInfl', 'lambda_t', 'lambda_t_norm', 'tau', 'SpoofDev_raw']
    for feat in new_features:
        feat_min = df[feat].min()
        feat_max = df[feat].max()
        feat_mean = df[feat].mean()
        logger.info("Feature '%s' -> min: %.4f, max: %.4f, mean: %.4f", feat, feat_min, feat_max, feat_mean)

    return df
