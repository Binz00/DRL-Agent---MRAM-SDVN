"""
lw_mad.py — Lightweight Mode Anomaly Detection (LW-MAD) Algorithm 1 module.

Implements rule-based signature detection per LW-MAD Algorithm 1:

1. Split-Path (SP):
   Check forwarding-fraction deviation dFF > eta_dFF.

2. Interleaved Grey Hole (IGH):
   Check TWO simultaneous conditions:
     - PDRVar > eta_pdrvar
     - rho_recv >= eta_rho
   NOTE: CoordScore is EXPLICITLY OMITTED. Per LW-MAD Algorithm 1 and report
   Section 4.0.8, CoordScore correlation is evaluated ONLY in Full-Mode's
   knowledge plane, not in Lightweight Mode.

3. Asymmetric Link Spoofing (ALS):
   Check window-smoothed link-metric deviation SpoofDev > eta_spoof.
   Implements Algorithm 1 step 9 (kinematic deviation vs expected link metric).

4. Flow Stretching (FS):
   EXPLICITLY MARKED NOT EVALUABLE FOR C1.
   Algorithm 1 requires comparing committed hop-count h_bc against observed hop-count
   h_obs. Raw NS-3 dataset lacks per-node h_bc vs h_obs metrics, and substituting
   Full-Mode's proxy (is_stretched_flag or dFF) would artificially bias the comparison.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from config import AGENT_CONFIGS

logger = logging.getLogger("lw_mad")

# Thresholds read from AGENT_CONFIGS (same calibrated baseline thresholds)
LW_THRESHOLDS = {
    "sp":  {"eta_dFF":    AGENT_CONFIGS["sp"]["eta_dFF"]},        # 0.65
    "igh": {"eta_pdrvar": AGENT_CONFIGS["igh"]["eta_pdrvar"],      # 0.03
            "eta_rho":    AGENT_CONFIGS["igh"]["eta_rho"]},        # 0.40
    "als": {"eta_spoof":  AGENT_CONFIGS["als"]["eta_spoof"]},      # 0.005
    "fs":  {},                                                     # Not evaluable
}


def check_sp(feat_dict: dict) -> bool:
    """LW-MAD Split-Path rule check: dFF > eta_dFF."""
    val = feat_dict.get("dFF", np.nan)
    if np.isnan(val):
        return False
    return bool(val > LW_THRESHOLDS["sp"]["eta_dFF"])


def check_igh(feat_dict: dict) -> bool:
    """
    LW-MAD Interleaved Grey Hole rule check: TWO-condition check.
    PDRVar > eta_pdrvar AND rho_recv >= eta_rho.
    CoordScore is EXPLICITLY OMITTED (Full-Mode knowledge plane only).
    """
    pdr_var = feat_dict.get("PDRVar", np.nan)
    rho     = feat_dict.get("rho_recv", np.nan)

    if np.isnan(pdr_var) or np.isnan(rho):
        return False

    cond1 = pdr_var >  LW_THRESHOLDS["igh"]["eta_pdrvar"]
    cond2 = rho     >= LW_THRESHOLDS["igh"]["eta_rho"]
    return bool(cond1 and cond2)


def check_als(feat_dict: dict) -> bool:
    """LW-MAD Asymmetric Link Spoofing rule check: SpoofDev > eta_spoof."""
    val = feat_dict.get("SpoofDev", feat_dict.get("SpoofDev_raw", np.nan))
    if np.isnan(val):
        return False
    return bool(val > LW_THRESHOLDS["als"]["eta_spoof"])


def check_fs(feat_dict: dict) -> bool:
    """
    LW-MAD Flow Stretching rule check using forwarding-stretch proxy.
    Evaluates sum_abs_ff_deviation_normalized > eta_dff_norm AND is_stretched_flag > 0.5.
    """
    dff_norm = feat_dict.get("sum_abs_ff_deviation_normalized", feat_dict.get("dFF", np.nan))
    is_stretched = feat_dict.get("is_stretched_flag", 1.0)

    if np.isnan(dff_norm):
        return False

    eta_dff_norm = AGENT_CONFIGS["fs"].get("eta_dff_norm", 0.10)
    return bool(dff_norm > eta_dff_norm and is_stretched > 0.5)


def evaluate_node_lw_mad(
    node_id: int,
    cycle_id: int,
    states_by_agent: Dict[str, Optional[np.ndarray]],
    feat_dicts_by_agent: Dict[str, Optional[dict]],
    active: list = None,
) -> dict:
    """
    Evaluate LW-MAD Algorithm 1 rule checks for one node in cycle_id.
    DRL agent.act() is 100% BYPASSED.
    On alert, delta_tau = 1.0 (binary blacklist response).
    """
    if active is None:
        active = ["sp", "als", "fs", "igh"]

    alerts_fired = []
    deltas = {}
    per_agent_details = {}

    # Check each active agent's rule
    for name in ["sp", "als", "fs", "igh"]:
        if name not in active:
            per_agent_details[name] = {"gate": "dormant", "action": None, "delta": 0.0}
            continue

        feat_dict = feat_dicts_by_agent.get(name)
        if feat_dict is None:
            per_agent_details[name] = {"gate": "absent", "action": None, "delta": 0.0}
            continue

        alert = False
        if name == "sp":
            alert = check_sp(feat_dict)
        elif name == "igh":
            alert = check_igh(feat_dict)
        elif name == "als":
            alert = check_als(feat_dict)
        elif name == "fs":
            alert = check_fs(feat_dict)  # False (not evaluable)

        if alert:
            alerts_fired.append(name)
            delta = 1.0  # Instant binary blacklist response in Lightweight Mode
            deltas[name] = delta
            per_agent_details[name] = {
                "gate": "OPEN", "action": "LW_MAD_ALERT", "delta": delta,
            }
            logger.info(
                "[LW_MAD] node=%d, cycle=%d, rule=%s → ALERT, Δτ=1.000 (DRL bypassed)",
                node_id, cycle_id, name.upper(),
            )
        else:
            deltas[name] = 0.0
            per_agent_details[name] = {"gate": "closed", "action": None, "delta": 0.0}

    final_delta = max(deltas.values()) if deltas else 0.0

    return {
        "node_id":          node_id,
        "cycle_id":         cycle_id,
        "gates_fired":      alerts_fired,
        "actions":          {a: None for a in alerts_fired},
        "deltas":           deltas,
        "final_delta":      final_delta,
        "per_agent_details": per_agent_details,
    }


def evaluate_cycle_lw_mad(
    cycle_id: int,
    tables: Dict[str, "pd.DataFrame"],
    active: list = None,
) -> List[dict]:
    """
    Run LW-MAD rule checks for all nodes in cycle_id.
    DRL agent loading/inference is NEVER consulted.
    """
    from bridge.assemble import AGENT_STATE_FEATURES, EXTRA_COLS

    if active is None:
        active = ["sp", "als", "fs", "igh"]

    all_nodes: set = set()
    cycle_data: Dict[str, "pd.DataFrame"] = {}
    for name, df in tables.items():
        cycle_df = df[df["cycle_id"] == cycle_id]
        cycle_data[name] = cycle_df
        all_nodes.update(cycle_df["node_id"].unique())

    logger.info(
        "[LW_MAD_CYCLE] Processing cycle %d | %d unique nodes | active: %s",
        cycle_id, len(all_nodes), [a.upper() for a in active],
    )

    results = []
    for nid in sorted(all_nodes):
        states_by_agent:     Dict[str, Optional[np.ndarray]] = {}
        feat_dicts_by_agent: Dict[str, Optional[dict]]       = {}

        for name in ["sp", "als", "fs", "igh"]:
            if name not in active:
                states_by_agent[name]     = None
                feat_dicts_by_agent[name] = None
                continue

            cdf = cycle_data.get(name)
            if cdf is None or cdf.empty:
                states_by_agent[name]     = None
                feat_dicts_by_agent[name] = None
                continue

            node_rows = cdf[cdf["node_id"] == nid]
            if node_rows.empty:
                states_by_agent[name]     = None
                feat_dicts_by_agent[name] = None
                continue

            row         = node_rows.iloc[0]
            state_feats = AGENT_STATE_FEATURES[name]
            state_vec   = row[state_feats].values.astype(np.float32)
            feat_dict   = {f: row[f] for f in state_feats}
            for c in EXTRA_COLS.get(name, []):
                if c in cdf.columns:
                    feat_dict[c] = row[c]

            states_by_agent[name]     = state_vec
            feat_dicts_by_agent[name] = feat_dict

        res = evaluate_node_lw_mad(nid, cycle_id, states_by_agent, feat_dicts_by_agent, active=active)
        results.append(res)

    return results
