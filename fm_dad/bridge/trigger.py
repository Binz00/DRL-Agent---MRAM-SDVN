"""
trigger.py — Core inference pipeline: gate → agent → max-combine (Part 5).

Public API:
    load_agents()                       → dict of loaded DQNAgent instances
    process_node(node_id, cycle_id,
                 states_by_agent, agents) → result dict
    process_cycle(cycle_id, tables,
                  agents)                → list of result dicts

Gate logic:
    SP / ALS / FS: single-condition gates  (one feature compared against one
                   threshold from config.py eta_* values — all currently
                   placeholders pending grid search).
    IGH:           three-condition AND gate  per Algorithm 4 (IGH-DM, line 9)
                   and Definition 3 (Eqs. 3.19–3.21).  ALL three conditions
                   must hold simultaneously — the report explicitly states
                   "Any single condition alone is explicable by non-attack
                   causes" (Section 3.3.5).  Thresholds (eta_pdrvar, eta_rho,
                   eta_coord) are also placeholders pending grid search, but
                   the feature/condition mapping is now structurally correct
                   regardless of threshold values.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

import sys
_FM_DAD_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(_FM_DAD_DIR))

from agent import DQNAgent
from config import AGENT_CONFIGS, MODEL_FILES, SHARED_HP

logger = logging.getLogger("pipeline")

# ---------------------------------------------------------------------------
# GATE_CONDITIONS — multi-condition AND gates, one list of (feature, op, threshold)
# tuples per agent.  The gate for an agent fires (returns True) only when EVERY
# condition in its list holds.  Single-item lists behave identically to the old
# single-threshold approach.
#
# SP / ALS / FS: single condition — thresholds are eta_* placeholders from config.py
# IGH:           three simultaneous conditions per Definition 3, Eqs. 3.19–3.21,
#                Algorithm 4 line 9
# ---------------------------------------------------------------------------

# Type alias: each condition is (feature_name, operator_string, threshold_value)
Condition = Tuple[str, str, float]

GATE_CONDITIONS: Dict[str, List[Condition]] = {

    # SP — Selective Packet Dropping gate (Algorithm 3, Stage 2)
    # δFF = abs_ff_deviation > η_FF  (Eq. 3.x, placeholder threshold)
    "sp": [
        ("dFF", ">", AGENT_CONFIGS["sp"]["eta_dFF"]),
    ],

    # ALS — Asymmetric Link Spoofing gate
    # SpoofDev > η_spoof  (placeholder threshold)
    "als": [
        ("SpoofDev", ">", AGENT_CONFIGS["als"]["eta_spoof"]),
    ],

    # FS — Flow Stretching gate (Algorithm 3, delay variant)
    # DelayInfl > η_delay  (placeholder threshold)
    "fs": [
        ("DelayInfl", ">", AGENT_CONFIGS["fs"]["eta_delay"]),
    ],

    # IGH — Inter-flow Greedy Hoarding gate (Algorithm 4 line 9, Definition 3)
    #
    # ALL THREE conditions must hold simultaneously.
    # Report rationale: "Any single condition alone is explicable by non-attack
    # causes" (Section 3.3.5).
    #
    # Eq. 3.19 — sustained PDR variance confirms an ON/OFF dropping pattern
    "igh": [
        ("PDRVar",     ">",  AGENT_CONFIGS["igh"]["eta_pdrvar"]),
        # Eq. 3.20 — node is RECEIVING packets but choosing not to forward
        #   (rho_recv >= eta_rho rules out upstream victim / congestion explanation)
        ("rho_recv",   ">=", AGENT_CONFIGS["igh"]["eta_rho"]),
        # Eq. 3.21 — dropping is coordinated with another node, not an isolated fault
        ("CoordScore", ">",  AGENT_CONFIGS["igh"]["eta_coord"]),
    ],
}

# Action names for logging
ACTION_NAMES = ["a0", "a1", "a2", "a3", "a4"]


# ---------------------------------------------------------------------------
# Agent loading
# ---------------------------------------------------------------------------

def load_agents() -> Dict[str, DQNAgent]:
    """
    Load all four trained DQN agents from disk in eval mode (no exploration).

    Returns:
        Dict mapping agent name to a DQNAgent with loaded weights and epsilon=0.
    """
    agents: Dict[str, DQNAgent] = {}
    for name in ["sp", "als", "fs", "igh"]:
        cfg   = AGENT_CONFIGS[name]
        agent = DQNAgent(cfg, SHARED_HP, device="cpu")
        model_path = str(_FM_DAD_DIR / MODEL_FILES[name])
        agent.load(model_path)
        agent.main_net.eval()
        agent.epsilon = 0.0   # pure greedy — no exploration
        agents[name] = agent
        logger.info(
            "[LOAD] Agent '%s' loaded from %s (eval mode, eps=0)",
            name.upper(), model_path,
        )
    return agents


# ---------------------------------------------------------------------------
# Gate check — evaluates ALL conditions with AND logic
# ---------------------------------------------------------------------------

def _check_gate(agent_name: str, state_dict: dict) -> bool:
    """
    Apply the detection gate for one agent on a node's features.

    For SP / ALS / FS: single condition (one feature compared against one
        threshold).  The gate fires when that condition holds.

    For IGH: three simultaneous conditions (Definition 3, Eqs. 3.19–3.21,
        Algorithm 4 line 9).  ALL must hold — any single condition alone is
        explicable by non-attack causes (report, Section 3.3.5).

    Returns True only if every condition in GATE_CONDITIONS[agent_name] holds.
    Returns False immediately if any monitored feature value is NaN.

    Args:
        agent_name : One of 'sp', 'als', 'fs', 'igh'.
        state_dict : Feature name → value for this node.

    Returns:
        bool: True if the gate fires (agent should run inference).
    """
    for feat, op, thresh in GATE_CONDITIONS[agent_name]:
        val = state_dict.get(feat, np.nan)
        if np.isnan(val):
            logger.debug(
                "[GATE] agent=%s feature=%s is NaN → gate closed",
                agent_name.upper(), feat,
            )
            return False
        if op == ">" and not (val > thresh):
            return False
        if op == ">=" and not (val >= thresh):
            return False
    return True


# ---------------------------------------------------------------------------
# Single-node processing
# ---------------------------------------------------------------------------

def process_node(
    node_id: int,
    cycle_id: int,
    states_by_agent: Dict[str, Optional[np.ndarray]],
    feature_dicts_by_agent: Dict[str, Optional[dict]],
    agents: Dict[str, DQNAgent],
) -> dict:
    """
    Run the full pipeline for one node: gate → agent → max-combine.

    For each agent, if the gate fires, run argmax Q(s) to pick an action,
    then map the action to Δτ via the agent's delta table (Eq. 3.45).
    The final Δτ is the MAX across all agents whose gate fired (Eq. 3.38).

    Args:
        node_id               : The node being evaluated.
        cycle_id              : The current cycle.
        states_by_agent       : agent_name → state vector (np array) or None.
        feature_dicts_by_agent: agent_name → {feature: value} dict for gate checks.
        agents                : Loaded DQNAgent instances.

    Returns:
        dict with keys: node_id, cycle_id, gates_fired, actions, deltas,
                        final_delta, per_agent_details.
    """
    gates_fired       = []
    actions:   dict   = {}
    deltas:    dict   = {}
    per_agent_details: dict = {}

    for name in ["sp", "als", "fs", "igh"]:
        state    = states_by_agent.get(name)
        feat_dict = feature_dicts_by_agent.get(name)

        if state is None or feat_dict is None:
            per_agent_details[name] = {"gate": "absent", "action": None, "delta": 0.0}
            logger.info(
                "[GATE] node=%d, cycle=%d, agent=%s → ABSENT (no state vector)",
                node_id, cycle_id, name.upper(),
            )
            continue

        gate_open = _check_gate(name, feat_dict)

        if gate_open:
            gates_fired.append(name)
            action = agents[name].act(state, epsilon=0.0)
            delta  = agents[name].deltas[action]
            actions[name] = action
            deltas[name]  = delta
            per_agent_details[name] = {
                "gate": "OPEN", "action": ACTION_NAMES[action], "delta": delta,
            }
            logger.info(
                "[AGENT] node=%d, cycle=%d, agent=%s → gate=OPEN, action=%s, Δτ=%.3f",
                node_id, cycle_id, name.upper(), ACTION_NAMES[action], delta,
            )
        else:
            deltas[name] = 0.0
            per_agent_details[name] = {"gate": "closed", "action": None, "delta": 0.0}
            logger.info(
                "[GATE] node=%d, cycle=%d, agent=%s → gate=CLOSED (condition(s) not met)",
                node_id, cycle_id, name.upper(),
            )

            # IGH: log per-condition diagnostics so threshold calibration is
            # observable without re-running at DEBUG level in production.
            if name == "igh":
                for feat, op, thresh in GATE_CONDITIONS["igh"]:
                    val    = feat_dict.get(feat, float("nan"))
                    if np.isnan(val):
                        status = "FAIL (NaN)"
                    elif op == ">" :
                        status = "PASS" if val >  thresh else f"FAIL ({val:.4f} not > {thresh})"
                    else:  # ">="
                        status = "PASS" if val >= thresh else f"FAIL ({val:.4f} not >= {thresh})"
                    logger.debug(
                        "[GATE] IGH condition %s %s %.4f | node_val=%.4f | %s",
                        feat, op, thresh,
                        val if not np.isnan(val) else -1.0,
                        status,
                    )

    # Max-combine (Eq. 3.38)
    final_delta = max(deltas.values()) if deltas else 0.0
    logger.info(
        "[COMBINE] node=%d, cycle=%d → final Δτ=%.3f (gates fired: %s)",
        node_id, cycle_id, final_delta,
        [g.upper() for g in gates_fired] if gates_fired else "none",
    )

    return {
        "node_id":          node_id,
        "cycle_id":         cycle_id,
        "gates_fired":      gates_fired,
        "actions":          actions,
        "deltas":           deltas,
        "final_delta":      final_delta,
        "per_agent_details": per_agent_details,
    }


# ---------------------------------------------------------------------------
# Cycle-level processing
# ---------------------------------------------------------------------------

def process_cycle(
    cycle_id: int,
    tables: Dict[str, "pd.DataFrame"],
    agents: Dict[str, DQNAgent],
) -> List[dict]:
    """
    Run process_node for every node present in the given cycle.

    Collects the union of all node_ids across all agent tables for the cycle,
    then processes each node.

    Args:
        cycle_id : Which cycle to process.
        tables   : agent_name → DataFrame (from assemble_agent_tables).
        agents   : Loaded DQNAgent instances.

    Returns:
        List of result dicts, one per node.
    """
    import pandas as pd
    from bridge.assemble import AGENT_STATE_FEATURES

    all_nodes: set = set()
    cycle_data: Dict[str, "pd.DataFrame"] = {}
    for name, df in tables.items():
        cycle_df = df[df["cycle_id"] == cycle_id]
        cycle_data[name] = cycle_df
        all_nodes.update(cycle_df["node_id"].unique())

    logger.info(
        "[CYCLE] Processing cycle %d | %d unique nodes across %d agents",
        cycle_id, len(all_nodes), len(tables),
    )

    results = []
    for nid in sorted(all_nodes):
        states_by_agent:        Dict[str, Optional[np.ndarray]] = {}
        feat_dicts_by_agent:    Dict[str, Optional[dict]]       = {}

        for name in ["sp", "als", "fs", "igh"]:
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

            states_by_agent[name]     = state_vec
            feat_dicts_by_agent[name] = feat_dict

        result = process_node(nid, cycle_id, states_by_agent, feat_dicts_by_agent, agents)
        results.append(result)

    return results


# ---------------------------------------------------------------------------
# Unit test — IGH gate AND semantics (run directly: python3 trigger.py)
# ---------------------------------------------------------------------------

def _test_igh_gate() -> None:
    """
    CHECK 7 — IGH gate AND semantics (Definition 3, Eqs. 3.19–3.21).

    Five cases, all testing that the three-condition AND gate behaves correctly:

      Case A: all three conditions pass          → gate fires   (True)
      Case B: only PDRVar fails                  → gate closed  (False)
      Case C: only rho_recv fails                → gate closed  (False)
      Case D: only CoordScore fails              → gate closed  (False)
      Case E: any feature is NaN                 → gate closed  (False)
    """
    # Build threshold references
    eta_pdrvar = AGENT_CONFIGS["igh"]["eta_pdrvar"]  # 0.05
    eta_rho    = AGENT_CONFIGS["igh"]["eta_rho"]     # 0.50
    eta_coord  = AGENT_CONFIGS["igh"]["eta_coord"]   # 0.50

    # Values that satisfy each condition individually
    good_pdrvar = eta_pdrvar + 0.10   # clearly above threshold
    good_rho    = eta_rho    + 0.10
    good_coord  = eta_coord  + 0.10

    # Values that fail each condition individually
    bad_pdrvar  = eta_pdrvar - 0.01   # just below
    bad_rho     = eta_rho    - 0.01
    bad_coord   = eta_coord  - 0.01

    cases = [
        # (description,                                 pdrvar,     rho_recv, coord_score, expected)
        ("Case A — all pass",                           good_pdrvar, good_rho, good_coord, True),
        ("Case B — PDRVar fails",                       bad_pdrvar,  good_rho, good_coord, False),
        ("Case C — rho_recv fails",                     good_pdrvar, bad_rho,  good_coord, False),
        ("Case D — CoordScore fails",                   good_pdrvar, good_rho, bad_coord,  False),
        ("Case E — NaN feature (PDRVar)",               float("nan"),good_rho, good_coord, False),
    ]

    all_passed = True
    print("=" * 60)
    print("CHECK 7 — IGH gate AND semantics")
    print("=" * 60)
    for desc, pdrvar, rho, coord, expected in cases:
        feat_dict = {
            "PDRVar":     pdrvar,
            "rho_recv":   rho,
            "CoordScore": coord,
            # Other features in state vector — not used by gate conditions
            "dFF": 0.0, "tau": 1.0, "d_bar": 5.0, "lambda_t_norm": 0.0, "FFc": 1.0,
        }
        result = _check_gate("igh", feat_dict)
        status = "PASS" if result == expected else "FAIL"
        if status == "FAIL":
            all_passed = False
        print(f"  [{status}] {desc}")
        if status == "FAIL":
            print(f"         expected={expected}, got={result}")

    print()
    if all_passed:
        print("✅  All IGH gate AND-semantics cases passed.")
    else:
        print("❌  Some cases FAILED — check GATE_CONDITIONS logic.")
    print("=" * 60)


if __name__ == "__main__":
    # Run the IGH gate unit test when executed directly
    logging.basicConfig(level=logging.WARNING)   # suppress INFO during test
    _test_igh_gate()
