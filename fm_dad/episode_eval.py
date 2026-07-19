"""
episode_eval.py — Trust-trajectory evaluator for the MCC-based reward term.

Simulates the full deployed pipeline (gate → argmax Q → MAX-combine → trust
accumulation → blacklist) over all cycles, using one live agent (under training)
plus three frozen agents loaded from their latest fine-tuned checkpoints.

Key design constraints (from the supervisor's implementation prompt):
  - Gate logic is IMPORTED from trigger.py (_check_gate, GATE_CONDITIONS).
    No second implementation of gate logic allowed (single source of truth).
  - Tables are loaded from data/agent_inputs/*_state.csv  — same CSVs as the
    validator (not regenerated).
  - Ground truth is the UNION across all node_attack_ground_truth_*.csv files
    (same logic as validate_pipeline.py).
  - Trust initialised to 1.0 per node; decremented by MAX-combined delta each
    cycle; clamped to [0, 1].
  - Blacklisting: trust < tau_min  → node is blacklisted.  Blacklisted nodes
    continue to be processed (same as the validator's min_trust semantics).
  - live agent is run with epsilon=0 and torch.no_grad() (eval mode). Train mode
    is restored after.

Usage:
    python3 episode_eval.py --self-test          # run oracle check
    python3 episode_eval.py --self-test --tau 0.3

Public API:
    evaluate_policy_epoch(agent_name, live_agent, frozen_agents,
                          tables, ground_truth, tau_min) → EpochOutcome
    load_tables()   → dict of DataFrames (reusable)
    load_gt()       → pd.DataFrame (union ground truth)
    load_frozen_agents(exclude=None) → dict of DQNAgent (fine-tuned checkpoints)
"""

from __future__ import annotations

import argparse
import glob
import logging
import math
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_FM_DAD_DIR = Path(__file__).parent
sys.path.insert(0, str(_FM_DAD_DIR))

from agent import DQNAgent
from config import (
    AGENT_CONFIGS,
    FINETUNE_DATA_FILES,
    FINETUNE_HP,
    FINETUNE_MODEL_FILES,
    SHARED_HP,
    get_logger,
)
# Import gate logic from trigger.py — single source of truth (do NOT reimplement)
from bridge.trigger import _check_gate, GATE_CONDITIONS
from bridge.assemble import AGENT_STATE_FEATURES, EXTRA_COLS

logger = get_logger("episode_eval")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_RAW_CSV_DIR = _FM_DAD_DIR / "data" / "raw_csvs"
_AGENT_INPUTS_DIR = _FM_DAD_DIR / "data" / "agent_inputs"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class EpochOutcome:
    """
    Result of one full trust-trajectory evaluation pass.

    counts        : (tp, fp, fn, tn) for variant X
    mcc           : float  MCC^X for this evaluation
    node_outcomes : (cycle_id, node_id) → {
                        "gate_fired": bool,
                        "action":     int or None,
                        "outcome":    "TP"|"FP"|"FN"|"TN"|"EXCLUDED"
                    }
                    Records the live agent's contribution only, per cycle visit.
                    NOTE: a node may appear in multiple cycles; each gets an entry.
    max_contributor: set of node_ids where the live agent's delta was the MAX
                     at least once across the trajectory  (used for D_i computation).
    node_deltas    : node_id → list of (live_delta, max_delta_all) per cycle,
                     used for the MAX-contribution approximation.
    """
    counts:         Tuple[int, int, int, int]
    mcc:            float
    node_outcomes:  Dict[Tuple[int, int], dict] = field(default_factory=dict)
    max_contributor: set = field(default_factory=set)
    node_deltas:    Dict[int, list] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# MCC
# ---------------------------------------------------------------------------

def mcc_from_counts(tp: int, fp: int, fn: int, tn: int) -> float:
    """Equation 4.1 — Matthews Correlation Coefficient."""
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return ((tp * tn) - (fp * fn)) / denom if denom > 0 else 0.0


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_tables() -> Dict[str, pd.DataFrame]:
    """
    Load the four agent state tables from data/agent_inputs/*_state.csv.
    These are the same CSVs validate_pipeline.py consumes (via pipeline_penalties.csv
    which is produced by bridge/run_pipeline.py from these inputs).
    """
    tables: Dict[str, pd.DataFrame] = {}
    for name in ("sp", "als", "fs", "igh"):
        path = _AGENT_INPUTS_DIR / f"{name}_state.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"[episode_eval] Agent input CSV not found: {path}\n"
                "Run: python3 bridge/run_pipeline.py  to generate it."
            )
        df = pd.read_csv(path)
        tables[name] = df
        logger.info("[load_tables] %s loaded: %d rows, cols=%s", name, len(df), list(df.columns))
    return tables


def load_gt() -> pd.DataFrame:
    """
    Build union ground truth from all node_attack_ground_truth_*.csv files.
    Mirrors validate_pipeline.py's Step 2 exactly.

    Returns a DataFrame with columns: node_id, is_attacker, attack_type.
    """
    gt_files = sorted(
        glob.glob(str(_RAW_CSV_DIR / "node_attack_ground_truth_*.csv")),
        key=lambda p: int(re.search(r"_(\d+)\.csv$", p).group(1)),
    )
    if not gt_files:
        raise FileNotFoundError(
            f"[episode_eval] No ground truth files in {_RAW_CSV_DIR}"
        )

    gt_all = pd.concat([pd.read_csv(f) for f in gt_files], ignore_index=True)
    gt_all.columns = gt_all.columns.str.strip()

    def _resolve(group):
        is_att = int(group["is_attacker"].max())
        if is_att:
            types = group.loc[group["is_attacker"] == 1, "attack_type"].unique()
            atype = types[0]
        else:
            atype = "NONE"
        return pd.Series({"is_attacker": is_att, "attack_type": atype})

    gt_base = gt_all.groupby("node_id", group_keys=False).apply(_resolve).reset_index()
    logger.info(
        "[load_gt] Union across %d cycles | attackers=%d | honest=%d",
        len(gt_files),
        int(gt_base["is_attacker"].sum()),
        int((gt_base["is_attacker"] == 0).sum()),
    )
    return gt_base


def load_frozen_agents(exclude: Optional[str] = None) -> Dict[str, DQNAgent]:
    """
    Load the other three agents from their FINETUNE_MODEL_FILES checkpoints.
    These must be the fine-tuned versions (+0.7260 set), not synthetic ones.

    Args:
        exclude : agent name to skip (the live one, currently being trained).

    Returns:
        Dict mapping agent name → DQNAgent (eval mode, eps=0, no_grad not set here
        — callers use torch.no_grad() when calling act()).
    """
    agents: Dict[str, DQNAgent] = {}
    for name in ("sp", "als", "fs", "igh"):
        if name == exclude:
            continue
        cfg   = AGENT_CONFIGS[name]
        agent = DQNAgent(cfg, FINETUNE_HP, device="cpu")

        ft_path  = _FM_DAD_DIR / FINETUNE_MODEL_FILES[name]
        syn_path = _FM_DAD_DIR / f"models/{name}.pt"

        if ft_path.exists():
            model_path = str(ft_path)
            mode_str   = "fine-tuned"
        elif syn_path.exists():
            model_path = str(syn_path)
            mode_str   = "synthetic (fine-tuned not found)"
            logger.warning(
                "[load_frozen_agents] %s: fine-tuned checkpoint not found (%s), "
                "falling back to synthetic (%s). Self-test may not match.",
                name.upper(), ft_path, syn_path,
            )
        else:
            raise FileNotFoundError(
                f"[episode_eval] No checkpoint found for '{name}'.\n"
                f"Checked: {ft_path}, {syn_path}"
            )

        # Assert the file mtime is available (confirms it was actually loaded)
        mtime = os.path.getmtime(model_path)
        agent.load(model_path)
        agent.main_net.eval()
        agent.epsilon = 0.0
        agents[name] = agent
        logger.info(
            "[load_frozen_agents] Loaded %s from %s (%s) | mtime=%.0f",
            name.upper(), model_path, mode_str, mtime,
        )
    return agents


# ---------------------------------------------------------------------------
# Core evaluator
# ---------------------------------------------------------------------------

def evaluate_policy_epoch(
    agent_name:    str,
    live_agent:    DQNAgent,
    frozen_agents: Dict[str, DQNAgent],
    tables:        Dict[str, pd.DataFrame],
    ground_truth:  pd.DataFrame,
    tau_min:       float,
) -> EpochOutcome:
    """
    Simulate the full deployed pipeline for variant X = agent_name.

    Mirrors validate_pipeline.py's classification semantics exactly:
      TP^X : X-attacker AND ends blacklisted
      FN^X : X-attacker AND does NOT end blacklisted
      FP^X : honest node (not any attacker) AND ends blacklisted
      TN^X : honest node AND not blacklisted
      Other-type attackers: EXCLUDED from this variant's FP/TN counts.

    Trust accumulation:
      - Init = 1.0 for every node
      - Per cycle: delta = MAX across all agents whose gate fires
      - trust = max(0.0, trust - delta)
      - Blacklisted when trust < tau_min  (same as validate_pipeline)
      - Blacklisted nodes continue to be processed (min_trust semantics)

    Gate logic:
      - Uses trigger.py's _check_gate() and GATE_CONDITIONS (single source of truth)
      - Feat_dict built per AGENT_STATE_FEATURES + EXTRA_COLS  (same as process_cycle)
      - live_agent: epsilon=0, torch.no_grad()  — train mode restored after
      - frozen_agents: epsilon=0, eval mode already set

    Args:
        agent_name    : The variant being evaluated (e.g. "fs").
        live_agent    : The in-training DQNAgent (greedy, no_grad during act).
        frozen_agents : Other 3 agents from fine-tuned checkpoints.
        tables        : Dict of DataFrames from load_tables().
        ground_truth  : Union GT DataFrame (node_id, is_attacker, attack_type).
        tau_min       : Blacklist threshold.

    Returns:
        EpochOutcome with counts, mcc, node_outcomes, max_contributor, node_deltas.
    """
    # Build combined agent dict: live + frozen
    all_agents: Dict[str, DQNAgent] = dict(frozen_agents)
    all_agents[agent_name] = live_agent

    # Put live agent into eval mode + no_grad; remember whether it was training
    was_training = live_agent.main_net.training
    live_agent.main_net.eval()

    # Determine all cycles (sorted)
    all_cycles: list = []
    for df in tables.values():
        if "cycle_id" in df.columns:
            all_cycles.extend(df["cycle_id"].unique().tolist())
    all_cycles = sorted(set(all_cycles))

    # Determine all nodes across all tables
    all_nodes: set = set()
    for df in tables.values():
        if "node_id" in df.columns:
            all_nodes.update(df["node_id"].unique().tolist())
    all_nodes_sorted = sorted(all_nodes)

    # ---------------------------------------------------------------------------
    # Trust state: initialised to 1.0 for every node
    # (matches validate_pipeline.py — cycle-1 rows in pipeline_penalties start at 1.0)
    # ---------------------------------------------------------------------------
    trust: Dict[int, float] = {nid: 1.0 for nid in all_nodes_sorted}

    # Per-node tracking for D_i computation
    node_max_deltas: Dict[int, list] = {nid: [] for nid in all_nodes_sorted}  # list of (live_delta, max_all)

    # Per-(cycle_id, node_id) outcome record for live agent
    node_outcomes: Dict[Tuple[int, int], dict] = {}

    # ---------------------------------------------------------------------------
    # Cycle loop — process cycles in temporal order
    # ---------------------------------------------------------------------------
    with torch.no_grad():
        for cycle_id in all_cycles:
            # Build cycle-level DataFrames for each agent
            cycle_data: Dict[str, pd.DataFrame] = {}
            cycle_nodes: set = set()
            for name, df in tables.items():
                cdf = df[df["cycle_id"] == cycle_id]
                cycle_data[name] = cdf
                cycle_nodes.update(cdf["node_id"].unique().tolist())

            # Process each node present in this cycle
            for nid in sorted(cycle_nodes):
                # Build state vector + feat_dict for each agent
                states_by_agent:    Dict[str, Optional[np.ndarray]] = {}
                feat_dicts_by_agent: Dict[str, Optional[dict]]      = {}

                for name in ("sp", "als", "fs", "igh"):
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

                # Gate → act → delta for each agent
                deltas:    Dict[str, float] = {}
                live_gate_fired = False
                live_action     = None
                live_delta      = 0.0

                for name in ("sp", "als", "fs", "igh"):
                    state    = states_by_agent.get(name)
                    feat_dict = feat_dicts_by_agent.get(name)

                    if state is None or feat_dict is None:
                        deltas[name] = 0.0
                        continue

                    gate_open = _check_gate(name, feat_dict)
                    if gate_open:
                        agent  = all_agents[name]
                        action = agent.act(state, epsilon=0.0)
                        delta  = agent.deltas[action]
                        deltas[name] = delta

                        if name == agent_name:
                            live_gate_fired = True
                            live_action     = action
                            live_delta      = delta
                    else:
                        deltas[name] = 0.0

                # MAX-combine (Eq. 3.38)
                max_delta = max(deltas.values()) if deltas else 0.0

                # Trust update
                trust[nid] = max(0.0, trust[nid] - max_delta)

                # Record live agent's contribution this cycle
                node_outcomes[(cycle_id, nid)] = {
                    "gate_fired": live_gate_fired,
                    "action":     live_action,
                    "outcome":    None,  # filled after final trust evaluation below
                }

                # Track per-node delta history for D_i MAX-contribution approximation
                node_max_deltas[nid].append((live_delta, max_delta))

    # Restore live agent train mode if it was in training
    if was_training:
        live_agent.main_net.train()

    # ---------------------------------------------------------------------------
    # Classification — mirror validate_pipeline.py _evaluate_at_tau exactly
    # ---------------------------------------------------------------------------
    gt_dict = {
        row["node_id"]: {"is_attacker": int(row["is_attacker"]), "attack_type": row["attack_type"]}
        for _, row in ground_truth.iterrows()
    }

    # Build final blacklist: trust < tau_min  (using min trust reached)
    # NOTE: trust[] already holds the min because we never restore trust.
    blacklisted: Dict[int, bool] = {nid: trust[nid] < tau_min for nid in trust}

    # Determine which nodes are MAX contributors of the live agent
    # Approximation: "X's delta was the max at least once across the trajectory"
    max_contributor_nodes: set = set()
    for nid, delta_history in node_max_deltas.items():
        for live_d, max_d in delta_history:
            if live_d > 0.0 and live_d >= max_d:
                max_contributor_nodes.add(nid)
                break

    # Classification counts for variant X = agent_name
    tp = fp = fn = tn = 0

    honest_mask_atypes = {
        nid for nid, info in gt_dict.items()
        if info["is_attacker"] == 0
    }
    target_mask = {
        nid for nid, info in gt_dict.items()
        if info["is_attacker"] == 1 and info["attack_type"].upper() == agent_name.upper()
    }

    for nid in sorted(trust.keys()):
        if nid not in gt_dict:
            continue
        is_target   = nid in target_mask
        is_honest   = nid in honest_mask_atypes
        is_detected = blacklisted[nid]

        if is_target:
            if is_detected:
                outcome = "TP"
                tp += 1
            else:
                outcome = "FN"
                fn += 1
        elif is_honest:
            if is_detected:
                outcome = "FP"
                fp += 1
            else:
                outcome = "TN"
                tn += 1
        else:
            outcome = "EXCLUDED"

        # Propagate final outcome to all cycle entries for this node
        for cycle_id in all_cycles:
            key = (cycle_id, nid)
            if key in node_outcomes:
                node_outcomes[key]["outcome"] = outcome

    mcc = mcc_from_counts(tp, fp, fn, tn)

    logger.info(
        "[episode_eval] variant=%s | TP=%d FP=%d FN=%d TN=%d | MCC=%.4f | tau_min=%.2f",
        agent_name.upper(), tp, fp, fn, tn, mcc, tau_min,
    )

    # Log how many nodes get nonzero D_i (gate fired + max contributor + TP/FP)
    n_nonzero_di = sum(
        1 for nid in max_contributor_nodes
        if blacklisted.get(nid) and nid in (target_mask | honest_mask_atypes)
    )
    logger.info(
        "[episode_eval] Max-contributor nodes: %d | nodes eligible for nonzero D_i: %d",
        len(max_contributor_nodes), n_nonzero_di,
    )

    return EpochOutcome(
        counts          = (tp, fp, fn, tn),
        mcc             = mcc,
        node_outcomes   = node_outcomes,
        max_contributor = max_contributor_nodes,
        node_deltas     = node_max_deltas,
    )


# ---------------------------------------------------------------------------
# Self-test oracle (Step 2 verification gate)
# ---------------------------------------------------------------------------

def self_test(tau_min: float = 0.3) -> bool:
    """
    Correctness oracle: load ALL FOUR fine-tuned checkpoints as 'frozen',
    evaluate each variant, and confirm MCC values match the current validation
    output to 4 decimal places.

    Expected (from validate_pipeline.py at tau_min=0.3):
        ALS  +0.8991
        FS   +0.5498
        IGH  +0.8783
        SP   +0.5766

    Any mismatch → FAIL. Caller should stop and fix before any training changes.

    Returns True if all pass, False otherwise.
    """
    EXPECTED = {
        "als": +0.8991,
        "fs":  +0.5498,
        "igh": +0.8783,
        "sp":  +0.5766,
    }
    TOLERANCE = 5e-4  # 4 significant decimal places

    print("=" * 64)
    print("episode_eval.py  --self-test  (Step 2 oracle)")
    print(f"tau_min = {tau_min}")
    print("=" * 64)

    tables        = load_tables()
    ground_truth  = load_gt()
    all_pass      = True

    for variant in ("als", "fs", "igh", "sp"):
        # Load all 4 checkpoints as frozen — use variant as "live" with frozen others
        frozen = load_frozen_agents(exclude=None)   # load all 4
        live   = frozen.pop(variant)                # pull out the "live" one

        outcome = evaluate_policy_epoch(
            agent_name    = variant,
            live_agent    = live,
            frozen_agents = frozen,
            tables        = tables,
            ground_truth  = ground_truth,
            tau_min       = tau_min,
        )
        tp, fp, fn, tn = outcome.counts
        mcc            = outcome.mcc
        expected       = EXPECTED[variant]
        diff           = abs(mcc - expected)
        status         = "PASS" if diff <= TOLERANCE else "FAIL"
        if status == "FAIL":
            all_pass = False

        print(
            f"  [{status}] {variant.upper():<4}  "
            f"MCC={mcc:+.4f}  expected={expected:+.4f}  "
            f"diff={diff:.4f}  "
            f"(TP={tp} FP={fp} FN={fn} TN={tn})"
        )

    print()
    if all_pass:
        print("✅  Self-test PASSED — episode_eval matches validate_pipeline.py.")
    else:
        print("❌  Self-test FAILED — evaluator has diverged from the deployed pipeline.")
        print("    Fix the evaluator before any training changes (hard gate per spec).")
    print("=" * 64)

    return all_pass


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="episode_eval.py — MCC trust-trajectory evaluator."
    )
    parser.add_argument(
        "--self-test", action="store_true",
        help="Run the Step 2 oracle: verify MCC matches validate_pipeline.py.",
    )
    parser.add_argument(
        "--tau", type=float, default=0.3,
        help="tau_min threshold for self-test (default 0.3).",
    )
    args = parser.parse_args()

    if args.self_test:
        ok = self_test(tau_min=args.tau)
        sys.exit(0 if ok else 1)
    else:
        parser.print_help()
        sys.exit(0)
