"""
grid_search_deltas.py — Per-agent delta (trust penalty magnitude) grid search.

For each agent X ∈ {SP, ALS, IGH, FS}:
  1. Load the fine-tuned checkpoint for X as the "live" agent.
  2. Load the other three fine-tuned checkpoints as frozen peers.
  3. For each candidate delta set in DELTA_CANDIDATES[X]:
       a. Temporarily replace agent.deltas with the candidate set.
       b. Call evaluate_policy_epoch() — full 28-cycle trust trajectory.
       c. Record MCC^X and confusion matrix counts.
       d. Restore original agent.deltas.
  4. Select the candidate set with highest MCC^X.
  5. Log all candidates and their scores.

Output:
  - Console: config.py-ready paste block with best deltas per agent.
  - CSV:     data/grid_search_delta_results.csv (all candidates, for the report).

Usage:
    python3 grid_search_deltas.py
    python3 grid_search_deltas.py --tau 0.4

Constraints (per spec):
  1. Never modify agent weights — only agent.deltas (a plain Python list) is swapped.
  2. Always restore agent.deltas immediately after each evaluation (try/finally).
  3. Current shared set [0.0, 0.05, 0.15, 0.30, 0.50] is in every candidate list.
  4. Each agent is searched independently — frozen peers keep pre-grid-search deltas.
  5. Import, do not reimplement: evaluate_policy_epoch, load_tables, load_gt,
     load_frozen_agents, mcc_from_counts — all from episode_eval.py.
  6. tau_min = 0.4 default (grid-search selected optimal), overridable via --tau.
  7. Every candidate is logged to CSV (not just the best).
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Path setup — make sure fm_dad/ is on sys.path when run from any directory
# ---------------------------------------------------------------------------
_FM_DAD_DIR = Path(__file__).parent
sys.path.insert(0, str(_FM_DAD_DIR))

from agent import DQNAgent
from config import (
    AGENT_CONFIGS,
    FINETUNE_HP,
    FINETUNE_MODEL_FILES,
    MODEL_FILES,
    get_logger,
)
from episode_eval import (
    evaluate_policy_epoch,
    load_frozen_agents,
    load_gt,
    load_tables,
    mcc_from_counts,
)

logger = get_logger("grid_search_deltas")

# ---------------------------------------------------------------------------
# Search space — per-agent candidate delta sets (Issue 2 fix)
# Constraint: δ0 = 0.0 (fixed), δ1 < δ2 < δ3 < δ4 for each agent.
# The current shared baseline [0.0, 0.05, 0.15, 0.30, 0.50] MUST appear in
# every agent's list so the grid search can select or replace it with evidence.
# ---------------------------------------------------------------------------
DELTA_CANDIDATES: Dict[str, List[List[float]]] = {
    "sp": [
        [0.0, 0.05, 0.10, 0.20, 0.40],
        [0.0, 0.05, 0.15, 0.30, 0.50],   # current (baseline)
        [0.0, 0.10, 0.20, 0.35, 0.50],
        [0.0, 0.10, 0.25, 0.40, 0.50],
    ],
    "als": [
        [0.0, 0.05, 0.10, 0.20, 0.40],
        [0.0, 0.05, 0.15, 0.30, 0.50],   # current (baseline)
        [0.0, 0.10, 0.20, 0.35, 0.50],
        [0.0, 0.15, 0.30, 0.45, 0.50],
    ],
    "igh": [
        [0.0, 0.05, 0.10, 0.20, 0.40],
        [0.0, 0.05, 0.15, 0.30, 0.50],   # current (baseline)
        [0.0, 0.10, 0.20, 0.35, 0.50],
        [0.0, 0.10, 0.25, 0.40, 0.50],
    ],
    "fs": [
        [0.0, 0.03, 0.08, 0.15, 0.30],   # smaller — FS has noisier signal
        [0.0, 0.05, 0.10, 0.20, 0.35],
        [0.0, 0.05, 0.15, 0.30, 0.50],   # current (baseline)
        [0.0, 0.08, 0.15, 0.25, 0.40],
    ],
}



# ---------------------------------------------------------------------------
# Agent loader — mirrors load_frozen_agents but for a single named agent
# ---------------------------------------------------------------------------

def load_one_agent(agent_name: str) -> DQNAgent:
    """
    Load the fine-tuned checkpoint for agent_name as the live agent.
    Falls back to the synthetic checkpoint if fine-tuned is not found.

    Returns a DQNAgent with epsilon=0.0 and main_net.eval() set.
    """
    cfg   = AGENT_CONFIGS[agent_name]
    agent = DQNAgent(cfg, FINETUNE_HP, device="cpu")

    ft_path  = _FM_DAD_DIR / FINETUNE_MODEL_FILES[agent_name]
    syn_path = _FM_DAD_DIR / MODEL_FILES[agent_name]

    if ft_path.exists():
        model_path = str(ft_path)
        mode_str   = "fine-tuned"
    elif syn_path.exists():
        model_path = str(syn_path)
        mode_str   = "synthetic (fine-tuned not found)"
        logger.warning(
            "[load_one_agent] %s: fine-tuned checkpoint not found (%s), "
            "falling back to synthetic (%s).",
            agent_name.upper(), ft_path, syn_path,
        )
    else:
        raise FileNotFoundError(
            f"[grid_search_deltas] No checkpoint found for '{agent_name}'.\n"
            f"Checked: {ft_path}, {syn_path}"
        )

    mtime = os.path.getmtime(model_path)
    agent.load(model_path)
    agent.main_net.eval()
    agent.epsilon = 0.0

    logger.info(
        "[load_one_agent] Loaded %s from %s (%s) | mtime=%.0f",
        agent_name.upper(), model_path, mode_str, mtime,
    )
    return agent


# ---------------------------------------------------------------------------
# Validation helper — confirm every candidate set obeys the ordering constraint
# ---------------------------------------------------------------------------

def _validate_candidates() -> None:
    """Raise AssertionError if any candidate violates δ1 < δ2 < δ3 < δ4."""
    for agent_name, candidates in DELTA_CANDIDATES.items():
        baseline = [0.0, 0.05, 0.15, 0.30, 0.50]
        assert baseline in candidates, (
            f"[grid_search_deltas] Baseline {baseline} missing from "
            f"{agent_name} candidates — regression guard violated."
        )
        for cand in candidates:
            assert cand[0] == 0.0, (
                f"[grid_search_deltas] {agent_name}: δ0 must be 0.0, got {cand}"
            )
            for i in range(1, len(cand) - 1):
                assert cand[i] < cand[i + 1], (
                    f"[grid_search_deltas] {agent_name}: candidate {cand} "
                    f"violates strict ordering at index {i}."
                )
    logger.info("[grid_search_deltas] All candidate sets pass ordering validation.")


# ---------------------------------------------------------------------------
# Core grid search
# ---------------------------------------------------------------------------

def run_delta_grid_search(
    tables,
    ground_truth,
    tau_min: float = 0.4,
) -> tuple:
    """
    Per-agent delta grid search.

    For each agent:
      1. Load the live agent from its fine-tuned checkpoint.
      2. Load the other three as frozen peers (pre-grid-search deltas intact).
      3. Sweep all candidate delta sets — swapping agent.deltas, evaluating,
         restoring immediately (try/finally).
      4. Select the candidate with highest MCC^X.

    Returns:
        best_per_agent : dict  agent_name -> {"deltas": list, "mcc": float}
        all_results    : list  all candidates with scores (for the CSV)
    """
    best_per_agent: dict = {}
    all_results:    list = []

    for agent_name, candidates in DELTA_CANDIDATES.items():
        logger.info("=" * 60)
        logger.info("[%s] Starting delta grid search (%d candidates)",
                    agent_name.upper(), len(candidates))

        # Step 1 — Load live agent from fine-tuned checkpoint
        live_agent = load_one_agent(agent_name)

        # Step 2 — Load other three frozen (each with their current deltas)
        # Constraint 4: frozen peers keep pre-grid-search deltas intact
        frozen_agents = load_frozen_agents(exclude=agent_name)

        best_mcc    = -float("inf")
        best_deltas: Optional[List[float]] = None
        results_rows: list = []

        for delta_set in candidates:
            # Swap deltas — no agent rebuild needed (constraint 1)
            original_deltas   = live_agent.deltas
            live_agent.deltas = delta_set

            try:
                outcome = evaluate_policy_epoch(
                    agent_name    = agent_name,
                    live_agent    = live_agent,
                    frozen_agents = frozen_agents,
                    tables        = tables,
                    ground_truth  = ground_truth,
                    tau_min       = tau_min,
                )
            finally:
                # Restore immediately — never leave agent in patched state
                # (constraint 2: even if evaluate_policy_epoch raises)
                live_agent.deltas = original_deltas

            tp, fp, fn, tn = outcome.counts
            results_rows.append({
                "agent":   agent_name,
                "deltas":  str(delta_set),
                "mcc":     outcome.mcc,
                "tp":      tp,
                "fp":      fp,
                "fn":      fn,
                "tn":      tn,
            })
            logger.info(
                "[%s] deltas=%s -> MCC=%.4f (TP=%d FP=%d FN=%d TN=%d)",
                agent_name.upper(), delta_set, outcome.mcc, tp, fp, fn, tn,
            )

            if outcome.mcc > best_mcc:
                best_mcc    = outcome.mcc
                best_deltas = delta_set

        best_per_agent[agent_name] = {"deltas": best_deltas, "mcc": best_mcc}
        all_results.extend(results_rows)

        logger.info(
            "[%s] BEST: %s -> MCC^%s = %.4f",
            agent_name.upper(), best_deltas, agent_name.upper(), best_mcc,
        )

    return best_per_agent, all_results


# ---------------------------------------------------------------------------
# End-to-end macro MCC check
# Loads all four agents simultaneously with their best-selected deltas and
# computes the average MCC — the true measure of whether per-agent calibration
# helps over the shared baseline.
# ---------------------------------------------------------------------------

def compute_baseline_macro_mcc(
    tables,
    ground_truth,
    tau_min: float = 0.3,
) -> float:
    """
    Compute macro MCC with current baseline deltas (unmodified shared deltas).
    """
    logger.info("=" * 60)
    logger.info("[baseline_mcc] Computing baseline macro MCC with shared deltas ...")

    all_agents: Dict[str, DQNAgent] = {}
    for name in ("sp", "als", "igh", "fs"):
        all_agents[name] = load_one_agent(name)

    mcc_values: list = []

    for agent_name in ("sp", "als", "igh", "fs"):
        live_agent    = all_agents[agent_name]
        frozen_agents = {n: ag for n, ag in all_agents.items() if n != agent_name}

        outcome = evaluate_policy_epoch(
            agent_name    = agent_name,
            live_agent    = live_agent,
            frozen_agents = frozen_agents,
            tables        = tables,
            ground_truth  = ground_truth,
            tau_min       = tau_min,
        )
        mcc_values.append(outcome.mcc)
        logger.info(
            "[baseline_mcc] %s -> MCC=%.4f", agent_name.upper(), outcome.mcc
        )

    macro = sum(mcc_values) / len(mcc_values)
    logger.info("[baseline_mcc] Baseline Macro MCC (shared deltas) = %.4f", macro)
    return macro


def compute_macro_mcc_with_best_deltas(
    best_per_agent: dict,
    tables,
    ground_truth,
    tau_min: float = 0.3,
) -> float:
    """
    Load all four agents with their best-selected deltas, evaluate each as
    'live' with the others as frozen peers, then return the average MCC.

    This is the end-to-end check that per-agent calibration helps macro MCC.
    """
    logger.info("=" * 60)
    logger.info("[macro_mcc] Computing macro MCC with best per-agent deltas ...")

    # Load all four agents once with their best deltas applied
    all_agents: Dict[str, DQNAgent] = {}
    for name in ("sp", "als", "igh", "fs"):
        agent = load_one_agent(name)
        agent.deltas = best_per_agent[name]["deltas"]   # apply best deltas
        all_agents[name] = agent

    mcc_values: list = []

    for agent_name in ("sp", "als", "igh", "fs"):
        live_agent    = all_agents[agent_name]
        frozen_agents = {n: ag for n, ag in all_agents.items() if n != agent_name}

        outcome = evaluate_policy_epoch(
            agent_name    = agent_name,
            live_agent    = live_agent,
            frozen_agents = frozen_agents,
            tables        = tables,
            ground_truth  = ground_truth,
            tau_min       = tau_min,
        )
        mcc_values.append(outcome.mcc)
        logger.info(
            "[macro_mcc] %s -> MCC=%.4f", agent_name.upper(), outcome.mcc
        )

    macro = sum(mcc_values) / len(mcc_values)
    logger.info("[macro_mcc] Macro MCC (best per-agent deltas) = %.4f", macro)
    return macro


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def write_csv(all_results: list, output_path: Path) -> None:
    """Write all candidate results to CSV (every candidate, not just best)."""
    if not all_results:
        logger.warning("[write_csv] No results to write.")
        return

    fieldnames = ["agent", "deltas", "mcc", "tp", "fp", "fn", "tn"]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)
    logger.info("[write_csv] Grid search results -> %s", output_path)


def print_config_ready_output(
    best_per_agent: dict,
    macro_mcc:      float,
    baseline_mcc:   float,
    tau_min:        float,
) -> None:
    """
    Print the config.py-ready paste block to the console, matching the
    format specified in the implementation prompt.
    """
    print()
    print("=" * 60)
    print("DELTA GRID SEARCH RESULTS — paste into config.py AGENT_CONFIGS")
    print("=" * 60)
    print(f"  (tau_min used = {tau_min})")
    print()

    for agent_name in ("sp", "als", "igh", "fs"):
        result = best_per_agent[agent_name]
        mcc_v  = result["mcc"]
        deltas = result["deltas"]
        print(f"  # {agent_name.upper()} best deltas "
              f"(MCC^{agent_name.upper()} = {mcc_v:+.4f})")
        print(f"  \"deltas\": {deltas},")
        print()

    print(f"Macro MCC with best per-agent deltas: {macro_mcc:+.4f}")
    print(f"(vs baseline with shared deltas:      +{baseline_mcc:.4f})")
    print("=" * 60)
    print()
    print("To apply: update each AGENT_CONFIGS[<name>][\"deltas\"] in config.py")
    print("and add comment: # grid-searched by grid_search_deltas.py (Issue 2 fix)")
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "grid_search_deltas.py — Per-agent delta (trust penalty magnitude) "
            "grid search. Evaluates all candidate delta sets per agent via full "
            "trust-trajectory simulation. Prints config.py-ready output."
        )
    )
    parser.add_argument(
        "--tau", type=float, default=0.3,
        help="tau_min blacklist threshold (default: 0.3, matching validate_pipeline.py).",
    )
    parser.add_argument(
        "--csv", type=str,
        default=str(_FM_DAD_DIR / "data" / "grid_search_delta_results.csv"),
        help="Output CSV path for all candidate results.",
    )
    args = parser.parse_args()

    tau_min  = args.tau
    csv_path = Path(args.csv)

    # Silence noisy sub-loggers during the grid search
    logging.getLogger("episode_eval").setLevel(logging.WARNING)
    logging.getLogger("bridge").setLevel(logging.WARNING)

    logger.info("=" * 60)
    logger.info("grid_search_deltas.py  (Issue 2 fix — per-agent delta calibration)")
    logger.info("tau_min = %.2f", tau_min)
    logger.info("=" * 60)

    # 0. Validate candidate sets before starting
    _validate_candidates()

    # 1. Load shared data once (reused across all agent iterations)
    logger.info("Loading tables ...")
    tables = load_tables()

    logger.info("Loading ground truth ...")
    ground_truth = load_gt()

    # 2. Compute baseline macro MCC on current dataset with shared deltas
    baseline_mcc = compute_baseline_macro_mcc(
        tables       = tables,
        ground_truth = ground_truth,
        tau_min      = tau_min,
    )

    # 3. Run per-agent grid search
    best_per_agent, all_results = run_delta_grid_search(
        tables       = tables,
        ground_truth = ground_truth,
        tau_min      = tau_min,
    )

    # 4. End-to-end macro MCC with all best deltas applied simultaneously
    macro_mcc = compute_macro_mcc_with_best_deltas(
        best_per_agent = best_per_agent,
        tables         = tables,
        ground_truth   = ground_truth,
        tau_min        = tau_min,
    )

    # 5. Write all candidate results to CSV
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_csv(all_results, csv_path)

    # 6. Print config.py-ready output with computed baseline MCC
    print_config_ready_output(best_per_agent, macro_mcc, baseline_mcc, tau_min)


if __name__ == "__main__":
    main()

