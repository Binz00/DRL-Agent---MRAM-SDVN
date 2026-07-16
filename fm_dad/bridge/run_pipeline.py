"""
run_pipeline.py — Batch evaluation entry point for the FM-DAD bridge (Part 5).

Reads:
  fm_dad/data/agent_inputs/*_state.csv
Runs:
  Gate checking, DQNAgent inference, and mock trust update for all nodes/cycles.
Writes:
  Logs to fm_dad/logs/pipeline.log and console.
  Prints a summary table at the end.
"""

import logging
from pathlib import Path
import sys
import pandas as pd

# Add parent directory to path to load modules correctly
_BRIDGE_DIR = Path(__file__).parent
_FM_DAD_DIR = _BRIDGE_DIR.parent
if str(_FM_DAD_DIR) not in sys.path:
    sys.path.insert(0, str(_FM_DAD_DIR))

from bridge.trigger import load_agents, process_cycle
from bridge.trust_client import apply_trust_delta, reset_mock_store, _get_mock_trust
from bridge.assemble import AGENT_INPUT_DIR

# ---------------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------------
log_path = _FM_DAD_DIR / "logs" / "pipeline.log"
log_path.parent.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("pipeline")
logger.setLevel(logging.INFO)

# Clear any existing handlers to prevent duplicate output
if logger.handlers:
    logger.handlers.clear()

file_handler = logging.FileHandler(str(log_path), mode="w")
file_handler.setFormatter(logging.Formatter("[%(asctime)s]%(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
logger.addHandler(file_handler)

console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter("[%(asctime)s]%(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
logger.addHandler(console_handler)


# ---------------------------------------------------------------------------
# Core Batch Runner
# ---------------------------------------------------------------------------

def run_batch_pipeline():
    """
    Load all agent state CSVs, run cycle-by-cycle evaluations, apply trust
    updates, and output the final stats summary.
    """
    logger.info("=" * 80)
    logger.info("FM-DAD BATCH INFERENCE PIPELINE RUN")
    logger.info("=" * 80)

    # 1. Reset mock trust store to start with fresh 1.0 trust scores
    reset_mock_store()

    # 2. Load agents
    logger.info("[LOAD] Loading trained DRL agents...")
    agents = load_agents()

    # 3. Load input CSVs
    logger.info("[LOAD] Loading agent input tables from %s ...", AGENT_INPUT_DIR)
    tables = {}
    for name in ["sp", "als", "fs", "igh"]:
        csv_path = Path(AGENT_INPUT_DIR) / f"{name}_state.csv"
        if not csv_path.exists():
            logger.error("Required agent state file %s does not exist! Run verify_bridge.py first.", csv_path)
            return
        tables[name] = pd.read_csv(csv_path)
        logger.info("  Loaded %s_state.csv: %d rows", name, len(tables[name]))

    # Determine unique cycles in ascending order
    all_cycles = set()
    for df in tables.values():
        all_cycles.update(df["cycle_id"].unique())
    sorted_cycles = sorted(all_cycles)

    logger.info("Cycles detected: %s", sorted_cycles)

    # Statistics accumulators
    cycle_stats = []
    penalties_data = []

    # 4. Process cycle by cycle
    for cycle in sorted_cycles:
        logger.info("-" * 80)
        logger.info("[CYCLE] START CYCLE %d", cycle)
        logger.info("-" * 80)

        # Run inference and gates for all nodes in the cycle
        cycle_results = process_cycle(cycle_id=cycle, tables=tables, agents=agents)

        # Apply trust updates and log detailed traces
        nodes_evaluated = len(cycle_results)
        gates_fired_count = 0
        penalty_applied_count = 0
        total_trust_reduction = 0.0

        for res in cycle_results:
            node_id = res["node_id"]
            gates_fired = res["gates_fired"]
            actions = res["actions"]
            final_delta = res["final_delta"]

            if gates_fired:
                gates_fired_count += 1
            if final_delta > 0:
                penalty_applied_count += 1
                total_trust_reduction += final_delta

            # Retrieve old trust score and apply update
            old_trust = _get_mock_trust(node_id)
            new_trust = apply_trust_delta(
                node_id=node_id,
                delta=final_delta,
                is_rsu=False,  # Assuming vehicles for batch run
                current_trust=old_trust
            )

            # Detailed trace log formatted for NS-3 evidence
            logger.info(
                "[TRACE] Cycle %d | Node %d | Gates Fired: %s | Actions: %s | Δτ: %.3f | Trust: %.4f → %.4f",
                cycle, node_id, 
                [g.upper() for g in gates_fired] if gates_fired else "NONE",
                {k.upper(): f"a{v}" for k, v in actions.items()} if actions else "NONE",
                final_delta, old_trust, new_trust
            )

            penalties_data.append({
                "cycle_id": cycle,
                "node_id": node_id,
                "trust_score_after": new_trust,
                "total_penalty": final_delta
            })

        cycle_stats.append({
            "cycle_id": cycle,
            "evaluated": nodes_evaluated,
            "gates_fired": gates_fired_count,
            "penalized": penalty_applied_count,
            "total_reduction": total_trust_reduction
        })
        logger.info("[CYCLE] END CYCLE %d | Evaluated: %d | Gates Fired: %d | Penalized: %d",
                    cycle, nodes_evaluated, gates_fired_count, penalty_applied_count)

    # 5. Print Summary Table
    logger.info("=" * 80)
    logger.info("PIPELINE EXECUTION SUMMARY")
    logger.info("=" * 80)
    logger.info("%-8s %-12s %-12s %-12s %-18s", "Cycle", "Evaluated", "Gates Fired", "Penalized", "Total Reduction")
    logger.info("-" * 80)
    for stat in cycle_stats:
        logger.info(
            "%-8d %-12d %-12d %-12d %-18.3f",
            stat["cycle_id"], stat["evaluated"], stat["gates_fired"], stat["penalized"], stat["total_reduction"]
        )
    logger.info("=" * 80)

    # 6. Export penalties CSV for validation script
    out_csv = _FM_DAD_DIR / "data" / "pipeline_penalties.csv"
    if penalties_data:
        pd.DataFrame(penalties_data).to_csv(out_csv, index=False)
        logger.info("[SUCCESS] Exported %d penalty records to %s", len(penalties_data), out_csv.name)


if __name__ == "__main__":
    run_batch_pipeline()
