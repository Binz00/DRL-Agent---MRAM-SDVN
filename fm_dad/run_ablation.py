"""
run_ablation.py — C2/C3/C4 ablation driver for FM-DAD (batch pipeline only).

Reads the four agent state CSVs from data/agent_inputs/ (written by verify_bridge.py),
runs the trigger/trust stage under the specified ablation condition, and writes:
  - data/results/<condition>_taumin<X>_<run_id>.csv
  - data/results/<condition>_taumin<X>_<run_id>.manifest.json

NOTE: This script reads data/agent_inputs/ AS-IS. Run verify_bridge.py once
before running any ablation conditions you want to compare against the same
snapshot. Do NOT change data/raw_csvs/ between a baseline and c2 run.

Attacker density note (Section 6, option b of ablation spec):
  The current NS-3 dataset has a FIXED ~9% attacker density (30 per type / 321 nodes).
  These results represent a single-density result, NOT a rho_a sweep. This is stated
  explicitly here and in every manifest file this script produces.

Usage:
    python3 run_ablation.py --condition baseline --tau-min 0.3
    python3 run_ablation.py --condition c2       --tau-min 0.3
    python3 run_ablation.py --condition c3       --tau-min 0.3
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_FM_DAD_DIR = Path(__file__).parent
if str(_FM_DAD_DIR) not in sys.path:
    sys.path.insert(0, str(_FM_DAD_DIR))

from bridge.trigger import load_agents, process_cycle
from bridge.trust_client import reset_mock_store, apply_trust_delta, _get_mock_trust
from bridge.assemble import AGENT_INPUT_DIR

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][ablation][%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ablation")

# ---------------------------------------------------------------------------
# Ablation condition registry
# C2: all 4 agents binary.
# C3: SP/ALS/FS binary, IGH stays graded (IGH behaviour is cycle-history dependent).
# Future C4/C5 entries go here — no other code changes needed.
# ---------------------------------------------------------------------------
CONDITIONS: Dict[str, dict] = {
    "baseline": dict(ablation_mode="graded",  ablation_binary_agents=None),
    "c2":       dict(ablation_mode="binary",   ablation_binary_agents=None),
    "c3":       dict(ablation_mode="binary",   ablation_binary_agents={"sp", "als", "fs"}),
}

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="FM-DAD C2/C3 Ablation Driver (batch pipeline, trigger-only stage)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--condition", choices=list(CONDITIONS.keys()), required=True,
        help="Ablation condition to run.",
    )
    p.add_argument(
        "--tau-min", type=float, default=0.3,
        help="Blacklisting trust threshold (tau_min). Results are tau_i < tau_min.",
    )
    p.add_argument(
        "--agent-inputs-dir", default=str(AGENT_INPUT_DIR), metavar="DIR",
        help="Directory containing *_state.csv files written by verify_bridge.py.",
    )
    p.add_argument(
        "--results-dir", default=str(_FM_DAD_DIR / "data" / "results"), metavar="DIR",
        help="Directory to write output CSV and manifest.",
    )
    return p


# ---------------------------------------------------------------------------
# Guard: check agent_inputs directory is ready
# ---------------------------------------------------------------------------

def _check_inputs(agent_inputs_dir: Path) -> Dict[str, pd.DataFrame]:
    """Load all four state CSVs.  Fail clearly if any are missing."""
    tables = {}
    for name in ["sp", "als", "fs", "igh"]:
        csv_path = agent_inputs_dir / f"{name}_state.csv"
        if not csv_path.exists():
            logger.error(
                "Required file %s not found.\n"
                "  → Run:  python3 verify_bridge.py\n"
                "  to regenerate data/agent_inputs/ from the current data/raw_csvs/.",
                csv_path,
            )
            sys.exit(1)
        tables[name] = pd.read_csv(csv_path)
        logger.info("  Loaded %s: %d rows", csv_path.name, len(tables[name]))
    return tables


# ---------------------------------------------------------------------------
# Run one ablation condition
# ---------------------------------------------------------------------------

def _git_commit() -> str:
    """Return current HEAD commit hash (short), or a timestamp fallback."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=_FM_DAD_DIR, capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def run_condition(
    condition: str,
    tau_min: float,
    agent_inputs_dir: Path,
    results_dir: Path,
) -> None:
    """
    Run a single ablation condition against the current data/agent_inputs/ snapshot.

    Writes:
      results_dir/<condition>_taumin<X>_<run_id>.csv
      results_dir/<condition>_taumin<X>_<run_id>.manifest.json
    """
    cond_cfg     = CONDITIONS[condition]
    ablation_mode          = cond_cfg["ablation_mode"]
    ablation_binary_agents = cond_cfg["ablation_binary_agents"]

    run_id    = time.strftime("%Y%m%d_%H%M%S")
    tau_str   = str(tau_min).replace(".", "")
    stem      = f"{condition}_taumin{tau_str}_{run_id}"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_csv   = results_dir / f"{stem}.csv"
    out_mfst  = results_dir / f"{stem}.manifest.json"

    logger.info("=" * 70)
    logger.info("FM-DAD ABLATION  condition=%s  tau_min=%.2f  run_id=%s",
                condition, tau_min, run_id)
    logger.info("Attacker density NOTE: fixed ~9%% (30 attackers / 321 nodes per type).")
    logger.info("This is a single-density result, NOT a rho_a sweep.")
    logger.info("=" * 70)

    # --- Load state CSVs ---
    logger.info("[LOAD] Reading agent state tables from %s ...", agent_inputs_dir)
    tables = _check_inputs(agent_inputs_dir)

    n_rows_per_agent = {name: len(df) for name, df in tables.items()}

    # Determine unique cycles
    all_cycles: set = set()
    for df in tables.values():
        all_cycles.update(df["cycle_id"].unique())
    sorted_cycles = sorted(all_cycles)
    n_cycles = len(sorted_cycles)
    logger.info("  %d cycles found: %s ... %s", n_cycles, sorted_cycles[0], sorted_cycles[-1])

    # --- Load agents (read-only — no checkpoint modification) ---
    logger.info("[LOAD] Loading DRL agents (read-only, eval mode) ...")
    agents = load_agents()

    # --- Fresh trust store ---
    reset_mock_store()
    trust: Dict[int, float] = {}  # node_id -> current trust (1.0 initial)

    # --- Accumulators ---
    output_rows:   List[dict] = []
    gate_fire_counts = {"sp": 0, "als": 0, "fs": 0, "igh": 0}
    t_start = time.perf_counter()

    # --- Cycle loop ---
    for cycle in sorted_cycles:
        cycle_results = process_cycle(
            cycle_id=cycle,
            tables=tables,
            agents=agents,
            active=None,  # all 4 agents active
            ablation_mode=ablation_mode,
            ablation_binary_agents=ablation_binary_agents,
        )

        for res in cycle_results:
            nid         = res["node_id"]
            final_delta = res["final_delta"]

            # Count gate fires per agent (for manifest verification)
            for agent_name in res["gates_fired"]:
                gate_fire_counts[agent_name] += 1

            # Trust update
            if nid not in trust:
                trust[nid] = 1.0
            trust_before = trust[nid]
            trust_after  = max(0.0, trust_before - final_delta)
            trust[nid]   = trust_after

            # Also push to mock store (for compatibility with validate_pipeline.py)
            old_mock = _get_mock_trust(nid)
            apply_trust_delta(node_id=nid, delta=final_delta,
                              is_rsu=False, current_trust=old_mock)

            # Per-agent detail rows
            for agent_name in ["sp", "als", "fs", "igh"]:
                det = res["per_agent_details"].get(agent_name, {})
                output_rows.append({
                    "cycle_id":        cycle,
                    "node_id":         nid,
                    "agent":           agent_name,
                    "gate_fired":      det.get("gate", "absent") == "OPEN",
                    "ablation_mode":   det.get("ablation_mode", ablation_mode),
                    "action":          det.get("action", ""),
                    "delta":           det.get("delta", 0.0),
                    "final_delta":     final_delta,
                    "trust_before":    round(trust_before, 6),
                    "trust_after":     round(trust_after, 6),
                    "blacklisted":     trust_after < tau_min,
                })

    elapsed = time.perf_counter() - t_start

    # --- Write output CSV ---
    if output_rows:
        pd.DataFrame(output_rows).to_csv(out_csv, index=False)
        logger.info("[OUTPUT] Written %d rows → %s", len(output_rows), out_csv)

    # --- Write manifest ---
    manifest = {
        "condition":              condition,
        "ablation_mode":          ablation_mode,
        "ablation_binary_agents": sorted(ablation_binary_agents) if ablation_binary_agents else None,
        "tau_min":                tau_min,
        "agent_inputs_dir":       str(agent_inputs_dir),
        "n_cycles_found":         n_cycles,
        "cycle_range":            [int(sorted_cycles[0]), int(sorted_cycles[-1])],
        "n_rows_per_agent":       n_rows_per_agent,
        "gate_fire_counts":       gate_fire_counts,
        "run_id":                 run_id,
        "elapsed_seconds":        round(elapsed, 3),
        "git_commit":             _git_commit(),
        "attacker_density_note":  (
            "Fixed ~9% per attack type (30 attackers / 321 nodes). "
            "Single-density result only — NOT a rho_a sweep."
        ),
    }
    with open(out_mfst, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("[OUTPUT] Manifest → %s", out_mfst)

    # --- Summary ---
    blacklisted = {nid for nid, ts in trust.items() if ts < tau_min}
    logger.info("=" * 70)
    logger.info("SUMMARY  condition=%s  tau_min=%.2f", condition, tau_min)
    logger.info("  Cycles processed : %d", n_cycles)
    logger.info("  Nodes blacklisted: %d (trust < %.2f)", len(blacklisted), tau_min)
    logger.info("  Gate fires       : SP=%d  ALS=%d  FS=%d  IGH=%d",
                gate_fire_counts["sp"], gate_fire_counts["als"],
                gate_fire_counts["fs"], gate_fire_counts["igh"])
    logger.info("  Wall-clock time  : %.2f s", elapsed)
    logger.info("  Output CSV       : %s", out_csv.name)
    logger.info("  Manifest         : %s", out_mfst.name)
    logger.info("=" * 70)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = _build_parser().parse_args()
    run_condition(
        condition        = args.condition,
        tau_min          = args.tau_min,
        agent_inputs_dir = Path(args.agent_inputs_dir),
        results_dir      = Path(args.results_dir),
    )


if __name__ == "__main__":
    main()
