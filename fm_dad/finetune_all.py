"""
finetune_all.py — Round-robin fine-tuning driver (Step 5).

Trains agents in order: FS → SP → ALS → IGH  (FS first: most headroom).
Each agent is fine-tuned against frozen latest checkpoints of the other three.
After one full round, validate_pipeline.py is run to report macro MCC.

A second round is performed only if round 1 improved macro MCC by > 0.01
relative to the pre-fine-tune baseline.

Usage:
    python3 finetune_all.py [--rounds 1] [--agents fs sp als igh]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import os
from pathlib import Path

_FM_DAD_DIR = Path(__file__).parent
sys.path.insert(0, str(_FM_DAD_DIR))

from config import get_logger

logger = get_logger("finetune_all")

# Fine-tune order per spec: FS first (most headroom), then SP, ALS, IGH.
DEFAULT_ORDER = ["fs", "sp", "als", "igh"]

# Macro MCC improvement threshold for triggering a second round.
SECOND_ROUND_DELTA = 0.01


def run_validate() -> float:
    """
    Run validate_pipeline.py and extract macro MCC from its output.
    Returns float macro MCC, or -1.0 if parsing fails.
    """
    result = subprocess.run(
        [sys.executable, str(_FM_DAD_DIR / "validate_pipeline.py")],
        capture_output=True,
        text=True,
        cwd=str(_FM_DAD_DIR),
    )
    output = result.stdout + result.stderr
    # Extract: "Macro-averaged MCC (all attacks): +0.7260"
    for line in output.splitlines():
        if "Macro-averaged MCC" in line:
            try:
                macro_mcc = float(line.split(":")[-1].strip())
                return macro_mcc
            except ValueError:
                pass
    logger.warning("[validate] Could not parse macro MCC from output:\n%s", output[-2000:])
    return -1.0


def finetune_agent(agent_name: str) -> None:
    """Run train.py --finetune for one agent and log completion."""
    logger.info("[finetune_all] === Starting fine-tune: %s ===", agent_name.upper())
    result = subprocess.run(
        [sys.executable, str(_FM_DAD_DIR / "train.py"),
         "--agent", agent_name,
         "--finetune"],
        cwd=str(_FM_DAD_DIR),
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"[finetune_all] train.py --agent {agent_name} --finetune "
            f"exited with code {result.returncode}"
        )
    logger.info("[finetune_all] === Finished fine-tune: %s ===", agent_name.upper())


def run_round(order: list[str], round_num: int) -> None:
    """Run one full fine-tune round over all agents in order."""
    logger.info(
        "[finetune_all] ====== ROUND %d: order=%s ======",
        round_num, " → ".join(a.upper() for a in order),
    )
    for agent_name in order:
        # frozen_agents inside train.py always reload latest FINETUNE_MODEL_FILES,
        # so successive agents automatically benefit from prior agents' improvements.
        finetune_agent(agent_name)
    logger.info("[finetune_all] ====== ROUND %d COMPLETE ======", round_num)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Round-robin fine-tuning driver (Step 5)."
    )
    parser.add_argument(
        "--rounds", type=int, default=1,
        help="Number of rounds to run (default: 1; second round auto-triggered if Δmacro > 0.01).",
    )
    parser.add_argument(
        "--agents", nargs="+", default=DEFAULT_ORDER,
        choices=["sp", "als", "igh", "fs"],
        help="Order of agents to fine-tune (default: fs sp als igh).",
    )
    args = parser.parse_args()

    order = args.agents

    # ------------------------------------------------------------------
    # Baseline: macro MCC before any fine-tuning
    # ------------------------------------------------------------------
    logger.info("[finetune_all] Measuring BASELINE macro MCC (pre-fine-tune)...")
    baseline_mcc = run_validate()
    logger.info("[finetune_all] Baseline macro MCC = %.4f", baseline_mcc)

    print(f"\n[finetune_all] PRE-FINE-TUNE  macro MCC = {baseline_mcc:+.4f}")
    print(f"[finetune_all] Fine-tune order: {' → '.join(a.upper() for a in order)}\n")

    # ------------------------------------------------------------------
    # Round 1
    # ------------------------------------------------------------------
    run_round(order, round_num=1)

    logger.info("[finetune_all] Measuring post-round-1 macro MCC...")
    round1_mcc = run_validate()
    logger.info("[finetune_all] Post-round-1 macro MCC = %.4f", round1_mcc)

    improvement_r1 = round1_mcc - baseline_mcc
    print(f"\n[finetune_all] POST-ROUND-1   macro MCC = {round1_mcc:+.4f}  "
          f"(Δ = {improvement_r1:+.4f})")

    # ------------------------------------------------------------------
    # Optional Round 2: only if round 1 improved by > SECOND_ROUND_DELTA
    # ------------------------------------------------------------------
    if improvement_r1 > SECOND_ROUND_DELTA:
        logger.info(
            "[finetune_all] Round 1 improved macro MCC by %.4f > %.4f — "
            "triggering optional Round 2.",
            improvement_r1, SECOND_ROUND_DELTA,
        )
        print(f"\n[finetune_all] Δmacro > {SECOND_ROUND_DELTA:.2f} — running optional Round 2...\n")
        run_round(order, round_num=2)

        logger.info("[finetune_all] Measuring post-round-2 macro MCC...")
        round2_mcc = run_validate()
        improvement_r2 = round2_mcc - round1_mcc
        print(f"\n[finetune_all] POST-ROUND-2   macro MCC = {round2_mcc:+.4f}  "
              f"(Δ from round 1 = {improvement_r2:+.4f})")
        final_mcc = round2_mcc
    else:
        logger.info(
            "[finetune_all] Round 1 improvement %.4f ≤ %.4f — no second round.",
            improvement_r1, SECOND_ROUND_DELTA,
        )
        final_mcc = round1_mcc

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("finetune_all.py — SUMMARY")
    print("=" * 60)
    print(f"  Baseline macro MCC : {baseline_mcc:+.4f}")
    print(f"  Final    macro MCC : {final_mcc:+.4f}")
    print(f"  Total improvement  : {final_mcc - baseline_mcc:+.4f}")
    print("=" * 60)

    # Success criterion: final MCC must not be worse than baseline
    if final_mcc < baseline_mcc - 1e-4:
        logger.error(
            "[finetune_all] Final macro MCC %.4f is WORSE than baseline %.4f!",
            final_mcc, baseline_mcc,
        )
        sys.exit(1)
    else:
        logger.info("[finetune_all] Final macro MCC %.4f ≥ baseline %.4f — SUCCESS.",
                    final_mcc, baseline_mcc)


if __name__ == "__main__":
    main()
