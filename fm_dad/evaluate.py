"""
evaluate.py — Stage 3 verification and evaluation for trained FM-DAD agents.

Implements Section 10, Steps 10–12 of the report:

    Step 10 : Train agents in order: SP → ALS → FS → IGH.
    Step 11 : For each trained agent verify:
                (a) Moving-average episode reward plot → models/<agent>_reward.png
                (b) Policy check on clear attacker / clear innocent / victim rows
                (c) Save trained weights → models/<agent>.pt
    Step 12 : Print final summary table:
              agent | final_avg_reward | attacker_acc | innocent_acc

Usage:
    python evaluate.py [--episodes 500]

All thresholds and hyperparameters are sourced from config.py.
"""

import os
import sys
import argparse
import numpy as np
import torch
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # headless backend — no display needed
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))

from config import (
    get_logger, SHARED_HP, AGENT_CONFIGS, DATA_FILES, MODEL_FILES, RANDOM_SEED,
)
from agent import DQNAgent
from train import train_agent

logger = get_logger("evaluate")

TRAINING_ORDER = ["sp", "als", "fs", "igh"]   # Step 10 ordering


# ---------------------------------------------------------------------------
# Reward-curve plot helper
# ---------------------------------------------------------------------------

def _moving_average(values: list, window: int = 20) -> np.ndarray:
    """
    Compute a simple moving average over a list of scalars.

    Args:
        values : List of per-episode rewards.
        window : Smoothing window size.

    Returns:
        np.ndarray: Smoothed values (same length as input, padded at start).

    Implements: Step 11(a) reward-curve smoothing.
    """
    arr = np.array(values, dtype=np.float64)
    out = np.full_like(arr, np.nan)
    for i in range(len(arr)):
        start = max(0, i - window + 1)
        out[i] = arr[start : i + 1].mean()
    return out


def save_reward_plot(agent_name: str, episode_rewards: list, out_path: str) -> None:
    """
    Save a reward-curve plot (raw + 20-episode moving average) as a PNG.

    Args:
        agent_name     : Short agent identifier (e.g. 'sp').
        episode_rewards: Per-episode total reward list from train_agent().
        out_path       : Absolute path to write the PNG file.

    Implements: Step 11(a) — save reward curve to models/<agent>_reward.png.
    """
    logger.info("[%s] Saving reward curve → %s", agent_name.upper(), out_path)

    episodes = list(range(1, len(episode_rewards) + 1))
    ma = _moving_average(episode_rewards, window=20)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(episodes, episode_rewards, alpha=0.25, color="#5b9bd5", linewidth=0.8, label="Episode reward")
    ax.plot(episodes, ma, color="#c00000", linewidth=2.0, label="20-ep moving avg")
    ax.set_title(f"FM-DAD — {agent_name.upper()} Agent Training Reward Curve", fontsize=13, fontweight="bold")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Episode Reward")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("[%s] Reward plot saved.", agent_name.upper())


# ---------------------------------------------------------------------------
# Policy check (Step 11b)
# ---------------------------------------------------------------------------

def run_policy_check(agent_name: str, agent: DQNAgent, n_sample: int = 20) -> dict:
    """
    Evaluate the trained policy on representative rows from the agent's CSV.

    Three groups are tested (matching the three-group structure in each CSV):
        Group 1 — Attackers (is_attacker=1)           → expect high actions (a3/a4)
        Group 2 — Innocent victims / false-pos risk   → expect low actions (a0/a1)
        Group 3 — Clear innocents (is_attacker=0)     → expect a0

    Selects up to n_sample rows from each group, converts state to tensor,
    uses greedy policy (epsilon=0), and records chosen actions.

    Args:
        agent_name : Agent identifier ('sp', 'als', 'igh', 'fs').
        agent      : Trained DQNAgent instance with loaded weights.
        n_sample   : Number of rows to sample per group.

    Returns:
        dict with keys:
            'attacker_actions'  : list of actions chosen on attacker rows
            'innocent_actions'  : list of actions chosen on clear innocent rows
            'victim_actions'    : list of actions chosen on victim/fp-risk rows
            'attacker_high_acc' : fraction with action ≥ 3 (strong response)
            'innocent_a0_acc'   : fraction with action == 0 (no response)
            'victim_low_acc'    : fraction with action ≤ 1 (conservative response)

    Implements: Step 11(b) policy check (Section 10 of the report).
    """
    cfg      = AGENT_CONFIGS[agent_name]
    features = cfg["features"]
    csv_path = DATA_FILES[agent_name]

    logger.info("[%s] Running policy check on %s ...", agent_name.upper(), csv_path)
    df = pd.read_csv(csv_path)

    # ---- Identify the three groups ------------------------------------------
    # Group 1: attackers
    g1 = df[df["is_attacker"] == 1]

    # Group 2: innocent victims / false-positive risk
    # Detection: innocent nodes that look suspicious (the tricky group):
    #   SP/IGH/FS: is_attacker=0 AND rho_recv < rho_recv_low  (upstream victims)
    #   ALS:       is_attacker=0 AND lambda_t > lambda_high    (high mobility)
    rho_low     = SHARED_HP["rho_recv_low"]
    lambda_high = SHARED_HP["lambda_high"]

    if agent_name == "als":
        g2 = df[(df["is_attacker"] == 0) & (df["lambda_t"] > lambda_high)]
    elif agent_name in ("sp", "igh", "fs"):
        g2 = df[(df["is_attacker"] == 0) & (df["rho_recv"] < rho_low)]
    else:
        g2 = df[df["is_attacker"] == 0].head(n_sample)

    # Group 3: clear innocents (not in the victim / fp-risk group)
    if agent_name == "als":
        g3 = df[(df["is_attacker"] == 0) & (df["lambda_t"] <= lambda_high)]
    elif agent_name in ("sp", "igh", "fs"):
        g3 = df[(df["is_attacker"] == 0) & (df["rho_recv"] >= rho_low)]
    else:
        g3 = df[df["is_attacker"] == 0]

    logger.info(
        "[%s] Group sizes | G1(attackers)=%d  G2(victims)=%d  G3(clear innocents)=%d",
        agent_name.upper(), len(g1), len(g2), len(g3),
    )

    def _get_actions(subset: pd.DataFrame) -> list:
        """Run greedy policy on a sampled subset of rows, return action list."""
        if len(subset) == 0:
            return []
        sampled = subset.sample(min(n_sample, len(subset)), random_state=RANDOM_SEED)
        actions = []
        for _, row in sampled.iterrows():
            state = row[features].values.astype(np.float32)
            a = agent.act(state, epsilon=0.0)   # greedy
            actions.append(a)
        return actions

    att_actions = _get_actions(g1)
    vic_actions = _get_actions(g2)
    inn_actions = _get_actions(g3)

    att_high  = sum(1 for a in att_actions if a >= 3) / max(len(att_actions), 1)
    inn_a0    = sum(1 for a in inn_actions if a == 0) / max(len(inn_actions), 1)
    vic_low   = sum(1 for a in vic_actions if a <= 1) / max(len(vic_actions), 1)

    logger.info(
        "[%s] Policy check | attacker→high(≥3): %.0f%%  innocent→a0: %.0f%%  victim→low(≤1): %.0f%%",
        agent_name.upper(), att_high * 100, inn_a0 * 100, vic_low * 100,
    )
    logger.info(
        "[%s] Attacker actions: %s", agent_name.upper(),
        [int(a) for a in att_actions],
    )
    logger.info(
        "[%s] Innocent actions: %s", agent_name.upper(),
        [int(a) for a in inn_actions],
    )
    logger.info(
        "[%s] Victim   actions: %s", agent_name.upper(),
        [int(a) for a in vic_actions],
    )

    return {
        "attacker_actions":  att_actions,
        "innocent_actions":  inn_actions,
        "victim_actions":    vic_actions,
        "attacker_high_acc": att_high,
        "innocent_a0_acc":   inn_a0,
        "victim_low_acc":    vic_low,
    }


# ---------------------------------------------------------------------------
# Per-agent final average reward helper
# ---------------------------------------------------------------------------

def _final_avg_reward(episode_rewards: list, window: int = 50) -> float:
    """
    Compute the mean reward over the last `window` episodes.

    Used in the summary table (Step 12).

    Args:
        episode_rewards : Full list of per-episode rewards.
        window          : How many tail episodes to average.

    Returns:
        float: Mean reward of last `window` episodes.

    Implements: Step 12 (final summary table — final avg reward column).
    """
    tail = episode_rewards[-window:] if len(episode_rewards) >= window else episode_rewards
    return float(np.mean(tail))


# ---------------------------------------------------------------------------
# Main evaluation entry point
# ---------------------------------------------------------------------------

def run_evaluation(n_episodes: int = None) -> None:
    """
    Run full Stage 3: train all four agents in order, verify, and summarise.

    Order: SP → ALS → FS → IGH  (per Step 10 of the report).

    For each agent:
        1. Train via train_agent() (Algorithm 2, Eqs. 3.58–3.63).
        2. Save reward curve plot → models/<agent>_reward.png.
        3. Run policy check (Step 11b).
        4. Reload trained weights into a fresh agent for clean inference.

    Prints final summary table (Step 12).

    Args:
        n_episodes : Training episodes per agent. Default from SHARED_HP.

    Implements: Steps 10–12 (Section 10 of the report).
    """
    os.makedirs("models", exist_ok=True)

    summary_rows = []

    for agent_name in TRAINING_ORDER:
        logger.info("")
        logger.info("=" * 70)
        logger.info("STAGE 3 | Training agent: %s", agent_name.upper())
        logger.info("=" * 70)

        # ---- Step 10: Train ------------------------------------------------
        episode_rewards = train_agent(
            agent_name = agent_name,
            n_episodes = n_episodes,
            smoke_test = False,
        )

        # ---- Step 11(a): Save reward plot ----------------------------------
        plot_path = f"models/{agent_name}_reward.png"
        save_reward_plot(agent_name, episode_rewards, plot_path)

        # ---- Step 11(b): Policy check (reload weights for clean inference) --
        cfg = AGENT_CONFIGS[agent_name]
        eval_agent = DQNAgent(agent_cfg=cfg, hp=SHARED_HP)
        eval_agent.load(MODEL_FILES[agent_name])

        policy_results = run_policy_check(agent_name, eval_agent)

        # ---- Step 11(c): weights already saved by train_agent() -----------
        logger.info("[%s] Weights already saved → %s", agent_name.upper(), MODEL_FILES[agent_name])

        # ---- Accumulate summary --------------------------------------------
        final_avg = _final_avg_reward(episode_rewards, window=50)
        summary_rows.append({
            "agent":              agent_name.upper(),
            "final_avg_reward":   round(final_avg, 4),
            "attacker_acc_%":     round(policy_results["attacker_high_acc"] * 100, 1),
            "innocent_a0_%":      round(policy_results["innocent_a0_acc"]   * 100, 1),
            "victim_low_%":       round(policy_results["victim_low_acc"]    * 100, 1),
        })

    # ---- Step 12: Print summary table ------------------------------------
    logger.info("")
    logger.info("=" * 70)
    logger.info("STEP 12 — FINAL SUMMARY TABLE")
    logger.info("=" * 70)
    header = f"{'Agent':<8} {'FinalAvgReward':>16} {'Attacker→≥a3 %':>16} {'Innocent→a0 %':>14} {'Victim→≤a1 %':>13}"
    logger.info(header)
    logger.info("-" * 70)
    for row in summary_rows:
        logger.info(
            "%-8s %16.4f %16.1f %14.1f %13.1f",
            row["agent"],
            row["final_avg_reward"],
            row["attacker_acc_%"],
            row["innocent_a0_%"],
            row["victim_low_%"],
        )
    logger.info("=" * 70)

    # Also print as plain text for easy copy
    print("\n" + "=" * 70)
    print("FINAL SUMMARY TABLE (Step 12)")
    print("=" * 70)
    print(f"{'Agent':<8} {'FinalAvgReward':>16} {'Attacker→≥a3%':>14} {'Innocent→a0%':>13} {'Victim→≤a1%':>12}")
    print("-" * 70)
    for row in summary_rows:
        print(
            f"{row['agent']:<8} {row['final_avg_reward']:>16.4f} "
            f"{row['attacker_acc_%']:>14.1f} {row['innocent_a0_%']:>13.1f} "
            f"{row['victim_low_%']:>12.1f}"
        )
    print("=" * 70)
    print("\nReward plots saved to:")
    for agent_name in TRAINING_ORDER:
        print(f"  models/{agent_name}_reward.png")
    print()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FM-DAD Stage 3: Train all agents and verify (Steps 10–12)."
    )
    parser.add_argument(
        "--episodes", type=int, default=None,
        help="Training episodes per agent (overrides config default of 500).",
    )
    args = parser.parse_args()

    run_evaluation(n_episodes=args.episodes)
