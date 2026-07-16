"""
train.py — Algorithm 2 training loop for one FM-DAD agent.

This file implements the full offline training procedure for a single agent
(identified by its name: 'sp', 'als', 'igh', or 'fs').

Algorithm 2 outline (Eqs. 3.58–3.63):
    For each episode:
        Shuffle transitions from the offline CSV.
        For each transition in the episode:
            1. Select action via epsilon-greedy (agent.act).
            2. Compute reward via the agent's reward function (rewards.py).
            3. Store (s, a, r, s', done) in the agent's buffer (agent.remember).
            4. If buffer has enough samples, call agent.learn() (Eqs. 3.58–3.63).
        Decay epsilon (linear schedule).
        Anneal beta for IS weights (Eq. 3.61).
        Log episode reward.

Usage:
    python train.py --agent sp [--episodes 500] [--smoke_test]

The --smoke_test flag uses a tiny dummy CSV (100 rows, 5 episodes) for
the Stage 1 smoke test. Real training requires synthetic CSVs from Stage 2.
"""

import argparse
import os
import random
import sys
import time

import numpy as np
import torch

# Add parent directory to path if running from inside fm_dad/
sys.path.insert(0, os.path.dirname(__file__))

from config import (
    get_logger,
    SHARED_HP,
    FINETUNE_HP,
    AGENT_CONFIGS,
    DATA_FILES,
    MODEL_FILES,
    FINETUNE_MODEL_FILES,
    FINETUNE_DATA_FILES,
    RANDOM_SEED,
)
from agent import DQNAgent
from data_loader import load_transitions
from rewards import REWARD_FN_MAP

logger = get_logger("train")


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seeds(seed: int) -> None:
    """Fix all random seeds for reproducibility (Section 11 of the report)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info("Seeds fixed to %d (torch, numpy, random).", seed)


# ---------------------------------------------------------------------------
# Epsilon schedule (linear decay, Eq. 3.58 / Section 8)
# ---------------------------------------------------------------------------

def compute_epsilon(episode: int, n_episodes: int, hp: dict) -> float:
    """
    Compute epsilon for the current episode using a linear decay schedule.

    Decays from eps0 to eps_min over eps_decay_frac * n_episodes episodes,
    then stays at eps_min for the remaining episodes.

    Args:
        episode    : Current episode index (0-based).
        n_episodes : Total number of training episodes.
        hp         : Shared hyperparameter dict.

    Returns:
        float: Current epsilon value.

    Implements: Linear epsilon-greedy decay (Section 8 of the report).
    """
    decay_episodes = int(hp["eps_decay_frac"] * n_episodes)
    if episode >= decay_episodes:
        return hp["eps_min"]
    fraction = episode / max(decay_episodes, 1)
    eps = hp["eps0"] - fraction * (hp["eps0"] - hp["eps_min"])
    return float(eps)


# ---------------------------------------------------------------------------
# Beta annealing for PER IS weights (Eq. 3.61)
# ---------------------------------------------------------------------------

def compute_beta(episode: int, n_episodes: int, hp: dict) -> float:
    """
    Anneal IS exponent beta from beta_per_init to 1.0 over training.

    Beta is linearly increased so that IS correction reaches full strength
    (beta=1) by the final episode, as required by Eq. 3.61.

    Args:
        episode    : Current episode index (0-based).
        n_episodes : Total number of training episodes.
        hp         : Shared hyperparameter dict.

    Returns:
        float: Current beta value ∈ [beta_per_init, 1.0].

    Implements: Beta annealing for Eq. 3.61 (IS correction).
    """
    fraction = episode / max(n_episodes - 1, 1)
    beta = hp["beta_per_init"] + fraction * (1.0 - hp["beta_per_init"])
    return float(beta)


# ---------------------------------------------------------------------------
# Reward dispatcher — calls the correct reward function per agent
# ---------------------------------------------------------------------------

def compute_reward_for_agent(agent_name: str, action: int, transition: dict, cfg: dict) -> float:
    """
    Dispatch reward computation to the correct attack-specific function.

    This wrapper extracts the required arguments from the transition dict and
    calls the appropriate function from rewards.py via REWARD_FN_MAP.

    Args:
        agent_name : One of 'sp', 'als', 'igh', 'fs'.
        action     : Chosen action index a_t.
        transition : Transition dict from data_loader.load_transitions().
        cfg        : Agent config dict (contains w1..w4 weights).

    Returns:
        float: Scalar reward r_t.

    Implements: Reward dispatch for Eq. 3.46 (per-agent reward functions).
    """
    is_attacker       = transition["is_attacker"]
    blockchain_reject = transition["blockchain_reject"]
    PDR_t             = transition["PDR_t"]
    d_bar_t           = transition["d_bar_t"]
    rho_recv          = transition["rho_recv"]
    lambda_t          = transition["lambda_t"]

    # Extract features from the state vector s (which contains the exact ordered features)
    s = transition["s"]
    features = cfg["features"]
    feat_map = {feat: float(s[i]) for i, feat in enumerate(features)}

    if agent_name == "igh":
        pdr_var     = feat_map["PDRVar"]
        coord_score = feat_map["CoordScore"]
        rho_recv_s  = feat_map["rho_recv"]
        return REWARD_FN_MAP["igh"](
            action, is_attacker, rho_recv_s, pdr_var, coord_score, d_bar_t, PDR_t, blockchain_reject, cfg
        )
    elif agent_name == "sp":
        dFF        = feat_map["dFF"]
        rho_recv_s = feat_map["rho_recv"]
        return REWARD_FN_MAP["sp"](
            action, is_attacker, rho_recv_s, dFF, d_bar_t, PDR_t, blockchain_reject, cfg
        )
    elif agent_name == "als":
        spoof_dev  = feat_map["SpoofDev"]
        lambda_t_s = feat_map["lambda_t"]
        return REWARD_FN_MAP["als"](
            action, is_attacker, lambda_t_s, spoof_dev, d_bar_t, PDR_t, blockchain_reject, cfg
        )
    elif agent_name == "fs":
        dFF        = feat_map["dFF"]
        delay_infl = feat_map["DelayInfl"]
        lambda_t_s = feat_map["lambda_t"]
        # Note: rho_recv is from the transition/CSV, not state vector features
        return REWARD_FN_MAP["fs"](
            action, is_attacker, rho_recv, lambda_t_s, dFF, delay_infl, d_bar_t, PDR_t, blockchain_reject, cfg
        )
    else:
        raise ValueError(f"Unknown agent name: {agent_name}")


# ---------------------------------------------------------------------------
# Main training loop (Algorithm 2)
# ---------------------------------------------------------------------------

def train_agent(
    agent_name:  str,
    n_episodes:  int     = None,
    smoke_test:  bool    = False,
    csv_path:    str     = None,
    finetune:    bool    = False,
) -> list[float]:
    """
    Run the full offline training loop for one FM-DAD agent (Algorithm 2).

    For each episode:
        - Shuffle transitions (simulates i.i.d. episode sampling from offline data).
        - For each transition: act -> compute reward -> remember -> learn.
        - Decay epsilon, anneal beta, log episode reward.

    Args:
        agent_name : One of 'sp', 'als', 'igh', 'fs'.
        n_episodes : Number of training episodes. Defaults to hp['n_episodes'].
        smoke_test : If True, uses minimal settings (5 episodes) for Stage 1 check.
        csv_path   : Path to CSV. If None, uses default from DATA_FILES / FINETUNE_DATA_FILES.
        finetune   : If True, loads pretrained synthetic weights and trains with
                     FINETUNE_HP on real NS-3 data. Saves to FINETUNE_MODEL_FILES.

    Returns:
        list[float]: Per-episode total reward history (for plotting in Stage 3).

    Implements: Algorithm 2 (Eqs. 3.58–3.63).
    """
    set_seeds(RANDOM_SEED)

    # Select hyperparameters and paths based on mode
    if finetune:
        hp           = FINETUNE_HP
        data_default = FINETUNE_DATA_FILES[agent_name]
        model_out    = FINETUNE_MODEL_FILES[agent_name]
        pretrain_src = MODEL_FILES[agent_name]   # synthetic weights to load
        logger.info(
            "=== FINE-TUNE MODE | loading pretrained weights from %s ===",
            pretrain_src,
        )
    else:
        hp           = SHARED_HP
        data_default = DATA_FILES[agent_name]
        model_out    = MODEL_FILES[agent_name]
        pretrain_src = None

    cfg = AGENT_CONFIGS[agent_name]

    # Override episodes for smoke test
    if smoke_test:
        n_episodes = 5
        logger.info("=== SMOKE TEST MODE: %d episodes ===", n_episodes)
    elif n_episodes is None:
        n_episodes = hp["n_episodes"]

    # CSV path
    if csv_path is None:
        csv_path = data_default

    logger.info(
        "=== Starting training | agent=%s, episodes=%d, csv=%s ===",
        agent_name, n_episodes, csv_path,
    )

    # ---- Load data ---------------------------------------------------------
    logger.info("[PIPELINE] Stage 1/4: Loading transitions from CSV...")
    transitions = load_transitions(csv_path, agent_name)
    logger.info("[PIPELINE] Transitions loaded: %d total.", len(transitions))

    if len(transitions) == 0:
        raise RuntimeError(f"No transitions loaded from {csv_path}")

    # ---- Initialise agent --------------------------------------------------
    logger.info("[PIPELINE] Stage 2/4: Initialising agent...")
    agent = DQNAgent(agent_cfg=cfg, hp=hp)
    logger.info("[PIPELINE] Agent initialised: %s", agent_name.upper())

    # Fine-tune: load synthetic-trained weights before training starts
    if finetune:
        if not os.path.exists(pretrain_src):
            raise FileNotFoundError(
                f"[FINETUNE] Pretrained model not found: {pretrain_src}\n"
                f"Run standard training first: python train.py --agent {agent_name}"
            )
        agent.load(pretrain_src)
        agent.target_net.load_state_dict(agent.main_net.state_dict())
        logger.info(
            "[FINETUNE] Loaded pretrained weights from %s | target net synced",
            pretrain_src,
        )

    # ---- Training loop (Algorithm 2) ---------------------------------------
    logger.info("[PIPELINE] Stage 3/4: Training loop starting...")
    episode_rewards: list[float] = []

    for ep in range(n_episodes):
        ep_start   = time.time()
        epsilon    = compute_epsilon(ep, n_episodes, hp)
        beta       = compute_beta(ep, n_episodes, hp)
        agent.beta = beta

        # Shuffle transitions within each episode (offline i.i.d. assumption)
        episode_transitions = transitions.copy()
        random.shuffle(episode_transitions)

        ep_reward  = 0.0
        ep_loss    = 0.0
        n_learns   = 0
        n_steps    = 0

        logger.info(
            "[Ep %d/%d] Starting | epsilon=%.4f, beta=%.4f, buffer_size=%d",
            ep + 1, n_episodes, epsilon, beta, len(agent.buffer),
        )

        for transition in episode_transitions:
            s      = transition["s"]
            s_next = transition["s_next"]
            done   = transition["done"]

            # Step 1: Select action (epsilon-greedy)
            action = agent.act(s, epsilon=epsilon)

            # Step 2: Compute reward from ground-truth columns
            reward = compute_reward_for_agent(agent_name, action, transition, cfg)
            ep_reward += reward

            # Step 3: Store in buffer
            agent.remember(s, action, reward, s_next, done)

            # Step 4: Learn if buffer is large enough
            loss = agent.learn(beta=beta)
            if loss is not None:
                ep_loss  += loss
                n_learns += 1

            n_steps += 1

        avg_loss    = ep_loss / max(n_learns, 1)
        ep_duration = time.time() - ep_start

        episode_rewards.append(ep_reward)

        logger.info(
            "[Ep %d/%d] DONE | total_reward=%.4f, steps=%d, learns=%d, "
            "avg_loss=%.6f, buffer=%d, time=%.2fs",
            ep + 1, n_episodes, ep_reward, n_steps, n_learns,
            avg_loss, len(agent.buffer), ep_duration,
        )

    # ---- Save model --------------------------------------------------------
    logger.info("[PIPELINE] Stage 4/4: Saving model...")
    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    agent.save(model_out)
    logger.info("Model saved → %s", model_out)
    logger.info("=== Training complete | agent=%s ===", agent_name)

    return episode_rewards


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train one FM-DAD agent (Algorithm 2, Eqs. 3.58–3.63)."
    )
    parser.add_argument(
        "--agent", required=True,
        choices=["sp", "als", "igh", "fs"],
        help="Which agent to train.",
    )
    parser.add_argument(
        "--episodes", type=int, default=None,
        help="Number of training episodes (overrides config default).",
    )
    parser.add_argument(
        "--smoke_test", action="store_true",
        help="Run a quick 5-episode smoke test on a dummy CSV.",
    )
    parser.add_argument(
        "--csv", default=None,
        help="Path to training CSV (overrides default from config).",
    )
    parser.add_argument(
        "--finetune", action="store_true",
        help=(
            "Fine-tune existing synthetic-trained model on real NS-3 data. "
            "Loads weights from MODEL_FILES[agent], trains with FINETUNE_HP, "
            "saves to FINETUNE_MODEL_FILES[agent]. "
            "Do not use --smoke_test together with --finetune."
        ),
    )
    args = parser.parse_args()

    if args.finetune and args.smoke_test:
        parser.error("--finetune and --smoke_test cannot be used together.")

    rewards = train_agent(
        agent_name  = args.agent,
        n_episodes  = args.episodes,
        smoke_test  = args.smoke_test,
        csv_path    = args.csv,
        finetune    = args.finetune,
    )

    logger.info("Episode rewards: %s", [round(r, 3) for r in rewards])

    # Plot rewards
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        arr = np.array(rewards, dtype=np.float64)
        ma = np.full_like(arr, np.nan)
        window = min(20, len(arr))
        for i in range(len(arr)):
            start = max(0, i - window + 1)
            ma[i] = arr[start : i + 1].mean()

        suffix = "_finetuned_reward.png" if args.finetune else "_reward.png"
        out_path = os.path.join("models", f"{args.agent}{suffix}")

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(range(1, len(rewards) + 1), rewards, alpha=0.25, color="#5b9bd5", linewidth=0.8, label="Episode reward")
        ax.plot(range(1, len(rewards) + 1), ma, color="#c00000", linewidth=2.0, label=f"{window}-ep moving avg")
        mode_str = "Fine-Tuning" if args.finetune else "Training"
        ax.set_title(f"FM-DAD — {args.agent.upper()} Agent {mode_str} Reward Curve", fontsize=13, fontweight="bold")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Total Episode Reward")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        logger.info("Reward plot saved to %s", out_path)
    except ImportError:
        logger.warning("matplotlib not installed, skipping reward plot.")
