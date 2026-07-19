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
import math
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

# episode_eval is imported lazily inside train_agent() when finetune=True
# to avoid circular import and to keep startup overhead minimal for non-finetune runs.


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
# MCC helpers (supervisor r_mcc patch) — lives in train.py per spec
# ---------------------------------------------------------------------------

def mcc_from_counts(tp: int, fp: int, fn: int, tn: int) -> float:
    """Equation 4.1 — MCC. Guard against zero denominator (degenerate case)."""
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return ((tp * tn) - (fp * fn)) / denom if denom > 0 else 0.0


def build_difference_rewards(outcome, agent_name: str) -> dict:
    """
    Build per-(cycle_id, node_id) difference rewards  D_i.

    D_i = MCC(actual) − MCC(counterfactual: live agent’s decision → a0).

    Only nodes where the live agent’s gate fired AND action > 0 AND the live
    agent was the MAX contributor (its delta ≥ every other agent’s delta at
    least once during the trajectory) can have non-zero D_i.

    Approximation (documented): “X’s delta was the max at least once” is
    used as the MAX-contribution condition rather than tracking which specific
    cycle’s delta was decisive. This is conservative: we may under-attribute
    D_i to some transitions, but we never false-attribute it.

    Counterfactual mapping:
      TP (attacker correctly blacklisted)
          → a0 would not have contributed → counterfactual: tp-1, fn+1
          D_i > 0  (real was better than a0)
      FP (honest node wrongly blacklisted)
          → a0 would not have contributed → counterfactual: fp-1, tn+1
          D_i < 0  (real was worse than a0  — penalises the false positive)
      FN / TN / gate closed / action==0 / not max-contributor
          → D_i = 0

    Returns:
        dict: (cycle_id, node_id) → D_i (float).  Only non-zero entries stored.
    """
    from episode_eval import EpochOutcome

    tp, fp, fn, tn = outcome.counts
    mcc_actual     = outcome.mcc

    d_rewards: dict = {}
    n_nonzero = 0

    for (cycle_id, node_id), rec in outcome.node_outcomes.items():
        # Skip if gate did not fire or action was a0 (delta=0)
        if not rec["gate_fired"]:
            continue
        action = rec["action"]
        if action is None or action == 0:
            continue

        final_outcome = rec["outcome"]
        if final_outcome not in ("TP", "FP"):
            # FN / TN / EXCLUDED — D_i = 0
            continue

        # MAX-contribution check: was this node’s live delta ≥ max at least once?
        if node_id not in outcome.max_contributor:
            continue

        # Compute counterfactual MCC (O(1) — closed-form over 4 integers)
        if final_outcome == "TP":
            # Without live agent, this TP becomes FN
            cf_mcc = mcc_from_counts(tp - 1, fp, fn + 1, tn)
        else:  # FP
            # Without live agent, this FP becomes TN
            cf_mcc = mcc_from_counts(tp, fp - 1, fn, tn + 1)

        d_i = mcc_actual - cf_mcc
        if d_i != 0.0:
            d_rewards[(cycle_id, node_id)] = d_i
            n_nonzero += 1

    logger.info(
        "[D_i] Variant=%s | non-zero D_i entries: %d / %d cycle-node visits",
        agent_name.upper(), n_nonzero, len(outcome.node_outcomes),
    )
    return d_rewards


# ---------------------------------------------------------------------------
# Main training loop (Algorithm 2)
# ---------------------------------------------------------------------------

def train_agent(
    agent_name:       str,
    n_episodes:       int     = None,
    smoke_test:       bool    = False,
    csv_path:         str     = None,
    finetune:         bool    = False,
    w5_override:      float   = None,   # overrides cfg["w5"] for Step 4 grid verification
    model_out_override: str   = None,   # redirects checkpoint output path
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

    # Allow CLI to redirect output path (e.g. for w5=0.0 regression check)
    if model_out_override is not None:
        model_out = model_out_override
        logger.info("[model_out] Output path OVERRIDDEN via CLI: %s", model_out)

    cfg = AGENT_CONFIGS[agent_name]

    # ------------------------------------------------------------------
    # (1−w5) rescaling of w1..w4 to keep sum(w1..w5)=1 (supervisor patch).
    # This is done at USE TIME on a local copy — config.py values are NEVER
    # mutated.  Only applies when finetune=True (w5 is a finetune hyperparameter).
    # When finetune=False, w5=0.0 so the scaling factor is 1.0 (no change).
    # w5_override (CLI --w5) lets Step 4 grid runs override w5 without editing config.
    # ------------------------------------------------------------------
    if w5_override is not None:
        w5 = float(w5_override)
        logger.info("[w5] w5 OVERRIDDEN via CLI: %.2f (config value ignored)", w5)
    else:
        w5 = cfg.get("w5", 0.0) if finetune else 0.0
    scale       = 1.0 - w5
    cfg_scaled  = dict(cfg)  # shallow copy
    for wi in ("w1", "w2", "w3", "w4"):
        if wi in cfg_scaled:
            cfg_scaled[wi] = cfg_scaled[wi] * scale
    logger.info(
        "[w5] agent=%s | w5=%.2f | scale=%.2f | "
        "w1=%.3f w2=%.3f w3=%.3f w4=%.3f (rescaled)",
        agent_name, w5, scale,
        cfg_scaled.get("w1", 0), cfg_scaled.get("w2", 0),
        cfg_scaled.get("w3", 0), cfg_scaled.get("w4", 0),
    )

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

    # ---- Fine-tune setup: episode_eval tables + frozen agents ---------------
    if finetune:
        from episode_eval import (
            evaluate_policy_epoch, load_tables, load_gt, load_frozen_agents,
            EpochOutcome,
        )
        import pandas as pd

        eval_tables      = load_tables()
        eval_gt          = load_gt()
        frozen_agents    = load_frozen_agents(exclude=agent_name)

        mcc_eval_every   = hp.get("mcc_eval_every", 5)
        best_mcc         = -2.0          # worst possible MCC is −1
        best_ep          = 0
        d_reward_lookup: dict = {}       # (cycle_id, node_id) → D_i; rebuilt periodically
        logger.info(
            "[FINETUNE] MCC checkpoint selection enabled | "
            "eval every %d episodes | w5=%.2f",
            mcc_eval_every, w5,
        )

    # ---- Training loop (Algorithm 2) ---------------------------------------
    logger.info("[PIPELINE] Stage 3/4: Training loop starting...")
    episode_rewards: list[float] = []

    for ep in range(n_episodes):
        ep_start   = time.time()
        epsilon    = compute_epsilon(ep, n_episodes, hp)
        beta       = compute_beta(ep, n_episodes, hp)
        agent.beta = beta

        # ------------------------------------------------------------------
        # MCC evaluation — every mcc_eval_every episodes (and final episode)
        # Runs BEFORE the episode's gradient steps to keep evaluation clean.
        # ------------------------------------------------------------------
        if finetune:
            is_final_ep   = (ep == n_episodes - 1)
            is_eval_ep    = (ep % mcc_eval_every == 0) or is_final_ep
            if is_eval_ep:
                outcome = evaluate_policy_epoch(
                    agent_name    = agent_name,
                    live_agent    = agent,
                    frozen_agents = frozen_agents,
                    tables        = eval_tables,
                    ground_truth  = eval_gt,
                    tau_min       = 0.3,   # grid-searched best tau_min
                )
                ep_mcc = outcome.mcc
                tp_ep, fp_ep, fn_ep, tn_ep = outcome.counts

                # Rebuild difference-reward lookup from new evaluation
                d_reward_lookup = build_difference_rewards(outcome, agent_name)

                logger.info(
                    "[MCC EVAL] ep=%d/%d | MCC^%s=%.4f | TP=%d FP=%d FN=%d TN=%d | "
                    "nonzero_D_i=%d",
                    ep + 1, n_episodes, agent_name.upper(), ep_mcc,
                    tp_ep, fp_ep, fn_ep, tn_ep, len(d_reward_lookup),
                )

                # Best-checkpoint selection: save if new best MCC
                if ep_mcc > best_mcc:
                    best_mcc = ep_mcc
                    best_ep  = ep + 1
                    os.makedirs(os.path.dirname(model_out), exist_ok=True)
                    agent.save(model_out)
                    logger.info(
                        "[BEST CKPT] New best MCC^%s=%.4f at ep=%d — saved to %s",
                        agent_name.upper(), best_mcc, best_ep, model_out,
                    )

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

        ep_rewards_list: list = []  # for per-episode reward variance logging

        for transition in episode_transitions:
            s      = transition["s"]
            s_next = transition["s_next"]
            done   = transition["done"]

            # Step 1: Select action (epsilon-greedy)
            action = agent.act(s, epsilon=epsilon)

            # Step 2: Compute reward from ground-truth columns
            # r_base uses the (1-w5)-rescaled cfg so that w1..w4 weights
            # already incorporate the budget reallocation.
            r_base = compute_reward_for_agent(agent_name, action, transition, cfg_scaled)

            # Difference-reward term: D_i from the last MCC evaluation.
            # Between evaluations d_reward_lookup is held fixed (per spec).
            # For non-finetune mode or before first eval, D_i = 0.
            d_i = 0.0
            if finetune:
                key = (transition.get("cycle_id"), transition.get("node_id"))
                d_i = d_reward_lookup.get(key, 0.0)

            reward     = r_base + w5 * d_i
            ep_reward += reward
            ep_rewards_list.append(reward)

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

        # Non-stationary reward variance monitoring (spec risk 1)
        if finetune and len(ep_rewards_list) > 1:
            import statistics
            ep_var = statistics.variance(ep_rewards_list)
            logger.info(
                "[Ep %d/%d] DONE | total_reward=%.4f, reward_var=%.6f, "
                "steps=%d, learns=%d, avg_loss=%.6f, buffer=%d, time=%.2fs",
                ep + 1, n_episodes, ep_reward, ep_var, n_steps, n_learns,
                avg_loss, len(agent.buffer), ep_duration,
            )
        else:
            logger.info(
                "[Ep %d/%d] DONE | total_reward=%.4f, steps=%d, learns=%d, "
                "avg_loss=%.6f, buffer=%d, time=%.2fs",
                ep + 1, n_episodes, ep_reward, n_steps, n_learns,
                avg_loss, len(agent.buffer), ep_duration,
            )

    # ---- Save model --------------------------------------------------------
    logger.info("[PIPELINE] Stage 4/4: Saving model...")
    if not finetune:
        # Standard training: always save at end
        os.makedirs(os.path.dirname(model_out), exist_ok=True)
        agent.save(model_out)
        logger.info("Model saved → %s", model_out)
    else:
        # Fine-tune: best checkpoint was already saved during evaluation loop.
        # If no eval ran (e.g. smoke test with 0 episodes), save now.
        if best_ep == 0:
            os.makedirs(os.path.dirname(model_out), exist_ok=True)
            agent.save(model_out)
            logger.info("Model saved (no eval ran) → %s", model_out)
        else:
            logger.info(
                "[FINETUNE DONE] Best checkpoint: ep=%d, MCC^%s=%.4f — already saved to %s",
                best_ep, agent_name.upper(), best_mcc, model_out,
            )
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
    parser.add_argument(
        "--w5", type=float, default=None,
        help=(
            "Override w5 for this run (supervisor r_mcc grid: {0.0, 0.1, 0.2, 0.3}). "
            "Does not modify config.py. Used for Step 4 regression verification."
        ),
    )
    parser.add_argument(
        "--model_out", type=str, default=None,
        help="Override output checkpoint path (e.g. 'models/fs_finetuned_w5_0.pt').",
    )
    args = parser.parse_args()

    if args.finetune and args.smoke_test:
        parser.error("--finetune and --smoke_test cannot be used together.")

    rewards = train_agent(
        agent_name        = args.agent,
        n_episodes        = args.episodes,
        smoke_test        = args.smoke_test,
        csv_path          = args.csv,
        finetune          = args.finetune,
        w5_override       = args.w5,
        model_out_override= args.model_out,
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
