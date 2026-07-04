"""
config.py — Central configuration for all FM-DAD agents and shared hyperparameters.

All values referenced throughout the codebase must live here; nothing is hardcoded
inside logic files. Final values will be determined via grid search; these are
placeholder defaults as specified in Sections 4–8 of the report.

Equations referenced:
  - Agent state vectors: Eqs. 3.41–3.44
  - Action space:        Eq.  3.45
  - Reward weights:      Eq.  3.46
  - Network topology:    Eq.  3.57
  - Training algorithm:  Eqs. 3.58–3.63 (Algorithm 2)
"""

import logging

# ---------------------------------------------------------------------------
# Logging setup — shared across all fm_dad modules
# ---------------------------------------------------------------------------
LOG_LEVEL = logging.DEBUG          # Set to logging.INFO to reduce verbosity

def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger with a consistent format."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(LOG_LEVEL)
    return logger


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = 42   # configurable; used in train.py via torch.manual_seed etc.


# ---------------------------------------------------------------------------
# Shared training hyperparameters (Algorithm 2, Eqs. 3.58–3.63)
# ---------------------------------------------------------------------------
SHARED_HP = {
    # -- RL core --
    "gamma":          0.95,      # discount factor
    "lr":             0.001,     # Adam learning rate
    "batch_size":     64,
    "n_episodes":     500,       # default episodes (smoke test uses fewer)

    # -- Replay buffer (Eqs. 3.59–3.61) --
    "buffer_capacity": 100_000,
    "buffer_min":      1_000,    # min transitions before training starts
    "alpha_per":       0.6,      # PER priority exponent
    "beta_per_init":   0.4,      # IS weight exponent initial value (anneals → 1.0)
    "eps_per":         1e-5,     # small constant to avoid zero priority

    # -- Huber loss (Eq. 3.62) --
    "delta_huber":     1.0,

    # -- Soft target update (Eq. 3.63 / Section 8) --
    "kappa":           0.005,

    # -- Epsilon-greedy exploration --
    "eps0":            1.0,
    "eps_min":         0.05,
    "eps_decay_frac":  0.80,     # linear decay over this fraction of total episodes

    # -- Network architecture (Eq. 3.57) --
    "hidden_layers":   2,        # number of shared trunk layers; search space {2,3,4}
    "hidden_size":     128,      # neurons per layer; search space {64,128,256}
    "output_size":     5,        # |A| = 5 for all agents (Eq. 3.45)

    # -- Reward thresholds (Section 6) --
    "rho_recv_low":    0.5,      # threshold for "rho_recv << 1"  (false-positive guard)
    "lambda_high":     0.7,      # threshold for "lambda_t high"  (false-positive guard)

    # -- QoS reference baselines (Eq. 3.56) --
    "d_ref":           50.0,     # reference end-to-end delay (ms)
    "PDR_ref":         1.0,      # reference packet delivery ratio (fraction)
}


# ---------------------------------------------------------------------------
# Per-agent configuration
# ---------------------------------------------------------------------------
# Each entry contains:
#   name        : short identifier (also used for CSV/model filenames)
#   features    : ordered list of state-vector feature names (Eqs. 3.41–3.44)
#   input_dim   : len(features) — duplicated explicitly for clarity
#   deltas      : trust-penalty magnitudes per action a0..a4 (Eq. 3.45)
#   reward_fn   : identifies which reward function to call from rewards.py
#   w1..w4      : reward weights for r_sec, r_fp, r_qos, r_end (Eq. 3.46)
# ---------------------------------------------------------------------------

AGENT_CONFIGS = {

    # -----------------------------------------------------------------------
    # IGH — Interleaved Grey Hole agent   (Eq. 3.41, state dim = 8)
    # -----------------------------------------------------------------------
    "igh": {
        "name":      "igh",
        "features":  ["FFc", "dFF", "rho_recv", "d_bar", "tau",
                      "PDRVar", "CoordScore", "lambda_t"],
        "input_dim": 8,
        "deltas":    [0.0, 0.05, 0.15, 0.30, 0.50],  # Eq. 3.45 placeholders
        "reward_fn": "igh",
        "w1": 0.4,   # weight for r_sec
        "w2": 0.3,   # weight for r_fp
        "w3": 0.2,   # weight for r_qos
        "w4": 0.1,   # weight for r_end
    },

    # -----------------------------------------------------------------------
    # SP — Selective Packet Dropping agent (Eq. 3.42, state dim = 5)
    # -----------------------------------------------------------------------
    "sp": {
        "name":      "sp",
        "features":  ["FFc", "dFF", "rho_recv", "tau", "lambda_t"],
        "input_dim": 5,
        "deltas":    [0.0, 0.05, 0.15, 0.30, 0.50],
        "reward_fn": "sp",
        "w1": 0.4,
        "w2": 0.3,
        "w3": 0.2,
        "w4": 0.1,
    },

    # -----------------------------------------------------------------------
    # ALS — Asymmetric Link Spoofing agent (Eq. 3.43, state dim = 4)
    # -----------------------------------------------------------------------
    "als": {
        "name":      "als",
        "features":  ["SpoofDev", "dFF", "tau", "lambda_t"],
        "input_dim": 4,
        "deltas":    [0.0, 0.05, 0.15, 0.30, 0.50],
        "reward_fn": "als",
        "w1": 0.4,
        "w2": 0.3,
        "w3": 0.2,
        "w4": 0.1,
    },

    # -----------------------------------------------------------------------
    # FS — Flow Stretching agent           (Eq. 3.44, state dim = 5)
    # -----------------------------------------------------------------------
    "fs": {
        "name":      "fs",
        "features":  ["FFc", "dFF", "DelayInfl", "tau", "lambda_t"],
        "input_dim": 5,
        "deltas":    [0.0, 0.05, 0.15, 0.30, 0.50],
        "reward_fn": "fs",
        "w1": 0.4,
        "w2": 0.3,
        "w3": 0.2,
        "w4": 0.1,
    },
}

# ---------------------------------------------------------------------------
# File paths  (relative to the fm_dad/ package root)
# ---------------------------------------------------------------------------
DATA_DIR   = "data"
MODELS_DIR = "models"

DATA_FILES = {
    "sp":  f"{DATA_DIR}/sp_train.csv",
    "als": f"{DATA_DIR}/als_train.csv",
    "igh": f"{DATA_DIR}/igh_train.csv",
    "fs":  f"{DATA_DIR}/fs_train.csv",
}

MODEL_FILES = {
    "sp":  f"{MODELS_DIR}/sp.pt",
    "als": f"{MODELS_DIR}/als.pt",
    "igh": f"{MODELS_DIR}/igh.pt",
    "fs":  f"{MODELS_DIR}/fs.pt",
}
