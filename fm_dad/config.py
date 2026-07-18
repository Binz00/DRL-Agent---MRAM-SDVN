"""
config.py — Central configuration for all FM-DAD agents and shared hyperparameters.

All values referenced throughout the codebase must live here; nothing is hardcoded
inside logic files. Final values will be determined via grid search; these are
placeholder defaults as specified in Sections 4–8 of the report.

Equations referenced:
  - Agent state vectors:  Eqs. 3.41–3.44
  - Action space:         Eq.  3.45
  - Reward weights:       Eq.  3.46
  - Graded r_sec (E^X):   Supervisor patch (replaces binary Eqs. 3.48/3.50/3.52/3.54)
  - Network topology:     Eq.  3.57
  - Training algorithm:   Eqs. 3.58–3.63 (Algorithm 2)

Supervisor review changes (Issue 1):
  - r_sec^X is now proportional: 1 - |a_t - a*(E^X)| / 4, where E^X ∈ [0,1]
    measures how strongly the state evidence exceeds the detection threshold.
  - Each agent config now includes per-feature detection thresholds (eta_*)
    and action-mapping thresholds e1 < e2 < e3 (subject to grid search).

Supervisor review changes (Issue 2):
  - eps_per corrected to 0.001 (grid-search range {0.01, 0.001}).
  - beta_per_init, eps_per, buffer_capacity are all now explicitly documented
    in the hyperparameter table below.

Supervisor review changes (Issue 3 — verified):
  - PDRVar computed over fixed window W = 10 for synthetic data.
    (Dynamic W* per Eq. 3.7 is applied only during real NS-3 preprocessing.)
  - CoordScore implements the Eq. 3.22 double maximum: max over partner
    nodes j ≠ i AND lag offsets τ ∈ [1, W] of the normalised cross-correlation
    of FFc series. Verified consistent with attacker CoordScore ≈ 0.998 in data.
"""

import logging

# ---------------------------------------------------------------------------
# Logging setup — shared across all fm_dad modules
# ---------------------------------------------------------------------------
LOG_LEVEL = logging.INFO          # Set to logging.DEBUG for full verbosity

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
    # NOTE (Issue 2): beta_per_init, eps_per, and buffer_capacity are in the
    # simulation-settings hyperparameter table in the report.
    "buffer_capacity": 100_000,  # B_max; search space {50000, 100000}
    "buffer_min":      1_000,    # min transitions before training starts
    "alpha_per":       0.6,      # PER priority exponent (α)
    "beta_per_init":   0.4,      # IS exponent initial value (β), annealed → 1.0
    "eps_per":         0.001,    # priority floor ε_per; search space {0.01, 0.001}

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
    # Issue 3 note: PDRVar and CoordScore were computed with W=10 for synthetic
    # data. CoordScore uses the Eq. 3.22 double maximum over partner nodes j≠i
    # AND lag offsets τ ∈ [1, W]. Dynamic W* applies to real NS-3 data only.
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
        # ---- E^IGH severity thresholds (graded r_sec, supervisor Issue 1) ----
        # E^IGH = mean of normalised PDRVar, CoordScore, rho_recv excesses.
        "eta_pdrvar": 0.03,  # PDRVar detection threshold (calibrated by grid search)
        "eta_coord":  0.30,  # CoordScore detection threshold
        "eta_rho":    0.30,  # rho_recv lower bound (= rho_recv_low)
        # a*(E^IGH) mapping thresholds: E<e1→a1, e1≤E<e2→a2, e2≤E<e3→a3, E≥e3→a4
        "e1": 0.76, "e2": 0.78, "e3": 0.80,  # calibrated from attacker severity distribution (25/50/75 percentiles), not from action-count tuning.
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
        # ---- E^SP severity threshold (graded r_sec, supervisor Issue 1) ----
        # E^SP = normalised dFF excess above eta_dFF.
        "eta_dFF": 0.65,  # dFF detection gate threshold (calibrated by grid search)
        "e1": 0.37, "e2": 0.58, "e3": 0.78,  # calibrated from attacker severity distribution (25/50/75 percentiles), not from action-count tuning.
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
        # ---- E^ALS severity threshold (graded r_sec, supervisor Issue 1) ----
        # E^ALS = normalised SpoofDev excess above eta_spoof.
        "eta_spoof": 0.005,  # SpoofDev detection threshold (calibrated by grid search)
        "e1": 0.42, "e2": 0.61, "e3": 0.81,  # calibrated from attacker severity distribution (25/50/75 percentiles), not from action-count tuning.
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
        # ---- E^FS severity thresholds (graded r_sec, supervisor Issue 1) ----
        # E^FS = mean of normalised dFF excess and DelayInfl excess.
        "eta_dFF":   0.20,  # dFF detection threshold (calibrated by grid search)
        "eta_delay": 1.50,  # DelayInfl detection threshold (spec: attackers > 1.3)
        "eta_hop":   1,     # hop-count excess gate threshold (placeholder, pending grid search)
        "delay_max": 2.00,  # upper clamp for DelayInfl normalisation
        "e1": 0.35, "e2": 0.50, "e3": 0.65,  # calibrated from attacker severity distribution (25/50/75 percentiles), not from action-count tuning.
    },
}

# ---------------------------------------------------------------------------
# File paths  (relative to the fm_dad/ working directory)
# ---------------------------------------------------------------------------
DATA_DIR   = "data/synthetic_data"   # real CSVs live in this subfolder
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

# ---------------------------------------------------------------------------
# Fine-tuning hyperparameters (supervisor instruction: short run, real NS-3 data)
# Inherits all values from SHARED_HP and overrides only what changes for fine-tuning.
# Lower lr preserves synthetic knowledge. Low eps0 starts mostly greedy.
# Short n_episodes per supervisor: "don't overtrain for a long time."
# ---------------------------------------------------------------------------
FINETUNE_HP = {
    **SHARED_HP,              # inherit everything
    "n_episodes":    100,     # short run — supervisor instruction
    "lr":            0.0001,  # lower — preserve synthetic-trained weights
    "eps0":          0.10,    # start mostly greedy — weights already learned
    "eps_min":       0.02,
    "eps_decay_frac": 0.50,
    "beta_per_init": 0.60,    # start closer to full IS correction
    "buffer_min":    200,     # less data — start training sooner
}

# Fine-tuned model output paths — separate from synthetic models
# so original weights are preserved for comparison
FINETUNE_MODEL_FILES = {
    "sp":  f"{MODELS_DIR}/sp_finetuned.pt",
    "als": f"{MODELS_DIR}/als_finetuned.pt",
    "igh": f"{MODELS_DIR}/igh_finetuned.pt",
    "fs":  f"{MODELS_DIR}/fs_finetuned.pt",
}

# Real NS-3 data paths for fine-tuning
# These CSVs come from the bridge pipeline run on real NS-3 simulation data
AGENT_INPUTS_DIR = "data/agent_inputs"
FINETUNE_DATA_FILES = {
    "sp":  f"{AGENT_INPUTS_DIR}/sp_state.csv",
    "als": f"{AGENT_INPUTS_DIR}/als_state.csv",
    "igh": f"{AGENT_INPUTS_DIR}/igh_state.csv",
    "fs":  f"{AGENT_INPUTS_DIR}/fs_state.csv",
}
