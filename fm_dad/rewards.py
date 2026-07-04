"""
rewards.py — Attack-specific reward functions for all four FM-DAD agents.

Master reward formula (Eq. 3.46):
    r_t = w1*r_sec - w2*r_fp - w3*r_qos - w4*r_end

Components:
    r_sec  — security reward (Eqs. 3.48, 3.50, 3.52, 3.54)
    r_fp   — false-positive penalty (Eqs. 3.49, 3.51, 3.53, 3.55)
    r_qos  — QoS degradation penalty (Eq. 3.56)
    r_end  — endogenous penalty (Eq. 3.47)

One compute_reward_<agent>() function per attack type.
All thresholds are pulled from SHARED_HP in config.py; nothing is hardcoded.

NO attack_type feature is used anywhere (R2).
"""

from config import get_logger, SHARED_HP

logger = get_logger("rewards")

# Convenience aliases for threshold constants
_RHO_LOW    = SHARED_HP["rho_recv_low"]   # "rho_recv << 1" threshold (false-positive guard)
_LAMBDA_HIGH = SHARED_HP["lambda_high"]   # "lambda_t high" threshold
_D_REF      = SHARED_HP["d_ref"]          # reference delay (ms)
_PDR_REF    = SHARED_HP["PDR_ref"]        # reference PDR


# ---------------------------------------------------------------------------
# Shared sub-components
# ---------------------------------------------------------------------------

def _r_qos(d_bar_t: float, PDR_t: float) -> float:
    """
    QoS degradation penalty term (Eq. 3.56).

    r_qos = (d_bar_t - d_ref)/d_ref + (PDR_ref - PDR_t)/PDR_ref

    Positive when network quality is worse than baseline (delay up, PDR down).

    Args:
        d_bar_t : Current mean per-hop delay (ms).
        PDR_t   : Current packet delivery ratio.

    Returns:
        float: r_qos value (can be negative if network is better than baseline).

    Implements: Eq. 3.56.
    """
    delay_term = (d_bar_t - _D_REF) / (_D_REF + 1e-9)
    pdr_term   = (_PDR_REF - PDR_t) / (_PDR_REF + 1e-9)
    r = delay_term + pdr_term
    logger.debug("_r_qos | d_bar_t=%.3f, PDR_t=%.3f -> r_qos=%.4f", d_bar_t, PDR_t, r)
    return r


def _r_end(action: int, blockchain_reject: int) -> float:
    """
    Endogenous penalty term (Eq. 3.47).

    r_end = 1 if the agent issued a non-zero trust reduction AND the blockchain
            could not act on it (blockchain_reject == 1), else 0.

    For offline training with synthetic data the blockchain_reject column is
    present in the CSV (default 0 = blockchain accepted the action).

    Args:
        action            : Chosen action index (0–4).
        blockchain_reject : Ground-truth flag from training CSV (0 or 1).

    Returns:
        float: 1.0 if non-zero action was rejected, else 0.0.

    Implements: Eq. 3.47.
    """
    r = 1.0 if (action > 0 and blockchain_reject == 1) else 0.0
    logger.debug(
        "_r_end | action=%d, blockchain_reject=%d -> r_end=%.1f",
        action, blockchain_reject, r,
    )
    return r


# ---------------------------------------------------------------------------
# IGH reward (Eqs. 3.48–3.49)
# ---------------------------------------------------------------------------

def compute_reward_igh(
    action:           int,
    is_attacker:      int,
    rho_recv:         float,
    d_bar_t:          float,
    PDR_t:            float,
    blockchain_reject: int,
    cfg:              dict,
) -> float:
    """
    Compute reward for the IGH (Interleaved Grey Hole) agent (Eqs. 3.46–3.49).

    r_sec (Eq. 3.48): 1 if action > a0 AND is_attacker == 1, else 0.
    r_fp  (Eq. 3.49): 1 if action > a0 AND is_attacker == 0
                        AND rho_recv < rho_recv_low (upstream victim), else 0.

    Master formula (Eq. 3.46):
        r_t = w1*r_sec - w2*r_fp - w3*r_qos - w4*r_end

    Args:
        action            : Chosen action index a_t ∈ {0..4}.
        is_attacker       : Ground-truth label (0 = innocent, 1 = IGH attacker).
        rho_recv          : Receipt ratio (state feature).
        d_bar_t           : Current mean per-hop delay for r_qos (Eq. 3.56).
        PDR_t             : Current PDR for r_qos (Eq. 3.56).
        blockchain_reject : Blockchain action flag for r_end (Eq. 3.47).
        cfg               : Agent config dict (contains w1..w4).

    Returns:
        float: Total scalar reward r_t.

    Implements: Eq. 3.46, Eq. 3.48 (r_sec), Eq. 3.49 (r_fp).
    """
    # r_sec — Eq. 3.48
    r_sec = 1.0 if (action > 0 and is_attacker == 1) else 0.0

    # r_fp  — Eq. 3.49
    upstream_victim = (is_attacker == 0) and (rho_recv < _RHO_LOW)
    r_fp  = 1.0 if (action > 0 and upstream_victim) else 0.0

    r_qos = _r_qos(d_bar_t, PDR_t)
    r_end = _r_end(action, blockchain_reject)

    r_t = cfg["w1"] * r_sec - cfg["w2"] * r_fp - cfg["w3"] * r_qos - cfg["w4"] * r_end
    logger.debug(
        "IGH reward | action=%d, is_attacker=%d, r_sec=%.1f, r_fp=%.1f, "
        "r_qos=%.4f, r_end=%.1f -> r_t=%.4f",
        action, is_attacker, r_sec, r_fp, r_qos, r_end, r_t,
    )
    return float(r_t)


# ---------------------------------------------------------------------------
# SP reward (Eqs. 3.50–3.51)
# ---------------------------------------------------------------------------

def compute_reward_sp(
    action:           int,
    is_attacker:      int,
    rho_recv:         float,
    d_bar_t:          float,
    PDR_t:            float,
    blockchain_reject: int,
    cfg:              dict,
) -> float:
    """
    Compute reward for the SP (Selective Packet Dropping) agent (Eqs. 3.46, 3.50–3.51).

    r_sec (Eq. 3.50): 1 if action > a0 AND is_attacker == 1
                        (confirmed via dFF > eta_FF — represented by is_attacker label
                         in the training data, which is set after dFF threshold check).
    r_fp  (Eq. 3.51): 1 if action > a0 AND is_attacker == 0
                        AND rho_recv < rho_recv_low (upstream victim), else 0.

    # ASSUMPTION: The spec says "confirmed via dFF > eta_FF". The dFF threshold
    # check is a preprocessing step already baked into the is_attacker label in
    # the training CSV. The reward function uses is_attacker directly, consistent
    # with how the spec treats all other agents.

    Args:
        action            : Chosen action index a_t ∈ {0..4}.
        is_attacker       : Ground-truth label (0 = innocent, 1 = SP attacker).
        rho_recv          : Receipt ratio (used for false-positive detection).
        d_bar_t           : Current mean per-hop delay for r_qos (Eq. 3.56).
        PDR_t             : Current PDR for r_qos (Eq. 3.56).
        blockchain_reject : Blockchain action flag for r_end (Eq. 3.47).
        cfg               : Agent config dict (contains w1..w4).

    Returns:
        float: Total scalar reward r_t.

    Implements: Eq. 3.46, Eq. 3.50 (r_sec), Eq. 3.51 (r_fp).
    """
    # r_sec — Eq. 3.50
    r_sec = 1.0 if (action > 0 and is_attacker == 1) else 0.0

    # r_fp  — Eq. 3.51
    upstream_victim = (is_attacker == 0) and (rho_recv < _RHO_LOW)
    r_fp  = 1.0 if (action > 0 and upstream_victim) else 0.0

    r_qos = _r_qos(d_bar_t, PDR_t)
    r_end = _r_end(action, blockchain_reject)

    r_t = cfg["w1"] * r_sec - cfg["w2"] * r_fp - cfg["w3"] * r_qos - cfg["w4"] * r_end
    logger.debug(
        "SP reward | action=%d, is_attacker=%d, r_sec=%.1f, r_fp=%.1f, "
        "r_qos=%.4f, r_end=%.1f -> r_t=%.4f",
        action, is_attacker, r_sec, r_fp, r_qos, r_end, r_t,
    )
    return float(r_t)


# ---------------------------------------------------------------------------
# ALS reward (Eqs. 3.52–3.53)
# ---------------------------------------------------------------------------

def compute_reward_als(
    action:           int,
    is_attacker:      int,
    lambda_t:         float,
    d_bar_t:          float,
    PDR_t:            float,
    blockchain_reject: int,
    cfg:              dict,
) -> float:
    """
    Compute reward for the ALS (Asymmetric Link Spoofing) agent (Eqs. 3.46, 3.52–3.53).

    r_sec (Eq. 3.52): 1 if action > a0 AND is_attacker == 1 (confirmed spoofing).
    r_fp  (Eq. 3.53): 1 if action > a0 AND is_attacker == 0
                        AND lambda_t > lambda_high (high mobility → mobility noise
                        mistaken for spoofing), else 0.

    Args:
        action            : Chosen action index a_t ∈ {0..4}.
        is_attacker       : Ground-truth label (0 = innocent, 1 = ALS attacker).
        lambda_t          : Topology change rate (state feature).
        d_bar_t           : Current mean per-hop delay for r_qos (Eq. 3.56).
        PDR_t             : Current PDR for r_qos (Eq. 3.56).
        blockchain_reject : Blockchain action flag for r_end (Eq. 3.47).
        cfg               : Agent config dict (contains w1..w4).

    Returns:
        float: Total scalar reward r_t.

    Implements: Eq. 3.46, Eq. 3.52 (r_sec), Eq. 3.53 (r_fp).
    """
    # r_sec — Eq. 3.52
    r_sec = 1.0 if (action > 0 and is_attacker == 1) else 0.0

    # r_fp  — Eq. 3.53
    mobility_noise = (is_attacker == 0) and (lambda_t > _LAMBDA_HIGH)
    r_fp  = 1.0 if (action > 0 and mobility_noise) else 0.0

    r_qos = _r_qos(d_bar_t, PDR_t)
    r_end = _r_end(action, blockchain_reject)

    r_t = cfg["w1"] * r_sec - cfg["w2"] * r_fp - cfg["w3"] * r_qos - cfg["w4"] * r_end
    logger.debug(
        "ALS reward | action=%d, is_attacker=%d, r_sec=%.1f, r_fp=%.1f, "
        "r_qos=%.4f, r_end=%.1f -> r_t=%.4f",
        action, is_attacker, r_sec, r_fp, r_qos, r_end, r_t,
    )
    return float(r_t)


# ---------------------------------------------------------------------------
# FS reward (Eqs. 3.54–3.55)
# ---------------------------------------------------------------------------

def compute_reward_fs(
    action:           int,
    is_attacker:      int,
    rho_recv:         float,
    lambda_t:         float,
    d_bar_t:          float,
    PDR_t:            float,
    blockchain_reject: int,
    cfg:              dict,
) -> float:
    """
    Compute reward for the FS (Flow Stretching) agent (Eqs. 3.46, 3.54–3.55).

    r_sec (Eq. 3.54): 1 if action > a0 AND is_attacker == 1
                        (dFF > eta_FF on Stage-1 flagged path — baked into label).
    r_fp  (Eq. 3.55): 1 if action > a0 AND is_attacker == 0
                        AND (rho_recv < rho_recv_low OR lambda_t > lambda_high).

    Args:
        action            : Chosen action index a_t ∈ {0..4}.
        is_attacker       : Ground-truth label (0 = innocent, 1 = FS attacker).
        rho_recv          : Receipt ratio (used in false-positive condition).
        lambda_t          : Topology change rate (used in false-positive condition).
        d_bar_t           : Current mean per-hop delay for r_qos (Eq. 3.56).
        PDR_t             : Current PDR for r_qos (Eq. 3.56).
        blockchain_reject : Blockchain action flag for r_end (Eq. 3.47).
        cfg               : Agent config dict (contains w1..w4).

    Returns:
        float: Total scalar reward r_t.

    Implements: Eq. 3.46, Eq. 3.54 (r_sec), Eq. 3.55 (r_fp).
    """
    # r_sec — Eq. 3.54
    r_sec = 1.0 if (action > 0 and is_attacker == 1) else 0.0

    # r_fp  — Eq. 3.55 (OR condition)
    fp_condition = (rho_recv < _RHO_LOW) or (lambda_t > _LAMBDA_HIGH)
    r_fp  = 1.0 if (action > 0 and is_attacker == 0 and fp_condition) else 0.0

    r_qos = _r_qos(d_bar_t, PDR_t)
    r_end = _r_end(action, blockchain_reject)

    r_t = cfg["w1"] * r_sec - cfg["w2"] * r_fp - cfg["w3"] * r_qos - cfg["w4"] * r_end
    logger.debug(
        "FS reward | action=%d, is_attacker=%d, r_sec=%.1f, r_fp=%.1f, "
        "r_qos=%.4f, r_end=%.1f -> r_t=%.4f",
        action, is_attacker, r_sec, r_fp, r_qos, r_end, r_t,
    )
    return float(r_t)


# ---------------------------------------------------------------------------
# Dispatcher — maps reward_fn id to the correct function
# ---------------------------------------------------------------------------

REWARD_FN_MAP = {
    "igh": compute_reward_igh,
    "sp":  compute_reward_sp,
    "als": compute_reward_als,
    "fs":  compute_reward_fs,
}
