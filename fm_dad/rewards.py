"""
rewards.py — Attack-specific reward functions for all four FM-DAD agents.

Master reward formula (Eq. 3.46):
    r_t = w1*r_sec - w2*r_fp - w3*r_qos - w4*r_end

Components:
    r_sec  — graded security reward   (supervisor patch; replaces binary Eqs. 3.48/3.50/3.52/3.54)
    r_fp   — false-positive penalty   (Eqs. 3.49, 3.51, 3.53, 3.55 — UNCHANGED)
    r_qos  — QoS degradation penalty  (Eq. 3.56)
    r_end  — endogenous penalty       (Eq. 3.47)

Supervisor review — Issue 1 (graded r_sec):
    The original specification used a binary r_sec = 1 for any non-zero action
    on a confirmed attacker.  This gave the same reward for a1 through a4,
    giving the agent no incentive to calibrate response magnitude.

    The updated formula is:
        r_sec^X(t) = [is_attacker] × (1 - |a_t - a*(E^X)| / 4)

    where E^X ∈ [0,1] is a continuous evidence severity score derived from the
    state vector features, and a*(E^X) ∈ {1,2,3,4} is the evidence-calibrated
    target action mapped via three thresholds e1 < e2 < e3 (grid-search params).

    r_fp is UNCHANGED — only r_sec changes.

One compute_reward_<agent>() function per attack type.
All thresholds are pulled from config.py; nothing is hardcoded.

NO attack_type feature is used anywhere (R2).
"""

from config import get_logger, SHARED_HP

logger = get_logger("rewards")

# Convenience aliases for shared threshold constants
_RHO_LOW     = SHARED_HP["rho_recv_low"]   # rho_recv < RHO_LOW → upstream victim
_LAMBDA_HIGH = SHARED_HP["lambda_high"]    # lambda_t > LAMBDA_HIGH → mobility noise
_D_REF       = SHARED_HP["d_ref"]          # reference delay (ms)
_PDR_REF     = SHARED_HP["PDR_ref"]        # reference PDR


# ---------------------------------------------------------------------------
# Shared sub-components — unchanged from original specification
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
# Graded r_sec helpers (supervisor Issue 1 — replaces binary Eqs. 3.48/3.50/3.52/3.54)
# ---------------------------------------------------------------------------

def _evidence_severity(raw: float, threshold: float, max_val: float) -> float:
    """
    Compute normalised evidence severity score E ∈ [0, 1].

    E = clamp((raw - threshold) / (max_val - threshold), 0, 1)

    E = 0 when raw == threshold (node barely exceeds detection gate).
    E = 1 when raw >= max_val  (maximum observed evidence strength).

    Args:
        raw       : Observed feature value (e.g. dFF, SpoofDev).
        threshold : Detection gate threshold (eta_*) from agent config.
        max_val   : Upper clamp value for normalisation (typically 1.0).

    Returns:
        float: Severity E ∈ [0, 1].

    Implements: E^X normalisation from supervisor's graded r_sec patch.
    """
    if max_val <= threshold:
        return 0.0
    return float(max(0.0, min(1.0, (raw - threshold) / (max_val - threshold))))


def _target_action(E: float, e1: float, e2: float, e3: float) -> int:
    """
    Map evidence severity E ∈ [0, 1] to target action a* ∈ {1, 2, 3, 4}.

    Thresholds e1 < e2 < e3 partition the severity range into four bands:
        E < e1           →  a* = 1  (small penalty — low-confidence detection)
        e1 ≤ E < e2      →  a* = 2  (moderate penalty)
        e2 ≤ E < e3      →  a* = 3  (large penalty)
        E ≥ e3           →  a* = 4  (maximum penalty — high-confidence detection)

    Placeholder defaults: e1=0.25, e2=0.50, e3=0.75 (uniform split).
    Final values determined by grid search (see config.py per-agent entries).

    Args:
        E          : Evidence severity ∈ [0, 1].
        e1, e2, e3 : Severity thresholds (e1 < e2 < e3) from agent config.

    Returns:
        int: Target action index a* ∈ {1, 2, 3, 4}.

    Implements: a*(E^X) mapping from supervisor's graded r_sec patch.
    """
    if E < e1:
        return 1
    elif E < e2:
        return 2
    elif E < e3:
        return 3
    else:
        return 4


def _graded_r_sec(
    action:      int,
    is_attacker: int,
    E:           float,
    e1:          float,
    e2:          float,
    e3:          float,
) -> float:
    """
    Compute the graded security reward r_sec^X (supervisor's updated formula).

    r_sec^X(t) = [is_attacker] × (1 - |a_t - a*(E^X)| / 4)

    When is_attacker == 1:
        r_sec = 1.0  when action perfectly matches a*(E)  (full credit)
        r_sec = 0.75 when action is 1 step from a*(E)     (partial credit)
        r_sec = 0.50 when 2 steps away
        r_sec = 0.25 when 3 steps away
        r_sec = 0.0  when 4 steps away                    (maximum mismatch)

    When is_attacker == 0:
        r_sec = 0.0  (no change from binary form)

    Replaces the original binary r_sec (Eqs. 3.48/3.50/3.52/3.54) with a
    proportional form that provides a gradient signal for action magnitude
    calibration. This fixes the a1-saturation problem observed at 100 episodes.

    Args:
        action      : Chosen action index a_t ∈ {0..4}.
        is_attacker : Ground-truth label (0 = innocent, 1 = attacker).
        E           : Evidence severity score ∈ [0, 1] computed from state.
        e1, e2, e3  : Severity thresholds for a*(E) mapping.

    Returns:
        float: Graded r_sec ∈ [0.0, 1.0].

    Implements: Graded r_sec from supervisor review
                (replaces binary Eqs. 3.48/3.50/3.52/3.54).
    """
    if is_attacker != 1:
        return 0.0
    a_star = _target_action(E, e1, e2, e3)
    return float(1.0 - abs(action - a_star) / 4.0)


# ---------------------------------------------------------------------------
# IGH reward (updated graded r_sec + unchanged r_fp Eq. 3.49)
# ---------------------------------------------------------------------------

def compute_reward_igh(
    action:            int,
    is_attacker:       int,
    rho_recv:          float,
    pdr_var:           float,
    coord_score:       float,
    d_bar_t:           float,
    PDR_t:             float,
    blockchain_reject: int,
    cfg:               dict,
) -> float:
    """
    Compute reward for the IGH (Interleaved Grey Hole) agent.

    E^IGH = mean of three normalised excesses (supervisor Issue 1):
        - PDRVar excess above eta_pdrvar
        - CoordScore excess above eta_coord
        - rho_recv excess above eta_rho

    r_sec (graded, supervisor patch):
        r_sec = 1 - |a_t - a*(E^IGH)| / 4  when is_attacker == 1, else 0.

    r_fp  (Eq. 3.49, UNCHANGED):
        r_fp = 1 if action > a0 AND is_attacker == 0 AND rho_recv < rho_recv_low.

    Master formula (Eq. 3.46):
        r_t = w1*r_sec - w2*r_fp - w3*r_qos - w4*r_end

    Args:
        action            : Chosen action index a_t ∈ {0..4}.
        is_attacker       : Ground-truth label (0 = innocent, 1 = IGH attacker).
        rho_recv          : Receipt ratio (state feature, used for E^IGH and r_fp).
        pdr_var           : PDRVar (state feature, used for E^IGH).
        coord_score       : CoordScore (state feature, used for E^IGH).
        d_bar_t           : Current mean per-hop delay for r_qos (Eq. 3.56).
        PDR_t             : Current PDR for r_qos (Eq. 3.56).
        blockchain_reject : Blockchain action flag for r_end (Eq. 3.47).
        cfg               : Agent config dict (contains thresholds and w1..w4).

    Returns:
        float: Total scalar reward r_t.

    Implements: Eq. 3.46, graded r_sec (supervisor patch), Eq. 3.49 (r_fp).
    """
    # E^IGH — Eq. 3.22-related severity (supervisor Issue 1)
    E_pdrvar = _evidence_severity(pdr_var,     cfg["eta_pdrvar"], 1.0)
    E_coord  = _evidence_severity(coord_score, cfg["eta_coord"],  1.0)
    E_rho    = _evidence_severity(rho_recv,    cfg["eta_rho"],    1.0)
    E_igh    = (E_pdrvar + E_coord + E_rho) / 3.0

    # r_sec — graded (supervisor patch replaces binary Eq. 3.48)
    r_sec = _graded_r_sec(action, is_attacker, E_igh, cfg["e1"], cfg["e2"], cfg["e3"])

    # r_fp — Eq. 3.49 (UNCHANGED)
    upstream_victim = (is_attacker == 0) and (rho_recv < _RHO_LOW)
    r_fp  = 1.0 if (action > 0 and upstream_victim) else 0.0

    r_qos = _r_qos(d_bar_t, PDR_t)
    r_end = _r_end(action, blockchain_reject)

    r_t = cfg["w1"] * r_sec - cfg["w2"] * r_fp - cfg["w3"] * r_qos - cfg["w4"] * r_end
    logger.debug(
        "IGH reward | action=%d, is_attacker=%d, E_igh=%.3f, a*=%d, "
        "r_sec=%.3f, r_fp=%.1f, r_qos=%.4f, r_end=%.1f -> r_t=%.4f",
        action, is_attacker, E_igh,
        _target_action(E_igh, cfg["e1"], cfg["e2"], cfg["e3"]),
        r_sec, r_fp, r_qos, r_end, r_t,
    )
    return float(r_t)


# ---------------------------------------------------------------------------
# SP reward (updated graded r_sec + unchanged r_fp Eq. 3.51)
# ---------------------------------------------------------------------------

def compute_reward_sp(
    action:            int,
    is_attacker:       int,
    rho_recv:          float,
    dFF:               float,
    d_bar_t:           float,
    PDR_t:             float,
    blockchain_reject: int,
    cfg:               dict,
) -> float:
    """
    Compute reward for the SP (Selective Packet Dropping) agent.

    E^SP = normalised dFF excess above eta_dFF (supervisor Issue 1).

    r_sec (graded, supervisor patch):
        r_sec = 1 - |a_t - a*(E^SP)| / 4  when is_attacker == 1, else 0.

    # NOTE: is_attacker is the TRUE ground-truth attacker label.
    # The dFF > eta_dFF gate decides IF the agent is invoked.
    # is_attacker decides WHETHER penalising was correct (r_sec)
    # or a false positive (r_fp).  They are always separate.

    r_fp  (Eq. 3.51, UNCHANGED):
        r_fp = 1 if action > a0 AND is_attacker == 0 AND rho_recv < rho_recv_low.

    Master formula (Eq. 3.46):
        r_t = w1*r_sec - w2*r_fp - w3*r_qos - w4*r_end

    Args:
        action            : Chosen action index a_t ∈ {0..4}.
        is_attacker       : Ground-truth label (0 = innocent, 1 = SP attacker).
        rho_recv          : Receipt ratio (used for r_fp false-positive detection).
        dFF               : Flow-table deviation (used for E^SP severity).
        d_bar_t           : Current mean per-hop delay for r_qos (Eq. 3.56).
        PDR_t             : Current PDR for r_qos (Eq. 3.56).
        blockchain_reject : Blockchain action flag for r_end (Eq. 3.47).
        cfg               : Agent config dict (contains thresholds and w1..w4).

    Returns:
        float: Total scalar reward r_t.

    Implements: Eq. 3.46, graded r_sec (supervisor patch), Eq. 3.51 (r_fp).
    """
    # E^SP — normalised dFF excess (supervisor Issue 1)
    E_sp = _evidence_severity(dFF, cfg["eta_dFF"], 1.0)

    # r_sec — graded (supervisor patch replaces binary Eq. 3.50)
    r_sec = _graded_r_sec(action, is_attacker, E_sp, cfg["e1"], cfg["e2"], cfg["e3"])

    # r_fp — Eq. 3.51 (UNCHANGED)
    upstream_victim = (is_attacker == 0) and (rho_recv < _RHO_LOW)
    r_fp  = 1.0 if (action > 0 and upstream_victim) else 0.0

    r_qos = _r_qos(d_bar_t, PDR_t)
    r_end = _r_end(action, blockchain_reject)

    r_t = cfg["w1"] * r_sec - cfg["w2"] * r_fp - cfg["w3"] * r_qos - cfg["w4"] * r_end
    logger.debug(
        "SP reward | action=%d, is_attacker=%d, E_sp=%.3f, a*=%d, "
        "r_sec=%.3f, r_fp=%.1f, r_qos=%.4f, r_end=%.1f -> r_t=%.4f",
        action, is_attacker, E_sp,
        _target_action(E_sp, cfg["e1"], cfg["e2"], cfg["e3"]),
        r_sec, r_fp, r_qos, r_end, r_t,
    )
    return float(r_t)


# ---------------------------------------------------------------------------
# ALS reward (updated graded r_sec + unchanged r_fp Eq. 3.53)
# ---------------------------------------------------------------------------

def compute_reward_als(
    action:            int,
    is_attacker:       int,
    lambda_t:          float,
    spoof_dev:         float,
    d_bar_t:           float,
    PDR_t:             float,
    blockchain_reject: int,
    cfg:               dict,
) -> float:
    """
    Compute reward for the ALS (Asymmetric Link Spoofing) agent.

    E^ALS = normalised SpoofDev excess above eta_spoof (supervisor Issue 1).

    r_sec (graded, supervisor patch):
        r_sec = 1 - |a_t - a*(E^ALS)| / 4  when is_attacker == 1, else 0.

    r_fp  (Eq. 3.53, UNCHANGED):
        r_fp = 1 if action > a0 AND is_attacker == 0 AND lambda_t > lambda_high.

    Master formula (Eq. 3.46):
        r_t = w1*r_sec - w2*r_fp - w3*r_qos - w4*r_end

    Args:
        action            : Chosen action index a_t ∈ {0..4}.
        is_attacker       : Ground-truth label (0 = innocent, 1 = ALS attacker).
        lambda_t          : Topology change rate (used for r_fp mobility check).
        spoof_dev         : SpoofDev metric (used for E^ALS severity).
        d_bar_t           : Current mean per-hop delay for r_qos (Eq. 3.56).
        PDR_t             : Current PDR for r_qos (Eq. 3.56).
        blockchain_reject : Blockchain action flag for r_end (Eq. 3.47).
        cfg               : Agent config dict (contains thresholds and w1..w4).

    Returns:
        float: Total scalar reward r_t.

    Implements: Eq. 3.46, graded r_sec (supervisor patch), Eq. 3.53 (r_fp).
    """
    # E^ALS — normalised SpoofDev excess (supervisor Issue 1)
    E_als = _evidence_severity(spoof_dev, cfg["eta_spoof"], 1.0)

    # r_sec — graded (supervisor patch replaces binary Eq. 3.52)
    r_sec = _graded_r_sec(action, is_attacker, E_als, cfg["e1"], cfg["e2"], cfg["e3"])

    # r_fp — Eq. 3.53 (UNCHANGED)
    mobility_noise = (is_attacker == 0) and (lambda_t > _LAMBDA_HIGH)
    r_fp  = 1.0 if (action > 0 and mobility_noise) else 0.0

    r_qos = _r_qos(d_bar_t, PDR_t)
    r_end = _r_end(action, blockchain_reject)

    r_t = cfg["w1"] * r_sec - cfg["w2"] * r_fp - cfg["w3"] * r_qos - cfg["w4"] * r_end
    logger.debug(
        "ALS reward | action=%d, is_attacker=%d, E_als=%.3f, a*=%d, "
        "r_sec=%.3f, r_fp=%.1f, r_qos=%.4f, r_end=%.1f -> r_t=%.4f",
        action, is_attacker, E_als,
        _target_action(E_als, cfg["e1"], cfg["e2"], cfg["e3"]),
        r_sec, r_fp, r_qos, r_end, r_t,
    )
    return float(r_t)


# ---------------------------------------------------------------------------
# FS reward (updated graded r_sec + unchanged r_fp Eq. 3.55)
# ---------------------------------------------------------------------------

def compute_reward_fs(
    action:            int,
    is_attacker:       int,
    rho_recv:          float,
    lambda_t:          float,
    dFF:               float,
    delay_infl:        float,
    d_bar_t:           float,
    PDR_t:             float,
    blockchain_reject: int,
    cfg:               dict,
) -> float:
    """
    Compute reward for the FS (Flow Stretching) agent.

    E^FS = mean of two normalised excesses (supervisor Issue 1):
        - dFF excess above eta_dFF
        - DelayInfl excess above eta_delay

    # NOTE: is_attacker is the TRUE ground-truth attacker label.
    # The detection gate (dFF > eta_dFF on Stage-1 flagged path) decides
    # IF the agent is invoked.  is_attacker decides WHETHER penalising
    # was correct (r_sec) or a false positive (r_fp). Always separate.

    r_sec (graded, supervisor patch):
        r_sec = 1 - |a_t - a*(E^FS)| / 4  when is_attacker == 1, else 0.

    r_fp  (Eq. 3.55, UNCHANGED):
        r_fp = 1 if action > a0 AND is_attacker == 0
                 AND (rho_recv < rho_recv_low OR lambda_t > lambda_high).

    Master formula (Eq. 3.46):
        r_t = w1*r_sec - w2*r_fp - w3*r_qos - w4*r_end

    Args:
        action            : Chosen action index a_t ∈ {0..4}.
        is_attacker       : Ground-truth label (0 = innocent, 1 = FS attacker).
        rho_recv          : Receipt ratio (used for r_fp OR condition).
        lambda_t          : Topology change rate (used for r_fp OR condition).
        dFF               : Flow-table deviation (used for E^FS severity).
        delay_infl        : DelayInfl = d_bar / d_ref (used for E^FS severity).
        d_bar_t           : Current mean per-hop delay for r_qos (Eq. 3.56).
        PDR_t             : Current PDR for r_qos (Eq. 3.56).
        blockchain_reject : Blockchain action flag for r_end (Eq. 3.47).
        cfg               : Agent config dict (contains thresholds and w1..w4).

    Returns:
        float: Total scalar reward r_t.

    Implements: Eq. 3.46, graded r_sec (supervisor patch), Eq. 3.55 (r_fp).
    """
    # E^FS — mean of normalised dFF and DelayInfl excesses (supervisor Issue 1)
    E_dff   = _evidence_severity(dFF,        cfg["eta_dFF"],   1.0)
    E_delay = _evidence_severity(delay_infl, cfg["eta_delay"], cfg["delay_max"])
    E_fs    = (E_dff + E_delay) / 2.0

    # r_sec — graded (supervisor patch replaces binary Eq. 3.54)
    r_sec = _graded_r_sec(action, is_attacker, E_fs, cfg["e1"], cfg["e2"], cfg["e3"])

    # r_fp — Eq. 3.55 (UNCHANGED, OR condition)
    fp_condition = (rho_recv < _RHO_LOW) or (lambda_t > _LAMBDA_HIGH)
    r_fp  = 1.0 if (action > 0 and is_attacker == 0 and fp_condition) else 0.0

    r_qos = _r_qos(d_bar_t, PDR_t)
    r_end = _r_end(action, blockchain_reject)

    r_t = cfg["w1"] * r_sec - cfg["w2"] * r_fp - cfg["w3"] * r_qos - cfg["w4"] * r_end
    logger.debug(
        "FS reward | action=%d, is_attacker=%d, E_fs=%.3f, a*=%d, "
        "r_sec=%.3f, r_fp=%.1f, r_qos=%.4f, r_end=%.1f -> r_t=%.4f",
        action, is_attacker, E_fs,
        _target_action(E_fs, cfg["e1"], cfg["e2"], cfg["e3"]),
        r_sec, r_fp, r_qos, r_end, r_t,
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
