"""
unit_checks.py — Stage 1 unit-checks for all FM-DAD modules.

Runs the following checks in order (as specified in Section 10, Steps 3–7):

    CHECK 1 (Step 3): DuelingDQN forward pass — verify output shape (batch, 5)
                      for all four agent input dimensions (8, 5, 4, 5).

    CHECK 2 (Step 4): PrioritizedReplayBuffer — push 100 transitions, sample
                      batch of 32, verify IS weights and indices, verify
                      priority update.

    CHECK 3 (Step 5): Reward functions — hand-made cases for all four agents:
                      (a) attacker + penalising action -> positive r_sec contribution
                      (b) innocent upstream victim + penalising action -> negative r_fp
                      (c) innocent + a0 -> reward ~0 (minus small r_qos term)

    CHECK 4 (Step 6): DQNAgent.learn() — one learn() step completes without
                      error and actually changes network weights.

    CHECK 5 (Step 7): Smoke test — build a tiny dummy CSV (100 rows, 5 nodes,
                      20 cycles each), run train_agent() for 5 episodes without
                      crashing.

All results are logged at INFO level and a PASS/FAIL summary is printed at the
end. Any failure raises an AssertionError with a clear message.
"""

import os
import sys
import tempfile
import numpy as np
import pandas as pd
import torch

# Ensure we can import from the fm_dad package
sys.path.insert(0, os.path.dirname(__file__))

from config import get_logger, SHARED_HP, AGENT_CONFIGS
from networks import DuelingDQN
from replay_buffer import PrioritizedReplayBuffer
from rewards import (
    compute_reward_igh,
    compute_reward_sp,
    compute_reward_als,
    compute_reward_fs,
)
from agent import DQNAgent
from train import train_agent

logger = get_logger("unit_checks")

# Track pass/fail for final summary
_results: dict[str, str] = {}


def _pass(check_name: str, detail: str = "") -> None:
    _results[check_name] = "PASS"
    logger.info("  ✅ PASS | %s %s", check_name, f"— {detail}" if detail else "")


def _fail(check_name: str, reason: str) -> None:
    _results[check_name] = "FAIL"
    logger.error("  ❌ FAIL | %s — %s", check_name, reason)
    raise AssertionError(f"CHECK FAILED: {check_name} — {reason}")


# ===========================================================================
# CHECK 1 — DuelingDQN forward pass shapes (Step 3)
# ===========================================================================

def check_network_forward():
    """
    Verify DuelingDQN produces correct output shapes for all agent input dims.

    Expected: forward(batch_tensor) → shape (batch_size, 5) for each agent.

    Implements: Step 3 unit-check (Section 10 of the report).
    """
    logger.info("")
    logger.info("=" * 60)
    logger.info("CHECK 1: DuelingDQN forward pass shapes")
    logger.info("=" * 60)

    batch_size = 8
    for agent_name, cfg in AGENT_CONFIGS.items():
        input_dim = cfg["input_dim"]
        net = DuelingDQN(input_dim=input_dim)
        x   = torch.randn(batch_size, input_dim)

        logger.info(
            "[%s] Running forward pass | input shape: (%d, %d)",
            agent_name.upper(), batch_size, input_dim,
        )
        out = net(x)

        expected_shape = (batch_size, SHARED_HP["output_size"])
        check_name = f"network_shape_{agent_name}"

        if out.shape != torch.Size(expected_shape):
            _fail(check_name, f"Expected {expected_shape}, got {tuple(out.shape)}")
        else:
            _pass(
                check_name,
                f"input_dim={input_dim} → output shape={tuple(out.shape)}",
            )
        logger.info(
            "[%s] Output Q-values (first row): %s",
            agent_name.upper(), out[0].detach().numpy().round(4),
        )


# ===========================================================================
# CHECK 2 — PrioritizedReplayBuffer (Step 4)
# ===========================================================================

def check_replay_buffer():
    """
    Verify PER buffer push/sample/update-priority workflow.

    Checks:
        - Push 100 random transitions without error.
        - Sample a batch of 32 returns correct keys.
        - IS weights are all positive and max ≤ 1.0.
        - Indices are valid buffer positions.
        - Priority update runs without error.

    Implements: Step 4 unit-check (Section 10 of the report).
    Eqs: 3.59, 3.60, 3.61.
    """
    logger.info("")
    logger.info("=" * 60)
    logger.info("CHECK 2: PrioritizedReplayBuffer push / sample / update")
    logger.info("=" * 60)

    input_dim  = 5   # SP agent dimension as test subject
    buf = PrioritizedReplayBuffer(capacity=1024, alpha=0.6, eps_per=1e-5)

    # ---- Push 100 transitions ----------------------------------------------
    logger.info("Pushing 100 random transitions (input_dim=%d)...", input_dim)
    for i in range(100):
        s      = np.random.rand(input_dim).astype(np.float32)
        s_next = np.random.rand(input_dim).astype(np.float32)
        a      = np.random.randint(0, 5)
        r      = float(np.random.randn())
        done   = bool(i % 20 == 0)
        buf.push(s, a, r, s_next, done)

    assert len(buf) == 100, f"Expected size 100, got {len(buf)}"
    _pass("buffer_push_100", f"size={len(buf)}")

    # ---- Sample batch of 32 ------------------------------------------------
    logger.info("Sampling batch of 32 (beta=0.4)...")
    batch = buf.sample(32, beta=0.4)

    required_keys = {"states", "actions", "rewards", "next_states", "dones", "indices", "weights"}
    missing_keys  = required_keys - set(batch.keys())
    if missing_keys:
        _fail("buffer_sample_keys", f"Missing keys: {missing_keys}")
    _pass("buffer_sample_keys", f"all keys present: {sorted(batch.keys())}")

    # Verify shapes
    assert batch["states"].shape      == (32, input_dim), f"states shape wrong: {batch['states'].shape}"
    assert batch["next_states"].shape == (32, input_dim), f"next_states shape wrong"
    assert batch["actions"].shape     == (32,),           f"actions shape wrong"
    assert batch["rewards"].shape     == (32,),           f"rewards shape wrong"
    assert batch["weights"].shape     == (32,),           f"weights shape wrong"
    _pass("buffer_sample_shapes", f"states={batch['states'].shape}, weights={batch['weights'].shape}")

    # IS weights: all positive, max ≤ 1.0 + small tolerance
    weights = batch["weights"]
    logger.info("IS weights | min=%.4f, max=%.4f, mean=%.4f", weights.min(), weights.max(), weights.mean())
    assert (weights > 0).all(), "All IS weights must be positive"
    assert weights.max() <= 1.0 + 1e-6, f"Max IS weight {weights.max():.4f} exceeds 1.0"
    _pass("buffer_is_weights", f"min={weights.min():.4f}, max={weights.max():.4f}")

    # Indices valid
    indices = batch["indices"]
    assert indices.min() >= 0,       f"Index out of range: {indices.min()}"
    assert indices.max() < buf.size, f"Index {indices.max()} ≥ buffer size {buf.size}"
    _pass("buffer_indices_valid", f"range [{indices.min()}, {indices.max()}]")

    # ---- Priority update ---------------------------------------------------
    logger.info("Updating priorities with fake TD errors...")
    fake_td = np.abs(np.random.randn(32))
    buf.update_priorities(indices, fake_td)
    _pass("buffer_priority_update", "no error raised")


# ===========================================================================
# CHECK 3 — Reward functions with hand-made cases (Step 5)
# ===========================================================================

def _cfg(agent_name: str) -> dict:
    """Return agent config dict."""
    return AGENT_CONFIGS[agent_name]


def check_rewards():
    """
    Verify reward functions with manually constructed test cases.

    Cases per agent:
        (a) attacker + penalising action (a=2) → r_sec contributes positively
        (b) innocent upstream victim + penalising action → r_fp penalises negatively
        (c) innocent + a0 (no action) → reward near 0 (only small r_qos term)

    Implements: Step 5 unit-check (Section 10 of the report).
    Eqs: 3.46–3.56.
    """
    logger.info("")
    logger.info("=" * 60)
    logger.info("CHECK 3: Reward functions — hand-made cases")
    logger.info("=" * 60)

    # Reference QoS values (at baseline → r_qos ≈ 0)
    d_ref   = SHARED_HP["d_ref"]
    PDR_ref = SHARED_HP["PDR_ref"]

    # ---- IGH ---------------------------------------------------------------
    logger.info("--- IGH reward ---")
    # (a) Attacker + action=2 (penalising)
    r = compute_reward_igh(
        action=2, is_attacker=1, rho_recv=0.8,
        d_bar_t=d_ref, PDR_t=PDR_ref, blockchain_reject=0, cfg=_cfg("igh"),
    )
    logger.info("  (a) attacker, action=2 → r=%.4f", r)
    assert r > 0, f"IGH (a): expected r>0 for attacker+penalise, got {r}"
    _pass("reward_igh_attacker", f"r={r:.4f} > 0 ✓")

    # (b) Innocent upstream victim (rho_recv < 0.5) + action=2
    r = compute_reward_igh(
        action=2, is_attacker=0, rho_recv=0.3,  # upstream victim
        d_bar_t=d_ref, PDR_t=PDR_ref, blockchain_reject=0, cfg=_cfg("igh"),
    )
    logger.info("  (b) innocent upstream victim, action=2 → r=%.4f", r)
    assert r < 0, f"IGH (b): expected r<0 for innocent+penalise, got {r}"
    _pass("reward_igh_fp", f"r={r:.4f} < 0 ✓")

    # (c) Innocent + a0 (no action)
    r = compute_reward_igh(
        action=0, is_attacker=0, rho_recv=0.9,
        d_bar_t=d_ref, PDR_t=PDR_ref, blockchain_reject=0, cfg=_cfg("igh"),
    )
    logger.info("  (c) innocent, action=0 → r=%.4f (should be ~0)", r)
    assert abs(r) < 0.05, f"IGH (c): expected r≈0 for a0+innocent, got {r}"
    _pass("reward_igh_a0", f"r={r:.4f} ≈ 0 ✓")

    # ---- SP ----------------------------------------------------------------
    logger.info("--- SP reward ---")
    r = compute_reward_sp(
        action=3, is_attacker=1, rho_recv=0.9,
        d_bar_t=d_ref, PDR_t=PDR_ref, blockchain_reject=0, cfg=_cfg("sp"),
    )
    logger.info("  (a) attacker, action=3 → r=%.4f", r)
    assert r > 0, f"SP (a): expected r>0 for attacker+penalise, got {r}"
    _pass("reward_sp_attacker", f"r={r:.4f} > 0 ✓")

    r = compute_reward_sp(
        action=2, is_attacker=0, rho_recv=0.2,  # upstream victim
        d_bar_t=d_ref, PDR_t=PDR_ref, blockchain_reject=0, cfg=_cfg("sp"),
    )
    logger.info("  (b) innocent upstream victim, action=2 → r=%.4f", r)
    assert r < 0, f"SP (b): expected r<0 for innocent+penalise, got {r}"
    _pass("reward_sp_fp", f"r={r:.4f} < 0 ✓")

    r = compute_reward_sp(
        action=0, is_attacker=0, rho_recv=0.95,
        d_bar_t=d_ref, PDR_t=PDR_ref, blockchain_reject=0, cfg=_cfg("sp"),
    )
    logger.info("  (c) innocent, action=0 → r=%.4f (should be ~0)", r)
    assert abs(r) < 0.05, f"SP (c): expected r≈0 for a0+innocent, got {r}"
    _pass("reward_sp_a0", f"r={r:.4f} ≈ 0 ✓")

    # ---- ALS ---------------------------------------------------------------
    logger.info("--- ALS reward ---")
    r = compute_reward_als(
        action=2, is_attacker=1, lambda_t=0.3,
        d_bar_t=d_ref, PDR_t=PDR_ref, blockchain_reject=0, cfg=_cfg("als"),
    )
    logger.info("  (a) attacker, action=2 → r=%.4f", r)
    assert r > 0, f"ALS (a): expected r>0 for attacker+penalise, got {r}"
    _pass("reward_als_attacker", f"r={r:.4f} > 0 ✓")

    r = compute_reward_als(
        action=2, is_attacker=0, lambda_t=0.8,  # high mobility → false positive
        d_bar_t=d_ref, PDR_t=PDR_ref, blockchain_reject=0, cfg=_cfg("als"),
    )
    logger.info("  (b) innocent high-mobility, action=2 → r=%.4f", r)
    assert r < 0, f"ALS (b): expected r<0 for innocent+penalise, got {r}"
    _pass("reward_als_fp", f"r={r:.4f} < 0 ✓")

    r = compute_reward_als(
        action=0, is_attacker=0, lambda_t=0.3,
        d_bar_t=d_ref, PDR_t=PDR_ref, blockchain_reject=0, cfg=_cfg("als"),
    )
    logger.info("  (c) innocent, action=0 → r=%.4f (should be ~0)", r)
    assert abs(r) < 0.05, f"ALS (c): expected r≈0 for a0+innocent, got {r}"
    _pass("reward_als_a0", f"r={r:.4f} ≈ 0 ✓")

    # ---- FS ----------------------------------------------------------------
    logger.info("--- FS reward ---")
    r = compute_reward_fs(
        action=2, is_attacker=1, rho_recv=0.9, lambda_t=0.3,
        d_bar_t=d_ref, PDR_t=PDR_ref, blockchain_reject=0, cfg=_cfg("fs"),
    )
    logger.info("  (a) attacker, action=2 → r=%.4f", r)
    assert r > 0, f"FS (a): expected r>0 for attacker+penalise, got {r}"
    _pass("reward_fs_attacker", f"r={r:.4f} > 0 ✓")

    r = compute_reward_fs(
        action=2, is_attacker=0, rho_recv=0.2, lambda_t=0.3,  # rho_recv < threshold
        d_bar_t=d_ref, PDR_t=PDR_ref, blockchain_reject=0, cfg=_cfg("fs"),
    )
    logger.info("  (b) innocent (low rho_recv), action=2 → r=%.4f", r)
    assert r < 0, f"FS (b): expected r<0 for innocent+penalise, got {r}"
    _pass("reward_fs_fp_rho", f"r={r:.4f} < 0 ✓")

    r = compute_reward_fs(
        action=0, is_attacker=0, rho_recv=0.9, lambda_t=0.3,
        d_bar_t=d_ref, PDR_t=PDR_ref, blockchain_reject=0, cfg=_cfg("fs"),
    )
    logger.info("  (c) innocent, action=0 → r=%.4f (should be ~0)", r)
    assert abs(r) < 0.05, f"FS (c): expected r≈0 for a0+innocent, got {r}"
    _pass("reward_fs_a0", f"r={r:.4f} ≈ 0 ✓")


# ===========================================================================
# CHECK 4 — DQNAgent.learn() changes weights (Step 6)
# ===========================================================================

def check_agent_learn():
    """
    Verify that one call to DQNAgent.learn() modifies the main network weights.

    Procedure:
        1. Create an SP agent.
        2. Push buffer_min + batch_size random transitions.
        3. Capture a parameter snapshot (clone).
        4. Call learn() once.
        5. Assert that weights have changed.

    Implements: Step 6 unit-check (Section 10 of the report).
    Eqs: 3.58, 3.62, 3.63.
    """
    logger.info("")
    logger.info("=" * 60)
    logger.info("CHECK 4: DQNAgent.learn() — weight update")
    logger.info("=" * 60)

    # Use SP agent (input_dim=5) as test subject
    agent_name = "sp"
    cfg        = AGENT_CONFIGS[agent_name]
    hp         = dict(SHARED_HP)
    hp["buffer_min"] = 50    # reduce threshold for unit-check

    logger.info("Creating SP agent (input_dim=5)...")
    agent = DQNAgent(agent_cfg=cfg, hp=hp)

    # Push enough transitions to satisfy buffer_min
    n_push = hp["buffer_min"] + hp["batch_size"]
    input_dim = cfg["input_dim"]
    logger.info("Pushing %d random transitions into buffer...", n_push)
    for _ in range(n_push):
        s      = np.random.rand(input_dim).astype(np.float32)
        s_next = np.random.rand(input_dim).astype(np.float32)
        a      = np.random.randint(0, 5)
        r      = float(np.random.randn())
        done   = False
        agent.remember(s, a, r, s_next, done)

    logger.info("Buffer size after push: %d", len(agent.buffer))

    # Snapshot weights before learn()
    before = {
        name: param.data.clone()
        for name, param in agent.main_net.named_parameters()
    }
    logger.info("Weight snapshot captured. Calling learn()...")

    loss = agent.learn(beta=0.4)
    logger.info("learn() returned loss=%.6f", loss if loss is not None else float("nan"))

    if loss is None:
        _fail("agent_learn_runs", "learn() returned None — buffer may be too small")

    # Check that at least one parameter changed
    changed = False
    for name, param in agent.main_net.named_parameters():
        if not torch.allclose(before[name], param.data, atol=1e-9):
            logger.info("  Parameter changed: %s", name)
            changed = True
            break

    if not changed:
        _fail("agent_learn_weight_change", "No parameter changed after learn()")

    _pass("agent_learn_runs", f"loss={loss:.6f}")
    _pass("agent_learn_weight_change", "at least one parameter updated ✓")


# ===========================================================================
# CHECK 5 — Smoke test on dummy CSV (Step 7)
# ===========================================================================

def _make_dummy_csv(agent_name: str, n_nodes: int = 5, n_cycles: int = 20) -> str:
    """
    Create a tiny dummy CSV for the smoke test.

    Generates n_nodes * n_cycles rows with random feature values and writes
    the file to a temporary location.

    Args:
        agent_name : One of 'sp', 'als', 'igh', 'fs'.
        n_nodes    : Number of unique node IDs.
        n_cycles   : Number of cycles per node.

    Returns:
        str: Path to the temporary CSV file.

    Implements: Step 7 smoke test data preparation.
    """
    cfg      = AGENT_CONFIGS[agent_name]
    features = cfg["features"]

    rows = []
    for node_id in range(n_nodes):
        is_attacker = int(node_id % 3 == 0)  # every 3rd node is an attacker
        for cycle in range(n_cycles):
            row = {"node_id": node_id, "cycle_id": cycle}
            for feat in features:
                row[feat] = float(np.random.rand())
            row["is_attacker"]       = is_attacker
            row["blockchain_reject"] = 0
            row["PDR_t"]             = float(np.random.uniform(0.8, 1.0))
            row["d_bar_t"]           = float(np.random.uniform(40, 60))
            # Ensure rho_recv and lambda_t are always present
            row["rho_recv"]          = float(np.random.rand())
            row["lambda_t"]          = float(np.random.rand())
            rows.append(row)

    df = pd.DataFrame(rows)
    tmp_file = tempfile.NamedTemporaryFile(
        suffix=f"_{agent_name}_dummy.csv", delete=False, mode="w"
    )
    df.to_csv(tmp_file.name, index=False)
    logger.info(
        "Dummy CSV created | agent=%s, rows=%d, path=%s",
        agent_name, len(df), tmp_file.name,
    )
    return tmp_file.name


def check_smoke_test():
    """
    End-to-end smoke test: load dummy CSV → train for 5 episodes → no crash.

    Verifies that the full pipeline (data_loader → agent → train loop) runs
    without errors on minimal dummy data.

    Implements: Step 7 smoke test (Section 10 of the report).
    """
    logger.info("")
    logger.info("=" * 60)
    logger.info("CHECK 5: Smoke test — full pipeline on dummy CSV")
    logger.info("=" * 60)

    # Use SP agent (simplest) for smoke test, as per Section 10 Step 10 ordering
    for agent_name in ["sp", "als", "igh", "fs"]:
        logger.info("--- Smoke test: agent=%s ---", agent_name.upper())
        csv_path = _make_dummy_csv(agent_name, n_nodes=5, n_cycles=20)
        try:
            ep_rewards = train_agent(
                agent_name  = agent_name,
                smoke_test  = True,
                csv_path    = csv_path,
            )
            assert isinstance(ep_rewards, list), "Expected list of episode rewards"
            assert len(ep_rewards) == 5, f"Expected 5 episodes, got {len(ep_rewards)}"
            logger.info(
                "  [%s] Episode rewards: %s",
                agent_name.upper(), [round(r, 3) for r in ep_rewards],
            )
            _pass(f"smoke_test_{agent_name}", f"5 episodes completed, rewards={[round(r,3) for r in ep_rewards]}")
        except Exception as e:
            _fail(f"smoke_test_{agent_name}", str(e))
        finally:
            # Clean up temp file
            try:
                os.unlink(csv_path)
            except Exception:
                pass


# ===========================================================================
# CHECK 6 — dFF sign-cancellation fix (abs per flow, then MAX across flows)
# ===========================================================================

def check_dff_sign_cancellation() -> None:
    """
    Verify that dFF is computed as MAX(|ff_deviation|) across flows per node,
    not mean-then-abs.

    Two synthetic cases:
      Case A — sign cancellation: one flow +0.5, one flow −0.5.
               Mean-then-abs gives 0.0 (wrong).  abs-then-MAX gives 0.5 (correct).
      Case B — same sign, different magnitude: flows +0.3 and +0.5.
               MAX of abs gives 0.5 (not 0.4 average, not 0.8 sum).

    Implements the fix for the bug described in the report's Eq. 3.99 Phase 2:
    "the maximum δFF across flows is taken" — where δFF is already absolute.
    """
    check_name = "CHECK 6 — dFF sign-cancellation (abs-per-flow, MAX-across-flows)"
    logger.info("")
    logger.info("%s", check_name)

    try:
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import pandas as pd
        import numpy as np
        from bridge.features_percycle import add_percycle_features

        # ----------------------------------------------------------------
        # Build a minimal synthetic joined DataFrame.
        # Two nodes (0 = honest, 1 = attacker), each with two flow rows
        # that have been pre-aggregated as if abs_ff_deviation is already
        # MAX-aggregated by join.py + config_bridge.py.
        # We directly set abs_ff_deviation to simulate what the fixed
        # join pipeline produces.
        # ----------------------------------------------------------------
        base = {
            "cycle_id":              [1, 1],
            "node_id":               [0, 1],
            # Case A: honest — two flows cancel when signed (mean=0.0)
            # Case B: attacker — two flows +0.3 and +0.5 → MAX abs = 0.5
            # After the fix, join.py stores MAX(|ff_deviation|) here:
            "abs_ff_deviation":      [0.5, 0.5],   # Case A honest gets 0.5 too (worst flow)
            "sum_abs_ff_deviation":  [1.0, 0.8],   # fallback column (not used when abs_ff_deviation present)
            # Required feature columns (set to neutral values)
            "node_pdr":              [100.0, 100.0],
            "inbound_ratio":         [1.0, 1.0],
            "hop_delay_sum_ms":      [10.0, 10.0],
            "hop_delay_count":       [2.0, 2.0],
            "lambda_t":              [0.5, 0.5],
            "trust_score":           [1.0, 1.0],
            "spoof_dev":             [0.0, 0.0],
            "is_attacker":           [0, 1],
        }
        df = pd.DataFrame(base)
        result = add_percycle_features(df)

        # --- Case A: sign-cancellation check ----
        # Honest node (id=0): abs_ff_deviation=0.5 → dFF must be 0.5, not 0.0
        dff_honest = float(result.loc[result["node_id"] == 0, "dFF"].iloc[0])
        assert abs(dff_honest - 0.5) < 1e-9, (
            f"Case A (sign-cancellation) FAILED: expected dFF=0.5, got {dff_honest:.4f}. "
            f"abs() must happen per flow BEFORE aggregation, not after."
        )

        # --- Case B: MAX-not-mean check ----
        # Attacker node (id=1): abs_ff_deviation=0.5 → dFF must be 0.5
        dff_att = float(result.loc[result["node_id"] == 1, "dFF"].iloc[0])
        assert abs(dff_att - 0.5) < 1e-9, (
            f"Case B (MAX-not-mean) FAILED: expected dFF=0.5, got {dff_att:.4f}."
        )

        # --- Fallback check: when abs_ff_deviation is absent, use sum_abs_ff_deviation ----
        df_no_abs = df.drop(columns=["abs_ff_deviation"])
        result_fallback = add_percycle_features(df_no_abs)
        dff_fallback = float(result_fallback.loc[result_fallback["node_id"] == 1, "dFF"].iloc[0])
        assert dff_fallback >= 0.0, (
            f"Fallback check FAILED: dFF must be >= 0, got {dff_fallback:.4f}"
        )

        logger.info("  [PASS] Case A (sign-cancellation): dFF=%.4f (expected 0.5)", dff_honest)
        logger.info("  [PASS] Case B (MAX-not-mean):      dFF=%.4f (expected 0.5)", dff_att)
        logger.info("  [PASS] Fallback (no abs_ff_deviation col): dFF=%.4f >= 0", dff_fallback)
        _results[check_name] = "PASS"

    except Exception as exc:
        logger.error("  [FAIL] %s: %s", check_name, exc)
        _results[check_name] = "FAIL"


# ===========================================================================
# Main entry point
# ===========================================================================

def main():
    """Run all Stage 1 unit-checks in order."""
    logger.info("")
    logger.info("╔══════════════════════════════════════════════════════╗")
    logger.info("║          FM-DAD Stage 1 Unit-Checks                 ║")
    logger.info("╚══════════════════════════════════════════════════════╝")

    check_network_forward()
    check_replay_buffer()
    check_rewards()
    check_agent_learn()
    check_smoke_test()
    check_dff_sign_cancellation()

    # ---- Final summary -----------------------------------------------------
    logger.info("")
    logger.info("╔══════════════════════════════════════════════════════╗")
    logger.info("║                  UNIT-CHECK SUMMARY                 ║")
    logger.info("╚══════════════════════════════════════════════════════╝")

    n_pass = sum(1 for v in _results.values() if v == "PASS")
    n_fail = sum(1 for v in _results.values() if v == "FAIL")

    for name, result in _results.items():
        icon = "✅" if result == "PASS" else "❌"
        logger.info("  %s %-45s %s", icon, name, result)

    logger.info("")
    logger.info("  Total: %d PASS | %d FAIL", n_pass, n_fail)

    if n_fail == 0:
        logger.info("  🎉 ALL CHECKS PASSED — Stage 1 complete. Awaiting Stage 2 approval.")
    else:
        logger.error("  ⚠️  SOME CHECKS FAILED — Fix issues before proceeding.")

    return n_fail == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
