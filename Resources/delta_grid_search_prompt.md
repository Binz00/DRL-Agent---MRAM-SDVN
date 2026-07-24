# Implementation Task: Per-Agent Delta Grid Search (Issue 2 Fix)

## Context

The current system uses one shared set of trust penalty magnitudes for all four agents:
```python
"deltas": [0.0, 0.05, 0.15, 0.30, 0.50]   # same for SP, ALS, IGH, FS
```
The report requires four independently calibrated per-attack delta sets, selected by
separate grid searches per agent. The constraint is: 0 < δ1^X < δ2^X < δ3^X < δ4^X
for each X ∈ {SP, ALS, IGH, FS}. δ0 = 0.0 always (no-penalty action, fixed).

## What this task builds

One new script: `grid_search_deltas.py`

No other files are modified in this task. config.py is updated only to record
the grid-searched results after the script runs — it is NOT modified by the script
itself (the script prints config.py-ready output for manual paste, same pattern
as grid_search_gate.py).

## Evaluation oracle

`episode_eval.py` is already built and verified. Use it as the evaluation function —
specifically `evaluate_policy_epoch()` with the current fine-tuned checkpoints.
Do NOT re-implement trust accumulation or blacklisting logic. Import and call
`evaluate_policy_epoch` directly.

This is important: delta values affect the trust trajectory (how fast trust drops),
so evaluation MUST simulate the full 28-cycle trajectory, not just one cycle.
`evaluate_policy_epoch` already does this correctly.

## Key facts about the existing code (read before writing anything)

**agent.py — how deltas are used:**
`self.deltas = agent_cfg["deltas"]` in `__init__`.
`delta = agent.deltas[action]` in `trigger.py process_node()`.
So swapping deltas during grid search = temporarily replace `agent.deltas` list,
evaluate, restore. No rebuild of the agent needed.

**episode_eval.py — public API:**
```python
evaluate_policy_epoch(
    agent_name,    # str: "sp"|"als"|"igh"|"fs"
    live_agent,    # DQNAgent (greedy, eps=0)
    frozen_agents, # dict of other 3 DQNAgents
    tables,        # dict from load_tables()
    ground_truth,  # DataFrame from load_gt()
    tau_min,       # float
) -> EpochOutcome   # .mcc, .counts (tp,fp,fn,tn)
```
Also use: `load_tables()`, `load_gt()`, `load_frozen_agents(exclude=agent_name)`

**config.py — current delta structure:**
```python
AGENT_CONFIGS["sp"]["deltas"]  = [0.0, 0.05, 0.15, 0.30, 0.50]
AGENT_CONFIGS["als"]["deltas"] = [0.0, 0.05, 0.15, 0.30, 0.50]
AGENT_CONFIGS["igh"]["deltas"] = [0.0, 0.05, 0.15, 0.30, 0.50]
AGENT_CONFIGS["fs"]["deltas"]  = [0.0, 0.05, 0.15, 0.30, 0.50]
```
After the grid search, each agent gets its own independently selected set.

**fs_state.csv column layout (confirmed):**
`node_id, cycle_id, FFc, dFF, DelayInfl, tau, lambda_t_norm, is_attacker,
rho_recv, node_pdr, sum_abs_ff_deviation_normalized`
State features (5): FFc, dFF, DelayInfl, tau, lambda_t_norm — in that order.
Extra cols: rho_recv, node_pdr, sum_abs_ff_deviation_normalized (gate-only, not state).

**tau_min for grid search:** use 0.4 (the currently grid-searched optimal value).

## Search space design

Each delta set is [0.0, δ1, δ2, δ3, δ4] with the constraint δ1 < δ2 < δ3 < δ4.
Search over these candidate sets per agent — they differ because attack severity
profiles differ:

```python
DELTA_CANDIDATES = {
    "sp": [
        [0.0, 0.05, 0.10, 0.20, 0.40],
        [0.0, 0.05, 0.15, 0.30, 0.50],   # current (baseline)
        [0.0, 0.10, 0.20, 0.35, 0.50],
        [0.0, 0.10, 0.25, 0.40, 0.50],
    ],
    "als": [
        [0.0, 0.05, 0.10, 0.20, 0.40],
        [0.0, 0.05, 0.15, 0.30, 0.50],   # current (baseline)
        [0.0, 0.10, 0.20, 0.35, 0.50],
        [0.0, 0.15, 0.30, 0.45, 0.50],
    ],
    "igh": [
        [0.0, 0.05, 0.10, 0.20, 0.40],
        [0.0, 0.05, 0.15, 0.30, 0.50],   # current (baseline)
        [0.0, 0.10, 0.20, 0.35, 0.50],
        [0.0, 0.10, 0.25, 0.40, 0.50],
    ],
    "fs": [
        [0.0, 0.03, 0.08, 0.15, 0.30],   # smaller — FS has noisier signal
        [0.0, 0.05, 0.10, 0.20, 0.35],
        [0.0, 0.05, 0.15, 0.30, 0.50],   # current (baseline)
        [0.0, 0.08, 0.15, 0.25, 0.40],
    ],
}
```

The current shared set [0.0, 0.05, 0.15, 0.30, 0.50] must appear in every agent's
candidate list as the baseline — the grid search must be able to select it,
confirming or replacing it with evidence.

## Script structure: `grid_search_deltas.py`

```python
"""
grid_search_deltas.py — Per-agent delta (trust penalty magnitude) grid search.

For each agent X ∈ {SP, ALS, IGH, FS}:
  1. Load the fine-tuned checkpoint for X as the "live" agent.
  2. Load the other three fine-tuned checkpoints as frozen peers.
  3. For each candidate delta set in DELTA_CANDIDATES[X]:
       a. Temporarily replace agent.deltas with the candidate set.
       b. Call evaluate_policy_epoch() — full 28-cycle trust trajectory.
       c. Record MCC^X and confusion matrix counts.
       d. Restore original agent.deltas.
  4. Select the candidate set with highest MCC^X.
  5. Log all candidates and their scores.

Output:
  - Console: config.py-ready paste block with best deltas per agent.
  - CSV:     data/grid_search_delta_results.csv (all candidates, for the report).

Usage:
    python3 grid_search_deltas.py
    python3 grid_search_deltas.py --tau 0.4
"""
```

### Core loop (write it this way exactly):

```python
for agent_name, candidates in DELTA_CANDIDATES.items():
    # Load live agent from fine-tuned checkpoint
    live_agent = load_one_agent(agent_name)

    # Load other three frozen
    frozen_agents = load_frozen_agents(exclude=agent_name)   # from episode_eval

    best_mcc      = -float("inf")
    best_deltas   = None
    results_rows  = []

    for delta_set in candidates:
        # Swap deltas — no agent rebuild needed
        original_deltas    = live_agent.deltas
        live_agent.deltas  = delta_set

        outcome = evaluate_policy_epoch(
            agent_name    = agent_name,
            live_agent    = live_agent,
            frozen_agents = frozen_agents,
            tables        = tables,        # loaded once before the loop
            ground_truth  = ground_truth,  # loaded once before the loop
            tau_min       = tau_min,
        )

        # Restore immediately — never leave agent in patched state
        live_agent.deltas = original_deltas

        tp, fp, fn, tn = outcome.counts
        results_rows.append({
            "agent":   agent_name,
            "deltas":  str(delta_set),
            "mcc":     outcome.mcc,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        })
        logger.info("[%s] deltas=%s → MCC=%.4f (TP=%d FP=%d FN=%d TN=%d)",
                    agent_name.upper(), delta_set, outcome.mcc, tp, fp, fn, tn)

        if outcome.mcc > best_mcc:
            best_mcc    = outcome.mcc
            best_deltas = delta_set

    best_per_agent[agent_name] = {"deltas": best_deltas, "mcc": best_mcc}
    logger.info("[%s] BEST: %s → MCC^%s = %.4f",
                agent_name.upper(), best_deltas, agent_name.upper(), best_mcc)
```

### Output format (console, config.py-ready):

```
============================================================
DELTA GRID SEARCH RESULTS — paste into config.py AGENT_CONFIGS
============================================================

  # SP best deltas (MCC^SP = +0.XXXX)
  "deltas": [0.0, 0.10, 0.20, 0.35, 0.50],

  # ALS best deltas (MCC^ALS = +0.XXXX)
  "deltas": [0.0, 0.05, 0.15, 0.30, 0.50],

  # IGH best deltas (MCC^IGH = +0.XXXX)
  "deltas": [0.0, 0.05, 0.15, 0.30, 0.50],

  # FS best deltas (MCC^FS = +0.XXXX)
  "deltas": [0.0, 0.03, 0.08, 0.15, 0.30],

Macro MCC with best per-agent deltas: +0.XXXX
(vs baseline with shared deltas:      +0.8869)
============================================================
```

The script must also print the macro MCC using the best-selected deltas for
ALL four agents simultaneously — load all four with their best deltas, run
`evaluate_policy_epoch` for each variant, average the four MCCs. This is the
true end-to-end check of whether the per-agent calibration helps macro MCC.

## Constraints (do not violate)

1. **Never modify agent weights** — only `agent.deltas` (a plain Python list) is
   swapped. The DQN networks themselves are untouched.
2. **Always restore `agent.deltas` immediately after each evaluation**, even if
   `evaluate_policy_epoch` raises an exception — use try/finally.
3. **The current shared set must be in every candidate list** — regression guard.
   If the grid search selects it for every agent, macro MCC should equal +0.8869.
4. **Search each agent independently** — while evaluating SP's delta candidates,
   ALS/IGH/FS use their current (pre-grid-search) deltas as frozen peers.
   Do not co-optimize across agents simultaneously.
5. **Import, do not reimplement** — `evaluate_policy_epoch`, `load_tables`,
   `load_gt`, `load_frozen_agents` all imported from `episode_eval.py`.
   `mcc_from_counts` imported from `episode_eval.py` too.
6. **tau_min = 0.4** (grid-search selected optimal) as default, overridable via
   `--tau` CLI argument.
7. **Log every candidate** — the CSV output must contain all candidates,
   not just the best, so the supervisor can see the full search space.

## Files to deliver

1. `fm_dad/grid_search_deltas.py` — the new script
2. Console output pasted into RESULTS.md showing:
   - All candidates per agent with their MCC scores
   - Best selected deltas per agent
   - Macro MCC with best per-agent deltas vs baseline +0.8869
3. Updated `config.py` — paste the best deltas into each agent's entry,
   add a comment: `# grid-searched by grid_search_deltas.py (Issue 2 fix)`

## Files that must NOT be modified

- `rewards.py`
- `episode_eval.py`
- `bridge/trigger.py`
- `bridge/assemble.py` AGENT_STATE_FEATURES
- `agent.py` (the DQNAgent class itself)
- Any model checkpoint (.pt files)
