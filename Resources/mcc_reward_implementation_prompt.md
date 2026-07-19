# Implementation Task: Per-Variant MCC Reward Term (r_mcc^X) via Difference Rewards

## Project Context (read first)

This is the FM-DAD project: a DRL-based defense for Software-Defined Vehicular Networks (NS-3, 321 nodes = 200 vehicles + 121 RSUs) against 4 multipath routing attacks: SP (Selective Packet Dropping), ALS (Asymmetric Link Spoofing), IGH (Interleaved Grey Hole), FS (Flow Stretching).

Architecture: deterministic rule-based detection gates decide IF an agent runs; 4 independent Dueling Double-DQN agents (PER + Huber loss) decide penalty magnitude (5 actions a0–a4 with trust deltas 0.0/0.05/0.15/0.30/0.50). Per-agent Δτ combined via MAX (Eq. 3.38), trust written per node, node blacklisted when trust < τ_min (currently 0.3, grid-search selected).

Current validation (28 cycles, macro MCC +0.7260): ALS +0.8991, IGH +0.8783, SP +0.5766, FS +0.5498. FP = 6/201 honest nodes.

**Goal of this task:** Add the supervisor's per-variant MCC reward term (r_mcc^X) to training — a population-level reward measuring network-wide detection quality (Matthews Correlation Coefficient) of each agent's decisions. The supervisor's LaTeX patch specifies r_mcc^X computed from TP/FP/FN/TN counts across all nodes after all trust writes and blacklisting decisions are finalized per epoch.

**The core problem this design solves:** the current training loop (train.py) is pure offline, single-agent, per-transition with immediate learning from globally shuffled data. r_mcc^X requires (a) cross-agent outcomes (MAX-combine over all 4 agents), (b) temporal trust accumulation across cycles, (c) population-level → per-transition credit assignment. The design below solves all three WITHOUT abandoning the offline single-agent training structure.

## Design Overview (do not deviate without flagging)

1. **`episode_eval.py` (NEW FILE)** — a trust-trajectory evaluator that simulates the full deployed pipeline (gate → argmax Q → MAX-combine → trust accumulation → blacklist) over all cycles, using the agent-under-training (live, greedy) plus the other 3 agents frozen at their latest checkpoints. Returns per-node outcomes (TP/FP/FN/TN classification under variant X) and confusion-matrix counts.
2. **Difference rewards** — per-node counterfactual credit: `D_i = MCC(actual) − MCC(node i's decision replaced by a0)`. O(1) per node since MCC is closed-form over 4 integers.
3. **Periodic re-evaluation** — recompute the evaluation every K episodes (config: `mcc_eval_every`, default 5) to track the improving policy without per-episode cost.
4. **Reward composition** — `reward = r_base + w5 * D_i` where r_base is the existing unchanged reward from rewards.py. rewards.py's four compute_reward_* functions must NOT be modified.
5. **MCC-based checkpoint selection** — save the checkpoint from the episode with best evaluated MCC, not the last episode.
6. **Frozen-policy round-robin** — train one agent at a time against frozen checkpoints of the others; rotate FS → SP → ALS → IGH.

## Critical Constraints (violating any of these is a failure)

- **R2 rule:** No `attack_type` feature may be used anywhere in state vectors or rewards.
- **State vectors are FIXED by the report** (Eq. 3.44/3.46): FS state = [FFc, dFF, DelayInfl, tau, lambda_t_norm]. Do NOT add any new feature to any AGENT_STATE_FEATURES entry. D_i is a REWARD-time quantity only, never a state feature.
- **rewards.py functions unchanged:** compute_reward_igh/sp/als/fs and their helpers stay byte-identical. The new term composes in train.py.
- **Threshold honesty principle:** all new hyperparameters (w5, mcc_eval_every) get grid-searched or explicitly documented as placeholders. Never hand-tuned to hit a target TP count. w5 grid MUST include 0.0 as a regression guard.
- **Weight budget:** the supervisor patch requires w1+w2+w3+w4+w5 = 1. Implementation approach: keep the RATIOS of the current w1..w4 fixed, rescale them by (1−w5), and grid-search only w5 ∈ {0.0, 0.1, 0.2, 0.3}. Document this in config.py comments.
- **Gate logic in trigger.py must be reused, not reimplemented.** episode_eval.py imports and calls trigger.py's `_check_gate` / GATE_CONDITIONS and the agents' `act()` — a second implementation of gate logic that could drift from the deployed one is forbidden (single source of truth).
- **Supervisor instruction:** do not overtrain fine-tuning. FINETUNE_HP is 100 episodes, lr=0.0001, eps0=0.10 — keep those unless a change is explicitly justified and logged.

## Existing File Map (what you're working with)

```
fm_dad/
  config.py            — SHARED_HP, FINETUNE_HP, AGENT_CONFIGS (per-agent features,
                          deltas, eta_* thresholds, e1/e2/e3, w1..w4), DATA_FILES,
                          MODEL_FILES, FINETUNE_MODEL_FILES, FINETUNE_DATA_FILES
  train.py             — Algorithm 2 offline loop; --finetune flag; per-transition:
                          act → compute_reward_for_agent → remember → learn;
                          episodes = full shuffled passes over all transitions
  rewards.py           — graded r_sec + r_fp + r_qos + r_end per agent; REWARD_FN_MAP
  data_loader.py       — load_transitions(csv_path, agent_name) → list of dicts with
                          keys: s, s_next, done, is_attacker, blockchain_reject,
                          PDR_t, d_bar_t, rho_recv, lambda_t
                          (compat renames: lambda_t_norm→lambda_t, PDR_t→FFc etc.)
  agent.py             — DQNAgent (Dueling DDQN, PER, act(), learn(), save(), load())
  bridge/trigger.py    — GATE_CONDITIONS (now OR-of-AND-groups structure:
                          Dict[str, List[List[Condition]]]), _check_gate(),
                          load_agents(), process_node(), process_cycle()
  bridge/assemble.py   — AGENT_STATE_FEATURES, EXTRA_COLS, assemble_agent_tables()
                          → data/agent_inputs/{sp,als,igh,fs}_state.csv
  validate_pipeline.py — MCC^X per Eq. 4.1; per-attack FP counts ONLY honest nodes;
                          ground truth = UNION across ALL node_attack_ground_truth_*.csv;
                          τ_min grid search {0.3,0.4,0.5} by macro MCC
```

Current FS gate (after recent fix): `"fs": [[("sum_abs_ff_deviation_normalized", ">", AGENT_CONFIGS["fs"]["eta_dff_norm"])]]` with eta_dff_norm = 0.1 (grid-searched). Do not change gates in this task.

## Implementation Steps (in this exact order — each step has a verification oracle)

### Step 1 — data_loader.py: carry cycle_id and node_id through transitions

Each transition dict must additionally contain `cycle_id` (int) and `node_id` (int) taken from the source CSV row. These are lookup keys only — they must NOT enter the state vector `s` or `s_next`.

**Verify:** print one transition; confirm both keys present and state vector length unchanged.

### Step 2 — episode_eval.py (NEW): the trust-trajectory evaluator

```python
def evaluate_policy_epoch(
    agent_name: str,              # which agent is live/under training
    live_agent: DQNAgent,         # current in-training network, used greedily (eps=0)
    frozen_agents: dict,          # other 3 agents loaded from latest checkpoints
    tables: dict,                 # agent_name -> per-cycle state DataFrame
                                  #   (same tables validate_pipeline consumes)
    ground_truth: pd.DataFrame,   # UNION ground truth across all cycles
                                  #   (reuse validate_pipeline's loader — do not re-derive)
    tau_min: float,
) -> EpochOutcome
```

Behavior (must mirror the deployed pipeline exactly):
- Process cycles IN ORDER (1..N). Maintain per-node trust, initialized as the deployed pipeline initializes it (check validate_pipeline.py for the initial trust value and reuse it).
- Per cycle, per node: build feat_dict per agent (state features + EXTRA_COLS — reuse the same construction as trigger.process_cycle), run `_check_gate`, if open run that agent's `act(state, epsilon=0.0)`, map to delta, MAX-combine across the four agents, subtract from trust.
- A node is blacklisted when trust < tau_min. Once blacklisted, follow whatever the validator does (check whether blacklisted nodes keep being processed — mirror it exactly).
- Classification under variant X (mirror validate_pipeline.py's fixed semantics):
  - TP^X: node is an X-attacker (per union GT) and ends blacklisted
  - FN^X: node is an X-attacker and does not end blacklisted
  - FP^X: node is HONEST (not an attacker of ANY type) and ends blacklisted
  - TN^X: node is honest and not blacklisted
  - Other-type attackers are EXCLUDED from X's FP/TN counts (this matches the validator fix).
- Return: confusion counts, the resulting MCC^X, AND a per-(cycle_id, node_id) record of the live agent's contribution: whether the live agent's gate fired, its chosen action, and whether the node's final outcome was TP/FP/FN/TN.

`EpochOutcome` should be a small dataclass: `counts: (tp, fp, fn, tn)`, `mcc: float`, `node_outcomes: dict[(cycle_id, node_id)] -> {"gate_fired": bool, "action": int, "outcome": "TP"|"FP"|"FN"|"TN"|"EXCLUDED"}`.

**Verify (correctness oracle — mandatory before proceeding):** load ALL FOUR fine-tuned checkpoints as "frozen", evaluate each variant, and confirm the four MCC values match the current validation output to 4 decimals: ALS +0.8991, FS +0.5498, IGH +0.8783, SP +0.5766 at tau_min=0.3. If they don't match, the evaluator has diverged from the deployed pipeline — STOP and fix before any training changes. Write this as a runnable check: `python episode_eval.py --self-test`.

### Step 3 — train.py: MCC-based checkpoint selection (independent win, do this before the reward term)

In `train_agent()` when `--finetune`:
- Every `mcc_eval_every` episodes (add to FINETUNE_HP, default 5) AND on the final episode, call `evaluate_policy_epoch` with the live agent + frozen others.
- Track best MCC^X seen; when a new best occurs, save the model to FINETUNE_MODEL_FILES[agent] (overwriting). Log: episode, MCC, counts.
- Final log line must state which episode's checkpoint was kept and its MCC.

**Verify:** re-run FS fine-tune. The selected checkpoint's MCC^FS (via episode_eval self-consistency) must be ≥ the pre-change FS MCC (+0.5498). Then run the real validate_pipeline.py to confirm end-to-end.

### Step 4 — difference-reward term in train.py

Add to train.py (NOT rewards.py):

```python
import math

def mcc_from_counts(tp, fp, fn, tn):
    denom = math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    return ((tp*tn) - (fp*fn)) / denom if denom > 0 else 0.0

def build_difference_rewards(outcome: EpochOutcome) -> dict:
    """(cycle_id, node_id) -> D_i.
    D_i = MCC(actual) - MCC(counterfactual: this node's live-agent decision -> a0).
    Only nodes where the live agent's gate fired AND action > 0 can have nonzero D_i:
      outcome TP  -> counterfactual tp-1, fn+1   (node would not have been penalized by X;
                     NOTE: only flip if X's delta was the MAX/deciding contribution —
                     if another frozen agent's delta alone would still blacklist the node,
                     the counterfactual outcome is unchanged and D_i = 0)
      outcome FP  -> counterfactual fp-1, tn+1   (same MAX-contribution caveat)
      outcome FN/TN or gate closed or action==0 -> D_i = 0
    """
```

The MAX-contribution caveat is important: because trust deltas MAX-combine, removing agent X's penalty only changes the outcome if X's delta was strictly greater than every other agent's delta for that node in the cycles that mattered. Simplification allowed: treat "X's delta was the max at least once on this node across the trajectory" as the condition, and document the approximation in a comment. Log how many nodes get nonzero D_i per evaluation.

In the transition loop:
```python
r_base = compute_reward_for_agent(agent_name, action, transition, cfg)   # UNCHANGED call
d_i    = d_reward_lookup.get((transition["cycle_id"], transition["node_id"]), 0.0)
reward = r_base + cfg["w5"] * d_i
```

d_reward_lookup is rebuilt at every evaluation (Step 3's cadence). Between evaluations it is held fixed.

config.py changes:
- Add `"w5": 0.2` to each AGENT_CONFIGS entry with a comment: `# MCC difference-reward weight (supervisor r_mcc patch); grid {0.0,0.1,0.2,0.3}; w1..w4 rescaled by (1-w5) at load time to keep sum=1`.
- Implement the (1−w5) rescaling of w1..w4 where cfg is consumed in train.py — do NOT mutate the config values on disk; rescale at use.
- Add `"mcc_eval_every": 5` to FINETUNE_HP.

**Verify:** fine-tune FS with w5=0.2 and with w5=0.0; run validate_pipeline.py on both. Report the two macro MCC values side by side. w5=0.0 must reproduce Step 3's result (regression guard).

### Step 5 — round-robin driver (small script, new file finetune_all.py)

```
order = ["fs", "sp", "als", "igh"]   # FS first (most headroom)
for name in order:
    run train_agent(name, finetune=True)   # frozen_agents always reload latest checkpoints
run validate_pipeline
```
One round only. Log macro MCC before/after. A second round is optional and only if round 1 improved macro MCC by > 0.01.

## Known Risks — handle these explicitly in code/logging

1. **Non-stationary reward vs PER:** D_i shifts every K episodes; PER may over-sample transitions whose stored priority reflects a stale reward. Mitigation: log per-episode reward variance; if training reward oscillates without trend after the first 30 episodes, double mcc_eval_every (10) and retry once before escalating.
2. **Degenerate MCC denominator:** with 0 predicted positives early in training, denom=0 → mcc=0. mcc_from_counts already guards this; make sure no division elsewhere.
3. **Frozen checkpoints must be the FINETUNE_MODEL_FILES versions** (the +0.7260 set), not the synthetic MODEL_FILES ones. Assert file mtimes/paths at load and log which files were loaded.
4. **episode_eval must use the SAME tables the validator uses.** Do not regenerate features differently; load the same data/agent_inputs/*_state.csv (or call the same assemble path the validator calls).
5. **Do not let evaluation leak into gradient steps:** evaluate_policy_epoch runs the live agent in eval mode with epsilon=0 and torch.no_grad(); restore train mode after.

## Deliverables

1. Modified: data_loader.py, train.py, config.py
2. New: episode_eval.py (with --self-test), finetune_all.py
3. Unchanged (assert in your summary): rewards.py, bridge/trigger.py gates, bridge/assemble.py AGENT_STATE_FEATURES
4. A short RESULTS.md: Step-2 oracle output, Step-3 best-checkpoint episodes + MCCs, Step-4 w5=0 vs w5=0.2 comparison, final validate_pipeline output after round-robin
5. All new hyperparameters listed in one table with their status (grid-searched value vs documented placeholder)

## Success Criteria

- episode_eval.py --self-test reproduces {ALS +0.8991, FS +0.5498, IGH +0.8783, SP +0.5766} exactly
- Macro MCC after the full sequence ≥ +0.7260 (never worse — w5=0.0 fallback guarantees this if honored)
- Zero changes to state vector definitions, gates, and rewards.py
- Every claim in RESULTS.md backed by a logged number, not an assertion
