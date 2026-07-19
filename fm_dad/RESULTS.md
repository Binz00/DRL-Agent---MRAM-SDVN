# RESULTS.md — MCC Reward Term Implementation (Steps 1–5)

## Protected Files (Unchanged)
- `rewards.py` — compute_reward_igh/sp/als/fs and all helpers byte-identical. Not touched.
- `bridge/trigger.py` — GATE_CONDITIONS and _check_gate logic unchanged.
- `bridge/assemble.py` — AGENT_STATE_FEATURES unchanged.

---

## Step 1 — data_loader.py: cycle_id + node_id in transitions

**Change:** Added `cycle_id: int` and `node_id: int` to every transition dict. Both are lookup keys for the D_i reward term — never enter state vectors `s` or `s_next`.

**Verification:**
```
Keys: ['s', 's_next', 'done', 'cycle_id', 'node_id', 'is_attacker', ...]
cycle_id: 1  <class 'int'>
node_id:  1  <class 'int'>
len(s):   5  (must be 5 for fs)
Step 1 PASS
```

---

## Step 2 — episode_eval.py: --self-test oracle

**Hard gate result:**
```
================================================================
tau_min = 0.3
================================================================
  [PASS] ALS   MCC=+0.8991  expected=+0.8991  diff=0.0000  (TP=30 FP=6 FN=0  TN=195)
  [PASS] FS    MCC=+0.5498  expected=+0.5498  diff=0.0000  (TP=15 FP=6 FN=15 TN=195)
  [PASS] IGH   MCC=+0.8783  expected=+0.8783  diff=0.0000  (TP=29 FP=6 FN=1  TN=195)
  [PASS] SP    MCC=+0.5766  expected=+0.5766  diff=0.0000  (TP=16 FP=6 FN=14 TN=195)
✅  Self-test PASSED — episode_eval matches validate_pipeline.py.
================================================================
```

All four variants match to 4 decimal places (diff=0.0000). Hard gate CLEARED. Proceeded to Step 3.

---

## Step 3 — MCC-based Checkpoint Selection (FS fine-tune)

**Configuration:**
- `mcc_eval_every = 5` (added to FINETUNE_HP)
- `w5 = 0.2` (active in cfg at run time; w1..w4 rescaled to 0.32/0.24/0.16/0.08)

**MCC^FS evaluation log (selected episodes):**
```
ep=1/100   MCC^FS=0.5167  TP=15 FP=8  FN=15 TN=193  (BEST, saved)
ep=6/100   MCC^FS=0.6309  TP=15 FP=2  FN=15 TN=199  (NEW BEST, saved)
ep=11/100  MCC^FS=0.5874  (no improvement)
ep=21/100  MCC^FS=0.6083  (no improvement)
...
ep=100/100 MCC^FS=0.5327  (no improvement)
```

**Result:**
```
[FINETUNE DONE] Best checkpoint: ep=6, MCC^FS=0.6309 — saved to models/fs_finetuned.pt
```

**Verification:** MCC^FS best = **+0.6309** ≥ pre-change baseline **+0.5498** ✅

---

## Step 4 — Difference Reward Term (w5=0.2 vs w5=0.0 comparison)

**w5=0.2 run** (episode_eval, not validate_pipeline — see Note below):
```
Best checkpoint: ep=6,  MCC^FS=+0.6309  (TP=15, FP=2, FN=15, TN=199)
```

**w5=0.0 run** (regression guard — CLI: `--w5 0.0 --model_out models/fs_finetuned_w5_0.pt`):
```
Best checkpoint: ep=11, MCC^FS=+0.6309  (TP=15, FP=2, FN=15, TN=199)
```

| w5  | Best ep | MCC^FS  | FP | Comment                            |
|-----|---------|---------|----|------------------------------------|
| 0.2 | 6       | +0.6309 | 2  | D_i term speeds convergence        |
| 0.0 | 11      | +0.6309 | 2  | Same peak; converges 5 eps later   |

**Regression guard:** w5=0.0 best MCC = +0.6309 ≥ +0.5498 baseline ✅

**Interpretation:** The D_i reward signal reduces false positives faster (ep=6 vs ep=11) but the MCC ceiling is the same. This is consistent with D_i reducing reward variance (FP transitions penalised earlier) rather than unlocking new discriminative capacity the gate cannot provide.

> **Note on validate_pipeline.py MCC values:** `validate_pipeline.py` reads from
> `data/pipeline_penalties.csv` — a static file from the last bridge pipeline run.
> It does NOT reflect the updated fine-tuned checkpoints until the bridge is
> re-run with the new agents. The MCC values reported above come from
> `episode_eval.py` (live simulation), which is the correct measurement for
> training evaluation per the spec. The pipeline_penalties.csv-based validate
> still shows +0.7260 (the pre-fine-tune state).

---

## Step 5 — finetune_all.py (Round-Robin Driver)

File created: `fm_dad/finetune_all.py`.  
Order: FS → SP → ALS → IGH (FS first per spec: most headroom).  
Auto-triggers second round only if Δmacro MCC > 0.01.

> **Status:** Ready to run with `python3 finetune_all.py`. Not yet executed — the
> round-robin produces meaningful results only after re-running the bridge pipeline
> to regenerate `pipeline_penalties.csv` from the updated checkpoints, making
> validate_pipeline.py's output reflect the fine-tuned agents.

---

## Hyperparameter Table

| Parameter       | Value   | Status                                    |
|-----------------|---------|-------------------------------------------|
| `w5`            | 0.2     | Placeholder; grid {0.0, 0.1, 0.2, 0.3}  |
| `mcc_eval_every`| 5       | Default per spec; double to 10 if reward variance oscillates |
| `tau_min`       | 0.3     | Grid-searched (Table 4.1)                 |
| `FINETUNE n_episodes` | 100 | Supervisor instruction (do not overtrain) |
| `lr`            | 0.0001  | Supervisor instruction                    |
| `eps0`          | 0.10    | Supervisor instruction                    |

---

## Files Delivered

| File | Action | Notes |
|------|--------|-------|
| `data_loader.py` | MODIFIED | Added cycle_id, node_id to transitions |
| `train.py` | MODIFIED | Step 3 (MCC checkpoint), Step 4 (D_i term), --w5/--model_out CLI |
| `config.py` | MODIFIED | w5=0.2 in all AGENT_CONFIGS; mcc_eval_every=5 in FINETUNE_HP |
| `episode_eval.py` | NEW | Trust-trajectory evaluator + --self-test oracle |
| `finetune_all.py` | NEW | Round-robin driver |
| `rewards.py` | **UNCHANGED** | Byte-identical |
| `bridge/trigger.py` | **UNCHANGED** | Gate conditions and _check_gate unchanged |
| `bridge/assemble.py` | **UNCHANGED** | AGENT_STATE_FEATURES unchanged |
