# IMPLEMENTATION TASK: Four Independent DRL Agents (FM-DAD) for MRAM-SDVN

## 1. PROJECT CONTEXT

You are implementing the Full-Mode DRL-Based Adaptive Defense (FM-DAD) module of a Final Year Project: "Multipath Routing Attack Mitigation for Software-Defined Vehicular Networks (MRAM-SDVN)".

The system defends against 4 multipath routing attacks:
- SP  = Selective Packet dropping
- ALS = Asymmetric Link Spoofing
- IGH = Interleaved Grey Hole
- FS  = Flow Stretching

The design uses FOUR completely separate and independent DQN agents, one per attack type. Each agent decides HOW MUCH to reduce a suspicious node's trust score (graded response, 5 levels). Detection itself is done by separate deterministic rule-based algorithms (not part of the DRL agents) — the agent is only invoked after detection fires.

This implementation follows a formal academic report. All equations referenced below (Eq. 3.xx) come from that report. DO NOT deviate from the specifications. Do not "improve" or "optimize" the design beyond what is specified. Faithfulness to the specification is the top priority.

## 2. ABSOLUTE RULES (NEVER VIOLATE)

R1. The four agents NEVER share network weights, target weights, or replay buffers. Each agent has its own θ^X, θ̂^X, and buffer B^X. Code CLASSES may be reused (one class, four instances), but no parameter/gradient sharing between instances.

R2. There is NO attack_type input feature anywhere. No agent ever receives an attack-type label in its state vector. Training data files are attack-specific by construction (separate file per agent).

R3. Do not change the state vector definitions, action space, reward structure, or training algorithm. They are fixed by the report (specifications in Sections 4–8 below).

R4. Implementation order is fixed (Section 10): first implement ALL code WITHOUT running training, then implement the synthetic data generator, then run training. Do not skip ahead.

R5. Use PyTorch. Do not use TensorFlow, Keras, or JAX.

R6. Every class and function must have a docstring that references the report equation it implements (e.g., "Implements Eq. 3.57 (dueling decomposition)"). This is required for supervisor review.

R7. Keep the code simple and readable. Prefer clarity over cleverness. A university panel will read this code. No unnecessary abstractions, no premature optimization.

## 3. LIBRARIES AND ENVIRONMENT

- Python 3.10+
- torch (PyTorch, CPU is fine; use CUDA if available via `torch.device`)
- numpy
- pandas (for reading/writing CSV training data)
- matplotlib (for reward curves / verification plots only)

No other ML libraries. No stable-baselines3, no gym/gymnasium dependency for the core agents (training data comes from offline CSV files, not a live environment).

## 4. THE FOUR AGENTS — CONFIGURATION TABLE

All four agents are instances of the SAME agent class with different configuration:

| Agent | Input size | State features (in this exact order) | State Eq. |
|-------|-----------|--------------------------------------|-----------|
| IGH | 8 | FFc, dFF, rho_recv, d_bar, tau, PDRVar, CoordScore, lambda_t | Eq. 3.41 |
| SP  | 5 | FFc, dFF, rho_recv, tau, lambda_t | Eq. 3.42 |
| ALS | 4 | SpoofDev, dFF, tau, lambda_t | Eq. 3.43 |
| FS  | 5 | FFc, dFF, DelayInfl, tau, lambda_t | Eq. 3.44 |

Notation used in code (ASCII names):
- FFc        = forwarding fraction per cycle (Eq. 3.91)
- dFF        = delta_FF, deviation from committed plan (Eq. 3.93)
- rho_recv   = receipt ratio (Eq. 3.18)
- d_bar      = mean per-hop delay (Amendment A1: NS-3 mean_delay_ms is the training-time proxy)
- tau        = current trust score (Eq. 3.37)
- PDRVar     = variance of FFc over detection window W* (Eq. 3.17)
- CoordScore = cross-node lagged correlation max over partners j and lags tau in [1, W*] (Eq. 3.22)
- SpoofDev   = windowed normalized metric-deviation severity (Eq. 3.16)
- DelayInfl  = d_bar / d_ref (Eq. 3.25)
- lambda_t   = topology change rate

Output size for ALL agents: 5 (actions a0..a4).

## 5. ACTION SPACE (Eq. 3.45) — SHARED STRUCTURE

A = {a0, a1, a2, a3, a4}. Action k maps to a fixed trust penalty delta_k^X:
- a0 -> 0.0 (no reduction, monitor only)
- a1 -> delta1 (small)
- a2 -> delta2 (moderate)
- a3 -> delta3 (large)
- a4 -> delta4 (maximum)

Constraint: 0 < delta1 < delta2 < delta3 < delta4, per agent.
Final values are TBD (grid search later). For now use placeholder defaults PER AGENT stored in config:
delta = [0.0, 0.05, 0.15, 0.30, 0.50]
Make these configurable per agent (they will differ after grid search).

## 6. REWARD FUNCTION (Eq. 3.46)

r_t = w1*r_sec - w2*r_fp - w3*r_qos - w4*r_end,  with w1+w2+w3+w4 = 1.

Placeholder defaults (configurable): w1=0.4, w2=0.3, w3=0.2, w4=0.1.

Shared components:
- r_qos (Eq. 3.56) = (d_bar_t - d_ref)/d_ref + (PDR_ref - PDR_t)/PDR_ref
- r_end (Eq. 3.47) = 1 if the agent issued a non-zero trust reduction but the blockchain could not act on it, else 0. For offline training with synthetic data, simulate this with a flag column `blockchain_reject` (default 0).

Attack-specific components (binary indicators computed from ground-truth columns in the training data):

IGH (Eqs. 3.48–3.49):
- r_sec = 1 if action > a0 AND node is a true IGH attacker (is_attacker == 1 with all Definition-3 conditions represented in data), else 0
- r_fp  = 1 if action > a0 AND node is innocent AND rho_recv << 1 (upstream victim), else 0

SP (Eqs. 3.50–3.51):
- r_sec = 1 if action > a0 AND is_attacker == 1 (confirmed via dFF > eta_FF), else 0
- r_fp  = 1 if action > a0 AND is_attacker == 0 AND rho_recv << 1, else 0

ALS (Eqs. 3.52–3.53):
- r_sec = 1 if action > a0 AND is_attacker == 1 (confirmed spoofing), else 0
- r_fp  = 1 if action > a0 AND is_attacker == 0 AND lambda_t is high (mobility noise mistaken for spoofing), else 0

FS (Eqs. 3.54–3.55):
- r_sec = 1 if action > a0 AND is_attacker == 1 (dFF > eta_FF on Stage-1 flagged path), else 0
- r_fp  = 1 if action > a0 AND is_attacker == 0 AND (rho_recv << 1 OR lambda_t high), else 0

Additional shaping (allowed, keep simple): when is_attacker == 1, scale r_sec by action magnitude appropriateness is NOT required — keep the binary form exactly as above. Do not invent extra reward terms.

Thresholds used in reward logic (configurable placeholders): rho_recv_low = 0.5 (for "rho_recv << 1"), lambda_high = 0.7 (normalized).

## 7. NETWORK ARCHITECTURE (Eqs. 3.57)

Dueling DQN per agent:
- Input layer: size = agent's state dim (8/5/4/5)
- Shared trunk: 2 hidden layers, 128 neurons each, ReLU (placeholder from search space {2,3,4} layers x {64,128,256}; make layer count and width configurable)
- Value stream: Linear(128 -> 1)
- Advantage stream: Linear(128 -> 5)
- Combine: Q(s,a) = V(s) + A(s,a) - mean_a(A(s,a))   [Eq. 3.57]

## 8. TRAINING ALGORITHM (Algorithm 2, Eqs. 3.58–3.63)

Per agent, fully independent:

- Double DQN target (Eq. 3.58): y = r + gamma * Q_target(s', argmax_a Q_main(s', a))
  (for terminal transitions: y = r)
- Prioritized Experience Replay:
  - priority p_i = |TD_error_i| + eps_per   (Eq. 3.59)
  - sampling prob P(i) = p_i^alpha / sum_k p_k^alpha   (Eq. 3.60)
  - IS weights w_i = (1/(N*P(i)))^beta / max_j w_j   (Eq. 3.61), beta annealed to 1 over training
- Huber loss with threshold delta_huber (Eq. 3.62)
- Objective: mean over batch of w_i * HuberLoss(y_i, Q(s_i,a_i))   (Eq. 3.63)
- Soft target update: theta_target = (1-kappa)*theta_target + kappa*theta_main
- Epsilon-greedy exploration, epsilon decayed from eps0 to eps_min

Hyperparameter placeholder defaults (ALL configurable via a config dict/file; final values come from grid search later):
- gamma = 0.95
- learning rate = 0.001, optimizer = Adam
- batch_size = 64
- buffer capacity B_max = 100000, B_min = 1000 (min buffer size before training starts)
- alpha_per = 0.6, beta_per initial = 0.4 (anneal to 1.0), eps_per = 1e-5
- delta_huber = 1.0
- kappa (soft update) = 0.005
- eps0 = 1.0, eps_min = 0.05, epsilon decay = linear over 80% of episodes
- N_ep (episodes) = configurable, default 500 for smoke tests

## 9. OFFLINE TRAINING DATA FORMAT

Training is OFFLINE from CSV files (no live environment). One CSV per agent:
- data/sp_train.csv, data/als_train.csv, data/igh_train.csv, data/fs_train.csv

Each row = one node's observation at one cycle. Required columns:
- The agent's state features (exact names from Section 4)
- is_attacker (0/1 ground truth, used ONLY by the reward function, NEVER in the state)
- blockchain_reject (0/1, default 0)
- PDR_t and d_bar_t network-level values for r_qos (plus constants d_ref, PDR_ref in config)

Transitions: consecutive rows for the SAME node_id form (s_t, s_{t+1}) pairs. Include node_id and cycle_id columns to build transitions correctly. A node's last cycle row is a terminal transition.

The action a_t is chosen by the agent during training (epsilon-greedy), the reward is computed on the fly from the ground-truth columns. (The CSV does NOT contain pre-recorded actions/rewards.)

## 10. IMPLEMENTATION ORDER (FOLLOW EXACTLY, STEP BY STEP)

### STAGE 1 — Code implementation only. NO TRAINING RUNS in this stage.

Step 1: Create project structure:
```
fm_dad/
  config.py            # all hyperparameters + 4 agent configs (features, input size, deltas)
  networks.py          # DuelingDQN class (Eq. 3.57)
  replay_buffer.py     # PrioritizedReplayBuffer (Eqs. 3.59-3.61)
  rewards.py           # reward functions, one per attack (Eqs. 3.46-3.56)
  agent.py             # DQNAgent class: act(), remember(), learn() (Eqs. 3.58, 3.62, 3.63)
  data_loader.py       # loads agent CSV, builds (s, s') transitions per node
  train.py             # Algorithm 2 training loop, runs for one agent given its config
  synthetic_data.py    # STAGE 2: generates the 4 synthetic CSVs
  evaluate.py          # STAGE 3: sanity checks + reward curve plots
  data/                # CSVs go here
  models/              # trained .pt files go here
```

Step 2: Implement config.py with a dict per agent: name, feature list, input_dim, deltas, reward function id. Shared hyperparameters in a separate dict.

Step 3: Implement networks.py (DuelingDQN). Unit-check: forward pass with a random tensor of each agent's input size returns shape (batch, 5).

Step 4: Implement replay_buffer.py. Unit-check: push 100 random transitions, sample a batch of 32, verify IS weights and indices are returned, verify priority update works.

Step 5: Implement rewards.py. Unit-check with hand-made cases:
- attacker + penalizing action -> positive reward contribution from r_sec
- innocent upstream victim + penalizing action -> negative contribution from r_fp
- innocent + a0 -> reward ~ 0 (minus small qos term)

Step 6: Implement agent.py (holds main net, target net, buffer, optimizer; methods: act(state, epsilon), remember(...), learn() implementing Double DQN + PER + Huber). Unit-check: one learn() step runs without error and changes weights.

Step 7: Implement data_loader.py and train.py (Algorithm 2 loop). Verify train.py runs end-to-end on a tiny DUMMY random CSV (100 rows) for 5 episodes without crashing. This is a smoke test only, not real training.

STOP after Stage 1. Report all unit-check results before continuing.

### STAGE 2 — Synthetic data generation.

Step 8: Implement synthetic_data.py. For each agent generate a CSV (default 200 nodes x 100 cycles, ~10-20% attacker nodes) with these behavior rules:

SP data: attackers have low FFc (0.2-0.6), high dFF (> 0.2), normal rho_recv (~1.0). Innocents: FFc ~ 0.9-1.0, dFF < 0.05. Include some "upstream victim" innocents: low FFc BUT low rho_recv (0.2-0.5) — these test the false-positive logic.

ALS data: attackers have high SpoofDev (> threshold), innocents low SpoofDev. Include innocents with moderately high SpoofDev AND high lambda_t (mobility noise cases).

FS data: attackers have DelayInfl > 1.3 and dFF > 0.15. Innocents DelayInfl ~ 0.9-1.1. Include high-mobility innocents with slightly elevated DelayInfl.

IGH data (hardest): attacker FFc alternates over cycles between ~1.0 (ON) and ~0.2 (OFF) in blocks of 3-5 cycles, giving HIGH PDRVar; pairs of attackers use staggered (lag-shifted) ON/OFF schedules so CoordScore is high; rho_recv stays ~1.0. Innocents have stable FFc (low PDRVar). Compute PDRVar and CoordScore in the generator over a window W (use W=10 for synthetic data; the dynamic W* selection of Eq. 3.7 is a preprocessing concern for real NS-3 data, out of scope here) and write them as columns. CoordScore for synthetic data: max over partner nodes and lags 1..W of the normalized cross-correlation of FFc series (Eq. 3.22 double maximum).

Add gaussian noise to all features. Values clipped to valid ranges (fractions in [0,1] where applicable).

Step 9: Validate generated data: print per-class feature means (attacker vs innocent) per agent, confirm separations exist.

### STAGE 3 — Training and verification.

Step 10: Train the SP agent first (simplest). Then ALS, FS, and IGH last.

Step 11: For each agent, verify:
- (a) moving-average episode reward increases over training (save plot to models/<agent>_reward.png)
- (b) policy check: feed 20 clear attacker rows -> agent should mostly pick a3/a4; feed 20 clear innocent rows -> agent should mostly pick a0; feed upstream-victim rows -> should mostly pick a0/a1
- (c) save trained weights to models/<agent>.pt

Step 12: Print a final summary table: agent | final avg reward | attacker-detection action accuracy | innocent-protection action accuracy.

## 11. NOTES FOR THE CODING AGENT

- If any specification seems ambiguous, choose the SIMPLEST interpretation consistent with the equations, and leave a code comment marking it: `# ASSUMPTION: ...`. Do not silently invent behavior.
- Set random seeds (torch, numpy, random) for reproducibility; seed configurable.
- All hyperparameters listed as "placeholder" must live in config.py — nothing hardcoded inside logic files.
- Keep each file under ~250 lines if possible. Readability matters more than compactness.
- Target runtime: full 4-agent training on synthetic data should finish in minutes on CPU.
