"""
agent.py — DQN Agent class for FM-DAD (one instance per attack type).

Implements Algorithm 2 and Eqs. 3.58–3.63 from the report:
    Eq. 3.58 — Double DQN target:  y = r + gamma * Q_target(s', argmax_a Q_main(s',a))
    Eq. 3.62 — Huber loss (element-wise)
    Eq. 3.63 — Weighted mean loss:  L = mean_i( w_i * HuberLoss(y_i, Q(s_i,a_i)) )

Each DQNAgent holds its OWN:
    - main network θ^X
    - target network θ̂^X
    - replay buffer B^X
    - Adam optimizer
No parameters, gradients, or buffer data are shared across agents (R1).
"""

from __future__ import annotations
from typing import Optional, List
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from config import get_logger, SHARED_HP
from networks import DuelingDQN
from replay_buffer import PrioritizedReplayBuffer

logger = get_logger("agent")


class DQNAgent:
    """
    Independent DRL agent for one FM-DAD attack type.

    Implements:
        act()       — epsilon-greedy action selection
        remember()  — store transition in the agent's own PER buffer (B^X)
        learn()     — one gradient-descent step using Double DQN + PER + Huber loss
                      (Algorithm 2, Eqs. 3.58–3.63)
        soft_update() — soft target network update (kappa parameter)

    All four agents (SP, ALS, IGH, FS) are separate instances of this class.
    They share NO weights, target weights, or buffer data (Rule R1).
    """

    def __init__(self, agent_cfg: dict, hp: dict = SHARED_HP, device: str = None) -> None:
        """
        Initialise a DRL agent for one attack type.

        Args:
            agent_cfg : Per-agent config dict from AGENT_CONFIGS in config.py.
            hp        : Shared hyperparameter dict (SHARED_HP).
            device    : Torch device string ('cpu' or 'cuda'). Auto-detected if None.

        Implements: Algorithm 2 initialisation (Eqs. 3.58–3.63).
        """
        self.cfg      = agent_cfg
        self.hp       = hp
        self.name     = agent_cfg["name"]
        self.input_dim = agent_cfg["input_dim"]
        self.n_actions = hp["output_size"]
        self.deltas    = agent_cfg["deltas"]   # Eq. 3.45 trust penalties

        # Device selection
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        logger.info(
            "[%s] DQNAgent init | input_dim=%d, n_actions=%d, device=%s",
            self.name, self.input_dim, self.n_actions, self.device,
        )

        # --- Main network θ^X (Eq. 3.57) ------------------------------------
        self.main_net = DuelingDQN(
            input_dim     = self.input_dim,
            output_size   = self.n_actions,
            hidden_layers = hp["hidden_layers"],
            hidden_size   = hp["hidden_size"],
        ).to(self.device)
        logger.info("[%s] Main network created: %s", self.name, self.main_net)

        # --- Target network θ̂^X (initially equal weights) (Eq. 3.58) -------
        self.target_net = DuelingDQN(
            input_dim     = self.input_dim,
            output_size   = self.n_actions,
            hidden_layers = hp["hidden_layers"],
            hidden_size   = hp["hidden_size"],
        ).to(self.device)
        self.target_net.load_state_dict(self.main_net.state_dict())
        self.target_net.eval()          # target net is never trained directly
        logger.info("[%s] Target network created (copy of main).", self.name)

        # --- Replay buffer B^X (Eqs. 3.59–3.61) ----------------------------
        self.buffer = PrioritizedReplayBuffer(
            capacity = hp["buffer_capacity"],
            alpha    = hp["alpha_per"],
            eps_per  = hp["eps_per"],
        )
        logger.info("[%s] PER buffer initialised (capacity=%d).", self.name, hp["buffer_capacity"])

        # --- Optimiser ------------------------------------------------------
        self.optimizer = optim.Adam(self.main_net.parameters(), lr=hp["lr"])
        logger.info("[%s] Adam optimiser created (lr=%.4f).", self.name, hp["lr"])

        # Epsilon state (managed externally by train.py, but stored here too)
        self.epsilon = hp["eps0"]

        # Beta state for IS annealing (Eq. 3.61)
        self.beta = hp["beta_per_init"]

        # Step counter (for target-network updates)
        self.learn_step = 0

    # -----------------------------------------------------------------------
    # act() — epsilon-greedy policy
    # -----------------------------------------------------------------------

    def act(self, state: np.ndarray, epsilon: float = None) -> int:
        """
        Select an action using an epsilon-greedy policy.

        With probability epsilon: choose a random action (exploration).
        Otherwise: choose argmax_a Q(s, a) using the main network (exploitation).

        Args:
            state   : State vector (numpy array, shape (input_dim,)).
            epsilon : Exploration rate. If None, uses self.epsilon.

        Returns:
            int: Chosen action index ∈ {0, 1, 2, 3, 4}.

        Implements: epsilon-greedy exploration (Algorithm 2, Section 8).
        """
        eps = epsilon if epsilon is not None else self.epsilon
        if np.random.rand() < eps:
            action = np.random.randint(0, self.n_actions)
            logger.debug("[%s] act | RANDOM action=%d (eps=%.3f)", self.name, action, eps)
        else:
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.main_net(state_t)
            action = int(q_values.argmax(dim=1).item())
            logger.debug(
                "[%s] act | GREEDY action=%d, Q=%s",
                self.name, action, q_values.cpu().numpy().round(3),
            )
        return action

    # -----------------------------------------------------------------------
    # remember() — store one transition in this agent's own buffer
    # -----------------------------------------------------------------------

    def remember(
        self,
        state:      np.ndarray,
        action:     int,
        reward:     float,
        next_state: np.ndarray,
        done:       bool,
    ) -> None:
        """
        Store a transition in the agent's private PER buffer (B^X).

        Args:
            state      : s_t
            action     : a_t ∈ {0..4}
            reward     : r_t (computed by rewards.py)
            next_state : s_{t+1}
            done       : terminal flag

        Rule R1: This buffer is NEVER accessed by any other agent.

        Implements: Buffer push with max-priority initialisation (Eq. 3.59).
        """
        logger.debug(
            "[%s] remember | action=%d, reward=%.4f, done=%s",
            self.name, action, reward, done,
        )
        self.buffer.push(state, action, reward, next_state, done)

    # -----------------------------------------------------------------------
    # learn() — one gradient-descent update step
    # -----------------------------------------------------------------------

    def learn(self, beta: float = None) -> Optional[float]:
        """
        Perform one Double DQN update step with Prioritized Experience Replay.

        Steps:
        1. Sample a batch from B^X using PER (Eqs. 3.60–3.61).
        2. Compute Double DQN targets (Eq. 3.58):
               y_i = r_i + gamma * Q_target(s'_i, argmax_a Q_main(s'_i, a))
               (terminal transitions: y_i = r_i)
        3. Compute IS-weighted Huber loss (Eqs. 3.62–3.63):
               L = mean_i( w_i * HuberLoss(y_i, Q(s_i, a_i)) )
        4. Backprop and Adam step.
        5. Update priorities with new |TD errors| (Eq. 3.59).
        6. Soft-update target network: θ̂ = (1-κ)θ̂ + κθ (Section 8).

        Args:
            beta : IS exponent override. If None, uses self.beta.

        Returns:
            float | None: Loss value (for logging), or None if buffer too small.

        Implements: Algorithm 2, Eqs. 3.58, 3.62–3.63.
        """
        if len(self.buffer) < self.hp["buffer_min"]:
            logger.debug(
                "[%s] learn | buffer too small (%d < %d), skipping.",
                self.name, len(self.buffer), self.hp["buffer_min"],
            )
            return None

        beta_val = beta if beta is not None else self.beta
        batch    = self.buffer.sample(self.hp["batch_size"], beta=beta_val)

        # ---- Unpack batch --------------------------------------------------
        states_np      = batch["states"]        # (B, input_dim)
        actions_np     = batch["actions"]        # (B,)
        rewards_np     = batch["rewards"]        # (B,)
        next_states_np = batch["next_states"]    # (B, input_dim)
        dones_np       = batch["dones"]          # (B,)
        indices        = batch["indices"]        # (B,)
        weights_np     = batch["weights"]        # (B,) IS weights Eq. 3.61

        # Move tensors to device
        states      = torch.FloatTensor(states_np).to(self.device)
        actions     = torch.LongTensor(actions_np).to(self.device)
        rewards     = torch.FloatTensor(rewards_np).to(self.device)
        next_states = torch.FloatTensor(next_states_np).to(self.device)
        dones       = torch.FloatTensor(dones_np).to(self.device)
        weights     = torch.FloatTensor(weights_np).to(self.device)

        logger.debug(
            "[%s] learn | batch sampled: states=%s, beta=%.4f",
            self.name, tuple(states.shape), beta_val,
        )

        # ---- Double DQN target (Eq. 3.58) ----------------------------------
        with torch.no_grad():
            # Action selection via main network
            next_actions = self.main_net(next_states).argmax(dim=1)   # (B,)
            # Value estimation via target network
            next_q       = self.target_net(next_states)                # (B, n_actions)
            next_q_sel   = next_q.gather(1, next_actions.unsqueeze(1)).squeeze(1)  # (B,)
            # y_i = r_i + gamma * Q_target(s', a*) * (1 - done_i)  [Eq. 3.58]
            y = rewards + self.hp["gamma"] * next_q_sel * (1.0 - dones)
        logger.debug("[%s] learn | targets computed, y[:4]=%s", self.name, y[:4].cpu().numpy())

        # ---- Current Q estimates for chosen actions -------------------------
        q_all     = self.main_net(states)                              # (B, n_actions)
        q_chosen  = q_all.gather(1, actions.unsqueeze(1)).squeeze(1)  # (B,)

        # ---- Huber loss element-wise (Eq. 3.62) ----------------------------
        huber = nn.HuberLoss(reduction="none", delta=self.hp["delta_huber"])
        td_errors_raw = (y - q_chosen).detach().cpu().numpy()         # for priority update
        element_losses = huber(q_chosen, y)                            # (B,)

        # ---- IS-weighted mean loss (Eq. 3.63) ------------------------------
        loss = (weights * element_losses).mean()

        # ---- Optimiser step ------------------------------------------------
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        loss_val = float(loss.item())
        logger.debug("[%s] learn | loss=%.6f", self.name, loss_val)

        # ---- Update priorities (Eq. 3.59) ----------------------------------
        self.buffer.update_priorities(indices, np.abs(td_errors_raw))

        # ---- Soft-update target network (Section 8) -----------------------
        self._soft_update()

        self.learn_step += 1
        return loss_val

    # -----------------------------------------------------------------------
    # _soft_update() — target network soft update
    # -----------------------------------------------------------------------

    def _soft_update(self) -> None:
        """
        Soft-update target network parameters.

        θ̂ ← (1 - κ) * θ̂ + κ * θ    (Section 8)

        where κ = kappa (small, e.g. 0.005) for gradual target tracking.

        Implements: Soft target update (Section 8 of the report).
        """
        kappa = self.hp["kappa"]
        for main_param, target_param in zip(
            self.main_net.parameters(), self.target_net.parameters()
        ):
            target_param.data.copy_(
                (1.0 - kappa) * target_param.data + kappa * main_param.data
            )
        logger.debug("[%s] _soft_update done (kappa=%.4f).", self.name, kappa)

    # -----------------------------------------------------------------------
    # save() / load() helpers
    # -----------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save main network weights to a .pt file."""
        torch.save(self.main_net.state_dict(), path)
        logger.info("[%s] Model saved -> %s", self.name, path)

    def load(self, path: str) -> None:
        """Load main network weights and sync target network."""
        self.main_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(self.main_net.state_dict())
        logger.info("[%s] Model loaded <- %s", self.name, path)
