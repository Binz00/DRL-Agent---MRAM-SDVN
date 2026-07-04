"""
replay_buffer.py — Prioritized Experience Replay buffer for FM-DAD agents.

Implements Eqs. 3.59–3.61 from the report:
    Eq. 3.59  priority:    p_i = |TD_error_i| + eps_per
    Eq. 3.60  sampling:    P(i) = p_i^alpha / sum_k( p_k^alpha )
    Eq. 3.61  IS weights:  w_i = (1/(N*P(i)))^beta / max_j(w_j)

Each agent owns exactly one buffer (R1: no sharing between agents).

Implementation note: a binary segment tree is used for O(log N) priority
updates and O(log N) prefix-sum sampling, matching the standard PER approach.
The tree stores raw priorities (before raising to alpha) to keep updates simple;
alpha is applied at sample time.
# ASSUMPTION: segment-tree approach; alpha applied during sampling, not storage.
"""

import numpy as np
from config import get_logger, SHARED_HP

logger = get_logger("replay_buffer")


class SegmentTree:
    """
    Min/Sum segment tree for O(log N) priority operations.

    Used internally by PrioritizedReplayBuffer to maintain:
      - Sum tree: fast prefix-sum queries for proportional sampling (Eq. 3.60)
      - Min tree: fast minimum query for max-weight normalisation (Eq. 3.61)

    Implements: Auxiliary data structure for Eqs. 3.60–3.61.
    """

    def __init__(self, capacity: int, operation, neutral_element: float) -> None:
        """
        Args:
            capacity        : Number of leaf nodes (must be a power of 2).
            operation       : Aggregation function (sum or min).
            neutral_element : Identity element for the operation.
        """
        self.capacity = capacity
        self.operation = operation
        self.neutral = neutral_element
        self.tree = [neutral_element] * (2 * capacity)
        logger.debug("SegmentTree init | capacity=%d", capacity)

    def _propagate(self, idx: int) -> None:
        """Propagate update from leaf up to root."""
        parent = idx // 2
        while parent >= 1:
            self.tree[parent] = self.operation(
                self.tree[2 * parent], self.tree[2 * parent + 1]
            )
            parent //= 2

    def update(self, idx: int, value: float) -> None:
        """Set leaf at position idx to value and propagate."""
        leaf = idx + self.capacity          # offset to leaf level
        self.tree[leaf] = value
        self._propagate(leaf)

    def query(self, start: int = 0, end: int = None) -> float:
        """Query aggregate over range [start, end)."""
        # For global sum/min, just return root
        return self.tree[1]

    def find_prefix(self, prefix_sum: float) -> int:
        """
        Find the leaf index whose cumulative sum first exceeds prefix_sum.
        Used for proportional sampling (Eq. 3.60).
        """
        idx = 1
        while idx < self.capacity:
            left = 2 * idx
            if self.tree[left] > prefix_sum:
                idx = left
            else:
                prefix_sum -= self.tree[left]
                idx = left + 1
        return idx - self.capacity          # convert back to [0, capacity)


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay buffer (Eqs. 3.59–3.61).

    Stores transitions (s, a, r, s', done) and supports weighted sampling
    proportional to TD-error priorities, with Importance Sampling correction.

    Implements:
        Eq. 3.59 — priority p_i = |TD_error| + eps_per
        Eq. 3.60 — sampling probability P(i) = p_i^alpha / sum_k(p_k^alpha)
        Eq. 3.61 — IS weight w_i = (1/(N*P(i)))^beta / max_j(w_j)
    """

    def __init__(
        self,
        capacity:   int   = SHARED_HP["buffer_capacity"],
        alpha:      float = SHARED_HP["alpha_per"],
        eps_per:    float = SHARED_HP["eps_per"],
    ) -> None:
        """
        Args:
            capacity : Maximum number of transitions to store (B_max).
            alpha    : Priority exponent (Eq. 3.60). 0 = uniform sampling.
            eps_per  : Small constant added to |TD| to avoid zero priority (Eq. 3.59).
        """
        # Round capacity up to nearest power of 2 for segment tree
        self.capacity = 1
        while self.capacity < capacity:
            self.capacity *= 2

        self.alpha   = alpha
        self.eps_per = eps_per
        self.size    = 0      # current number of stored transitions
        self.ptr     = 0      # next write position (circular)

        # Storage arrays
        self.states      = [None] * self.capacity
        self.actions     = [None] * self.capacity
        self.rewards     = [None] * self.capacity
        self.next_states = [None] * self.capacity
        self.dones       = [None] * self.capacity

        # Segment trees for O(log N) operations
        self._sum_tree = SegmentTree(self.capacity, lambda a, b: a + b, 0.0)
        self._min_tree = SegmentTree(self.capacity, min, float("inf"))

        # Max priority seen so far — new transitions receive this priority
        self._max_priority = 1.0

        logger.info(
            "PrioritizedReplayBuffer init | capacity=%d, alpha=%.2f, eps_per=%.1e",
            self.capacity, alpha, eps_per,
        )

    def push(
        self,
        state:      "np.ndarray",
        action:     int,
        reward:     float,
        next_state: "np.ndarray",
        done:       bool,
    ) -> None:
        """
        Store one transition.  New transitions get the current max priority so
        they are guaranteed to be sampled at least once before their TD error
        is known (Eq. 3.59 initialisation convention).

        Args:
            state      : State vector s_t.
            action     : Discrete action index a_t.
            reward     : Scalar reward r_t.
            next_state : Next state vector s_{t+1}.
            done       : Episode termination flag.

        Implements: Eq. 3.59 (priority initialisation for new transitions).
        """
        idx = self.ptr
        self.states[idx]      = state
        self.actions[idx]     = action
        self.rewards[idx]     = reward
        self.next_states[idx] = next_state
        self.dones[idx]       = done

        # Assign max priority so new transition is sampled
        priority = self._max_priority ** self.alpha
        self._sum_tree.update(idx, priority)
        self._min_tree.update(idx, priority)

        self.ptr  = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        logger.debug("Buffer push | ptr=%d, size=%d, priority=%.4f", idx, self.size, priority)

    def sample(
        self,
        batch_size: int,
        beta:       float,
    ) -> dict:
        """
        Sample a batch of transitions with Prioritized Experience Replay.

        Sampling is proportional to p_i^alpha (Eq. 3.60).
        IS weights are computed per Eq. 3.61 and normalised by the maximum weight.

        Args:
            batch_size : Number of transitions to sample.
            beta       : IS exponent (Eq. 3.61); annealed from beta_init to 1.0.

        Returns:
            dict with keys:
                'states', 'actions', 'rewards', 'next_states', 'dones',
                'indices'  — buffer indices (for priority update after learn step),
                'weights'  — IS weights w_i (Eq. 3.61), shape (batch_size,)

        Implements: Eqs. 3.60–3.61 (proportional sampling + IS correction).
        """
        assert self.size >= batch_size, (
            f"Buffer has {self.size} transitions, need at least {batch_size}."
        )

        total_priority = self._sum_tree.query()  # sum of p_i^alpha over all stored
        logger.debug(
            "Buffer sample | size=%d, batch=%d, beta=%.4f, total_priority=%.4f",
            self.size, batch_size, beta, total_priority,
        )

        indices  = []
        weights  = []
        segment  = total_priority / batch_size

        min_prob = self._min_tree.query() / total_priority   # P(i) for max-priority item
        max_weight = (1.0 / (self.size * min_prob + 1e-8)) ** beta

        for i in range(batch_size):
            # Stratified sampling: one sample per segment
            lo = segment * i
            hi = segment * (i + 1)
            prefix = np.random.uniform(lo, hi)
            idx = self._sum_tree.find_prefix(prefix)
            idx = max(0, min(idx, self.size - 1))  # clamp to valid range
            indices.append(idx)

            # IS weight — Eq. 3.61
            p_i     = self._sum_tree.tree[idx + self._sum_tree.capacity] / total_priority
            w_i     = ((1.0 / (self.size * p_i + 1e-8)) ** beta) / max_weight
            weights.append(w_i)

        indices = np.array(indices, dtype=np.int64)
        weights = np.array(weights, dtype=np.float32)

        batch = {
            "states":      np.stack([self.states[i]      for i in indices]),
            "actions":     np.array([self.actions[i]     for i in indices], dtype=np.int64),
            "rewards":     np.array([self.rewards[i]     for i in indices], dtype=np.float32),
            "next_states": np.stack([self.next_states[i] for i in indices]),
            "dones":       np.array([self.dones[i]       for i in indices], dtype=np.float32),
            "indices":     indices,
            "weights":     weights,
        }
        logger.debug("Buffer sample done | indices=%s, max_w=%.4f", indices[:4], weights.max())
        return batch

    def update_priorities(self, indices: "np.ndarray", td_errors: "np.ndarray") -> None:
        """
        Update priorities after a learning step using new TD errors.

        p_i = |TD_error_i| + eps_per  (Eq. 3.59)

        Args:
            indices   : Buffer indices returned by sample().
            td_errors : Absolute TD errors for each sampled transition.

        Implements: Eq. 3.59 (priority update).
        """
        for idx, td_err in zip(indices, td_errors):
            priority = (float(abs(td_err)) + self.eps_per) ** self.alpha
            self._sum_tree.update(int(idx), priority)
            self._min_tree.update(int(idx), priority)
            if priority > self._max_priority:
                self._max_priority = priority
        logger.debug(
            "Priority update | %d transitions, max_priority now %.4f",
            len(indices), self._max_priority,
        )

    def __len__(self) -> int:
        """Return number of stored transitions."""
        return self.size
