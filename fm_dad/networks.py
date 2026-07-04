"""
networks.py — Dueling DQN network architecture for FM-DAD agents.

Implements Eq. 3.57 from the report:
    Q(s, a) = V(s) + A(s, a) - mean_a[ A(s, a) ]

The same class is instantiated separately for each of the four agents
(IGH, SP, ALS, FS). No weight sharing occurs between instances.

References:
    Eq. 3.57  — Dueling decomposition of Q-values
    Section 7 — Network architecture specification
"""

from __future__ import annotations
from typing import List
import torch
import torch.nn as nn
from config import get_logger, SHARED_HP

logger = get_logger("networks")


class DuelingDQN(nn.Module):
    """
    Dueling Deep Q-Network as specified in Eq. 3.57.

    Architecture:
        Input  -> [Shared Trunk: hidden_layers x hidden_size, ReLU]
               -> Value stream:    Linear(hidden_size -> 1)
               -> Advantage stream: Linear(hidden_size -> output_size)
        Q(s,a) = V(s) + A(s,a) - mean_a(A(s,a))

    Args:
        input_dim    (int): Dimension of the state vector (4, 5, or 8 per agent).
        output_size  (int): Number of discrete actions |A| = 5.
        hidden_layers (int): Number of hidden layers in the shared trunk.
        hidden_size   (int): Width (neurons) of each hidden layer.

    Implements: Eq. 3.57 (dueling decomposition).
    """

    def __init__(
        self,
        input_dim:     int,
        output_size:   int  = SHARED_HP["output_size"],
        hidden_layers: int  = SHARED_HP["hidden_layers"],
        hidden_size:   int  = SHARED_HP["hidden_size"],
    ) -> None:
        super().__init__()
        logger.debug(
            "DuelingDQN.__init__ | input_dim=%d, output_size=%d, "
            "hidden_layers=%d, hidden_size=%d",
            input_dim, output_size, hidden_layers, hidden_size,
        )

        # --- Shared trunk (Eq. 3.57: shared layers) -------------------------
        trunk_layers: List[nn.Module] = []
        in_features = input_dim
        for layer_idx in range(hidden_layers):
            trunk_layers.append(nn.Linear(in_features, hidden_size))
            trunk_layers.append(nn.ReLU())
            logger.debug(
                "  Trunk layer %d: Linear(%d -> %d) + ReLU",
                layer_idx, in_features, hidden_size,
            )
            in_features = hidden_size
        self.trunk = nn.Sequential(*trunk_layers)

        # --- Value stream V(s)  (Eq. 3.57) -----------------------------------
        self.value_stream = nn.Linear(hidden_size, 1)
        logger.debug("  Value stream:     Linear(%d -> 1)", hidden_size)

        # --- Advantage stream A(s,a) (Eq. 3.57) ------------------------------
        self.advantage_stream = nn.Linear(hidden_size, output_size)
        logger.debug(
            "  Advantage stream: Linear(%d -> %d)", hidden_size, output_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass implementing Eq. 3.57:
            Q(s,a) = V(s) + A(s,a) - mean_a(A(s,a))

        Args:
            x (torch.Tensor): State batch of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Q-value estimates of shape (batch_size, output_size).

        Implements: Eq. 3.57 (dueling decomposition).
        """
        logger.debug("DuelingDQN.forward | input shape: %s", tuple(x.shape))

        # Shared feature representation
        features = self.trunk(x)                          # (batch, hidden_size)

        # Value branch
        V = self.value_stream(features)                   # (batch, 1)

        # Advantage branch
        A = self.advantage_stream(features)               # (batch, output_size)

        # Dueling combination — Eq. 3.57
        # Subtract row-wise mean to centre advantages (improves stability)
        Q = V + A - A.mean(dim=1, keepdim=True)           # (batch, output_size)

        logger.debug(
            "DuelingDQN.forward | V shape: %s, A shape: %s, Q shape: %s",
            tuple(V.shape), tuple(A.shape), tuple(Q.shape),
        )
        return Q
