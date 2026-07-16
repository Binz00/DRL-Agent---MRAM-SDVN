"""
trust_client.py — Mock/real trust chaincode client (Part 5).

Public API:
    apply_trust_delta(node_id, delta, is_rsu, current_trust) → new_trust

When USE_MOCK_CHAINCODE is True (default), calls are logged but do not touch
a real Hyperledger Fabric network.  When the real Fabric SDK is integrated
later, set USE_MOCK_CHAINCODE = False and implement _call_chaincode().
"""

import logging

logger = logging.getLogger("pipeline")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
USE_MOCK_CHAINCODE = True  # Flip to False when real Fabric SDK is ready


# ---------------------------------------------------------------------------
# Mock trust store (in-memory, per-run)
# ---------------------------------------------------------------------------
_mock_trust_store: dict = {}  # node_id → current trust score


def _get_mock_trust(node_id: int, default: float = 1.0) -> float:
    """Return the current mock trust score for a node (default 1.0)."""
    return _mock_trust_store.get(node_id, default)


def _set_mock_trust(node_id: int, new_trust: float) -> None:
    """Set the mock trust score for a node, clamped to [0, 1]."""
    _mock_trust_store[node_id] = max(0.0, min(1.0, new_trust))


def reset_mock_store() -> None:
    """Clear the mock trust store (call between runs if needed)."""
    _mock_trust_store.clear()


# ---------------------------------------------------------------------------
# Real chaincode stub (to be implemented with Fabric SDK)
# ---------------------------------------------------------------------------

def _call_chaincode(node_id: int, delta: float, is_rsu: bool) -> float:
    """
    Call the Hyperledger Fabric chaincode to edit a node's trust score.

    EditTrustScore(id, -delta)        for RSU nodes
    EditVehicleTrustScore(id, -delta) for vehicle nodes

    Args:
        node_id : The node ID to update.
        delta   : The trust penalty (positive value to subtract).
        is_rsu  : True if the node is an RSU, False for vehicle.

    Returns:
        float: The new trust score returned by the chaincode.

    Raises:
        NotImplementedError: Until the real Fabric SDK is integrated.
    """
    raise NotImplementedError(
        "Real chaincode integration not yet available. "
        "Set USE_MOCK_CHAINCODE = True to use the mock."
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def apply_trust_delta(
    node_id: int,
    delta: float,
    is_rsu: bool = False,
    current_trust: float = None,
) -> float:
    """
    Apply a trust penalty to a node (Eq. 3.38 trust update).

    Calls the chaincode (or mock) to subtract delta from the node's trust.
    The trust score stays within [0.0, 1.0].

    Args:
        node_id       : Node to update.
        delta         : Trust reduction amount (>= 0).
        is_rsu        : True if RSU, False if vehicle.
        current_trust : Optional current trust (used for mock; real chaincode
                        reads from ledger).  If None, defaults to 1.0.

    Returns:
        float: New trust score after applying the penalty.
    """
    node_type = "RSU" if is_rsu else "Vehicle"

    if USE_MOCK_CHAINCODE:
        # Mock path
        old = current_trust if current_trust is not None else _get_mock_trust(node_id)
        new = max(0.0, min(1.0, old - delta))
        _set_mock_trust(node_id, new)

        chaincode_fn = "EditTrustScore" if is_rsu else "EditVehicleTrustScore"
        logger.info(
            "[TRUST] MOCK %s(node=%d, -%.3f) | %s | old=%.4f → new=%.4f",
            chaincode_fn, node_id, delta, node_type, old, new,
        )
        return new
    else:
        # Real chaincode path
        new = _call_chaincode(node_id, delta, is_rsu)
        logger.info(
            "[TRUST] REAL chaincode call | node=%d, delta=%.3f, %s → new=%.4f",
            node_id, delta, node_type, new,
        )
        return new
