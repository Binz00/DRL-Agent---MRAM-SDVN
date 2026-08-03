from __future__ import annotations

"""
trust_client.py — real trust chaincode client (Reduce design).

reduce_rsu_trust(node_id, amount)           → new score (float) or None
reduce_vehicle_trust(node_id, amount)       → new score (float) or None
batch_reduce_rsu_trust(updates: dict)       → dict[int, float] or None
batch_reduce_vehicle_trust(updates: dict)   → dict[int, float] or None

Single-node functions: send one POSITIVE reduction amount per call.
Batch functions: send {node_id: amount, ...} for an entire cycle in one Fabric
transaction — dramatically faster when multiple nodes are penalized per cycle.
Never raises — a Fabric hiccup logs and returns None so the cycle loop survives.
"""

import os
import re
import json
import logging
import subprocess

from bridge.fabric_config import (
    BIN_DIR, CFG_DIR, CHANNEL_NAME, ORDERER, ORDERER_HOSTNAME,
    PEER_ORG1, PEER_ORG2, TLS_CERT_ORG1, TLS_CERT_ORG2,
    MSP_PATH_ORG1, ORDERER_CA, LOCAL_MSPID, RSU_CC, VEHICLE_CC,
)

logger = logging.getLogger("pipeline")


# ---------------------------------------------------------------------------
# Real Fabric trust client (blockchain integration)
# ---------------------------------------------------------------------------

def _env() -> dict:
    env = os.environ.copy()
    env.update({
        "PATH":                        f"{BIN_DIR}:{env.get('PATH','')}",
        "FABRIC_CFG_PATH":             CFG_DIR,
        "CORE_PEER_TLS_ENABLED":       "true",
        "CORE_PEER_LOCALMSPID":        LOCAL_MSPID,
        "CORE_PEER_ADDRESS":           PEER_ORG1,
        "CORE_PEER_TLS_ROOTCERT_FILE": TLS_CERT_ORG1,
        "CORE_PEER_MSPCONFIGPATH":     MSP_PATH_ORG1,
    })
    return env


def _reduce_trust(chaincode: str, function: str, node_id: int, amount: float):
    """Reduce the on-chain score by `amount` (positive). Returns new score or None. Never raises."""
    ctor = {"function": function, "Args": [str(node_id), str(float(amount))]}
    cmd = [
        "peer", "chaincode", "invoke",
        "-o", ORDERER,
        "--ordererTLSHostnameOverride", ORDERER_HOSTNAME,
        "--tls", "--cafile", ORDERER_CA,
        "-C", CHANNEL_NAME, "-n", chaincode,
        "--peerAddresses", PEER_ORG1, "--tlsRootCertFiles", TLS_CERT_ORG1,
        "--peerAddresses", PEER_ORG2, "--tlsRootCertFiles", TLS_CERT_ORG2,
        "-c", json.dumps(ctor),
        "--waitForEvent",
    ]
    try:
        r = subprocess.run(cmd, env=_env(), capture_output=True, text=True, timeout=30)
        out = r.stderr + r.stdout
        if r.returncode != 0:
            logger.error("%s.%s(%d, -%.6f) FAILED rc=%d | %s",
                         chaincode, function, node_id, amount, r.returncode, out)
            return None
        m = re.search(r'payload:"(-?\d+(?:\.\d+)?)"', out)
        new = float(m.group(1)) if m else None
        logger.info("%s.%s(%d, -%.6f) OK new=%s",
                    chaincode, function, node_id, amount, new)
        return new
    except subprocess.TimeoutExpired:
        logger.error("%s.%s(%d) timeout", chaincode, function, node_id)
        return None
    except Exception as e:
        logger.error("%s.%s(%d) exception: %s", chaincode, function, node_id, e)
        return None
def _batch_reduce_trust(chaincode: str, function: str, updates: dict) -> dict | None:
    """
    Call a batch chaincode function with {node_id: amount, ...} in one transaction.

    `updates` is a dict of int node_id → float reduction_amount (positive values).
    The chaincode receives this as a single JSON string argument and applies all
    reductions atomically in one Fabric transaction.

    Returns a dict of int node_id → float new_score on success.
    Returns None if the Fabric call fails entirely.
    Nodes the chaincode rejected (returned score -1) are excluded from the result.
    Never raises.
    """
    if not updates:
        return {}

    # Fabric chaincode Args are always strings — serialize the whole dict as JSON.
    # Keys must be strings for JSON; values are float reduction amounts.
    updates_json = json.dumps({str(k): float(v) for k, v in updates.items()})
    ctor = {"function": function, "Args": [updates_json]}

    cmd = [
        "peer", "chaincode", "invoke",
        "-o", ORDERER,
        "--ordererTLSHostnameOverride", ORDERER_HOSTNAME,
        "--tls", "--cafile", ORDERER_CA,
        "-C", CHANNEL_NAME, "-n", chaincode,
        "--peerAddresses", PEER_ORG1, "--tlsRootCertFiles", TLS_CERT_ORG1,
        "--peerAddresses", PEER_ORG2, "--tlsRootCertFiles", TLS_CERT_ORG2,
        "-c", json.dumps(ctor),
        "--waitForEvent",
    ]
    try:
        r = subprocess.run(cmd, env=_env(), capture_output=True, text=True, timeout=60)
        out = r.stderr + r.stdout
        if r.returncode != 0:
            logger.error("%s.%s BATCH(%d nodes) FAILED rc=%d | %s",
                         chaincode, function, len(updates), r.returncode, out)
            return None

        # The chaincode payload is an escaped JSON string in the peer CLI output.
        # peer prints it as:  payload:"{\"205\":0.88,\"211\":0.76}"
        # The regex captures the inner content; unescape before parsing.
        m = re.search(r'payload:"(.*?)"(?:\s|$)', out)
        if not m:
            logger.warning("%s.%s BATCH: could not parse payload from: %s", chaincode, function, out)
            return None

        raw = json.loads(m.group(1).replace('\\"', '"'))
        # Convert string keys back to int, drop any -1 sentinel values (rejected IDs)
        result = {int(k): v for k, v in raw.items() if v >= 0}
        logger.info("%s.%s BATCH(%d nodes) OK — %d succeeded, %d rejected",
                    chaincode, function, len(updates), len(result),
                    len(updates) - len(result))
        return result

    except subprocess.TimeoutExpired:
        logger.error("%s.%s BATCH(%d nodes) timeout", chaincode, function, len(updates))
        return None
    except Exception as e:
        logger.error("%s.%s BATCH exception: %s", chaincode, function, e)
        return None

def reduce_rsu_trust(node_id: int, amount: float):
    return _reduce_trust(RSU_CC, "ReduceTrustScore", node_id, amount)


def reduce_vehicle_trust(node_id: int, amount: float):
    return _reduce_trust(VEHICLE_CC, "ReduceVehicleTrustScore", node_id, amount)


def batch_reduce_rsu_trust(updates: dict) -> dict | None:
    """Reduce multiple RSU trust scores in one Fabric transaction.
    updates: {node_id (int): reduction_amount (float), ...}
    Returns {node_id (int): new_score (float)} or None on total failure."""
    return _batch_reduce_trust(RSU_CC, "BatchReduceTrustScores", updates)


def batch_reduce_vehicle_trust(updates: dict) -> dict | None:
    """Reduce multiple vehicle trust scores in one Fabric transaction.
    updates: {node_id (int): reduction_amount (float), ...}
    Returns {node_id (int): new_score (float)} or None on total failure."""
    return _batch_reduce_trust(VEHICLE_CC, "BatchReduceVehicleTrustScores", updates)


# ---------------------------------------------------------------------------
# Mock trust API — used by run_pipeline.py (batch mode, no Fabric required)
# ---------------------------------------------------------------------------

_mock_trust_store: dict = {}  # node_id → current trust score (in-memory, per-run)


def _get_mock_trust(node_id: int, default: float = 1.0) -> float:
    """Return the current mock trust score for a node (default 1.0 for unseen nodes)."""
    return _mock_trust_store.get(node_id, default)


def _set_mock_trust(node_id: int, new_trust: float) -> None:
    """Set the mock trust score for a node, clamped to [0.0, 1.0]."""
    _mock_trust_store[node_id] = max(0.0, min(1.0, new_trust))


def reset_mock_store() -> None:
    """Clear the mock trust store — call between independent batch runs."""
    _mock_trust_store.clear()


def apply_trust_delta(
    node_id: int,
    delta: float,
    is_rsu: bool = False,
    current_trust: float = None,
) -> float:
    """
    Apply a trust penalty delta to a node using the in-memory mock store.

    Used exclusively by run_pipeline.py (batch mode).  Does NOT touch the
    Hyperledger Fabric ledger — the real blockchain calls are made by
    stream_pipeline._update_trust() via reduce_rsu_trust / reduce_vehicle_trust.

    Args:
        node_id       : Node being penalized.
        delta         : Penalty amount (positive = reduce trust).
        is_rsu        : Unused — kept for API compatibility with run_pipeline.py.
        current_trust : If provided, used as the starting value; otherwise the
                        mock store is queried (defaults to 1.0 for first-seen nodes).

    Returns:
        new_trust (float) — updated trust score clamped to [0.0, 1.0].
    """
    if current_trust is None:
        current_trust = _get_mock_trust(node_id)
    new_trust = max(0.0, current_trust - delta)
    _set_mock_trust(node_id, new_trust)
    logger.info(
        "apply_trust_delta: node=%d delta=%.4f %.4f → %.4f",
        node_id, delta, current_trust, new_trust,
    )
    return new_trust