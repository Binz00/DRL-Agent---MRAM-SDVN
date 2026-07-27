"""
trust_client.py — real trust chaincode client (Reduce design).

reduce_rsu_trust(node_id, amount)     → new score (float) or None
reduce_vehicle_trust(node_id, amount) → new score (float) or None

Sends a POSITIVE reduction amount to ReduceTrustScore / ReduceVehicleTrustScore,
which subtract it from the current on-chain score (the source of truth).
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


def reduce_rsu_trust(node_id: int, amount: float):
    return _reduce_trust(RSU_CC, "ReduceTrustScore", node_id, amount)


def reduce_vehicle_trust(node_id: int, amount: float):
    return _reduce_trust(VEHICLE_CC, "ReduceVehicleTrustScore", node_id, amount)