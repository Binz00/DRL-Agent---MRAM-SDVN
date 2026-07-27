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



