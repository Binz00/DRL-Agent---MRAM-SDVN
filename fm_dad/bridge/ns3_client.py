import socket
import time
import logging

logger = logging.getLogger("stream.ns3")

NS3_CONTROL_SOCKET = "/tmp/vanet_verify.sock"
NS3_SOCKET_TIMEOUT_SECONDS = 2.0
NS3_REVOKE_RETRIES = 3
NS3_RETRY_DELAY_SECONDS = 0.2


def request_ns3_revocation(routing_node_id: int) -> str:
    """
    Send one REVOKE_NODE command to ns-3. Returns the ns-3 status string
    (e.g. 'QUEUED') on success, raises on failure after retries.
    Pass the routing node_id as-is — ns-3 does the routing→global conversion.
    """
    if routing_node_id < 0:
        raise ValueError("routing_node_id cannot be negative")

    command = f"REVOKE_NODE:{routing_node_id}\n".encode("utf-8")
    last_error = None

    for attempt in range(1, NS3_REVOKE_RETRIES + 1):
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
                client.settimeout(NS3_SOCKET_TIMEOUT_SECONDS)
                client.connect(NS3_CONTROL_SOCKET)
                client.sendall(command)
                response = client.recv(1024)

            if not response:
                raise ConnectionError("ns-3 closed the socket without a response")

            status = response.decode("utf-8").strip()

        except (OSError, ConnectionError) as exc:
            last_error = exc
            if attempt < NS3_REVOKE_RETRIES:
                logger.warning(
                    "Node %d: revocation attempt %d/%d failed: %s; retrying",
                    routing_node_id, attempt, NS3_REVOKE_RETRIES, exc,
                )
                time.sleep(NS3_RETRY_DELAY_SECONDS)
                continue
            break

        if status.startswith(("QUEUED", "ALREADY_QUEUED")):
            return status
        # Connection worked but ns-3 rejected the command — retrying won't help.
        raise RuntimeError(f"ns-3 rejected revocation: {status}")

    raise RuntimeError(
        f"could not revoke routing node {routing_node_id}: {last_error}"
    )