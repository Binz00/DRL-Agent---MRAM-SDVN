"""
hopcount.py — hop-count excess feature for the FS gate (Eq. 3.23 proxy).

Builds a per-flow directed graph from the committed flow table
(verified_flow_rules_cycle_{N}.csv) and the observed forwarding table
(observed_forwarding_fractions_{N}.csv), takes the longest src→dst path
in each, and computes:

    hop_excess_flow = h_obs_longest - h_bc_longest

Broadcast to every node that forwarded that flow, MAX-aggregated across a
node's flows if it's on more than one (same worst-case-MAX convention as
dFF elsewhere in this bridge, for the same reason: avoid diluting a real
stretch signal by averaging it against a normal flow).

LIMITATIONS (intentional, not hidden):
  - This is a per-cycle topology snapshot, not the per-packet observed
    hop count Eq. 3.23 specifies. NS-3 doesn't emit per-packet path length
    today; this is the closest derivable proxy from current CSVs.
  - Δh_min(v) (mobility-adjusted threshold, Eq. 3.9) is NOT applied here.
    hop_excess is a raw integer difference. The gate threshold is a
    placeholder (AGENT_CONFIGS["fs"]["eta_hop"]) pending grid search —
    same status as every other eta_* in this codebase.

GOTCHA — flow_id indexing mismatch between the two source files:
  verified_flow_rules is 1-indexed (NS-3 fid+1); observed_forwarding_fractions
  is 0-indexed (raw fid). Normalised here by subtracting 1 from
  verified_flow_rules' flow_id, then cross-checked against (src,dst) ==
  (flow_source,flow_destination) as a safety assertion. A mismatch is logged
  and that flow is skipped, not guessed.
"""

import logging
from pathlib import Path
from collections import defaultdict
from typing import Optional

import pandas as pd

from bridge.config_bridge import RAW_CSV_FOLDER

logger = logging.getLogger("bridge")

EDGE_FLOOR = 0.05  # ignore negligible-fraction edges (noise floor)
FLOW_RULES_PATTERN = "verified_flow_rules_cycle_{cycle}.csv"
OFF_PATTERN = "observed_forwarding_fractions_{cycle}.csv"


def _longest_path_len(edges: dict, src, dst, max_depth: int = 50) -> Optional[int]:
    """Longest simple src->dst path length in hops. DFS, visited-set per path."""
    best = {"len": None}

    def dfs(node, depth, visited):
        if depth > max_depth:
            return
        if node == dst:
            if best["len"] is None or depth > best["len"]:
                best["len"] = depth
            return
        for nxt, w in edges.get(node, []):
            if w <= EDGE_FLOOR or nxt in visited:
                continue
            dfs(nxt, depth + 1, visited | {nxt})

    dfs(src, 0, {src})
    return best["len"]


def _build_edge_map(df: pd.DataFrame, from_col: str, to_col: str, weight_col: str) -> dict:
    edges = defaultdict(list)
    for row in df.itertuples(index=False):
        edges[getattr(row, from_col)].append((getattr(row, to_col), getattr(row, weight_col)))
    return edges


def compute_hop_excess(cycle_no: int, folder: str = RAW_CSV_FOLDER) -> pd.DataFrame:
    """Returns DataFrame[cycle_id, node_id, hop_excess], one row per node
    that forwarded a flow this cycle. Nodes not on any flow are absent
    (outer-join in join.py fills them NaN — same as every other sparse feature)."""
    fr_path = Path(folder) / FLOW_RULES_PATTERN.format(cycle=cycle_no)
    off_path = Path(folder) / OFF_PATTERN.format(cycle=cycle_no)

    if not fr_path.exists() or not off_path.exists():
        logger.warning("[hopcount] Missing %s or %s — skipping cycle %d",
                        fr_path.name, off_path.name, cycle_no)
        return pd.DataFrame(columns=["cycle_id", "node_id", "hop_excess"])

    fr = pd.read_csv(fr_path)
    off = pd.read_csv(off_path)
    fr.columns = fr.columns.str.strip()
    off.columns = off.columns.str.strip()
    fr["flow_id_norm"] = fr["flow_id"] - 1   # correct the indexing gotcha

    node_hop_excess: dict = defaultdict(lambda: None)
    n_flows_matched = 0

    for fid in off["flow_id"].unique():
        off_flow = off[off["flow_id"] == fid]
        fr_flow = fr[fr["flow_id_norm"] == fid]
        if fr_flow.empty:
            continue

        src, dst = off_flow["flow_source"].iloc[0], off_flow["flow_destination"].iloc[0]
        fr_src, fr_dst = fr_flow["src"].iloc[0], fr_flow["dst"].iloc[0]
        if (src, dst) != (fr_src, fr_dst):
            logger.warning("[hopcount] cycle=%d flow=%d src/dst mismatch off=(%s,%s) "
                            "flow_rules=(%s,%s) — skipping", cycle_no, fid, src, dst, fr_src, fr_dst)
            continue

        obs_edges = _build_edge_map(off_flow, "forwarding_node_id", "next_hop_id", "observed_ff")
        bc_edges = _build_edge_map(fr_flow, "from_node", "to_node", "delta_value")
        h_obs = _longest_path_len(obs_edges, src, dst)
        h_bc = _longest_path_len(bc_edges, fr_src, fr_dst)
        if h_obs is None or h_bc is None:
            continue  # unreachable this cycle — skip, don't fabricate a value

        hop_excess = h_obs - h_bc
        n_flows_matched += 1
        for nid in set(off_flow["forwarding_node_id"]) | set(fr_flow["from_node"]):
            prev = node_hop_excess[nid]
            node_hop_excess[nid] = hop_excess if prev is None else max(prev, hop_excess)

    result = pd.DataFrame(
        [{"cycle_id": cycle_no, "node_id": nid, "hop_excess": v}
         for nid, v in node_hop_excess.items() if v is not None],
        columns=["cycle_id", "node_id", "hop_excess"],
    )
    logger.info("[hopcount] cycle=%d: %d/%d flows matched, hop_excess computed for %d nodes",
                cycle_no, n_flows_matched, off["flow_id"].nunique(), len(result))
    return result
