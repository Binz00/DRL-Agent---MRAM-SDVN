"""
config_bridge.py — All configuration for the FM-DAD data bridge.

Nothing in join.py is hardcoded. All file patterns, column-name maps,
folder paths, and aggregation rules live here.
"""

import pathlib

# ---------------------------------------------------------------------------
# Folder paths  (resolved relative to this file so the package is portable)
# ---------------------------------------------------------------------------
_BRIDGE_DIR = pathlib.Path(__file__).parent        # fm_dad/bridge/
_FM_DAD_DIR = _BRIDGE_DIR.parent                   # fm_dad/

RAW_CSV_FOLDER = str(_FM_DAD_DIR / "data" / "raw_csvs")
LOG_FILE       = str(_FM_DAD_DIR / "logs" / "bridge.log")
DELAY_REF_FILE = str(_FM_DAD_DIR / "data" / "raw_csvs" / "delay_reference.csv")

LAMBDA_REF = 5000.0  # PLACEHOLDER — set from the max lambda_t across a full simulation run once available.


# ---------------------------------------------------------------------------
# File-name patterns — {cycle} is replaced with the integer cycle number
# ---------------------------------------------------------------------------
FILE_PATTERNS = {
    "off":          "observed_forwarding_fractions_{cycle}.csv",
    "hop_delay":    "vanet_hop_delay_{cycle}.csv",
    "flow_metrics": "per_flow_metrics_{cycle}.csv",
    "planned":      "verified_planned_inbound_cycle_{cycle}.csv",
    "anomaly":      "ff_node_anomaly_scores_{cycle}.csv",
    "trust":        "node_trust_scores_{cycle}.csv",
    "als":          "als_link_metrics_{cycle}.csv",
    "ground_truth": "node_attack_ground_truth_{cycle}.csv",
}

# Reference pattern used to auto-detect which cycle numbers exist in the folder.
# Any file that is guaranteed to appear once per cycle works here.
CYCLE_DETECTION_PATTERN = "node_attack_ground_truth_{cycle}.csv"
CYCLE_DETECTION_REGEX   = r"node_attack_ground_truth_(\d+)\.csv"


# ---------------------------------------------------------------------------
# Column map: file key → raw node-id column name (None = no node-id column)
# ---------------------------------------------------------------------------
NODE_ID_COL = {
    "off":          "forwarding_node_id",   # renamed → node_id
    "hop_delay":    "node_id",
    "flow_metrics": None,                   # per-flow only; broadcast via off mapping
    "planned":      "node",                 # renamed → node_id
    "anomaly":      "node_id",
    "trust":        "node_id",
    "als":          "node_id",
    "ground_truth": "node_id",
}

# Non-standard cycle-id column names (default assumed: "cycle_id")
CYCLE_ID_COL = {
    "planned":      "cycle",
    "flow_metrics": "Cycle No.",
}

# Non-standard flow-id column names (default assumed: "flow_id")
FLOW_ID_COL = {
    "flow_metrics": "Flow No.",
}


# ---------------------------------------------------------------------------
# Unsafe column names in per_flow_metrics → safe Python identifiers
# ---------------------------------------------------------------------------
RENAME_UNSAFE = {
    "PDR_k (%)":        "PDR_k_pct",
    "d_k (ms)":         "d_k_ms",
    "e2e_latency (ms)": "e2e_latency_ms",
    "Jitter (ms)":      "Jitter_ms",
    "cpu_overhead (ms)":"cpu_overhead_ms",
}


# ---------------------------------------------------------------------------
# Aggregation rules: how to collapse multiple (cycle, flow, node) rows
# to one (cycle, node) row
# ---------------------------------------------------------------------------
AGG_SUM = frozenset([
    "unique_inbound_count",
    "unique_outbound_to_next_hop",
    "total_inbound",
    "total_outbound",
    "hop_delay_sum_ms",
    "hop_delay_count",
    "planned_inbound_by_subflow",
    "planned_inbound_by_mainflow",
])

AGG_MEAN = frozenset([
    "observed_ff",
    "node_pdr",
    "mean_delay_ms",
    "PDR_k_pct",
    "d_k_ms",
    "e2e_latency_ms",
    "Jitter_ms",
    "lambda_t",
    "cpu_overhead_ms",
    "inbound_ratio",
    # sum_abs_ff_deviation: NS-3 pre-summed absolute deviation across flows.
    # Kept in MEAN as-is (separate review pending for SUM vs MEAN correctness).
    "sum_abs_ff_deviation",
    # NOTE: "ff_deviation" (signed) is intentionally NOT listed here.
    # Aggregating the signed value then taking abs() causes sign cancellation:
    # a node with one over-forwarding flow (+0.5) and one dropping flow (−0.5)
    # would produce mean=0.0, completely hiding the attack.
    # Instead, abs() is applied per flow row in join.py (producing abs_ff_deviation)
    # and that column is MAX-aggregated below — capturing the worst-case departure
    # from the committed forwarding plan (report Eq. 3.99, Phase 2 procedure).
])


AGG_MAX = frozenset([
    # abs_ff_deviation: per-flow |ff_deviation|, computed in join.py before
    # aggregation (see load_cycle() Step 1).  MAX across flows per node gives
    # the worst-case departure from the committed plan — matching Phase 2 of
    # the report's SP-DM procedure (Eq. 3.99).
    "abs_ff_deviation",
    # NOTE: 'detected' from ff_node_anomaly_scores is intentionally EXCLUDED.
    # NS-3 sets 'detected' using: sum_abs_ff_deviation_normalized > 0.5
    # This is the raw forwarding-fraction threshold the supervisor identified as wrong
    # (Algorithm 3, Stage 2 should use δFF > η_FF, not a raw fraction threshold).
    # The Python bridge computes its own gate in trigger.py using:
    #   dFF = abs_ff_deviation > eta_dFF  (correct per-plan deviation, Eq. 3.99)
    # The NS-3 pre-flag is therefore both wrong and unused — drop it.
])

# Columns that are join/routing artifacts — dropped after aggregation
# so they don't cause naming conflicts during the final outer join.
DROP_POST_AGG = frozenset([
    "flow_id",
    "flow_source",
    "flow_destination",
    "sim_time_s",
    "next_hop_id",
    "next_hop_type",
    "forwarding_node_type",
    "node_type",
    "prev_hop_id",
    "mean_pdr_flow",
    "pdr_deviation",
    # 'detected' from ff_node_anomaly_scores: NS-3 pre-flag using wrong threshold
    # (sum_abs_ff_deviation_normalized > 0.5 instead of δFF > η_FF).
    # Bridge gate in trigger.py uses dFF = abs(ff_deviation) > eta_dFF instead.
    "detected",
    "sum_abs_ff_deviation_normalized",
    "threshold",
])


# ---------------------------------------------------------------------------
# Dynamic window W* selection (Part 3 — windowed features)
# ---------------------------------------------------------------------------
# W* candidates: higher lambda_t (more topology change) → shorter window
WINDOW_CANDIDATES = (10, 15, 20)

# Thresholds on the cycle-wide median lambda_t:
#   lambda_t >= LAMBDA_W_HIGH  →  W* = 10  (high mobility, short memory)
#   lambda_t >= LAMBDA_W_MED   →  W* = 15  (moderate mobility)
#   lambda_t <  LAMBDA_W_MED   →  W* = 20  (low mobility, long memory)
LAMBDA_W_HIGH = 0.6   # lambda_t_norm >= 0.6 → W* = 10  (high mobility)
LAMBDA_W_MED  = 0.2   # lambda_t_norm >= 0.2 → W* = 15  (moderate mobility)
                      # lambda_t_norm <  0.2 → W* = 20  (low mobility)
