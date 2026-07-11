"""
join.py — Core join logic for the FM-DAD NS-3 data bridge (Part 1).

Public API:
    load_cycle(cycle_no, folder)   → DataFrame, one row per (cycle_id, node_id)
    load_all_cycles(folder)        → DataFrame for all detected cycles combined

Design:
    1. Each raw CSV is loaded and column names are normalised (cycle_id, node_id,
       flow_id) so downstream code never sees source-specific column names.
    2. Multi-row files (per flow-node) are aggregated to per-(cycle, node) using
       the rules in config_bridge.AGG_SUM / AGG_MEAN / AGG_MAX.
    3. per_flow_metrics has no node_id column; it is broadcast to every node on
       each flow using the flow→node mapping from observed_forwarding_fractions.
    4. All per-node DataFrames are outer-joined on (cycle_id, node_id) so that
       no node is dropped even if some source files are missing.
    5. Intermediate join-artifact columns (flow_id, sim_time_s, …) are dropped
       after aggregation to prevent naming conflicts in the final join.
"""

import logging
import re
from pathlib import Path
from typing import Optional

import pandas as pd

from bridge.config_bridge import (
    RAW_CSV_FOLDER,
    LOG_FILE,
    FILE_PATTERNS,
    CYCLE_DETECTION_REGEX,
    NODE_ID_COL,
    CYCLE_ID_COL,
    FLOW_ID_COL,
    RENAME_UNSAFE,
    AGG_SUM,
    AGG_MEAN,
    AGG_MAX,
    DROP_POST_AGG,
)


# ---------------------------------------------------------------------------
# Logger — writes to console AND fm_dad/logs/bridge.log
# ---------------------------------------------------------------------------

def _get_logger() -> logging.Logger:
    """Return the bridge logger, creating handlers once."""
    log = logging.getLogger("bridge")
    if log.handlers:
        return log

    log.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "[%(asctime)s][bridge][%(levelname)s] %(message)s", "%H:%M:%S"
    )

    # Console
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    log.addHandler(ch)

    # File
    Path(LOG_FILE).parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(LOG_FILE, mode="a")
    fh.setFormatter(fmt)
    log.addHandler(fh)

    return log


logger = _get_logger()


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _load_raw(cycle_no: int, file_key: str, folder: str) -> Optional[pd.DataFrame]:
    """
    Read one raw CSV for the given cycle and file key.

    Returns None (and logs a warning) when the file is absent.
    Logs file path and row count on success.
    """
    filename = FILE_PATTERNS[file_key].format(cycle=cycle_no)
    path = Path(folder) / filename

    if not path.exists():
        logger.warning("  [%s] NOT FOUND: %s", file_key, path.name)
        return None

    df = pd.read_csv(path)
    logger.info("  [%s] Loaded %d rows  ← %s", file_key, len(df), path.name)
    return df


def _normalise_columns(df: pd.DataFrame, file_key: str) -> pd.DataFrame:
    """
    Normalise cycle_id, node_id, and flow_id column names to standard tokens.

    Also renames unsafe column names in per_flow_metrics (e.g. 'PDR_k (%)')
    to safe Python identifiers (e.g. 'PDR_k_pct') defined in RENAME_UNSAFE.

    Args:
        df       : Raw DataFrame just read from disk.
        file_key : Key in FILE_PATTERNS identifying which source this is.

    Returns:
        DataFrame with standardised column names (mutated copy).
    """
    df = df.copy()

    # cycle_id
    raw_cycle = CYCLE_ID_COL.get(file_key, "cycle_id")
    if raw_cycle in df.columns and raw_cycle != "cycle_id":
        df.rename(columns={raw_cycle: "cycle_id"}, inplace=True)

    # node_id
    raw_node = NODE_ID_COL.get(file_key)
    if raw_node and raw_node in df.columns and raw_node != "node_id":
        df.rename(columns={raw_node: "node_id"}, inplace=True)

    # flow_id
    raw_flow = FLOW_ID_COL.get(file_key, "flow_id")
    if raw_flow in df.columns and raw_flow != "flow_id":
        df.rename(columns={raw_flow: "flow_id"}, inplace=True)

    # unsafe column names in per_flow_metrics
    df.rename(
        columns={k: v for k, v in RENAME_UNSAFE.items() if k in df.columns},
        inplace=True,
    )

    return df


def _build_agg_dict(df: pd.DataFrame, keys: list) -> dict:
    """
    Build a {column: aggregation_function} dict for pandas groupby.

    Uses the rules from config_bridge:
        AGG_SUM  columns → 'sum'
        AGG_MEAN columns → 'mean'
        AGG_MAX  columns → 'max'
        All other columns → 'first' (take any representative value)

    Columns that are groupby keys are excluded.

    Args:
        df   : DataFrame about to be grouped.
        keys : List of groupby key column names to exclude.

    Returns:
        dict mapping column name → aggregation function string.
    """
    agg = {}
    for col in df.columns:
        if col in keys:
            continue
        if col in AGG_SUM:
            agg[col] = "sum"
        elif col in AGG_MEAN:
            agg[col] = "mean"
        elif col in AGG_MAX:
            agg[col] = "max"
        else:
            agg[col] = "first"
    return agg


def _aggregate_to_node(df: pd.DataFrame, file_key: str) -> pd.DataFrame:
    """
    Collapse a per-(cycle_id, node_id) group to one row per node per cycle.

    Logs row counts before and after aggregation.
    After aggregation, drops join-artifact columns defined in DROP_POST_AGG.

    Args:
        df       : DataFrame already normalised by _normalise_columns.
        file_key : Source key used only for logging.

    Returns:
        Aggregated DataFrame with one row per (cycle_id, node_id).
    """
    keys = ["cycle_id", "node_id"]
    n_before = len(df)
    agg = _build_agg_dict(df, keys)

    if not agg:
        return df

    result = df.groupby(keys, as_index=False).agg(agg)
    logger.info(
        "  [%s] Aggregated %d → %d rows (per node)", file_key, n_before, len(result)
    )

    # Drop artifact columns that would cause naming conflicts in the final join
    to_drop = [c for c in result.columns if c in DROP_POST_AGG]
    if to_drop:
        result.drop(columns=to_drop, inplace=True)
        logger.info("  [%s] Dropped artifact columns: %s", file_key, to_drop)

    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_cycle(cycle_no: int, folder: str = RAW_CSV_FOLDER) -> pd.DataFrame:
    """
    Load all CSV files for one cycle and return a single merged DataFrame.

    Each row in the returned DataFrame represents one (cycle_id, node_id)
    observation with columns from every available source file merged in.

    Join strategy:
        • Per-(cycle, flow, node) files (off, hop_delay, planned, anomaly) are
          first aggregated to per-(cycle, node) using the rules in config_bridge.
        • per_flow_metrics has no node_id; it is broadcast to every node on each
          flow via the flow→node mapping from observed_forwarding_fractions, then
          aggregated to per-(cycle, node).
        • Scalar per-node files (trust, als, ground_truth) are used as-is.
        • All resulting DataFrames are outer-joined on (cycle_id, node_id) so no
          node is lost. Missing values are represented as NaN.

    Args:
        cycle_no : Integer cycle number (matches the _{N} suffix in filenames).
        folder   : Directory that contains the raw CSV files.

    Returns:
        pd.DataFrame: One row per (cycle_id, node_id). May contain NaN where a
        source file was absent or a node appeared in only some sources.
    """
    logger.info("=== load_cycle(%d) | folder=%s ===", cycle_no, folder)

    # --- 1. Load and normalise every available file --------------------------
    raw: dict[str, pd.DataFrame] = {}
    for key in FILE_PATTERNS:
        df = _load_raw(cycle_no, key, folder)
        if df is not None:
            raw[key] = _normalise_columns(df, key)

    if not raw:
        logger.warning("Cycle %d: no files found — returning empty DataFrame", cycle_no)
        return pd.DataFrame()

    # --- 2. Aggregate multi-row files to per-(cycle, node) -------------------
    per_node: dict[str, pd.DataFrame] = {}

    for key in ["off", "hop_delay", "planned", "anomaly"]:
        if key in raw:
            per_node[key] = _aggregate_to_node(raw[key], key)

    # --- 3. Scalar per-node files (already one row per node) -----------------
    for key in ["trust", "als", "ground_truth"]:
        if key in raw:
            per_node[key] = raw[key]
            logger.info(
                "  [%s] %d rows (already per-node, no aggregation needed)",
                key, len(raw[key]),
            )

    # --- 4. Broadcast per_flow_metrics to nodes via off's flow→node mapping --
    if "flow_metrics" in raw:
        if "off" in raw:
            # Build a minimal (cycle_id, flow_id, node_id) lookup from 'off'
            flow_node = (
                raw["off"][["cycle_id", "flow_id", "node_id"]]
                .drop_duplicates()
            )
            n_pairs = len(flow_node)

            fm = raw["flow_metrics"]  # already has cycle_id, flow_id after normalise
            broadcast = flow_node.merge(fm, on=["cycle_id", "flow_id"], how="left")
            logger.info(
                "  [flow_metrics] Broadcast to %d (cycle, flow, node) rows"
                " from %d flow-node pairs", len(broadcast), n_pairs,
            )
            per_node["flow_metrics"] = _aggregate_to_node(broadcast, "flow_metrics")
        else:
            logger.warning(
                "  [flow_metrics] Skipped: 'off' file missing — "
                "cannot build flow→node mapping"
            )

    # --- 5. Outer-join all per-node DataFrames on (cycle_id, node_id) --------
    if not per_node:
        logger.warning("Cycle %d: no per-node frames to join", cycle_no)
        return pd.DataFrame()

    logger.info("  Joining %d source(s) on (cycle_id, node_id) …", len(per_node))

    combined: Optional[pd.DataFrame] = None
    for key, df in per_node.items():
        if combined is None:
            combined = df
        else:
            before = len(combined)
            combined = combined.merge(df, on=["cycle_id", "node_id"], how="outer")
            logger.info(
                "  After joining [%s]: %d → %d rows", key, before, len(combined)
            )

    if combined is None or combined.empty:
        logger.warning("Cycle %d: combined DataFrame is empty after join", cycle_no)
        return pd.DataFrame()

    # --- 6. Diagnostics -------------------------------------------------------
    n_nodes = combined["node_id"].nunique()
    null_keys = combined[
        combined["node_id"].isna() | combined["cycle_id"].isna()
    ]
    if not null_keys.empty:
        logger.warning(
            "  Cycle %d: %d rows have null key (node_id or cycle_id) — investigate",
            cycle_no, len(null_keys),
        )
    else:
        logger.info(
            "  Cycle %d complete | %d unique nodes, %d total rows, 0 null keys",
            cycle_no, n_nodes, len(combined),
        )

    return combined


def load_all_cycles(folder: str = RAW_CSV_FOLDER) -> pd.DataFrame:
    """
    Auto-detect all cycle numbers in the raw CSV folder and load them all.

    Cycle numbers are discovered by scanning the folder for files matching
    CYCLE_DETECTION_REGEX (node_attack_ground_truth_{N}.csv) and extracting N.
    All detected cycles are loaded via load_cycle() and concatenated.

    Args:
        folder : Directory containing the raw CSV files.

    Returns:
        pd.DataFrame: Combined table for all cycles,
        one row per (cycle_id, node_id). Empty if no cycles found.
    """
    folder_path = Path(folder)
    if not folder_path.exists():
        logger.error("Raw CSV folder does not exist: %s", folder)
        return pd.DataFrame()

    pattern_re = re.compile(CYCLE_DETECTION_REGEX)
    cycles = sorted(
        int(m.group(1))
        for f in folder_path.iterdir()
        if (m := pattern_re.match(f.name))
    )

    if not cycles:
        logger.warning(
            "No cycles detected in %s  "
            "(place your CSVs there and re-run)", folder
        )
        return pd.DataFrame()

    logger.info("Detected %d cycle(s): %s", len(cycles), cycles)

    parts = []
    for c in cycles:
        df = load_cycle(c, folder)
        if not df.empty:
            parts.append(df)

    if not parts:
        logger.error("No data loaded for any cycle.")
        return pd.DataFrame()

    combined = pd.concat(parts, ignore_index=True)
    logger.info(
        "load_all_cycles done | cycles=%d, total_rows=%d, columns=%d",
        len(cycles), len(combined), len(combined.columns),
    )
    return combined
