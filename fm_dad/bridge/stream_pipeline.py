"""
stream_pipeline.py — Real-time streaming pipeline for FM-DAD (Part 5b).

Processes each NS-3 cycle immediately when both sentinel files arrive in --watch-dir,
instead of waiting for all cycles to finish (batch mode).

Sentinels:
    /tmp/drl_cycle_{N}_ready_ns3   — written by NS-3 after its CSVs for cycle N
    /tmp/drl_cycle_{N}_ready_mid   — written by middleware after its CSVs for cycle N
    Cycle N is processed ONLY when BOTH sentinels are present.

Outputs (written to --output-dir):
    pipeline_penalties.csv   — appended after every cycle (same format as batch)
    live_blacklist.csv       — overwritten after every cycle (current trust snapshot)
    live_trust_scores.csv    — overwritten after every cycle (simple node→trust dump)

Usage:
    # Live mode — watch /tmp/ for sentinel files from NS-3 + middleware
    python3 bridge/stream_pipeline.py \\
        --watch-dir /tmp/drl_agent \\
        --output-dir fm_dad/data \\
        --tau 0.4 \\
        --max-cycles 58 \\
        --timeout 300

    # Replay mode — test against existing CSVs without NS-3
    python3 bridge/stream_pipeline.py \\
        --replay fm_dad/data/raw_csvs \\
        --output-dir fm_dad/data/stream_test \\
        --tau 0.4

Architecture:
    - Imports bridge/join.py, features_percycle.py, features_windowed.py,
      assemble.py, trigger.py, config.py unchanged.
    - Uses episode_eval.load_frozen_agents() for model loading.
    - One rolling deque(maxlen=20) keeps cycle history for windowed features.
    - Trust state accumulates across the entire run (never resets).

IGH dormancy rule:
    W_MIN = 10  (minimum history cycles before IGH gate can fire)
    Cycles 1 to W_MIN-1: run SP, ALS, FS only.
    Cycle W_MIN onward:   run all four agents.
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
import sys
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Set

import pandas as pd

# ---------------------------------------------------------------------------
# Path setup — allow running as  python3 bridge/stream_pipeline.py
# ---------------------------------------------------------------------------
_BRIDGE_DIR = Path(__file__).parent
_FM_DAD_DIR = _BRIDGE_DIR.parent
if str(_FM_DAD_DIR) not in sys.path:
    sys.path.insert(0, str(_FM_DAD_DIR))

from bridge.join import load_cycle
from bridge.features_percycle import add_percycle_features
from bridge.features_windowed import add_windowed_features
from bridge.assemble import assemble_agent_tables
from bridge.trigger import process_cycle as trigger_process_cycle, load_agents
from bridge.config_bridge import RAW_CSV_FOLDER, CYCLE_DETECTION_REGEX
from config import AGENT_CONFIGS
from episode_eval import load_frozen_agents

from bridge.trust_client import reduce_rsu_trust, reduce_vehicle_trust


def node_type(node_id: int) -> str:
    """Return 'rsu' or 'vehicle' for a node_id.
    IMPLEMENT THIS from your NS-3 / ground-truth metadata. RSU IDs (0–99) and
    vehicle IDs (0–199) overlap, so a bare node_id cannot disambiguate on its own."""
    raise NotImplementedError("map node_id -> 'rsu' | 'vehicle'")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][stream][%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("stream")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
W_MIN = 10          # minimum cycle history before IGH gate can fire
ALL_AGENTS  = ["sp", "als", "fs", "igh"]
EARLY_AGENTS = ["sp", "als", "fs"]   # IGH dormant before W_MIN


# ---------------------------------------------------------------------------
# Global state (module-level, reset by reset_state())
# ---------------------------------------------------------------------------
# Rolling cycle buffer — maxlen=20 matches the largest W* candidate
_cycle_buffer: deque = deque(maxlen=20)

# Trust state — node_id → current trust score (lazy-init to 1.0, never resets)
_trust: Dict[int, float] = {}

# Blacklist state — node_id → first_blacklist_cycle
_blacklisted: Dict[int, int] = {}

# Sentinel tracking
_ns3_ready:  Set[int] = set()
_mid_ready:  Set[int] = set()
_processed:  Set[int] = set()

# Runtime config (set by run_streaming / run_replay)
_watch_dir:  str = "/tmp/drl_agent"
_output_dir: Path = _FM_DAD_DIR / "data"
_tau_min:    float = 0.4
_max_cycles: Optional[int] = None
_agents:     Optional[Dict] = None
_is_replay:  bool = False

# Ground truth state (loaded at startup, IGH refreshed per-cycle)
# _gt_static  : dict node_id -> {is_attacker, attack_type} for SP/ALS/FS + honest nodes
# _all_igh_nodes : union of IGH attacker node_ids across ALL cycles
_gt_static:    Dict[int, dict] = {}
_all_igh_nodes: Set[int] = set()

# Output file paths (resolved in _init_outputs)
_penalties_csv:       Optional[Path] = None
_blacklist_csv:       Optional[Path] = None
_trust_csv:           Optional[Path] = None
_trust_history_csv:   Optional[Path] = None
_penalties_written:     bool = False   # tracks whether header needs to be written
_trust_history_written: bool = False   # tracks whether header needs to be written


def reset_state() -> None:
    """Reset all mutable global state for a fresh run (used by replay mode)."""
    global _penalties_written, _trust_history_written, _is_replay
    _cycle_buffer.clear()
    _trust.clear()
    _blacklisted.clear()
    _ns3_ready.clear()
    _mid_ready.clear()
    _processed.clear()
    _gt_static.clear()
    _all_igh_nodes.clear()
    _penalties_written = False
    _trust_history_written = False
    _is_replay = False


# ---------------------------------------------------------------------------
# Ground truth helpers
# ---------------------------------------------------------------------------

def _load_static_gt(watch_dir: str) -> None:
    """
    Scan all node_attack_ground_truth_*.csv files in watch_dir at startup.
    Populates:
        _gt_static — SP/ALS/FS attackers + honest nodes (IGH nodes excluded).
    Note: _all_igh_nodes is populated incrementally by _load_cycle_igh() on
    each cycle — no full startup scan needed.
    """
    folder = Path(watch_dir)
    gt_re  = re.compile(r"node_attack_ground_truth_(\d+)\.csv$")
    files  = sorted(f for f in folder.iterdir() if gt_re.match(f.name))

    if not files:
        logger.warning("[GT] No ground truth files found in %s — GT labels unavailable.", watch_dir)
        return

    # Build static GT from first file (SP/ALS/FS don't rotate — any cycle file works).
    # IGH nodes are excluded here; they are handled per-cycle by _load_cycle_igh().
    # We identify which nodes are EVER IGH by scanning all files once, cheaply.
    all_igh: Set[int] = set()
    for f in files:
        try:
            df = pd.read_csv(f)
            df.columns = df.columns.str.strip()
            mask = (df["is_attacker"] == 1) & (df["attack_type"] == "IGH")
            all_igh.update(df.loc[mask, "node_id"].tolist())
        except Exception as exc:
            logger.warning("[GT] Could not read %s: %s", f.name, exc)

    try:
        df_ref = pd.read_csv(files[0])
        df_ref.columns = df_ref.columns.str.strip()
        non_igh = df_ref[~df_ref["node_id"].isin(all_igh)]
        for r in non_igh.to_dict("records"):
            nid = int(r["node_id"])
            _gt_static[nid] = {
                "is_attacker": int(r["is_attacker"]),
                "attack_type": str(r["attack_type"]),
            }
    except Exception as exc:
        logger.warning("[GT] Could not build static GT from %s: %s", files[0].name, exc)

    logger.info(
        "[GT] Static GT loaded: %d non-IGH nodes from %d GT files",
        len(_gt_static), len(files),
    )


def _load_cycle_igh(cycle_no: int, watch_dir: str) -> Set[int]:
    """
    Returns set of node_ids that are ACTIVE IGH attackers in cycle_no.
    Reads node_attack_ground_truth_{cycle_no}.csv from watch_dir.
    Also incrementally updates _all_igh_nodes with any new IGH nodes seen.
    Returns empty set if file not found (logs warning).
    """
    global _all_igh_nodes
    path = Path(watch_dir) / f"node_attack_ground_truth_{cycle_no}.csv"
    if not path.exists():
        logger.warning("[GT] node_attack_ground_truth_%d.csv not found", cycle_no)
        return set()
    try:
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()
        active_igh = set(
            df[(df["is_attacker"] == 1) & (df["attack_type"] == "IGH")]["node_id"].tolist()
        )
        _all_igh_nodes.update(active_igh)   # incremental accumulation
        return active_igh
    except Exception as exc:
        logger.warning("[GT] Could not read GT for cycle %d: %s", cycle_no, exc)
        return set()


# ---------------------------------------------------------------------------
# Trust update (mirrors validate_pipeline.py exactly)
# ---------------------------------------------------------------------------

def _update_trust(
    results: List[dict],
    cycle_no: int,
    cycle_gt: Optional[Dict[int, dict]] = None,
) -> None:
    """
    Apply trust deltas from agent results.

    trust NEVER resets between cycles — accumulates across the entire run.
    Blacklisted nodes continue to be processed each cycle (min-trust semantics).
    Writes non-zero delta nodes to live_trust_history.csv.

    Args:
        results   : per-node agent results from trigger_process_cycle.
        cycle_no  : current cycle number.
        cycle_gt  : per-cycle ground truth dict (node_id -> {is_attacker, attack_type}).
                    Used to label each penalized node in live_trust_history.csv.
                    If None, gt_label is written as 'UNKNOWN'.
    """
    global _trust_history_written
    history_rows = []

    for r in results:
        nid = r["node_id"]
        final_delta = r["final_delta"]

        if nid not in _trust:
            _trust[nid] = 1.0       # lazy init: first-seen node starts at 1.0

        trust_before = _trust[nid]
        trust_after  = max(0.0, trust_before - final_delta)
        _trust[nid]  = trust_after

        if trust_after < _tau_min and nid not in _blacklisted:
            _blacklisted[nid] = cycle_no
            logger.info(
                "[BLACKLIST] node=%d blacklisted at cycle=%d trust=%.4f",
                nid, cycle_no, trust_after,
            )

        if final_delta > 0:
            # Send the reduction amount to the blockchain (the source of truth).
            # The chaincode subtracts it from the current on-chain score.
            if node_type(nid) == "rsu":
                reduce_rsu_trust(nid, final_delta)
            else:
                reduce_vehicle_trust(nid, final_delta)

            # Determine ground truth label for this specific cycle
            if cycle_gt is not None:
                info = cycle_gt.get(nid, {})
                gt_label = "ATTACKER" if info.get("is_attacker", 0) else "HONEST"
            else:
                gt_label = "UNKNOWN"

            history_rows.append({
                "cycle_id":      cycle_no,
                "node_id":       nid,
                "trust_before":  trust_before,
                "delta_applied": final_delta,
                "trust_after":   trust_after,
                "blacklisted":   trust_after < _tau_min,
                "gt_label":      gt_label,
            })

    if history_rows and _trust_history_csv is not None:
        write_header = not _trust_history_written or not _trust_history_csv.exists()
        with open(_trust_history_csv, "a", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["cycle_id", "node_id", "trust_before", "delta_applied",
                            "trust_after", "blacklisted", "gt_label"],
            )
            if write_header:
                writer.writeheader()
            writer.writerows(history_rows)
        _trust_history_written = True


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def _init_outputs(output_dir: Path) -> None:
    """Create output directory and initialise file paths."""
    global _penalties_csv, _blacklist_csv, _trust_csv, _trust_history_csv
    output_dir.mkdir(parents=True, exist_ok=True)
    _penalties_csv       = output_dir / "pipeline_penalties.csv"
    _blacklist_csv       = output_dir / "live_blacklist.csv"
    _trust_csv           = output_dir / "live_trust_scores.csv"
    _trust_history_csv   = output_dir / "live_trust_history.csv"


def _append_penalties(results: List[dict], cycle_no: int) -> None:
    """
    Append per-node per-agent penalty rows to pipeline_penalties.csv.

    Columns: cycle_id, node_id, agent, gate_fired, action, delta, final_delta
    File is created with header on first write, appended thereafter.
    """
    global _penalties_written
    rows = []
    for r in results:
        nid         = r["node_id"]
        final_delta = r["final_delta"]
        for agent_name in ALL_AGENTS:
            details = r["per_agent_details"].get(agent_name, {})
            gate    = details.get("gate", "absent")
            action  = details.get("action")
            delta   = details.get("delta", 0.0)
            rows.append({
                "cycle_id":    cycle_no,
                "node_id":     nid,
                "agent":       agent_name,
                "gate_fired":  gate == "OPEN",
                "action":      action if action is not None else "",
                "delta":       delta,
                "final_delta": final_delta,
            })

    if not rows:
        return

    write_header = not _penalties_written or not _penalties_csv.exists()
    with open(_penalties_csv, "a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["cycle_id", "node_id", "agent", "gate_fired",
                        "action", "delta", "final_delta"],
        )
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
    _penalties_written = True


def _write_live_blacklist(cycle_no: int) -> None:
    """
    Overwrite live_blacklist.csv with the current trust + blacklist snapshot.

    Columns: node_id, current_trust, blacklisted, cycles_seen, first_blacklist_cycle
    first_blacklist_cycle = -1 means never blacklisted.
    """
    rows = []
    for nid, trust_score in sorted(_trust.items()):
        rows.append({
            "node_id":              nid,
            "current_trust":        round(trust_score, 6),
            "blacklisted":          trust_score < _tau_min,
            "cycles_seen":          cycle_no,
            "first_blacklist_cycle": _blacklisted.get(nid, -1),
        })
    pd.DataFrame(rows).to_csv(_blacklist_csv, index=False)


def _write_live_trust(cycle_no: int) -> None:
    """
    Overwrite live_trust_scores.csv: simple node_id → trust_score dump.
    """
    rows = [{"node_id": nid, "trust_score": round(ts, 6)}
            for nid, ts in sorted(_trust.items())]
    pd.DataFrame(rows).to_csv(_trust_csv, index=False)


# ---------------------------------------------------------------------------
# Per-cycle log summary
# ---------------------------------------------------------------------------

def _log_cycle_summary(cycle_no: int, results: List[dict], active: List[str]) -> None:
    """Print a concise per-cycle summary line to the logger."""
    penalized    = sum(1 for r in results if r["final_delta"] > 0)
    att_below    = sum(1 for nid, ts in _trust.items()
                       if ts < _tau_min and nid in _blacklisted)
    hon_below    = sum(1 for nid, ts in _trust.items()
                       if ts < _tau_min and nid not in _blacklisted)
    total_bl     = len(_blacklisted)

    if "igh" in active:
        agent_str = "SP,ALS,FS,IGH"
    else:
        cycles_left = W_MIN - cycle_no
        agent_str   = f"SP,ALS,FS (IGH dormant, need {cycles_left} more cycle{'s' if cycles_left > 1 else ''})"

    logger.info(
        "[CYCLE %2d] agents=%s | penalized=%d | blacklisted_so_far=%d | trust<tau: %d nodes",
        cycle_no, agent_str, penalized, total_bl, att_below + hon_below,
    )


# ---------------------------------------------------------------------------
# Core streaming processing function
# ---------------------------------------------------------------------------

def process_cycle_streaming(cycle_no: int) -> None:
    """
    Process one cycle immediately when both sentinels are confirmed.

    Steps:
      1. load_cycle(cycle_no, folder=_watch_dir)
      2. add_percycle_features(df)
      3. Append to _cycle_buffer
      4. df_window = pd.concat(list(_cycle_buffer))
      5. add_windowed_features(df_window)
      6. Extract current cycle rows: df_current = df_windowed[cycle_id == cycle_no]
      7. assemble_agent_tables(df_current)
      8. Build per-cycle GT: merge _gt_static with active IGH set for this cycle
      9. Determine active agents: all four active
     10. trigger_process_cycle(cycle_no, tables, agents, active=active_agents)
     11. _update_trust(results, cycle_no, cycle_gt)
     12. _append_penalties(results, cycle_no)
     13. _write_live_blacklist(cycle_no)
     14. _write_live_trust(cycle_no)
     15. _log_cycle_summary(cycle_no, results, active_agents)
     16. Delete both sentinel files after successful processing (live mode only).
    """
    logger.info("[STREAM] === Starting cycle %d ===", cycle_no)

    # Step 1: Load raw CSVs for this cycle
    try:
        df = load_cycle(cycle_no, folder=_watch_dir)
    except Exception as exc:
        logger.error("[STREAM] load_cycle(%d) failed: %s", cycle_no, exc)
        return

    if df.empty:
        logger.warning("[STREAM] cycle %d produced empty DataFrame — skipping", cycle_no)
        return

    # Step 2: Per-cycle features
    df = add_percycle_features(df)

    # Step 3: Append to rolling buffer
    _cycle_buffer.append(df)

    # Step 4-5: Windowed features need the full history window
    df_window   = pd.concat(list(_cycle_buffer), ignore_index=True)
    df_windowed = add_windowed_features(df_window)

    # Step 6: Extract only the current cycle's rows
    df_current = df_windowed[df_windowed["cycle_id"] == cycle_no].copy()
    if df_current.empty:
        logger.warning("[STREAM] cycle %d: no rows after windowing — skipping", cycle_no)
        return

    # Step 7: Build agent state tables
    tables = assemble_agent_tables(df_current)

    # Step 8: Build per-cycle ground truth.
    # Start from the static dict (SP/ALS/FS + honest nodes), then overlay
    # the active/inactive IGH status for THIS specific cycle.
    active_igh = _load_cycle_igh(cycle_no, _watch_dir)
    cycle_gt: Dict[int, dict] = dict(_gt_static)  # copy static entries
    for nid in _all_igh_nodes:
        if nid in active_igh:
            cycle_gt[nid] = {"is_attacker": 1, "attack_type": "IGH"}
        else:
            # IGH node inactive this cycle — treat as honest
            cycle_gt[nid] = {"is_attacker": 0, "attack_type": "NONE"}
    logger.info(
        "[GT] cycle %d: %d IGH active / %d total IGH nodes",
        cycle_no, len(active_igh), len(_all_igh_nodes),
    )

    # Step 9: Active agents — all four agents active
    active_agents = ALL_AGENTS

    # Step 10: Run agent inference
    results = trigger_process_cycle(cycle_no, tables, _agents, active=active_agents)

    # Step 11: Update trust state, passing per-cycle GT for history labeling
    _update_trust(results, cycle_no, cycle_gt=cycle_gt)

    # Steps 12-14: Write outputs
    _append_penalties(results, cycle_no)
    _write_live_blacklist(cycle_no)
    _write_live_trust(cycle_no)

    # Step 15: Summary log
    _log_cycle_summary(cycle_no, results, active_agents)

    # Step 16: Delete both sentinel files after successful processing (live mode only).
    if not _is_replay:
        for suffix in ["ns3", "mid"]:
            sentinel = Path(_watch_dir) / f"drl_cycle_{cycle_no}_ready_{suffix}"
            try:
                sentinel.unlink(missing_ok=True)
                logger.info("[STREAM] Deleted sentinel: %s", sentinel.name)
            except Exception as e:
                logger.warning("[STREAM] Could not delete sentinel %s: %s",
                               sentinel.name, e)

    logger.info("[STREAM] === Cycle %d complete ===", cycle_no)


# ---------------------------------------------------------------------------
# Sentinel watcher (watchdog-based, polling fallback)
# ---------------------------------------------------------------------------

def _make_handler():
    """Return a FileSystemEventHandler that triggers cycle processing on sentinels."""
    try:
        from watchdog.events import FileSystemEventHandler

        class SentinelHandler(FileSystemEventHandler):
            def on_created(self, event):
                if event.is_directory:
                    return
                filename = Path(event.src_path).name

                m = re.match(r"drl_cycle_(\d+)_ready_ns3$", filename)
                if m:
                    c = int(m.group(1))
                    if c not in _processed:
                        _ns3_ready.add(c)

                m = re.match(r"drl_cycle_(\d+)_ready_mid$", filename)
                if m:
                    c = int(m.group(1))
                    if c not in _processed:
                        _mid_ready.add(c)

                # Fire when BOTH sentinels present for the same cycle
                ready = (_ns3_ready & _mid_ready) - _processed
                for c in sorted(ready):
                    if _max_cycles is not None and c > _max_cycles:
                        continue
                    process_cycle_streaming(c)
                    _processed.add(c)

        return SentinelHandler()

    except ImportError:
        return None


def _poll_for_sentinels(watch_dir: str, timeout: Optional[int]) -> None:
    """
    Polling fallback when watchdog is unavailable.
    Scans the watch directory every second for sentinel files.
    """
    logger.info("[STREAM] watchdog not available — using polling fallback")
    watch_path  = Path(watch_dir)
    ns3_re      = re.compile(r"drl_cycle_(\d+)_ready_ns3$")
    mid_re      = re.compile(r"drl_cycle_(\d+)_ready_mid$")
    start_time  = time.time()
    last_activity = time.time()

    while True:
        for f in watch_path.iterdir():
            m = ns3_re.match(f.name)
            if m:
                c = int(m.group(1))
                if c not in _processed:
                    _ns3_ready.add(c)
            m = mid_re.match(f.name)
            if m:
                c = int(m.group(1))
                if c not in _processed:
                    _mid_ready.add(c)

        ready = (_ns3_ready & _mid_ready) - _processed
        if ready:
            last_activity = time.time()
            for c in sorted(ready):
                if _max_cycles is not None and c > _max_cycles:
                    continue
                process_cycle_streaming(c)
                _processed.add(c)

        if _max_cycles is not None and len(_processed) >= _max_cycles:
            logger.info("[STREAM] Reached max-cycles=%d — stopping.", _max_cycles)
            break

        if timeout is not None and (time.time() - last_activity) > timeout:
            logger.info("[STREAM] Timeout (%ds with no new cycle) — stopping.", timeout)
            break

        time.sleep(1.0)


# ---------------------------------------------------------------------------
# Live mode entry point
# ---------------------------------------------------------------------------

def run_streaming(
    watch_dir: str,
    output_dir: Path,
    tau: float,
    max_cycles: Optional[int],
    timeout: Optional[int],
) -> None:
    """Start the watchdog/polling sentinel watcher in live mode."""
    global _watch_dir, _output_dir, _tau_min, _max_cycles, _agents, _is_replay

    reset_state()
    _is_replay  = False
    _watch_dir  = watch_dir
    _output_dir = output_dir
    _tau_min    = tau
    _max_cycles = max_cycles
    _init_outputs(output_dir)

    logger.info("[STREAM] Loading DRL agents...")
    _agents = load_frozen_agents()
    logger.info("[STREAM] Agents loaded. Watching %s ...", watch_dir)

    logger.info("[STREAM] Loading static ground truth from %s ...", watch_dir)
    _load_static_gt(watch_dir)

    handler = _make_handler()
    if handler is not None:
        try:
            from watchdog.observers import Observer
            observer = Observer()
            observer.schedule(handler, str(watch_dir), recursive=False)
            observer.start()
            logger.info("[STREAM] watchdog observer started.")

            start_time    = time.time()
            last_activity = time.time()
            try:
                while True:
                    time.sleep(1.0)

                    if _max_cycles is not None and len(_processed) >= _max_cycles:
                        logger.info("[STREAM] Reached max-cycles=%d — stopping.", _max_cycles)
                        break

                    if timeout is not None:
                        now_processed = len(_processed)
                        if now_processed > 0:
                            last_activity = time.time()
                        if (time.time() - last_activity) > timeout:
                            logger.info("[STREAM] Timeout (%ds no new cycle) — stopping.", timeout)
                            break

            finally:
                observer.stop()
                observer.join()
        except Exception as exc:
            logger.warning("[STREAM] watchdog observer error: %s — falling back to polling", exc)
            _poll_for_sentinels(watch_dir, timeout)
    else:
        _poll_for_sentinels(watch_dir, timeout)

    logger.info("[STREAM] Done. Processed %d cycles. Blacklisted: %d nodes.",
                len(_processed), len(_blacklisted))


# ---------------------------------------------------------------------------
# Replay mode entry point
# ---------------------------------------------------------------------------

def run_replay(
    replay_dir: str,
    output_dir: Path,
    tau: float,
) -> None:
    """
    Replay mode: process existing CSVs in cycle order, simulating sentinel arrival.
    Used for correctness verification before NS-3 integration.

    Streaming output must be identical to batch output on the same data.
    """
    global _watch_dir, _output_dir, _tau_min, _max_cycles, _agents, _is_replay

    reset_state()
    _is_replay  = True
    _watch_dir  = replay_dir
    _output_dir = output_dir
    _tau_min    = tau
    _max_cycles = None
    _init_outputs(output_dir)

    logger.info("[REPLAY] Loading DRL agents...")
    _agents = load_frozen_agents()
    logger.info("[REPLAY] Agents loaded. Scanning %s ...", replay_dir)

    logger.info("[REPLAY] Loading static ground truth from %s ...", replay_dir)
    _load_static_gt(replay_dir)

    # Discover all cycles via ground-truth file pattern (same as load_all_cycles)
    folder_path = Path(replay_dir)
    pattern_re  = re.compile(CYCLE_DETECTION_REGEX)
    cycles = sorted(
        int(m.group(1))
        for f in folder_path.iterdir()
        if (m := pattern_re.match(f.name))
    )

    if not cycles:
        logger.error("[REPLAY] No cycles found in %s — nothing to do.", replay_dir)
        return

    logger.info("[REPLAY] Found %d cycles: %s", len(cycles), cycles)

    for c in cycles:
        process_cycle_streaming(c)
        _processed.add(c)

    logger.info("[REPLAY] Done. Processed %d cycles. Blacklisted: %d nodes.",
                len(_processed), len(_blacklisted))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="FM-DAD Real-Time Streaming Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Mode selection
    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--watch-dir", default="/tmp/drl_agent", metavar="DIR",
        help="Directory to watch for sentinel files (live mode).",
    )
    mode.add_argument(
        "--replay", metavar="DIR",
        help="Directory with existing CSVs to replay in order (replay mode).",
    )

    p.add_argument(
        "--output-dir", default=str(_FM_DAD_DIR / "data"), metavar="DIR",
        help="Directory for output CSVs (pipeline_penalties, live_blacklist, live_trust).",
    )
    p.add_argument(
        "--tau", type=float, default=0.4, metavar="FLOAT",
        help="Blacklisting trust threshold (tau_min).",
    )
    p.add_argument(
        "--max-cycles", type=int, default=None, metavar="N",
        help="Stop after N cycles (live mode only).",
    )
    p.add_argument(
        "--timeout", type=int, default=None, metavar="SECS",
        help="Stop if no new cycle arrives within SECS seconds (live mode only).",
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()

    if args.replay:
        run_replay(
            replay_dir  = args.replay,
            output_dir  = Path(args.output_dir),
            tau         = args.tau,
        )
    else:
        run_streaming(
            watch_dir   = args.watch_dir,
            output_dir  = Path(args.output_dir),
            tau         = args.tau,
            max_cycles  = args.max_cycles,
            timeout     = args.timeout,
        )


if __name__ == "__main__":
    main()
