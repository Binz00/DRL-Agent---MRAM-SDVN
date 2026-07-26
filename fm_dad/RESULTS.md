# RESULTS.md — Real-Time Streaming Pipeline (stream_pipeline.py)

## Delivery Summary

| File | Status | Description |
| :--- | :--- | :--- |
| `fm_dad/bridge/stream_pipeline.py` | **NEW** | Real-time streaming pipeline watcher + processor |
| `fm_dad/bridge/trigger.py` | **MODIFIED** | Added optional `active` parameter to `process_cycle()` |

---

## Architecture & Integration

- **Sentinel Watcher**: Monitors `--watch-dir` (default `/tmp`) for `drl_cycle_{N}_ready_ns3` and `drl_cycle_{N}_ready_mid`. Cycles are processed only when both sentinels are confirmed.
- **Rolling Buffer**: Uses `collections.deque(maxlen=20)` to maintain a sliding window of historical cycles for windowed features (`PDRVar`, `CoordScore`, `SpoofDev`).
- **IGH Dormancy**: For cycles $1 \dots W_{\text{MIN}}-1$ ($W_{\text{MIN}}=10$), IGH remains dormant (`active=["sp", "als", "fs"]`).
- **Live Trust Accumulation**: Node trust scores accumulate across all cycles without resetting between cycles.
- **Output Files**:
  1. `pipeline_penalties.csv` — appended per cycle
  2. `live_blacklist.csv` — overwritten snapshot of blacklisted nodes
  3. `live_trust_scores.csv` — overwritten snapshot of current trust scores

---

## Protected Files (Unchanged)
- `bridge/join.py`
- `bridge/features_percycle.py`
- `bridge/features_windowed.py`
- `bridge/assemble.py`
- `bridge/config_bridge.py`
- `rewards.py`
- `episode_eval.py`
- Gate conditions and `_check_gate` in `bridge/trigger.py`
