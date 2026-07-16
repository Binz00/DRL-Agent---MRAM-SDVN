"""
server.py — FastAPI HTTP server for real-time node evaluation (Part 5).

Exposes:
  POST /evaluate - Runs gate + agent inference + trust update for a single node.
  GET /health    - Health check.

Logs every request and response to fm_dad/logs/pipeline.log and console.
"""

import logging
from pathlib import Path
from typing import Dict, List, Any
import sys

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Add parent directory to path to load modules correctly
_BRIDGE_DIR = Path(__file__).parent
_FM_DAD_DIR = _BRIDGE_DIR.parent
if str(_FM_DAD_DIR) not in sys.path:
    sys.path.insert(0, str(_FM_DAD_DIR))

from bridge.trigger import load_agents, process_node
from bridge.trust_client import apply_trust_delta

# ---------------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------------
log_path = _FM_DAD_DIR / "logs" / "pipeline.log"
log_path.parent.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("pipeline")
logger.setLevel(logging.INFO)

# Avoid adding duplicate handlers if reloaded
if not logger.handlers:
    file_handler = logging.FileHandler(str(log_path))
    file_handler.setFormatter(logging.Formatter("[%(asctime)s]%(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("[%(asctime)s]%(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(console_handler)

# Load the agents once at server startup
logger.info("[LOAD] Pre-loading DRL agents for real-time inference...")
AGENTS = load_agents()
logger.info("[LOAD] All agents loaded successfully.")

app = FastAPI(title="FM-DAD Real-time Evaluation Bridge")


# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------

class EvaluateRequest(BaseModel):
    cycle_id: int
    node_id: int
    features: Dict[str, float]  # raw or pre-computed feature name -> value
    is_rsu: bool = False
    current_trust: float = 1.0


class EvaluateResponse(BaseModel):
    node_id: int
    cycle_id: int
    gate_fired: List[str]
    actions: Dict[str, int]
    final_delta: float
    new_trust: float


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/evaluate", response_model=EvaluateResponse)
def evaluate(req: EvaluateRequest):
    """
    Run evaluation for a single node.
    
    Accepts cycle_id, node_id, features dictionary, is_rsu and current_trust.
    """
    logger.info(
        "[HTTP] Received POST /evaluate | node_id=%d, cycle_id=%d",
        req.node_id, req.cycle_id
    )
    
    # Reconstruct states_by_agent and feature_dicts_by_agent from the input features
    from bridge.assemble import AGENT_STATE_FEATURES
    import numpy as np

    states_by_agent = {}
    feature_dicts_by_agent = {}

    for name in ["sp", "als", "fs", "igh"]:
        state_feats = AGENT_STATE_FEATURES[name]
        
        # Check if all required features for this agent are provided
        missing = [f for f in state_feats if f not in req.features]
        if missing:
            # If features are missing, we cannot construct the state vector
            states_by_agent[name] = None
            feature_dicts_by_agent[name] = None
            continue
            
        # Build state vector and feature dictionary
        vec = np.array([req.features[f] for f in state_feats], dtype=np.float32)
        states_by_agent[name] = vec
        feature_dicts_by_agent[name] = {f: req.features[f] for f in state_feats}

    # Run the core evaluation pipeline
    try:
        res = process_node(
            node_id=req.node_id,
            cycle_id=req.cycle_id,
            states_by_agent=states_by_agent,
            feature_dicts_by_agent=feature_dicts_by_agent,
            agents=AGENTS
        )
    except Exception as e:
        logger.error("[HTTP] Error processing node: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

    # Apply trust penalty
    final_delta = res["final_delta"]
    new_trust = apply_trust_delta(
        node_id=req.node_id,
        delta=final_delta,
        is_rsu=req.is_rsu,
        current_trust=req.current_trust
    )

    response_data = EvaluateResponse(
        node_id=req.node_id,
        cycle_id=req.cycle_id,
        gate_fired=res["gates_fired"],
        actions=res["actions"],
        final_delta=final_delta,
        new_trust=new_trust
    )

    logger.info(
        "[HTTP] Responding to POST /evaluate | node_id=%d, final_delta=%.3f, new_trust=%.4f",
        req.node_id, final_delta, new_trust
    )
    
    return response_data
