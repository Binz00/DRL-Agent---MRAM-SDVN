import copy
import csv
import itertools
import math
import logging
from pathlib import Path
import pandas as pd
import numpy as np

from config import (
    AGENT_CONFIGS, FINETUNE_MODEL_FILES, MODEL_FILES,
    FINETUNE_HP, SHARED_HP, get_logger
)
import bridge.trigger as trigger_module
from bridge.trigger import load_agents, process_cycle
from bridge.assemble import assemble_agent_tables, AGENT_STATE_FEATURES

from bridge.join import load_all_cycles
from bridge.features_percycle import add_percycle_features
from bridge.features_windowed import add_windowed_features
from bridge.config_bridge import RAW_CSV_FOLDER

GATE_SEARCH_SPACE = {
    "sp": {
        "eta_dFF":  [0.20, 0.35, 0.50, 0.65, 0.80],
    },
    "als": {
        "eta_spoof": [0.001, 0.003, 0.005, 0.010, 0.050],
    },
    "fs": {
        "eta_dff_norm": [0.10, 0.30, 0.50, 0.70, 0.90],
    },
    "igh": {
        "eta_pdrvar": [0.01, 0.03, 0.05, 0.10, 0.15],
        "eta_coord":  [0.30, 0.40, 0.50, 0.60, 0.70],
        "eta_rho":    [0.30, 0.40, 0.50],
    },
}

def compute_mcc(tp, fp, fn, tn):
    numerator   = tp * tn - fp * fn
    denominator = math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    return numerator / denominator if denominator > 0 else 0.0

def load_best_agents(logger):
    """Load finetuned models if available, else fall back to synthetic."""
    from agent import DQNAgent
    agents = {}
    for name in ["sp", "als", "fs", "igh"]:
        cfg  = AGENT_CONFIGS[name]
        hp   = FINETUNE_HP
        agent = DQNAgent(cfg, hp, device="cpu")
        ft_path  = Path(FINETUNE_MODEL_FILES[name])
        syn_path = Path(MODEL_FILES[name])
        if ft_path.exists():
            agent.load(str(ft_path))
            logger.info("[LOAD] %s — finetuned model", name.upper())
        elif syn_path.exists():
            agent.load(str(syn_path))
            logger.info("[LOAD] %s — synthetic model (finetuned not found)", name.upper())
        else:
            raise FileNotFoundError(f"No model found for {name}")
        agent.main_net.eval()
        agent.epsilon = 0.0
        agents[name] = agent
    return agents

def evaluate_thresholds(agents, tables, gt, tau_min=0.30) -> dict:
    """
    Run process_cycle for all cycles with current GATE_CONDITIONS,
    accumulate trust reductions, compute MCC^X per attack type.
    Returns dict: {attack_type: mcc, "macro": macro_mcc, "fp": fp_count}
    """
    # Accumulate trust reductions per node across all cycles
    trust = {nid: 1.0 for nid in gt["node_id"].unique()}

    all_cycle_ids = sorted(
        set().union(*[set(df["cycle_id"].unique()) for df in tables.values()])
    )
    for cycle_id in all_cycle_ids:
        results = process_cycle(cycle_id, tables, agents)
        for r in results:
            nid = r["node_id"]
            if nid in trust:
                trust[nid] = max(0.0, trust[nid] - r["final_delta"])

    # Build detected set
    detected = {nid for nid, tau in trust.items() if tau < tau_min}

    # Compute MCC^X per attack type
    honest_mask = gt["is_attacker"] == 0
    attack_types = sorted(gt.loc[gt["is_attacker"]==1, "attack_type"].unique())
    mcc_results = {}
    for atype in attack_types:
        is_target  = (gt["is_attacker"]==1) & (gt["attack_type"]==atype)
        relevant   = is_target | honest_mask
        rel_gt     = gt[relevant].copy()
        rel_gt["detected_flag"] = rel_gt["node_id"].isin(detected)
        rel_tgt    = is_target[relevant]
        rel_det    = rel_gt["detected_flag"]
        tp = int(( rel_tgt &  rel_det).sum())
        fp = int((~rel_tgt &  rel_det).sum())
        fn = int(( rel_tgt & ~rel_det).sum())
        tn = int((~rel_tgt & ~rel_det).sum())
        mcc_results[atype] = {"tp":tp,"fp":fp,"fn":fn,"tn":tn,
                               "mcc": compute_mcc(tp,fp,fn,tn)}
    macro = sum(v["mcc"] for v in mcc_results.values()) / len(mcc_results)
    fp_total = list(mcc_results.values())[0]["fp"]  # shared across agents
    return {"per_attack": mcc_results, "macro": macro, "fp": fp_total}

def _patch_gate_conditions(agent_name: str, threshold_overrides: dict):
    """
    Patch trigger.GATE_CONDITIONS for one agent with new threshold values.
    Returns the original conditions for that agent so they can be restored.
    """
    import copy
    original = copy.deepcopy(trigger_module.GATE_CONDITIONS[agent_name])

    def patch_list(item_list):
        new_list = []
        for item in item_list:
            if isinstance(item, list):
                new_list.append(patch_list(item))
            else:
                feat, op, thresh = item
                override_key = None
                for k in threshold_overrides:
                    # Match by threshold name pattern: eta_dFF → dFF, eta_spoof → SpoofDev etc.
                    if feat.lower() in k.lower() or k.lower().replace("eta_","") in feat.lower():
                        override_key = k
                        break
                new_thresh = threshold_overrides.get(override_key, thresh) if override_key else thresh
                new_list.append((feat, op, new_thresh))
        return new_list

    trigger_module.GATE_CONDITIONS[agent_name] = patch_list(original)
    return original

def _restore_gate_conditions(agent_name: str, original):
    trigger_module.GATE_CONDITIONS[agent_name] = original

def run_grid_search(agents, tables, gt, tau_min=0.30, logger=None):
    """
    Search each agent's gate thresholds independently.
    For each agent, sweep its thresholds while others stay at current values.
    Select best per-agent threshold by highest per-agent MCC^X.
    """
    results_log = []   # all candidates — written to CSV for the report
    best_per_agent = {}

    for agent_name, param_grid in GATE_SEARCH_SPACE.items():
        logger.info("=== Grid search: %s ===", agent_name.upper())
        param_names  = list(param_grid.keys())
        param_values = list(param_grid.values())
        best_mcc  = -float("inf")
        best_cand = None

        for combo in itertools.product(*param_values):
            overrides = dict(zip(param_names, combo))
            original  = _patch_gate_conditions(agent_name, overrides)
            try:
                eval_result = evaluate_thresholds(agents, tables, gt, tau_min)
            finally:
                _restore_gate_conditions(agent_name, original)

            # Score by this agent's own MCC^X
            attack_map = {"sp":"SP","als":"ALS","igh":"IGH","fs":"FS"}
            atype = attack_map[agent_name]
            agent_mcc = eval_result["per_attack"].get(
                atype, {"mcc": 0.0}
            )["mcc"]

            row = {
                "agent": agent_name,
                **overrides,
                "mcc_agent":  agent_mcc,
                "macro_mcc":  eval_result["macro"],
                "fp":         eval_result["fp"],
            }
            results_log.append(row)

            logger.info(
                "[%s] %s → MCC^%s=%.4f, macro=%.4f, FP=%d",
                agent_name.upper(), overrides, atype.upper(),
                agent_mcc, eval_result["macro"], eval_result["fp"]
            )

            if agent_mcc > best_mcc:
                best_mcc  = agent_mcc
                best_cand = overrides.copy()

        best_per_agent[agent_name] = {"thresholds": best_cand, "mcc": best_mcc}
        logger.info(
            "[%s] BEST: %s → MCC^%s = %.4f",
            agent_name.upper(), best_cand, atype.upper(), best_mcc
        )

    return best_per_agent, results_log

def write_outputs(best_per_agent, results_log, logger):
    # 1. Full grid search log CSV — all candidates, for the report
    log_path = Path("data/grid_search_gate_results.csv")
    if results_log:
        # Get all possible keys across all rows
        keys = []
        for row in results_log:
            for k in row.keys():
                if k not in keys:
                    keys.append(k)
        
        with open(log_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(results_log)
        logger.info("Grid search log → %s", log_path)

    # 2. Best thresholds summary — print to console in config.py-ready format
    print("\n" + "="*60)
    print("GRID SEARCH RESULTS — paste into config.py AGENT_CONFIGS")
    print("="*60)
    for agent_name, result in best_per_agent.items():
        print(f"\n  # {agent_name.upper()} best thresholds "
              f"(MCC^{agent_name.upper()} = {result['mcc']:+.4f})")
        for k, v in result["thresholds"].items():
            print(f"  \"{k}\": {v},")
    print("="*60)

if __name__ == "__main__":
    logger = get_logger("grid_search_gate")
    logger.setLevel(logging.INFO)
    
    # Optional: disable noisy pipeline logs during grid search
    logging.getLogger("pipeline").setLevel(logging.WARNING)
    logging.getLogger("bridge").setLevel(logging.WARNING)

    logger.info("Loading ground truth...")
    GT_FILE = Path("data/raw_csvs/node_attack_ground_truth_1.csv")
    gt = pd.read_csv(GT_FILE)
    gt.columns = gt.columns.str.strip()

    logger.info("Loading bridge data...")
    df_joined   = load_all_cycles(RAW_CSV_FOLDER)
    df_features = add_percycle_features(df_joined)
    df_windowed = add_windowed_features(df_features)
    BASE_TABLES = assemble_agent_tables(df_windowed)
    
    logger.info("Loading agents...")
    agents = load_best_agents(logger)
    
    logger.info("Running gate threshold grid search...")
    best, log = run_grid_search(agents, BASE_TABLES, gt, tau_min=0.30, logger=logger)
    write_outputs(best, log, logger)
