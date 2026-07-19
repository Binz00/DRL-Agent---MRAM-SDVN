"""
plot_validation.py — Visualise FM-DAD pipeline validation metrics.

Reads:
  - data/pipeline_mcc_validation.csv   (node-level: min trust, TP/FP/FN/TN)
  - data/pipeline_penalties.csv        (cycle-level: trust_score_after per cycle)

Four panels (matching report Eq. 4.1 and Eq. 3.80):
  1. MCC^X per attack variant  (Eq. 4.1)
  2. Confusion-matrix counts   (TP / FP / FN / TN) per attack variant
  3. Trust distribution per cycle — attackers vs honest nodes below τ_min
  4. Min-trust-reached distribution — attacker vs honest (histogram overlay)

Output: plots/validation_curves.png

Usage:
    python3 plot_validation.py
"""

import os
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
FM_DAD_DIR   = os.path.dirname(os.path.abspath(__file__))
MCC_CSV      = os.path.join(FM_DAD_DIR, "data", "pipeline_mcc_validation.csv")
PEN_CSV      = os.path.join(FM_DAD_DIR, "data", "pipeline_penalties.csv")
PLOTS_DIR    = os.path.join(FM_DAD_DIR, "plots")
OUT_PNG      = os.path.join(PLOTS_DIR, "validation_curves.png")

TAU_MIN = 0.3  # must match validate_pipeline.py

# ---------------------------------------------------------------------------
# Colour palette (matches evaluate.py training plots)
# ---------------------------------------------------------------------------
GREEN  = "#2a9d8f"
RED    = "#c00000"
BLUE   = "#5b9bd5"
AMBER  = "#f4a261"
PURPLE = "#9b72cf"
GREY   = "#aaaaaa"

BG_DARK  = "#1a1a2e"
BG_PANEL = "#16213e"
TEXT_COL = "#e0e0f0"


def _style_ax(ax, title: str, xlabel: str = "", ylabel: str = "") -> None:
    ax.set_facecolor(BG_PANEL)
    ax.set_title(title, fontsize=11, fontweight="bold", color=TEXT_COL, pad=8)
    ax.set_xlabel(xlabel, fontsize=9, color=GREY)
    ax.set_ylabel(ylabel, fontsize=9, color=GREY)
    ax.tick_params(colors=GREY, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333355")
    ax.grid(True, alpha=0.2, color=GREY, linestyle="--")


def compute_mcc(tp, fp, fn, tn):
    """Implements Equation 4.1 from the report."""
    num   = tp * tn - fp * fn
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return num / denom if denom > 0 else 0.0


def load_mcc_data(mcc_csv: str) -> pd.DataFrame:
    df = pd.read_csv(mcc_csv)
    return df


def load_penalty_data(pen_csv: str) -> pd.DataFrame:
    df = pd.read_csv(pen_csv)
    return df


def build_per_attack_stats(df_mcc: pd.DataFrame) -> pd.DataFrame:
    """Compute TP/FP/FN/TN and MCC per attack type from node-level CSV."""
    records = []
    attack_types = sorted(df_mcc.loc[df_mcc["is_attacker"] == 1, "attack_type"].unique())
    honest_mask = df_mcc["is_attacker"] == 0

    for atype in attack_types:
        is_target = (df_mcc["is_attacker"] == 1) & (df_mcc["attack_type"] == atype)
        detected  = df_mcc["detected"].astype(bool)

        # Restrict to target attackers (positives) and honest nodes (negatives).
        # Exclude attackers of other types from the confusion matrix.
        relevant = is_target | honest_mask
        rel_detected = detected[relevant]
        rel_is_target = is_target[relevant]

        tp = int(( rel_is_target &  rel_detected).sum())
        fp = int((~rel_is_target &  rel_detected).sum())
        fn = int(( rel_is_target & ~rel_detected).sum())
        tn = int((~rel_is_target & ~rel_detected).sum())
        mcc = compute_mcc(tp, fp, fn, tn)

        records.append({"attack": atype, "TP": tp, "FP": fp, "FN": fn, "TN": tn, "MCC": mcc})

    return pd.DataFrame(records)


def build_cycle_trust(df_pen: pd.DataFrame, df_mcc: pd.DataFrame) -> pd.DataFrame:
    """For each cycle, count attackers and honest nodes with trust_score_after < TAU_MIN."""
    attacker_ids = set(df_mcc.loc[df_mcc["is_attacker"] == 1, "node_id"])
    honest_ids   = set(df_mcc.loc[df_mcc["is_attacker"] == 0, "node_id"])
    total_att    = len(attacker_ids)
    total_hon    = len(honest_ids)

    records = []
    for cycle in sorted(df_pen["cycle_id"].unique()):
        cyc = df_pen[df_pen["cycle_id"] == cycle]
        att_below = int(cyc[cyc["node_id"].isin(attacker_ids)]["trust_score_after"].lt(TAU_MIN).sum())
        hon_below = int(cyc[cyc["node_id"].isin(honest_ids) ]["trust_score_after"].lt(TAU_MIN).sum())
        records.append({
            "cycle":        cycle,
            "att_below":    att_below,
            "hon_below":    hon_below,
            "att_below_pct": att_below / total_att * 100 if total_att > 0 else 0,
            "hon_below_pct": hon_below / total_hon * 100 if total_hon > 0 else 0,
        })
    return pd.DataFrame(records)


def plot_all(stats: pd.DataFrame, cycle_trust: pd.DataFrame,
             df_mcc: pd.DataFrame, out_path: str) -> None:

    attacks = stats["attack"].tolist()
    x       = np.arange(len(attacks))
    bar_w   = 0.35

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.patch.set_facecolor(BG_DARK)
    fig.suptitle(
        f"FM-DAD Pipeline — Validation Metrics  (τ_min = {TAU_MIN}, Eq. 3.80 / 4.1)",
        fontsize=14, fontweight="bold", color=TEXT_COL, y=0.98,
    )

    # ------------------------------------------------------------------ #
    # Panel 1 — MCC^X per attack variant                                  #
    # ------------------------------------------------------------------ #
    ax1 = axes[0, 0]
    _style_ax(ax1, "MCC per Attack Variant  (Eq. 4.1)", ylabel="MCC")

    colours = [GREEN if v >= 0 else RED for v in stats["MCC"]]
    bars = ax1.bar(x, stats["MCC"], color=colours, alpha=0.85, zorder=3)

    ax1.axhline(0,   color=GREY,  linewidth=1.0, linestyle="--", alpha=0.7, label="Random (0)")
    ax1.axhline(1,   color=GREEN, linewidth=0.8, linestyle=":",  alpha=0.5, label="Perfect (+1)")
    ax1.axhline(-1,  color=RED,   linewidth=0.8, linestyle=":",  alpha=0.5, label="Worst  (−1)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(attacks, color=GREY)
    ax1.set_ylim(-1.2, 1.2)

    for bar, val in zip(bars, stats["MCC"]):
        offset = 0.04 if val >= 0 else -0.08
        ax1.text(bar.get_x() + bar.get_width() / 2, val + offset,
                 f"{val:+.4f}", ha="center", va="bottom", fontsize=10,
                 fontweight="bold", color=TEXT_COL)

    macro = stats["MCC"].mean()
    ax1.axhline(macro, color=AMBER, linewidth=1.5, linestyle="-.",
                label=f"Macro avg ({macro:+.4f})")

    ax1.legend(fontsize=7, facecolor=BG_PANEL,
               labelcolor=TEXT_COL, edgecolor="#333355")

    # ------------------------------------------------------------------ #
    # Panel 2 — Confusion-matrix counts per attack variant                #
    # ------------------------------------------------------------------ #
    ax2 = axes[0, 1]
    _style_ax(ax2, "Confusion Matrix Counts per Attack Variant", ylabel="Node Count")

    w = 0.20
    offsets = [-1.5*w, -0.5*w, 0.5*w, 1.5*w]
    cols_cm = [GREEN, RED, AMBER, BLUE]
    labels  = ["TP", "FP", "FN", "TN"]

    for i, (col, label, off) in enumerate(zip(cols_cm, labels, offsets)):
        vals = stats[label].values
        ax2.bar(x + off, vals, w, color=col, alpha=0.85, label=label, zorder=3)
        for j, v in enumerate(vals):
            ax2.text(j + off, v + 0.3, str(v),
                     ha="center", va="bottom", fontsize=8, color=TEXT_COL)

    ax2.set_xticks(x)
    ax2.set_xticklabels(attacks, color=GREY)
    patches = [mpatches.Patch(color=c, label=l)
               for c, l in zip(cols_cm, labels)]
    ax2.legend(handles=patches, fontsize=8, facecolor=BG_PANEL,
               labelcolor=TEXT_COL, edgecolor="#333355")

    # ------------------------------------------------------------------ #
    # Panel 3 — Trust distribution per cycle                              #
    # ------------------------------------------------------------------ #
    ax3 = axes[1, 0]
    _style_ax(ax3, f"Nodes below τ_min={TAU_MIN} per Cycle",
              xlabel="Cycle", ylabel="% of nodes below τ_min")

    cycles = cycle_trust["cycle"].tolist()
    cx     = np.arange(len(cycles))

    ax3.plot(cx, cycle_trust["att_below_pct"], color=RED, linewidth=2.2,
             marker="o", markersize=7, zorder=4, label="Attackers below τ_min")
    ax3.fill_between(cx, cycle_trust["att_below_pct"], alpha=0.12, color=RED)

    ax3.plot(cx, cycle_trust["hon_below_pct"], color=BLUE, linewidth=2.2,
             marker="s", markersize=7, zorder=4, label="Honest nodes below τ_min")
    ax3.fill_between(cx, cycle_trust["hon_below_pct"], alpha=0.12, color=BLUE)

    ax3.set_xticks(cx)
    ax3.set_xticklabels([f"C{c}" for c in cycles], color=GREY)
    ax3.set_ylim(0, max(cycle_trust[["att_below_pct", "hon_below_pct"]].max()) * 1.25 + 5)

    for i, row in cycle_trust.iterrows():
        ci = list(cycle_trust["cycle"]).index(row["cycle"])
        ax3.text(ci, row["att_below_pct"] + 0.8, f"{row['att_below_pct']:.0f}%",
                 ha="center", fontsize=7, color=RED)
        ax3.text(ci, row["hon_below_pct"] + 0.8, f"{row['hon_below_pct']:.0f}%",
                 ha="center", fontsize=7, color=BLUE)

    ax3.legend(fontsize=8, facecolor=BG_PANEL,
               labelcolor=TEXT_COL, edgecolor="#333355")

    # ------------------------------------------------------------------ #
    # Panel 4 — Min-trust-reached histogram: attackers vs honest          #
    # ------------------------------------------------------------------ #
    ax4 = axes[1, 1]
    _style_ax(ax4, "Min Trust Reached — Attacker vs Honest",
              xlabel="Minimum Trust Score Reached", ylabel="Node Count")

    att_trust = df_mcc.loc[df_mcc["is_attacker"] == 1, "min_trust_reached"]
    hon_trust = df_mcc.loc[df_mcc["is_attacker"] == 0, "min_trust_reached"]

    bins = np.linspace(0, 1, 21)
    ax4.hist(att_trust, bins=bins, color=RED,  alpha=0.70,
             label="Attacker nodes", zorder=3)
    ax4.hist(hon_trust, bins=bins, color=BLUE, alpha=0.55,
             label="Honest nodes",   zorder=3)

    ax4.axvline(TAU_MIN, color=AMBER, linewidth=2.0, linestyle="--",
                label=f"τ_min = {TAU_MIN}  (blacklist)")

    ax4.legend(fontsize=8, facecolor=BG_PANEL,
               labelcolor=TEXT_COL, edgecolor="#333355")
    ax4.set_xlim(0, 1)

    # ------------------------------------------------------------------ #
    # Save
    # ------------------------------------------------------------------ #
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(PLOTS_DIR, exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor=BG_DARK)
    plt.close(fig)
    print(f"[SUCCESS] Validation curves saved → {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    for path, name in [(MCC_CSV, "pipeline_mcc_validation.csv"),
                       (PEN_CSV, "pipeline_penalties.csv")]:
        if not os.path.exists(path):
            print(f"[ERROR] {name} not found at {path}.")
            print("Run  python3 validate_pipeline.py  first.")
            return

    df_mcc   = load_mcc_data(MCC_CSV)
    df_pen   = load_penalty_data(PEN_CSV)

    global TAU_MIN
    if "tau_min" in df_mcc.columns:
        TAU_MIN = float(df_mcc["tau_min"].iloc[0])
        print(f"[LOAD] Loaded dynamically selected τ_min = {TAU_MIN}")

    stats    = build_per_attack_stats(df_mcc)
    cyc_trust = build_cycle_trust(df_pen, df_mcc)

    print("Per-attack MCC stats:")
    print(stats.to_string(index=False))
    print()

    plot_all(stats, cyc_trust, df_mcc, OUT_PNG)


if __name__ == "__main__":
    main()
