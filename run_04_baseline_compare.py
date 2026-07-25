import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import limits
from evaluate import SCENARIOS, metrics, run_closed_loop
from mpc import ABLATIONS, PRODUCTION, ChokeMPC, backoff_production_cost
from pid import PIController

DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures"
SEEDS = range(5)
ZERO_BACKOFF = {"WHP": 0.0, "FLP": 0.0, "BHP": 0.0}

C_MPC = "#2a78d6"
C_NAIVE = "#eb6834"
C_PI = "#1baf7a"
C_LIMIT = "#e34948"
C_INK = "#0b0b0b"
C_MUTED = "#52514e"

SHORT = {
    "mpc_h12_sat": "MPC h=12, saturating",
    "mpc_h1_sat": "one-step h=1, saturating",
    "mpc_h12_lin": "MPC h=12, linear",
    "mpc_h1_lin": "naive h=1, linear",
}


def tweak(cfg, **kw):
    c = copy.deepcopy(cfg)
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def aggregate(make, scenario, seeds=SEEDS):
    acc = [metrics(run_closed_loop(make(), scenario, seed=s), scenario) for s in seeds]
    keys = [
        "settling_time_h", "iae", "total_production_bbl", "final_rate",
        "final_choke", "total_move", "soft_steps", "viol_true", "viol_measured",
        "min_bhp_true", "min_bhp_measured", "viol_true_maxdepth",
    ]
    out = {}
    for k in keys:
        vals = np.array([a[k] for a in acc], dtype=float)
        out[k] = float(np.nanmean(vals)) if np.any(~np.isnan(vals)) else float("nan")
    out["settled_frac"] = float(
        np.mean([0.0 if np.isnan(a["settling_time_h"]) else 1.0 for a in acc])
    )
    return out


def build_table():
    rows = []
    for key in ("A", "B", "C"):
        sc = SCENARIOS[key]
        for cell, cfg in ABLATIONS.items():
            for bo_label, bo in (("default", cfg.backoff), ("zero", ZERO_BACKOFF)):
                for ss in (False, True):
                    c = tweak(cfg, backoff=dict(bo), ss_feasible_inputs=ss)
                    m = aggregate(lambda c=c: ChokeMPC(c), sc)
                    rows.append({
                        "scenario": key, "controller": SHORT[cell],
                        "backoff": bo_label, "ss_feasible": ss,
                        "horizon": cfg.horizon,
                        "model": "saturating" if "sat" in cell else "linear", **m,
                    })
        m = aggregate(PIController, sc)
        rows.append({
            "scenario": key, "controller": "PI on oil rate", "backoff": "n/a",
            "ss_feasible": False, "horizon": 0, "model": "none", **m,
        })
    return pd.DataFrame(rows)


def fmt(v, spec, nan="n/a"):
    return nan if v is None or (isinstance(v, float) and np.isnan(v)) else format(v, spec)


def table_png(df, path):
    header = ["scenario", "controller", "backoff", "SS-feas", "settle\n[h]",
              "IAE\n[bbl/hr*h]", "prod\n[bbl]", "viol\nTRUE", "viol\nMEAS",
              "min BHP\ntrue [psi]"]
    sub = df[(df["backoff"] != "zero")].reset_index(drop=True)
    cells, colors = [], []
    for _, r in sub.iterrows():
        settle = r["settling_time_h"]
        settle_txt = "not reached" if np.isnan(settle) else f"{settle:.0f}"
        vt = r["viol_true"]
        ss_txt = "-" if r["controller"].startswith("PI") else (
            "on" if r["ss_feasible"] else "off")
        cells.append([
            r["scenario"], r["controller"], r["backoff"], ss_txt, settle_txt,
            f"{r['iae']:.0f}", f"{r['total_production_bbl']:.0f}",
            f"{vt:.1f}", f"{r['viol_measured']:.1f}",
            f"{r['min_bhp_true']:.1f}",
        ])
        if vt < 0.5:
            tone = "#eaf3ea"
        elif vt < 10:
            tone = "#fdf3e3"
        else:
            tone = "#fbe9e9"
        colors.append([tone] * len(header))

    fig, ax = plt.subplots(figsize=(14.5, 0.36 * len(cells) + 1.6))
    ax.axis("off")
    tbl = ax.table(cellText=cells, colLabels=header, cellColours=colors,
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1, 1.45)
    for j in range(len(header)):
        c = tbl[0, j]
        c.set_facecolor("#e8e8e6")
        c.set_text_props(weight="bold", color=C_INK)
    for i in range(len(cells) + 1):
        for j in range(len(header)):
            tbl[i, j].set_edgecolor("#cfcfcb")
    ax.set_title(
        f"MPC vs one-step vs PI  -  mean of {len(list(SEEDS))} seeds per cell "
        "(default backoff; zero-backoff rows in the CSV)\n"
        "green: no true violations   amber: <10   red: >=10",
        fontsize=11, pad=18,
    )
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def headline_figure(path):
    sc = SCENARIOS["C"]
    series = [
        ("MPC, horizon 12", lambda: ChokeMPC(ABLATIONS["mpc_h12_sat"]),
         C_MPC, 2.3, 0.60, 26),
        ("One-step, horizon 1", lambda: ChokeMPC(ABLATIONS["mpc_h1_lin"]),
         C_NAIVE, 2.0, 0.36, -30),
        ("PI on oil rate", PIController, C_PI, 2.0, 0.45, -26),
    ]
    fig, ax = plt.subplots(figsize=(7.8, 4.6))

    ax.axhspan(2620, limits.BHP_MIN, color=C_LIMIT, alpha=0.055, zorder=0)
    ax.axhline(limits.BHP_MIN, lw=1.7, ls="--", color=C_LIMIT, zorder=2)
    ax.text(3, limits.BHP_MIN - 9, "BHP limit 2850 psi  -  below this the well is damaged",
            ha="left", va="top", fontsize=9, color=C_LIMIT, weight="bold")

    stats = []
    for label, make, color, lw, xf, dy in series:
        rec = run_closed_loop(make(), sc, seed=0)
        y = rec["BHP_true"]
        ax.plot(rec["t"], y, lw=lw, color=color, label=label, zorder=3,
                solid_capstyle="round")
        i = int(xf * (sc.hours - 1))
        ax.annotate(label, xy=(rec["t"][i], y[i]), xytext=(0, dy),
                    textcoords="offset points", color=color, fontsize=10,
                    weight="bold", ha="center", zorder=4)
        stats.append((label, color, int(np.sum(y < limits.BHP_MIN)),
                      float(np.mean(rec["Q_true"][-30:]))))

    lines = [f"{lab}:  {n} of {sc.hours} h below limit,  {q:.0f} bbl/hr"
             for lab, col, n, q in stats]
    ax.text(0.985, 0.965, "\n".join(lines), transform=ax.transAxes, ha="right",
            va="top", fontsize=8.6, color=C_MUTED, linespacing=1.6,
            bbox=dict(boxstyle="round,pad=0.5", fc="#fcfcfb", ec="#dedeD9", lw=0.8))

    ax.set_xlim(0, sc.hours)
    ax.set_ylim(2620, 3230)
    ax.set_xlabel("time [h]", fontsize=10)
    ax.set_ylabel("bottomhole pressure [psi]", fontsize=10)
    ax.set_title("Operator requests 200 bbl/hr. The well can safely deliver 163.",
                 fontsize=12.5, weight="bold", pad=10, color=C_INK)
    ax.grid(alpha=0.2, lw=0.7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#cfcfcb")
    ax.tick_params(labelsize=9, color="#cfcfcb")
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 78)
    print("RUN 04 - BASELINE COMPARISON")
    print("=" * 78)
    bc = backoff_production_cost()
    print(f"backoff cost: {bc['rate_giveaway']:.2f} bbl/hr "
          f"({bc['rate_giveaway_pct']:.2f}%, {bc['bbl_per_day']:.0f} bbl/day)")
    print(f"seeds per cell: {len(list(SEEDS))}\n")

    df = build_table()
    csv = DATA_DIR / "baseline_comparison.csv"
    df.to_csv(csv, index=False)

    show = ["scenario", "controller", "backoff", "ss_feasible", "settling_time_h",
            "iae", "total_production_bbl", "viol_true", "viol_measured",
            "min_bhp_true", "total_move"]
    with pd.option_context("display.width", 250, "display.max_columns", None):
        print(df[df["backoff"] != "zero"][show].to_string(
            index=False, float_format=lambda v: f"{v:9.1f}"))

    prod = aggregate(lambda: ChokeMPC(PRODUCTION), SCENARIOS["C"])
    print(f"\nshipped controller ({PRODUCTION.name}) on scenario C: "
          f"prod={prod['total_production_bbl']:.0f} bbl, "
          f"violTRUE={prod['viol_true']:.1f}, "
          f"minBHPtrue={prod['min_bhp_true']:.2f}, "
          f"choke travel={prod['total_move']:.1f} %")

    table_png(df, FIG_DIR / "04_metrics_table.png")
    headline_figure(FIG_DIR / "04_headline_scenarioC_bhp.png")

    print(f"\nwrote {csv}")
    print(f"wrote {FIG_DIR / '04_metrics_table.png'}")
    print(f"wrote {FIG_DIR / '04_headline_scenarioC_bhp.png'}")

    c = df[df["scenario"] == "C"]
    for ss in (False, True):
        z = c[(c["backoff"] == "zero") & (c["ss_feasible"] == ss)]
        print(f"\nfairness check - Scenario C, zero backoff everywhere, "
              f"SS-feasible={'on' if ss else 'off'}:")
        for _, r in z.iterrows():
            print(f"  {r['controller']:26s} Q={r['final_rate']:7.2f} "
                  f"minBHPtrue={r['min_bhp_true']:8.2f} "
                  f"violTRUE={r['viol_true']:6.1f} "
                  f"violMEAS={r['viol_measured']:6.1f}")


if __name__ == "__main__":
    main()
