import copy
import sys
import time
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
from mpc import ABLATIONS, PRODUCTION, IDENTIFIED_SATURATING, ChokeMPC

DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures"

TAU_BHP = IDENTIFIED_SATURATING["BHP"][2]
HORIZONS = (1, 6, 12, 18, 24, 36, 48, 60)
SEEDS = range(5)
BASE = ABLATIONS["mpc_h12_sat"]

C_SWEEP = "#2a78d6"
C_SS = "#1baf7a"
C_LIMIT = "#e34948"
C_INK = "#0b0b0b"
C_MUTED = "#52514e"


class _Timed:
    def __init__(self, controller):
        self.controller = controller
        self.times = []

    def reset(self, *a, **kw):
        return self.controller.reset(*a, **kw)

    def compute(self, measurement, target):
        t0 = time.perf_counter()
        u = self.controller.compute(measurement, target)
        self.times.append(time.perf_counter() - t0)
        return u

    @property
    def last_info(self):
        return self.controller.last_info


def horizon_config(h, ss=False):
    cfg = copy.deepcopy(BASE)
    cfg.horizon = h
    cfg.control_horizon = 3
    cfg.ss_feasible_inputs = ss
    cfg.backoff_start = None
    cfg.__post_init__()
    return cfg


def evaluate(cfg, scenario, seeds=SEEDS):
    acc, times = [], []
    for s in seeds:
        controller = _Timed(ChokeMPC(cfg))
        rec = run_closed_loop(controller, scenario, seed=s)
        m = metrics(rec, scenario)
        m["peak_choke"] = float(np.max(rec["u"]))
        acc.append(m)
        times.extend(controller.times)
    keys = ("viol_true", "viol_measured", "peak_choke", "total_move",
            "final_rate", "min_bhp_true", "settling_time_h", "iae",
            "total_production_bbl")
    out = {}
    for k in keys:
        vals = np.array([a[k] for a in acc], dtype=float)
        out[k] = float(np.nanmean(vals)) if np.any(~np.isnan(vals)) else float("nan")
    out["ms_per_step"] = 1000.0 * float(np.median(times))
    out["n_candidates"] = int(BASE.grid_points ** 3)
    return out


def build_table():
    rows = []
    for key in ("A", "C"):
        sc = SCENARIOS[key]
        for h in HORIZONS:
            m = evaluate(horizon_config(h), sc)
            rows.append({"scenario": key, "controller": f"h={h}", "horizon": h,
                         "horizon_over_tau": h / TAU_BHP, "ss_feasible": False, **m})
        m = evaluate(PRODUCTION, sc)
        rows.append({"scenario": key, "controller": "h=12 + SS-feasible", "horizon": 12,
                     "horizon_over_tau": 12 / TAU_BHP, "ss_feasible": True, **m})
    return pd.DataFrame(rows)


def sweep_figure(df, path):
    c = df[(df["scenario"] == "C") & (~df["ss_feasible"])].sort_values("horizon")
    ss = df[(df["scenario"] == "C") & (df["ss_feasible"])].iloc[0]
    x = c["horizon_over_tau"].to_numpy()

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.3))

    ax = axes[0]
    ax.plot(x, c["viol_true"], "o-", lw=2.2, ms=6, color=C_SWEEP, zorder=3)
    ax.axhline(0, lw=1.8, ls="--", color=C_SS, zorder=2)
    ax.text(x[-1], 4, "steady-state-feasible inputs, h=12", ha="right", va="bottom",
            fontsize=9, color=C_SS, weight="bold")
    ax.axvline(1.0, lw=1.0, ls=":", color=C_MUTED, zorder=1)
    ax.text(1.06, 92, "horizon = 1 x tau", fontsize=8.5, color=C_MUTED)
    ax.annotate("103", xy=(x[0], 103), xytext=(6, 0), textcoords="offset points",
                fontsize=9, color=C_SWEEP, weight="bold", va="center")
    ax.set_ylabel("BHP violations [samples of 200]", fontsize=10)
    ax.set_title("Violations collapse at ~1 x tau", fontsize=11, weight="bold",
                 color=C_INK)

    ax = axes[1]
    ax.plot(x, c["peak_choke"], "o-", lw=2.2, ms=6, color=C_SWEEP, zorder=3)
    ax.axhline(ss["peak_choke"], lw=1.8, ls="--", color=C_SS, zorder=2)
    ax.text(x[0], ss["peak_choke"] - 0.9, "steady-state-feasible inputs, h=12",
            ha="left", va="top", fontsize=9, color=C_SS, weight="bold")
    ax.axhline(limits.FIRST_BINDING_CHOKE, lw=1.4, ls=":", color=C_LIMIT, zorder=2)
    ax.text(x[-1], limits.FIRST_BINDING_CHOKE + 0.7,
            f"BHP-binding choke {limits.FIRST_BINDING_CHOKE:.1f}%",
            ha="right", va="bottom", fontsize=9, color=C_LIMIT)
    ax.set_ylim(65.5, 101.5)
    ax.set_ylabel("peak choke reached [%]", fontsize=10)
    ax.set_title("Overshoot does not: it plateaus 9 pts high", fontsize=11,
                 weight="bold", color=C_INK)

    ax = axes[2]
    ax.plot(x, c["ms_per_step"], "o-", lw=2.2, ms=6, color=C_SWEEP, zorder=3)
    ax.set_ylabel("solve time [ms per control step]", fontsize=10)
    ax.set_title("Cost is linear, and irrelevant", fontsize=11, weight="bold",
                 color=C_INK)
    ax.text(0.03, 0.93, f"control interval is 3 600 000 ms\nworst case here is "
                        f"{c['ms_per_step'].max():.0f} ms",
            transform=ax.transAxes, fontsize=8.8, color=C_MUTED, va="top",
            linespacing=1.5,
            bbox=dict(boxstyle="round,pad=0.45", fc="#fcfcfb", ec="#dedeD9", lw=0.8))

    for ax in axes:
        ax.set_xlabel("prediction horizon / tau_BHP", fontsize=10)
        ax.grid(alpha=0.2, lw=0.7)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#cfcfcb")
        ax.tick_params(labelsize=9, color="#cfcfcb")

    fig.suptitle("Scenario C, 200 bbl/hr requested. Prediction horizon swept with the "
                 "steady-state feasibility test OFF; mean of 5 seeds.",
                 fontsize=11.5, y=1.02, color=C_INK)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 82)
    print("RUN 04b - PREDICTION HORIZON SWEEP")
    print("=" * 82)
    print(f"tau_BHP = {TAU_BHP:.3f} h, control interval {limits.TS_HOURS:g} h")
    print(f"steady-state feasibility OFF for the sweep, backoff ON, "
          f"{len(list(SEEDS))} seeds\n")

    df = build_table()
    csv = DATA_DIR / "horizon_sweep.csv"
    df.to_csv(csv, index=False)

    for key in ("A", "C"):
        sc = SCENARIOS[key]
        print(f"--- Scenario {sc.name} ---")
        hdr = (f"{'controller':>20s} {'h/tau':>6s} {'violT':>6s} {'violM':>6s} "
               f"{'peakU':>7s} {'travel':>7s} {'Qfinal':>7s} {'minBHP':>8s} "
               f"{'ms/step':>8s}")
        print(hdr)
        print("-" * len(hdr))
        for _, r in df[df["scenario"] == key].iterrows():
            print(f"{r['controller']:>20s} {r['horizon_over_tau']:6.2f} "
                  f"{r['viol_true']:6.1f} {r['viol_measured']:6.1f} "
                  f"{r['peak_choke']:7.2f} {r['total_move']:7.1f} "
                  f"{r['final_rate']:7.2f} {r['min_bhp_true']:8.2f} "
                  f"{r['ms_per_step']:8.2f}")
        print()

    c = df[(df["scenario"] == "C") & (~df["ss_feasible"])].sort_values("horizon")
    zero = c[c["viol_true"] < 0.5]
    if len(zero):
        h0 = int(zero.iloc[0]["horizon"])
        print(f"violations first reach ~zero at horizon {h0} "
              f"= {h0 / TAU_BHP:.2f} x tau_BHP")
    ss = df[(df["scenario"] == "C") & (df["ss_feasible"])].iloc[0]
    print(f"long-horizon plateau: peak choke {c['peak_choke'].iloc[-1]:.1f} %, "
          f"travel {c['total_move'].iloc[-1]:.0f} %")
    print(f"steady-state-feasible h=12: peak choke {ss['peak_choke']:.1f} %, "
          f"travel {ss['total_move']:.0f} %, "
          f"production {ss['total_production_bbl']:.0f} bbl")

    png = FIG_DIR / "04b_horizon_sweep.png"
    sweep_figure(df, png)
    print(f"\nwrote {csv}")
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
