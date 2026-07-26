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
from mpc import PRODUCTION, ChokeMPC, backoff_production_cost, infeasibility_report

DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures"
SEED = 0

C_TRUE = "#2a78d6"
C_MEAS = "#9db8d4"
C_TARGET = "#52514e"
C_LIMIT = "#e34948"
C_CHOKE = "#4a3aa7"
C_SOFT = "#eb6834"


def record_frame(rec):
    cols = {"Time_hr": rec["t"], "Choke_pct": rec["u"], "Target_bbl_hr": rec["target"]}
    for k, label in (("Q", "OilRate"), ("WHP", "WHP"), ("FLP", "FLP"), ("BHP", "BHP")):
        cols[f"{label}_meas"] = rec[f"{k}_meas"]
        if f"{k}_true" in rec:
            cols[f"{label}_true"] = rec[f"{k}_true"]
    cols["Mode"] = rec["modes"]
    return pd.DataFrame(cols)


def plot_scenario(rec, scenario, path):
    t = rec["t"]
    fig, axes = plt.subplots(6, 1, figsize=(11.5, 14), sharex=True)

    ax = axes[0]
    ax.plot(t, rec["Q_meas"], lw=0.8, color=C_MEAS, label="oil rate (measured)")
    if "Q_true" in rec:
        ax.plot(t, rec["Q_true"], lw=1.8, color=C_TRUE, label="oil rate (true)")
    ax.plot(t, rec["target"], lw=1.6, ls="--", color=C_TARGET, label="target rate")
    if not scenario.feasible:
        ax.axhline(limits.MAX_SAFE_RATE, lw=1.2, ls=":", color=C_LIMIT)
        ax.text(t[-1], limits.MAX_SAFE_RATE, f" max safe {limits.MAX_SAFE_RATE:.0f} ",
                ha="right", va="bottom", fontsize=8, color=C_LIMIT)
    ax.set_ylabel("oil rate\n[bbl/hr]")
    ax.legend(fontsize=8, loc="lower right", ncol=3)
    ax.grid(alpha=0.25)

    for ax, key, label in (
        (axes[1], "WHP", "WHP [psi]"),
        (axes[2], "FLP", "FLP [psi]"),
        (axes[3], "BHP", "BHP [psi]"),
    ):
        lim = limits.PRESSURE_LIMITS[key]
        ax.plot(t, rec[f"{key}_meas"], lw=0.8, color=C_MEAS, label="measured")
        if f"{key}_true" in rec:
            ax.plot(t, rec[f"{key}_true"], lw=1.8, color=C_TRUE, label="true")
        ax.axhline(lim, lw=1.6, ls="--", color=C_LIMIT)
        ax.text(t[0], lim, f" {key} limit {lim:g} psi ", va="bottom",
                fontsize=8, color=C_LIMIT)
        lo = min(np.min(rec[f"{key}_meas"]), lim)
        hi = max(np.max(rec[f"{key}_meas"]), lim)
        pad = 0.08 * max(hi - lo, 1.0)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_ylabel(label)
        ax.legend(fontsize=8, loc="best")
        ax.grid(alpha=0.25)

    ax = axes[4]
    ax.step(t, rec["u"], where="post", lw=1.6, color=C_CHOKE)
    ax.axhline(limits.FIRST_BINDING_CHOKE, lw=1.2, ls="--", color=C_LIMIT)
    ax.text(t[0], limits.FIRST_BINDING_CHOKE,
            f" BHP-binding choke {limits.FIRST_BINDING_CHOKE:.1f}% ",
            va="bottom", fontsize=8, color=C_LIMIT)
    ax.set_ylabel("choke\nposition [%]")
    ax.grid(alpha=0.25)

    ax = axes[5]
    soft = np.array([1.0 if m == "soft" else 0.0 for m in rec["modes"]])
    backoff = np.array([1.0 if m == "hard+backoff" else 0.0 for m in rec["modes"]])
    ax.fill_between(t, 0, backoff, step="post", color=C_TRUE, alpha=0.45,
                    label="hard + backoff satisfied")
    ax.fill_between(t, 0, soft, step="post", color=C_SOFT, alpha=0.75,
                    label="soft fallback active")
    ax.set_ylim(-0.05, 1.15)
    ax.set_yticks([0, 1])
    ax.set_ylabel("controller\nmode")
    ax.legend(fontsize=8, loc="upper right", ncol=2)
    ax.grid(alpha=0.25)

    axes[-1].set_xlabel("time [h]")
    fig.suptitle(f"Scenario {scenario.name}\n{scenario.description}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    cfg = PRODUCTION
    print("=" * 78)
    print("RUN 03 - CLOSED-LOOP SCENARIOS")
    print("=" * 78)
    print(f"controller: {cfg.name}")
    print(f"horizon {cfg.horizon}, control horizon {cfg.control_horizon}, "
          f"move weight {cfg.move_weight}")
    print(f"steady-state-feasible inputs: {cfg.ss_feasible_inputs}")
    print(f"backoff: {cfg.backoff}  applied from step {cfg.backoff_start}")
    bc = backoff_production_cost(backoff=cfg.backoff)
    print(f"backoff production cost: {bc['rate_giveaway']:.2f} bbl/hr "
          f"({bc['rate_giveaway_pct']:.2f}%, {bc['bbl_per_day']:.0f} bbl/day)")

    rows = []
    for key in ("A", "B", "C"):
        sc = SCENARIOS[key]
        rec = run_closed_loop(ChokeMPC(cfg), sc, seed=SEED)
        m = metrics(rec, sc)
        rows.append({"controller": cfg.name, **m})

        df = record_frame(rec)
        csv = DATA_DIR / f"scenario_{key}.csv"
        df.to_csv(csv, index=False)
        png = FIG_DIR / f"03_scenario_{key}.png"
        plot_scenario(rec, sc, png)

        st = m["settling_time_h"]
        st_txt = "not reached" if np.isnan(st) else f"{st:.0f} h"
        print(f"\n--- Scenario {sc.name} ---")
        print(f"  {sc.description}")
        adv = infeasibility_report(sc.targets[-1], backoff=cfg.backoff)
        if adv is not None:
            print(f"  ADVISORY: {adv['message']}")
        print(f"  settling to +/-2% of target : {st_txt}")
        print(f"  IAE                         : {m['iae']:.1f} bbl/hr*h")
        print(f"  final rate / choke          : {m['final_rate']:.2f} bbl/hr "
              f"/ {m['final_choke']:.2f} %")
        print(f"  total production            : {m['total_production_bbl']:.0f} bbl "
              f"over {sc.hours} h")
        print(f"  min BHP true / measured     : {m['min_bhp_true']:.2f} / "
              f"{m['min_bhp_measured']:.2f} psi")
        print(f"  violations true / measured  : {m['viol_true']} / "
              f"{m['viol_measured']}  of {sc.hours}")
        print(f"  soft-fallback steps         : {m['soft_steps']}")
        print(f"  wrote {csv.name}, {png.name}")

    out = pd.DataFrame(rows)
    out.to_csv(DATA_DIR / "scenario_summary.csv", index=False)
    print(f"\nwrote {DATA_DIR / 'scenario_summary.csv'}")
    print(f"figures written to {FIG_DIR}")


if __name__ == "__main__":
    main()
