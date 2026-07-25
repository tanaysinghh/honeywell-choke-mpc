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
from evaluate import OUTPUT_KEYS, SCENARIOS, metrics, run_closed_loop
from mpc import PRODUCTION, ChokeMPC

DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures"

N_RUNS = 100
MISMATCH = 0.20
MISMATCH_LEVELS = (0.0, 0.05, 0.10, 0.20, 0.30, 0.40)
MISMATCH_SEED = 20260725

C_DIST = "#2a78d6"
C_SS = "#1baf7a"
C_LIMIT = "#e34948"
C_INK = "#0b0b0b"
C_MUTED = "#52514e"


def draw_mismatch(rng, magnitude):
    if magnitude <= 0.0:
        return {}
    lo, hi = 1.0 - magnitude, 1.0 + magnitude
    return {
        "gain_scale": {k: float(rng.uniform(lo, hi)) for k in OUTPUT_KEYS},
        "tau_scale": {k: float(rng.uniform(lo, hi)) for k in OUTPUT_KEYS},
    }


def monte_carlo(scenario, magnitude, n_runs=N_RUNS, seed=MISMATCH_SEED):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_runs):
        kw = draw_mismatch(rng, magnitude)
        rec = run_closed_loop(ChokeMPC(PRODUCTION), scenario, seed=i, plant_kwargs=kw)
        m = metrics(rec, scenario)
        rows.append({
            "run": i,
            "scenario": scenario.key,
            "mismatch": magnitude,
            "gain_bhp": kw.get("gain_scale", {}).get("BHP", 1.0),
            "tau_bhp": kw.get("tau_scale", {}).get("BHP", 1.0),
            "viol_true": m["viol_true"],
            "viol_measured": m["viol_measured"],
            "depth_true": m["viol_true_maxdepth"],
            "depth_measured": m["viol_measured_maxdepth"],
            "min_bhp_true": m["min_bhp_true"],
            "production_bbl": m["total_production_bbl"],
            "final_rate": m["final_rate"],
            "final_choke": m["final_choke"],
            "settling_time_h": m["settling_time_h"],
            "soft_steps": m["soft_steps"],
            "hours": scenario.hours,
        })
    return pd.DataFrame(rows)


def summarise(df):
    n = len(df)
    hours = float(df["hours"].iloc[0])
    return {
        "runs": n,
        "run_violation_rate": float((df["viol_true"] > 0).mean()),
        "run_violation_rate_meas": float((df["viol_measured"] > 0).mean()),
        "sample_violation_rate": float(df["viol_true"].sum() / (n * hours)),
        "sample_violation_rate_meas": float(df["viol_measured"].sum() / (n * hours)),
        "worst_depth_psi": float(df["depth_true"].max()),
        "p95_depth_psi": float(np.percentile(df["depth_true"], 95)),
        "worst_min_bhp": float(df["min_bhp_true"].min()),
        "production_mean": float(df["production_bbl"].mean()),
        "production_std": float(df["production_bbl"].std(ddof=1)),
        "production_p05": float(np.percentile(df["production_bbl"], 5)),
        "production_p95": float(np.percentile(df["production_bbl"], 95)),
        "production_min": float(df["production_bbl"].min()),
        "soft_steps_max": float(df["soft_steps"].max()),
    }


def distribution_figure(df, path):
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.3))

    ax = axes[0]
    v = df["min_bhp_true"].to_numpy()
    ax.hist(v, bins=24, color=C_DIST, alpha=0.85, edgecolor="white", lw=0.6)
    ax.axvline(limits.BHP_MIN, lw=2.0, ls="--", color=C_LIMIT, zorder=3)
    ax.text(limits.BHP_MIN + 0.6, ax.get_ylim()[1] * 0.94, "  BHP limit\n  2850 psi",
            color=C_LIMIT, fontsize=9, weight="bold", va="top")
    ax.set_xlabel("worst true BHP reached in the run [psi]", fontsize=10)
    ax.set_ylabel(f"runs (of {len(df)})", fontsize=10)
    ax.set_title(f"Every run clears the limit  (closest: "
                 f"{v.min() - limits.BHP_MIN:+.1f} psi)",
                 fontsize=11, weight="bold", color=C_INK)

    ax = axes[1]
    p = df["production_bbl"].to_numpy()
    ax.hist(p, bins=24, color=C_SS, alpha=0.85, edgecolor="white", lw=0.6)
    ax.axvline(p.mean(), lw=2.0, ls="--", color=C_MUTED, zorder=3)
    ax.text(p.mean(), ax.get_ylim()[1] * 0.94,
            f"  mean {p.mean():,.0f} bbl\n  sd {p.std(ddof=1):,.0f}",
            color=C_MUTED, fontsize=9, weight="bold", va="top")
    ax.set_xlabel(f"total production over {int(df['hours'].iloc[0])} h [bbl]",
                  fontsize=10)
    ax.set_ylabel(f"runs (of {len(df)})", fontsize=10)
    ax.set_title("Production spread is the cost of not knowing the well",
                 fontsize=11, weight="bold", color=C_INK)

    for ax in axes:
        ax.grid(alpha=0.2, lw=0.7, axis="y")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#cfcfcb")
        ax.tick_params(labelsize=9, color="#cfcfcb")

    fig.suptitle(f"Scenario C, {len(df)} Monte Carlo runs: measurement noise plus "
                 f"independent +/-{MISMATCH:.0%} gain and time-constant mismatch "
                 f"on every output",
                 fontsize=11.5, y=1.02, color=C_INK)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def envelope_figure(env, path):
    x = 100.0 * env["mismatch"].to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.3))

    ax = axes[0]
    ax.plot(x, 100.0 * env["run_violation_rate"], "o-", lw=2.2, ms=6, color=C_DIST,
            zorder=3, label="true")
    ax.plot(x, 100.0 * env["run_violation_rate_meas"], "o--", lw=1.6, ms=5,
            color=C_MUTED, zorder=2, label="measured")
    ax.set_ylabel("runs with any BHP violation [%]", fontsize=10)
    ax.set_title("Robustness envelope", fontsize=11, weight="bold", color=C_INK)
    ax.legend(fontsize=9, frameon=False, loc="upper left")

    ax = axes[1]
    ax.plot(x, env["worst_depth_psi"], "o-", lw=2.2, ms=6, color=C_DIST, zorder=3,
            label="worst of 100 runs")
    ax.plot(x, env["p95_depth_psi"], "o--", lw=1.6, ms=5, color=C_MUTED, zorder=2,
            label="95th percentile")
    ax.axhline(PRODUCTION.backoff["BHP"], lw=1.6, ls=":", color=C_LIMIT, zorder=2)
    ax.text(x[0], PRODUCTION.backoff["BHP"] + 0.5,
            f"backoff margin {PRODUCTION.backoff['BHP']:.0f} psi  -  the well is still "
            f"inside the margin the controller reserved",
            ha="left", va="bottom", fontsize=9, color=C_LIMIT)
    ax.set_ylim(-0.6, 18.5)
    ax.set_ylabel("BHP excursion depth below limit [psi]", fontsize=10)
    ax.set_title("Excursion depth when it does happen", fontsize=11, weight="bold",
                 color=C_INK)
    ax.legend(fontsize=9, frameon=False, loc="center left")

    for ax in axes:
        ax.set_xlabel("plant-model mismatch magnitude [+/- %, gain and tau]",
                      fontsize=10)
        ax.grid(alpha=0.2, lw=0.7)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#cfcfcb")
        ax.tick_params(labelsize=9, color="#cfcfcb")

    fig.suptitle(f"Scenario C, {N_RUNS} runs per mismatch level, shipped controller "
                 f"({PRODUCTION.name})", fontsize=11.5, y=1.02, color=C_INK)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 82)
    print("RUN 05 - MONTE CARLO ROBUSTNESS")
    print("=" * 82)
    print(f"controller: {PRODUCTION.name}")
    print(f"{N_RUNS} runs per case, independent uniform gain and tau mismatch "
          f"per output")
    print(f"measurement noise and colored process drift on in every run\n")

    runs = []
    print(f"--- headline: +/-{MISMATCH:.0%} mismatch, all three scenarios ---")
    hdr = (f"{'scen':>5s} {'runs w/ viol':>13s} {'sample rate':>12s} "
           f"{'worst depth':>12s} {'worst minBHP':>13s} {'prod mean':>10s} "
           f"{'prod sd':>8s} {'prod p05':>9s}")
    print(hdr)
    print("-" * len(hdr))
    for key in ("A", "B", "C"):
        df = monte_carlo(SCENARIOS[key], MISMATCH)
        df["case"] = "headline"
        runs.append(df)
        s = summarise(df)
        print(f"{key:>5s} {s['run_violation_rate']:12.0%} "
              f"{s['sample_violation_rate']:11.2%} "
              f"{s['worst_depth_psi']:11.2f}  {s['worst_min_bhp']:13.2f} "
              f"{s['production_mean']:10.0f} {s['production_std']:8.0f} "
              f"{s['production_p05']:9.0f}")

    all_runs = pd.concat(runs, ignore_index=True)

    print(f"\n--- mismatch sweep on scenario C ---")
    hdr = (f"{'+/- %':>6s} {'runs w/ viol TRUE':>18s} {'MEAS':>7s} "
           f"{'sample rate':>12s} {'worst depth':>12s} {'p95 depth':>10s} "
           f"{'worst minBHP':>13s} {'prod mean':>10s}")
    print(hdr)
    print("-" * len(hdr))
    env_rows, sweep = [], []
    for mag in MISMATCH_LEVELS:
        df = monte_carlo(SCENARIOS["C"], mag)
        df["case"] = "sweep"
        sweep.append(df)
        s = summarise(df)
        env_rows.append({"mismatch": mag, **s})
        print(f"{100 * mag:6.0f} {s['run_violation_rate']:17.0%} "
              f"{s['run_violation_rate_meas']:6.0%} "
              f"{s['sample_violation_rate']:11.2%} "
              f"{s['worst_depth_psi']:11.2f} {s['p95_depth_psi']:10.2f} "
              f"{s['worst_min_bhp']:13.2f} {s['production_mean']:10.0f}")
    env = pd.DataFrame(env_rows)

    runs_csv = DATA_DIR / "montecarlo_runs.csv"
    env_csv = DATA_DIR / "montecarlo_envelope.csv"
    pd.concat([all_runs] + sweep, ignore_index=True).to_csv(runs_csv, index=False)
    env.to_csv(env_csv, index=False)

    headline = all_runs[all_runs["scenario"] == "C"]
    distribution_figure(headline, FIG_DIR / "05_montecarlo.png")
    envelope_figure(env, FIG_DIR / "05_robustness_envelope.png")

    safe = env[env["run_violation_rate"] == 0.0]["mismatch"]
    if len(safe):
        print(f"\nzero true violations in {N_RUNS} runs up to "
              f"+/-{100 * safe.max():.0f}% mismatch")
    breach = env[env["run_violation_rate"] > 0.0]
    if len(breach):
        r = breach.iloc[0]
        print(f"first breach at +/-{100 * r['mismatch']:.0f}%: "
              f"{r['run_violation_rate']:.0%} of runs, worst depth "
              f"{r['worst_depth_psi']:.2f} psi "
              f"({100 * r['worst_depth_psi'] / PRODUCTION.backoff['BHP']:.0f}% of the "
              f"backoff margin)")

    print(f"\nwrote {runs_csv}")
    print(f"wrote {env_csv}")
    print(f"wrote {FIG_DIR / '05_montecarlo.png'}")
    print(f"wrote {FIG_DIR / '05_robustness_envelope.png'}")


if __name__ == "__main__":
    main()
