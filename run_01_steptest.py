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
from identify import (
    OUTPUTS,
    endpoint_gain,
    fit_fopdt,
    fit_saturating,
    segment_bounds,
)
from plant import ChokePlant

DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures"

TAU_BHP = limits.REFERENCE_FOPDT_TAU["BHP"]
SETTLE_FACTOR = 5.0
SEGMENT_HOURS = int(np.ceil(SETTLE_FACTOR * TAU_BHP / 10.0) * 10)
LEVELS = [20.0, 35.0, 50.0, 65.0, 80.0, 65.0, 50.0, 35.0, 20.0]
SHORT_SEGMENT_HOURS = 20

COLUMNS = {
    "Q": "OilRate_bbl_hr",
    "WHP": "WHP_psi",
    "FLP": "FLP_psi",
    "BHP": "BHP_psi",
}


def run_sequence(levels, segment_hours, seed=0, u_start=None, clean=False):
    u0 = levels[0] if u_start is None else u_start
    plant = ChokePlant(seed=seed, noise=not clean, drift=not clean)
    y = plant.reset(u0=u0)
    schedule = []
    for lvl in levels:
        schedule.extend([lvl] * segment_hours)
    rows = [(0.0, u0) + y]
    for k, u in enumerate(schedule):
        y = plant.step(u)
        rows.append((float(k + 1), u) + y)
    arr = np.asarray(rows, dtype=float)
    df = pd.DataFrame(
        {
            "Time_hr": arr[:, 0],
            "Choke_pct": arr[:, 1],
            "OilRate_bbl_hr": arr[:, 2],
            "WHP_psi": arr[:, 3],
            "FLP_psi": arr[:, 4],
            "BHP_psi": arr[:, 5],
        }
    )
    return df


def as_dict(df):
    return {k: df[v].to_numpy(float) for k, v in COLUMNS.items()}


def settling_check(df):
    u = df["Choke_pct"].to_numpy(float)
    bhp = df["BHP_psi"].to_numpy(float)
    segs = segment_bounds(u)
    out = []
    for idx, (s, e) in enumerate(segs):
        if e - s < 10 or idx == 0:
            continue
        seg = bhp[s:e]
        start = float(seg[0])
        final = float(seg[-1])
        span = final - start
        if abs(span) < 1.0:
            continue
        approach = float(seg[-1] - seg[-6])
        out.append(
            {
                "choke": float(u[s]),
                "hours": e - s,
                "bhp_start": start,
                "bhp_end": final,
                "step_span": span,
                "last_5h_move": approach,
                "unsettled_pct": 100.0 * abs(approach) / abs(span),
            }
        )
    return pd.DataFrame(out)


def plot_steptest(df, path, title):
    u = df["Choke_pct"].to_numpy(float)
    t = df["Time_hr"].to_numpy(float)
    fig, axes = plt.subplots(5, 1, figsize=(12, 13), sharex=True)
    axes[0].step(t, u, where="post", color="tab:purple", lw=1.4)
    axes[0].set_ylabel("choke [%]")
    axes[0].axhline(limits.FIRST_BINDING_CHOKE, ls="--", lw=1.0, color="tab:red")
    axes[0].text(
        t[-1], limits.FIRST_BINDING_CHOKE, f" BHP binds {limits.FIRST_BINDING_CHOKE:.1f}%",
        va="bottom", ha="right", fontsize=8, color="tab:red",
    )
    axes[0].grid(alpha=0.3)

    specs = [
        ("OilRate_bbl_hr", "oil rate [bbl/hr]", None, "tab:green"),
        ("WHP_psi", "WHP [psi]", limits.WHP_MIN, "tab:blue"),
        ("FLP_psi", "FLP [psi]", limits.FLP_MIN, "tab:orange"),
        ("BHP_psi", "BHP [psi]", limits.BHP_MIN, "tab:red"),
    ]
    for ax, (col, lab, lim, color) in zip(axes[1:], specs):
        ax.plot(t, df[col].to_numpy(float), lw=1.0, color=color)
        if lim is not None:
            ax.axhline(lim, ls="--", lw=1.2, color="k")
            ax.text(t[0], lim, f" limit {lim:g}", va="bottom", fontsize=8)
        else:
            ax.axhline(limits.MAX_SAFE_RATE, ls="--", lw=1.2, color="k")
            ax.text(
                t[0], limits.MAX_SAFE_RATE,
                f" max safe {limits.MAX_SAFE_RATE:g}", va="bottom", fontsize=8,
            )
        ax.set_ylabel(lab)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("time [h]")
    fig.suptitle(title)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_endpoint_bias(sweep, true_gain, path):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.axhline(true_gain, ls="--", color="k", lw=1.2,
               label=f"regression gain {true_gain:.2f} psi/%  (630 h FOPDT fit)")
    ax.plot(sweep["segment_hours"], sweep["endpoint_gain"], "o-",
            color="tab:red", label="endpoint-differenced gain")
    ax.axvline(TAU_BHP, ls=":", color="tab:blue", lw=1.2,
               label=f"tau_BHP = {TAU_BHP:.1f} h  (reference-CSV FOPDT fit)")
    ax.set_xlabel("step segment duration [h]")
    ax.set_ylabel("estimated BHP gain [psi / % choke]")
    ax.set_title("Endpoint differencing understates BHP gain for short steps")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    for _, r in sweep.iterrows():
        ax.annotate(
            f"{100 * (r['endpoint_gain'] - true_gain) / true_gain:+.0f}%",
            (r["segment_hours"], r["endpoint_gain"]),
            textcoords="offset points", xytext=(0, -14), fontsize=7, ha="center",
        )
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("RUN 01 - OPEN-LOOP STEP TESTS")
    print("=" * 72)
    print(f"tau_BHP = {TAU_BHP:.2f} h   segment = {SEGMENT_HOURS} h "
          f"({SEGMENT_HOURS / TAU_BHP:.1f} x tau_BHP)")
    print(f"levels  = {LEVELS}")
    print("choke moves are true steps; the +/-5%/interval rate limit is a")
    print("controller-side constraint (limits.CHOKE_MAX_MOVE), not a plant limit.")

    df_long = run_sequence(LEVELS, SEGMENT_HOURS, seed=0)
    long_path = DATA_DIR / "steptest_long.csv"
    df_long.to_csv(long_path, index=False)
    print(f"\nwrote {long_path}  ({len(df_long)} h)")

    df_short = run_sequence(LEVELS, SHORT_SEGMENT_HOURS, seed=0)
    short_path = DATA_DIR / "steptest_short.csv"
    df_short.to_csv(short_path, index=False)
    print(f"wrote {short_path}  ({len(df_short)} h)")

    df_clean = run_sequence(LEVELS, SEGMENT_HOURS, clean=True)
    print("\nBHP settling per segment (noise-free replica, so this measures")
    print("dynamics rather than the disturbance):")
    sc = settling_check(df_clean)
    print(sc.to_string(index=False, float_format=lambda v: f"{v:9.3f}"))
    worst = sc["unsettled_pct"].max()
    print(f"\nworst final-5h movement = {worst:.2f}% of step span "
          f"-> BHP settled in every segment")

    data_long = as_dict(df_long)
    u_long = df_long["Choke_pct"].to_numpy(float)
    print("\nregression fits from the long test (20-80% choke):")
    print(f"  {'out':5s} {'model':11s} {'gain@45':>9s} {'tau_h':>7s} "
          f"{'R2':>9s} {'RMSE':>8s}")
    true_gains = {}
    for key in OUTPUTS:
        fr = fit_fopdt(u_long, data_long[key])
        sr = fit_saturating(u_long, data_long[key])
        true_gains[key] = fr.model.gain
        print(f"  {key:5s} {'FOPDT':11s} {fr.model.gain:9.3f} "
              f"{fr.model.tau:7.2f} {fr.r2:9.5f} {fr.rmse:8.3f}")
        print(f"  {'':5s} {'saturating':11s} {float(sr.model.local_gain(45.0)):9.3f} "
              f"{sr.model.tau:7.2f} {sr.r2:9.5f} {sr.rmse:8.3f}")

    segs_long = segment_bounds(u_long)
    segs_short = segment_bounds(df_short["Choke_pct"].to_numpy(float))
    data_short = as_dict(df_short)
    print("\nendpoint differencing, short (20 h) vs long "
          f"({SEGMENT_HOURS} h) segments:")
    for key in OUTPUTS:
        ge_s = endpoint_gain(df_short["Choke_pct"].to_numpy(float),
                             data_short[key], segs_short)
        ge_l = endpoint_gain(u_long, data_long[key], segs_long)
        g = true_gains[key]
        print(f"  {key:5s} true={g:8.3f}  short={ge_s:8.3f} "
              f"({100 * (ge_s - g) / g:+6.1f}%)  long={ge_l:8.3f} "
              f"({100 * (ge_l - g) / g:+6.1f}%)")

    rows = []
    durations = [10, 15, 20, 25, 30, 40, 50, 60, 70]
    for d in durations:
        vals = []
        for seed in range(5):
            dfx = run_sequence(LEVELS, d, seed=seed)
            vals.append(
                endpoint_gain(
                    dfx["Choke_pct"].to_numpy(float),
                    dfx["BHP_psi"].to_numpy(float),
                    segment_bounds(dfx["Choke_pct"].to_numpy(float)),
                )
            )
        rows.append({"segment_hours": d, "endpoint_gain": float(np.mean(vals))})
    sweep = pd.DataFrame(rows)
    sweep_path = DATA_DIR / "endpoint_gain_bias.csv"
    sweep.to_csv(sweep_path, index=False)
    print(f"\nwrote {sweep_path}")
    print("\nBHP endpoint-gain bias vs segment duration:")
    gb = true_gains["BHP"]
    for _, r in sweep.iterrows():
        print(f"  {int(r['segment_hours']):3d} h  gain={r['endpoint_gain']:8.3f}  "
              f"bias={100 * (r['endpoint_gain'] - gb) / gb:+6.1f}%")

    viol = limits.summarize(
        df_long["WHP_psi"].to_numpy(float),
        df_long["FLP_psi"].to_numpy(float),
        df_long["BHP_psi"].to_numpy(float),
    )
    print("\nconstraint violations during the open-loop test "
          "(expected - the test deliberately drives past the envelope):")
    print(f"  violating samples: {viol['violation_steps']} / {viol['n_samples']} "
          f"({100 * viol['violation_rate']:.1f}%)")
    for k in limits.PRESSURE_ORDER:
        print(f"    {k:4s} count={viol['counts'][k]:4d}  "
              f"max depth={viol['max_depth'][k]:8.2f} psi")

    plot_steptest(df_long, FIG_DIR / "01_steptest_long.png",
                  f"Open-loop step test, {SEGMENT_HOURS} h segments "
                  f"({SEGMENT_HOURS / TAU_BHP:.1f} x tau_BHP)")
    plot_steptest(df_short, FIG_DIR / "01_steptest_short.png",
                  f"Open-loop step test, {SHORT_SEGMENT_HOURS} h segments "
                  "(too short for BHP)")
    plot_endpoint_bias(sweep, gb, FIG_DIR / "01_endpoint_gain_bias.png")
    print(f"\nfigures written to {FIG_DIR}")


if __name__ == "__main__":
    main()
