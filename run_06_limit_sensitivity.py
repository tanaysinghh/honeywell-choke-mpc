import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory

import limits
from evaluate import SCENARIOS, metrics, run_closed_loop
from mpc import (PRODUCTION, ChokeMPC, binding_chokes, build_models,
                 infeasibility_report, rate_ceiling)

DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures"

SHIPPED = {"WHP": 200.0, "FLP": 145.0, "BHP": 2850.0}
SWEEPS = (
    ("BHP", np.arange(2800.0, 2901.0, 10.0)),
    ("WHP", np.arange(180.0, 221.0, 5.0)),
    ("FLP", np.arange(135.0, 161.0, 2.5)),
)
SEEDS = (0, 1, 2, 3, 4)
SCENARIO_KEYS = ("A", "B", "C")

C_BIND = {"BHP": "#2a78d6", "WHP": "#d98c1f", "FLP": "#2e9e6b"}
C_SHIPPED = "#e34948"
C_GRID = "#b8b6b2"


def limit_set(values):
    return limits.override(**values)


def envelope(models):
    return binding_chokes(models), rate_ceiling(models, None), rate_ceiling(
        models, PRODUCTION.backoff)


def evaluate_config(models):
    per, hard, held = envelope(models)
    base = {
        "limit_WHP": limits.PRESSURE_LIMITS["WHP"],
        "limit_FLP": limits.PRESSURE_LIMITS["FLP"],
        "limit_BHP": limits.PRESSURE_LIMITS["BHP"],
        "binds_first": hard["binding"],
        "binding_choke": hard["choke"],
        "max_safe_rate": hard["rate"],
        "backoff_choke": held["choke"],
        "backoff_rate": held["rate"],
        "choke_WHP": per["WHP"],
        "choke_FLP": per["FLP"],
        "choke_BHP": per["BHP"],
    }
    rows = []
    for key in SCENARIO_KEYS:
        sc = SCENARIOS[key]
        adv = infeasibility_report(sc.targets[-1], models, PRODUCTION.backoff)
        agg = {"viol_true": 0, "viol_meas": 0, "soft": 0}
        depth = 0.0
        min_bhp = np.inf
        margin = np.inf
        margin_at = None
        rate = []
        for seed in SEEDS:
            rec = run_closed_loop(ChokeMPC(PRODUCTION), sc, seed=seed)
            m = metrics(rec, sc)
            agg["viol_true"] += int(m["viol_true"])
            agg["viol_meas"] += int(m["viol_measured"])
            agg["soft"] += int(m["soft_steps"])
            depth = max(depth, float(m["viol_true_maxdepth"]))
            min_bhp = min(min_bhp, float(m["min_bhp_true"]))
            rate.append(float(m["final_rate"]))
            for k in limits.PRESSURE_ORDER:
                mg = float(np.min(rec[f"{k}_true"])) - limits.PRESSURE_LIMITS[k]
                if mg < margin:
                    margin, margin_at = mg, k
        rows.append({
            **base,
            "scenario": key,
            "target": float(sc.targets[-1]),
            "target_feasible": adv is None,
            "target_binding": None if adv is None else adv["binding"],
            "n_seeds": len(SEEDS),
            "samples": sc.hours * len(SEEDS),
            "viol_true": agg["viol_true"],
            "viol_meas": agg["viol_meas"],
            "viol_true_maxdepth": depth,
            "min_bhp_true": float(min_bhp),
            "worst_margin": float(margin),
            "worst_margin_output": margin_at,
            "final_rate": float(np.mean(rate)),
            "soft_steps": agg["soft"],
        })
    return rows


def sweep_figure(df, path):
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.4), sharey="row")
    for col, (name, _) in enumerate(SWEEPS):
        sub = df[(df["swept"] == name) & (df["scenario"] == "C")].sort_values("value")
        tot = (df[df["swept"] == name].groupby("value", as_index=False)
               .agg(viol_true=("viol_true", "sum"), samples=("samples", "sum"),
                    worst_margin=("worst_margin", "min")))
        x = sub["value"].to_numpy(float)
        y = sub["max_safe_rate"].to_numpy(float)

        ax = axes[0, col]
        ax.plot(x, y, lw=1.6, color="#52514e", zorder=2)
        for b in sorted(set(sub["binds_first"])):
            m = sub["binds_first"].to_numpy() == b
            ax.scatter(x[m], y[m], s=46, color=C_BIND[b], zorder=3,
                       edgecolor="white", linewidth=0.7, label=f"{b} binds first")
        xs = SHIPPED[name]
        ys = float(sub.loc[np.isclose(x, xs), "max_safe_rate"].iloc[0])
        ax.scatter([xs], [ys], marker="*", s=340, color=C_SHIPPED, zorder=4,
                   edgecolor="white", linewidth=0.8, label="shipped operating point")
        dx, ha = (-10, "right") if name == "BHP" else (10, "left")
        ax.annotate(f"{xs:g} psi\n{ys:.1f} bbl/hr", xy=(xs, ys),
                    xytext=(dx, -32), textcoords="offset points",
                    fontsize=8, color=C_SHIPPED, ha=ha)
        ax.set_title(f"{name} limit swept, other two at shipped values", fontsize=10)
        ax.set_xlabel(f"{name} limit [psi]")
        ax.grid(alpha=0.25, color=C_GRID)
        ax.legend(fontsize=7.5, loc="best")

        ax = axes[1, col]
        w = 0.55 * float(np.diff(x)[0])
        ax.bar(tot["value"], tot["worst_margin"], width=w,
               color=C_BIND[name], alpha=0.85)
        ax.axhline(0.0, lw=1.4, color=C_SHIPPED)
        ax.set_xlabel(f"{name} limit [psi]")
        ax.grid(alpha=0.25, color=C_GRID)
        n = int(tot["samples"].sum())
        v = int(tot["viol_true"].sum())
        ok = v == 0
        ax.text(0.5, 0.90, f"{v} true violations over {n} controlled hours"
                if ok else f"{v} TRUE VIOLATIONS over {n} controlled hours",
                transform=ax.transAxes, ha="center", va="top", fontsize=8.5,
                color="#2e9e6b" if ok else C_SHIPPED,
                bbox=dict(boxstyle="round,pad=0.35",
                          fc="#eaf5ef" if ok else "#fbe9e9",
                          ec="#2e9e6b" if ok else C_SHIPPED, lw=0.8))

    axes[0, 0].set_ylabel("maximum safe rate\n[bbl/hr]")
    axes[1, 0].set_ylabel("worst true pressure\nmargin held [psi]")
    axes[1, 0].set_ylim(-2.6, 17.2)
    for col in range(3):
        axes[1, col].text(
            0.015, -1.55, "limit - a violation is any bar below this line",
            transform=blended_transform_factory(axes[1, col].transAxes,
                                                axes[1, col].transData),
            fontsize=7.5, color=C_SHIPPED, va="center")
    fig.suptitle("Sensitivity to the assumed numeric pressure limits\n"
                 "shipped controller, unmodified, scenarios A+B+C x 5 seeds at "
                 "every limit value", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("RUN 06 - SENSITIVITY TO THE ASSUMED NUMERIC PRESSURE LIMITS")
    print("=" * 78)
    print("The problem statement names WHP, FLP and BHP as active constraints but")
    print("gives no numeric values; 200 / 145 / 2850 psi are an assumption of this")
    print("submission. This sweeps each independently over a plausible range and")
    print("re-runs the shipped controller unmodified at every point.")
    print(f"controller: {PRODUCTION.name}")
    print(f"scenarios: {', '.join(SCENARIO_KEYS)}   seeds: {list(SEEDS)}")

    models = build_models(PRODUCTION.structure, PRODUCTION.ts)
    rows = []
    for name, values in SWEEPS:
        print(f"\n--- {name} limit swept over {values[0]:g}-{values[-1]:g} psi "
              f"({len(values)} points) ---")
        print(f"  {'limit':>7s} {'binds':>6s} {'choke':>7s} {'max safe':>9s} "
              f"{'backoff':>8s} {'A':>4s} {'B':>4s} {'C':>4s} {'viol':>5s}")
        for v in values:
            cfg = dict(SHIPPED)
            cfg[name] = float(v)
            with limit_set(cfg):
                out = evaluate_config(models)
            for r in out:
                r["swept"] = name
                r["value"] = float(v)
            rows.extend(out)

            head = out[0]
            feas = {r["scenario"]: r["target_feasible"] for r in out}
            viol = sum(r["viol_true"] for r in out)
            mark = "  <-- shipped" if np.isclose(v, SHIPPED[name]) else ""
            print(f"  {v:7.1f} {head['binds_first']:>6s} "
                  f"{head['binding_choke']:7.2f} {head['max_safe_rate']:9.2f} "
                  f"{head['backoff_rate']:8.2f} "
                  f"{'ok' if feas['A'] else 'INF':>4s} "
                  f"{'ok' if feas['B'] else 'INF':>4s} "
                  f"{'ok' if feas['C'] else 'INF':>4s} "
                  f"{viol:5d}{mark}")

    df = pd.DataFrame(rows)
    csv = DATA_DIR / "limit_sensitivity.csv"
    df.to_csv(csv, index=False)

    print("\n" + "=" * 78)
    print("NON-BHP ACTIVE CONSTRAINT")
    print("=" * 78)
    alt = df[df["binds_first"] != "BHP"]
    if alt.empty:
        print("  no swept limit made WHP or FLP bind before BHP")
    else:
        keys = alt[["swept", "value", "binds_first"]].drop_duplicates()
        print(f"  {len(keys)} of {len(df[['swept','value']].drop_duplicates())} "
              f"swept points move the active constraint off BHP:")
        for _, k in keys.iterrows():
            s = alt[(alt["swept"] == k["swept"]) & (alt["value"] == k["value"])]
            h = s.iloc[0]
            print(f"    {k['swept']} = {k['value']:7.1f} psi -> {k['binds_first']} "
                  f"binds at {h['binding_choke']:.2f} % choke, max safe rate "
                  f"{h['max_safe_rate']:.2f} bbl/hr, "
                  f"{int(s['viol_true'].sum())} true violations")

        worst = keys.iloc[-1]
        s = alt[(alt["swept"] == worst["swept"]) & (alt["value"] == worst["value"])]
        cfg = dict(SHIPPED)
        cfg[worst["swept"]] = float(worst["value"])
        print(f"\n  detail, {worst['swept']} = {worst['value']:g} psi "
              f"({worst['binds_first']} active, no controller change):")
        with limit_set(cfg):
            for _, r in s.iterrows():
                sc = SCENARIOS[r["scenario"]]
                adv = infeasibility_report(sc.targets[-1], models, PRODUCTION.backoff)
                print(f"    scenario {r['scenario']}, target {r['target']:.0f} "
                      f"bbl/hr: settled {r['final_rate']:.2f} bbl/hr, "
                      f"min BHP {r['min_bhp_true']:.1f} psi, "
                      f"{int(r['viol_true'])} true violations "
                      f"in {int(r['samples'])} h")
                if adv is not None:
                    print(f"      ADVISORY: {adv['message']}")

    print("\n" + "=" * 78)
    print("HEADLINE")
    print("=" * 78)
    pts = df[["swept", "value"]].drop_duplicates()
    hours = int(df["samples"].sum())
    print(f"  {len(pts)} limit sets x {len(SCENARIO_KEYS)} scenarios x "
          f"{len(SEEDS)} seeds = {hours} controlled hours")
    print(f"  max safe rate spans {df['max_safe_rate'].min():.2f} - "
          f"{df['max_safe_rate'].max():.2f} bbl/hr")
    print(f"  binding constraint takes values: "
          f"{sorted(set(df['binds_first']))}")
    print(f"  TRUE violations, all configurations : {int(df['viol_true'].sum())}")
    print(f"  MEASURED violations, all configs    : {int(df['viol_meas'].sum())}")
    print(f"  worst true excursion depth          : "
          f"{df['viol_true_maxdepth'].max():.3f} psi")
    w = df.loc[df["worst_margin"].idxmin()]
    print(f"  smallest true margin ever held      : {w['worst_margin']:.2f} psi "
          f"on {w['worst_margin_output']} ({w['swept']}={w['value']:g}, "
          f"scenario {w['scenario']})")

    png = FIG_DIR / "06_limit_sensitivity.png"
    sweep_figure(df, png)
    print(f"\nwrote {csv.name}, {png.name}")


if __name__ == "__main__":
    main()
