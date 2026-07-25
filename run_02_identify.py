import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import identify
import limits
from identify import (OUTPUTS, FOPDTModel, identify as fit_all, load_reference,
                      r_squared, report, residual_plots)
from mpc import IDENTIFIED_FOPDT, IDENTIFIED_SATURATING, build_models

DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures"
TS = limits.TS_HOURS

DATASETS = {
    "long": ("steptest_long.csv", "own step test, 9 levels x 70 h, 20-80% choke"),
    "short": ("steptest_short.csv", "own step test, 9 levels x 20 h (deliberately "
                                    "shorter than tau_BHP)"),
    "reference": ("reference_steptest.csv", "supplied reference step test, 120 h"),
}


def load(name):
    return load_reference(DATA_DIR / DATASETS[name][0])


def cross_validate(models, u, data):
    out = {}
    for key in OUTPUTS:
        y = np.asarray(data[key], dtype=float)
        yhat = np.asarray(models[key].simulate(u, y[0]), dtype=float)
        resid = y - yhat
        out[key] = {
            "r2": float(r_squared(y, yhat)),
            "rmse": float(np.sqrt(np.mean(resid**2))),
            "bias": float(np.mean(resid)),
            "max_abs": float(np.max(np.abs(resid))),
        }
    return out


def print_cv(title, cv):
    print(f"\n  {title}")
    head = f"    {'output':6s} {'R2':>9s} {'RMSE':>8s} {'mean resid':>11s} {'max |e|':>8s}"
    print(head)
    print("    " + "-" * (len(head) - 4))
    for key in OUTPUTS:
        s = cv[key]
        print(f"    {key:6s} {s['r2']:9.5f} {s['rmse']:8.3f} {s['bias']:11.3f} "
              f"{s['max_abs']:8.3f}")


def shipped_models():
    return {
        "saturating": build_models("saturating", TS),
        "fopdt": build_models("fopdt", TS),
    }


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 82)
    print("RUN 02 - IDENTIFICATION AND VALIDATION")
    print("=" * 82)
    print(f"control interval {TS:g} h, saturation scale "
          f"{identify.DEFAULT_SATURATION_SCALE:g}")
    print("models are fitted by simulation-error minimisation, not endpoint "
          "differencing")

    u_long, d_long, t_long = load("long")
    results = fit_all(u_long, d_long, ts=TS)
    report(results, u_long, d_long,
           title=f"FIT on {DATASETS['long'][0]} ({DATASETS['long'][1]})")

    rows = []
    for key in OUTPUTS:
        for mk in ("fopdt", "saturating"):
            fr = results[key][mk]
            m = fr.model
            rows.append({
                "dataset": "long", "role": "fit", "output": key, "model": mk,
                "gain_at_45": (m.gain if isinstance(m, FOPDTModel)
                               else float(m.local_gain(45.0))),
                "tau_h": m.tau, "r2": fr.r2, "rmse": fr.rmse,
                "resid_ac1": fr.resid_ac1,
            })

    print("\n" + "-" * 82)
    print("HELD-OUT VALIDATION - the shipped model coefficients, never refitted,")
    print("simulated open-loop from the first sample of each dataset")
    print("-" * 82)

    models = shipped_models()
    for ds in ("long", "short", "reference"):
        u, data, _ = load(ds)
        print(f"\n{DATASETS[ds][0]}  -  {DATASETS[ds][1]}")
        for mk in ("saturating", "fopdt"):
            cv = cross_validate(models[mk], u, data)
            print_cv(f"{mk} model", cv)
            for key in OUTPUTS:
                rows.append({
                    "dataset": ds, "role": "validate", "output": key, "model": mk,
                    "gain_at_45": np.nan, "tau_h": models[mk][key].tau,
                    "r2": cv[key]["r2"], "rmse": cv[key]["rmse"],
                    "resid_ac1": np.nan,
                })

    print("\n" + "-" * 82)
    print("MODEL STRUCTURE COMPARISON on the held-out reference step test")
    print("-" * 82)
    u_ref, d_ref, t_ref = load("reference")
    cv_sat = cross_validate(models["saturating"], u_ref, d_ref)
    cv_lin = cross_validate(models["fopdt"], u_ref, d_ref)
    head = (f"  {'output':6s} {'RMSE sat':>10s} {'RMSE lin':>10s} "
            f"{'improvement':>12s}")
    print(head)
    print("  " + "-" * (len(head) - 2))
    for key in OUTPUTS:
        a, b = cv_sat[key]["rmse"], cv_lin[key]["rmse"]
        print(f"  {key:6s} {a:10.3f} {b:10.3f} {100.0 * (b - a) / b:11.1f}%")

    print("\n  shipped coefficients used by the controller:")
    print(f"    {'output':6s} {'base':>10s} {'span':>10s} {'tau_h':>8s} "
          f"{'| linear gain':>14s} {'tau_h':>8s}")
    for key in OUTPUTS:
        b, s, tau, _ = IDENTIFIED_SATURATING[key]
        g, taul, _, _ = IDENTIFIED_FOPDT[key]
        print(f"    {key:6s} {b:10.3f} {s:10.3f} {tau:8.3f} "
              f"| {g:12.4f} {taul:8.3f}")

    df = pd.DataFrame(rows)
    csv = DATA_DIR / "identification_validation.csv"
    df.to_csv(csv, index=False)

    png_fit = residual_plots(
        u_long, d_long, results, FIG_DIR / "02_fit_long.png", time=t_long,
        title="Identification fit - own step test (70 h per level, 5.7 x tau_BHP)")

    ref_results = fit_all(u_ref, d_ref, ts=TS)
    png_ref = residual_plots(
        u_ref, d_ref, ref_results, FIG_DIR / "02_fit_reference.png", time=t_ref,
        title="Identification fit - supplied reference step test (120 h)")

    worst_sat = min(cv_sat[k]["r2"] for k in OUTPUTS)
    print(f"\nworst held-out R2 across all four outputs, saturating model: "
          f"{worst_sat:.5f}")
    print(f"\nwrote {csv}")
    print(f"wrote {png_fit}")
    print(f"wrote {png_ref}")


if __name__ == "__main__":
    main()
