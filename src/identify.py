import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

OUTPUTS = ("Q", "WHP", "FLP", "BHP")
OUTPUT_UNITS = {"Q": "bbl/hr", "WHP": "psi", "FLP": "psi", "BHP": "psi"}
REFERENCE_COLUMNS = {
    "Q": "OilRate_bbl_hr",
    "WHP": "WHP_psi",
    "FLP": "FLP_psi",
    "BHP": "BHP_psi",
}
DEFAULT_SATURATION_SCALE = 140.0


def _alpha(tau, ts):
    return 1.0 - math.exp(-ts / max(tau, 1e-6))


def _delayed(u, d):
    if d <= 0:
        return np.asarray(u, dtype=float)
    u = np.asarray(u, dtype=float)
    out = np.empty_like(u)
    out[:d] = u[0]
    out[d:] = u[:-d]
    return out


@dataclass
class FOPDTModel:
    gain: float
    tau: float
    bias: float
    delay: int = 0
    ts: float = 1.0

    def steady(self, u):
        return self.bias + self.gain * np.asarray(u, dtype=float)

    def simulate(self, u, y0):
        u = _delayed(u, self.delay)
        n = len(u)
        y = np.empty(n)
        y[0] = y0
        a = _alpha(self.tau, self.ts)
        for k in range(n - 1):
            y[k + 1] = y[k] + a * (self.bias + self.gain * u[k] - y[k])
        return y


@dataclass
class SaturatingModel:
    base: float
    span: float
    tau: float
    scale: float = DEFAULT_SATURATION_SCALE
    ts: float = 1.0

    def _phi(self, u):
        u = np.asarray(u, dtype=float)
        return (1.0 - np.exp(-u / self.scale)) / (1.0 - math.exp(-100.0 / self.scale))

    def steady(self, u):
        return self.base + self.span * self._phi(u)

    def local_gain(self, u):
        u = np.asarray(u, dtype=float)
        denom = 1.0 - math.exp(-100.0 / self.scale)
        return self.span * np.exp(-u / self.scale) / (self.scale * denom)

    def simulate(self, u, y0):
        u = np.asarray(u, dtype=float)
        n = len(u)
        y = np.empty(n)
        y[0] = y0
        a = _alpha(self.tau, self.ts)
        f = self.steady(u)
        for k in range(n - 1):
            y[k + 1] = y[k] + a * (f[k] - y[k])
        return y


@dataclass
class FitResult:
    name: str
    model: object
    yhat: np.ndarray = field(repr=False)
    residual: np.ndarray = field(repr=False)
    r2: float = 0.0
    rmse: float = 0.0
    resid_ac1: float = 0.0


def r_squared(y, yhat):
    y = np.asarray(y, dtype=float)
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def _finalize(name, model, u, y, y0):
    yhat = model.simulate(u, y0)
    res = np.asarray(y, dtype=float) - yhat
    ac1 = 0.0
    if len(res) > 2 and res.std() > 0:
        ac1 = float(np.corrcoef(res[1:], res[:-1])[0, 1])
    return FitResult(
        name=name,
        model=model,
        yhat=yhat,
        residual=res,
        r2=r_squared(y, yhat),
        rmse=float(np.sqrt(np.mean(res**2))),
        resid_ac1=ac1,
    )


def fit_fopdt(u, y, ts=1.0, delays=(0, 1, 2, 3), rising=None):
    u = np.asarray(u, dtype=float)
    y = np.asarray(y, dtype=float)
    y0 = float(y[0])
    if rising is None:
        rising = np.corrcoef(u, y)[0, 1] >= 0
    g0 = 1.8 if rising else -2.0
    lo = [0.0, 1e-2, -1e6] if rising else [-1e3, 1e-2, -1e6]
    hi = [1e3, 200.0, 1e6] if rising else [0.0, 200.0, 1e6]
    best = None
    for d in delays:
        def resid(p, d=d):
            return FOPDTModel(p[0], p[1], p[2], d, ts).simulate(u, y0) - y

        r = least_squares(
            resid, x0=[g0, 8.0, float(y.mean() - g0 * u.mean())], bounds=(lo, hi)
        )
        cand = _finalize(
            f"FOPDT(d={d})", FOPDTModel(r.x[0], r.x[1], r.x[2], d, ts), u, y, y0
        )
        if best is None or cand.rmse < best.rmse:
            best = cand
    return best


def fit_saturating(u, y, ts=1.0, scale=DEFAULT_SATURATION_SCALE, fit_scale=False):
    u = np.asarray(u, dtype=float)
    y = np.asarray(y, dtype=float)
    y0 = float(y[0])
    rising = np.corrcoef(u, y)[0, 1] >= 0
    s0 = 200.0 if rising else -200.0

    if fit_scale:
        def resid(p):
            return SaturatingModel(p[0], p[1], p[2], p[3], ts).simulate(u, y0) - y

        r = least_squares(
            resid,
            x0=[float(y.mean()), s0, 8.0, scale],
            bounds=(
                [-1e6, 0.0 if rising else -1e6, 1e-2, 20.0],
                [1e6, 1e6 if rising else 0.0, 200.0, 1e6],
            ),
        )
        model = SaturatingModel(r.x[0], r.x[1], r.x[2], r.x[3], ts)
    else:
        def resid(p):
            return SaturatingModel(p[0], p[1], p[2], scale, ts).simulate(u, y0) - y

        r = least_squares(
            resid,
            x0=[float(y.mean()), s0, 8.0],
            bounds=(
                [-1e6, 0.0 if rising else -1e6, 1e-2],
                [1e6, 1e6 if rising else 0.0, 200.0],
            ),
        )
        model = SaturatingModel(r.x[0], r.x[1], r.x[2], scale, ts)
    return _finalize("saturating", model, u, y, y0)


def endpoint_gain(u, y, segments):
    ups, downs = [], []
    for (s0, e0), (s1, e1) in zip(segments[:-1], segments[1:]):
        du = float(np.mean(u[s1:e1]) - np.mean(u[s0:e0]))
        if abs(du) < 1e-9:
            continue
        dy = float(np.mean(y[e1 - 3 : e1]) - np.mean(y[e0 - 3 : e0]))
        (ups if du > 0 else downs).append(dy / du)
    allg = ups + downs
    return float(np.mean(allg)) if allg else float("nan")


def segment_bounds(u):
    u = np.asarray(u, dtype=float)
    edges = [0] + [i for i in range(1, len(u)) if u[i] != u[i - 1]] + [len(u)]
    return [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]


def identify(u, data, ts=1.0, scale=DEFAULT_SATURATION_SCALE):
    results = {}
    for key in OUTPUTS:
        y = np.asarray(data[key], dtype=float)
        results[key] = {
            "fopdt": fit_fopdt(u, y, ts=ts),
            "saturating": fit_saturating(u, y, ts=ts, scale=scale),
        }
    return results


def report(results, u=None, data=None, title="identification"):
    segs = segment_bounds(u) if u is not None else None
    print(f"\n{title}")
    head = (
        f"  {'output':6s} {'model':12s} {'gain@45':>9s} {'tau_h':>7s} "
        f"{'R2':>8s} {'RMSE':>8s} {'ac1':>6s}"
    )
    print(head)
    print("  " + "-" * (len(head) - 2))
    for key in OUTPUTS:
        for mk in ("fopdt", "saturating"):
            fr = results[key][mk]
            m = fr.model
            g = m.gain if isinstance(m, FOPDTModel) else float(m.local_gain(45.0))
            print(
                f"  {key:6s} {fr.name:12s} {g:9.3f} {m.tau:7.2f} "
                f"{fr.r2:8.5f} {fr.rmse:8.3f} {fr.resid_ac1:6.2f}"
            )
    if segs is not None and data is not None and len(segs) > 2:
        print("\n  endpoint-differencing vs regression gain (bias check):")
        for key in OUTPUTS:
            y = np.asarray(data[key], dtype=float)
            ge = endpoint_gain(u, y, segs)
            gr = results[key]["fopdt"].model.gain
            err = 100.0 * (ge - gr) / gr if gr != 0 else float("nan")
            print(
                f"    {key:6s} endpoint={ge:8.3f}  regression={gr:8.3f}  "
                f"bias={err:+7.1f}%"
            )


def residual_plots(u, data, results, path, time=None, title="Model fit and residuals"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(next(iter(data.values())))
    t = np.arange(n, dtype=float) if time is None else np.asarray(time, dtype=float)
    fig, axes = plt.subplots(len(OUTPUTS), 2, figsize=(13, 11), sharex=True)
    for i, key in enumerate(OUTPUTS):
        y = np.asarray(data[key], dtype=float)
        fr = results[key]["saturating"]
        fo = results[key]["fopdt"]
        ax = axes[i, 0]
        ax.plot(t, y, ".", ms=3, color="0.55", label="data")
        ax.plot(t, fo.yhat, "-", lw=1.0, color="tab:orange",
                label=f"FOPDT R2={fo.r2:.4f}")
        ax.plot(t, fr.yhat, "-", lw=1.4, color="tab:blue",
                label=f"saturating R2={fr.r2:.4f}")
        ax.set_ylabel(f"{key} [{OUTPUT_UNITS[key]}]")
        ax.legend(fontsize=7, loc="best")
        ax.grid(alpha=0.3)

        ax = axes[i, 1]
        ax.axhline(0.0, color="0.3", lw=0.8)
        ax.plot(t, fr.residual, "-", lw=0.9, color="tab:blue")
        ax.set_ylabel(f"residual [{OUTPUT_UNITS[key]}]")
        ax.grid(alpha=0.3)
        ax.text(
            0.99, 0.05,
            f"RMSE={fr.rmse:.3f}  ac1={fr.resid_ac1:.2f}",
            transform=ax.transAxes, ha="right", fontsize=8,
        )
    axes[-1, 0].set_xlabel("time [h]")
    axes[-1, 1].set_xlabel("time [h]")
    axes[0, 0].set_title("model vs data")
    axes[0, 1].set_title("residuals (saturating model)")
    fig.suptitle(title)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def load_reference(path):
    import pandas as pd

    df = pd.read_csv(path)
    u = df["Choke_pct"].to_numpy(float)
    data = {k: df[v].to_numpy(float) for k, v in REFERENCE_COLUMNS.items()}
    return u, data, df["Time_hr"].to_numpy(float)
