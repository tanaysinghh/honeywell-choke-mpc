import math
from pathlib import Path

import numpy as np

OUTPUTS = ("Q", "WHP", "FLP", "BHP")

SATURATION_SCALE = 140.0

STATIC_BASE = {"Q": 23.25, "WHP": 333.82, "FLP": 227.46, "BHP": 3487.83}
STATIC_SPAN = {"Q": 183.39, "WHP": -160.96, "FLP": -98.39, "BHP": -837.69}

TAU_HOURS = {"Q": 5.42, "WHP": 9.33, "FLP": 6.52, "BHP": 13.12}

MEAS_SIGMA = {"Q": 0.365, "WHP": 0.316, "FLP": 0.344, "BHP": 1.955}
DRIFT_SIGMA = {"Q": 1.11, "WHP": 1.01, "FLP": 0.97, "BHP": 5.22}
DRIFT_TAU_HOURS = 5.0

REFERENCE_IC = {"Q": 90.0, "WHP": 250.0, "FLP": 180.0, "BHP": 3000.0}
REFERENCE_CHOKE = 30.0

TS_HOURS = 1.0
CHOKE_MIN = 0.0
CHOKE_MAX = 100.0

_PHI_NORM = 1.0 - math.exp(-CHOKE_MAX / SATURATION_SCALE)


def _phi(u):
    return (1.0 - math.exp(-u / SATURATION_SCALE)) / _PHI_NORM


def _expand(value):
    if isinstance(value, dict):
        return {k: float(value.get(k, 1.0)) for k in OUTPUTS}
    return {k: float(value) for k in OUTPUTS}


class ChokePlant:
    def __init__(
        self,
        seed=None,
        noise=True,
        drift=True,
        gain_scale=1.0,
        tau_scale=1.0,
        bias=None,
    ):
        self._noise = bool(noise)
        self._drift = bool(drift)
        self._gain_scale = _expand(gain_scale)
        self._tau_scale = _expand(tau_scale)
        self._bias = _expand(0.0) if bias is None else _expand(bias)
        self._tau = {
            k: max(TAU_HOURS[k] * self._tau_scale[k], 1e-3) for k in OUTPUTS
        }
        self._alpha = {k: 1.0 - math.exp(-TS_HOURS / self._tau[k]) for k in OUTPUTS}
        self._drift_rho = math.exp(-TS_HOURS / DRIFT_TAU_HOURS)
        self._drift_kick = math.sqrt(max(1.0 - self._drift_rho**2, 0.0))
        self._rng = np.random.default_rng(seed)
        self._x = None
        self._d = None
        self._u = None
        self._t = 0.0
        self.reset()

    def _steady(self, u, key):
        span = STATIC_SPAN[key] * self._gain_scale[key]
        return STATIC_BASE[key] + span * _phi(u) + self._bias[key]

    def reset(self, u0=REFERENCE_CHOKE, y0=None, seed=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        u0 = min(max(float(u0), CHOKE_MIN), CHOKE_MAX)
        self._u = u0
        self._t = 0.0
        if y0 is None:
            self._x = {k: self._steady(u0, k) for k in OUTPUTS}
        elif isinstance(y0, dict):
            self._x = {k: float(y0[k]) for k in OUTPUTS}
        else:
            self._x = {k: float(v) for k, v in zip(OUTPUTS, y0)}
        self._d = {k: 0.0 for k in OUTPUTS}
        if self._drift:
            for k in OUTPUTS:
                self._d[k] = float(self._rng.normal(0.0, DRIFT_SIGMA[k]))
        return self._measure()

    def _measure(self):
        out = []
        for k in OUTPUTS:
            v = self._x[k] + self._d[k]
            if self._noise:
                v += float(self._rng.normal(0.0, MEAS_SIGMA[k]))
            out.append(v)
        return tuple(out)

    def step(self, choke_position):
        u = float(choke_position)
        if not math.isfinite(u):
            raise ValueError("choke_position must be finite")
        u = min(max(u, CHOKE_MIN), CHOKE_MAX)
        self._u = u
        for k in OUTPUTS:
            target = self._steady(u, k)
            self._x[k] += self._alpha[k] * (target - self._x[k])
            if self._drift:
                w = float(self._rng.normal(0.0, DRIFT_SIGMA[k]))
                self._d[k] = self._drift_rho * self._d[k] + self._drift_kick * w
        self._t += TS_HOURS
        return self._measure()


def _reference_path():
    return Path(__file__).resolve().parents[1] / "data" / "reference_steptest.csv"


def _replay(plant, choke):
    rows = [plant.reset(u0=choke[0], y0=REFERENCE_IC)]
    for k in range(len(choke) - 1):
        rows.append(plant.step(choke[k]))
    return np.asarray(rows, dtype=float)


def _rmse_report():
    import pandas as pd

    path = _reference_path()
    df = pd.read_csv(path)
    choke = df["Choke_pct"].to_numpy(float)
    ref = df[["OilRate_bbl_hr", "WHP_psi", "FLP_psi", "BHP_psi"]].to_numpy(float)

    det = _replay(ChokePlant(noise=False, drift=False), choke)
    det_rmse = np.sqrt(np.mean((det - ref) ** 2, axis=0))

    trials = 200
    stoch = np.empty((trials, len(OUTPUTS)))
    for i in range(trials):
        sim = _replay(ChokePlant(seed=i), choke)
        stoch[i] = np.sqrt(np.mean((sim - ref) ** 2, axis=0))

    print(f"reference: {path}")
    print(f"samples: {len(df)}  choke steps: {sorted(set(choke.tolist()))}\n")
    header = f"{'output':8s} {'det RMSE':>9s} {'meas sigma':>11s} {'stoch RMSE p5-p95':>20s}"
    print(header)
    print("-" * len(header))
    for j, k in enumerate(OUTPUTS):
        lo, hi = np.percentile(stoch[:, j], [5, 95])
        print(
            f"{k:8s} {det_rmse[j]:9.3f} {MEAS_SIGMA[k]:11.3f} "
            f"{lo:9.3f} - {hi:8.3f}"
        )

    nrmse = det_rmse / ref.std(axis=0)
    print("\nnormalized deterministic RMSE (fraction of signal std):")
    for j, k in enumerate(OUTPUTS):
        print(f"  {k:8s} {nrmse[j]:6.4f}")

    print("\nsteady-state map:")
    p = ChokePlant(noise=False, drift=False)
    print(f"  {'choke':>6s} {'Q':>9s} {'WHP':>9s} {'FLP':>9s} {'BHP':>10s}")
    for u in (30, 45, 55, 65, 68.9, 80, 100):
        vals = [p._steady(u, k) for k in OUTPUTS]
        print(
            f"  {u:6.1f} {vals[0]:9.2f} {vals[1]:9.2f} "
            f"{vals[2]:9.2f} {vals[3]:10.2f}"
        )


if __name__ == "__main__":
    _rmse_report()
