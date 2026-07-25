from dataclasses import dataclass, field

import numpy as np

import limits
from plant import ChokePlant

OUTPUT_KEYS = ("Q", "WHP", "FLP", "BHP")
SETTLE_BAND = 0.02


@dataclass
class Scenario:
    key: str
    name: str
    u0: float
    hours: int
    targets: list
    change_hour: int = 0
    feasible: bool = True
    description: str = ""

    def target_series(self):
        if len(self.targets) == 1:
            return np.full(self.hours, float(self.targets[0]))
        out = np.empty(self.hours)
        out[: self.change_hour] = float(self.targets[0])
        out[self.change_hour :] = float(self.targets[1])
        return out


SCENARIOS = {
    "A": Scenario(
        key="A",
        name="A: startup 20% choke -> 120 bbl/hr",
        u0=20.0,
        hours=150,
        targets=[120.0],
        change_hour=0,
        feasible=True,
        description="cold start from 20% choke to a 120 bbl/hr target",
    ),
    "B": Scenario(
        key="B",
        name="B: 100 -> 150 bbl/hr target change",
        u0=33.65,
        hours=180,
        targets=[100.0, 150.0],
        change_hour=40,
        feasible=True,
        description="settled at 100 bbl/hr, step target to 150 bbl/hr at t=40 h",
    ),
    "C": Scenario(
        key="C",
        name="C: 200 bbl/hr requested (infeasible)",
        u0=30.0,
        hours=200,
        targets=[200.0],
        change_hour=0,
        feasible=False,
        description="operator requests 200 bbl/hr; BHP limits the well to ~163",
    ),
}


def true_trajectory(choke_applied, u0, hours):
    twin = ChokePlant(noise=False, drift=False)
    y = twin.reset(u0=u0)
    rows = []
    for t in range(hours):
        rows.append(twin.step(choke_applied[t]))
    return np.asarray(rows, dtype=float)


def run_closed_loop(controller, scenario, seed=0, plant_kwargs=None, truth=True):
    plant = ChokePlant(seed=seed, **(plant_kwargs or {}))
    y = plant.reset(u0=scenario.u0)
    controller.reset(scenario.u0, y)
    targets = scenario.target_series()

    u_hist = np.empty(scenario.hours)
    meas = np.empty((scenario.hours, 4))
    modes = []
    for t in range(scenario.hours):
        u = controller.compute(y, targets[t])
        y = plant.step(u)
        u_hist[t] = u
        meas[t] = y
        modes.append(getattr(controller, "last_info", {}).get("mode", "?"))

    rec = {
        "t": np.arange(scenario.hours, dtype=float),
        "u": u_hist,
        "target": targets,
        "modes": modes,
        "scenario": scenario.key,
    }
    for i, k in enumerate(OUTPUT_KEYS):
        rec[f"{k}_meas"] = meas[:, i]
    if truth and not (plant_kwargs or {}):
        tr = true_trajectory(u_hist, scenario.u0, scenario.hours)
        for i, k in enumerate(OUTPUT_KEYS):
            rec[f"{k}_true"] = tr[:, i]
    elif truth:
        twin = ChokePlant(noise=False, drift=False, **(plant_kwargs or {}))
        twin.reset(u0=scenario.u0)
        tr = np.asarray([twin.step(u_hist[t]) for t in range(scenario.hours)], float)
        for i, k in enumerate(OUTPUT_KEYS):
            rec[f"{k}_true"] = tr[:, i]
    return rec


def _violations(rec, suffix):
    keys = [f"{k}_{suffix}" for k in ("WHP", "FLP", "BHP")]
    if any(k not in rec for k in keys):
        return None
    whp, flp, bhp = (rec[k] for k in keys)
    below = np.zeros(len(bhp), dtype=bool)
    depth = np.zeros(len(bhp))
    per = {}
    for name, arr in (("WHP", whp), ("FLP", flp), ("BHP", bhp)):
        lim = limits.PRESSURE_LIMITS[name]
        d = np.maximum(lim - arr, 0.0)
        per[name] = int((d > 0).sum())
        below |= d > 0
        depth = np.maximum(depth, d)
    idx = np.flatnonzero(below)
    return {
        "count": int(below.sum()),
        "rate": float(below.mean()),
        "per_output": per,
        "max_depth": float(depth.max()),
        "first": int(idx[0]) if len(idx) else -1,
        "last": int(idx[-1]) if len(idx) else -1,
        "indices": idx,
        "longest_run": _longest_run(below),
    }


def _longest_run(mask):
    best = cur = 0
    for v in mask:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return int(best)


def settling_time(q, target, start=0, band=SETTLE_BAND):
    tol = band * abs(float(np.atleast_1d(target)[-1]))
    ref = float(np.atleast_1d(target)[-1])
    inside = np.abs(q - ref) <= tol
    for i in range(start, len(q)):
        if inside[i:].all():
            return float(i - start)
    return float("nan")


def metrics(rec, scenario):
    q_meas = rec["Q_meas"]
    q_true = rec.get("Q_true")
    tgt = rec["target"]
    start = scenario.change_hour if len(scenario.targets) > 1 else 0
    ts = limits.TS_HOURS

    src = q_true if q_true is not None else q_meas
    out = {
        "scenario": scenario.key,
        "settling_time_h": settling_time(src, tgt, start),
        "iae": float(np.sum(np.abs(src[start:] - tgt[start:])) * ts),
        "total_production_bbl": float(np.sum(src) * ts),
        "final_rate": float(np.mean(src[-20:])),
        "final_choke": float(np.mean(rec["u"][-20:])),
        "total_move": float(np.sum(np.abs(np.diff(rec["u"])))),
        "soft_steps": int(sum(1 for m in rec["modes"] if m == "soft")),
    }
    for suffix, label in (("true", "true"), ("meas", "measured")):
        v = _violations(rec, suffix)
        if v is None:
            continue
        out[f"viol_{label}"] = v["count"]
        out[f"viol_{label}_rate"] = v["rate"]
        out[f"viol_{label}_maxdepth"] = v["max_depth"]
        out[f"viol_{label}_longest_run"] = v["longest_run"]
        out[f"viol_{label}_first"] = v["first"]
        key = f"BHP_{suffix}"
        if key in rec:
            out[f"min_bhp_{label}"] = float(np.min(rec[key]))
    return out
