import contextlib
from dataclasses import dataclass

TS_HOURS = 1.0

CHOKE_MIN = 0.0
CHOKE_MAX = 100.0
CHOKE_MAX_MOVE = 5.0

WHP_MIN = 200.0
FLP_MIN = 145.0
BHP_MIN = 2850.0

PRESSURE_LIMITS = {"WHP": WHP_MIN, "FLP": FLP_MIN, "BHP": BHP_MIN}
PRESSURE_ORDER = ("WHP", "FLP", "BHP")

IDENTIFIED_GAIN = {"Q": 1.83, "WHP": -1.61, "FLP": -0.98, "BHP": -8.40}
IDENTIFIED_TAU = {"Q": 5.42, "WHP": 9.33, "FLP": 6.52, "BHP": 13.12}

BINDING_CHOKE = {"BHP": 69.00, "WHP": 77.16, "FLP": 78.25}
BINDING_ORDER = ("BHP", "WHP", "FLP")

FIRST_BINDING = "BHP"
FIRST_BINDING_CHOKE = 69.00
MAX_SAFE_RATE = 163.05
MAX_SAFE_RATE_CURVATURE_RANGE = (162.7, 163.2)


@dataclass(frozen=True)
class Margins:
    whp: float
    flp: float
    bhp: float

    @property
    def worst(self):
        return min(self.whp, self.flp, self.bhp)

    @property
    def binding(self):
        pairs = (("WHP", self.whp), ("FLP", self.flp), ("BHP", self.bhp))
        return min(pairs, key=lambda p: p[1])[0]

    def as_dict(self):
        return {"WHP": self.whp, "FLP": self.flp, "BHP": self.bhp}


@contextlib.contextmanager
def override(**values):
    """Temporarily install a different set of numeric pressure limits.

    The numeric values 200 / 145 / 2850 psi are an assumption: the problem
    statement names WHP, FLP and BHP as active constraints but gives no numbers
    anywhere. Every consumer reads PRESSURE_LIMITS and the WHP_MIN/FLP_MIN/
    BHP_MIN scalars at call time rather than capturing them at import, so
    patching this module is enough to move the constraint set seen by the
    controller, the steady-state feasibility screen, the backoff tier and the
    violation counter simultaneously. Nothing is recompiled and no tuning
    constant is touched, which is what makes run_06_limit_sensitivity.py a test
    of the method rather than of a particular calibration.
    """
    global WHP_MIN, FLP_MIN, BHP_MIN
    saved_map = dict(PRESSURE_LIMITS)
    saved_scalar = (WHP_MIN, FLP_MIN, BHP_MIN)
    try:
        PRESSURE_LIMITS.update({k: float(v) for k, v in values.items()})
        WHP_MIN = PRESSURE_LIMITS["WHP"]
        FLP_MIN = PRESSURE_LIMITS["FLP"]
        BHP_MIN = PRESSURE_LIMITS["BHP"]
        yield
    finally:
        PRESSURE_LIMITS.clear()
        PRESSURE_LIMITS.update(saved_map)
        WHP_MIN, FLP_MIN, BHP_MIN = saved_scalar


def margins(whp, flp, bhp):
    return Margins(whp - WHP_MIN, flp - FLP_MIN, bhp - BHP_MIN)


def is_violation(whp, flp, bhp):
    return margins(whp, flp, bhp).worst < 0.0


def violation_detail(whp, flp, bhp):
    m = margins(whp, flp, bhp).as_dict()
    return {k: -v for k, v in m.items() if v < 0.0}


def clamp_choke(u, u_prev=None):
    if u_prev is not None:
        lo = u_prev - CHOKE_MAX_MOVE
        hi = u_prev + CHOKE_MAX_MOVE
        u = min(max(u, lo), hi)
    return min(max(u, CHOKE_MIN), CHOKE_MAX)


def summarize(whp, flp, bhp):
    n = len(bhp)
    counts = {k: 0 for k in PRESSURE_ORDER}
    depths = {k: 0.0 for k in PRESSURE_ORDER}
    steps = 0
    for i in range(n):
        d = violation_detail(whp[i], flp[i], bhp[i])
        if d:
            steps += 1
        for k, v in d.items():
            counts[k] += 1
            depths[k] = max(depths[k], v)
    return {
        "n_samples": n,
        "violation_steps": steps,
        "violation_rate": steps / n if n else 0.0,
        "counts": counts,
        "max_depth": depths,
    }
