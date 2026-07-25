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

BHP_BINDING_CHOKE = 68.9
MAX_SAFE_RATE = 162.9


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
