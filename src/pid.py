from dataclasses import dataclass

import limits

DEFAULT_KC = 0.55
DEFAULT_TI = 6.0


@dataclass
class PIConfig:
    kc: float = DEFAULT_KC
    ti: float = DEFAULT_TI
    ts: float = limits.TS_HOURS
    u_min: float = limits.CHOKE_MIN
    u_max: float = limits.CHOKE_MAX
    max_move: float = limits.CHOKE_MAX_MOVE
    name: str = "PI on oil rate"


class PIController:
    """Velocity-form PI on oil rate, representing current manual practice.

    Deliberately has no constraint awareness: it sees only oil rate and its
    target, and will drive the choke straight through the BHP limit if the
    target demands it. Anti-windup is inherent to the velocity form, since
    the integral term acts on the applied (clamped and rate-limited) choke
    rather than on an unbounded internal accumulator. The same +/-5 % per
    interval slew limit as the MPC is applied.

    Tuned by SIMC/IMC on the identified oil-rate model (K = 1.83 bbl/hr per
    %, tau = 4.56 h, Ts = 1 h) with a detuning factor for the noise level.
    """

    def __init__(self, config=None):
        self.config = config or PIConfig()
        self.u_prev = 0.0
        self._e_prev = None
        self.last_info = {}

    def reset(self, u0, measurement=None):
        self.u_prev = float(u0)
        self._e_prev = None
        self.last_info = {}
        return self.u_prev

    def compute(self, measurement, target):
        q = float(measurement[0])
        e = float(target) - q
        if self._e_prev is None:
            self._e_prev = e
        cfg = self.config
        du = cfg.kc * ((e - self._e_prev) + (cfg.ts / cfg.ti) * e)
        du = max(-cfg.max_move, min(cfg.max_move, du))
        u = min(max(self.u_prev + du, cfg.u_min), cfg.u_max)
        self._e_prev = e
        self.last_info = {
            "mode": "pi",
            "error": e,
            "du": u - self.u_prev,
            "u_prev": self.u_prev,
            "u": u,
            "target": float(target),
            "soft": False,
        }
        self.u_prev = u
        return u
