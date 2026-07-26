import math
from dataclasses import dataclass, field

import numpy as np

import limits
from identify import OUTPUTS, FOPDTModel, SaturatingModel

PRESSURES = limits.PRESSURE_ORDER

IDENTIFIED_SATURATING = {
    "Q": (23.6103, 182.9253, 4.5596, 140.0),
    "WHP": (333.8066, -161.2052, 8.0859, 140.0),
    "FLP": (227.2177, -98.0204, 5.7953, 140.0),
    "BHP": (3487.7920, -836.7140, 12.3422, 140.0),
}

IDENTIFIED_FOPDT = {
    "Q": (1.8289, 4.6315, 37.3304, 0),
    "WHP": (-1.6097, 8.2671, 321.5981, 0),
    "FLP": (-0.9799, 5.8968, 219.8549, 0),
    "BHP": (-8.3501, 12.5538, 3424.0771, 0),
}

DEFAULT_BACKOFF = {"WHP": 5.0, "FLP": 5.0, "BHP": 15.0}
DEFAULT_SOFT_WEIGHT = {"WHP": 1.0, "FLP": 1.0, "BHP": 1.0}
BACKOFF_START_HOURS = 6.0


def build_models(structure="saturating", ts=limits.TS_HOURS):
    if structure == "saturating":
        return {
            k: SaturatingModel(base=b, span=s, tau=t, scale=c, ts=ts)
            for k, (b, s, t, c) in IDENTIFIED_SATURATING.items()
        }
    if structure == "fopdt":
        return {
            k: FOPDTModel(gain=g, tau=t, bias=b, delay=d, ts=ts)
            for k, (g, t, b, d) in IDENTIFIED_FOPDT.items()
        }
    raise ValueError(f"unknown model structure: {structure!r}")


def steady_choke_for_rate(models, target, lo=0.0, hi=100.0, tol=1e-4):
    q = models["Q"]
    if float(q.steady(hi)) <= target:
        return hi
    if float(q.steady(lo)) >= target:
        return lo
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if float(q.steady(mid)) < target:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


def max_feasible_choke(models, backoff=None, lo=0.0, hi=100.0, tol=1e-4):
    backoff = backoff or {}
    def ok(u):
        for k in PRESSURES:
            if float(models[k].steady(u)) < limits.PRESSURE_LIMITS[k] + backoff.get(k, 0.0):
                return False
        return True

    if ok(hi):
        return hi
    if not ok(lo):
        return lo
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if ok(mid):
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return lo


def binding_choke(model, limit, lo=0.0, hi=100.0, tol=1e-5):
    """Choke position at which a falling pressure output reaches its limit.

    NaN when the constraint never binds anywhere in the choke range.
    """
    if float(model.steady(hi)) >= limit:
        return float("nan")
    if float(model.steady(lo)) <= limit:
        return lo
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if float(model.steady(mid)) > limit:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


def binding_chokes(models=None):
    models = models or build_models("saturating")
    return {
        k: binding_choke(models[k], limits.PRESSURE_LIMITS[k])
        for k in PRESSURES
    }


def rate_ceiling(models=None, backoff=None):
    """Maximum steady oil rate admissible under the current pressure limits.

    Reports which constraint is responsible as well as the rate, so an
    operator is told why the well is capped and not merely that it is. Every
    limit is read from limits.PRESSURE_LIMITS at call time rather than
    captured at import, so changing a limit moves the reported ceiling and the
    reported binding constraint with no retuning anywhere.
    """
    models = models or build_models("saturating")
    backoff = dict(backoff or {})
    u = max_feasible_choke(models, backoff)
    margin = {
        k: float(models[k].steady(u)) - limits.PRESSURE_LIMITS[k] - backoff.get(k, 0.0)
        for k in PRESSURES
    }
    binding = min(PRESSURES, key=lambda k: margin[k])
    wide_open = u >= limits.CHOKE_MAX - 1e-6 and margin[binding] > 1e-6
    return {
        "choke": u,
        "rate": float(models["Q"].steady(u)),
        "binding": None if wide_open else binding,
        "binding_limit": None if wide_open else limits.PRESSURE_LIMITS[binding],
        "binding_backoff": 0.0 if wide_open else backoff.get(binding, 0.0),
        "margin": margin,
    }


def infeasibility_report(target, models=None, backoff=None, tol=1e-3):
    """Operator-facing advisory for a target the well cannot deliver.

    None when the target is achievable. Otherwise it states the achievable
    rate and the constraint responsible, at the hard limit and again under the
    controller's backoff policy, because those are different numbers: the
    first is what the well can physically give, the second is what the
    controller will actually hold. An operator asking why a 200 bbl/hr request
    is producing 161 needs both, and needs the constraint named rather than
    having to infer it from which pressure trace is flattest.
    """
    models = models or build_models("saturating")
    backoff = DEFAULT_BACKOFF if backoff is None else backoff
    hard = rate_ceiling(models, None)
    held = rate_ceiling(models, backoff)
    target = float(target)
    if target <= hard["rate"] + tol:
        return None

    if hard["binding"] is None:
        head = (f"Target {target:.1f} bbl/hr infeasible. Maximum rate "
                f"{hard['rate']:.2f} bbl/hr at a fully open choke; no pressure "
                f"constraint is active.")
    else:
        head = (f"Target {target:.1f} bbl/hr infeasible. Maximum safe rate "
                f"{hard['rate']:.2f} bbl/hr, limited by {hard['binding']} at "
                f"{hard['binding_limit']:.0f} psi.")
    if held["binding"] is None:
        tail = (f"Controller will hold {held['rate']:.2f} bbl/hr at "
                f"{held['choke']:.2f} % choke.")
    else:
        tail = (f"Controller will hold {held['rate']:.2f} bbl/hr at "
                f"{held['choke']:.2f} % choke, keeping a "
                f"{held['binding_backoff']:.0f} psi {held['binding']} backoff "
                f"margin.")
    return {
        "target": target,
        "achievable_rate": hard["rate"],
        "achievable_choke": hard["choke"],
        "binding": hard["binding"],
        "binding_limit": hard["binding_limit"],
        "held_rate": held["rate"],
        "held_choke": held["choke"],
        "held_binding": held["binding"],
        "shortfall": target - hard["rate"],
        "message": head + " " + tail,
    }


def backoff_production_cost(models=None, backoff=None):
    models = models or build_models("saturating")
    backoff = DEFAULT_BACKOFF if backoff is None else backoff
    u_hard = max_feasible_choke(models, None)
    u_soft = max_feasible_choke(models, backoff)
    q_hard = float(models["Q"].steady(u_hard))
    q_soft = float(models["Q"].steady(u_soft))
    return {
        "choke_hard": u_hard,
        "choke_backoff": u_soft,
        "choke_giveaway": u_hard - u_soft,
        "rate_hard": q_hard,
        "rate_backoff": q_soft,
        "rate_giveaway": q_hard - q_soft,
        "rate_giveaway_pct": 100.0 * (q_hard - q_soft) / q_hard if q_hard else 0.0,
        "bbl_per_day": 24.0 * (q_hard - q_soft),
    }


@dataclass
class MPCConfig:
    horizon: int = 12
    control_horizon: int = 3
    move_weight: float = 1.0
    structure: str = "saturating"
    grid_points: int = 11
    backoff: dict = field(default_factory=lambda: dict(DEFAULT_BACKOFF))
    backoff_start: int = None
    soft_weight: dict = field(default_factory=lambda: dict(DEFAULT_SOFT_WEIGHT))
    soft_move_weight: float = 0.01
    bias_gain: float = 0.3
    ss_feasible_inputs: bool = True
    ts: float = limits.TS_HOURS
    name: str = "mpc"

    def __post_init__(self):
        self.control_horizon = max(1, min(self.control_horizon, self.horizon))
        if self.backoff_start is None:
            self.backoff_start = int(round(BACKOFF_START_HOURS / self.ts))
        self.backoff_start = min(max(0, self.backoff_start), max(0, self.horizon - 1))


ABLATIONS = {
    "mpc_h12_sat": MPCConfig(horizon=12, control_horizon=3, structure="saturating",
                             ss_feasible_inputs=False, name="MPC h=12, saturating"),
    "mpc_h1_sat": MPCConfig(horizon=1, control_horizon=1, structure="saturating",
                            ss_feasible_inputs=False, name="one-step h=1, saturating"),
    "mpc_h12_lin": MPCConfig(horizon=12, control_horizon=3, structure="fopdt",
                             ss_feasible_inputs=False, name="MPC h=12, linear FOPDT"),
    "mpc_h1_lin": MPCConfig(horizon=1, control_horizon=1, structure="fopdt",
                            ss_feasible_inputs=False, name="naive h=1, linear FOPDT"),
}

PRODUCTION = MPCConfig(
    horizon=12,
    control_horizon=3,
    structure="saturating",
    ss_feasible_inputs=True,
    name="MPC h=12, saturating + SS-feasible inputs",
)


class ChokeMPC:
    """Receding-horizon choke controller.

    Sees the plant only through the measurement tuple handed to compute();
    it never imports or inspects the simulator. Prediction uses the models
    identified in identify.py.

    The one-step baseline is this same class with horizon=1 and
    control_horizon=1. Model, cost function, candidate grid, move limits,
    constraint handling, backoff, offset-free correction and the soft
    fallback are all shared. Horizon length is the only difference, so the
    2x2 ablation in ABLATIONS separates horizon length from model structure
    with nothing else varying.

    Constraints are applied in three tiers. The hard pressure limits are
    enforced over every step of the prediction horizon in all tiers,
    including the backoff tier. Only the extra backoff margin is windowed,
    enforced from backoff_start onward, because BHP has a 12.3 h time
    constant and cannot recover a 15 psi margin within one interval;
    demanding the margin immediately would make the backoff tier permanently
    infeasible and collapse the controller onto the zero-margin limit. The
    limit itself is actuatable at every step and so is never windowed. If no
    candidate satisfies the hard limits the controller switches to a soft
    mode that minimises weighted predicted violation instead of tracking
    error, so it degrades rather than throwing.

    Windowing the limit rather than the margin is a subtle and severe error.
    A tier that skips the first backoff_start steps leaves them entirely
    unconstrained whenever that tier is satisfiable, and the ladder never
    reaches the stricter tier because the looser one already succeeded. If
    backoff_start is additionally tied to the horizon the hole widens as the
    horizon grows, and prediction length becomes actively harmful: at
    horizon 36 with a half-horizon window the controller held a choke whose
    steady BHP was 2795 psi for the whole run, 166 of 200 samples in
    violation, while every plan it evaluated looked feasible.

    backoff_start is therefore a fixed physical quantity, not a fraction of
    the horizon: BACKOFF_START_HOURS is the time BHP needs to recover the
    margin after a full-rate close, about 6 h for a 15 psi margin given a
    -8.49 psi/% gain, a 5%/interval move limit and tau 12.3 h. It is the same
    number of steps for every configuration, so no cell is handicapped, and
    it is clamped to horizon-1 so a one-step controller must satisfy the
    margin immediately - a real consequence of the short horizon, and
    exactly what the ablation is meant to expose.

    ss_feasible_inputs is kept OFF in the four ABLATIONS cells, so that the
    2x2 varies horizon and model structure and nothing else. It is itself a
    form of long-range reasoning - a steady-state, i.e. infinite-horizon,
    admissibility test on each candidate input - so switching it on inside a
    horizon=1 cell would smuggle the very information the ablation is trying
    to isolate. It is reported instead as an explicit third axis, and the
    shipped controller is PRODUCTION, which has it on.

    Two additions keep the specified horizon of 12 usable. The horizon is
    only 0.97 x tau_BHP, so a 12-step prediction does not reach the steady
    state a candidate choke position implies; without help the controller
    opens to 84% during startup, sees no violation inside the window, and
    then has to retreat. ss_feasible_inputs therefore restricts candidates
    to inputs that are also feasible at steady state. Constraining only the
    terminal input is not enough: under a receding horizon the controller
    opens wide now and plans to close later, never executes the close, and
    the choke zigzags on the move limit. Requiring every planned input to be
    steady-state feasible cuts total choke travel about fivefold on the
    infeasible scenario at a ~1.5% production cost, and leaves the feasible
    scenarios unchanged because the restriction only binds near the limit.

    bias_gain low-pass filters the offset-free disturbance estimate; taking
    the raw one-step innovation makes the estimated constraint boundary jump
    with every noisy sample and the choke chatters against it.

    Both apply identically to every configuration including the one-step
    baseline.

    move_weight is deliberately low. A move penalty is asymmetric across
    horizons: a one-step controller sees only the first interval of a move's
    benefit (about 20% of it for oil rate) while paying the full penalty, so
    a weight tuned for the long horizon silently disables the short one. The
    horizon-12 cost is flat in this parameter (IAE 334-351 over move_weight
    0.5 to 20, with the infeasible scenario unchanged), so nothing is gained
    by raising it, while the one-step baseline stops tracking above about 2.
    At 1.0 both are near their best and the one-step cell actually settles
    scenario A faster than the horizon-12 cell.
    """

    def __init__(self, config=None, models=None):
        self.config = config or MPCConfig()
        self.models = models or build_models(self.config.structure, self.config.ts)
        self._alpha = {
            k: 1.0 - math.exp(-self.config.ts / max(self.models[k].tau, 1e-6))
            for k in OUTPUTS
        }
        self._candidates = self._build_candidates()
        self.u_prev = 0.0
        self._x = None
        self._bias = {k: 0.0 for k in OUTPUTS}
        self.last_info = {}
        self._ceiling = rate_ceiling(self.models, None)

    def _build_candidates(self):
        g = max(2, self.config.grid_points)
        grid = np.linspace(-limits.CHOKE_MAX_MOVE, limits.CHOKE_MAX_MOVE, g)
        m = self.config.control_horizon
        mesh = np.meshgrid(*([grid] * m), indexing="ij")
        return np.stack([a.ravel() for a in mesh], axis=1)

    def reset(self, u0, measurement=None):
        self.u_prev = float(u0)
        if measurement is None:
            self._x = {k: float(self.models[k].steady(u0)) for k in OUTPUTS}
        else:
            self._x = dict(zip(OUTPUTS, [float(v) for v in measurement]))
        self._bias = {k: 0.0 for k in OUTPUTS}
        self.last_info = {}
        self._ceiling = rate_ceiling(self.models, None)
        return self.u_prev

    def infeasibility(self, target):
        return infeasibility_report(target, self.models, self.config.backoff)

    def _u_trajectory(self, cand):
        n = cand.shape[0]
        m = self.config.control_horizon
        moves = np.cumsum(cand, axis=1) + self.u_prev
        moves = np.clip(moves, limits.CHOKE_MIN, limits.CHOKE_MAX)
        traj = np.empty((n, self.config.horizon))
        traj[:, :m] = moves[:, : min(m, self.config.horizon)]
        if self.config.horizon > m:
            traj[:, m:] = moves[:, -1][:, None]
        return traj

    def _predict(self, traj):
        n, h = traj.shape
        preds = {}
        for k in OUTPUTS:
            model = self.models[k]
            a = self._alpha[k]
            x = np.full(n, self._x[k], dtype=float)
            out = np.empty((n, h))
            for j in range(h):
                x = x + a * (np.asarray(model.steady(traj[:, j]), dtype=float) - x)
                out[:, j] = x
            preds[k] = out + self._bias[k]
        return preds

    def _feasible_mask(self, preds, backoff, start=0, u_plan=None):
        mask = np.ones(preds["Q"].shape[0], dtype=bool)
        for k in PRESSURES:
            lim = limits.PRESSURE_LIMITS[k]
            bo = backoff.get(k, 0.0)
            mask &= np.all(preds[k] >= lim, axis=1)
            if bo > 0.0 and start < preds[k].shape[1]:
                mask &= np.all(preds[k][:, start:] >= lim + bo, axis=1)
            if u_plan is not None:
                ss = np.asarray(self.models[k].steady(u_plan), dtype=float)
                mask &= np.all(ss + self._bias[k] >= lim + bo, axis=1)
        return mask

    def _tracking_cost(self, preds, cand, target):
        err = preds["Q"] - target
        return np.sum(err * err, axis=1) + self.config.move_weight * np.sum(
            cand * cand, axis=1
        )

    def _violation_cost(self, preds, cand):
        total = np.zeros(preds["Q"].shape[0])
        for k in PRESSURES:
            short = np.maximum(limits.PRESSURE_LIMITS[k] - preds[k], 0.0)
            total += self.config.soft_weight.get(k, 1.0) * np.sum(short * short, axis=1)
        return total + self.config.soft_move_weight * np.sum(cand * cand, axis=1)

    def compute(self, measurement, target):
        meas = dict(zip(OUTPUTS, [float(v) for v in measurement]))
        if self._x is None:
            self._x = dict(meas)
        g = self.config.bias_gain
        for k in OUTPUTS:
            self._bias[k] += g * ((meas[k] - self._x[k]) - self._bias[k])

        cand = self._candidates
        traj = self._u_trajectory(cand)
        preds = self._predict(traj)
        u_plan = traj if self.config.ss_feasible_inputs else None

        zero_backoff = {k: 0.0 for k in PRESSURES}
        tiers = (
            ("hard+backoff", self.config.backoff, self.config.backoff_start),
            ("hard", zero_backoff, 0),
        )
        chosen = None
        for tier_name, bo, start in tiers:
            mask = self._feasible_mask(preds, bo, start, u_plan)
            if np.any(mask):
                cost = self._tracking_cost(preds, cand, target)
                cost = np.where(mask, cost, np.inf)
                idx = int(np.argmin(cost))
                chosen = (tier_name, idx, int(mask.sum()), float(cost[idx]))
                break
        if chosen is None:
            cost = self._violation_cost(preds, cand)
            idx = int(np.argmin(cost))
            chosen = ("soft", idx, 0, float(cost[idx]))

        tier_name, idx, n_feasible, cost = chosen
        u = float(traj[idx, 0])
        u = limits.clamp_choke(u, self.u_prev)

        pred_min = {k: float(np.min(preds[k][idx])) for k in PRESSURES}
        self.last_info = {
            "mode": tier_name,
            "n_feasible": n_feasible,
            "n_candidates": int(cand.shape[0]),
            "cost": cost,
            "u_prev": self.u_prev,
            "u": u,
            "du": u - self.u_prev,
            "target": float(target),
            "predicted_q_end": float(preds["Q"][idx, -1]),
            "predicted_min": pred_min,
            "bias": dict(self._bias),
            "soft": tier_name == "soft",
            "target_infeasible": float(target) > self._ceiling["rate"] + 1e-3,
            "rate_ceiling": self._ceiling["rate"],
            "ceiling_binding": self._ceiling["binding"],
        }

        for k in OUTPUTS:
            self._x[k] = self._x[k] + self._alpha[k] * (
                float(self.models[k].steady(u)) - self._x[k]
            )
        self.u_prev = u
        return u


def make_controller(key, **overrides):
    if key not in ABLATIONS:
        raise KeyError(f"unknown ablation {key!r}; options: {sorted(ABLATIONS)}")
    base = ABLATIONS[key]
    cfg = MPCConfig(
        horizon=overrides.get("horizon", base.horizon),
        control_horizon=overrides.get("control_horizon", base.control_horizon),
        move_weight=overrides.get("move_weight", base.move_weight),
        structure=overrides.get("structure", base.structure),
        grid_points=overrides.get("grid_points", base.grid_points),
        backoff=dict(overrides.get("backoff", base.backoff)),
        backoff_start=overrides.get("backoff_start", None),
        bias_gain=overrides.get("bias_gain", base.bias_gain),
        ss_feasible_inputs=overrides.get(
            "ss_feasible_inputs", base.ss_feasible_inputs
        ),
        soft_weight=dict(overrides.get("soft_weight", base.soft_weight)),
        soft_move_weight=overrides.get("soft_move_weight", base.soft_move_weight),
        ts=overrides.get("ts", base.ts),
        name=overrides.get("name", base.name),
    )
    return ChokeMPC(cfg)


def naive_controller(structure="saturating", **overrides):
    overrides.setdefault("name", f"one-step h=1, {structure}")
    return make_controller(
        "mpc_h1_sat" if structure == "saturating" else "mpc_h1_lin", **overrides
    )
