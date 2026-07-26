"""Adapter and conformance test for the official simulator.

The official simulator is released after this submission round. Every
controller in this repository is written against one interface only:

    Q, WHP, FLP, BHP = simulator.step(choke_position)

`SimulatorAdapter` wraps any object exposing that call so it can stand in for
`plant.ChokePlant` without touching controller code, and `check_conformance`
verifies a candidate simulator against the contract the controllers rely on
before a single closed-loop run is attempted.

Substituting the official simulator is one import change per `run_*.py`:

    from plant import ChokePlant as Simulator

becomes

    from official_simulator import OfficialSim as Simulator

and, if its call signature differs at all, wrap it:

    from simulator_adapter import SimulatorAdapter
    Simulator = lambda **kw: SimulatorAdapter(OfficialSim(**kw))

Run the conformance test first:

    python src/simulator_adapter.py

or, against a candidate:

    python src/simulator_adapter.py official_simulator:OfficialSim
"""

import importlib
import inspect
import math
import sys

OUTPUT_ORDER = ("Q", "WHP", "FLP", "BHP")
PRESSURE_OUTPUTS = ("WHP", "FLP", "BHP")

CHOKE_MIN = 0.0
CHOKE_MAX = 100.0
OPERATING_RANGE = (20.0, 80.0)
SETTLE_STEPS = 150
AVERAGE_STEPS = 20
GRID_STEP = 10.0


class ConformanceError(AssertionError):
    """Raised when a candidate simulator violates the required contract."""


class SimulatorAdapter:
    """Wrap any object exposing `step(u) -> (Q, WHP, FLP, BHP)`.

    The adapter is deliberately thin: it validates at the boundary and does
    nothing else. Controllers already assume a 4-tuple of finite floats in the
    order (Q, WHP, FLP, BHP), so a simulator that returns a dict, a numpy
    array, a namedtuple, or the outputs in a different order is normalised
    here rather than by scattering conversions through the control code.

    `output_order` lets a simulator that returns the four outputs in another
    order be reordered without modifying it. If the wrapped object returns a
    mapping, keys are used directly and `output_order` selects them.

    `reset` is forwarded when the wrapped object provides it and is otherwise a
    no-op, so a stateless or self-initialising simulator needs no shim.
    """

    def __init__(self, simulator, output_order=OUTPUT_ORDER):
        if not hasattr(simulator, "step"):
            raise ConformanceError(
                f"{type(simulator).__name__} has no step() method")
        if tuple(sorted(output_order)) != tuple(sorted(OUTPUT_ORDER)):
            raise ConformanceError(
                f"output_order must be a permutation of {OUTPUT_ORDER}, "
                f"got {tuple(output_order)}")
        self._sim = simulator
        self._order = tuple(output_order)
        self._index = [self._order.index(k) for k in OUTPUT_ORDER]

    @property
    def wrapped(self):
        return self._sim

    def reset(self, *args, **kwargs):
        fn = getattr(self._sim, "reset", None)
        if fn is None:
            return None
        return self._normalise(fn(*args, **kwargs), allow_none=True)

    def step(self, choke_position):
        u = float(choke_position)
        if not math.isfinite(u):
            raise ConformanceError("choke_position must be finite")
        return self._normalise(self._sim.step(u))

    def _normalise(self, value, allow_none=False):
        if value is None:
            if allow_none:
                return None
            raise ConformanceError("step() returned None")
        if hasattr(value, "keys"):
            missing = [k for k in OUTPUT_ORDER if k not in value]
            if missing:
                raise ConformanceError(
                    f"step() mapping is missing {missing}")
            return tuple(float(value[k]) for k in OUTPUT_ORDER)
        seq = list(value)
        if len(seq) != 4:
            raise ConformanceError(
                f"step() must return 4 outputs, got {len(seq)}")
        out = tuple(float(seq[i]) for i in self._index)
        for k, v in zip(OUTPUT_ORDER, out):
            if not math.isfinite(v):
                raise ConformanceError(f"step() returned non-finite {k}: {v}")
        return out


def _instantiate(factory, **kwargs):
    if inspect.isclass(factory) or callable(factory):
        return factory(**kwargs)
    raise ConformanceError(f"{factory!r} is not callable")


def _accepts(factory, name):
    target = factory.__init__ if inspect.isclass(factory) else factory
    try:
        sig = inspect.signature(target)
    except (TypeError, ValueError):
        return False
    if any(p.kind is inspect.Parameter.VAR_KEYWORD
           for p in sig.parameters.values()):
        return True
    return name in sig.parameters


def _make(factory, **kwargs):
    sim = _instantiate(factory, **kwargs)
    return sim if isinstance(sim, SimulatorAdapter) else SimulatorAdapter(sim)


def _settle(factory, u, steps=SETTLE_STEPS, average=AVERAGE_STEPS, **kwargs):
    sim = _make(factory, **kwargs)
    sim.reset()
    tail = []
    for i in range(steps):
        y = sim.step(u)
        if i >= steps - average:
            tail.append(y)
    n = len(tail)
    return tuple(sum(row[j] for row in tail) / n for j in range(4))


def _check_returns_four_floats(factory, results):
    sim = _make(factory)
    sim.reset()
    y = sim.step(50.0)
    ok = isinstance(y, tuple) and len(y) == 4 and all(
        isinstance(v, float) and math.isfinite(v) for v in y)
    detail = ", ".join(f"{k}={v:.4g}" for k, v in zip(OUTPUT_ORDER, y))
    results.append(("returns four finite floats (Q, WHP, FLP, BHP)", ok, detail))


def _check_choke_range(factory, results):
    failures = []
    seen = {}
    for u in (CHOKE_MIN, 1.0, 50.0, 99.0, CHOKE_MAX):
        try:
            sim = _make(factory)
            sim.reset()
            y = None
            for _ in range(5):
                y = sim.step(u)
            seen[u] = y
        except Exception as exc:
            failures.append(f"u={u:g}: {type(exc).__name__}: {exc}")
    ok = not failures
    detail = ("accepted 0, 1, 50, 99, 100 %" if ok
              else "; ".join(failures))
    results.append((f"accepts u over {CHOKE_MIN:g}-{CHOKE_MAX:g} %", ok, detail))


def _check_determinism(factory, results, seed=1234):
    if not _accepts(factory, "seed"):
        results.append(("deterministic under a fixed seed", None,
                        "skipped: no seed parameter exposed"))
        return
    plan = [30.0, 35.0, 40.0, 45.0, 50.0, 45.0, 40.0] * 6
    runs = []
    for _ in range(2):
        sim = _make(factory, seed=seed)
        sim.reset()
        runs.append([sim.step(u) for u in plan])
    worst = max(abs(a[j] - b[j])
                for ra, rb in [(runs[0], runs[1])]
                for a, b in zip(ra, rb)
                for j in range(4))
    ok = worst == 0.0
    results.append(("deterministic under a fixed seed", ok,
                    f"{len(plan)} steps, seed {seed}, max |difference| "
                    f"{worst:.3e}"))


def _check_monotonicity(factory, results, lo=None, hi=None, step=GRID_STEP):
    lo = OPERATING_RANGE[0] if lo is None else lo
    hi = OPERATING_RANGE[1] if hi is None else hi
    kwargs = {"noise": False} if _accepts(factory, "noise") else {}
    if _accepts(factory, "drift"):
        kwargs["drift"] = False

    grid = []
    u = lo
    while u <= hi + 1e-9:
        grid.append(round(u, 6))
        u += step
    curves = {k: [] for k in OUTPUT_ORDER}
    for point in grid:
        y = _settle(factory, point, **kwargs)
        for k, v in zip(OUTPUT_ORDER, y):
            curves[k].append(v)

    span = f"{lo:g}-{hi:g} % in {step:g} % steps"
    dq = [b - a for a, b in zip(curves["Q"], curves["Q"][1:])]
    ok_q = all(d > 0.0 for d in dq)
    results.append((f"dQ/du > 0 over {span}", ok_q,
                    f"min increment {min(dq):+.4f} bbl/hr, "
                    f"Q {curves['Q'][0]:.2f} -> {curves['Q'][-1]:.2f}"))

    for k in PRESSURE_OUTPUTS:
        dp = [b - a for a, b in zip(curves[k], curves[k][1:])]
        ok = all(d < 0.0 for d in dp)
        results.append((f"d{k}/du < 0 over {span}", ok,
                        f"max increment {max(dp):+.4f} psi, "
                        f"{k} {curves[k][0]:.2f} -> {curves[k][-1]:.2f}"))
    return curves


def check_conformance(factory, verbose=True, lo=None, hi=None):
    """Check a candidate simulator against the contract the controllers assume.

    `factory` is anything callable that returns a fresh simulator: a class, or
    a lambda binding constructor arguments. A fresh instance is built for each
    check, so a simulator that cannot be re-instantiated will fail loudly here
    rather than silently corrupting a closed-loop run.

    Checks performed:

    1. `step(u)` returns four finite floats in the order (Q, WHP, FLP, BHP).
    2. `u` is accepted across 0-100 % without raising.
    3. Given a fixed seed the simulator is reproducible. Skipped, not failed,
       when no `seed` parameter is exposed - a deterministic simulator with no
       seed argument is conformant.
    4. The steady-state map is monotone over the operating range:
       `dQ/du > 0` and `dWHP/du`, `dFLP/du`, `dBHP/du` all `< 0`. Noise and
       drift are switched off when the simulator exposes those arguments;
       otherwise each point is settled and averaged to suppress them.

    Monotonicity is the check that matters most. The controller's steady-state
    feasibility screen and its bisection for the maximum feasible choke both
    assume a monotone map; a simulator that violates it will not merely degrade
    performance, it will make the screen select the wrong branch.

    Returns `(passed, results)` where `results` is a list of
    `(check, passed_or_None, detail)`; `None` marks a skipped check.
    """
    results = []
    _check_returns_four_floats(factory, results)
    _check_choke_range(factory, results)
    _check_determinism(factory, results)
    _check_monotonicity(factory, results, lo=lo, hi=hi)

    passed = all(r[1] is not False for r in results)
    if verbose:
        width = max(len(r[0]) for r in results)
        print("=" * 78)
        print("SIMULATOR CONFORMANCE TEST")
        print("=" * 78)
        for name, ok, detail in results:
            mark = "SKIP" if ok is None else ("PASS" if ok else "FAIL")
            print(f"  [{mark}] {name:<{width}s}  {detail}")
        print("-" * 78)
        n_pass = sum(1 for r in results if r[1] is True)
        n_skip = sum(1 for r in results if r[1] is None)
        n_fail = sum(1 for r in results if r[1] is False)
        print(f"  {n_pass} passed, {n_skip} skipped, {n_fail} failed")
        print(f"  VERDICT: {'CONFORMANT' if passed else 'NOT CONFORMANT'}")
    return passed, results


def _resolve(spec):
    module_name, _, attr = spec.partition(":")
    if not attr:
        raise SystemExit(f"expected module:attribute, got {spec!r}")
    module = importlib.import_module(module_name)
    return getattr(module, attr)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv:
        target = _resolve(argv[0])
        label = argv[0]
    else:
        from plant import ChokePlant

        target = ChokePlant
        label = "plant:ChokePlant (the surrogate shipped with this submission)"
    print(f"candidate: {label}\n")
    passed, _ = check_conformance(target)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
