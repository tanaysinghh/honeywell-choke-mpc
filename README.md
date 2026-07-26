# Autonomous Production Choke Controller

Constraint-aware MPC for a single naturally flowing oil well. Control interval
`Ts = 1 h`, choke `0-100 %`, slew limit `±5 %` per interval.

**Headline result.** When an operator requests 200 bbl/hr from a well that can
safely deliver 163, the shipped controller settles at 158 bbl/hr with **zero
constraint violations** and holds BHP at 2864 psi against a 2850 psi limit. A
tuned PI controller on the same target reaches 199 bbl/hr by driving BHP to
2676 psi — 174 psi below the limit, for 177 of 200 hours. See
`figures/04_headline_scenarioC_bhp.png`.

Robustness: **0 true violations in 300 Monte Carlo runs** with measurement
noise and independent ±20 % gain and time-constant mismatch on every output;
the envelope stays clean to ±30 %.

## Plant interface and swapping in the official simulator

The controller only ever calls

```python
Q, WHP, FLP, BHP = simulator.step(choke_position)
```

`src/plant.py` exposes exactly `step(u)` and `reset(...)`; every other attribute
is underscore-private. No controller module imports anything from `plant.py`.
Replacing the surrogate with the official simulator is a **one-line change** in
each `run_*.py` script:

```python
from plant import ChokePlant as Simulator      # <- replace this import
```

Anything the surrogate needs but the real simulator does not have (noise
settings, gain/tau mismatch for Monte Carlo) is passed to the *constructor*, not
to `step()`, so the call signature is identical either way.

`src/mpc.py` and `src/pid.py` import `limits` and `identify` only. Neither
imports `plant`, and the invariant is checkable in one command:

```
grep -l plant src/mpc.py src/pid.py      # no output
```

`src/evaluate.py` (a shared harness, not in the original file list — it exists
so `run_03/04/04b/05` do not each reimplement the closed loop and the metrics)
does import `plant`, because it *is* the test rig. It reports true, noise-free
plant state by replaying the applied choke sequence through a second
`ChokePlant` with noise and drift off. That replay is valid because plant state
depends only on the choke sequence — noise and drift are additive at the output
— and it uses nothing but `reset()` and `step()`, so the one-line swap survives.

## Identified model

Fitted to `data/reference_steptest.csv` by simulation-error minimisation over
all 120 samples (not endpoint differencing — see Finding 1). First-order, no
deadtime; second-order and deadtime 1-3 h were tested and all fit worse.

| output | gain @45 % | tau (h) | meas. sigma | R2 (reference CSV) |
|---|---|---|---|---|
| Oil rate | **+1.83** bbl/hr per % | 5.42 | 0.37 | 0.9977 |
| WHP | **−1.61** psi per % | 9.33 | 0.32 | 0.9968 |
| FLP | **−0.98** psi per % | 6.52 | 0.34 | 0.9936 |
| BHP | **−8.40** psi per % | 13.12 | 1.96 | 0.9951 |

The static map is a saturating choke characteristic — one shared normalised
`phi(u)` across all four outputs, since all four are driven by the same flow
path — so the model stays monotone and physically sensible when extrapolated to
`u = 100`. Local gain falls from 2.07 to 1.26 bbl/hr/% across 0-100 %.

Residuals sit at roughly 3x the white-noise sigma with lag-1 autocorrelation
0.77-0.86, i.e. the reference data contains a *coloured* disturbance, not just
measurement noise. The surrogate reproduces this with an Ornstein-Uhlenbeck
drift term (`tau_d = 5 h`) on top of white noise, so a noisy realisation of the
surrogate sits the same distance from the CSV as the CSV sits from its own
noise-free trajectory.

### Held-out validation

`run_02_identify.py` fits on the 630 h own step test and validates the shipped
coefficients — never refitted — open-loop against three datasets. Worst R2
across all four outputs and all three datasets is **0.9889**.

| output | R2 on own 70 h test | R2 on own 20 h test | R2 on supplied reference |
|---|---|---|---|
| Oil rate | 0.99868 | 0.99753 | 0.99559 |
| WHP | 0.99879 | 0.99801 | 0.99119 |
| FLP | 0.99685 | 0.99483 | 0.98894 |
| BHP | 0.99868 | 0.99793 | 0.99207 |

The saturating structure earns its place mainly on BHP: on the held-out
reference test it cuts BHP RMSE by **26.6 %** versus linear FOPDT (6.13 vs
8.35 psi) and oil-rate RMSE by 11.1 %, while WHP and FLP are a wash (−3.6 %,
−0.1 %) — those two are close to linear over the tested range. Since BHP is the
binding constraint, that is exactly where the extra structure is worth having.

## Operating envelope

All three pressure constraints are **lower** limits.

> **The numeric values 200 / 145 / 2850 psi are an assumption of this
> submission.** The problem statement names WHP, FLP and BHP as active
> constraints and refers to "WHP limits, FLP limits, BHP limits" but gives no
> numbers anywhere; they were presumably carried in the simulator that was
> never released. Because every result below depends on them,
> `run_06_limit_sensitivity.py` sweeps each one independently and re-runs the
> shipped controller unmodified at every point - see
> [Sensitivity to the assumed pressure limits](#sensitivity-to-the-assumed-pressure-limits).

| constraint | limit | choke at which it binds | oil rate there |
|---|---|---|---|
| **BHP** | 2850 psi | **69.00 %** | **163.0 bbl/hr** |
| WHP | 200 psi | 77.16 % | 175.4 bbl/hr |
| FLP | 145 psi | 78.25 % | 177.0 bbl/hr |

**BHP binds first, at 69.00 % choke, giving a maximum achievable safe rate of
163 bbl/hr.** WHP and FLP bind ~8 percentage points later and never become
active under correct BHP control; at the BHP-binding point FLP still has 7.5 psi
of margin and WHP has 10.9 psi.

These are recomputed from the **final** identified model (fitted on the 630 h
own step test). An earlier fit to the reference CSV alone put BHP binding at
68.90 % / 162.9 bbl/hr; the 0.1-point shift is well inside the curvature
sensitivity band below and changes no conclusion.

### The 163 bbl/hr ceiling is robust to the saturation assumption

Saturation curvature is *not identifiable* from the reference data — fitting the
curvature parameter freely drives it to its lower bound on all four outputs,
i.e. over 30-65 % the plant is indistinguishable from linear. The saturating
shape is therefore an imposed modelling assumption, adopted for safe
extrapolation above the tested range.

It does not affect the conclusion. Sweeping the curvature from pure-linear to
aggressive saturation:

Refitting the model to the 630 h step test with the curvature held fixed at each
value (simply *rescaling* the fitted coefficients would prove nothing - `phi` is
shared by BHP and rate, so rescaling it is a pure reparameterisation of `u` and
leaves the locus exactly invariant by construction):

| curvature scale | BHP binds at | max safe rate |
|---|---|---|
| linear (no saturation) | 68.75 % | 163.07 bbl/hr |
| 250 | 68.85 % | 163.06 bbl/hr |
| **140 (adopted)** | **69.00 %** | **163.05 bbl/hr** |
| 90 | 69.29 % | 163.01 bbl/hr |
| 55 | 70.20 % | 162.92 bbl/hr |

The binding *choke position* moves by 1.5 percentage points across that range,
but the maximum safe *rate* stays within **162.9-163.1 bbl/hr** - a spread of
0.1 %. The same sweep against the shorter reference CSV alone gives a wider but
equally conclusive **162.7-163.2 bbl/hr**; `limits.MAX_SAFE_RATE_CURVATURE_RANGE`
retains that wider band as the conservative union of both datasets. Oil rate and BHP
saturate together, so their locus is pinned by the data even though neither
individual curve's curvature is. The rate ceiling is an identified result; the
choke position at which it occurs is assumption-dependent.

## Controller

Receding-horizon MPC, prediction horizon 12, control horizon 3, cost = squared
tracking error on oil rate + move suppression. Candidate moves are enumerated on
an 11-point grid over `±5 %` per interval, `11^3 = 1331` candidates, evaluated
vectorised. Median solve time is **3.3 ms** against a 3 600 000 ms control
interval.

Constraints are applied as a three-tier ladder:

1. **hard + backoff** — every pressure ≥ its limit at every step of the
   horizon, *plus* a margin (WHP 5, FLP 5, BHP 15 psi) from step 6 onward;
2. **hard** — every pressure ≥ its limit at every step;
3. **soft** — if neither is satisfiable, minimise weighted predicted violation
   instead of tracking error, so the controller degrades rather than going
   infeasible.

Offset-free tracking comes from an output-disturbance estimate, low-pass
filtered (`bias_gain = 0.3`); the raw one-step innovation makes the estimated
constraint boundary jump with every noisy sample and the choke chatters.

The shipped controller adds a **steady-state feasibility test on every planned
input** (`ss_feasible_inputs`). See Finding 6 — it, not horizon length, is what
removes the startup overshoot.

### Baseline fairness

The naive one-step controller is the *same class* with `horizon=1`,
`control_horizon=1`. Model, cost function, candidate grid, move limits,
constraint ladder, backoff, offset-free correction and soft fallback are all
shared code paths. Prediction horizon is the only difference. Two things had to
be corrected to keep that true:

- **Move weight.** A move penalty is asymmetric across horizons: a one-step
  controller sees only the first interval of a move's benefit (~20 % for oil
  rate) while paying the full penalty. A weight of 20, tuned for h=12, silently
  disabled the h=1 baseline (IAE 2966 vs 346). Sweeping 0.5-20 showed the h=12
  cost is *flat* (IAE 334-351) so nothing is gained by raising it. At the chosen
  `move_weight = 1.0` the one-step cell actually settles scenario A **faster**
  than h=12 (18.2 h vs 26.2 h).
- **`ss_feasible_inputs` is off in all four ablation cells.** It is itself a
  form of long-range reasoning — a steady-state, i.e. infinite-horizon,
  admissibility test — so enabling it inside a horizon=1 cell would smuggle in
  the very information the ablation isolates. It is reported as an explicit
  third axis instead.

## Scenario results

Shipped controller, seed 0 (`run_03_scenarios.py`):

| scenario | settling | IAE | final rate | min BHP (true) | violations | production |
|---|---|---|---|---|---|---|
| A: startup 20 % → 120 bbl/hr | 9 h | 330 | 119.7 | 3041.8 | **0 / 150** | 17 763 bbl |
| B: 100 → 150 bbl/hr | 11 h | 367 | 149.3 | 2904.7 | **0 / 180** | 24 718 bbl |
| C: 200 bbl/hr (infeasible) | n/a | 8648 | 158.2 | **2864.0** | **0 / 200** | 31 352 bbl |

Scenario C never enters the soft tier — the backoff tier remains satisfiable
throughout, which is the design intent.

### Ablation, scenario C, mean of 5 seeds

Violations are counted two ways: **true** = noise-free plant state below limit,
**measured** = what the controller saw.

| controller | backoff | SS-feas | viol TRUE | viol MEAS | min BHP | production |
|---|---|---|---|---|---|---|
| MPC h=12, saturating | on | off | **0.2** | 1.6 | 2853.1 | 31 830 |
| MPC h=12, linear | on | off | **1.2** | 1.8 | 2852.4 | 31 850 |
| one-step h=1, saturating | on | off | **103.0** | 94.6 | 2841.2 | 32 378 |
| naive h=1, linear | on | off | **104.8** | 95.6 | 2841.2 | 32 375 |
| MPC h=12, saturating | on | **on** | **0.0** | 0.0 | 2864.4 | 31 287 |
| PI on oil rate | n/a | n/a | **179.0** | 180.0 | 2676.3 | 38 605 |

Scenarios A and B are unconstrained in practice — every controller including PI
records zero violations — so they measure tracking, not safety. There the MPC
leads on IAE (A: 340 vs 362 one-step vs 562 PI; B: 368 vs 402 vs 583) but the
one-step cell settles A faster. **The safety argument rests entirely on
scenario C.**

Measured violations *undercount* for an unsafe controller (one-step: 103 true
vs 95 measured) and *overcount* for a safe one (h=12: 0.2 true vs 1.6 measured,
noise dipping across a limit the plant respects). Reporting measured violations
alone would flatter the wrong controller.

## Findings

### 1. Endpoint differencing understates the BHP gain by ~28 %

Differencing segment endpoints gives a BHP gain of −6.08 psi/%, against
−8.40 psi/% from full-trajectory regression — a 28 % understatement (`identify.endpoint_gain`; an earlier ad-hoc estimate put this at ~23 %). The cause
is that the reference step test holds each choke level for 20-30 h while
`tau_BHP = 13.1 h`, so BHP has only reached 78-90 % of its final value when the
next step is applied. The error is systematically toward *underestimating* the
gain, which is the dangerous direction: it makes the well look less
BHP-constrained than it is and puts the apparent limit well past the
true 69.0 % choke / 163 bbl/hr.

`run_01_steptest.py` reproduces the mechanism under controlled conditions by
sweeping segment duration (`figures/01_endpoint_gain_bias.png`):

| segment duration | 10 h | 15 h | 20 h | 30 h | 50 h | 70 h |
|---|---|---|---|---|---|---|
| BHP gain bias | −33 % | −19 % | −12 % | −5 % | −2 % | −1 % |

Bias falls below 2 % only beyond ~4x tau. The fast outputs are far less
affected at the same durations (oil rate −4 %, FLP −4 % at 20 h), which is why
the distortion shows up specifically on the constraint that matters.

### 2. The reference step test violates the ±5 %/interval move limit

The reference sequence steps 30→40→55→45→65 %, i.e. moves of 10, 15, 10 and
20 % in a single 1 h interval, all exceeding the stated ±5 % limit. The limit
therefore **cannot** be a plant/actuator property — enforcing it inside
`step()` would make the supplied reference data unreproducible.

Consequently the slew rate lives in the **controller**, as
`limits.CHOKE_MAX_MOVE`, applied via `limits.clamp_choke(u, u_prev)`.
`plant.step()` clamps only to the physical range `[0, 100]`. Own step tests in
`run_01_steptest.py` likewise use true steps, matching the reference protocol
and keeping the identification excitation clean.

### 3. Constraint ordering: BHP first at 69.0 %, WHP and FLP at 77-78 %

BHP is the only constraint a correctly designed controller ever needs to
respect actively; WHP (77.16 %) and FLP (78.25 %) bind roughly 8 points
further open. They are still enforced as hard constraints in the MPC, but they
should never become active — if they do, it indicates BHP handling has failed
or the plant has drifted from the identified model. Across every closed-loop
run in this repository, WHP and FLP never bound.

### 4. Window the *margin*, never the *limit*

The backoff margin cannot be demanded immediately: BHP has a 12.3 h time
constant and cannot recover 15 psi within one 1 h interval, so a
margin-from-step-0 rule makes the backoff tier permanently infeasible and
collapses the controller onto the zero-margin limit. The margin therefore
applies only from `backoff_start` onward.

The first implementation windowed the **whole constraint** rather than just the
margin, i.e. tier 1 checked `pressure >= limit + margin` on `[backoff_start:]`
and checked nothing at all on `[0:backoff_start]`. Because the ladder returns as
soon as a tier is satisfiable, the stricter full-horizon tier was never reached,
and the early steps were left completely unconstrained.

With `backoff_start` additionally tied to the horizon (`horizon // 2`) the hole
widened as the horizon grew, and **prediction length became actively harmful**:

| horizon | unconstrained steps | true violations of 200 | min BHP |
|---|---|---|---|
| 12 | 6 | 0.2 | 2853 |
| 24 | 12 | 55.4 | 2837 |
| 36 | 18 | **166.2** | **2813** |

At horizon 36 the controller held a choke whose steady-state BHP was 2795 psi
for the entire run while every plan it evaluated looked feasible. The bug is
invisible at the specified horizon of 12 and only appears when the horizon is
swept — which is the argument for sweeping it.

Fixed in two parts: hard limits are now enforced at **every** step in **every**
tier, and `backoff_start` is a fixed physical quantity
(`BACKOFF_START_HOURS = 6`, the time BHP needs to recover a 15 psi margin after
a full-rate close) rather than a fraction of the horizon. It is the same number
of steps for every configuration, so no cell is handicapped. Results at
horizon 12 are bit-identical before and after the fix.

### 5. Violations collapse at a prediction horizon of ~1 x tau_BHP

`run_04b_horizon_sweep.py`, scenario C, backoff on, SS-feasibility off,
5 seeds. `tau_BHP = 12.34 h`.

| horizon | h / tau | viol TRUE | peak choke | choke travel | solve time |
|---|---|---|---|---|---|
| 1 | 0.08 | 103.0 | 99.2 % | 453 % | 0.24 ms |
| 6 | 0.49 | 89.8 | 88.4 % | 354 % | 1.4 ms |
| **12** | **0.97** | **0.2** | 85.4 % | 356 % | 3.3 ms |
| 18 | 1.46 | 0.0 | 83.2 % | 410 % | 5.0 ms |
| 24 | 1.94 | 0.0 | 80.2 % | 405 % | 6.0 ms |
| 36 | 2.92 | 0.0 | 78.2 % | 375 % | 8.1 ms |
| 48 | 3.89 | 0.4 | 78.2 % | 365 % | 15.4 ms |
| 60 | 4.86 | 0.0 | 78.0 % | 375 % | 19.3 ms |

Violations fall off a cliff between 0.5 and 1.0 tau and are gone by 1.5 tau.
The specified horizon of 12 sits at 0.97 tau — right on the knee, and the
correct choice. Solve time grows linearly with the horizon and is irrelevant at
every point: the worst case, 19 ms, is 5 parts per million of the control
interval.

### 6. A longer horizon fixes violations but *not* overshoot

The same sweep shows peak choke plateauing at **78 %** no matter how long the
horizon gets — still 9 points past the 69.0 % BHP-binding position. Horizon
length cannot fix this, for a structural reason: with `control_horizon = 3` the
plan is `u0, u1, u2` held thereafter, so a candidate can open **+5 % now** and
promise to close over the next two intervals. The held tail is steady-state
feasible, the plan passes, only the first move is executed, and next interval
the controller re-plans and does it again. Peak choke exceeds the feasible
position by roughly `2 x 5 %` — 69.0 + 10 ≈ 79 %, matching the observed 78 %. It
is receding-horizon gaming, and it is invariant to horizon length.

Requiring **every planned input** — not merely the terminal one — to be
steady-state feasible removes it:

| | peak choke | choke travel | viol TRUE | production |
|---|---|---|---|---|
| horizon 12, SS-feasibility off | 85.4 % | 356 % | 0.2 | 31 830 bbl |
| horizon 60, SS-feasibility off | 78.0 % | 375 % | 0.0 | 31 850 bbl |
| **horizon 12, SS-feasibility on** | **68.2 %** | **63 %** | **0.0** | 31 287 bbl |

Constraining only the *terminal* input is not enough — that was tried first and
is exactly the loophole above. The full-plan version cuts choke travel
**5.7-fold** at a 1.7 % production cost, and leaves scenarios A and B
untouched, because the restriction only binds near the constraint.

### 7. The backoff is necessary at any horizon, but only *actuatable* at a long one

Zero-backoff ablation across all four cells, scenario C, 5 seeds:

| controller | backoff off | backoff on | change |
|---|---|---|---|
| MPC h=12, saturating | 90.2 | **0.2** | −99.8 % |
| MPC h=12, linear | 91.8 | **1.2** | −98.7 % |
| one-step h=1, saturating | 103.8 | 103.0 | −0.8 % |
| naive h=1, linear | 105.0 | 104.8 | −0.2 % |

With no margin anywhere, every cell violates heavily — a long horizon alone
buys only ~13 % (90 vs 104). The margin is what produces safety, and the
one-step controller **cannot use it**: it changes nothing at h=1 and changes
138-fold at h=12. A one-step controller cannot plan the several intervals of
closing needed to open a 15 psi margin on a 12.3 h time constant, so the
requirement is simply unsatisfiable for it and the ladder drops to the
zero-margin tier every interval.

This also disposes of the fairness objection directly: with the backoff removed
from every controller, the h=12 cells are *still* better than the h=1 cells, so
the backoff rule is not what is producing the gap.

The margin is not free. It costs **2.08 points of choke** (69.00 → 66.91 %) and
**3.28 bbl/hr** (163.05 → 159.77, −2.01 %, −79 bbl/day) of steady production.
That is the price of the safety result and it should be stated as such.

## Step-test design

`run_01_steptest.py` runs a 630 h open-loop test: levels
20 → 35 → 50 → 65 → 80 → 65 → 50 → 35 → 20 %, **70 h per segment
(5.3 x tau_BHP)**, covering increasing and decreasing steps across 20-80 %.
Verified settling: BHP moves less than 0.25 % of its step span over the final
5 h of every segment. A 20 h-per-segment variant is generated alongside it
purely to demonstrate Finding 1.

Identification on the long test (R2 against the saturating structure): oil rate
0.9987, WHP 0.9988, FLP 0.9969, BHP 0.9987.

The test deliberately drives to 80 % choke, past the envelope, to excite the
saturating region — 13 % of samples violate a constraint by design. This is an
open-loop characterisation run, not a control result.

## Robustness

`run_05_montecarlo.py`, shipped controller, 100 runs per case. Every run has
measurement noise and coloured process drift; mismatch is drawn independently
per output for both gain and time constant.

At **±20 %** mismatch, 100 runs per scenario:

| scenario | runs with any true violation | worst true BHP | production |
|---|---|---|---|
| A | **0 / 100** | 2860.6 | 17 765 ± 128 bbl |
| B | **0 / 100** | 2861.3 | 23 949 ± 1265 bbl |
| C | **0 / 100** | 2858.2 | 30 513 ± 3653 bbl |

The closest any of the 300 runs came to the limit was **+8.2 psi**.

Mismatch sweep on scenario C gives the robustness envelope:

| mismatch | runs with viol (TRUE) | (MEASURED) | worst depth | mean production |
|---|---|---|---|---|
| ±0 % | 0 % | 5 % | 0.00 psi | 31 285 bbl |
| ±5 % | 0 % | 3 % | 0.00 psi | 31 441 bbl |
| ±10 % | 0 % | 1 % | 0.00 psi | 31 247 bbl |
| ±20 % | 0 % | 2 % | 0.00 psi | 30 513 bbl |
| ±30 % | 0 % | 4 % | 0.00 psi | 29 727 bbl |
| ±40 % | **8 %** | 16 % | 7.86 psi | 28 901 bbl |

**The controller tolerates ±30 % gain and time-constant error with zero true
violations.** It first breaks at ±40 %, and even then degrades gracefully: 8 %
of runs touch the limit, the 95th-percentile excursion is 0.23 psi and the
deepest across 100 runs is 7.86 psi — still inside the 15 psi margin the
controller had reserved. The margin absorbs the failure rather than the well
doing so.

Two things worth noting. The measured violation rate is non-zero even at
**zero** mismatch (5 % of runs) — those are noisy samples, not excursions,
which is why true and measured are reported separately throughout. And
production variance grows steadily with mismatch (sd 54 → 6273 bbl) while
safety does not degrade at all until ±40 %: uncertainty is paid for in
*throughput*, not in constraint violations, which is the correct trade for this
asset.

## Sensitivity to the assumed pressure limits

The three numeric limits are the largest undocumented assumption in this
submission, so they are treated as an input and swept rather than defended.
`run_06_limit_sensitivity.py` varies each limit independently over a plausible
range with the other two at their shipped values, and at every point re-runs
**the shipped controller with no change of any kind** - no retuning, no
re-identification, no change to the backoff, the horizon or the feasibility
screen. 31 limit sets x scenarios A, B, C x 5 seeds = **82 150 controlled
hours**.

| BHP limit | binds first | binding choke | max safe rate | true violations |
|---|---|---|---|---|
| 2800 | BHP | 76.17 % | 173.98 | 0 |
| 2820 | BHP | 73.25 % | 169.61 | 0 |
| **2850** | **BHP** | **69.00 %** | **163.05** | **0** |
| 2880 | BHP | 64.86 % | 156.49 | 0 |
| 2900 | BHP | 62.17 % | 152.12 | 0 |

| WHP limit | binds first | binding choke | max safe rate | true violations |
|---|---|---|---|---|
| 180 | BHP | 69.00 % | 163.05 | 0 |
| **200** | **BHP** | **69.00 %** | **163.05** | **0** |
| 210 | BHP | 69.00 % | 163.05 | 0 |
| 215 | **WHP** | 66.07 % | 158.42 | 0 |
| 220 | **WHP** | 62.56 % | 152.75 | 0 |

| FLP limit | binds first | binding choke | max safe rate | true violations |
|---|---|---|---|---|
| 135 | BHP | 69.00 % | 163.05 | 0 |
| **145** | **BHP** | **69.00 %** | **163.05** | **0** |
| 152.5 | BHP | 69.00 % | 163.05 | 0 |
| 155 | **FLP** | 66.04 % | 158.38 | 0 |
| 160 | **FLP** | 60.32 % | 149.05 | 0 |

`figures/06_limit_sensitivity.png`; full grid in `data/limit_sensitivity.csv`.

**The claim holds.** Across the whole plausible limit space:

- **0 true violations** and **0 measured violations** in 82 150 controlled hours
- worst true excursion depth **0.000 psi**
- the smallest true margin ever held was **+3.60 psi** (WHP, at a WHP limit of
  220 psi, scenario C)
- the maximum safe rate moves over **149.05-173.98 bbl/hr**, a 17 % spread
- the *active constraint* changes identity: BHP at 26 of the 31 points, WHP at
  2, FLP at 3

The specific numbers change; the method does not. 163 bbl/hr is a consequence
of the assumed limits, not a property of the well, and the README should be
read that way. What survives the sweep is the structure: BHP binds first over
most of the plausible space, the binding choke position and the rate ceiling
track the limit smoothly and predictably, and the controller stays feasible
throughout.

### A different active constraint, without retuning

The sweep deliberately includes cases where the constraint the controller was
designed around is *not* the one that binds. At WHP >= 215 psi and FLP >= 155
psi, BHP is no longer first. Nothing in the controller is specialised to BHP -
the prediction, the three-tier ladder, the backoff and the steady-state
feasibility screen all iterate over `limits.PRESSURE_ORDER` and read
`limits.PRESSURE_LIMITS` at call time - so the active constraint simply
changes and the controller tracks it.

At **FLP = 160 psi**, FLP binds at 60.32 % choke and the ceiling falls to
149.05 bbl/hr. That is below scenario B's 150 bbl/hr target, so a scenario that
is feasible as shipped becomes infeasible, and the controller detects it and
says so:

```
scenario B, target 150 bbl/hr: settled 139.39 bbl/hr, min BHP 2951.5 psi,
  0 true violations in 900 h
  ADVISORY: Target 150.0 bbl/hr infeasible. Maximum safe rate 149.05 bbl/hr,
  limited by FLP at 160 psi. Controller will hold 139.72 bbl/hr at 54.82 %
  choke, keeping a 5 psi FLP backoff margin.
```

The advisory names FLP, not BHP, with no code change.

### Operator advisory on infeasible targets

`mpc.infeasibility_report(target)` returns `None` when a target is achievable
and otherwise an operator-facing message naming both the achievable rate and
the constraint responsible. It reports two numbers deliberately:

- the **hard-limit** ceiling, what the well can physically deliver
- the rate the controller **will actually hold**, which is lower because of the
  backoff margin

An operator asking "why am I asking for 200 and getting 160" needs the second
number and needs the constraint named, rather than having to infer it from
which pressure trace looks flattest. Shown in the scenario C output of
`run_03_scenarios.py`:

```
--- Scenario C: 200 bbl/hr requested (infeasible) ---
  operator requests 200 bbl/hr; BHP limits the well to ~163
  ADVISORY: Target 200.0 bbl/hr infeasible. Maximum safe rate 163.05 bbl/hr,
  limited by BHP at 2850 psi. Controller will hold 159.77 bbl/hr at 66.91 %
  choke, keeping a 15 psi BHP backoff margin.
```

Every step also carries `target_infeasible`, `rate_ceiling` and
`ceiling_binding` in `ChokeMPC.last_info`, so a host system can raise the
condition without polling a separate function.

## Conformance to the eight Simulator Assumptions

The problem statement's Simulator Assumptions are the closest thing to a
specification available for a simulator that was never released, so conformance
to them is part of the defence of the surrogate. Verified in
`notebook/Autonomous_Choke_Control.ipynb`, Appendix A (executed, with output).

| # | Assumption | Status | Evidence |
|---|---|---|---|
| 1 | Single well | **Conforms** | One scalar state per output, a single `_x`/`_d` pair, no well index or per-well dimension. |
| 2 | Naturally flowing | **Conforms** | No energy-input term. `dQ/du > 0` and `dWHP/du`, `dFLP/du`, `dBHP/du` all `< 0` across the whole range - opening the choke raises rate and draws every pressure down, as for a well on reservoir energy. |
| 3 | Single choke | **Conforms** | `step(self, choke_position)` - exactly one scalar actuator. |
| 4 | No gas lift | **Conforms** | No gas injection input or state; token `gas` absent from `src/plant.py`. |
| 5 | No ESP | **Conforms** | No pump speed input, no added-head term; tokens `esp`, `pump` absent. |
| 6 | No facility network | **Conforms** | FLP is an output of the *same single flow path* via the shared `phi(u)`, not a shared node with backpressure coupling. No manifold or commingling state. |
| 7 | No changing reservoir properties | **Conforms - verified** | See below. |
| 8 | No changing GOR / water cut | **Conforms** | No phase-split, GOR or water-cut state. Oil rate is a direct time-invariant function of choke; tokens `gor`, `water` absent. |

### Assumption 7 verified in detail

The surrogate carries an Ornstein-Uhlenbeck term. For the assumption to hold it
must be an **unmeasured output disturbance** - zero-mean and mean-reverting
around a **fixed** static map - not a drift in the plant. If it drifted the map,
the assumption would be violated and the term would have to be removed.

**Analytically.** `d[t+1] = rho*d[t] + sqrt(1-rho^2)*w`, `rho = exp(-Ts/tau_d) = 0.8187`,
`w ~ N(0, sigma)`:

- `|rho| < 1` so the process is stationary and mean-reverting;
- `E[d] = rho*E[d]` gives `E[d] = 0` as the unique fixed point;
- `Var[d] = rho^2*Var[d] + (1-rho^2)*sigma^2` gives `Var[d] = sigma^2` exactly;
- `d[0] ~ N(0, sigma)`, i.e. initialised *at* the stationary distribution, so
  there is no burn-in transient either.

**Structurally.** The state recursion is `x <- x + alpha*(steady(u) - x)`, and
`steady(u)` depends only on fixed constants and `u`. The disturbance appears
**only** in `_measure()`, as `y = x + d + noise`. It never enters the state update.

**Empirically:**

| test | result |
|---|---|
| Static map after 100 000 drift-on steps | **bitwise identical** (steady BHP at 68.9 %: 2849.9767460431 both times) |
| `max abs(x with drift - x without drift)`, 400 steps, same input | **0.000e+00**, while `max abs(d)` reached 16.7 |
| Mean of `d` over 400 000 steps at fixed choke | within **1.3 standard errors of zero** on all four outputs |
| Lag-1 autocorrelation of `d` | 0.8175-0.8212 vs `exp(-Ts/tau_d) = 0.8187` |
| Sample sd of `d` | matches `DRIFT_SIGMA` to <1 % on all four outputs |
| Linear trend in `d` per 100 000 steps | 0.0003-0.0056, i.e. none |
| First-half vs second-half mean of `d` | agree to <0.02 |
| Long-run mean of the **output** vs the fixed map value | agree to **0.03 psi** on BHP |

**Confirmed: a zero-mean, mean-reverting, stationary unmeasured output
disturbance around a fixed static map. It does not drift the plant. Assumption 7
holds; no change required.**

### Declared model-validity caveat

One honest limitation, a calibration-range artifact rather than an assumption
violation: the static map is calibrated over **20-80 % choke**, and extrapolating
to a fully closed choke gives `Q(0) = 23.25 bbl/hr` rather than zero. A real
choke at 0 % passes no flow. This affects no result here - the controller
operates between 20 % and 100 % and never approaches 0 % in any scenario - but
the surrogate should not be used below its calibration range without adding a
physical shut-in condition.

## Outputs

| file | contents |
|---|---|
| `data/steptest_long.csv` | 630 h step test, 70 h segments |
| `data/steptest_short.csv` | 180 h step test, 20 h segments |
| `data/endpoint_gain_bias.csv` | gain bias vs segment duration |
| `data/identification_validation.csv` | fit and held-out validation, per output |
| `data/scenario_{A,B,C}.csv` | closed-loop trends, shipped controller |
| `data/scenario_summary.csv` | per-scenario metrics |
| `data/baseline_comparison.csv` | full sweep: 4 cells x 3 scenarios x backoff x SS-feas, + PI |
| `data/horizon_sweep.csv` | horizon 1-60, scenarios A and C |
| `data/montecarlo_runs.csv` | 700 Monte Carlo runs, one row each |
| `data/montecarlo_envelope.csv` | robustness envelope vs mismatch magnitude |
| `data/limit_sensitivity.csv` | 31 limit sets x 3 scenarios, envelope + violations |
| `figures/01_steptest_long.png` | step test trends, limits dashed |
| `figures/01_steptest_short.png` | same, undersettled variant |
| `figures/01_endpoint_gain_bias.png` | Finding 1 evidence |
| `figures/02_fit_long.png` | model vs data and residuals, own step test |
| `figures/02_fit_reference.png` | same, supplied reference test |
| `figures/03_scenario_{A,B,C}.png` | six trends per scenario, limits dashed |
| **`figures/04_headline_scenarioC_bhp.png`** | **headline: MPC vs one-step vs PI on BHP** |
| `figures/04_metrics_table.png` | metrics table, colour-coded by true violations |
| `figures/04b_horizon_sweep.png` | Findings 5 and 6 |
| `figures/05_montecarlo.png` | min-BHP and production distributions at ±20 % |
| `figures/05_robustness_envelope.png` | Finding: envelope vs mismatch magnitude |
| `figures/04c_fairness_zero_backoff.png` | zero-backoff fairness evidence |
| `figures/06_limit_sensitivity.png` | max safe rate and margin held vs each assumed limit |
| **`notebook/Autonomous_Choke_Control.ipynb`** | **primary submission artifact, executed** |

All figures are 150 dpi PNGs in `figures/`.

## Running

```
python src/plant.py             # surrogate vs reference CSV, RMSE table
python run_01_steptest.py       # step tests, settling check, gain-bias sweep
python run_02_identify.py       # fit + held-out validation, R2 per output
python run_03_scenarios.py      # scenarios A, B, C with the shipped controller
python run_04_baseline_compare.py   # ablation + PI, metrics table, headline figure
python run_04b_horizon_sweep.py     # prediction horizon 1-60
python run_05_montecarlo.py         # 700 runs, robustness envelope
python run_06_limit_sensitivity.py  # sweep the assumed pressure limits
```

The primary submission artifact is `notebook/Autonomous_Choke_Control.ipynb`,
which reads top to bottom as an analysis and is committed with all outputs and
figures executed and embedded. It recomputes the fast results live and embeds
the pre-rendered figures for the expensive sweeps, each reproducible with one
script above.

`run_04_baseline_compare.py` and `run_05_montecarlo.py` take a few minutes each;
everything else is seconds.

## Source layout

| file | role |
|---|---|
| `src/plant.py` | behavioural surrogate; exposes only `step()` and `reset()` |
| `src/limits.py` | operating envelope, identified constants, violation checks |
| `src/identify.py` | FOPDT and saturating model fitting, R2, residual plots |
| `src/mpc.py` | receding-horizon MPC, ablation configs, shipped config |
| `src/pid.py` | velocity-form PI baseline on oil rate, no constraint awareness |
| `src/evaluate.py` | shared closed-loop harness, scenarios, metrics, true-state replay |
