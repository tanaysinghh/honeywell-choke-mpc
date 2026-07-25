# Autonomous Production Choke Controller

Constraint-aware MPC for a single naturally flowing oil well. Control interval
`Ts = 1 h`, choke `0-100 %`, slew limit `±5 %` per interval.

Status: plant surrogate and identification complete (`src/plant.py`,
`src/limits.py`, `src/identify.py`, `run_01_steptest.py`). Controllers and
scenario studies to follow.

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

## Operating envelope

All three pressure constraints are **lower** limits.

| constraint | limit | choke at which it binds | oil rate there |
|---|---|---|---|
| **BHP** | 2850 psi | **68.90 %** | **162.9 bbl/hr** |
| WHP | 200 psi | 77.33 % | 175.7 bbl/hr |
| FLP | 145 psi | 78.16 % | 177.0 bbl/hr |

**BHP binds first, at 68.90 % choke, giving a maximum achievable safe rate of
163 bbl/hr.** WHP and FLP bind ~8.5 percentage points later and never become
active under correct BHP control; at the BHP-binding point FLP still has 7.5 psi
of margin and WHP has 11.3 psi.

### The 163 bbl/hr ceiling is robust to the saturation assumption

Saturation curvature is *not identifiable* from the reference data — fitting the
curvature parameter freely drives it to its lower bound on all four outputs,
i.e. over 30-65 % the plant is indistinguishable from linear. The saturating
shape is therefore an imposed modelling assumption, adopted for safe
extrapolation above the tested range.

It does not affect the conclusion. Sweeping the curvature from pure-linear to
aggressive saturation:

| curvature scale | BHP binds at | max safe rate |
|---|---|---|
| linear (no saturation) | 67.70 % | 162.7 bbl/hr |
| 250 | 68.33 % | 162.8 bbl/hr |
| **140 (adopted)** | **68.90 %** | **162.9 bbl/hr** |
| 90 | 69.75 % | 163.0 bbl/hr |
| 55 | 71.68 % | 163.2 bbl/hr |

The binding *choke position* moves by 4 percentage points across that range, but
the maximum safe *rate* stays within **162.7-163.2 bbl/hr**. Oil rate and BHP
saturate together, so their locus is pinned by the data even though neither
individual curve's curvature is. The rate ceiling is an identified result; the
choke position at which it occurs is assumption-dependent.

## Findings

### 1. Endpoint differencing understates the BHP gain by ~23 %

Differencing segment endpoints gives a BHP gain near −6.5 psi/%, against
−8.40 psi/% from full-trajectory regression — a 23 % understatement. The cause
is that the reference step test holds each choke level for 20-30 h while
`tau_BHP = 13.1 h`, so BHP has only reached 78-90 % of its final value when the
next step is applied. The error is systematically toward *underestimating* the
gain, which is the dangerous direction: it makes the well look less
BHP-constrained than it is and puts the apparent limit at 71 % choke /
167 bbl/hr instead of the true 68.9 % / 163 bbl/hr.

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

### 3. Constraint ordering: BHP first at 68.9 %, WHP and FLP at 77-78 %

BHP is the only constraint a correctly designed controller ever needs to
respect actively; WHP (77.33 %) and FLP (78.16 %) bind roughly 8.5 points
further open. They are still enforced as hard constraints in the MPC, but they
should never become active — if they do, it indicates BHP handling has failed
or the plant has drifted from the identified model.

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

## Outputs

| file | contents |
|---|---|
| `data/steptest_long.csv` | 630 h step test, 70 h segments |
| `data/steptest_short.csv` | 180 h step test, 20 h segments |
| `data/endpoint_gain_bias.csv` | gain bias vs segment duration |
| `figures/01_steptest_long.png` | step test trends, limits dashed |
| `figures/01_steptest_short.png` | same, undersettled variant |
| `figures/01_endpoint_gain_bias.png` | Finding 1 evidence |

All figures are 150 dpi PNGs in `figures/`.

## Running

```
python src/plant.py          # surrogate vs reference CSV, RMSE table
python run_01_steptest.py    # step tests, settling check, gain-bias sweep
```
