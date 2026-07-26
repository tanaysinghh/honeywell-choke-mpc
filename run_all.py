import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT / "data" / "logs"

STAGES = (
    ("run_01_steptest.py", "step tests, settling check, gain-bias sweep"),
    ("run_02_identify.py", "identification, held-out validation"),
    ("run_03_scenarios.py", "scenarios A, B, C with the shipped controller"),
    ("run_04_baseline_compare.py", "ablation + PI, metrics table, headline figure"),
    ("run_04b_horizon_sweep.py", "prediction horizon 1-60"),
    ("run_05_montecarlo.py", "700 Monte Carlo runs, robustness envelope"),
    ("run_06_limit_sensitivity.py", "sensitivity to the assumed pressure limits"),
)


def human(seconds):
    if seconds < 60.0:
        return f"{seconds:.1f} s"
    return f"{int(seconds // 60)} m {seconds % 60:04.1f} s"


def run_stage(script, note, index, total):
    label = f"[{index}/{total}] {script}"
    print(f"{label:<44s} {note}")
    sys.stdout.flush()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log = LOG_DIR / f"{Path(script).stem}.log"
    t0 = time.perf_counter()
    with open(log, "w", encoding="utf-8") as fh:
        proc = subprocess.run([sys.executable, str(ROOT / script)], cwd=str(ROOT),
                              stdout=fh, stderr=subprocess.STDOUT, text=True)
    dt = time.perf_counter() - t0
    status = "ok" if proc.returncode == 0 else f"FAILED (exit {proc.returncode})"
    print(f"{'':<44s} -> {status} in {human(dt)}   log: {log.relative_to(ROOT)}")
    sys.stdout.flush()
    return proc.returncode, dt, log


def main():
    print("=" * 78)
    print("AUTONOMOUS CHOKE CONTROL - FULL REPRODUCTION")
    print("=" * 78)
    print(f"python {sys.version.split()[0]}")
    print(f"root   {ROOT}")
    print(f"stages {len(STAGES)}; per-stage output is written to "
          f"{LOG_DIR.relative_to(ROOT)}/")
    print("-" * 78)

    total = time.perf_counter()
    timings = []
    for i, (script, note) in enumerate(STAGES, start=1):
        code, dt, log = run_stage(script, note, i, len(STAGES))
        timings.append((script, dt))
        if code != 0:
            print("-" * 78)
            print(f"ABORTED at {script}. Last 25 lines of {log.name}:")
            print("-" * 78)
            tail = log.read_text(encoding="utf-8").splitlines()[-25:]
            print("\n".join(tail))
            return code
    elapsed = time.perf_counter() - total

    print("-" * 78)
    print("all stages completed")
    width = max(len(s) for s, _ in timings)
    for script, dt in sorted(timings, key=lambda p: -p[1]):
        share = 100.0 * dt / elapsed if elapsed else 0.0
        print(f"  {script:<{width}s}  {human(dt):>12s}  {share:5.1f} %")
    print("-" * 78)
    print(f"TOTAL RUNTIME: {human(elapsed)}")
    print(f"figures in {ROOT / 'figures'}")
    print(f"data in    {ROOT / 'data'}")
    print("\nthe notebook is committed already executed; to re-execute it:")
    print("  jupyter nbconvert --to notebook --execute --inplace \\")
    print("      notebook/Autonomous_Choke_Control.ipynb")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
