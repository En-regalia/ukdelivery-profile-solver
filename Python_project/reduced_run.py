# reduced_run.py - reruns the pipeline on a reduced-volume orders file from scenario_reduction.py.
# Keep it in the same folder as the scripts.
#
# Steps: build matrix, anneal, sweep, planning estimate. Depot assignment is skipped: the
# existing depot assignment file (DEPOT_ASSIGNED in .env) is reused as it is.
#
# Run, for example:
#   python reduced_run.py "C:/.../Volume reduction/Run_2026-11-27_10-00/Reduced orders/Orders_Blanket_10_2026-11-27_10-00.csv"
# or with no argument, and paste the path when asked.
#
# The reduced file stands in for both ORDERS_PATH (build_matrix) and C_OUTPUT_PATH (sweep and
# planning estimate) for this run only; .env is not changed. Everything this run writes goes
# into its own folder, so the full-volume matrix and runs are left alone:
#   <scenario Run folder>/Rerun_<reduced file name>/
#       Matrix/            matrix and unassigned-areas check built from the reduced orders
#       Anneal output/     Run_<stamp>/ as anneal_regional writes it
#       Sweep output/
#       Planning output/
# Use --out to put that folder somewhere else.
import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()                      # read .env before any script reads its settings


def clean(text):
    return (text or "").strip().strip('"').strip("'")


def parse_args():
    p = argparse.ArgumentParser(
        description="Rerun matrix, anneal, sweep and planning estimate on a reduced-volume orders file.")
    p.add_argument("orders", nargs="?",
                   help="reduced orders CSV (Reduced orders/Orders_<mode>_<pct>_<stamp>.csv). "
                        "Asked for if left out.")
    p.add_argument("--out", help="folder for this run's outputs "
                                 "(default: Rerun_<file name> in the scenario Run folder)")
    return p.parse_args()


def ask_orders_path():
    while True:
        path = Path(clean(input("Please enter path to the reduced orders csv file: ")))
        if path.is_file():
            return path
        print(f"File not found: {path}")


def default_out_dir(orders):
    """Rerun_<stem> next to the "Reduced orders" folder, i.e. inside the scenario Run folder."""
    base = orders.parent.parent if orders.parent.name == "Reduced orders" else orders.parent
    return base / f"Rerun_{orders.stem}"


args = parse_args()
ORDERS = Path(clean(args.orders)) if args.orders else ask_orders_path()
if not ORDERS.is_file():
    kind = "a folder" if ORDERS.is_dir() else "not found"
    raise SystemExit(f"Reduced orders file {kind}: {ORDERS}")
ORDERS = ORDERS.resolve()

DEPOT_ASSIGNED = Path(clean(os.getenv("DEPOT_ASSIGNED")))
if not clean(os.getenv("DEPOT_ASSIGNED")) or not DEPOT_ASSIGNED.is_file():
    raise SystemExit(f"Depot assignment file not found: DEPOT_ASSIGNED={os.getenv('DEPOT_ASSIGNED')}\n"
                     "Run full_run.py (or Matrix/assign_depot.py) first, or point DEPOT_ASSIGNED in .env at it.")

OUT_DIR = Path(clean(args.out)) if args.out else default_out_dir(ORDERS)
folders = {
    "A_OUTPUT_DIR": OUT_DIR / "Anneal output",
    "S_OUTPUT_DIR": OUT_DIR / "Sweep output",
    "P_OUTPUT_DIR": OUT_DIR / "Planning output",
}
for folder in [*folders.values(), OUT_DIR / "Matrix"]:
    folder.mkdir(parents=True, exist_ok=True)

# The scripts read these at import, so they are set before the imports below.
# os.environ wins over .env: the scripts' own load_dotenv() calls do not override it.
os.environ["ORDERS_PATH"] = str(ORDERS)                  # build_matrix
os.environ["C_OUTPUT_PATH"] = str(ORDERS)                # sweep and planning estimate (orders prompt default)
os.environ["MATRIX_OUT"] = str(OUT_DIR / "Matrix" / f"Matrix_{ORDERS.stem}.csv")
os.environ["CHECK_OUT"] = str(OUT_DIR / "Matrix" / f"Unassigned_areas_{ORDERS.stem}.csv")
for key, folder in folders.items():
    os.environ[key] = str(folder)

import matrix_common

import Matrix.build_matrix
import anneal_regional
import sweep_plan
import planning_estimate

anneal_regional.AUTO_OPEN = False  # unattended: skip os.startfile

# Every step uses the matrix built from the reduced orders; no matrix prompt.
matrix_common.MATRIX_PATH = Path(os.environ["MATRIX_OUT"])

print(f"Reduced orders:  {ORDERS}")
print(f"Depot assignment (reused): {DEPOT_ASSIGNED}")
print(f"Outputs:         {OUT_DIR}")


def step(title, fn):
    print(f"\n=== {title} ===")
    fn()                           # an exception or SystemExit stops the run here


def sweep_newest_anneal():
    """Step 3 without the "Enter the number to use" prompt: the sweep takes the newest anneal run.

    Same as in full_run.py. Step 2 has just written its Run_<stamp> folder into this run's own
    anneal folder, so the newest run is that one. planning_estimate.AUTO is switched on for the
    sweep only and put back afterwards, so step 4 still asks as it does in full_run.py.
    """
    expected = anneal_regional.OUTPUT_DIR / f"Run_{anneal_regional.RUN_STAMP}"   # folder step 2 wrote
    saved_auto = planning_estimate.AUTO
    saved_choose = planning_estimate.choose_plan_source

    def choose_newest():
        chosen = saved_choose()
        print(f"Sweep auto-picked anneal run: {chosen}")
        if chosen != expected:
            print(f"WARNING: this is not the run step 2 just made ({expected}). "
                  "Check the anneal output folder before using the sweep results.")
        return chosen

    planning_estimate.AUTO = True
    planning_estimate.choose_plan_source = choose_newest
    try:
        sweep_plan.main()
    finally:
        planning_estimate.AUTO = saved_auto
        planning_estimate.choose_plan_source = saved_choose


step("1. Build matrix (reduced orders)", Matrix.build_matrix.main)

if not matrix_common.MATRIX_PATH.exists():
    raise SystemExit(f"Matrix not found: {matrix_common.MATRIX_PATH}")
print(f"\nUsing matrix for all steps: {matrix_common.MATRIX_PATH}")

step("2. Anneal", anneal_regional.main)
step("3. Sweep", sweep_newest_anneal)
step("4. Planning estimate", planning_estimate.main)
print(f"\nDone. Outputs in: {OUT_DIR}")
