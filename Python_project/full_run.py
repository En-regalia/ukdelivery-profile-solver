# full_run.py - runs the whole pipeline. Keep it in the same folder as the scripts.
import os

from dotenv import load_dotenv
load_dotenv()                      # read .env before any script reads its settings

import matrix_common
from matrix_common import ask_skip, select_matrix_path

import Matrix.assign_depot
import Matrix.build_matrix
import anneal_regional
import sweep_plan
import planning_estimate

anneal_regional.AUTO_OPEN = False  # unattended: skip os.startfile

# Ask once which matrix to use. The anneal, sweep and planning estimate all call
# select_matrix_path(), which returns this answer instead of asking again.
# must_exist=False: the built matrix is (re)written by step 2, so it may not exist yet.
matrix_common.MATRIX_PATH = select_matrix_path(must_exist=False)


def step(title, fn):
    print(f"\n=== {title} ===")
    fn()                           # an exception or SystemExit stops the run here


def sweep_newest_anneal():
    """Step 4 without the "Enter the number to use" prompt: the sweep takes the newest anneal run.

    Step 3 has just written its Run_<stamp> folder, so the newest run is that one.
    planning_estimate.AUTO is switched on for the sweep only and put back afterwards,
    so step 5 still asks as it does now (settings.toml [planning] auto decides there).
    """
    expected = anneal_regional.OUTPUT_DIR / f"Run_{anneal_regional.RUN_STAMP}"   # folder step 3 wrote
    saved_auto = planning_estimate.AUTO
    saved_choose = planning_estimate.choose_plan_source

    def choose_newest():
        chosen = saved_choose()    # lists the 5 newest runs, then takes the newest (AUTO is on)
        print(f"Sweep auto-picked anneal run: {chosen}")
        if chosen != expected:
            print(f"WARNING: this is not the run step 3 just made ({expected}). "
                  "Check the anneal output folder before using the sweep results.")
        return chosen

    planning_estimate.AUTO = True
    planning_estimate.choose_plan_source = choose_newest
    try:
        sweep_plan.main()
    finally:
        planning_estimate.AUTO = saved_auto
        planning_estimate.choose_plan_source = saved_choose


if ask_skip(5):
    print("Depot assignment/Anneal Skipped.")
else:
    step("1. Depot assignment", Matrix.assign_depot.main)

step("2. Build matrix", Matrix.build_matrix.main)

if not matrix_common.MATRIX_PATH.exists():
    raise SystemExit(f"Matrix not found: {matrix_common.MATRIX_PATH}")
print(f"\nUsing matrix for all steps: {matrix_common.MATRIX_PATH}")

step("3. Anneal", anneal_regional.main)
step("4. Sweep", sweep_newest_anneal)
step("5. Planning estimate", planning_estimate.main) 
