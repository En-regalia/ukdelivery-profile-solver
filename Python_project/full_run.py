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


if ask_skip(5):
    print("Depot assignment/Anneal Skipped.")
else:
    step("1. Depot assignment", Matrix.assign_depot.main)

step("2. Build matrix", Matrix.build_matrix.main)

if not matrix_common.MATRIX_PATH.exists():
    raise SystemExit(f"Matrix not found: {matrix_common.MATRIX_PATH}")
print(f"\nUsing matrix for all steps: {matrix_common.MATRIX_PATH}")

step("3. Anneal", anneal_regional.main)
step("4. Sweep", sweep_plan.main)
step("5. Planning estimate", planning_estimate.main) 
