# run_all.py - keep it in the same folder as the scripts
import os

from dotenv import load_dotenv
load_dotenv()                      # read .env before any script reads its settings

from matrix_common import ask_skip, select_matrix_path

import Matrix.assign_depot
import Matrix.build_matrix
import anneal_regional
import sweep_plan
import planning_estimate

BUILT_POSTCODE_PATH = os.environ.get("MATRIX_OUT")
CURRENT_POSTCODE_PATH = os.environ.get("MATRIX_CURRENT")
matrix_path = select_matrix_path()

anneal_regional.AUTO_OPEN = False  # unattended: skip os.startfile

#AREA_OPTIONS = {
#    "1": ("Built Postcode areas (assign_depot output)", BUILT_POSTCODE_PATH),
#    "2": ("Existing Postcode areas (Matrix V1)", CURRENT_POSTCODE_PATH),
#}

def step(title, fn):
    print(f"\n=== {title} ===")
    fn()                           # an exception or SystemExit stops the run here

if ask_skip(5):
    print("Step 1: Depot assignment Skipped. Using existing/actual postcode areas")
else:
    step("1. Depot assignment", Matrix.assign_depot.main)

step("2. Build matrix", Matrix.build_matrix.main)
step("3. Anneal", anneal_regional.main,
     matrix_path=matrix_path)
step("4. Sweep", sweep_plan.main, 
     matrix_path=matrix_path)
step("5. Planning estimate", planning_estimate.main, 
     matrix_path=matrix_path)