"""Build the day-volume Matrix file the anneal solver reads.
One row per postcode area in the depot helper file, with kg per weekday."""
import os
from dotenv import load_dotenv

from pathlib import Path

import pandas as pd
import glob

# ---- Settings -------------------------------------------------------------
ORDERS_PATH = Path(os.environ["ORDERS_PATH"])
DEPOT_ASSIGNED = Path(os.environ["DEPOT_ASSIGNED"])
MATRIX_OUT = Path(os.environ["MATRIX_OUT"])
CHECK_OUT = Path(os.environ["CHECK_OUT"])

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
AREA = "Postcode Area"
DEPOT = "Depot"
EAST, NORTH = "GRIDE_Ave", "GRIDN_Ave"
DATE_COL = "date"
WEIGHT_COL = "Weight"
STATE_COL = "stateText"
EXCLUDE_STATES = ["Cancelled"]


# ---- Helpers --------------------------------------------------------------
def read_csv_any(path):
    for enc in ("utf-8-sig", "cp1252"):
        try:
            return pd.read_csv(path, dtype=str, keep_default_na=False, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="latin-1")


def to_num(s):
    return pd.to_numeric(s.str.replace(",", "", regex=False).str.strip(),
                         errors="coerce").fillna(0.0)


# ---- Main -----------------------------------------------------------------
def main():
    orders = read_csv_any(ORDERS_PATH)
    helper = read_csv_any(DEPOT_ASSIGNED)

    for col in [AREA, DEPOT, EAST, NORTH]:
        if col not in helper.columns:
            raise SystemExit(f"Depot helper is missing column: {col}")

    # Orders: kg per postcode area per weekday
    orders[AREA] = orders[AREA].str.strip().str.upper()
    orders["kg"] = to_num(orders[WEIGHT_COL])
    orders = orders[~orders[STATE_COL].str.strip().isin(EXCLUDE_STATES)]
    dates = pd.to_datetime(orders[DATE_COL].str.strip(), format="%Y%m%d", errors="coerce")
    orders["Weekday"] = dates.dt.day_name()

    weekend = orders["Weekday"].isin(["Saturday", "Sunday"])
    if weekend.any():
        print(f"Warning: {weekend.sum()} orders on weekends excluded "
              f"({orders.loc[weekend, 'kg'].sum():,.0f} kg)")
    orders = orders[~weekend & orders["Weekday"].notna()]

    pivot = (orders.pivot_table(index=AREA, columns="Weekday", values="kg",
                                aggfunc="sum", fill_value=0)
             .reindex(columns=DAYS, fill_value=0))

    # Helper: one row per area, with depot and grid reference
    helper[AREA] = helper[AREA].str.strip().str.upper()
    helper[DEPOT] = helper[DEPOT].str.strip()
    for c in (EAST, NORTH):
        helper[c] = to_num(helper[c])

    dupes = helper[helper[AREA].duplicated(keep=False)]
    if not dupes.empty:
        raise SystemExit(f"Duplicate areas in depot helper:\n{dupes[AREA].unique()}")
    blank = helper[helper[DEPOT] == ""]
    if not blank.empty:
        print(f"Warning: {len(blank)} areas have no depot: {list(blank[AREA])}")

    matrix = helper.set_index(AREA)[[DEPOT, EAST, NORTH]].join(pivot, how="left")
    matrix[DAYS] = matrix[DAYS].fillna(0).round(2)
    matrix = matrix[[DEPOT] + DAYS + [EAST, NORTH]].reset_index()

    # Order areas missing from the helper: volume that would be lost
    unassigned = pivot[~pivot.index.isin(helper[AREA])].copy()
    unassigned["Total kg"] = unassigned.sum(axis=1).round(0)
    unassigned = unassigned.reset_index()
    if not unassigned.empty:
        unassigned.to_csv(CHECK_OUT, index=False, encoding="utf-8-sig")
        print(f"WARNING: {len(unassigned)} postcode areas in the orders are not in "
              f"the depot helper ({unassigned['Total kg'].sum():,.0f} kg). "
              f"Listed in {CHECK_OUT}. These are NOT in the matrix.")

    MATRIX_OUT.parent.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(MATRIX_OUT, index=False)

    # Sanity check: kg per depot per day
    by_depot = matrix.groupby(DEPOT)[DAYS].sum().round(0)
    by_depot["Total"] = by_depot.sum(axis=1)
    print("\nkg per depot per day")
    print(by_depot.to_string())
    print(f"\nAreas in matrix: {len(matrix)}  |  areas with volume: "
          f"{(matrix[DAYS].sum(axis=1) > 0).sum()}")
    print(f"Total kg in matrix: {matrix[DAYS].sum().sum():,.0f}")
    print(f"Saved: {MATRIX_OUT}")


if __name__ == "__main__":
    main()