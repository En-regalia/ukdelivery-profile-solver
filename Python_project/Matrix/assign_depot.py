#"""Assign each postcode area to the depot with the shortest straight-line distance on the British National Grid. A bonus (km) keeps the current depot unless another depot is clearly closer."""


import os
from dotenv import load_dotenv
from pathlib import Path

import numpy as np
import pandas as pd

# ---- Settings -------------------------------------------------------------
load_dotenv()

DEPOT_PATH = Path(os.environ["DEPOT_PATH"])
DEPOT_INFO_PATH = Path(os.environ["DEPOT_INFO_PATH"])
DEPOT_OUT = Path(os.environ["DEPOT_OUT"])

AREA = "Postcode Area"
DEPOT = "Depot"
EAST, NORTH = "GRIDE_Ave", "GRIDN_Ave"

PREFERRED_BONUS_KM = 15     # current depot treated as this many km closer
MAX_KM = 150                 # flag areas further than this from their chosen depot

locked = {"Bristol Outbase"}  # areas that are locked to their current depot


# ---- Helpers --------------------------------------------------------------
def read_csv_any(path):
    """Read a CSV whatever encoding it was saved in."""
    for enc in ("utf-8-sig", "cp1252"):
        try:
            return pd.read_csv(path, dtype=str, keep_default_na=False, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="latin-1")


def to_num(s):
    return pd.to_numeric(s.str.replace(",", "", regex=False).str.strip(),
                         errors="coerce").fillna(0.0)


def load_depot_locations(path):
    """Read depot names and BNG grid references from the depot info file."""
    info = read_csv_any(path)
    info.columns = [c.strip() for c in info.columns]
    info["Title"] = info["Title"].str.strip()
    info["E"] = to_num(info["X (Easting)"])
    info["N"] = to_num(info["Y (Northing)"])
    info = info[(info["E"] > 0) & (info["N"] > 0)]

    dupes = info.loc[info["Title"].duplicated(), "Title"].tolist()
    if dupes:
        raise SystemExit(f"Duplicate depot titles in depot info: {dupes}")
    return {t: (e, n) for t, e, n in zip(info["Title"], info["E"], info["N"])}


def assign_depots(helper, depot_xy):
    names = list(depot_xy)
    dep = np.array([depot_xy[n] for n in names], dtype=float)
    pts = helper[[EAST, NORTH]].to_numpy(dtype=float)

    dist = np.hypot(pts[:, None, 0] - dep[None, :, 0],
                    pts[:, None, 1] - dep[None, :, 1]) / 1000   # km
    score = dist.copy()
    for j, n in enumerate(names):
        score[helper[DEPOT].to_numpy() == n, j] -= PREFERRED_BONUS_KM

    for i in range(len(helper)):
        current_depot = helper.iloc[i][DEPOT]
        cd_column = names.index(current_depot)
        if current_depot in locked:
            score[i,cd_column] = -np.inf
        else:
            for d in locked:
                locked_column = names.index(d)
                score[i,locked_column] = np.inf

                    

    best = score.argmin(axis=1)
    out = helper.copy()
    out["Current depot"] = helper[DEPOT]
    out[DEPOT] = [names[k] for k in best]
    out["Km to depot"] = dist[np.arange(len(best)), best].round(1)
    out["Changed"] = out[DEPOT] != out["Current depot"]
    out["Review"] = np.where(out["Km to depot"] > MAX_KM,
                             "Beyond max distance - check", "")
    return out


# ---- Main -----------------------------------------------------------------
def main():
    depot_xy = load_depot_locations(DEPOT_INFO_PATH)
    helper = read_csv_any(DEPOT_PATH)
    helper[DEPOT] = helper[DEPOT].str.strip()
    helper[EAST] = to_num(helper[EAST])
    helper[NORTH] = to_num(helper[NORTH])

    # Names must match exactly between the two files
    helper_depots = set(helper[DEPOT]) - {""}
    no_location = helper_depots - set(depot_xy)
    no_areas = set(depot_xy) - helper_depots
    print(f"Depot locations loaded: {', '.join(depot_xy)}")
    if no_location:
        raise SystemExit(f"Depots in helper with no location in depot info: {sorted(no_location)}")
    if no_areas:
        print(f"Note: depot info has locations with no areas in the helper: {sorted(no_areas)}")

    result = assign_depots(helper, depot_xy)

    print("\nAreas per depot after assignment")
    print(result.groupby(DEPOT).size().to_string())
    print(f"\nAreas changing depot: {result['Changed'].sum()}")
    if result["Changed"].any():
        print(result.loc[result["Changed"],
                         [AREA, "Current depot", DEPOT, "Km to depot"]]
              .to_string(index=False))
    print(f"\nBeyond {MAX_KM} km: {(result['Review'] != '').sum()}")

    result.to_csv(DEPOT_OUT, index=False, encoding="utf-8-sig")
    print(f"Saved: {DEPOT_OUT}")


if __name__ == "__main__":
    main()