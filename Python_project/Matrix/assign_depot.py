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
DEPOT_OUT = Path(os.environ["DEPOT_ASSIGNED"])

AREA = "Postcode Area"
DEPOT = "Depot"
EAST, NORTH = "GRIDE_Ave", "GRIDN_Ave"

PREFERRED_BONUS_KM = 15     # current depot treated as this many km closer
MAX_KM = 150                 # flag areas further than this from their chosen depot

# ---- Depot settings --------------------------------------------------------------

# Only these depots are assigned areas. Leave empty to use every depot in the depot info file.
ACTIVE_DEPOTS = ["Watford", "Doncaster", "Bristol Outbase", "Warrington"]

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

    # Locked depots keep their own areas and take on no others. Areas whose current
    # depot is not active (or blank) have no bonus and go to the nearest active depot.
    locked_cols = [names.index(d) for d in locked if d in names]
    for i in range(len(helper)):
        current_depot = helper.iloc[i][DEPOT]
        if current_depot in locked and current_depot in names:
            score[i, names.index(current_depot)] = -np.inf
        else:
            score[i, locked_cols] = np.inf

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
    print(f"Depot locations loaded: {', '.join(depot_xy)}")

    # Only the active depots take part. Names must match the depot info Title exactly.
    if ACTIVE_DEPOTS:
        missing = [d for d in ACTIVE_DEPOTS if d not in depot_xy]
        if missing:
            raise SystemExit(f"ACTIVE_DEPOTS not found in depot info (check spelling): {missing}")
        depot_xy = {d: depot_xy[d] for d in ACTIVE_DEPOTS}
    print(f"Active depots: {', '.join(depot_xy)}")
    inactive_locked = sorted(set(locked) - set(depot_xy))
    if inactive_locked:
        print(f"Note: locked depots that are not active are ignored: {inactive_locked}")

    helper_depots = set(helper[DEPOT]) - {""}
    no_areas = set(depot_xy) - helper_depots
    if no_areas:
        print(f"Note: active depots with no areas in the helper: {sorted(no_areas)}")
    for d in sorted(helper_depots - set(depot_xy)):
        print(f"Note: {(helper[DEPOT] == d).sum()} areas at {d} (not active) "
              f"will move to the nearest active depot")

    result = assign_depots(helper, depot_xy)

    print("\nAreas per depot after assignment")
    print(result.groupby(DEPOT).size().to_string())
    print(f"\nAreas changing depot: {result['Changed'].sum()}")
    if result["Changed"].any():
        print(result.loc[result["Changed"],
                         [AREA, "Current depot", DEPOT, "Km to depot"]]
              .to_string(index=False))
    print(f"\nBeyond {MAX_KM} km: {(result['Review'] != '').sum()}")

    DEPOT_OUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(DEPOT_OUT, index=False, encoding="utf-8-sig")

    print(f"Saved: {DEPOT_OUT}")


if __name__ == "__main__":
    main()