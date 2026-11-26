"""Assign each postcode area to a depot by straight-line distance on the British National Grid.

Two methods (settings.toml [assignment] method):

"distance" (default): the depot-assignment stage.
  1. Nearest-depot test across every active depot that is not closing.
  2. Tolerance band: an area is eligible only for its nearest depot, and only if that depot is
     at least TOLERANCE_KM nearer than the next-nearest. Areas not clearly nearer to any depot
     are left unassigned, not forced.
  3. Max distance: an area further than MAX_KM from its nearest depot is flagged, not assigned.
  4. Fill by capacity: each depot takes its eligible areas largest margin first, and stops
     when the next area would take it over its kg or pallets per day. Capacity is the depot's
     [fleet] in settings.toml, by vehicle type, times the [vehicles] kg and pallets.
  5. Route-km check: before each addition, the closed loop depot -> area -> depot (routing.vehicle_route)
     must fit within the depot's route cap. An area that breaks it is skipped.
  6. Fallback (FALLBACK_TO_NEAREST, default on): every area rules 2 to 5 could not place goes to
     its nearest depot anyway (or, if that depot is at its MAX_KG_PER_WEEK target, to the next-nearest
     depot with room, or the pallet network, per OVER_TARGET), flagged with the reason ("may need pallet network"), so no postcode
     is left uncovered. The anneal and the planning estimate then decide: the anneal treats far
     letters as external (pallet network), and the planning estimate's route cap and fleet checks
     list any orders that still do not fit. The flagged areas are on the "Flagged" sheet.
     With the fallback off, these areas get the depot "Leftover" and go to the pallet network.
     Only an area with no grid reference stays "Leftover" either way.

"nearest": the old method. Nearest active depot, with a bonus for the current depot, and
  locked depots that keep their own areas and take on no others.
"""

import os
import sys
from datetime import datetime
from dotenv import load_dotenv
from pathlib import Path

import numpy as np
import pandas as pd

# ---- Settings -------------------------------------------------------------
load_dotenv()
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # settings.py sits one folder up
from settings import SETTINGS, ACTIVE_DEPOTS as _ACTIVE, LOCKED_DEPOTS, CLOSING_DEPOTS, fleet_caps

DEPOT_PATH = Path(os.environ["DEPOT_PATH"])
DEPOT_INFO_PATH = Path(os.environ["DEPOT_INFO_PATH"])
DEPOT_OUT = Path(os.environ["DEPOT_ASSIGNED"])
ORDERS_PATH = Path(os.environ["ORDERS_PATH"]) if os.getenv("ORDERS_PATH") else None   # volume for the capacity limits
RUN_STAMP = datetime.now().strftime("%Y-%m-%d_%H-%M")   # set when the script starts

AREA = "Postcode Area"
DEPOT = "Depot"
EAST, NORTH = "GRIDE_Ave", "GRIDN_Ave"

S = SETTINGS["assignment"]
METHOD = S.get("method", "distance")          # "distance" (assignment stage) or "nearest" (old method)

# Distance stage
TOLERANCE_KM = S["tolerance_km"]              # nearest depot must be at least this much nearer than the next
MAX_KM = S["max_km"]                          # further than this from the nearest depot: flagged, not assigned
MAX_ROUTE_KM = S["max_route_km"]              # closed loop depot -> area -> depot must fit within this
ROUTE_CAPS_FROM_ORDERS = S.get("route_caps_from_orders", True)   # per-depot cap as planning_estimate works it out
ROUTE_CAP_KM = dict(S.get("route_cap_km", {}))   # per-depot cap set by hand: overrides the one from the orders
# Capacity per depot per day comes from settings.toml: [fleet] (vehicles by type) x [vehicles] (kg, pallets)
VEHICLE_TYPES = SETTINGS["vehicles"]
FALLBACK_TO_NEAREST = S.get("fallback_to_nearest", True)   # leftovers go to their nearest depot, flagged
MAX_KG_PER_WEEK = dict(S.get("max_kg_per_week", {}))        # depot -> kg-per-week target (fill and fallback respect it)
OVER_TARGET = S.get("over_target", "next")                  # area over a depot's target: "next" depot or "pallet" network
PALLET_COL = S.get("pallet_col", "").strip()  # pallets column in the orders file ("" = no pallet check)

# Orders file columns (same as Matrix/build_matrix.py)
DATE_COL, WEIGHT_COL, STATE_COL = "date", "Weight", "stateText"
EXCLUDE_STATES = ["Cancelled"]
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]

# Old "nearest" method
PREFERRED_BONUS_KM = S["preferred_bonus_km"]  # current depot treated as this many km closer

# ---- Depot settings --------------------------------------------------------------

# Depot lists come from settings.toml [depots]. Closing depots are never assigned areas.
# An empty active list means every depot in the depot info file (except closing ones).
ACTIVE_DEPOTS = [d for d in _ACTIVE if d not in CLOSING_DEPOTS]
locked = set(LOCKED_DEPOTS)   # "nearest" method only: depots that keep their current areas and take no others

LEFTOVER = "Leftover"            # depot name for areas with no depot in the assigned helper file

# Leftover reasons
R_TOO_FAR = "Too far"            # nearest depot beyond MAX_KM
R_UNASSIGNED = "Unassigned"      # not at least TOLERANCE_KM nearer to any depot
R_CAPACITY = "Over capacity"     # depot full (kg or pallets per day)
R_ROUTE = "Route km"             # depot -> area -> depot over the route cap
R_NO_GRID = "No grid reference"


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


# ---- Distance stage -------------------------------------------------------
def depot_capacity(depots):
    """depot -> (vehicles, kg per day, pallets per day), from [fleet] and [vehicles] in settings.toml.

    e.g. Watford 14 HGV + 4 Van + 2 26t = 14 x 8,000 + 4 x 1,000 + 2 x 12,000 kg per day.
    """
    fleet = fleet_caps()
    out = {}
    for d in depots:
        if d not in fleet:
            raise SystemExit(f"No fleet for {d}: add it to [fleet] in settings.toml")
        unknown = set(fleet[d]) - set(VEHICLE_TYPES)
        if unknown:
            raise SystemExit(f"[fleet] {d} uses vehicle types not in [vehicles]: {sorted(unknown)}")
        n = sum(fleet[d].values())
        kg = sum(k * VEHICLE_TYPES[t]["kg"] for t, k in fleet[d].items())
        pal = sum(k * VEHICLE_TYPES[t]["pallets"] for t, k in fleet[d].items())
        out[d] = (n, kg, pal)
    return out


def orders_route_caps(helper, depot_xy):
    """Per-depot truck cap worked out as planning_estimate.py does: ROUTE_PCT percentile of the
    depot's actual HGV route km in the raw orders file (C_OUTPUT_PATH), never above the HGV max_km.

    Uses the real depot positions from the depot info file. Returns {} if the raw file is not set.
    """
    raw = (os.getenv("C_OUTPUT_PATH") or "").strip().strip('"').strip("'")
    if not ROUTE_CAPS_FROM_ORDERS or not raw or not Path(raw).exists():
        if ROUTE_CAPS_FROM_ORDERS:
            print(f"Note: C_OUTPUT_PATH not set or not found, so route caps are MAX_ROUTE_KM ({MAX_ROUTE_KM} km) "
                  "or [assignment.route_cap_km]")
        return {}
    import planning_estimate as pe       # imported here: it is only needed for this step
    print(f"Route caps from actual routes in {raw}")
    orders = pe.load_orders(raw)
    coords = {a: (e / 1000, n / 1000) for a, e, n in zip(helper[AREA], helper[EAST], helper[NORTH])
              if e > 0 and n > 0}
    dxy = {d: (e / 1000, n / 1000) for d, (e, n) in depot_xy.items()}
    caps = pe.route_caps(pe.actual_route_rows(orders, coords, dxy))
    return {d: c["HGV"] for d, c in caps.items()}


ROUTE_CAP = {}    # depot -> km, filled by run_distance_stage


def route_cap(depot):
    return min(MAX_ROUTE_KM, ROUTE_CAP_KM.get(depot, ROUTE_CAP.get(depot, MAX_ROUTE_KM)))


def load_area_volume(path, areas):
    """Average kg and pallets per weekday for each area: total over the file / weekdays in the file.

    The anneal spreads an area's volume across its delivery days, so the depot limit is checked
    against the average day, not the busiest day of the current plan.
    """
    vol = pd.DataFrame({"kg": 0.0, "pal": 0.0}, index=pd.Index(areas, name=AREA))
    if path is None:
        print("WARNING: ORDERS_PATH is not set in .env, so kg and pallet limits are not checked.")
        return vol, False
    o = read_csv_any(path)
    o[AREA] = o[AREA].str.strip().str.upper()
    o = o[~o[STATE_COL].str.strip().isin(EXCLUDE_STATES)]
    dates = pd.to_datetime(o[DATE_COL].str.strip(), format="%Y%m%d", errors="coerce")
    o = o[dates.dt.day_name().isin(WEEKDAYS)]
    n_days = dates[o.index].dt.normalize().nunique()
    if n_days == 0:
        raise SystemExit(f"No weekday orders with readable dates in {path}")
    o["kg"] = to_num(o[WEIGHT_COL])
    has_pal = bool(PALLET_COL) and PALLET_COL in o.columns
    if PALLET_COL and not has_pal:
        print(f"WARNING: pallet column '{PALLET_COL}' not in the orders file, so the pallet limit is not checked.")
    elif not PALLET_COL:
        print("Note: no pallet column set ([assignment] pallet_col), so the pallet limit is not checked.")
    o["pal"] = to_num(o[PALLET_COL]) if has_pal else 0.0
    per = o.groupby(AREA)[["kg", "pal"]].sum() / n_days
    vol.update(per)
    print(f"Volume: {n_days} weekdays in the orders file, {per['kg'].sum():,.0f} kg per day on average")
    return vol, has_pal


def distance_stage(helper, depot_xy, volume, check_pallets=True):
    """Assignment table (one row per area) and leftovers list. See the module docstring for the rules."""
    from routing import vehicle_route    # imported here so the "nearest" method does not need it

    names = list(depot_xy)
    if len(names) < 2:
        raise SystemExit("The distance stage needs at least two depots to compare")
    dep = np.array([depot_xy[n] for n in names], dtype=float)
    pts = helper[[EAST, NORTH]].to_numpy(dtype=float)
    dist = np.hypot(pts[:, None, 0] - dep[None, :, 0],
                    pts[:, None, 1] - dep[None, :, 1]) / 1000   # km
    order = np.argsort(dist, axis=1)
    near, nxt = order[:, 0], order[:, 1]
    rows_i = np.arange(len(helper))
    d1, d2 = dist[rows_i, near], dist[rows_i, nxt]

    t = pd.DataFrame({
        AREA: helper[AREA].to_numpy(),
        "Current depot": helper[DEPOT].to_numpy(),
        "Nearest depot": [names[k] for k in near],
        "Km to nearest": d1.round(1),
        "Next depot": [names[k] for k in nxt],
        "Km to next": d2.round(1),
        "Margin km": (d2 - d1).round(1),
    })
    t["kg per day"] = volume.reindex(t[AREA])["kg"].fillna(0).round(1).to_numpy()
    t["Pallets per day"] = volume.reindex(t[AREA])["pal"].fillna(0).round(2).to_numpy()
    t[DEPOT] = ""
    t["Km to chosen"] = np.nan
    t["Loop km"] = np.nan
    t["Reason"] = ""

    no_grid = (pts[:, 0] <= 0) | (pts[:, 1] <= 0)
    t.loc[no_grid, "Reason"] = R_NO_GRID
    t.loc[~no_grid & (d1 > MAX_KM), "Reason"] = R_TOO_FAR
    t.loc[(t["Reason"] == "") & (d2 - d1 < TOLERANCE_KM), "Reason"] = R_UNASSIGNED

    cap = depot_capacity(names)
    target = {d: MAX_KG_PER_WEEK[d] / 5 for d in names if d in MAX_KG_PER_WEEK}   # kg per day
    for d, kg_day in target.items():                 # the fill stops at the target as well as the fleet
        cap[d] = (cap[d][0], min(cap[d][1], kg_day), cap[d][2])
    load = {d: [0.0, 0.0] for d in names}
    full = set()
    # Rule 4: per depot, largest margin first. Each area is eligible for one depot only (its nearest).
    eligible = t[t["Reason"] == ""].sort_values(["Nearest depot", "Margin km", AREA],
                                                ascending=[True, False, True])
    for i, r in eligible.iterrows():
        d = r["Nearest depot"]
        dxy = (depot_xy[d][0] / 1000, depot_xy[d][1] / 1000)
        axy = (helper.at[i, EAST] / 1000, helper.at[i, NORTH] / 1000)
        loop = vehicle_route([r[AREA]], {r[AREA]: axy}, dxy)[1]
        t.at[i, "Loop km"] = round(loop, 1)
        if d in full:
            t.at[i, "Reason"] = R_CAPACITY
            continue
        if loop > route_cap(d):                      # rule 5: skip, keep filling
            t.at[i, "Reason"] = R_ROUTE
            continue
        kg, pal = load[d][0] + r["kg per day"], load[d][1] + r["Pallets per day"]
        if kg > cap[d][1] or (check_pallets and pal > cap[d][2]):
            full.add(d)                              # rule 4: stop adding to this depot
            t.at[i, "Reason"] = R_CAPACITY
            continue
        load[d] = [kg, pal]
        t.at[i, DEPOT] = d
        t.at[i, "Km to chosen"] = r["Km to nearest"]

    # Rule 6: fallback to the nearest depot, flagged. Capacity is not checked here on purpose:
    # the planning estimate's fleet and route-cap checks decide what actually fits.
    t["Flag"] = ""
    fallback_load = {d: [0.0, 0.0] for d in names}
    if FALLBACK_TO_NEAREST:
        # A depot with a kg target (MAX_KG_PER_WEEK) takes fallback areas only while it stays within it.
        # Fleet capacity is not checked here: the planning estimate decides what fits on vehicles.
        fb = t[(t["Reason"] != "") & (t["Reason"] != R_NO_GRID)].sort_values(["Margin km", AREA],
                                                                             ascending=[False, True])
        for i, r in fb.iterrows():
            kg, why = r["kg per day"], r["Reason"].lower()
            choice, flag = None, None
            for rank, k in enumerate(order[i]):
                d = names[k]
                if d in target and load[d][0] + fallback_load[d][0] + kg > target[d]:
                    if OVER_TARGET == "pallet":
                        flag = f"Over {d} target: pallet network ({why})"
                        break
                    continue                          # "next": try the next-nearest depot
                if rank > 0 and (dist[i, k] > MAX_KM or 2 * dist[i, k] > route_cap(d)):
                    flag = f"Over {names[order[i][0]]} target, next depot too far: pallet network ({why})"
                    break                             # never move an area to a far depot just to cut a target
                choice = d
                flag = (f"Fallback ({why}): may need pallet network" if rank == 0 else
                        f"Over {names[order[i][0]]} target: moved to {d} ({why})")
                break
            t.at[i, "Flag"] = flag or "Over every depot target: pallet network"
            if choice:
                t.at[i, DEPOT] = choice
                t.at[i, "Km to chosen"] = round(float(dist[i, names.index(choice)]), 1)
                fallback_load[choice][0] += kg
                fallback_load[choice][1] += r["Pallets per day"]
    t.loc[t["Reason"] == R_NO_GRID, "Flag"] = "No depot: no grid reference"

    t["Changed"] = t[DEPOT] != t["Current depot"]
    leftovers = t[t["Reason"] != ""].copy()
    loads = pd.DataFrame([{
        "Depot": d, "Vehicles": cap[d][0], "Route cap km": route_cap(d),
        "kg target per week": MAX_KG_PER_WEEK.get(d),
        "Areas": int((t[DEPOT] == d).sum()),
        "kg per day": round(load[d][0], 0), "kg cap": cap[d][1],
        "kg used": round(load[d][0] / cap[d][1], 3) if cap[d][1] else None,
        "Pallets per day": round(load[d][1], 1) if check_pallets else None,
        "Pallet cap": cap[d][2] if check_pallets else None,
        "Full": "Yes" if d in full else "",
        "Leftovers (over capacity)": int(((t["Nearest depot"] == d) & (t["Reason"] == R_CAPACITY)).sum()),
        "Leftovers (route km)": int(((t["Nearest depot"] == d) & (t["Reason"] == R_ROUTE)).sum()),
        "Fallback areas": int(((t[DEPOT] == d) & (t["Reason"] != "")).sum()),
        "Fallback kg per day": round(fallback_load[d][0], 0),
        "kg per day incl. fallback": round(load[d][0] + fallback_load[d][0], 0),
        "kg used incl. fallback": round((load[d][0] + fallback_load[d][0]) / cap[d][1], 3) if cap[d][1] else None,
    } for d in names])
    return t, leftovers, loads


def check_assignment(t, depot_xy):
    """Acceptance checks. Returns rows of (check, result, detail)."""
    a = t[(t[DEPOT] != "") & (t["Reason"] == "")]          # placed by rules 2 to 5 (not fallback)
    bad_tol = a[(a[DEPOT] != a["Nearest depot"]) | (a["Margin km"] < TOLERANCE_KM)]
    rows = [{"Check": f"Assigned only to the nearest depot, at least {TOLERANCE_KM} km nearer than the next "
                      "(fallback areas excepted, they are flagged)",
             "Result": "Pass" if bad_tol.empty else "FAIL",
             "Detail": ", ".join(bad_tol[AREA]) or f"{len(a)} areas assigned"}]
    fb = t[t["Reason"] != ""]
    bad_fb = fb[(fb[DEPOT] != "") & (fb[DEPOT] != fb["Nearest depot"]) & ~fb["Flag"].str.startswith("Over ")]
    rows.append({"Check": "Fallback areas go to their nearest depot (unless it is at its kg target)",
                 "Result": "Pass" if bad_fb.empty else "FAIL",
                 "Detail": ", ".join(bad_fb[AREA]) or f"{int((fb[DEPOT] != '').sum())} fallback areas"})
    if {"Watford", "Bristol Outbase"} <= set(depot_xy):
        w, b = (np.array(depot_xy[d], dtype=float) for d in ("Watford", "Bristol Outbase"))
        at_w = t[t[DEPOT] == "Watford"]
        # Bristol side of the perpendicular bisector of Watford-Bristol: nearer Bristol than Watford
        pts = at_w[["_E", "_N"]].to_numpy(dtype=float)
        bristol_side = np.hypot(*(pts - b).T) < np.hypot(*(pts - w).T)
        moved = at_w["Flag"].str.startswith("Over ").to_numpy()
        bad = at_w.loc[bristol_side & ~moved, AREA]
        rows.append({"Check": "No Watford area on the Bristol side of the Watford-Bristol midpoint "
                              "(areas moved past a full Bristol target excepted, they are flagged)",
                     "Result": "Pass" if bad.empty else "FAIL", "Detail": ", ".join(bad) or ""})
        crossed = at_w.loc[bristol_side & moved, AREA]
        if len(crossed):
            rows.append({"Check": "Bristol-side areas moved to Watford because Bristol is at its target",
                         "Result": "Flagged", "Detail": ", ".join(crossed)})
    left = t[t[DEPOT] == ""]
    unflagged = left[left["Flag"] == ""]
    rows.append({"Check": "Every postcode area has a depot, or is flagged for the pallet network",
                 "Result": "Pass" if unflagged.empty else "FAIL",
                 "Detail": ", ".join(unflagged[AREA]) or
                           (f"pallet network: {', '.join(left[AREA])}" if len(left) else "")})
    no_reason = t[(t["Flag"] != "") & (t["Reason"] == "")]
    rows.append({"Check": "Every flagged area has a reason",
                 "Result": "Pass" if no_reason.empty else "FAIL",
                 "Detail": ", ".join(no_reason[AREA]) or f"{int((t['Flag'] != '').sum())} flagged"})
    return pd.DataFrame(rows)


def settings_rows(depot_xy, check_pallets):
    return pd.DataFrame([
        {"Setting": "Run", "Value": RUN_STAMP},
        {"Setting": "Depots in the nearest-depot test", "Value": ", ".join(depot_xy)},
        {"Setting": "Closing depots (left out)", "Value": ", ".join(CLOSING_DEPOTS)},
        {"Setting": "TOLERANCE_KM", "Value": TOLERANCE_KM},
        {"Setting": "MAX_KM", "Value": MAX_KM},
        {"Setting": "MAX_ROUTE_KM", "Value": MAX_ROUTE_KM},
        {"Setting": "Route caps from actual routes", "Value": ", ".join(f"{k} {v}" for k, v in ROUTE_CAP.items()) or "none"},
        {"Setting": "Route caps set by hand", "Value": ", ".join(f"{k} {v}" for k, v in ROUTE_CAP_KM.items()) or "none"},
        {"Setting": "Vehicle sizes ([vehicles])", "Value": ", ".join(f"{t} {v['kg']} kg / {v['pallets']} pallets" for t, v in VEHICLE_TYPES.items())},
        {"Setting": "Pallet limit", "Value": "checked" if check_pallets else "not checked (no pallet column)"},
        {"Setting": "Orders file (volume)", "Value": str(ORDERS_PATH or "not set")},
    ])


def run_distance_stage(helper, depot_xy):
    ROUTE_CAP.update(orders_route_caps(helper, depot_xy))
    volume, has_pal = load_area_volume(ORDERS_PATH, helper[AREA].str.strip().str.upper())
    volume.index = helper[AREA].to_numpy()
    t, leftovers, loads = distance_stage(helper, depot_xy, volume, check_pallets=has_pal)
    t["_E"], t["_N"] = helper[EAST].to_numpy(), helper[NORTH].to_numpy()
    checks = check_assignment(t, depot_xy)
    t = t.drop(columns=["_E", "_N"])

    print("\nDepot loads (capacity per day from [fleet] x [vehicles] in settings.toml)")
    print(loads.to_string(index=False))
    print(f"\nFlagged (rules 2 to 5 could not place them): {len(leftovers)}"
          + ("  ->  given to their nearest depot" if FALLBACK_TO_NEAREST else "  ->  depot 'Leftover'"))
    if not leftovers.empty:
        print(leftovers.groupby("Reason").size().to_string())
        print(leftovers[[AREA, "Current depot", DEPOT, "Km to nearest", "Next depot",
                         "Margin km", "kg per day", "Reason"]].to_string(index=False))
    print("\nChecks")
    print(checks.to_string(index=False))

    cols = [AREA, "Current depot", DEPOT, "Km to chosen", "Nearest depot", "Km to nearest", "Next depot",
            "Km to next", "Margin km", "Loop km", "kg per day", "Pallets per day", "Changed", "Reason", "Flag"]
    out = DEPOT_OUT.parent / f"Depot_assignment_{RUN_STAMP}.xlsx"
    out.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out) as xl:
        t[cols].rename(columns={DEPOT: "Assigned depot"}).to_excel(xl, sheet_name="Assignment", index=False)
        leftovers[cols].rename(columns={DEPOT: "Assigned depot"}).to_excel(xl, sheet_name="Flagged", index=False)
        loads.to_excel(xl, sheet_name="Depot loads", index=False)
        checks.to_excel(xl, sheet_name="Checks", index=False)
        settings_rows(depot_xy, has_pal).to_excel(xl, sheet_name="Settings", index=False)
    print(f"Saved: {out}")

    # Helper file for build_matrix: same columns as the input, depot replaced. Areas with no depot
    # (fallback off, or no grid reference) get LEFTOVER, which is not active, so the anneal does not
    # plan them and the planning estimate sends their orders to the pallet network.
    result = helper.copy()
    result["Current depot"] = helper[DEPOT]
    result[DEPOT] = t[DEPOT].replace("", LEFTOVER).to_numpy()
    result["Km to depot"] = t["Km to chosen"].to_numpy()
    result["Changed"] = t["Changed"].to_numpy()
    result["Review"] = t["Flag"].to_numpy()
    return result


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
    else:
        depot_xy = {d: xy for d, xy in depot_xy.items() if d not in CLOSING_DEPOTS}
    print(f"Active depots: {', '.join(depot_xy)}  |  method: {METHOD}")

    helper_depots = set(helper[DEPOT]) - {""}
    no_areas = set(depot_xy) - helper_depots
    if no_areas:
        print(f"Note: active depots with no areas in the helper: {sorted(no_areas)}")
    for d in sorted(helper_depots - set(depot_xy)):
        print(f"Note: {(helper[DEPOT] == d).sum()} areas at {d} (not active) will be reassigned")

    if METHOD == "distance":
        result = run_distance_stage(helper, depot_xy)
    elif METHOD == "nearest":
        inactive_locked = sorted(set(locked) - set(depot_xy))
        if inactive_locked:
            print(f"Note: locked depots that are not active are ignored: {inactive_locked}")
        result = assign_depots(helper, depot_xy)
        print(f"\nBeyond {MAX_KM} km: {(result['Review'] != '').sum()}")
    else:
        raise SystemExit(f'[assignment] method must be "distance" or "nearest", not "{METHOD}"')

    print("\nAreas per depot after assignment")
    print(result[DEPOT].value_counts().sort_index().to_string())
    print(f"\nAreas changing depot: {result['Changed'].sum()}")

    DEPOT_OUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(DEPOT_OUT, index=False, encoding="utf-8-sig")
    print(f"Saved: {DEPOT_OUT}")


if __name__ == "__main__":
    main()
