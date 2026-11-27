"""Assign each postcode area to a depot by straight-line distance on the British National Grid.

Three methods (settings.toml [assignment] method):

"preference" (default): Watford first. Bristol only covers what Watford cannot reasonably do.
  A depot can reasonably reach an area when the closed loop depot -> area -> depot fits within its
  route cap, it is no more than REACH_RATIO times as far as the area's nearest depot (the cover
  depot is left out of that comparison), and its vehicles would not be driving past the cover depot
  to get there. "Driving past" (PAST_COVER_KM): going depot -> cover depot -> area is at most
  PAST_COVER_KM longer than going straight depot -> area, so the area lies beyond Bristol as seen
  from that depot. Areas in COVER_ALWAYS always stay with the cover depot.
  1. PREFERRED_DEPOT (Watford) takes every area it can reach, nearest to it first, up to its
     MAX_FLEET capacity.
  2. The other depots except the cover depot (Warrington, Doncaster) take what is left: each area
     goes to its nearest depot that can reach it, nearest first, up to that depot's MAX_FLEET
     capacity. An area over a full depot tries its next reachable depot.
  3. COVER_DEPOT (Bristol) keeps or takes what no other depot can reach (or could, but is full).
     These areas stay with it even over its capacity, flagged, so nothing goes back to a far depot.
  4. Anything still left (no depot can reach it) goes to its nearest depot, flagged "may need
     pallet network".
  MAX_FLEET capacity: [assignment.max_fleet] vehicles by type x the [vehicles] kg and pallets x the
  [report] fill_target (90%). A depot without a max_fleet entry uses its [fleet] and capacity basis.
  The output lists vehicles needed per depot against today's [fleet] and the max fleet.

"distance": the depot-assignment stage.
  1. Nearest-depot test across every active depot that is not closing.
  2. Tolerance band: an area is eligible only for its nearest depot, and only if that depot is
     at least TOLERANCE_KM nearer than the next-nearest. Areas not clearly nearer to any depot
     are left unassigned, not forced.
  3. Max distance: an area further than MAX_KM from its nearest depot is flagged, not assigned.
  4. Fill by capacity: each depot takes its eligible areas largest margin first (or, for a depot
     in FILL_FURTHEST_FROM, furthest from the depot it names first), and stops
     when the next area would take it over its kg or pallets per day. Capacity is the depot's
     [fleet] in settings.toml, by vehicle type, times the [vehicles] kg and pallets.
  5. Route-km check: before each addition, the closed loop depot -> area -> depot (routing.vehicle_route)
     must fit within the depot's route cap. An area that breaks it is skipped.
  6. Fallback (FALLBACK_TO_NEAREST, default on): every area rules 2 to 5 could not place goes to
     its nearest depot anyway (or, if that depot is full, e.g. Bristol at its capacity or target, to the
     next-nearest depot with room, however far, or the pallet network if OVER_TARGET = "pallet"), flagged with the reason ("may need pallet network"), so no postcode
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
# Capacity per depot per day comes from settings.toml [fleet] (vehicles by type) times a load per vehicle:
#   "actual":  the depot's average kg and pallets per actual route of that type in the raw orders
#              (C_OUTPUT_PATH), i.e. what its vehicles really carry with the stops and km they run.
#   "nominal": the full [vehicles] kg and pallets (a vehicle filled to the limit every day).
VEHICLE_TYPES = SETTINGS["vehicles"]
CAPACITY_BASIS = S.get("capacity_basis", "nominal")                       # every depot not listed below
CAPACITY_BASIS_BY_DEPOT = dict(S.get("capacity_basis_by_depot", {}))      # e.g. {"Bristol Outbase": "actual"}


def basis_for(depot):
    return CAPACITY_BASIS_BY_DEPOT.get(depot, CAPACITY_BASIS)


def strict_depot(depot):
    """A depot whose capacity is also enforced on fallback areas: one with the "actual" basis or a
    kg target. Other depots take fallback areas regardless of capacity, and the planning estimate's
    fleet check decides what fits, as before."""
    return basis_for(depot) == "actual" or depot in MAX_KG_PER_WEEK
FALLBACK_TO_NEAREST = S.get("fallback_to_nearest", True)   # leftovers go to their nearest depot, flagged
MAX_KG_PER_WEEK = dict(S.get("max_kg_per_week", {}))        # depot -> kg-per-week target (fill and fallback respect it)
OVER_TARGET = S.get("over_target", "next")                  # area over a depot's target: "next" depot or "pallet" network
FILL_FURTHEST_FROM = dict(S.get("fill_furthest_from", {}))  # depot -> other depot: fill areas furthest from it first
PALLET_COL = S.get("pallet_col", "").strip()  # pallets column in the orders file ("" = no pallet check)

# Preference method
PREFERRED_DEPOT = S.get("preferred_depot", "Watford")         # takes every area it can reasonably reach first
COVER_DEPOT = S.get("cover_depot", "Bristol Outbase")         # only keeps or takes what no other depot can reach
REACH_RATIO = S.get("reach_ratio", 2.0)                       # at most this many times the nearest depot's km
PAST_COVER_KM = S.get("past_cover_km", 35.0)                  # "driving past Bristol" detour limit, km
COVER_ALWAYS = [str(a).strip().upper() for a in S.get("cover_always", [])]  # areas or letters always Bristol
MAX_FLEET = {d: dict(v) for d, v in S.get("max_fleet", {}).items()}   # depot -> {type: vehicles}, the ceiling
FILL_TARGET = SETTINGS["report"]["fill_target"]               # each max_fleet vehicle counts at this fill
MAKE_MAP = S.get("make_map", True)                            # draw the depot coverage map at the end
MAP_AREA_CAP_KM = S.get("map_area_cap_km", 30.0)              # coverage map: an area reaches at most this far from its grid ref

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
ACTUAL_LOAD = {}  # (depot, type) -> (average kg, average pallets) per actual route; filled from the orders


def load_per_vehicle(depot, vtype):
    """(kg, pallets, source) one vehicle of this type at this depot is counted as carrying per day."""
    nominal = (VEHICLE_TYPES[vtype]["kg"], VEHICLE_TYPES[vtype]["pallets"], "nominal")
    if basis_for(depot) != "actual":
        return nominal
    if (depot, vtype) in ACTUAL_LOAD:
        return (*ACTUAL_LOAD[(depot, vtype)], "actual")
    if vtype == "26t" and (depot, "HGV") in ACTUAL_LOAD:   # actual routes are only split van / HGV
        kg, pal = ACTUAL_LOAD[(depot, "HGV")]
        f_kg = VEHICLE_TYPES["26t"]["kg"] / VEHICLE_TYPES["HGV"]["kg"]
        f_pal = VEHICLE_TYPES["26t"]["pallets"] / VEHICLE_TYPES["HGV"]["pallets"]
        return (kg * f_kg, pal * f_pal, "actual HGV, scaled")
    return nominal


def depot_capacity(depots):
    """depot -> (vehicles, kg per day, pallets per day): [fleet] vehicles x load per vehicle.

    With capacity_basis = "actual", e.g. Bristol 2 HGV + 1 Van = 2 x (Bristol's average kg per actual
    HGV route) + 1 x (its average kg per actual van route). With "nominal", 2 x 8,000 + 1 x 1,000.
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
        per = {t: load_per_vehicle(d, t) for t, k in fleet[d].items() if k}
        kg = sum(fleet[d][t] * per[t][0] for t in per)
        pal = sum(fleet[d][t] * per[t][1] for t in per)
        out[d] = (n, round(kg), round(pal, 1))
    return out


def orders_route_caps(helper, depot_xy):
    """Per-depot truck cap worked out as planning_estimate.py does: ROUTE_PCT percentile of the
    depot's actual HGV route km in the raw orders file (C_OUTPUT_PATH), never above the HGV max_km.

    Uses the real depot positions from the depot info file. Returns {} if the raw file is not set.
    """
    raw = (os.getenv("C_OUTPUT_PATH") or "").strip().strip('"').strip("'")
    if not raw or not Path(raw).exists():
        print(f"Note: C_OUTPUT_PATH not set or not found, so route caps are MAX_ROUTE_KM ({MAX_ROUTE_KM} km) "
              "or [assignment.route_cap_km], and capacity uses the nominal [vehicles] sizes")
        return {}
    import planning_estimate as pe       # imported here: it is only needed for this step
    print(f"Route caps from actual routes in {raw}")
    orders = pe.load_orders(raw)
    coords = {a: (e / 1000, n / 1000) for a, e, n in zip(helper[AREA], helper[EAST], helper[NORTH])
              if e > 0 and n > 0}
    dxy = {d: (e / 1000, n / 1000) for d, (e, n) in depot_xy.items()}
    rows = pe.actual_route_rows(orders, coords, dxy)
    actual = pd.DataFrame(rows)
    if not actual.empty:
        for (d, vt), g in actual.groupby(["Depot", "Type"]):
            ACTUAL_LOAD[(d, vt)] = (float(g["KG"].mean()), float(g["Pallets"].mean()))
    if not ROUTE_CAPS_FROM_ORDERS:
        return {}
    caps = pe.route_caps(rows)
    return {d: c["HGV"] for d, c in caps.items()}


ROUTE_CAP = {}    # depot -> km, filled by run_distance_stage
LAST_WORKBOOK = [None]   # the run's Depot_assignment workbook, for the coverage map


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


def fill_key(t, dist, names):
    """Fill order within each depot, largest first: margin over the next-nearest depot, or for a
    depot in FILL_FURTHEST_FROM, the distance from the depot it names."""
    key = t["Margin km"].to_numpy(dtype=float).copy()
    for depot, other in FILL_FURTHEST_FROM.items():
        if depot not in names or other not in names:
            print(f"Note: fill_furthest_from {depot} -> {other} ignored (not an active depot)")
            continue
        mask = (t["Nearest depot"] == depot).to_numpy()
        key[mask] = dist[mask, names.index(other)]
    return key


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
    # A depot in FILL_FURTHEST_FROM fills furthest from the named depot first instead (e.g. Bristol
    # furthest from Watford), so the areas it cuts at its target are the ones nearest that depot.
    t["_order"] = fill_key(t, dist, names)
    eligible = t[t["Reason"] == ""].sort_values(["Nearest depot", "_order", AREA],
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
        # At a "strict" depot (actual capacity basis or a kg target, e.g. Bristol) fallback areas respect
        # the same capacity as the fill: an area whose nearest strict depot is full goes to the
        # next-nearest depot with room, however far (OVER_TARGET = "next"), flagged. Other depots take
        # fallback areas regardless of capacity.
        fb = t[(t["Reason"] != "") & (t["Reason"] != R_NO_GRID)].sort_values(["_order", AREA],
                                                                             ascending=[False, True])
        for i, r in fb.iterrows():
            kg, why = r["kg per day"], r["Reason"].lower()
            choice, flag = None, None
            for rank, k in enumerate(order[i]):
                d = names[k]
                # A depot that is full (the fill stopped there, or this area would take it over its
                # capacity or target) takes no fallback areas. Not topping up a full depot with a later,
                # smaller area keeps the fill order (e.g. Bristol furthest from Watford first).
                over = strict_depot(d) and (
                    load[d][0] + fallback_load[d][0] + kg > cap[d][1] or
                    (check_pallets and load[d][1] + fallback_load[d][1] + r["Pallets per day"] > cap[d][2]))
                if (d in full and strict_depot(d)) or over:
                    full.add(d)
                    if OVER_TARGET == "pallet":
                        flag = f"{d} full: pallet network ({why})"
                        break
                    continue                          # "next": try the next-nearest depot
                # The next-nearest depot with room takes the area even if it is beyond MAX_KM or the
                # route cap: the flag says so, and the planning estimate's route cap decides what fits.
                far = rank > 0 and (dist[i, k] > MAX_KM or 2 * dist[i, k] > route_cap(d))
                choice = d
                flag = (f"Fallback ({why}): may need pallet network" if rank == 0 else
                        f"{names[order[i][0]]} full: moved to {d} ({why})"
                        + (f", {dist[i, k]:.0f} km away: may need pallet network" if far else ""))
                break
            t.at[i, "Flag"] = flag or "Every depot full: pallet network"
            if choice:
                t.at[i, DEPOT] = choice
                t.at[i, "Km to chosen"] = round(float(dist[i, names.index(choice)]), 1)
                fallback_load[choice][0] += kg
                fallback_load[choice][1] += r["Pallets per day"]
    t.loc[t["Reason"] == R_NO_GRID, "Flag"] = "No depot: no grid reference"

    t = t.drop(columns="_order")
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


# ---- Preference method ----------------------------------------------------
R_COVER = "Cover depot"          # no other depot can reach it (or those that can are full)
R_OUT_OF_REACH = "Out of reach"  # no depot can reach it within its route cap and the reach ratio


def max_fleet_capacity(depots, check_pallets):
    """depot -> (vehicles, kg per day, pallets per day, source). A depot in MAX_FLEET: its max fleet x
    [vehicles] size x FILL_TARGET. Any other depot: depot_capacity ([fleet] x its capacity basis)."""
    fleet_cap = depot_capacity([d for d in depots if d not in MAX_FLEET])
    out = {}
    for d in depots:
        if d in MAX_FLEET:
            unknown = set(MAX_FLEET[d]) - set(VEHICLE_TYPES)
            if unknown:
                raise SystemExit(f"[assignment.max_fleet] {d} uses vehicle types not in [vehicles]: {sorted(unknown)}")
            f = MAX_FLEET[d]
            kg = sum(n * VEHICLE_TYPES[t]["kg"] * FILL_TARGET for t, n in f.items())
            pal = sum(n * VEHICLE_TYPES[t]["pallets"] * FILL_TARGET for t, n in f.items())
            out[d] = (sum(f.values()), round(kg), round(pal, 1), f"max fleet at {FILL_TARGET:.0%} fill")
        else:
            n, kg, pal = fleet_cap[d]
            out[d] = (n, kg, pal, f"[fleet], {basis_for(d)} load")
        if d in MAX_KG_PER_WEEK:                       # a kg target still caps the fill
            out[d] = (out[d][0], min(out[d][1], round(MAX_KG_PER_WEEK[d] / 5)), out[d][2], out[d][3] + ", kg target")
    return out


def preference_stage(helper, depot_xy, volume, check_pallets=True):
    """Watford-first assignment table, flagged list and depot loads. See the module docstring."""
    from routing import vehicle_route    # imported here so the "nearest" method does not need it

    names = list(depot_xy)
    for role, d in (("preferred_depot", PREFERRED_DEPOT), ("cover_depot", COVER_DEPOT)):
        if d not in names:
            raise SystemExit(f"[assignment] {role} '{d}' is not an active depot")
    if PREFERRED_DEPOT == COVER_DEPOT:
        raise SystemExit("[assignment] preferred_depot and cover_depot must be different depots")
    P, C = names.index(PREFERRED_DEPOT), names.index(COVER_DEPOT)
    others = [k for k in range(len(names)) if k not in (P, C)]

    dep = np.array([depot_xy[n] for n in names], dtype=float)
    pts = helper[[EAST, NORTH]].to_numpy(dtype=float)
    dist = np.hypot(pts[:, None, 0] - dep[None, :, 0],
                    pts[:, None, 1] - dep[None, :, 1]) / 1000   # km
    order = np.argsort(dist, axis=1)
    near, nxt = order[:, 0], order[:, 1]
    rows_i = np.arange(len(helper))
    d1, d2 = dist[rows_i, near], dist[rows_i, nxt]
    no_grid = (pts[:, 0] <= 0) | (pts[:, 1] <= 0)

    # Closed loop depot -> area -> depot for every area and depot, and whether the depot can reach it
    areas = helper[AREA].to_numpy()
    loop = np.zeros_like(dist)
    for i in np.where(~no_grid)[0]:
        axy = {areas[i]: (pts[i, 0] / 1000, pts[i, 1] / 1000)}
        for k, n in enumerate(names):
            loop[i, k] = vehicle_route([areas[i]], axy, (dep[k, 0] / 1000, dep[k, 1] / 1000))[1]
    caps_km = np.array([route_cap(n) for n in names])

    # Reach ratio against the nearest depot other than the cover depot, so being close to Bristol
    # no longer stops Watford reaching an area. The "driving past Bristol" test does that job instead.
    not_c = [k for k in range(len(names)) if k != C]
    d1_not_c = dist[:, not_c].min(axis=1)
    ratio_ok = dist <= REACH_RATIO * d1_not_c[:, None] + 1e-9

    # Driving past Bristol: depot -> Bristol -> area is at most PAST_COVER_KM longer than depot -> area.
    depot_to_c = np.hypot(dep[:, 0] - dep[C, 0], dep[:, 1] - dep[C, 1]) / 1000   # km, each depot to Bristol
    detour = depot_to_c[None, :] + dist[:, [C]] - dist                          # km, area x depot
    past_c = detour <= PAST_COVER_KM
    past_c[:, C] = False                                                         # Bristol never drives past itself

    # Areas always kept by Bristol: an exact area ("TQ6") or a whole letter group ("CF").
    letters = pd.Series(areas).astype(str).str.upper().str.extract(r"^([A-Z]+)")[0].fillna("").to_numpy()
    always_c = np.array([(str(a).upper() in COVER_ALWAYS) or (l in COVER_ALWAYS)
                         for a, l in zip(areas, letters)])

    reach = (loop <= caps_km[None, :]) & ratio_ok & ~past_c & ~no_grid[:, None]
    reach[always_c, :] = False
    reach[always_c, C] = ~no_grid[always_c]            # only Bristol can reach a cover_always area

    t = pd.DataFrame({
        AREA: areas,
        "Current depot": helper[DEPOT].to_numpy(),
        "Nearest depot": [names[k] for k in near],
        "Km to nearest": d1.round(1),
        "Next depot": [names[k] for k in nxt],
        "Km to next": d2.round(1),
        "Margin km": (d2 - d1).round(1),
        f"Km to {PREFERRED_DEPOT}": dist[:, P].round(1),
        f"{PREFERRED_DEPOT} loop km": loop[:, P].round(1),
        f"{PREFERRED_DEPOT} can reach": reach[:, P],
        f"{PREFERRED_DEPOT} detour via {COVER_DEPOT} km": detour[:, P].round(1),
        f"Past {COVER_DEPOT} for {PREFERRED_DEPOT}": past_c[:, P] | always_c,
        "Depots that can reach": [", ".join(names[k] for k in order[i] if reach[i, k]) for i in rows_i],
    })
    t["kg per day"] = volume.reindex(t[AREA])["kg"].fillna(0).round(1).to_numpy()
    t["Pallets per day"] = volume.reindex(t[AREA])["pal"].fillna(0).round(2).to_numpy()
    t[DEPOT] = ""
    t["Km to chosen"] = np.nan
    t["Loop km"] = np.nan
    t["Rule"] = ""
    t["Reason"] = ""
    t["Flag"] = ""
    kg_a, pal_a = t["kg per day"].to_numpy(), t["Pallets per day"].to_numpy()

    cap = max_fleet_capacity(names, check_pallets)
    load = {d: [0.0, 0.0] for d in names}
    full = set()

    def fits(i, k):
        d = names[k]
        return (load[d][0] + kg_a[i] <= cap[d][1] and
                (not check_pallets or load[d][1] + pal_a[i] <= cap[d][2]))

    def place(i, k, rule, reason="", flag=""):
        d = names[k]
        t.at[i, DEPOT] = d
        t.at[i, "Km to chosen"] = round(float(dist[i, k]), 1)
        t.at[i, "Loop km"] = round(float(loop[i, k]), 1)
        t.at[i, "Rule"], t.at[i, "Reason"], t.at[i, "Flag"] = rule, reason, flag
        load[d][0] += kg_a[i]
        load[d][1] += pal_a[i]

    placed = lambda i: t.at[i, DEPOT] != ""

    # 1. Preferred depot takes every area it can reach, nearest to it first, up to its capacity.
    for i in sorted(np.where(reach[:, P])[0], key=lambda i: (dist[i, P], areas[i])):
        if fits(i, P):
            place(i, P, f"1 {PREFERRED_DEPOT} first"
                        + (f" (nearer: {names[near[i]]})" if near[i] not in (P, C) else ""))
        else:
            full.add(PREFERRED_DEPOT)

    # 2. The other depots take what is left: nearest reachable depot first, nearest areas first.
    #    An area over a full depot comes up again later in the list for its next reachable depot.
    pairs = sorted(((dist[i, k], areas[i], i, k) for i in rows_i for k in others
                    if reach[i, k] and not placed(i)))
    for _, _, i, k in pairs:
        if placed(i):
            continue
        if fits(i, k):
            nearer_full = [names[j] for j in others if reach[i, j] and dist[i, j] < dist[i, k]]
            place(i, k, "2 nearest depot that can reach it", reason=R_CAPACITY if nearer_full else "",
                  flag=f"{', '.join(nearer_full)} full: moved to {names[k]}" if nearer_full else "")
        else:
            full.add(names[k])

    # 3. Cover depot keeps or takes what no other depot can reach, or what those depots are too full
    #    for. It keeps them even over its capacity (flagged), so nothing is pushed back to a far depot.
    for i in sorted((i for i in rows_i if not placed(i) and reach[i, C]),
                    key=lambda i: (dist[i, C], areas[i])):
        could = [names[k] for k in [P] + others if reach[i, k]]
        why = (f"{', '.join(could)} full" if could
               else "on the cover_always list" if always_c[i]
               else f"{PREFERRED_DEPOT} would drive past {COVER_DEPOT}" if past_c[i, P]
               else "no other depot can reach it")
        over = not fits(i, C)
        place(i, C, f"3 {COVER_DEPOT} cover", R_COVER,
              f"{COVER_DEPOT} cover ({why})" + (f", over {COVER_DEPOT} capacity" if over else ""))
        if over:
            full.add(COVER_DEPOT)

    # 4. Out of reach of every depot: nearest depot anyway, flagged.
    for i in rows_i:
        if placed(i) or no_grid[i]:
            continue
        k = near[i]
        place(i, k, "4 fallback to nearest", R_OUT_OF_REACH,
              f"Fallback (no depot can reach it within its route cap, {REACH_RATIO}x the nearest "
              f"and without driving past {COVER_DEPOT}): "
              "may need pallet network")
    t.loc[no_grid, "Reason"] = R_NO_GRID
    t.loc[no_grid, "Flag"] = "No depot: no grid reference"

    t["Changed"] = t[DEPOT] != t["Current depot"]
    leftovers = t[t["Flag"] != ""].copy()
    fleet = fleet_caps()
    veh_kg = VEHICLE_TYPES["HGV"]["kg"] * FILL_TARGET
    before = (t.groupby("Current depot")["kg per day"].sum() * 5).to_dict()
    loads = pd.DataFrame([{
        "Depot": d, "Role": ("preferred" if d == PREFERRED_DEPOT else "cover" if d == COVER_DEPOT else ""),
        "Route cap km": route_cap(d),
        "Areas": int((t[DEPOT] == d).sum()),
        "kg per week before (current depots)": round(before.get(d, 0)),
        "kg per week after": round(5 * load[d][0]),
        "kg per day": round(load[d][0]), "kg cap per day": cap[d][1], "Capacity from": cap[d][3],
        "kg used": round(load[d][0] / cap[d][1], 3) if cap[d][1] else None,
        "Pallets per day": round(load[d][1], 1) if check_pallets else None,
        "Pallet cap": cap[d][2] if check_pallets else None,
        "Full or over": "Yes" if d in full else "",
        f"HGVs needed at {FILL_TARGET:.0%} fill (kg per day / {veh_kg:,.0f})": round(load[d][0] / veh_kg, 1),
        "Vehicles today ([fleet])": sum(fleet.get(d, {}).values()),
        "Vehicles max ([assignment.max_fleet])": sum(MAX_FLEET[d].values()) if d in MAX_FLEET else None,
        "Areas flagged": int(((t[DEPOT] == d) & (t["Flag"] != "")).sum()),
    } for d in names])
    t.attrs["reach"], t.attrs["names"], t.attrs["cap"], t.attrs["load"] = reach, names, cap, load
    return t, leftovers, loads


def check_preference(t):
    """Acceptance checks for the preference method. Returns rows of (check, result, detail)."""
    names, reach = t.attrs["names"], t.attrs["reach"]
    cap, load = t.attrs["cap"], t.attrs["load"]
    k_of = {d: k for k, d in enumerate(names)}
    rows = []
    by_rule = t[t["Rule"].str[:1].isin(["1", "2", "3"])]
    bad = [a for i, a, d in zip(by_rule.index, by_rule[AREA], by_rule[DEPOT]) if not reach[i, k_of[d]]]
    rows.append({"Check": f"Areas placed by rules 1 to 3 are within their depot's route cap, {REACH_RATIO}x "
                          f"the nearest depot's km, and not past {COVER_DEPOT}",
                 "Result": "Pass" if not bad else "FAIL", "Detail": ", ".join(bad) or f"{len(by_rule)} areas"})
    at_p = t[t[DEPOT] == PREFERRED_DEPOT]
    strays = at_p[~at_p[f"{PREFERRED_DEPOT} can reach"]]
    rows.append({"Check": f"No {PREFERRED_DEPOT} area beyond its reach (strays)",
                 "Result": "Pass" if strays.empty else "Flagged",
                 "Detail": ", ".join(strays[AREA]) or ""})
    C = k_of[COVER_DEPOT]
    gained = t[(t[DEPOT] == COVER_DEPOT) & (t["Current depot"] != COVER_DEPOT)]
    bad = []
    for i, r in gained.iterrows():
        for k, d in enumerate(names):
            if k != C and reach[i, k] and load[d][0] + r["kg per day"] <= cap[d][1]:
                bad.append(f"{r[AREA]} ({d} had room)")
                break
    rows.append({"Check": f"{COVER_DEPOT} gains only areas no other depot could reach or take",
                 "Result": "Pass" if not bad else "FAIL",
                 "Detail": ", ".join(bad) or (f"gained: {', '.join(gained[AREA])}" if len(gained) else "gains none")})
    before = 5 * t.loc[t["Current depot"] == COVER_DEPOT, "kg per day"].sum()
    after = 5 * t.loc[t[DEPOT] == COVER_DEPOT, "kg per day"].sum()
    rows.append({"Check": f"{COVER_DEPOT} does not grow (kg per week, before vs after)",
                 "Result": "Pass" if after <= before + 0.5 else "Flagged",
                 "Detail": f"{before:,.0f} before, {after:,.0f} after"})
    over = [d for d in names if cap[d][1] and load[d][0] > cap[d][1] + 0.5]
    rows.append({"Check": "Depots over their capacity", "Result": "Pass" if not over else "Flagged",
                 "Detail": ", ".join(f"{d} {load[d][0]:,.0f} of {cap[d][1]:,.0f} kg per day" for d in over)})
    unplaced = t[(t[DEPOT] == "") & (t["Flag"] == "")]
    rows.append({"Check": "Every postcode area has a depot, or is flagged",
                 "Result": "Pass" if unplaced.empty else "FAIL", "Detail": ", ".join(unplaced[AREA])})
    no_reason = t[(t["Flag"] != "") & (t["Reason"] == "")]
    rows.append({"Check": "Every flagged area has a reason", "Result": "Pass" if no_reason.empty else "FAIL",
                 "Detail": ", ".join(no_reason[AREA]) or f"{int((t['Flag'] != '').sum())} flagged"})
    return pd.DataFrame(rows)


def check_assignment(t, depot_xy):
    """Acceptance checks. Returns rows of (check, result, detail)."""
    a = t[(t[DEPOT] != "") & (t["Reason"] == "")]          # placed by rules 2 to 5 (not fallback)
    bad_tol = a[(a[DEPOT] != a["Nearest depot"]) | (a["Margin km"] < TOLERANCE_KM)]
    rows = [{"Check": f"Assigned only to the nearest depot, at least {TOLERANCE_KM} km nearer than the next "
                      "(fallback areas excepted, they are flagged)",
             "Result": "Pass" if bad_tol.empty else "FAIL",
             "Detail": ", ".join(bad_tol[AREA]) or f"{len(a)} areas assigned"}]
    fb = t[t["Reason"] != ""]
    bad_fb = fb[(fb[DEPOT] != "") & (fb[DEPOT] != fb["Nearest depot"]) & ~fb["Flag"].str.contains(" full: moved to ")]
    rows.append({"Check": "Fallback areas go to their nearest depot (unless it is full)",
                 "Result": "Pass" if bad_fb.empty else "FAIL",
                 "Detail": ", ".join(bad_fb[AREA]) or f"{int((fb[DEPOT] != '').sum())} fallback areas"})
    if {"Watford", "Bristol Outbase"} <= set(depot_xy):
        w, b = (np.array(depot_xy[d], dtype=float) for d in ("Watford", "Bristol Outbase"))
        at_w = t[t[DEPOT] == "Watford"]
        # Bristol side of the perpendicular bisector of Watford-Bristol: nearer Bristol than Watford
        pts = at_w[["_E", "_N"]].to_numpy(dtype=float)
        bristol_side = np.hypot(*(pts - b).T) < np.hypot(*(pts - w).T)
        moved = at_w["Flag"].str.contains(" full: moved to ").to_numpy()
        bad = at_w.loc[bristol_side & ~moved, AREA]
        rows.append({"Check": "No Watford area on the Bristol side of the Watford-Bristol midpoint "
                              "(areas moved past a full Bristol excepted, they are flagged)",
                     "Result": "Pass" if bad.empty else "FAIL", "Detail": ", ".join(bad) or ""})
        crossed = at_w.loc[bristol_side & moved, AREA]
        if len(crossed):
            rows.append({"Check": "Bristol-side areas moved to Watford because Bristol is full",
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


def capacity_rows(depot_xy):
    """How each depot's daily capacity was built: vehicles by type x load per vehicle."""
    fleet = fleet_caps()
    rows = []
    for d in depot_xy:
        if METHOD == "preference" and d in MAX_FLEET:
            for t, n in MAX_FLEET[d].items():
                if n:
                    kg, pal = VEHICLE_TYPES[t]["kg"] * FILL_TARGET, VEHICLE_TYPES[t]["pallets"] * FILL_TARGET
                    rows.append({"Depot": d, "Type": t, "Vehicles": n, "kg per vehicle": round(kg),
                                 "Pallets per vehicle": round(pal, 1),
                                 "Source": f"max fleet at {FILL_TARGET:.0%} fill",
                                 "Nominal kg": VEHICLE_TYPES[t]["kg"], "Fill vs nominal": FILL_TARGET,
                                 "kg per day": round(n * kg), "kg per week": round(5 * n * kg)})
            continue
        for t, n in fleet.get(d, {}).items():
            if not n:
                continue
            kg, pal, src = load_per_vehicle(d, t)
            rows.append({"Depot": d, "Type": t, "Vehicles": n, "kg per vehicle": round(kg),
                         "Pallets per vehicle": round(pal, 1), "Source": src,
                         "Nominal kg": VEHICLE_TYPES[t]["kg"], "Fill vs nominal": round(kg / VEHICLE_TYPES[t]["kg"], 2),
                         "kg per day": round(n * kg), "kg per week": round(5 * n * kg)})
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
        {"Setting": "Capacity basis", "Value": f"{CAPACITY_BASIS}; " + (", ".join(f"{k} {v}" for k, v in CAPACITY_BASIS_BY_DEPOT.items()) or "no depot overrides")},
        {"Setting": "Vehicle sizes ([vehicles])", "Value": ", ".join(f"{t} {v['kg']} kg / {v['pallets']} pallets" for t, v in VEHICLE_TYPES.items())},
        {"Setting": "Pallet limit", "Value": "checked" if check_pallets else "not checked (no pallet column)"},
        {"Setting": "Orders file (volume)", "Value": str(ORDERS_PATH or "not set")},
        {"Setting": "Method", "Value": METHOD},
    ] + ([
        {"Setting": "Preferred depot", "Value": PREFERRED_DEPOT},
        {"Setting": "Cover depot", "Value": COVER_DEPOT},
        {"Setting": "Reach ratio", "Value": f"{REACH_RATIO} (nearest depot other than {COVER_DEPOT})"},
        {"Setting": f"Driving past {COVER_DEPOT} (past_cover_km)", "Value": PAST_COVER_KM},
        {"Setting": f"Always {COVER_DEPOT} (cover_always)", "Value": ", ".join(COVER_ALWAYS) or "none"},
        {"Setting": "Max fleet", "Value": "; ".join(f"{d} " + ", ".join(f"{t} {n}" for t, n in f.items())
                                                    for d, f in MAX_FLEET.items()) or "none"},
        {"Setting": "Max fleet vehicles count at", "Value": f"{FILL_TARGET:.0%} fill ([report] fill_target)"},
    ] if METHOD == "preference" else []))


def run_distance_stage(helper, depot_xy):
    ROUTE_CAP.update(orders_route_caps(helper, depot_xy))
    volume, has_pal = load_area_volume(ORDERS_PATH, helper[AREA].str.strip().str.upper())
    volume.index = helper[AREA].to_numpy()
    if METHOD == "preference":
        t, leftovers, loads = preference_stage(helper, depot_xy, volume, check_pallets=has_pal)
        checks = check_preference(t)
    else:
        t, leftovers, loads = distance_stage(helper, depot_xy, volume, check_pallets=has_pal)
        t["_E"], t["_N"] = helper[EAST].to_numpy(), helper[NORTH].to_numpy()
        checks = check_assignment(t, depot_xy)
        t = t.drop(columns=["_E", "_N"])

    print("\nDepot loads (capacity per day: " + ("[assignment.max_fleet] at the fill target, else [fleet]"
          if METHOD == "preference" else "[fleet] vehicles x load per vehicle") + ", see the Capacity sheet)")
    print(loads.to_string(index=False))
    if METHOD == "preference":
        print(f"\nFlagged: {len(leftovers)}")
    else:
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
    if METHOD == "preference":
        cols[3:3] = ["Rule"]
        cols += [f"Km to {PREFERRED_DEPOT}", f"{PREFERRED_DEPOT} loop km", f"{PREFERRED_DEPOT} can reach",
                 "Depots that can reach"]
    out = DEPOT_OUT.parent / f"Depot_assignment_{RUN_STAMP}.xlsx"
    LAST_WORKBOOK[0] = out
    out.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out) as xl:
        t[cols].rename(columns={DEPOT: "Assigned depot"}).to_excel(xl, sheet_name="Assignment", index=False)
        leftovers[cols].rename(columns={DEPOT: "Assigned depot"}).to_excel(xl, sheet_name="Flagged", index=False)
        loads.to_excel(xl, sheet_name="Depot loads", index=False)
        checks.to_excel(xl, sheet_name="Checks", index=False)
        capacity_rows(depot_xy).to_excel(xl, sheet_name="Capacity", index=False)
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
    result["_kg"] = t["kg per day"].to_numpy()
    return result


# ---- Coverage map ---------------------------------------------------------
DEPOT_COLOURS = ["#1f5fa8", "#d9731a", "#2e8b57", "#a83279", "#7a5c2e", "#5a5a9e", "#c0392b", "#16a0a0"]
DEPOT_INFO_XY = {}   # every depot in the depot info file (closing ones too), for the "current depots" map


def coverage_cells(pts, outline, cap_km):
    """One approximate area per postcode area: the land nearer its grid reference than any other
    area's (a Voronoi cell), clipped to the UK outline (if loaded) and to cap_km around the grid
    reference, so a lone area far from its neighbours does not fill half the country.
    Returns a GeoSeries in the same row order as pts."""
    import geopandas as gpd
    import shapely
    from shapely.geometry import MultiPoint, Point

    xy = pts[[EAST, NORTH]].to_numpy(dtype=float)
    uniq = np.unique(xy, axis=0)                     # areas sharing a grid reference share a cell
    land = outline.union_all() if outline is not None else MultiPoint(uniq).envelope.buffer(cap_km * 1000)
    cells = shapely.voronoi_polygons(MultiPoint(uniq), extend_to=land.envelope.buffer(cap_km * 1000)).geoms
    cell_gdf = gpd.GeoDataFrame(geometry=list(cells), crs=27700)
    pt_gdf = gpd.GeoDataFrame({"_i": range(len(pts))}, geometry=gpd.points_from_xy(xy[:, 0], xy[:, 1]), crs=27700)
    hit = gpd.sjoin(pt_gdf, cell_gdf, predicate="within").drop_duplicates("_i").set_index("_i")["index_right"]
    geoms = []
    for i, (e, n) in enumerate(xy):
        cell = cell_gdf.geometry.iloc[hit[i]] if i in hit.index else Point(e, n).buffer(1000)
        geoms.append(cell.intersection(land).intersection(Point(e, n).buffer(cap_km * 1000)))
    return gpd.GeoSeries(geoms, index=pts.index, crs=27700)


def draw_coverage_map(result, depot_xy, xlsx_path):
    """Map of each depot and the postcode areas it covers: current depots beside the new assignment.

    Each postcode area is drawn as an approximate area (see coverage_cells) filled in its depot's
    colour, with a dark line where one depot's patch meets another's. No area labels, so the coverage
    reads at a glance; the Assignment sheet lists every area. Depots are white stars. Flagged
    areas are hatched, except the cover depot's routine "cover" flags. Saved as a PNG next to the workbook and added to it as a "Map" sheet. The UK outline comes
    from MAP_IMAGE_PATH in .env, if set (without it the areas are not clipped to the coast).
    """
    os.environ.setdefault("MAP_IMAGE_PATH", "")          # maps_common reads it at import
    from maps_common import MAP_DPI, add_maps_sheet, load_uk_outline, plt
    import geopandas as gpd
    from matplotlib.patches import Patch

    outline = load_uk_outline()
    pts = result[(result[EAST] > 0) & (result[NORTH] > 0)].copy()
    kg = pts["_kg"].to_numpy(dtype=float) if "_kg" in pts else np.zeros(len(pts))
    shown = list(depot_xy) + sorted((set(pts["Current depot"]) | set(pts[DEPOT])) - set(depot_xy) - {"", LEFTOVER})
    colour = {d: DEPOT_COLOURS[i % len(DEPOT_COLOURS)] for i, d in enumerate(shown)}
    colour[LEFTOVER] = colour[""] = "#9a9a9a"
    # Hatched: flagged areas, except Bristol's routine cover areas (flagged on every run by design)
    review = pts["Review"].fillna("")
    flagged = ((review != "") & ~review.str.startswith(f"{COVER_DEPOT} cover (")).to_numpy()
    all_xy = {**DEPOT_INFO_XY, **depot_xy}
    cells = gpd.GeoDataFrame(pts[["Current depot", DEPOT]].reset_index(drop=True),
                             geometry=coverage_cells(pts, outline, MAP_AREA_CAP_KM).reset_index(drop=True), crs=27700)
    cells = cells[~cells.geometry.is_empty]
    fig, axes = plt.subplots(1, 2, figsize=(16, 11))
    for ax, col, title in ((axes[0], "Current depot", "Current depots"), (axes[1], DEPOT, "New assignment")):
        if outline is not None:
            outline.plot(ax=ax, color="#eeeeee", edgecolor="white", linewidth=0.3)
        cells.plot(ax=ax, color=[colour.get(d, "#9a9a9a") for d in cells[col]], edgecolor="white", linewidth=0.15)
        cells.dissolve(by=col).boundary.plot(ax=ax, color="#222222", linewidth=0.9)   # depot borders
        if col == DEPOT and flagged.any():
            cells[flagged[cells.index]].plot(ax=ax, facecolor="none", edgecolor="black", hatch="////", linewidth=0)
        depots_here = [d for d in shown if d in set(pts[col])]
        for d in depots_here:
            if d in all_xy:
                e, n = all_xy[d]
                ax.plot([e], [n], marker="*", markersize=22, markerfacecolor="white",
                        markeredgecolor="black", markeredgewidth=1, ls="", zorder=10)
                ax.annotate(d, (e, n), xytext=(10, -14), textcoords="offset points", fontsize=10,
                            fontweight="bold", zorder=11)
        legend = [Patch(facecolor=colour[d], label=f"{d}: {(pts[col] == d).sum()} areas, "
                                                    f"{5 * kg[(pts[col] == d).to_numpy()].sum():,.0f} kg/wk")
                  for d in depots_here]
        if col == DEPOT and flagged.any():
            legend.append(Patch(facecolor="none", edgecolor="black", hatch="////", label="Flagged (see Flagged sheet; routine Bristol cover not hatched)"))
        ax.legend(handles=legend, loc="upper right", fontsize=9, frameon=True)
        minx, miny, maxx, maxy = cells.total_bounds
        pad = 20000
        ax.set_xlim(minx - pad, maxx + pad)
        ax.set_ylim(miny - pad, maxy + pad)
        ax.set_aspect("equal")
        ax.set_axis_off()
        ax.set_title(title, fontsize=14, fontweight="bold")
    fig.suptitle(f"Depot coverage by postcode area ({METHOD} method, run {RUN_STAMP})",
                 fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    png = Path(xlsx_path).with_name(f"Depot_coverage_map_{RUN_STAMP}.png")
    png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=MAP_DPI)
    plt.close(fig)
    on_sheet = Path(xlsx_path).exists()
    if on_sheet:
        add_maps_sheet(xlsx_path, png, sheet="Map")
    print(f"Saved: {png}" + (" (also on the Map sheet)" if on_sheet else ""))


# ---- Main -----------------------------------------------------------------
def main():
    depot_xy = load_depot_locations(DEPOT_INFO_PATH)
    DEPOT_INFO_XY.update(depot_xy)
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

    if METHOD in ("distance", "preference"):
        result = run_distance_stage(helper, depot_xy)
    elif METHOD == "nearest":
        inactive_locked = sorted(set(locked) - set(depot_xy))
        if inactive_locked:
            print(f"Note: locked depots that are not active are ignored: {inactive_locked}")
        result = assign_depots(helper, depot_xy)
        print(f"\nBeyond {MAX_KM} km: {(result['Review'] != '').sum()}")
    else:
        raise SystemExit(f'[assignment] method must be "preference", "distance" or "nearest", not "{METHOD}"')

    print("\nAreas per depot after assignment")
    print(result[DEPOT].value_counts().sort_index().to_string())
    print(f"\nAreas changing depot: {result['Changed'].sum()}")

    DEPOT_OUT.parent.mkdir(parents=True, exist_ok=True)
    result.drop(columns="_kg", errors="ignore").to_csv(DEPOT_OUT, index=False, encoding="utf-8-sig")
    print(f"Saved: {DEPOT_OUT}")

    # Final step: map of each depot and the areas it covers
    if MAKE_MAP:
        try:
            draw_coverage_map(result, depot_xy,
                              LAST_WORKBOOK[0] or DEPOT_OUT.parent / f"Depot_assignment_{RUN_STAMP}.xlsx")
        except Exception as e:                 # the assignment is saved either way
            print(f"Coverage map not drawn: {e}")


if __name__ == "__main__":
    main()
