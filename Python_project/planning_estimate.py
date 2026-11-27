"""Example-week simulation: actual routes vs the solver's new day plan, van and truck fleet.

For the lowest, median and peak network weeks (by kg):
- Before: actual routes (unique assignedRouteUID) per depot per day, plus a
  vehicle estimate from kg and pallets.
- After: each order moves to its area's new day (same rule as map_days).
  Outliers and external letters go to "Pallet network" with no vehicles.
- Vehicles, packed per depot per day, within the depot's fleet:
    Van: 1000 kg / 5 pallets, route ceiling 500 km (hard).
    HGV (8t): 8000 kg / 15 pallets. Routes are built at this size first.
    26t: 12000 kg / 20 pallets. A route is upgraded from HGV to 26t only when that
    lets it absorb another route's orders (so one truck fewer), and the 26t fleet
    has room. Orders over 8t go straight onto 26t routes.
    Both truck types use the depot's real-route cap (90th percentile of actual HGV routes).
- Comparator: each planned route is compared with ONE real route, taken whole as it ran
  (real routes are never combined): the real route with the most drops in the planned
  route's areas, provided more than half its drops are there (comparator_min_share).
  Same week and day first, at any depot; then the same week on any day; else
  "No comparable route". Shown on the Routes sheet and the example route maps.
  Route km is straight-line (depot -> stops -> depot), so it understates road
  distance. The ceilings apply to these straight-line km.
  FLEET_CAP is enforced: no depot-day uses more vans, 8t HGVs or 26ts than its fleet.
  Any order that cannot be placed within a route ceiling or the fleet goes to the
  pallet network (the "Overflow" sheet), with its reason: "route cap", "fleet", or
  "too big" (over 26t limits; counted with the route cap orders).

  Van selection maximises the number of orders on the vans. Van-eligible orders
  (fit a van on their own) are taken smallest first, by kg then pallets. Each goes
  into the van it fills most tightly, provided the van's route stays within 500 km.
  Orders no van can take go to the trucks.

  Truck steps:
  1. Pack the orders onto 8t routes by postcode area (first fit, largest orders first),
     within the route cap. Orders over 8t go to 26t routes instead.
  2. Consolidate: empty a route into the others. A route that can't take an order at 8t
     is upgraded to 26t if the 26t fleet has room. Repeat until no route can be emptied.
  3. Fleet: if there are still more 26t or 8t routes than the fleet allows, the routes
     carrying the most orders are kept and the rest of their orders are placed into them
     where they fit within the cap. Whatever cannot be placed is the fleet overflow.
  A new truck is only opened for an order whose own closed loop is within the cap.
  Added-km limit: van selection and both consolidations only put an order on a
  non-empty vehicle if it adds at most MAX_ADDED_KM (60) to that vehicle's loop, so
  far-apart areas do not share a vehicle. Per depot and day the limit rises by 20 km
  while the trucks are over the fleet, up to 150 km. A day still over the fleet there
  is flagged in Summary ("Fleet cap unresolved") and trimmed as in step 3. The final
  limit per depot and day is printed and shown in Daily and Summary.
Both plans use the same packing and the same fleet limits, so the comparison
is like for like.

Actual routes are typed as vans when the whole route fits a van (under 1000 kg
and 5 pallets, as on the vehicle-type rule), otherwise as HGV routes (26t routes
are not split out of the actual data). The truck route cap comes from the 90th
percentile of actual HGV routes per depot, and is used for both 8t and 26t.

Route km (added): each packed vehicle's straight-line closed loop, depot -> stops
-> depot, from routing.py (nearest neighbour then 2-opt). Route km includes both
depot legs; Stop km is stop-to-stop only. Km columns are added to Weeks, Routes
and Summary, and the Route detail sheet lists every stop. Totals only in the
route summary; no route order sheet beyond that detail.
"""
import os
import random
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

import pandas as pd

from matrix_common import AREA, DAYS, map_days
from routing import DEPOT_KEY, load_grid_refs, route_legs, sum_route_km, vehicle_route
from maps_common import MAP_DPI, load_uk_outline
import matplotlib.pyplot as plt

# ---- dir paths and environment variables -------------------------------------------------------------
# set to your file
load_dotenv()

ANNEAL_FOLDER = Path(os.getenv("A_OUTPUT_DIR"))        # folder with Anneal_*_Regional_*.xlsx
OUTPUT_DIR = Path(os.getenv("P_OUTPUT_DIR"))
SWEEP_FOLDER = Path(os.getenv("S_OUTPUT_DIR")) if os.getenv("S_OUTPUT_DIR") else None   # sweep output folder
RUN_STAMP = datetime.now().strftime("%Y-%m-%d_%H-%M")

# ---- Settings -------------------------------------------------------------

# Values come from settings.toml ([vehicles], [routes], [fleet], [planning], [depots]).
from settings import SETTINGS, TEAM_CODES, fleet_caps
VEHICLE_TYPES = {t: dict(spec) for t, spec in SETTINGS["vehicles"].items()}
ROUTE_PCT = SETTINGS["routes"]["route_pct"]   # truck cap: this percentile of the depot's actual HGV route km
ROUTE_CAP = {}            # depot -> {"HGV": km}; filled in main() from the actual routes (26t uses the same cap)
FLEET_CAP = fleet_caps()  # vehicles available per depot per day, by type; a depot or type left out has no limit
# Added-km limit: an order may join a non-empty vehicle only if it adds at most this many km to
# that vehicle's closed loop, so far-apart areas do not share a vehicle. Per depot and day the
# limit starts at MAX_ADDED_KM and rises by ADDED_KM_STEP while the trucks are over the fleet,
# up to ADDED_KM_CEILING. A day still over the fleet there is flagged "unresolved" and trimmed
# to the fleet as before (fit_fleet: the extra orders go to the pallet network as "fleet").
MAX_ADDED_KM = SETTINGS["routes"]["max_added_km"]
ADDED_KM_STEP = SETTINGS["routes"]["added_km_step"]
ADDED_KM_CEILING = SETTINGS["routes"]["added_km_ceiling"]
# Comparator: a planned route is compared with the one real route that has more than this
# share of its drops in the planned route's areas (most such drops wins). Real routes are never combined.
COMPARATOR_MIN_SHARE = SETTINGS["planning"].get("comparator_min_share", 0.5)
ROUTE_DETAIL = SETTINGS["planning"]["route_detail"]   # writes a "Route detail" sheet: every stop, leg and cumulative km (After plan)
PLANNED_COLOUR = "#1f5fa8"  # single planned route on the example maps
ACTUAL_COLOUR = "#9a9a9a"   # actual routes, drawn behind the planned ones
DAY_MAP_COLOURS = ["#1f5fa8", "#e07b00", "#2a9d3c", "#c0392b", "#8e5a9b", "#8c564b",
                   "#d6458f", "#17a2b8", "#b8860b", "#2c3e50"]   # planned routes on the full-day maps
DAY_MAPS = SETTINGS["planning"]["day_maps"]   # one map per depot, day and week of that depot's example routes: all planned routes
EXAMPLE_MIN_DROPS = SETTINGS["planning"]["example_min_drops"]   # example maps only use routes with at least this many drops

C_OUTPUT_PATH = (os.getenv("C_OUTPUT_PATH") or "").strip().strip('"').strip("'") or None   # raw orders CSV: default for the orders prompt
PROMPT_SECONDS = SETTINGS["planning"]["prompt_seconds"]   # the consolidated file is used if nothing is typed within this time

# Raw file column positions (0-based): I, J, K, Y, AG, AN
DATE_COL, KG_COL, PAL_COL, TEAM_COL, ROUTE_COL, AREA_COL = 8, 9, 10, 24, 32, 39
DEPOT_CODES = TEAM_CODES  # team code -> depot, settings.toml [depots.team_codes]
PALLET = "Pallet network"

AUTO = SETTINGS["planning"]["auto"]  # True = unattended: use the most recent run without asking

# ---- Choose Anneal File run --------------------------------------------------------------
def ask_path(prompt):
    # Strip the quotes Windows adds when you copy a path
    return Path(input(prompt).strip().strip('"').strip("'"))

PLAN_GLOB = "Anneal_*_Regional_*.xlsx"   # anneal and sweep plan files (sweep: Anneal_*_Regional_Sweep_*)


def choose_run(folder, title):
    """Pick one of the five most recent runs in folder (Run_ subfolders or files saved directly).

    Run_ folders holding no plan file (a run stopped early, or one that planned no depots)
    are skipped and named, so they cannot be picked and send every area to the pallet network.
    """
    found = [d for d in folder.iterdir() if d.is_dir() and d.name.startswith("Run_")]
    runs = [d for d in found if any(d.glob(PLAN_GLOB))]
    empty = sorted(d.name for d in found if d not in runs)
    if empty:
        print(f"Skipped {len(empty)} {title} run folder(s) with no {PLAN_GLOB} file: {', '.join(empty[-5:])}"
              + (" ..." if len(empty) > 5 else ""))
    if any(folder.glob(PLAN_GLOB)):
        runs.append(folder)                # single-depot runs save straight into this folder
    if not runs:
        raise SystemExit(f"No {title} runs with a {PLAN_GLOB} file found in {folder}")
    runs = sorted(runs, key=lambda d: d.stat().st_mtime, reverse=True)[:5]
    print(f"\nMost recent {title} runs:")
    for i, d in enumerate(runs, start=1):
        saved = datetime.fromtimestamp(d.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        label = d.name if d != folder else f"(single-depot files in this folder)"
        print(f"  {i}. {label}  (saved {saved})")
    if AUTO:
        chosen = runs[0]                   # newest run, already sorted first
        print(f"Using most recent {title} run: {chosen.name if chosen != folder else folder.name}")
        return chosen
    choice = input("Enter the number to use: ").strip()
    if not choice.isdigit() or not 1 <= int(choice) <= len(runs):
        raise SystemExit("Not a valid choice")
    return runs[int(choice) - 1]


def choose_anneal_run():
    return choose_run(ANNEAL_FOLDER, "anneal")

# ---- Loading --------------------------------------------------------------
def num(s):
    s = s.astype(str).str.replace(",", "", regex=False).str.strip()
    return pd.to_numeric(s, errors="coerce").fillna(0.0)


def load_orders(path):
    raw = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    o = pd.DataFrame({
        "date": pd.to_datetime(raw.iloc[:, DATE_COL].str.strip(), format="%Y%m%d", errors="coerce"),
        "team": raw.iloc[:, TEAM_COL].str.strip(),
        "route": raw.iloc[:, ROUTE_COL].str.strip(),
        "area": raw.iloc[:, AREA_COL].str.strip(),
        "kg": num(raw.iloc[:, KG_COL]),
        "pal": num(raw.iloc[:, PAL_COL]),
    })
    print(f"Rows read: {len(o):,}")
    print(f"Unreadable dates dropped: {o['date'].isna().sum()}")
    o = o.dropna(subset=["date"])
    o["depot"] = o["team"].map(DEPOT_CODES)
    unknown = o.loc[o["depot"].isna(), "team"].value_counts()
    if len(unknown):
        print(f"Unknown team codes dropped: {unknown.to_dict()}")
    o = o[o["depot"].notna()].copy()
    o["day"] = o["date"].dt.day_name()
    iso = o["date"].dt.isocalendar()
    o["wk"] = iso["year"].astype(int).astype(str) + "-W" + iso["week"].astype(int).astype(str).str.zfill(2)
    return o.reset_index(drop=True)


def timed_input(prompt, seconds, default):
    """Typed line, or default if nothing is typed within `seconds`.

    The timer stops once typing starts, so a half-typed answer is not cut short.
    Windows console (msvcrt); on other systems, select() on stdin.
    """
    print(prompt, end="", flush=True)
    try:
        import msvcrt
    except ImportError:
        import select
        import sys
        ready, _, _ = select.select([sys.stdin], [], [], seconds)
        if ready:
            return sys.stdin.readline().strip()
        print()
        return default

    import time
    buf, deadline = [], time.time() + seconds
    while True:
        if msvcrt.kbhit():
            ch = msvcrt.getwch()
            if ch in ("\r", "\n"):
                print()
                return "".join(buf).strip()
            if ch == "\b":
                if buf:
                    buf.pop()
                    print("\b \b", end="", flush=True)
            else:
                buf.append(ch)
                print(ch, end="", flush=True)
        elif not buf and time.time() >= deadline:
            print()
            return default
        else:
            time.sleep(0.05)


def choose_plan_source():
    """The plan to use: an anneal run. A consolidated plan file is not yet defined."""
    return choose_anneal_run()


def choose_planning_source():
    """Planning step only: anneal or sweep output. Waits for a valid answer (no default)."""
    if SWEEP_FOLDER is None:
        raise SystemExit("S_OUTPUT_DIR is not set in .env, so sweep runs cannot be chosen")
    while True:
        choice = input("Plan source: 1 = Anneal, 2 = Sweep: ").strip()
        if choice == "1":
            return choose_run(ANNEAL_FOLDER, "anneal")
        if choice == "2":
            return choose_run(SWEEP_FOLDER, "sweep")
        print("Please type 1 or 2")


def check_csv(path):
    """Stop with a clear message if the orders path is not a file."""
    p = Path(path)
    if not p.is_file():
        kind = "a folder" if p.is_dir() else "not found"
        raise SystemExit(f"Raw csv {kind}: {p}\n"
                         "Check C_OUTPUT_PATH in .env points to the orders CSV file itself.")
    return p


def choose_orders_path():
    """Raw orders CSV. C_OUTPUT_PATH is used after the timer unless another path is typed."""
    if not C_OUTPUT_PATH:
        return check_csv(ask_path("Please enter path to raw csv file: "))
    print(f"\nRaw csv file: {C_OUTPUT_PATH}")
    answer = timed_input(f"Press Enter to use it, or type another path "
                         f"(used after {PROMPT_SECONDS} seconds): ", PROMPT_SECONDS, "")
    if answer == "":
        return check_csv(C_OUTPUT_PATH)
    return check_csv(answer.strip().strip('"').strip("'"))


def load_plan(source):
    """depot -> (area -> new days, set of pallet-network areas).

    source is a folder of Anneal_*_Regional_*.xlsx files, or one such file.
    """
    source = Path(source)
    files = [source] if source.is_file() else sorted(source.glob(PLAN_GLOB))
    if not files:
        raise SystemExit(f"No plan files ({PLAN_GLOB}) in {source}\n"
                         "Without a plan every order would go to the pallet network. "
                         "Pick a run folder that holds the anneal or sweep workbooks.")
    plans = {}
    for f in files:
        depot = str(pd.read_excel(f, sheet_name="Summary").loc[0, "Depot"]).strip()
        dist = pd.read_excel(f, sheet_name="Districts", dtype=str).fillna("")
        outl = pd.read_excel(f, sheet_name="Outliers", dtype=str).fillna("")
        new = {}
        for _, r in dist.iterrows():
            days = [d.strip() for d in r["New days"].split(",") if d.strip()]
            if days:
                new[r[AREA].strip()] = days
        status = outl["Status"].str.strip()
        pallet_areas = set(outl.loc[status.str.match(r"(External|Excluded)"), AREA].str.strip())
        plans[depot] = (new, pallet_areas)
        print(f"Plan: {depot} - {len(new)} areas with days, "
              f"{len(pallet_areas)} pallet-network areas")
    return plans


def pick_weeks(o):
    days_by_wk = o.groupby("wk")["day"].agg(set)
    complete = [wk for wk, days in days_by_wk.items() if set(DAYS) <= days]
    excluded = sorted(set(days_by_wk.index) - set(complete))
    if excluded:
        print(f"Excluded weeks without orders on every weekday: {excluded}")
    if not complete:
        raise SystemExit("No week has orders on all five weekdays")
    kg = o[o["wk"].isin(complete)].groupby("wk")["kg"].sum().sort_values()
    n, k, mid = len(kg), 10, len(kg) // 2
    if n < 3 * k:
        print(f"Only {n} complete weeks, so the bands will be smaller or overlap")
    bands = {"Bottom 10": list(kg.index[:k]),
             "Middle 10": list(kg.index[max(0, mid - k // 2): mid + k // 2]),
             "Top 10": list(kg.index[max(0, n - k):])}
    seen, picks = set(), []
    for band, wks in bands.items():
        for wk in wks:
            if wk not in seen:
                seen.add(wk)
                picks.append((band, wk))
    for band in bands:
        ws = [wk for b, wk in picks if b == band]
        if ws:
            print(f"{band}: {len(ws)} weeks, {kg[ws].min():,.0f} to {kg[ws].max():,.0f} kg")
    return picks


# ---- Vehicle types ----------------------------------------------------------
# A vehicle is {"type": "HGV", "26t" or "Van", "kg": float, "pal": float, "items": [(area, kg, pal), ...]}.
# Items are whole orders, so an order is never split between vehicles.
# A truck starts as "HGV" (8t) and becomes "26t" only when an upgrade is needed (see take).

def order_fits_type(kg, pal, vtype):
    """True if one order fits an empty vehicle of this type."""
    spec = VEHICLE_TYPES[vtype]
    return kg <= spec["kg"] and pal <= spec["pallets"]


def fits_type(orders, vtype):
    """Boolean mask: which orders in a DataFrame fit an empty vehicle of this type."""
    spec = VEHICLE_TYPES[vtype]
    return (orders["kg"] <= spec["kg"]) & (orders["pal"] <= spec["pallets"])


def route_is_van(kg, pal):
    """Actual routes: a route is a van route if the whole load fits a van."""
    return order_fits_type(kg, pal, "Van")


def caps_for(depot):
    """Hard route km ceiling per vehicle type for one depot (HGV and 26t: depot truck cap; Van: 500 km)."""
    caps = {vt: spec["max_km"] for vt, spec in VEHICLE_TYPES.items()}
    caps.update(ROUTE_CAP.get(depot, {}))
    caps["26t"] = caps["HGV"]
    return caps


def fleet_for(depot):
    """Vehicles available per day per type for one depot. None = not capped."""
    limits = FLEET_CAP.get(depot, {})
    return {vt: limits.get(vt) for vt in VEHICLE_TYPES}


# ---- Packing --------------------------------------------------------------

def areas_of(v):
    """area -> kg for one vehicle."""
    out = {}
    for area, kg, _ in v["items"]:
        out[area] = out.get(area, 0.0) + kg
    return out


def fits(v, kg, pal):
    """True if the order fits in the vehicle as it is now (its current type's size)."""
    spec = VEHICLE_TYPES[v["type"]]
    return v["kg"] + kg <= spec["kg"] and v["pal"] + pal <= spec["pallets"]


def add_item(v, item):
    _, kg, pal = item
    v["items"].append(item)
    v["kg"] += kg
    v["pal"] += pal


def copy_vehicle(v):
    return {"type": v["type"], "kg": v["kg"], "pal": v["pal"], "items": list(v["items"])}


def can_take(v, kg, pal, trucks, lim26):
    """True if truck v can take the order as it is, or as a 26t upgrade of an 8t route.

    An upgrade needs room in the 26t fleet: fewer 26t trucks in `trucks` than lim26
    (None = no limit). trucks is the set of trucks in the plan being built.
    """
    if fits(v, kg, pal):
        return True
    if v["type"] != "HGV":
        return False
    if lim26 is not None and sum(1 for t in trucks if t["type"] == "26t") >= lim26:
        return False
    spec = VEHICLE_TYPES["26t"]
    return v["kg"] + kg <= spec["kg"] and v["pal"] + pal <= spec["pallets"]


def take(v, item):
    """Add an order to a truck, upgrading an 8t route to 26t if its load now needs it."""
    add_item(v, item)
    h8 = VEHICLE_TYPES["HGV"]
    if v["type"] == "HGV" and (v["kg"] > h8["kg"] or v["pal"] > h8["pallets"]):
        v["type"] = "26t"


def route_km_fn(coords, depot_xy):
    """Returns route_km(areas): closed-loop km for a set of areas, cached."""
    cache = {}

    def route_km(areas):
        key = frozenset(areas)
        if key not in cache:
            cache[key] = vehicle_route(sorted(key), coords, depot_xy)[1] if key else 0.0
        return cache[key]

    return route_km


def added_ok(current, area, route_km, limit):
    """True if adding `area` to a vehicle serving `current` areas adds at most `limit` km to its loop.

    An empty vehicle always passes (its first stop is limited by the route cap instead).
    limit None means no limit.
    """
    if limit is None or not current or area in current:
        return True
    return route_km(current | {area}) - route_km(current) <= limit


def select_van_loads(eligible, n_vans, route_km, cap, limit=None):
    """Choose the van-eligible orders that go on vans: as many orders as possible.

    Orders are taken smallest first (kg, then pallets). Each goes into the van, among
    the n_vans vans of the day, that it fills most tightly, provided the van's closed
    route stays within cap (500 km, hard) and the order adds at most `limit` km to the
    van's loop (added_ok). An order no van can take is returned unplaced, for the trucks.
    n_vans None means no fleet limit.
    Returns (van loads, unplaced index labels).
    """
    spec = VEHICLE_TYPES["Van"]
    if n_vans is None:
        n_vans = len(eligible)
    loads = [{"type": "Van", "kg": 0.0, "pal": 0.0, "items": []} for _ in range(n_vans)]
    unplaced = []
    ordered = eligible.sort_values(["kg", "pal"], kind="stable")
    for idx, area, kg, pal in zip(ordered.index, ordered["area"], ordered["kg"], ordered["pal"]):
        best = None
        for v in loads:
            if not fits(v, kg, pal):
                continue
            current = set(areas_of(v))
            if route_km(current | {area}) > cap or not added_ok(current, area, route_km, limit):
                continue
            left = spec["kg"] - (v["kg"] + kg)
            if best is None or left < best[0]:
                best = (left, v)
        if best is None:
            unplaced.append(idx)
        else:
            add_item(best[1], (area, kg, pal))
    return [v for v in loads if v["items"]], unplaced


def place(orders, route_km, cap, vtype="HGV"):
    """Pack orders into vehicles of one size with no fleet limit, first fit within each postcode area.

    vtype: the route size, "HGV" (8t) or "26t". cap: hard route km ceiling. A new route is
    only opened for an order whose own closed loop is within cap, so a single far-away stop
    cannot start a route over the ceiling. Orders must already fit an empty vehicle of vtype.
    Returns (vehicles, orders blocked by the route cap as (area, kg, pal) tuples).
    """
    vehicles, cap_over = [], []
    area_order = orders.groupby("area")["kg"].sum().sort_values(ascending=False).index
    for area in area_order:
        own = []
        items = orders[orders["area"] == area].sort_values("kg", ascending=False)
        for kg, pal in zip(items["kg"], items["pal"]):
            v = next((v for v in own if fits(v, kg, pal)
                      and route_km(set(areas_of(v)) | {area}) <= cap), None)
            if v is None:
                if route_km({area}) > cap:
                    cap_over.append((area, kg, pal))
                    continue
                v = {"type": vtype, "kg": 0.0, "pal": 0.0, "items": []}
                own.append(v)
            add_item(v, (area, kg, pal))
        vehicles.extend(own)
    return vehicles, cap_over


def consolidate(vehicles, coords, route_km, caps, limit=None):
    """Remove vans by moving their orders into the other vans.

    Smallest vehicles are tried first, and their orders largest first. Each order
    goes to the vehicle where the closed route adds the fewest km, provided the
    load fits, the route stays within the cap and the order adds at most `limit` km.
    A vehicle is removed only if every one of its orders is placed. Repeats until no vehicle can be emptied.
    """
    changed = True
    while changed:
        changed = False
        for src in sorted(vehicles, key=lambda v: v["kg"]):
            cap = caps[src["type"]]
            trial = [copy_vehicle(v) for v in vehicles if v is not src]
            placed_all = True
            for item in sorted(src["items"], key=lambda it: -it[1]):
                area, kg, pal = item
                if area not in coords:         # no grid reference: distance unknown, never moved
                    placed_all = False
                    break
                best = None
                for dst in trial:
                    if dst["type"] != src["type"]:
                        continue
                    if not fits(dst, kg, pal):
                        continue
                    current = set(areas_of(dst))
                    new_km = route_km(current | {area})
                    if new_km > cap:
                        continue
                    add = new_km - route_km(current)
                    if current and limit is not None and add > limit:
                        continue
                    if best is None or add < best[0]:
                        best = (add, dst)
                if best is None:
                    placed_all = False
                    break
                add_item(best[1], item)
            if placed_all:
                vehicles = trial
                changed = True
                break
    return vehicles


def consolidate_trucks(trucks, coords, route_km, cap, lim26, limit=None):
    """Empty trucks into other trucks, so the fewest trucks are used.

    Smallest trucks are emptied first, and their orders largest first. Each order goes
    to the truck where the closed route adds the fewest km, and at most `limit` km. A truck that can't take an
    order at 8t is upgraded to 26t if the 26t fleet has room (see can_take). A truck is
    removed only if every one of its orders is placed. Repeats until no truck can be emptied.
    """
    changed = True
    while changed:
        changed = False
        for src in sorted(trucks, key=lambda v: v["kg"]):
            trial = [copy_vehicle(v) for v in trucks if v is not src]
            placed_all = True
            for item in sorted(src["items"], key=lambda it: -it[1]):
                area, kg, pal = item
                if area not in coords:         # no grid reference: distance unknown, never moved
                    placed_all = False
                    break
                best = None
                for dst in trial:
                    if not can_take(dst, kg, pal, trial, lim26):
                        continue
                    current = set(areas_of(dst))
                    new_km = route_km(current | {area})
                    if new_km > cap:
                        continue
                    add = new_km - route_km(current)
                    if current and limit is not None and add > limit:
                        continue
                    if best is None or add < best[0]:
                        best = (add, dst)
                if best is None:
                    placed_all = False
                    break
                take(best[1], item)
            if placed_all:
                trucks = trial
                changed = True
                break
    return trucks


def fit_fleet(trucks, fleet, route_km, cap):
    """Trim trucks to the fleet. Returns (kept trucks, fleet overflow as (area, kg, pal) tuples).

    26t trucks over the 26t fleet are dropped first, then 8t trucks over the 8t fleet.
    A dropped truck's orders go into the kept trucks where they fit within the cap (an 8t
    kept truck may be upgraded only if the 26t fleet has room). The truck with the fewest
    orders is dropped first. Orders that still cannot be placed are the fleet overflow.
    """
    lim8, lim26 = fleet["HGV"], fleet["26t"]
    kept = list(trucks)
    overflow = []

    def n_type(t):
        return sum(1 for v in kept if v["type"] == t)

    def drop_one(vtype):
        cands = [v for v in kept if v["type"] == vtype]
        dropped = min(cands, key=lambda v: (len(v["items"]), v["kg"]))
        kept[:] = [v for v in kept if v is not dropped]
        for item in sorted(dropped["items"], key=lambda it: (it[1], it[2])):
            area, kg, pal = item
            best = None
            for dst in kept:
                if not can_take(dst, kg, pal, kept, lim26):
                    continue
                current = set(areas_of(dst))
                new_km = route_km(current | {area})
                if new_km > cap:
                    continue
                add = new_km - route_km(current)
                if best is None or add < best[0]:
                    best = (add, dst)
            if best is None:
                overflow.append(item)
            else:
                take(best[1], item)

    while lim26 is not None and n_type("26t") > lim26:
        drop_one("26t")
    while lim8 is not None and n_type("HGV") > lim8:
        drop_one("HGV")
    return kept, overflow


def over_fleet(trucks, fleet):
    """True if there are more 8t or 26t trucks than the depot's fleet allows."""
    _, n8, n26 = type_counts(trucks)
    return ((fleet["HGV"] is not None and n8 > fleet["HGV"])
            or (fleet["26t"] is not None and n26 > fleet["26t"]))


def pack(orders, coords, depot_xy, caps, fleet):
    """Pack one depot-day, relaxing the added-km limit while the trucks are over the fleet.

    Tries MAX_ADDED_KM, then steps of ADDED_KM_STEP up to ADDED_KM_CEILING. The first limit
    whose trucks fit the fleet is used. If none does, the ceiling run is used and the day is
    flagged unresolved: fit_fleet then trims it to the fleet, sending the rest to overflow.

    Returns (vehicles, overflow DataFrame, final added-km limit, unresolved flag).
    """
    route_km = route_km_fn(coords, depot_xy)       # shared cache across the retries
    limit = MAX_ADDED_KM
    while True:
        vehicles, overflow, over = pack_at(orders, coords, route_km, caps, fleet, limit)
        if not over or limit >= ADDED_KM_CEILING:
            return vehicles, overflow, limit, over
        limit = min(limit + ADDED_KM_STEP, ADDED_KM_CEILING)


def pack_at(orders, coords, route_km, caps, fleet, limit):
    """orders: rows with area, kg, pal for one depot and day.

    caps: type -> hard route km ceiling (trucks: depot cap; Van: 500 km).
    fleet: type -> most vehicles on the day (None = no limit).

    limit: added-km limit for placing an order on a non-empty vehicle (added_ok). It applies
    to the van selection and both consolidations. Stage 1 truck packing (place) only puts
    an order with others from its own postcode area, so it adds no km there. fit_fleet,
    the last-resort trim to the fleet, uses the route cap only.

    1. Van-eligible orders: select as many as possible onto the van fleet, with each
       van's route within 500 km.
    2. Everything else is packed onto 8t routes by area within the cap. Orders over 8t
       go onto 26t routes. Orders over 26t limits are "too big".
    3. Consolidation empties trucks into others. A truck that cannot take an order at 8t
       is upgraded to 26t if the 26t fleet has room (consolidate_trucks).
    4. Trucks are trimmed to the fleet (fit_fleet). Orders that still do not fit are
       fleet overflow.

    Returns (vehicles, overflow DataFrame with Reason "route cap", "too big" or "fleet",
    True if the trucks were over the fleet before fit_fleet).
    """
    spec = VEHICLE_TYPES["Van"]
    eligible = (orders["kg"] <= spec["kg"]) & (orders["pal"] <= spec["pallets"])

    van_loads, unplaced_van = select_van_loads(orders[eligible], fleet["Van"], route_km, caps["Van"], limit)
    rest = pd.concat([orders[~eligible], orders.loc[unplaced_van]])
    fits_8 = fits_type(rest, "HGV")
    fits_26 = fits_type(rest, "26t")
    too_big = rest[~fits_26]
    hgv, cap_over_8 = place(rest[fits_8], route_km, caps["HGV"], "HGV")
    heavy, cap_over_26 = place(rest[~fits_8 & fits_26], route_km, caps["26t"], "26t")

    trucks = consolidate_trucks(hgv + heavy, coords, route_km, caps["HGV"], fleet["26t"], limit)
    over = over_fleet(trucks, fleet)
    trucks, fleet_over = fit_fleet(trucks, fleet, route_km, caps["HGV"])
    vans = consolidate(van_loads, coords, route_km, caps, limit)

    rows = ([("route cap", a, k, p) for a, k, p in cap_over_8 + cap_over_26]
            + [("too big", a, k, p) for a, k, p in too_big[["area", "kg", "pal"]].itertuples(index=False, name=None)]
            + [("fleet", a, k, p) for a, k, p in fleet_over])
    overflow = pd.DataFrame(rows, columns=["Reason", "area", "kg", "pal"])
    return vans + trucks, overflow, over


def vehicle_rows(week, depot, day, plan, vehicles, coords, depot_xy):
    rows = []
    # trucks listed first (26t, then 8t), then vans, each by weight
    for i, v in enumerate(sorted(vehicles, key=lambda v: (v["type"] == "Van", v["type"] != "26t", -v["kg"])), start=1):
        areas = areas_of(v)
        order, km, stop_km, unlocated = vehicle_route(areas, coords, depot_xy)
        spec = VEHICLE_TYPES[v["type"]]
        loads = {}
        for area, kg, pal in v["items"]:
            k, p_, n = loads.get(area, (0.0, 0.0, 0))
            loads[area] = (k + kg, p_ + pal, n + 1)
        rows.append({"Week": week, "Depot": depot, "Day": day, "Plan": plan,
                     "Vehicle": i, "Type": v["type"], "Order": order, "Areas": ", ".join(sorted(areas)),
                     "Loads": loads,   # area -> (kg, pallets, orders): one drop per area; for the report only
                     "Drops": max(len(order) - 2, 0),
                     "KG": round(v["kg"]), "Pallets": round(v["pal"], 1),
                     "KG fill %": round(v["kg"] / spec["kg"], 2),
                     "Pallet fill %": round(v["pal"] / spec["pallets"], 2),
                     "Route km": round(km, 1),
                     "Over van 500 km": v["type"] == "Van" and km > VEHICLE_TYPES["Van"]["max_km"],
                     "Stop km": round(stop_km, 1),
                     "Areas without grid ref": len(unlocated)})
    return rows


# ---- Simulation -----------------------------------------------------------
def target_day(row, plan_new, pallet_areas, cache):
    if row["area"] in pallet_areas or row["area"] not in plan_new:
        return PALLET
    pattern = tuple(plan_new[row["area"]])
    key = (row["day"], pattern)
    if key not in cache:
        cache[key] = map_days([row["day"]], list(pattern))[row["day"]]
    return cache[key]


def type_counts(vehicles):
    """(vans, 8t HGVs, 26ts) in a list of vehicle dicts."""
    vans = sum(1 for v in vehicles if v["type"] == "Van")
    t26 = sum(1 for v in vehicles if v["type"] == "26t")
    return vans, len(vehicles) - vans - t26, t26


def overflow_counts(over):
    """(route cap orders, route cap kg, fleet orders, fleet kg) in an overflow table.

    "too big" orders (over the 26t limits) are counted in the route cap group.
    """
    r = over[over["Reason"].isin(["route cap", "too big"])]
    f = over[over["Reason"] == "fleet"]
    return len(r), round(r["kg"].sum()), len(f), round(f["kg"].sum())


def area_depots(plans):
    """area -> depot whose plan holds it (its Districts or Outliers sheet)."""
    out = {}
    for depot, (new, pallet_areas) in sorted(plans.items()):
        for area in list(new) + sorted(pallet_areas):
            if area in out and out[area] != depot:
                print(f"Area {area} is in the plans of both {out[area]} and {depot}; {out[area]} kept")
                continue
            out[area] = depot
    return out


def empty_pack():
    return [], pd.DataFrame(columns=["Reason", "area", "kg", "pal"]), MAX_ADDED_KM, False


_NOTED = set()   # depots with no plan already reported


def simulate(label, wk, wo, plans, coords, depot_xy):
    """Before: each order on its actual depot and day. After: each order on the depot whose
    plan holds its postcode area, on that area's new day (map_days).

    So a closing depot (no plan) keeps its "before" vehicles as a comparator, and its areas
    are served "after" by the depots they were assigned to. Areas in no plan, and outliers,
    go to the pallet network. Both sides use the same packing (pack).
    """
    out = {"Daily": [], "Vehicles": [], "Moves": [], "Pallet_areas": [], "Overflow": [],
           "Checks": [], "Orders": []}
    weekday = wo[wo["day"].isin(DAYS)].copy()
    out["Checks"].append({"Week": label, "Week ID": wk,
                          "Raw kg": round(wo["kg"].sum()),
                          "Weekend kg (not simulated)": round(wo.loc[~wo["day"].isin(DAYS), "kg"].sum())})

    area_dep = area_depots(plans)
    weekday["new_depot"] = [area_dep.get(a, PALLET) for a in weekday["area"]]
    if len(weekday) and not (weekday["new_depot"] != PALLET).any():
        order_areas = sorted(set(weekday["area"]))[:10]
        plan_areas = sorted(area_dep)[:10]
        raise SystemExit(f"{label}: no postcode area in the orders matches any area in the plan, "
                         f"so every order would go to the pallet network.\n"
                         f"  Orders areas (first 10): {order_areas}\n"
                         f"  Plan areas (first 10):   {plan_areas}\n"
                         f"  Plan depots: {sorted(plans) or 'none'}\n"
                         "Check the plan run folder and that the orders file's area column "
                         "(column AN) holds postcode districts in the same format as the plan.")
    unmatched = sorted(set(weekday.loc[weekday["new_depot"] == PALLET, "area"]))
    if unmatched:
        print(f"{label}: {len(unmatched)} areas in no plan, sent to pallet network: {unmatched}")
    caches = {d: {} for d in plans}
    weekday["new_day"] = [
        PALLET if nd == PALLET else target_day(r, plans[nd][0], plans[nd][1], caches[nd])
        for nd, (_, r) in zip(weekday["new_depot"], weekday.iterrows())]
    weekday.loc[weekday["new_day"] == PALLET, "new_depot"] = PALLET
    out["Orders"].append(weekday.assign(Week=label))

    no_plan = sorted(set(weekday["depot"]) - set(plans) - _NOTED)
    _NOTED.update(no_plan)
    for depot in no_plan:
        print(f"{label} {depot}: no plan (closing or not active): kept in 'before' as a comparator, "
              f"its areas served 'after' by their new depots")
    depots = sorted(set(weekday["depot"]) | set(plans))
    for depot in depots:
        if depot not in depot_xy:
            raise SystemExit(f"No depot position for '{depot}'. Check the depot names in the matrix "
                             f"or the depot info file match [depots.team_codes]: {sorted(depot_xy)}")

    for depot in depots:
        caps = caps_for(depot)
        fleet = fleet_for(depot)
        own = weekday[weekday["depot"] == depot]          # before: its actual orders
        served = weekday[weekday["new_depot"] == depot]   # after: the areas its plan holds
        if own.empty and served.empty:
            continue
        moved_out_total = moved_in_total = 0.0
        limits = {"Before": [], "After": []}
        for day in DAYS:
            before = own[own["day"] == day]
            after = served[served["new_day"] == day]
            veh_b, over_b, lim_b, unres_b = (pack(before, coords, depot_xy[depot], caps, fleet)
                                             if not before.empty else empty_pack())
            veh_a, over_a, lim_a, unres_a = (pack(after, coords, depot_xy[depot], caps, fleet)
                                             if not after.empty else empty_pack())
            limits["Before"].append(f"{day[:3]} {lim_b:.0f}{' UNRESOLVED' if unres_b else ''}")
            limits["After"].append(f"{day[:3]} {lim_a:.0f}{' UNRESOLVED' if unres_a else ''}")
            # moved out: leaves this depot-day for another day or depot (not the pallet network)
            moved_out = before.loc[((before["new_day"] != day) | (before["new_depot"] != depot))
                                   & (before["new_day"] != PALLET), "kg"].sum()
            moved_in = after.loc[(after["day"] != day) | (after["depot"] != depot), "kg"].sum()
            moved_out_total += moved_out
            moved_in_total += moved_in
            pallet = before[before["new_day"] == PALLET]

            rows_b = vehicle_rows(label, depot, day, "Before", veh_b, coords, depot_xy[depot])
            rows_a = vehicle_rows(label, depot, day, "After", veh_a, coords, depot_xy[depot])
            out["Vehicles"] += rows_b + rows_a

            # Actual routes for the day, split by whether the whole route fits a van
            rt = before.groupby("route")[["kg", "pal"]].sum()
            van_routes = int(sum(route_is_van(k, p) for k, p in zip(rt["kg"], rt["pal"])))
            vans_b, hgvs_b, t26_b = type_counts(veh_b)
            vans_a, hgvs_a, t26_a = type_counts(veh_a)
            rc_b_n, rc_b_kg, fl_b_n, fl_b_kg = overflow_counts(over_b)
            rc_a_n, rc_a_kg, fl_a_n, fl_a_kg = overflow_counts(over_a)

            out["Daily"].append({
                "Week": label, "Depot": depot, "Day": day,
                "Routes before": before["route"].nunique(),
                "Van routes before": van_routes,
                "HGV routes before": before["route"].nunique() - van_routes,
                "Vehicles before": len(veh_b),
                "Vans before": vans_b,
                "HGVs before": hgvs_b,
                "26t before": t26_b,
                "Vans over 500 km before": sum(1 for r in rows_b if r["Over van 500 km"]),
                "Over route cap before orders": rc_b_n,
                "Over route cap before KG": rc_b_kg,
                "Over fleet before orders": fl_b_n,
                "Over fleet before KG": fl_b_kg,
                "KG before": round(before["kg"].sum()),
                "Pallets before": round(before["pal"].sum(), 1),
                "Pallet network KG": round(pallet["kg"].sum()),
                "Moved out KG": round(moved_out),
                "Moved in KG": round(moved_in),
                "Vehicles after": len(veh_a),
                "Vans after": vans_a,
                "HGVs after": hgvs_a,
                "26t after": t26_a,
                "Vans over 500 km after": sum(1 for r in rows_a if r["Over van 500 km"]),
                "Over route cap after orders": rc_a_n,
                "Over route cap after KG": rc_a_kg,
                "Over fleet after orders": fl_a_n,
                "Over fleet after KG": fl_a_kg,
                "KG after": round(after["kg"].sum()),
                "Pallets after": round(after["pal"].sum(), 1),
                "Added km limit before": lim_b,
                "Added km limit after": lim_a,
                "Fleet unresolved before": int(unres_b),
                "Fleet unresolved after": int(unres_a),
            })

            for _, r in over_a.iterrows():
                out["Overflow"].append({"Week": label, "Depot": depot, "Day": day, "Area": r["area"],
                                        "Reason": r["Reason"], "KG": round(r["kg"]),
                                        "Pallets": round(r["pal"], 1)})

        out["Checks"].append({"Week": label, "Week ID": wk, "Depot": depot,
                              "Simulated KG": round(own["kg"].sum()),
                              "Moved out KG": round(moved_out_total),
                              "Moved in KG": round(moved_in_total),
                              "Moved difference": round(moved_out_total - moved_in_total, 1)})

    # Moves and pallet-network areas, by the depot the orders came from
    o = weekday
    mv = (o[((o["new_day"] != o["day"]) | (o["new_depot"] != o["depot"])) & (o["new_day"] != PALLET)]
          .groupby(["depot", "area", "day", "new_depot", "new_day"])
          .agg(kg=("kg", "sum"), orders=("kg", "size")).reset_index())
    for _, r in mv.iterrows():
        out["Moves"].append({"Week": label, "Depot": r["depot"], "Area": r["area"],
                             "Old day": r["day"], "New depot": r["new_depot"], "New day": r["new_day"],
                             "KG": round(r["kg"]), "Orders": int(r["orders"])})

    pa = (o[o["new_day"] == PALLET].groupby(["depot", "area", "day"])
          .agg(kg=("kg", "sum"), pal=("pal", "sum")).reset_index())
    for _, r in pa.iterrows():
        out["Pallet_areas"].append({"Week": label, "Depot": r["depot"], "Area": r["area"],
                                    "Day": r["day"], "KG": round(r["kg"]),
                                    "Pallets": round(r["pal"], 1)})

    # Network check: kg moved out of depot-days equals kg moved in (should be 0)
    out_kg = sum(c.get("Moved out KG", 0) for c in out["Checks"] if "Depot" in c)
    in_kg = sum(c.get("Moved in KG", 0) for c in out["Checks"] if "Depot" in c)
    out["Checks"].append({"Week": label, "Week ID": wk, "Depot": "All depots (should be 0)",
                          "Moved out KG": out_kg, "Moved in KG": in_kg,
                          "Moved difference": round(out_kg - in_kg, 1)})
    return out


def route_detail_rows(label, res, coords, depot_xy):
    """Every point of every After route, with leg and cumulative km."""
    rows = []
    for v in res["Vehicles"]:
        if v["Plan"] != "After" or not v["Order"]:
            continue
        for leg in route_legs(v["Order"], coords, depot_xy[v["Depot"]]):
            rows.append({"Week": label, "Depot": v["Depot"], "Day": v["Day"],
                         "Vehicle": v["Vehicle"], "Type": v["Type"], **leg})
    return rows


def route_rows(label, res, actual=()):
    daily = pd.DataFrame(res["Daily"])
    if daily.empty:
        return []
    vehicles = pd.DataFrame(res["Vehicles"])
    pallets = pd.DataFrame(res["Pallet_areas"])
    after = vehicles[vehicles["Plan"] == "After"] if not vehicles.empty else vehicles
    rows = []
    for depot in sorted(daily["Depot"].unique()):
        for day in DAYS:
            if not after.empty:
                g = after[(after["Depot"] == depot) & (after["Day"] == day)].sort_values("Vehicle")
                for _, r in g.iterrows():
                    rows.append({"Week": label, "Depot": depot, "Day": day, "Route": int(r["Vehicle"]),
                                 "Type": r["Type"], "Weight": int(r["KG"]), "Pallets": r["Pallets"],
                                 "Drops": int(r["Drops"]),
                                 "Route km": r["Route km"], "Stop km": r["Stop km"],
                                 "Over van 500 km": bool(r["Over van 500 km"]),
                                 "Areas without grid ref": int(r["Areas without grid ref"]),
                                 **comparator_fields(r, actual)})
            pn = pallets[(pallets["Depot"] == depot) & (pallets["Day"] == day)] if not pallets.empty else pallets
            rows.append({"Week": label, "Depot": depot, "Day": day, "Route": "Pallet Network",
                         "Type": "", "Weight": int(pn["KG"].sum()) if not pn.empty else 0,
                         "Pallets": round(pn["Pallets"].sum(), 1) if not pn.empty else 0,
                         "Drops": None, "Route km": None, "Stop km": None,
                         "Over van 500 km": None, "Areas without grid ref": None})
    return rows


def summary_rows(label, res):
    daily = pd.DataFrame(res["Daily"])
    if daily.empty:
        return []
    vehicles = pd.DataFrame(res["Vehicles"])
    after = vehicles[vehicles["Plan"] == "After"] if not vehicles.empty else vehicles
    rows = []

    def daily_pivot(col):
        return daily.pivot_table(index="Depot", columns="Day", values=col, aggfunc="sum")

    actual = daily_pivot("Routes before")

    def planned(vtype=None):
        if after.empty:
            return pd.DataFrame(index=actual.index)
        sub = after if vtype is None else after[after["Type"] == vtype]
        if sub.empty:
            return pd.DataFrame(index=actual.index)
        return sub.groupby(["Depot", "Day"]).size().unstack()

    bases = [("Actual routes", actual),
             ("Actual van routes", daily_pivot("Van routes before")),
             ("Actual HGV routes", daily_pivot("HGV routes before")),
             ("Planned vehicles", planned()),
             ("Planned vans", planned("Van")),
             ("Planned HGVs", planned("HGV")),
             ("Planned 26t", planned("26t")),
             ("Vans over 500 km (after, check)", daily_pivot("Vans over 500 km after")),
             ("Orders over route cap (after)", daily_pivot("Over route cap after orders")),
             ("Orders over fleet (after)", daily_pivot("Over fleet after orders")),
             ("Orders over route cap (before)", daily_pivot("Over route cap before orders")),
             ("Orders over fleet (before)", daily_pivot("Over fleet before orders")),
             ("Added km limit (after)", daily_pivot("Added km limit after")),
             ("Added km limit (before)", daily_pivot("Added km limit before")),
             ("Fleet cap unresolved (after, flag)", daily_pivot("Fleet unresolved after")),
             ("Fleet cap unresolved (before, flag)", daily_pivot("Fleet unresolved before"))]
    for basis, counts in bases:
        counts = counts.reindex(index=actual.index, columns=DAYS).fillna(0).astype(int)
        for depot, row in counts.iterrows():
            rows.append({"Week": label, "Basis": basis, "Depot": depot,
                         **{d: int(row[d]) for d in DAYS}, "Total": int(row.sum())})

    if not after.empty:
        km = sum_route_km(after, ["Depot", "Day"]).pivot(index="Depot", columns="Day", values="Route km")
        km = km.reindex(index=actual.index, columns=DAYS).fillna(0.0)
        for depot, row in km.iterrows():
            rows.append({"Week": label, "Basis": "Planned route km", "Depot": depot,
                         **{d: round(float(row[d]), 1) for d in DAYS},
                         "Total": round(float(row.sum()), 1)})

    # Check only: packing stops at the fleet and the route ceilings, so these should always be zero.
    # A day the added-km relaxation could not bring within the fleet is in "Fleet cap unresolved"
    # instead; fit_fleet trims it, with the extra orders in "Orders over fleet".
    for vtype, name in [("Van", "Vans over fleet cap"), ("HGV", "HGVs over fleet cap"), ("26t", "26t over fleet cap")]:
        over = planned(vtype).reindex(index=actual.index, columns=DAYS).fillna(0)
        for depot, row in over.iterrows():
            cap = FLEET_CAP.get(depot, {}).get(vtype)
            if cap is None:
                continue
            excess = {d: max(0, int(row[d]) - cap) for d in DAYS}
            rows.append({"Week": label, "Basis": name, "Depot": depot,
                         **excess, "Total": sum(excess.values())})
    return rows


def actual_route_rows(orders, coords, depot_xy):
    """Each real route (week, depot, day, route ID) measured as one closed straight-line loop.

    Week is in the key so a route ID reused in another week is a separate route.
    """
    rows = []
    o = orders[orders["day"].isin(DAYS)]
    for (wk, depot, day, route), g in o.groupby(["wk", "depot", "day", "route"]):
        if depot not in depot_xy:
            continue
        areas = sorted(set(g["area"]))
        _, km, _, unlocated = vehicle_route(areas, coords, depot_xy[depot])
        kg, pal = g["kg"].sum(), g["pal"].sum()
        rows.append({"Week": wk, "Depot": depot, "Day": day, "Route": route,
                     "Type": "Van" if route_is_van(kg, pal) else "HGV",
                     "Areas": ", ".join(areas), "KG": round(kg), "Pallets": round(pal, 1),
                     "Route km": round(km, 1), "Areas without grid ref": len(unlocated),
                     "Drops": len(g),
                     # area -> order lines on this route; used to pick comparators, not written out
                     "_area_drops": g["area"].value_counts().to_dict()})
    return rows


def public(rows):
    """Rows without the internal "_" keys, for writing to Excel."""
    return [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]


def route_caps(rows):
    """Per depot: ROUTE_PCT percentile of actual HGV route km, never above the HGV max_km.

    Van caps are not set here: vans use their fixed 500 km ceiling (see caps_for).
    The 26t uses this same truck cap (see caps_for).
    """
    df = pd.DataFrame(rows)
    caps = {}
    if df.empty:
        return caps
    hgv = df[df["Type"] == "HGV"]
    for depot, g in hgv.groupby("Depot"):
        caps[depot] = {"HGV": min(round(float(g["Route km"].quantile(ROUTE_PCT)), 1),
                                  VEHICLE_TYPES["HGV"]["max_km"])}
    return caps


def route_km_summary(rows):
    """Median, 90th percentile and longest real route per depot and type."""
    df = pd.DataFrame(rows)
    if df.empty:
        return []
    out = []
    for (depot, vtype), g in df.groupby(["Depot", "Type"]):
        k = g["Route km"]
        out.append({"Depot": depot, "Type": vtype, "Routes": len(k), "Median km": round(k.median(), 1),
                    "90th percentile km": round(k.quantile(0.9), 1), "Longest km": round(k.max(), 1)})
    return out


def drops(v):
    """Stops on a planned route: the order list minus the depot at each end."""
    return max(len(v["Order"]) - 2, 0)


def pick_example_routes(vehicles, n_random=2, seed=1, min_drops=EXAMPLE_MIN_DROPS):
    """Per depot and vehicle type: the longest planned route, then n_random others (seeded).

    Vans, 8t HGVs and 26ts are sampled separately, so one type cannot crowd out the others.
    Only routes with more than 3 drops (min_drops = 4) are considered, so the examples
    show multi-stop routes rather than single long runs. A depot and type with no
    such route falls back to all its routes.
    """
    picks = []
    if not vehicles:
        return picks
    rng = random.Random(seed)
    by_group = {}
    for v in vehicles:
        by_group.setdefault((v["Depot"], v["Type"]), []).append(v)
    for depot, vtype in sorted(by_group):
        recs = by_group[(depot, vtype)]
        multi = [v for v in recs if drops(v) >= min_drops]
        if not multi:
            print(f"{depot} {vtype}: no planned route with more than 3 drops, using all {vtype} routes for examples")
            multi = recs
        longest = max(multi, key=lambda v: v["Route km"])
        others = [v for v in multi if v is not longest]
        picks.append(("Longest", longest))
        picks += [("Random", s) for s in rng.sample(others, min(n_random, len(others)))]
    return picks


def split_areas(text):
    """Areas column ("AL, SG, ...") back to a set."""
    return {a for a in (text or "").split(", ") if a}


def comparator_route(v, actual):
    """The one real route to compare a planned route with, or (None, "").

    Real routes are taken whole, as they ran, and never combined. A real route qualifies
    if more than COMPARATOR_MIN_SHARE of its drops (order lines) are in the planned
    route's areas; its other drops stay on it. Of those, the one with the most drops in
    the planned areas wins; a tie goes to the closest kg, then the route ID.
    Search: same week and day first (any depot, as areas move depot), then the same
    week on any day. Returns (route row, "Same day" or "Other day").
    """
    areas = split_areas(v["Areas"])
    if not areas:
        return None, ""

    def best(rows):
        scored = []
        for r in rows:
            ad = r.get("_area_drops") or {}
            total = sum(ad.values())
            inside = sum(n for a, n in ad.items() if a in areas)
            if total and inside / total > COMPARATOR_MIN_SHARE:
                scored.append((-inside, abs(r["KG"] - v["KG"]), str(r["Route"]), r))
        return min(scored, key=lambda t: t[:3])[3] if scored else None

    week = [r for r in actual if r["Week"] == v["Week"]]
    r = best([r for r in week if r["Day"] == v["Day"]])
    if r is not None:
        return r, "Same day"
    r = best(week)
    return (r, "Other day") if r is not None else (None, "")


def comparator_fields(v, actual):
    """Comparator columns for the Routes sheet."""
    r, basis = comparator_route(v, actual)
    if r is None:
        return {"Comparator route": "No comparable route", "Comparator basis": "",
                "Comparator depot": "", "Comparator day": "", "Comparator KG": None,
                "Comparator pallets": None, "Comparator drops": None,
                "Comparator drops in areas %": None, "Comparator route km": None}
    areas = split_areas(v["Areas"])
    ad = r["_area_drops"]
    inside = sum(n for a, n in ad.items() if a in areas)
    return {"Comparator route": r["Route"], "Comparator basis": basis,
            "Comparator depot": r["Depot"], "Comparator day": r["Day"], "Comparator KG": r["KG"],
            "Comparator pallets": r["Pallets"], "Comparator drops": r["Drops"],
            "Comparator drops in areas %": round(100 * inside / r["Drops"]),
            "Comparator route km": r["Route km"]}


def fill_pct(kg, pal, vtypes):
    """kg and pallet fill, as % of the combined capacity of the given vehicle types."""
    cap_kg = sum(VEHICLE_TYPES[t]["kg"] for t in vtypes)
    cap_pal = sum(VEHICLE_TYPES[t]["pallets"] for t in vtypes)
    return (100 * kg / cap_kg if cap_kg else 0.0), (100 * pal / cap_pal if cap_pal else 0.0)


def compare_footer(planned, actual):
    """Footer lines: planned against actual route count, km, kg and fill."""
    def line(label, rows):
        kg, pal = sum(r["KG"] for r in rows), sum(r["Pallets"] for r in rows)
        kg_f, pal_f = fill_pct(kg, pal, [r["Type"] for r in rows])
        return (f"{label}: {len(rows)} route{'s' if len(rows) != 1 else ''}, "
                f"{sum(r['Route km'] for r in rows):,.0f} km, {kg:,.0f} kg, "
                f"fill {kg_f:.0f}% kg / {pal_f:.0f}% pallets")
    return line("Planned", planned) + "\n" + line("Actual", actual)


class MapFrame:
    """One map figure: outline, depot star, loops in British National Grid metres."""

    def __init__(self, depot, coords, depot_xy, outline):
        self.depot, self.coords, self.depot_xy = depot, coords, depot_xy
        self.fig, self.ax = plt.subplots(figsize=(7, 7.6))
        if outline is not None:
            outline.plot(ax=self.ax, color="#dedede", edgecolor="white", linewidth=0.3)
        self.xs, self.ys = [], []

    def to_xy(self, order, depot=None):
        # coords are km; the outline is in metres (EPSG 27700), so convert here
        d = self.depot_xy[depot or self.depot]
        return [(d[0] * 1000, d[1] * 1000) if node == DEPOT_KEY
                else (self.coords[node][0] * 1000, self.coords[node][1] * 1000) for node in order]

    def loop(self, order, colour, lw, z, label=None, dashed=False, dot=14, depot=None):
        xy = self.to_xy(order, depot)
        xs, ys = [p[0] for p in xy], [p[1] for p in xy]
        self.ax.plot(xs, ys, color=colour, linewidth=lw, linestyle="--" if dashed else "-",
                     zorder=z, label=label)
        self.ax.scatter(xs[1:-1], ys[1:-1], s=dot, color=colour, linewidths=0, zorder=z)
        self.xs += xs
        self.ys += ys
        return xy

    def actual(self, r, label=True):
        """One actual route, grey and dashed, behind the planned routes (order from vehicle_route).

        Drawn from its own depot, which can differ from the map's depot for a comparator.
        """
        depot = r["Depot"] if r["Depot"] in self.depot_xy else self.depot
        order = vehicle_route(sorted(split_areas(r["Areas"])), self.coords, self.depot_xy[depot])[0]
        if not order:
            return
        xy = self.loop(order, ACTUAL_COLOUR, 0.9, 1, dashed=True, dot=8, depot=depot)
        if label:
            # label at the stop furthest from the depot
            far = max(xy[1:-1], key=lambda p: (p[0] - xy[0][0]) ** 2 + (p[1] - xy[0][1]) ** 2)
            self.ax.annotate(f"{r['Route']} {r['Route km']:.0f} km", far, xytext=(3, 3),
                             textcoords="offset points", fontsize=6, color="#555555", zorder=5)

    def save(self, title, footer, out_png, legend_cols=1):
        ax = self.ax
        d = self.to_xy([DEPOT_KEY])[0]
        ax.scatter([d[0]], [d[1]], marker="*", s=220, color="#c00000", zorder=6)
        xs, ys = self.xs or [d[0]], self.ys or [d[1]]
        # square view centred on the routes, so wide or tall days still fill the figure
        half = max(max(xs) - min(xs), max(ys) - min(ys)) * 0.6 + 5000
        cx, cy = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2
        ax.set_xlim(cx - half, cx + half)
        ax.set_ylim(cy - half, cy + half)
        ax.set_aspect("equal")
        ax.set_axis_off()
        ax.set_title(title, fontsize=11, fontweight="bold")
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc="upper right", fontsize=7, ncol=legend_cols, framealpha=0.85)
        self.fig.text(0.5, 0.012, footer, ha="center", va="bottom", fontsize=8, linespacing=1.5)
        self.fig.tight_layout(rect=(0, 0.06, 1, 1))     # keep the footer clear of the map
        self.fig.savefig(out_png, dpi=MAP_DPI)
        plt.close(self.fig)
        return out_png


def draw_route_map(kind, v, coords, depot_xy, outline, out_png, actual_routes=()):
    """One planned route (blue) with its comparator real route (grey, behind); see comparator_route.

    Footer: planned against the comparator's km, kg and fill.
    """
    depot, order, km = v["Depot"], v["Order"], v["Route km"]
    if not order:
        print(f"No located areas on {depot} {v['Day']} {v['Type']} vehicle {v['Vehicle']}, map skipped")
        return None
    m = MapFrame(depot, coords, depot_xy, outline)
    if actual_routes:
        m.ax.plot([], [], color=ACTUAL_COLOUR, linestyle="--", linewidth=0.9,
                  label="Actual: " + ", ".join(f"{r['Route']} ({r['Depot']} {r['Day'][:3]})"
                                               for r in actual_routes))
    for r in actual_routes:
        m.actual(r)
    m.loop(order, PLANNED_COLOUR, 1.4, 3, label="Planned", dot=18)
    title = (f"{depot} {v['Day']} {v['Type']} vehicle {v['Vehicle']} ({kind.lower()}, week {v['Week']})\n"
             f"{km:.0f} km loop, {drops(v)} drops, {v['KG']:,.0f} kg")
    return m.save(title, compare_footer([v], actual_routes), out_png)


def draw_day_map(depot, day, week, planned, actual_routes, coords, depot_xy, outline, out_png):
    """All planned routes for one depot, day and week in different colours; actual routes grey behind."""
    m = MapFrame(depot, coords, depot_xy, outline)
    if actual_routes:
        m.ax.plot([], [], color=ACTUAL_COLOUR, linestyle="--", linewidth=0.9,
                  label=f"Actual ({len(actual_routes)})")
    for r in actual_routes:
        m.actual(r, label=False)
    for i, v in enumerate(sorted(planned, key=lambda v: v["Vehicle"])):
        m.loop(v["Order"], DAY_MAP_COLOURS[i % len(DAY_MAP_COLOURS)], 1.3, 3,
               label=f"{v['Type']} {v['Vehicle']}: {v['Route km']:.0f} km, {v['KG']:,.0f} kg")
    title = f"{depot} {day}, week {week}: all planned routes"
    return m.save(title, compare_footer(planned, actual_routes), out_png,
                  legend_cols=2 if len(planned) > 8 else 1)


def safe_name(name):
    return name.replace(" ", "_").replace("/", "_")


def date_tag(week, day):
    """Sortable start for a map file name, e.g. 2026-W42_1-Mon.

    Week first, then day number, so Windows lists the maps in date order
    (all of week 42, Monday to Friday, then week 43) instead of by depot or day name.
    """
    return f"{week}_{DAYS.index(day) + 1}-{day[:3]}"


# ---- Main -----------------------------------------------------------------
def main():
    orders = load_orders(choose_orders_path())
    plan_source = choose_planning_source()
    plans = load_plan(plan_source)
    coords, depot_xy = load_grid_refs()
    print("Depot positions (km): " + ", ".join(f"{d} ({x:.1f}, {y:.1f})" for d, (x, y) in depot_xy.items()))
    missing = sorted(set(orders["area"]) - set(coords))
    if missing:
        print(f"Areas with no grid reference (never merged across, no route km): {missing}")

    picks = pick_weeks(orders)
    band_of = {wk: band for band, wk in picks}
    tables = {"Band averages": [], "Pivot": [], "Weeks": [], "Summary": [], "Daily": [],
              "Routes": [], "Moves": [], "Overflow": [], "Checks": []}
    if ROUTE_DETAIL:
        tables["Route detail"] = []
    actual = actual_route_rows(orders, coords, depot_xy)
    tables["Actual route km"] = public(actual)
    tables["Actual route summary"] = route_km_summary(actual)
    ROUTE_CAP.update(route_caps(actual))
    for row in tables["Actual route summary"]:
        row["Cap km used"] = caps_for(row["Depot"])[row["Type"]]
    cap_text = ", ".join(f"{d} HGV {c['HGV']:.1f} km" for d, c in ROUTE_CAP.items()) or "none, using HGV max_km"
    print(f"Route caps: {cap_text}; 26t uses the same truck cap; vans {VEHICLE_TYPES['Van']['max_km']:.0f} km (hard)")
    planned_vehicles, all_vehicles, all_orders = [], [], []
    for band, wk in picks:
        wo = orders[orders["wk"] == wk]
        tables["Weeks"].append({"Band": band, "Week ID": wk,
                                "KG": round(wo["kg"].sum()), "Routes": wo["route"].nunique()})
        res = simulate(wk, wk, wo, plans, coords, depot_xy)
        planned_vehicles += [v for v in res["Vehicles"] if v["Plan"] == "After" and v["Order"]]
        all_vehicles += res["Vehicles"]
        all_orders += res["Orders"]
        veh = pd.DataFrame(res["Vehicles"])
        km = {} if veh.empty else sum_route_km(veh, ["Plan"]).set_index("Plan")["Route km"].to_dict()
        tables["Weeks"][-1]["Route km before"] = round(km.get("Before", 0.0), 1)
        tables["Weeks"][-1]["Route km after"] = round(km.get("After", 0.0), 1)
        tables["Routes"] += route_rows(wk, res, actual)
        if ROUTE_DETAIL:
            tables["Route detail"] += route_detail_rows(wk, res, coords, depot_xy)
        tables["Summary"] += summary_rows(wk, res)
        tables["Daily"] += res["Daily"]
        tables["Moves"] += res["Moves"]
        tables["Overflow"] += res["Overflow"]
        tables["Checks"] += res["Checks"]

    summary = pd.DataFrame(tables["Summary"])
    summary.insert(1, "Band", summary["Week"].map(band_of))
    tables["Summary"] = summary.to_dict("records")
    averages = (summary.groupby(["Band", "Depot", "Basis"])[DAYS + ["Total"]]
                .mean().round(1).reset_index())
    tables["Band averages"] = averages.to_dict("records")
    pivot = summary.pivot_table(index=["Basis", "Depot", "Band"], values=DAYS, aggfunc="mean").round(1)
    pivot = pivot.reindex(columns=DAYS)
    tables["Pivot"] = pivot.reset_index().to_dict("records")

    run_dir = OUTPUT_DIR / f"Run_{RUN_STAMP}"
    run_dir.mkdir(parents=True, exist_ok=True)

    example_routes = pick_example_routes(planned_vehicles)
    if example_routes:
        example_dir = OUTPUT_DIR / f"Run_{RUN_STAMP}" / "Example_maps"
        example_dir.mkdir(parents=True, exist_ok=True)
        outline = load_uk_outline()
        for kind, v in example_routes:
            depot_dir = example_dir / safe_name(v["Depot"])      # one folder per depot
            depot_dir.mkdir(exist_ok=True)
            name = (f"{date_tag(v['Week'], v['Day'])}_{v['Depot']}_{v['Type']}_vehicle_{v['Vehicle']}_{kind}.png")
            comp, _ = comparator_route(v, actual)
            png = draw_route_map(kind, v, coords, depot_xy, outline, depot_dir / safe_name(name),
                                 [comp] if comp is not None else [])
        print(f"Days Maps Saved {png}")

        if DAY_MAPS:
            # every weekday, for each depot and week that has an example route
            day_dir = run_dir / "Day maps"
            day_dir.mkdir(exist_ok=True)
            for depot, week in sorted({(v["Depot"], v["Week"]) for _, v in example_routes}):
                day_dir_depot = day_dir / safe_name(depot)        # one folder per depot
                day_dir_depot.mkdir(exist_ok=True)
                for day in DAYS:
                    planned = [v for v in planned_vehicles
                               if (v["Depot"], v["Day"], v["Week"]) == (depot, day, week)]
                    if not planned:
                        continue
                    act = [r for r in actual if (r["Depot"], r["Day"], r["Week"]) == (depot, day, week)]
                    png = draw_day_map(depot, day, week, planned, act, coords, depot_xy, outline,
                                       day_dir_depot / safe_name(f"{date_tag(week, day)}_{depot}.png"))
            print(f"Day maps Saved to: {day_dir}")

    out = run_dir / f"Example_weeks_{RUN_STAMP}.xlsx"
    with pd.ExcelWriter(out) as xl:
        for name, rows in tables.items():
            pd.DataFrame(rows).to_excel(xl, sheet_name=name, index=False)
    print(f"Saved {out}")

    if SETTINGS["planning"]["customer_report"]:
        import customer_report
        rep = customer_report.build(run_dir, RUN_STAMP, band_of, all_vehicles, pd.concat(all_orders),
                                    tables["Daily"], orders, plan_source)
        print(f"Saved {rep}")


if __name__ == "__main__":
    main()