"""Example-week simulation: actual routes vs the solver's new day plan.

For the lowest, median and peak network weeks (by kg):
- Before: actual routes (unique assignedRouteUID) per depot per day, plus a
  vehicle estimate from kg and pallets.
- After: each order moves to its area's new day (same rule as map_days).
  Outliers and external letters go to "Pallet network" with no vehicles.
- Vehicles: packed per depot per day at 8000 kg / 15 pallets.
  Stage 1 packs within each postcode area (first fit, largest orders first).
  Stage 2 empties the smallest vehicles by moving their orders into other
  vehicles on the same day, where the load fits, the closed route stays within
  MAX_ROUTE_KM, and the added route km is smallest.
  Each depot's vehicle count is checked against FLEET_CAP.
Both plans use the same packing, so the comparison is like for like.

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

VEH_KG = 8000
VEH_PALLETS = 15
MAX_ROUTE_KM = 400.0      # ceiling: max closed-loop km for one vehicle (depot -> stops -> depot)
ROUTE_PCT = 0.9           # per depot, the cap is this percentile of that depot's real route km (never above MAX_ROUTE_KM)
ROUTE_CAP = {}            # depot -> cap km; filled in main() from the actual routes
FLEET_CAP = {             # vehicles available per depot per day (names as in the matrix)
    "Doncaster": 6,
    "Warrington": 8,
    "Watford": 18,
    "Bristol Outbase": 3,
}                         # depots not listed here are not capped (Solihull is not given yet)
ROUTE_DETAIL = True       # writes a "Route detail" sheet: every stop, leg and cumulative km (After plan)

C_OUTPUT_PATH = (os.getenv("C_OUTPUT_PATH") or "").strip().strip('"').strip("'") or None   # raw orders CSV: default for the orders prompt
PROMPT_SECONDS = 3        # the consolidated file is used if nothing is typed within this time

# Raw file column positions (0-based): I, J, K, Y, AG, AN
DATE_COL, KG_COL, PAL_COL, TEAM_COL, ROUTE_COL, AREA_COL = 8, 9, 10, 24, 32, 39
DEPOT_CODES = {"1001601": "Watford", "1001599": "Solihull",
               "1001598": "Bristol Outbase", "1001593": "Doncaster",
               "1001597": "Warrington"}
PALLET = "Pallet network";

# ---- Choose Anneal File run --------------------------------------------------------------
def ask_path(prompt):
    # Strip the quotes Windows adds when you copy a path
    return Path(input(prompt).strip().strip('"').strip("'"))

def choose_run(folder, title):
    """Pick one of the five most recent runs in folder (Run_ subfolders or files saved directly)."""
    runs = [d for d in folder.iterdir() if d.is_dir() and d.name.startswith("Run_")]
    if any(folder.glob("Anneal_*_Regional_*.xlsx")):
        runs.append(folder)                # single-depot runs save straight into this folder
    if not runs:
        raise SystemExit(f"No {title} runs found in {folder}")
    runs = sorted(runs, key=lambda d: d.stat().st_mtime, reverse=True)[:5]
    print(f"\nMost recent {title} runs:")
    for i, d in enumerate(runs, start=1):
        saved = datetime.fromtimestamp(d.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        label = d.name if d != folder else f"(single-depot files in this folder)"
        print(f"  {i}. {label}  (saved {saved})")
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
    files = [source] if source.is_file() else sorted(source.glob("Anneal_*_Regional_*.xlsx"))
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
        print(f"Plan: {depot} ({f.name}) - {len(new)} areas with days, "
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


# ---- Packing --------------------------------------------------------------
# A vehicle is {"kg": float, "pal": float, "items": [(area, kg, pal), ...]}.
# Items are whole orders, so an order is never split between vehicles.

def areas_of(v):
    """area -> kg for one vehicle."""
    out = {}
    for area, kg, _ in v["items"]:
        out[area] = out.get(area, 0.0) + kg
    return out


def fits(v, kg, pal):
    return v["kg"] + kg <= VEH_KG and v["pal"] + pal <= VEH_PALLETS


def add_item(v, item):
    _, kg, pal = item
    v["items"].append(item)
    v["kg"] += kg
    v["pal"] += pal


def copy_vehicle(v):
    return {"kg": v["kg"], "pal": v["pal"], "items": list(v["items"])}


def route_km_fn(coords, depot_xy):
    """Returns route_km(areas): closed-loop km for a set of areas, cached."""
    cache = {}

    def route_km(areas):
        key = frozenset(areas)
        if key not in cache:
            cache[key] = vehicle_route(sorted(key), coords, depot_xy)[1] if key else 0.0
        return cache[key]

    return route_km


def pack(orders, coords, depot_xy, cap):
    """orders: rows with area, kg, pal for one depot and day. cap: max route km. Returns vehicles."""
    route_km = route_km_fn(coords, depot_xy)
    vehicles = []

    # Stage 1: first fit, largest orders first, within each postcode area.
    # An order joins a vehicle only if its route stays within cap.
    area_order = orders.groupby("area")["kg"].sum().sort_values(ascending=False).index
    for area in area_order:
        own = []
        items = orders[orders["area"] == area].sort_values("kg", ascending=False)
        for kg, pal in zip(items["kg"], items["pal"]):
            v = next((v for v in own if fits(v, kg, pal)
                      and route_km(set(areas_of(v)) | {area}) <= cap), None)
            if v is None:
                v = {"kg": 0.0, "pal": 0.0, "items": []}
                own.append(v)
            add_item(v, (area, kg, pal))
        vehicles.extend(own)

    # Stage 2: empty the smallest vehicles where every order can be placed elsewhere
    return consolidate(vehicles, coords, route_km, cap)


def consolidate(vehicles, coords, route_km, cap):
    """Remove vehicles by moving their orders into the other vehicles on the day.

    Smallest vehicles are tried first, and their orders largest first. Each order
    goes to the vehicle where the closed route adds the fewest km, provided the
    load fits and the route stays within cap. A vehicle is removed only
    if every one of its orders is placed. Repeats until no vehicle can be emptied.
    """
    changed = True
    while changed:
        changed = False
        for src in sorted(vehicles, key=lambda v: v["kg"]):
            trial = [copy_vehicle(v) for v in vehicles if v is not src]
            placed_all = True
            for item in sorted(src["items"], key=lambda it: -it[1]):
                area, kg, pal = item
                if area not in coords:         # no grid reference: distance unknown, never moved
                    placed_all = False
                    break
                best = None
                for dst in trial:
                    if not fits(dst, kg, pal):
                        continue
                    current = set(areas_of(dst))
                    new_km = route_km(current | {area})
                    if new_km > cap:
                        continue
                    add = new_km - route_km(current)
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


def vehicle_rows(week, depot, day, plan, vehicles, coords, depot_xy):
    rows = []
    for i, v in enumerate(sorted(vehicles, key=lambda v: -v["kg"]), start=1):
        areas = areas_of(v)
        order, km, stop_km, unlocated = vehicle_route(areas, coords, depot_xy)
        rows.append({"Week": week, "Depot": depot, "Day": day, "Plan": plan,
                     "Vehicle": i, "Order": order, "Areas": ", ".join(sorted(areas)),
                     "KG": round(v["kg"]), "Pallets": round(v["pal"], 1),
                     "KG fill %": round(v["kg"] / VEH_KG, 2),
                     "Pallet fill %": round(v["pal"] / VEH_PALLETS, 2),
                     "Route km": round(km, 1),
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


def simulate(label, wk, wo, plans, coords, depot_xy):
    out = {"Daily": [], "Vehicles": [], "Moves": [], "Pallet_areas": [], "Checks": []}
    weekday = wo[wo["day"].isin(DAYS)]
    out["Checks"].append({"Week": label, "Week ID": wk,
                          "Raw kg": round(wo["kg"].sum()),
                          "Weekend kg (not simulated)": round(wo.loc[~wo["day"].isin(DAYS), "kg"].sum())})

    for depot in sorted(weekday["depot"].unique()):
        if depot not in plans:
            print(f"{label} {depot}: no anneal plan, skipped")
            continue
        if depot not in depot_xy:
            raise SystemExit(f"No depot position for '{depot}'. Check the Depot names in the matrix "
                             f"match DEPOT_CODES: {sorted(depot_xy)}")
        new, pallet_areas = plans[depot]
        cap = ROUTE_CAP.get(depot, MAX_ROUTE_KM)
        o = weekday[weekday["depot"] == depot].copy()
        if o.empty:
            continue
        unmatched = set(o["area"]) - set(new) - pallet_areas
        if unmatched:
            print(f"{label} {depot}: {len(unmatched)} areas not in plan, sent to pallet network: "
                  f"{sorted(unmatched)}")
        cache = {}
        o["new_day"] = o.apply(target_day, axis=1, plan_new=new,
                               pallet_areas=pallet_areas, cache=cache)

        moved_out_total = moved_in_total = 0.0
        for day in DAYS:
            before = o[o["day"] == day]
            after = o[o["new_day"] == day]
            veh_b = pack(before, coords, depot_xy[depot], cap)
            veh_a = pack(after, coords, depot_xy[depot], cap)
            moved_out = before.loc[(before["new_day"] != day) & (before["new_day"] != PALLET), "kg"].sum()
            moved_in = after.loc[after["day"] != day, "kg"].sum()
            moved_out_total += moved_out
            moved_in_total += moved_in
            pallet = before[before["new_day"] == PALLET]
            out["Daily"].append({
                "Week": label, "Depot": depot, "Day": day,
                "Routes before": before["route"].nunique(),
                "Vehicles before": len(veh_b),
                "KG before": round(before["kg"].sum()),
                "Pallets before": round(before["pal"].sum(), 1),
                "Pallet network KG": round(pallet["kg"].sum()),
                "Moved out KG": round(moved_out),
                "Moved in KG": round(moved_in),
                "Vehicles after": len(veh_a),
                "KG after": round(after["kg"].sum()),
                "Pallets after": round(after["pal"].sum(), 1),
            })
            out["Vehicles"] += vehicle_rows(label, depot, day, "Before", veh_b, coords, depot_xy[depot])
            out["Vehicles"] += vehicle_rows(label, depot, day, "After", veh_a, coords, depot_xy[depot])

        mv = (o[(o["new_day"] != o["day"]) & (o["new_day"] != PALLET)]
              .groupby(["area", "day", "new_day"])
              .agg(kg=("kg", "sum"), orders=("kg", "size")).reset_index())
        for _, r in mv.iterrows():
            out["Moves"].append({"Week": label, "Depot": depot, "Area": r["area"],
                                 "Old day": r["day"], "New day": r["new_day"],
                                 "KG": round(r["kg"]), "Orders": int(r["orders"])})

        pa = (o[o["new_day"] == PALLET].groupby(["area", "day"])
              .agg(kg=("kg", "sum"), pal=("pal", "sum")).reset_index())
        for _, r in pa.iterrows():
            out["Pallet_areas"].append({"Week": label, "Depot": depot, "Area": r["area"],
                                        "Day": r["day"], "KG": round(r["kg"]),
                                        "Pallets": round(r["pal"], 1)})

        out["Checks"].append({"Week": label, "Week ID": wk, "Depot": depot,
                              "Simulated KG": round(o["kg"].sum()),
                              "Moved out KG": round(moved_out_total),
                              "Moved in KG": round(moved_in_total),
                              "Moved difference": round(moved_out_total - moved_in_total, 1)})
    return out


def route_detail_rows(label, res, coords, depot_xy):
    """Every point of every After route, with leg and cumulative km."""
    rows = []
    for v in res["Vehicles"]:
        if v["Plan"] != "After" or not v["Order"]:
            continue
        for leg in route_legs(v["Order"], coords, depot_xy[v["Depot"]]):
            rows.append({"Week": label, "Depot": v["Depot"], "Day": v["Day"],
                         "Vehicle": v["Vehicle"], **leg})
    return rows


def route_rows(label, res):
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
                                 "Weight": int(r["KG"]), "Pallets": r["Pallets"],
                                 "Route km": r["Route km"], "Stop km": r["Stop km"],
                                 "Areas without grid ref": int(r["Areas without grid ref"])})
            pn = pallets[(pallets["Depot"] == depot) & (pallets["Day"] == day)] if not pallets.empty else pallets
            rows.append({"Week": label, "Depot": depot, "Day": day, "Route": "Pallet Network",
                         "Weight": int(pn["KG"].sum()) if not pn.empty else 0,
                         "Pallets": round(pn["Pallets"].sum(), 1) if not pn.empty else 0,
                         "Route km": None, "Stop km": None, "Areas without grid ref": None})
    return rows


def summary_rows(label, res):
    daily = pd.DataFrame(res["Daily"])
    if daily.empty:
        return []
    vehicles = pd.DataFrame(res["Vehicles"])
    after = vehicles[vehicles["Plan"] == "After"] if not vehicles.empty else vehicles
    rows = []
    actual = daily.pivot_table(index="Depot", columns="Day", values="Routes before", aggfunc="sum")
    planned = (after.groupby(["Depot", "Day"]).size().unstack() if not after.empty
               else pd.DataFrame(index=actual.index))
    for basis, counts in [("Actual routes", actual), ("Planned vehicles", planned)]:
        counts = counts.reindex(columns=DAYS).fillna(0).astype(int)
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
        over = planned.reindex(index=actual.index, columns=DAYS).fillna(0)
        for depot, row in over.iterrows():
            cap = FLEET_CAP.get(depot)
            if cap is None:
                continue
            excess = {d: max(0, int(row[d]) - cap) for d in DAYS}
            rows.append({"Week": label, "Basis": "Vehicles over fleet cap", "Depot": depot,
                         **excess, "Total": sum(excess.values())})
    return rows

def actual_route_rows(orders, coords, depot_xy):
    """Each real route (depot, day, route ID) measured as one closed straight-line loop."""
    rows = []
    o = orders[orders["day"].isin(DAYS)]
    for (depot, day, route), g in o.groupby(["depot", "day", "route"]):
        if depot not in depot_xy:
            continue
        areas = sorted(set(g["area"]))
        _, km, _, unlocated = vehicle_route(areas, coords, depot_xy[depot])
        rows.append({"Depot": depot, "Day": day, "Route": route,
                     "Areas": ", ".join(areas), "KG": round(g["kg"].sum()),
                     "Route km": round(km, 1), "Areas without grid ref": len(unlocated)})
    return rows


def route_caps(rows):
    """Per depot: ROUTE_PCT percentile of real route km, never above MAX_ROUTE_KM."""
    df = pd.DataFrame(rows)
    caps = {}
    if df.empty:
        return caps
    for depot, g in df.groupby("Depot"):
        caps[depot] = min(round(float(g["Route km"].quantile(ROUTE_PCT)), 1), MAX_ROUTE_KM)
    return caps


def route_km_summary(rows):
    """Median, 90th percentile and longest real route per depot."""
    df = pd.DataFrame(rows)
    if df.empty:
        return []
    out = []
    for depot, g in df.groupby("Depot"):
        k = g["Route km"]
        out.append({"Depot": depot, "Routes": len(k), "Median km": round(k.median(), 1),
                    "90th percentile km": round(k.quantile(0.9), 1), "Longest km": round(k.max(), 1)})
    return out

def pick_example_routes(vehicles, n_random=2, seed=1):
    """Per depot: the longest planned route across all simulated weeks, then n_random others (seeded)."""
    picks = []
    if not vehicles:
        return picks
    rng = random.Random(seed)
    by_depot = {}
    for v in vehicles:
        by_depot.setdefault(v["Depot"], []).append(v)
    for depot in sorted(by_depot):
        recs = by_depot[depot]
        longest = max(recs, key=lambda v: v["Route km"])
        others = [v for v in recs if v is not longest]
        picks.append(("Longest", longest))
        picks += [("Random", s) for s in rng.sample(others, min(n_random, len(others)))]
    return picks


def draw_route_map(kind, v, coords, depot_xy, outline, out_png):
    """Straight-line map of one planned route: depot, stops in visiting order, back to depot."""
    depot, order, km = v["Depot"], v["Order"], v["Route km"]
    if not order:
        print(f"No located areas on {depot} {v['Day']} vehicle {v['Vehicle']}, map skipped")
        return None

    # coords are km; the outline is in metres (EPSG 27700), so convert here
    xy = [(depot_xy[depot][0] * 1000, depot_xy[depot][1] * 1000) if node == DEPOT_KEY
          else (coords[node][0] * 1000, coords[node][1] * 1000) for node in order]
    xs, ys = [p[0] for p in xy], [p[1] for p in xy]
    stops = [p for node, p in zip(order, xy) if node != DEPOT_KEY]

    fig, ax = plt.subplots(figsize=(7, 7))
    if outline is not None:
        outline.plot(ax=ax, color="#dedede", edgecolor="white", linewidth=0.3)
    ax.plot(xs, ys, color="#1f5fa8", linewidth=1.2, zorder=2)
    if stops:
        ax.scatter([p[0] for p in stops], [p[1] for p in stops], s=18,
                   color="#1f5fa8", linewidths=0, zorder=3)
    ax.scatter([xy[0][0]], [xy[0][1]], marker="*", s=220, color="#c00000", zorder=4)

    span = max(max(xs) - min(xs), max(ys) - min(ys))
    pad = max(span * 0.1, 5000)
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)
    ax.set_aspect("equal")
    ax.set_axis_off()
    n_areas = len(v["Areas"].split(", ")) if v["Areas"] else 0
    ax.set_title(f"{depot} {v['Day']} vehicle {v['Vehicle']} ({kind.lower()}, week {v['Week']})\n"
                 f"{km:.0f} km loop, {n_areas} areas, {v['KG']:,.0f} kg",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_png, dpi=MAP_DPI)
    plt.close(fig)
    return out_png


# ---- Main -----------------------------------------------------------------
def main():
    orders = load_orders(choose_orders_path())
    plans = load_plan(choose_planning_source())
    coords, depot_xy = load_grid_refs()
    print("Depot positions (km): " + ", ".join(f"{d} ({x:.1f}, {y:.1f})" for d, (x, y) in depot_xy.items()))
    missing = sorted(set(orders["area"]) - set(coords))
    if missing:
        print(f"Areas with no grid reference (never merged across, no route km): {missing}")

    picks = pick_weeks(orders)
    band_of = {wk: band for band, wk in picks}
    tables = {"Band averages": [], "Pivot": [], "Weeks": [], "Summary": [], "Daily": [],
              "Routes": [], "Moves": [], "Checks": []}
    if ROUTE_DETAIL:
        tables["Route detail"] = []
    actual = actual_route_rows(orders, coords, depot_xy)
    tables["Actual route km"] = actual
    tables["Actual route summary"] = route_km_summary(actual)
    ROUTE_CAP.update(route_caps(actual))
    for row in tables["Actual route summary"]:
        row["Cap km used"] = ROUTE_CAP.get(row["Depot"], MAX_ROUTE_KM)
    print("Route cap km per depot: " + (", ".join(f"{d} {c:.1f}" for d, c in ROUTE_CAP.items())
                                       or "none, using MAX_ROUTE_KM"))
    planned_vehicles = []
    for band, wk in picks:
        wo = orders[orders["wk"] == wk]
        tables["Weeks"].append({"Band": band, "Week ID": wk,
                                "KG": round(wo["kg"].sum()), "Routes": wo["route"].nunique()})
        res = simulate(wk, wk, wo, plans, coords, depot_xy)
        planned_vehicles += [v for v in res["Vehicles"] if v["Plan"] == "After" and v["Order"]]
        veh = pd.DataFrame(res["Vehicles"])
        km = {} if veh.empty else sum_route_km(veh, ["Plan"]).set_index("Plan")["Route km"].to_dict()
        tables["Weeks"][-1]["Route km before"] = round(km.get("Before", 0.0), 1)
        tables["Weeks"][-1]["Route km after"] = round(km.get("After", 0.0), 1)
        tables["Routes"] += route_rows(wk, res)
        if ROUTE_DETAIL:
            tables["Route detail"] += route_detail_rows(wk, res, coords, depot_xy)
        tables["Summary"] += summary_rows(wk, res)
        tables["Daily"] += res["Daily"]
        tables["Moves"] += res["Moves"]
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
        outline = load_uk_outline()
        for kind, v in example_routes:
            name = f"{v['Depot']}_{v['Day']}_week_{v['Week']}_vehicle_{v['Vehicle']}_{kind}.png"
            png = draw_route_map(kind, v, coords, depot_xy, outline,
                                 run_dir / name.replace(" ", "_").replace("/", "_"))
            if png:
                print(f"Saved {png}")

    out = run_dir / f"Example_weeks_{RUN_STAMP}.xlsx"
    with pd.ExcelWriter(out) as xl:
        for name, rows in tables.items():
            pd.DataFrame(rows).to_excel(xl, sheet_name=name, index=False)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()