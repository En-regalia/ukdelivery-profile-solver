"""Example-week simulation: actual routes vs the solver's new day plan.

For the lowest, median and peak network weeks (by kg):
- Before: actual routes (unique assignedRouteUID) per depot per day, plus a
  vehicle estimate from kg and pallets.
- After: each order moves to its area's new day (same rule as map_days).
  Outliers and external letters go to "Pallet network" with no vehicles.
- Vehicles: packed per depot per day at 8000 kg / 15 pallets.
  Stage 1 packs within each postcode area only.
  Stage 2 merges leftover vehicles with nearby areas on the same day
  (same letter first, then any area within CAP_KM).
Both plans use the same packing, so the comparison is like for like.
"""
import os
import re
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

import numpy as np
import pandas as pd

from matrix_common import AREA, DAYS, EAST, NORTH, load_matrix, map_days, select_matrix_path

# ---- dir paths and environment variables -------------------------------------------------------------
# set to your file
load_dotenv()

ANNEAL_FOLDER = Path(os.getenv("A_OUTPUT_DIR"))        # folder with Anneal_*_Regional_*.xlsx
OUTPUT_DIR = Path(os.getenv("P_OUTPUT_DIR"))
RUN_STAMP = datetime.now().strftime("%Y-%m-%d_%H-%M")

# ---- Settings -------------------------------------------------------------

VEH_KG = 8000
VEH_PALLETS = 15
CAP_KM = 120.0            # max straight-line km between areas that can share a vehicle

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

def choose_anneal_run():
    runs = [d for d in ANNEAL_FOLDER.iterdir() if d.is_dir() and d.name.startswith("Run_")]
    if any(ANNEAL_FOLDER.glob("Anneal_*_Regional_*.xlsx")):
        runs.append(ANNEAL_FOLDER)         # single-depot runs save straight into this folder
    if not runs:
        raise SystemExit(f"No anneal runs found in {ANNEAL_FOLDER}")
    runs = sorted(runs, key=lambda d: d.stat().st_mtime, reverse=True)[:5]
    print("\nMost recent anneal runs:")
    for i, d in enumerate(runs, start=1):
        saved = datetime.fromtimestamp(d.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        label = d.name if d != ANNEAL_FOLDER else "(single-depot files in this folder)"
        print(f"  {i}. {label}  (saved {saved})")
    choice = input("Enter the number to use: ").strip()
    if not choice.isdigit() or not 1 <= int(choice) <= len(runs):
        raise SystemExit("Not a valid choice")
    return runs[int(choice) - 1]

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


def load_plan(folder):
    """depot -> (area -> new days, set of pallet-network areas)."""
    plans = {}
    for f in sorted(Path(folder).glob("Anneal_*_Regional_*.xlsx")):
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


def area_coords():
    m = load_matrix(select_matrix_path())
    m = m[(m[EAST] > 0) & (m[NORTH] > 0)]
    c = m.groupby(AREA)[[EAST, NORTH]].mean() / 1000          # km
    return {a: (x, y) for a, x, y in zip(c.index, c[EAST], c[NORTH])}


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
def letters(area):
    m = re.match(r"[A-Z]+", area)
    return m.group(0) if m else area


def vehicle_distance(a, b, coords):
    best = np.inf
    for x in a["areas"]:
        for y in b["areas"]:
            if x in coords and y in coords:
                d = np.hypot(coords[x][0] - coords[y][0], coords[x][1] - coords[y][1])
                best = min(best, d)
    return best


def pack(orders, coords):
    """orders: rows with area, kg, pal for one depot and day. Returns vehicles."""
    vehicles = []

    def fits(v, kg, pal):
        return v["kg"] + kg <= VEH_KG and v["pal"] + pal <= VEH_PALLETS

    # Stage 1: pack within each postcode area only
    area_order = orders.groupby("area")["kg"].sum().sort_values(ascending=False).index
    for area in area_order:
        own = []
        items = orders[orders["area"] == area].sort_values("kg", ascending=False)
        for kg, pal in zip(items["kg"], items["pal"]):
            v = next((v for v in own if fits(v, kg, pal)), None)
            if v is None:
                v = {"kg": 0.0, "pal": 0.0, "areas": {}}
                own.append(v)
            v["kg"] += kg
            v["pal"] += pal
            v["areas"][area] = v["areas"].get(area, 0.0) + kg
        vehicles.extend(own)

    # Stage 2: merge vehicles with nearby areas on the same day, same letter first
    for same_letter in (True, False):
        while True:
            pair = None
            for a in sorted(vehicles, key=lambda v: v["kg"]):
                best = None
                for b in vehicles:
                    if b is a or not fits(b, a["kg"], a["pal"]):
                        continue
                    if same_letter and not ({letters(x) for x in a["areas"]}
                                            & {letters(y) for y in b["areas"]}):
                        continue
                    d = vehicle_distance(a, b, coords)
                    if d <= CAP_KM and (best is None or d < best[0]):
                        best = (d, b)
                if best:
                    pair = (a, best[1])
                    break
            if pair is None:
                break
            a, b = pair
            b["kg"] += a["kg"]
            b["pal"] += a["pal"]
            for k, val in a["areas"].items():
                b["areas"][k] = b["areas"].get(k, 0.0) + val
            vehicles[:] = [v for v in vehicles if v is not a]
    return vehicles


def vehicle_rows(week, depot, day, plan, vehicles):
    rows = []
    for i, v in enumerate(sorted(vehicles, key=lambda v: -v["kg"]), start=1):
        rows.append({"Week": week, "Depot": depot, "Day": day, "Plan": plan,
                     "Vehicle": i, "Areas": ", ".join(sorted(v["areas"])),
                     "KG": round(v["kg"]), "Pallets": round(v["pal"], 1),
                     "KG fill %": round(v["kg"] / VEH_KG, 2),
                     "Pallet fill %": round(v["pal"] / VEH_PALLETS, 2)})
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


def simulate(label, wk, wo, plans, coords):
    out = {"Daily": [], "Vehicles": [], "Moves": [], "Pallet_areas": [], "Checks": []}
    weekday = wo[wo["day"].isin(DAYS)]
    out["Checks"].append({"Week": label, "Week ID": wk,
                          "Raw kg": round(wo["kg"].sum()),
                          "Weekend kg (not simulated)": round(wo.loc[~wo["day"].isin(DAYS), "kg"].sum())})

    for depot in sorted(weekday["depot"].unique()):
        if depot not in plans:
            print(f"{label} {depot}: no anneal plan, skipped")
            continue
        new, pallet_areas = plans[depot]
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
            veh_b = pack(before, coords)
            veh_a = pack(after, coords)
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
            out["Vehicles"] += vehicle_rows(label, depot, day, "Before", veh_b)
            out["Vehicles"] += vehicle_rows(label, depot, day, "After", veh_a)

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
                                 "Weight": int(r["KG"]), "Pallets": r["Pallets"]})
            pn = pallets[(pallets["Depot"] == depot) & (pallets["Day"] == day)] if not pallets.empty else pallets
            rows.append({"Week": label, "Depot": depot, "Day": day, "Route": "Pallet Network",
                         "Weight": int(pn["KG"].sum()) if not pn.empty else 0,
                         "Pallets": round(pn["Pallets"].sum(), 1) if not pn.empty else 0})
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
    return rows

# ---- Main -----------------------------------------------------------------
def main():
    orders = load_orders(ask_path("Please enter path to raw csv file: "))
    plans = load_plan(choose_anneal_run())
    coords = area_coords()
    missing = sorted(set(orders["area"]) - set(coords))
    if missing:
        print(f"Areas with no grid reference (never merged across): {missing}")

    picks = pick_weeks(orders)
    band_of = {wk: band for band, wk in picks}
    tables = {"Band averages": [], "Pivot": [], "Weeks": [], "Summary": [], "Routes": [], "Moves": [], "Checks": []}
    for band, wk in picks:
        wo = orders[orders["wk"] == wk]
        tables["Weeks"].append({"Band": band, "Week ID": wk,
                                "KG": round(wo["kg"].sum()), "Routes": wo["route"].nunique()})
        res = simulate(wk, wk, wo, plans, coords)
        tables["Routes"] += route_rows(wk, res)
        tables["Summary"] += summary_rows(wk, res)
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

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / f"Example_weeks_{RUN_STAMP}.xlsx"
    with pd.ExcelWriter(out) as xl:
        for name, rows in tables.items():
            pd.DataFrame(rows).to_excel(xl, sheet_name=name, index=False)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()