"""Quality sweep: rebalances the anneal's day plan against the real sample of orders.

Uses the same orders file as planning_estimate.py and the anneal plan folder
(A_OUTPUT_DIR). The anneal itself is not changed.

Moves: a district takes the day pattern of one of its nearest neighbours
(a boundary move). Every pattern is one the anneal already uses, so the day
rules hold. A move is kept only if the objective falls.

Objective, summed over every sample week and depot:
  - vehicle spread between the busiest and quietest day (target SPREAD_VEH)
  - vehicles above the depot's fleet cap (hard: large penalty)
  - kg spread above the anneal's 5% tolerance (soft)
  - km proxy: 2 x straight-line distance to the depot x visits per week
The search uses fast proxies (vehicles from kg and pallets; km from visits).
The final comparison re-runs the real packing and route km (planning_estimate.simulate)
for the Before and After plans, so the reported figures are not proxies.

Writes Anneal_<Depot>_Regional_Sweep_<stamp>.xlsx to S_OUTPUT_DIR, in the anneal's
sheet layout, so planning_estimate.load_plan can read it. The Map and Daily loads
sheets, and the day maps picture, are regenerated from the swept plan, using the
anneal's own map code. A "Sweep changes" sheet lists every district whose days changed.
Output goes to a Run_<stamp> folder when several depots run, as the anneal does.
"""
import os
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

import numpy as np
import pandas as pd

import planning_estimate as PE
from matrix_common import AREA, DAYS, DEPOT, EAST, NORTH, map_days
from routing import grid_distance_km, load_grid_refs
from maps_common import add_maps_sheet, load_uk_outline, plot_depot_maps

# ---- dir paths and environment variables ----------------------------------------------------------
load_dotenv()
_missing = [v for v in ("A_OUTPUT_DIR", "P_OUTPUT_DIR", "S_OUTPUT_DIR") if not os.getenv(v)]
if _missing:
    raise SystemExit(f"Missing in .env: {', '.join(_missing)}. Add them, e.g. "
                     "S_OUTPUT_DIR=C:\\path\\to\\sweep output folder")
S_OUTPUT_DIR = Path(os.getenv("S_OUTPUT_DIR"))      # sweep output folder (new variable)
RUN_STAMP = datetime.now().strftime("%Y-%m-%d_%H-%M")

# ---- Settings ---------------------------------------------------------------------------------------
# Values come from settings.toml [sweep] and [patterns]; see the notes there.
from settings import SETTINGS
SW = SETTINGS["sweep"]
SPREAD_VEH = SW["spread_veh"]
SPREAD_TOL = SW["spread_tol"]
NEIGHBOURS = SETTINGS["patterns"]["neighbours"]
MAX_PASSES = SW["max_passes"]

# Objective weights: starting points, tune on real weeks
V_SPREAD = SW["v_spread"]
V_OVER = SW["v_over"]
CAP_PEN = SW["cap_pen"]
KG_PEN = SW["kg_pen"]
KM_W = SW["km_w"]

MAKE_MAPS = SW["make_maps"]

HGV = PE.VEHICLE_TYPES["HGV"]   # 18t truck: the sweep's vehicle proxy
# The proxy treats every vehicle as an 8t truck, so the cap is HGV + 26t fleet (vans left out)
TRUCK_CAP = {d: fl.get("HGV", 0) + fl.get("26t", 0) for d, fl in PE.FLEET_CAP.items()}

def _day_map(pattern):
    """5 x 5 matrix: old day (row) -> new day (column) under map_days for this pattern."""
    m = np.zeros((5, 5))
    for o, d in enumerate(DAYS):
        m[o, DAYS.index(map_days([d], list(pattern))[d])] = 1.0
    return m


# ---- Depot data -----------------------------------------------------------------------------------
def build_depot(depot, plan_new, orders, weeks, coords, depot_pt):
    """Arrays for one depot. plan_new: area -> list of days in the anneal plan."""
    areas = sorted(plan_new)
    ai = {a: i for i, a in enumerate(areas)}
    wi = {w: i for i, w in enumerate(weeks)}
    K = np.zeros((len(weeks), len(areas), 5))           # kg by week, area, old day
    P = np.zeros((len(weeks), len(areas), 5))           # pallets by week, area, old day
    for wk, area, day, kg, pal in zip(orders["wk"], orders["area"], orders["day"],
                                       orders["kg"], orders["pal"]):
        i = ai.get(area)
        if i is None:
            continue                                     # pallet network: not part of the plan
        K[wi[wk], i, DAYS.index(day)] += kg
        P[wi[wk], i, DAYS.index(day)] += pal

    # Allowed patterns are the ones this depot's anneal plan already uses, so the sweep
    # only moves districts between patterns the anneal chose. Nothing is rejected here.
    keys = [tuple(d for d in DAYS if d in plan_new[a]) for a in areas]
    allowed = sorted(set(keys))
    idx = {p: i for i, p in enumerate(allowed)}
    pat = [idx[k] for k in keys]

    dist = np.array([grid_distance_km(coords[a], depot_pt) if a in coords else 0.0 for a in areas])
    xy = np.array([coords.get(a, (np.nan, np.nan)) for a in areas], dtype=float)
    nb = []
    for i in range(len(areas)):
        d = np.hypot(xy[:, 0] - xy[i, 0], xy[:, 1] - xy[i, 1])
        d[i] = np.inf
        d[~np.isfinite(d)] = np.inf
        nb.append([int(j) for j in np.argsort(d)[:NEIGHBOURS] if np.isfinite(d[j])])

    return {"areas": areas, "K": K, "P": P, "pat": np.array(pat), "dist": dist, "nb": nb,
            "active": K.sum(axis=(0, 2)) > 0, "kg_total": K.sum(axis=(0, 2)),
            "cap": TRUCK_CAP.get(depot, float("inf")), "weeks": len(weeks),
            "allowed": allowed,
            "DAY_MAP": np.stack([_day_map(p) for p in allowed]),
            "N_DAYS": np.array([len(p) for p in allowed])}


def evaluate(pat, D):
    """Objective for a pattern per district. Lower is better."""
    MM = D["DAY_MAP"][pat]
    kg = np.einsum("wao,aod->wd", D["K"], MM)
    pal = np.einsum("wao,aod->wd", D["P"], MM)
    veh = np.maximum(np.ceil(kg / HGV["kg"]), np.ceil(pal / HGV["pallets"]))
    spread_v = veh.max(axis=1) - veh.min(axis=1)
    over = np.maximum(0, veh - D["cap"]).sum()
    avg = kg.mean(axis=1)
    kg_spread = np.where(avg > 0, (kg.max(axis=1) - kg.min(axis=1)) / np.where(avg > 0, avg, 1), 0)
    km = D["weeks"] * (2 * D["dist"] * D["N_DAYS"][pat] * D["active"]).sum()
    return (V_SPREAD * spread_v.sum() + V_OVER * np.maximum(0, spread_v - SPREAD_VEH).sum()
            + CAP_PEN * over + KG_PEN * np.maximum(0, kg_spread - SPREAD_TOL).sum()
            + KM_W * km)


def sweep_depot(D):
    """Greedy boundary moves. Returns (new pattern indices, objective before, after)."""
    pat = D["pat"].copy()
    days_kept = D["N_DAYS"][D["pat"]]                       # each district keeps its number of days
    start = best = evaluate(pat, D)
    for _ in range(MAX_PASSES):
        improved = False
        for a in np.argsort(-D["kg_total"]):                 # largest districts first
            for n in D["nb"][a]:
                cand = pat[n]
                if cand == pat[a] or D["N_DAYS"][cand] != days_kept[a]:
                    continue
                trial = pat.copy()
                trial[a] = cand
                j = evaluate(trial, D)
                if j < best - 1e-9:
                    pat, best, improved = trial, j, True
                    break
        if not improved:
            break
    return pat, start, best


# ---- Exact comparison -----------------------------------------------------------------------------
def week_metrics(label, res, plan_name):
    daily = pd.DataFrame(res["Daily"])
    veh = pd.DataFrame(res["Vehicles"])
    rows = []
    if daily.empty:
        return rows
    for depot, g in daily.groupby("Depot"):
        vehs = g.set_index("Day")["Vehicles after"].reindex(DAYS).fillna(0).astype(int)
        kgs = g.set_index("Day")["KG after"].reindex(DAYS).fillna(0)
        cap = TRUCK_CAP.get(depot)
        km = 0.0
        if not veh.empty:
            km = veh[(veh["Depot"] == depot) & (veh["Plan"] == "After")]["Route km"].sum()
        avg = kgs.mean()
        rows.append({"Week": label, "Plan": plan_name, "Depot": depot,
                     **{f"Vehicles {d}": int(vehs[d]) for d in DAYS},
                     "Vehicle spread": int(vehs.max() - vehs.min()),
                     "Vehicles over cap": int((vehs - cap).clip(lower=0).sum()) if cap else 0,
                     "KG spread %": round(100 * (kgs.max() - kgs.min()) / avg, 1) if avg else 0.0,
                     "Route km": round(float(km), 1)})
    return rows


# ---- Output ---------------------------------------------------------------------------------------
def day_kg(orders_depot, areas, pat, D):
    """kg per area per new day over all weekday orders for the depot.

    Uses every weekday order, not just the sample weeks, so the scale matches the
    anneal's matrix. Returns an (areas, 5) array in DAYS order.
    """
    K = np.zeros((len(areas), 5))
    ai = {a: i for i, a in enumerate(areas)}
    for area, day, kg in zip(orders_depot["area"], orders_depot["day"], orders_depot["kg"]):
        i = ai.get(area)
        if i is not None:
            K[i, DAYS.index(day)] += kg
    return np.einsum("ao,aod->ad", K, D["DAY_MAP"][pat])


def write_depot(source, depot, new_before, new_after, kg_day, areas, coords, run_dir, outline):
    source = Path(source)
    files = [source] if source.is_file() else sorted(source.glob("Anneal_*_Regional_*.xlsx"))
    src = next(f for f in files
               if str(pd.read_excel(f, sheet_name="Summary").loc[0, "Depot"]).strip() == depot)
    sheets = pd.read_excel(src, sheet_name=None)
    dist = sheets["Districts"]
    changed = []
    for i, row in dist.iterrows():
        a = str(row[AREA]).strip()
        if a not in new_after:
            continue
        before_txt = ", ".join(new_before.get(a, []))
        after_txt = ", ".join(new_after[a])
        dist.at[i, "New days"] = after_txt
        if before_txt != after_txt:
            changed.append({"Postcode Area": a, "Before new days": before_txt,
                            "After new days": after_txt})
    sheets["Districts"] = dist
    sheets["Sweep changes"] = pd.DataFrame(changed)

    # Map sheet and Daily loads "After kg" follow the swept plan, as the anneal writes them
    map_rows = [{AREA: a, DEPOT: depot, "Weekday": d, "Weight": round(float(kg_day[i, DAYS.index(d)]), 1)}
                for i, a in enumerate(areas) for d in new_after[a]]
    map_df = pd.DataFrame(map_rows, columns=[AREA, DEPOT, "Weekday", "Weight"])
    map_df["_o"] = map_df["Weekday"].map(DAYS.index)
    sheets["Map"] = (map_df.sort_values([AREA, "_o"]).drop(columns="_o")
                     .rename(columns={DEPOT: "tblTeams.Best_postcode.Best Depot"}))
    if "Daily loads" in sheets:
        totals = kg_day.sum(axis=0)
        sheets["Daily loads"]["After kg"] = [round(float(totals[DAYS.index(d)]))
                                             for d in sheets["Daily loads"]["Day"]]

    run_dir.mkdir(parents=True, exist_ok=True)
    out = run_dir / f"Anneal_{depot.replace(' ', '_')}_Regional_Sweep_{RUN_STAMP}.xlsx"
    with pd.ExcelWriter(out) as xl:
        for name, df in sheets.items():
            df.to_excel(xl, sheet_name=name, index=False)

    if MAKE_MAPS:
        # coords are in km; the outline and the anneal's maps use metres (British National Grid)
        depot_df = pd.DataFrame([{AREA: a, EAST: coords[a][0] * 1000, NORTH: coords[a][1] * 1000}
                                 for a in areas if a in coords])
        png = run_dir / f"Maps_{depot.replace(' ', '_')}_Sweep_{RUN_STAMP}.png"
        plot_depot_maps(depot, map_rows, depot_df, png, outline)
        add_maps_sheet(out, png)
        print(f"Saved map {png}")
    return out, len(changed)


# ---- Main -----------------------------------------------------------------------------------------
def main():
    orders = PE.load_orders(PE.choose_orders_path())
    folder = PE.choose_plan_source()
    plans = PE.load_plan(folder)
    coords, depot_xy = load_grid_refs()

    complete = [wk for wk, g in orders.groupby("wk") if set(DAYS) <= set(g["day"])]
    weeks = sorted(complete)
    print(f"Sample weeks: {len(weeks)}")

    plans_after = {}
    after_kg = {}
    for depot, (new, pallet_areas) in plans.items():
        o = orders[(orders["depot"] == depot) & orders["wk"].isin(weeks) & orders["day"].isin(DAYS)]
        D = build_depot(depot, new, o, weeks, coords, depot_xy[depot])
        pat, start, best = sweep_depot(D)
        new_after = {a: list(D["allowed"][pat[i]]) for i, a in enumerate(D["areas"])}
        plans_after[depot] = (new_after, pallet_areas)
        o_all = orders[(orders["depot"] == depot) & orders["day"].isin(DAYS)]
        after_kg[depot] = (D["areas"], day_kg(o_all, D["areas"], pat, D))
        moved = int(np.sum(pat != D["pat"]))
        print(f"{depot}: {moved} districts moved, objective {start:,.0f} -> {best:,.0f}")

    rows = []
    for wk in weeks:
        wo = orders[orders["wk"] == wk]
        rb = PE.simulate(wk, wk, wo, plans, coords, depot_xy)
        ra = PE.simulate(wk, wk, wo, plans_after, coords, depot_xy)
        rows += week_metrics(wk, rb, "Before") + week_metrics(wk, ra, "After")
    comparison = pd.DataFrame(rows)

    # One dated folder per multi-depot run; a single depot saves to the output folder (as the anneal)
    run_dir = S_OUTPUT_DIR / f"Run_{RUN_STAMP}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving this run to: {run_dir}")
    outline = load_uk_outline() if MAKE_MAPS else None
    for depot in plans:
        areas, kg_day = after_kg[depot]
        out, n = write_depot(folder, depot, plans[depot][0], plans_after[depot][0],
                             kg_day, areas, coords, run_dir, outline)
        print(f"Saved {out} ({n} districts changed)")

    out_cmp = run_dir / f"Sweep_comparison_{RUN_STAMP}.xlsx"
    summary = (comparison.groupby(["Depot", "Plan"])
               .agg(**{"Mean vehicle spread": ("Vehicle spread", "mean"),
                       "Mean KG spread %": ("KG spread %", "mean"),
                       "Total vehicles over cap": ("Vehicles over cap", "sum"),
                       "Total route km": ("Route km", "sum")})
               .round(1).reset_index())
    with pd.ExcelWriter(out_cmp) as xl:
        summary.to_excel(xl, sheet_name="Summary", index=False)
        comparison.to_excel(xl, sheet_name="Weeks", index=False)
    print(f"Saved {out_cmp}")


if __name__ == "__main__":
    main()