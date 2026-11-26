"""Volume reduction scenarios: vehicles, route km and fleet cap when customer volume falls.

Builds on planning_estimate.py and changes nothing in it (or in matrix_common, routing or
anneal_regional). For the same example weeks as planning_estimate (pick_weeks: Bottom,
Middle and Top 10), each week's weekday orders are reduced by 5%, 10% and 20% in four ways,
and each reduced week is run through planning_estimate.simulate() with the anneal plan as
it stands. The day pattern is not re-solved and the truck route cap (ROUTE_CAP) is computed
once, from the unreduced actual routes, and held fixed for every scenario.

Reduction modes (each takes one week's weekday orders and a share p):
  Blanket            kg and pallets on every order scaled by (1 - p). Keeps the pallets-to-kg
                     ratio exactly. No seed. The baseline mode.
  Random orders      within each depot and day (the order's own depot and day), drop
                     round(p x n) orders with random.Random(seed). Removed orders take their
                     pallets with them. 20 seeds by default.
  Area, furthest     per depot, areas sorted by straight-line km from the depot, furthest
  first              first; whole areas removed until at least p of that depot's weekday kg
                     is gone. Areas with no grid reference go last. Areas are never split.
  Area, nearest      the same, nearest first. Removes the short local loops, so it saves
  first              the least route km.
  "Depot" for the area modes is the depot that serves the area after the change (the plan
  that holds it, as simulate() assigns it), with distance from that depot. An area in no
  plan (pallet network) is grouped under the order's own depot.

Checks: the 0% run of each mode must reproduce simulate() on the unreduced week exactly
(the After figures of planning_estimate for the same weeks). If it does not, the script
stops and says where.

Also written (turn off with --export-orders none / --customer-reports none):
  Reduced orders/Orders_<mode>_<pct>_<stamp>.csv   the whole orders CSV in its own format, reduced;
                                                   point C_OUTPUT_PATH at it to rerun the other steps
  Reduced orders/Removed_<mode>_<pct>_<stamp>.csv  the orders taken out
  Customer_report_<stamp>_<mode>_<pct>.xlsx        customer report at the reduced volume
  Random orders uses the first seed for these.

No prompts. Run, for example:
  python scenario_reduction.py --plan "C:/.../Anneal output/Run_2026-11-20_10-00" --matrix built

Settings: the orders CSV is C_OUTPUT_PATH from .env unless --orders is given. The workbook
goes to P_OUTPUT_DIR/Run_<stamp>/Scenario_reduction_<stamp>.xlsx (or --out instead of
P_OUTPUT_DIR). The matrix is "built" (MATRIX_OUT), "current" (MATRIX_CURRENT) or a path;
use the same one planning_estimate used, or the baseline will not reproduce.
"""
import argparse
import math
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

MODE_BLANKET = "Blanket"
MODE_RANDOM = "Random orders"
MODE_FAR = "Area, furthest first"
MODE_NEAR = "Area, nearest first"
MODES = [MODE_BLANKET, MODE_RANDOM, MODE_FAR, MODE_NEAR]
MODE_TEXT = {
    MODE_BLANKET: "kg and pallets on every weekday order scaled by (1 - p); pallets-to-kg ratio kept "
                  "exactly; deterministic, no seed",
    MODE_RANDOM: "within each depot and day (the order's own depot and day), round(p x n) orders dropped "
                 "with random.Random(seed); removed orders take their pallets; one row per seed",
    MODE_FAR: "per serving depot, whole areas removed furthest from the depot first (straight-line km) "
              "until at least p of that depot's weekday kg is gone; areas with no grid ref last; "
              "areas never split. One bound of the range.",
    MODE_NEAR: "as the furthest-first mode but nearest first: removes the short, dense local loops, so "
               "it saves the least route km. The other bound of the range.",
}


# ---- Arguments and environment (before planning_estimate is imported) ----------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--plan", required=True, help="anneal run folder (or one Anneal_*_Regional_*.xlsx)")
    p.add_argument("--orders", help="orders CSV (default: C_OUTPUT_PATH in .env)")
    p.add_argument("--matrix", default="built",
                   help='"built" (MATRIX_OUT), "current" (MATRIX_CURRENT) or a path. Default: built')
    p.add_argument("--out", help="output folder (default: P_OUTPUT_DIR in .env)")
    p.add_argument("--pcts", default="5,10,20", help="reductions in percent, comma separated")
    p.add_argument("--seeds", type=int, default=20, help="random-orders seeds (default 20)")
    p.add_argument("--timing-seeds", type=int, default=3,
                   help="seeds run first to time the run (default 3); the rest follow without asking")
    p.add_argument("--seed-start", type=int, default=1, help="first seed (default 1)")
    p.add_argument("--export-orders", choices=["all", "none"], default="all",
                   help="write the reduced order file (same columns as the orders CSV) and the removed "
                        "orders for every mode and reduction (random orders: first seed). Default: all")
    p.add_argument("--customer-reports", choices=["all", "none"], default="all",
                   help="write a customer report (customer_report.build) for every mode and reduction "
                        "(random orders: first seed). Default: all")
    return p.parse_args(argv)


def prepare_environment(args):
    """planning_estimate reads A_OUTPUT_DIR and P_OUTPUT_DIR, and matrix_common reads MATRIX_OUT and
    MATRIX_CURRENT, at import time. .env wins; anything still missing is filled from the arguments
    so the import cannot fail on a path this script does not use."""
    load_dotenv(Path(__file__).with_name(".env"))
    load_dotenv()
    plan = Path(args.plan)
    os.environ.setdefault("A_OUTPUT_DIR", str(plan.parent if plan.is_file() else plan))
    if args.out:
        os.environ["P_OUTPUT_DIR"] = args.out
    if not os.getenv("P_OUTPUT_DIR"):
        raise SystemExit("P_OUTPUT_DIR is not set in .env and --out was not given")
    if args.matrix not in ("built", "current"):
        os.environ.setdefault("MATRIX_OUT", args.matrix)
        os.environ.setdefault("MATRIX_CURRENT", args.matrix)
    for key in ("MATRIX_OUT", "MATRIX_CURRENT"):
        if not os.getenv(key):
            raise SystemExit(f"{key} is not set in .env (needed by matrix_common at import)")
    os.environ.setdefault("MAP_IMAGE_PATH", "")   # maps_common reads it at import; no maps are drawn here


def import_sources():
    """Import the checked scripts. Stop and report on failure; never patch them."""
    try:
        import matrix_common
        import planning_estimate as pe
        import routing
    except SystemExit as e:
        raise SystemExit(f"Import of the source scripts stopped: {e}") from None
    except Exception as e:  # noqa: BLE001 - report any import failure plainly
        raise SystemExit(f"Import of the source scripts failed: {type(e).__name__}: {e}") from None
    return matrix_common, pe, routing


# ---- Route cache ------------------------------------------------------------------------------
class RouteCache:
    """Memo for routing.vehicle_route, keyed by (depot position, frozenset of areas).

    vehicle_route sorts and de-duplicates its areas before routing, so the result depends only on
    the set of areas and the depot: the cache returns exactly what a fresh call would. The day is
    not in the key because it does not change the km; keying by (depot, day, areas) would give the
    same values with fewer hits. planning_estimate builds a new cache for every pack() call, so
    without this the same loops are re-routed in every scenario.
    """

    def __init__(self, fn):
        self.fn, self.store, self.hits, self.misses = fn, {}, 0, 0

    def __call__(self, areas, coords, depot_xy):
        key = (tuple(depot_xy), frozenset(areas))
        if key in self.store:
            self.hits += 1
        else:
            self.misses += 1
            self.store[key] = self.fn(areas, coords, depot_xy)
        order, km, stop_km, unlocated = self.store[key]
        return list(order), km, stop_km, list(unlocated)


# ---- Reductions -------------------------------------------------------------------------------
def reduce_blanket(week, p):
    out = week.copy()
    out["kg"] = out["kg"] * (1 - p)
    out["pal"] = out["pal"] * (1 - p)
    return out


def reduce_random(week, p, seed):
    """Drop round(p x n) orders within each (own depot, day), seeded. Python's round() is used as
    specified: it rounds halves to even (round(0.5) = 0, round(1.5) = 2)."""
    rng = random.Random(seed)
    drop = []
    for _, g in week.groupby(["depot", "day"], sort=True):
        k = round(p * len(g))
        if k:
            drop += rng.sample(sorted(g.index), k)
    return week.drop(index=drop)


def area_groups(week, area_dep):
    """Serving depot per order: the plan that holds its area, else the order's own depot."""
    return week["area"].map(area_dep).fillna(week["depot"])


def reduce_area(week, p, area_dep, coords, depot_xy, furthest):
    """Remove whole areas per serving depot, by distance from that depot, until >= p of its kg.

    Returns (reduced week, list of per-depot dicts with the areas removed and the share achieved).
    """
    grp = area_groups(week, area_dep)
    removed_idx, info = [], []
    for depot in sorted(grp.unique()):
        sub = week[grp == depot]
        total = sub["kg"].sum()
        by_area = sub.groupby("area")["kg"].sum()
        xy = depot_xy.get(depot)
        located, unlocated = [], []
        for area in sorted(by_area.index):
            if area in coords and xy is not None:
                located.append((math.hypot(coords[area][0] - xy[0], coords[area][1] - xy[1]), area))
            else:
                unlocated.append(area)
        located.sort(key=lambda t: (-t[0], t[1]) if furthest else (t[0], t[1]))
        order = [a for _, a in located] + unlocated      # no grid ref: last in both directions
        gone, areas = 0.0, []
        if p > 0 and total > 0:
            for area in order:
                if gone >= p * total - 1e-9:
                    break
                areas.append(area)
                gone += by_area[area]
        removed_idx += list(sub.index[sub["area"].isin(areas)])
        info.append({"Depot": depot, "Depot kg": total, "Removed kg": gone,
                     "Areas removed": areas, "Areas without grid ref": unlocated})
    return week.drop(index=removed_idx), info


# ---- Metrics ------------------------------------------------------------------------------------
AFTER_DAILY = ["Vehicles after", "Vans after", "HGVs after", "26t after", "Vans over 500 km after",
               "Over route cap after orders", "Over route cap after KG", "Over fleet after orders",
               "Over fleet after KG", "KG after", "Pallets after", "Pallet network KG",
               "Added km limit after", "Fleet unresolved after"]
VEH_KEYS = ["Week", "Depot", "Day", "Vehicle", "Type", "Areas", "KG", "Pallets", "Route km", "Stop km"]


def metric_rows(res, pe, week_orders):
    """One row per depot and day of the After plan."""
    after = [v for v in res["Vehicles"] if v["Plan"] == "After"]
    km, n_type = {}, {}
    for v in after:
        key = (v["Depot"], v["Day"])
        km[key] = km.get(key, 0.0) + v["Route km"]
        n_type[key + (v["Type"],)] = n_type.get(key + (v["Type"],), 0) + 1
    orders_after = res["Orders"][0]
    n_orders = orders_after.groupby(["new_depot", "new_day"]).size().to_dict()
    rows = []
    for d in res["Daily"]:
        key = (d["Depot"], d["Day"])
        over = 0
        for vtype in ("Van", "HGV", "26t"):
            cap = pe.FLEET_CAP.get(d["Depot"], {}).get(vtype)
            if cap is not None:
                over += max(0, n_type.get(key + (vtype,), 0) - cap)
        rows.append({
            "Week": d["Week"], "Depot": d["Depot"], "Day": d["Day"],
            "Planned vehicles": d["Vehicles after"], "Vans": d["Vans after"],
            "HGVs": d["HGVs after"], "26t": d["26t after"],
            "Planned route km": round(km.get(key, 0.0), 1),
            "Vehicles over fleet cap": over,
            "Fleet unresolved": d["Fleet unresolved after"],
            "Added km limit": d["Added km limit after"],
            "Pallet network KG (plan areas, by own depot)": d["Pallet network KG"],
            "Overflow KG (route cap + fleet)": d["Over route cap after KG"] + d["Over fleet after KG"],
            "Overflow orders": d["Over route cap after orders"] + d["Over fleet after orders"],
            "KG after": d["KG after"], "Pallets after": d["Pallets after"],
            "Orders after": int(n_orders.get(key, 0)),
        })
    return rows


def compare_baseline(ref, base):
    """Rows comparing the unreduced simulate() (planning_estimate) with the 0% scenario run."""
    rows, ok = [], True
    rd = {(d["Depot"], d["Day"]): d for d in ref["Daily"]}
    bd = {(d["Depot"], d["Day"]): d for d in base["Daily"]}
    for key in sorted(set(rd) | set(bd)):
        for col in AFTER_DAILY:
            a, b = rd.get(key, {}).get(col), bd.get(key, {}).get(col)
            same = a == b
            ok &= same
            if not same or col in ("Vehicles after", "KG after", "Pallet network KG"):
                rows.append({"Week": ref["Daily"][0]["Week"] if ref["Daily"] else "", "Depot": key[0],
                             "Day": key[1], "Check": col, "planning_estimate": a, "Scenario 0%": b,
                             "Match": same})
    rv = sorted(tuple(str(v[k]) for k in VEH_KEYS) for v in ref["Vehicles"] if v["Plan"] == "After")
    bv = sorted(tuple(str(v[k]) for k in VEH_KEYS) for v in base["Vehicles"] if v["Plan"] == "After")
    same = rv == bv
    ok &= same
    rows.append({"Week": ref["Daily"][0]["Week"] if ref["Daily"] else "", "Depot": "All", "Day": "All",
                 "Check": "After vehicles: type, areas, kg, pallets, route km (every vehicle)",
                 "planning_estimate": len(rv), "Scenario 0%": len(bv), "Match": same})
    return rows, ok


# ---- Run ----------------------------------------------------------------------------------------
def main(argv=None):
    import pandas as pd

    args = parse_args(argv)
    prepare_environment(args)
    matrix_common, pe, routing = import_sources()
    DAYS = matrix_common.DAYS

    matrix_common.MATRIX_PATH = Path({"built": os.environ["MATRIX_OUT"],
                                      "current": os.environ["MATRIX_CURRENT"]}.get(args.matrix, args.matrix))
    if not matrix_common.MATRIX_PATH.exists():
        raise SystemExit(f"Matrix not found: {matrix_common.MATRIX_PATH}")
    orders_path = pe.check_csv(args.orders or pe.C_OUTPUT_PATH or "")
    pcts = [float(x) / 100 for x in args.pcts.split(",") if x.strip()]
    seeds = list(range(args.seed_start, args.seed_start + args.seeds))
    started = time.time()

    orders = pe.load_orders(orders_path)
    plans = pe.load_plan(args.plan)
    coords, depot_xy = pe.load_grid_refs()
    picks = pe.pick_weeks(orders)
    band_of = {wk: band for band, wk in picks}

    # ROUTE_CAP once, from the unreduced actual routes, as planning_estimate.main does. Held fixed.
    actual = pe.actual_route_rows(orders, coords, depot_xy)
    pe.ROUTE_CAP.update(pe.route_caps(actual))
    route_cap_fixed = {d: dict(c) for d, c in pe.ROUTE_CAP.items()}
    print("Route caps (fixed for every scenario): "
          + (", ".join(f"{d} {c['HGV']:.1f} km" for d, c in route_cap_fixed.items()) or "none"))

    area_dep = pe.area_depots(plans)
    weeks = {wk: orders[orders["wk"] == wk] for _, wk in picks}

    # ---- 1. Baseline: planning_estimate's simulate on the unreduced week (original routing),
    #         against the 0% run of every mode (cached routing). Must match exactly.
    print("\nBaseline check: 0% of every mode against planning_estimate on the same weeks")
    ref = {wk: pe.simulate(wk, wk, wo, plans, coords, depot_xy) for wk, wo in weeks.items()}
    cache = RouteCache(routing.vehicle_route)
    pe.vehicle_route = cache          # runtime memo only: planning_estimate.py on disk is unchanged
    check_rows, all_ok, base_res = [], True, {}
    for wk, wo in weeks.items():
        weekday = wo[wo["day"].isin(DAYS)].copy()
        zero = {MODE_BLANKET: reduce_blanket(weekday, 0.0),
                MODE_RANDOM: reduce_random(weekday, 0.0, seeds[0]),
                MODE_FAR: reduce_area(weekday, 0.0, area_dep, coords, depot_xy, True)[0],
                MODE_NEAR: reduce_area(weekday, 0.0, area_dep, coords, depot_xy, False)[0]}
        for mode, red in zero.items():
            if not red.equals(weekday):
                raise SystemExit(f"STOP: {mode} at 0% changed the orders for week {wk}")
        base_res[wk] = pe.simulate(wk, wk, zero[MODE_BLANKET], plans, coords, depot_xy)
        rows, ok = compare_baseline(ref[wk], base_res[wk])
        for r in rows:
            r["Band"] = band_of[wk]
        check_rows += rows
        all_ok &= ok
    if pe.ROUTE_CAP != route_cap_fixed:
        raise SystemExit("STOP: ROUTE_CAP changed during the run")
    if not all_ok:
        bad = [r for r in check_rows if not r["Match"]]
        for r in bad[:20]:
            print(f"  MISMATCH {r['Week']} {r['Depot']} {r['Day']} {r['Check']}: "
                  f"{r['planning_estimate']} vs {r['Scenario 0%']}")
        raise SystemExit(f"STOP: the 0% baseline does not reproduce planning_estimate "
                         f"({len(bad)} mismatches). No scenarios run.")
    print(f"Baseline reproduces planning_estimate exactly for {len(weeks)} weeks")

    baseline_rows = []
    base_index = {}
    for wk in weeks:
        for r in metric_rows(base_res[wk], pe, None):
            r["Band"] = band_of[wk]
            baseline_rows.append(r)
            base_index[(wk, r["Depot"], r["Day"])] = r
    base_orders = {wk: base_res[wk]["Orders"][0] for wk in weeks}

    # ---- 2. Scenarios
    scen_rows, removed_rows, dropped_rows = [], [], []
    report_runs = {}      # (mode, p) -> simulate() results over all weeks, for the customer reports

    def run_one(wk, weekday, mode, p, seed):
        if mode == MODE_BLANKET:
            red, info = reduce_blanket(weekday, p), None
        elif mode == MODE_RANDOM:
            red, info = reduce_random(weekday, p, seed), None
        else:
            red, info = reduce_area(weekday, p, area_dep, coords, depot_xy, mode == MODE_FAR)
        res = pe.simulate(wk, wk, red, plans, coords, depot_xy)
        if mode != MODE_RANDOM or seed == seeds[0]:
            kept = report_runs.setdefault((mode, p), {"Vehicles": [], "Orders": [], "Daily": []})
            for k in kept:
                kept[k] += res[k]
        so = res["Orders"][0]
        bo = base_orders[wk]
        for r in metric_rows(res, pe, None):
            b = base_index[(wk, r["Depot"], r["Day"])]
            kg_b = b["KG after"]
            scen_rows.append({
                "Band": band_of[wk], "Week": wk, "Depot": r["Depot"], "Day": r["Day"], "Mode": mode,
                "Reduction %": round(p * 100), "Seed": seed if mode == MODE_RANDOM else "",
                **{k: r[k] for k in ("Planned vehicles", "Vans", "HGVs", "26t")},
                "Vehicles saved": b["Planned vehicles"] - r["Planned vehicles"],
                "Planned route km": r["Planned route km"],
                "Route km change": round(r["Planned route km"] - b["Planned route km"], 1),
                "Vehicles over fleet cap": r["Vehicles over fleet cap"],
                "Fleet unresolved": r["Fleet unresolved"],
                "Added km limit": r["Added km limit"],
                "Pallet network KG (plan areas, by own depot)": r["Pallet network KG (plan areas, by own depot)"],
                "Overflow KG (route cap + fleet)": r["Overflow KG (route cap + fleet)"],
                "KG after": r["KG after"], "Baseline KG after": kg_b,
                "Actual KG removed %": round(100 * (1 - r["KG after"] / kg_b), 2) if kg_b else None,
                "Orders after": r["Orders after"], "Baseline orders after": b["Orders after"],
            })
        # week-level checks by the depot the reduction is defined on
        group = ["depot"] if mode != MODE_FAR and mode != MODE_NEAR else None
        if group:
            kb = bo.groupby("depot")["kg"].sum()
            ks = so.groupby("depot")["kg"].sum().reindex(kb.index).fillna(0.0)
            for depot in kb.index:
                removed_rows.append({"Band": band_of[wk], "Week": wk, "Mode": mode,
                                     "Reduction %": round(p * 100),
                                     "Seed": seed if mode == MODE_RANDOM else "",
                                     "Depot (own)": depot, "Target %": round(p * 100, 2),
                                     "Actual KG removed %": round(100 * (1 - ks[depot] / kb[depot]), 2)
                                     if kb[depot] else None,
                                     "Areas removed": "", "Pallet network KG removed": None})
        else:
            pallet_areas = set(bo.loc[bo["new_day"] == pe.PALLET, "area"])
            for i in info:
                pn = bo[(area_groups(bo, area_dep) == i["Depot"]) & bo["area"].isin(i["Areas removed"])
                        & bo["area"].isin(pallet_areas) & (bo["new_day"] == pe.PALLET)]["kg"].sum()
                removed_rows.append({"Band": band_of[wk], "Week": wk, "Mode": mode,
                                     "Reduction %": round(p * 100), "Seed": "",
                                     "Depot (own)": i["Depot"] + " (serving)", "Target %": round(p * 100, 2),
                                     "Actual KG removed %": round(100 * i["Removed kg"] / i["Depot kg"], 2)
                                     if i["Depot kg"] else None,
                                     "Areas removed": ", ".join(i["Areas removed"]),
                                     "Pallet network KG removed": round(pn)})
        n_b = bo.groupby("depot").size()
        n_s = so.groupby("depot").size().reindex(n_b.index).fillna(0).astype(int)
        for depot in n_b.index:
            dropped_rows.append({"Band": band_of[wk], "Week": wk, "Mode": mode, "Reduction %": round(p * 100),
                                 "Seed": seed if mode == MODE_RANDOM else "", "Depot (own)": depot,
                                 "Orders before": int(n_b[depot]), "Orders after": int(n_s[depot]),
                                 "Orders dropped": int(n_b[depot] - n_s[depot])})

    def run_phase(seed_list, with_deterministic):
        for wk, wo in weeks.items():
            weekday = wo[wo["day"].isin(DAYS)].copy()
            for p in pcts:
                if with_deterministic:
                    run_one(wk, weekday, MODE_BLANKET, p, None)
                    run_one(wk, weekday, MODE_FAR, p, None)
                    run_one(wk, weekday, MODE_NEAR, p, None)
                for s in seed_list:
                    run_one(wk, weekday, MODE_RANDOM, p, s)

    n_time = min(args.timing_seeds, len(seeds))
    t0 = time.time()
    run_phase(seeds[:n_time], True)
    t_first = time.time() - t0
    sims_first = len(weeks) * len(pcts) * (3 + n_time)
    per_sim = t_first / max(sims_first, 1)
    rest = len(seeds) - n_time
    est = per_sim * len(weeks) * len(pcts) * rest
    print(f"\nTiming: {sims_first} simulations ({n_time} seeds + 3 fixed modes) in {t_first:.0f} s "
          f"({per_sim:.2f} s each). Remaining {rest} seeds: about {est / 60:.1f} min "
          f"(cached routes make later runs faster). Route cache: {cache.hits:,} hits, {cache.misses:,} routed.")
    t1 = time.time()
    if rest:
        run_phase(seeds[n_time:], False)
    t_rest = time.time() - t1
    print(f"Remaining seeds done in {t_rest:.0f} s")

    if pe.ROUTE_CAP != route_cap_fixed:
        raise SystemExit("STOP: ROUTE_CAP changed during the run")

    # ---- 3. Summary: per seed, weekly totals averaged over weeks; then mean/min/max over seeds
    scen = pd.DataFrame(scen_rows)
    base = pd.DataFrame(baseline_rows)
    removed = pd.DataFrame(removed_rows)
    summary = summarise(pd, scen, base, band_of)

    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    run_dir = Path(os.environ["P_OUTPUT_DIR"]) / f"Run_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    export_rows = []
    if args.export_orders == "all":
        export_rows = export_orders(pd, pe, DAYS, orders_path, orders, pcts, seeds[0], area_dep, coords,
                                    depot_xy, weeks, scen, run_dir, stamp)
    report_rows = []
    if args.customer_reports == "all":
        report_rows = customer_reports(pd, pe, report_runs, removed, band_of, orders, args.plan, run_dir,
                                       stamp, seeds[0])
    out = run_dir / f"Scenario_reduction_{stamp}.xlsx"

    checks = pd.concat([
        pd.DataFrame(check_rows).assign(Section="Baseline reproduction"),
        removed.assign(Section="KG removed against target"),
        pd.DataFrame(dropped_rows).assign(Section="Orders dropped"),
        pd.DataFrame(export_rows).assign(Section="Order file export"),
    ], ignore_index=True)
    checks.insert(0, "Section", checks.pop("Section"))

    notes = notes_rows(args, orders_path, matrix_common.MATRIX_PATH, pcts, seeds, n_time, route_cap_fixed,
                       pe, picks, t_first, t_rest, cache, time.time() - started, depot_xy)
    base_cols = ["Band"] + [c for c in base.columns if c != "Band"]
    with pd.ExcelWriter(out) as xl:
        base[base_cols].to_excel(xl, sheet_name="Baseline", index=False)
        scen.to_excel(xl, sheet_name="Scenarios", index=False)
        summary.to_excel(xl, sheet_name="Summary", index=False)
        checks.to_excel(xl, sheet_name="Checks", index=False)
        pd.DataFrame(notes, columns=["Item", "Value"]).to_excel(xl, sheet_name="Notes", index=False)
        if report_rows:
            pd.DataFrame(report_rows).to_excel(xl, sheet_name="Reports", index=False)
    print(f"Saved {out}")
    return out


def summarise(pd, scen, base, band_of):
    """Mean, min and max across seeds, per band (and all weeks), mode, depot and reduction.

    Each seed's figure is the weekly total (sum of the five days) averaged over the weeks in the
    band, so it reads as "per week". Deterministic modes have one value (min = mean = max).
    All depots is the sum over depots before averaging.
    """
    keys = ["Mode", "Reduction %", "Seed", "Week", "Band", "Depot"]
    vals = {"Planned vehicles": "sum", "Vehicles saved": "sum", "Planned route km": "sum",
            "Route km change": "sum", "Vehicles over fleet cap": "sum", "Fleet unresolved": "sum",
            "Pallet network KG (plan areas, by own depot)": "sum", "Overflow KG (route cap + fleet)": "sum",
            "KG after": "sum", "Baseline KG after": "sum"}
    s = scen.copy()
    s["Seed"] = s["Seed"].astype(str)
    wk = s.groupby(keys, as_index=False).agg(vals)
    allw = wk.groupby(["Mode", "Reduction %", "Seed", "Week", "Band"], as_index=False).agg(vals)
    allw["Depot"] = "All depots"
    wk = pd.concat([wk, allw], ignore_index=True)
    wk_all = wk.assign(Band="All weeks")
    wk = pd.concat([wk, wk_all], ignore_index=True)
    per_seed = wk.groupby(["Band", "Mode", "Reduction %", "Depot", "Seed"], as_index=False)[list(vals)].mean()
    per_seed["Route km change %"] = 100 * per_seed["Route km change"] / (
        per_seed["Planned route km"] - per_seed["Route km change"]).where(lambda x: x != 0)
    per_seed["Actual KG removed %"] = 100 * (1 - per_seed["KG after"] / per_seed["Baseline KG after"]
                                             .where(lambda x: x != 0))

    # baseline per week, for reference columns
    b = base.copy()
    bw = b.groupby(["Week", "Band", "Depot"], as_index=False)[["Planned vehicles", "Planned route km",
                                                                "Vehicles over fleet cap"]].sum()
    ball = bw.groupby(["Week", "Band"], as_index=False)[["Planned vehicles", "Planned route km",
                                                          "Vehicles over fleet cap"]].sum()
    ball["Depot"] = "All depots"
    bw = pd.concat([bw, ball], ignore_index=True)
    bw = pd.concat([bw, bw.assign(Band="All weeks")], ignore_index=True)
    bmean = bw.groupby(["Band", "Depot"], as_index=False).mean(numeric_only=True).rename(columns={
        "Planned vehicles": "Baseline vehicles per week", "Planned route km": "Baseline route km per week",
        "Vehicles over fleet cap": "Baseline over fleet cap per week"})

    metrics = ["Planned vehicles", "Vehicles saved", "Planned route km", "Route km change",
               "Route km change %", "Vehicles over fleet cap", "Pallet network KG (plan areas, by own depot)",
               "Overflow KG (route cap + fleet)", "Actual KG removed %"]
    g = per_seed.groupby(["Band", "Mode", "Reduction %", "Depot"])
    out = g.size().rename("Seeds").reset_index()
    for m in metrics:
        agg = g[m].agg(["mean", "min", "max"]).reset_index(drop=True)
        out[f"{m} mean"] = agg["mean"].round(2).values
        out[f"{m} min"] = agg["min"].round(2).values
        out[f"{m} max"] = agg["max"].round(2).values
    out = out.merge(bmean, on=["Band", "Depot"], how="left")
    band_order = {"All weeks": 0, "Bottom 10": 1, "Middle 10": 2, "Top 10": 3}
    mode_order = {m: i for i, m in enumerate(MODES)}
    out["_b"] = out["Band"].map(band_order)
    out["_m"] = out["Mode"].map(mode_order)
    out["_d"] = (out["Depot"] != "All depots").astype(int)
    out = out.sort_values(["_b", "Reduction %", "_d", "Depot", "_m"]).drop(columns=["_b", "_m", "_d"])
    lead = ["Band", "Reduction %", "Depot", "Mode", "Seeds", "Baseline vehicles per week",
            "Baseline route km per week", "Baseline over fleet cap per week"]
    return out[lead + [c for c in out.columns if c not in lead]]


def DEPOT_GRID_KM():
    import routing
    return routing.DEPOT_GRID_KM


def routing_version():
    """Last git commit that touched routing.py, plus whether the working copy differs from it."""
    import subprocess
    here = Path(__file__).parent
    try:
        log = subprocess.run(["git", "log", "-1", "--format=%h %ad %s", "--date=short", "--", "routing.py"],
                             cwd=here, capture_output=True, text=True, timeout=10).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "routing.py"],
                               cwd=here, capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown (git not available)"
    if not log:
        return "unknown (not a git checkout)"
    return log + (" (with local changes)" if dirty else "")


def notes_rows(args, orders_path, matrix_path, pcts, seeds, n_time, route_cap, pe, picks,
               t_first, t_rest, cache, total_s, depot_xy):
    rows = [("Run", datetime.now().strftime("%Y-%m-%d %H:%M")),
            ("Orders CSV", str(orders_path)),
            ("Plan (anneal run)", str(Path(args.plan))),
            ("Matrix", str(matrix_path)),
            ("Reductions %", ", ".join(f"{p * 100:g}" for p in pcts)),
            ("Random-orders seeds", f"{seeds[0]} to {seeds[-1]} ({len(seeds)} seeds); random.Random(seed), "
                                    f"one generator per week, depot-day groups in sorted order"),
            ("Rounding", "round(p x n) is Python's round(): halves go to the even number"),
            ("Weeks", f"{len(picks)} weeks from planning_estimate.pick_weeks (Bottom, Middle, Top 10), not re-picked"),
            ("Day pattern", "anneal plan used as it stands; not re-solved for any scenario"),
            ("ROUTE_CAP", "computed once from the unreduced actual routes, fixed for every scenario"),
            ]
    rows += [(f"ROUTE_CAP {d}", f"HGV and 26t {c['HGV']:.1f} km") for d, c in sorted(route_cap.items())]
    rows += [("Routing version", routing_version()),
             ("Depot positions", "routing.DEPOT_GRID_KM where set, else the kg-weighted centre of the depot's "
                                 "areas in the matrix (depot info file for a depot with no areas). Set in "
                                 "DEPOT_GRID_KM: " + (", ".join(sorted(DEPOT_GRID_KM())) or "none"))]
    rows += [(f"Depot position {d}", f"{x:.2f}, {y:.2f} km") for d, (x, y) in sorted(depot_xy.items())]
    rows += [(f"FLEET_CAP {d}", ", ".join(f"{t} {n}" for t, n in caps.items()))
             for d, caps in sorted(pe.FLEET_CAP.items())]
    rows += [(f"Vehicle {t}", f"{s['kg']:g} kg, {s['pallets']:g} pallets, {s['max_km']:g} km")
             for t, s in pe.VEHICLE_TYPES.items()]
    rows += [("Added km limit", f"{pe.MAX_ADDED_KM:g} km, +{pe.ADDED_KM_STEP:g} up to {pe.ADDED_KM_CEILING:g} km"),
             ("Route percentile", f"{pe.ROUTE_PCT:g}")]
    rows += [(f"Mode: {m}", t) for m, t in MODE_TEXT.items()]
    rows += [("Range to quote", "Area, furthest first and Area, nearest first are reported as the two bounds. "
                                "Check the Summary: Blanket removes no stops at all, so it can save less "
                                "than either area mode; if Blanket or Random orders fall outside the two "
                                "bounds, quote the wider range"),
             ("Area modes overshoot", "whole areas are removed until at least p is gone, so the actual share "
                                      "removed is above the target; compare modes on Actual KG removed %"),
             ("Pallet network", "handled by planning_estimate.simulate: plan outliers and areas in no plan "
                                "go to the pallet network with no vehicles; orders over the route cap or fleet "
                                "are the overflow"),
             ("Vehicles saved", "baseline (0%) planned vehicles minus scenario planned vehicles, per depot-day; "
                                "Summary sums the five days, so it is vehicle-days per week"),
             ("Vehicles over fleet cap", "per depot-day, planned vehicles above FLEET_CAP, summed over types"),
             ("Route km", "straight-line closed loops (routing.py), as in planning_estimate"),
             ("Route cache", f"vehicle_route memo by depot position and set of areas: {cache.misses:,} routed, "
                             f"{cache.hits:,} reused. Same values as a fresh call (baseline check uses the "
                             f"uncached routing for planning_estimate's side)"),
             ("Reduced order files", "Orders_<mode>_<pct>.csv: the whole orders CSV, same columns and header, "
                                     "with the reduction applied week by week to weekday orders by the same "
                                     "rule as the scenarios (random orders: first seed), so the example weeks "
                                     "match the Scenarios sheet exactly. Weekend rows, and rows planning_estimate "
                                     "drops (unreadable date, unknown team code), are kept unchanged. Blanket "
                                     "rewrites the kg and pallet columns as plain numbers. Area modes pick the "
                                     "areas per week, so a borderline area can be removed in one week and kept "
                                     "in another. Removed_<mode>_<pct>.csv: the rows taken out (Blanket: each "
                                     "order's removed share). Point C_OUTPUT_PATH at an Orders file to run "
                                     "the other steps on the reduced volume"),
             ("Customer reports", "Customer_report_<stamp>_<mode>_<pct>.xlsx: customer_report.build on the "
                                  "scenario's simulate() results, so before (current depots and days) and after "
                                  "(new plan) are both at the reduced volume. Usual days per area come from the "
                                  "unreduced order file. The Reports sheet lists them"),
             ("Timing", f"first {n_time} seeds + fixed modes {t_first:.0f} s; remaining seeds {t_rest:.0f} s; "
                        f"whole run {total_s:.0f} s")]
    return rows



# ---- Reduced order files and customer reports ------------------------------------------------
def slug(mode, p):
    return {MODE_BLANKET: "blanket", MODE_RANDOM: "random", MODE_FAR: "area_furthest",
            MODE_NEAR: "area_nearest"}[mode] + f"_{round(p * 100)}pct"


def number_text(values):
    """Plain number text for the kg and pallet columns (no thousands separator, up to 3 decimals)."""
    return [f"{v:.3f}".rstrip("0").rstrip(".") for v in values]


def export_orders(pd, pe, DAYS, orders_path, orders, pcts, seed, area_dep, coords, depot_xy, weeks, scen,
                  run_dir, stamp):
    """Write Orders_<mode>_<pct>.csv and Removed_<mode>_<pct>.csv in the raw orders format.

    Every week in the file is reduced (weekday rows), with the same functions as the scenarios.
    Rows of the raw file are matched to load_orders rows by position: load_orders keeps the raw row
    order and only drops rows with an unreadable date or an unknown team code.
    """
    raw = pd.read_csv(orders_path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    date = pd.to_datetime(raw.iloc[:, pe.DATE_COL].str.strip(), format="%Y%m%d", errors="coerce")
    keep = date.notna() & raw.iloc[:, pe.TEAM_COL].str.strip().map(pe.DEPOT_CODES).notna()
    raw_idx = raw.index[keep.to_numpy()]
    if len(raw_idx) != len(orders) or not (
            raw.iloc[raw_idx, pe.AREA_COL].str.strip().to_numpy() == orders["area"].to_numpy()).all():
        raise SystemExit("STOP: could not match the raw orders file to load_orders rows; no order files written")
    kg_name, pal_name = raw.columns[pe.KG_COL], raw.columns[pe.PAL_COL]
    weekday_all = orders[orders["day"].isin(DAYS)]
    out_dir = run_dir / "Reduced orders"
    out_dir.mkdir(exist_ok=True)
    rows = []
    for mode in MODES:
        for p in pcts:
            parts = []
            for wk, wo in weekday_all.groupby("wk", sort=True):
                w = wo.copy()
                if mode == MODE_BLANKET:
                    parts.append(reduce_blanket(w, p))
                elif mode == MODE_RANDOM:
                    parts.append(reduce_random(w, p, seed))
                else:
                    parts.append(reduce_area(w, p, area_dep, coords, depot_xy, mode == MODE_FAR)[0])
            reduced = pd.concat(parts)
            gone = weekday_all.index.difference(reduced.index)
            kept_raw = raw.drop(index=raw_idx[gone])
            if mode == MODE_BLANKET:
                scaled = raw_idx[weekday_all.index]
                removed_raw = raw.loc[scaled].copy()
                kept_raw.loc[scaled, kg_name] = number_text(weekday_all["kg"] * (1 - p))
                kept_raw.loc[scaled, pal_name] = number_text(weekday_all["pal"] * (1 - p))
                removed_raw[kg_name] = number_text(weekday_all["kg"] * p)
                removed_raw[pal_name] = number_text(weekday_all["pal"] * p)
            else:
                removed_raw = raw.loc[raw_idx[gone]]
            name = slug(mode, p)
            f_out, f_rem = out_dir / f"Orders_{name}_{stamp}.csv", out_dir / f"Removed_{name}_{stamp}.csv"
            kept_raw.to_csv(f_out, index=False, encoding="utf-8-sig")
            removed_raw.to_csv(f_rem, index=False, encoding="utf-8-sig")
            print(f"Saved {f_out.name} and {f_rem.name}")

            # check: reading the file back gives the scenario's reduced weekday kg in every example week
            back = pe.load_orders(f_out)
            back = back[back["day"].isin(DAYS)]
            for wk in weeks:
                want = reduced.loc[reduced["wk"] == wk, "kg"].sum()
                got = back.loc[back["wk"] == wk, "kg"].sum()
                rows.append({"Mode": mode, "Reduction %": round(p * 100), "Week": wk,
                             "File": f_out.name, "Weekday KG in file": round(got, 1),
                             "Weekday KG in scenario": round(want, 1), "Match": abs(got - want) < 0.5})
            rows.append({"Mode": mode, "Reduction %": round(p * 100), "Week": "All weeks in file",
                         "File": f_out.name, "Rows in": len(raw), "Rows out": len(kept_raw),
                         "Rows removed": len(raw) - len(kept_raw),
                         "Weekday KG in file": round(back["kg"].sum(), 1),
                         "Weekday KG original": round(weekday_all["kg"].sum(), 1),
                         "Actual KG removed %": round(100 * (1 - back["kg"].sum() / weekday_all["kg"].sum()), 2)
                         if weekday_all["kg"].sum() else None})
    bad = [r for r in rows if r.get("Match") is False]
    if bad:
        print(f"WARNING: {len(bad)} exported week(s) do not match the scenario kg; see Checks")
    return rows


def customer_reports(pd, pe, report_runs, removed, band_of, orders, plan, run_dir, stamp, seed):
    """customer_report.build for each mode and reduction, with the scenario named on the front sheet."""
    try:
        import customer_report
    except Exception as e:  # noqa: BLE001
        print(f"Customer reports skipped: customer_report import failed: {type(e).__name__}: {e}")
        return []
    base_text = customer_report.R["scenario"]
    rows = []
    try:
        for (mode, p), res in sorted(report_runs.items(), key=lambda t: (MODES.index(t[0][0]), t[0][1])):
            r = removed[(removed["Mode"] == mode) & (removed["Reduction %"] == round(p * 100))]
            if mode == MODE_RANDOM:
                r = r[r["Seed"].astype(str) == str(seed)]
            actual = r["Actual KG removed %"].mean()
            plain = {MODE_BLANKET: f"every order {p * 100:g}% lighter",
                     MODE_RANDOM: f"{p * 100:g}% of orders removed at random on each depot and day",
                     MODE_FAR: f"whole postcode areas furthest from each depot removed until at least "
                               f"{p * 100:g}% of its weight is gone",
                     MODE_NEAR: f"whole postcode areas nearest each depot removed until at least "
                                f"{p * 100:g}% of its weight is gone"}[mode]
            got = f" ({actual:.1f}% removed on average)" if pd.notna(actual) and mode in (MODE_FAR, MODE_NEAR) else ""
            customer_report.R["scenario"] = (f"{base_text} Volume scenario: {plain}{got}. Before and after "
                                             f"are both at this lower volume.")
            out = customer_report.build(run_dir, f"{stamp}_{slug(mode, p)}", band_of, res["Vehicles"],
                                        pd.concat(res["Orders"]), res["Daily"], orders, plan)
            print(f"Saved {out.name}")
            rows.append({"Mode": mode, "Reduction %": round(p * 100),
                         "Seed": seed if mode == MODE_RANDOM else "",
                         "Average KG removed % (per depot)": round(actual, 2) if pd.notna(actual) else None,
                         "Report": out.name})
    finally:
        customer_report.R["scenario"] = base_text
    return rows


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
