"""Simulated annealing for the regional day-pattern method.

Out-of-area letters are removed first (external pallet network). Weighted
letters are grouped geographically, then one day pattern is searched per group.
Cost = day radius + service term (days per week) + penalties for exceeding the
peak limit or for uneven days. The peak limit is the Letters method's peak,
calculated at run time from the same active districts.
"""
import math
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from matrix_common import (MATRIX_PATH, AREA, DEPOT, DAYS, EAST, NORTH,
                           NEIGHBOURS, PATTERNS, district_new_loads, load_matrix,
                           build_adjacency)
from balence_greedyV1 import (find_outliers, zones_letters,
                              evaluate as letters_evaluate)

# ---- Settings -------------------------------------------------------------
DEPOT_NAME = "Watford"
OUTPUT_DIR = Path(r"C:\Users\micranne\OneDrive - DPDHL\Documents\Porcelanosa\Porject Swichback\Matix files\Output")
RUN_STAMP = datetime.now().strftime("%Y-%m-%d_%H-%M")   # set when the script starts

# Out-of-area letters: external pallet network, removed before planning
EXTERNAL_LETTERS = ["BB", "CA", "DH", "EH", "G", "L", "LS", "ML", "SK", "LE", "LN"]

GROUP_SIZE = 3             # about this many letter zones per group
PEAK_TOLERANCE = 0.03      # peak may rise this far above the Letters baseline peak
PEAK_PENALTY = 500.0       # km added per 100% of peak excess above the limit

SPREAD_TOLERANCE = 0.05    # busiest minus quietest day, as a share of the average day
SPREAD_PENALTY = 500.0     # km added per 100% of spread excess above the tolerance

DAY_LOSS_KM = 10.0         # km cost per day short of a full week, averaged over kg
MAX_DAY_COUNT = max(len(p) for p in PATTERNS)   # 3 with the current rules

MAX_COPY_KM = 15.0         # zero-volume codes further than this get no days

ITERATIONS = 200000        # moves per restart
RESTARTS = 5
T_START = 10.0             # km: early moves may worsen the cost by about this much
T_END = 0.05               # km: late moves must almost always improve
SEED = 0
AUTO_OPEN = True


# ---- Day radius -----------------------------------------------------------
def day_radius_km(M, xy):
    total, kg = 0.0, 0.0
    for j in range(M.shape[1]):
        w = M[:, j]
        s = w.sum()
        if s <= 0:
            continue
        cx = (w * xy[:, 0]).sum() / s
        cy = (w * xy[:, 1]).sum() / s
        total += (w * np.hypot(xy[:, 0] - cx, xy[:, 1] - cy)).sum()
        kg += s
    return total / kg / 1000 if kg else 0.0


# ---- Grouping (weighted letters only; zero-weight letters join afterwards) -
def build_groups(active, df):
    w = active["Weight"].to_numpy()
    xy = active[[EAST, NORTH]].to_numpy()
    lab = active["Letters"].to_numpy()
    zones = list(pd.unique(lab))
    idx_by_zone = {z: np.flatnonzero(lab == z) for z in zones}
    centres = np.array([np.average(xy[idx_by_zone[z]], axis=0,
                                   weights=w[idx_by_zone[z]]) for z in zones])
    kgs = np.array([w[idx_by_zone[z]].sum() for z in zones])

    R = max(1, min(len(zones), round(len(zones) / GROUP_SIZE)))
    km = KMeans(n_clusters=R, n_init=10, random_state=0)
    km.fit(centres, sample_weight=kgs)
    groups = [[zones[k] for k in range(len(zones)) if km.labels_[k] == r]
              for r in range(R)]
    group_centres = np.array([
        np.average(centres[km.labels_ == r], axis=0,
                   weights=kgs[km.labels_ == r])
        for r in range(R)])

    # Zero-weight letters join the nearest group centre afterwards
    for L, g in df.groupby("Letters"):
        if L in idx_by_zone:
            continue
        pts = g[[EAST, NORTH]].to_numpy()
        pts = pts[(pts[:, 0] > 0) & (pts[:, 1] > 0)]
        if not len(pts):
            continue
        c = pts.mean(axis=0)
        r = int(np.argmin(np.hypot(*(group_centres - c).T)))
        groups[r].append(L)
    return groups, idx_by_zone


def precompute(active, groups, idx_by_zone):
    """For each group and each pattern, the district-by-day kg matrix."""
    group_rows, M = [], {}
    for r, zs in enumerate(groups):
        rows = [i for z in zs if z in idx_by_zone for i in idx_by_zone[z]]
        group_rows.append(np.array(rows, dtype=int))
        for p in PATTERNS:
            block = np.zeros((len(rows), len(DAYS)))
            for k, i in enumerate(rows):
                _, new = district_new_loads(active.iloc[i], p)
                block[k] = [new[d] for d in DAYS]
            M[(r, p)] = block
    return group_rows, M


# ---- Annealing ------------------------------------------------------------
def assemble(choice, group_rows, M, n):
    full = np.zeros((n, len(DAYS)))
    for r, pi in enumerate(choice):
        if len(group_rows[r]):
            full[group_rows[r]] = M[(r, PATTERNS[pi])]
    return full


def evaluate(choice, group_rows, M, n, xy, limit):
    full = assemble(choice, group_rows, M, n)
    day_tot = full.sum(axis=0)
    peak = day_tot.max()
    spread = (peak - day_tot.min()) / day_tot.mean()
    dr = day_radius_km(full, xy)

    # kg-weighted average number of delivery days across the depot
    total_kg, weighted_days = 0.0, 0.0
    for r, pi in enumerate(choice):
        p = PATTERNS[pi]
        kg = M[(r, p)].sum()
        total_kg += kg
        weighted_days += len(p) * kg
    avg_days = weighted_days / total_kg if total_kg else 0.0
    service = DAY_LOSS_KM * (MAX_DAY_COUNT - avg_days)

    peak_excess = max(0.0, peak / limit - 1)
    spread_excess = max(0.0, spread - SPREAD_TOLERANCE)
    cost = (dr + service + PEAK_PENALTY * peak_excess
            + SPREAD_PENALTY * spread_excess)
    return {"cost": cost, "dr": dr, "peak": peak, "spread": spread,
            "avg_days": avg_days,
            "feasible": peak <= limit and spread <= SPREAD_TOLERANCE}


def better(a, b):
    """Feasible results first, then the shorter day radius."""
    return (not a["feasible"], a["dr"]) < (not b["feasible"], b["dr"])


def anneal(R, group_rows, M, n, xy, limit, rng):
    choice = [int(rng.integers(len(PATTERNS))) for _ in range(R)]
    cur = evaluate(choice, group_rows, M, n, xy, limit)
    best_choice, best = list(choice), cur

    for it in range(ITERATIONS):
        temp = T_START * (T_END / T_START) ** (it / ITERATIONS)
        cand = list(choice)
        if R < 2 or rng.random() < 0.7:
            cand[int(rng.integers(R))] = int(rng.integers(len(PATTERNS)))
        else:
            a, b = rng.choice(R, size=2, replace=False)
            cand[int(a)], cand[int(b)] = cand[int(b)], cand[int(a)]

        new = evaluate(cand, group_rows, M, n, xy, limit)
        delta = new["cost"] - cur["cost"]
        if delta <= 0 or rng.random() < math.exp(-delta / temp):
            choice, cur = cand, new
            if better(cur, best):
                best_choice, best = list(choice), cur
    return best_choice, best


# ---- Outputs --------------------------------------------------------------
def write_outputs(depot_df, active, zero, excluded, outliers, groups,
                  group_pattern, letter_group, best, xy, reference_peak):
    load = {d: 0.0 for d in DAYS}
    active_days, district_rows, map_rows = [], [], []

    for i, row in active.iterrows():
        g = letter_group[row["Letters"]]
        old, new = district_new_loads(row, group_pattern[g])
        days = [d for d in DAYS if new[d] > 0]
        active_days.append(days)
        for d in DAYS:
            load[d] += new[d]
        district_rows.append({
            AREA: row[AREA], "Group": g + 1,
            "Weight kg": round(row["Weight"], 2),
            "Old days": ", ".join(old), "New days": ", ".join(days),
            **{d: round(new[d], 2) for d in DAYS},
            "Assigned from": "", "Distance km": None, "Review": "",
        })
        for d in days:
            map_rows.append({AREA: row[AREA], DEPOT: row[DEPOT],
                             "Weekday": d, "Weight": round(new[d], 2)})

    assert len(active_days) == len(xy) == len(active), (
        f"length mismatch: days {len(active_days)}, xy {len(xy)}, active {len(active)}")

    # Zero-volume codes copy the nearest weighted district's days, within MAX_COPY_KM
    for _, z in zero.iterrows():
        assigned, dist, days, group = "", None, [], ""
        review = "No grid reference - assign manually"
        if z[EAST] != 0 and z[NORTH] != 0:
            d = np.hypot(xy[:, 0] - z[EAST], xy[:, 1] - z[NORTH])
            k = int(np.argmin(d))
            dist = round(d[k] / 1000, 1)
            if dist <= MAX_COPY_KM:
                assigned = active.at[k, AREA]
                days = active_days[k]
                group = letter_group[active.at[k, "Letters"]] + 1
                review = "Zero volume - pattern copied from nearest district, check"
            else:
                review = (f"Outside depot area ({dist} km from nearest weighted district) "
                          "- no days assigned, check")
        district_rows.append({
            AREA: z[AREA], "Group": group, "Weight kg": 0.0, "Old days": "",
            "New days": ", ".join(days), **{d: 0.0 for d in DAYS},
            "Assigned from": assigned, "Distance km": dist, "Review": review,
        })
        for d in days:
            map_rows.append({AREA: z[AREA], DEPOT: z[DEPOT],
                             "Weekday": d, "Weight": 0.0})

    # Excluded isolation outliers: listed, no days, no map points
    for _, r in excluded.iterrows():
        district_rows.append({
            AREA: r[AREA], "Group": "", "Weight kg": round(r["Weight"], 2),
            "Old days": ", ".join(d for d in DAYS if r[d] > 0), "New days": "",
            **{d: 0.0 for d in DAYS}, "Assigned from": "", "Distance km": None,
            "Review": "Excluded - outlier, no days assigned",
        })

    districts_df = pd.DataFrame(district_rows)

    map_df = pd.DataFrame(map_rows, columns=[AREA, DEPOT, "Weekday", "Weight"])
    map_df["_o"] = map_df["Weekday"].map(DAYS.index)
    map_df = (map_df.sort_values([AREA, "_o"]).drop(columns="_o")
              .rename(columns={DEPOT: "tblTeams.Best_postcode.Best Depot"}))

    group_rows = []
    for r, zs in enumerate(groups):
        mask = active["Letters"].isin(zs)
        group_rows.append({"Group": r + 1, "Letters": ", ".join(sorted(zs)),
                           "Pattern": ", ".join(group_pattern[r]),
                           "Districts": int(mask.sum()),
                           "Weight kg": round(active.loc[mask, "Weight"].sum(), 0)})

    before = {d: depot_df[d].sum() for d in DAYS}
    excl_kg = {d: excluded[d].sum() for d in DAYS}
    daily_df = pd.DataFrame([{"Day": d, "Before kg": round(before[d], 0),
                              "Excluded kg": round(excl_kg[d], 0),
                              "After kg": round(load[d], 0)} for d in DAYS])

    summary = pd.DataFrame([{
        "Groups": len(groups), "Group size": GROUP_SIZE,
        "Day radius km": round(best["dr"], 2),
        "Peak kg": round(best["peak"], 0),
        "Spread": round(best["spread"], 3),
        "Avg days": round(best["avg_days"], 2),
        "Reference peak kg": round(reference_peak, 0),
        "Peak limit kg": round(reference_peak * (1 + PEAK_TOLERANCE), 0),
        "Within limits": best["feasible"],
        "Restarts": RESTARTS, "Iterations per restart": ITERATIONS,
    }])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / f"Anneal_{DEPOT_NAME.replace(' ', '_')}_Regional_{RUN_STAMP}.xlsx"
    with pd.ExcelWriter(out) as xl:
        summary.to_excel(xl, sheet_name="Summary", index=False)
        pd.DataFrame(group_rows).to_excel(xl, sheet_name="Groups", index=False)
        districts_df.to_excel(xl, sheet_name="Districts", index=False)
        map_df.to_excel(xl, sheet_name="Map", index=False)
        daily_df.to_excel(xl, sheet_name="Daily loads", index=False)
        outliers.to_excel(xl, sheet_name="Outliers", index=False)
    print(f"Saved {out}")
    if AUTO_OPEN:
        try:
            os.startfile(out)
        except OSError as e:
            print(f"Could not open {out}: {e}")


# ---- Main -----------------------------------------------------------------
def main():
    df = load_matrix(MATRIX_PATH)
    df["Letters"] = df[AREA].str.extract(r"^([A-Z]+)")[0]
    depot_all = df[df[DEPOT] == DEPOT_NAME].reset_index(drop=True)

    # Out-of-area letters go to the external pallet network: removed from the plan
    is_ext = depot_all["Letters"].isin(EXTERNAL_LETTERS)
    external = depot_all[is_ext & (depot_all["Weight"] > 0)][
        [AREA, DEPOT, "Letters", "Weight"]].copy()
    external["Status"] = "External pallet network - not in plan"
    external_kg = external["Weight"].sum()
    depot_df = depot_all[~is_ext].reset_index(drop=True)

    weighted = depot_df[depot_df["Weight"] > 0].reset_index(drop=True)
    zero = depot_df[depot_df["Weight"] <= 0].reset_index(drop=True)

    # Isolation outliers (excluded from the plan, listed on the Outliers tab)
    outliers = find_outliers(weighted)
    excluded_names = set(outliers[AREA]) if not outliers.empty else set()
    excluded = weighted[weighted[AREA].isin(excluded_names)].reset_index(drop=True)
    active = weighted[~weighted[AREA].isin(excluded_names)].reset_index(drop=True)

    outliers = pd.concat([outliers, external], ignore_index=True)
    print(f"External pallet network: {len(external)} weighted districts, "
          f"{external_kg:,.0f} kg removed from the plan")

    # Baseline: the Letters method on exactly the same active districts
    adj = build_adjacency(active[[EAST, NORTH]].to_numpy(), NEIGHBOURS)
    letters_labels = zones_letters(active, active["Weight"].sum(), adj)
    letters_metrics, _ = letters_evaluate(active, letters_labels, adj)
    reference_peak = letters_metrics["Peak kg"]
    limit = reference_peak * (1 + PEAK_TOLERANCE)
    print(f"Letters baseline peak: {reference_peak:,.0f} kg")

    n = len(active)
    xy = active[[EAST, NORTH]].to_numpy()

    groups, idx_by_zone = build_groups(active, depot_df)
    group_rows, M = precompute(active, groups, idx_by_zone)
    R = len(groups)
    print(f"{DEPOT_NAME}: {len(idx_by_zone)} weighted letters, {R} groups, "
          f"peak limit {limit:,.0f} kg")

    rng = np.random.default_rng(SEED)
    best_choice, best = None, None
    for rs in range(RESTARTS):
        choice, res = anneal(R, group_rows, M, n, xy, limit, rng)
        print(f"  restart {rs + 1}: day radius {res['dr']:.2f} km, "
              f"peak {res['peak']:,.0f} kg, spread {res['spread']:.1%}, "
              f"avg days {res['avg_days']:.2f}, "
              f"{'within' if res['feasible'] else 'over'} limits")
        if best is None or better(res, best):
            best_choice, best = choice, res

    group_pattern = {r: PATTERNS[pi] for r, pi in enumerate(best_choice)}
    letter_group = {L: r for r, zs in enumerate(groups) for L in zs}

    print(f"\nBest: day radius {best['dr']:.2f} km, peak {best['peak']:,.0f} kg, "
          f"spread {best['spread']:.1%}, avg days {best['avg_days']:.2f}, "
          f"{'within' if best['feasible'] else 'NOT within'} the limits")
    for r, zs in enumerate(groups):
        print(f"  Group {r + 1}: {', '.join(sorted(zs))} -> "
              f"{', '.join(group_pattern[r])}")

    write_outputs(depot_df, active, zero, excluded, outliers, groups,
                  group_pattern, letter_group, best, xy, reference_peak)


if __name__ == "__main__":
    main()