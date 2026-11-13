"""Simulated annealing for the regional day-pattern method.

Runs one depot, or every depot when DEPOT_NAME is None. Remote letters (far from
the depot's main body with few weighted districts) are removed as external.
Weighted letters are grouped geographically, then one day pattern is searched
per group. Cost = day radius + peak term + service term (days per week) +
penalties. The peak limit is the Letters method's peak, calculated at run time
from the same active districts.
"""
import math
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from matrix_common import (AREA, DAYS, DEPOT, EAST, NORTH,
                           PATTERNS, district_new_loads, load_matrix,
                           select_matrix_path, zone_loads)


# ---- Settings -------------------------------------------------------------
DEPOT_NAME = None          # a depot name, or None = every depot
OUTPUT_DIR = Path(r"C:\Users\micranne\OneDrive - DPDHL\Documents\Porcelanosa\project_swichback\Matix files\Output")
RUN_STAMP = datetime.now().strftime("%Y-%m-%d_%H-%M")   # set when the script starts

# Remote letters: weighted centre further than REMOTE_KM from the depot's main
# body, and fewer than MIN_DISTRICTS weighted districts, are treated as external
REMOTE_KM = 150.0
MIN_DISTRICTS = 2

GROUP_SIZE = 3             # about this many letter zones per group
PEAK_TOLERANCE = 0.03      # peak may rise this far above the Letters baseline peak
PEAK_PENALTY = 500.0       # km added per 100% of peak excess above the limit
PEAK_COST_KM = 1000.0      # km of day radius a 100% peak reduction is worth (0 = ignore)

SPREAD_TOLERANCE = 0.05    # busiest minus quietest day, as a share of the average day
SPREAD_PENALTY = 500.0     # km added per 100% of spread excess above the tolerance

SPACING_DAYS_KM = 5.0      # km per unit of uneven spacing between a group's delivery days (0 = ignore)
SPACING_IDEAL_GAP = 2      # ideal gap in days between consecutive deliveries (every other day)

DAY_LOSS_KM = 10.0         # km cost per day short of a full week, averaged over kg
MAX_DAY_COUNT = max(len(p) for p in PATTERNS)   # 3 with the current rules

ISOLATION_KM = None        # e.g. 10 excludes districts averaging over 10 km to neighbours
PERCENTILE = 98            # used when ISOLATION_KM is None

MAX_COPY_KM = 15.0         # zero-volume codes further than this get no days

ITERATIONS = 200000        # moves per restart
RESTARTS = 5
T_START = 10.0             # km: early moves may worsen the cost by about this much
T_END = 0.05               # km: late moves must almost always improve
SEED = 0
AUTO_OPEN = True           # opens the workbook only when a single depot is run

from sklearn.neighbors import NearestNeighbors

# ---- Local helpers -----------------------------------------------------------
def service_score(pattern):
    score = len(pattern) * 10
    if set(pattern) == {"Monday", "Friday"}:
        score -= 3      # allowed, but the weekend gap ranks lower
    return score

def spacing_penalty(pattern):
    """Average distance of the gaps between a pattern's days from the ideal gap.
    Mon/Wed/Fri = 0, Mon/Thu = 1, Mon/Fri = 2, Mon/Tue/Wed = 1."""
    idx = sorted(DAYS.index(d) for d in pattern)
    gaps = [b - a for a, b in zip(idx, idx[1:])]
    return sum(abs(g - SPACING_IDEAL_GAP) for g in gaps) / len(gaps)


SPACING_PEN = {p: spacing_penalty(p) for p in PATTERNS}

def assign_patterns(df, labels):
    w = df["Weight"].to_numpy()
    zone_idx = {z: np.flatnonzero(labels == z) for z in pd.unique(labels)}
    order = sorted(zone_idx, key=lambda z: -w[zone_idx[z]].sum())
    load = {d: 0.0 for d in DAYS}
    zone_pattern = {}
    for z in order:
        rows = df.iloc[zone_idx[z]]
        best = None
        for p in PATTERNS:
            trial = zone_loads(rows, p)
            new = {d: load[d] + trial[d] for d in DAYS}
            key = (max(new.values()), -service_score(p),
                   max(new.values()) - min(new.values()))
            if best is None or key < best[0]:
                best = (key, p, trial)
        _, pattern, trial = best
        zone_pattern[z] = pattern
        for d in DAYS:
            load[d] += trial[d]
    return zone_pattern, load


def letters_baseline_peak(active):
    labels = active["Letters"].to_numpy(dtype=object)
    _, load = assign_patterns(active, labels)
    return max(load.values())


def find_outliers(df):
    """Districts far from their nearest neighbours, or from their letter's centre."""
    active = df[df["Weight"] > 0].copy()
    if len(active) < 6:
        return pd.DataFrame()

    xy = active[[EAST, NORTH]].to_numpy()
    nn = NearestNeighbors(n_neighbors=5).fit(xy)
    dist, _ = nn.kneighbors(xy)
    active["Isolation km"] = dist[:, 1:].mean(axis=1) / 1000

    cx, cy = {}, {}
    for letter, g in active.groupby("Letters"):
        cx[letter] = np.average(g[EAST], weights=g["Weight"])
        cy[letter] = np.average(g[NORTH], weights=g["Weight"])
    active["Centre dist km"] = (np.hypot(
        active[EAST] - active["Letters"].map(cx),
        active[NORTH] - active["Letters"].map(cy)) / 1000)

    threshold = (ISOLATION_KM if ISOLATION_KM is not None
                 else np.percentile(active["Isolation km"], PERCENTILE))
    out = active[active["Isolation km"] > threshold].copy()
    out["Threshold km"] = round(threshold, 1)
    out = out.sort_values("Isolation km", ascending=False)
    out = out[[AREA, DEPOT, "Letters", "Weight", "Isolation km",
               "Centre dist km", "Threshold km"]].round(
        {"Weight": 0, "Isolation km": 1, "Centre dist km": 1})
    out["Status"] = "Excluded - no days assigned"
    return out

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


# ---- Remote letters -------------------------------------------------------
def find_remote_letters(depot_all):
    """Letters far from the depot's main body. Those with fewer weighted
    districts than MIN_DISTRICTS are external; the rest are kept in the plan."""
    lab = depot_all[depot_all["Weight"] > 0].copy()
    lab["wx"] = lab[EAST] * lab["Weight"]
    lab["wy"] = lab[NORTH] * lab["Weight"]
    lc = lab.groupby("Letters").agg(
        wx=("wx", "sum"), wy=("wy", "sum"),
        kg=("Weight", "sum"), districts=(AREA, "count"))
    lc["x"] = lc["wx"] / lc["kg"]
    lc["y"] = lc["wy"] / lc["kg"]
    main_c = np.average(lc[["x", "y"]].to_numpy(), axis=0,
                        weights=lc["kg"].to_numpy())
    lc["km"] = np.hypot(lc["x"] - main_c[0], lc["y"] - main_c[1]) / 1000

    remote = lc[lc["km"] > REMOTE_KM]
    external = list(remote.index[remote["districts"] < MIN_DISTRICTS])
    kept = list(remote.index[remote["districts"] >= MIN_DISTRICTS])
    return external, kept


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

    # kg-weighted average number of delivery days, and spacing penalty, across the depot
    total_kg, weighted_days, weighted_spacing = 0.0, 0.0, 0.0
    for r, pi in enumerate(choice):
        p = PATTERNS[pi]
        kg = M[(r, p)].sum()
        total_kg += kg
        weighted_days += len(p) * kg
        weighted_spacing += SPACING_PEN[p] * kg
    avg_days = weighted_days / total_kg if total_kg else 0.0
    spacing = weighted_spacing / total_kg if total_kg else 0.0
    service = DAY_LOSS_KM * (MAX_DAY_COUNT - avg_days)

    peak_excess = max(0.0, peak / limit - 1)
    spread_excess = max(0.0, spread - SPREAD_TOLERANCE)
    cost = (dr + service + SPACING_DAYS_KM * spacing
            + PEAK_COST_KM * peak / limit
            + PEAK_PENALTY * peak_excess + SPREAD_PENALTY * spread_excess)
    return {"cost": cost, "dr": dr, "peak": peak, "spread": spread,
            "avg_days": avg_days, "spacing": spacing,
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
                  group_pattern, letter_group, best, xy, reference_peak,
                  depot_name, run_dir):
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
        "Depot": depot_name,
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

    run_dir.mkdir(parents=True, exist_ok=True)
    out = run_dir / f"Anneal_{depot_name.replace(' ', '_')}_Regional_{RUN_STAMP}.xlsx"
    with pd.ExcelWriter(out) as xl:
        summary.to_excel(xl, sheet_name="Summary", index=False)
        pd.DataFrame(group_rows).to_excel(xl, sheet_name="Groups", index=False)
        districts_df.to_excel(xl, sheet_name="Districts", index=False)
        map_df.to_excel(xl, sheet_name="Map", index=False)
        daily_df.to_excel(xl, sheet_name="Daily loads", index=False)
        outliers.to_excel(xl, sheet_name="Outliers", index=False)
    print(f"Saved {out}")
    return out


# ---- Per-depot run --------------------------------------------------------
def run_depot(df, depot_name, run_dir):
    depot_all = df[df[DEPOT] == depot_name].reset_index(drop=True)
    if depot_all.empty:
        print(f"\n{depot_name}: no rows, skipping")
        return None

    # Remote letters: far from the main body with few weighted districts
    ext_list, kept = find_remote_letters(depot_all)
    print(f"\n{'=' * 60}\n{depot_name}")
    print(f"Removed as external: {ext_list or 'none'}")
    print(f"Far but kept (several districts): {kept or 'none'}")

    is_ext = depot_all["Letters"].isin(ext_list)
    external = depot_all[is_ext & (depot_all["Weight"] > 0)][
        [AREA, DEPOT, "Letters", "Weight"]].copy()
    external["Status"] = "External pallet network - review"
    external_kg = external["Weight"].sum()
    depot_df = depot_all[~is_ext].reset_index(drop=True)

    weighted = depot_df[depot_df["Weight"] > 0].reset_index(drop=True)
    zero = depot_df[depot_df["Weight"] <= 0].reset_index(drop=True)
    if weighted.empty:
        print(f"{depot_name}: no weighted districts, skipping")
        return None

    # Isolation outliers (excluded from the plan, listed on the Outliers tab)
    outliers = find_outliers(weighted)
    excluded_names = set(outliers[AREA]) if not outliers.empty else set()
    excluded = weighted[weighted[AREA].isin(excluded_names)].reset_index(drop=True)
    active = weighted[~weighted[AREA].isin(excluded_names)].reset_index(drop=True)
    if active.empty:
        print(f"{depot_name}: all districts excluded, skipping")
        return None

    outliers = pd.concat([outliers, external], ignore_index=True)
    print(f"External pallet network: {len(external)} districts, "
          f"{external_kg:,.0f} kg removed from the plan")

    # Peak limit: Letters baseline on exactly the same active districts
    reference_peak = letters_baseline_peak(active)
    limit = reference_peak * (1 + PEAK_TOLERANCE)
    print(f"Letters baseline peak: {reference_peak:,.0f} kg")

    n = len(active)
    xy = active[[EAST, NORTH]].to_numpy()

    groups, idx_by_zone = build_groups(active, depot_df)
    group_rows, M = precompute(active, groups, idx_by_zone)
    R = len(groups)
    print(f"{len(idx_by_zone)} weighted letters, {R} groups, "
          f"peak limit {limit:,.0f} kg")

    rng = np.random.default_rng(SEED)
    best_choice, best = None, None
    for rs in range(RESTARTS):
        choice, res = anneal(R, group_rows, M, n, xy, limit, rng)
        print(f"  restart {rs + 1}: day radius {res['dr']:.2f} km, "
              f"peak {res['peak']:,.0f} kg, spread {res['spread']:.1%}, "
              f"avg days {res['avg_days']:.2f}, spacing {res['spacing']:.2f}, "
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

    return write_outputs(depot_df, active, zero, excluded, outliers, groups,
                         group_pattern, letter_group, best, xy, reference_peak,
                         depot_name, run_dir)


# ---- Main -----------------------------------------------------------------
def main():
    matrix_path = select_matrix_path()
    print(f"Using matrix: {matrix_path}")
    df = load_matrix(matrix_path)
    df["Letters"] = df[AREA].str.extract(r"^([A-Z]+)")[0]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    depots = sorted(df[DEPOT].unique()) if DEPOT_NAME is None else [DEPOT_NAME]
    print(f"Depots to run: {', '.join(depots)}")

    # One dated folder per multi-depot run; a single depot saves to the output folder
    if len(depots) > 1:
        run_dir = OUTPUT_DIR / f"Run_{RUN_STAMP}"
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving this run to: {run_dir}")
    else:
        run_dir = OUTPUT_DIR

    outputs = []
    for depot in depots:
        out = run_depot(df, depot, run_dir)
        if out is not None:
            outputs.append(out)

    print(f"\nFinished: {len(outputs)} of {len(depots)} depots saved to {run_dir}")
    if AUTO_OPEN and len(outputs) == 1:
        try:
            os.startfile(outputs[0])
        except OSError as e:
            print(f"Could not open {outputs[0]}: {e}")
    elif AUTO_OPEN and len(outputs) > 1:
        try:
            os.startfile(run_dir)      # open the folder so all files are visible
        except OSError as e:
            print(f"Could not open {run_dir}: {e}")


if __name__ == "__main__":
    main()