"""Baseline route lengths per depot from the planned Anneal workbooks.

Run from the editor (no command line). For each Anneal_*.xlsx under the output
folder, orders each group's postcode areas into a closed loop (nearest
neighbour + 2-opt) on British National Grid straight-line distances. Prints
average, maximum and 95th percentile route length by depot, and saves
route_lengths.csv in the output folder.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from matrix_common import AREA, EAST, NORTH, load_matrix, select_matrix_path

# ---- Settings -------------------------------------------------------------
PCT = 95                   # percentile reported as the route cap
# ---------------------------------------------------------------------------

load_dotenv()
OUTPUT_DIR = Path(os.getenv("A_OUTPUT_DIR"))


def loop_km(order, d):
    n = len(order)
    return sum(d[order[k], order[(k + 1) % n]] for k in range(n))


def two_opt(order, d):
    n = len(order)
    improved = True
    while improved:
        improved = False
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                a, b = order[i - 1], order[i]
                c, e = order[j], order[(j + 1) % n]
                if d[a, b] + d[c, e] > d[a, c] + d[b, e] + 1e-9:
                    order[i:j + 1] = order[i:j + 1][::-1]
                    improved = True
    return order


def closed_tour_km(pts):
    """Shortest closed loop through the points, in km (heuristic)."""
    n = len(pts)
    if n < 2:
        return 0.0
    d = np.hypot(pts[:, None, 0] - pts[None, :, 0],
                 pts[:, None, 1] - pts[None, :, 1]) / 1000
    best = None
    for start in range(n):
        left = set(range(n)) - {start}
        order = [start]
        while left:
            nxt = min(left, key=lambda k: d[order[-1], k])
            order.append(nxt)
            left.remove(nxt)
        length = loop_km(order, d)
        if best is None or length < best[0]:
            best = (length, order)
    order = two_opt(best[1], d)
    return loop_km(order, d)


def main():
    matrix = load_matrix(select_matrix_path())
    xy = matrix.drop_duplicates(AREA).set_index(AREA)[[EAST, NORTH]]

    records = []
    latest = {}
    for path in sorted(OUTPUT_DIR.rglob("Anneal_*.xlsx")):
        depot = pd.read_excel(path, sheet_name="Summary")["Depot"].iloc[0]
        latest[depot] = path          # file names carry the run timestamp, so the last one is newest

    for depot, path in latest.items():
        dist = pd.read_excel(path, sheet_name="Districts")
        dist = dist[pd.to_numeric(dist["Group"], errors="coerce").notna()]
        dist[AREA] = dist[AREA].astype(str).str.strip()

        for g, rows in dist.groupby("Group"):
            areas = [a for a in rows[AREA] if a in xy.index]
            pts = xy.loc[areas, [EAST, NORTH]].to_numpy(float)
            pts = pts[(pts[:, 0] > 0) & (pts[:, 1] > 0)]
            records.append({"Depot": depot, "Group": int(g),
                            "Districts": len(pts),
                            "Route km": round(closed_tour_km(pts), 1)})
        print(f"Using {path.name}: {depot}")

    routes = pd.DataFrame(records)
    summary = routes.groupby("Depot")["Route km"].agg(
        Routes="count", Average="mean", Maximum="max",
        P95=lambda s: np.percentile(s, PCT)).round(1)
    print("\n", summary.to_string())

    out_file = OUTPUT_DIR / "route_lengths.csv"
    while True:
        try:
            routes.to_csv(out_file, index=False)
            print(f"\nSaved {out_file}")
            break
        except PermissionError:
            try:
                input(f"\n{out_file.name} is open. Close it, then press Enter to retry (Ctrl+C to cancel)... ")
            except KeyboardInterrupt:
                print("\nSave cancelled. Results were printed above but not saved.")
                break


if __name__ == "__main__":
    main()