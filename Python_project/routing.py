"""Route distances for the day-pattern planning tools.

Straight-line distances in km from the grid references in the matrix. Each
vehicle route is a closed loop: depot -> stops -> depot. Straight-line distance
understates road distance, and by different amounts in rural and urban areas,
so the totals are for comparing options, not for mileage.

Depot positions: the matrix has no depot grid references. Unless a depot is
set in DEPOT_GRID_KM below, its position is the kg-weighted centre of the
areas it serves. That is a proxy, and it moves the legs if the real depot
sits away from its areas. Fill in DEPOT_GRID_KM with real refs when available.

Used by planning_estimate.py and the quality sweep. The anneal does not use it.
"""
import math

import numpy as np

from matrix_common import AREA, DEPOT, EAST, NORTH, load_matrix, select_matrix_path

WEIGHT = "Weight"             # column load_matrix adds: sum of the day kg

IMPROVE_TOL = 1e-9            # km; a 2-opt reversal must save at least this much
DEPOT_KEY = "__DEPOT__"       # placeholder node for the depot in a route

# Optional real depot positions, in km: {"Depot name as in the matrix": (east_km, north_km)}
DEPOT_GRID_KM = {}


# ---- Distance -------------------------------------------------------------
def grid_distance_km(a, b):
    """Straight-line distance between two (x, y) points, both in km."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def load_grid_refs():
    """One matrix load. Returns (area_coords, depot_coords), both in km.

    area_coords: area -> (x, y), averaged over matrix rows with grid references.
    depot_coords: depot -> (x, y), kg-weighted centre of its areas unless set
    in DEPOT_GRID_KM.
    """
    m = load_matrix(select_matrix_path())
    m = m[(m[EAST] > 0) & (m[NORTH] > 0)]

    c = m.groupby(AREA)[[EAST, NORTH]].mean() / 1000
    area_coords = {a: (x, y) for a, x, y in zip(c.index, c[EAST], c[NORTH])}

    depot_coords = {}
    for name, g in m.groupby(DEPOT):
        w = g[WEIGHT].to_numpy(dtype=float)
        if w.sum() > 0:
            x, y = np.average(g[EAST], weights=w), np.average(g[NORTH], weights=w)
        else:
            x, y = g[EAST].mean(), g[NORTH].mean()
        depot_coords[name] = (x / 1000, y / 1000)
    depot_coords.update(DEPOT_GRID_KM)
    return area_coords, depot_coords


def load_area_coords():
    """Area -> (x, y) km. Kept for callers that only need the areas."""
    return load_grid_refs()[0]


def path_km(path, coords):
    return sum(grid_distance_km(coords[a], coords[b]) for a, b in zip(path, path[1:]))


# ---- Route order ----------------------------------------------------------
def nearest_neighbour_loop(stops, pts):
    """Closed loop from the depot: always go to the nearest unvisited stop."""
    remaining = sorted(set(stops))
    path = [DEPOT_KEY]
    while remaining:
        last = pts[path[-1]]
        nxt = min(remaining, key=lambda s: grid_distance_km(last, pts[s]))
        path.append(nxt)
        remaining.remove(nxt)
    path.append(DEPOT_KEY)
    return path


def two_opt_loop(path, pts):
    """Reverse stop segments while that shortens the closed loop.

    The depot is at both ends and never moves. Reversal range: indices 1..n-2.
    """
    path = list(path)
    n = len(path)

    def d(a, b):
        return grid_distance_km(pts[a], pts[b])

    improved = True
    while improved:
        improved = False
        for i in range(1, n - 2):
            for j in range(i + 1, n - 1):
                before = d(path[i - 1], path[i]) + d(path[j], path[j + 1])
                after = d(path[i - 1], path[j]) + d(path[i], path[j + 1])
                if after < before - IMPROVE_TOL:
                    path[i:j + 1] = path[i:j + 1][::-1]
                    improved = True
    return path


def vehicle_route(areas, coords, depot_xy):
    """Closed route for one vehicle: depot -> stops -> depot.

    Returns (order, route_km, stop_km, unlocated):
      order      depot, stops in visiting order, depot
      route_km   full loop, including both depot legs
      stop_km    stop-to-stop distance only (route_km minus the two depot legs)
      unlocated  areas with no grid reference; left out of the route and listed
                 so the totals are not silently low
    """
    unlocated = sorted({a for a in areas if a not in coords})
    located = sorted({a for a in areas if a in coords})
    if not located:
        return [], 0.0, 0.0, unlocated

    pts = dict(coords)
    pts[DEPOT_KEY] = depot_xy
    order = nearest_neighbour_loop(located, pts)
    order = two_opt_loop(order, pts)
    stop_km = path_km(order[1:-1], pts) if len(located) > 1 else 0.0
    route_km = path_km(order, pts)
    return order, route_km, stop_km, unlocated


def route_legs(order, coords, depot_xy):
    """One row per point on a route, for checking by hand.

    Seq 0 is the depot; the last row is the depot again. Leg km is the straight
    line from the previous point. Cumulative km at the end equals route_km.
    """
    pts = dict(coords)
    pts[DEPOT_KEY] = depot_xy
    rows, cum, prev = [], 0.0, None
    for seq, node in enumerate(order):
        leg = grid_distance_km(pts[prev], pts[node]) if prev is not None else 0.0
        cum += leg
        x, y = pts[node]
        rows.append({"Seq": seq, "Stop": "Depot" if node == DEPOT_KEY else node,
                     "Easting km": round(x, 2), "Northing km": round(y, 2),
                     "Leg km": round(leg, 2), "Cumulative km": round(cum, 2)})
        prev = node
    return rows


# ---- Summaries ------------------------------------------------------------
def sum_route_km(vehicle_table, by, column="Route km"):
    """Total of `column` by the given columns, e.g. ["Week", "Depot", "Day"].

    vehicle_table is a DataFrame with one row per vehicle.
    """
    return vehicle_table.groupby(by, as_index=False)[column].sum()
