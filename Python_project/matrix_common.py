"""Shared loader, day rules and neighbour checks used by both balancing approaches."""
import itertools
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from dotenv import load_dotenv

import sys
import time

# ---- Shared settings ------------------------------------------------------

SHEET_NAME = 0            # used only if the matrix is an Excel file

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
AREA = "Postcode Area"
DEPOT = "Depot"
EAST = "GRIDE_Ave"
NORTH = "GRIDN_Ave"

MAX_DAYS = 2              # maximum number of days in a pattern
NEIGHBOURS = 4            # a district is "next to" its 4 nearest districts

# ---- environment variables ------------------------------------------------------

load_dotenv()
BUILT_MATRIX_PATH = Path(os.environ["MATRIX_OUT"])  # built matrix (build_matrix output)
CURRENT_MATRIX_PATH = Path(os.environ["MATRIX_CURRENT"])  # existing matrix (Matrix V1)

# ---- Matrix sources -------------------------------------------------------



MATRIX_OPTIONS = {
    "1": ("Built matrix (build_matrix output)", BUILT_MATRIX_PATH),
    "2": ("Existing matrix (Matrix V1)", CURRENT_MATRIX_PATH),
}


def select_matrix_path():
    """Ask in the terminal which matrix to use. Repeats until a valid, existing file is chosen."""
    print("\nWhich matrix should the solver use?")
    for key, (label, path) in MATRIX_OPTIONS.items():
        print(f"  {key}. {label}\n     {path}")
    while True:
        choice = input("Enter 1 or 2: ").strip()
        if choice not in MATRIX_OPTIONS:
            print("Please enter 1 or 2.")
            continue
        path = MATRIX_OPTIONS[choice][1]
        if not path.exists():
            print(f"File not found: {path}")
            continue
        return path

# ---- Patterns -------------------------------------------------------------
def adjacent_pairs(pattern):
    idx = sorted(DAYS.index(d) for d in pattern)
    return sum(1 for a, b in zip(idx, idx[1:]) if b - a == 1)


# Patterns are ordered tuples in day order. Two-day patterns must not be
# consecutive (no Mon/Tue, Tue/Wed, etc.). Mon/Fri is allowed.
ALL_PATTERNS = [c for k in range(2, MAX_DAYS + 1)
                for c in itertools.combinations(DAYS, k)]
PATTERNS = [p for p in ALL_PATTERNS if len(p) > 2 or adjacent_pairs(p) == 0]


# ---- Day mapping (next available day, wrapping to the first) --------------
def map_days(old_days, new_days):
    """Keep a delivery day if the pattern has it. Otherwise move it to the
    nearest pattern day by calendar distance; on a tie, take the earlier day."""
    mapping = {}
    for d in old_days:
        if d in new_days:
            mapping[d] = d
        else:
            mapping[d] = min(
                new_days,
                key=lambda n: (abs(DAYS.index(n) - DAYS.index(d)), DAYS.index(n)))
    return mapping


def district_new_loads(row, pattern):
    old = [d for d in DAYS if row[d] > 0]
    mapping = map_days(old, pattern)
    new = {d: 0.0 for d in DAYS}
    for d in old:
        new[mapping[d]] += row[d]
    return old, new


def zone_loads(rows, pattern):
    loads = {d: 0.0 for d in DAYS}
    for _, row in rows.iterrows():
        _, new = district_new_loads(row, pattern)
        for d in DAYS:
            loads[d] += new[d]
    return loads


# ---- Loading --------------------------------------------------------------
def clean_number(series):
    s = series.astype(str).str.strip().str.replace(",", "", regex=False)
    s = s.replace({"-": "0", "nan": "0", "": "0"})
    return pd.to_numeric(s, errors="coerce").fillna(0.0)


def load_matrix(path):
    path = Path(path)
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path, thousands=",", dtype=str)
    else:
        df = pd.read_excel(path, sheet_name=SHEET_NAME, dtype=str)

    df.columns = [str(c).strip() for c in df.columns]
    depot_col = next(c for c in df.columns if "Depot" in c)
    df = df.rename(columns={depot_col: DEPOT})

    df[AREA] = df[AREA].astype(str).str.strip()
    df[DEPOT] = df[DEPOT].astype(str).str.strip()
    for c in DAYS + [EAST, NORTH]:
        df[c] = clean_number(df[c])

    df["Weight"] = df[DAYS].sum(axis=1)
    return df


# ---- Adjacency ------------------------------------------------------------
def build_adjacency(xy, k=NEIGHBOURS):
    nn = NearestNeighbors(n_neighbors=min(k + 1, len(xy))).fit(xy)
    _, idx = nn.kneighbors(xy)
    adj = [set() for _ in range(len(xy))]
    for i, row in enumerate(idx):
        for j in row[1:]:
            adj[i].add(int(j))
            adj[int(j)].add(i)
    return adj


def count_disconnected(labels, adj):
    bad = 0
    for z in pd.unique(labels):
        members = [int(i) for i in np.flatnonzero(labels == z)]
        if len(members) < 2:
            continue
        inside = set(members)
        seen = {members[0]}
        stack = [members[0]]
        while stack:
            i = stack.pop()
            for j in adj[i]:
                if j in inside and j not in seen:
                    seen.add(j)
                    stack.append(j)
        if len(seen) < len(members):
            bad += 1
    return bad

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


def closed_tour_km(pts, starts=20):
    """Closed loop through the points in km (heuristic). Points with no grid reference are ignored."""
    pts = pts[(pts[:, 0] > 0) & (pts[:, 1] > 0)]
    n = len(pts)
    if n < 2:
        return 0.0
    d = np.hypot(pts[:, None, 0] - pts[None, :, 0],
                 pts[:, None, 1] - pts[None, :, 1]) / 1000
    best = None
    for start in range(min(n, starts)):
        left = set(range(n)) - {start}
        order = [start]
        while left:
            nxt = min(left, key=lambda k: d[order[-1], k])
            order.append(nxt)
            left.remove(nxt)
        length = loop_km(order, d)
        if best is None or length < best[0]:
            best = (length, order)
    return loop_km(two_opt(best[1], d), d)


def ask_skip(seconds=5, key="s"):
    """True if the key is pressed within `seconds`; False if the time runs out."""
    print(f"Press S to skip step 1 (depot assignment). "
          f"It runs in {seconds} seconds if nothing is pressed.", flush=True)
    try:
        import msvcrt                      # Windows
    except ImportError:
        msvcrt = None

    if msvcrt:
        deadline = time.time() + seconds
        while time.time() < deadline:
            if msvcrt.kbhit() and msvcrt.getwch().lower() == key:
                print()
                return True
            time.sleep(0.05)
        print()
        return False

    # Other systems: S must be followed by Enter
    import select
    ready, _, _ = select.select([sys.stdin], [], [], seconds)
    if ready and sys.stdin.readline().strip().lower() == key:
        return True
    print()
    return False