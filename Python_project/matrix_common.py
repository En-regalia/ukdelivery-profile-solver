"""Shared loader, day rules and neighbour checks used by both balancing approaches."""
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

# ---- Shared settings ------------------------------------------------------
MATRIX_PATH = r"C:\Users\micranne\OneDrive - DPDHL\Documents\Porcelanosa\Porject Swichback\Matix files\Matrix V1.csv"
SHEET_NAME = 0            # used only if the matrix is an Excel file

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
AREA = "Postcode Area"
DEPOT = "Depot"
EAST = "GRIDE_Ave"
NORTH = "GRIDN_Ave"

MAX_DAYS = 3              # maximum number of days in a pattern
NEIGHBOURS = 4            # a district is "next to" its 4 nearest districts


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