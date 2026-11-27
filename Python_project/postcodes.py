"""Delivery points: each order's own postcode, from the ONS UPRN Directory (ONSUD).

Without this, every order in a postcode district sits on the district's average grid
reference, so ten SA1 orders cost 0 km between them. With it, each order is a stop at
its full postcode's position, and routing.vehicle_route measures the km between them.

Lookup file
  ONSUD_DIR (.env): the folder holding the ONSUD csv files (ONSUD_E128_SEP_2026_NE.csv ...,
  one per region; any sub-folder is searched). Only three columns are read: PCDS (postcode),
  GRIDGB1E and GRIDGB1N (grid reference, metres). Each postcode's point is the average of its
  addresses. The result is saved once as postcode_grid.csv (POSTCODE_GRID in .env, else
  ONSUD_DIR/postcode_grid.csv, about 1.7m rows) and read from there on later runs. Delete it to
  rebuild after a new ONSUD release.
  Neither set: every order stays on its district centre, as before.

Stops
  A stop key is "<district>|<postcode>", e.g. "SA1|SA1 1AA", so the district it belongs to is
  always known (routing.area_of). An order falls back to its district centre (stop key = the
  district itself) when: it has no postcode, the postcode is not in the lookup (e.g. Northern
  Ireland, new builds), or the postcode point is more than [routes] postcode_max_km_from_area
  from its district centre (a typo or mismatched district). The run prints how many did.
"""
import os
from pathlib import Path

import pandas as pd

from settings import SETTINGS

COLS = ["PCDS", "GRIDGB1E", "GRIDGB1N"]
MAX_KM_FROM_AREA = SETTINGS.get("routes", {}).get("postcode_max_km_from_area", 30.0)
CHUNK_ROWS = 2_000_000


def env_path(name):
    v = (os.getenv(name) or "").strip().strip('"').strip("'")
    return Path(v) if v else None


def norm_postcode(s):
    """'sa11aa', 'SA1  1AA' -> 'SA1 1AA'. Empty or too short -> ''."""
    s = s.astype(str).str.upper().str.replace(r"[^A-Z0-9]", "", regex=True)
    ok = s.str.len().between(5, 7)
    return (s.str[:-3] + " " + s.str[-3:]).where(ok, "")


def grid_path():
    p = env_path("POSTCODE_GRID")
    if p:
        return p
    d = env_path("ONSUD_DIR")
    return d / "postcode_grid.csv" if d else None


def build_grid(onsud_dir, out):
    """Average grid reference per postcode over every ONSUD csv in onsud_dir. Writes out."""
    files = sorted(f for f in Path(onsud_dir).rglob("*.csv")
                   if f.name.upper().startswith("ONSUD") and f.resolve() != Path(out).resolve())
    if not files:
        raise SystemExit(f"No ONSUD*.csv files under ONSUD_DIR: {onsud_dir}")
    print(f"Building the postcode lookup from {len(files)} ONSUD file(s); this runs once and takes a few minutes")
    parts = []
    for f in files:
        rows = 0
        for chunk in pd.read_csv(f, usecols=lambda c: c.strip().upper() in COLS, dtype=str,
                                 chunksize=CHUNK_ROWS, keep_default_na=False, encoding="utf-8-sig"):
            chunk.columns = [c.strip().upper() for c in chunk.columns]
            pc = norm_postcode(chunk["PCDS"])
            e = pd.to_numeric(chunk["GRIDGB1E"], errors="coerce")
            n = pd.to_numeric(chunk["GRIDGB1N"], errors="coerce")
            ok = (pc != "") & (e > 0) & (n > 0)
            g = pd.DataFrame({"pc": pc[ok], "e": e[ok], "n": n[ok]}).groupby("pc").agg(
                e=("e", "sum"), n=("n", "sum"), k=("e", "size"))
            parts.append(g)
            rows += len(chunk)
        print(f"  {f.name}: {rows:,} addresses")
    allg = pd.concat(parts).groupby(level=0).sum()
    out_df = pd.DataFrame({"Postcode": allg.index,
                           "Easting": (allg["e"] / allg["k"]).round(1),
                           "Northing": (allg["n"] / allg["k"]).round(1),
                           "Addresses": allg["k"].astype(int)})
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out, index=False)
    print(f"Saved postcode lookup: {out} ({len(out_df):,} postcodes)")
    return out_df


def load_grid():
    """Postcode -> (x, y) km, or {} when no lookup is set up."""
    out = grid_path()
    if out is None:
        return {}
    if not out.exists():
        d = env_path("ONSUD_DIR")
        if d is None or not d.exists():
            print(f"Postcode lookup {out} not found and ONSUD_DIR is not a folder: orders stay on district centres")
            return {}
        df = build_grid(d, out)
    else:
        df = pd.read_csv(out, dtype={"Postcode": str})
    return {p: (e / 1000, n / 1000) for p, e, n in zip(df["Postcode"], df["Easting"], df["Northing"])}


_GRID = None


def grid():
    global _GRID
    if _GRID is None:
        _GRID = load_grid()
    return _GRID


def add_stops(orders, coords, label="orders"):
    """Add a "stop" column to orders and each stop's position to coords (in place). Returns orders.

    coords: area -> (x, y) km, as from routing.load_grid_refs; stop points are added to it.
    """
    g = grid()
    o = orders
    if not g or "postcode" not in o.columns:
        if g and "postcode" not in o.columns:
            print(f"{label}: no dropoffPostCode column, so orders stay on district centres")
        o["stop"] = o["area"]
        return o
    pc = o["postcode"]
    stops = o["area"].copy()
    counts = {"no postcode": 0, "not in lookup": 0, f"over {MAX_KM_FROM_AREA:g} km from district": 0}
    max2 = MAX_KM_FROM_AREA ** 2
    for i, area, p in zip(o.index, o["area"], pc):
        if not p:
            counts["no postcode"] += 1
            continue
        pt = g.get(p)
        if pt is None:
            counts["not in lookup"] += 1
            continue
        c = coords.get(area)
        if c is not None and (pt[0] - c[0]) ** 2 + (pt[1] - c[1]) ** 2 > max2:
            counts[f"over {MAX_KM_FROM_AREA:g} km from district"] += 1
            continue
        key = f"{area}|{p}"
        coords[key] = pt
        stops.at[i] = key
    o["stop"] = stops
    fb = sum(counts.values())
    detail = ", ".join(f"{k} {v:,}" for k, v in counts.items() if v)
    print(f"{label}: {len(o) - fb:,} of {len(o):,} orders placed at their own postcode"
          + (f"; {fb:,} on the district centre ({detail})" if fb else ""))
    return o


def fallback_counts(orders):
    """(orders at their own postcode, orders on a district centre)."""
    if "stop" not in orders.columns:
        return 0, len(orders)
    own = orders["stop"].str.contains("|", regex=False)
    return int(own.sum()), int((~own).sum())
