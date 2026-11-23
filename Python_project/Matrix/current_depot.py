import os
import shutil
from pathlib import Path
from datetime import datetime
import json

import pandas as pd
from dotenv import load_dotenv

load_dotenv()
ORDERS_PATH = Path(os.environ["ORDERS_PATH"])
DEPOT_PATH = Path(os.environ["DEPOT_PATH"])
TEAM_TO_DEPOT = json.loads(os.environ["TEAM_TO_DEPOT"])

STAMP = datetime.now().strftime("%Y-%m-%d_%H-%M")

AREA, DEPOT, EAST, NORTH = "Postcode Area", "Depot", "GRIDE_Ave", "GRIDN_Ave"
TEAM, STATE = "teamNames", "stateText"


def read_csv_any(path):
    for enc in ("utf-8-sig", "cp1252"):
        try:
            return pd.read_csv(path, dtype=str, keep_default_na=False, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="latin-1")


def to_num(s):
    return pd.to_numeric(s.str.replace(",", "", regex=False).str.strip(),
                         errors="coerce").fillna(0.0)


def main():
    # 1. Depot per postcode area from the orders
    orders = read_csv_any(ORDERS_PATH)
    orders = orders[~orders[STATE].str.strip().eq("Cancelled")].copy()
    orders[AREA] = orders[AREA].str.strip().str.upper()
    code = orders[TEAM].str.extract(r"(\d{7})", expand=False)
    orders["Orders depot"] = code.map(TEAM_TO_DEPOT)

    unmapped = orders.loc[orders["Orders depot"].isna(), TEAM].value_counts()
    if not unmapped.empty:
        print("Team names not in TEAM_TO_DEPOT (ignored):")
        print(unmapped.to_string())

    counts = (orders.dropna(subset=["Orders depot"])
              .groupby([AREA, "Orders depot"]).size().reset_index(name="orders"))
    counts["Orders share"] = (counts["orders"] /
                              counts.groupby(AREA)["orders"].transform("sum")).round(2)
    top = (counts.sort_values("orders", ascending=False)
           .drop_duplicates(AREA)[[AREA, "Orders depot", "Orders share"]])

    # 2. Existing helper: keep its grid references and previous depot, back it up
    if DEPOT_PATH.exists():
        old = read_csv_any(DEPOT_PATH)
        old[AREA] = old[AREA].str.strip().str.upper()
        old[DEPOT] = old[DEPOT].str.strip()
        for c in (EAST, NORTH):
            old[c] = to_num(old[c])
        backup = DEPOT_PATH.with_name(f"{DEPOT_PATH.stem}_backup_{STAMP}{DEPOT_PATH.suffix}")
        shutil.copy2(DEPOT_PATH, backup)
        print(f"Old helper backed up to {backup}")
        old = (old[[AREA, DEPOT, EAST, NORTH]].drop_duplicates(AREA)
               .rename(columns={DEPOT: "Previous depot"}))
    else:
        old = pd.DataFrame(columns=[AREA, "Previous depot", EAST, NORTH])

    # 3. Combine: orders decide the depot, the old helper supplies grid references
    out = old.merge(top, on=AREA, how="outer")
    out[DEPOT] = out["Orders depot"].fillna(out["Previous depot"]).fillna("")
    out["Changed"] = out[DEPOT] != out["Previous depot"].fillna("")
    for c in (EAST, NORTH):
        out[c] = out[c].fillna(0.0)

    no_grid = out[(out[EAST] <= 0) | (out[NORTH] <= 0)]
    print(f"\nAreas changing depot: {out['Changed'].sum()}")
    print(f"Areas with no grid reference (fill these before assign_depot.py): "
          f"{len(no_grid)}")
    if not no_grid.empty:
        print(no_grid[[AREA, DEPOT]].to_string(index=False))

    out = out[[AREA, DEPOT, EAST, NORTH, "Previous depot", "Changed", "Orders share"]]
    DEPOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(DEPOT_PATH, index=False, encoding="utf-8-sig")
    print(f"\nSaved: {DEPOT_PATH}")


if __name__ == "__main__":
    main()