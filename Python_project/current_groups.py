"""Current delivery matrix, grouped geographically into letter groups.
Each group's pattern is the days it currently receives. No optimisation."""
from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans

from matrix_common import MATRIX_PATH, AREA, DAYS, DEPOT, EAST, NORTH, load_matrix

# ---- Settings -------------------------------------------------------------
DEPOT_NAME = "Watford"     # a depot name, or None = every depot
GROUP_SIZE = 3.5           # about this many letters per group (63 letters -> 18 groups)
MIN_DAY_SHARE = 0.05       # a day is in a group's pattern if it carries at least this share of its kg
OUTPUT_DIR = Path(r"C:\Users\micranne\OneDrive - DPDHL\Documents\Porcelanosa\project_swichback\Matix files\Output")
RUN_STAMP = datetime.now().strftime("%Y-%m-%d_%H-%M")


def group_depot(d):
    d = d[d["Weight"] > 0].copy()
    d["Letters"] = d[AREA].str.extract(r"^([A-Z]+)")[0]
    d["wx"] = d[EAST] * d["Weight"]
    d["wy"] = d[NORTH] * d["Weight"]

    # Each letter's kg-weighted centre
    lc = d.groupby("Letters").agg(kg=("Weight", "sum"), wx=("wx", "sum"),
                                  wy=("wy", "sum"), districts=(AREA, "count"))
    lc["x"] = lc["wx"] / lc["kg"]
    lc["y"] = lc["wy"] / lc["kg"]

    # Cluster letters into geographic groups, weighted by kg
    R = max(1, round(len(lc) / GROUP_SIZE))
    km = KMeans(n_clusters=R, n_init=10, random_state=0)
    km.fit(lc[["x", "y"]].to_numpy(), sample_weight=lc["kg"].to_numpy())

    rows = []
    for r in range(R):
        members = list(lc.index[km.labels_ == r])
        if not members:
            continue
        g = d[d["Letters"].isin(members)]
        day_kg = g[DAYS].sum()
        share = day_kg / day_kg.sum()
        pattern = [day for day in DAYS if share[day] >= MIN_DAY_SHARE]
        rows.append({
            "Letters": ", ".join(sorted(members)),
            "Pattern": ", ".join(pattern),
            "Districts": int(len(g)),
            "Weight kg": int(round(g["Weight"].sum())),
        })

    out = (pd.DataFrame(rows)
             .sort_values("Weight kg", ascending=False)
             .reset_index(drop=True))
    out.insert(0, "Group", range(1, len(out) + 1))
    return out


def main():
    df = load_matrix(MATRIX_PATH)
    depots = sorted(df[DEPOT].unique()) if DEPOT_NAME is None else [DEPOT_NAME]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    out = OUTPUT_DIR / f"Current_groups_{RUN_STAMP}.xlsx"
    with pd.ExcelWriter(out) as xl:
        for depot in depots:
            g = group_depot(df[df[DEPOT] == depot])
            print(f"\n{depot}: {len(g)} groups")
            print(g.to_string(index=False))
            g.to_excel(xl, sheet_name=depot[:31], index=False)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()