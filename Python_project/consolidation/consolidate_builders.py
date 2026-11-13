"""Consolidate builder orders: same builder, same full postcode, same ISO week.
Unmatched orders and orders in EXCLUDE_STATES pass through unchanged."""
from pathlib import Path

import pandas as pd

# ---- Settings -------------------------------------------------------------
RAW_PATH = Path(r"C:\Users\micranne\OneDrive - DPDHL\Documents\Porcelanosa\project_swichback\Python_project\Data\aug_oct_orders.csv")
KEYWORD_PATH = Path(r"C:\Users\micranne\OneDrive - DPDHL\Documents\Porcelanosa\project_swichback\Python_project\consolidation\builder_keywords.csv")
OUTPUT_PATH = Path(r"C:\Users\micranne\OneDrive - DPDHL\Documents\Porcelanosa\project_swichback\Python_project\Data\Consolidation_output\raw_orders_consolidated.csv")
SUMMARY_PATH = Path(r"C:\Users\micranne\OneDrive - DPDHL\Documents\Porcelanosa\project_swichback\Python_project\Data\Consolidation_output\consolidation_summary.csv")

# Columns searched for builder keywords (a match in any of them counts)
MATCH_COLUMNS = ["dropoffContactName", "dropoffCustomerReference",
                 "customerReference", "internalComment", "reference"]
POSTCODE_COL = "dropoffPostCode"
DATE_COL = "date"                  # yyyymmdd
WEIGHT_COL = "Weight"
PALLET_COL = "Number of pallets"
STATE_COL = "stateText"
EXCLUDE_STATES = ["Cancelled"]     # never consolidated, passed through unchanged


# ---- Helpers --------------------------------------------------------------
def to_num(s):
    return pd.to_numeric(s.str.replace(",", "", regex=False).str.strip(),
                         errors="coerce").fillna(0.0)


def fmt(x):
    return str(int(x)) if float(x).is_integer() else f"{x:.2f}"


def load_keywords(path):
    kw = pd.read_csv(path, dtype=str, keep_default_na=False)
    kw["Keyword"] = kw["Keyword"].str.strip().str.lower()
    kw["Builder"] = kw["Builder"].str.strip()
    kw = kw[kw["Keyword"] != ""]
    # Longest keyword first, so "barratt redrow" is tried before "redrow"
    kw = kw.sort_values("Keyword", key=lambda s: s.str.len(), ascending=False)
    return list(zip(kw["Keyword"], kw["Builder"]))


def match_builder(text, pairs):
    t = text.lower()
    for keyword, builder in pairs:
        if keyword in t:
            return builder
    return ""


# ---- Main -----------------------------------------------------------------
def main():
    raw = pd.read_csv(RAW_PATH, dtype=str, keep_default_na=False)

    required = MATCH_COLUMNS + [POSTCODE_COL, DATE_COL, WEIGHT_COL,
                                PALLET_COL, STATE_COL]
    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise SystemExit(f"Missing columns in raw file: {missing}")

    pairs = load_keywords(KEYWORD_PATH)

    # Helper frame: the raw file itself is only edited at the kept rows
    work = pd.DataFrame(index=raw.index)
    text = raw[MATCH_COLUMNS].agg(" | ".join, axis=1)
    work["builder"] = text.map(lambda t: match_builder(t, pairs))
    work["pc"] = raw[POSTCODE_COL].str.upper().str.replace(r"\s+", "", regex=True)

    dates = pd.to_datetime(raw[DATE_COL].str.strip(), format="%Y%m%d", errors="coerce")
    iso = dates.dt.isocalendar()
    work["week"] = iso["year"].astype(str) + "-W" + iso["week"].astype(str).str.zfill(2)

    work["kg"] = to_num(raw[WEIGHT_COL])
    work["pallets"] = to_num(raw[PALLET_COL])
    state = raw[STATE_COL].str.strip()

    eligible = ((work["builder"] != "") & (work["pc"] != "")
                & dates.notna() & ~state.isin(EXCLUDE_STATES))

    # Group matched, eligible orders by builder, full postcode and ISO week
    drop_idx, groups_merged = [], 0
    for _, g in work[eligible].groupby(["builder", "pc", "week"]):
        if len(g) < 2:
            continue
        groups_merged += 1
        keep = g["kg"].idxmax()                      # largest order by kg
        raw.loc[keep, WEIGHT_COL] = fmt(g["kg"].sum())
        raw.loc[keep, PALLET_COL] = fmt(g["pallets"].sum())
        drop_idx.extend(g.index.drop(keep))

    dropped = work.loc[drop_idx]
    out = raw.drop(index=drop_idx)
    out.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    # Summary by builder
    matched = work[work["builder"] != ""]
    orders_in = matched.groupby("builder").size().rename("Orders in")
    merged = dropped.groupby("builder").agg(
        **{"Orders merged": ("kg", "size"),
           "kg moved": ("kg", "sum"),
           "Pallets moved": ("pallets", "sum")})
    summary = pd.concat([orders_in, merged], axis=1).fillna(0)
    summary["Orders out"] = summary["Orders in"] - summary["Orders merged"]
    summary[["kg moved", "Pallets moved"]] = summary[["kg moved", "Pallets moved"]].round(0)
    summary.index.name = "Builder"
    summary = summary.reset_index()
    summary = summary[["Builder", "Orders in", "Orders merged", "Orders out",
                       "kg moved", "Pallets moved"]]
    summary.to_csv(SUMMARY_PATH, index=False, encoding="utf-8-sig")

    print(summary.to_string(index=False))
    print()
    print(f"Raw orders in:            {len(raw)}")
    print(f"Builder orders matched:   {len(matched)}")
    print(f"Unmatched (unchanged):    {(work['builder'] == '').sum()}")
    print(f"Excluded by state:        {state.isin(EXCLUDE_STATES).sum()}")
    print(f"Groups consolidated:      {groups_merged}")
    print(f"Orders out:               {len(out)}")
    print(f"\nSaved: {OUTPUT_PATH}\nSaved: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()