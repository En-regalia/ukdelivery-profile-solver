"""Customer reporting workbook, built at the end of planning_estimate.py.

One workbook, Customer_report_<stamp>.xlsx, in the planning run folder:
  Customer summary   front sheet in plain words: before, after and change
  Summary            bottom, middle and top 10 weeks: average kg and pallets per weekday
  Origin split       weight each depot takes from other depots' regions
  Vehicle mix        vans and trucks per depot and band, and the order size table
  Fill               average vehicle fill per depot and weekday against the target
  Detail - ...       analyst working sheets behind the front sheets

Before = every order on its actual depot and day. After = every order on the depot
whose plan holds its postcode area, on the area's new day. Both are packed the same
way (planning_estimate.pack), so the comparison is like for like. Closing depots stay
in "before" as a comparator.

Settings: settings.toml [report], [cost], [data], [depots], [fleet], [vehicles].
Each step's own Excel output (assignment, anneal, sweep, Example_weeks) is unchanged.
"""
import calendar
import os
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from matrix_common import DAYS
from settings import (SETTINGS, SETTINGS_PATH, CLOSING_DEPOTS, LOCKED_DEPOTS,
                      fleet_caps, is_active)

R = SETTINGS["report"]
C = SETTINGS["cost"]
PALLET = "Pallet network"
BANDS = ["Bottom 10", "Middle 10", "Top 10"]
ALL = "All depots"
TYPES = ["Van", "HGV", "26t"]
TRUCKS = ["HGV", "26t"]
TYPE_NAME = {"Van": "Vans", "HGV": "18t HGVs", "26t": "26t HGVs"}
FILL_TARGET = R["fill_target"]


# ---- Data views -------------------------------------------------------------------------
def depot_status(depot):
    if depot in CLOSING_DEPOTS:
        return "Closing (comparator)"
    if not is_active(depot):
        return "Not active"
    return "Active, locked areas" if depot in LOCKED_DEPOTS else "Active"


def order_view(O, plan):
    """Orders as delivered in one plan: Week, Depot, Day, area, kg, pal.

    Before: actual depot and day. After: new depot and day; pallet-network orders left out.
    """
    if plan == "Before":
        v = O.rename(columns={"depot": "Depot", "day": "Day"})
    else:
        v = O[O["new_day"] != PALLET].rename(columns={"new_depot": "Depot", "new_day": "Day"})
    return v[["Week", "Depot", "Day", "area", "kg", "pal"]]


def drop_table(V):
    """One row per drop: a postcode area on one vehicle (Week, Plan, Depot, Day, Vehicle, area, kg, pal, orders)."""
    rows = []
    for w, p, d, day, veh, loads in zip(V["Week"], V["Plan"], V["Depot"], V["Day"], V["Vehicle"], V["Loads"]):
        for area, (kg, pal, n) in loads.items():
            rows.append((w, p, d, day, veh, area, kg, pal, n))
    return pd.DataFrame(rows, columns=["Week", "Plan", "Depot", "Day", "Vehicle", "area", "kg", "pal", "orders"])


def old_patterns(orders_all):
    """area -> set of weekdays carrying at least old_pattern_min_share of its weekday kg (whole order file)."""
    wd = orders_all[orders_all["day"].isin(DAYS)]
    kg = wd.groupby(["area", "day"])["kg"].sum().unstack(fill_value=0.0)
    share = kg.div(kg.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    return {a: {d for d in kg.columns if share.at[a, d] >= R["old_pattern_min_share"]} for a in kg.index}


# ---- Measures -------------------------------------------------------------------------------
class Measures:
    """Service and efficiency measures for one scope (all depots or one depot) and plan."""

    def __init__(self, O, V, drops, D, patterns, n_weeks):
        self.O, self.V, self.drops, self.D = O, V, drops, D
        self.patterns, self.n_weeks = patterns, n_weeks

    def scope(self, plan, depot=None):
        ov = order_view(self.O, plan)
        v = self.V[self.V["Plan"] == plan]
        dr = self.drops[self.drops["Plan"] == plan]
        if depot is not None:
            ov, v, dr = ov[ov["Depot"] == depot], v[v["Depot"] == depot], dr[dr["Depot"] == depot]
        return ov, v, dr

    def get(self, plan, depot=None):
        ov, v, dr = self.scope(plan, depot)
        n_days = self.n_weeks * len(DAYS)
        m = {}
        m["kg_week"] = v["KG"].sum() / self.n_weeks
        m["pal_week"] = v["Pallets"].sum() / self.n_weeks
        m["days_covered"] = ov.groupby(["Week", "area"])["Day"].nunique().mean() if not ov.empty else np.nan
        in_old = [d in self.patterns.get(a, set()) for a, d in zip(ov["area"], ov["Day"])]
        m["stability"] = (ov.loc[in_old, "kg"].sum() / ov["kg"].sum()) if ov["kg"].sum() else np.nan
        m["veh_day"] = len(v) / n_days
        m["veh_week"] = len(v) / self.n_weeks
        tonnes = v["KG"].sum() / 1000
        m["km_tonne"] = v["Route km"].sum() / tonnes if tonnes else np.nan
        m["km_week"] = v["Route km"].sum() / self.n_weeks
        fill = np.maximum(v["KG fill %"], v["Pallet fill %"])
        m["fill"] = fill.mean() if len(v) else np.nan
        m["kg_fill"] = v["KG fill %"].mean() if len(v) else np.nan
        m["pal_fill"] = v["Pallet fill %"].mean() if len(v) else np.nan
        m["drops_veh"] = v["Drops"].mean() if len(v) else np.nan
        visits = dr.groupby(["Week", "Depot", "Day", "area"])["Vehicle"].nunique()
        m["multi_share"] = (visits > 1).mean() if len(visits) else np.nan
        m["dup_week"] = (visits - 1).clip(lower=0).sum() / self.n_weeks
        d = self.D if depot is None else self.D[self.D["Depot"] == depot]
        sfx = "before" if plan == "Before" else "after"
        m["fleet_overflow_kg_week"] = d[f"Over fleet {sfx} KG"].sum() / self.n_weeks
        m["routecap_overflow_kg_week"] = d[f"Over route cap {sfx} KG"].sum() / self.n_weeks
        m["actual_routes_day"] = d["Routes before"].sum() / n_days if plan == "Before" else np.nan
        return m


MEASURES = [  # key, label, kind (pct / num / int), decimals, plain explanation
    ("days_covered", "Delivery days per postcode area per week", "num", 2,
     "How many weekdays each postcode area receives a delivery, on average"),
    ("stability", "Volume delivered on one of the area's usual days", "pct", 1,
     "Share of kg delivered on a weekday the area already had in its pattern (service stability)"),
    ("veh_day", "Vehicles per day", "num", 1, "Average vehicles used per working day, all depots"),
    ("veh_week", "Vehicles per week", "num", 1, "Vehicle-days per week (one vehicle used on one day = 1)"),
    ("km_tonne", "Kilometres per tonne", "num", 1, "Route km driven per tonne delivered (lower is better)"),
    ("fill", "Average vehicle fill", "pct", 1,
     "Per vehicle, the higher of its kg fill and pallet fill; target shown below"),
    ("drops_veh", "Drops per vehicle", "num", 1, "Postcode areas visited per vehicle"),
    ("multi_share", "Area-days served by more than one vehicle", "pct", 1,
     "Share of postcode-area days where two or more vehicles visit the same area"),
    ("dup_week", "Duplicate vehicle visits per week", "num", 1,
     "Extra visits to an area on a day it is already served by another vehicle"),
    ("kg_week", "Weight delivered by vehicle per week (kg)", "int", 0, "Excludes the pallet network"),
    ("pal_week", "Pallets delivered by vehicle per week", "int", 0, "Excludes the pallet network"),
    ("km_week", "Route km per week", "int", 0, "Straight-line km, depot to stops and back"),
]


def change(before, after, kind=None):
    if before is None or after is None or pd.isna(before) or pd.isna(after):
        return None
    return after - before


# ---- Cost -------------------------------------------------------------------------------------
def working_days(year, month=None):
    hols = {date.fromisoformat(h) for h in C.get("bank_holidays", [])}
    months = [month] if month else range(1, 13)
    n = 0
    for m in months:
        for d in range(1, calendar.monthrange(year, m)[1] + 1):
            day = date(year, m, d)
            if day.weekday() < 5 and day not in hols:
                n += 1
    return n


def sample_dates(weeks):
    out = []
    for wk in weeks:
        y, w = wk.split("-W")
        out += [date.fromisocalendar(int(y), int(w), d) for d in range(1, 6)]
    return out


def sample_budget(weeks):
    """Budget for the sample's working days, or None if no budget is set. Returns (amount, method text)."""
    dates = sample_dates(weeks)
    year = int(C.get("budget_year") or dates[0].year)
    if C.get("mode", "annual") == "monthly":
        monthly = C.get("monthly", {})
        names = list(calendar.month_abbr)[1:]
        if not any(monthly.get(n, 0) for n in names):
            return None, "No monthly budget entered in settings.toml [cost.monthly]"
        total = sum(monthly.get(names[d.month - 1], 0) / working_days(year, d.month) for d in dates)
        return total, (f"Monthly budget: each of the {len(dates)} sample days gets its month's figure "
                       f"divided by that month's working days in {year}")
    annual = C.get("annual_total", 0)
    if not annual:
        return None, "No annual budget entered in settings.toml [cost] annual_total"
    wd = working_days(year)
    return annual / wd * len(dates), (f"Annual budget {C.get('currency', '£')}{annual:,.0f} / {wd} working days "
                                      f"in {year} x {len(dates)} working days in the sample")


def cost_rows(mb, ma, weeks):
    """Cost per kg and per pallet, before and after. After = budget scaled by vehicle-days and km."""
    budget, method = sample_budget(weeks)
    if budget is None:
        return None, method
    vs = C.get("vehicle_share", 0.7)
    ratio = (vs * (ma["veh_week"] / mb["veh_week"] if mb["veh_week"] else 1)
             + (1 - vs) * (ma["km_week"] / mb["km_week"] if mb["km_week"] else 1))
    n = len(weeks)
    cost_b, cost_a = budget, budget * ratio
    kg_b, kg_a = mb["kg_week"] * n, ma["kg_week"] * n
    pal_b, pal_a = mb["pal_week"] * n, ma["pal_week"] * n
    rows = {"cost_sample": (cost_b, cost_a),
            "cost_kg": (cost_b / kg_b if kg_b else np.nan, cost_a / kg_a if kg_a else np.nan),
            "cost_pallet": (cost_b / pal_b if pal_b else np.nan, cost_a / pal_a if pal_a else np.nan)}
    method += (f". After = before x ({vs:.0%} x vehicle-days ratio + {1 - vs:.0%} x km ratio) "
               f"= x {ratio:.3f}. An estimate, not a quote.")
    return rows, method


# ---- Fleet --------------------------------------------------------------------------------------
def fleet_tables(V, D, n_weeks):
    """Fleet allocation per active depot and type, with flags for review; plus the total fleet."""
    caps = fleet_caps()
    after = V[V["Plan"] == "After"]
    days = [(w, d) for w in sorted(V["Week"].unique()) for d in DAYS]
    rows = []
    depots = sorted(set(after["Depot"]) | {d for d in caps if is_active(d)})
    for depot in depots:
        if not is_active(depot):
            continue
        dd = D[D["Depot"] == depot]
        short_days = ((dd["Fleet unresolved after"] > 0) | (dd["Over fleet after orders"] > 0)).sum()
        short_share = short_days / len(days) if days else 0
        for t in TYPES:
            cap = caps.get(depot, {}).get(t)
            used = (after[(after["Depot"] == depot) & (after["Type"] == t)]
                    .groupby(["Week", "Day"]).size().reindex(days, fill_value=0))
            if cap is None and used.sum() == 0:
                continue
            q = float(used.quantile(R["allocation_spare_pct"]))
            spare = (cap - int(np.ceil(q))) if cap is not None else None
            if cap is None:
                flag = "No cap set"
            elif t in TRUCKS and short_share >= R["allocation_short_share"]:
                flag = f"Short on {short_share:.0%} of days: consider adding"
            elif spare is not None and spare >= 1 and (t != "Van" or cap > 0):
                flag = f"Spare: {spare} could be released"
            else:
                flag = "Within allocation"
            rows.append({"Depot": depot, "Vehicle type": TYPE_NAME[t], "Cap per day": cap,
                         "Average used": round(used.mean(), 2), f"P{int(R['allocation_spare_pct'] * 100)} used": round(q, 1),
                         "Most used": int(used.max()), "Days at cap %": (used >= cap).mean() if cap else None,
                         "Days short of fleet %": short_share if t in TRUCKS else None,
                         "Flag (for review, not applied)": flag})
    alloc = pd.DataFrame(rows)

    # Total fleet: sum of caps over active depots (closing excluded); before comparator includes them
    tot = []
    before = V[V["Plan"] == "Before"]
    for t in TYPES:
        cap_after = sum(caps.get(d, {}).get(t, 0) for d in caps if is_active(d))
        before_depots = sorted(before["Depot"].unique())
        uncapped = [d for d in before_depots if caps.get(d, {}).get(t) is None]
        cap_before = sum(caps.get(d, {}).get(t, 0) for d in before_depots)
        ub = before[before["Type"] == t].groupby(["Week", "Day"]).size().reindex(days, fill_value=0)
        ua = after[after["Type"] == t].groupby(["Week", "Day"]).size().reindex(days, fill_value=0)
        tot.append({"Vehicle type": TYPE_NAME[t],
                    "Fleet before (depots operating)": cap_before,
                    "No cap set before": ", ".join(uncapped),
                    "Fleet after (active depots)": cap_after,
                    "Average used before": round(ub.mean(), 2), "Most used before": int(ub.max()),
                    "Average used after": round(ua.mean(), 2), "Most used after": int(ua.max())})
    return alloc, pd.DataFrame(tot)


# ---- Workbook helpers -----------------------------------------------------------------------------
HEAD_FILL = PatternFill("solid", fgColor="1F4E79")
SUB_FILL = PatternFill("solid", fgColor="D9E2F3")
NOTE_FONT = Font(italic=True, color="595959", size=9)


class Sheet:
    """Writes titled tables down an openpyxl sheet."""

    def __init__(self, wb, title, tab=None, first=False):
        self.ws = wb.active if first else wb.create_sheet()
        self.ws.title = title[:31]
        if tab:
            self.ws.sheet_properties.tabColor = tab
        self.row = 1
        self.widths = {}

    def title(self, text, size=14):
        c = self.ws.cell(self.row, 1, text)
        c.font = Font(bold=True, size=size, color="1F4E79")
        self.row += 2 if size >= 14 else 1

    def text(self, text, italic=False, bold=False):
        c = self.ws.cell(self.row, 1, text)
        c.font = NOTE_FONT if italic else Font(bold=bold)
        c.alignment = Alignment(wrap_text=False)
        self.row += 1

    def gap(self, n=1):
        self.row += n

    def table(self, df, pct=(), num1=(), num2=(), money=(), heading=None):
        if heading:
            c = self.ws.cell(self.row, 1, heading)
            c.font = Font(bold=True, size=11)
            self.row += 1
        if df is None or df.empty:
            self.text("No data", italic=True)
            self.gap()
            return
        for j, col in enumerate(df.columns, start=1):
            c = self.ws.cell(self.row, j, str(col))
            c.font = Font(bold=True, color="FFFFFF")
            c.fill = HEAD_FILL
            c.alignment = Alignment(wrap_text=True, vertical="top")
            self.widths[j] = max(self.widths.get(j, 8), min(len(str(col)) * 0.9 + 2, 30))
        self.row += 1
        cur = money_fmt()
        for rec in df.itertuples(index=False):
            for j, (col, val) in enumerate(zip(df.columns, rec), start=1):
                if isinstance(val, (np.floating, float)) and np.isnan(val):
                    val = None
                if isinstance(val, np.generic):
                    val = val.item()
                c = self.ws.cell(self.row, j, val)
                if col in pct:
                    c.number_format = "0.0%"
                elif col in money:
                    c.number_format = cur
                elif col in num2:
                    c.number_format = "#,##0.00"
                elif col in num1:
                    c.number_format = "#,##0.0"
                elif isinstance(val, (int, float)):
                    c.number_format = "#,##0"
                if isinstance(val, str):
                    self.widths[j] = max(self.widths.get(j, 8), min(len(val) + 2, 70))
                else:
                    self.widths[j] = max(self.widths.get(j, 8), 11)
            self.row += 1
        self.gap()

    def finish(self, freeze=None):
        for j, w in self.widths.items():
            self.ws.column_dimensions[get_column_letter(j)].width = w
        if freeze:
            self.ws.freeze_panes = freeze


def money_fmt():
    cur = C.get("currency", "£")
    return f'"{cur}"#,##0.000'


# ---- Sheets ----------------------------------------------------------------------------------------
def band_summary(O, band_of):
    """Average kg and pallets per weekday, by band, depot and plan."""
    rows = []
    for plan in ["Before", "After"]:
        ov = order_view(O, plan)
        if plan == "After":   # show the pallet network as its own line
            pn = O[O["new_day"] == PALLET].rename(columns={"depot": "Depot", "day": "Day"})
            pn = pn.assign(Depot=PALLET)[["Week", "Depot", "Day", "area", "kg", "pal"]]
            ov = pd.concat([ov, pn])
        ov = ov.assign(Band=ov["Week"].map(band_of))
        for band in BANDS:
            b = ov[ov["Band"] == band]
            n = b["Week"].nunique() or 1
            depots = [ALL] + sorted(d for d in b["Depot"].unique() if d != PALLET) + \
                     ([PALLET] if (b["Depot"] == PALLET).any() else [])
            for depot in depots:
                g = b if depot == ALL else b[b["Depot"] == depot]
                if depot == ALL:
                    g = g[g["Depot"] != PALLET]
                for col, label in [("kg", "Weight kg"), ("pal", "Pallets")]:
                    per = g.groupby("Day")[col].sum().reindex(DAYS, fill_value=0) / n
                    rows.append({"Band": band, "Weeks": n, "Depot": depot, "Plan": plan, "Measure": label,
                                 **{d: round(per[d], 0 if col == "kg" else 1) for d in DAYS},
                                 "Week total": round(per.sum(), 0 if col == "kg" else 1)})
    df = pd.DataFrame(rows)
    order = df.assign(_b=df["Band"].map(BANDS.index),
                      _d=df["Depot"].map(lambda d: 0 if d == ALL else 2 if d == PALLET else 1),
                      _p=df["Plan"].map({"Before": 0, "After": 1}))
    return (order.sort_values(["_b", "Measure", "_d", "Depot", "_p"], kind="stable")
            .drop(columns=["_b", "_d", "_p"]))


def origin_split(O, n_weeks):
    """Weight each depot receives from other depots' regions, as % of the originating depot's volume."""
    origin_total = O.groupby("depot")["kg"].sum()
    moved = O[(O["new_depot"] != O["depot"]) & (O["new_depot"] != PALLET)]
    g = (moved.groupby(["new_depot", "depot"]).agg(kg=("kg", "sum"), pal=("pal", "sum"), orders=("kg", "size"))
         .reset_index())
    t1 = pd.DataFrame({
        "Receiving depot": g["new_depot"], "From depot (origin)": g["depot"],
        "Origin status": g["depot"].map(depot_status),
        "KG per week": (g["kg"] / n_weeks).round(0), "Pallets per week": (g["pal"] / n_weeks).round(1),
        "Orders per week": (g["orders"] / n_weeks).round(1),
        "Origin depot KG per week": (g["depot"].map(origin_total) / n_weeks).round(0),
        "% of origin depot's volume": g["kg"] / g["depot"].map(origin_total)})
    t1 = t1.sort_values(["Receiving depot", "KG per week"], ascending=[True, False])

    # Where each depot's volume goes (rows: origin; columns: destination), % of origin volume
    dest = O["new_depot"].where(O["new_depot"] != PALLET, PALLET)
    ct = O.assign(dest=dest).pivot_table(index="depot", columns="dest", values="kg", aggfunc="sum", fill_value=0)
    cols = sorted(c for c in ct.columns if c != PALLET) + ([PALLET] if PALLET in ct.columns else [])
    ct = ct[cols]
    pct = ct.div(ct.sum(axis=1), axis=0)
    t2 = pct.reset_index().rename(columns={"depot": "From depot"})
    t2.insert(1, "KG per week", (ct.sum(axis=1).values / n_weeks).round(0))
    return t1, t2, cols


def vehicle_mix(V, band_of):
    rows = []
    Vb = V.assign(Band=V["Week"].map(band_of))
    for band in BANDS:
        b = Vb[Vb["Band"] == band]
        n = b["Week"].nunique() or 1
        for depot in [ALL] + sorted(b["Depot"].unique()):
            g = b if depot == ALL else b[b["Depot"] == depot]
            for plan in ["Before", "After"]:
                p = g[g["Plan"] == plan]
                rec = {"Band": band, "Depot": depot, "Plan": plan}
                for t in TYPES:
                    q = p[p["Type"] == t]
                    rec[f"{TYPE_NAME[t]} per week"] = round(len(q) / n, 1)
                for t in TYPES:
                    q = p[p["Type"] == t]
                    rec[f"{TYPE_NAME[t]} kg per week"] = round(q["KG"].sum() / n)
                    rec[f"{TYPE_NAME[t]} pallets per week"] = round(q["Pallets"].sum() / n, 1)
                rows.append(rec)
    return pd.DataFrame(rows)


def size_table(O, drops, n_weeks):
    """Order sizes, and drop sizes before and after: how condensing changes small deliveries."""
    edges = [0] + list(R["order_size_bands_kg"]) + [np.inf]
    labels = [f"{int(a):,} to {int(b):,} kg" if b != np.inf else f"Over {int(a):,} kg"
              for a, b in zip(edges, edges[1:])]
    labels[0] = f"Up to {int(edges[1]):,} kg"

    def bands(kg):
        return pd.cut(kg, edges, labels=labels, right=True, include_lowest=True)

    t = pd.DataFrame({"Size (kg)": labels})
    o = O[O["day"].isin(DAYS)]
    oc = bands(o["kg"]).value_counts().reindex(labels, fill_value=0)
    t["Orders per week"] = (oc.values / n_weeks).round(1)
    t["Orders %"] = oc.values / max(oc.sum(), 1)
    for plan in ["Before", "After"]:
        d = drops[drops["Plan"] == plan]
        dc = bands(d["kg"]).value_counts().reindex(labels, fill_value=0)
        t[f"Drops per week {plan.lower()}"] = (dc.values / n_weeks).round(1)
        t[f"Drops % {plan.lower()}"] = dc.values / max(dc.sum(), 1)
    small = R["small_order_kg"]
    summary = []
    for plan in ["Before", "After"]:
        d = drops[drops["Plan"] == plan]
        summary.append({"Plan": plan, "Drops per week": round(len(d) / n_weeks, 1),
                        "Orders per drop": round(d["orders"].sum() / len(d), 2) if len(d) else None,
                        "Average drop kg": round(d["kg"].mean(), 0) if len(d) else None,
                        f"Drops up to {small:,} kg %": (d["kg"] <= small).mean() if len(d) else None})
    return t, pd.DataFrame(summary)


def fill_table(V):
    rows = []
    V = V.assign(Fill=np.maximum(V["KG fill %"], V["Pallet fill %"]))
    for depot in [ALL] + sorted(V["Depot"].unique()):
        g = V if depot == ALL else V[V["Depot"] == depot]
        for plan in ["Before", "After"]:
            p = g[g["Plan"] == plan]
            if p.empty:
                continue
            per = p.groupby("Day")["Fill"].mean().reindex(DAYS)
            avg = p["Fill"].mean()
            rows.append({"Depot": "National average" if depot == ALL else depot, "Plan": plan,
                         **{d: per[d] for d in DAYS}, "Average fill": avg,
                         "KG fill": p["KG fill %"].mean(), "Pallet fill": p["Pallet fill %"].mean(),
                         "Target": FILL_TARGET, "Gap to target": avg - FILL_TARGET,
                         "Vehicles": len(p)})
    return pd.DataFrame(rows)


def step_outputs(plan_source):
    """Key figures from each step's own output, for the detail sheet. Missing files are skipped."""
    out = {}
    assigned = (os.getenv("DEPOT_ASSIGNED") or "").strip().strip('"')
    if assigned and Path(assigned).exists():
        a = pd.read_csv(assigned, dtype=str, keep_default_na=False, encoding="utf-8-sig")
        if {"Current depot", "Depot"} <= set(a.columns):
            ct = pd.crosstab(a["Current depot"].replace("", "(none)"), a["Depot"])
            out["Depot assignment: postcode areas, current depot (rows) to new depot (columns)"] = \
                ct.reset_index().rename(columns={"Current depot": "Current depot"})
            if "Review" in a.columns:
                rv = a[a["Review"].str.strip() != ""]
                if not rv.empty:
                    out["Depot assignment: areas flagged for review"] = rv[
                        [c for c in ["Postcode Area", "Current depot", "Depot", "Km to depot", "Review"] if c in a.columns]]
    src = Path(plan_source) if plan_source else None
    if src and src.exists():
        files = [src] if src.is_file() else sorted(src.glob("Anneal_*_Regional_*.xlsx"))
        summ = []
        for f in files:
            try:
                s = pd.read_excel(f, sheet_name="Summary")
                s.insert(0, "File", f.name)
                summ.append(s)
            except Exception as e:   # a damaged file should not stop the report
                print(f"Could not read {f.name} Summary: {e}")
        if summ:
            out["Day plan (anneal or sweep) per depot: Summary sheets"] = pd.concat(summ, ignore_index=True)
        folder = src if src.is_dir() else src.parent
        for f in sorted(folder.glob("Sweep_comparison_*.xlsx"))[-1:]:
            out[f"Sweep comparison ({f.name})"] = pd.read_excel(f, sheet_name="Summary")
    return out


def settings_rows(d=None, prefix=""):
    d = SETTINGS if d is None else d
    rows = []
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict) and not (prefix.startswith("fleet") or prefix.startswith("vehicles")):
            rows += settings_rows(v, key + ".")
        else:
            rows.append({"Setting": key, "Value": str(v)})
    return rows


# ---- Build ---------------------------------------------------------------------------------------------
def build(run_dir, stamp, band_of, vehicles, orders_sim, daily, orders_all, plan_source):
    """Write Customer_report_<stamp>.xlsx. Returns its path."""
    V = pd.DataFrame(vehicles)
    O = orders_sim.copy()
    D = pd.DataFrame(daily)
    weeks = sorted(band_of)
    n_weeks = len(weeks)
    drops = drop_table(V)
    patterns = old_patterns(orders_all)
    M = Measures(O, V, drops, D, patterns, n_weeks)
    mb, ma = M.get("Before"), M.get("After")
    cost, cost_method = cost_rows(mb, ma, weeks)
    alloc, total_fleet = fleet_tables(V, D, n_weeks)
    t_origin, t_dest, dest_cols = origin_split(O, n_weeks)

    wb = Workbook()

    # ---- 1. Customer summary (front) ----
    s = Sheet(wb, "Customer summary", tab="2E7D32", first=True)
    s.title("Delivery plan: before and after")
    s.text(R["scenario"])
    dates = sample_dates(weeks)
    s.text(f"Order data as at {SETTINGS['data']['source_date']}. "
           f"Sample: {n_weeks} weeks ({min(dates):%d %b %Y} to {max(dates):%d %b %Y}): the 10 lightest, "
           f"10 middle and 10 busiest weeks by weight.", italic=True)
    s.gap()

    rows = []
    for key, label, kind, dec, note in MEASURES:
        b, a = mb[key], ma[key]
        rows.append({"Measure": label, "Before": b, "After": a, "Change": change(b, a, kind), "What it means": note,
                     "_kind": kind})
    rows.insert(6, {"Measure": "Fill target", "Before": FILL_TARGET, "After": FILL_TARGET, "Change": None,
                    "What it means": "As high as possible; 90% is the working target", "_kind": "pct"})
    if cost:
        cur = C.get("currency", "£")
        for key, label in [("cost_kg", f"Cost per kg ({cur})"), ("cost_pallet", f"Cost per pallet ({cur})")]:
            b, a = cost[key]
            rows.append({"Measure": label, "Before": b, "After": a, "Change": a - b,
                         "What it means": "Estimate from the delivery budget; see Detail - Cost", "_kind": "money"})
    df = pd.DataFrame(rows)
    _write_measure_table(s, df, "Key measures (all depots)")

    # depot table
    caps = fleet_caps()
    flags = alloc.groupby("Depot")["Flag (for review, not applied)"].agg(
        lambda x: "; ".join(sorted({f for f in x if f != "Within allocation"})) or "Within allocation") \
        if not alloc.empty else pd.Series(dtype=str)
    drow = []
    for depot in sorted(set(V["Depot"])):
        b, a = M.get("Before", depot), M.get("After", depot)
        cap = caps.get(depot)
        drow.append({"Depot": depot, "Status": depot_status(depot),
                     "KG per week before": round(b["kg_week"]), "KG per week after": round(a["kg_week"]),
                     "Change %": (a["kg_week"] / b["kg_week"] - 1) if b["kg_week"] else None,
                     "Vehicles per day before": round(b["veh_day"], 1),
                     "Vehicles per day after": round(a["veh_day"], 1),
                     "Fleet per day": sum(cap.values()) if cap and is_active(depot) else None,
                     "Average fill after": a["fill"],
                     "Fleet allocation (for review)": flags.get(depot, "") if is_active(depot) else ""})
    s.table(pd.DataFrame(drow), pct=("Change %", "Average fill after"), num1=("Vehicles per day before", "Vehicles per day after"),
            heading="By depot")

    s.title("In plain words", size=11)
    for line in plain_words(M, mb, ma, t_dest, dest_cols, n_weeks, cost):
        s.text("• " + line)
    s.gap()
    s.title("How these figures are measured", size=11)
    for line in [
        "Before: each order on its current depot and delivery day. After: each order on its new depot and "
        "the postcode area's new day. Both are packed onto vehicles the same way, so the comparison is like for like.",
        "Visits, drops and days are measured by postcode area, not by delivery site. Two customers in the same "
        "postcode area on one vehicle count as one drop.",
        "Kilometres are straight-line distances, depot to stops and back. Road distances are longer; use the "
        "figures to compare before and after, not as mileage.",
        f"Fill is the higher of each vehicle's weight fill and pallet fill. Target {FILL_TARGET:.0%}.",
        f"A postcode area's usual days are the weekdays that carried at least {R['old_pattern_min_share']:.0%} "
        "of its weight across the order file.",
        "Closing depots appear in 'before' only, as a comparator. Volume to the pallet network is not on vehicles.",
        "Detail sheets behind this one are working figures for the planning team.",
    ]:
        s.text(line, italic=True)
    s.ws.column_dimensions["A"].width = 48
    s.finish()

    # ---- 2. Summary by band ----
    s = Sheet(wb, "Summary", tab="1F4E79")
    s.title("Average weight and pallets per weekday: lightest, middle and busiest 10 weeks")
    s.text("Averages per week within each band. After = new depot and new day; the pallet network is shown "
           "as its own line and is not in 'All depots'.", italic=True)
    s.gap()
    s.table(band_summary(O, band_of))
    s.finish(freeze="A4")

    # ---- 3. Origin split ----
    s = Sheet(wb, "Origin split", tab="1F4E79")
    s.title("Weight taken from other regions")
    s.text("Per receiving depot: weight it takes on from other depots' current areas, as a share of the "
           "originating depot's total volume (sample weeks, weekdays).", italic=True)
    s.gap()
    s.table(t_origin, pct=("% of origin depot's volume",), heading="By receiving depot")
    s.table(t_dest, pct=tuple(dest_cols), heading="Where each depot's current volume goes (share of its volume)")
    s.finish()

    # ---- 4. Vehicle mix ----
    s = Sheet(wb, "Vehicle mix", tab="1F4E79")
    s.title("Vans and HGVs per depot, before and after")
    s.text("Average per week in each band. 18t HGVs are upgraded to 26t only when that saves a truck.", italic=True)
    s.gap()
    s.table(vehicle_mix(V, band_of))
    sizes, size_sum = size_table(O, drops, n_weeks)
    s.title("Order sizes, and how condensing changes them", size=11)
    s.text("Orders are unchanged by the plan. A drop is all of one postcode area's orders on one vehicle: "
           "with fewer delivery days, small orders combine into larger drops.", italic=True)
    s.table(sizes, pct=("Orders %", "Drops % before", "Drops % after"))
    s.table(size_sum, pct=tuple(c for c in size_sum.columns if c.endswith("%")), num2=("Orders per drop",))
    s.finish()

    # ---- 5. Fill ----
    s = Sheet(wb, "Fill", tab="1F4E79")
    s.title(f"Average vehicle fill per day against the {FILL_TARGET:.0%} target")
    s.text("Fill per vehicle = the higher of its kg fill and pallet fill (as in the planning sheets). "
           "Averaged over every vehicle on that weekday in the sample. As high as possible is better.", italic=True)
    s.gap()
    ft = fill_table(V)
    s.table(ft, pct=tuple(DAYS) + ("Average fill", "KG fill", "Pallet fill", "Target", "Gap to target"))
    s.finish()

    # ---- Detail sheets ----
    s = Sheet(wb, "Detail - Measures by depot", tab="808080")
    s.title("DETAIL (working): service and efficiency measures per depot")
    rows = []
    for depot in [None] + sorted(set(V["Depot"])):
        b, a = M.get("Before", depot), M.get("After", depot)
        for key, label, kind, dec, note in MEASURES + [
                ("kg_fill", "KG fill", "pct", 1, ""), ("pal_fill", "Pallet fill", "pct", 1, ""),
                ("fleet_overflow_kg_week", "KG per week over fleet (to pallet network)", "int", 0, ""),
                ("routecap_overflow_kg_week", "KG per week over route cap (to pallet network)", "int", 0, ""),
                ("actual_routes_day", "Actual routes per day (reference)", "num", 1, "")]:
            rows.append({"Depot": depot or ALL, "Measure": label, "Before": b[key], "After": a[key],
                         "Change": change(b[key], a[key], kind), "_kind": kind})
    _write_measure_table(s, pd.DataFrame(rows), None, depot_col=True)
    s.finish(freeze="A3")

    s = Sheet(wb, "Detail - Fleet allocation", tab="808080")
    s.title("DETAIL (working): fleet allocation, for review")
    s.text("Flags suggest where vehicles could move between depots. Nothing is moved automatically: "
           "moves are a planning decision.", italic=True)
    s.text(f"Short: needed more trucks than the cap on at least {R['allocation_short_share']:.0%} of days. "
           f"Spare: the cap is at least one vehicle above use on the "
           f"{R['allocation_spare_pct']:.0%} busiest-day level.", italic=True)
    s.gap()
    s.table(total_fleet, heading="Total fleet: sum of per-depot caps (closing depots excluded after)")
    s.table(alloc, pct=("Days at cap %", "Days short of fleet %"), heading="By depot and vehicle type (after)")
    s.finish()

    s = Sheet(wb, "Detail - Cost", tab="808080")
    s.title("DETAIL (working): cost per kg and per pallet")
    s.text(cost_method, italic=True)
    if cost:
        cur = C.get("currency", "£")
        crow = [{"Item": f"Budget for the sample ({cur})", "Before": cost["cost_sample"][0], "After": cost["cost_sample"][1]},
                {"Item": f"Cost per kg ({cur})", "Before": cost["cost_kg"][0], "After": cost["cost_kg"][1]},
                {"Item": f"Cost per pallet ({cur})", "Before": cost["cost_pallet"][0], "After": cost["cost_pallet"][1]},
                {"Item": "KG on vehicles in the sample", "Before": mb["kg_week"] * n_weeks, "After": ma["kg_week"] * n_weeks},
                {"Item": "Pallets on vehicles in the sample", "Before": mb["pal_week"] * n_weeks, "After": ma["pal_week"] * n_weeks}]
        s.gap()
        s.table(pd.DataFrame(crow), money=("Before", "After"))
    s.finish()

    s = Sheet(wb, "Detail - Step outputs", tab="808080")
    s.title("DETAIL (working): figures from each step's own output")
    s.text(f"Plan used: {plan_source}", italic=True)
    s.gap()
    for heading, t in step_outputs(plan_source).items():
        s.table(t, heading=heading)
    s.finish()

    s = Sheet(wb, "Detail - Settings", tab="808080")
    s.title("DETAIL (working): settings used for this run")
    s.text(f"From {SETTINGS_PATH}", italic=True)
    s.gap()
    s.table(pd.DataFrame(settings_rows()))
    s.finish()

    out = Path(run_dir) / f"Customer_report_{stamp}.xlsx"
    wb.save(out)
    return out


def _write_measure_table(s, df, heading, depot_col=False):
    """Before / After / Change columns formatted per row (percent, number or money)."""
    if heading:
        c = s.ws.cell(s.row, 1, heading)
        c.font = Font(bold=True, size=11)
        s.row += 1
    cols = (["Depot"] if depot_col else []) + ["Measure", "Before", "After", "Change"] + \
           (["What it means"] if "What it means" in df.columns else [])
    for j, col in enumerate(cols, start=1):
        c = s.ws.cell(s.row, j, col)
        c.font = Font(bold=True, color="FFFFFF")
        c.fill = HEAD_FILL
        s.widths[j] = max(s.widths.get(j, 8), 14 if col in ("Before", "After", "Change") else 20)
    s.row += 1
    fmt = {"pct": "0.0%", "num": "#,##0.0", "int": "#,##0", "money": money_fmt()}
    for rec in df.to_dict("records"):
        for j, col in enumerate(cols, start=1):
            val = rec.get(col)
            if isinstance(val, float) and np.isnan(val):
                val = None
            if isinstance(val, np.generic):
                val = val.item()
            if col == "Change" and rec["_kind"] == "pct" and val is not None:
                c = s.ws.cell(s.row, j, val * 100)          # percentage points
                c.number_format = '+0.0" pts";-0.0" pts";0.0" pts"'
                continue
            c = s.ws.cell(s.row, j, val)
            if col in ("Before", "After", "Change"):
                c.number_format = fmt[rec["_kind"]] if not (col == "Change" and rec["_kind"] == "num") else "+#,##0.0;-#,##0.0"
            if isinstance(val, str):
                s.widths[j] = max(s.widths.get(j, 8), min(len(val) + 2, 90))
        s.row += 1
    s.gap()


def plain_words(M, mb, ma, t_dest, dest_cols, n_weeks, cost):
    lines = []
    for depot in CLOSING_DEPOTS:
        row = t_dest[t_dest["From depot"] == depot]
        if row.empty:
            continue
        r = row.iloc[0]
        parts = [f"{r[c]:.0%} to {c}" for c in dest_cols if c != depot and r.get(c, 0) >= 0.005]
        lines.append(f"{depot} closes. Its {r['KG per week']:,.0f} kg a week goes " + ", ".join(parts) + ".")
    for depot in LOCKED_DEPOTS:
        b, a = M.get("Before", depot), M.get("After", depot)
        if b["kg_week"]:
            lines.append(f"{depot}: weekly weight on its vehicles goes from {b['kg_week']:,.0f} kg to "
                         f"{a['kg_week']:,.0f} kg ({a['kg_week'] / b['kg_week'] - 1:+.0%}).")
    lines.append(f"Each postcode area is served on {ma['days_covered']:.1f} days a week on average, "
                 f"against {mb['days_covered']:.1f} now.")
    lines.append(f"{ma['stability']:.0%} of the weight is delivered on a day the area already has today "
                 f"({mb['stability']:.0%} now).")
    lines.append(f"Vehicles per day: {ma['veh_day']:.1f} against {mb['veh_day']:.1f} now; "
                 f"{ma['drops_veh']:.1f} drops per vehicle against {mb['drops_veh']:.1f}.")
    lines.append(f"Average fill {ma['fill']:.0%} against {mb['fill']:.0%} now (target {FILL_TARGET:.0%}).")
    lines.append(f"Kilometres per tonne: {ma['km_tonne']:.1f} against {mb['km_tonne']:.1f} now.")
    lines.append(f"Areas visited by more than one vehicle on the same day: {ma['multi_share']:.0%} of area-days, "
                 f"against {mb['multi_share']:.0%} now.")
    if cost:
        cur = C.get("currency", "£")
        lines.append(f"Estimated cost per kg {cur}{cost['cost_kg'][1]:.3f} against {cur}{cost['cost_kg'][0]:.3f} now; "
                     f"per pallet {cur}{cost['cost_pallet'][1]:.2f} against {cur}{cost['cost_pallet'][0]:.2f}.")
    return lines
