"""Measure the current delivery matrix with the same metrics as the planner,
without any optimisation. Each district keeps its existing days."""
import numpy as np
import pandas as pd

from matrix_common import MATRIX_PATH, AREA, DAYS, DEPOT, EAST, NORTH, load_matrix

DEPOT_NAME = "Watford"     # the depot to measure


def day_radius_km(M, xy):
    """Average distance from each day's kg-weighted centre, weighted by kg."""
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


def spacing_of(row_days):
    """Distance of gaps between a district's days from the ideal gap of 2.
    Districts with fewer than two days have no gaps and count as 0."""
    idx = sorted(row_days)
    gaps = [b - a for a, b in zip(idx, idx[1:])]
    return sum(abs(g - 2) for g in gaps) / len(gaps) if gaps else 0.0


df = load_matrix(MATRIX_PATH)
d = df[(df[DEPOT] == DEPOT_NAME) & (df["Weight"] > 0)].reset_index(drop=True)

M = d[DAYS].to_numpy(dtype=float)
xy = d[[EAST, NORTH]].to_numpy()
kg = d["Weight"].to_numpy()

day_tot = M.sum(axis=0)
peak = day_tot.max()
spread = (peak - day_tot.min()) / day_tot.mean()
dr = day_radius_km(M, xy)

days_per = (M > 0).sum(axis=1)
avg_days = (days_per * kg).sum() / kg.sum()

spacing = sum(
    spacing_of([j for j in range(len(DAYS)) if M[i, j] > 0]) * kg[i]
    for i in range(len(d))) / kg.sum()

print(f"\n{DEPOT_NAME}: current matrix, {len(d)} weighted districts, "
      f"{kg.sum():,.0f} kg")
print(f"Day radius:  {dr:.2f} km")
print(f"Peak day:    {peak:,.0f} kg")
print(f"Spread:      {spread:.1%}")
print(f"Avg days:    {avg_days:.2f}")
print(f"Spacing:     {spacing:.2f}")

print("\nDaily totals (kg):")
for day, tot in zip(DAYS, day_tot):
    print(f"  {day:<10} {tot:>12,.0f}")

print("\nDistricts by number of days:")
for n_days, count in pd.Series(days_per).value_counts().sort_index().items():
    print(f"  {n_days} day(s): {count}")