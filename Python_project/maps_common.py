"""Day maps shared by anneal_regional.py and sweep_plan.py.

One row of five day maps per depot, from a Map-sheet style table of
(Postcode Area, Depot, Weekday, Weight) rows and the depot's grid references.
The optional UK outline is read from the path in the .env file.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

import numpy as np
import pandas as pd
from openpyxl import load_workbook
from openpyxl.drawing.image import Image as XLImage

import matplotlib
matplotlib.use("Agg")                  # Stop pop ups when running in a terminal
import matplotlib.pyplot as plt
import geopandas as gpd

from matrix_common import AREA, DAYS, DEPOT, EAST, NORTH

# ---- Settings -------------------------------------------------------------
MAP_DPI = 150

# ---- env variables -----------------------------------------------------------
load_dotenv()
UK_OUTLINE_PATH = Path(os.getenv("MAP_IMAGE_PATH"))  # loaded from env var or None for dots only


def load_uk_outline():
    if not UK_OUTLINE_PATH:
        return None
    try:
        return gpd.read_file(UK_OUTLINE_PATH).to_crs(27700)
    except Exception as e:
        print(f"Outline not loaded ({e}); maps will show dots only")
        return None


def plot_depot_maps(depot_name, map_rows, depot_df, out_png, outline=None):
    """One row of five day maps for a depot, using the same data as the Map sheet."""
    loads = pd.DataFrame(map_rows, columns=[AREA, DEPOT, "Weekday", "Weight"])
    coords = depot_df.drop_duplicates(AREA).set_index(AREA)[[EAST, NORTH]]
    pts = loads.join(coords, on=AREA)
    pts = pts[(pts[EAST] > 0) & (pts[NORTH] > 0)]
    peak = pts.loc[pts["Weight"] > 0, "Weight"].max()
    peak = peak if peak and peak > 0 else 1.0

    fig, axes = plt.subplots(1, len(DAYS), figsize=(4.2 * len(DAYS), 5.2))
    for ax, day in zip(axes, DAYS):
        if outline is not None:
            outline.plot(ax=ax, color="#dedede", edgecolor="white", linewidth=0.3)

        d = pts[pts["Weekday"] == day]
        copied = d[d["Weight"] <= 0]        # zero-volume codes given the nearest district's day
        weighted = d[d["Weight"] > 0]
        ax.scatter(copied[EAST], copied[NORTH], s=6,
                   color="#9cc3e6", linewidths=0)
        ax.scatter(weighted[EAST], weighted[NORTH],
                   s=6 + 40 * np.sqrt(weighted["Weight"] / peak),
                   color="#1f5fa8", linewidths=0)

        if outline is not None:
            minx, miny, maxx, maxy = outline.total_bounds
            ax.set_xlim(minx, maxx)
            ax.set_ylim(miny, maxy)
        ax.set_aspect("equal")
        ax.set_axis_off()
        kg = weighted["Weight"].sum()
        ax.set_title(f"{day.upper()}\n{kg:,.0f} kg | {weighted[AREA].nunique()} districts",
                     fontsize=11, fontweight="bold")

    fig.suptitle(f"{depot_name} - delivery districts by day", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_png, dpi=MAP_DPI)
    plt.close(fig)
    return out_png


def add_maps_sheet(xlsx_path, png_path, sheet="Maps"):
    wb = load_workbook(xlsx_path)
    ws = wb.create_sheet(sheet)
    img = XLImage(str(png_path))
    img.width, img.height = img.width * 0.5, img.height * 0.5
    ws.add_image(img, "A1")
    wb.save(xlsx_path)
