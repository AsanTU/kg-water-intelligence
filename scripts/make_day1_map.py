"""Day 1: stats + map of the Sokuluk cropland mask."""
import json, sys
import numpy as np, rasterio, geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

MASK = "data/processed/sokuluk_cropland.tif"
BOUND = "data/boundaries/sokuluk_district.geojson"
OUT = "reports/figures/sokuluk_cropland_map.png"

with rasterio.open(MASK) as s:
    a = s.read(1); b = s.bounds
    print("CRS", s.crs, "shape", s.shape, "res", s.res, "dtype", s.dtypes, "nodata", s.nodata)
    print("bounds", b)
vals, cnt = np.unique(a, return_counts=True)
print(dict(zip(vals.tolist(), cnt.tolist())))
g = gpd.read_file(BOUND).to_crs("EPSG:4326")
# pixels are in degrees: compute true per-row pixel area (m2), restrict to inside the district
from rasterio.features import rasterize
with rasterio.open(MASK) as s:
    inside = rasterize(g.geometry, out_shape=s.shape, transform=s.transform, fill=0, default_value=1, dtype="uint8").astype(bool)
    lats = b.top - (np.arange(a.shape[0]) + 0.5) * s.res[1]
R = 6371008.8
row_m2 = np.radians(s.res[0]) * np.radians(s.res[1]) * R**2 * np.cos(np.radians(lats))
area = np.broadcast_to(row_m2[:, None], a.shape)
crop_km2 = area[(a == 1) & inside].sum() / 1e6
dist_km2 = area[inside].sum() / 1e6
print(f"inside-district px={inside.sum()}; cropland px={int(((a==1)&inside).sum())}")
print(f"cropland {crop_km2:.0f} km2 of district {dist_km2:.0f} km2 = {crop_km2/dist_km2:.1%}")
print("cropland px outside boundary:", int(((a == 1) & ~inside).sum()))
crop = crop_km2
print("boundary area km2 (EPSG:32643):", round(g.to_crs(32643).area.sum()/1e6))
print("boundary bounds", g.total_bounds)

cmap = ListedColormap(["#e9e4d4", "#2e8b57"])
fig, ax = plt.subplots(figsize=(8, 11))
ax.imshow(a, cmap=cmap, vmin=0, vmax=1, interpolation="nearest",
          extent=[b.left, b.right, b.bottom, b.top])
g.boundary.plot(ax=ax, color="#222", linewidth=1)
ax.set_aspect(1 / np.cos(np.radians(42.8)))  # correct degree distortion
ax.set_xlabel("Longitude (°E)"); ax.set_ylabel("Latitude (°N)")
ax.set_title("Sokuluk District, Chüy Region\nCropland mask (ESA WorldCover 2021, 10 m)")
ax.legend(handles=[Patch(color="#2e8b57", label=f"Cropland ({crop_km2:,.0f} km²)"),
                   Patch(color="#e9e4d4", label="Other land")], loc="lower right")
fig.text(0.01, 0.01, "Source: ESA WorldCover 2021 v200 (CC BY 4.0); boundary: geoBoundaries ADM2", fontsize=8)
fig.tight_layout(rect=(0, 0.02, 1, 1)); fig.savefig(OUT, dpi=200)
print("saved", OUT)
