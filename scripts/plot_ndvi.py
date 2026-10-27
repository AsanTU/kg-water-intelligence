"""Day 3: map + histogram of the Sokuluk cropland NDVI raster -> reports/figures/sokuluk_ndvi_map.png"""
import json
import numpy as np, rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

SURF, INK, INK2, GRID, OFF = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1", "#e9e8e2"
meta = json.load(open("data/processed/sokuluk_ndvi_meta.json"))
with rasterio.open("data/processed/sokuluk_ndvi.tif") as s:
    a = s.read(1); b = s.bounds; res = s.res[0]; nod = s.nodata
valid = a != nod; v = a[valid]
cmap = LinearSegmentedColormap.from_list("ndvi", ["#f4f9ee", "#cfe8b8", "#8ccd7a", "#3fa35a", "#17703a", "#0a4a26"])   # one hue (green), light -> dark
img = np.ma.masked_where(~valid, a)

fig = plt.figure(figsize=(12, 8.6)); fig.set_facecolor(SURF)
ax = fig.add_axes([.03, .04, .224, .78]); ax.set_facecolor(OFF)
im = ax.imshow(img, cmap=cmap, vmin=0, vmax=.9, extent=[b.left / 1e3, b.right / 1e3, b.bottom / 1e3, b.top / 1e3], interpolation="nearest")
ax.set_xticks([]); ax.set_yticks([]); [sp.set_visible(False) for sp in ax.spines.values()]
x0, y0 = b.left / 1e3 + 4, b.bottom / 1e3 + 4; ax.plot([x0, x0 + 10], [y0, y0], color=INK, lw=3); ax.text(x0 + 5, y0 + 2, "10 km", ha="center", fontsize=10, color=INK)
cax = fig.add_axes([.285, .14, .012, .46]); cb = fig.colorbar(im, cax=cax); cb.set_label("NDVI", color=INK2, fontsize=11); cb.ax.tick_params(colors=INK2, labelsize=10, length=0); cb.outline.set_visible(False)

hx = fig.add_axes([.42, .38, .54, .40]); hx.set_facecolor(SURF)
hx.hist(v, bins=np.linspace(0, 1, 51), color="#3fa35a", edgecolor=SURF, lw=.6)
hx.axvline(np.median(v), color=INK, lw=1.4, ls=(0, (4, 3))); hx.text(np.median(v) + .01, hx.get_ylim()[1] * .94, f"median {np.median(v):.2f}", fontsize=10, color=INK, va="top")
for sp_ in ("top", "right", "left"): hx.spines[sp_].set_visible(False)
hx.spines["bottom"].set_color(GRID); hx.set_yticks([]); hx.tick_params(colors=INK2, length=0, labelsize=10); hx.set_xlim(0, 1)
hx.set_xlabel("NDVI of cropland pixels (10 m)", color=INK2, fontsize=10.5); hx.set_title("How green is the farmland?", loc="left", fontsize=12, color=INK, fontweight="bold")
lines = [f"Date: {meta['date']}  (Sentinel-2 L2A, tile {', '.join(t.replace('MGRS-', '') for t in meta['tile'])})", f"Scene cloud cover: {', '.join(str(c) for c in meta['scene_cloud_percent'])} %",
         f"Clear cropland pixels: {meta['valid_share_of_cropland']*100:.0f} % ({meta['valid_area_km2']:.0f} km²)",
         f"Mean NDVI {meta['ndvi_mean']:.2f} · 10th–90th percentile {meta['ndvi_p10']:.2f}–{meta['ndvi_p90']:.2f}",
         "Grey = not cropland or masked (cloud, shadow, snow)", "Sentinel-2 (ESA/Copernicus) · cropland mask: ESA WorldCover 2021"]
for i, ln in enumerate(lines): fig.text(.42, .31 - i * .043, ln, fontsize=10.5, color=INK if i < 4 else INK2)
fig.text(.02, .945, "Sokuluk cropland: NDVI (vegetation greenness)", fontsize=17, fontweight="bold", color=INK)
fig.text(.02, .905, f"Peak-season, cloud-free image · {meta['date']} · darker green = denser, healthier vegetation (NDVI is greenness, not water stress)", fontsize=10.5, color=INK2)
fig.savefig("reports/figures/sokuluk_ndvi_map.png", dpi=200); print("saved map; valid pixels", v.size, "mean", round(float(v.mean()), 3))
