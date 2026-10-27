"""Day 3: NDVI time series over Sokuluk cropland (Sentinel-2, 80 m, cloud-masked) with weather dry spells shaded."""
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

SURF, INK, INK2, GRID, GREEN, SPELL = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1", "#2e8b57", "#d9d8d2"
d = pd.read_csv("data/processed/sokuluk_ndvi_daily.csv", parse_dates=["date"]).set_index("date")
sp = pd.read_csv("reports/dry_spells.csv", parse_dates=["start", "end"]); sp = sp[(sp.end >= d.index.min()) & (sp.start <= d.index.max())]
good = d[d.valid_frac >= .3]
fig, ax = plt.subplots(figsize=(12, 5.4)); fig.subplots_adjust(left=.055, right=.975, top=.80, bottom=.14)
ax.set_facecolor(SURF); fig.set_facecolor(SURF)
for s, e in zip(sp.start, sp.end): ax.axvspan(s, e, color=SPELL, alpha=.8, lw=0, zorder=0)
seg = (good.index.to_series().diff() > pd.Timedelta(days=60)).cumsum()
for _, g_ in good.groupby(seg.values): ax.plot(g_.index, g_.ndvi_mean, color=GREEN, lw=1.8, zorder=2)   # break the line across the winter gap
ax.scatter(good.index, good.ndvi_mean, s=14, color=GREEN, zorder=3)
low = d[d.valid_frac < .3]; ax.scatter(low.index, low.ndvi_mean, s=14, facecolors="none", edgecolors=GREEN, lw=1, zorder=3)   # sparse-clear dates, hollow
# winter gap (no clear scenes: snow / cloud)
gaps = good.index.to_series().diff(); gs = gaps[gaps > pd.Timedelta(days=60)]
for e_, dt in gs.items():
    s_ = e_ - dt; ax.annotate("no clear data\n(snow / cloud)", ((s_ + (e_ - s_) / 2), .18), ha="center", va="center", fontsize=9.5, color=INK2)
pk = good.ndvi_mean.idxmax(); ax.annotate(f"peak {good.ndvi_mean[pk]:.2f}\n{pk:%-d %b %Y}", (pk, good.ndvi_mean[pk]), xytext=(10, 6), textcoords="offset points", fontsize=9.5, color=INK)
ax.set_ylim(0, .85); ax.set_xlim(d.index.min() - pd.Timedelta(days=10), d.index.max() + pd.Timedelta(days=10))
for s_ in ("top", "right", "left"): ax.spines[s_].set_visible(False)
ax.spines["bottom"].set_color(GRID); ax.tick_params(colors=INK2, length=0, labelsize=10); ax.grid(axis="y", color=GRID, lw=.8); ax.set_axisbelow(True)
ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10])); ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
fig.text(.055, .93, "Sokuluk cropland: NDVI (vegetation greenness) over time", fontsize=16, fontweight="bold", color=INK, ha="left")
fig.text(.055, .875, "Sentinel-2, mean NDVI of cloud-free cropland pixels (80 m) · hollow dot = under 30% of cropland clear · grey = weather dry spell (≥ 14 days under 1 mm)", fontsize=9.6, color=INK2, ha="left")
fig.savefig("reports/figures/sokuluk_ndvi_timeseries.png", dpi=200); print("saved", len(good), "good dates;", len(low), "sparse")
