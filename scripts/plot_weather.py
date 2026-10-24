"""Day 2: temperature + precipitation charts and dry-spell table from data/processed/sokuluk_weather_daily.csv."""
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

CSV = "data/processed/sokuluk_weather_daily.csv"
SURF, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
TEMP, RAIN, SPELL = "#eb6834", "#2a78d6", "#d9d8d2"
DRY_MM, MIN_DAYS, SEASON = 1.0, 14, (4, 10)   # dry day < 1 mm; spell >= 14 days; growing season Apr-Oct

d = pd.read_csv(CSV, parse_dates=["date"]).set_index("date")
span = f"{d.index.min():%-d %b %Y} – {d.index.max():%-d %b %Y}"

# ---- dry spells (growing season only)
in_season = d.index.month.isin(range(SEASON[0], SEASON[1] + 1))
dry = (d.precipitation_mm < DRY_MM) & in_season
grp = (dry != dry.shift()).cumsum()
spells = [(g.index[0], g.index[-1], len(g)) for _, g in d[dry].groupby(grp[dry]) if len(g) >= MIN_DAYS]
sp = pd.DataFrame(spells, columns=["start", "end", "days"]); sp["precip_mm"] = [round(d.loc[s:e, "precipitation_mm"].sum(), 1) for s, e, _ in spells]
sp.to_csv("reports/dry_spells.csv", index=False)
print(sp.to_string(index=False)); print("hot days (max >= 35 °C):", int((d.temperature_2m_max_c >= 35).sum()))

def style(ax, title, sub):
    ax.set_facecolor(SURF); ax.figure.set_facecolor(SURF)
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID); ax.tick_params(colors=INK2, length=0, labelsize=10)
    ax.grid(axis="y", color=GRID, lw=.8); ax.set_axisbelow(True)
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7])); ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax.figure.text(.055, .93, title, fontsize=16, fontweight="bold", color=INK, ha="left")
    ax.figure.text(.055, .875, sub, fontsize=10.5, color=INK2, ha="left")

# ---- temperature
fig, ax = plt.subplots(figsize=(12, 5.4)); fig.subplots_adjust(left=.055, right=.975, top=.80, bottom=.14)
ax.fill_between(d.index, d.temperature_2m_min_c, d.temperature_2m_max_c, color=TEMP, alpha=.18, lw=0)
ax.plot(d.index, d.temperature_2m_c, color=TEMP, lw=1.6)
ax.axhline(0, color=MUTED, lw=1, ls=(0, (4, 3)))
style(ax, "Sokuluk cropland: daily air temperature", f"2 m air temperature in °C, ERA5-Land, cropland-weighted mean · {span} · line = daily mean, band = daily min–max")
ax.set_ylim(-31, 45)
hot = d.temperature_2m_max_c.idxmax(); cold = d.temperature_2m_min_c.idxmin()
ax.annotate(f"hottest: {d.temperature_2m_max_c[hot]:.0f} °C ({hot:%-d %b %Y})", (hot, d.temperature_2m_max_c[hot]), xytext=(-8, 4), textcoords="offset points", ha="right", va="bottom", fontsize=9.5, color=INK)
ax.annotate(f"coldest: {str(round(d.temperature_2m_min_c[cold])).replace('-', '−')} °C ({cold:%-d %b %Y})", (cold, d.temperature_2m_min_c[cold]), xytext=(10, 0), textcoords="offset points", ha="left", va="center", fontsize=9.5, color=INK)
ax.text(d.index[0], .6, "0 °C", color=MUTED, fontsize=9, va="bottom")
fig.savefig("reports/figures/sokuluk_temperature.png", dpi=200); plt.close(fig)

# ---- precipitation + dry spells: one small panel per year (same scale), so years can be compared
years = sorted(d.index.year.unique()); ymax = np.ceil(d.precipitation_mm.max() / 10) * 10
fig, axes = plt.subplots(len(years), 1, figsize=(12, 9.6), sharey=True); fig.subplots_adjust(left=.105, right=.975, top=.885, bottom=.07, hspace=.28)
for ax, yr in zip(axes, years):
    dy = d[d.index.year == yr]; ax.set_facecolor(SURF)
    for s_, e_, n_ in spells:
        if s_.year == yr:
            ax.axvspan(s_ - pd.Timedelta(hours=12), e_ + pd.Timedelta(hours=12), color=SPELL, alpha=.85, lw=0, zorder=0)
            ax.text(s_ + (e_ - s_) / 2, ymax * .97, f"{n_}", ha="center", va="top", fontsize=8.5, color=INK, zorder=3)
    ax.bar(dy.index, dy.precipitation_mm, width=1.0, color=RAIN, lw=0, zorder=2, edgecolor="none")
    ax.set_xlim(pd.Timestamp(f"{yr}-01-01"), pd.Timestamp(f"{yr}-12-31")); ax.set_ylim(0, ymax); ax.set_yticks([0, 20, 40])
    for sp_ in ("top", "right", "left"): ax.spines[sp_].set_visible(False)
    ax.spines["bottom"].set_color(GRID); ax.tick_params(colors=INK2, length=0, labelsize=9); ax.grid(axis="y", color=GRID, lw=.7); ax.set_axisbelow(True)
    ax.xaxis.set_major_locator(mdates.MonthLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter("%b" if ax is axes[-1] else ""))
    tot = dy.precipitation_mm.sum(); season = dy[dy.index.month.isin(range(SEASON[0], SEASON[1] + 1))].precipitation_mm.sum()
    ax.text(-.04, .5, str(yr), transform=ax.transAxes, ha="right", va="center", fontsize=12, fontweight="bold", color=INK)
    ax.text(1.0, .97, f"year {tot:.0f} mm · Apr–Oct {season:.0f} mm", transform=ax.transAxes, ha="right", va="top", fontsize=8.5, color=INK2)
wet = d.precipitation_mm.idxmax(); aw = axes[years.index(wet.year)]
aw.annotate(f"wettest day: {d.precipitation_mm[wet]:.0f} mm ({wet:%-d %b %Y})", (wet, d.precipitation_mm[wet]), xytext=(-8, -2), textcoords="offset points", ha="right", va="center", fontsize=8.5, color=INK)
fig.set_facecolor(SURF)
fig.text(.105, .955, "Sokuluk cropland: daily precipitation by year", fontsize=16, fontweight="bold", color=INK, ha="left")
fig.text(.105, .925, f"Precipitation in mm per day, ERA5-Land, cropland-weighted · {span} · grey = dry spell (≥ {MIN_DAYS} days under {DRY_MM:g} mm, Apr–Oct; number = days)", fontsize=9.6, color=INK2, ha="left")
fig.savefig("reports/figures/sokuluk_precipitation.png", dpi=200); plt.close(fig)
print("saved reports/figures/sokuluk_temperature.png, sokuluk_precipitation.png")
