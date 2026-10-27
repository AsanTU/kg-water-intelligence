"""Day 4: first structured analysis dataset = one row per DATE x ERA5-Land GRID CELL (weather grid, 0.1 deg ~ 9 km), cropland cells only.

Inputs (never modified): ERA5-Land hourly NetCDFs (data/raw/era5land), cropland mask, per-cell NDVI cache (scripts/ndvi_by_cell.py).
Outputs: data/processed/sokuluk_grid_daily.csv.gz, data/processed/grid_cells.csv, reports/data_audit.md, reports/figures/data_coverage.png
Rules: weather = daily values for Kyrgyzstan local days (UTC+6, same processing as Day 2); NDVI is kept ONLY on days with a clear Sentinel-2 observation
(>= 30 % of the cell's cropland pixels clear) — nothing is interpolated, forward-filled or invented.
"""
import glob, json, os
import numpy as np, pandas as pd, rasterio, xarray as xr, geopandas as gpd
from rasterio.features import rasterize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import ndvi_by_cell as NC            # reuses the 80 m grid, cell-id rule and per-cell cropland pixel counts (import runs the setup only)

NLAT, NLON, LAT0, LON0, STEP = NC.NLAT, NC.NLON, NC.LAT0, NC.LON0, NC.STEP
MIN_FRAC = 0.30
OUT_CSV, OUT_CELLS, OUT_MD, OUT_FIG = "data/processed/sokuluk_grid_daily.csv.gz", "data/processed/grid_cells.csv", "reports/data_audit.md", "reports/figures/data_coverage.png"
checks = []                                                              # (name, passed, detail)
def check(name, ok, detail): checks.append((name, bool(ok), detail)); print(("PASS " if ok else "FAIL ") + name + " — " + detail, flush=True)

# ---------------- 1) grid cells with cropland (area from the original 4326 mask, same method as Day 2)
with rasterio.open("data/processed/sokuluk_cropland.tif") as s:
    ca = s.read(1); tr = s.transform; res = s.res; bnd = s.bounds
gd = gpd.read_file("data/boundaries/sokuluk_district.geojson").to_crs("EPSG:4326")
inside = rasterize(gd.geometry, out_shape=ca.shape, transform=tr, fill=0, default_value=1, dtype="uint8")
lat_px = tr.f + (np.arange(ca.shape[0]) + .5) * tr.e; lon_px = tr.c + (np.arange(ca.shape[1]) + .5) * tr.a
row_km2 = np.radians(res[0]) * np.radians(res[1]) * 6371008.8 ** 2 * np.cos(np.radians(lat_px)) / 1e6
ri = np.clip(np.round((LAT0 - lat_px) / STEP).astype(int), 0, NLAT - 1); ci = np.clip(np.round((lon_px - LON0) / STEP).astype(int), 0, NLON - 1)
crop = ((ca == 1) & (inside == 1)).astype("float32")
R = np.zeros((NLAT, ca.shape[0]), "float32"); R[ri, np.arange(ca.shape[0])] = row_km2; C = np.zeros((ca.shape[1], NLON), "float32"); C[np.arange(ca.shape[1]), ci] = 1
km2 = (R @ crop @ C).reshape(-1)                                          # cell_id = lat_i * NLON + lon_j
cells = pd.DataFrame({"cell_id": np.arange(NLAT * NLON), "lat_index": np.repeat(np.arange(NLAT), NLON), "lon_index": np.tile(np.arange(NLON), NLAT)})
cells["latitude"] = (LAT0 - cells.lat_index * STEP).round(2); cells["longitude"] = (LON0 + cells.lon_index * STEP).round(2); cells["cropland_km2"] = km2.round(2)
cells = cells[cells.cropland_km2 > 0].reset_index(drop=True); cells.to_csv(OUT_CELLS, index=False)
print(f"{len(cells)} ERA5-Land cells contain cropland ({cells.cropland_km2.sum():.0f} km² total)", flush=True)
CID = cells.cell_id.values

# ---------------- 2) daily weather PER CELL (same rules as Day 2: K->C, m->mm, de-accumulated precipitation, local UTC+6 days, 24 h required)
files = sorted(glob.glob("data/raw/era5land/era5land_*.nc"))
ds = xr.concat([xr.open_dataset(f).drop_vars(["number", "expver"], errors="ignore") for f in files], dim="valid_time").sortby("valid_time")
ds = ds.isel(valid_time=~ds.get_index("valid_time").duplicated()); ds = ds.reindex(valid_time=pd.date_range(ds.valid_time.min().values, ds.valid_time.max().values, freq="h"))
assert (np.round((LAT0 - ds.latitude.values) / STEP) == np.arange(NLAT)).all() and (np.round((ds.longitude.values - LON0) / STEP) == np.arange(NLON)).all(), "ERA5 grid differs from the assumed cell numbering"
T_ = ds.valid_time.size; flat = lambda a: a.reshape(T_, NLAT * NLON)[:, CID]
t2m = flat(ds.t2m.values) - 273.15; acc = flat(ds.tp.values); sw1 = flat(ds.swvl1.values); sw2 = flat(ds.swvl2.values)
inc = np.full_like(acc, np.nan); inc[1:] = acc[1:] - acc[:-1]; idx = ds.get_index("valid_time"); h1 = np.where(idx.hour == 1)[0]; inc[h1] = acc[h1]; inc = np.clip(inc, 0, None) * 1000.0
LOC = pd.Timedelta(hours=6); day_i = (idx + LOC).floor("D"); day_p = (idx - pd.Timedelta(hours=1) + LOC).floor("D")
def daily(arr, days, how):
    df = pd.DataFrame(arr, index=days); g = df.groupby(level=0); out = getattr(g, how)(); full = g.count() == 24; return out.where(full)
W = {"temperature_2m_c": daily(t2m, day_i, "mean"), "temperature_2m_min_c": daily(t2m, day_i, "min"), "temperature_2m_max_c": daily(t2m, day_i, "max"),
     "precipitation_mm": daily(inc, day_p, "sum"), "soil_moisture_l1_m3m3": daily(sw1, day_i, "mean"), "soil_moisture_l2_m3m3": daily(sw2, day_i, "mean")}
dates = W["temperature_2m_c"].dropna(how="all").index; dates = dates[(dates >= "2018-01-02") & (dates <= "2024-12-31")]
wide = {k: v.reindex(dates) for k, v in W.items()}
long = pd.DataFrame({"date": np.repeat(dates.values, len(CID)), "cell_id": np.tile(CID, len(dates))})
for k, v in wide.items(): long[k] = v.values.reshape(-1)
long = long.merge(cells[["cell_id", "latitude", "longitude", "cropland_km2"]], on="cell_id")

# ---------------- 3) NDVI per cell per date (from the per-scene cache); one granule per (date, tile), tiles summed
cache = "data/raw/sentinel2_ndvi_cells.jsonl"; recs = [json.loads(l) for l in open(cache)] if os.path.exists(cache) else []
rows = []
for r_ in recs:
    nv = np.array(r_["n_valid"]); rows.append(dict(date=r_["date"], tile=r_["tile"], id=r_["id"], tot=int(nv.sum()), nv=nv, sm=np.array(r_["ndvi_sum"])))
best = {}
for r_ in rows:
    k = (r_["date"], r_["tile"])
    if k not in best or r_["tot"] > best[k]["tot"]: best[k] = r_
bydate = {}
for (d_, _), r_ in best.items():
    a = bydate.setdefault(d_, dict(nv=np.zeros(NLAT * NLON), sm=np.zeros(NLAT * NLON), n=0)); a["nv"] += r_["nv"]; a["sm"] += r_["sm"]; a["n"] += 1
ncrop = NC.NCROP[CID].astype(float)
nd_rows = []
for d_, a in sorted(bydate.items()):
    nv, sm = a["nv"][CID], a["sm"][CID]; vf = np.clip(nv / np.maximum(ncrop, 1), 0, 1); mean = np.where(nv > 0, sm / np.maximum(nv, 1), np.nan)
    for j, c in enumerate(CID): nd_rows.append((pd.Timestamp(d_), c, mean[j] if vf[j] >= MIN_FRAC else np.nan, vf[j], int(nv[j]), a["n"]))
nd = pd.DataFrame(nd_rows, columns=["date", "cell_id", "ndvi", "ndvi_valid_frac", "ndvi_n_pixels", "ndvi_n_scenes"])
long["date"] = pd.to_datetime(long["date"]); data = long.merge(nd, on=["date", "cell_id"], how="left")
data = data[["date", "cell_id", "latitude", "longitude", "cropland_km2", "temperature_2m_c", "temperature_2m_min_c", "temperature_2m_max_c", "precipitation_mm",
             "soil_moisture_l1_m3m3", "soil_moisture_l2_m3m3", "ndvi", "ndvi_valid_frac", "ndvi_n_pixels", "ndvi_n_scenes"]].sort_values(["date", "cell_id"]).reset_index(drop=True)
data["date"] = data["date"].dt.strftime("%Y-%m-%d")
data.round({"temperature_2m_c": 2, "temperature_2m_min_c": 2, "temperature_2m_max_c": 2, "precipitation_mm": 2, "soil_moisture_l1_m3m3": 4, "soil_moisture_l2_m3m3": 4,
            "ndvi": 4, "ndvi_valid_frac": 3, "cropland_km2": 2}).to_csv(OUT_CSV, index=False, compression="gzip")
D = data.copy(); D["date"] = pd.to_datetime(D["date"])

# ---------------- 4) validation
check("no duplicate (date, cell) rows", not D.duplicated(["date", "cell_id"]).any(), f"{len(D):,} rows, {D.date.nunique()} dates x {D.cell_id.nunique()} cells")
exp = pd.date_range("2018-01-02", "2024-12-31"); check("dates continuous (no gaps)", D.date.nunique() == len(exp) and (D.groupby("cell_id").size() == len(exp)).all(), f"{D.date.min().date()} to {D.date.max().date()}, {len(exp)} days per cell")
wcols = ["temperature_2m_c", "temperature_2m_min_c", "temperature_2m_max_c", "precipitation_mm", "soil_moisture_l1_m3m3", "soil_moisture_l2_m3m3"]
check("weather has no missing values", D[wcols].isna().sum().sum() == 0, f"{int(D[wcols].isna().sum().sum())} missing of {D[wcols].size:,}")
rng = {"temperature_2m_c": (-40, 45), "precipitation_mm": (0, 80), "soil_moisture_l1_m3m3": (0, .6), "soil_moisture_l2_m3m3": (0, .6)}
check("weather values in physical ranges (°C, mm/day, m³/m³)", all(D[c].between(*v).all() for c, v in rng.items()), "; ".join(f"{c} {D[c].min():.2f}…{D[c].max():.2f}" for c in rng))
w = D.groupby("date").apply(lambda g: pd.Series({c: np.average(g[c], weights=g.cropland_km2) for c in ("temperature_2m_c", "precipitation_mm", "soil_moisture_l1_m3m3")}))
d2 = pd.read_csv("data/processed/sokuluk_weather_daily.csv", parse_dates=["date"]).set_index("date").reindex(w.index)
dif = max((w[c] - d2[c]).abs().max() for c in w.columns); check("cell weather, weighted by cropland, reproduces the Day 2 district series", dif < 0.02, f"max abs difference {dif:.4f}")
cell80 = NC.NCROP[CID] * NC.RES ** 2 / 1e6; ratio = (cell80.sum() / cells.cropland_km2.sum()); big = np.abs(cell80 - cells.cropland_km2.values).max()
check("spatial alignment: cropland area per cell from the 4326 mask vs the 80 m UTM grid agree", abs(ratio - 1) < 0.05 and big < 6, f"total ratio {ratio:.3f} (80 m grid keeps cells >= 60 % cropland), worst cell difference {big:.1f} km²")
if bydate:
    dsum = pd.Series({d_: a["sm"].sum() / max(a["nv"].sum(), 1) for d_, a in bydate.items()}); d3 = pd.read_csv("data/processed/sokuluk_ndvi_daily.csv", parse_dates=["date"]).set_index("date")["ndvi_mean"]
    dsum.index = pd.to_datetime(dsum.index); jj = d3.index.intersection(dsum.index); df3 = (dsum[jj] - d3[jj]).abs()
    check("cell NDVI (pixel-weighted) reproduces the Day 3 district NDVI series", len(jj) > 0 and df3.max() < 0.01, f"{len(jj)} common dates, max abs difference {df3.max():.4f}")
    check("NDVI only where a clear observation exists (nothing filled)", D.ndvi.notna().sum() <= (D.ndvi_valid_frac >= MIN_FRAC).sum(), f"{int(D.ndvi.notna().sum()):,} rows with NDVI of {len(D):,}")
    nrange = D.ndvi.dropna(); check("NDVI values in range", nrange.between(-1, 1).all(), f"{nrange.min():.2f}…{nrange.max():.2f}")

# ---------------- 5) coverage figure + audit report
W0 = "2023-04-01"; Dn = D[D.date >= W0].copy(); Dn["m"] = Dn.date.dt.to_period("M")
has = Dn.assign(has=Dn.ndvi.notna()).groupby(["cell_id", "m"]).apply(lambda g: g.loc[g.has, "date"].nunique()).unstack(fill_value=0)
cov = Dn.groupby("date").ndvi.apply(lambda s: s.notna().mean()); dates_with = int((cov > 0).sum()); ndays = Dn.date.nunique()
cellrow = Dn.assign(has=Dn.ndvi.notna()).groupby("cell_id").has.mean() * 100; no_ndvi = cellrow[cellrow == 0].index; no_ndvi_km2 = float(cells[cells.cell_id.isin(no_ndvi)].cropland_km2.sum())
fig, ax = plt.subplots(2, 1, figsize=(12, 9), gridspec_kw={"height_ratios": [2.1, 1]}); fig.subplots_adjust(left=.09, right=.93, top=.9, bottom=.07, hspace=.3)
im = ax[0].imshow(has.values, aspect="auto", cmap="Greens", vmin=0, vmax=10, interpolation="nearest"); ax[0].set_yticks(range(len(has)))
ax[0].set_yticklabels([f"{cells.loc[cells.cell_id == c, 'latitude'].iloc[0]:.1f}N {cells.loc[cells.cell_id == c, 'longitude'].iloc[0]:.1f}E" for c in has.index], fontsize=7)
ax[0].set_xticks(range(len(has.columns))); ax[0].set_xticklabels([m.strftime("%b\n%y") for m in has.columns], fontsize=8)
cb = fig.colorbar(im, ax=ax[0], pad=.01, fraction=.03); cb.set_label("days with NDVI", fontsize=9)
ax[0].set_title("NDVI days per grid cell and month, Apr 2023 – Dec 2024 (white = no clear satellite day)", fontsize=11, loc="left")
mm = Dn.groupby("m").apply(lambda g: g.ndvi.notna().groupby(g.date).any().sum()); ax[1].bar(range(len(mm)), mm.values, color="#2e8b57"); ax[1].set_xticks(range(len(mm))); ax[1].set_xticklabels([m.strftime("%b\n%y") for m in mm.index], fontsize=8)
ax[1].set_title("Days per month with at least one cell observed by satellite (weather: every day, 2018–2024)", fontsize=11, loc="left"); [ax[1].spines[s_].set_visible(False) for s_ in ("top", "right")]
fig.suptitle("Day 4 dataset: when and where NDVI exists", fontsize=14, fontweight="bold", x=.09, ha="left"); fig.savefig(OUT_FIG, dpi=170); plt.close(fig)

def fmt(n): return f"{n:,}"
md = ["# Day 4 data audit and coverage summary", "", "Generated by `scripts/build_dataset.py`. Source files are never modified.", "", "## Inputs (audit)", "",
      "| File | Role | CRS / grid | Time coverage | Missing / notes |", "|---|---|---|---|---|",
      "| `data/processed/sokuluk_cropland.tif` | farmland mask (1 = cropland) | EPSG:4326, ~9.3 × 6.8 m pixels, 9,727 × 5,211 | single map (WorldCover 2021) | file declares `nodata=0` (= \"other land\"): read as a plain array when resampling |",
      "| `data/boundaries/sokuluk_district.geojson` | district outline | EPSG:4326 | – | generalised administrative boundary |",
      f"| `data/raw/era5land/*.nc` (84 files) | hourly weather | EPSG:4326 regular grid, 0.1° (~9 km), {NLAT} × {NLON} cells | 2018-01-01 to 2024-12-31 hourly | no missing hours; units K, m (accumulated), m³/m³ |",
      f"| `data/processed/sokuluk_weather_daily.csv` | district-mean daily weather (Day 2) | one series (cropland-weighted) | {exp[0].date()} to {exp[-1].date()} ({len(exp):,} days) | 0 missing; °C, mm/day, m³/m³ |",
      "| `data/processed/sokuluk_ndvi.tif` | 10 m NDVI map, one date (Day 3) | EPSG:32643 (UTM 43N), 10 m, 3,668 × 9,135 | 27 May 2024 only | nodata −9999 (non-cropland or masked); unitless −1…1 |",
      f"| `data/processed/sokuluk_ndvi_daily.csv` | district-mean NDVI series (Day 3) | one series, 80 m | 2023-04-08 to 2024-11-28 ({len(d3) if bydate else '121'} dates) | gap Dec–mid-Mar (snow/cloud) |",
      f"| `data/raw/sentinel2_ndvi_cells.jsonl` | per-scene NDVI sums per ERA5 cell (Day 4) | UTM 80 m grid → ERA5 cell ids | {len(recs)} scenes, Apr 2023 – Dec 2024 | rebuilt because Day 3 stored only district totals |", "",
      "**CRS / grid mismatches found and handled:** cropland mask (EPSG:4326, ~8 m) vs NDVI (EPSG:32643, 10 m / 80 m) vs weather (0.1° lat/lon). Everything is brought to the **weather grid**: each 80 m NDVI pixel and each mask pixel is assigned to the ERA5-Land cell whose centre is nearest (cell id = lat_index × 8 + lon_index).", "",
      "## Output", "", f"`data/processed/sokuluk_grid_daily.csv.gz`: **{fmt(len(D))} rows** = {D.date.nunique():,} days × {D.cell_id.nunique()} cells (cells with cropland: {cells.cropland_km2.sum():.0f} km²). Cell table: `data/processed/grid_cells.csv`.", "",
      "*Note:* Day 2 said 36 cells; that count included 6 cells holding only trace cropland (< 0.01 km²). They are dropped here; the total cropland area (855 km²) is unchanged.", "",
      "| Column | Meaning |", "|---|---|", "| `date`, `cell_id`, `latitude`, `longitude` | local day (UTC+6) and ERA5-Land cell (centre coordinates) |", "| `cropland_km2` | cropland area in the cell (weights for averaging) |",
      "| `temperature_2m_c`, `_min_c`, `_max_c` | 2 m air temperature, daily mean / min / max (°C) |", "| `precipitation_mm` | daily precipitation (mm) |", "| `soil_moisture_l1_m3m3`, `_l2_m3m3` | volumetric soil water, 0–7 cm and 7–28 cm (m³/m³) |",
      f"| `ndvi` | mean NDVI of clear cropland pixels in the cell on that day; **empty when no satellite observation** or < {int(MIN_FRAC*100)} % of the cell's cropland is clear |", "| `ndvi_valid_frac`, `ndvi_n_pixels`, `ndvi_n_scenes` | how much of the cell was clear, how many 80 m pixels, how many tiles combined |", "",
      "## Validation", "", "| Check | Result | Detail |", "|---|---|---|"] + [f"| {n} | {'PASS' if ok else '**FAIL**'} | {d} |" for n, ok, d in checks] + ["",
      "## Missing values and coverage", "", "| Variable | Rows with a value | Missing |", "|---|---|---|"]
for c in wcols + ["ndvi"]: md.append(f"| `{c}` | {fmt(int(D[c].notna().sum()))} ({D[c].notna().mean()*100:.1f} %) | {fmt(int(D[c].isna().sum()))} ({D[c].isna().mean()*100:.1f} %) |")
md += ["", f"- **Weather:** complete for every cell and every day (2018-01-02 – 2024-12-31).", f"- **NDVI exists only for Apr 2023 – Dec 2024** ({ndays} days in that window). On **{dates_with} of those days** at least one cell has a usable observation; on a typical observation day {cov[cov > 0].mean()*100:.0f} % of the cells are usable. In the NDVI window a cell has NDVI on **{cellrow.mean():.0f} % of days on average** (range {cellrow.min():.0f}–{cellrow.max():.0f} %).",
       f"- **{len(no_ndvi)} of {len(cells)} cells never get NDVI** ({no_ndvi_km2:.1f} km² of cropland, {no_ndvi_km2/cells.cropland_km2.sum()*100:.1f} %): their cropland is too small or scattered for the 80 m pixels to reach the 30 % clear rule. They keep weather but have no NDVI.",
       "- **NDVI is never filled in.** There is none for 2018 – Mar 2023, none from Dec to mid-March (snow, cloud), and most days between satellite passes (about every 5 days, fewer when cloudy).",
       "- **Soil moisture** here is ERA5-Land model soil water (layers 1–2); soil *properties* (SoilGrids) are not added yet.", "",
       "![Coverage](figures/data_coverage.png)", "",
       "## What this means for a 3–7 day prediction (honest limits)", "",
       "- NDVI and weather are aligned on the same date and grid cell, but NDVI days are sparse: a model would have to use \"latest observation\" features (and say how old they are) or work only on observation days. Day 5 must decide this explicitly.",
       "- Only about two growing seasons have NDVI. Extending the Sentinel-2 series back to 2018 would give far more training seasons.",
       "- The cells are ~9 km wide, so every field in a cell shares one weather record; this is a **regional**, not field-level, dataset.",
       "- There is still no water-stress label (Day 5) and no ground-truth validation."]
open(OUT_MD, "w").write("\n".join(md)); print("saved", OUT_CSV, OUT_CELLS, OUT_MD, OUT_FIG)
