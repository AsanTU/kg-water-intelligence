"""Day 2: hourly ERA5-Land NetCDFs -> clean daily CSV for Sokuluk.

Processing (documented in README):
 1. Concatenate monthly files, drop duplicate hours, reindex to a continuous hourly UTC axis (gaps stay NaN).
 2. Units: t2m K -> °C; tp m -> mm. tp is an accumulation since 00 UTC, so de-accumulate to hourly increments
    (hour 01 UTC = the accumulated value itself; other hours = difference to the previous hour; negatives clipped to 0).
 3. Spatial mean over the ERA5-Land cells, WEIGHTED BY CROPLAND AREA in each cell (from the WorldCover mask), so the series
    describes the weather over the farmland, not the mountain tops.
 4. Daily values for Kyrgyzstan local days (UTC+6, no DST). Instant values (temperature, soil water) are assigned by their
    timestamp; a precipitation increment is assigned by the START of its hour. A day needs all 24 hours, otherwise NaN.
Usage: python3 scripts/process_era5land.py [glob] [start_date end_date]   (default: all files, all days)
"""
import glob, sys
import numpy as np, pandas as pd, xarray as xr, rasterio, geopandas as gpd
from rasterio.features import rasterize

PATTERN = sys.argv[1] if len(sys.argv) > 1 else "data/raw/era5land/era5land_*.nc"
START, END = (sys.argv[2], sys.argv[3]) if len(sys.argv) > 3 else (None, None)
OUT = "data/processed/sokuluk_weather_daily.csv"
MASK, BOUND = "data/processed/sokuluk_cropland.tif", "data/boundaries/sokuluk_district.geojson"

files = sorted(glob.glob(PATTERN)); assert files, f"no files match {PATTERN}"
ds = xr.concat([xr.open_dataset(f).drop_vars(["number", "expver"], errors="ignore") for f in files], dim="valid_time").sortby("valid_time")
ds = ds.isel(valid_time=~ds.get_index("valid_time").duplicated())
ds = ds.reindex(valid_time=pd.date_range(ds.valid_time.min().values, ds.valid_time.max().values, freq="h"))
print(f"{len(files)} files | {ds.valid_time.size} hourly steps | {str(ds.valid_time.min().values)[:16]} -> {str(ds.valid_time.max().values)[:16]} UTC")
print("grid:", ds.latitude.size, "lat x", ds.longitude.size, "lon;", float(ds.latitude.min()), "-", float(ds.latitude.max()), "N,", float(ds.longitude.min()), "-", float(ds.longitude.max()), "E")

# ---- cropland weights per ERA5-Land cell
gdf = gpd.read_file(BOUND).to_crs("EPSG:4326")
with rasterio.open(MASK) as s:
    a = s.read(1); tr = s.transform; res = s.res
    inside = rasterize(gdf.geometry, out_shape=s.shape, transform=tr, fill=0, default_value=1, dtype="uint8")
crop = ((a == 1) & (inside == 1)).astype("float32")
lat_px = tr.f + (np.arange(a.shape[0]) + .5) * tr.e; lon_px = tr.c + (np.arange(a.shape[1]) + .5) * tr.a
row_km2 = np.radians(res[0]) * np.radians(res[1]) * 6371008.8**2 * np.cos(np.radians(lat_px)) / 1e6
lat_c, lon_c = ds.latitude.values, ds.longitude.values
ri = np.abs(lat_px[:, None] - lat_c[None, :]).argmin(1); ci = np.abs(lon_px[:, None] - lon_c[None, :]).argmin(1)
R = np.zeros((lat_c.size, a.shape[0]), "float32"); R[ri, np.arange(a.shape[0])] = row_km2
C = np.zeros((a.shape[1], lon_c.size), "float32"); C[np.arange(a.shape[1]), ci] = 1
Wt = R @ crop @ C                                    # cropland km² per ERA5-Land cell
print(f"cropland km² per cell (total {Wt.sum():.0f}); cells with cropland: {(Wt > 0).sum()} of {Wt.size}")
print(pd.DataFrame(Wt, index=lat_c.round(1), columns=lon_c.round(1)).round(0).astype(int).to_string())
wn = Wt / Wt.sum()
wmean = lambda v: np.tensordot(v, wn, axes=([1, 2], [0, 1]))

t_idx = ds.valid_time.to_index()
t2m = wmean(ds.t2m.values) - 273.15
tp_acc = ds.tp.values
inc = np.full_like(tp_acc, np.nan); inc[1:] = tp_acc[1:] - tp_acc[:-1]
h1 = np.where(t_idx.hour == 1)[0]; inc[h1] = tp_acc[h1]                      # reset after 00 UTC
inc = np.clip(inc, 0, None) * 1000.0                                          # mm
tp_mm = wmean(inc)
sw1, sw2 = wmean(ds.swvl1.values), wmean(ds.swvl2.values)

LOCAL = pd.Timedelta(hours=6)
inst = pd.DataFrame({"t": t2m, "sw1": sw1, "sw2": sw2}, index=(t_idx + LOCAL).floor("D"))
prec = pd.DataFrame({"p": tp_mm}, index=(t_idx - pd.Timedelta(hours=1) + LOCAL).floor("D"))
gi, gp = inst.groupby(level=0), prec.groupby(level=0)
daily = pd.DataFrame({
    "temperature_2m_c": gi.t.mean(), "temperature_2m_min_c": gi.t.min(), "temperature_2m_max_c": gi.t.max(),
    "precipitation_mm": gp.p.sum(min_count=24), "soil_moisture_l1_m3m3": gi.sw1.mean(), "soil_moisture_l2_m3m3": gi.sw2.mean()})
full = gi.t.count() == 24
daily.loc[~full.reindex(daily.index, fill_value=False), ["temperature_2m_c", "temperature_2m_min_c", "temperature_2m_max_c", "soil_moisture_l1_m3m3", "soil_moisture_l2_m3m3"]] = np.nan
daily.loc[~(gp.p.count() == 24).reindex(daily.index, fill_value=False), "precipitation_mm"] = np.nan
daily = daily.dropna(how="all").loc[START:END]; daily.index.name = "date"
daily = daily.round({"temperature_2m_c": 2, "temperature_2m_min_c": 2, "temperature_2m_max_c": 2, "precipitation_mm": 2, "soil_moisture_l1_m3m3": 4, "soil_moisture_l2_m3m3": 4})
daily.to_csv(OUT)
print(f"\nsaved {OUT}: {len(daily)} days, {daily.index.min().date()} -> {daily.index.max().date()}")
print("missing per column:\n" + daily.isna().sum().to_string()); print(daily.describe().round(2).T[["min", "mean", "max"]].to_string())
