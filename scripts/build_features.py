"""Day 7: reproducible, leakage-safe feature dataset for forecasting soil-moisture anomalies 3-7 days ahead.

One row per ERA5-Land grid cell and FORECAST DATE t (growing season Apr-Oct). Features use only information up to and including day t (a daily value is complete at the
end of local day t, so the forecast is issued at the end of day t). The label uses days t+3 ... t+7 only.

Leakage rules implemented here:
  * the "normal" for the anomaly (mean and std per cell and day of year, +-15 days) is computed from the TRAINING YEARS ONLY (2018-2022) and then applied to every year;
  * features are trailing windows ending at t; NDVI is the latest observation at least NDVI_LATENCY days old, never older than NDVI_MAX_AGE days, never filled when missing;
  * splits are by calendar year (train 2018-2022, validation 2023, test 2024).

Run:  python3 scripts/build_features.py      (inputs: data/processed/sokuluk_grid_daily.csv.gz; outputs listed at the bottom)
"""
import json
import numpy as np, pandas as pd

TRAIN_YEARS, VAL_YEAR, TEST_YEAR = [2018, 2019, 2020, 2021, 2022], 2023, 2024
WIN, LEAD_MIN, LEAD_MAX, THRESH = 15, 3, 7, -1.0          # normal window +-15 d; label window t+3..t+7; dry if mean z <= -1
NDVI_LATENCY, NDVI_MAX_AGE, SEASON = 2, 21, (4, 10)        # NDVI usable 2 days after acquisition, at most 21 days old; forecast dates Apr-Oct
SD_FLOOR = 0.01                                             # m3/m3
MIN_POS_CELLDAYS, MIN_EPISODES, EPISODE_FRAC = 200, 3, 0.10  # split adequacy, fixed BEFORE looking at results
GRID = "data/processed/sokuluk_grid_daily.csv.gz"
OUT = dict(features="data/processed/water_stress_features.parquet", defs="data/processed/water_stress_feature_definitions.csv", splits="data/processed/water_stress_splits.csv",
           baseline="data/processed/soil_moisture_normal_train2018_2022.csv", quality="reports/feature_dataset_quality.json")

def split_of(year): return "train" if year in TRAIN_YEARS else "val" if year == VAL_YEAR else "test" if year == TEST_YEAR else "unused"

def load_grid(path=GRID):
    return pd.read_csv(path, parse_dates=["date"]).sort_values(["cell_id", "date"]).reset_index(drop=True)

def normal_from_years(cell_df, col, years, win=WIN):
    """mean and std of `col` per day of year (1..365), pooling +-win days of the given years only."""
    doy = np.minimum(cell_df.date.dt.dayofyear.values, 365); yr = cell_df.date.dt.year.values; M = np.full((len(years), 365), np.nan)
    for i, y in enumerate(years): m = yr == y; M[i, doy[m] - 1] = cell_df[col].values[m]
    k = np.ones(2 * win + 1); circ = lambda A: np.apply_along_axis(lambda r: np.convolve(r, k, mode="valid"), 1, np.concatenate([A[:, -win:], A, A[:, :win]], axis=1))
    V = ~np.isnan(M); S, Q, N = circ(np.nan_to_num(M)).sum(0), circ(np.nan_to_num(M) ** 2).sum(0), circ(V.astype(float)).sum(0)
    mu = S / N; sd = np.maximum(np.sqrt(np.maximum(Q / N - mu ** 2, 1e-6)), SD_FLOOR); return mu, sd

def normals_table(d, years=TRAIN_YEARS):
    rows = []
    for c, g in d.groupby("cell_id"):
        m1, s1 = normal_from_years(g, "soil_moisture_l1_m3m3", years); m2, s2 = normal_from_years(g, "soil_moisture_l2_m3m3", years)
        rows.append(pd.DataFrame({"cell_id": c, "doy": np.arange(1, 366), "mu_l1": m1, "sd_l1": s1, "mu_l2": m2, "sd_l2": s2}))
    return pd.concat(rows, ignore_index=True)

def cell_features(g, nt):
    """features + label for ONE cell. g = that cell's daily rows sorted by date (continuous days); nt = normals rows of that cell indexed by doy.
    Every feature at row i uses rows <= i only; label columns use rows i+3..i+7 only."""
    g = g.copy(); doy = np.minimum(g.date.dt.dayofyear.values, 365)
    g["z_l1"] = (g.soil_moisture_l1_m3m3.values - nt.mu_l1.values[doy - 1]) / nt.sd_l1.values[doy - 1]
    g["z_l2"] = (g.soil_moisture_l2_m3m3.values - nt.mu_l2.values[doy - 1]) / nt.sd_l2.values[doy - 1]
    r = lambda s, n, how="mean": getattr(s.rolling(n, min_periods=n), how)()
    f = pd.DataFrame({"forecast_date": g.date.values, "cell_id": g.cell_id.values, "latitude": g.latitude.values, "longitude": g.longitude.values, "cropland_km2": g.cropland_km2.values})
    f["doy_sin"], f["doy_cos"] = np.sin(2 * np.pi * doy / 365), np.cos(2 * np.pi * doy / 365)
    f["sm_l2_t"] = g.soil_moisture_l2_m3m3.values; f["z_l1_t"] = g.z_l1.values; f["z_l2_t"] = g.z_l2.values
    z2 = g.z_l2; f["z_l2_mean3"] = r(z2, 3).values; f["z_l2_mean7"] = r(z2, 7).values; f["z_l2_delta3"] = (z2 - z2.shift(3)).values
    f["temp_mean_7d"] = r(g.temperature_2m_c, 7).values; f["temp_max_7d"] = r(g.temperature_2m_max_c, 7, "max").values
    f["precip_sum_7d"] = r(g.precipitation_mm, 7, "sum").values; f["precip_sum_14d"] = r(g.precipitation_mm, 14, "sum").values
    p = g.precipitation_mm.values; idx = np.arange(len(p)); last = np.maximum.accumulate(np.where(p >= 1, idx, -1)); f["days_since_rain"] = np.where(last < 0, 60, np.minimum(idx - last, 60))
    fut = pd.concat([z2.shift(-k) for k in range(LEAD_MIN, LEAD_MAX + 1)], axis=1); f["z_l2_future_mean_t3_t7"] = fut.mean(axis=1, skipna=False).values
    f["label_dry_t3_t7"] = np.where(np.isnan(f.z_l2_future_mean_t3_t7), np.nan, (f.z_l2_future_mean_t3_t7 <= THRESH).astype(float))
    return f

def add_ndvi(f, obs):
    """latest NDVI observation usable at the forecast date (acquired >= NDVI_LATENCY days earlier, <= NDVI_MAX_AGE days old). Never interpolated."""
    f = f.sort_values("forecast_date").copy(); f["_key"] = f.forecast_date - pd.Timedelta(days=NDVI_LATENCY)
    o = obs.rename(columns={"date": "ndvi_obs_date", "ndvi": "ndvi_last", "ndvi_valid_frac": "ndvi_valid_frac_last"}).sort_values("ndvi_obs_date")
    m = pd.merge_asof(f, o[["cell_id", "ndvi_obs_date", "ndvi_last", "ndvi_valid_frac_last"]], left_on="_key", right_on="ndvi_obs_date", by="cell_id", tolerance=pd.Timedelta(days=NDVI_MAX_AGE - NDVI_LATENCY))
    m["ndvi_age_days"] = (m.forecast_date - m.ndvi_obs_date).dt.days; return m.drop(columns="_key")

def build(d=None, normals=None, verbose=True):
    d = load_grid() if d is None else d; normals = normals_table(d) if normals is None else normals
    obs = d.loc[d.ndvi.notna(), ["cell_id", "date", "ndvi", "ndvi_valid_frac"]]; parts = []
    for c, g in d.groupby("cell_id"): parts.append(add_ndvi(cell_features(g, normals[normals.cell_id == c].set_index("doy").sort_index()), obs[obs.cell_id == c]))
    f = pd.concat(parts, ignore_index=True).sort_values(["forecast_date", "cell_id"]).reset_index(drop=True)
    q = {"rows_before_filters": int(len(f))}
    f["year"] = f.forecast_date.dt.year; f["split"] = f.year.map(split_of)
    ins = f.forecast_date.dt.month.between(*SEASON); q["excluded_outside_growing_season"] = int((~ins).sum()); f = f[ins]
    need = ["z_l2_t", "z_l2_mean7", "z_l2_delta3", "temp_mean_7d", "temp_max_7d", "precip_sum_14d", "label_dry_t3_t7"]; ok = f[need].notna().all(axis=1)
    q["excluded_missing_history_or_label"] = int((~ok).sum()); q["excluded_dates_examples"] = sorted(f.loc[~ok, "forecast_date"].dt.strftime("%Y-%m-%d").unique().tolist())[:10]; f = f[ok]
    f["label_dry_t3_t7"] = f.label_dry_t3_t7.astype(int); f["label_window_start"] = f.forecast_date + pd.Timedelta(days=LEAD_MIN); f["label_window_end"] = f.forecast_date + pd.Timedelta(days=LEAD_MAX)
    f = f.reset_index(drop=True); q["rows"] = int(len(f)); q["positive_rate_pct"] = round(float(f.label_dry_t3_t7.mean() * 100), 2)
    q["missing_values"] = {c: int(v) for c, v in f.isna().sum().items() if v}; return f, q

FEATURES = ["latitude", "longitude", "cropland_km2", "doy_sin", "doy_cos", "sm_l2_t", "z_l1_t", "z_l2_t", "z_l2_mean3", "z_l2_mean7", "z_l2_delta3", "temp_mean_7d", "temp_max_7d",
            "precip_sum_7d", "precip_sum_14d", "days_since_rain", "ndvi_last", "ndvi_age_days", "ndvi_valid_frac_last"]
DEFS = [("latitude, longitude, cropland_km2", "static", "cell centre and cropland area", "constant", "Day 4 grid"),
        ("doy_sin, doy_cos", "calendar", "sin/cos of day of year of t", "date t", "calendar"),
        ("sm_l2_t", "soil", "ERA5-Land volumetric soil water 7-28 cm on day t (m3/m3)", "day t", "ERA5-Land swvl2"),
        ("z_l1_t, z_l2_t", "soil", "anomaly z of layers 0-7 / 7-28 cm on day t, vs the TRAINING-YEARS normal for that cell and day of year", "day t", "ERA5-Land + normals 2018-2022"),
        ("z_l2_mean3, z_l2_mean7", "soil", "mean of z_l2 over days t-2..t and t-6..t", "days t-6..t", "ERA5-Land + normals"),
        ("z_l2_delta3", "soil", "z_l2 on day t minus z_l2 on day t-3", "days t-3, t", "ERA5-Land + normals"),
        ("temp_mean_7d", "weather", "mean 2 m air temperature over days t-6..t (degC)", "days t-6..t", "ERA5-Land"),
        ("temp_max_7d", "weather", "maximum of the daily maximum temperature over days t-6..t (degC)", "days t-6..t", "ERA5-Land"),
        ("precip_sum_7d, precip_sum_14d", "weather", "total precipitation over days t-6..t and t-13..t (mm)", "days t-13..t", "ERA5-Land"),
        ("days_since_rain", "weather", "days since the last day with >= 1 mm precipitation, capped at 60", "days <= t", "ERA5-Land"),
        ("ndvi_last", "vegetation", f"latest NDVI of the cell acquired >= {NDVI_LATENCY} days before t and <= {NDVI_MAX_AGE} days old; empty if none (never filled)", f"obs date <= t-{NDVI_LATENCY}", "Sentinel-2 (Apr 2023+ only)"),
        ("ndvi_age_days, ndvi_obs_date, ndvi_valid_frac_last", "vegetation", "age of that observation, its acquisition date, and the clear share of the cell's cropland", f"obs date <= t-{NDVI_LATENCY}", "Sentinel-2"),
        ("z_l2_future_mean_t3_t7", "TARGET (continuous)", "mean z_l2 over days t+3..t+7", "days t+3..t+7", "ERA5-Land + normals"),
        ("label_dry_t3_t7", "TARGET (binary)", f"1 if z_l2_future_mean_t3_t7 <= {THRESH}", "days t+3..t+7", "derived"),
        ("label_window_start, label_window_end, split, year", "bookkeeping", "dates of the label window; split = train 2018-22 / val 2023 / test 2024", "-", "-")]

def episodes(f, frac=EPISODE_FRAC):
    """number of district-level positive episodes: runs of consecutive forecast dates on which >= frac of the cells are positive."""
    fr = f.groupby("forecast_date").label_dry_t3_t7.mean(); ep, on = 0, False
    for v in fr.values:
        if v >= frac and not on: ep += 1; on = True
        if v < frac: on = False
    return ep, int((fr >= frac).sum())

def split_table(f):
    rows = []
    for y, g in f.groupby("year"):
        ep, nd = episodes(g); pos = int(g.label_dry_t3_t7.sum())
        rows.append({"year": int(y), "split": split_of(int(y)), "rows": len(g), "positive_cell_days": pos, "positive_rate_pct": round(pos / len(g) * 100, 2), "positive_episodes": ep, "days_with_>=10%_cells_positive": nd,
                     "adequate": bool(pos >= MIN_POS_CELLDAYS and ep >= MIN_EPISODES)})
    return pd.DataFrame(rows)

def auc(score, y):
    from scipy.stats import rankdata
    y = np.asarray(y); n1 = int(y.sum()); n0 = len(y) - n1
    return np.nan if n1 == 0 or n0 == 0 else (rankdata(score)[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

def persistence_metrics(g):
    """persistence baseline: dry today (z_l2_t) predicts dry in 3-7 days. Score = -z_l2_t; binary rule z_l2_t <= -1."""
    y = g.label_dry_t3_t7.values; pred = (g.z_l2_t.values <= THRESH).astype(int); tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else np.nan; rec = tp / (tp + fn) if tp + fn else np.nan; f1 = 2 * prec * rec / (prec + rec) if prec and rec else np.nan
    return {"rows": len(g), "base_rate_%": round(y.mean() * 100, 2), "AUC": auc(-g.z_l2_t.values, y), "precision": prec, "recall": rec, "F1": f1, "lift_vs_base": prec / y.mean() if y.mean() > 0 and prec == prec else np.nan}

def bootstrap_auc(g, n=1000, block=14, seed=0):
    """AUC 95% interval resampling 14-day blocks of forecast dates (all cells of a date stay together)."""
    rng = np.random.default_rng(seed); dates = np.sort(g.forecast_date.unique()); blocks = [dates[i:i + block] for i in range(0, len(dates), block)]; by = {d_: i for i, d_ in enumerate(dates)}
    idx_by_block = [np.where(g.forecast_date.isin(b).values)[0] for b in blocks]; s, y = -g.z_l2_t.values, g.label_dry_t3_t7.values; out = []
    for _ in range(n):
        pick = rng.integers(0, len(blocks), len(blocks)); ii = np.concatenate([idx_by_block[k] for k in pick]); a = auc(s[ii], y[ii])
        if a == a: out.append(a)
    return (float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))) if out else (np.nan, np.nan)

def main():
    d = load_grid(); normals = normals_table(d); f, q = build(d, normals)
    f.to_parquet(OUT["features"], index=False); normals.round(5).to_csv(OUT["baseline"], index=False)
    pd.DataFrame(DEFS, columns=["feature(s)", "group", "definition", "observation window", "source"]).to_csv(OUT["defs"], index=False)
    st = split_table(f); st.to_csv(OUT["splits"], index=False); q["splits"] = st.to_dict("records")
    json.dump(q, open(OUT["quality"], "w"), indent=1, default=str)
    print(f"rows {q['rows']:,} | positive rate {q['positive_rate_pct']} % | excluded outside season {q['excluded_outside_growing_season']:,}, missing history/label {q['excluded_missing_history_or_label']}")
    print(st.to_string(index=False)); print("saved", *OUT.values(), sep="\n  ")

if __name__ == "__main__": main()
