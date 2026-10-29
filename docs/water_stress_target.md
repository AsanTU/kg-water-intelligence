# Water-stress target — provisional definition (Day 6)

**Status:** provisional, for the first research prototype. Evidence and figures: [`notebooks/06_water_stress_target.ipynb`](../notebooks/06_water_stress_target.ipynb). Exploratory anomaly values: `data/processed/soil_moisture_anomaly_loyo.csv.gz`.

## 1. What the target measures

**Unusually dry root-zone soil for the place and time of year** — a *soil-moisture anomaly*. It is an **indicator of agricultural drought conditions**, not proof that a field needs irrigation, and not a measurement of plant stress.

It is deliberately **not** defined from NDVI. Low NDVI also results from crop type, sowing date, ripening and harvest, disease, and the season itself (Day 5), so a field is never called "water-stressed" only because its NDVI is low. NDVI is used only as an *independent check* (section 5).

## 2. How it is calculated

1. **Variable:** ERA5-Land volumetric soil water, layer 2 (7–28 cm), m³/m³, daily, per ERA5-Land grid cell (30 cells with cropland). *Planned:* layer 3 (28–100 cm), closer to a crop's root zone, once downloaded.
2. **Normal for the date:** for each cell and each day of year, the mean μ and standard deviation σ of the soil moisture of that day ±15 days in the *other* years (σ floored at 0.01 m³/m³).
3. **Anomaly:** `z = (soil moisture − μ) / σ`. Negative z = drier than usual for the date.
4. **Season:** only the growing season (April–October) is used. In winter the soil is frozen or snow-covered and the model value says little about crop water.
5. **Dry day / dry event:** a *dry day* has `z ≤ −1`; a *dry event* is ≥ 7 consecutive dry days.
6. **Forecast label (for a later model):** at forecast date *t* (Apr–Oct), `label = 1` if the mean anomaly over days *t+3 … t+7* is `≤ −1`, otherwise 0 — "will the soil be unusually dry 3–7 days from now?".

**Baseline rule for training:** in exploration the normal for a year is computed from the other years (leave-one-year-out). For model training and testing the normal must be computed **from the training years only**; a test year must never be part of its own baseline.

## 3. Data used

| Role | Data | Notes |
|---|---|---|
| **Target** | ERA5-Land soil water, layers 1–2 (layer 3 planned) | ~9 km, model output, 2018–2024, complete (0 missing) |
| Predictors (later) | ERA5-Land precipitation, temperature; soil-moisture history | information up to day *t* only |
| **Independent check** | Sentinel-2 NDVI (Apr 2023 – Dec 2024 so far) | never used to define the target |
| Units | cropland mask / cell weights | Day 1 / Day 4 |

## 4. What the data shows so far

- **Strong seasonal cycle** — layer 2 averages 0.40 m³/m³ in March and 0.21 in July–August. An absolute value is therefore meaningless as a stress signal; only the anomaly for the date is usable.
- **Plausible at the year level.** Mean growing-season anomaly: 2021 −0.50 and 2023 −0.41 (dry), 2024 +0.80 (wet), other years near 0. These are the years the Day 2 rain totals called driest (253 and 290 mm) and wettest (467 mm).
- **Dry events (z ≤ −1 for ≥ 7 days, Apr–Oct):** 6 in 7 years — Oct 2020, May–Jul 2021 (42 days), Apr 2022, Apr 2023, and two in May–June 2023.
- **Base rate of the forecast label:** 13.2 % of Apr–Oct cell-days (by year: 4, 14, 12, 23, 12, 25, 2 %). The trivial **persistence rule** ("dry today, so dry later") already reaches **AUC 0.93** (0.87–0.96 by year), so a model must beat that.
- **Sensitivity:** the number of events depends on the choice (threshold −1.5 / −1 / −0.5 with ≥ 7 days: 2 / 6 / 13), so the thresholds are provisional.
- **Follows recent rain:** Spearman ρ = +0.53 with rain of the prior 30 days (temperature −0.10). During the 16 rain-based dry spells of Day 2 the mean anomaly is −0.27 (negative in 9 of 16; strongly negative in spring 2023 and Oct 2020, but mild or even positive in many summer spells, where the soil is already dry or snowmelt still contributes). The long July–August 2019 rain spell (36 days) is *not* flagged as a dry event.
- **Consistency with vegetation — weak.** 2024 minus 2023, by month: NDVI and anomaly move in the same direction (district r = +0.73, n = 7 months; better than rain +0.20 or temperature −0.23). But the cell-level pooled correlation (ρ = +0.59) is the spring-vs-autumn pattern: **within a month it disappears** (median ρ = −0.11, positive in 3 of 7 months; month-demeaned ρ = −0.06). So the anomaly has **not** been shown to explain differences between cells.

## 5. Limitations

1. **Not independent of the predictors.** ERA5-Land soil moisture is *computed by a land-surface model from the same rain and temperature* we would use as features. Predicting it from weather partly re-learns that model, and agreement with rain is partly built in. Strong results here would not prove skill at detecting real stress.
2. **Coarse, model-based, not ground truth:** ~9 km cells, one value for many fields, crops and soils; no field-level observation.
3. **Irrigation is invisible.** Much of Chüy's cropland is irrigated from canals (snowmelt-fed); the model has no irrigation, so "dry soil" may be watered fields. The target describes *rain-fed dryness*, not what a farmer experiences.
4. **Depth:** 7–28 cm is shallower than a cereal root zone; the 28–100 cm layer should be preferred.
5. **Short baseline:** 6 other years per date; the normal and σ are noisy, and σ is floored.
6. **Freezing/snow** limit the winter signal (hence April–October only). **Mixed information in NDVI:** it cannot validate the target by itself (section 6).

## 6. How it will be validated (independent observations)

| Independent evidence | What it checks | Resolution | Status |
|---|---|---|---|
| Satellite soil moisture, e.g. Copernicus Sentinel-1 surface soil moisture / ESA CCI / SMAP | do independent retrievals agree with the ERA5-Land anomaly? | 1–36 km | to obtain |
| Evapotranspiration / land-surface temperature anomalies (e.g. WaPOR, MODIS/Landsat) | does the land actually transpire less / run hotter when flagged? | 30 m – 1 km | to obtain |
| **NDVI anomaly vs a multi-year baseline** | do flagged periods coincide with below-normal greenness, *within a month and across cells*? | 10–80 m | needs Sentinel-2 back to 2018 |
| Station rain / drought indices (SPI, SPEI) from Kyrgyz Hydromet | independent check of dryness and of the ERA5 rain bias | point | to request |
| Yield statistics, irrigation records, farmer or agronomist reports | does the flag relate to real losses or irrigation need? | district / field | to request |

**Planned tests:** event-level hit rate and false-alarm rate against NDVI-anomaly events; within-month rank correlations across cells; agreement with at least one independent soil-moisture product; all on held-out years. **The target should be revised or rejected if** it shows no agreement with any independent source, or if it is merely re-detecting the rain deficit.

## 7. Rules for using it in modelling

- Baseline and thresholds fixed from training years only; test years held out in time.
- Features use information up to day *t* only (no future weather, no future soil moisture).
- Report the persistence baseline (AUC ≈ 0.93) and the gain over it at 3–7 days; do not claim stress detection from agreement with ERA5-Land alone.
- Describe results as "probability of unusually dry soil" until independent validation exists.

## 8. Next steps

1. Download ERA5-Land layer 3 (28–100 cm) and re-check the target at root-zone depth.
2. Extend Sentinel-2 NDVI back to 2018 to build a seasonal NDVI baseline for an independent anomaly.
3. Obtain at least one independent soil-moisture or evapotranspiration product; ask for station/agronomic data.
4. Day 7: a reproducible pipeline that builds weather, soil-moisture anomaly and satellite features in one run.
