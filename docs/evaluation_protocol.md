# Evaluation protocol — soil-moisture drought forecast, 3–7 days ahead (Day 7)

Pipeline: [`scripts/build_features.py`](../scripts/build_features.py) · audit and results: [`notebooks/07_feature_dataset.ipynb`](../notebooks/07_feature_dataset.ipynb) · target definition: [`water_stress_target.md`](water_stress_target.md).
Reproduce: `python3 scripts/build_features.py`, then execute the notebook.

## 1. What is predicted

| | |
|---|---|
| Unit | one ERA5-Land grid cell (30 cells with cropland) on one **forecast date t**, April–October |
| Question | "Will the root-zone soil (7–28 cm) be unusually dry 3–7 days from now?" |
| **Label** | `label_dry_t3_t7 = 1` if the mean soil-moisture anomaly z over days **t+3 … t+7** is ≤ −1 (also stored continuous as `z_l2_future_mean_t3_t7`) |
| Anomaly z | (soil moisture − normal) / spread for the cell and day of year; **the normal and spread come from the training years 2018–2022 only** and are then applied to every year (`data/processed/soil_moisture_normal_train2018_2022.csv`) |
| Forecast time | the **end of local day t**: a daily value is complete only when the day is over, so features may use days ≤ t and nothing later |

## 2. Features and their timing

Every feature is a trailing window ending at t (full list with windows in `data/processed/water_stress_feature_definitions.csv`): current soil moisture and anomaly (days t, t−2…t, t−6…t, change over 3 days), temperature and precipitation over the previous 7 and 14 days, days since rain, calendar (sin/cos of day of year), and static cell attributes. **NDVI** is the latest observation acquired at least **2 days** before t (processing latency) and at most **21 days** old, with its acquisition date and age stored; it is **never interpolated or filled**. The target is stored in separate columns, with `label_window_start/end` recording its dates.

## 3. Leakage audit (automated, all pass)

| Test | What it proves |
|---|---|
| T1 truncation | deleting all data after t and recomputing 19 features for 240 random rows gives identical values (0 differences); the label cannot even be computed without future data |
| T2 normal isolation | changing 2023/2024 soil-moisture values changes neither the normal nor any training-year feature or label |
| T3 NDVI timing | every NDVI value is 2–21 days old, none from the future |
| T4 label window | always t+3 … t+7: a 3-day gap after t, 4-day span |
| T5 chronology | last training label day (7 Nov 2022) precedes the first validation date; same for validation → test |
| T6 persistence | the baseline reads only `z_l2_t`: shuffling every other feature leaves its AUC unchanged |

**A leak found and fixed:** the Day 6 anomaly used a leave-one-year-out normal, which includes *future* years. Day 6 reported a persistence AUC of 0.93; with the leakage-safe normal the pooled value is **0.912** (by year the Day 6 value was higher by 0.00–0.07, mostly in training years). Day 6's anomaly file is still reproducible (max difference 0.0005 = rounding) and remains valid as a *descriptive* anomaly, but it must not be used for modelling.

## 4. Splits

Chronological by year — never random by row. **Train 2018–2022, validation 2023, test 2024.**

| Year | Split | Rows | Positive cell-days | Positive rate | Independent episodes | Adequate |
|---|---|---|---|---|---|---|
| 2018 | train | 6,420 | 137 | 2.1 % | 4 | no |
| 2019 | train | 6,420 | 693 | 10.8 % | 5 | yes |
| 2020 | train | 6,420 | 511 | 8.0 % | 5 | yes |
| 2021 | train | 6,420 | 1,135 | 17.7 % | 8 | yes |
| 2022 | train | 6,420 | 431 | 6.7 % | 6 | yes |
| **2023** | **validation** | 6,420 | 1,523 | 23.7 % | 3 | **yes** |
| **2024** | **test** | 6,420 | 114 | 1.8 % | 1 | **no** |

**Adequacy rule (fixed before the results were looked at):** a year is adequate for evaluation if it has ≥ 200 positive cell-days **and** ≥ 3 independent positive episodes (an episode = a run of consecutive forecast dates on which ≥ 10 % of cells are positive; cell-days within one dry spell are strongly dependent).

**Decision:** the planned split is kept. **2023 is adequate. 2024 is not** — a wet year with one short dry episode (114 positive cell-days). Its results are reported but are **descriptive only**; years are not moved between splits to obtain a more comfortable test set. NDVI is absent from every training row (NDVI exists only from April 2023: available on 79.5 % of 2023 rows and 83.0 % of 2024 rows), so a first model cannot use it.

## 5. Metrics

Primary: **AUC** (ranking of dry-soon cell-days) with a 95 % interval from resampling 14-day blocks of dates (cells of one date stay together). Secondary, at the rule z ≤ −1: **precision, recall, F1** and **lift** over the base rate. Always report the base rate, the number of positives and the number of episodes alongside any score. Models and baselines are evaluated on **exactly the same rows** (the split rows of the saved dataset).

## 6. Persistence baseline ("dry today → dry later")

Score = −`z_l2_t`; binary rule `z_l2_t ≤ −1`; no fitting, so it uses no training labels.

| Split | Base rate | AUC (95 % interval) | Precision | Recall | F1 | Lift |
|---|---|---|---|---|---|---|
| train 2018–22 (in-sample) | 9.1 % | 0.878 | 0.54 | 0.60 | 0.56 | 5.9 |
| **validation 2023** | 23.7 % | **0.939** (0.89–0.97) | 0.76 | 0.82 | 0.79 | 3.2 |
| test 2024 *(not adequate)* | 1.8 % | 0.961 (0.93–1.00) | 0.56 | 0.62 | 0.59 | 31.5 |

Persistence is a **strong** baseline because soil moisture changes slowly. Its weakness is timing: the share of cells that are dry *today* trails the share that *will* be dry in 3–7 days by several days (figure `features_persistence_timeline.png`), so a useful model must anticipate onset and recovery from rain and temperature. Day 8 succeeds only if it beats persistence on the same validation rows (and, descriptively, the test rows).

## 7. Rules for Day 8 and later

- Choose features, hyperparameters and thresholds on **validation (2023)** only; look at 2024 once, at the end.
- In any cross-validation across training years, **recompute the normal inside each fold from that fold's training years** (the stored normal contains all of 2018–2022).
- No feature may use data after t; re-run the leakage tests after adding features.
- Report the base rate, positives and episodes with every score; do not claim skill from the 2024 numbers.
- Describe results as "probability of unusually dry soil", not "water stress", until independent validation exists (see the target definition).

## 8. Dataset quality and limitations

- **Quality:** 44,940 rows × 28 columns (30 cells × 214 forecast dates × 7 years); no duplicates; **no missing values except NDVI** (34,508 rows without NDVI = all of 2018–2022 plus rows with no observation within 21 days); positive-label rate 10.11 %; **excluded:** 31,740 rows outside April–October (dormant season); 0 rows excluded for missing history or label. Saved: `water_stress_features.parquet`, `water_stress_feature_definitions.csv`, `water_stress_splits.csv`, `reports/feature_dataset_quality.json`.
- **Few independent events:** only 1–8 positive episodes per year; row counts overstate the information. Seven years means results vary a lot by year (persistence AUC 0.82–0.96).
- **2024 test set too thin** (section 4); **2023 is a single validation year.**
- **The target is model-made.** ERA5-Land soil moisture is computed from the same rain and temperature used as features, so predicting it partly re-learns the land model; this does not show that real fields are stressed. Independent validation (satellite soil moisture, evapotranspiration, an NDVI baseline, station and agronomic records) is still required.
- **Coarse resolution:** ~9 km cells; no irrigation; the 7–28 cm layer is shallower than a crop's root zone.
- **No NDVI for training**, and no forecast weather: features describe the past only, so any skill must come from persistence of soil moisture plus recent rain and temperature.
