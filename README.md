🇰🇬 KG Water Intelligence

An ML project exploring whether agricultural water stress in Kyrgyzstan can be predicted using weather, satellite, and environmental data.

Initial scope: Chüy region.

Initial goal: build a baseline model that estimates the probability of water stress for agricultural fields.

## Data Sources

### 1. ERA5-Land
Source: Copernicus Climate Data Store
Purpose: weather and soil-water conditions
Variables:
- 2m temperature
- precipitation
- volumetric soil water
- solar radiation

### 2. Sentinel-2
Source: Copernicus Data Space Ecosystem
Purpose: vegetation and surface monitoring
Initial derived feature:
- NDVI

### 3. SoilGrids
Source: ISRIC
Purpose: soil characteristics
Potential variables:
- clay
- sand
- silt
- organic carbon
- bulk density

## Initial hypothesis

Weather + soil-water + vegetation + soil characteristics
may contain enough information to detect agricultural
water stress.

## Study Area

### Region
Chüy Valley, Kyrgyzstan

### Initial study area

Bounding box:

- North: 43.10° N
- South: 42.70° N
- West: 74.20° E
- East: 74.90° E

### Initial crop

Wheat

### Why Chüy?

Chüy is one of Kyrgyzstan's major agricultural regions
and has extensive irrigated agriculture.

### Initial research question

Can weather, satellite, and soil/environmental data
be used to detect or predict water stress in
wheat-growing areas of the Chüy Valley?

### Important limitation

The bounding box contains both agricultural and
non-agricultural land. An agricultural/land-cover
mask will be applied before modeling.

---

## Day 1 — Study area and cropland mask

**Study area:** Sokuluk District, Chüy Region (≈2,011 km²), boundary from geoBoundaries ADM2
(`data/boundaries/sokuluk_district.geojson`). This replaces the rough bounding box above for all processing.

**Cropland mask:** `data/processed/sokuluk_cropland.tif` — ESA WorldCover 2021 v200 (tile N42E072), clipped to
Sokuluk, `1 = cropland` (class 40), `0 = other land`. EPSG:4326, ~0.00008333° pixels (≈9.3 × 6.8 m at this latitude).

**Result:** ~855 km² of cropland, about 42.5% of the district, concentrated in the northern irrigated plain.
The southern mountain/foothill part is almost entirely non-cropland.

![Sokuluk cropland mask](reports/figures/sokuluk_cropland_map.png)

Reproduce the map and statistics: `python3 scripts/make_day1_map.py`

### Data sources (Day 1)
| Data | Source | License |
|---|---|---|
| Admin boundary | geoBoundaries ADM2 (KGZ) | CC BY 4.0 |
| Land cover | ESA WorldCover 2021 v200, 10 m | CC BY 4.0 |

### Limitations
- WorldCover class 40 means *cropland*, not *wheat*, and not necessarily *irrigated*. Crop type is unknown.
- It is a single-year (2021) snapshot; fields change with rotation and abandonment. Overall accuracy is
  roughly 75–80% globally and lower for small fields; edges between fields, orchards and grassland are noisy.
- The geoBoundaries polygon is generalized (administrative, not survey-grade); the mask is clipped to it.
- The raster stores `nodata = 0`, which is the same value as "other land". Re-write the file without a
  nodata value (or use 255) before using it in tools that honour nodata.
- No water-stress label exists yet. Defining how stress is measured (e.g. soil-moisture anomaly, NDVI
  anomaly, evapotranspiration deficit) is the next step, before any model training.

### Next
Collect ERA5-Land weather and soil-water data and SoilGrids soil properties for Sokuluk; decide the water-stress definition.


---

## Day 2 — Historical weather for Sokuluk

**Data:** ERA5-Land hourly reanalysis (Copernicus Climate Data Store, ECMWF), ~9 km grid, downloaded for the box 42.4–43.3 °N, 74.0–74.7 °E.
**Period:** 1 Apr 2023 – 31 Dec 2024 (641 days, no gaps). Longer history (2018–2022) is not downloaded yet.
**Output:** `data/processed/sokuluk_weather_daily.csv`

| Column | Meaning | Unit |
|---|---|---|
| `date` | local day, Kyrgyzstan time (UTC+6) | |
| `temperature_2m_c` / `_min_c` / `_max_c` | 2 m air temperature: daily mean / min / max of hourly values | °C |
| `precipitation_mm` | daily total precipitation | mm/day |
| `soil_moisture_l1_m3m3` / `_l2_m3m3` | volumetric soil water, layer 1 (0–7 cm) and layer 2 (7–28 cm), daily mean | m³/m³ |

### Processing (`scripts/download_era5land.py`, `scripts/process_era5land.py`)
1. One CDS request per month (a full year exceeds the request-size limit); 2 m temperature, total precipitation, soil water layers 1–2.
2. K → °C; m → mm. ERA5-Land precipitation is an accumulation that resets at 00 UTC, so it is converted to hourly amounts (negatives clipped to 0) and summed per local day.
3. The 36 grid cells that contain cropland are averaged **weighted by the cropland area in each cell** (from the Day 1 mask, 855 km² in total), so the series describes the weather over the farmland, not the mountains.
4. A day is kept only if all 24 hours are present. Checked: no missing values, no date gaps.

### First look (`scripts/plot_weather.py`)
![Temperature](reports/figures/sokuluk_temperature.png)
![Precipitation](reports/figures/sokuluk_precipitation.png)

- Strong seasonal cycle: daily mean from about −20 °C (Dec 2023) to about 31 °C (Jul 2023); hottest day 38 °C (13 Jul 2023).
- Growing season (Apr–Oct) 2023 was **drier and hotter** than 2024: ~290 mm vs ~467 mm of rain, and 18 vs 7 days with a maximum ≥ 35 °C.
- Five **dry spells** (≥ 14 days under 1 mm, Apr–Oct) — listed in `reports/dry_spells.csv`: May and Jun 2023, Jul–Aug 2023, late May–Jun 2024, Jul–Aug 2024.

### What this data can and cannot tell us about water stress
**Can:** show when it was hot and dry, how long dry spells lasted, and roughly how wet the top soil was — the weather side of water stress, with daily resolution.

**Cannot:**
- Tell whether a *specific field* needs water. ERA5-Land cells are ~9 km wide, so one number covers many fields, crops and irrigation schedules.
- See **irrigation**. Much of Chüy's cropland is irrigated from canals, so a dry spell does not mean dry fields. This is the biggest gap.
- Measure the plant itself. Crop stress needs satellite vegetation data (Day 3, Sentinel-2 NDVI) and a way to validate it.
- Be treated as station-quality. It is a model product: temperature mixes valley and foothill elevations, and ERA5-Land precipitation can have biases in mountain regions — the 2024 total (~756 mm) looks high for the Chüy plain and should be checked against weather-station data before it is used quantitatively.

### Next
Day 3: Sentinel-2 vegetation (NDVI) over the cropland mask. Later: align weather, vegetation and cropland in space and time, then define a defensible water-stress indicator before any model.
