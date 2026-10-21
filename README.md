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
