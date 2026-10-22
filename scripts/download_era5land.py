"""Day 2: download hourly ERA5-Land for the Sokuluk area from the Copernicus Climate Data Store (CDS).

CDS limits request size (one month x 4 variables x hourly fits; a whole year does not), so this makes ONE REQUEST PER MONTH,
newest first, 3 in parallel. Resumable: finished months are skipped. Needs ~/.cdsapirc and the accepted ERA5-Land licence.
Usage: python3 scripts/download_era5land.py [first_year last_year]      (default 2018 2024)
"""
import calendar, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import cdsapi

OUT = Path("data/raw/era5land")
AREA = [43.3, 74.0, 42.4, 74.7]  # N, W, S, E — covers Sokuluk (42.43-43.24 N, 74.13-74.56 E) with one ERA5-Land cell of margin
VARIABLES = ["2m_temperature", "total_precipitation", "volumetric_soil_water_layer_1", "volumetric_soil_water_layer_2"]
WORKERS = 3

def fetch(ym):
    year, month = ym
    target = OUT / f"era5land_{year}-{month:02d}.nc"
    if target.exists() and target.stat().st_size > 0:
        return f"skip {target.name}"
    part = target.with_suffix(".part")
    ndays = calendar.monthrange(year, month)[1]
    cdsapi.Client(quiet=True).retrieve("reanalysis-era5-land", {
        "variable": VARIABLES, "year": str(year), "month": f"{month:02d}",
        "day": [f"{d:02d}" for d in range(1, ndays + 1)], "time": [f"{h:02d}:00" for h in range(24)], "area": AREA,
        "data_format": "netcdf", "download_format": "unarchived",
    }, str(part))
    part.rename(target)
    return f"saved {target.name} ({target.stat().st_size/1e6:.1f} MB)"

def main(y0, y1):
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(y, m) for y in range(y1, y0 - 1, -1) for m in range(12, 0, -1)]
    print(f"{len(jobs)} monthly requests, {WORKERS} in parallel", flush=True)
    with ThreadPoolExecutor(WORKERS) as ex:
        for i, msg in enumerate(ex.map(fetch, jobs), 1):
            print(f"[{i}/{len(jobs)}] {msg}", flush=True)

if __name__ == "__main__":
    a = [int(x) for x in sys.argv[1:3]]
    main(*(a if len(a) == 2 else (2018, 2024)))
