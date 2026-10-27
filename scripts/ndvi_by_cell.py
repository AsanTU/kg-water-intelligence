"""Day 4: Sentinel-2 NDVI aggregated PER ERA5-Land GRID CELL (the weather grid), per scene -> data/raw/sentinel2_ndvi_cells.jsonl (resumable)

Same method as scripts/ndvi_sentinel2.py (80 m grid, SCL 4/5 only, cropland pixels = >= 60 % cropland), but instead of one district total it keeps,
for each scene, the number of clear cropland pixels and the NDVI sum in each of the 80 ERA5-Land cells (10 lat x 8 lon, 0.1 deg). A pixel belongs to the
ERA5 cell whose centre is nearest — the same rule used for the cropland weights in Day 2. Cell id = lat_index * 8 + lon_index (lat index 0 = 43.3 N, lon index 0 = 74.0 E).
Usage: python3 scripts/ndvi_by_cell.py [start end max_cloud]    default 2023-04-01 2024-12-31 40
"""
import sys, json, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np, rasterio
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import reproject, transform_bounds
from rasterio.windows import from_bounds
from pystac_client import Client

START, END, MAXCLOUD = (sys.argv[1], sys.argv[2], float(sys.argv[3])) if len(sys.argv) > 3 else ("2023-04-01", "2024-12-31", 40.0)
RES, CROP_FRAC = 80.0, 0.6
BBOX = [74.12, 42.43, 74.56, 43.25]; MASK = "data/processed/sokuluk_cropland.tif"; CACHE = Path("data/raw/sentinel2_ndvi_cells.jsonl")
LAT0, LON0, STEP, NLAT, NLON = 43.3, 74.0, 0.1, 10, 8
ENV = dict(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", AWS_NO_SIGN_REQUEST="YES", GDAL_HTTP_MAX_RETRY="4", GDAL_HTTP_RETRY_DELAY="1", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif")

l, b, r, t = transform_bounds("EPSG:4326", "EPSG:32643", *BBOX)
l, t = np.floor(l / RES) * RES, np.ceil(t / RES) * RES; r, b = np.ceil(r / RES) * RES, np.floor(b / RES) * RES
GW, GH = int((r - l) / RES), int((t - b) / RES); GT = from_origin(l, t, RES, RES)
frac = np.zeros((GH, GW), "float32")
with rasterio.open(MASK) as s:   # plain array: the file declares nodata=0 which would break "average"
    reproject(s.read(1).astype("float32"), frac, src_transform=s.transform, src_crs=s.crs, dst_transform=GT, dst_crs="EPSG:32643", resampling=Resampling.average, src_nodata=None, dst_nodata=None)
CROP = frac >= CROP_FRAC
# pixel centre (UTM) -> lon/lat -> nearest ERA5-Land cell id
xs = l + (np.arange(GW) + .5) * RES; ys = t - (np.arange(GH) + .5) * RES; X, Y = np.meshgrid(xs, ys)
lon, lat = Transformer.from_crs("EPSG:32643", "EPSG:4326", always_xy=True).transform(X, Y)
li, lj = np.round((LAT0 - lat) / STEP).astype(int), np.round((lon - LON0) / STEP).astype(int)
CELL = np.where((li >= 0) & (li < NLAT) & (lj >= 0) & (lj < NLON), li * NLON + lj, -1)
NCELL = NLAT * NLON; NCROP = np.bincount(CELL[CROP & (CELL >= 0)], minlength=NCELL)
print(f"grid {GW}x{GH} @ {RES:.0f} m; cropland px {int(CROP.sum())}; cells with cropland: {(NCROP > 0).sum()}", flush=True)

def read_band(href):
    with rasterio.Env(**ENV), rasterio.open(href) as src:
        win = from_bounds(l, b, r, t, transform=src.transform)
        return src.read(1, window=win, out_shape=(GH, GW), boundless=True, fill_value=0, resampling=Resampling.nearest)

def process(item):
    try:
        a = item.assets; off = (a["red"].extra_fields.get("raster:bands") or [{}])[0].get("offset", 0.0)
        red, nir, scl = read_band(a["red"].href), read_band(a["nir"].href), read_band(a["scl"].href)
        ok = CROP & (CELL >= 0) & np.isin(scl, (4, 5)) & (red > 0) & (nir > 0)
        rf, nf = (red[ok] + off) / 1e4, (nir[ok] + off) / 1e4; nd = (nf - rf) / np.maximum(nf + rf, 1e-6); c = CELL[ok]
        return dict(id=item.id, date=item.datetime.strftime("%Y-%m-%d"), tile=item.properties.get("grid:code"), cloud=round(item.properties["eo:cloud_cover"], 1),
                    n_valid=np.bincount(c, minlength=NCELL).tolist(), ndvi_sum=[round(float(v), 4) for v in np.bincount(c, weights=nd, minlength=NCELL)])
    except Exception as e:
        return dict(id=item.id, error=str(e)[:200])

def main():
    items = list(Client.open("https://earth-search.aws.element84.com/v1").search(
        collections=["sentinel-2-l2a"], bbox=BBOX, datetime=f"{START}/{END}", query={"eo:cloud_cover": {"lt": MAXCLOUD}}, max_items=2000).items())
    done = {j["id"] for j in map(json.loads, CACHE.read_text().splitlines()) if "error" not in j} if CACHE.exists() else set()
    todo = [i for i in items if i.id not in done]; print(f"{len(items)} scenes, {len(done)} cached, {len(todo)} to process", flush=True); t0 = time.time()
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(8) as ex, CACHE.open("a") as f:
        for k, res in enumerate(ex.map(process, todo), 1):
            if "error" in res: print("ERR", res, flush=True); continue
            f.write(json.dumps(res) + "\n"); f.flush()
            if k % 20 == 0 or k == len(todo): print(f"{k}/{len(todo)}  {time.time()-t0:.0f}s", flush=True)
    print("done", flush=True)

if __name__ == "__main__": main()
