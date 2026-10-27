"""Day 3: Sentinel-2 L2A NDVI over the Sokuluk cropland mask -> data/processed/sokuluk_ndvi_daily.csv

Source: Sentinel-2 L2A COGs on AWS (Earth Search STAC, open, no key). NDVI = (NIR-Red)/(NIR+Red) from B08/B04 reflectance
((DN + offset)/10000, offset from the STAC metadata). Read at 80 m (COG overview level 8) onto one UTM-43N grid covering the district.
Cropland pixels = grid cells with >= 60 % cropland (WorldCover mask averaged to 80 m). Clouds/shadow/snow removed with the SCL band
(keep classes 4 vegetation, 5 bare soil). Scenes from both tiles on the same date are merged (valid-pixel weighted mean).
Usage: python3 scripts/ndvi_sentinel2.py [start end max_cloud]    default 2023-04-01 2024-12-31 40
"""
import sys, json, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd, rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import reproject, transform_bounds
from rasterio.windows import from_bounds
from pystac_client import Client

START, END, MAXCLOUD = (sys.argv[1], sys.argv[2], float(sys.argv[3])) if len(sys.argv) > 3 else ("2023-04-01", "2024-12-31", 40.0)
RES, CROP_FRAC, MIN_VALID = 80.0, 0.6, 0.15
BBOX = [74.12, 42.43, 74.56, 43.25]; MASK = "data/processed/sokuluk_cropland.tif"
CACHE = Path("data/raw/sentinel2_ndvi_scenes.jsonl"); OUT = "data/processed/sokuluk_ndvi_daily.csv"
ENV = dict(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", AWS_NO_SIGN_REQUEST="YES", GDAL_HTTP_MAX_RETRY="4", GDAL_HTTP_RETRY_DELAY="1", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif")

# ---- target grid (UTM 43N, 80 m) and cropland fraction on it
l, b, r, t = transform_bounds("EPSG:4326", "EPSG:32643", *BBOX)
l, t = np.floor(l / RES) * RES, np.ceil(t / RES) * RES; r, b = np.ceil(r / RES) * RES, np.floor(b / RES) * RES
GW, GH = int((r - l) / RES), int((t - b) / RES); GT = from_origin(l, t, RES, RES)
frac = np.zeros((GH, GW), "float32")
with rasterio.open(MASK) as s:   # read as a plain array: the file declares nodata=0, which would make "average" ignore the non-cropland zeros
    reproject(s.read(1).astype("float32"), frac, src_transform=s.transform, src_crs=s.crs, dst_transform=GT, dst_crs="EPSG:32643",
              resampling=Resampling.average, src_nodata=None, dst_nodata=None)
CROP = frac >= CROP_FRAC
print(f"grid {GW}x{GH} @ {RES:.0f} m; cropland cells: {int(CROP.sum())} (~{CROP.sum()*RES**2/1e6:.0f} km²)", flush=True)

def read_band(href):
    with rasterio.Env(**ENV), rasterio.open(href) as src:
        win = from_bounds(l, b, r, t, transform=src.transform)
        return src.read(1, window=win, out_shape=(GH, GW), boundless=True, fill_value=0, resampling=Resampling.nearest)

def process(item):
    try:
        a = item.assets; off = (a["red"].extra_fields.get("raster:bands") or [{}])[0].get("offset", 0.0)
        red, nir, scl = read_band(a["red"].href), read_band(a["nir"].href), read_band(a["scl"].href)
        ok = CROP & np.isin(scl, (4, 5)) & (red > 0) & (nir > 0)
        rf, nf = (red[ok] + off) / 1e4, (nir[ok] + off) / 1e4
        nd = (nf - rf) / np.maximum(nf + rf, 1e-6)
        return dict(id=item.id, date=item.datetime.strftime("%Y-%m-%d"), tile=item.properties.get("grid:code"), cloud=round(item.properties["eo:cloud_cover"], 1),
                    n_crop=int(CROP.sum()), n_valid=int(ok.sum()), ndvi_sum=float(nd.sum()), ndvi_sumsq=float((nd ** 2).sum()))
    except Exception as e:
        return dict(id=item.id, error=str(e)[:200])

def main():
    items = list(Client.open("https://earth-search.aws.element84.com/v1").search(
        collections=["sentinel-2-l2a"], bbox=BBOX, datetime=f"{START}/{END}", query={"eo:cloud_cover": {"lt": MAXCLOUD}}, max_items=2000).items())
    done = {}
    if CACHE.exists(): done = {j["id"]: j for j in map(json.loads, CACHE.read_text().splitlines()) if "error" not in j}
    todo = [i for i in items if i.id not in done]
    print(f"{len(items)} scenes, {len(done)} cached, {len(todo)} to process", flush=True); CACHE.parent.mkdir(parents=True, exist_ok=True); t0 = time.time()
    with ThreadPoolExecutor(8) as ex, CACHE.open("a") as f:
        for k, res in enumerate(ex.map(process, todo), 1):
            if "error" in res: print("ERR", res, flush=True); continue
            f.write(json.dumps(res) + "\n"); f.flush(); done[res["id"]] = res
            if k % 20 == 0 or k == len(todo): print(f"{k}/{len(todo)}  {time.time()-t0:.0f}s", flush=True)
    df = pd.DataFrame([done[i.id] for i in items if i.id in done])
    # the same tile can appear as 2 overlapping granules on one date (…_0 / …_1): keep the one with most valid cropland pixels (avoids double counting)
    df = df.sort_values("n_valid").groupby(["date", "tile"], as_index=False).tail(1)
    g = df.groupby("date").agg(n_valid=("n_valid", "sum"), ndvi_sum=("ndvi_sum", "sum"), n_scenes=("id", "count"), cloud_mean=("cloud", "mean")).reset_index()
    ncrop = int(CROP.sum()); g["valid_frac"] = (g.n_valid / ncrop).clip(upper=1)   # NB: overlapping scenes on one date may double-count; clipped
    g["ndvi_mean"] = g.ndvi_sum / g.n_valid
    out = g[g.valid_frac >= MIN_VALID][["date", "ndvi_mean", "valid_frac", "n_scenes"]].round({"ndvi_mean": 4, "valid_frac": 3})
    out.to_csv(OUT, index=False); print(f"saved {OUT}: {len(out)} dates (of {len(g)} with data; kept valid_frac >= {MIN_VALID})")

if __name__ == "__main__": main()
