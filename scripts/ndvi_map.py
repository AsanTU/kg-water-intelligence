"""Day 3: NDVI map of Sokuluk cropland for ONE cloud-free peak-season date at 10 m -> data/processed/sokuluk_ndvi.tif (+ _meta.json)

Source: Sentinel-2 L2A COGs (AWS Earth Search STAC, open, no key). NDVI = (B08 - B04) / (B08 + B04) on surface reflectance ((DN + offset)/10000).
Masks: SCL classes 4 (vegetation) and 5 (bare soil) only; cloud/shadow/snow classes (3, 8, 9, 10, 11) dilated by 2 px; and the Day 1 WorldCover cropland mask.
Grid: UTM 43N (EPSG:32643), 10 m. The cropland mask (EPSG:4326, read as a plain array because the file declares nodata=0) is resampled with nearest neighbour.
Usage: python3 scripts/ndvi_map.py [item_id ...]     default: the two 27 May 2024 scenes
"""
import json, sys
from concurrent.futures import ThreadPoolExecutor
import numpy as np, rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import reproject, transform_bounds
from rasterio.windows import from_bounds
from scipy.ndimage import binary_dilation
from pystac_client import Client

IDS = sys.argv[1:] or ["S2B_43TDH_20240527_1_L2A", "S2B_43TDG_20240527_0_L2A"]   # default = the delivered map (27 May 2024)
RES, BBOX, MASK = 10.0, [74.12, 42.43, 74.56, 43.25], "data/processed/sokuluk_cropland.tif"
OUT, META = "data/processed/sokuluk_ndvi.tif", "data/processed/sokuluk_ndvi_meta.json"
ENV = dict(GDAL_CACHEMAX=1024, GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", AWS_NO_SIGN_REQUEST="YES", GDAL_HTTP_MAX_RETRY="4", GDAL_HTTP_RETRY_DELAY="1", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif")

l, b, r, t = transform_bounds("EPSG:4326", "EPSG:32643", *BBOX)
l, t = np.floor(l / RES) * RES, np.ceil(t / RES) * RES; r, b = np.ceil(r / RES) * RES, np.floor(b / RES) * RES
GW, GH = int((r - l) / RES), int((t - b) / RES); GT = from_origin(l, t, RES, RES)
print(f"grid {GW}x{GH} @ {RES:.0f} m (UTM 43N)", flush=True)

STRIP = 512   # rows per task; each HTTPS connection to the bucket is slow (~100 KB/s), so many strips are fetched in parallel
def _strip(href, r0, r1):
    with rasterio.Env(**ENV), rasterio.open(href) as src:
        win = from_bounds(l, t - r1 * RES, r, t - r0 * RES, transform=src.transform)
        return r0, src.read(1, window=win, out_shape=(r1 - r0, GW), boundless=True, fill_value=0, resampling=Resampling.nearest)
def read_bands(hrefs):
    outs = {k: np.zeros((GH, GW), "uint16") for k in hrefs}
    with ThreadPoolExecutor(24) as ex:
        futs = [(k, ex.submit(_strip, h, r0, min(r0 + STRIP, GH))) for k, h in hrefs.items() for r0 in range(0, GH, STRIP)]
        for n, (k, f) in enumerate(futs, 1):
            r0, a = f.result(); outs[k][r0:r0 + a.shape[0]] = a
            if n % 20 == 0: print(f"  strips {n}/{len(futs)}", flush=True)
    return outs

with rasterio.open(MASK) as s:
    crop = np.zeros((GH, GW), "uint8")
    reproject(s.read(1), crop, src_transform=s.transform, src_crs=s.crs, dst_transform=GT, dst_crs="EPSG:32643", resampling=Resampling.nearest, src_nodata=None, dst_nodata=None)
crop = crop == 1; print(f"cropland pixels: {crop.sum():,} (~{crop.sum()*RES**2/1e6:.0f} km²)", flush=True)

items = list(Client.open("https://earth-search.aws.element84.com/v1").search(collections=["sentinel-2-l2a"], ids=IDS).items()); assert items, "scene not found"
ndvi = np.full((GH, GW), np.nan, "float32"); clear = np.zeros((GH, GW), bool)
for it in items:
    a = it.assets; off = (a["red"].extra_fields.get("raster:bands") or [{}])[0].get("offset", 0.0)
    bands = read_bands({"red": a["red"].href, "nir": a["nir"].href, "scl": a["scl"].href}); red, nir, scl = bands["red"].astype("float32"), bands["nir"].astype("float32"), bands["scl"]
    bad = binary_dilation(np.isin(scl, (3, 8, 9, 10, 11)), iterations=2)
    ok = np.isin(scl, (4, 5)) & ~bad & (red > 0) & (nir > 0) & ~clear     # first scene wins where granules overlap
    rf, nf = (red[ok] + off) / 1e4, (nir[ok] + off) / 1e4
    ndvi[ok] = np.clip((nf - rf) / np.maximum(nf + rf, 1e-6), -1, 1); clear |= ok
    print(f"{it.id}: cloud {it.properties['eo:cloud_cover']:.1f}%, clear pixels {ok.sum():,}", flush=True)

out = np.where(crop & clear, ndvi, -9999).astype("float32"); v = out[out != -9999]
prof = dict(driver="GTiff", dtype="float32", count=1, width=GW, height=GH, crs="EPSG:32643", transform=GT, nodata=-9999, compress="lzw", predictor=3, tiled=True, blockxsize=256, blockysize=256)
with rasterio.open(OUT, "w", **prof) as dst: dst.write(out, 1); dst.update_tags(source="Sentinel-2 L2A " + ",".join(IDS), formula="(B08-B04)/(B08+B04)", mask="SCL 4/5, cloud dilated 2px, WorldCover cropland")
meta = dict(items=IDS, date=items[0].datetime.strftime("%Y-%m-%d"), scene_cloud_percent=[round(i.properties["eo:cloud_cover"], 1) for i in items], tile=sorted({i.properties.get("grid:code", "") for i in items}),
            cropland_pixels=int(crop.sum()), valid_pixels=int(v.size), valid_share_of_cropland=round(float(v.size / crop.sum()), 4), valid_area_km2=round(float(v.size * RES ** 2 / 1e6), 1),
            ndvi_mean=round(float(v.mean()), 4), ndvi_median=round(float(np.median(v)), 4), ndvi_p10=round(float(np.percentile(v, 10)), 4), ndvi_p90=round(float(np.percentile(v, 90)), 4),
            ndvi_min=round(float(v.min()), 4), ndvi_max=round(float(v.max()), 4), crs="EPSG:32643", resolution_m=RES, shape=[GH, GW])
json.dump(meta, open(META, "w"), indent=1); print(json.dumps(meta, indent=1)); print("saved", OUT)
