"""Make the district boundary file used for Figure 3.

The district outlines come from Statistics Korea (SGIS, "센서스용 행정구역 경계",
2023, 시군구). This package does not redistribute them (see README Section 6).
Download them, then run this script once from the package root:

    python code/prepare_boundaries.py PATH/TO/bnd_all_00_2023_2Q.zip

The argument can be the downloaded zip file (bnd_all_00_2023_2Q.zip), the
district zip inside it (bnd_sigungu_00_2023_2Q.zip), or the unzipped .shp file.

The script
  1. reads the district code (SIGUNGU_CD), the district name (SIGUNGU_NM) and the outlines,
  2. simplifies the outlines to about 100 m (enough for a national map),
  3. repairs outlines that cross themselves after simplification,
  4. adds the province name from the first two digits of the district code,
and writes data/geo/sigungu_boundaries.geojson.
"""
import argparse
import zipfile
from pathlib import Path

import geopandas as gpd
import pyogrio

import config as C

PROJECTED_CRS = "EPSG:5179"   # Korea 2000 / Unified CS (UTM-K), units in metres; the SGIS files use it
SHP_NAME = "bnd_sigungu_00_2023_2Q"

# The first two digits of the SGIS district code give the province.
# Names are written as in the KEPCO surplus electricity file.
PROVINCES = {
    "11": "서울특별시", "21": "부산광역시", "22": "대구광역시", "23": "인천광역시",
    "24": "광주광역시", "25": "대전광역시", "26": "울산광역시", "29": "세종특별자치시",
    "31": "경기도", "32": "강원특별자치도", "33": "충청북도", "34": "충청남도",
    "35": "전북특별자치도", "36": "전라남도", "37": "경상북도", "38": "경상남도",
    "39": "제주특별자치도",
}
# District names that differ between the SGIS file and the KEPCO file
DISTRICT_RENAMES = {"세종시": "세종특별자치시"}


def source_path(p: Path) -> str:
    """Path that GDAL can read: the .shp itself, or the .shp inside one or two zip files."""
    if p.suffix.lower() == ".shp":
        return str(p)
    if p.suffix.lower() != ".zip":
        raise SystemExit(f"Give the downloaded .zip file or the .shp file, not {p.name}")
    names = zipfile.ZipFile(p).namelist()
    if f"{SHP_NAME}.shp" in names:                       # the district zip
        return f"/vsizip/{p.resolve()}/{SHP_NAME}.shp"
    if f"{SHP_NAME}.zip" in names:                       # the zip with all levels
        return f"/vsizip/{{/vsizip/{p.resolve()}/{SHP_NAME}.zip}}/{SHP_NAME}.shp"
    raise SystemExit(f"{p.name} does not contain {SHP_NAME}.shp or {SHP_NAME}.zip. Contents: {names}")


def polygons_only(geoms):
    """Repair shapes whose simplified outline crosses itself, keeping only polygon parts."""
    import shapely
    from shapely.geometry import MultiPolygon, Polygon

    def fix(g):
        g = shapely.make_valid(g)
        parts = [p for p in shapely.get_parts(g) if isinstance(p, (Polygon, MultiPolygon))]
        polys = [q for p in parts for q in (p.geoms if isinstance(p, MultiPolygon) else [p])]
        return polys[0] if len(polys) == 1 else MultiPolygon(polys)

    return geoms.apply(fix)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", type=Path, help="downloaded SGIS zip file, or the district .shp file")
    ap.add_argument("--tolerance", type=float, default=100.0, help="simplification tolerance in metres")
    args = ap.parse_args()
    if not args.source.is_file():
        raise SystemExit(f"Not a file: {args.source}")

    src = source_path(args.source)
    info = pyogrio.read_info(src)
    print(f"source: {info['features']} shapes, CRS {info['crs']}, columns {list(info['fields'])}")
    gdf = gpd.read_file(src, columns=["BASE_DATE", "SIGUNGU_CD", "SIGUNGU_NM"])
    print(f"base date of the outlines: {sorted(gdf['BASE_DATE'].unique())}")

    gdf = gdf.to_crs(PROJECTED_CRS)
    gdf["province"] = gdf["SIGUNGU_CD"].str[:2].map(PROVINCES)
    if gdf["province"].isna().any():
        bad = gdf.loc[gdf["province"].isna(), ["SIGUNGU_CD", "SIGUNGU_NM"]].values.tolist()
        raise SystemExit(f"Unknown province code for {bad}. A different version of the outlines may have "
                         "been downloaded (see README Section 6).")
    gdf["district"] = gdf["SIGUNGU_NM"].replace(DISTRICT_RENAMES)
    gdf["geometry"] = gdf.geometry.simplify(args.tolerance, preserve_topology=True)
    gdf["geometry"] = polygons_only(gdf.geometry)
    gdf = gdf[["SIGUNGU_CD", "province", "district", "geometry"]].sort_values("SIGUNGU_CD").to_crs("EPSG:4326")

    out = C.DISTRICT_GEOJSON_FILE
    out.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(out, driver="GeoJSON", COORDINATE_PRECISION=5)
    print(f"wrote {len(gdf)} districts to {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
