"""Make the district boundary file used for Figure 3.

The district outlines are third-party data that this package does not
redistribute (see README Section 6). Download them, then run this script once:

    python code/prepare_boundaries.py PATH/TO/SIG.shp

The script
  1. reads the district code column (SIG_CD) and the outlines,
  2. keeps one shape per district code,
  3. simplifies the outlines to about 100 m (enough for a national map),
  4. repairs outlines that cross themselves after simplification,
  5. adds the province and district names from data/geo/sigungu_codes.csv,
and writes data/geo/sigungu_boundaries.geojson.

It also accepts a GeoJSON file in which the outlines repeat for every month
(columns 년도 and 월); it then reads one month only.
"""
import argparse
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pyogrio

import config as C

PROJECTED_CRS = "EPSG:5179"   # Korea 2000 / Unified CS, units in metres
CODE_CANDIDATES = ["SIG_CD", "Code"]
YEAR_COL, MONTH_COL = "년도", "월"
CODES_FILE = C.DATA_DIR / "geo" / "sigungu_codes.csv"


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
    ap.add_argument("source", type=Path, help="district boundary file (.shp or .geojson)")
    ap.add_argument("--tolerance", type=float, default=100.0, help="simplification tolerance in metres")
    ap.add_argument("--code-column", default=None, help="district code column (default: SIG_CD or Code)")
    ap.add_argument("--source-crs", default=None, help="CRS of the source if the file does not say, e.g. EPSG:5179")
    ap.add_argument("--encoding", default=None, help="text encoding of a shapefile, e.g. cp949")
    args = ap.parse_args()

    if not args.source.is_file():
        raise SystemExit(f"Not a file: {args.source}\nGive the path of the .shp or .geojson file itself, not its folder.")
    opts = {"encoding": args.encoding} if args.encoding else {}
    info = pyogrio.read_info(args.source, **opts)   # header only, fast even for a large file
    fields = list(info["fields"])
    print(f"source: {info['features']} shapes, CRS {info['crs']}, columns {fields}")

    code_col = args.code_column or next((c for c in CODE_CANDIDATES if c in fields), None)
    if code_col not in fields:
        raise SystemExit(f"No district code column found. Choose it with --code-column. Columns: {fields}")

    where = None
    if YEAR_COL in fields and MONTH_COL in fields:
        first = pyogrio.read_dataframe(args.source, columns=[YEAR_COL, MONTH_COL], read_geometry=False)
        y, m = first[[YEAR_COL, MONTH_COL]].drop_duplicates().sort_values([YEAR_COL, MONTH_COL]).iloc[0]
        where = f'"{YEAR_COL}" = {int(y)} AND "{MONTH_COL}" = {int(m)}'
        print(f"file repeats outlines per month; reading only {int(y)}-{int(m):02d}")

    gdf = gpd.read_file(args.source, columns=[code_col], where=where, **opts)
    if gdf.crs is None:
        if args.source_crs is None:
            raise SystemExit("The source file has no CRS. Give it with --source-crs (EPSG:5179 for UTM-K).")
        gdf = gdf.set_crs(args.source_crs)
    n_read = len(gdf)

    gdf["SIG_CD"] = gdf[code_col].astype(str).str.strip()
    gdf = gdf[["SIG_CD", "geometry"]].to_crs(PROJECTED_CRS)
    if gdf["SIG_CD"].duplicated().any():
        gdf = gdf.dissolve(by="SIG_CD", as_index=False)   # one shape per district
    gdf["geometry"] = gdf.geometry.simplify(args.tolerance, preserve_topology=True)
    gdf["geometry"] = polygons_only(gdf.geometry)

    names = pd.read_csv(CODES_FILE, dtype={"SIG_CD": str})
    gdf = gdf.merge(names, on="SIG_CD", how="outer", indicator=True)
    no_name = gdf.loc[gdf["_merge"] == "left_only", "SIG_CD"].tolist()
    no_shape = gdf.loc[gdf["_merge"] == "right_only", ["SIG_CD", "province", "district"]].values.tolist()
    if no_name:
        print(f"Note: {len(no_name)} codes in the source are not in {CODES_FILE.name} and are dropped: {no_name}")
    if no_shape:
        print(f"Warning: {len(no_shape)} districts in {CODES_FILE.name} have no outline in the source: {no_shape}\n"
              "  The source may be a different version of the boundaries (see README Section 6).")
    gdf = gdf[gdf["_merge"] == "both"].drop(columns="_merge")
    gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs=PROJECTED_CRS)
    gdf = gdf[["SIG_CD", "province", "district", "geometry"]].sort_values("SIG_CD").to_crs("EPSG:4326")

    out = C.DISTRICT_GEOJSON_FILE
    gdf.to_file(out, driver="GeoJSON", COORDINATE_PRECISION=5)
    print(f"read {n_read} shapes, wrote {len(gdf)} districts to {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
