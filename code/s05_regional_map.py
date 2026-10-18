"""Step 5. Regional map of households and surplus electricity (Figure 3).

Needs data/geo/sigungu_boundaries.geojson (made by prepare_boundaries.py from the
downloaded outlines, see README Section 6). If the file is
missing, this step is skipped. No other result depends on it.

Produces
  figures/fig03_regional_map.png
  tables/map_district_averages.csv
"""
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

import common as M
import config as C


def harmonise_districts(df):
    """Align KEPCO district names with the boundary file."""
    df = df.copy()
    # Sejong has no sub-districts in the boundary file
    df.loc[df["province"] == "세종특별자치시", "district"] = "세종특별자치시"
    # Gunwi-gun moved from Gyeongsangbuk-do to Daegu in July 2023
    df.loc[df["district"] == "군위군", "province"] = "대구광역시"
    # Bucheon's three former gu are one city in the boundary file
    df.loc[df["district"].str.contains("소사구|오정구|원미구", na=False), "district"] = "부천시"
    return df


def main() -> None:
    if not C.DISTRICT_GEOJSON_FILE.exists():
        print("Step 5 skipped: district outlines not found. Figure 3 needs them; see README Section 6.")
        return
    import geopandas as gpd

    M.ensure_output_dirs()
    s = harmonise_districts(M.load_surplus())
    # monthly district totals, then the average month over 2021-2023
    monthly = s.groupby(["province", "district", "year", "month"])[["households", "surplus_kwh"]].sum()
    avg = monthly.groupby(["province", "district"]).mean().reset_index()

    geo = gpd.read_file(C.DISTRICT_GEOJSON_FILE)
    gdf = geo.merge(avg, on=["province", "district"], how="outer", indicator=True)
    no_shape = gdf.loc[gdf["_merge"] == "right_only", ["province", "district"]].values.tolist()
    no_data = gdf.loc[gdf["_merge"] == "left_only", ["province", "district"]].values.tolist()
    if no_shape:
        print(f"Warning: {len(no_shape)} districts have data but no shape, e.g. {no_shape[:10]}")
    if no_data:
        print(f"Note: {len(no_data)} districts have a shape but no surplus data, e.g. {no_data[:10]}")
    gdf = gdf[gdf["_merge"] == "both"].drop(columns="_merge")
    M.save_table(gdf.drop(columns="geometry"), "map_district_averages", index=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    for ax, col, cmap, label in [(ax1, "households", plt.cm.Oranges, "Number of household"),
                                 (ax2, "surplus_kwh", plt.cm.Reds, "Surplus electricity (kWh)")]:
        norm = mcolors.Normalize(vmin=gdf[col].min(), vmax=gdf[col].max())
        gdf.plot(column=col, cmap=cmap, norm=norm, linewidth=0.8, edgecolor="black", ax=ax)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        cbar = plt.colorbar(sm, ax=ax, shrink=0.5)
        cbar.set_label(label, rotation=270, labelpad=20)
        if col == "surplus_kwh":
            import matplotlib.ticker as mticker
            cbar.formatter = mticker.ScalarFormatter(useMathText=True)
            cbar.formatter.set_powerlimits((6, 6))
            cbar.update_ticks()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
    M.savefig(fig, "fig03_regional_map")
    print(f"Step 5 done: Figure 3 ({len(gdf)} districts)")


if __name__ == "__main__":
    main()
