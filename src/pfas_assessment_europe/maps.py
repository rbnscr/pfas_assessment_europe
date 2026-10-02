"""Create maps of PFAS monitoring coverage and concentration hotspots.

This module provides visualisations of PFAS monitoring locations across
European HydroBASINS regions. It generates:

- A map showing the number of monitored substances at each monitoring site
- Maps identifying sites with concentrations at or above the 85th percentile
  for selected PFAS substances

The generated maps are saved as PNG or PDF files in the specified output
directory.
"""

import logging
from pathlib import (
    Path,
)

import cartopy.crs as ccrs
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import (
    Rectangle,
)
from pfas_assessment_europe.constants import HYBAS_RIVER_RENAME

logger = logging.getLogger(__name__)


def eu_map(
    gdf: gpd.GeoDataFrame, 
    basins: gpd.GeoDataFrame, 
    save_path: Path
    ) -> None:
    """Create and save a map of PFAS monitoring coverage across Europe.

    The function selects monitoring sites from observations collected after
    2018 and calculates the number of unique substances monitored at each
    site. The sites are spatially joined to HydroBASINS level-04 regions,
    which are displayed as the background map.

    Selected basin regions are highlighted and labelled. Basin labels are
    positioned either at representative points or at manually specified
    coordinates. Monitoring sites are coloured according to the number of
    monitored substances.

    The resulting map is saved as ``main_introduction_map.jpg`` in ``save_path``.

    Args:
        gdf: GeoDataFrame containing PFAS observations. It must include
            ``year``, ``substance``, and ``geometry`` columns and have a
            defined coordinate reference system.
        basins: GeoDataFrame containing basin geometries and a ``HYBAS_ID``
            column. The basin geometries are transformed to the CRS of
            ``gdf``.
        save_path: Directory in which the map image is saved.

    Returns:
        None. The map is saved to disk and a completion message is written to
        the logger.
    """
    logger.info("--- Starting EU Map ---")
    lev04_rename = HYBAS_RIVER_RENAME
    id_name = "HYBAS_ID"
    basins["HYBAS_ID_ID"] = basins["HYBAS_ID"].copy()

    basins[id_name] = basins[id_name].astype("object")
    basins.loc[~basins[id_name].isin(lev04_rename.keys()), id_name] = "Other"
    basins[id_name] = basins[id_name].replace(lev04_rename)
    basins[id_name] = basins[id_name].astype("string")
    europe = basins.to_crs(gdf.crs)

    gdf_geom = (
        gdf[gdf.year > 2018]
        .groupby("geometry", as_index=False)
        .agg(count_subst=("substance", "nunique"))
    )
    gdf_geom = gpd.GeoDataFrame(gdf_geom, geometry="geometry", crs=gdf.crs)
    joined = gpd.sjoin(
        gdf_geom[["count_subst", "geometry"]],
        europe[["HYBAS_ID", "geometry"]],
        how="left",
        predicate="within",
    )

    europe_with_points = europe.loc[joined.index_right.dropna().unique()]
    basins_with_name = europe[europe.HYBAS_ID_ID.isin(europe_with_points.HYBAS_ID_ID)]

    logger.info("Number of monitoring geometries: %d", len(gdf_geom))
    logger.info(
        "Monitoring bounds: %s",
        gdf_geom.total_bounds,
    )
    logger.info(
        "Basin bounds: %s",
        europe.total_bounds,
    )

    logger.info("Start plotting")
    fig, ax = plt.subplots(
        figsize=(7, 8), # 9.5, 8
        subplot_kw={
            "projection": ccrs.PlateCarree(central_longitude=10),
            "frameon": False,
        },
    )
    ax.set_extent([-15, 32, 34, 63], crs=ccrs.PlateCarree())

    europe.plot(
        ax=ax,
        color="white",
        edgecolor="gray",
        linewidth=0.2,
        transform=ccrs.PlateCarree(),
    )

    # Highlight selected basins
    basins_with_name.plot(
        ax=ax,
        color="whitesmoke",
        edgecolor="black",
        linewidth=0.4,
        transform=ccrs.PlateCarree(),
        zorder=1,
    )

    gdf_geom.sort_values(by="count_subst").plot(
        ax=ax,
        column="count_subst",
        cmap="viridis",
        markersize=7,
        alpha=0.7,
        transform=ccrs.PlateCarree(),
        zorder=2,
    )

    # Colorbar
    norm = mpl.colors.Normalize(
        vmin=gdf_geom["count_subst"].min(), vmax=gdf_geom["count_subst"].max()
    )

    sm = mpl.cm.ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array([])

    cbar = fig.colorbar(
        sm, ax=ax, orientation="horizontal", fraction=0.01, pad=0.04, aspect=50
    )
    cbar.set_label("Number of monitored substances", fontsize=12)
    # ax.set_autoscale_on(False)
    # Format:
    # "Name": [(label_x, label_y), (arrow_target_x, arrow_target_y)]
    manual_labels = {
        "Maas": [(2.946, 52.127), (3.865, 50.500)],
        "Tiber": [(11.864, 40.159), (14.427, 41.239)],
        "Po": [(14.520, 43.315), (11.4499, 44.9918)],
        "Garonne": [(-3.914, 44.609), (1.096, 44.906)],
        "Rhône / Ebro": [(4.883, 40.511), (5.019, 44.132)],
        "Loire": [(-4.550, 46.697), (1.956, 46.654)],
        "Seine": [(-1.421, 50.000), (3.264, 48.492)],
        "Brittany / Normandy": [(-8.818, 49.104), (-2.85, 48.481)],
        "Rhine": [(3.442, 53.939), (7.98, 49.234)],
        "Weser / Ems": [(8.345, 55.097), (7.642, 52.127)],
        "United Kingdom": [(1.705, 56.743), (-1.919, 54.316)],
        "Sicily": [(17.574, 35.562), (14.137, 37.476)],
        "Corsica": [(7.199, 37.321), (9.098, 40.063)],
        # "Lower Danube" : [(-5.5, 50.5), (24.496 , 44.418)],
        # "Upper Danube" : [(-5.5, 50.5), (14.528 , 47.896)],
        # "Elbe" : [(-5.5, 50.5), (12.648 , 51.163)],
        # "Oder" : [(-5.5, 50.5), (17.115 , 51.684)],
        "Nemunas": [(19.198, 57.00), (23.09, 56.545)],
        "Baltic (Southern Sweden)": [(12.477, 60.954), (15.307, 58.0)],
        # "Baltic (Western Finland)" : [(-5.5, 50.5), (6.2, 50.9)],
        # "Norway" : [(-5.5, 50.5), (6.2, 50.9)],
        # "Newa" : [(-5.5, 50.5), (6.2, 50.9)],
        "Daugava": [(30.049, 53.099), (27.897, 56.038)],
        "Nyoman": [(26.076, 51.500), (25.269, 54.318)],
        # "Narva" : [(-5.5, 50.5), (26.78 , 57.975)],
        # "Duero" : [(-5.5, 50.5), (-4.942 , 41.661)],
        # "Sil" : [(-5.5, 50.5), (-7.599 , 42.471)],
        # "Mediterranean Balkans" : [(-5.5, 50.5), (23.771 , 41.414)],
        "Prut": [(29.663, 48.309), (27.893, 46.938)],
        # "Tysa" : [(-5.5, 50.5), (6.2, 50.9)],
        # "Drava" : [(-5.5, 50.5), (15.0 , 46.599)],
        "Mura-Drava-Danube": [(24.429, 50.352), (19.108, 45.736)],
        # "Sava" : [(-5.5, 50.5), (6.2, 50.9)]
        "Tagus" : [(-5, 39.75), (-8, 40)], # No arrow, but moving the label
        "Guadiana" : [(-5, 38.75), (-8, 40)],
    }

    no_arrow = [
        "Elbe",
        "Narva",
        "Oder",
        "Tysa",
        "Duero",
        "Sil",
        "Sava",
        "Mediterranean Balkans",
        "Drava",
        "Lower Danube",
        "Upper Danube",
        "Guadiana",
        "Tagus"
    ]

    for _, row in basins_with_name.iterrows():
        name = row[id_name]
        rep = row.geometry.representative_point()
        arrow_x, arrow_y = rep.x, rep.y

        # override if manually defined
        if name in manual_labels:
            (label_x, label_y), (arrow_x, arrow_y) = manual_labels[name]
        else:
            label_x, label_y = arrow_x, arrow_y
        if name in no_arrow:
            custom_arrow_props = None
        else:
            custom_arrow_props = dict(arrowstyle="-", color="black", lw=0.6)
        ax.annotate(
            name,
            xy=(arrow_x, arrow_y),
            xytext=(label_x, label_y),
            textcoords=ccrs.PlateCarree()._as_mpl_transform(ax),
            xycoords=ccrs.PlateCarree()._as_mpl_transform(ax),
            ha="center",
            fontsize=11,
            arrowprops=custom_arrow_props,
            zorder=3,
        )

    gl = ax.gridlines(
        draw_labels=True, linewidth=0.5, color="gray", alpha=0.5, linestyle="--"
    )

    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 10}
    gl.ylabel_style = {"size": 10}
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    logger.info(f"Saving map to {save_path / "main_introduction_map.jpg"}")
    fig.savefig(save_path / "main_introduction_map.jpg", bbox_inches="tight", dpi=600)
    plt.close(fig)


def hotspot_maps(
    gdf: gpd.GeoDataFrame, basins: gpd.GeoDataFrame, save_path: Path
) -> None:
    """Create and save PFAS concentration hotspot maps.

    For each selected PFAS substance, the function calculates the 85th
    percentile of concentrations above the limit of detection. Sampling sites
    with concentrations at or above this threshold are classified as
    concentration hotspots.

    A separate map is generated for each selected substance. Each map shows
    all sampling sites for the substance, hotspot sites, and the HydroBASINS
    regions containing sampling locations. The resulting maps are saved as
    JPG files named ``appendix_hotspot_<substance>.jpg`` in ``save_path``.

    Args:
        gdf: GeoDataFrame containing PFAS observations. It must include
            ``substance``, ``conc``, ``less_than``, and ``geometry`` columns and
            have a defined coordinate reference system.
        basins: GeoDataFrame containing basin geometries and a ``HYBAS_ID``
            column. The basin geometries are transformed to the CRS of
            ``gdf``.
        save_path: Directory in which the hotspot maps are saved.

    Returns:
        None. One JPG map is saved for each selected substance.

    Notes:
        Observations marked as below the limit of detection are excluded from
        hotspot classification.
    """
    logger.info("Start plotting Hotspot maps")
    gdf_dropped = gdf[~gdf["less_than"]].copy()
    df_plot = gdf_dropped[["substance", "conc"]].copy()
    df_plot["substance"] = df_plot["substance"].replace(
        {
            "L_PFBS": "Linear PFBS",
            "L_PFHxS": "Linear PFHxS",
            "L_PFHpS": "Linear PFHpS",
            "L_PFOS": "Linear PFOS",
            "L_PFOA": "Linear PFOA",
        }
    )
    stats = df_plot.groupby(["substance"]).agg(
        # nmeas=("conc", "count"),
        # median=("conc", "median"),
        # mean=("conc", "mean"),
        q85=("conc", lambda x: np.quantile(x.dropna(), 0.85)),
        # max=("conc", "max")
    )

    id_name = "HYBAS_ID"
    basins["HYBAS_ID_ID"] = basins["HYBAS_ID"]
    europe = basins.to_crs(gdf.crs)
    europe["ADMIN"] = europe[id_name]

    all_substances = [
        "6:2 FTS",
        "EtFOSAA",
        "FOSA",
        "HFPO-DA",
        "N-MeFOSAA",
        "PFBA",
        "PFBS",
        "PFDS",
        "PFDoDA",
        "PFHpA",
        "PFHxA",
        "PFHxS",
        "PFOA",
        "PFOS",
        "PFPeA",
        "PFPeS",
        "PFTeDA",
        "PFTrDA",
    ]
    logger.info(f"Hotspot maps are created for substances: {all_substances}")
    hotspots = []

    for substance in all_substances:
        substance = substance.replace("L_", "Linear ")
        # substance_write = substance.replace(":", "-")
        threshold_value_q85 = stats.loc[substance, "q85"]
        logger.info(
            f"Threshold value (85th percentile) of {substance}: {threshold_value_q85} ng/L"
        )
        gdf_substance_timeframe = gdf[gdf.substance == substance].copy()
        hotspots.append(
            gdf_substance_timeframe[
                (~gdf_substance_timeframe["less_than"])
                & (gdf_substance_timeframe.conc >= threshold_value_q85)
            ]
        )
    gdf_h = gpd.GeoDataFrame(
        pd.concat(hotspots, ignore_index=True), crs=hotspots[0].crs
    )
    gdf_high_geoms = gdf_h.drop_duplicates(subset=["geometry", "substance"])
    gdf_all_geoms = gdf.drop_duplicates(subset=["geometry", "substance"]).to_crs(
        gdf_high_geoms.crs
    )

    for subst in all_substances:
        logger.info(f"Plotting substance: {subst}")
        gdf_subst_high = gdf_high_geoms[gdf_high_geoms.substance == subst].copy()
        gdf_all_geom = gdf_all_geoms[gdf_all_geoms.substance == subst].copy()

        logger.info(f"Number of sites indicating a hotspot: {len(gdf_subst_high)}")
        logger.info(f"Number of sites: {len(gdf_all_geom)}")

        # europe_joined = gdf_all_geom.sjoin(europe, predicate="within")

        # eu_crs = europe.crs
        gdf_crs = gdf_all_geom.crs
        # europe_joined = gdf_all_geom.to_crs("EPSG:3035").sjoin_nearest(europe.to_crs("EPSG:3035"), how="left")

        # europe_joined = europe_joined.to_crs(gdf_crs)
        within = gpd.sjoin(
            gdf_all_geom.to_crs("EPSG:3035"),
            europe[["ADMIN", "geometry"]].to_crs("EPSG:3035"),
            how="left",
            predicate="within",
        )
    
        missing = within[within["ADMIN"].isna()].drop(columns="ADMIN")
    
        if "index_right" in missing.columns:
            missing = missing.drop(columns="index_right")
    
        nearest = gpd.sjoin_nearest(
            missing.to_crs("EPSG:3035"),
            europe[["ADMIN", "geometry"]].to_crs("EPSG:3035"),
            how="left",
        )
        europe_joined = pd.concat([within[within["ADMIN"].notna()], nearest])
        # europe_joined = europe_joined.to_crs(gdf_all_geom.crs)
        # print(europe_joined)
        # print(gdf_all_geom.crs)
        # gdf_all_geom = gdf_all_geom.to_crs(gdf_crs)
        europe_with_points = europe.loc[europe_joined.index_right.unique()]
        scale = 1
        fig, ax = plt.subplots(
            figsize=(14*scale, 12*scale),
            subplot_kw={
                "projection": ccrs.PlateCarree(central_longitude=10),
                "frameon": False,
            },
        )

        ax.set_autoscale_on(False)
        europe.plot(
            ax=ax,
            color="white",
            edgecolor="gray",
            transform=ccrs.PlateCarree(),
            linewidth=0.2,
        )

        europe[europe.HYBAS_ID_ID.isin(europe_with_points.HYBAS_ID_ID)].plot(
            ax=ax,
            color="whitesmoke",
            edgecolor="black",
            transform=ccrs.PlateCarree(),
            linewidth=0.4,
        )

        gdf_all_geom.plot(
            ax=ax,
            color="#1b9e77",
            markersize=2,
            alpha=1,
            label="Concentration $<$ 85th percentile",
            transform=ccrs.PlateCarree(),
        )

        gdf_subst_high.plot(
            ax=ax,
            color="#d95f02",
            markersize=3,
            alpha=1,
            label="Concentration ≥ 85th percentile",
            transform=ccrs.PlateCarree(),
        )

        gl = ax.gridlines(
            crs=ccrs.PlateCarree(),
            draw_labels=True,
            linewidth=0.5,
            color="gray",
            alpha=0.5,
            linestyle="--",
        )

        gl.top_labels = False
        gl.right_labels = False

        gl.xlabel_style = {"size": 10, "color": "black"}
        gl.ylabel_style = {"size": 10, "color": "black"}
        gl.xpadding = 5
        gl.ypadding = 5

        rect = Rectangle(
            (0.2, 0.08),
            0.6,
            0.83,
            transform=fig.transFigure,
            facecolor="none",
            edgecolor="none",
            linewidth=0,
        )
        fig.add_artist(rect)

        plt.title(f"{subst}")
        plt.legend(
            fontsize=10,
            markerscale=4,
            loc="upper left",
            frameon=True,
            title="Sampling site classification",
        )
        ax.set_extent([-15, 32, 34, 72], crs=ccrs.PlateCarree())
        subst_name = subst.replace(":", "")
        logger.info(f"Saving map to {save_path / f"appendix_hotspot_{subst_name}.jpg"}")
        plt.savefig(
            save_path / f"appendix_hotspot_{subst_name}.jpg",
            dpi = 300,
            bbox_inches="tight",
            pad_inches=0.02,
            transparent=False
        )
        plt.close(fig)
    logger.info("--- Finished plotting hotspot maps ---")


def main() -> None:
    input_path = Path("data/input/")
    gdf: gpd.GeoDataFrame = gpd.read_file(input_path.joinpath("pfas_data.gpkg"))
    # print(gdf.crs)
    # gdf = gpd.read_file(input_path.joinpath("pfas_data.gpkg"))
    basins: gpd.GeoDataFrame = gpd.read_file(
        input_path.joinpath("hybas_eu_lev04_v1c.shp")
    )
    gdf["month"] = pd.to_datetime(gdf["date"], format="ISO8601").dt.month
    gdf["dayofyear"] = pd.to_datetime(gdf["date"], format="ISO8601").dt.dayofyear
    # gdf_dropped = gdf[~gdf["less_than"]].copy()

    save_path = Path("results/")

    basins = basins.to_crs(gdf.crs)  # pyright: ignore[reportArgumentType]
    gdf_timeframe = gdf[gdf["year"] > 2018]
    eu_map(gdf_timeframe, basins, save_path)
    hotspot_maps(gdf_timeframe, basins, save_path)


if __name__ == "__main__":
    main()
