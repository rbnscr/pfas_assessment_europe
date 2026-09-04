"""Summarise and visualise PFAS detections, concentrations, and sampling sites.

This module generates general information about PFAS monitoring data,
including detection frequencies, concentration distributions, the number of
substances detected at each sampling site, and the number of monitored
substances per site.

The module produces combined plots showing substance detection frequencies and
concentration distributions, together with a plot comparing detected and
monitored substances across sampling sites. Generated figures are saved as PDF
files in the specified output directory.
"""

import logging
from pathlib import (
    Path,
)

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import (
    FuncFormatter,
)

logger = logging.getLogger(__name__)


def general_info(gdf, gdf_timeframe, save_path: Path):
    """Generate summary statistics and plots for PFAS monitoring data.

    The function calculates substance-specific detection frequencies and
    summary information for both the complete dataset and a selected time
    period. It excludes measurements below the limit of detection when
    calculating concentration distributions and derives concentrations for
    PFAS4 and the available PFAS sum at each sampling event.

    The function also calculates how many substances were detected and
    monitored at each sampling site. The resulting summaries are passed to
    :func:`combined_plot` for visualisation.

    Args:
        gdf: GeoDataFrame containing the complete PFAS dataset. It must contain
            ``year`` and ``geometry`` columns.
        gdf_timeframe: GeoDataFrame containing the subset of observations to be
            analysed. It must contain ``substance``, ``less_than``, ``conc``,
            ``geometry``, ``dayofyear``, and ``year`` columns.
        save_path: Directory in which the generated plots are saved.

    Returns:
        None. Summary information is written to the log, and plots are saved
        to ``save_path``.
    """
    logger.info("--- Start general information generation ---")
    counts = (
        gdf_timeframe.groupby("substance")["less_than"]
        .value_counts()
        .unstack(fill_value=0)
    )
    counts = counts.rename(columns={True: "Below", False: "Above"})

    counts["total_n"] = counts["Below"] + counts["Above"]
    counts["frequency"] = counts["Above"] / counts["total_n"] * 100
    counts = counts.sort_values(by="frequency", ascending=False)
    counts = counts.rename(
        index={
            "L_PFBS": "Linear PFBS",
            "L_PFHxS": "Linear PFHxS",
            "L_PFHpS": "Linear PFHpS",
            "L_PFOS": "Linear PFOS",
            "L_PFOA": "Linear PFOA",
        }
    )
    start_year = gdf.year.min()
    end_year = gdf.year.max()
    n_meas_full = len(gdf)
    n_loc_full = gdf.geometry.nunique()

    start_year_tf = gdf_timeframe.year.min()
    end_year_tf = gdf_timeframe.year.max()
    n_meas_full_tf = len(gdf_timeframe)
    n_loc_full_tf = gdf_timeframe.geometry.nunique()

    logger.info("-- Full Dataset information --")
    logger.info(f"Number of samples: {n_meas_full:,}")
    logger.info(f"Number of site: {n_loc_full:,}")
    logger.info(f"Start year: {start_year:.0f}")
    logger.info(f"End year: {end_year:.0f}")

    logger.info("-- Cropped Dataset information --")
    logger.info(f"Number of samples: {n_meas_full_tf:,}")
    logger.info(f"Number of site: {n_loc_full_tf:,}")
    logger.info(f"Start year: {start_year_tf:.0f}")
    logger.info(f"End year: {end_year_tf:.0f}")

    gdf_timeframe_dropped = gdf_timeframe[~gdf_timeframe.less_than]
    df_plot = gdf_timeframe_dropped[["substance", "conc"]]

    pfas_four = ["PFOS", "PFOA", "PFHxS", "PFNA"]

    pfas_sum = [
        "PFBA",
        "PFPeA",
        "PFHxA",
        "PFHpA",
        "PFOA",
        "PFNA",
        "PFDA",
        "PFUnDA",
        "PFDoDA",
        "PFTrDA",
        "PFBS",
        "PFPeS",
        "PFHxS",
        "PFHpS",
        "PFOS",
        "PFDS",
        "PFNS",
        "PFUnDS",
        "PFDoDS",
        "PFTrDS",
    ]

    gdf_timeframe_for_sum = gdf_timeframe.copy()
    gdf_timeframe_for_sum.loc[gdf_timeframe_for_sum.less_than, "conc"] = np.nan

    # gdf_timeframe_for_sum["dayofyear"] = pd.to_datetime(gdf_timeframe_for_sum["date"], format='ISO8601').dt.dayofyear

    gdf_timeframe_for_sum["site"] = (
        gdf_timeframe_for_sum["dayofyear"].astype(int).astype(str)
        + "_"
        + gdf_timeframe_for_sum["year"].astype(int).astype(str)
        + "_"
        + gdf_timeframe_for_sum["geometry"].astype(str)
    )

    pfas_wide = gdf_timeframe_for_sum.pivot_table(
        index="site", columns="substance", values="conc", aggfunc="median"
    )

    pfas_wide["PFAS4"] = pfas_wide[pfas_four].sum(axis=1, min_count=1)
    pfas_wide["PFAS_avail"] = pfas_wide[pfas_sum].sum(axis=1, min_count=1)

    df_pfas4_long = (
        pfas_wide[["PFAS4"]]
        .rename(columns={"PFAS4": "conc"})
        .assign(substance=r"$\Sigma_4$PFAS")
        .reset_index()
    )

    df_pfassum_long = (
        pfas_wide[["PFAS_avail"]]
        .rename(columns={"PFAS_avail": "conc"})
        .assign(substance=r"$\Sigma_{20}$PFAS")
        .reset_index()
    )

    # append to plot dataframe
    df_plot = pd.concat(
        [
            df_plot,
            df_pfas4_long[["substance", "conc"]],
            df_pfassum_long[["substance", "conc"]],
        ],
        ignore_index=True,
    )

    df_plot["substance"] = df_plot["substance"].replace(
        {
            "L_PFBS": "Linear PFBS",
            "L_PFHxS": "Linear PFHxS",
            "L_PFHpS": "Linear PFHpS",
            "L_PFOS": "Linear PFOS",
            "L_PFOA": "Linear PFOA",
        }
    )

    sub_detected = (
        gdf_timeframe[~gdf_timeframe["less_than"]]
        .groupby("geometry")["substance"]
        .nunique()
        .reset_index(name="n_substances")
    )

    geom_detected = sub_detected.groupby("n_substances").size()

    # total (True + False)
    sub_total = (
        gdf_timeframe.groupby("geometry")["substance"]
        .nunique()
        .reset_index(name="n_substances")
    )

    geom_total = sub_total.groupby("n_substances").size()
    geom_counts = (
        pd.concat([geom_detected, geom_total], axis=1).fillna(0)
    ).sort_index()

    geom_counts.columns = ["Detected", "Total"]

    combined_plot(df_plot, counts, geom_counts, save_path)
    logger.info("--- Finished general information generation ---")


def combined_plot(df_plot, counts, geom_counts, save_path: Path) -> None:
    """Create and save summary plots for PFAS substances and sampling sites.

    The first figure contains two panels: a bar chart of substance detection
    frequencies and a boxplot of PFAS concentrations. The second figure shows
    the number of detected and monitored substances for sampling sites grouped
    by the total number of monitored substances.

    The generated figures are saved as:

    - ``substances_combined_alt41.pdf``: Detection frequencies and
      concentration distributions.
    - ``substances_combined_alt42.pdf``: Detected and monitored substances by
      sampling site.

    Args:
        df_plot: DataFrame containing concentration observations. It must
            contain ``substance`` and ``conc`` columns.
        counts: DataFrame containing substance detection statistics. It must
            contain ``frequency`` and ``total_n`` columns, with substances in
            the index.
        geom_counts: DataFrame containing counts of sampling sites grouped by
            the number of monitored substances. It must contain ``Detected``
            and ``Total`` columns.
        save_path: Directory in which the generated PDF figures are saved.

    Returns:
        None. The figures are saved to ``save_path``.
    """
    logger.info("--- Plotting general information ---")
    color_detect = "#88CCF1"
    sort_counts = [
        "TFA",
        "PFBA",
        "PFPeA",
        "PFHxA",
        "PFHpA",
        "PFOA",
        "PFNA",
        "PFDA",
        "PFUnDA",
        "PFDoDA",
        "PFTrDA",
        "PFTeDA",
        "PFHxDA",
        "PFODA",
        "PFBS",
        "PFPeS",
        "PFHxS",
        "PFHpS",
        "PFOS",
        "PFNS",
        "PFDS",
        "PFUnDS",
        "PFDoDS",
        "PFTrDS",
        "Linear PFOA",
        "Linear PFBS",
        "Linear PFHpS",
        "Linear PFHxS",
        "Linear PFOS",
        "6:2 FTCA",
        "4:2 FTS",
        "6:2 FTS",
        "8:2 FTS",
        "HFPO-DA",
        "ADONA",
        "FOSA",
        "N-Et-FOSA",
        "N-MeFOSAA",
        "EtFOSAA",
    ]
    df_plot_connect_new = df_plot[df_plot.substance.isin(sort_counts)]
    counts = counts.reindex(sort_counts)
    fig, axs = plt.subplots(
        2, 1, figsize=(9.5, 11), gridspec_kw={"height_ratios": [1.2, 1]}
    )

    ax_top = axs[0]

    counts.frequency.plot(kind="bar", ax=ax_top, width=0.8, color=color_detect)

    for i, (_, row) in enumerate(counts.iterrows()):

        total = row.total_n
        freq = row.frequency
        ax_top.text(
            i,
            # freq / 1.05,
            freq + 0.5,
            f"{int(total):,}",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=90,
        )

    ax_top.set_xlabel("")
    ax_top.set_ylabel("Detection frequency (%)")

    ax_top.spines[["right", "top"]].set_visible(False)
    ax_top.set_ylim([0, 100])

    ax_top.grid(axis="y", alpha=0.15, linewidth=0.6)

    ax_bottom = axs[1]

    sns.boxplot(
        data=df_plot_connect_new,
        ax=ax_bottom,
        x="substance",
        y="conc",
        # fliersize=1,
        flierprops={"marker": "x", "markersize": 1, "alpha": 1},
        order=sort_counts,
        color="black",
        linewidth=0.8,
        boxprops=dict(facecolor="none", edgecolor="black"),
        whiskerprops=dict(color="black", linewidth=0.8),
        capprops=dict(color="black", linewidth=0.8),
        medianprops=dict(color="black", linewidth=1.2),
    )

    ax_bottom.set_ylabel("Concentration (ng L$^{-1}$)")
    ax_bottom.set_xlabel("Substance / Group")
    ax_bottom.set_yscale("log")
    ax_bottom.tick_params(axis="x", rotation=90)
    ax_bottom.grid(axis="y", alpha=0.15, linewidth=0.6)

    ax_bottom.spines[["right", "top"]].set_visible(False)

    ax_top.text(
        -0.06,
        1.05,
        "a",
        transform=ax_top.transAxes,
        fontsize=11,
        fontweight="bold",
        va="top",
    )
    ax_bottom.text(
        -0.06,
        1.05,
        "b",
        transform=ax_bottom.transAxes,
        fontsize=11,
        fontweight="bold",
        va="top",
    )

    plt.subplots_adjust(left=0.1, hspace=0.28, right=0.95, top=0.88, bottom=0.05)
    plt.savefig(save_path / "substances_combined_alt41.pdf", bbox_inches="tight")
    plt.close(fig)
    # plt.show()

    fig, ax = plt.subplots(figsize=(13, 8))

    geom_counts.plot(
        kind="bar",
        ax=ax,
        width=0.8,
        color=[color_detect, "#D0D0D0"],
        # edgecolor="k",
        edgecolor="none",
        linewidth=0.2,
    )

    ax.grid(axis="y", alpha=0.15, linewidth=0.6)

    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:,.0f}"))

    ax.tick_params(axis="x", labelrotation=0)  # , labelsize=11)
    ax.tick_params(axis="y", labelrotation=0)  # , labelsize=11)
    ax.spines[["right", "top"]].set_visible(False)
    ax.grid(axis="y", alpha=0.15, linewidth=0.6)
    ax.legend(
        ["Detected", "Monitored"],
        title="",
        frameon=False,
        fontsize=12,
        loc="upper right",
    )
    ax.set_xlabel("Number of substances")  # , fontsize=11
    ax.set_ylabel("Number of sampling sites")  # , fontsize=11
    plt.subplots_adjust(left=0.1, hspace=0.28, right=0.95, top=0.88, bottom=0.05)
    plt.savefig(save_path / "substances_combined_alt42.pdf", bbox_inches="tight")
    # plt.show()


def main() -> None:
    input_path = Path("data/input/")
    # gdf: gpd.GeoDataFrame = gpd.read_file(input_path.joinpath("test_cutout.gpkg"))
    gdf = gpd.read_file(input_path.joinpath("pfas_data.gpkg"))
    basins: gpd.GeoDataFrame = gpd.read_file(
        input_path.joinpath("hybas_eu_lev12_v1c.shp")
    )
    gdf["month"] = pd.to_datetime(gdf["date"], format="ISO8601").dt.month
    gdf["dayofyear"] = pd.to_datetime(gdf["date"], format="ISO8601").dt.dayofyear
    save_path = Path("results/")

    basins = basins.to_crs(gdf.crs)  # pyright: ignore[reportArgumentType]
    gdf_timeframe = gdf[(gdf.year > 2018)]

    general_info(gdf, gdf_timeframe, save_path)


if __name__ == "__main__":
    main()
