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
from sklearn.cluster import DBSCAN


logger = logging.getLogger(__name__)

# 'cluster' function adapted from the PFAS DataHub https://colab.research.google.com/drive/1AL9Jw3AzcIpbhckIr2B0zEnc5NQ8nP9Z#scrollTo=lzMKd6HWOdAU 
def cluster(df, max_dist):
    df = df.copy()
    kms_per_radian = 6371.0088
    eps_rad = max_dist / kms_per_radian
    # represent points consistently as (lat, lon) and convert to radians to fit using haversine metric
    coords = df[['lat', 'lon']].values
    db = DBSCAN(eps=eps_rad, min_samples=1, algorithm='ball_tree', metric='haversine').fit(np.radians(coords))
    cluster_labels = db.labels_
    df['cluster_label'] = cluster_labels
    num_clusters = len(set(cluster_labels))

    # all done, print outcome
    logger.info(f'Clustered {len(df):,} points down to {num_clusters} clusters, for {100*(1 - float(num_clusters) / len(df)):.2f}% compression.')
    return df
# End of 'cluster' function

def general_info(gdf_timeframe, save_path: Path):
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

    start_year_tf = gdf_timeframe.year.min()
    end_year_tf = gdf_timeframe.year.max()
    n_meas_full_tf = len(gdf_timeframe)
    n_loc_full_tf = gdf_timeframe.geometry.nunique()
    n_quant_meas_full_tf = len(gdf_timeframe[~gdf_timeframe.less_than])
    n_loc_meas_full_tf = gdf_timeframe[~gdf_timeframe.less_than].geometry.nunique()

    logger.info("-- Dataset information --")
    logger.info(f"Start year: {start_year_tf:.0f}")
    logger.info(f"End year: {end_year_tf:.0f}")
    logger.info(f"Number of samples: {n_meas_full_tf:,}")
    logger.info(f"Number of site: {n_loc_full_tf:,}")
    logger.info(fr"Samples above LOD: {n_quant_meas_full_tf:,} ({n_quant_meas_full_tf/n_meas_full_tf*100:.1f}~\%)")
    logger.info(f"Number of sites with samples over LOD: {n_loc_meas_full_tf:,}")

    # On the PFAS Data Hub website is stated, that counting each individual coordinate as sampling location leads to an overestimation of the sampling. They show an example of clustering using K-MEANS, with a 200m minimal distance between 2 clusters
    df_cluster = gdf_timeframe.drop_duplicates(subset=['lat', 'lon'])
    clustered = cluster(df_cluster, max_dist=0.2)

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

    pfas_wide["PFAS4"] = pfas_wide[pfas_four].sum(axis=1, min_count=4)
    pfas_wide["PFAS_avail"] = pfas_wide[pfas_sum].sum(axis=1, min_count=10)

    res = []
    for i in range(1,21):
        dat = np.nan
        dat = pfas_wide[pfas_sum].sum(axis=1, min_count=i).dropna()
        # print(dat)
        res_dict = {
            "i" : i,
            "n" : len(dat),
            "median" : dat.median(),
            "mean" : dat.mean(),
            "exceed100" : len(dat[dat>100])
        }
        res.append(res_dict)
    # pd.DataFrame(res).to_excel("pfassum.xlsx", index=False)
    # logger.info("'Sum of 20 PFAS' dependency on number of included substances (i)")
    # logger.info(pd.DataFrame(res).set_index("i"))

    # print(pfas_wide)
    df_pfas4_long = (
        pfas_wide[["PFAS4"]]
        .rename(columns={"PFAS4": "conc"})
        .assign(substance="Sum of 4 PFAS")
        .reset_index()
    )
    # print(df_pfas4_long)
    # pfas_wide[pfas_four].to_excel("pfas4_input.xlsx")
    # pfas_wide[pfas_sum].to_excel("pfas20_input.xlsx")
    # df_pfas4_long.to_excel("pfas4.xlsx")
    df_pfassum_long = (
        pfas_wide[["PFAS_avail"]]
        .rename(columns={"PFAS_avail": "conc"})
        .assign(substance="Sum of 20 PFAS")
        .reset_index()
    )

    logger.info(f"Number of matches for 'Sum of 4 PFAS': {len(df_pfas4_long.dropna())}")
    logger.info(f"Number of 'Sum of 4 PFAS' > 2ng/L: {len(df_pfas4_long[df_pfas4_long.conc > 2])}")
    logger.info(f"Number of 'Sum of 4 PFAS' > 20ng/L: {len(df_pfas4_long[df_pfas4_long.conc > 20])}")
    logger.info(f"Number of matches for 'Sum of 20 PFAS': {len(df_pfassum_long.dropna())}")

    # append to plot dataframe
    df_plot = pd.concat(
        [
            df_plot,
            df_pfas4_long[["substance", "conc"]],
            df_pfassum_long[["substance", "conc"]],
        ],
        ignore_index=True
    )
    count = ((df_plot["substance"] == "TFA") & (df_plot["conc"] > 9000)).sum()
    logger.info("Number of TFA > 9,000 ng/L: %d", count)

    df_plot["substance"] = df_plot["substance"].replace(
        {
            "L_PFBS": "Linear PFBS",
            "L_PFHxS": "Linear PFHxS",
            "L_PFHpS": "Linear PFHpS",
            "L_PFOS": "Linear PFOS",
            "L_PFOA": "Linear PFOA",
        }
    )
    logger.info("Substance/Group Median:")
    logger.info(df_plot.groupby('substance')['conc'].median())
    logger.info("Substance/Group Mean:")
    logger.info(df_plot.groupby('substance')['conc'].mean())

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
    print("Sampling site(s) with most detected substances:\n", sub_detected[sub_detected.n_substances == max(sub_detected.n_substances)])
    # print(sub_total[sub_total.n_substances == max(sub_total.n_substances)])
    # exit()

    geom_total = sub_total.groupby("n_substances").size()
    geom_counts = (
        pd.concat([geom_detected, geom_total], axis=1).fillna(0)
    ).sort_index()

    geom_counts.columns = ["Detected", "Monitored"]

    combined_plot(df_plot, counts, geom_counts, save_path)
    logger.info("--- Finished general information generation ---")


def combined_plot(df_plot, counts, geom_counts, save_path: Path) -> None:
    """Create and save summary plots for PFAS substances and sampling sites.

    The first figure contains two panels: a bar chart of substance detection
    frequencies and a boxplot of PFAS concentrations. The second figure shows
    the number of detected and monitored substances for sampling sites grouped
    by the total number of monitored substances.

    The generated figures are saved as:

    - ``main_substances_combined_1.pdf``: Detection frequencies and
      concentration distributions.
    - ``main_substances_combined_2.pdf``: Detected and monitored substances by
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
        "Linear PFOS",
        "6:2 FTCA",
        "4:2 FTS",
        "6:2 FTS",
        "8:2 FTS",
        "HFPO-DA",
        "DONA",
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

    ax_bottom.set_ylabel("Concentration (ngL$^{-1}$)")
    ax_bottom.set_xlabel("Substance")
    ax_bottom.set_yscale("log")
    ax_bottom.tick_params(axis="x", rotation=90)
    ax_bottom.grid(axis="y", alpha=0.15, linewidth=0.6)

    ax_bottom.spines[["right", "top"]].set_visible(False)

    ax_top.text(
        -0.07,
        1.05,
        "a",
        transform=ax_top.transAxes,
        fontsize=11,
        fontweight="bold",
        va="top",
    )
    ax_bottom.text(
        -0.07,
        1.05,
        "b",
        transform=ax_bottom.transAxes,
        fontsize=11,
        fontweight="bold",
        va="top",
    )

    plt.subplots_adjust(
        left=0.1, 
        hspace=0.35, # 0.28
        right=0.95, 
        top=0.88, 
        bottom=0.05
        )
    plt.savefig(save_path / "main_substances_combined_1.pdf", bbox_inches="tight")
    plt.close(fig)
    # plt.show()
    
    fig, ax = plt.subplots(figsize=(7, 4.5)) # 13, 8
    logger.info("\n"+geom_counts.rename_axis("n_substances").reset_index().to_string(index = False, formatters={ 
        "Detected": lambda x: f"{x:,.0f}", 
        "Monitored": lambda x: f"{x:,.0f}", 
        }))

    # window, that contains ~50% of monitored substances
    s = geom_counts["Monitored"].sort_index()
    values = s.to_numpy()
    target = values.sum() / 2

    left = 0
    window_sum = 0
    best = None # (length, start position, end position, sum)

    for right, value in enumerate(values):
        window_sum += value

        while window_sum >= target:
            length = right - left + 1
            if best is None or length < best[0]:
                best = (length, left, right, window_sum)

            window_sum -= values[left]
            left += 1

    if best is None:
        print("No range found.")
    else:
        _, start, end, range_sum = best
        print("Index range:", s.index[start], "to", s.index[end])
        print("Range sum:", range_sum)
    # End

    # Plotting
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

    ax.tick_params(axis="x", labelrotation=90)  # , labelsize=11)
    ax.tick_params(axis="y", labelrotation=0)  # , labelsize=11)
    ax.spines[["right", "top"]].set_visible(False)
    ax.grid(axis="y", alpha=0.15, linewidth=0.6)
    # ax.legend(
    #     ["Detected", "Monitored"],
    #     title="",
    #     frameon=False,
    #     fontsize=12,
    #     loc="upper right",
    # )
    ax.legend(
        ["Detected", "Monitored"],
        title="",
        frameon=False,
        fontsize=12,
        loc="upper center",
        # bbox_to_anchor=(0.5, -0.40),
        ncol = 2
    )
    ax.set_xlabel("Number of substances")  # , fontsize=11
    ax.set_ylabel("Number of sampling sites")  # , fontsize=11
    plt.subplots_adjust(left=0.1, hspace=0.28, right=0.95, top=0.88, bottom=0.05)
    plt.savefig(save_path / "main_substances_combined_2.pdf", bbox_inches="tight")
    # plt.show()


def main() -> None:
    input_path = Path("data/input/")
    # gdf: gpd.GeoDataFrame = gpd.read_file(input_path.joinpath("test_cutout.gpkg"))
    gdf = gpd.read_file(input_path.joinpath("pfas_data.gpkg"))
    # basins: gpd.GeoDataFrame = gpd.read_file(
    #     input_path.joinpath("hybas_eu_lev12_v1c.shp")
    # )
    # gdf["month"] = pd.to_datetime(gdf["date"], format="ISO8601").dt.month
    gdf["dayofyear"] = pd.to_datetime(gdf["date"], format="ISO8601").dt.dayofyear
    save_path = Path("results/")

    # basins = basins.to_crs(gdf.crs)  # pyright: ignore[reportArgumentType]
    gdf_timeframe = gdf[(gdf.year > 2018)]

    general_info(gdf_timeframe, save_path)


if __name__ == "__main__":
    main()
