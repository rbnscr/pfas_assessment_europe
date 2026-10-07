"""Evaluate and visualise ecological risks associated with PFAS concentrations.

This module calculates risk quotients (RQ) for PFAS substances using
substance-specific freshwater advisory values. It assigns observations to
ecological risk categories, spatially associates observations with
HydroBASINS basins, and summarises the affected basin area by risk
category.

The module provides visualisations including:

- Boxplots of risk quotients by substance
- Stacked bar plots of basin areas by risk category
- Stacked bar plots of the fraction of basin area by risk category

Generated figures are saved as PDF files in the specified output directory.
"""

import logging
import warnings
from pathlib import (
    Path,
)

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from dateutil.parser._parser import (
    UnknownTimezoneWarning,
)
from matplotlib.ticker import (
    FuncFormatter,
    LogLocator,
)

logger = logging.getLogger(__name__)


RISK_LABELS = [
    "unknown risk",
    "negligible risk",
    "low risk",
    "medium risk",
    "high risk",
]

RISK_COLORS = {
    "unknown risk": "#D3D3D3",
    "negligible risk": "#60F074",
    "low risk": "#6080F0",
    "medium risk": "#F0C960",
    "high risk": "#F06085",
}


def ere_boxplot(res_df_dropped, save_path: Path) -> None:
    """Create and save a boxplot of ecological risk quotients.

    Risk quotients are plotted separately for each substance. The y-axis uses a 
    logarithmic scale, with reference lines indicating risk quotients of 
    0.01, 0.1, and 1.

    The resulting figure is saved as
    ``main_risk_quotient_boxplot.pdf`` in ``save_path``.

    Args:
        res_df_dropped: DataFrame containing ecological risk-quotient results.
            It must contain ``substance`` and ``RQ`` columns. Rows with missing
            risk quotients are excluded from the boxplot.
        save_path: Directory in which the boxplot PDF is saved.

    Returns:
        None.
    """
    sort_substances = [
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
            "DONA",
            "FOSA",
            "N-Et-FOSA",
            "N-MeFOSAA",
            "EtFOSAA",
        ]
    sort_substances = [substance for substance in sort_substances if substance in res_df_dropped.substance.unique()]
    logger.info("Creating boxplot")
    # RQ Boxplot
    fig, ax = plt.subplots(figsize=(10, 6))

    # logger.info("\n"+geom_counts.rename_axis("n_substances").reset_index().to_string(index = False, formatters={ 
    #         "Detected": lambda x: f"{x:,.0f}", 
    #         "Monitored": lambda x: f"{x:,.0f}", 
    #         }))

    sns.boxplot(
        data=res_df_dropped.dropna(subset="RQ"),
        ax=ax,
        x="substance",
        y="RQ",
        flierprops={"marker": "x", "markersize": 1, "alpha": 1},
        order=sort_substances,
        color="black",
        linewidth=0.8,
        boxprops=dict(facecolor="none", edgecolor="black"),
        whiskerprops=dict(color="black", linewidth=0.8),
        capprops=dict(color="black", linewidth=0.8),
        medianprops=dict(color="black", linewidth=1.2),
    )

    ax.axhline(1, color="#F06085", lw=0.5, linestyle=":")
    ax.axhline(0.1, color="#F0C960", lw=0.5, linestyle=":")
    ax.axhline(0.01, color="#6080F0", lw=0.5, linestyle=":")

    ax.set_ylabel("Risk quotient, RQ (-)", fontsize=10)
    ax.set_xlabel("Substance", fontsize=10)
    ax.set_yscale("log")
    ax.tick_params(axis="x", rotation=90)

    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0,), numticks=20))
    ax.grid(axis="y", alpha=0.15, linewidth=0.6)

    for y, label in [(1, "RQ = 1"), (0.1, "RQ = 0.1"), (0.01, "RQ = 0.01")]:
        ax.text(
            1.01,
            y,
            label,
            transform=ax.get_yaxis_transform(),
            va="center",
            ha="left",
            fontsize=10,
            color="k",
        )

    plt.tight_layout()
    logger.info(f"Saving boxplot to {save_path}")
    plt.savefig(save_path / "main_risk_quotient_boxplot.pdf", bbox_inches="tight")
    plt.close(fig)


def ere_dual_plot(pivot, pivot_fraction, save_path: Path) -> None:
    """Create and save stacked bar plots of basin-area risk categories.

    The function creates a two-panel figure. The first panel shows the total
    basin area assigned to each ecological risk category for each
    substance. The second panel shows the corresponding fraction of the basin area.

    Risk categories are colored according to ``RISK_COLORS``. The resulting
    figure is saved as ``main_risk_eval_by_basins.pdf`` in ``save_path``.

    Args:
        pivot: DataFrame containing basin areas by substance and risk
            category. The index should contain substances and the columns
            should contain risk categories.
        pivot_fraction: DataFrame containing the fraction of basin area by
            substance and risk category. Its index and columns should
            correspond to those of ``pivot``.
        save_path: Directory in which the plot PDF is saved.

    Returns:
        None.
    """
    logger.info("Creating dual plot for ecological risk evaluation")
    colors = RISK_COLORS

    fig, axes = plt.subplots(
        2, 1, figsize=(10, 10), gridspec_kw={"height_ratios": [1.5, 1]}
    )

    ax1 = axes[0]
    ax2 = axes[1]

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UnknownTimezoneWarning)
        bottom = None
        for col in pivot.columns:
            ax1.bar(
                pivot.index, pivot[col], bottom=bottom, label=col, color=colors[col]
            )

            if bottom is None:
                bottom = pivot[col].copy()
            else:
                bottom += pivot[col]

        bottom = None

        for col in pivot_fraction.columns:
            ax2.bar(
                pivot_fraction.index,
                pivot_fraction[col],
                bottom=bottom,
                label=col,
                color=colors[col],
            )

            if bottom is None:
                bottom = pivot_fraction[col].copy()
            else:
                bottom += pivot_fraction[col]

    ax1.set_ylabel("Basin area (km²)")
    ax2.set_ylabel("Fraction of basin area (-)")
    ax1.tick_params(labelbottom=False)
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:,.0f}"))
    ax2.tick_params(axis="x", labelrotation=90)
    ax2.set_xlabel("Substance")

    ax2.legend(
        title="Risk Category",
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.40),
        ncol=5,
        labels=[
            "Unknown risk (< LOQ)",
            "Negligible risk",
            "Low risk",
            "Medium risk",
            "High risk",
        ],
    )

    ax1.grid(axis="y", alpha=0.15, linewidth=0.6)

    ax2.grid(axis="y", alpha=0.15, linewidth=0.6)

    plt.tight_layout()
    ax1.text(
        -0.07,
        1.03,
        "a",
        transform=ax1.transAxes,
        fontsize=11,
        fontweight="bold",
        va="top",
    )
    ax2.text(
        -0.07,
        1.03,
        "b",
        transform=ax2.transAxes,
        fontsize=11,
        fontweight="bold",
        va="top",
    )

    logger.info(f"Saving plot to {save_path}")
    plt.savefig(save_path / "main_risk_eval_by_basins.pdf")
    plt.close(fig)

def ere_single_plot(pivot, save_path: Path) -> None:
    """Create and save stacked bar plots of basin-area risk categories.

    Risk categories are colored according to ``RISK_COLORS``. The resulting
    figure is saved as ``appendix_risk_eval_by_basins_single.pdf`` in ``save_path``.

    Args:
        pivot: DataFrame containing basin areas by substance and risk
            category. The index should contain substances and the columns
            should contain risk categories.
        save_path: Directory in which the plot PDF is saved.

    Returns:
        None.
    """
    logger.info("Creating plot for ecological risk evaluation")
    colors = RISK_COLORS

    fig, ax = plt.subplots(
        figsize=(7, 5)
    )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UnknownTimezoneWarning)
        bottom = None
        for col in pivot.columns:
            ax.bar(
                pivot.index, pivot[col], bottom=bottom, label=col, color=colors[col]
            )

            if bottom is None:
                bottom = pivot[col].copy()
            else:
                bottom += pivot[col]

    ax.set_ylabel("Basin area (km²)")
    ax.tick_params(axis="x", labelrotation=90)
    ax.set_xlabel("Substance")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:,.0f}"))
    ax.grid(axis="y", alpha=0.15, linewidth=0.6)
    
    ax.legend(
        title="Risk Category",
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.4),
        ncol=3,
        labels=[
            "Unknown risk (< LOQ)",
            "Negligible risk",
            "Low risk",
            "Medium risk",
            "High risk",
        ],
    )

    # plt.tight_layout()
    # plt.subplots_adjust(
    #     left=0.14, 
    #     bottom=0.38, 
    #     right=0.99, 
    #     top=0.99, 
    #     wspace=None, 
    #     hspace=None
    #     )
    plt.subplots_adjust(
            left=0.17, 
            bottom=0.40, 
            right=0.98, 
            top=0.99, 
            wspace=None, 
            hspace=None
            )
    logger.info(f"Saving plot to {save_path}")

    plt.savefig(save_path / "appendix_risk_eval_by_basins_single.pdf")
    plt.close(fig)

def eco_risk_eval(gdf, basins, save_path) -> None:
    """Evaluate and visualize ecological risk by PFAS substance and basin.

    The function calculates risk quotients using substance-specific freshwater
    advisory values, assigns each observation to an ecological risk category,
    and spatially joins observations to HydroBASINS polygons.

    It generates:

    - A boxplot of risk quotients by substance.
    - A summary of basin area assigned to each risk category.
    - A summary of the fraction of basin area assigned to each risk category.
    - A stacked-bar plot showing absolute basin-area risk.

    Observations marked as below the limit of detection are assigned to the
    ``unknown risk`` category for the spatial risk assessment. They are
    excluded from the risk-quotient boxplot.

    The generated figures are saved in ``save_path`` by
    :func:`ere_boxplot` and :func:`ere_dual_plot`.

    Args:
        gdf: GeoDataFrame containing PFAS observations. It must contain
            ``geometry``, ``substance``, ``conc``, and ``less_than`` columns.
        basins: GeoDataFrame containing basin polygons. It must contain
            ``HYBAS_ID`` and ``SUB_AREA`` columns.
        save_path: Directory in which the generated figures are saved.

    Returns:
        None. The ecological risk results are logged and the figures are
        saved to disk.
    """
    logger.info("--- Starting ecological risk evaluation ---")
    freshwater_advisory_value: dict = {
        "PFBA": 4400,
        # "PFBA" : 0.0044, # original value by NORMAN Database
        "PFNA": 0.44,
        "PFDA": 0.628571429,
        "PFOS": 0.65,
        "PFUnDA": 1.1,
        "PFDoDA": 1.466667,
        "PFTrDA": 2.66667,
        "PFHpS": 3.384615385,
        "PFOA": 4.4,
        "PFHxS": 7.33333,
        "PFHpA": 8.712871287,
        "PFPeS": 14.6422649,
        "PFTeDA": 14.66667,
        "PFPeA": 146.666667,
        "PFHxA": 440,
        "TFA": 109797.44,
        "PFBS": 4400,
        "PFDS": 162.34,
        "6:2 FTS": 621.84,
        "8:2 FTS": 252.69,
        "HFPO-DA": 1351.22,
        "FOSA": 166.74,
        "PPFBS": 451.9866233,
        "EtFOSAA": 300.8,
        "N-MeFOSAA": 410.93,
        "4:2 FTS": 2970.01,
        "PFODA": 220,
        "PFHxDA": 220,
        "PFNS": 216.39,
        "PFDoDS": 116.73,
        "6:2 FTCA": 433.34,
        "N-Et-FOSA": 191.29,
        "DONA" : 1746.02
    }

    df_sel = gdf[["geometry", "substance", "conc", "less_than"]].copy()
    # -1 is used only to assign below-LOD observations
    # to the "unknown risk" category; it is not a concentration.
    df_sel.loc[df_sel["less_than"], "conc"] = -1

    df_sel = df_sel.loc[df_sel["substance"].isin(freshwater_advisory_value),].copy()

    df_sel.loc[:, "advisory_value"] = df_sel["substance"].map(freshwater_advisory_value)

    df_sel.loc[:, "RQ"] = df_sel["conc"] / df_sel["advisory_value"]

    df_sel = df_sel.drop(columns="advisory_value")
    res_df = df_sel.copy()

    gdf_joined = gpd.sjoin(
        basins,
        res_df[["geometry", "substance", "conc", "less_than", "RQ"]],
        how="left",
        predicate="contains",
    )

    gdf_joined = gdf_joined.dropna(subset="RQ")
    res_df_drop: gpd.GeoDataFrame = gdf_joined.sort_values(
        by="RQ", ascending=False
    ).copy()
    bins = [-np.inf, 0, 0.01, 0.1, 1, np.inf]
    labels = RISK_LABELS

    # RQ < 0
    # 0 <= RQ < 0.01
    # 0.01 <= RQ < 0.1
    # 0.1 <= RQ < 1
    # RQ >= 1

    res_df_drop["risk_category"] = pd.cut(
        res_df_drop["RQ"], bins=bins, labels=labels, right=False, include_lowest=True
    ) 

    res_df_drop["risk_category"] = pd.Categorical(
        res_df_drop["risk_category"], categories=labels, ordered=True
    )

    res_df_dropped = res_df.loc[~res_df["less_than"]].copy()

    counts_by_substance = ( 
        res_df_drop 
        .groupby(["substance", "risk_category"], observed=False) 
        .size() 
        .unstack("risk_category", fill_value=0) 
        .reindex(columns=labels, fill_value=0) 
        .astype(int) 
        )
    median_by_substance = ( 
        res_df_drop[res_df_drop["RQ"]>0]
        .groupby(["substance"])["RQ"]
        .median() 
        )

    with pd.option_context("display.max_colwidth", None):
        logger.info(
            "Counts\n%s",
            counts_by_substance.to_string()
            )
        logger.info(
            "Median\n%s",
            median_by_substance.to_string()
            )

    ere_boxplot(res_df_dropped, save_path)

    """
    counts = (
        res_df_drop
        .groupby(["substance", "HYBAS_ID", "risk_category"])
        .size()
        .reset_index(name="count")
    )

    substance_hybas = (
        res_df_drop
        .groupby(["substance", "HYBAS_ID"])
        .size()
        .reset_index(name="count")
    )

    with pd.ExcelWriter("substance_hybas_risk_counts.xlsx") as writer:
        counts.to_excel(writer, sheet_name="Risk counts", index=False)
        substance_hybas.to_excel(
            writer,
            sheet_name="Substance HYBAS counts",
            index=False
        )

    combination_sizes = ( 
        res_df_drop 
        .groupby(["substance", "HYBAS_ID"]) 
        .size() 
        )
    
    # x = 2
    results = []
    for x in range(0, 101, 5):
        valid_combinations = combination_sizes[combination_sizes >= x].index 

        res_df_filtered = ( 
            res_df_drop
            .set_index(["substance", "HYBAS_ID"])
            .loc[valid_combinations]
            .reset_index() 
            )


        # pivot, pivot_fraction = prepare_df_for_ere(res_df_drop)
        pivot, pivot_fraction = prepare_df_for_ere(res_df_filtered)
        # entry = {
        #     "x": x,
        #     "df" : pivot
        # }
        # results.append(entry)
        pivot_long = ( 
            pivot 
            .reset_index() 
            .melt( 
                id_vars="substance", 
                var_name="risk_category", 
                value_name="area" 
                ) 
            ) 
        pivot_long["x"] = x 
        results.append(pivot_long)

    # Combine all thresholds 
    plot_df = pd.concat(results, ignore_index=True) 
    # plot_df = plot_df[plot_df.risk_category == "high risk"]

    # Plot 
    g = sns.relplot( 
        data=plot_df, 
        x="x",
        y="area", 
        hue="risk_category", 
        col="substance", 
        col_wrap=4, 
        kind="line", 
        marker="o", 
        height=4, 
        aspect=1 
        ) 
    g.set_axis_labels("Minimum combination size (x)", "Area (km²)") 
    plt.yscale("log")
    plt.tight_layout() 
    plt.savefig(save_path / "extra_ere_basin_effect.pdf")
    """

    pivot, pivot_fraction = prepare_df_for_ere(res_df_drop)
    # print(pivot, pivot_fraction)
    # ere_dual_plot(pivot, pivot_fraction, save_path)
    ere_single_plot(pivot, save_path)

    logger.info(f"Area (km²)\n{pivot.sort_values(by="high risk", ascending=False).to_string()}")
    logger.info(f"Total area (km²)\n{pivot.sum(axis=1).sort_values(ascending=False).to_string()}")

    logger.info("--- Finished ecological risk evaluation ---")

def prepare_df_for_ere(res_df_drop):
    df_worst = res_df_drop.sort_values(
        "risk_category", ascending=False
    ).drop_duplicates(subset=["HYBAS_ID", "substance"])

    total_area = (
        df_worst.groupby("substance")["SUB_AREA"]
        .sum()
        .reset_index(name="total_area_all")
    )

    summary_basin_rq = (
        df_worst.groupby(["substance", "risk_category"], observed=True)
        .agg(cat_area_km=("SUB_AREA", "sum"))
        .reset_index()
        .copy()
    )

    summary_basin_rq = summary_basin_rq.merge(total_area, on="substance").copy()

    summary_basin_rq["fraction_area"] = (
        summary_basin_rq["cat_area_km"] / summary_basin_rq["total_area_all"]
    )

    pivot = summary_basin_rq.pivot(
        index="substance", columns="risk_category", values="cat_area_km"
    ).fillna(0)

    pivot_fraction = summary_basin_rq.pivot(
        index="substance", columns="risk_category", values="fraction_area"
    ).fillna(0)

    # logger.info(pivot)
    risk_order = [
        "unknown risk",
        "negligible risk",
        "low risk",
        "medium risk",
        "high risk",
    ]
    sort_substances = [
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
            "DONA",
            "FOSA",
            "N-Et-FOSA",
            "N-MeFOSAA",
            "EtFOSAA",
        ]
    pivot = pivot[risk_order]
    pivot = pivot.reindex(sort_substances).dropna()
    pivot_fraction = pivot_fraction.reindex(sort_substances).dropna()
    return pivot,pivot_fraction


def main() -> None:
    input_path = Path("data/input/")
    # gdf: gpd.GeoDataFrame = gpd.read_file(input_path.joinpath("test_cutout.gpkg"))
    gdf = gpd.read_file(input_path.joinpath("pfas_data.gpkg"))
    basins: gpd.GeoDataFrame = gpd.read_file(
        input_path.joinpath("hybas_eu_lev12_v1c.shp")
    )

    save_path = Path("results/")
    basins = basins.to_crs(gdf.crs)  # pyright: ignore[reportArgumentType]
    eco_risk_eval(gdf, basins, save_path)


if __name__ == "__main__":
    main()
