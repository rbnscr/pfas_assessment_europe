"""Analyse and visualise PFAS concentration ratios across European basins.

This module provides utilities to calculate substance concentration ratios, perform paired statistical tests, and generate boxplots and heatmaps for PFAS data.
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
from scipy.stats import (
    ttest_rel,
    wilcoxon,
)

logger = logging.getLogger(__name__)


def boxplot_ratios(df: pd.DataFrame, substances: list[tuple]):
    """Calculate concentration ratios for pairs of substances.

    Rows containing missing values for either substance in a pair are excluded.
    The resulting ratios are returned in a long-format DataFrame suitable for
    plotting with Seaborn.

    Args:
        df: DataFrame containing substance concentration columns and a
            ``HYBAS_ID`` column.
        substances: Iterable of two-element tuples. Each tuple contains the
            numerator and denominator substance column names.

    Returns:
        A DataFrame with the following columns:

        - ``value``: Calculated concentration ratio.
        - ``ratio``: Ratio name in the format ``"substance1/substance2"``.
        - ``HYBAS_ID``: Basin identifier associated with each observation.
    """
    dfs = []

    for sub1, sub2 in substances:
        ratio_name = f"{sub1}/{sub2}"
        df_ = df.dropna(subset=[sub1, sub2]).copy()
        ratio = (df_[sub1] / df_[sub2]).to_frame(name="value")
        ratio["ratio"] = ratio_name
        ratio["HYBAS_ID"] = df_["HYBAS_ID"].values
        dfs.append(ratio)
    df_ratios = pd.concat(dfs, ignore_index=True)
    return df_ratios


def test_significant_difference(df: pd.DataFrame, substances: list[tuple]):
    """Test whether the first substance has higher concentrations than the second.

    For each substance pair, this function performs a one-sided paired Wilcoxon
    signed-rank test on the original concentrations and a one-sided paired
    t-test on log10-transformed concentrations.

    The alternative hypothesis for both tests is that the first substance has
    higher concentrations than the second substance.

    Args:
        df: DataFrame containing paired substance concentration columns.
        substances: Iterable of two-element tuples. Each tuple contains the
            first and second substance column names to compare.

    Returns:
        A DataFrame containing one row per substance comparison. The result
        includes sample size, medians, median differences, median ratios,
        test statistics, p-values, and significance indicators for both tests.
    """
    results = []
    for sub1, sub2 in substances:
        df_ = df.dropna(subset=[sub1, sub2]).copy()
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="One or more sample arguments is too small.*"
            )
            stat_wilc, p_wilc = wilcoxon(
                df_[sub1],
                df_[sub2],
                alternative="greater",  # One-sided test, because we have reasonal suspicion, that PFCA are higher. Wilcoxon, because of paired observations
            )
            stat_tt, p_tt = ttest_rel(
                np.log10(df_[sub1]), np.log10(df_[sub2]), alternative="greater"
            )
        results.append(
            {
                "comparison": f"{sub1} > {sub2}",
                "n": len(df_),
                "median_substance1": df_[sub1].median(),
                "median_substance2": df_[sub2].median(),
                "median_difference": (df_[sub1] - df_[sub2]).median(),
                "median_ratio": (df_[sub1] / df_[sub2]).median(),
                "stat_wilc": stat_wilc,
                "p_value_wilc": p_wilc,
                "significant_wilc": p_wilc < 0.05,  # substance1 larger than substance2
                "stat_tt": stat_tt,
                "p_value_tt": p_tt,
                "significant_tt": p_tt < 0.05,
            }
        )
    return pd.DataFrame(results)


def conc_ratio(gdf, basins, save_path: Path):
    """Generate PFAS concentration-ratio plots by hydrological basin.

    Sampling locations are spatially joined to level-04 HydroBASINS polygons.
    Locations that do not fall within a basin are assigned to the nearest basin.
    The function then calculates median PFAS concentrations per sampling event,
    computes concentration ratios for predefined PFAS pairs, performs paired
    statistical tests, and saves a boxplot and heatmap as PDF files.

    The generated files are:

    - ``appendix_boxplot_concentration_ratios.pdf``
    - ``appendix_heatmap_concentration_ratios.pdf``

    Only observations from years after 2018 and observations not marked as
    below the reporting limit are used for concentration-ratio calculations.

    Args:
        gdf: GeoDataFrame containing PFAS observations. It must contain
            ``year``, ``dayofyear``, ``geometry``, ``less_than``, ``substance``,
            and ``conc`` columns.
        basins: GeoDataFrame containing basin polygons and a ``HYBAS_ID``
            column.
        save_path: Directory in which the generated PDF plots are saved.

    Returns:
        None. The plots are saved to ``save_path``.
    """
    logger.info("--- Starting Concentration Ratios ---")
    lev04_rename = {
        2040020320: "Garonne",
        2040016230: "Rhône / Ebro",
        2040021030: "Loire",
        2040022150: "Seine",
        2040021040: "Brittany / Normandy",
        2040022160: "Maas",
        2040023010: "Rhine",
        2040023020: "Weser / Ems",
        2040048790: "United Kingdom",
        2040014550: "Tiber",
        2040046500: "Sicily",
        2040012730: "Po",
        2040047500: "Corsica",
        2040543160: "Lower Danube",
        2040539930: "Upper Danube",
        2040024170: "Elbe",
        2040026060: "Oder",
        2040026930: "Nemunas",
        2040031500: "Baltic (Southern Sweden)",
        2040028670: "Baltic (Western Finland)",
        2040033480: "Norway",
        2040028310: "Newa",
        2040027320: "Daugava",
        2040026920: "Nyoman",
        2040027330: "Narva",
        2040019150: "Duero",
        2040019160: "Sil",
        2040009230: "Mediterranean Balkans",
        2040008490: "Prut",
        2040548500: "Tysa",
        2040540100: "Drava",
        2040548700: "Mura-Drava-Danube",
        2040555780: "Sava",
    }
    id_name = "HYBAS_ID"
    basins[id_name] = basins[id_name].astype("object")
    basins.loc[~basins[id_name].isin(lev04_rename.keys()), id_name] = "Other"
    basins[id_name] = basins[id_name].replace(lev04_rename)
    basins[id_name] = basins[id_name].astype("string")

    within = gpd.sjoin(
        gdf[gdf.year > 2018].to_crs("EPSG:3035"),
        basins[[id_name, "geometry"]].to_crs("EPSG:3035"),
        how="left",
        predicate="within",
    )

    missing = within[within[id_name].isna()].drop(columns=id_name)

    if "index_right" in missing.columns:
        missing = missing.drop(columns="index_right")

    nearest = gpd.sjoin_nearest(
        missing.to_crs("EPSG:3035"),
        basins[[id_name, "geometry"]].to_crs("EPSG:3035"),
        how="left",
    )
    gdf_joined = pd.concat([within[within[id_name].notna()], nearest])

    gdf_joined["site"] = (
        gdf_joined["dayofyear"].astype(int).astype(str)
        + "_"
        + gdf_joined["year"].astype(int).astype(str)
        + "_"
        + gdf_joined["geometry"].astype(str)
    )
    pfas_wide = gdf_joined[~gdf_joined.less_than].pivot_table(
        index="site", columns="substance", values="conc", aggfunc="median"
    )
    geom = gdf_joined.groupby("site")["HYBAS_ID"].first()
    pfas_wide = pfas_wide.join(geom)

    subst_for_bp_rat = [
        ("PFNA", "PFDA"),
        ("PFUnDA", "PFDoDA"),
        ("PFTrDA", "PFTeDA"),
        ("PFBA", "PFBS"),
        ("PFPeA", "PFPeS"),
        ("PFHxA", "PFHxS"),
        ("PFHpA", "PFHpS"),
        ("PFOA", "PFOS"),
        ("PFNA", "PFNS"),
        ("PFDA", "PFDS"),
        ("PFUnDA", "PFUnDS"),
        ("PFDoDA", "PFDoDS"),
        ("PFTrDA", "PFTrDS"),
    ]

    df_ratios = boxplot_ratios(df=pfas_wide, substances=subst_for_bp_rat)
    df_greater = test_significant_difference(df=pfas_wide, substances=subst_for_bp_rat)
    logger.info(df_greater)

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.boxplot(
        data=df_ratios,
        ax=ax,
        x="ratio",
        y="value",
        flierprops={"marker": "x", "markersize": 1, "alpha": 1},
        color="black",
        linewidth=0.8,
        boxprops=dict(facecolor="none", edgecolor="black"),
        whiskerprops=dict(color="black", linewidth=0.8),
        capprops=dict(color="black", linewidth=0.8),
        medianprops=dict(color="black", linewidth=1.2),
    )

    ax.set_ylabel("Concentration ratios (-)")
    ax.set_xlabel("Substances")
    ax.set_yscale("log")
    ax.tick_params(axis="x", rotation=90)
    ax.grid(axis="y", alpha=0.15, linewidth=0.6)

    ax.axvline(x=2.5, color="k", linewidth=0.2)

    plt.tight_layout()
    plt.savefig(
        save_path / "appendix_boxplot_concentration_ratios.pdf", bbox_inches="tight"
    )

    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 11))

    heatmap_data = df_ratios.groupby(["HYBAS_ID", "ratio"])["value"].median().unstack()

    ratio_order = []
    for sub1, sub2 in subst_for_bp_rat:
        ratio_name = f"{sub1}/{sub2}"
        if ratio_name in heatmap_data.columns:
            ratio_order.append(ratio_name)

    heatmap_data = heatmap_data[ratio_order]
    heatmap_log2 = np.log2(heatmap_data)

    sns.heatmap(
        heatmap_log2,
        cmap="coolwarm",
        center=0,
        linewidths=0.3,
        cbar_kws={"label": "log$_2$(median ratio)"},
        annot=True,
        fmt=".2f",
        annot_kws={"fontsize": 10},
    )

    ax.set_xlabel("Substances")
    ax.set_ylabel("Basin name")

    ax.tick_params(axis="x", rotation=90, labelsize=11)

    ax.tick_params(axis="y", rotation=0, labelsize=11)

    plt.tight_layout()
    plt.savefig(
        save_path / "appendix_heatmap_concentration_ratios.pdf", bbox_inches="tight"
    )
    plt.close(fig)
    logger.info("--- Finished Concentration Ratios ---")


def main() -> None:
    input_path = Path("data/input/")

    gdf = gpd.read_file(input_path.joinpath("pfas_data.gpkg"))
    basins: gpd.GeoDataFrame = gpd.read_file(
        input_path.joinpath("hybas_eu_lev04_v1c.shp")
    )
    gdf["month"] = pd.to_datetime(gdf["date"], format="ISO8601").dt.month
    gdf["dayofyear"] = pd.to_datetime(gdf["date"], format="ISO8601").dt.dayofyear
    save_path = Path("results/")

    basins = basins.to_crs(gdf.crs)  # pyright: ignore[reportArgumentType]
    gdf_timeframe = gdf[(gdf.year > 2018)]

    conc_ratio(gdf_timeframe, basins, save_path)


if __name__ == "__main__":
    main()
