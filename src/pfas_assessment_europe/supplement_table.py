"""Create supplementary PFAS assessment tables for Excel export.

This module calculates supplementary statistics for PFAS monitoring data,
including concentration-ratio counts by basin, Wilcoxon test results, country-
level detection statistics, substance-level summary statistics, predicted
no-effect concentrations, and inferred limits of detection.

The resulting tables are written to an Excel workbook containing separate
worksheets for each statistical summary.
"""

import logging
from pathlib import (
    Path,
)

import geopandas as gpd
import numpy as np
import pandas as pd

from pfas_assessment_europe.concentration_ratios import (
    boxplot_ratios,
    test_significant_difference,
)

# from pfas_assessment_europe.constants import HYBAS_RIVER_RENAME

logger = logging.getLogger(__name__)


def create_supplementary_table(
    gdf_timeframe: gpd.GeoDataFrame,
    # basins: gpd.GeoDataFrame,
    additional_information: dict,
    save_path: Path,
) -> None:
    """Calculate and export supplementary PFAS assessment tables.

    The function spatially assigns PFAS observations to HydroBASINS regions
    and European countries. It then calculates concentration-ratio counts,
    Wilcoxon test results, country-level detection statistics, substance-level
    summary statistics, predicted no-effect concentrations, and inferred
    minimum and maximum detection or quantification limits.

    The resulting tables are written to the file: 
    ``appendix_SupplementaryInformation-B.xlsx``

    Each worksheet includes a caption in its first row. The data table starts
    in the second row.

    Args:
        gdf_timeframe: GeoDataFrame containing PFAS observations. It must
            include ``geometry``, ``substance``, ``conc``, ``less_than``,
            ``dayofyear``, and ``year`` columns.
        basins: GeoDataFrame containing HydroBASINS geometries and a
            ``HYBAS_ID`` column.
        countries: GeoDataFrame containing country geometries and ``ADMIN``
            and ``CONTINENT`` columns.
        save_path: Directory in which the Excel workbook is saved.

    Returns:
        None. The supplementary tables are written to an Excel workbook in
        ``save_path``.
    """
    logger.info("--- Starting computation for Excel Table in SI B ---")
    # Create subset
    gdf_timeframe_less = gdf_timeframe[gdf_timeframe.less_than]

    logger.info("Computing sheet 'Ratio counts'")
    # Number of observations where two substances are present at the same time and geometry in a basin

    # id_name = "HYBAS_ID"
    # # Rename basins
    # lev04_rename = HYBAS_RIVER_RENAME
    # basins[id_name] = basins[id_name].astype("object")
    # basins["id"] = basins[id_name]
    # basins[id_name] = basins[id_name].replace(lev04_rename)
    # basins[id_name] = basins[id_name].astype("string")

    # within = gpd.sjoin(
    #     gdf_timeframe.to_crs("EPSG:3035"),
    #     basins[[id_name, "geometry"]].to_crs("EPSG:3035"),
    #     how="left",
    #     predicate="within",
    # )

    # missing = within[within[id_name].isna()].drop(columns=id_name)

    # if "index_right" in missing.columns:
    #     missing = missing.drop(columns="index_right")

    # nearest = gpd.sjoin_nearest(
    #     missing.to_crs("EPSG:3035"),
    #     basins[[id_name, "geometry"]].to_crs("EPSG:3035"),
    #     how="left",
    # )

    # # Combine back
    # gdf_joined = pd.concat([within[within[id_name].notna()], nearest])
    gdf_joined = gdf_timeframe.copy()
    gdf_joined["site"] = (
        gdf_joined["dayofyear"].astype(int).astype(str)
        + "_"
        + gdf_joined["year"].astype(int).astype(str)
        + "_"
        + gdf_joined["geometry"].astype(str)
    )

    # Create pivot table for substances
    gdf_joined_pivot = gdf_joined[~gdf_joined.less_than].pivot_table(
        index="site", columns="substance", values="conc", aggfunc="median"
    )

    # Get geometry groupby for joining
    geom = gdf_joined.groupby("site")["basin"].first()

    # Create wide table with geometry info
    gdf_joined_pivot_wide = gdf_joined_pivot.join(geom)

    # Define substances for ratio analysis
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

    df_ratios = boxplot_ratios(df=gdf_joined_pivot_wide, substances=subst_for_bp_rat)
    df_greater = test_significant_difference(
        df=gdf_joined_pivot_wide, substances=subst_for_bp_rat
    )
    df_greater = df_greater[["comparison", "n", "stat_wilc", "p_value_wilc"]]
    df_greater = df_greater.rename(
        columns={
            "comparison": "Comparison",
            "n": "Observations",
            "stat_wilc": "Test statistic",
            "p_value_wilc": "p value",
        }
    )

    df_greater = df_greater.set_index("Comparison").dropna()
    # Create ratio counts table
    count_ratios = (
        df_ratios.groupby(["basin", "ratio"])["value"]
        .count()
        .unstack()
        .fillna(0)
        .astype(int)
    )

    # Define ratio order
    ratio_order = []
    for sub1, sub2 in subst_for_bp_rat:
        ratio_name = f"{sub1}/{sub2}"
        if ratio_name in count_ratios.columns:
            ratio_order.append(ratio_name)

    count_ratios = count_ratios[ratio_order]
    count_ratios.columns = [col.replace("/", " / ") for col in count_ratios.columns]

    count_ratios.loc["Total"] = count_ratios.sum(axis=0)
    count_ratios["Total"] = count_ratios.sum(axis=1)
    count_ratios.index.name = "Basin"

    logger.info("Sheets Ratio counts, and Wilcoxon test done")

    # Create country statistics
    # logger.info("Start computing country statistics")
    """
    europe = countries[countries["CONTINENT"].isin(["Europe", "Asia"])]
    europe = europe.to_crs(gdf_timeframe.crs)

    europe_no_russia = europe[
        europe["ADMIN"] != "Russia"
    ]  # Done due to wrong assignment. There are no sammpling stations in Russia in this dataset

    within = gpd.sjoin(
        gdf_timeframe.to_crs("EPSG:3035"),
        europe_no_russia[["ADMIN", "geometry"]].to_crs("EPSG:3035"),
        how="left",
        predicate="within",
    )

    missing = within[within["ADMIN"].isna()].drop(columns="ADMIN")

    if "index_right" in missing.columns:
        missing = missing.drop(columns="index_right")

    nearest = gpd.sjoin_nearest(
        missing.to_crs("EPSG:3035"),
        europe_no_russia[["ADMIN", "geometry"]].to_crs("EPSG:3035"),
        how="left",
    )

    # combine back
    gdf_joined = pd.concat([within[within["ADMIN"].notna()], nearest])
    """

    # counts = gdf_joined.groupby(["country", "less_than"]).size().unstack(fill_value=0)
    # counts = counts.rename(columns={True: "true", False: "false"})
    # counts = counts.fillna(0)

    # counts_display_country = counts.copy()
    # counts_display_country["total"] = (
    #     counts_display_country["true"] + counts_display_country["false"]
    # )
    # counts_display_country["ratio_false"] = (
    #     counts_display_country["false"] / counts_display_country["total"]
    # )

    # counts_display_country_new = counts_display_country.reset_index().copy()
    # counts_display_country_new = counts_display_country_new.rename(
    #     columns={
    #         "country": "Country",
    #         "false": "Detected",
    #         "true": "Non-detected",
    #         "total": "Total",
    #         "ratio_false": "Detection frequency",
    #     }
    # ).set_index("Country")
    # logger.info("Country stats done")

    # logger.info("Basin statistics")
    # counts_basin = gdf_joined.groupby(["HYBAS_ID", "less_than"]).size().unstack(fill_value=0)
    # counts_basin = counts_basin.rename(columns={True: "true", False: "false"})
    # counts_basin = counts_basin.fillna(0)

    # counts_display_basin = counts_basin.copy()
    # counts_display_basin["total"] = (
    #     counts_display_basin["true"] + counts_display_basin["false"]
    # )
    # counts_display_basin["ratio_false"] = (
    #     counts_display_basin["false"] / counts_display_basin["total"]
    # )

    # counts_display_basin_new = counts_display_basin.reset_index().copy()
    # counts_display_basin_new = counts_display_basin_new.rename(
    #     columns={
    #         "HYBAS_ID": "Basin",
    #         "false": "Detected",
    #         "true": "Non-detected",
    #         "total": "Total",
    #         "ratio_false": "Detection frequency",
    #     }
    # ).set_index("Basin")
    logger.info("Start computing Substance stats")
    print("Index unique:", gdf_timeframe.index.is_unique) 
    print("Duplicate index rows:", gdf_timeframe.index.duplicated().sum())

    # Helper functions for statistics
    def median_detected(x):
        """Return the median concentration for detected observations."""
        mask = gdf_timeframe.loc[x.index, "less_than"].eq(False)
        return x[mask].median()

    def mean_detected(x):
        """Return the mean concentration for detected observations."""
        mask = gdf_timeframe.loc[x.index, "less_than"].eq(False)
        return x[mask].mean()

    def q85_detected(x):
        """Return the 85th percentile concentration for detected observations."""
        mask = gdf_timeframe.loc[x.index, "less_than"].eq(False)
        return x[mask].quantile(0.85)

    # Create substance statistics
    stats = gdf_timeframe.groupby(["substance"]).agg(
        n_sites=("geometry", "nunique"),
        nmeas=("conc", "count"),
        detfreq=("less_than", lambda x: x.eq(False).mean()),
        median=("conc", median_detected),
        mean=("conc", mean_detected),
        q85=("conc", q85_detected),
        max=("conc", "max"),
    )
    sort = [
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
        # "Linear PFBS",
        # "Linear PFHpS",
        # "Linear PFHxS",
        "Linear PFOS",
        "6:2 FTCA",
        "4:2 FTS",
        "6:2 FTS",
        "8:2 FTS",
        "HFPO-DA",
        # "ADONA",
        "DONA",
        "FOSA",
        "N-Et-FOSA",
        "N-MeFOSAA",
        "EtFOSAA",
    ]
    substance_stats = stats.copy()
    substance_stats_new = substance_stats.reset_index().copy()
    substance_stats_new = substance_stats_new.rename(
        columns={
            "substance": "Substance",
            "n_sites": "Sites",
            "nmeas": "Samples",
            "detfreq": "Detection frequency",
            "median": "Median",
            "mean": "Mean",
            "q85": "85%-percentile",
            "max": "Maximum",
        }
    ).set_index("Substance")
    substance_stats_new = substance_stats_new.rename(
        index={
            "L_PFBS": "Linear PFBS",
            "L_PFHpS": "Linear PFHpS",
            "L_PFHxS": "Linear PFHxS",
            "L_PFOA": "Linear PFOA",
            "L_PFOS": "Linear PFOS",
        }
    )
    substance_stats_new = substance_stats_new.reindex(sort)
    logger.info("Computing Substance stats done")

    logger.info("Computing PNEC and LOD sheets")
    # Create LOD and PNEC tables
    advisory_val = {
        "PFBA": 4400,
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
        # "ADONA": 146.666667,
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

    res_lod = []
    for subst in sorted(gdf_timeframe.substance.unique()):
        gdf_tf_less_sub = gdf_timeframe_less[gdf_timeframe_less.substance == subst]
        raw_num = np.nan
        if subst in advisory_val.keys():
            raw_num = advisory_val[subst]
        res_lod.append(
            {
                "Substance": subst,
                "Minimum (ng/L)": gdf_tf_less_sub.conc.min(),
                "Minimum n (-)": len(
                    gdf_tf_less_sub[gdf_tf_less_sub.conc == gdf_tf_less_sub.conc.min()]
                ),
                "Maximum (ng/L)": gdf_tf_less_sub.conc.max(),
                "Maximum n (-)": len(
                    gdf_tf_less_sub[gdf_tf_less_sub.conc == gdf_tf_less_sub.conc.max()]
                ),
                "PNEC (ng/L)": raw_num,
            }
        )

    df_res_lod = pd.DataFrame(res_lod)
    df_res_lod = df_res_lod.set_index("Substance")
    df_res_lod = df_res_lod.rename(
        index={
            "L_PFBS": "Linear PFBS",
            "L_PFHpS": "Linear PFHpS",
            "L_PFHxS": "Linear PFHxS",
            "L_PFOA": "Linear PFOA",
            "L_PFOS": "Linear PFOS",
        }
    )

    df_res_pnec = df_res_lod["PNEC (ng/L)"].dropna().copy()
    df_res_lod = df_res_lod.drop(columns="PNEC (ng/L)")

    logger.info("PNEC and LOD computation done")
    logger.info("'Sum of 20 PFAS' dependency of number of included substances computation")
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

    res = []
    for i in range(1,21):
        dat = pfas_wide[pfas_sum].sum(axis=1, min_count=i)
        res_dict = {
            "Included substances" : i,
            "Number of matching samples" : len(dat.dropna()),
            "Median (ng/L)" : dat.median(),
            "Mean (ng/L)" : dat.mean(),
            "Number of 100ng/L exceedance" : len(dat[dat>100])
        }
        res.append(res_dict)
    df_pfas_sum_dependency = pd.DataFrame(res).set_index("Included substances").fillna("-")
    logger.info("Computation done.")
    if additional_information == {}:
        exit()
    logger.info("Prepare sheets from 'Additional Information'")
    logger.info("Prepare factor analysis results table")
    factor_analysis_res = pd.DataFrame(additional_information["Factor Analysis"]).set_index("Group ID")
    factor_analysis_res["Substances"] = factor_analysis_res["Substances"].apply(", ".join).apply(lambda x: x.replace("L_", "Linear "))
    factor_analysis_res["Bartlett's p"] = factor_analysis_res["Bartlett's p"].apply(
        lambda x: "<0.05" if x < 0.05 else x
    )
    round_cols = ["Explained Variance 1", "KMO criterion", "Explained Variance 2", "Explained Variance 3", "Sum Explained Variance", "Max absolute factor correlation"]
    factor_analysis_res[round_cols] = factor_analysis_res[round_cols].round(3)
    

    # Prepare data for Excel export
    dict_to_excel = [
        {
            "file": count_ratios,
            "sheet": "Ratio counts",
            "cap": "Number of observations per PFAS ratio and basin. Total counts are given for the respective row and column.",
        },
        {
            "file": df_greater,
            "sheet": "Wilcoxon test",
            "cap": "Results from Wilcoxon statistical test of PFAS concentrations",
        },
        # {
        #     "file": counts_display_country_new,
        #     "sheet": "Country statistics",
        #     "cap": "Counts of samples resulting in detected or non-detected PFAS per country and the respective detection frequency. Time period: 2019-2025.",
        # },
        # {
        #     "file": counts_display_basin_new,
        #     "sheet": "Basin statistics",
        #     "cap" : "Counts of samples resulting in detected or non-detected PFAS per basin and the respective detection frequency. Time period: 2019-2025."
        # },
        {
            "file": substance_stats_new,
            "sheet": "Substance statistics",
            "cap": "General statistics of the used PFAS dataset collection. Concentrations in ng/L. Time period: 2019-2025. For abbreviations refer to Table A.1.",
        },
        {
            "file": df_res_pnec,
            "sheet": "PNEC",
            "cap": "Predicted no-effect concentrations (PNEC) of respective substances.",
        },
        {
            "file": df_res_lod,
            "sheet": "LOD",
            "cap": "Inferred minimum and maximum limits of detectability / quantification for 2019 - 2025.",
        },
        {
            "file": df_pfas_sum_dependency,
            "sheet": "PFAS20 dependency",
            "cap": "Dependency of the 'Sum of 20 PFAS' parameter on the minimum number of included substances.",
        },
        {
            "file": factor_analysis_res,
            "sheet": "PCAFA",
            "cap": "Group definitions, test statistics and results from the PCA/FA.",
        },
    ]
    # table_numbers = ["B.1", "B.2", "B.3", "B.4", "B.5", "B.6"] 

    table_numbers = [f"B.{i}" for i in range(1, len(dict_to_excel) + 1)]

    table_list = [ f"{item['sheet']} - Table {number}: {item['cap']}" for item, number in zip(dict_to_excel, table_numbers) ]

    metadata = {
        "Title" : ["Supplementary Information B","Fingerprinting PFAS pollution in European surface waters"],
        "Authors" : ["Robin Schröder\u1d43", "Tobias Licha\u1d47","Martina Flörke\u1d43"],
        "Affiliations": ["\u1d43 Engineering Hydrology and Water Resources Management, Faculty of Civil and Environmental Engineering, Ruhr University Bochum, Universitätsstraße 150, 44801 Bochum, Germany","\u1d47 Institute of Geology, Mineralogy & Geophysics, Dept. Hydrogeology and Environmental Geology, Ruhr University Bochum, Universitätsstraße 150, 44801 Bochum, Germany"],
        "Contents": table_list,
        "Contact" : "robin.schroeder@hydrology.ruhr-uni-bochum.de" 
    }
    
    metadata_rows = []

    for key, value in metadata.items():
        if isinstance(value, list):
            for i, entry in enumerate(value):
                metadata_rows.append({
                    "Key": key if i == 0 else "",
                    "Value": entry
                })
        else:
            metadata_rows.append({
                "Key": key,
                "Value": value
            })

    metadata_df = pd.DataFrame(metadata_rows)

    sheet_counter = 1
    with pd.ExcelWriter(
        save_path / "appendix_SupplementaryInformation-B.xlsx"
    ) as writer:
        logger.info(
            f"Start writing Excel file to {save_path / "appendix_SupplementaryInformation-B.xlsx"}"
        )
        # Write metadata
        logger.info(f"Writing sheet: README")
        metadata_df.to_excel( 
            writer, 
            sheet_name="README", 
            index=False, 
            header=False 
            )

        for entry in dict_to_excel:
            logger.info(f"Writing sheet: {entry["sheet"]}")
            df_out: pd.DataFrame = entry["file"]
            sheet_name = entry["sheet"]
            df_out.to_excel(writer, sheet_name=sheet_name, startrow=1)
            ws = writer.sheets[sheet_name]
            ws["A1"] = f"Table B.{sheet_counter}: {entry["cap"]}"
            sheet_counter += 1
        # Columns dims
        readme = writer.book["README"] 
        readme.column_dimensions["A"].width = 20 
        readme.column_dimensions["B"].width = 100

        writer.book["Ratio counts"].column_dimensions["A"].width = 25
        for sheet in ["B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N"]:
            writer.book["Ratio counts"].column_dimensions[sheet].width = 17

        writer.book["Wilcoxon test"].column_dimensions["A"].width = 18
        writer.book["Wilcoxon test"].column_dimensions["B"].width = 15
        writer.book["Wilcoxon test"].column_dimensions["C"].width = 15
        writer.book["Wilcoxon test"].column_dimensions["D"].width = 15

        # writer.book["Country statistics"].column_dimensions["A"].width = 25
        # writer.book["Country statistics"].column_dimensions["B"].width = 20
        # writer.book["Country statistics"].column_dimensions["C"].width = 20
        # writer.book["Country statistics"].column_dimensions["D"].width = 20
        # writer.book["Country statistics"].column_dimensions["E"].width = 20

        # writer.book["Basin statistics"].column_dimensions["A"].width = 25
        # writer.book["Basin statistics"].column_dimensions["B"].width = 20
        # writer.book["Basin statistics"].column_dimensions["C"].width = 20
        # writer.book["Basin statistics"].column_dimensions["D"].width = 20
        # writer.book["Basin statistics"].column_dimensions["E"].width = 20

        writer.book["Substance statistics"].column_dimensions["A"].width = 15
        writer.book["Substance statistics"].column_dimensions["B"].width = 20
        writer.book["Substance statistics"].column_dimensions["C"].width = 20
        writer.book["Substance statistics"].column_dimensions["D"].width = 20
        writer.book["Substance statistics"].column_dimensions["E"].width = 20
        writer.book["Substance statistics"].column_dimensions["F"].width = 20
        writer.book["Substance statistics"].column_dimensions["G"].width = 20
        writer.book["Substance statistics"].column_dimensions["H"].width = 20

        writer.book["PNEC"].column_dimensions["A"].width = 15
        writer.book["PNEC"].column_dimensions["B"].width = 15

        writer.book["LOD"].column_dimensions["A"].width = 15
        writer.book["LOD"].column_dimensions["B"].width = 15
        writer.book["LOD"].column_dimensions["C"].width = 15
        writer.book["LOD"].column_dimensions["D"].width = 15

        writer.book["PFAS20 dependency"].column_dimensions["A"].width = 20
        writer.book["PFAS20 dependency"].column_dimensions["B"].width = 30
        writer.book["PFAS20 dependency"].column_dimensions["C"].width = 15
        writer.book["PFAS20 dependency"].column_dimensions["D"].width = 15
        writer.book["PFAS20 dependency"].column_dimensions["E"].width = 30


        writer.book["PCAFA"].column_dimensions["A"].width = 10
        writer.book["PCAFA"].column_dimensions["B"].width = 75
        for sheet in ["C", "D", "E", "F"]:
            writer.book["PCAFA"].column_dimensions[sheet].width = 14
        for sheet in [ "G", "H", "I", "J"]:
            writer.book["PCAFA"].column_dimensions[sheet].width = 22
        writer.book["PCAFA"].column_dimensions["K"].width = 30
    logger.info("Finished Excel file")


if __name__ == "__main__":
    gdf_tf = gpd.read_file("data/input/pfas_data.gpkg")
    gdf_tf["dayofyear"] = pd.to_datetime(gdf_tf["date"], format="ISO8601").dt.dayofyear
    basins = gpd.read_file("data/input/hybas_eu_lev04_v1c.shp")
    # countries = gpd.read_file("data/input/ne_10m_admin_0_countries.shp")
    basins = basins.to_crs(gdf_tf.crs)
    # countries = countries.to_crs(gdf_tf.crs)

    create_supplementary_table(gdf_tf, basins, {}, Path("results"))
