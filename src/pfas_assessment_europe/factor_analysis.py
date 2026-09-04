"""Perform PCA, factor rotation, clustering, and visualisation of PFAS data.

This module provides functions for analysing PFAS concentration data using
principal component analysis (PCA), factor rotations, hierarchical clustering,
and geographic grouping. It also generates biplots, and
consensus matrices for visualising relationships among substances and sampling
regions.
"""

import logging
import warnings
from itertools import (
    cycle,
)
from pathlib import (
    Path,
)

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from adjustText import (
    adjust_text,
)
from factor_analyzer.factor_analyzer import (
    calculate_bartlett_sphericity,
    calculate_kmo,
)
from factor_analyzer.rotator import (
    Rotator,
)
from matplotlib.lines import (
    Line2D,
)
from scipy.cluster.hierarchy import (
    fcluster,
    linkage,
)
from scipy.spatial import (
    ConvexHull,
)
from shapely import (
    wkt,
)
from sklearn.decomposition import (
    PCA,
)
from sklearn.pipeline import (
    Pipeline,
)
from sklearn.preprocessing import (
    PowerTransformer,
    StandardScaler,
)

plt.rcParams.update({"font.size": 12})

logger = logging.getLogger(__name__)


def plot_convex_hull(ax, df, xcol, ycol, color, alpha=0.15):
    """Plot the convex hull surrounding a set of two-dimensional points.

    A convex hull is only calculated when the input contains at least three
    points. If fewer than three points are available, the function returns
    without modifying the axes.

    Args:
        ax: Matplotlib axes on which the convex hull should be plotted.
        df: DataFrame containing the point coordinates.
        xcol: Name of the column containing x-coordinate values.
        ycol: Name of the column containing y-coordinate values.
        color: Fill color for the convex hull polygon.
        alpha: Transparency of the polygon fill. Defaults to ``0.15``.

    Returns:
        None. The convex hull is drawn directly on ``ax``.
    """
    if len(df) < 3:
        return  # cannot compute hull

    points = df[[xcol, ycol]].values
    hull = ConvexHull(points)
    hull_points = points[hull.vertices]

    ax.fill(hull_points[:, 0], hull_points[:, 1], color=color, alpha=alpha, zorder=0)


def full_pca_analysis(
    gdf_dropped: gpd.GeoDataFrame,
    groups: dict,
    save_path: Path,
    basins: gpd.GeoDataFrame,
    id_name: str,
):
    """Run grouped PCA analyses and save rotated PCA biplots.

    For each substance group, this function selects sampling locations where
    all substances in the group are present, aggregates repeated observations
    by sampling event using the median concentration, and removes incomplete
    rows. The data are transformed using a Yeo-Johnson power transformation
    and standardized before PCA.

    The function calculates Kaiser-Meyer-Olkin and Bartlett sphericity
    statistics, applies varimax rotation to the PCA loadings, calculates
    rotated sample scores, and estimates factor correlations using quartimin
    rotation. Substances are additionally grouped using hierarchical
    clustering with Ward's linkage method.

    Sampling locations are spatially joined to the supplied basin geometries.
    For each substance group, the function generates a biplot showing:

    - Rotated sample scores.
    - Rotated substance loadings.
    - Basin-specific colors and markers.
    - Convex hulls around observations from each basin.
    - Labels adjusted to reduce overlap.

    The primary group named ``"A"`` is saved as
    ``pcafa_biplot_A.pdf``. Biplots for other groups are saved using the
    ``appendix_pcafa_biplot_<group>.pdf`` naming scheme.

    Args:
        gdf_dropped: GeoDataFrame containing PFAS observations. It must contain
            ``geometry``, ``substance``, ``conc``, ``dayofyear``, and ``year``
            columns. Rows with missing substances are excluded from the
            availability calculations.
        groups: Mapping from group names to iterables of substance names.
            Every substance listed in a group must be present at a location
            for that location to be included in the corresponding analysis.
        save_path: Directory in which the PCA biplot PDF files are saved.
        basins: GeoDataFrame containing basin geometries used to assign
            sampling locations to regions.
        id_name: Name of the column in ``basins`` containing basin or region
            identifiers.

    Returns:
        A list of dictionaries. Each dictionary corresponds to one substance
        group and maps substance names to hierarchical-clustering labels.
    """
    comps = gdf_dropped.dropna(subset="substance")
    counts = comps.groupby(["geometry", "substance"]).size().unstack(fill_value=0)
    counts[counts > 0] = 1
    counts.loc["Total"] = counts.sum()

    cluster_results = []

    plt.rcParams.update({"font.size": 12})

    for group_n, substances in groups.items():
        pipeline = Pipeline(
            [
                ("power", PowerTransformer(method="yeo-johnson", standardize=False)),
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=3)),
            ]
        )

        logger.info(f"{group_n} : {substances}")
        group_locations = counts[counts[substances].eq(1).all(axis=1)].index.to_list()
        gdf_analysis = gdf_dropped[gdf_dropped["geometry"].isin(group_locations)].copy()
        df = gdf_analysis[gdf_analysis["substance"].isin(substances)].copy()
        logger.info(f"Number of locations: {len(group_locations)}")

        # New index
        df["site"] = (
            df["dayofyear"].astype(int).astype(str)
            + "_"
            + df["year"].astype(int).astype(str)
            + "_"
            + df["geometry"].astype(str)
        )

        # In case there are multiple values at the same location and date, calculate median
        df = df.pivot_table(
            index="site", columns="substance", values="conc", aggfunc="median"
        )
        df = df.dropna(axis=0)
        matching_locations = df.index.str.rsplit("_", n=1).str[-1].nunique()
        logger.info(f"Number of matching locations: {matching_locations}")
        logger.info(f"Rows of data for PCA: {len(df)}")

        compounds = df.columns

        X_scaled = pipeline[:-1].fit_transform(df)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            _, kmo_model = calculate_kmo(X_scaled)
        logger.info(f"KMO criterion: {kmo_model}")  # Should be at least 0.5, good > 0.7
        chi2, p_value = calculate_bartlett_sphericity(X_scaled)
        logger.info(
            f"Bartlett's test: χ² = {chi2:.2f}, p = {p_value:.3e}"
        )  # Significant p-value (p < 0.05) -> appropriate for PCA / FA

        X_pca = pipeline.fit_transform(df)
        pca = pipeline.named_steps["pca"]
        rotator = Rotator(method="varimax")

        pattern = rotator.fit_transform(pca.components_.T)

        scores_rot = X_pca @ rotator.rotation_  # (locations × components)

        # Flip so that loadings with largest magnitude are positive. This was done to improve readability of bi-plots
        flip = np.abs(pattern.min(axis=0)) > np.abs(pattern.max(axis=0))
        pattern[:, flip] *= -1
        scores_rot[:, flip] *= -1

        x, y, z = pattern.T

        points = df.index.str.rsplit("_", n=1).str[-1].map(wkt.loads)
        points_gdf = gpd.GeoDataFrame(
            {
                "factor1": scores_rot[:, 0],
                "factor2": scores_rot[:, 1],
                "factor3": scores_rot[:, 2],
            },
            geometry=points.values,
            crs=gdf_dropped.crs,
        )
        points_country = gpd.sjoin(points_gdf, basins, how="left", predicate="within")
        countries = sorted(points_country[id_name].dropna().unique())
        if len(countries) > 10:
            palette = sns.color_palette("tab20", 20)
        else:
            palette = sns.color_palette("tab10", 10)

        country_colors = dict(zip(countries, palette))
        markers = [
            "s",
            "o",
            "D",
            "^",
            "v",
            "<",
            ">",
            "P",
            "X",
            "H",
            "*",
            "d",
            "p",
            "h",
            "8",
            "^",
            "v",
            "1",
        ]
        marker_cycle = cycle(markers)
        country_markers = {country: next(marker_cycle) for country in countries}

        # Initialise max_corr
        max_corr = 0.0
        rotator_for_phi = Rotator(method="quartimin")  # oblique rotation
        rotator_for_phi.fit_transform(pca.components_.T)
        phi = rotator_for_phi.phi_
        max_corr = np.max(np.abs(phi[np.triu_indices_from(phi, k=1)]))

        logger.info(f"Maximum absolute factor correlation: {max_corr:.3f}")

        # Clustering
        Z = linkage(pattern, method="ward")
        clusters = fcluster(Z, t=0.6, criterion="distance")
        compound_clusters = pd.DataFrame({"compound": compounds, "cluster": clusters})
        cluster_results.append(
            compound_clusters.set_index("compound")["cluster"].to_dict()
        )

        fig_scale = 0.63
        fig, axes = plt.subplots(
            1, 2, figsize=(18 * fig_scale, 8 * fig_scale), constrained_layout=True
        )

        text_offset = 1  # 1.1
        scaling_secondary_axis = 5
        color_secondary_axis = "blue"

        def forward(x):
            return x / scaling_secondary_axis

        def inverse(x):
            return x * scaling_secondary_axis

        # Left subplot: X–Z
        ax1 = axes[0]

        secax_x1 = ax1.secondary_xaxis("top", functions=(forward, inverse))
        secax_y1 = ax1.secondary_yaxis("right", functions=(forward, inverse))
        texts = []
        for i, compound in enumerate(compounds):
            ax1.arrow(
                0,
                0,
                x[i] * scaling_secondary_axis,
                z[i] * scaling_secondary_axis,
                color=color_secondary_axis,
                head_width=0.7 * max(abs(z)) / scaling_secondary_axis,
                length_includes_head=True,
                overhang=1.0,
                alpha=0.7,
                label="_nolegend_",
                zorder=2,
            )
            ha = "left" if x[i] > 0 else "right"
            va = "bottom" if z[i] > 0 else "top"

            txt = ax1.text(
                x[i] * scaling_secondary_axis * text_offset,
                z[i] * scaling_secondary_axis * text_offset,
                compound,
                ha=ha,
                va=va,
                fontsize=14,
                zorder=3,
                color=color_secondary_axis,
            )
            texts.append(txt)
        for country, group in points_country.groupby(id_name):
            ax1.scatter(
                group["factor1"],
                group["factor3"],
                color=country_colors[country],
                marker=country_markers[country],
                label=country,
                alpha=0.4,
                zorder=1,
            )
            plot_convex_hull(ax1, group, "factor1", "factor3", country_colors[country])
            txt = ax1.text(
                group["factor1"].mean(),
                group["factor3"].mean(),
                country,
                fontsize=9,
                ha="center",
                va="center",
                zorder=4,
            )
            texts.append(txt)
        adjust_text(
            texts,
            ax=ax1,
            expand_points=(1.2, 1.2),
            expand_text=(1.2, 1.2),
            force_text=0.5,
        )

        ax1.set_xlabel("Rotated component 1 (samples)")
        ax1.set_ylabel("Rotated component 3 (samples)")
        ax1.axhline(0, color="gray", lw=1)
        ax1.axvline(0, color="gray", lw=1)

        # Hide ax1 top/right
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)

        secax_x1.set_xlabel("Rotated component 1 (substances)")
        secax_y1.set_ylabel("Rotated component 3 (substances)")
        # X (top)
        secax_x1.xaxis.label.set_color(color_secondary_axis)
        secax_x1.tick_params(axis="x", colors=color_secondary_axis)
        secax_x1.spines["top"].set_color(color_secondary_axis)

        # Y (right)
        secax_y1.yaxis.label.set_color(color_secondary_axis)
        secax_y1.tick_params(axis="y", colors=color_secondary_axis)
        secax_y1.spines["right"].set_color(color_secondary_axis)

        # Right subplot: Y–Z
        ax2 = axes[1]
        texts = []
        secax_x2 = ax2.secondary_xaxis("top", functions=(forward, inverse))
        secax_y2 = ax2.secondary_yaxis("right", functions=(forward, inverse))
        for i, compound in enumerate(compounds):

            ax2.arrow(
                0,
                0,
                y[i] * scaling_secondary_axis,
                z[i] * scaling_secondary_axis,
                color=color_secondary_axis,
                head_width=0.7 * max(abs(z)) / scaling_secondary_axis,
                length_includes_head=True,
                overhang=1.0,
                alpha=0.7,
                label="_nolegend_",
                zorder=2,
            )

            ha = "left" if y[i] > 0 else "right"
            va = "bottom" if z[i] > 0 else "top"

            txt = ax2.text(
                y[i] * scaling_secondary_axis * text_offset,
                z[i] * scaling_secondary_axis * text_offset,
                compound,
                ha=ha,
                va=va,
                fontsize=14,
                zorder=3,
                color=color_secondary_axis,
            )
            texts.append(txt)

        for country, group in points_country.groupby(id_name):
            ax2.scatter(
                group["factor2"],
                group["factor3"],
                color=country_colors[country],
                marker=country_markers[country],
                label=country,
                alpha=0.4,
                zorder=1,
            )
            plot_convex_hull(ax2, group, "factor2", "factor3", country_colors[country])
            txt = ax2.text(
                group["factor2"].mean(),
                group["factor3"].mean(),
                country,
                fontsize=9,
                ha="center",
                va="center",
                zorder=4,
            )
            texts.append(txt)
        adjust_text(
            texts,
            ax=ax2,
            expand_points=(1.2, 1.2),
            expand_text=(1.2, 1.2),
            force_text=0.5,
        )
        ax2.set_xlabel("Rotated component 2 (samples)")
        ax2.set_ylabel("Rotated component 3 (samples)")

        ax2.axhline(0, color="gray", lw=1)
        ax2.axvline(0, color="gray", lw=1)

        # Hide ax1 top/right
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)

        secax_x2.set_xlabel("Rotated component 2 (substances)")
        secax_y2.set_ylabel("Rotated component 3 (substances)")
        # X (top)
        secax_x2.xaxis.label.set_color(color_secondary_axis)
        secax_x2.tick_params(axis="x", colors=color_secondary_axis)
        secax_x2.spines["top"].set_color(color_secondary_axis)

        # Y (right)
        secax_y2.yaxis.label.set_color(color_secondary_axis)
        secax_y2.tick_params(axis="y", colors=color_secondary_axis)
        secax_y2.spines["right"].set_color(color_secondary_axis)
        handles = [
            Line2D(
                [0],
                [0],
                marker=country_markers[c],
                color="w",
                markerfacecolor=country_colors[c],
                markeredgecolor=country_colors[c],
                markersize=6,
                linestyle="None",
                label=c,
            )
            for c in countries
        ]

        n_sub = len(countries)

        # choose n_cols
        if n_sub <= 5:
            n_legend_col = n_sub
        elif n_sub in [6, 8, 10]:
            n_legend_col = n_sub / 2
        else:
            n_legend_col = 5

        fig.legend(
            handles=handles,
            title="Regions / Basins",
            bbox_to_anchor=(0.5, -0.01),
            ncols=n_legend_col,
            frameon=False,
            loc="upper center",
        )
        if group_n == "A":
            plt.savefig(save_path / f"pcafa_biplot_{group_n}.pdf", bbox_inches="tight")
        else:
            plt.savefig(
                save_path / f"appendix_pcafa_biplot_{group_n}.pdf", bbox_inches="tight"
            )
        plt.close(fig)

    return cluster_results


def plot_consensus_matrix(cluster_results, save_path: Path) -> None:
    """Plot and save a consensus matrix for repeated clustering results.

    The consensus matrix shows the frequency with which pairs of substances
    are assigned to the same cluster across multiple clustering runs. Pairs
    that never occur together in a clustering run are displayed in grey.

    The resulting heatmap is saved as
    ``appendix_consensus_matrix.pdf`` in ``save_path``.

    Args:
        cluster_results: Iterable of dictionaries containing clustering
            assignments. Each dictionary must map feature or substance names
            to cluster labels.
        save_path: Directory in which the consensus-matrix PDF is saved.

    Returns:
        None. The consensus heatmap is saved to disk.
    """
    features = sorted({f for r in cluster_results for f in r})
    n = len(features)
    plt.rcParams.update({"font.size": 12})
    co_mat = np.zeros((n, n))
    co_run_count = np.zeros((n, n))  # number of runs where both features exist

    #  map feature name -> index
    idx = {f: i for i, f in enumerate(features)}

    # compute co-occurrence and count of runs where both features exist
    for r in cluster_results:
        present = list(r.keys())
        for i, fi in enumerate(present):
            for j, fj in enumerate(present):
                ii, jj = idx[fi], idx[fj]
                co_run_count[ii, jj] += 1
                if r[fi] == r[fj]:
                    co_mat[ii, jj] += 1

    consensus = co_mat / np.maximum(co_run_count, 1)

    # create mask for pairs that never co-occurred
    mask_never = co_run_count == 0

    # set these to NaN so they can be grey
    consensus_masked = consensus.copy()
    consensus_masked[mask_never] = np.nan

    # plot
    cmap = sns.color_palette("crest", as_cmap=True)
    cmap.set_bad(color="lightgrey")  # grey for never co-occurred pairs

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        consensus_masked,
        cmap=cmap,
        xticklabels=features,
        yticklabels=features,
        square=True,
        cbar_kws={"label": "Co-clustering frequency"},
        linewidth=0.1,
        linecolor="white",
        annot=True,
        fmt=".1f",
    )

    plt.tight_layout()
    plt.savefig(save_path / "appendix_consensus_matrix.pdf", bbox_inches="tight")

    plt.close()


def factor_analysis(
    gdf_dropped: gpd.GeoDataFrame, basins: gpd.GeoDataFrame, save_path: Path
):
    """Run grouped factor analyses and generate PCA visualizations.

    The function prepares HydroBASINS identifiers, defines multiple groups of
    PFAS substances, and runs a principal component and factor analysis for
    each group using :func:`full_pca_analysis`. It then creates a consensus
    matrix summarising the clustering results across the analyzed groups.

    The generated output includes PCA/factor-analysis biplots for each group
    and a consensus-matrix heatmap.

    Args:
        gdf_dropped: GeoDataFrame containing PFAS observations. It must contain
            the columns required by :func:`full_pca_analysis`, including
            ``geometry``, ``substance``, ``conc``, ``dayofyear``, and ``year``.
        basins: GeoDataFrame containing HydroBASINS geometries and a
            ``HYBAS_ID`` column used to assign observations to regions.
        save_path: Directory in which the generated plots are saved.

    Returns:
        None. Analysis results are written to log output, and generated figures
        are saved to ``save_path``.
    """
    logger.info("--- Starting Factor Analysis ---")
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

    groups_dict = {
        "A": ["PFBA", "PFBS", "PFHpA", "PFHxS", "PFHxA", "PFOA", "PFOS", "PFPeA"],
        "A01": [
            "PFBA",
            "PFBS",
            "PFHpA",
            "PFHxS",
            "PFHxA",
            "PFOA",
            "PFOS",
            "PFPeA",
            "PFNA",
            "PFDA",
        ],
        "A02": [
            "PFBA",
            "PFBS",
            "PFHpA",
            "PFHxS",
            "PFHxA",
            "PFOA",
            "PFOS",
            "PFPeA",
            "L_PFOS",
            "L_PFOA",
        ],
        "A03": [
            "PFBA",
            "PFBS",
            "PFHpA",
            "PFHxS",
            "PFHxA",
            "PFOA",
            "PFOS",
            "PFPeA",
            "6:2 FTS",
        ],
        "A04": [
            "PFBA",
            "PFBS",
            "PFHpA",
            "PFHxS",
            "PFHxA",
            "PFOA",
            "PFOS",
            "PFPeA",
            "PFPeS",
        ],
        "A05": [
            "PFBA",
            "PFBS",
            "PFHpA",
            "PFHxS",
            "PFHxA",
            "PFOA",
            "PFOS",
            "PFPeA",
            "PFNA",
            "PFPeS",
        ],
        "A06": [
            "PFBA",
            "PFBS",
            "PFHpA",
            "PFHxS",
            "PFHxA",
            "PFOA",
            "PFOS",
            "PFPeA",
            "HFPO-DA",
        ],
        "A07": [
            "PFBA",
            "PFBS",
            "PFHpA",
            "PFHxS",
            "PFHxA",
            "PFOA",
            "PFOS",
            "PFPeA",
            "FOSA",
        ],
        "B": ["PFOA", "PFOS", "PFHxA", "PFBS", "PFHpA", "PFHxS", "PFHpS"],
        "C": [
            "L_PFBS",
            "L_PFHxS",
            "L_PFOS",
            "PFBA",
            "PFDA",
            "PFHpA",
            "PFHxA",
            "PFNA",
            "PFOA",
            "PFPeA",
            "PFPeS",
        ],
        "D": [
            "EtFOSAA",
            "L_PFBS",
            "L_PFHxS",
            "L_PFOS",
            "PFBA",
            "PFDA",
            "PFHpA",
            "PFHxA",
            "PFNA",
            "PFOA",
            "PFPeA",
            "PFPeS",
        ],
        "E": [
            "EtFOSAA",
            "FOSA",
            "N-MeFOSAA",
            "L_PFBS",
            "L_PFHxS",
            "L_PFOS",
            "PFBA",
            "PFDA",
            "PFHpA",
            "PFHxA",
            "PFNA",
            "PFOA",
            "PFPeA",
            "PFPeS",
        ],
        "F": ["FOSA", "L_PFOS", "PFBA", "PFOA", "PFHpA", "PFHxA", "PFPeA", "PFPeS"],
        "G": [
            "EtFOSAA",
            "FOSA",
            "L_PFOS",
            "PFBA",
            "PFOA",
            "PFHpA",
            "PFHxA",
            "PFPeA",
            "PFPeS",
        ],
        "H": ["PFBA", "PFBS", "PFHpA", "PFHxA", "PFOA", "PFOS", "PFPeA", "TFA"],
        "I": [
            "L_PFHpS",
            "PFNA",
            "L_PFBS",
            "L_PFOS",
            "PFDA",
            "PFPeS",
            "L_PFHxS",
            "PFBA",
            "PFHpA",
            "PFOA",
            "PFPeA",
        ],
    }
    cluster_results = full_pca_analysis(
        gdf_dropped, groups_dict, save_path, basins, id_name
    )
    plot_consensus_matrix(cluster_results, save_path)
    logger.info("--- Finished Factor Analysis ---")


def main() -> None:
    input_path = Path("data/input/")
    gdf: gpd.GeoDataFrame = gpd.read_file(input_path.joinpath("pfas_data.gpkg"))
    gdf = gdf[gdf.year > 2018]
    # gdf = gpd.read_file(input_path.joinpath("pfas_data.gpkg"))
    basins: gpd.GeoDataFrame = gpd.read_file(
        input_path.joinpath("hybas_eu_lev04_v1c.shp")
    )
    gdf_dropped = gdf[~gdf["less_than"]]
    gdf_dropped["month"] = pd.to_datetime(
        gdf_dropped["date"], format="ISO8601"
    ).dt.month
    gdf_dropped["dayofyear"] = pd.to_datetime(
        gdf_dropped["date"], format="ISO8601"
    ).dt.dayofyear
    save_path = Path("results/")

    basins = basins.to_crs(gdf.crs)  # pyright: ignore[reportArgumentType]

    factor_analysis(gdf_dropped, basins, save_path)


if __name__ == "__main__":
    main()
