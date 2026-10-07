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
import pickle
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
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

from pfas_assessment_europe.constants import HYBAS_RIVER_RENAME

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

    For each substance group, the function generates a biplot showing:

    - Rotated sample scores.
    - Rotated substance loadings.
    - Basin-specific colors and markers.
    - Convex hulls around observations from each basin.
    - Labels adjusted to reduce overlap.

    The primary group named ``"A"`` is saved as
    ``main_pcafa_biplot_A.pdf``. Biplots for other groups are saved using the
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
    report_results = []

    plt.rcParams.update({"font.size": 12})

    report_sum_expl_max = 0.0
    report_sum_expl_min = 1.0
    report_max_corr = 0.0
    report_third_comp_max = 0.0

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
        df_reference = df.copy()
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
        logger.info(f"PCA explained variance (ratio): {pca.explained_variance_ratio_}")
        logger.info(f"PCA sum explained variance (ratio): {np.sum(pca.explained_variance_ratio_)}")

        report_sum_expl_max = max(np.sum(pca.explained_variance_ratio_), report_sum_expl_max)
        report_sum_expl_min = min(np.sum(pca.explained_variance_ratio_), report_sum_expl_min)
        report_third_comp_max = max(pca.explained_variance_ratio_[2], report_third_comp_max)
        rotator = Rotator(method="varimax")

        pattern = rotator.fit_transform(pca.components_.T)

        scores_rot = X_pca @ rotator.rotation_  # (locations × components)

        # Flip so that loadings with largest magnitude are positive. This was done to improve readability of bi-plots
        flip = np.abs(pattern.min(axis=0)) > np.abs(pattern.max(axis=0))
        pattern[:, flip] *= -1
        scores_rot[:, flip] *= -1

        x, y, z = pattern.T

        points_region = pd.DataFrame(
            scores_rot,
            index = df.index,
            columns = ["factor1", "factor2", "factor3"]
        )
        site_basin = df_reference[["site", "basin"]].drop_duplicates()
        basin_by_site = site_basin.set_index("site")["basin"]
        points_region["basin"] = basin_by_site.reindex(points_region.index)

        logger.info(points_region.basin.unique())

        regions = sorted(points_region[id_name].dropna().unique())
        if len(regions) > 10:
            palette = sns.color_palette("tab20", 20)
        else:
            palette = sns.color_palette("tab10", 10)

        region_colors = dict(zip(regions, palette))
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
        region_markers = {region: next(marker_cycle) for region in regions}

        # Initialise max_corr
        max_corr = 0.0
        rotator_for_phi = Rotator(method="quartimin")  # oblique rotation
        rotator_for_phi.fit_transform(pca.components_.T)
        phi = rotator_for_phi.phi_
        max_corr = np.max(np.abs(phi[np.triu_indices_from(phi, k=1)]))

        logger.info(f"Maximum absolute factor correlation: {max_corr:.3f}")

        report_max_corr = max(max_corr, report_max_corr)

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
        for region, group in points_region.groupby(id_name):
            ax1.scatter(
                group["factor1"],
                group["factor3"],
                color=region_colors[region],
                marker=region_markers[region],
                label=region,
                alpha=0.4,
                zorder=1,
            )
            plot_convex_hull(ax1, group, "factor1", "factor3", region_colors[region])
            txt = ax1.text(
                group["factor1"].mean(),
                group["factor3"].mean(),
                region,
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

        for region, group in points_region.groupby(id_name):
            ax2.scatter(
                group["factor2"],
                group["factor3"],
                color=region_colors[region],
                marker=region_markers[region],
                label=region,
                alpha=0.4,
                zorder=1,
            )
            plot_convex_hull(ax2, group, "factor2", "factor3", region_colors[region])
            txt = ax2.text(
                group["factor2"].mean(),
                group["factor3"].mean(),
                region,
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

        ax1.text(
            -0.08,
            1.05,
            "a",
            transform=ax1.transAxes,
            fontsize=11,
            fontweight="bold",
            va="top",
        )
        ax2.text(
            -0.08,
            1.05,
            "b",
            transform=ax2.transAxes,
            fontsize=11,
            fontweight="bold",
            va="top",
        )
        handles = [
            Line2D(
                [0],
                [0],
                marker=region_markers[c],
                color="w",
                markerfacecolor=region_colors[c],
                markeredgecolor=region_colors[c],
                markersize=6,
                linestyle="None",
                label=c,
            )
            for c in regions
        ]

        n_sub = len(regions)

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
            plt.savefig(save_path / f"main_pcafa_biplot_{group_n}.pdf", bbox_inches="tight")
        else:
            plt.savefig(
                save_path / f"appendix_pcafa_biplot_{group_n}.pdf", bbox_inches="tight"
            )
        plt.close(fig)

        report_results.append(
            {
                "Group ID" : group_n,
                "Substances" : substances,
                "Sampling sites": matching_locations,
                "Samples" : len(df),
                "KMO criterion" : kmo_model,
                "Bartlett's p" : p_value,
                "Explained Variance 1" : pca.explained_variance_ratio_[0],
                "Explained Variance 2" : pca.explained_variance_ratio_[1],
                "Explained Variance 3" : pca.explained_variance_ratio_[2],
                "Sum Explained Variance" : np.sum(pca.explained_variance_ratio_),
                "Max absolute factor correlation" : max_corr,
            }
        )

    logger.info(f"Minimum cumulative variance explained: {report_sum_expl_min}")
    logger.info(f"Maximum cumulative variance explained: {report_sum_expl_max}")
    logger.info(f"Maximum third component across groups: {report_third_comp_max}")
    logger.info(f"Maximum absolute factor correlation across groups: {report_max_corr:.3f}")

    return cluster_results, report_results


def plot_consensus_matrix(cluster_results, save_path: Path) -> None:
    sort_substances = [
        "TFA", "PFBA", "PFPeA", "PFHxA", "PFHpA", "PFOA", "PFNA", "PFDA",
        "PFUnDA", "PFDoDA", "PFTrDA", "PFTeDA", "PFHxDA", "PFODA",
        "PFBS", "PFPeS", "PFHxS", "PFHpS", "PFOS", "PFNS", "PFDS",
        "PFUnDS", "PFDoDS", "PFTrDS", "Linear PFOA", "Linear PFOS", "6:2 FTCA", "4:2 FTS",
        "6:2 FTS", "8:2 FTS", "HFPO-DA", "DONA", "FOSA", "N-Et-FOSA",
        "N-MeFOSAA", "EtFOSAA",
    ]

    # These substances appear first, in this order.
    preferred_order = [
        "PFPeA", "PFHxA", "PFHpA",
        "PFOA", "PFBA", "PFBS", 
        "PFDA", "PFNA",
        "PFOS", "PFHxS", "PFPeS", "PFHpS", 
        "Linear PFOA", "Linear PFOS"
    ]

    # Each group must be contiguous in the displayed order for its rectangle
    # to enclose just that group's block.
    groups = [
        {
            "substances": ["PFPeA", "PFHxA", "PFHpA"],
            "label": "Fingerprint 1: Short-chain PFCAs",
            "color": "red",
            "text_x_offset": 14,
        },
        {
            "substances": ["PFPeA", "PFHxA", "PFHpA", "PFOA", "PFBA"],
            "label": "Occasional co-loading of \nshort- and long-chain PFCAs",
            "color": "black",
            "text_x_offset": 14,
            "text_y_offset": 0.5,
            "linestyle" : "--",
            "linewidth" : 1
        },
        {
            "substances": ["PFOS", "PFHxS"],
            "label": "Fingerprint 2: Long-chain PFSAs",
            "text_x_offset" : 14,
            "color": "red",
        },
        {
            "substances": ["PFOS", "PFHxS", "PFPeS", "PFHpS"],
            "label": "Co-loading of \nshort- and long-chain PFSAs",
            "color": "black",
            "text_x_offset": 14,
            "text_y_offset": 0.5,
            "linestyle" : "--",
            "linewidth" : 1
        },
        {
            "substances": ["PFBA", "PFBS"],
            "label": "Fingerprint 3: Short-chain PFAAs",
            "color": "red",
            "text_x_offset": 14,
        },
        {
            "substances": ["PFNA", "PFDA"],
            "label": "Fingerprint 4: Long-chain PFCAs",
            "color": "red",
            "text_x_offset": 14,
            "text_y_offset": -0.5,
        },
        {
            "substances": ["PFOS", "PFHxS", "PFDA", "PFNA"],
            "label": "Occasional co-loading of \nFingerprint 2 and 4 substances",
            "color": "black",
            "text_x_offset": 14,
            "text_y_offset": -0.5,
            "linestyle" : "--",
            "linewidth" : 1
        },
    ]

    # Normalize names such as L_PFOA to Linear PFOA.
    normalized_results = [
        {
            key.replace("L_", "Linear ", 1) if key.startswith("L_") else key: value
            for key, value in record.items()
        }
        for record in cluster_results
    ]

    present_features = {feature for record in normalized_results for feature in record}

    # Requested order first, then the original preferred order, then any
    # remaining features alphabetically.
    features = (
        [f for f in preferred_order if f in present_features]
        + [
            f for f in sort_substances
            if f in present_features and f not in preferred_order
        ]
        + sorted(present_features - set(sort_substances) - set(preferred_order))
    )

    n = len(features)
    if n == 0:
        raise ValueError("No features found in cluster_results.")

    idx = {feature: i for i, feature in enumerate(features)}
    co_mat = np.zeros((n, n), dtype=float)
    co_run_count = np.zeros((n, n), dtype=float)

    for record in normalized_results:
        present = [feature for feature in record if feature in idx]

        for fi in present:
            for fj in present:
                i, j = idx[fi], idx[fj]
                co_run_count[i, j] += 1
                if record[fi] == record[fj]:
                    co_mat[i, j] += 1

    consensus = co_mat / np.maximum(co_run_count, 1)

    # Mask pairs that never appeared together and the upper triangle.
    mask_never = co_run_count == 0
    mask_upper = np.triu(np.ones((n, n), dtype=bool), k=1)
    mask = mask_never | mask_upper

    consensus_masked = consensus.copy()
    consensus_masked[mask_never] = np.nan

    cmap = sns.color_palette("crest", as_cmap=True)
    # cmap.set_bad(color="lightgrey")

    fig, ax = plt.subplots(figsize=(max(12, n * 0.45), max(8, n * 0.45)))

    sns.heatmap(
        consensus_masked,
        mask=mask,
        cmap=cmap,
        xticklabels=features,
        yticklabels=features,
        square=True,
        cbar_kws={"label": "Co-clustering frequency"},
        linewidth=0.1,
        linecolor="white",
        annot=True,
        fmt=".1f",
        ax=ax,
    )

    # Outline each group's diagonal block and label it to the right.
    for group in groups:
        members = [s for s in group["substances"] if s in idx]
        if not members:
            continue

        positions = sorted(idx[s] for s in members)
        start, end = positions[0], positions[-1]

        if positions != list(range(start, end + 1)):
            raise ValueError(
                f"Group {group['label']!r} is not contiguous in the feature order. "
                "Adjust preferred_order so its substances are next to each other."
            )
        linestyle = group.get("linestyle", "-")
        linewidth = group.get("linewidth", 2.5)
        size = end - start + 1
        ax.add_patch(
            patches.Rectangle(
                (start, start),
                size,
                size,
                fill=False,
                edgecolor=group["color"],
                linewidth = linewidth,
                linestyle = linestyle,
                clip_on=False,
            )
        )

        text_x = group.get("text_x_offset", n + 1)
        text_y_offset = group.get("text_y_offset", 0)
        center_y = start + size / 2
        ax.annotate(
            group["label"],
            xy=(end + 1, center_y + text_y_offset),
            xytext=(text_x, center_y + text_y_offset),
            ha="left",
            va="center",
            color=group["color"],
            arrowprops={"arrowstyle": "-", "color": group["color"]},
            annotation_clip=False,
        )

    # Make room for group labels placed to the right.
    # ax.set_xlim(0, n + 5)
    fig.tight_layout()
    fig.savefig(save_path / "appendix_consensus_matrix.pdf", bbox_inches="tight")
    plt.close(fig)

# def plot_consensus_matrix(cluster_results, save_path: Path) -> None:
#     """Plot and save a consensus matrix for repeated clustering results.

#     The consensus matrix shows the frequency with which pairs of substances
#     are assigned to the same cluster across multiple clustering runs. Pairs
#     that never occur together in a clustering run are displayed in grey.

#     The resulting heatmap is saved as
#     ``appendix_appendix_consensus_matrix.pdf`` in ``save_path``.

#     Args:
#         cluster_results: Iterable of dictionaries containing clustering
#             assignments. Each dictionary must map feature or substance names
#             to cluster labels.
#         save_path: Directory in which the consensus-matrix PDF is saved.

#     Returns:
#         None. The consensus heatmap is saved to disk.
#     """
#     sort_substances = [
#             "TFA",
#             "PFBA",
#             "PFPeA",
#             "PFHxA",
#             "PFHpA",
#             "PFOA",
#             "PFNA",
#             "PFDA",
#             "PFUnDA",
#             "PFDoDA",
#             "PFTrDA",
#             "PFTeDA",
#             "PFHxDA",
#             "PFODA",
#             "PFBS",
#             "PFPeS",
#             "PFHxS",
#             "PFHpS",
#             "PFOS",
#             "PFNS",
#             "PFDS",
#             "PFUnDS",
#             "PFDoDS",
#             "PFTrDS",
#             "Linear PFOA",
#             "Linear PFBS",
#             "Linear PFHpS",
#             "Linear PFHxS",
#             "Linear PFOS",
#             "6:2 FTCA",
#             "4:2 FTS",
#             "6:2 FTS",
#             "8:2 FTS",
#             "HFPO-DA",
#             # "ADONA",
#             "DONA",
#             "FOSA",
#             "N-Et-FOSA",
#             "N-MeFOSAA",
#             "EtFOSAA",
#         ]
#     cluster_results = [ 
#         { 
#             key.replace("L_", "Linear ", 1) if key.startswith("L_") else key: value for key, value in record.items() 
#             } for record in cluster_results 
#         ]

#     features = sorted({f for r in cluster_results for f in r})
#     features = [substance for substance in sort_substances if substance in features]
#     # print(features)
#     # print(cluster_results)
#     n = len(features)
#     plt.rcParams.update({"font.size": 12})
#     co_mat = np.zeros((n, n))
#     co_run_count = np.zeros((n, n))  # number of runs where both features exist

#     #  map feature name -> index
#     idx = {f: i for i, f in enumerate(features)}

#     # compute co-occurrence and count of runs where both features exist
#     for r in cluster_results:
#         present = list(r.keys())
#         for i, fi in enumerate(present):
#             for j, fj in enumerate(present):
#                 ii, jj = idx[fi], idx[fj]
#                 co_run_count[ii, jj] += 1
#                 if r[fi] == r[fj]:
#                     co_mat[ii, jj] += 1

#     consensus = co_mat / np.maximum(co_run_count, 1)

#     # create mask for pairs that never co-occurred
#     mask_never = co_run_count == 0

#     # set these to NaN so they can be grey
#     consensus_masked = consensus.copy()
#     consensus_masked[mask_never] = np.nan
#     # print(consensus_masked)
#     # plot
#     cmap = sns.color_palette("crest", as_cmap=True)
#     cmap.set_bad(color="lightgrey")  # grey for never co-occurred pairs

#     plt.figure(figsize=(10, 8))
#     sns.heatmap(
#         consensus_masked,
#         cmap=cmap,
#         xticklabels=features,
#         yticklabels=features,
#         square=True,
#         cbar_kws={"label": "Co-clustering frequency"},
#         linewidth=0.1,
#         linecolor="white",
#         annot=True,
#         fmt=".1f",
#     )

#     plt.tight_layout()
#     plt.savefig(save_path / "appendix_consensus_matrix.pdf", bbox_inches="tight")

#     plt.close()


def factor_analysis(
    gdf_dropped: gpd.GeoDataFrame, 
    save_path: Path
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
        save_path: Directory in which the generated plots are saved.

    Returns:
        None. Analysis results are written to log output, and generated figures
        are saved to ``save_path``.
    """
    logger.info("--- Starting Factor Analysis ---")
    # lev04_rename = HYBAS_RIVER_RENAME
    # id_name = "HYBAS_ID"
    # basins[id_name] = basins[id_name].astype("object")
    # basins.loc[~basins[id_name].isin(lev04_rename.keys()), id_name] = "Other"
    # basins[id_name] = basins[id_name].replace(lev04_rename)
    # basins[id_name] = basins[id_name].astype("string")
    id_name = "basin"
    groups_dict = {
        "A": ["PFBA", "PFBS", "PFHpA", "PFHxS", "PFHxA", "PFOA", "PFOS", "PFPeA"],
        "A1": [
            "PFBA",
            "PFBS",
            "PFHpA",
            "PFHxS",
            "PFHxA",
            "PFOA",
            "PFOS",
            "PFPeA",
            "PFNA", #
            "PFDA",#
        ],
        "A2": [
            "PFBA",
            "PFBS",
            "PFHpA",
            "PFHxS",
            "PFHxA",
            "PFOA",
            "PFOS",
            "PFPeA",
            "L_PFOS",#
            "L_PFOA",#
        ],
        "A3": [
            "PFBA",
            "PFBS",
            "PFHpA",
            "PFHxS",
            "PFHxA",
            "PFOA",
            "PFOS",
            "PFPeA",
            "6:2 FTS",#
        ],
        "A4": [
            "PFBA",
            "PFBS",
            "PFHpA",
            "PFHxS",
            "PFHxA",
            "PFOA",
            "PFOS",
            "PFPeA",
            "PFPeS",#
        ],
        # "A05": [
        #     "PFBA",
        #     "PFBS",
        #     "PFHpA",
        #     "PFHxS",
        #     "PFHxA",
        #     "PFOA",
        #     "PFOS",
        #     "PFPeA",
        #     "PFNA",#
        #     "PFPeS",#
        # ],
        "A5": [
            "PFBA",
            "PFBS",
            "PFHpA",
            "PFHxS",
            "PFHxA",
            "PFOA",
            "PFOS",
            "PFPeA",
            "PFDA"#
        ],
        "A6": [
            "PFBA",
            "PFBS",
            "PFHpA",
            "PFHxS",
            "PFHxA",
            "PFOA",
            "PFOS",
            "PFPeA",
            "PFNA",#
        ],
        "A7": [
            "PFBA",
            "PFBS",
            "PFHpA",
            "PFHxS",
            "PFHxA",
            "PFOA",
            "PFOS",
            "PFPeA",
            "HFPO-DA",#
        ],
        "A8": [
            "PFBA",
            "PFBS",
            "PFHpA",
            "PFHxS",
            "PFHxA",
            "PFOA",
            "PFOS",
            "PFPeA",
            "FOSA",#
        ],        
        # "A08": [
        #     "PFBA",
        #     "PFBS",
        #     "PFHpA",
        #     "PFHxS",
        #     "PFHxA",
        #     "PFOA",
        #     "PFOS",
        #     "PFPeA",
        #     "HFPO-DA",#
        # ],
        "B": [
            "PFOA", 
            "PFOS", 
            "PFHxA", 
            "PFBS", 
            "PFHpA", 
            "PFHxS", 
            "PFHpS"#
            ],
        "C":["N-MeFOSAA", "PFBS", "PFHpA", "PFHxA", "PFHxS", "PFOA", "PFOS", "PFPeA"], # N-MeFOSAA
        # "C1":["6:2 FTS", "EtFOSAA", "FOSA", "L_PFOS", "N-MeFOSAA", "PFPeS"],
        # "D":["6:2 FTS", "EtFOSAA", "FOSA", "L_PFOS", "N-MeFOSAA", "PFPeS"], # EtFOSAA
        "D":["EtFOSAA", "PFBA", "PFBS", "PFHpA", "PFHxA", "PFHxS", "PFOA", "PFPeA"],
        # "E": ["DONA", "PFBA", "PFHxA", "PFOA", "PFOS", "PFPeA"], # DONA
        "E":["DONA", "PFBA", "PFHpA", "PFHxA", "PFOA", "PFOS", "PFPeA"],
        # "F":["6:2 FTS","8:2 FTS", "EtFOSAA", "L_PFOS", "N-MeFOSAA", "PFPeS"], # 8:2 FTS
        # "F1":["6:2 FTS", "8:2 FTS", "EtFOSAA", "FOSA", "L_PFOS", "N-MeFOSAA", "PFPeS"],
        # "C": [
        #     # "L_PFBS",
        #     # "L_PFHxS",
        #     "L_PFOS",
        #     "PFBA",
        #     "PFDA",
        #     "PFHpA",
        #     "PFHxA",
        #     "PFNA",
        #     "PFOA",
        #     "PFPeA",
        #     "PFPeS",
        # ],
        # "D": [
        #     "EtFOSAA",
        #     # "L_PFBS",
        #     # "L_PFHxS",
        #     "L_PFOS",
        #     "PFBA",
        #     "PFDA",
        #     "PFHpA",
        #     "PFHxA",
        #     "PFNA",
        #     "PFOA",
        #     "PFPeA",
        #     "PFPeS",
        # ],
        # "E": [
        #     "EtFOSAA",
        #     "FOSA",
        #     "N-MeFOSAA",
        #     # "L_PFBS",
        #     # "L_PFHxS",
        #     "L_PFOS",
        #     "PFBA",
        #     "PFDA",
        #     "PFHpA",
        #     "PFHxA",
        #     "PFNA",
        #     "PFOA",
        #     "PFPeA",
        #     "PFPeS",
        # ],
        # "F": ["FOSA", "L_PFOS", "PFBA", "PFOA", "PFHpA", "PFHxA", "PFPeA", "PFPeS"],
        # "G": [
        #     "EtFOSAA",
        #     "FOSA",
        #     "L_PFOS",
        #     "PFBA",
        #     "PFOA",
        #     "PFHpA",
        #     "PFHxA",
        #     "PFPeA",
        #     "PFPeS",
        # ],
        # "H": ["PFBA", "PFBS", "PFHpA", "PFHxA", "PFOA", "PFOS", "PFPeA", "TFA"],
        "F": ["PFBA", "PFOS", "PFHxS", "PFOA", "PFHxA", "PFPeA", "PFHpA", "TFA"], #TFA
        # "I": [
        #     # "L_PFHpS",
        #     "PFNA",
        #     # "L_PFBS",
        #     "L_PFOS",
        #     "PFDA",
        #     "PFPeS",
        #     # "L_PFHxS",
        #     "PFBA",
        #     "PFHpA",
        #     "PFOA",
        #     "PFPeA",
        # ],
        "G": ["PFDA", "PFHpA", "PFHxA", "PFHxS", "PFNA", "PFOA", "PFPeA", "PFPeS"]
    }
    cluster_results, report_results = full_pca_analysis(
        gdf_dropped, 
        groups_dict, 
        save_path, 
        # basins, 
        id_name
    )
    # with open("cluster_results.pkl", "wb") as f:
    #     pickle.dump(cluster_results, f)
    plot_consensus_matrix(cluster_results, save_path)
    logger.info("--- Finished Factor Analysis ---")

    return report_results


def main() -> None:
    # input_path = Path("data/input/")
    # gdf: gpd.GeoDataFrame = gpd.read_file(input_path.joinpath("pfas_data.gpkg"))
    # gdf = gdf[gdf.year > 2018]
    # # gdf = gpd.read_file(input_path.joinpath("pfas_data.gpkg"))
    # basins: gpd.GeoDataFrame = gpd.read_file(
    #     input_path.joinpath("hybas_eu_lev04_v1c.shp")
    # )
    # gdf_dropped = gdf[~gdf["less_than"]]
    # gdf_dropped["month"] = gdf_dropped["date"].dt.month
    # gdf_dropped["dayofyear"] = gdf_dropped["date"].dt.dayofyear
    # # gdf_dropped["month"] = pd.to_datetime(
    # #     gdf_dropped["date"], format="ISO8601"
    # # ).dt.month
    # # gdf_dropped["dayofyear"] = pd.to_datetime(
    # #     gdf_dropped["date"], format="ISO8601"
    # # ).dt.dayofyear
    save_path = Path("results/")

    # basins = basins.to_crs(gdf.crs)  # pyright: ignore[reportArgumentType]

    # factor_analysis(gdf_dropped, basins, save_path) 
    file_path = Path("cluster_results.pkl")
    # with file_path.open("rb") as f:
    #     cluster_results = pickle.load(f)
    plot_consensus_matrix(cluster_results, save_path)


if __name__ == "__main__":
    main()
