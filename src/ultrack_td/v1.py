from typing import Optional, Literal, Sequence, Union

import tracksdata as td
import polars as pl
from numpy.typing import ArrayLike

from ultrack_td import UltrackMultiHypotheses
try:
    from ultrack.config import MainConfig
    from ultrack.utils import labels_to_contours
except ImportError as e:
    raise ImportError(
        "`ultrack` is not installed. Please install it with `pip install ultrack-td[v1]`.\n"
        "If this doesn't work follow the instructions at: https://royerlab.github.io/ultrack/install.html"
    ) from e


def _validate_config(config: MainConfig) -> None:
    if config.tracking_config.link_function != "power":
        raise ValueError(
            "`ultrack_td.v1.track`'s `config.tracking_config.link_function` must be 'power'. "
            f"Found `{config.tracking_config.link_function}`."
        )
    
    if config.tracking_config.image_border_size is not None:
        raise NotImplementedError(
            "`ultrack_td.v1.track`'s `config.tracking_config.image_border_size` is not supported yet.\n"
            "Please apply the shifts yourself in the tracksdata graph."
        )
    
    if config.tracking_config.window_size is not None:
        raise NotImplementedError(
            "`ultrack_td.v1.track`'s `config.tracking_config.window_size` is not supported yet.\n"
            "To process large datasets use `tracksdata`'s API directly."
        )


def track(
    *,
    graph: td.graph.BaseGraph,
    config: MainConfig,
    labels: Optional[ArrayLike] = None,
    sigma: Optional[Union[Sequence[float], float]] = None,
    foreground: Optional[ArrayLike] = None,
    contours: Optional[ArrayLike] = None,
    images: Sequence[ArrayLike] = tuple(),
    scale: Optional[Sequence[float]] = None,
    vector_field: Optional[Union[ArrayLike, Sequence[ArrayLike]]] = None,
    overwrite: Literal["all", "links", "solutions", "none", bool] = "none",
) -> None:
    """
    All-in-one function for cell tracking, it accepts multiple inputs (labels or contours)
    and run all intermediate steps, computing segmentation hypothesis, linking and solving the ILP.
    The results must be queried using the export function of preference.

    Note: Either `labels` or `foreground` and `contours` can be used as input, but not both.

    Parameters
    ----------
    config : MainConfig
        Tracking configuration parameters.
    labels : Optional[ArrayLike], optional
        Segmentation labels of shape (T, (Z), Y, X), by default None
    sigma : Optional[Union[Sequence[float], float]], optional
        Edge smoothing parameter (gaussian blur) for labels to contours conversion, by default None
    foreground : Optional[ArrayLike], optional
        Foreground probability array of shape (T, (Z), Y, X), by default None
    contours : Optional[ArrayLike], optional
        Contours array of shape (T, (Z), Y, X), by default None
    images : Sequence[ArrayLike]
        DEPRECATED: This argument is not fully compatible with `ultrack`.
        Use tracksdata's API directly.
    scale : Sequence[float]
        Optional scaling for nodes' distances.
    vector_field : Union[ArrayLike, Sequence[ArrayLike]]
        DEPRECATED: This argument is not fully compatible with `ultrack`.
        Only the `None` option is supported.
        Use tracksdata's API directly.
    overwrite : Literal["all", "links", "solutions", "none"], optional
        DEPRECATED: This argument is not fully compatible with `ultrack`.
        Only the `none` or `False` option is supported.
        Use tracksdata's API directly.
    """

    _validate_config(config)

    if len(images) > 0:
        raise ValueError(
            "`ultrack_td.v1.track`'s `images` argument is not supported yet.\n"
            "Please extract the `images` features yourself and tracking using `tracksdata`'s API."
        )

    if vector_field is not None:
        raise NotImplementedError(
            "ultrack's vector field API is not supported yet.\n"
            "Please apply the shifts yourself in the tracksdata graph."
        )
    
    if (overwrite != "none" and overwrite != False):
        raise ValueError(
            "`ultrack_td.v1.track`'s `overwrite` option must be a boolean or 'none'. "
            f"Found `{overwrite}`."
        )

    if labels is not None and (foreground is not None or contours is not None):
        raise ValueError(
            "`labels` and `foreground` or `contours` cannot be supplied at the same time."
        )

    if labels is not None:
        foreground, contours = labels_to_contours(labels, sigma=sigma)

    elif foreground is None or contours is None:
        raise ValueError(
            "Both `foreground` and `contours` must be supplied when not using `labels`."
        )
    
    if foreground.dtype != bool:
        raise ValueError(
            f"Unlike `ultrack`, we require `foreground` to be a boolean array. Got `{foreground.dtype}`."
        )

    UltrackMultiHypotheses(
        min_num_pixels=config.segmentation_config.min_area,
        max_num_pixels=config.segmentation_config.max_area,
        min_frontier=config.segmentation_config.min_frontier,
    ).add_nodes(
        graph=graph,
        foreground=foreground,
        contours=contours,
    )

    spatial_cols = ["z", "y", "x"][-len(scale):]
    if scale is not None:
        node_attrs = graph.node_attrs(attr_keys=[td.DEFAULT_ATTR_KEYS.NODE_ID, *spatial_cols])
        node_attrs = node_attrs.with_columns(
            (pl.col(c) * s).alias(f"{c}_scaled")
            for c, s in zip(spatial_cols, scale, strict=True)
        )
        # updating the spatial columns to the scaled ones
        spatial_cols = [f"{c}_scaled" for c in spatial_cols]
        for c in spatial_cols:
            graph.add_node_attr_key(c, -1.0)
        graph.update_node_attrs(
            attrs=node_attrs.select(*spatial_cols).to_dict(as_series=False),
            node_ids=node_attrs[td.DEFAULT_ATTR_KEYS.NODE_ID],
        )

    td.edges.DistanceEdges(
        distance_threshold=config.linking_config.max_distance,
        n_neighbors=config.linking_config.max_neighbors,
        attr_keys=spatial_cols,
    ).add_edges(graph=graph)

    td.edges.IoUEdgeAttr(output_key="iou").add_edge_attrs(graph=graph)

    td.solvers.ILPSolver(
        edge_weight=-td.EdgeAttr("iou").pow(config.tracking_config.power) - config.tracking_config.bias,
        appearance_weight=-config.tracking_config.appear_weight,
        disappearance_weight=-config.tracking_config.disappear_weight,
        division_weight=-config.tracking_config.division_weight,
        gap=config.tracking_config.solution_gap,
        num_threads=config.tracking_config.n_threads,
        timeout=config.tracking_config.time_limit,
    ).solve(graph=graph)
