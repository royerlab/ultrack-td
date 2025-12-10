from functools import partial
from typing import Any, override

import numpy as np
import tracksdata as td
from numpy.typing import NDArray
from tracksdata.graph import BaseGraph
from tracksdata.nodes._base_nodes import BaseNodesOperator
from tracksdata.nodes._mask import Mask
from tracksdata.utils._logging import LOG
from tracksdata.utils._multiprocessing import multiprocessing_apply

from ultrack_td.ultrack_td_ext import (
    compute_segmentation_hypotheses_double,
    compute_segmentation_hypotheses_float,
    compute_segmentation_hypotheses_int,
    compute_segmentation_hypotheses_int_8,
    compute_segmentation_hypotheses_int_16,
    compute_segmentation_hypotheses_int_32,
    compute_segmentation_hypotheses_uint,
    compute_segmentation_hypotheses_uint_8,
    compute_segmentation_hypotheses_uint_16,
    compute_segmentation_hypotheses_uint_32,
    overlap_dict_from_segments,
)

_compute_segmentation_hypotheses_funcs = {
    np.float32: compute_segmentation_hypotheses_float,
    np.float64: compute_segmentation_hypotheses_double,
    np.int8: compute_segmentation_hypotheses_int_8,
    np.int16: compute_segmentation_hypotheses_int_16,
    np.int32: compute_segmentation_hypotheses_int_32,
    np.int64: compute_segmentation_hypotheses_int,
    np.uint8: compute_segmentation_hypotheses_uint_8,
    np.uint16: compute_segmentation_hypotheses_uint_16,
    np.uint32: compute_segmentation_hypotheses_uint_32,
    np.uint64: compute_segmentation_hypotheses_uint,
    np.uint: compute_segmentation_hypotheses_uint,
}


def compute_segmentation_hypotheses(
    foreground: NDArray[np.bool_],
    contours: NDArray,
    min_num_pixels: int,
    max_num_pixels: int,
    min_frontier: float = float("-inf"),
):
    """
    Compute segmentation hypotheses for a given foreground and contours.

    Parameters
    ----------
    foreground : NDArray[np.bool_]
        Foreground mask of shape (T, Z, Y, X).
    contours : NDArray
        Contours of shape (T, Z, Y, X).
    min_num_pixels : int
        Minimum number of pixels for a valid segmentation hypothesis. Must be greater than 0.
    max_num_pixels : int
        Maximum number of pixels for a valid segmentation hypothesis. Must be greater than min_num_pixels.
    min_frontier : float
        Minimum frontier for a valid segmentation hypothesis. Must be greater than 0.

    Returns
    -------
    list[Segment]
        List of segmentation hypotheses.
    """

    if min_num_pixels < 0:
        raise ValueError(f"min_num_pixels must be greater than or equal to 0. Got {min_num_pixels}")

    if max_num_pixels < min_num_pixels:
        raise ValueError(
            f"max_num_pixels must be greater than or equal to min_num_pixels. Got {max_num_pixels} and {min_num_pixels}"
        )

    dtype = contours.dtype
    try:
        compute_segmentation_hypotheses_func = _compute_segmentation_hypotheses_funcs[dtype.type]
    except KeyError as e:
        raise ValueError(
            f"Unsupported dtype: {dtype.type}. Expected one of {list(_compute_segmentation_hypotheses_funcs.keys())}"
        ) from e

    return compute_segmentation_hypotheses_func(
        foreground=foreground,
        contours=contours,
        min_num_pixels=min_num_pixels,
        max_num_pixels=max_num_pixels,
        min_frontier=min_frontier,
    )


class UltrackMultiHypotheses(BaseNodesOperator):
    _default_attr_keys = ("bbox", "z", "y", "x", "num_pixels")

    def __init__(
        self,
        min_num_pixels: int,
        max_num_pixels: int,
        min_frontier: float = float("-inf"),
    ) -> None:
        super().__init__()
        self._min_num_pixels = min_num_pixels
        self._max_num_pixels = max_num_pixels
        self._min_frontier = min_frontier

    def _init_node_attrs(self, graph: BaseGraph) -> None:
        """
        Initialize the node attributes for the segmentation hypotheses.

        Parameters
        ----------
        graph : BaseGraph
            The graph to add the segmentation hypotheses to.
        """
        for key in self._default_attr_keys[1:]:  # skipping "bbox"
            if key not in graph.node_attr_keys:
                graph.add_node_attr_key(key, -1.0)

        if "bbox" not in graph.node_attr_keys:
            graph.add_node_attr_key("bbox", np.zeros(6, dtype=np.int32))

    @override
    def add_nodes(
        self,
        graph: BaseGraph,
        *,
        foreground: NDArray[np.bool_],
        contours: NDArray[np.float32],
        t: int | None = None,
    ) -> None:
        """
        Add segmentation hypotheses to the graph.

        Parameters
        ----------
        graph : BaseGraph
            The graph to add the segmentation hypotheses to.
        foreground : NDArray[np.bool_]
            The foreground mask.
        contours : NDArray[np.float32]
            The contours.
        t : int | None, optional
            The time point to add the segmentation hypotheses to. If None, all time points will be processed.
        """
        self._init_node_attrs(graph)

        if t is None:
            time_points = range(foreground.shape[0])
        else:
            time_points = [t]

        for nodes_data, overlap_dict in multiprocessing_apply(
            func=partial(self._nodes_per_time, foreground=foreground, contours=contours),
            sequence=time_points,
            desc="Adding segmentation hypotheses",
        ):
            old_ids = [n.pop("tmp_id") for n in nodes_data]
            node_ids = graph.bulk_add_nodes(nodes_data)
            id_map = dict(zip(old_ids, node_ids, strict=True))
            overlaps = [(id_map[n_id], id_map[o_id]) for n_id, overlaps in overlap_dict.items() for o_id in overlaps]
            graph.bulk_add_overlaps(overlaps=overlaps)

    def _nodes_per_time(
        self,
        t: int,
        *,
        foreground: NDArray[np.bool_],
        contours: NDArray[np.float32],
    ) -> tuple[list[dict[str, Any]], dict[int, list[int]]]:
        """
        Compute segmentation hypotheses for a given time point.

        Parameters
        ----------
        t : int
            The time point to compute the segmentation hypotheses for.
        foreground : NDArray[np.bool_]
            The foreground mask.
        contours : NDArray[np.float32]
            The contours.

        Returns
        -------
        tuple[list[dict[str, Any]], dict[int, list[int]]]
            - The nodes data to add to the graph.
            - The overlap dictionary, where each key is a node id and it maps to a
              list of all its overlaps (ancestors in the hierarchy)
        """
        nodes_data = []

        if foreground.shape != contours.shape:
            raise ValueError(
                f"Foreground and contours must have the same shape. Got {foreground.shape} and {contours.shape}"
            )

        foreground = np.asarray(foreground[t])
        contours = np.asarray(contours[t])

        if foreground.ndim == 2:
            foreground = foreground[None, ...]
            contours = contours[None, ...]

        if foreground.ndim != 3:
            raise ValueError(f"Foreground and contours must be 3D. Got {foreground.ndim}D array")

        LOG.info("Computing segmentation hypotheses for time point %d", t)

        hypotheses = compute_segmentation_hypotheses(
            foreground=foreground,
            contours=contours,
            min_num_pixels=self._min_num_pixels,
            max_num_pixels=self._max_num_pixels,
            min_frontier=self._min_frontier,
        )

        LOG.info("Found %d hypotheses for time point %d", len(hypotheses), t)

        for segm in hypotheses:
            attrs = {key: getattr(segm, key) for key in self._default_attr_keys}
            attrs["tmp_id"] = segm.id
            attrs[td.DEFAULT_ATTR_KEYS.T] = t
            attrs[td.DEFAULT_ATTR_KEYS.MASK] = Mask(segm.mask, segm.bbox)
            nodes_data.append(attrs)

        if len(nodes_data) == 0:
            LOG.warning("No valid nodes found for time point %d", t)
            overlap_dict = {}
        else:
            overlap_dict = overlap_dict_from_segments(hypotheses)

        LOG.info("Found %d overlaps for time point %d", len(overlap_dict), t)

        return nodes_data, overlap_dict
