from typing import override, Any
from functools import partial

import tracksdata as td
import numpy as np
from numpy.typing import NDArray
from tracksdata.nodes._mask import Mask
from tracksdata.graph import BaseGraph
from tracksdata.nodes._base_nodes import BaseNodesOperator
from tracksdata.utils._multiprocessing import multiprocessing_apply
from tracksdata.utils._logging import LOG

from .ultrack_td_ext import compute_segmentation_hypotheses, overlap_dict_from_segments


class UltrackMultiHypotheses(BaseNodesOperator):
    _default_attr_keys = ["bbox", "z", "y", "x", "num_pixels"]

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
        TODO
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
            overlaps = [
                (id_map[n_id], id_map[o_id])
                for n_id, overlaps in overlap_dict.items()
                for o_id in overlaps
            ]
            graph.bulk_add_overlaps(overlaps=overlaps)

    def _nodes_per_time(
        self,
        t: int,
        *,
        foreground: NDArray[np.bool_],
        contours: NDArray[np.float32],
    ) -> tuple[list[dict[str, Any]], dict[int, list[int]]]:
        """
        TODO
        """
        nodes_data = []

        if foreground.shape != contours.shape:
            raise ValueError(f"Foreground and contours must have the same shape. Got {foreground.shape} and {contours.shape}")

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
            attrs = {
                key: getattr(segm, key) for key in self._default_attr_keys
            }
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
