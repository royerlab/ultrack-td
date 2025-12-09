import napari
import numpy as np
import dask.array as da
from scipy.ndimage import gaussian_filter
import tracksdata as td

from ultrack_td import UltrackMultiHypotheses


def main() -> None:

    viewer = napari.Viewer()
    viewer.window.resize(1800, 1000)

    img_layer, = viewer.open(
        "http://public.czbiohub.org/royerlab/zebrahub/imaging/single-objective/ZSNS001_tail.ome.zarr/",
        plugin="napari-ome-zarr",
        rendering="attenuated_mip",
        gamma=0.7,
        contrast_limits=(0, 500),
    )

    scale_idx = 0
    n_timepoints = 5

    scale = np.asarray([1, *[s * 2 ** scale_idx for s in img_layer.scale[-3:]]])

    img = img_layer.data[scale_idx]
    img = da.rechunk(img, chunks=(1, *img.shape[1:]))
    img = img.map_blocks(gaussian_filter, sigma=(2 / scale), dtype=np.float32)

    local_data = img[:n_timepoints].compute()

    foreground = local_data > 10
    contours = -local_data + local_data.max()

    graph = td.graph.InMemoryGraph()

    UltrackMultiHypotheses(
        min_num_pixels=500,
        max_num_pixels=100_000,
        min_frontier=0.0,
    ).add_nodes(
        graph=graph,
        foreground=foreground,
        contours=contours,
    )

    td.edges.DistanceEdges(
        distance_threshold=25.0,
        n_neighbors=5,
    ).add_edges(graph=graph)

    td.edges.IoUEdgeAttr(output_key="iou").add_edge_attrs(graph=graph)

    solution = td.solvers.ILPSolver(
        edge_weight=-td.EdgeAttr("iou").pow(4),
        appearance_weight=0.002,
        disappearance_weight=0.01,
        division_weight=0.001,
    ).solve(graph=graph)

    tracks_df, tracks_graph, segms = td.functional.to_napari_format(
        solution, shape=local_data.shape, solution_key=None, mask_key=td.DEFAULT_ATTR_KEYS.MASK,
    )

    viewer.add_image(local_data, blending="additive", colormap="magenta", scale=scale)
    viewer.add_tracks(tracks_df, graph=tracks_graph)
    viewer.add_labels(segms, opacity=0.5)

    napari.run()


if __name__ == "__main__":
    main()
