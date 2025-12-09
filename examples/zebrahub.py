import napari
import numpy as np
import tracksdata as td
from rich import print

from ultrack_td.v1 import track
from ultrack.config import load_config
from ultrack.imgproc import robust_invert, detect_foreground
from ultrack.utils.array import array_apply, create_zarr


def main() -> None:
    """
    This is an example how the tracksdata implementation is compatible with the original `ultrack` API
    Through the `ultrack_td.v1.track` module.

    For reference, the original `ultrack` example is available at:
    https://github.com/royerlab/ultrack/blob/main/examples/zebrahub/zebrahub.ipynb
    """
    config = load_config("config.toml")
    print(config)

    viewer = napari.Viewer()
    viewer.window.resize(1800, 1000)

    img_layer, = viewer.open(
        "http://public.czbiohub.org/royerlab/zebrahub/imaging/single-objective/ZSNS001_tail.ome.zarr/",
        plugin="napari-ome-zarr",
        rendering="attenuated_mip",
        gamma=0.7,
        contrast_limits=(0, 500),
    )

    voxel_size = img_layer.scale[1:]  # (z, y, x) pixel size
    start_idx = 400  # starting frame
    viewer.dims.set_point(0, start_idx + 5)

    image = img_layer.data[0]
    image = image[start_idx:(start_idx + 10)]  # processing only a subset of time points

    foreground = create_zarr(image.shape, np.uint8, store_or_path="detection.zarr", overwrite=True)
    array_apply(
        image,
        out_array=foreground,
        func=detect_foreground,
        sigma=25.0,
        voxel_size=voxel_size,
    )

    contours = create_zarr(image.shape, np.float16, store_or_path="boundaries.zarr", overwrite=True)
    array_apply(
        image,
        out_array=contours,
        func=robust_invert,
        voxel_size=voxel_size,
    )

    viewer.add_image(contours, visible=False, translate=(start_idx, 0, 0, 0), scale=voxel_size)
    viewer.add_labels(
        foreground,
        visible=True,
        translate=(start_idx, 0, 0, 0),
        scale=voxel_size,
    ).contour = 2

    graph = td.graph.InMemoryGraph()

    track(
        graph=graph,
        config=config,
        foreground=foreground,
        contours=contours,
        scale=voxel_size,
    )

    solution_graph = graph.filter(td.NodeAttr("solution") == True, td.EdgeAttr("solution") == True)

    tracks_df, tracks_graph, segms = td.functional.to_napari_format(
        solution_graph, shape=image.shape, solution_key=None, mask_key=td.DEFAULT_ATTR_KEYS.MASK,
    )

    viewer.add_tracks(tracks_df, graph=tracks_graph)
    viewer.add_labels(segms, opacity=0.5)

    napari.run()


if __name__ == "__main__":
    main()
