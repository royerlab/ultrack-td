import time
import pytest
import edt
import napari
import rustworkx as rx
import numpy as np
from ultrack_td import compute_segmentation_hypotheses, overlap_dict_from_segments


def _validate_hierarchy(components) -> None:
    segm_hierarchy = rx.PyDiGraph()
    for component in components:
        segm_hierarchy.add_node(component.id - 1)

    for component in components:
        if component.parent_id != component.id:
            segm_hierarchy.add_edge(component.id - 1, component.parent_id - 1, None)

    expected_hierarchy = rx.PyDiGraph()
    expected_hierarchy.add_nodes_from(range(5))
    expected_hierarchy.add_edges_from([
        (0, 2, None),
        (1, 2, None),
        (3, 4, None),
        (2, 4, None),
    ])

    assert rx.is_isomorphic(segm_hierarchy, expected_hierarchy)


def _validate_overlap_dict(overlap_dict: dict[int, list[int]]) -> None:
    expected_n_overlaps = {
        0: 1,
        1: 2,
        2: 2,
    }
    seen_n_overlaps = {0: 0, 1: 0, 2: 0}

    for overlaps in overlap_dict.values():
        n_overlaps = len(overlaps)
        seen_n_overlaps[n_overlaps] += 1

    assert seen_n_overlaps == expected_n_overlaps


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint])
def test_multi_hypotheses_overlap(
    dtype: np.dtype,
    interactive: bool = False,
) -> None:
    # Generate an initial image with two overlapping circles
    x, y = np.indices((80, 80))
    x1, y1, x2, y2 = 27, 25, 44, 52
    r1, r2 = 15, 20
    mask_circle1 = (x - x1) ** 2 + (y - y1) ** 2 < r1**2
    mask_circle2 = (x - x2) ** 2 + (y - y2) ** 2 < r2**2
    # mask_circle1 = np.zeros_like(mask_circle1)  # FIXME
    foreground = np.logical_or(mask_circle1, mask_circle2)

    x3, y3 = 64, 34
    r3 = 16
    mask_circle3 = (x - x3) ** 2 + (y - y3) ** 2 < r3**2
    foreground = np.logical_or(foreground, mask_circle3)

    distance = edt.edt(foreground).astype(np.float32)

    foreground = foreground[None, ...]
    distance = distance[None, ...]

    contour = -distance
    contour = contour - contour.min()
    contour = contour.astype(dtype)

    start = time.time()
    components = compute_segmentation_hypotheses(
        foreground=foreground,
        contours=contour,
        min_num_pixels=0,
        max_num_pixels=1_000_000,
        min_frontier=0.0,
    )
    end = time.time()
    print("Time taken:", end - start)

    assert len(components) == 5
    _validate_hierarchy(components)

    overlap_dict = overlap_dict_from_segments(components)
    print(overlap_dict)
    _validate_overlap_dict(overlap_dict)

    if interactive:
        viewer = napari.Viewer()
        viewer.add_image(contour, colormap="magma")
        viewer.add_labels(foreground, opacity=0.5, visible=False)

        for component in components:
            # print("-" * 100)
            print(component.num_pixels)
            viewer.add_labels(
                component.mask * component.id,
                translate=component.bbox[:3],
            )
        napari.run()


if __name__ == "__main__":
    test_multi_hypotheses_overlap(interactive=True)
