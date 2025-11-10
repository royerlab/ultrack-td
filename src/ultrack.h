#include <vector>
#include <algorithm>
#include <cstring>
#include <numeric>
#include <nanobind/ndarray.h>
#include <nanobind/nanobind.h>
#include "union_find.h"

namespace nb = nanobind;

using namespace nb::literals;

struct Segment {
    nb::ndarray<nb::numpy, bool> mask;
    nb::ndarray<nb::numpy, int> bbox;
    int num_pixels;
    int z;
    int y;
    int x;

    static Segment from_visited_and_bbox(
        const std::vector<int>& visited,
        int min_z, int min_y, int min_x,
        int max_z, int max_y, int max_x,
        int depth, int height, int width
    ) {
        size_t mask_depth = max_z - min_z + 1;
        size_t mask_height = max_y - min_y + 1;
        size_t mask_width = max_x - min_x + 1;

        bool *mask_data = new bool[mask_depth * mask_height * mask_width];
        std::memset(mask_data, 0, mask_depth * mask_height * mask_width * sizeof(bool));
        for (int idx : visited) {
            int z = idx / (height * width) - min_z;
            int y = (idx % (height * width)) / width - min_y;
            int x = idx % width - min_x;
            mask_data[z * mask_height * mask_width + y * mask_width + x] = true;
        }

        size_t shape[3] = {mask_depth, mask_height, mask_width};
        nb::capsule mask_owner(mask_data, [](void *p) noexcept {
            delete[] (bool *) p;
        });
        auto mask = nb::ndarray<nb::numpy, bool>(mask_data, 3, shape, mask_owner);

        int *bbox_data = new int[6]{min_z, min_y, min_x, max_z, max_y, max_x};
        size_t bbox_shape[1] = {6};
        nb::capsule bbox_owner(bbox_data, [](void *p) noexcept {
            delete[] (int *) p;
        });
        auto bbox = nb::ndarray<nb::numpy, int>(bbox_data, 1, bbox_shape, bbox_owner);

        return Segment{
            .mask = mask,
            .bbox = bbox,
            .num_pixels = static_cast<int>(visited.size()),
            .z = min_z,
            .y = min_y,
            .x = min_x,
        };
    }

    static Segment from_visited(
        const std::vector<int> &visited,
        int depth, int height, int width
    ) {
        int min_z = depth - 1;
        int min_y = height - 1;
        int min_x = width - 1;
        int max_z = 0;
        int max_y = 0;
        int max_x = 0;
        for (int idx : visited) {
            int z = idx / (height * width);
            int y = (idx % (height * width)) / width;
            int x = idx % width;
            min_z = std::min(min_z, z);
            min_y = std::min(min_y, y);
            min_x = std::min(min_x, x);
            max_z = std::max(max_z, z);
            max_y = std::max(max_y, y);
            max_x = std::max(max_x, x);
        }
        return Segment::from_visited_and_bbox(
            visited, min_z, min_y, min_x,
            max_z, max_y, max_x,
            depth, height, width
        );
    }
};


std::vector<size_t> argsort(const std::vector<float> &array)
{
    std::vector<size_t> indices(array.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(),
              [&array](int left, int right) -> bool {
                  return array[left] < array[right];
              });

    return indices;
}


int hierarchical_watershed(
    std::vector<Segment> &segments,
    const std::vector<int> &visited,
    const std::vector<int> &edges,
    const std::vector<float> &weights,
    int min_num_pixels,
    int max_num_pixels,
    float min_frontier,
    int depth,
    int height,
    int width
) {
    std::vector<size_t> sorted_indices = argsort(weights);

    int num_segments = 0;
    UnionFind uf(visited);

    for (size_t i = 0; i < sorted_indices.size(); i++)
    {
        int idx = sorted_indices[i];
        int u = edges[idx * 2];
        int v = edges[idx * 2 + 1];
        bool is_new = uf.unite(u, v);
        if (is_new && weights[idx] > min_frontier)
        {
            int size = uf.get_size(u);
            if (size > min_num_pixels && size < max_num_pixels)
            {
                segments.push_back(
                    Segment::from_visited(
                        uf.get_component(u), depth, height, width
                    )
                );
                num_segments++;
            }
        }
    }
    return num_segments;
}


template <typename T>
void compute_connected_components(
    std::vector<Segment> &segments,
    const bool *fg_data,
    const T *ctr_data,
    bool *seen_data,
    int depth,
    int height,
    int width,
    int min_num_pixels,
    int max_num_pixels,
    float min_frontier,
    int cur_idx
) {
    std::vector<int> queue = {cur_idx};
    std::vector<int> visited;

    std::vector<int> edges;
    std::vector<float> weights;

    int offsets[18] = {
        0, 0, 1,
        0, 1, 0,
        1, 0, 0,
        0, -1, 0,
        0, 0, -1,
        -1, 0, 0,
    };

    int min_z = depth - 1;
    int min_y = height - 1;
    int min_x = width - 1;
    int max_z = 0;
    int max_y = 0;
    int max_x = 0;

    while (!queue.empty())
    {
        int idx = queue.back();
        queue.pop_back();
        seen_data[idx] = true;
        visited.push_back(idx);

        int cur_z = idx / (height * width);
        int cur_y = (idx % (height * width)) / width;
        int cur_x = idx % width;

        min_z = std::min(min_z, cur_z);
        min_y = std::min(min_y, cur_y);
        min_x = std::min(min_x, cur_x);
        max_z = std::max(max_z, cur_z);
        max_y = std::max(max_y, cur_y);
        max_x = std::max(max_x, cur_x);
        for (int i = 0; i < 6; i++) {
            int nz = cur_z + offsets[i * 3];
            int ny = cur_y + offsets[i * 3 + 1];
            int nx = cur_x + offsets[i * 3 + 2];
            if (
                nz >= 0 && nz < depth &&
                ny >= 0 && ny < height &&
                nx >= 0 && nx < width
            ) {
                int nidx = nz * height * width + ny * width + nx;
                if (fg_data[nidx] && !seen_data[nidx]) {
                    seen_data[nidx] = true;
                    queue.push_back(nidx);

                    edges.push_back(idx);
                    edges.push_back(nidx);

                    float w = 0.5f * (ctr_data[idx] + ctr_data[nidx]);
                    weights.push_back(w);
                }
            }
        }
    }

    int num_segments = hierarchical_watershed(
        segments, visited, edges, weights,
        min_num_pixels, max_num_pixels, min_frontier,
        depth, height, width
    );

    if (num_segments == 0) {
        segments.push_back(
            Segment::from_visited_and_bbox(
                visited, min_z, min_y, min_x,
                max_z, max_y, max_x, depth, height, width
            )
        );
    }
}


template <typename T>
std::vector<Segment> compute_segmentation_hypotheses(
    const nb::ndarray<bool>& foreground,
    const nb::ndarray<T>& contours,
    int min_num_pixels,
    int max_num_pixels,
    float min_frontier
) {
    size_t depth = foreground.shape(0);
    size_t height = foreground.shape(1);
    size_t width = foreground.shape(2);

    bool *seen_data = new bool[depth * height * width];
    std::memset(seen_data, 0, depth * height * width * sizeof(bool));

    bool *fg_data = foreground.data();
    T *ctr_data = contours.data();

    std::vector<Segment> segments;

    for (int z = 0; z < depth; z++) {
        int z_step = z * height * width;
        for (int y = 0; y < height; y++) {
            int y_step = y * width;
            for (int x = 0; x < width; x++) {
                int idx = z_step + y_step + x;
                if (fg_data[idx] && !seen_data[idx]) {
                    compute_connected_components(
                        segments, fg_data, ctr_data, seen_data,
                        depth, height, width,
                        min_num_pixels, max_num_pixels, min_frontier, idx
                    );
                }
            }
        }
    }

    delete[] seen_data;
    return segments;
}
