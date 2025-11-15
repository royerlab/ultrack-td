#include <vector>
#include <algorithm>
#include <cstring>
#include <numeric>
#include <iostream>
#include <nanobind/ndarray.h>
#include <nanobind/nanobind.h>
#include "bimap.h"
#include "tree.h"
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


int _update_minima(
    int i,
    std::vector<int> &minima,
    BinaryTree &tree
) {
    /*
    Assigns minima depending on the change in weight between the current node and its parent.
    */
    if (i < tree.num_leaves)
        return 0;

    if (minima.at(i) == 0) {
        int p = tree.parent(i);
        minima.at(i) = (tree.weight(i) < tree.weight(p)) ? 1 : 0;
    }
    return minima.at(i);
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

    std::vector<int> c_to_tree_idx(visited.size());
    std::iota(c_to_tree_idx.begin(), c_to_tree_idx.end(), 0);

    BiMap bimap(visited);
    BinaryTree tree(visited.size());
    std::vector<int> minima(2 * visited.size() - 1, 0);

    std::vector<int> local_edges = bimap.apply_backward(edges);

    int num_segments = 0;
    UnionFind uf(visited.size());

    for (size_t i = 0; i < sorted_indices.size(); i++)
    {
        int idx = sorted_indices[i];
        int u = local_edges[idx * 2];
        int v = local_edges[idx * 2 + 1];

        int c_u = uf.find(u);
        int c_v = uf.find(v);
        if (c_u == c_v) continue;

        int size_u = uf.get_size(c_u);
        int size_v = uf.get_size(c_v);

        int c_new = uf.unite(c_u, c_v);

        int t_u = c_to_tree_idx.at(c_u);
        int t_v = c_to_tree_idx.at(c_v);

        int t_new = tree.add_node(t_u, t_v, weights[idx]);
        c_to_tree_idx.at(c_new) = t_new;

        // evaluating if it's a watershed
        int min_u = _update_minima(t_u, minima, tree);
        int min_v = _update_minima(t_v, minima, tree);

        minima.at(t_new) = min_u + min_v;
        if (i == sorted_indices.size() - 1 || (min_u > 0 && min_v > 0)) // it's a watershed
        {
            if (weights[idx] < min_frontier) continue;

            int size = uf.get_size(c_new);
            if (
                size_u > min_num_pixels && size_v > min_num_pixels &&
                size > min_num_pixels && size < max_num_pixels
            ) {
                std::vector<int> local_component = uf.get_component(c_new);
                std::vector<int> component = bimap.apply_forward(local_component);

                segments.push_back(
                    Segment::from_visited(
                        component, depth, height, width
                    )
                );
                num_segments++;
            }
        }
    }

    std::cout << num_segments << std::endl;

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
