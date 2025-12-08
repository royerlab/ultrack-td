#include <cfloat>
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
    int id;
    int parent_id;

    static Segment from_visited_and_bbox(
        const std::vector<int>& visited,
        int min_z, int min_y, int min_x,
        int max_z, int max_y, int max_x,
        int depth, int height, int width,
        int id = -1, int parent_id = -1
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
            .id = id,
            .parent_id = parent_id,
        };
    }

    static Segment from_visited(
        const std::vector<int> &visited,
        int depth, int height, int width,
        int id = -1, int parent_id = -1
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
            depth, height, width,
            id, parent_id
        );
    }
};


template <typename T>
std::vector<size_t> argsort(const std::vector<T> &array)
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
    float parent_weight,
    std::vector<int> &minima,
    BinaryTree &tree
) {
    /*
    Assigns minima depending on the change in weight between the current node and its parent.
    */
    if (i < tree.num_leaves)
        return 0;

    if (minima[i] == 0) {
        minima[i] = (tree.weight(i) < parent_weight) ? 1 : 0;
    }
    return minima[i];
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
    std::vector<int> local_edges = bimap.apply_backward(edges);

    int num_segments = 0;
    int num_leaves = visited.size();
    int num_nodes = 2 * visited.size() - 1;

    BinaryTree tree(visited.size());
    UnionFind uf(visited.size());

    std::vector<int> areas(num_nodes, 0);
    std::fill(areas.begin(), areas.begin() + num_leaves, 1);
    
    std::vector<int> mst_edges(num_nodes - 1, -1);
    std::vector<float> mst_weights(num_nodes - num_leaves, 0.0f);

    for (size_t i = 0; i < sorted_indices.size(); i++)
    {
        int idx = sorted_indices[i];
        int u = local_edges[idx * 2];
        int v = local_edges[idx * 2 + 1];

        int c_u = uf.find(u);
        int c_v = uf.find(v);
        if (c_u == c_v) continue;

        int c_new = uf.unite(c_u, c_v);

        int t_u = c_to_tree_idx[c_u];
        int t_v = c_to_tree_idx[c_v];

        int t_new = tree.add_node(t_u, t_v, weights[idx]);
        c_to_tree_idx[c_new] = t_new;

        areas[t_new] = areas[t_u] + areas[t_v];

        int mst_idx = t_new - visited.size();
        mst_edges[2 * mst_idx] = u;
        mst_edges[2 * mst_idx + 1] = v;

        mst_weights[mst_idx] = weights[idx]; // TODO: is this really necessary
    }


    std::fill(areas.begin(), areas.begin() + visited.size(), 1);

    for (int i = num_leaves; i < num_nodes - 1; i++)
    { // skipping root on purpose
        if (fabs(tree.weight(i) - tree.weight(tree.parent(i))) < FLT_EPSILON)
        {
            areas[i] = std::max(
                areas[tree.left_child(i)],
                areas[tree.right_child(i)]
            );
        }
    }


    // mst edges are the minimum of the attributes of the two children
    for (int i = num_nodes - 1; i >= num_leaves; i--)
    {
        areas[i] = std::min(areas[tree.left_child(i)], areas[tree.right_child(i)]);
    }

    std::vector<int> sliced_areas(areas.begin() + visited.size(), areas.end());

    sorted_indices = argsort(sliced_areas);
    std::vector<int> minima(num_nodes, 0);

    // resetting data structures for the second pass
    tree = BinaryTree(visited.size());
    uf = UnionFind(visited.size());
    std::iota(c_to_tree_idx.begin(), c_to_tree_idx.end(), 0);

    for (size_t i = 0; i < sorted_indices.size(); i++)
    {
        int idx = sorted_indices[i];
        int u = mst_edges[idx * 2];
        int v = mst_edges[idx * 2 + 1];

        int c_u = uf.find(u);
        int c_v = uf.find(v);
        if (c_u == c_v) continue;

        int size_u = uf.get_size(c_u);
        int size_v = uf.get_size(c_v);

        int t_u = c_to_tree_idx[c_u];
        int t_v = c_to_tree_idx[c_v];

        float parent_weight = sliced_areas[idx];
        int min_u = _update_minima(t_u, parent_weight, minima, tree);
        int min_v = _update_minima(t_v, parent_weight, minima, tree);

        bool is_watershed = min_u > 0 && min_v > 0;

        int size_new = size_u + size_v;
        int t_new = tree.add_node(t_u, t_v, parent_weight);

        minima[t_new] = min_u + min_v;

        if (is_watershed)
        {
            bool merge_exceeds_size = size_new >= max_num_pixels && (size_u < max_num_pixels || size_v < max_num_pixels);

            if (mst_weights[idx] < min_frontier && !merge_exceeds_size) continue;

            if (size_u >= min_num_pixels && size_u < max_num_pixels &&
                size_v >= min_num_pixels && size_v < max_num_pixels)
            {
                for (int c : {c_u, c_v})
                {
                    int t_id = c_to_tree_idx[c];
                    std::vector<int> local_component = uf.get_component(c);
                    std::vector<int> component = bimap.apply_forward(local_component);
                    segments.push_back(
                        Segment::from_visited(
                            component, depth, height, width,
                            t_id, t_new
                        )
                    );

                    num_segments++;
                }
            }
        }

        // finishing merging the two components
        int c_new = uf.unite(c_u, c_v);
        c_to_tree_idx[c_new] = t_new;
    }

    if (visited.size() < max_num_pixels || num_segments == 0)
    {
        segments.push_back(
            Segment::from_visited(
                visited, depth, height, width,
                num_nodes - 1, num_nodes - 1
            )
        );
        num_segments++;
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
    seen_data[cur_idx] = true;
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
                if (fg_data[nidx])
                {
                    if (!seen_data[nidx]) { // avoiding adding the same pixel twice
                        seen_data[nidx] = true;
                        queue.push_back(nidx);
                    }

                    if (idx < nidx) // making sure we don't add the same edge twice
                    {
                        edges.push_back(idx);
                        edges.push_back(nidx);

                        float w = 0.5f * (ctr_data[idx] + ctr_data[nidx]);
                        weights.push_back(w);
                    }
                }
            }
        }
    }

    int num_segments = hierarchical_watershed(
        segments, visited, edges, weights,
        min_num_pixels, max_num_pixels, min_frontier,
        depth, height, width
    );
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
    int id_offset = 0;

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
                    int max_id = 0;
                    for (Segment &segment : segments) {
                        segment.id += id_offset;
                        segment.parent_id += id_offset;
                        max_id = std::max(max_id, segment.id);
                    }
                    id_offset = max_id + 1;
                }
            }
        }
    }

    delete[] seen_data;
    return segments;
}
