#pragma once

#include <unordered_map>
#include <vector>
#include <algorithm>
#include <numeric>
#include "bimap.h"
#include "counting_map.h"
#include "segment.h"
#include "tree.h"
#include "union_find.h"

namespace nb = nanobind;

using namespace nb::literals;


template <typename T>
std::vector<size_t> argsort(const std::vector<T> &array)
{
    std::vector<size_t> indices(array.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(),
              [&array](size_t left, size_t right) -> bool {
                  return array[left] < array[right];
              });
    return indices;
}


template <typename WeightType>
int _update_minima(
    int i,
    WeightType parent_weight,
    std::vector<int> &minima,
    BinaryTree<WeightType> &tree
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
    int width,
    long *id_offset_ptr
) {
    std::vector<size_t> sorted_indices = argsort(weights);

    std::vector<int> c_to_tree_idx(visited.size());
    std::iota(c_to_tree_idx.begin(), c_to_tree_idx.end(), 0);

    BiMap bimap(visited);
    std::vector<int> local_edges = bimap.apply_backward(edges);

    int num_segments = 0;
    int num_leaves = visited.size();
    int num_nodes = 2 * visited.size() - 1;

    BinaryTree<float> tree(visited.size());
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

    // Tolerance for comparing floating-point weights
    // Since weights are averaged from contour data, we need a reasonable tolerance
    constexpr float WEIGHT_TOLERANCE = 1e-6f;

    // eliminating difference between flat zones
    for (int i = num_leaves; i < num_nodes - 1; i++)
    { // skipping root on purpose
        if (fabs(tree.weight(i) - tree.weight(tree.parent(i))) <= WEIGHT_TOLERANCE)
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
    BinaryTree<int> area_tree(visited.size());
    UnionFind<true> uf_tracked(visited.size());  // Use component tracking for O(1) get_component
    std::iota(c_to_tree_idx.begin(), c_to_tree_idx.end(), 0);

    CountingMap<long> id_map(*id_offset_ptr);

    for (size_t i = 0; i < sorted_indices.size(); i++)
    {
        int idx = sorted_indices[i];
        int parent_weight = sliced_areas[idx];

        int u = mst_edges[idx * 2];
        int v = mst_edges[idx * 2 + 1];

        int c_u = uf_tracked.find(u);
        int c_v = uf_tracked.find(v);
        if (c_u == c_v) continue;

        int size_u = uf_tracked.get_size(c_u);
        int size_v = uf_tracked.get_size(c_v);

        int t_u = c_to_tree_idx[c_u];
        int t_v = c_to_tree_idx[c_v];

        int min_u = _update_minima(t_u, parent_weight, minima, area_tree);
        int min_v = _update_minima(t_v, parent_weight, minima, area_tree);

        int size_new = size_u + size_v;
        int t_new = area_tree.add_node(t_u, t_v, parent_weight);

        minima[t_new] = min_u + min_v;

        bool is_watershed = min_u > 0 && min_v > 0;

        if (is_watershed)
        {
            bool merge_exceeds_size = size_new >= max_num_pixels && (size_u < max_num_pixels || size_v < max_num_pixels);

            // if merge is allowed, we skip adding children segments
            // merge must not exceed size constraints to be allowed
            if (!(mst_weights[idx] < min_frontier && !merge_exceeds_size))
            {
                for (auto [c, size] : { std::pair{c_u, size_u}, std::pair{c_v, size_v} })
                {
                    if (size >= min_num_pixels && size < max_num_pixels)
                    {
                        int t_id = c_to_tree_idx[c];

                        // Apply forward mapping directly to the list to avoid intermediate vector allocation
                        const auto& component_list = uf_tracked.get_component_list(c);
                        std::vector<int> component = bimap.apply_forward(component_list);

                        segments.push_back(
                            Segment::from_visited(
                                component, depth, height, width,
                                id_map.get(t_id), id_map.get(t_new)
                            )
                        );

                        num_segments++;
                    }
                }
            }
        }

        // finishing merging the two components
        int c_new = uf_tracked.unite(c_u, c_v);
        c_to_tree_idx[c_new] = t_new;
    }

    if (visited.size() < max_num_pixels || num_segments == 0)
    {
        segments.push_back(
            Segment::from_visited(
                visited, depth, height, width,
                id_map.get(num_nodes - 1), id_map.get(num_nodes - 1)
            )
        );
        num_segments++;
    }

    *id_offset_ptr = id_map.next_value();

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
    int cur_idx,
    long *id_offset_ptr
) {

    seen_data[cur_idx] = true;
    std::vector<int> queue = {cur_idx};
    std::vector<int> visited;

    std::vector<int> edges;
    std::vector<float> weights;

    constexpr std::array<int, 18> offsets = {
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

    hierarchical_watershed(
        segments, visited, edges, weights,
        min_num_pixels, max_num_pixels, min_frontier,
        depth, height, width, id_offset_ptr
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
    long id_offset = 1;

    for (int z = 0; z < depth; z++) {
        int z_step = z * height * width;
        for (int y = 0; y < height; y++) {
            int y_step = y * width;
            for (int x = 0; x < width; x++) {
                int idx = z_step + y_step + x;
                if (fg_data[idx] && !seen_data[idx])
                {
                    compute_connected_components(
                        segments, fg_data, ctr_data, seen_data,
                        depth, height, width,
                        min_num_pixels, max_num_pixels, min_frontier, idx, &id_offset
                    );

                }
            }
        }
    }

    delete[] seen_data;
    return segments;
}


std::unordered_map<long, std::vector<long>> overlap_dict_from_segments(
    const std::vector<Segment> &segments
) {

    std::unordered_map<long, const Segment *> segment_dict;
    for (const Segment &segment : segments) {
        segment_dict[segment.id] = &segment;
    }

    std::unordered_map<long, std::vector<long>> overlap_dict;

    for (const Segment &segment : segments) {
        std::vector<long> overlap_ids;
        long current_id = segment.id;
        const Segment *current = &segment;
        while ((current->parent_id != current->id) &&
               (segment_dict.find(current->parent_id) != segment_dict.end()))
        {
            overlap_ids.push_back(current->parent_id);
            current = segment_dict[current->parent_id];
        }
        overlap_dict.insert({current_id, overlap_ids});
    }

    return overlap_dict;
}
