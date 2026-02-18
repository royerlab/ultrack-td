#pragma once

#include <vector>
#include <cstring>
#include <nanobind/ndarray.h>
#include <nanobind/nanobind.h>

namespace nb = nanobind;

using namespace nb::literals;

struct Segment {
    nb::ndarray<nb::numpy, bool> mask;
    nb::ndarray<nb::numpy, int> bbox;
    int num_pixels;
    float z;
    float y;
    float x;
    long id;
    long parent_id;

    static Segment from_visited_and_bbox(
        const std::vector<int> &visited,
        int min_z, int min_y, int min_x,
        int max_z, int max_y, int max_x,
        int depth, int height, int width,
        long id = -1, long parent_id = -1
    ) {
        size_t mask_depth = max_z - min_z + 1;
        size_t mask_height = max_y - min_y + 1;
        size_t mask_width = max_x - min_x + 1;
        size_t mask_yx_size = mask_height * mask_width;
        size_t yx_size = height * width;
        float avg_z = 0.0f;
        float avg_y = 0.0f;
        float avg_x = 0.0f;

        bool *mask_data = new bool[mask_depth * mask_height * mask_width];
        std::memset(mask_data, 0, mask_depth * mask_height * mask_width * sizeof(bool));
        for (int idx : visited) {
            int z = idx / yx_size;
            int y = (idx % yx_size) / width;
            int x = idx % width;
            mask_data[(z - min_z) * mask_yx_size + (y - min_y) * mask_width + (x - min_x)] = true;
            avg_z += z;
            avg_y += y;
            avg_x += x;
        }

        size_t num_pixels = visited.size();
        avg_z /= num_pixels;
        avg_y /= num_pixels;
        avg_x /= num_pixels;

        // Acquire GIL before creating ndarrays
        nb::gil_scoped_acquire acquire;

        size_t shape[3] = {mask_depth, mask_height, mask_width};
        nb::capsule mask_owner(mask_data, [](void *p) noexcept {
            delete[] (bool *) p;
        });
        auto mask = nb::ndarray<nb::numpy, bool>(mask_data, 3, shape, mask_owner);

        int *bbox_data = new int[6]{min_z, min_y, min_x, max_z + 1, max_y + 1, max_x + 1};
        size_t bbox_shape[1] = {6};
        nb::capsule bbox_owner(bbox_data, [](void *p) noexcept {
            delete[] (int *) p;
        });
        auto bbox = nb::ndarray<nb::numpy, int>(bbox_data, 1, bbox_shape, bbox_owner);

        return Segment{
            .mask = mask,
            .bbox = bbox,
            .num_pixels = static_cast<int>(num_pixels),
            .z = avg_z,
            .y = avg_y,
            .x = avg_x,
            .id = id,
            .parent_id = parent_id,
        };
    }

    static Segment from_visited(
        const std::vector<int> &visited,
        int depth, int height, int width,
        long id = -1, long parent_id = -1
    ) {
        size_t yx_size = height * width;
        int min_z = depth - 1;
        int min_y = height - 1;
        int min_x = width - 1;
        int max_z = 0;
        int max_y = 0;
        int max_x = 0;
        for (int idx : visited) {
            int z = idx / yx_size;
            int y = (idx % yx_size) / width;
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
