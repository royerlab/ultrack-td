#include <vector>
#include <cstring>
#include <nanobind/ndarray.h>
#include <nanobind/nanobind.h>

namespace nb = nanobind;

using namespace nb::literals;

void inspect(const nb::ndarray<>& a);

struct Segment {
    nb::ndarray<bool> mask;
    nb::ndarray<int> bbox;
    int num_pixels;
    int z;
    int y;
    int x;
};


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

    auto seen = nb::ndarray<nb::numpy, bool, nb::ndim<2>>(seen_data, {depth, height, width});

    std::vector<Segment> segments;

    for (int z = 0; z < depth; z++) {
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                if (fg_data[z * height * width + y * width + x]) {
                    //  segments.push_back(Segment(foreground, z, y, x));
                }
            }
        }
    }

    return segments;
}
