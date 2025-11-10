#include <nanobind/ndarray.h>
#include <nanobind/nanobind.h>

namespace nb = nanobind;

using namespace nb::literals;

void inspect(const nb::ndarray<>& a);

template <typename T>
void compute_segmentation_hypotheses(
    const nb::ndarray<bool>& foreground,
    const nb::ndarray<T>& contours,
    int min_num_pixels,
    int max_num_pixels,
    float min_frontier
) {
    printf("Computing segmentation hypotheses...\n");
}