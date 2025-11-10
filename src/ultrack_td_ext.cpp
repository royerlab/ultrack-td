#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include "cpp/ultrack.h"

namespace nb = nanobind;

using namespace nb::literals;

NB_MODULE(ultrack_td_ext, m) {
    m.doc() = "This is a \"hello world\" example with nanobind";
    m.def("inspect", inspect, "a"_a);
    m.def("compute_segmentation_hypotheses", compute_segmentation_hypotheses<float>, "foreground"_a, "contours"_a, "min_num_pixels"_a, "max_num_pixels"_a, "min_frontier"_a);
    // TODO other types
}
