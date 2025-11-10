#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>
#include "cpp/ultrack.h"

namespace nb = nanobind;

using namespace nb::literals;

NB_MODULE(ultrack_td_ext, m) {
    m.doc() = "This is a \"hello world\" example with nanobind";
    m.def("inspect", inspect, "a"_a);

    nb::class_<Segment>(m, "Segment")
    .def(nb::init<nb::ndarray<bool>, nb::ndarray<int>, int, int, int>())
    .def_ro("mask", &Segment::mask)
    .def_ro("bbox", &Segment::bbox)
    .def_ro("num_pixels", &Segment::num_pixels)
    .def_ro("z", &Segment::z)
    .def_ro("y", &Segment::y)
    .def_ro("x", &Segment::x);

    m.def("compute_segmentation_hypotheses", compute_segmentation_hypotheses<float>, "foreground"_a, "contours"_a, "min_num_pixels"_a, "max_num_pixels"_a, "min_frontier"_a);
    // TODO other types
}
