#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/unordered_map.h>
#include "ultrack.h"

namespace nb = nanobind;

using namespace nb::literals;

NB_MODULE(ultrack_td_ext, m) {
    nb::class_<Segment>(m, "Segment")
    .def(nb::init<nb::ndarray<nb::numpy, bool>, nb::ndarray<nb::numpy, int>, int, int, int>())
    .def_prop_ro("mask", [](const Segment &s) -> nb::ndarray<nb::numpy, bool> { return s.mask; }, nb::rv_policy::reference)
    .def_prop_ro("bbox", [](const Segment &s) -> nb::ndarray<nb::numpy, int> { return s.bbox; }, nb::rv_policy::reference)
    .def_ro("num_pixels", &Segment::num_pixels)
    .def_ro("z", &Segment::z)
    .def_ro("y", &Segment::y)
    .def_ro("x", &Segment::x)
    .def_ro("id", &Segment::id)
    .def_ro("parent_id", &Segment::parent_id);

    m.def("compute_segmentation_hypotheses_float", compute_segmentation_hypotheses<float>, "foreground"_a, "contours"_a, "min_num_pixels"_a, "max_num_pixels"_a, "min_frontier"_a);
    m.def("compute_segmentation_hypotheses_double", compute_segmentation_hypotheses<double>, "foreground"_a, "contours"_a, "min_num_pixels"_a, "max_num_pixels"_a, "min_frontier"_a);
    m.def("compute_segmentation_hypotheses_int_8", compute_segmentation_hypotheses<int8_t>, "foreground"_a, "contours"_a, "min_num_pixels"_a, "max_num_pixels"_a, "min_frontier"_a);
    m.def("compute_segmentation_hypotheses_int_16", compute_segmentation_hypotheses<int16_t>, "foreground"_a, "contours"_a, "min_num_pixels"_a, "max_num_pixels"_a, "min_frontier"_a);
    m.def("compute_segmentation_hypotheses_int_32", compute_segmentation_hypotheses<int32_t>, "foreground"_a, "contours"_a, "min_num_pixels"_a, "max_num_pixels"_a, "min_frontier"_a);
    m.def("compute_segmentation_hypotheses_int", compute_segmentation_hypotheses<int>, "foreground"_a, "contours"_a, "min_num_pixels"_a, "max_num_pixels"_a, "min_frontier"_a);
    m.def("compute_segmentation_hypotheses_uint_8", compute_segmentation_hypotheses<uint8_t>, "foreground"_a, "contours"_a, "min_num_pixels"_a, "max_num_pixels"_a, "min_frontier"_a);
    m.def("compute_segmentation_hypotheses_uint_16", compute_segmentation_hypotheses<uint16_t>, "foreground"_a, "contours"_a, "min_num_pixels"_a, "max_num_pixels"_a, "min_frontier"_a);
    m.def("compute_segmentation_hypotheses_uint_32", compute_segmentation_hypotheses<uint32_t>, "foreground"_a, "contours"_a, "min_num_pixels"_a, "max_num_pixels"_a, "min_frontier"_a);
    m.def("compute_segmentation_hypotheses_uint", compute_segmentation_hypotheses<unsigned int>, "foreground"_a, "contours"_a, "min_num_pixels"_a, "max_num_pixels"_a, "min_frontier"_a);

    m.def("overlap_dict_from_segments", overlap_dict_from_segments, "segments"_a);
}
