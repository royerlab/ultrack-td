#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include "cpp/ultrack.h"

namespace nb = nanobind;

using namespace nb::literals;

NB_MODULE(ultrack_td_ext, m) {
    m.doc() = "This is a \"hello world\" example with nanobind";
    m.def("add", [](int a, int b) { return a + b; }, "a"_a, "b"_a);
    m.def("inspect", inspect, "a"_a);
}
