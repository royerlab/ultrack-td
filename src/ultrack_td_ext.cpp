#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

namespace nb = nanobind;

using namespace nb::literals;

NB_MODULE(ultrack_td_ext, m) {
    m.doc() = "This is a \"hello world\" example with nanobind";
    m.def("add", [](int a, int b) { return a + b; }, "a"_a, "b"_a);
    m.def("inspect", [](const nb::ndarray<>& a) {
        printf("Array data pointer : %p\n", a.data());
        printf("Array dimension : %zu\n", a.ndim());
        for (size_t i = 0; i < a.ndim(); ++i) {
            printf("Array dimension [%zu] : %zu\n", i, a.shape(i));
            printf("Array stride    [%zu] : %zd\n", i, a.stride(i));
        }
        printf("Device ID = %u (cpu=%i, cuda=%i)\n", a.device_id(),
            int(a.device_type() == nb::device::cpu::value),
            int(a.device_type() == nb::device::cuda::value)
        );
        printf("Array dtype: int16=%i, uint32=%i, float32=%i\n",
            a.dtype() == nb::dtype<int16_t>(),
            a.dtype() == nb::dtype<uint32_t>(),
            a.dtype() == nb::dtype<float>()
        );
    });
}
