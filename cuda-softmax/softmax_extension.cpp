#include <torch/extension.h>

at::Tensor softmax_cuda(at::Tensor input);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("softmax_cuda", &softmax_cuda,
        "Row-wise softmax using kernels from softmax.cu");
}
