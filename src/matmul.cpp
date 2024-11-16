#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include "matmul.h"

#define CHECK_CUDA(x) \
  TORCH_CHECK(x.is_cuda(), #x, " must be a CUDAtensor ")
#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x, " must be contiguous ")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

torch::Tensor matmul(const torch::Tensor a, const torch::Tensor b)
{
    CHECK_INPUT(a);
    CHECK_INPUT(b);

    unsigned m = a.size(0), n = b.size(1), k = b.size(0);
    torch::Tensor c = torch::zeros({m, n}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    float *p_c = c.data_ptr<float>();
    float *p_a = a.data_ptr<float>();
    float *p_b = b.data_ptr<float>();
    matmul_launcher(p_c, p_a, p_b, m, n, k);
    return c;
}

PYBIND11_MODULE(torch_ops_matmul, m)
{
    m.def("forward", matmul);
}

TORCH_LIBRARY(torch_ops_matmul, m)
{
    m.def("forward", matmul);
}
