#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include "matmul.h"

void matmul(torch::Tensor c, torch::Tensor a, torch::Tensor b) {
    float *p_c = c.data_ptr<float>();
    float *p_a = a.data_ptr<float>();
    float *p_b = b.data_ptr<float>();
    int n = c.size(0);
    matmul_launcher(p_c, p_a, p_b, n);
}

PYBIND11_MODULE(torch_ops_matmul, m) {
    m.def("forward", matmul);
}

TORCH_LIBRARY(torch_ops_matmul, m) {
    m.def("forward", matmul);
}