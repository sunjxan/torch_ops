#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include "matmul.h"

torch::Tensor forward(const torch::Tensor a, const torch::Tensor b) {
    unsigned m = a.size(0), n = b.size(1), k = b.size(0);
    torch::Tensor c = torch::zeros({m, n}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    float *p_c = c.data_ptr<float>();
    float *p_a = a.data_ptr<float>();
    float *p_b = b.data_ptr<float>();
    matmul_launcher(p_c, p_a, p_b, m, n, k);
    return c;
}

torch::Tensor backward_left(const torch::Tensor a, const torch::Tensor b) {
    unsigned m = a.size(0), n = b.size(1), k = b.size(0);
    torch::Tensor c = torch::zeros({m, k}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    for (unsigned i = 0; i < k; ++i) {
        float sum = 0;
        for (unsigned j = 0; j < n; ++j) {
            sum += b[i][j].item<float>();
        }
        for (unsigned j = 0; j < m; ++j) {
            c[j][i] = sum;
        }
    }
    return c;
}

torch::Tensor backward_right(const torch::Tensor a, const torch::Tensor b) {
    unsigned m = a.size(0), n = b.size(1), k = b.size(0);
    torch::Tensor c = torch::zeros({k, n}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    for (unsigned i = 0; i < k; ++i) {
        float sum = 0;
        for (unsigned j = 0; j < m; ++j) {
            sum += a[j][i].item<float>();
        }
        for (unsigned j = 0; j < n; ++j) {
            c[i][j] = sum;
        }
    }
    return c;
}

PYBIND11_MODULE(torch_ops_matmul, m) {
    m.def("forward", forward);
    m.def("backward_left", backward_left);
    m.def("backward_right", backward_right);
}

TORCH_LIBRARY(torch_ops_matmul, m) {
    m.def("forward", forward);
    m.def("backward_left", backward_left);
    m.def("backward_right", backward_right);
}
