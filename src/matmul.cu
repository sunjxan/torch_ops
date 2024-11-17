#include <pybind11/pybind11.h>
#include <torch/extension.h>

#define CHECK_CUDA(x) \
  TORCH_CHECK(x.is_cuda(), #x, " must be a CUDAtensor ")
#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x, " must be contiguous ")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

#define DIVUP(m, n) ((m + n - 1) / n)

template<typename T>
__global__ void matmul_kernel(T *c, const T *a, const T *b, unsigned m, unsigned n, unsigned k)
{
    unsigned ix = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix < m && iy < n) {
        T sum = 0;
        for (unsigned t = 0; t < k; ++t) {
            sum += a[ix * k + t] * b[t * n + iy];
        }
        c[ix * n + iy] = sum;
    }
}

template<typename T>
void matmul_launcher(T *c, const T *a, const T *b, unsigned m, unsigned n, unsigned k)
{
    dim3 block_size(32, 32);
    dim3 grid_size(DIVUP(m, block_size.x), DIVUP(n, block_size.y));
    matmul_kernel<<<grid_size, block_size>>>(c, a, b, m, n, k);
}

at::Tensor matmul(const at::Tensor a, const at::Tensor b)
{
    CHECK_INPUT(a);
    CHECK_INPUT(b);

    unsigned m = a.size(0), n = b.size(1), k = b.size(0);
    at::Tensor c = at::zeros({m, n}, at::dtype(a.scalar_type()).device(torch::kCUDA));

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(a.scalar_type(), "matmul operation", ([&] {
        const scalar_t *p_a = a.data_ptr<scalar_t>();
        const scalar_t *p_b = b.data_ptr<scalar_t>();
        scalar_t *p_c = c.data_ptr<scalar_t>();
        matmul_launcher(p_c, p_a, p_b, m, n, k);
    }));

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
