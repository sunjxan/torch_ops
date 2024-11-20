#include <pybind11/pybind11.h>
#include <torch/extension.h>

#define CHECK_CUDA(x) \
  TORCH_CHECK(x.is_cuda(), #x, " must be a CUDAtensor")
#define CHECK_2D(x) \
  TORCH_CHECK(x.dim() == 2, #x, " must be a matrix")
#define CHECK_DTYPE(x) \
  TORCH_CHECK(x.scalar_type() == at::ScalarType::Float || x.scalar_type() == at::ScalarType::Double || \
      x.scalar_type() == at::ScalarType::Half || x.scalar_type() == at::ScalarType::BFloat16, \
      #x, " dtype must be float32 or float64 or float16 or bfloat16")
#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x, " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_2D(x);         \
  CHECK_DTYPE(x);       \
  CHECK_CONTIGUOUS(x)
#define CHECK_INPUT_SAME_DTYPE(x, y) \
  TORCH_CHECK(x.scalar_type() == y.scalar_type(), #x, " and ", #y, " must be same dtype")

#define DIVUP(m, n) ((m + n - 1) / n)

template<typename T>
__global__ void forward_kernel(T *c, const T *a, const T *b, unsigned m, unsigned n, unsigned k)
{
    unsigned ix = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix < m && iy < n) {
        T sum = 0.0f;
        for (unsigned t = 0; t < k; ++t) {
            sum += a[ix * k + t] * b[t * n + iy];
        }
        c[ix * n + iy] = sum;
    }
}

template<typename T>
void forward_impl(T *c, const T *a, const T *b, unsigned m, unsigned n, unsigned k)
{
    dim3 block_size(32, 32);
    dim3 grid_size(DIVUP(m, block_size.x), DIVUP(n, block_size.y));
    forward_kernel<<<grid_size, block_size>>>(c, a, b, m, n, k);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error %d: %s\n", err, cudaGetErrorString(err));
    }
}

at::Tensor forward(const at::Tensor &a, const at::Tensor &b)
{
    CHECK_INPUT(a);
    CHECK_INPUT(b);
    CHECK_INPUT_SAME_DTYPE(a, b);

    unsigned m = a.size(0), n = b.size(1), k = b.size(0);
    at::ScalarType mat_dtype = a.scalar_type();
    at::Tensor c = at::zeros({m, n}, at::dtype(mat_dtype).device(at::kCUDA));

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, mat_dtype, "matmul operation forward", ([&] {
        const scalar_t *p_a = a.data_ptr<scalar_t>();
        const scalar_t *p_b = b.data_ptr<scalar_t>();
        scalar_t *p_c = c.data_ptr<scalar_t>();
        forward_impl(p_c, p_a, p_b, m, n, k);
    }));

    return c;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("forward", forward);
}

TORCH_LIBRARY(TORCH_EXTENSION_NAME, m)
{
    m.def("forward", forward);
}
