#define DIVUP(m, n) ((m + n - 1) / n)

__global__ void matmul_kernel(float* c, const float* a, const float* b, unsigned m, unsigned n, unsigned k)
{
    unsigned ix = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix < m && iy < n) {
        float sum = 0;
        for (unsigned t = 0; t < k; ++t) {
            sum += a[ix * k + t] * b[t * n + iy];
        }
        c[ix * n + iy] = sum;
    }
}

void matmul_launcher(float* c, const float* a, const float* b, unsigned m, unsigned n, unsigned k)
{
    dim3 block_size(32, 32);
    dim3 grid_size(DIVUP(m, block_size.x), DIVUP(n, block_size.y));
    matmul_kernel<<<grid_size, block_size>>>(c, a, b, m, n, k);
}
