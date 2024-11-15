__global__ void matmul_kernel(float *c, float *a, float *b, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

void matmul_launcher(float *c, float *a, float *b, int n)
{
    matmul_kernel<<<(n+127)/128, 128>>>(c, a, b, n);
}