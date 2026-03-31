#include <cuda_runtime.h>
__global__ void resize_kernel(const unsigned char *in, unsigned char *out, int in_w, int in_h, int out_w, int out_h, int c)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= out_w || y >= out_h)
        return;
    float scale_x = (float)in_w / out_w;
    float scale_y = (float)in_h / out_h;
    int src_x = (int)(x * scale_x);
    int src_y = (int)(y * scale_y);
    for (int ch = 0; ch < c; ++ch)
    {
        out[(y * out_w + x) * c + ch] = in[(src_y * in_w + src_x) * c + ch];
    }
}
void resize_cuda(const unsigned char *in, unsigned char *out, int in_w, int in_h, int out_w, int out_h, int channels)
{
    size_t in_bytes = in_w * in_h * channels;
    size_t out_bytes = out_w * out_h * channels;
    unsigned char *d_in = nullptr, *d_out = nullptr;
    cudaMalloc(&d_in, in_bytes);
    cudaMalloc(&d_out, out_bytes);
    cudaMemcpy(d_in, in, in_bytes, cudaMemcpyHostToDevice);
    dim3 block(16, 16);
    dim3 grid((out_w + 15) / 16, (out_h + 15) / 16);
    resize_kernel<<<grid, block>>>(d_in, d_out, in_w, in_h, out_w, out_h, channels);
    cudaDeviceSynchronize();
    cudaMemcpy(out, d_out, out_bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_in);
    cudaFree(d_out);
}