#include "winograd.cuh"

// F(2x2, 3x3) Winograd算法的变换矩阵
__constant__ float G[4][3] = {
    {1.0f, 0.0f, 0.0f}, 
    {0.5f, 0.5f, 0.5f}, 
    {0.5f, -0.5f, 0.5f}, 
    {0.0f, 0.0f, 1.0f}
};

__constant__ float B_T[4][4] = {
    {1.0f, 0.0f, -1.0f, 0.0f}, 
    {0.0f, 1.0f, 1.0f, 0.0f}, 
    {0.0f, -1.0f, 1.0f, 0.0f}, 
    {0.0f, 1.0f, 0.0f, -1.0f}
};

__constant__ float B[4][4] = {
    {1.0f,  0.0f,  0.0f,  0.0f}, 
    {0.0f,  1.0f, -1.0f,  1.0f}, 
    {-1.0f, 1.0f,  1.0f,  0.0f}, 
    {0.0f,  0.0f,  0.0f, -1.0f}
};

__constant__ float A_T[2][4] = {
    {1.0f, 1.0f, 1.0f, 0.0f}, 
    {0.0f, 1.0f, -1.0f, -1.0f}
};

// Step 4: 寄存器优化的Winograd卷积核函数
// 使用 __launch_bounds__ 优化寄存器分配和线程占用率
__global__ __launch_bounds__(256, 4)
void winograd_conv_kernel(const float* __restrict__ image,
                          const float* __restrict__ filter,
                          float* __restrict__ output,
                          int N, int C, int H, int W, int K, int outH, int outW) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num_tiles = N * K * (outH / 2) * (outW / 2);
    if (idx >= num_tiles) return;

    // 将线程索引分解为(n, k, tile_y, tile_x)
    int p_local = idx % ((outH / 2) * (outW / 2));
    int k = (idx / ((outH / 2) * (outW / 2))) % K;
    int n = idx / (K * (outH / 2) * (outW / 2));
    int tile_y = p_local / (outW / 2);
    int tile_x = p_local % (outW / 2);

    float m[4][4] = {{0.0f}};

    // 滤波器预取指针优化
    const float* filter_base = filter + k * C * 9;

    // 使用优化的内存访问模式遍历输入通道
    for (int c = 0; c < C; ++c) {
        // === 优化的滤波器变换 ===
        const float* g = filter_base + c * 9;
        float u_kc[4][4];
        
        // 将滤波器权重加载到寄存器中
        float g0 = g[0], g1 = g[1], g2 = g[2];
        float g3 = g[3], g4 = g[4], g5 = g[5];
        float g6 = g[6], g7 = g[7], g8 = g[8];
        
        // 优化的直接计算（恢复到正确版本）
        u_kc[0][0] = g0;
        u_kc[0][1] = 0.5f * (g0 + g1 + g2);
        u_kc[0][2] = 0.5f * (g0 - g1 + g2);
        u_kc[0][3] = g2;
        
        float t1 = g0 + g3 + g6;
        float t2 = g1 + g4 + g7;
        float t3 = g2 + g5 + g8;
        u_kc[1][0] = 0.5f * t1;
        u_kc[1][1] = 0.25f * (t1 + t2 + t3);
        u_kc[1][2] = 0.25f * (t1 - t2 + t3);
        u_kc[1][3] = 0.5f * t3;
        
        float t4 = g0 - g3 + g6;
        float t5 = g1 - g4 + g7;
        float t6 = g2 - g5 + g8;
        u_kc[2][0] = 0.5f * t4;
        u_kc[2][1] = 0.25f * (t4 + t5 + t6);
        u_kc[2][2] = 0.25f * (t4 - t5 + t6);
        u_kc[2][3] = 0.5f * t6;
        
        u_kc[3][0] = g6;
        u_kc[3][1] = 0.5f * (g6 + g7 + g8);
        u_kc[3][2] = 0.5f * (g6 - g7 + g8);
        u_kc[3][3] = g8;

        // === 使用直接计算优化的图像变换 ===
        int h_start = tile_y * 2;
        int w_start = tile_x * 2;
        
        // 直接加载和变换计算
        int base_idx = (n * C + c) * H * W + h_start * W + w_start;
        
        // 将4x4图像块直接加载到寄存器中
        float d00 = image[base_idx], d01 = image[base_idx + 1], d02 = image[base_idx + 2], d03 = image[base_idx + 3];
        float d10 = image[base_idx + W], d11 = image[base_idx + W + 1], d12 = image[base_idx + W + 2], d13 = image[base_idx + W + 3];
        float d20 = image[base_idx + 2*W], d21 = image[base_idx + 2*W + 1], d22 = image[base_idx + 2*W + 2], d23 = image[base_idx + 2*W + 3];
        float d30 = image[base_idx + 3*W], d31 = image[base_idx + 3*W + 1], d32 = image[base_idx + 3*W + 2], d33 = image[base_idx + 3*W + 3];
        
        // B^T * d * B 变换的直接计算
        float v_ncp[4][4];
        v_ncp[0][0] = d00 - d02 - d20 + d22;
        v_ncp[0][1] = d01 + d02 - d21 - d22;
        v_ncp[0][2] = d02 - d01 - d22 + d21;
        v_ncp[0][3] = d01 - d03 - d21 + d23;
        
        v_ncp[1][0] = d10 + d20 - d12 - d22;
        v_ncp[1][1] = d11 + d12 + d21 + d22;
        v_ncp[1][2] = d12 - d11 + d22 - d21;
        v_ncp[1][3] = d11 - d13 + d21 - d23;
        
        v_ncp[2][0] = d20 - d10 - d22 + d12;
        v_ncp[2][1] = d21 + d22 - d11 - d12;
        v_ncp[2][2] = d22 - d21 - d12 + d11;
        v_ncp[2][3] = d21 - d23 - d11 + d13;
        
        v_ncp[3][0] = d10 - d12 - d30 + d32;
        v_ncp[3][1] = d11 + d12 - d31 - d32;
        v_ncp[3][2] = d12 - d11 - d32 + d31;
        v_ncp[3][3] = d11 - d13 - d31 + d33;

        // === 逐元素相乘并累加 ===
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                m[i][j] += u_kc[i][j] * v_ncp[i][j];
            }
        }
    }

    // === 输出变换（恢复到简单正确版本） ===
    float Y[2][2];
    
    // A^T * m * A 的直接计算
    Y[0][0] = m[0][0] + m[0][1] + m[0][2] + m[1][0] + m[1][1] + m[1][2] + m[2][0] + m[2][1] + m[2][2];
    Y[0][1] = m[0][1] - m[0][2] - m[0][3] + m[1][1] - m[1][2] - m[1][3] + m[2][1] - m[2][2] - m[2][3];
    Y[1][0] = m[1][0] + m[1][1] + m[1][2] - m[2][0] - m[2][1] - m[2][2] - m[3][0] - m[3][1] - m[3][2];
    Y[1][1] = m[1][1] - m[1][2] - m[1][3] - m[2][1] + m[2][2] + m[2][3] - m[3][1] + m[3][2] + m[3][3];

    // === 直接写入输出 ===
    int h0 = tile_y * 2;
    int w0 = tile_x * 2;
    
    // 对内部tiles进行直接写入，无边界检查优化
    if (h0 + 1 < outH && w0 + 1 < outW) {
        int out_idx = (n * K + k) * outH * outW + h0 * outW + w0;
        output[out_idx] = Y[0][0];
        output[out_idx + 1] = Y[0][1];
        output[out_idx + outW] = Y[1][0];
        output[out_idx + outW + 1] = Y[1][1];
    } else {
        // 边界tiles的边界检查后备方案
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            #pragma unroll
            for (int j = 0; j < 2; ++j) {
                int h = h0 + i;
                int w = w0 + j;
                if (h < outH && w < outW) {
                    output[((n * K + k) * outH + h) * outW + w] = Y[i][j];
                }
            }
        }
    }
}

void winograd_conv(thrust::device_vector<float>& image,
                   thrust::device_vector<float>& filter, 
                   thrust::device_vector<float>& out,
                   thrust::device_vector<float>& U,
                   thrust::device_vector<float>& V, 
                   thrust::device_vector<float>& M,
                   int H, int W, int C, int K, int N) {
    const int outH = H - 2;
    const int outW = W - 2;
    
    // 使用最优block大小以获得更好性能
    const int threads_per_block = 256;
    int num_tiles = N * K * (outH / 2) * (outW / 2);
    int grid_size = (num_tiles + threads_per_block - 1) / threads_per_block;

    winograd_conv_kernel<<<grid_size, threads_per_block>>>(
        image.data().get(), filter.data().get(), out.data().get(),
        N, C, H, W, K, outH, outW
    );

    cudaDeviceSynchronize();
}
