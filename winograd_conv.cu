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

// Step 6: 数学计算优化的Winograd卷积核函数
// 使用 __launch_bounds__ 控制寄存器使用和并发性
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
        // === 数学优化的滤波器变换 ===
        const float* g = filter_base + c * 9;
        float u_kc[4][4];
        
        // 一次性将所有滤波器权重加载到寄存器中
        float g0 = g[0], g1 = g[1], g2 = g[2];
        float g3 = g[3], g4 = g[4], g5 = g[5];
        float g6 = g[6], g7 = g[7], g8 = g[8];
        
        // 预计算公共项以减少算术运算
        float g0_plus_g2 = g0 + g2, g0_minus_g2 = g0 - g2;
        float g3_plus_g5 = g3 + g5, g3_minus_g5 = g3 - g5;
        float g6_plus_g8 = g6 + g8, g6_minus_g8 = g6 - g8;
        float g1_plus_g4_plus_g7 = g1 + g4 + g7;
        float g1_minus_g4_plus_g7 = g1 - g4 + g7;
        
        // 行0：简化计算
        u_kc[0][0] = g0;
        u_kc[0][1] = 0.5f * (g0_plus_g2 + g1);
        u_kc[0][2] = 0.5f * (g0_plus_g2 - g1);
        u_kc[0][3] = g2;
        
        // 行1：优化计算
        float t1_base = g0 + g3 + g6;
        float t3_base = g2 + g5 + g8;
        u_kc[1][0] = 0.5f * t1_base;
        u_kc[1][1] = 0.25f * (t1_base + g1_plus_g4_plus_g7 + t3_base);
        u_kc[1][2] = 0.25f * (t1_base - g1_plus_g4_plus_g7 + t3_base);
        u_kc[1][3] = 0.5f * t3_base;
        
        // 行2：优化计算  
        float t4_base = g0 - g3 + g6;
        float t6_base = g2 - g5 + g8;
        u_kc[2][0] = 0.5f * t4_base;
        u_kc[2][1] = 0.25f * (t4_base + g1_minus_g4_plus_g7 + t6_base);
        u_kc[2][2] = 0.25f * (t4_base - g1_minus_g4_plus_g7 + t6_base);
        u_kc[2][3] = 0.5f * t6_base;
        
        // 行3：简化计算
        u_kc[3][0] = g6;
        u_kc[3][1] = 0.5f * (g6_plus_g8 + g7);
        u_kc[3][2] = 0.5f * (g6_plus_g8 - g7);
        u_kc[3][3] = g8;

        // === 数学优化的图像变换 ===
        int h_start = tile_y * 2;
        int w_start = tile_x * 2;
        
        // 使用优化的索引进行直接加载和变换计算
        int base_idx = (n * C + c) * H * W + h_start * W + w_start;
        
        // 使用局部性优化加载4x4图像块
        float d00 = image[base_idx], d01 = image[base_idx + 1], d02 = image[base_idx + 2], d03 = image[base_idx + 3];
        base_idx += W;
        float d10 = image[base_idx], d11 = image[base_idx + 1], d12 = image[base_idx + 2], d13 = image[base_idx + 3];
        base_idx += W;
        float d20 = image[base_idx], d21 = image[base_idx + 1], d22 = image[base_idx + 2], d23 = image[base_idx + 3];
        base_idx += W;
        float d30 = image[base_idx], d31 = image[base_idx + 1], d32 = image[base_idx + 2], d33 = image[base_idx + 3];
        
        // 预计算图像变换的公共项
        float d00_minus_d02 = d00 - d02, d01_plus_d02 = d01 + d02, d02_minus_d01 = d02 - d01, d01_minus_d03 = d01 - d03;
        float d10_minus_d12 = d10 - d12, d11_plus_d12 = d11 + d12, d12_minus_d11 = d12 - d11, d11_minus_d13 = d11 - d13;
        float d20_minus_d22 = d20 - d22, d21_plus_d22 = d21 + d22, d22_minus_d21 = d22 - d21, d21_minus_d23 = d21 - d23;
        float d30_minus_d32 = d30 - d32, d31_plus_d32 = d31 + d32, d32_minus_d31 = d32 - d31, d31_minus_d33 = d31 - d33;
        
        // 优化的 B^T * d * B 变换计算
        float v_ncp[4][4];
        v_ncp[0][0] = d00_minus_d02 - d20_minus_d22;
        v_ncp[0][1] = d01_plus_d02 - d21_plus_d22;
        v_ncp[0][2] = d02_minus_d01 - d22_minus_d21;
        v_ncp[0][3] = d01_minus_d03 - d21_minus_d23;
        
        v_ncp[1][0] = d10_minus_d12 + d20_minus_d22;
        v_ncp[1][1] = d11_plus_d12 + d21_plus_d22;
        v_ncp[1][2] = d12_minus_d11 + d22_minus_d21;
        v_ncp[1][3] = d11_minus_d13 + d21_minus_d23;
        
        v_ncp[2][0] = d20_minus_d22 - d10_minus_d12;
        v_ncp[2][1] = d21_plus_d22 - d11_plus_d12;
        v_ncp[2][2] = d22_minus_d21 - d12_minus_d11;
        v_ncp[2][3] = d21_minus_d23 - d11_minus_d13;
        
        v_ncp[3][0] = d10_minus_d12 - d30_minus_d32;
        v_ncp[3][1] = d11_plus_d12 - d31_plus_d32;
        v_ncp[3][2] = d12_minus_d11 - d32_minus_d31;
        v_ncp[3][3] = d11_minus_d13 - d31_minus_d33;

        // === 逐元素相乘并累加 ===
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                m[i][j] += u_kc[i][j] * v_ncp[i][j];
            }
        }
    }

    // === 数学优化的输出变换 ===
    float Y[2][2];
    
    // 预计算常用项
    float m01_plus_m02 = m[0][1] + m[0][2];
    float m01_minus_m02 = m[0][1] - m[0][2];
    float m11_plus_m12 = m[1][1] + m[1][2];
    float m11_minus_m12 = m[1][1] - m[1][2];
    float m21_plus_m22 = m[2][1] + m[2][2];
    float m21_minus_m22 = m[2][1] - m[2][2];
    float m31_plus_m32 = m[3][1] + m[3][2];
    float m31_minus_m32 = m[3][1] - m[3][2];
    
    // 优化的 A^T * m * A 计算
    Y[0][0] = m[0][0] + m01_plus_m02 + m[1][0] + m11_plus_m12 + m[2][0] + m21_plus_m22;
    Y[0][1] = m01_minus_m02 - m[0][3] + m11_minus_m12 - m[1][3] + m21_minus_m22 - m[2][3];
    Y[1][0] = m[1][0] + m11_plus_m12 - m[2][0] - m21_plus_m22 - m[3][0] - m31_plus_m32;
    Y[1][1] = m11_minus_m12 - m[1][3] - m21_minus_m22 + m[2][3] - m31_minus_m32 + m[3][3];

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

    // 计算共享内存大小：此优化不需要
    const int shared_memory_size = 0;

    winograd_conv_kernel<<<grid_size, threads_per_block, shared_memory_size>>>(
        image.data().get(), filter.data().get(), out.data().get(),
        N, C, H, W, K, outH, outW
    );

    cudaDeviceSynchronize();
}
