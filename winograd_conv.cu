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

// Step 1: 循环展开优化的Winograd卷积核函数
__global__
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

    // 遍历所有输入通道
    for (int c = 0; c < C; ++c) {
        // === 滤波器变换 ===
        const float* g = filter + (k * C + c) * 9;
        float u_kc[4][4];
        float temp_g[4][3];
        
        // 计算 G * g，使用循环展开优化
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            temp_g[i][0] = G[i][0] * g[0] + G[i][1] * g[3] + G[i][2] * g[6];
            temp_g[i][1] = G[i][0] * g[1] + G[i][1] * g[4] + G[i][2] * g[7];
            temp_g[i][2] = G[i][0] * g[2] + G[i][1] * g[5] + G[i][2] * g[8];
        }
        
        // 计算 (G * g) * G^T，使用循环展开优化
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            u_kc[i][0] = temp_g[i][0];
            u_kc[i][1] = 0.5f * (temp_g[i][0] + temp_g[i][1] + temp_g[i][2]);
            u_kc[i][2] = 0.5f * (temp_g[i][0] - temp_g[i][1] + temp_g[i][2]);
            u_kc[i][3] = temp_g[i][2];
        }

        // === 图像变换 ===
        int h_start = tile_y * 2;
        int w_start = tile_x * 2;
        float d[4][4];
        
        // 加载图像数据，优化内存合并访问
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                d[i][j] = image[(n * C + c) * H * W + (h_start + i) * W + (w_start + j)];
            }
        }
        
        float v_ncp[4][4];
        float temp_d[4][4];
        
        // 计算 B^T * d，使用循环展开优化
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                temp_d[i][j] = B_T[i][0] * d[0][j] + B_T[i][1] * d[1][j] + B_T[i][2] * d[2][j] + B_T[i][3] * d[3][j];
            }
        }
        
        // 计算 (B^T * d) * B，使用循环展开优化
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                v_ncp[i][j] = temp_d[i][0] * B[0][j] + temp_d[i][1] * B[1][j] + temp_d[i][2] * B[2][j] + temp_d[i][3] * B[3][j];
            }
        }

        // === 逐元素相乘并累加 ===
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                m[i][j] += u_kc[i][j] * v_ncp[i][j];
            }
        }
    }

    // === 输出变换 ===
    float temp_m[2][4];
    
    // 计算 A^T * m，使用循环展开优化
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            temp_m[i][j] = A_T[i][0] * m[0][j] + A_T[i][1] * m[1][j] + A_T[i][2] * m[2][j] + A_T[i][3] * m[3][j];
        }
    }
    
    float Y[2][2];
    // 计算 (A^T * m) * A，使用循环展开优化
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        Y[i][0] = temp_m[i][0] + temp_m[i][1] + temp_m[i][2];
        Y[i][1] = temp_m[i][1] - temp_m[i][2] - temp_m[i][3];
    }

    // === 写入输出并进行边界检查 ===
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            int h = tile_y * 2 + i;
            int w = tile_x * 2 + j;
            if (h < outH && w < outW) {
                output[((n * K + k) * outH + h) * outW + w] = Y[i][j];
            }
        }
    }
}

// Winograd卷积主函数接口
void winograd_conv(thrust::device_vector<float>& image,
                   thrust::device_vector<float>& filter, 
                   thrust::device_vector<float>& out,
                   thrust::device_vector<float>& U,
                   thrust::device_vector<float>& V, 
                   thrust::device_vector<float>& M,
                   int H, int W, int C, int K, int N) {
    const int outH = H - 2;
    const int outW = W - 2;
    
    // 设置GPU执行参数
    const int threads_per_block = 256;
    int num_tiles = N * K * (outH / 2) * (outW / 2);
    int grid_size = (num_tiles + threads_per_block - 1) / threads_per_block;

    // 启动CUDA核函数
    winograd_conv_kernel<<<grid_size, threads_per_block>>>(
        image.data().get(), filter.data().get(), out.data().get(),
        N, C, H, W, K, outH, outW
    );

    cudaDeviceSynchronize();
}
