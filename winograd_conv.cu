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

// Step 2: 直接矩阵计算优化的Winograd卷积核函数
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
        // === 滤波器变换 - 直接计算优化 ===
        const float* g = filter + (k * C + c) * 9;
        float u_kc[4][4];
        
        // 直接计算 G * g * G^T（避免中间临时数组temp_g）
        // 行0: g[0], 0.5*(g[0]+g[1]+g[2]), 0.5*(g[0]-g[1]+g[2]), g[2]
        u_kc[0][0] = g[0]; 
        u_kc[0][1] = 0.5f * (g[0] + g[1] + g[2]);
        u_kc[0][2] = 0.5f * (g[0] - g[1] + g[2]);
        u_kc[0][3] = g[2];
        
        // 行1: 使用预计算的中间变量
        float t1 = g[0] + g[3] + g[6];
        float t2 = g[1] + g[4] + g[7];
        float t3 = g[2] + g[5] + g[8];
        u_kc[1][0] = 0.5f * t1;
        u_kc[1][1] = 0.5f * 0.5f * (t1 + t2 + t3);
        u_kc[1][2] = 0.5f * 0.5f * (t1 - t2 + t3);
        u_kc[1][3] = 0.5f * t3;
        
        // 行2: 使用预计算的中间变量
        float t4 = g[0] - g[3] + g[6];
        float t5 = g[1] - g[4] + g[7];
        float t6 = g[2] - g[5] + g[8];
        u_kc[2][0] = 0.5f * t4;
        u_kc[2][1] = 0.5f * 0.5f * (t4 + t5 + t6);
        u_kc[2][2] = 0.5f * 0.5f * (t4 - t5 + t6);
        u_kc[2][3] = 0.5f * t6;
        
        // 行3: g[6], 0.5*(g[6]+g[7]+g[8]), 0.5*(g[6]-g[7]+g[8]), g[8]
        u_kc[3][0] = g[6];
        u_kc[3][1] = 0.5f * (g[6] + g[7] + g[8]);
        u_kc[3][2] = 0.5f * (g[6] - g[7] + g[8]);
        u_kc[3][3] = g[8];

        // === 图像变换 - 直接计算优化 ===
        int h_start = tile_y * 2;
        int w_start = tile_x * 2;
        float d[4][4];
        
        // 高效加载图像数据
        int base_idx = (n * C + c) * H * W + h_start * W + w_start;
        d[0][0] = image[base_idx];
        d[0][1] = image[base_idx + 1];
        d[0][2] = image[base_idx + 2];
        d[0][3] = image[base_idx + 3];
        d[1][0] = image[base_idx + W];
        d[1][1] = image[base_idx + W + 1];
        d[1][2] = image[base_idx + W + 2];
        d[1][3] = image[base_idx + W + 3];
        d[2][0] = image[base_idx + 2*W];
        d[2][1] = image[base_idx + 2*W + 1];
        d[2][2] = image[base_idx + 2*W + 2];
        d[2][3] = image[base_idx + 2*W + 3];
        d[3][0] = image[base_idx + 3*W];
        d[3][1] = image[base_idx + 3*W + 1];
        d[3][2] = image[base_idx + 3*W + 2];
        d[3][3] = image[base_idx + 3*W + 3];
        
        float v_ncp[4][4];
        
        // 直接计算 B^T * d * B（避免中间临时数组temp_d）
        // 使用优化的直接公式计算 B^T * d * B
        v_ncp[0][0] = d[0][0] - d[0][2] - d[2][0] + d[2][2];
        v_ncp[0][1] = d[0][1] + d[0][2] - d[2][1] - d[2][2];
        v_ncp[0][2] = d[0][2] - d[0][1] - d[2][2] + d[2][1];
        v_ncp[0][3] = d[0][1] - d[0][3] - d[2][1] + d[2][3];
        
        v_ncp[1][0] = d[1][0] + d[2][0] - d[1][2] - d[2][2];
        v_ncp[1][1] = d[1][1] + d[1][2] + d[2][1] + d[2][2];
        v_ncp[1][2] = d[1][2] - d[1][1] + d[2][2] - d[2][1];
        v_ncp[1][3] = d[1][1] - d[1][3] + d[2][1] - d[2][3];
        
        v_ncp[2][0] = d[2][0] - d[1][0] - d[2][2] + d[1][2];
        v_ncp[2][1] = d[2][1] + d[2][2] - d[1][1] - d[1][2];
        v_ncp[2][2] = d[2][2] - d[2][1] - d[1][2] + d[1][1];
        v_ncp[2][3] = d[2][1] - d[2][3] - d[1][1] + d[1][3];
        
        v_ncp[3][0] = d[1][0] - d[1][2] - d[3][0] + d[3][2];
        v_ncp[3][1] = d[1][1] + d[1][2] - d[3][1] - d[3][2];
        v_ncp[3][2] = d[1][2] - d[1][1] - d[3][2] + d[3][1];
        v_ncp[3][3] = d[1][1] - d[1][3] - d[3][1] + d[3][3];

        // === 逐元素相乘并累加 ===
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                m[i][j] += u_kc[i][j] * v_ncp[i][j];
            }
        }
    }

    // === 输出变换 - 直接计算优化 ===
    float Y[2][2];
    
    // 直接计算 A^T * m * A（避免中间临时数组temp_m）
    // Y = A^T * m * A，其中 A^T = [[1,1,1,0], [0,1,-1,-1]]
    Y[0][0] = m[0][0] + m[0][1] + m[0][2] + m[1][0] + m[1][1] + m[1][2] + m[2][0] + m[2][1] + m[2][2];
    Y[0][1] = m[0][1] - m[0][2] - m[0][3] + m[1][1] - m[1][2] - m[1][3] + m[2][1] - m[2][2] - m[2][3];
    Y[1][0] = m[1][0] + m[1][1] + m[1][2] - m[2][0] - m[2][1] - m[2][2] - m[3][0] - m[3][1] - m[3][2];
    Y[1][1] = m[1][1] - m[1][2] - m[1][3] - m[2][1] + m[2][2] + m[2][3] - m[3][1] + m[3][2] + m[3][3];

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

void winograd_conv(thrust::device_vector<float>& image,
                   thrust::device_vector<float>& filter, 
                   thrust::device_vector<float>& out,
                   thrust::device_vector<float>& U,
                   thrust::device_vector<float>& V, 
                   thrust::device_vector<float>& M,
                   int H, int W, int C, int K, int N) {
    const int outH = H - 2;
    const int outW = W - 2;
    
    // Optimize block size for better occupancy
    const int threads_per_block = 128; // Reduced from 256 for better cache usage
    int num_tiles = N * K * (outH / 2) * (outW / 2);
    int grid_size = (num_tiles + threads_per_block - 1) / threads_per_block;

    winograd_conv_kernel<<<grid_size, threads_per_block>>>(
        image.data().get(), filter.data().get(), out.data().get(),
        N, C, H, W, K, outH, outW
    );

    cudaDeviceSynchronize();
}
