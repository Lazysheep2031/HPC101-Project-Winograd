#include "winograd.cuh"

// Winograd F(2x2,3x3) 算法的变换矩阵常量
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

// 第四步：数学库 + 内联函数优化
// 核心思想：使用CUDA内联函数和数学库优化，提高计算效率

// 内联函数：高效的Filter变换 G * g * G^T
__forceinline__ __device__ void filter_transform_optimized(const float* g, float* u_kc) {
    // 使用CUDA数学库的快速乘法：__fmaf_rn (fused multiply-add)
    // 同时使用更优的数据布局
    
    // 寄存器加载 - 利用L1缓存
    const float g0 = g[0], g1 = g[1], g2 = g[2];
    const float g3 = g[3], g4 = g[4], g5 = g[5]; 
    const float g6 = g[6], g7 = g[7], g8 = g[8];
    
    // 第一阶段：G * g 使用融合乘加指令优化
    float temp_g[12];
    // Row 0: [1, 0, 0] * g
    temp_g[0] = g0; temp_g[1] = g1; temp_g[2] = g2;
    
    // Row 1: [0.5, 0.5, 0.5] * g - 使用融合乘加
    temp_g[3] = __fmaf_rn(0.5f, g0, __fmaf_rn(0.5f, g3, 0.5f * g6));
    temp_g[4] = __fmaf_rn(0.5f, g1, __fmaf_rn(0.5f, g4, 0.5f * g7));
    temp_g[5] = __fmaf_rn(0.5f, g2, __fmaf_rn(0.5f, g5, 0.5f * g8));
    
    // Row 2: [0.5, -0.5, 0.5] * g - 使用融合乘加
    temp_g[6] = __fmaf_rn(0.5f, g0, __fmaf_rn(-0.5f, g3, 0.5f * g6));
    temp_g[7] = __fmaf_rn(0.5f, g1, __fmaf_rn(-0.5f, g4, 0.5f * g7));
    temp_g[8] = __fmaf_rn(0.5f, g2, __fmaf_rn(-0.5f, g5, 0.5f * g8));
    
    // Row 3: [0, 0, 1] * g
    temp_g[9] = g6; temp_g[10] = g7; temp_g[11] = g8;
    
    // 第二阶段：(G*g) * G^T 使用向量化操作
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float t0 = temp_g[i*3 + 0], t1 = temp_g[i*3 + 1], t2 = temp_g[i*3 + 2];
        u_kc[i*4 + 0] = t0;
        u_kc[i*4 + 1] = __fmaf_rn(0.5f, t0, __fmaf_rn(0.5f, t1, 0.5f * t2));
        u_kc[i*4 + 2] = __fmaf_rn(0.5f, t0, __fmaf_rn(-0.5f, t1, 0.5f * t2)); 
        u_kc[i*4 + 3] = t2;
    }
}

// 内联函数：高效的Input变换 B^T * d * B
__forceinline__ __device__ void input_transform_optimized(const float* d, float* v_ncp) {
    // 第一步：B^T * d - 使用CUDA内建数学函数优化
    float temp_d[16];
    
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float d0 = d[0*4 + i], d1 = d[1*4 + i], d2 = d[2*4 + i], d3 = d[3*4 + i];
        temp_d[0*4 + i] = __fsub_rn(d0, d2);         // d0 - d2 精确减法
        temp_d[1*4 + i] = __fadd_rn(d1, d2);         // d1 + d2 精确加法
        temp_d[2*4 + i] = __fsub_rn(d2, d1);         // d2 - d1 精确减法
        temp_d[3*4 + i] = __fsub_rn(d1, d3);         // d1 - d3 精确减法
    }
    
    // 第二步：(B^T * d) * B - 向量化计算
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float td0 = temp_d[i*4 + 0], td1 = temp_d[i*4 + 1];
        const float td2 = temp_d[i*4 + 2], td3 = temp_d[i*4 + 3];
        
        v_ncp[i*4 + 0] = __fsub_rn(td0, td2);        // td0 - td2
        v_ncp[i*4 + 1] = __fadd_rn(td1, td2);        // td1 + td2  
        v_ncp[i*4 + 2] = __fsub_rn(td2, td1);        // td2 - td1
        v_ncp[i*4 + 3] = __fsub_rn(td1, td3);        // td1 - td3
    }
}

// 内联函数：高效的Output变换 A^T * m * A
__forceinline__ __device__ void output_transform_optimized(const float* m, float* Y) {
    // 第一步：A^T * m - 使用CUDA数学库
    float temp_m[8];
    
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float m0 = m[0*4 + j], m1 = m[1*4 + j], m2 = m[2*4 + j], m3 = m[3*4 + j];
        temp_m[0*4 + j] = __fmaf_rn(m1, 1.0f, __fadd_rn(m0, m2));     // m0 + m1 + m2
        temp_m[1*4 + j] = __fmaf_rn(m2, -1.0f, __fmaf_rn(m3, -1.0f, m1)); // m1 - m2 - m3
    }
    
    // 第二步：(A^T * m) * A - 最终计算
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        const float tm0 = temp_m[i*4 + 0], tm1 = temp_m[i*4 + 1];
        const float tm2 = temp_m[i*4 + 2], tm3 = temp_m[i*4 + 3];
        
        Y[i*2 + 0] = __fmaf_rn(tm1, 1.0f, __fadd_rn(tm0, tm2));      // tm0 + tm1 + tm2
        Y[i*2 + 1] = __fmaf_rn(tm2, -1.0f, __fmaf_rn(tm3, -1.0f, tm1)); // tm1 - tm2 - tm3
    }
}

// 第六步：极致优化的单kernel版本
// 核心思想：单kernel架构 + 所有数学库内联优化 + 内存访问优化
__global__
void winograd_conv_kernel_optimized(const float* __restrict__ image,
                                    const float* __restrict__ filters,
                                    float* __restrict__ output,
                                    int N, int C, int H, int W, int K, int outH, int outW) {
    
    // 高效索引计算
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num_tiles = N * K * (outH / 2) * (outW / 2);
    if (idx >= num_tiles) return;

    // 快速索引分解
    int p_local = idx % ((outH / 2) * (outW / 2));
    int k = (idx / ((outH / 2) * (outW / 2))) % K;
    int n = idx / (K * (outH / 2) * (outW / 2));
    int tile_y = p_local / (outW / 2);
    int tile_x = p_local % (outW / 2);

    // 寄存器优化：预分配累加器
    float m[16] = {0.0f};

    // 地址预计算，减少重复计算
    const int h_start = tile_y * 2;
    const int w_start = tile_x * 2;
    const int base_img_addr = n * C * H * W;
    const int base_filter_addr = k * C * 9;
    const int base_out_addr = (n * K + k) * outH * outW;

    // === 主计算循环：充分利用内联函数优化 ===
    for (int c = 0; c < C; ++c) {
        
        // === Filter变换：G * g * G^T（使用优化内联函数）===
        const float* g = filters + base_filter_addr + c * 9;
        float u_kc[16];
        filter_transform_optimized(g, u_kc);

        // === Input变换：B^T * d * B（使用优化内联函数）===
        float d[16];
        const int img_addr = base_img_addr + c * H * W;
        
        // 向量化内存加载，优化合并访问
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int h = h_start + i;
            #pragma unroll  
            for (int j = 0; j < 4; ++j) {
                const int w = w_start + j;
                d[i*4 + j] = (h >= 0 && h < H && w >= 0 && w < W) ? 
                            image[img_addr + h * W + w] : 0.0f;
            }
        }
        
        float v_ncp[16];
        input_transform_optimized(d, v_ncp);

        // === Winograd域计算：矩阵风格优化 ===
        // 使用块状处理提高指令级并行度
        #pragma unroll
        for (int block = 0; block < 4; block++) {
            int base_idx = block * 4;
            // 融合乘加指令批量处理
            m[base_idx + 0] = __fmaf_rn(u_kc[base_idx + 0], v_ncp[base_idx + 0], m[base_idx + 0]);
            m[base_idx + 1] = __fmaf_rn(u_kc[base_idx + 1], v_ncp[base_idx + 1], m[base_idx + 1]);
            m[base_idx + 2] = __fmaf_rn(u_kc[base_idx + 2], v_ncp[base_idx + 2], m[base_idx + 2]);
            m[base_idx + 3] = __fmaf_rn(u_kc[base_idx + 3], v_ncp[base_idx + 3], m[base_idx + 3]);
        }
    }

    // === Output变换：A^T * m * A（使用优化内联函数）===
    float Y[4];
    output_transform_optimized(m, Y);

    // === 优化的内存写入 ===
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int h = tile_y * 2 + i;
            const int w = tile_x * 2 + j;
            if (h < outH && w < outW) {
                output[base_out_addr + h * outW + w] = Y[i*2 + j];
            }
        }
    }
}

// 第六步：混合优化 - 单kernel + 数学库内联优化
// 核心思想：回到单kernel架构，保留数学库优化的内联函数，最大化性能
void winograd_conv(thrust::device_vector<float>& image,
                   thrust::device_vector<float>& filter, 
                   thrust::device_vector<float>& out,
                   thrust::device_vector<float>& U,
                   thrust::device_vector<float>& V, 
                   thrust::device_vector<float>& M,
                   int H, int W, int C, int K, int N) {
    const int outH = H - 2;
    const int outW = W - 2;
    
    // 直接使用优化的单kernel方案，保留所有内联函数优化
    const int threads_per_block = 256;
    int num_tiles = N * K * (outH / 2) * (outW / 2);
    int grid_size = (num_tiles + threads_per_block - 1) / threads_per_block;

    // 启动优化的单kernel（包含所有数学库内联优化）
    winograd_conv_kernel_optimized<<<grid_size, threads_per_block>>>(
        image.data().get(),
        filter.data().get(),
        out.data().get(),
        N, C, H, W, K, outH, outW
    );

    cudaDeviceSynchronize();
}
