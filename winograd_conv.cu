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

// 核心思想：将每个 2x2 输出瓦片分配给一个线程，通过手工优化矩阵变换消除通用矩阵乘法开销
__global__
void winograd_conv_kernel(const float* __restrict__ image,
                          const float* __restrict__ filter,
                          float* __restrict__ output,
                          int N, int C, int H, int W, int K, int outH, int outW) {
    
    // 线程索引计算与任务分配
    // 每个线程负责处理一个 2x2 的输出瓦片，避免线程间的数据竞争
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num_tiles = N * K * (outH / 2) * (outW / 2);  // 总瓦片数量
    if (idx >= num_tiles) return;

    // 多维索引分解
    // 将线性线程索引分解为 (batch, filter, tile_y, tile_x) 四维坐标
    // 这种分解方式保证了内存访问的局部性
    int p_local = idx % ((outH / 2) * (outW / 2));    // 当前batch和filter内的瓦片索引
    int k = (idx / ((outH / 2) * (outW / 2))) % K;     // 输出通道索引
    int n = idx / (K * (outH / 2) * (outW / 2));       // batch索引
    int tile_y = p_local / (outW / 2);                 // 瓦片的Y坐标
    int tile_x = p_local % (outW / 2);                 // 瓦片的X坐标

    // 寄存器优化的累加器初始化
    // 使用寄存器数组存储 Winograd 域的中间结果，避免shared memory的bank conflicts
    float m[16] = {0.0f};  // 4x4 的 Winograd 域累加器

    // 地址预计算优化
    // 预先计算基地址，减少循环内的地址计算开销，提高内存访问效率
    int h_start = tile_y * 2;                          // 输入图像中瓦片的起始H坐标
    int w_start = tile_x * 2;                          // 输入图像中瓦片的起始W坐标
    int base_img_addr = (n * C) * H * W;               // 当前batch图像的基地址
    int base_flt_addr = k * C * 9;                     // 当前输出通道filter的基地址
    int base_out_addr = (n * K + k) * outH * outW;     // 当前输出位置的基地址

    // Winograd算法需要对所有输入通道进行处理，然后在Winograd域进行累加
    for (int c = 0; c < C; ++c) {
        
        //Filter变换
        // 核心优化：展开矩阵乘法，消除通用矩阵乘法的循环开销
        const float* g = filter + base_flt_addr + c * 9;  // 当前3x3 filter
        float u_kc[16];  // 存储变换后的4x4 filter
        
        // 将3x3 filter加载到寄存器，避免重复内存访问
        float g0 = g[0], g1 = g[1], g2 = g[2];
        float g3 = g[3], g4 = g[4], g5 = g[5];
        float g6 = g[6], g7 = g[7], g8 = g[8];
        
        // G * g 变换
        // 手工计算每一行，避免三重循环的开销
        float temp_g[12];
        // 行0: [1, 0, 0] 乘以 g 的每一列
        temp_g[0] = g0; temp_g[1] = g1; temp_g[2] = g2;
        // 行1: [0.5, 0.5, 0.5] 乘以 g 的每一列
        temp_g[3] = 0.5f*(g0+g3+g6); temp_g[4] = 0.5f*(g1+g4+g7); temp_g[5] = 0.5f*(g2+g5+g8);
        // 行2: [0.5, -0.5, 0.5] 乘以 g 的每一列
        temp_g[6] = 0.5f*(g0-g3+g6); temp_g[7] = 0.5f*(g1-g4+g7); temp_g[8] = 0.5f*(g2-g5+g8);
        // 行3: [0, 0, 1] 乘以 g 的每一列
        temp_g[9] = g6; temp_g[10] = g7; temp_g[11] = g8;
        
        // (G*g) * G^T 变换
        // 使用 pragma unroll 展开循环，减少分支开销
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            float t0 = temp_g[i*3 + 0], t1 = temp_g[i*3 + 1], t2 = temp_g[i*3 + 2];
            u_kc[i*4 + 0] = t0;                           // G^T 的列0: [1, 0, 0]
            u_kc[i*4 + 1] = 0.5f * (t0 + t1 + t2);       // G^T 的列1: [0.5, 0.5, 0.5]
            u_kc[i*4 + 2] = 0.5f * (t0 - t1 + t2);       // G^T 的列2: [0.5, -0.5, 0.5]
            u_kc[i*4 + 3] = t2;                           // G^T 的列3: [0, 0, 1]
        }

        // Input变换
        // 高效的输入数据加载与边界处理
        float d[16];  // 4x4 输入补丁
        int img_addr = base_img_addr + c * H * W;  // 当前通道图像的基地址
        
        // 加载4x4输入补丁，同时处理边界情况
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int h = h_start + i;
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                int w = w_start + j;
                // 边界检查：超出边界的位置用0填充
                d[i*4 + j] = (h >= 0 && h < H && w >= 0 && w < W) ? 
                            image[img_addr + h * W + w] : 0.0f;
            }
        }
        
        // 手工优化的Input变换
        float v_ncp[16];  // 存储变换后的4x4 input
        
        // ：B^T * d 变换 
        // 利用 B^T 矩阵的稀疏性，手工计算每个元素
        float temp_d[16];
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            float d0 = d[0*4 + i], d1 = d[1*4 + i], d2 = d[2*4 + i], d3 = d[3*4 + i];
            temp_d[0*4 + i] = d0 - d2;          // B^T 行0: [1, 0, -1, 0]
            temp_d[1*4 + i] = d1 + d2;          // B^T 行1: [0, 1, 1, 0]  
            temp_d[2*4 + i] = -d1 + d2;         // B^T 行2: [0, -1, 1, 0]
            temp_d[3*4 + i] = d1 - d3;          // B^T 行3: [0, 1, 0, -1]
        }
        
        // (B^T * d) * B 变换
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            float td0 = temp_d[i*4 + 0], td1 = temp_d[i*4 + 1], td2 = temp_d[i*4 + 2], td3 = temp_d[i*4 + 3];
            v_ncp[i*4 + 0] = td0 - td2;         // B 列0: [1, 0, -1, 0]^T
            v_ncp[i*4 + 1] = td1 + td2;         // B 列1: [0, 1, 1, 0]^T
            v_ncp[i*4 + 2] = -td1 + td2;        // B 列2: [0, -1, 1, 0]^T
            v_ncp[i*4 + 3] = td1 - td3;         // B 列3: [0, 1, 0, -1]^T
        }

        // Winograd域的逐元素乘法与累加
        // 向量化的逐元素乘法
        // 在Winograd变换域进行逐元素乘法，这是算法的核心计算
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            m[i] += u_kc[i] * v_ncp[i];  // 对应元素相乘并累加到结果矩阵
        }
    }

    // Output变换
    // 输出变换
    float Y[4];  // 最终的2x2输出瓦片
    
    // A^T * m 变换 
    // 手工计算，利用A^T矩阵的结构特性
    float temp_m[8];
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        float m0 = m[0*4 + j], m1 = m[1*4 + j], m2 = m[2*4 + j], m3 = m[3*4 + j];
        temp_m[0*4 + j] = m0 + m1 + m2;        // A^T 行0: [1, 1, 1, 0]
        temp_m[1*4 + j] = m1 - m2 - m3;        // A^T 行1: [0, 1, -1, -1]
    }
    
    // (A^T * m) * A 变换
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        float tm0 = temp_m[i*4 + 0], tm1 = temp_m[i*4 + 1], tm2 = temp_m[i*4 + 2], tm3 = temp_m[i*4 + 3];
        Y[i*2 + 0] = tm0 + tm1 + tm2;          // A 列0: [1, 1, 1, 0]^T
        Y[i*2 + 1] = tm1 - tm2 - tm3;          // A 列1: [0, 1, -1, -1]^T
    }

    // 结果写回
    // 优化的输出写入与边界检查
    // 将2x2的结果瓦片写回到输出特征图的正确位置
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            int h = tile_y * 2 + i;
            int w = tile_x * 2 + j;
            if (h < outH && w < outW) {  // 边界检查，避免越界写入
                output[base_out_addr + h * outW + w] = Y[i*2 + j];
            }
        }
    }
}

// Winograd卷积主函数 - 负责启动GPU kernel并管理执行配置
void winograd_conv(thrust::device_vector<float>& image,
                   thrust::device_vector<float>& filter, 
                   thrust::device_vector<float>& out,
                   thrust::device_vector<float>& U,
                   thrust::device_vector<float>& V, 
                   thrust::device_vector<float>& M,
                   int H, int W, int C, int K, int N) {
    const int outH = H - 2;  // 输出特征图高度（输入减去filter尺寸再加1）
    const int outW = W - 2;  // 输出特征图宽度
    
    // GPU执行配置优化：
    // 选择256线程/块 - 在寄存器使用和并行度之间达到最佳平衡
    const int threads_per_block = 256;
    
    // 计算总的工作量：每个2x2输出瓦片需要一个线程
    int num_tiles = N * K * (outH / 2) * (outW / 2);
    
    // 计算所需的线程块数量，向上取整确保覆盖所有工作
    int grid_size = (num_tiles + threads_per_block - 1) / threads_per_block;

    // 启动GPU kernel - 使用最优配置执行Winograd卷积
    winograd_conv_kernel<<<grid_size, threads_per_block>>>(
        image.data().get(),   // 输入特征图
        filter.data().get(),  // 卷积核
        out.data().get(),     // 输出特征图
        N, C, H, W, K, outH, outW
    );

    // 同步GPU执行，确保计算完成后再返回
    cudaDeviceSynchronize();
}
