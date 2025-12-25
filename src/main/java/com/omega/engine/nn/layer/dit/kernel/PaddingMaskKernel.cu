#define SHARED_SIZE 256 * 4

extern "C"
__global__ void set_mask(const size_t size, const float *mask, float *out, const int W)
{

    for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < (size); pos += blockDim.x * gridDim.x) {
    	int w = pos % W;
    	out[pos] = mask[w];
  	}

}

extern "C"
__global__ void set_ids(const size_t size, const float *x, const float *idskeep, float *out, const int FT, const int T, const int W)
{

    for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < (size); pos += blockDim.x * gridDim.x) {
		int len = T * W;
    	int b = pos / len;
    	int t = pos % len / W;
    	int w = pos % len % W;
    	int t_idx = (int) idskeep[b * T + t];
    	out[b * FT * W + t_idx * W + w] = x[pos];
  	}

}

extern "C"
__global__ void set_ids_back(const size_t size, float *dx, const float *idskeep, float *dout, const int FT, const int T, const int W)
{

    for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < (size); pos += blockDim.x * gridDim.x) {
		int len = T * W;
    	int b = pos / len;
    	int t = pos % len / W;
    	int w = pos % len % W;
    	int t_idx = (int) idskeep[b * T + t];
    	dx[pos] = dout[b * FT * W + t_idx * W + w];
    	dout[b * FT * W + t_idx * W + w] = 0;
  	}

}

extern "C"
__global__ void mask_diff(const float *dout, float *dw, const int rows, const int cols) {
    // 每个线程块处理一列
    int col = blockIdx.x;
    
    if (col < cols) {
        // 分配共享内存，每个线程块使用一个共享变量
        __shared__ float shared_sum;
        
        // 初始化共享变量
        if (threadIdx.x == 0) {
            shared_sum = 0.0f;
        }
        __syncthreads();
        
        // 每个线程处理一部分工作负载
        unsigned int thread_id = threadIdx.x;
        unsigned int threads_per_block = blockDim.x;
        
        // 线程负责的行
        for (int row = thread_id; row < rows; row += threads_per_block) {
            atomicAdd(&shared_sum, dout[row * cols + col]);
        }
        __syncthreads();
        
        // 一个线程将结果写入全局内存
        if (threadIdx.x == 0) {
            dw[col] = shared_sum;
        }
    }
}

extern "C"
__global__ void mask_diff2(
    const float* __restrict__ inp,  // 输入：(M, N) 行优先存储
    float* __restrict__ out,        // 输出：(N,)
    int M,                      // 行数
    int N                       // 列数
) {
    // 共享内存：缓存当前线程块的列数据（按块加载，减少全局内存访问）
    __shared__ float s_data[SHARED_SIZE];

    // 线程索引：tx=列块内索引（0~255），对应处理1列
    const int tx = threadIdx.x;
    // 全局列索引 = 块索引×块大小 + 列块内索引
    const int col = blockIdx.x * blockDim.x + tx;
    if (col >= N) return;  // 超出列范围直接退出

    float col_sum = 0.0;
    const int block_rows = 4;  // 每次加载4行到共享内存

    // 分块遍历行维度，共享内存缓存数据
    for (int row_block = 0; row_block < M; row_block += block_rows) {
        // 步骤1：加载4行数据到共享内存（批量加载，合并访问）
        for (int br = 0; br < block_rows; br++) {
            int row = row_block + br;
            if (row < M) {
                // 共享内存索引：br*blockDim.x + tx（按行存储，避免bank冲突）
                s_data[br * blockDim.x + tx] = inp[row * N + col];
            } else {
                s_data[br * blockDim.x + tx] = 0.0;  // 超出范围置0
            }
        }
        __syncthreads();  // 等待共享内存加载完成

        // 步骤2：共享内存内批量累加（无需再访问全局内存）
        for (int br = 0; br < block_rows; br++) {
            col_sum += s_data[br * blockDim.x + tx];
        }
        __syncthreads();  // 等待累加完成
    }

    // 写入结果（单线程写单列，无竞争）
    out[col] = col_sum;
}
