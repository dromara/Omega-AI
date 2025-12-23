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
