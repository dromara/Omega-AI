
extern "C"
__global__ void token_drop(const size_t size, const float *x, float *param, float *mask, float *out, const int len, float prob)
{

    for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < (size); pos += blockDim.x * gridDim.x) {
    	int b = pos / len;
    	int w = pos % len;
    	if(mask[b] < prob){
			out[pos] = param[w];
		}else{
			out[pos] = x[pos];
		}
  	}

}

extern "C"
__global__ void token_drop_class(const size_t size, const float *x, float param, float *mask, float *out, const int len, float prob)
{

    for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < (size); pos += blockDim.x * gridDim.x) {
    	int b = pos / len;
    	if(mask[b] < prob){
			out[pos] = param;
		}else{
			out[pos] = x[pos];
		}
  	}

}

extern "C"
__global__ void timestep_embedding(int N, const float *t, float *freqs, float *out, const int d_model) {
    int idx = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (idx < N) {
       int n = idx / (d_model / 2);
       int w = idx % (d_model / 2);
       float args = t[n] * freqs[w];
       out[n * d_model + w] = cos(args);
       out[n * d_model + (d_model / 2) + w] = sin(args);
    }
}

extern "C"
__global__ void generateRandomUniqueIntegers(int B, int N, int T, float *output, int seeds) {
    // 每个block处理一个batch元素
    int batch_idx = blockIdx.x;

    if (batch_idx >= B) return;
    
    // 当前batch的输出起始位置
    int output_offset = batch_idx * N;

    // 为每个batch生成不同的种子
    int seed = seeds + batch_idx;
    
    // 使用共享内存存储已生成的随机数，确保唯一性
    extern __shared__ int shared_mem[];
    int *local_set = shared_mem;
    
    // 初始化共享内存为-1
    if (threadIdx.x < N) {
        local_set[threadIdx.x] = -1;
    }
    __syncthreads();
    
    // 使用Fisher-Yates洗牌算法的变体
    int count = 0;
    
    while (count < N) {
        // 每个线程生成一个随机数
        if (threadIdx.x == 0) {
            // 使用XORShift算法生成高质量随机数
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            
            // 生成0到T范围内的随机数
            int r = seed % (T + 1);
            
            // 检查是否已存在
            bool exists = false;
            for (int i = 0; i < count; i++) {
                if (local_set[i] == r) {
                    exists = true;
                    break;
                }
            }
            
            if (!exists) {
                local_set[count] = r;
                output[output_offset + count] = r;
                count++;
            }

        }
        __syncthreads();
    }
}

extern "C"
__global__ void img_token_drop(const size_t size, const float *x, float *idskeep, float *out, const int xT, const int T, const int W)
{

    for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < (size); pos += blockDim.x * gridDim.x) {
		int len = T * W;
    	int b = pos / len;
    	int t = pos % len / W;
    	int w = pos % len % W;
    	int ids = (int) idskeep[b * T + t];
    	out[pos] = x[b * xT * W + ids * W + w];
  	}

}

extern "C"
__global__ void img_token_drop_back(const size_t size, float *dx, float *idskeep, const float *dout, const int xT, const int T, const int W)
{

    for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < (size); pos += blockDim.x * gridDim.x) {
		int len = T * W;
    	int b = pos / len;
    	int t = pos % len / W;
    	int w = pos % len % W;
    	int ids = (int) idskeep[b * T + t];
    	dx[b * xT * W + ids * W + w] = dout[pos];
  	}

}