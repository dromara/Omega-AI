#define BLOCK 1024
#define FLT_MAX 3.402823466e+38F

extern "C"
__global__ void loss(float *input, float *label, float *output, int batch, int n)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (id >= batch) return;

	float sum = 0;

	for(int i = 0;i<n;i++){
        sum += - label[id * n + i] * log(input[id * n + i]);
    }
    
	output[id] = sum;
}

extern "C"
__global__ void nl_loss(float *input, float *label, float *output, int batch, int n)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (id >= batch) return;

	float sum = 0;

	for(int i = 0;i<n;i++){
        sum += - label[id * n + i] * input[id * n + i];
    }
    
	output[id] = sum;
}

extern "C"
__global__ void log_softmax_nl_loss(float *input, float *label, float *output, int batch, int n)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (id >= batch) return;
	float max = -FLT_MAX;
	float sum = 0;
	float loss_sum = 0;
	float EPSILON = 1e-12f;
	
	for(int i = 0;i<n;i++) {
		if(max <= input[id * n + i]) {
			max = input[id * n + i];
		}
	}
	for(int i = 0;i<n;i++){
        sum += expf(input[id * n + i] - max);
    }
	for(int i = 0;i<n;i++){
        //loss_sum += - ((input[id * n + i] - max) - logf(sum)) * label[id * n + i];
        float vl = fmax(expf(input[id * n + i] - max) / sum, EPSILON);
        loss_sum -= logf(vl) * label[id * n + i];
    }
    output[id] = loss_sum;
}

extern "C"
__global__ void log_softmax_nl_loss_igone(float *input, float *label, float *output, int batch, int n,int igonre)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (id >= batch) return;
	float max = -FLT_MAX;
	float sum = 0;
	float loss_sum = 0;
	float EPSILON = 1e-12f;
	if(label[id * n + igonre] == 1){
        output[id] = 0.0f;
        return;
    }
	for(int i = 0;i<n;i++) {
		if(max <= input[id * n + i]) {
			max = input[id * n + i];
		}
	}
	for(int i = 0;i<n;i++){
        sum += expf(input[id * n + i] - max);
    }
	for(int i = 0;i<n;i++){
        //loss_sum += - ((input[id * n + i] - max) - logf(sum)) * label[id * n + i];
        float vl = fmax(expf(input[id * n + i] - max) / sum, EPSILON);
        loss_sum -= logf(vl) * label[id * n + i];
    }
    output[id] = loss_sum;
}

extern "C"
__global__ void log_softmax_nl_loss_igone_idx(float *input, float *label, float *output, int batch, int n,int igonre)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (id >= batch) return;
	float max = -FLT_MAX;
	float sum = 0;
	float loss_sum = 0;
	float EPSILON = 1e-12f;
	int tx = label[id];
	if(tx == igonre){
        output[id] = 0.0f;
        return;
    }
	for(int i = 0;i<n;i++) {
		if(max <= input[id * n + i]) {
			max = input[id * n + i];
		}
	}
	for(int i = 0;i<n;i++){
        sum += expf(input[id * n + i] - max);
    }
    float vl = fmax(expf(input[id * n + tx] - max) / sum, EPSILON);
    output[id] = -logf(vl);
}

extern "C"
__global__ void check(float *input, float *label, float *output, int batch, int n)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (id >= batch) return;

	for(int i = 0;i<n;i++){
        output[id * n + i] =  - label[id * n + i] * log(input[id * n + i]);
    }

}

extern "C"
__global__ void loss_back(float *output, float *currentLabel, float *diff, int n, int batch)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (id >= n) return;
	
	diff[id] = (output[id] - currentLabel[id]) / batch;    

}

extern "C"
__global__ void loss_back2(float *input, float *currentLabel, float *diff, int batch, int n)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (id >= batch) return;
	float max = -FLT_MAX;
	float sum = 0;
	float EPSILON = 1e-12f;
	for(int i = 0;i<n;i++) {
		if(max <= input[id * n + i]) {
			max = input[id * n + i];
		}
	}
	for(int i = 0;i<n;i++){
        float e = expf(input[id * n + i] - max);
        sum += e;
        diff[id * n + i] = e;
    }
	for(int i = 0;i<n;i++){
        //diff[id * n + i] = ((diff[id * n + i] / sum) - currentLabel[id * n + i]) / batch;
        float vl = fmax(diff[id * n + i] / sum, EPSILON);
        diff[id * n + i] = vl - currentLabel[id * n + i];
    } 
}

extern "C"
__global__ void loss_back_igonre2(float *input, float *currentLabel, float *diff, int batch, int n,int igonre)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (id >= batch) return;
	float max = -FLT_MAX;
	float sum = 0;
	float EPSILON = 1e-12f;
	if(currentLabel[id * n + igonre] == 1){
		for(int i = 0;i<n;i++){
        	diff[id * n + i] = 0.0f;
        }
        return;
    }
	for(int i = 0;i<n;i++) {
		if(max <= input[id * n + i]) {
			max = input[id * n + i];
		}
	}
	for(int i = 0;i<n;i++){
        float e = expf(input[id * n + i] - max);
        sum += e;
        diff[id * n + i] = e;
    }
	for(int i = 0;i<n;i++){
        //diff[id * n + i] = ((diff[id * n + i] / sum) - currentLabel[id * n + i]) / batch;
        float vl = fmax(diff[id * n + i] / sum, EPSILON);
        diff[id * n + i] = vl - currentLabel[id * n + i];
    }
}

extern "C"
__global__ void loss_back_igonre2_idx(float *input, float *currentLabel, float *diff, int batch, int n,int igonre)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (id >= batch) return;
	float max = -FLT_MAX;
	float sum = 0;
	float EPSILON = 1e-12f;
	int ix = currentLabel[id];
	if(currentLabel[id * n + igonre] == 1){
		for(int i = 0;i<n;i++){
        	diff[id * n + i] = 0.0f;
        }
        return;
    }
	for(int i = 0;i<n;i++) {
		if(max <= input[id * n + i]) {
			max = input[id * n + i];
		}
	}
	for(int i = 0;i<n;i++){
        float e = expf(input[id * n + i] - max);
        sum += e;
        diff[id * n + i] = e;
    }
    
	for(int i = 0;i<n;i++){
        //diff[id * n + i] = ((diff[id * n + i] / sum) - currentLabel[id * n + i]) / batch;
        float indicator = i == ix ? 1.0f : 0.0f;
        float vl = fmax(diff[id * n + i] / sum, EPSILON);
        diff[id * n + i] = vl - indicator;
    }
}

__device__ float warpReduceMax(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

__device__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

extern "C"
__global__ void softmax_forward_kernel7(float* out, const float* inp, int N, int C) {
    // out is (N, C) just like inp. Each row of inp will get softmaxed.
    // same as kernel4, but optimised for very large Cs with advanced unrolling

    // The trick is to read into a register array (all indices known at compile time)
    // and always read UNROLL_FACTOR values to maximise memory level parallelism
    // even if we would be out of bounds, we set the index to min(C-1, idx)
    // so we just do some unnecessary reads (obviously bad for small C)
    // the writes are in a separate loop with a conditional check for out of bounds
    // making it separate is necessary to convince the compiler to do the right thing
    const int UNROLL_FACTOR = 8;
    const int warpsPerBlock = blockDim.x / 32;

    extern __shared__ float shared[];
    int idx = blockIdx.x;
    int tid = threadIdx.x;
    int warpId = threadIdx.x / 32; // warp index within a block
    int laneId = threadIdx.x % 32; // thread index within a warp

    // shared[] must be allocated to have 2 * warpsPerBlock elements
    // first half for max values, the second half for sum values
    float* maxvals = shared;
    float* sumvals = &shared[warpsPerBlock];

    if (tid >= C) {
        maxvals[warpId] = -INFINITY;
        sumvals[warpId] = 0.0f;
        return;
    }

    const float* x = inp + idx * C; // input
    float* y = out + idx * C; // output

    // first, thread coarsening by directly accessing global memory in series
    float maxval = -INFINITY;
    for (int i = tid; i < C; i += blockDim.x * UNROLL_FACTOR) {
        #pragma unroll
        for (int u = 0; u < UNROLL_FACTOR; u++) {
            maxval = fmaxf(maxval, x[min(C - 1, i + u*blockDim.x)]);
        }
    }

    // now within-warp reductions for maxval
    maxval = warpReduceMax(maxval);
    // the 0th thread of each warp writes the maxval of that warp to shared memory
    if (laneId == 0) maxvals[warpId] = maxval;
    __syncthreads();
    // now the 0th thread reduces the maxvals in shared memory, i.e. across warps
    if (tid == 0) {
        float val = maxvals[tid];
        #pragma unroll
        for (int i = 1; i < warpsPerBlock; i++) {
            val = fmaxf(val, maxvals[i]);
        }
        // store the final max in the first position
        maxvals[0] = val;
    }
    __syncthreads();
    // broadcast the max to all threads
    float offset = maxvals[0];

    // compute expf and write the result to global memory
    // + thread coarsening for sum
    float sumval = 0.0f;
    for (int i = tid; i < C; i += blockDim.x * UNROLL_FACTOR) {
        float reg_array[UNROLL_FACTOR];
        #pragma unroll
        for (int u = 0; u < UNROLL_FACTOR; u++) {
            reg_array[u] = __ldcs(&x[min(C - 1, i + u*blockDim.x)]);
        }
        #pragma unroll
        for (int u = 0; u < UNROLL_FACTOR; u++) {
            if (i + u*blockDim.x < C) {
                float output = expf(reg_array[u] - offset);
                y[min(C - 1, i + u*blockDim.x)] = output; // compiler likes redundant min()?!
                sumval += output; // combined into the same loop unlike kernel3
            }
        }
    }

    // okay now we calculated exp(x - max(x))
    // step 2: sum all the values and divide by the sum

    // within-warp reduction for sumval
    sumval = warpReduceSum(sumval);
    // write sumval to shared memory
    if (laneId == 0) sumvals[warpId] = sumval;
    __syncthreads();
    // inter-thread reduction of sum
    if (tid == 0) {
        float val = sumvals[tid];
        #pragma unroll
        for (int i = 1; i < warpsPerBlock; ++i) {
            val += sumvals[i];
        }
        sumvals[0] = val;
    }
    __syncthreads();
    // broadcast the sum to all threads
    float sum = sumvals[0];

    // divide the whole row by the sum
    for (int i = tid; i < C; i += blockDim.x * UNROLL_FACTOR) {
        float reg_array[UNROLL_FACTOR];
        #pragma unroll
        for (int u = 0; u < UNROLL_FACTOR; u++) {
            reg_array[u] = y[min(C - 1, i + u*blockDim.x)];
        }
        #pragma unroll
        for (int u = 0; u < UNROLL_FACTOR; u++) {
            if (i + u*blockDim.x < C) {
                y[i + u*blockDim.x] = reg_array[u] / sum;
                
            }
        }
    }
}

extern "C"
__global__ void crossentropy_forward_kernel(float* losses,
                            const float* probs, const int* targets,
                            int B, int T, int V) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < B * T) {
        int b = i / T;
        int t = i % T;
        const float* probs_bt = probs + b * T * V + t * V;  // probs[b,t,:]
        int ix = targets[b * T + t];
        losses[b * T + t] += -logf(probs_bt[ix]); 
    }
}

extern "C"
__global__ void crossentropy_forward_kernel_igone(float* losses,
                            const float* probs, const int* targets,
                            int B, int T, int V, int igone) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < B * T) {
        int b = i / T;
        int t = i % T;
        int ix = targets[b * T + t];
        if(ix == igone){
        	losses[b * T + t] += 0; 
        }else{
        	const float* probs_bt = probs + b * T * V + t * V;  // probs[b,t,:]
        	losses[b * T + t] += -logf(probs_bt[ix]); 
        }
    }
}

extern "C"
__global__ void crossentropy_softmax_backward_kernel(float* dlogits, const float* probs, const int* targets,int B, int T, int V) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < B * T * V) {
        int b = i / (T * V);
        int t = (i / V) % T;
        int v = i % V;
        float* dlogits_bt = dlogits + b * T * V + t * V;
        const float* probs_bt = probs + b * T * V + t * V;
        int ix = targets[b * T + t];
        float p = probs_bt[v];
        float indicator = v == ix ? 1.0f : 0.0f;
        dlogits_bt[v] = (p - indicator);
    }
}

extern "C"
__global__ void crossentropy_softmax_igone_backward_kernel(float* dlogits, const float* probs, const int* targets,
	int B, int T, int V,int igone) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < B * T * V) {
        int b = i / (T * V);
        int t = (i / V) % T;
        int v = i % V;
        float* dlogits_bt = dlogits + b * T * V + t * V;
        const float* probs_bt = probs + b * T * V + t * V;
        int ix = targets[b * T + t];
        float p = probs_bt[v];
        printf("p:%f",p);
        float indicator = v == ix ? 1.0f : 0.0f;
        if(ix == igone){
        	dlogits_bt[v] = 0;
        }else{
        	dlogits_bt[v] = (p - indicator);
        }
    }
}


extern "C"
__global__ void cross_softmax_forward_kernel(float* loss,float* out, const float* inp, const float* label, int igone, int N, int C) {
    const int UNROLL_FACTOR = 8;
    const int warpsPerBlock = blockDim.x / 32;

    extern __shared__ float shared[];
    int idx = blockIdx.x;
    int tid = threadIdx.x;
    int warpId = threadIdx.x / 32; // warp index within a block
    int laneId = threadIdx.x % 32; // thread index within a warp

    // shared[] must be allocated to have 2 * warpsPerBlock elements
    // first half for max values, the second half for sum values
    float* maxvals = shared;
    float* sumvals = &shared[warpsPerBlock];
	
    if (tid >= C) {
        maxvals[warpId] = -INFINITY;
        sumvals[warpId] = 0.0f;
        return;
    }
	
	int tx = label[idx];
	loss[idx] = 0;
    const float* x = inp + idx * C; // input
    float* y = out + idx * C; // output

    // first, thread coarsening by directly accessing global memory in series
    float maxval = -INFINITY;
    for (int i = tid; i < C; i += blockDim.x * UNROLL_FACTOR) {
        #pragma unroll
        for (int u = 0; u < UNROLL_FACTOR; u++) {
            maxval = fmaxf(maxval, x[min(C - 1, i + u*blockDim.x)]);
        }
    }

    // now within-warp reductions for maxval
    maxval = warpReduceMax(maxval);
    // the 0th thread of each warp writes the maxval of that warp to shared memory
    if (laneId == 0) maxvals[warpId] = maxval;
    __syncthreads();
    // now the 0th thread reduces the maxvals in shared memory, i.e. across warps
    if (tid == 0) {
        float val = maxvals[tid];
        #pragma unroll
        for (int i = 1; i < warpsPerBlock; i++) {
            val = fmaxf(val, maxvals[i]);
        }
        // store the final max in the first position
        maxvals[0] = val;
    }
    __syncthreads();
    // broadcast the max to all threads
    float offset = maxvals[0];

    // compute expf and write the result to global memory
    // + thread coarsening for sum
    float sumval = 0.0f;
    for (int i = tid; i < C; i += blockDim.x * UNROLL_FACTOR) {
        float reg_array[UNROLL_FACTOR];
        #pragma unroll
        for (int u = 0; u < UNROLL_FACTOR; u++) {
            reg_array[u] = __ldcs(&x[min(C - 1, i + u*blockDim.x)]);
        }
        #pragma unroll
        for (int u = 0; u < UNROLL_FACTOR; u++) {
            if (i + u*blockDim.x < C) {
                float output = expf(reg_array[u] - offset);
                y[min(C - 1, i + u*blockDim.x)] = output; // compiler likes redundant min()?!
                sumval += output; // combined into the same loop unlike kernel3
            }
        }
    }

    // okay now we calculated exp(x - max(x))
    // step 2: sum all the values and divide by the sum

    // within-warp reduction for sumval
    sumval = warpReduceSum(sumval);
    // write sumval to shared memory
    if (laneId == 0) sumvals[warpId] = sumval;
    __syncthreads();
    // inter-thread reduction of sum
    if (tid == 0) {
        float val = sumvals[tid];
        #pragma unroll
        for (int i = 1; i < warpsPerBlock; ++i) {
            val += sumvals[i];
        }
        sumvals[0] = val;
    }
    __syncthreads();
    // broadcast the sum to all threads
    float sum = sumvals[0];

    // divide the whole row by the sum
    for (int i = tid; i < C; i += blockDim.x * UNROLL_FACTOR) {
        float reg_array[UNROLL_FACTOR];
        #pragma unroll
        for (int u = 0; u < UNROLL_FACTOR; u++) {
            reg_array[u] = y[min(C - 1, i + u*blockDim.x)];
        }
        #pragma unroll
        for (int u = 0; u < UNROLL_FACTOR; u++) {
            if (i + u*blockDim.x < C) {
            	y[i + u*blockDim.x] = reg_array[u] / sum;
            }
        }
        
    }
    
    if(tx == igone){
		loss[idx] = 0;
	}else{
		loss[idx] = -logf(y[tx]);
	}
}

extern "C"
__global__ void cross_softmax_backward_kernel(float* out, const float* inp, const float* label, int igone, int N, int C) {
    const int UNROLL_FACTOR = 8;

    int idx = blockIdx.x;
    int tid = threadIdx.x;

    if (tid >= C) {
        return;
    }
	int tx = label[idx];
    const float* x = inp + idx * C; // input
    float* y = out + idx * C; // output
	
	if(tx == igone){
		// divide the whole row by the sum
	    for (int i = tid; i < C; i += blockDim.x * UNROLL_FACTOR) {
	        #pragma unroll
	        for (int u = 0; u < UNROLL_FACTOR; u++) {
	            if (i + u*blockDim.x < C) {
	            	y[i + u*blockDim.x] = 0;
	            }
	        }
	    }
	}else{
		// divide the whole row by the sum
	    for (int i = tid; i < C; i += blockDim.x * UNROLL_FACTOR) {
	        #pragma unroll
	        for (int u = 0; u < UNROLL_FACTOR; u++) {
	            if (i + u*blockDim.x < C) {
	            	float indicator = i + u*blockDim.x == tx ? 1.0f : 0.0f;
	            	y[i + u*blockDim.x] = (x[i + u*blockDim.x] - indicator) / N;
	            }
	        }
	    }
	}

}