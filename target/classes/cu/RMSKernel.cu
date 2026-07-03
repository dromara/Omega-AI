#define BLOCK 1024 
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

__device__ void atomicAddX(float* addr, float val) {
    atomicAdd(addr, val);
}

extern "C"
__global__ void rmsnorm_forward_kernel(float* __restrict__ out, float* __restrict__ smean, float* __restrict__ rms,const float*  __restrict__ weight,
                                    const float*  __restrict__ inp, int N, int C) {
    namespace cg = cooperative_groups;
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    __shared__ float shared_sum2[32]; // warps will be writing into shared memeory after warp-reduce
    int num_warps = blockDim.x / 32;
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int idx = blockIdx.x; // simpoy one block per row
    // the row of input that this group of threads is responsible for
    const float* x = inp + idx * C;
    // thread coarsening through the row, reduce the sum in series
    float thread_sum2 = 0.0; // stores sum(x**2)
    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        float xi = x[i];
        thread_sum2 += xi * xi;
    }
    // warp-level reduction
    float warp_sum2 = cg::reduce(warp, thread_sum2, cg::plus<float>{}); // sum(x**2)
    // store the warp-level reduction in shared memory (we could have lane_id == 0 guard but not needed)
    shared_sum2[warp_id] = warp_sum2;
    __syncthreads();
    // load results from shared memory to threads, pad with zeros for threads that are out of bounds
    warp_sum2 = (lane_id < num_warps) ? shared_sum2[lane_id] : 0.0f;
    // now reduce the warp-level reductions
    float block_sum2 = cg::reduce(warp, warp_sum2, cg::plus<float>{}); // sum(x**2)
    // mean
    block_sum2 /= C; // mean(x**2)
    float sm = block_sum2;
    float rsqrt = rsqrtf(block_sum2 + 1e-5f);
    // store the mean, no need to cache it
    if(threadIdx.x == 0 && smean != nullptr) {
        __stcs(smean + idx, sm);
    }
    if(threadIdx.x == 0 && rms != nullptr) {
        __stcs(rms + idx, rsqrt);
    }
    // final normalization and scaling by weight/bias
    float* o = out + idx * C;
    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        float n = x[i] * rsqrt;
        __stcs(o+i, n * weight[i]);
    }
}

extern "C"
__global__ void rmsnorm_backward_kernel(float* __restrict__ out, float* __restrict__ dweight, float* __restrict__ smean, float* __restrict__ rms,
 const float*  __restrict__ inp, const float*  __restrict__ delta, const float* __restrict__ weight, int N, int C) {
 	extern __shared__ float shared[]; // size = C
    namespace cg = cooperative_groups;
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    __shared__ float shared_sum2[32]; // warps will be writing into shared memeory after warp-reduce
    int num_warps = blockDim.x / 32;
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int idx = blockIdx.x; // simpoy one block per row
    
    float* dweight_shared = shared;
    
    // the row of input that this group of threads is responsible for
    const float* x = inp + idx * C;
    const float* d = delta + idx * C;
    
     // init shared memory to zero
    #pragma unroll 4
    for(int i = threadIdx.x; i < C; i+= blockDim.x){
       dweight_shared[i] = 0.0f;
    }
    __syncthreads();
    
    float b = -0.5 * powf(smean[idx] + 1e-5f, -1.5);
    
    // thread coarsening through the row, reduce the sum in series
    float drms_sum = 0.0; // stores sum(x * d * weight)
    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        drms_sum += x[i] * d[i] * weight[i];
    }
    // warp-level reduction
    float warp_sum2 = cg::reduce(warp, drms_sum, cg::plus<float>{}); // sum(x * d * weight)
    // store the warp-level reduction in shared memory (we could have lane_id == 0 guard but not needed)
    shared_sum2[warp_id] = warp_sum2;
    __syncthreads();
    // load results from shared memory to threads, pad with zeros for threads that are out of bounds
    warp_sum2 = (lane_id < num_warps) ? shared_sum2[lane_id] : 0.0f;
    // now reduce the warp-level reductions
    float block_sum2 = cg::reduce(warp, warp_sum2, cg::plus<float>{}); // sum(x * d * weight)
	
    float* o = out + idx * C;
    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        float n = rms[idx] * d[i] * weight[i] + block_sum2 * b / C * 2 * x[i];
        dweight_shared[i] = x[i] * rms[idx] * d[i];
        __stcs(o+i, n);
    }
    
    __syncthreads();

    for(int i = threadIdx.x; i < C; i+= blockDim.x) {
        atomicAddX(&dweight[i], (float)dweight_shared[i]);
    }
}

extern "C"
__global__ void rmsnorm_forward_kernel1(float *out, const float *inp, const float *weight, int N, int C)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float eps = 1e-5f;

    if (idx < N) {
        // seek to the input position inp[idx,:]
        const float *x = inp + idx * C;
        // calculate the rms (root mean square)
        float rms = 0.0f;
        for (int i = 0; i < C; i++)
        {
            rms += x[i] * x[i];
        }
        rms = sqrtf(rms / C + eps);
        // seek to the output position in out[idx,:]
        float *out_idx = out + idx * C;
        for (int i = 0; i < C; i++)
        {
            float n = x[i] / rms;              // normalized output
            float o = n * weight[i]; // scale and shift it
            out_idx[i] = o;                    // write
        }
    }
}

extern "C"
__global__ void rmsnorm_backward_kernel1(float *dinp, float *dweight,const float *dout, const float *inp, const float *weight,int N, int C)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N)
        return;

    float eps = 1e-5f;
    const float *dout_bt = dout + idx * C;
    const float *inp_bt = inp + idx * C;
    float *dinp_bt = dinp + idx * C;

    // Calculate the rms
    float rms = 0.0f;
    for (int i = 0; i < C; i++)
    {
        rms += inp_bt[i] * inp_bt[i];
    }
    rms = sqrtf(rms / C + eps);

    // First, calculate the gradients for the weights and biases
    for (int i = 0; i < C; i++)
    {
        float norm = inp_bt[i] / rms;
        atomicAdd(&dweight[i], norm * dout_bt[i]);
    }

    // Calculate drms
    float drms = 0.0f;
    for (int i = 0; i < C; i++)
    {
        drms += inp_bt[i] * dout_bt[i] * weight[i];
    }
    drms = drms * (-1.0f / (rms * rms * rms * C));

    // Now, calculate the gradients for the inputs
    for (int i = 0; i < C; i++)
    {
        float norm = inp_bt[i] / rms;
        dinp_bt[i] = dout_bt[i] * weight[i] / rms + drms * inp_bt[i];
    }
}

extern "C"
__global__ void rmsnorm_forward_kernel2(float *__restrict__ out, const float *__restrict__ inp,
                                        const float *__restrict__ weight,
                                        int N, int C)
{
    namespace cg = cooperative_groups;
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

    // Calculate thread index within grid (each warp handles one row)
    int idx = blockIdx.x * warp.meta_group_size() + warp.meta_group_rank();
    if (idx >= N)
    {
        return;
    }

    // Pointer to input row
    const float *x = inp + idx * C;

    // RMS Calculation: First calculate sum of squares
    float sum_squares = 0.0f;
    for (int i = warp.thread_rank(); i < C; i += warp.size())
    {
        sum_squares += x[i] * x[i];
    }

    // Reduce sum across threads within the warp
    sum_squares = cg::reduce(warp, sum_squares, cg::plus<float>{});

    // Calculate RMS
    float rms = sqrtf(sum_squares / C + 1e-5f);

    // Final normalization and scaling by weight/bias
    float *o = out + idx * C;
    for (int c = warp.thread_rank(); c < C; c += warp.size())
    {
        float n = __ldcs(x + c) / rms;          // Normalized output
        __stcs(o + c, n * weight[c]); // Scale, shift and write output
    }
}

extern "C"
__global__ void rmsnorm_backward_kernel2(float *__restrict__ dinp, float *__restrict__ dweight,
                                         const float *__restrict__ dout, const float *__restrict__ inp, const float *__restrict__ weight,
                                         int N, int C)
{
    namespace cg = cooperative_groups;
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

    // Calculate thread index within grid (each warp handles one row)
    int idx = blockIdx.x * warp.meta_group_size() + warp.meta_group_rank();
    if (idx >= N)
        return;

    const float eps = 1e-5f;
    const float *dout_bt = dout + idx * C;
    const float *inp_bt = inp + idx * C;
    float *dinp_bt = dinp + idx * C;

    // Compute the RMS using cooperative group reduction
    float sum_squares = 0.0f;
    for (int i = warp.thread_rank(); i < C; i += warp.size())
    {
        sum_squares += inp_bt[i] * inp_bt[i];
    }
    sum_squares = cg::reduce(warp, sum_squares, cg::plus<float>());
    float rms = sqrtf(sum_squares / C + eps);

    // Calculate the gradients for the weights and biases (accumulated across threads)
    for (int i = warp.thread_rank(); i < C; i += warp.size())
    {
        float norm = inp_bt[i] / rms;
        // Accumulate gradient for bias and weight using atomicAdd with warp-level synchronization
        atomicAdd(&dweight[i], norm * dout_bt[i]);
    }

    // Compute drms (gradient with respect to rms)
    float drms = 0.0f;
    for (int i = warp.thread_rank(); i < C; i += warp.size())
    {
        drms += inp_bt[i] * dout_bt[i] * weight[i];
    }
    drms = cg::reduce(warp, drms, cg::plus<float>());
    drms = drms * (-1.0f / (rms * rms * rms * C));

    // Step 4: Compute gradients for inputs
    for (int i = warp.thread_rank(); i < C; i += warp.size())
    {
        dinp_bt[i] = dout_bt[i] * weight[i] / rms + drms * inp_bt[i];
    }
}
