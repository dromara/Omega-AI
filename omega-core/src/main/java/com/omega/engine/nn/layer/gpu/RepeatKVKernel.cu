#define BLOCK 1024 
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

extern "C"
__global__ void repeat_once_forward(float *out, const float *x,
 int B, int T, int num_kv_heads, int num_queries_per_kv, int head_dim){

    int b = blockIdx.x;
    int t = blockIdx.y;
    int kv_head = blockIdx.z;
    int d = threadIdx.x;

    if (d < head_dim){
        // Each thread will now handle one specific kv_head and repeat it
        for (int rep = 0; rep < num_queries_per_kv; rep++){
            int out_head = kv_head * num_queries_per_kv + rep;

            // Calculate input and output indices
            int in_index = ((b * T + t) * num_kv_heads + kv_head) * head_dim + d;
            int out_index = ((b * T + t) * (num_kv_heads * num_queries_per_kv) + out_head) * head_dim + d;

            // Copy values for both k and v
            out[out_index] = x[in_index];
        }
    }
}

extern "C"
__global__ void repeat_kv_forward(float *k_out, float *v_out, const float *k, const float *v,
 int B, int T, int num_kv_heads, int num_queries_per_kv, int head_dim){

    int b = blockIdx.x;
    int t = blockIdx.y;
    int kv_head = blockIdx.z;
    int d = threadIdx.x;

    if (d < head_dim){
        // Each thread will now handle one specific kv_head and repeat it
        for (int rep = 0; rep < num_queries_per_kv; rep++){
            int out_head = kv_head * num_queries_per_kv + rep;

            // Calculate input and output indices
            int in_index = ((b * T + t) * num_kv_heads + kv_head) * head_dim + d;
            int out_index = ((b * T + t) * (num_kv_heads * num_queries_per_kv) + out_head) * head_dim + d;

            // Copy values for both k and v
            k_out[out_index] = k[in_index];
            v_out[out_index] = v[in_index];
        }
    }
}

extern "C"
__global__ void repeat_once_backward(float *diff, const float *delta,
    int B, int T, int num_kv_heads, int num_queries_per_kv, int head_dim){
    
    int b = blockIdx.x;
    int t = blockIdx.y;
    int kv_head = blockIdx.z; // Each block processes a specific kv_head
    int d = threadIdx.x;      // handles individual elements in the head_dim

    if (d < head_dim) // Guard against over-indexing
    {
        // accumulation variables for dk and dv
        float d_accum = 0.0f;

        //  gradients from repeated queries
        for (int rep = 0; rep < num_queries_per_kv; rep++)
        {
            int out_head = kv_head * num_queries_per_kv + rep;
            int in_index = ((b * T + t) * num_kv_heads * num_queries_per_kv + out_head) * head_dim + d;

            // Sum gradients from the repeated heads
            d_accum += delta[in_index];
        }

        // Writing accumulated gradients to dk and dv
        int out_index = ((b * T + t) * num_kv_heads + kv_head) * head_dim + d;
        diff[out_index] = d_accum;
    }
}

extern "C"
__global__ void repeat_kv_backward(float *dk, float *dv, const float *dk_rep, const float *dv_rep,
    int B, int T, int num_kv_heads, int num_queries_per_kv, int head_dim){
    
    int b = blockIdx.x;
    int t = blockIdx.y;
    int kv_head = blockIdx.z; // Each block processes a specific kv_head
    int d = threadIdx.x;      // handles individual elements in the head_dim

    if (d < head_dim) // Guard against over-indexing
    {
        // accumulation variables for dk and dv
        float dk_accum = 0.0f;
        float dv_accum = 0.0f;

        //  gradients from repeated queries
        for (int rep = 0; rep < num_queries_per_kv; rep++)
        {
            int out_head = kv_head * num_queries_per_kv + rep;
            int in_index = ((b * T + t) * num_kv_heads * num_queries_per_kv + out_head) * head_dim + d;

            // Sum gradients from the repeated heads
            dk_accum += dk_rep[in_index];
            dv_accum += dv_rep[in_index];
        }

        // Writing accumulated gradients to dk and dv
        int out_index = ((b * T + t) * num_kv_heads + kv_head) * head_dim + d;
        dk[out_index] = dk_accum;
        dv[out_index] = dv_accum;
    }
}