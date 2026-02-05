/**
 * Flash Attention 2 - CUDA Kernel Implementation
 * Optimized for JCuda integration
 */

#include <cuda_runtime.h>

// Tile size for shared memory tiling
#define TILE_SIZE 32
#define BLOCK_SIZE_R 32
#define BLOCK_SIZE_C 32
#define BK 4
#define TM 4
#define TN 4

#define FLT_MAX 3.402823466e+38F

extern "C" __global__ void D_computation_reduction_kernel(
    const float *d_output, 
    const float *output, 
    int batch_size, 
    int num_heads, 
    int seq_len,
    int head_dim,
    float *D)
{
    extern __shared__ float shmm[];
    unsigned int tid = threadIdx.x;
    unsigned int BATCH_IDX = blockIdx.x / (num_heads * seq_len);
    unsigned int HEAD_IDX = (blockIdx.x / seq_len) % num_heads;
    unsigned int SEQ_IDX = blockIdx.x % seq_len;
    
    const int base_hbm_offset = BATCH_IDX * (num_heads * seq_len * head_dim) + 
                                HEAD_IDX * (seq_len * head_dim) + 
                                SEQ_IDX * head_dim;
    
    float thread_sum = 0.0f;
    for (int i = tid; i < head_dim; i += blockDim.x) {
        thread_sum += d_output[base_hbm_offset + i] * output[base_hbm_offset + i];
    }
    
    shmm[tid] = thread_sum;
    __syncthreads();
    
    // reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shmm[tid] += shmm[tid + stride];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        D[BATCH_IDX * num_heads * seq_len + HEAD_IDX * seq_len + SEQ_IDX] = shmm[0];
    }
}

extern "C" __global__ void flash_attention2_backward_kernel(
    const float* __restrict__ query,
    const float* __restrict__ key,
    const float* __restrict__ value,
    const float* __restrict__ output,
    const float* __restrict__ d_output,
    const float* __restrict__ logsumexp,
    const float* __restrict__ d,
    float* __restrict__ d_query,
    float* __restrict__ d_key,
    float* __restrict__ d_value,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim
) {
    extern __shared__ unsigned char shm[];
    float *q_buff = (float*)shm;
    float *k_buff = (float*)(shm + sizeof(float)*BLOCK_SIZE_R*head_dim);
    float *v_buff = (float*)(shm + sizeof(float)*BLOCK_SIZE_R*head_dim + sizeof(float)*BLOCK_SIZE_C*head_dim);
    float *d_k_buff = (float*)(shm + sizeof(float)*BLOCK_SIZE_R*head_dim + sizeof(float)*BLOCK_SIZE_C*head_dim*2);
    float *d_v_buff = (float*)(shm + sizeof(float)*BLOCK_SIZE_R*head_dim + sizeof(float)*BLOCK_SIZE_C*head_dim*3);
    float *logsumexp_d_shm = (float*)(shm + sizeof(float)*BLOCK_SIZE_R*head_dim + sizeof(float)*BLOCK_SIZE_C*head_dim*4);
    float *p_buff = (float*)(shm + sizeof(float)*BLOCK_SIZE_R*head_dim + sizeof(float)*BLOCK_SIZE_C*head_dim*4 + sizeof(float)*BLOCK_SIZE_R);
    float *d_o_buff = q_buff;

    unsigned int tid = threadIdx.x;
    unsigned int T_r = (seq_len + BLOCK_SIZE_R - 1) / BLOCK_SIZE_R;
    unsigned int T_c = (seq_len + BLOCK_SIZE_C - 1) / BLOCK_SIZE_C;

    unsigned int BATCH_IDX = blockIdx.x / (num_heads * T_c);
    unsigned int HEAD_IDX = (blockIdx.x / T_c) % num_heads;
    unsigned int KV_TILE_IDX = blockIdx.x % T_c;

    int WARP_ID = tid / 32;
    int LANE_ID = tid % 32;
    int warps_per_block = blockDim.x / 32;
    int rows_per_warp = (BLOCK_SIZE_R + warps_per_block - 1) / warps_per_block;
    int warp_start_row = WARP_ID * rows_per_warp;
    int warp_end_row = min(warp_start_row + rows_per_warp, BLOCK_SIZE_R);

    const float sqrt_head_dim = sqrtf((float)head_dim);

    const int base_hbm_offset = BATCH_IDX * (num_heads * seq_len * head_dim) + 
                                HEAD_IDX * (seq_len * head_dim) + 
                                KV_TILE_IDX * BLOCK_SIZE_C * head_dim;
    float4 *k_buff_f4 = reinterpret_cast<float4 *>(k_buff);
    float4 *v_buff_f4 = reinterpret_cast<float4 *>(v_buff);
    float4 *d_k_buff_f4 = reinterpret_cast<float4 *>(d_k_buff);
    float4 *d_v_buff_f4 = reinterpret_cast<float4 *>(d_v_buff);
    // each block handles one k,v tile, so every thread in block handles at least one element of that k,v tile
    // load kv tile into shm
    #pragma unroll
    for (int x=tid; x < BLOCK_SIZE_C * head_dim / 4; x += blockDim.x)
    {
        int local_row = (x*4) / head_dim;  // which row within the block [0, BLOCK_SIZE_C)
        int local_col = (x*4) % head_dim;  // which head dimension [0, head_dim)
        
        int global_seq_idx = KV_TILE_IDX * BLOCK_SIZE_C + local_row;
        
        // check bounds
        if (global_seq_idx < seq_len && local_col + 3 < head_dim) {
            int idx = base_hbm_offset + local_row * head_dim + local_col;
            k_buff_f4[x] = __ldg(reinterpret_cast<const float4 *>(&key[idx]));
            v_buff_f4[x] = __ldg(reinterpret_cast<const float4 *>(&value[idx]));
        } else {
            k_buff_f4[x] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            v_buff_f4[x] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }

        // also initialize the derivative buffers to zero
        d_k_buff_f4[x] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        d_v_buff_f4[x] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }

    __syncthreads();

    const int q_hbm_base = BATCH_IDX * (num_heads * seq_len * head_dim) + 
                            HEAD_IDX * (seq_len * head_dim);
    float4 *q_buff_f4 = reinterpret_cast<float4 *>(q_buff);
    // iterate over q tiles
    for (int i = 0; i < T_r; ++i)
    {
        // Load q, d_o, logsumexp into shm
        #pragma unroll
        for(int x=tid; x < BLOCK_SIZE_R * head_dim / 4; x += blockDim.x)
        {
            int local_row = (x*4) / head_dim;  // Which row within the block [0, BLOCK_SIZE_R)
            int local_col = (x*4) % head_dim;  // Which head dimension [0, head_dim)
            
            int global_seq_idx = i * BLOCK_SIZE_R + local_row;
            
            if (global_seq_idx < seq_len) {
                int q_idx = q_hbm_base +
                            global_seq_idx * head_dim + local_col;
                q_buff_f4[x] = __ldg(reinterpret_cast<const float4 *>(&query[q_idx]));


            } else {
                q_buff_f4[x] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
            if (local_col == 0) {
                logsumexp_d_shm[local_row] = logsumexp[BATCH_IDX * num_heads * seq_len + 
                                                        HEAD_IDX * seq_len + 
                                                        global_seq_idx];
            }
        }
        __syncthreads();

        // compute S_ij, P_ij
        for (int row = warp_start_row; row < warp_end_row; ++row)
        {
            int global_q_idx = i * BLOCK_SIZE_R + row;
            if (global_q_idx >= seq_len) continue;

            #pragma unroll
            for(int col = LANE_ID; col < BLOCK_SIZE_C; col += 32)
            {
                int global_kv_idx = KV_TILE_IDX * BLOCK_SIZE_C + col;
                if (global_kv_idx >= seq_len) {
                    continue;
                }

                float dot_product = 0.0f;
                // #pragma unroll
                #pragma unroll
                for (int d = 0; d < head_dim; d+=4)
                {
                    float4 q_val_f4 = *reinterpret_cast<float4*>(&q_buff[row * head_dim + d]);
                    float4 k_val_f4 = *reinterpret_cast<float4*>(&k_buff[col * head_dim + d]);

                    dot_product = __fmaf_rn(q_val_f4.x, k_val_f4.x, dot_product);
                    dot_product = __fmaf_rn(q_val_f4.y, k_val_f4.y, dot_product);
                    dot_product = __fmaf_rn(q_val_f4.z, k_val_f4.z, dot_product);
                    dot_product = __fmaf_rn(q_val_f4.w, k_val_f4.w, dot_product);

                }
                dot_product /= sqrt_head_dim; // scale
                dot_product = __expf(dot_product - logsumexp_d_shm[row]); // softmax
                p_buff[row * BLOCK_SIZE_C + col] = dot_product;
            }
        }
        __syncthreads();

        #pragma unroll
        for(int x=tid; x < BLOCK_SIZE_R * head_dim / 4; x += blockDim.x)
        {
            int local_row = (x*4) / head_dim;  // Which row within the block [0, BLOCK_SIZE_R)
            int local_col = (x*4) % head_dim;  // Which head dimension [0, head_dim)
            
            int global_seq_idx = i * BLOCK_SIZE_R + local_row;
            
            if (global_seq_idx < seq_len && local_col + 3 < head_dim) {
                int o_idx = q_hbm_base +  
                            global_seq_idx * head_dim + local_col;
                q_buff_f4[x] = __ldg(reinterpret_cast<const float4 *>(&d_output[o_idx]));
            } 
            else 
            {
                q_buff_f4[x] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }

            if (local_col == 0) {
                logsumexp_d_shm[local_row] = d[BATCH_IDX * num_heads * seq_len + 
                                                        HEAD_IDX * seq_len + 
                                                        global_seq_idx];
            }
        }
        __syncthreads();

        // compute dV_j
        #pragma unroll
        for(int x=tid; x < BLOCK_SIZE_C * head_dim; x += blockDim.x)
        {
            int local_c = x / head_dim;  // Which row within the block [0, BLOCK_SIZE_C)
            int local_h = x % head_dim;  // Which head dimension [0, head_dim)

            float d_v_sum = 0.0f;
            #pragma unroll
            for (int r=0; r < BLOCK_SIZE_R; ++r)
            {
                int global_q_idx = i * BLOCK_SIZE_R + r;
                int global_kv_idx = KV_TILE_IDX * BLOCK_SIZE_C + local_c;

                if (global_q_idx < seq_len && global_kv_idx < seq_len) {
                    float p_ij = p_buff[r * BLOCK_SIZE_C + local_c];
                    d_v_sum += p_ij * d_o_buff[r * head_dim + local_h];
                }
            }

            d_v_buff[local_c * head_dim + local_h] += d_v_sum;
        }
        __syncthreads();

        for (int row = warp_start_row; row < warp_end_row; ++row)
        {
            int global_q_idx = i * BLOCK_SIZE_R + row;
            if (global_q_idx >= seq_len) continue;

            #pragma unroll
            for(int col = LANE_ID; col < BLOCK_SIZE_C; col += 32)
            {
                int global_kv_idx = KV_TILE_IDX * BLOCK_SIZE_C + col;
                if (global_kv_idx >= seq_len) {
                    continue;
                }

                float d_p_ij = 0.0f;
                #pragma unroll
                for (int d = 0; d < head_dim; ++d)
                {
                    d_p_ij += d_o_buff[row * head_dim + d] * v_buff[col * head_dim + d]; 
                }
                p_buff[row * BLOCK_SIZE_C + col] = (d_p_ij - logsumexp_d_shm[row]) * p_buff[row * BLOCK_SIZE_C + col] / sqrt_head_dim;
            }
        }
        __syncthreads();

        // accum dQ_i
        #pragma unroll
        for(int x=tid; x < BLOCK_SIZE_R * head_dim; x += blockDim.x)
        {
            int local_r = x / head_dim;  // Which row within the block [0, BLOCK_SIZE_R)
            int local_h = x % head_dim;  // Which head dimension [0, head_dim)

            // dQ_i computation - atomics
            int global_q_idx = i * BLOCK_SIZE_R + local_r;

            if (global_q_idx < seq_len) {
                int q_idx = q_hbm_base +  
                            global_q_idx * head_dim + local_h;
                q_buff[x] = query[q_idx];
            }
            else {
                q_buff[x] = 0.0f;
            }

            float d_s_k_sum = 0.0f;
            #pragma unroll
            for (int c=0; c < BLOCK_SIZE_C; ++c)
            {
                int global_kv_idx = KV_TILE_IDX * BLOCK_SIZE_C + c;
                if (global_q_idx < seq_len && global_kv_idx < seq_len) {
                    d_s_k_sum += p_buff[local_r * BLOCK_SIZE_C + c] * k_buff[c * head_dim + local_h];
                }
            }

            if (global_q_idx < seq_len) {
                atomicAdd(&d_query[q_hbm_base + global_q_idx * head_dim + local_h], d_s_k_sum);
            }
        }
        __syncthreads();

        // accum dK_j
        #pragma unroll
        for(int x=tid; x < BLOCK_SIZE_C * head_dim; x += blockDim.x)
        {
            int local_c = x / head_dim;  // Which row within the block [0, BLOCK_SIZE_C)
            int local_h = x % head_dim;  // Which head dimension [0, head_dim)

            // dK_j computation
            int global_kv_idx = KV_TILE_IDX * BLOCK_SIZE_C + local_c;
            float d_s_q_sum = 0.0f;
            #pragma unroll
            for (int r=0; r < BLOCK_SIZE_R; ++r)
            {
                int global_q_idx = i * BLOCK_SIZE_R + r;
                if (global_q_idx < seq_len && global_kv_idx < seq_len) {
                    d_s_q_sum += p_buff[r * BLOCK_SIZE_C + local_c] * q_buff[r * head_dim + local_h];
                }
            }

            d_k_buff[local_c * head_dim + local_h] += d_s_q_sum;
        }
        __syncthreads();
    }

    #pragma unroll
    for(int x = tid; x < BLOCK_SIZE_C * head_dim; x += blockDim.x)
    {
        int local_c = x / head_dim;
        int local_h = x % head_dim;
        int global_kv_idx = KV_TILE_IDX * BLOCK_SIZE_C + local_c;
        if (global_kv_idx < seq_len) {
            int idx = base_hbm_offset + local_c * head_dim + local_h;
            d_key[idx] = d_k_buff[local_c * head_dim + local_h]; 
            d_value[idx] = d_v_buff[local_c * head_dim + local_h];
        }
    }
}