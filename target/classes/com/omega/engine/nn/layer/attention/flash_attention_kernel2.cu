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

extern "C" __global__ void flash_attention2_forward_kernel_optim(
    const float* __restrict__ query,
    const float* __restrict__ key,
    const float* __restrict__ value,
    float* __restrict__ output,
    float* __restrict__ logsumexp,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim
) {
    extern __shared__ unsigned char shm[];
    float *q_buff = (float*)shm;
    float *kv_buff = (float*)(shm + sizeof(float)*BLOCK_SIZE_R*head_dim);
    float *s_buff = (float*)(shm + sizeof(float)*BLOCK_SIZE_R*head_dim + sizeof(float)*BLOCK_SIZE_C*head_dim);
    float *o_buff = (float*)(shm + sizeof(float)*BLOCK_SIZE_R*head_dim + sizeof(float)*BLOCK_SIZE_C*head_dim + sizeof(float)*BLOCK_SIZE_R*BLOCK_SIZE_C);
    float *logsumexp_shm = (float*)(shm + sizeof(float)*BLOCK_SIZE_R*head_dim*2 + sizeof(float)*BLOCK_SIZE_C*head_dim + sizeof(float)*BLOCK_SIZE_R*BLOCK_SIZE_C);
    float *maxes_shm = (float*)(shm + sizeof(float)*BLOCK_SIZE_R*head_dim*2 + sizeof(float)*BLOCK_SIZE_C*head_dim + sizeof(float)*BLOCK_SIZE_R*BLOCK_SIZE_C + sizeof(float)*BLOCK_SIZE_R);
    float *exp_norm_coeffs = (float*)(shm + sizeof(float)*BLOCK_SIZE_R*head_dim*2 + sizeof(float)*BLOCK_SIZE_C*head_dim + sizeof(float)*BLOCK_SIZE_R*BLOCK_SIZE_C + sizeof(float)*2*BLOCK_SIZE_R);
    
    int tid = threadIdx.x;
    const int T_r = (seq_len + BLOCK_SIZE_R - 1) / BLOCK_SIZE_R; //q,o
    const int T_c = (seq_len + BLOCK_SIZE_C - 1) / BLOCK_SIZE_C; //k,v
    
    const int BATCH_IDX = blockIdx.x / (num_heads*T_r);
    const int HEAD_IDX = (blockIdx.x / T_r) % num_heads;
    const int Q_TILE_IDX = blockIdx.x % T_r;

    // thread indexing for tiled computation GEMM
    unsigned int threadRow = threadIdx.x / (BLOCK_SIZE_C / TN); // [0, ..., (BLOCK_SIZE_R/TM)-1]
    unsigned int threadCol = threadIdx.x % (BLOCK_SIZE_C / TN); // [0, ..., (BLOCK_SIZE_C/TN)-1]

    int WARP_ID = tid / 32;
    int LANE_ID = tid % 32;

    const float sqrt_head_dim = sqrtf((float)head_dim);

    const int base_hbm_offset = BATCH_IDX * (num_heads * seq_len * head_dim) + 
                                HEAD_IDX * (seq_len * head_dim) + 
                                Q_TILE_IDX * BLOCK_SIZE_R * head_dim;

    float4 *q_buff_f4 = reinterpret_cast<float4 *>(q_buff);
    // load Q tile to SHM
    #pragma unroll
    for(int x = tid; x < BLOCK_SIZE_R * head_dim / 4; x += blockDim.x)
    {
        int local_row = x*4 / (head_dim);
        int local_col = (x*4) % (head_dim);
        
        int global_seq_idx = Q_TILE_IDX * BLOCK_SIZE_R + local_row;
        
        if (global_seq_idx < seq_len && local_col + 3 < head_dim) {
            int idx = base_hbm_offset + local_row * head_dim + local_col;
            q_buff_f4[x] = __ldg(reinterpret_cast<const float4*>(&query[idx]));
        } else {
            q_buff_f4[x] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }

        if(local_col == 0 && local_row < BLOCK_SIZE_R)
        {
            logsumexp_shm[local_row] = 0.0f;
            maxes_shm[local_row] = -FLT_MAX;
            o_buff[local_row * head_dim] = 0.0f; 
        }
    }

    const int base_hbm_offset_kv = BATCH_IDX * (num_heads * seq_len * head_dim) + 
                                   HEAD_IDX * (seq_len * head_dim);

    for(int j=0; j < T_c; ++j)
    {
        // load K,V tiles into shared memory
        int kv_block_start = j * BLOCK_SIZE_C;
        
        #pragma unroll
        for(int x = tid; x < BLOCK_SIZE_C * head_dim / 4; x += blockDim.x)
        {
            int local_row = (x * 4) / head_dim;
            int local_col = (x * 4) % head_dim;
            
            int global_seq_idx = kv_block_start + local_row;
            
            if (global_seq_idx < seq_len && local_col + 3 < head_dim) {
                int idx = base_hbm_offset_kv + global_seq_idx * head_dim + local_col;

                float4 k_val_f4 = __ldg(reinterpret_cast<const float4*>(&key[idx]));

                // K transposed in shared memory
                kv_buff[(local_col + 0) * BLOCK_SIZE_C + local_row] = k_val_f4.x;
                kv_buff[(local_col + 1) * BLOCK_SIZE_C + local_row] = k_val_f4.y;
                kv_buff[(local_col + 2) * BLOCK_SIZE_C + local_row] = k_val_f4.z;
                kv_buff[(local_col + 3) * BLOCK_SIZE_C + local_row] = k_val_f4.w;
            } else if (global_seq_idx < seq_len) {
                for(int d = local_col; d < head_dim && d < local_col + 4; ++d) {
                    int idx = base_hbm_offset_kv + global_seq_idx * head_dim + d;
                    kv_buff[d * BLOCK_SIZE_C + local_row] = __ldg(&key[idx]);
                }
            } else {
                // Zero padding
                for(int d = local_col; d < local_col + 4 && d < head_dim; ++d) {
                    kv_buff[d * BLOCK_SIZE_C + local_row] = 0.0f;
                }
            }
        } 
        __syncthreads();

        float threadS[TM * TN] = {0.0f};
        float regQ[TM] = {0.0f};
        float regK[TN] = {0.0f};

        // outer loop: tile over head_dim dimension (BK chunks)
        for (unsigned int bkIdx = 0; bkIdx < head_dim; bkIdx += BK) {
            // Q is already in shared memory, K is already transposed
            // inner computation: accumulate partial dot products
            #pragma unroll
            for (unsigned int dotIdx = 0; dotIdx < BK; ++dotIdx) {
                unsigned int d = bkIdx + dotIdx;
                if (d >= head_dim) break;

                // load Q values for this thread's TM registers
                #pragma unroll
                for (unsigned int i = 0; i < TM; ++i) {
                    unsigned int qRow = threadRow * TM + i;
                    if (qRow < BLOCK_SIZE_R) {
                        regQ[i] = q_buff[qRow * head_dim + d];
                    }
                }

                // load K^T values for this thread's TN registers  
                #pragma unroll
                for (unsigned int i = 0; i < TN; ++i) {
                    unsigned int kCol = threadCol * TN + i;
                    if (kCol < BLOCK_SIZE_C) {
                        regK[i] = kv_buff[d * BLOCK_SIZE_C + kCol];
                    }
                }

                // accumulate outer product into registers
                #pragma unroll
                for (unsigned int resIdx_M = 0; resIdx_M < TM; ++resIdx_M) {
                    #pragma unroll
                    for (unsigned int resIdx_N = 0; resIdx_N < TN; ++resIdx_N) {
                        threadS[resIdx_M * TN + resIdx_N] += regQ[resIdx_M] * regK[resIdx_N];
                    }
                }
            }
        }

        // write results to shared memory and apply scaling + bounds checking
        #pragma unroll
        for (unsigned int resIdx_M = 0; resIdx_M < TM; ++resIdx_M) {
            unsigned int globalRow = threadRow * TM + resIdx_M;
            if (globalRow >= BLOCK_SIZE_R) continue;
            
            // niepotrzebne i guess
           unsigned int global_q_idx = Q_TILE_IDX * BLOCK_SIZE_R + globalRow;
            if (global_q_idx >= seq_len) continue;

            #pragma unroll
            for (unsigned int resIdx_N = 0; resIdx_N < TN; ++resIdx_N) {
                unsigned int globalCol = threadCol * TN + resIdx_N;
                if (globalCol >= BLOCK_SIZE_C) continue;

                unsigned int global_kv_idx = j * BLOCK_SIZE_C + globalCol;
                if (global_kv_idx >= seq_len) {
                    s_buff[globalRow * BLOCK_SIZE_C + globalCol] = -FLT_MAX;
                } else {
                    s_buff[globalRow * BLOCK_SIZE_C + globalCol] = 
                        threadS[resIdx_M * TN + resIdx_N] / sqrt_head_dim;
                }
            }
        }
        __syncthreads();

        // each warp processes one row
        #pragma unroll
        for(int row = WARP_ID; row < BLOCK_SIZE_R; row += (blockDim.x / 32))
        {
            int global_seq_idx = Q_TILE_IDX * BLOCK_SIZE_R + row;
            if (global_seq_idx >= seq_len) continue;

            float row_max = -FLT_MAX;
            #pragma unroll
            for(int col = LANE_ID; col < BLOCK_SIZE_C; col += 32)
            {
                row_max = fmaxf(row_max, s_buff[row * BLOCK_SIZE_C + col]);   
            }

            // warp reduction to find max
            #pragma unroll
            for(int offset = 16; offset > 0; offset >>= 1)
            {
                row_max = fmaxf(row_max, __shfl_xor_sync(0xFFFFFFFF, row_max, offset));
            }

            float new_max = 0.0f;
            float coeff = 0.0f;
            if (LANE_ID == 0)
            {
                new_max = fmaxf(maxes_shm[row], row_max);
                coeff = __expf(maxes_shm[row] - new_max);
                maxes_shm[row] = new_max;
                exp_norm_coeffs[row] = coeff;
            }

            new_max = __shfl_sync(0xFFFFFFFF, new_max, 0);
            coeff = __shfl_sync(0xFFFFFFFF, coeff, 0);

            #pragma unroll
            for(int col = LANE_ID; col < BLOCK_SIZE_C; col += 32)
            {
                float val = s_buff[row * BLOCK_SIZE_C + col];
                s_buff[row * BLOCK_SIZE_C + col] = __expf(val - new_max);
            }

            float row_sum = 0.0f;
            #pragma unroll
            for(int col = LANE_ID; col < BLOCK_SIZE_C; col += 32)
            {
                row_sum += s_buff[row * BLOCK_SIZE_C + col];
            }

            // warp reduction to find sum
            #pragma unroll
            for(int offset = 16; offset > 0; offset >>= 1)
            {
                row_sum += __shfl_xor_sync(0xFFFFFFFF, row_sum, offset);
            }

            if (LANE_ID == 0)
            {
                logsumexp_shm[row] = coeff * logsumexp_shm[row] + row_sum;
            }
        }
        __syncthreads();




        #pragma unroll
        for(int x = tid; x < BLOCK_SIZE_C * head_dim / 4; x += blockDim.x)
        {
            int local_row = (x * 4) / head_dim;
            int local_col = (x * 4) % head_dim;
            
            int global_seq_idx = kv_block_start + local_row;
            
            if (global_seq_idx < seq_len && local_col + 3 < head_dim) {
                int idx = base_hbm_offset_kv + global_seq_idx * head_dim + local_col;
                float4 v_val_f4 = __ldg(reinterpret_cast<const float4*>(&value[idx]));
                *reinterpret_cast<float4*>(&kv_buff[local_row * head_dim + local_col]) = v_val_f4;
            } else if (global_seq_idx < seq_len) {
                for(int d = local_col; d < head_dim && d < local_col + 4; ++d) {
                    int idx = base_hbm_offset_kv + global_seq_idx * head_dim + d;
                    kv_buff[local_row * head_dim + d] = __ldg(&value[idx]);
                }
            } else {
                // zero padding
                for(int d = local_col; d < local_col + 4 && d < head_dim; ++d) {
                    kv_buff[local_row * head_dim + d] = 0.0f;
                }
            }
        } 
        __syncthreads();

        float threadO[TM * BK] = {0.0f};
        #pragma unroll
        for (unsigned int resIdx_M = 0; resIdx_M < TM; ++resIdx_M) {
            unsigned int row = threadRow * TM + resIdx_M;
            if (row >= BLOCK_SIZE_R) continue;
            
            unsigned int global_seq_idx = Q_TILE_IDX * BLOCK_SIZE_R + row;
            if (global_seq_idx >= seq_len) continue;

            float coeff = exp_norm_coeffs[row];
            
            #pragma unroll
            for (unsigned int dIdx = threadCol * BK; dIdx < head_dim; dIdx += (BLOCK_SIZE_C / TN) * BK) {
                float accum[BK] = {0.0f};
                
                #pragma unroll
                for (unsigned int c = 0; c < BLOCK_SIZE_C; ++c) {
                    float s_val = s_buff[row * BLOCK_SIZE_C + c];
                    
                    #pragma unroll
                    for (unsigned int i = 0; i < BK; ++i) {
                        unsigned int d = dIdx + i;
                        if (d < head_dim) {
                            accum[i] += s_val * kv_buff[c * head_dim + d];
                        }
                    }
                }

                #pragma unroll
                for (unsigned int i = 0; i < BK; ++i) {
                    unsigned int d = dIdx + i;
                    if (d < head_dim) {
                        unsigned int idx = row * head_dim + d;
                        o_buff[idx] = o_buff[idx] * coeff + accum[i];
                    }
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for(int x = tid; x < BLOCK_SIZE_R * head_dim; x += blockDim.x)
    {
        int local_row = x / head_dim;
        int local_col = x % head_dim;
        int global_seq_idx = Q_TILE_IDX * BLOCK_SIZE_R + local_row;
        if (global_seq_idx < seq_len) {
            int idx = base_hbm_offset + local_row * head_dim + local_col;
            output[idx] = o_buff[local_row * head_dim + local_col] / logsumexp_shm[local_row];

            if(local_col == 0)
            {
                logsumexp[BATCH_IDX * num_heads * seq_len + 
                               HEAD_IDX * seq_len + 
                               global_seq_idx] = __logf(logsumexp_shm[local_row]) + maxes_shm[local_row];
            }
        }
    }
}



