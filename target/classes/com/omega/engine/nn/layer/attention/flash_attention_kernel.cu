/**
 * Flash Attention 2 - CUDA Kernel Implementation
 * Optimized for JCuda integration
 */

#include <cuda_runtime.h>

// Tile size for shared memory tiling
#define TILE_SIZE 32

// Device inline functions for warp-level reductions
__device__ inline float warpReduceMax(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val = max(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

__device__ inline float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

/**
 * Flash Attention 2 Forward Kernel
 *
 * Parameters:
 * Q:     Query tensor    [batch, heads, seq_len, head_dim]
 * K:     Key tensor      [batch, heads, seq_len, head_dim]
 * V:     Value tensor    [batch, heads, seq_len, head_dim]
 * out:   Output tensor   [batch, heads, seq_len, head_dim]
 *
 * scale: Scaling factor (typically 1/sqrt(head_dim))
 * batch_size, num_heads, seq_len, head_dim: tensor dimensions
 */
extern "C"
__global__ void flashAttentionForward(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ out,
    const float scale,
    const int batch_size,
    const int num_heads,
    const int seq_len,
    const int head_dim) {

    // Shared memory for tiles
    __shared__ float Q_tile[TILE_SIZE][TILE_SIZE];
    __shared__ float K_tile[TILE_SIZE][TILE_SIZE];
    __shared__ float V_tile[TILE_SIZE][TILE_SIZE];

    // Compute global indices
    int batch_idx = blockIdx.z;
    int head_idx = blockIdx.y;
    int seq_block = blockIdx.x;

    int seq_idx = seq_block * TILE_SIZE + threadIdx.y;
    int head_dim_idx = threadIdx.x;

    // Only proceed if this thread is within valid bounds
    if (batch_idx >= batch_size || head_idx >= num_heads ||
        seq_idx >= seq_len || head_dim_idx >= head_dim) {
        return;
    }

    // Compute base indices for this batch and head
    int batch_head_offset = (batch_idx * num_heads + head_idx) * seq_len * head_dim;

    // Register storage for output accumulation
    float acc[TILE_SIZE];
    for (int i = 0; i < TILE_SIZE; i++) {
        acc[i] = 0.0f;
    }

    // For online softmax: track running maximum and sum
    float row_max = -INFINITY;
    float row_sum = 0.0f;

    // Iterate over K tiles in outer loop (Flash Attention key optimization)
    int num_k_tiles = (seq_len + TILE_SIZE - 1) / TILE_SIZE;

    for (int k_tile_idx = 0; k_tile_idx < num_k_tiles; k_tile_idx++) {

        // Load K tile (transposed for efficient matrix multiplication)
        // K shape: [seq_len, head_dim], load as [head_dim, seq_len]
        int k_row = k_tile_idx * TILE_SIZE + threadIdx.x;
        int k_col = threadIdx.y;
        int k_idx = batch_head_offset + k_row * head_dim + k_col;

        float k_val = 0.0f;
        if (k_row < seq_len && k_col < head_dim) {
            k_val = K[k_idx];
        }
        K_tile[threadIdx.x][threadIdx.y] = k_val;

        // Load V tile
        // V shape: [seq_len, head_dim]
        int v_row = k_tile_idx * TILE_SIZE + threadIdx.y;
        int v_col = threadIdx.x;
        int v_idx = batch_head_offset + v_row * head_dim + v_col;

        float v_val = 0.0f;
        if (v_row < seq_len && v_col < head_dim) {
            v_val = V[v_idx];
        }
        V_tile[threadIdx.y][threadIdx.x] = v_val;

        __syncthreads();

        // Load Q tile
        int q_row = seq_block * TILE_SIZE + threadIdx.y;
        int q_col = threadIdx.x;
        int q_idx = batch_head_offset + q_row * head_dim + q_col;

        float q_val = 0.0f;
        if (q_row < seq_len && q_col < head_dim) {
            q_val = Q[q_idx];
        }
        Q_tile[threadIdx.y][threadIdx.x] = q_val;

        __syncthreads();

        // Compute S = Q * K^T for this tile
        float S[TILE_SIZE];
        #pragma unroll
        for (int i = 0; i < TILE_SIZE; i++) {
            S[i] = 0.0f;
            #pragma unroll
            for (int j = 0; j < TILE_SIZE; j++) {
                S[i] += Q_tile[threadIdx.y][j] * K_tile[j][threadIdx.x];
            }
            S[i] *= scale;
        }

        // Online softmax update
        // Find new maximum across all tiles processed so far
        float new_max = row_max;
        #pragma unroll
        for (int i = 0; i < TILE_SIZE; i++) {
            int k_seq_idx = k_tile_idx * TILE_SIZE + threadIdx.x;
            if (k_seq_idx < seq_len) {
                new_max = max(new_max, S[i]);
            }
        }

        new_max = warpReduceMax(new_max);

        // Compute rescaling factors
        float old_scale = expf(row_max - new_max);

        // Update accumulated output with rescaling
        float tile_sum = 0.0f;
        #pragma unroll
        for (int i = 0; i < TILE_SIZE; i++) {
            int k_seq_idx = k_tile_idx * TILE_SIZE + threadIdx.x;
            if (k_seq_idx < seq_len) {
                float p = expf(S[i] - new_max);
                tile_sum += p;

                // Rescale previous accumulation
                acc[i] = acc[i] * old_scale;

                // Add new contribution: p * V
                #pragma unroll
                for (int j = 0; j < TILE_SIZE; j++) {
                    acc[i] += p * V_tile[threadIdx.x][j];
                }
            }
        }

        // Update row statistics
        float reduced_tile_sum = warpReduceSum(tile_sum);
        row_sum = row_sum * old_scale + reduced_tile_sum;
        row_max = new_max;

        __syncthreads();
    }

    // Normalize and write output
    int out_idx = batch_head_offset + seq_idx * head_dim + head_dim_idx;
    if (seq_idx < seq_len && head_dim_idx < head_dim) {
        out[out_idx] = acc[threadIdx.y] / row_sum;
    }
}

/**
 * Standard Scaled Dot-Product Attention Kernel
 * Used for comparison and verification
 */
extern "C" __global__ void standardAttentionForward(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ out,
    const float scale,
    const int batch_size,
    const int num_heads,
    const int seq_len,
    const int head_dim) {

    int batch_idx = blockIdx.z;
    int head_idx = blockIdx.y;
    int seq_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch_idx >= batch_size || head_idx >= num_heads || seq_idx >= seq_len) {
        return;
    }

    int batch_head_offset = (batch_idx * num_heads + head_idx) * seq_len * head_dim;

    // Compute attention scores
    for (int head_dim_idx = 0; head_dim_idx < head_dim; head_dim_idx++) {
        float sum = 0.0f;

        for (int k_idx = 0; k_idx < seq_len; k_idx++) {
            // Compute Q * K^T
            float qk = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                int q_idx = batch_head_offset + seq_idx * head_dim + d;
                int k_idx_actual = batch_head_offset + k_idx * head_dim + d;
                qk += Q[q_idx] * K[k_idx_actual];
            }

            // Apply scaling and softmax (simplified - needs proper softmax across sequence)
            float attn_weight = expf(qk * scale);

            // Accumulate V weighted by attention
            int v_idx = batch_head_offset + k_idx * head_dim + head_dim_idx;
            sum += attn_weight * V[v_idx];
        }

        // Write output
        int out_idx = batch_head_offset + seq_idx * head_dim + head_dim_idx;
        out[out_idx] = sum;
    }
}

/**
 * Flash Attention 2 Backward Kernel
 *
 * This kernel computes gradients for Q, K, V using the rematerialization technique.
 * It recomputes the forward pass values on-the-fly to save memory.
 *
 * Parameters:
 * Q:           Query tensor [batch, heads, seq_len, head_dim]
 * K:           Key tensor   [batch, heads, seq_len, head_dim]
 * V:           Value tensor [batch, heads, seq_len, head_dim]
 * Output:      Forward output [batch, heads, seq_len, head_dim]
 * GradOutput:  Gradient w.r.t. output [batch, heads, seq_len, head_dim]
 * GradQ:       Output gradient w.r.t. Q [batch, heads, seq_len, head_dim]
 * GradK:       Output gradient w.r.t. K [batch, heads, seq_len, head_dim]
 * GradV:       Output gradient w.r.t. V [batch, heads, seq_len, head_dim]
 * scale:       Scaling factor
 */
extern "C" __global__ void flashAttentionBackward(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    const float* __restrict__ Output,
    const float* __restrict__ GradOutput,
    float* __restrict__ GradQ,
    float* __restrict__ GradK,
    float* __restrict__ GradV,
    const float scale,
    const int batch_size,
    const int num_heads,
    const int seq_len,
    const int head_dim) {

    // Shared memory for tiles
    __shared__ float Q_tile[TILE_SIZE][TILE_SIZE];
    __shared__ float K_tile[TILE_SIZE][TILE_SIZE];
    __shared__ float V_tile[TILE_SIZE][TILE_SIZE];
    __shared__ float Output_tile[TILE_SIZE][TILE_SIZE];
    __shared__ float GradOutput_tile[TILE_SIZE][TILE_SIZE];

    // Compute global indices
    int batch_idx = blockIdx.z;
    int head_idx = blockIdx.y;
    int seq_block = blockIdx.x;

    int seq_idx = seq_block * TILE_SIZE + threadIdx.y;
    int head_dim_idx = threadIdx.x;

    // Boundary check
    if (batch_idx >= batch_size || head_idx >= num_heads ||
        seq_idx >= seq_len || head_dim_idx >= head_dim) {
        return;
    }

    // Compute base indices for this batch and head
    int batch_head_offset = (batch_idx * num_heads + head_idx) * seq_len * head_dim;

    // Initialize gradient accumulators
    float grad_q_accum = 0.0f;
    float grad_k_accum = 0.0f;
    float grad_v_accum = 0.0f;

    // Recompute forward pass to get attention weights
    // We need to recompute because we didn't store intermediate values
    int num_k_tiles = (seq_len + TILE_SIZE - 1) / TILE_SIZE;

    // First pass: recompute forward statistics and gradients
    for (int k_tile_idx = 0; k_tile_idx < num_k_tiles; k_tile_idx++) {

        // Load K tile
        int k_row = k_tile_idx * TILE_SIZE + threadIdx.x;
        int k_col = threadIdx.y;
        int k_idx = batch_head_offset + k_row * head_dim + k_col;
        float k_val = (k_row < seq_len && k_col < head_dim) ? K[k_idx] : 0.0f;
        K_tile[threadIdx.x][threadIdx.y] = k_val;

        // Load V tile
        int v_row = k_tile_idx * TILE_SIZE + threadIdx.y;
        int v_col = threadIdx.x;
        int v_idx = batch_head_offset + v_row * head_dim + v_col;
        float v_val = (v_row < seq_len && v_col < head_dim) ? V[v_idx] : 0.0f;
        V_tile[threadIdx.y][threadIdx.x] = v_val;

        // Load Output tile (from forward pass)
        int out_row = seq_block * TILE_SIZE + threadIdx.y;
        int out_col = threadIdx.x;
        int out_idx = batch_head_offset + out_row * head_dim + out_col;
        float out_val = (out_row < seq_len && out_col < head_dim) ? Output[out_idx] : 0.0f;
        Output_tile[threadIdx.y][threadIdx.x] = out_val;

        // Load GradOutput tile
        int grad_out_row = seq_block * TILE_SIZE + threadIdx.y;
        int grad_out_col = threadIdx.x;
        int grad_out_idx = batch_head_offset + grad_out_row * head_dim + grad_out_col;
        float grad_out_val = (grad_out_row < seq_len && grad_out_col < head_dim) ? GradOutput[grad_out_idx] : 0.0f;
        GradOutput_tile[threadIdx.y][threadIdx.x] = grad_out_val;

        // Load Q tile
        int q_row = seq_block * TILE_SIZE + threadIdx.y;
        int q_col = threadIdx.x;
        int q_idx = batch_head_offset + q_row * head_dim + q_col;
        float q_val = (q_row < seq_len && q_col < head_dim) ? Q[q_idx] : 0.0f;
        Q_tile[threadIdx.y][threadIdx.x] = q_val;

        __syncthreads();

        // Compute S = Q * K^T for this tile (recompute forward scores)
        float S[TILE_SIZE];
        #pragma unroll
        for (int i = 0; i < TILE_SIZE; i++) {
            S[i] = 0.0f;
            #pragma unroll
            for (int j = 0; j < TILE_SIZE; j++) {
                S[i] += Q_tile[threadIdx.y][j] * K_tile[j][threadIdx.x];
            }
            S[i] *= scale;
        }

        // Compute online softmax statistics (recompute forward pass)
        float row_max = -INFINITY;
        #pragma unroll
        for (int i = 0; i < TILE_SIZE; i++) {
            int k_seq_idx = k_tile_idx * TILE_SIZE + threadIdx.x;
            if (k_seq_idx < seq_len) {
                row_max = max(row_max, S[i]);
            }
        }
        row_max = warpReduceMax(row_max);

        float row_sum = 0.0f;
        #pragma unroll
        for (int i = 0; i < TILE_SIZE; i++) {
            int k_seq_idx = k_tile_idx * TILE_SIZE + threadIdx.x;
            if (k_seq_idx < seq_len) {
                row_sum += expf(S[i] - row_max);
            }
        }
        row_sum = warpReduceSum(row_sum);

        // Compute attention weights: P = softmax(S)
        float P[TILE_SIZE];
        #pragma unroll
        for (int i = 0; i < TILE_SIZE; i++) {
            int k_seq_idx = k_tile_idx * TILE_SIZE + threadIdx.x;
            P[i] = (k_seq_idx < seq_len) ? expf(S[i] - row_max) / row_sum : 0.0f;
        }

        // Compute gradient w.r.t. V: dV = P^T * GradOutput
        // Each thread accumulates contribution to grad_v
        #pragma unroll
        for (int i = 0; i < TILE_SIZE; i++) {
            int k_seq_idx = k_tile_idx * TILE_SIZE + threadIdx.x;
            if (k_seq_idx < seq_len) {
                grad_v_accum += P[i] * GradOutput_tile[threadIdx.y][threadIdx.x];
            }
        }

        // Compute gradient w.r.t. attention weights: dP = GradOutput * V^T
        float dP[TILE_SIZE];
        #pragma unroll
        for (int i = 0; i < TILE_SIZE; i++) {
            dP[i] = 0.0f;
            #pragma unroll
            for (int j = 0; j < TILE_SIZE; j++) {
                dP[i] += GradOutput_tile[threadIdx.y][j] * V_tile[threadIdx.x][j];
            }
        }

        // Compute gradient w.r.t. S: dS = dP - P * sum(dP)
        float dP_sum = 0.0f;
        #pragma unroll
        for (int i = 0; i < TILE_SIZE; i++) {
            int k_seq_idx = k_tile_idx * TILE_SIZE + threadIdx.x;
            if (k_seq_idx < seq_len) {
                dP_sum += dP[i];
            }
        }
        dP_sum = warpReduceSum(dP_sum);

        float dS[TILE_SIZE];
        #pragma unroll
        for (int i = 0; i < TILE_SIZE; i++) {
            int k_seq_idx = k_tile_idx * TILE_SIZE + threadIdx.x;
            if (k_seq_idx < seq_len) {
                dS[i] = (dP[i] - P[i] * dP_sum) / row_sum;
            }
        }

        // Accumulate gradients for Q and K
        // grad_q: each thread contributes to its head_dim position
        // grad_k: need to accumulate across sequence dimension

        // Gradient w.r.t. Q: dQ = dS * K
        #pragma unroll
        for (int i = 0; i < TILE_SIZE; i++) {
            int k_seq_idx = k_tile_idx * TILE_SIZE + threadIdx.x;
            if (k_seq_idx < seq_len) {
                grad_q_accum += dS[i] * K_tile[threadIdx.x][threadIdx.y];
            }
        }

        // Gradient w.r.t. K: dK = Q^T * dS
        #pragma unroll
        for (int i = 0; i < TILE_SIZE; i++) {
            int k_seq_idx = k_tile_idx * TILE_SIZE + threadIdx.x;
            if (k_seq_idx < seq_len) {
                grad_k_accum += Q_tile[threadIdx.y][threadIdx.x] * dS[i];
            }
        }

        __syncthreads();
    }

    // Write gradients (atomic add might be needed if multiple threads write to same location)
    int q_idx = batch_head_offset + seq_idx * head_dim + head_dim_idx;
    atomicAdd(&GradQ[q_idx], grad_q_accum * scale);

    int k_idx = batch_head_offset + seq_idx * head_dim + head_dim_idx;
    atomicAdd(&GradK[k_idx], grad_k_accum * scale);

    int v_idx = batch_head_offset + seq_idx * head_dim + head_dim_idx;
    atomicAdd(&GradV[v_idx], grad_v_accum);
}


/**
 * Simplified backward kernel for single-thread per element (easier to verify)
 * This is less efficient but clearer for understanding the gradient computation
 */
extern "C" __global__ void flashAttentionBackwardSimple(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    const float* __restrict__ Output,
    const float* __restrict__ GradOutput,
    float* __restrict__ GradQ,
    float* __restrict__ GradK,
    float* __restrict__ GradV,
    const float scale,
    const int batch_size,
    const int num_heads,
    const int seq_len,
    const int head_dim) {

    int batch_idx = blockIdx.z;
    int head_idx = blockIdx.y;
    int seq_idx = blockIdx.x;
    int head_dim_idx = threadIdx.x;

    if (batch_idx >= batch_size || head_idx >= num_heads ||
        seq_idx >= seq_len || head_dim_idx >= head_dim) {
        return;
    }

    int batch_head_offset = (batch_idx * num_heads + head_idx) * seq_len * head_dim;
    int base_idx = batch_head_offset + seq_idx * head_dim + head_dim_idx;

    // Compute gradient for V
    // dV_ij = sum_k(P_ki * GradOutput_kj) where P is attention weight matrix
    float grad_v = 0.0f;
    for (int k = 0; k < seq_len; k++) {
        // Recompute attention score
        float s = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            int q_idx = batch_head_offset + seq_idx * head_dim + d;
            int k_idx = batch_head_offset + k * head_dim + d;
            s += Q[q_idx] * K[k_idx];
        }
        s *= scale;

        // Compute softmax (need all scores for this row)
        float max_s = s;
        for (int k2 = 0; k2 < seq_len; k2++) {
            float s2 = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                int q_idx = batch_head_offset + seq_idx * head_dim + d;
                int k_idx = batch_head_offset + k2 * head_dim + d;
                s2 += Q[q_idx] * K[k_idx];
            }
            s2 *= scale;
            max_s = max(max_s, s2);
        }

        float sum_exp = 0.0f;
        for (int k2 = 0; k2 < seq_len; k2++) {
            float s2 = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                int q_idx = batch_head_offset + seq_idx * head_dim + d;
                int k_idx = batch_head_offset + k2 * head_dim + d;
                s2 += Q[q_idx] * K[k_idx];
            }
            s2 *= scale;
            sum_exp += expf(s2 - max_s);
        }

        float p = expf(s - max_s) / sum_exp;
        int grad_out_idx = batch_head_offset + k * head_dim + head_dim_idx;
        grad_v += p * GradOutput[grad_out_idx];
    }

    GradV[base_idx] = grad_v;

    // Compute gradient for Q
    float grad_q = 0.0f;
    for (int k = 0; k < seq_len; k++) {
        // Similar recomputation for attention weights
        float s = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            int q_idx = batch_head_offset + seq_idx * head_dim + d;
            int k_idx = batch_head_offset + k * head_dim + d;
            s += Q[q_idx] * K[k_idx];
        }
        s *= scale;

        // Compute softmax
        float max_s = s;
        for (int k2 = 0; k2 < seq_len; k2++) {
            float s2 = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                int q_idx = batch_head_offset + seq_idx * head_dim + d;
                int k_idx = batch_head_offset + k2 * head_dim + d;
                s2 += Q[q_idx] * K[k_idx];
            }
            s2 *= scale;
            max_s = max(max_s, s2);
        }

        float sum_exp = 0.0f;
        for (int k2 = 0; k2 < seq_len; k2++) {
            float s2 = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                int q_idx = batch_head_offset + seq_idx * head_dim + d;
                int k_idx = batch_head_offset + k2 * head_dim + d;
                s2 += Q[q_idx] * K[k_idx];
            }
            s2 *= scale;
            sum_exp += expf(s2 - max_s);
        }

        float p = expf(s - max_s) / sum_exp;

        // Gradient contribution from K
        int k_idx_base = batch_head_offset + k * head_dim + head_dim_idx;
        grad_q += p * GradOutput[base_idx] * K[k_idx_base] * scale;
    }

    GradQ[base_idx] = grad_q;

    // Compute gradient for K
    float grad_k = 0.0f;
    for (int q = 0; q < seq_len; q++) {
        float s = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            int q_idx = batch_head_offset + q * head_dim + d;
            int k_idx = batch_head_offset + seq_idx * head_dim + d;
            s += Q[q_idx] * K[k_idx];
        }
        s *= scale;

        // Compute softmax
        float max_s = s;
        for (int k2 = 0; k2 < seq_len; k2++) {
            float s2 = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                int q_idx = batch_head_offset + q * head_dim + d;
                int k_idx = batch_head_offset + k2 * head_dim + d;
                s2 += Q[q_idx] * K[k_idx];
            }
            s2 *= scale;
            max_s = max(max_s, s2);
        }

        float sum_exp = 0.0f;
        for (int k2 = 0; k2 < seq_len; k2++) {
            float s2 = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                int q_idx = batch_head_offset + q * head_dim + d;
                int k_idx = batch_head_offset + k2 * head_dim + d;
                s2 += Q[q_idx] * K[k_idx];
            }
            s2 *= scale;
            sum_exp += expf(s2 - max_s);
        }

        float p = expf(s - max_s) / sum_exp;

        // Gradient contribution from Q
        int q_idx_base = batch_head_offset + q * head_dim + head_dim_idx;
        grad_k += p * GradOutput[q_idx_base] * Q[q_idx_base] * scale;
    }

    GradK[base_idx] = grad_k;
}


/**
 * Kernel to compute attention statistics (for debugging/profiling)
 */
extern "C" __global__ void computeAttentionStats(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* stats,
    const float scale,
    const int batch_size,
    const int num_heads,
    const int seq_len,
    const int head_dim) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid == 0) {
        // Store some basic stats in the stats array
        // stats[0]: total elements in Q
        // stats[1]: memory per attention matrix (in bytes)
        // stats[2]: total operations for standard attention
        // stats[3]: total operations for flash attention (approximate)

        stats[0] = (float)(batch_size * num_heads * seq_len * head_dim);
        stats[1] = (float)(batch_size * num_heads * seq_len * seq_len * sizeof(float));
        stats[2] = (float)(batch_size * num_heads * seq_len * seq_len * head_dim * 2);
        stats[3] = (float)(batch_size * num_heads * seq_len * head_dim * 4);
    }
}



template<int BLOCK_SIZE_R, int BLOCK_SIZE_C, int BK, int TM, int TN>
__global__ void flash_attention2_forward_kernel_optim(
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