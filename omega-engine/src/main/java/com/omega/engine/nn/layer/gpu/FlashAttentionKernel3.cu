#define BLOCK 1024 

#include <cuda.h>
#include <cuda_runtime.h> 

__device__ void kernel_compute_statistics(
    float* scores,
    float* local_rowmax,
    float global_rowmax_old,
    float global_rowsum_old,
    float* global_rowmax_new,
    float* global_rowsum_new,
    int num_rows_per_tile,
    int thread_idx,
    int local_row_idx,
    int dimension)
{
    if(thread_idx % dimension == 0){ 
        //compute rowsums for S_ij
        float l_ij = 0.0f;
        float m_i_new = 0.0f;
        float l_i_new = 0.0f;
        for (int i = 0; i < num_rows_per_tile; i++){
            l_ij += scores[local_row_idx * num_rows_per_tile + i]; //l_ij doenst need to be written to SRAM
        }
        //compute new global rowmax statistics 
        m_i_new = fmax(global_rowmax_old, local_rowmax[local_row_idx]);
        global_rowmax_new[local_row_idx] = m_i_new; //Reuse the shared memory allocated to Q_i, since we dont use it anymore after computing S_ij

        //compute new global rowsum statistics
        l_i_new = expf(global_rowmax_old - m_i_new) * global_rowsum_old + expf(local_rowmax[local_row_idx] - m_i_new) * l_ij; 
        global_rowsum_new[local_row_idx] = l_i_new;  
    }
}

__device__ void kernel_reduction_max( //TODO: not actually reduction yet
    float* scores, 
    float* local_rowmax, 
    int num_rows_per_tile, 
    int dimension,  
    int thread_idx,
    int block_idx)
{
    if(thread_idx < num_rows_per_tile){ //S_ij is square, so num_rows_per_tile accounts for both row and column dimension
        float max_val = -INFINITY;
        for (int i = 0; i < num_rows_per_tile; i++){
            auto s_ij = scores[thread_idx * num_rows_per_tile + i];
            max_val = fmax(max_val, s_ij);
        }
        local_rowmax[thread_idx] = max_val; //Resue Q_i allocated memory
    }   
}

__device__ void inner_product_matmul(
    float* Q_i, 
    float* K_j, 
    float* scores, 
    int num_rows_per_block,
    int dimension, 
    int thread_idx, 
    int thread_idx_limit,
    float scaling_factor)
{
    if (thread_idx < thread_idx_limit){
        //each threads computes one output value
        float temp = 0.0f;
        int local_matrix_row_index = thread_idx / num_rows_per_block;
        for(int k = 0; k < dimension; k++){
            temp += Q_i[local_matrix_row_index * dimension + k] * K_j[(thread_idx % num_rows_per_block) * dimension + k]; //Q_i * K^T_j
        }
        scores[thread_idx] = scaling_factor * temp;
    }
}

__device__ float outer_product_matmul(
    float* scores,
    float* V_j, 
    int num_rows_per_block,
    int dimension,
    int thread_idx,
    int thread_idx_limit
    )
{
    if(thread_idx < thread_idx_limit){ //TODO: fix edge case for when last tile does not have same amount of rows
        float temp = 0.0f;
        for (int k = 0; k < num_rows_per_block; k++){
            temp += scores[(thread_idx / dimension) * num_rows_per_block + k] * V_j[k * dimension + (thread_idx % dimension)];
        }
        return temp;
    };
    return 0.0f;
}

__device__ float* shared_memory_proxy()
{
    extern __shared__ unsigned char memory[];
    return reinterpret_cast<float*>(memory);
}

extern "C"
__global__ void forward_attention_kernel(
    float* query,
    float* key,
    float* value,
    float* outputs,
    float* rowmax_statistics, 
    float* rowsum_statistics,
    int batch_size, int sequence_length, int dimension,
    int block_size,
    int num_rows_per_block,
    int num_blocks_per_sample)
{
    //SRAM
    extern __shared__ float sharedMemory[];
    //float* sharedMemory = shared_memory_proxy<float>();
    float* Q_i = &sharedMemory[0];
    float* K_j = Q_i + block_size;
    float* V_j = Q_i + 2*block_size; 
    float* scores = &sharedMemory[0]; //Reuse Q_i allocated SRAM space
    float* local_rowmax = scores + num_rows_per_block * num_rows_per_block; 
    float* global_rowmax_new = local_rowmax + num_rows_per_block; 
    float* global_rowsum_new = global_rowmax_new + num_rows_per_block; 
    
    //compute indexes
    int batch_idx = blockIdx.x; 
    int local_row_idx = threadIdx.x / dimension; 
    int col_idx = threadIdx.x % dimension; // global_col_idx == local_col_idx in this sense

    //scaling factor
    float scaling_factor = 1.0f / (sqrtf(static_cast<float>(dimension)));
	
	int onceLen = num_rows_per_block * dimension;
	
    if(batch_idx < batch_size && local_row_idx < num_rows_per_block){ 
        for(int j = 0; j < num_blocks_per_sample; j++){
            //Load K_j, V_j to SRAM
            
            int offset = batch_idx * onceLen + (j * num_rows_per_block + local_row_idx) * dimension + col_idx;
            
            K_j[threadIdx.x] = key[offset]; // K_j
            V_j[threadIdx.x] = value[offset]; // V_j - Not very coalessed for when we do our matmuls later.... 

            for(int i = 0; i < num_blocks_per_sample; i++){ //i gives us which tile we are on for Q along the row-axis

                int global_row_idx_i = i * num_rows_per_block + local_row_idx;
				
				int qidx = batch_idx * onceLen + global_row_idx_i * dimension + col_idx;
				
                //Load Q_i, m_i, l_i to SRAM - O_i is unecessary 
                Q_i[threadIdx.x] = query[qidx]; 
                
                //Compute attention scores Q_i*K^T_j
                __syncthreads(); //necessary because utilized threads all come from the first row in the tile, but some of them operate on values from other rows in the tile
                inner_product_matmul(Q_i, K_j, scores, num_rows_per_block, dimension, threadIdx.x, num_rows_per_block * num_rows_per_block, scaling_factor); 
                // __syncthreads(); 
 
                //compute statistics - brute force it for now...
                kernel_reduction_max(scores, local_rowmax, num_rows_per_block, dimension, threadIdx.x, blockIdx.x);
                __syncthreads();

                if(threadIdx.x < num_rows_per_block*num_rows_per_block){ 
                    scores[threadIdx.x] = expf(scores[threadIdx.x] - local_rowmax[threadIdx.x / num_rows_per_block]); //P_ij 
                }
                __syncthreads();

                float global_rowmax_old = rowmax_statistics[batch_idx * num_rows_per_block + global_row_idx_i];
                float global_rowsum_old = rowsum_statistics[batch_idx * num_rows_per_block + global_row_idx_i]; 
                kernel_compute_statistics(scores, local_rowmax, global_rowmax_old, global_rowsum_old, global_rowmax_new, global_rowsum_new, num_rows_per_block, threadIdx.x, local_row_idx, dimension); //pretty sure its good
                __syncthreads();
            
                //compute attention outputs (from here on out its all element-wise so we dont need to sync threads)
                auto m_i_new = global_rowmax_new[local_row_idx];

                float old_output_adjusted = (global_rowsum_old * expf(global_rowmax_old - m_i_new)) * outputs[qidx]; 
                float local_attention_adjusted = outer_product_matmul(scores, V_j, num_rows_per_block, dimension, threadIdx.x, num_rows_per_block * dimension); //TODO: num_rows_per_block*dimension doesnt account for edge_case of non-divisible total-rows
                local_attention_adjusted = expf(local_rowmax[local_row_idx] - m_i_new) * local_attention_adjusted; 

                //Write to global memory (HBM)
                outputs[qidx] = (1 / (global_rowsum_new[local_row_idx])) * (old_output_adjusted + local_attention_adjusted); 
                if(threadIdx.x < num_rows_per_block){
					int ridx = batch_idx * num_rows_per_block + (i * num_rows_per_block + threadIdx.x % num_rows_per_block);
                    rowmax_statistics[ridx] = global_rowmax_new[threadIdx.x];
                    rowsum_statistics[ridx] = global_rowsum_new[threadIdx.x]; 
                }
            }
        }
    }
}


