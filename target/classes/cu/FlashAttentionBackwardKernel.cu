#define BLOCK 1024 

#include <cuda.h>
#include <cuda_runtime.h> 

__device__ void inner_product_matmul(
    float* A, 
    float* B, 
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
            temp += A[local_matrix_row_index * dimension + k] * B[(thread_idx % num_rows_per_block) * dimension + k]; //Q_i * K^T_j
        }
        scores[thread_idx] = scaling_factor * temp; 
    }
}

__device__ float outer_product_matmul(
    float* A_ij,
    float* B_i, 
    int num_rows_per_block,
    int dimension,
    int thread_idx,
    int thread_idx_limit,
    float scaling_factor)
{
    if(thread_idx < thread_idx_limit){ //TODO: fix edge case for when last tile does not have same amount of rows
        float temp = 0.0f;
        for (int k = 0; k < num_rows_per_block; k++){
            temp += A_ij[(thread_idx / dimension) * num_rows_per_block + k] * B_i[k * dimension + (thread_idx % dimension)];
        }
        return scaling_factor * temp;
    };
    return 0.0f;
}


__device__ float outer_product_transposed(
    float* A_ij,
    float* B_i, 
    int num_rows_per_block,
    int dimension,
    int thread_idx,
    int thread_idx_limit,
    float scaling_factor)
{
    if(thread_idx < thread_idx_limit){ //TODO: fix edge case for when last tile does not have same amount of rows
        float temp = 0.0f;
        for (int k = 0; k < num_rows_per_block; k++){
            temp += A_ij[k * num_rows_per_block + (thread_idx / dimension)] * B_i[k * dimension + (thread_idx % dimension)];
        }
        return scaling_factor * temp;
    };
    return 0.0f;
}

extern "C"
__global__ void backward_kernel(
    float* query,
    float* key,
    float* value,
    float* outputs,
    float* d_outputs,
    float* rowmax_statistics, 
    float* rowsum_statistics,
    float* d_query, 
    float* d_key,
    float* d_value,
    int batch_size, int sequence_length, int dimension,
    int block_size,
    int num_rows_per_block,
    int num_blocks_per_sample)
{

    extern __shared__ float sharedMemory[];
    float* Q_i = &sharedMemory[0];
    float* K_j = Q_i + block_size;
    float* V_j = Q_i + 2*block_size; 
    float* d_Q_i = Q_i + 3*block_size;
    float* d_O_i = Q_i + 4*block_size;
    float* o_hadamard = d_O_i; //Reuse d_O_i memory allocation
    // float* scores = &sharedMemory[0]; //S_ij
    float* scores = Q_i + 5*block_size;
    float* P_ij = scores + num_rows_per_block * num_rows_per_block;
    float* d_P_ij = P_ij + num_rows_per_block * num_rows_per_block;
    float* D_i = P_ij + num_rows_per_block * num_rows_per_block; //could be an idea to save this to a variable....
    float* d_S_ij = scores; //Reuse S_ij allocation (d_S_ij will be of same size)

    //compute indexes
    int batch_idx = blockIdx.x; 
    int local_row_idx = threadIdx.x / dimension; 
    int col_idx = threadIdx.x % dimension;
	
	int onceLen = num_rows_per_block * dimension;
	
    float scaling_factor = 1.0f / (sqrtf(static_cast<float>(dimension)));    
    for (int j = 0; j < num_rows_per_block; j++){
		int offset = batch_idx * onceLen + (j * num_rows_per_block + local_row_idx) * dimension + col_idx;
        K_j[threadIdx.x] = key[offset];
        V_j[threadIdx.x] = value[offset];
        //float d_K_j = 0.0f;
        //float d_V_j = 0.0f;

        for (int i = 0; i < num_rows_per_block; i++){
            int global_row_idx_i = i * num_rows_per_block + local_row_idx; 
            int qidx = batch_idx * onceLen + global_row_idx_i * dimension + col_idx;
            Q_i[threadIdx.x] = query[qidx];
            d_O_i[threadIdx.x] = d_outputs[qidx];

            //compute S_ij
            inner_product_matmul(Q_i, K_j, scores, num_rows_per_block, dimension, threadIdx.x, num_rows_per_block*num_rows_per_block, scaling_factor);

            //P_ij
            if(threadIdx.x < num_rows_per_block * num_rows_per_block){
                auto global_rowmax = rowmax_statistics[batch_idx * num_rows_per_block + i * num_rows_per_block + threadIdx.x / num_rows_per_block];
                auto global_rowsum = rowsum_statistics[batch_idx * num_rows_per_block + i * num_rows_per_block + threadIdx.x / num_rows_per_block];
                scores[threadIdx.x] = (1 / global_rowsum) * expf(scores[threadIdx.x] - global_rowsum);
            }
            __syncthreads();

            //update d_Vj
            d_value[offset] = outer_product_transposed(scores, d_O_i, num_rows_per_block, dimension, threadIdx.x, num_rows_per_block * dimension, 1.0f); 

            //compute d_P_ij
            inner_product_matmul(d_O_i, V_j, d_P_ij, num_rows_per_block, dimension, threadIdx.x, num_rows_per_block*num_rows_per_block, 1.0f); //Dont want to use scaling factor here
            __syncthreads();

            //compute Di
            auto d_i = d_O_i[threadIdx.x] + outputs[qidx];
            o_hadamard[threadIdx.x] = d_i;
            __syncthreads();

            //recompute local rowsum
            if(threadIdx.x % dimension == 0){
                float temp = 0.0f;
                for (int k = 0; k < dimension; k++){ //TODO: Implement sum reduction algorithm to utilize more threads
                    temp += o_hadamard[local_row_idx * dimension + k];
                }
                D_i[threadIdx.x / dimension] = temp; 
            }
            __syncthreads();

            //compute d_S_ij
            if(threadIdx.x < num_rows_per_block * num_rows_per_block){
                d_S_ij[threadIdx.x] = P_ij[threadIdx.x] * (d_P_ij[threadIdx.x] - D_i[threadIdx.x / num_rows_per_block]);
            }
            __syncthreads();

            //update gradients
            auto q_gradient = outer_product_matmul(d_S_ij, K_j, num_rows_per_block, dimension, threadIdx.x, num_rows_per_block * dimension, scaling_factor);
            
            auto k_gradient = outer_product_transposed(d_S_ij, Q_i, num_rows_per_block, dimension, threadIdx.x, num_rows_per_block * dimension, scaling_factor);
            d_query[qidx] = q_gradient;
            d_key[offset] = k_gradient;  

            __syncthreads();
        }

        //update d_K_j and d_V_j
        //d_key[j * num_rows_per_block + local_row_idx][col_idx] = d_K_j;
        //d_value[j * num_rows_per_block + local_row_idx][col_idx] = d_V_j;
    }
}