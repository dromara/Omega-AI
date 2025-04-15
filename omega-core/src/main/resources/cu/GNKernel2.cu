#define BLOCK 1024 


extern "C"
__global__ void groupnorm_forward_kernel2(const float* x, const float* gamma, const float* beta,
    float* out, float* mean, float* var,
    int B, int C, int img_size, int groupSize, int n_groups)
{
    int b = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (b >= B) return;

	int once = groupSize * img_size;

	for(int g = 0;g<n_groups;g++) {
		float sum = 0.0f;
		float sum2 = 0.0f;
		for(int gs = 0;gs<once;gs++) {
			float val = x[b * n_groups * once + g * once + gs];
			sum += val;
			sum2 += val * val;
		}
		float mean_val = sum / once;
		mean[b * n_groups + g] = mean_val;
		var[b * n_groups + g] = sum2 / once - mean_val * mean_val;
		for(int gs = 0;gs<groupSize * img_size;gs++) {
			out[b * n_groups * once + g * once + gs] = (float) ((x[b * n_groups * once + g * once + gs] - mean_val) / sqrtf(var[b * n_groups + g] + 1e-5f));
		}
	}
	for(int c = 0;c<C;c++) {
		for(int i = 0;i<img_size;i++) {
			float x_norm = out[b * C * img_size + c * img_size + i];
			out[b * C * img_size + c * img_size + i] = gamma[c] * x_norm + beta[c];
		}
	}
	
}