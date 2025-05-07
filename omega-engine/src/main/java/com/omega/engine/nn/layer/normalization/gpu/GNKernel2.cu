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
		float rstd = 1 / sqrtf(sum2 / once - mean_val * mean_val + 1e-5f);
		mean[b * n_groups + g] = mean_val;
		var[b * n_groups + g] = rstd;
		for(int gs = 0;gs<groupSize * img_size;gs++) {
			out[b * n_groups * once + g * once + gs] = (x[b * n_groups * once + g * once + gs] - mean_val) * rstd;
		}
	}
	for(int c = 0;c<C;c++) {
		for(int i = 0;i<img_size;i++) {
			float x_norm = out[b * C * img_size + c * img_size + i];
			out[b * C * img_size + c * img_size + i] = gamma[c] * x_norm + beta[c];
		}
	}
	
}

extern "C"
__global__ void groupnorm_backward_kernel1(float* dinp, float* dweight, float* dbias,
                        const float* dout, const float* inp, const float* weight, const float* mean, const float* rstd,
                        int B, int T, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B*T) return;
    int b = idx / T;
    int t = idx % T;

    const float* dout_bt = dout + b * T * C + t * C;
    const float* inp_bt = inp + b * T * C + t * C;
    float* dinp_bt = dinp + b * T * C + t * C;
    const float mean_bt = mean[b * T + t];
    const float rstd_bt = rstd[b * T + t];

    // first: two reduce operations
    float dnorm_mean = 0.0f;
    float dnorm_norm_mean = 0.0f;
    for (int i = 0; i < C; i++) {
        float norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
        float dnorm_i = weight[i] * dout_bt[i];
        dnorm_mean += dnorm_i;
        dnorm_norm_mean += dnorm_i * norm_bti;
    }
    dnorm_mean = dnorm_mean / C;
    dnorm_norm_mean = dnorm_norm_mean / C;

    // now iterate again and accumulate all the gradients
    for (int i = 0; i < C; i++) {
        float norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
        float dnorm_i = weight[i] * dout_bt[i];
        // gradient contribution to bias
        atomicAdd(&dbias[i], dout_bt[i]);
        // gradient contribution to weight
        atomicAdd(&dweight[i], norm_bti * dout_bt[i]);
        // gradient contribution to input
        float dval = 0.0f;
        dval += dnorm_i; // term 1
        dval -= dnorm_mean; // term 2
        dval -= norm_bti * dnorm_norm_mean; // term 3
        dval *= rstd_bt; // final scale
        dinp_bt[i] += dval;
    }
}