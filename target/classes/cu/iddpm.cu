#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

extern "C"
__global__ void add_kernel(
    float* x_start,
    float* xt,
    float* t,
    float* output,
    float* a,
    float* b,
    int N, int W
) {
    int idx = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (idx < N) {
       int n = idx / W;
       int tidx = t[n];
       output[idx] = a[tidx] * x_start[idx] + b[tidx] * xt[idx];
    }
}

extern "C"
__global__ void extract_into_tensor(
    float* a,
    float* t,
    float* output,
    int N, int W
) {
    int idx = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (idx < N) {
       int n = idx / W;
       int tidx = t[n];
       output[idx] = a[tidx];
    }
}

extern "C"
__global__ void model_variance(
    float* var,
    float* max_log,
    float* min_log,
    float* output,
    int N
) {
    int idx = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (idx < N) {
		float frac = (var[idx] + 1) / 2;
		float model_log_variance = frac * max_log[idx] + (1 - frac) * min_log[idx];
        output[idx] = expf(model_log_variance);
    }
}

extern "C"
__global__ void model_log_variance(
    float* var,
    float* max_log,
    float* min_log,
    float* output,
    int N
) {
    int idx = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (idx < N) {
		float frac = (var[idx] + 1) / 2;
		float model_log_variance = frac * max_log[idx] + (1 - frac) * min_log[idx];
        output[idx] = model_log_variance;
    }
}

extern "C"
__global__ void sub_kernel(
    float* x_start,
    float* xt,
    float* t,
    float* output,
    float* a,
    float* b,
    int N, int W
) {
    int idx = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (idx < N) {
       int n = idx / W;
       int tidx = t[n];
       output[idx] = a[tidx] * x_start[idx] - b[tidx] * xt[idx];
    }
}

extern "C"
__global__ void normal_kl(
    float* mean1,
    float* logvar1,
    float* mean2,
    float* logvar2,
    float* output,
    int N
) {
    int idx = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (idx < N) {
		float lv1 = logvar1[idx];
		float lv2 = logvar2[idx];
		float m1 = mean1[idx];
		float m2 = mean2[idx];
        output[idx] = 0.5 * (-1.0 + lv2 - lv1 + exp(lv1 - lv2) + pow((m1 - m2), 2) * exp(-lv2)) / log(2.0);
    }
}

extern "C"
__global__ void normal_kl_back(
    float* mean1,
    float* logvar1,
    float* mean2,
    float* logvar2,
    float* dlogvar2,
    int N
) {
    int idx = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (idx < N) {
		float lv1 = logvar1[idx];
		float lv2 = logvar2[idx];
		float m1 = mean1[idx];
		float m2 = mean2[idx];
		float delta = 0.5 / logf(2.0) / N;
		dlogvar2[idx] = delta - delta * exp(lv1 - lv2) - delta * pow((m1 - m2), 2) * exp(-lv2);
    }
}

__device__ float approx_standard_normal_cdf(float x) {
  return 0.5 * (1.0 + tanh(sqrt(2.0 / M_PI) * (x + 0.044715 * pow(x, 3))));
}

__device__ float approx_standard_normal_cdf_back(float x,float delta) {
  float tmp = tanh(sqrtf(2.0 / M_PI) * (x + 0.044715 * pow(x, 3)));
  float d1 = 0.5 * delta * (1 - tmp * tmp) * sqrtf(2.0 / M_PI);
  return d1 + (0.044715 * 3 * pow(x, 2) * d1);
}

extern "C"
__global__ void discretized_gaussian_log_likelihood(
    float* x,
    float* means,
    float* log_scales,
    float* output,
    int N
) {
    int idx = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (idx < N) {
		float x_v = x[idx];
		float centered_x = x_v - means[idx];
		float inv_stdv = exp(-log_scales[idx] * 0.5);
		float plus_in = inv_stdv * (centered_x + 1.0/255.0);
		float cdf_plus = approx_standard_normal_cdf(plus_in);
		float min_in = inv_stdv * (centered_x - 1.0/255.0);
		float cdf_min = approx_standard_normal_cdf(min_in);
		float tmp = 1e-12f;
		if(cdf_plus > 1e-12f){
			tmp = cdf_plus;
		}
		float log_cdf_plus = log(tmp);
		float log_one_minus_cdf_min = 1.0 - cdf_min;
		if(log_one_minus_cdf_min < 1e-12f){
			log_one_minus_cdf_min = 1e-12f;
		}
		log_one_minus_cdf_min = log(log_one_minus_cdf_min);
		
		float cdf_delta = cdf_plus - cdf_min;
		if(cdf_delta < 1e-12f){
			cdf_delta = 1e-12f;
		}
		float log_probs = 0.0f;
		
		if(x_v < -0.999f){
			log_probs = log_cdf_plus;
		}else if(x_v > 0.999){
			log_probs = log_one_minus_cdf_min;
		}else{
			log_probs = log(cdf_delta);
		}
		
        output[idx] = - (log_probs / log(2.0));

    }
}

extern "C"
__global__ void discretized_gaussian_log_likelihood_back(
    float* x,
    float* means,
    float* log_scales,
    float* dlogvar,
    int N
) {
    int idx = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (idx < N) {
		float x_v = x[idx];
		float centered_x = x_v - means[idx];
		float inv_stdv = expf(-log_scales[idx] * 0.5);
		float plus_in = inv_stdv * (centered_x + 1.0/255.0);
		float cdf_plus = approx_standard_normal_cdf(plus_in);
		float min_in = inv_stdv * (centered_x - 1.0/255.0);
		float cdf_min = approx_standard_normal_cdf(min_in);
		
		float cdf_delta = cdf_plus - cdf_min;
		
		float d_log_probs = -1.0f / N / logf(2.0);
		
		float tmp = 1e-12f;
		if(cdf_plus > 1e-12f){
			tmp = cdf_plus;
		}
		
		if(x_v < -0.999f){
			float dtmp = 0.0f;
			if(cdf_plus > 1e-12f){
				dtmp = d_log_probs * 1 / cdf_plus;
			}
			float d = approx_standard_normal_cdf_back(plus_in, dtmp);
			d_log_probs = d * (centered_x + 1.0/255.0);
		}else if(x_v > 0.999){
			float dtmp = 0.0f;
			if(1.0 - cdf_min > 1e-12f){
				dtmp = d_log_probs * 1 / (1.0 - cdf_min);
			}
			float d = approx_standard_normal_cdf_back(min_in, -dtmp);
			d_log_probs = d * (centered_x - 1.0/255.0);
		}else{
			float dtmp = 0.0f;
			if(cdf_delta > 1e-12f){
				dtmp = d_log_probs * 1 / cdf_delta;
			}
			float d1 = approx_standard_normal_cdf_back(plus_in, dtmp);
			d1 = d1 * (centered_x + 1.0/255.0);
			float d2 = approx_standard_normal_cdf_back(min_in, -dtmp);
			d2 = d2 * (centered_x - 1.0/255.0);
			d_log_probs = (d1 + d2);
		}
		
		dlogvar[idx] = d_log_probs * inv_stdv * -0.5;
		
    }
}

extern "C"
__global__ void where_kernel(
    float* a,
    float* b,
    float* t,
    float* output,
    int N, int W
) {
    int idx = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (idx < N) {
       int n = idx / W;
       int tidx = t[n];
       if(tidx == 0){
			output[idx] = a[idx];
	   }else{
	   		output[idx] = b[idx];
	   }
    }
}

extern "C"
__global__ void var_back(
    float* maxLog,
    float* minLog,
    float* delta,
    float* diff,
    int N
) {
    int idx = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (idx < N) {
        diff[idx] = (delta[idx] * maxLog[idx] - delta[idx] * minLog[idx]) / 2;
    }
}

extern "C"
__global__ void mean_kernel(
    float* x,
    float* output,
    int N, int W
) {
    int idx = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (idx < N) {
		float sum = 0;
		for(int i = 0;i<W;i++){
			sum += x[idx * W + i];
		}
		output[idx] = sum / W;
    }
}

extern "C"
__global__ void get_score_from_velocity(
    float* vt,
    float* xt,
    float t,
    float* score,
    int N, int W
) {
    int idx = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (idx < N) {
       int n = idx / W;
       float alpha_t = 1 - t;
       float d_alpha_t = -1;
       float sigma_t = t;
       float d_sigma_t  = 1;
       float reverse_alpha_ratio = alpha_t / d_alpha_t;
       float var = sigma_t * sigma_t - reverse_alpha_ratio * d_sigma_t * sigma_t;
       score[idx] = (reverse_alpha_ratio * vt[idx] - xt[idx]) / var;
    }
}

extern "C"
__global__ void p_sample(
    float* v_cur,
    float* x_cur,
    float* s_cur,
    float* deps,
    float diffusion,
    float dt,
    float* x_next,
    int N, int W
) {
    int idx = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (idx < N) {
       int n = idx / W;
	   float d_cur = v_cur[idx] - 0.5f * diffusion * s_cur[idx];
	   x_next[idx] =  x_cur[idx] + d_cur * dt + sqrtf(diffusion) * deps[idx];
    }
}

extern "C"
__global__ void p_sample_last(
    float* v_cur,
    float* x_cur,
    float* s_cur,
    float diffusion,
    float dt,
    float* x_next,
    int N, int W
) {
    int idx = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (idx < N) {
       int n = idx / W;
	   float d_cur = v_cur[idx] - 0.5f * diffusion * s_cur[idx];
	   x_next[idx] =  x_cur[idx] + d_cur * dt;
    }
}

extern "C"
__global__ void q_sample(
    float* latend,
    float* noise,
    float* t,
    float* output,
    float* target,
    int N, int W
) {
    int idx = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (idx < N) {
       int n = idx / W;
       float tf = t[n];
       float alpha_t = 1 - tf;
       float sigma_t = tf;
       float d_alpha_t = -1;
       float d_sigma_t =  1;
       
	   output[idx] = alpha_t * latend[idx] + sigma_t * noise[idx];
	   target[idx] = d_alpha_t * latend[idx] + d_sigma_t * noise[idx];
    }
}

extern "C"
__global__ void q_sample_no_target(
    float* latend,
    float* noise,
    float* t,
    float* output,
    int N, int W
) {
    int idx = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (idx < N) {
       int n = idx / W;
       float tf = t[n];
       float alpha_t = 1 - tf;
       float sigma_t = tf;
       
	   output[idx] = alpha_t * latend[idx] + sigma_t * noise[idx];
    }
}