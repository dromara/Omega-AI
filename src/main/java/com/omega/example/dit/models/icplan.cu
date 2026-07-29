#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

extern "C"
__global__ void compute_xt(
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
       float alpha_t = tf;
       float sigma_t = 1 - tf;
       
	   output[idx] = alpha_t * latend[idx] + sigma_t * noise[idx];
    }
}

extern "C"
__global__ void compute_ut(
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
       float d_alpha_t = 1;
       float d_sigma_t = -1;
       
	   output[idx] = d_alpha_t * latend[idx] + d_sigma_t * noise[idx];
    }
}

extern "C"
__global__ void compute_v(
    float* x,
    float* z,
    float* t,
    float* output,
    float t_eps,
    int N, int W
) {
    int idx = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (idx < N) {
       int n = idx / W;
       float tf = t[n];
       float alpha_t = tf;
       float sigma_t = 1 - tf;
       if(sigma_t < t_eps){
		  sigma_t = t_eps;
	   }
	   output[idx] = (x[idx] - z[idx]) / sigma_t;
    }
}

extern "C"
__global__ void compute_dv(
    float* delta,
    float* t,
    float* dx,
    float t_eps,
    int N, int W
) {
    int idx = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (idx < N) {
       int n = idx / W;
       float tf = t[n];
       float sigma_t = 1 - tf;
       if(sigma_t < t_eps){
		  sigma_t = t_eps;
	   }
	   dx[idx] = delta[idx] / sigma_t;
    }
}

extern "C"
__global__ void compute_z_next(
	float* v_pred,
    float* z,
    float t,
    float t_next,
    float* output,
    int N
) {
    int idx = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (idx < N) {
	   output[idx] = z[idx] + (t_next - t) * v_pred[idx];
    }
}

extern "C"
__global__ void cosine_similarity_loss(
    float* x1,
    float* norm1,
    float* x2,
    float* norm2,
    float* out,
    int N,
    int C,
    int W
) {
    int idx = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (idx < N) {
	   int n = idx / C / W;
	   int c = idx % C;
	   int w = idx % W;
	   int x_idx = n * C * W + c * W + w;
	   int n_idx = n * W + w;
       out[idx] = 1 - (x1[x_idx] / norm1[n_idx]) * (x2[x_idx] / norm2[n_idx]);
    }
}

extern "C"
__global__ void cosine_similarity_loss_dim1(
    float* x1,
    float* norm1,
    float* x2,
    float* norm2,
    float* out,
    int N,
    int C,
    int W
) {
    int idx = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (idx < N) {
	   int n = idx / W;
	   int w = idx % W;
	   float o = 0.0f;
	   for(int c = 0;c<C;c++){
	     int x_idx = n * C * W + c * W + w;
	     o += (x1[x_idx] / norm1[idx]) * (x2[x_idx] / norm2[idx]);
	   }
       out[idx] = 1 - o;
    }
}

extern "C"
__global__ void cosine_similarity_loss_back1(
	float delta,
    float* x1,
    float* norm1,
    float* x2,
    float* norm2,
    float* dx1,
    int N,
    int C,
    int W
) {
    int idx = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (idx < N) {
	   int n = idx / W;
	   int w = idx % W;
	   for(int c = 0;c<C;c++){
	      int x_idx = n * C * W + c * W + w;
	       dx1[x_idx] = -delta * (x2[x_idx] / norm2[idx]) / norm1[idx];
	   }
    }
}

extern "C"
__global__ void cosine_similarity_loss_back2(
	float delta,
    float* x1,
    float* norm1,
    float* x2,
    float* norm2,
    float* dnorm1,
    int N,
    int C,
    int W
) {
    int idx = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (idx < N) {
	   int n = idx / W;
	   int w = idx % W;
	   
	   float out = 0.0f;
	   
	   for(int c = 0;c<C;c++){
	      int x_idx = n * C * W + c * W + w;
	      out += -delta * (-x1[x_idx] / norm1[idx] / norm1[idx]) * (x2[x_idx] / norm2[idx]);
	   }
	   
	   dnorm1[idx] = out;
    }
}

extern "C"
__global__ void cosine_similarity(
    float* x1,
    float* x2,
    float* out,
    int outer,
    int N,
    int C,
    int H,
    int W,
    int dim,
    float eps
) {
    int idx = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (idx < outer) {
       int outN = (dim == 0) ? 1 : N;
       int outC = (dim == 1) ? 1 : C;
       int outH = (dim == 2) ? 1 : H;
       int outW = (dim == 3) ? 1 : W;
       int tmp = idx;
       int ow = tmp % outW;
       tmp /= outW;
       int oh = tmp % outH;
       tmp /= outH;
       int oc = tmp % outC;
       tmp /= outC;
       int on = tmp % outN;
       int base = ((on * C + oc) * H + oh) * W + ow;
       int dimSize = W;
       int stride = 1;
       if (dim == 0) {
          dimSize = N;
          stride = C * H * W;
       } else if (dim == 1) {
          dimSize = C;
          stride = H * W;
       } else if (dim == 2) {
          dimSize = H;
          stride = W;
       }
       float dot = 0.0f;
       float n1 = 0.0f;
       float n2 = 0.0f;
       for (int i = 0; i < dimSize; i++) {
          int offset = base + i * stride;
          float a = x1[offset];
          float b = x2[offset];
          dot += a * b;
          n1 += a * a;
          n2 += b * b;
       }
       n1 = fmaxf(sqrtf(n1), eps);
       n2 = fmaxf(sqrtf(n2), eps);
       out[idx] = dot / (n1 * n2);
    }
}

extern "C"
__global__ void cosine_similarity_back(
    float* grad_out,
    float* x1,
    float* x2,
    float* dx1,
    int outer,
    int N,
    int C,
    int H,
    int W,
    int dim,
    float eps
) {
    int idx = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (idx < outer) {
       int outN = (dim == 0) ? 1 : N;
       int outC = (dim == 1) ? 1 : C;
       int outH = (dim == 2) ? 1 : H;
       int outW = (dim == 3) ? 1 : W;
       int tmp = idx;
       int ow = tmp % outW;
       tmp /= outW;
       int oh = tmp % outH;
       tmp /= outH;
       int oc = tmp % outC;
       tmp /= outC;
       int on = tmp % outN;
       int base = ((on * C + oc) * H + oh) * W + ow;
       int dimSize = W;
       int stride = 1;
       if (dim == 0) {
          dimSize = N;
          stride = C * H * W;
       } else if (dim == 1) {
          dimSize = C;
          stride = H * W;
       } else if (dim == 2) {
          dimSize = H;
          stride = W;
       }
       float dot = 0.0f;
       float n1_raw = 0.0f;
       float n2_raw = 0.0f;
       for (int i = 0; i < dimSize; i++) {
          int offset = base + i * stride;
          float a = x1[offset];
          float b = x2[offset];
          dot += a * b;
          n1_raw += a * a;
          n2_raw += b * b;
       }

       float n1_sqrt = sqrtf(n1_raw);
       float n2_sqrt = sqrtf(n2_raw);
       float n1 = fmaxf(n1_sqrt, eps);
       float n2 = fmaxf(n2_sqrt, eps);
       float cosv = dot / (n1 * n2);
       float go = grad_out[idx];
       float inv = 1.0f / (n1 * n2);
       float scale = (n1_sqrt > eps) ? (cosv / (n1 * n1)) : 0.0f;

       for (int i = 0; i < dimSize; i++) {
          int offset = base + i * stride;
          float a = x1[offset];
          float b = x2[offset];
          dx1[offset] = go * (b * inv - a * scale);
       }
    }
}

extern "C"
__global__ void latend_norm(
    float* x1,
    float* mean,
    float* std,
    int N,
    int C,
    int W
) {
    int idx = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (idx < N) {
	   int c = idx % (W * C) / W;
	   x1[idx] = (x1[idx] - mean[c]) / std[c];
    }
}

extern "C"
__global__ void latend_un_norm(
    float* x1,
    float* mean,
    float* std,
    int N,
    int C,
    int W
) {
    int idx = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (idx < N) {
	   int c = idx % (W * C) / W;
	   x1[idx] = x1[idx] * std[c] + mean[c];
    }
}


extern "C"
__global__ void expand_mask(
    float* a,
    float* b,
    float* mask,
    float* out,
    int N,
    int W,
    float maskRatio
) {
    int idx = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (idx < N) {
	   int bidx = idx / W;
	   if(mask[idx] < maskRatio){
	        out[idx] = b[bidx];
	   }else{
			out[idx] = a[bidx];
	   }
    }
}

extern "C"
__global__ void expand_(
    float* a,
    float* out,
    int N,
    int W
) {
    int idx = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (idx < N) {
	   int b = idx / W;
	   out[idx] = a[b];
    }
}

extern "C"
__global__ void compute_xt_ft(
    float* latend,
    float* noise,
    float* t,
    float* output,
    int N, int C, int gridH, int gridW, int patchSize
) {
    int idx = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (idx < N) {
	   int onceSize = C * gridH * patchSize * gridW * patchSize;
	   int o_hw = gridH * patchSize * gridW * patchSize;
       int b = idx / onceSize;
       int hw = idx % onceSize % o_hw;
       int ph = hw / (gridW * patchSize);
       int pw = hw % (gridW * patchSize);
       int h = ph / patchSize;
       int w = pw / patchSize;
       float tf = t[b * gridH * gridW + h * gridW + w];
       float alpha_t = tf;
       float sigma_t = 1 - tf;
	   output[idx] = alpha_t * latend[idx] + sigma_t * noise[idx];
    }
}

extern "C"
__global__ void compute_ut_ft(
    float* latend,
    float* noise,
    float* t,
    float* output,
    int N, int C, int gridH, int gridW, int patchSize
) {
    int idx = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (idx < N) {
       float d_alpha_t = 1;
       float d_sigma_t = -1;
	   output[idx] = d_alpha_t * latend[idx] + d_sigma_t * noise[idx];
    }
}

extern "C"
__global__ void compute_xt_ft_offset(
    float* latend,
    float* noise,
    float* t,
    float* output,
    int N, int C, int gridH, int gridW, int patchSize, int offset
) {
    int idx = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (idx < N) {
	   int onceSize = C * gridH * patchSize * gridW * patchSize;
	   int o_hw = gridH * patchSize * gridW * patchSize;
       int b = idx / onceSize;
       int hw = idx % onceSize % o_hw;
       int ph = hw / (gridW * patchSize);
       int pw = hw % (gridW * patchSize);
       int h = ph / patchSize;
       int w = pw / patchSize;
       float tf = t[b * (offset + gridH * gridW) + offset + h * gridW + w];
       float alpha_t = tf;
       float sigma_t = 1 - tf;
	   output[idx] = alpha_t * latend[idx] + sigma_t * noise[idx];
    }
}

extern "C"
__global__ void compute_ut_ft_offset(
    float* latend,
    float* noise,
    float* t,
    float* output,
    int N, int C, int gridH, int gridW, int patchSize, int offset
) {
    int idx = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (idx < N) {
       float d_alpha_t = 1;
       float d_sigma_t = -1;
	   output[idx] = d_alpha_t * latend[idx] + d_sigma_t * noise[idx];
    }
}