#define BLOCK 1024 


extern "C"
__global__ void culOutput_cov(float* x,float* mean,float* std,float* gama,float* beta,float* z,float* out,int N,int number,int height,int width)
{
    
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < N; index += blockDim.x * gridDim.x) {
		
		int c = index / (height * width);
		
		float mean_val = mean[c];
		float std_val = std[c];
		float gama_val = gama[c];
		float beta_val = beta[c];
	
		for(int n = 0;n<number;n++) {	
			
			float z_val = (x[n * N + index] - mean_val) / std_val;
			
			z[n * N + index] = z_val;
			
			out[n * N + index] = z_val * gama_val + beta_val;
			
		}	
		
	}

}

extern "C"
__global__ void culOutput_full(float* x,float* mean,float* std,float* gama,float* beta,float* z,float* out,int number,int channel,int height,int width)
{
    
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < width; index += blockDim.x * gridDim.x) {
		
		float mean_val = mean[index];
		float std_val = std[index];
		float gama_val = gama[index];
		float beta_val = beta[index];
	
		for(int n = 0;n<number;n++) {	
			
			float z_val = (x[n * width + index] - mean_val) / std_val;
			
			z[n * width + index] = z_val;
			
			out[n * width + index] = z_val * gama_val + beta_val;
			
		}	
		
	}

}


extern "C"
__global__ void computeDelta(float* delta,float* gama,float* deltaGama,float* deltaBeta,float* z,float* diff,int number,int channel,int height,int width)
{
    
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < channel; index += blockDim.x * gridDim.x) {

		float gama_val = gama[index];
		
		float deltaGama_val = 0;
		
		float deltaBeta_val = 0;
		
		for(int n = 0;n<number;n++) {	
			for(int h = 0;h<height;h++) {
				for(int w = 0;w<width;w++) {
					
					float delta_val = delta[n * channel * height * width + index * height * width + h * width + w];
					
					deltaGama_val += delta_val * z[n * channel * height * width + index * height * width + h * width + w];
					
					deltaBeta_val += delta_val;
					
					diff[n * channel * height * width + index * height * width + h * width + w] = delta_val * gama_val;
	
				}
			}
		}	
		
		deltaGama[index] = deltaGama_val;
		deltaBeta[index] = deltaBeta_val;
		
	}

}

extern "C"
__global__ void meanDzSum(float* x,float* dz,float* mean,float* var,float* std,float* dvar,float* dmu,float scale,int number,int channel,int height,int width,float eta)
{
    
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < channel; index += blockDim.x * gridDim.x) {
		
		float std_val = std[index];
		float mean_val = mean[index];
		
		float dvar_val = 0.0f;
		float dmu_val = 0.0f;
		float dmu2_val = 0.0f;
		
		for(int n = 0;n<number;n++) {	
			for(int h = 0;h<height;h++) {
				for(int w = 0;w<width;w++) {
					float x_val = x[n * channel * height * width + index * height * width + h * width + w];
					float dz_val = dz[n * channel * height * width + index * height * width + h * width + w];
					
					dvar_val += (x_val - mean_val) * dz_val;
					dmu_val += -1.0f * dz_val / std_val;
					dmu2_val += -2.0f * (x_val - mean_val) * scale;
				}
			}
		}	
		
		dvar[index] = dvar_val * -0.5 * powf(var[index] + eta, -1.5);
		dmu[index] = dmu_val + dmu2_val * dvar[index];
		
	}

}


extern "C"
__global__ void computeDiff(float* x,float* dvar,float* dmu,float* mean,float* std,float* diff,float scale,int number,int channel,int height,int width)
{
    
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < channel; index += blockDim.x * gridDim.x) {
		
		float dvar_val = dvar[index];
		float dmu_val = dmu[index];
		float std_val = std[index];
		float mean_val = mean[index];
		
		for(int n = 0;n<number;n++) {	
			for(int h = 0;h<height;h++) {
				for(int w = 0;w<width;w++) {
					int xIndex = n * channel * height * width + index * height * width + h * width + w;
					diff[xIndex] = diff[xIndex] / std_val + 2.0 * dvar_val * (x[xIndex] - mean_val) * scale + dmu_val * scale;
				}
			}
		}	
		
	}

}


extern "C"
__global__ void computeDgama(float* delta,float* deltaGama,float* deltaBeta,float* z,float* diff,float* gama,int number,int channel,int height,int width)
{
    
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < channel; index += blockDim.x * gridDim.x) {
	
		float gama_val = gama[index];
	
		float deltaGama_val = 0;
		
		float deltaBeta_val = 0;
		
		for(int n = 0;n<number;n++) {	
			for(int h = 0;h<height;h++) {
				for(int w = 0;w<width;w++) {
					
					float delta_val = delta[n * channel * height * width + index * height * width + h * width + w];
					
					deltaGama_val += delta_val * z[n * channel * height * width + index * height * width + h * width + w];
					
					deltaBeta_val += delta_val;
					
					diff[n * channel * height * width + index * height * width + h * width + w] = delta_val * gama_val;
					
				}
			}
		}	
		
		deltaGama[index] = deltaGama_val;
		deltaBeta[index] = deltaBeta_val;
		
	}

}

extern "C"
__global__ void computeDParams(float* delta,float* deltaGama,float* deltaBeta,float* z,int number,int channel,int height,int width)
{
    
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < channel; index += blockDim.x * gridDim.x) {
	
		float deltaGama_val = 0;
		
		float deltaBeta_val = 0;
		
		for(int n = 0;n<number;n++) {	
			for(int h = 0;h<height;h++) {
				for(int w = 0;w<width;w++) {
					float delta_val = delta[n * channel * height * width + index * height * width + h * width + w];					
					deltaGama_val += delta_val * z[n * channel * height * width + index * height * width + h * width + w];
					deltaBeta_val += delta_val;
				}
			}
		}	
		
		deltaGama[index] = deltaGama_val;
		deltaBeta[index] = deltaBeta_val;
		
	}

}

extern "C"
__global__ void meanDelta(float* delta,float* out,float* meanDelta,float* meanDeltaDot,int number,int channel,int height,int width,float scale)
{
    
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < channel; index += blockDim.x * gridDim.x) {

		float meanDelta_val = 0;
		
		float meanDeltaDot_val = 0;
		
		for(int n = 0;n<number;n++) {	
			for(int h = 0;h<height;h++) {
				for(int w = 0;w<width;w++) {
					
					int xIndex = n * channel * height * width + index * height * width + h * width + w;
					
					meanDelta_val += delta[xIndex] * scale;
					
					meanDeltaDot_val += delta[xIndex] * out[xIndex] * scale;
					
				}
			}
		}	
		
		meanDelta[index] = meanDelta_val;
		meanDeltaDot[index] = meanDeltaDot_val;
		
	}

}


extern "C"
__global__ void dx_fn(float* delta,float* out,float* diff,float* meanDelta,float* meanDeltaDot,float* std,int number,int channel,int height,int width,float eta)
{
    
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < channel; index += blockDim.x * gridDim.x) {

		for(int n = 0;n<number;n++) {	
			for(int h = 0;h<height;h++) {
				for(int w = 0;w<width;w++) {
				
					int xIndex = n * channel * height * width + index * height * width + h * width + w;
					
					diff[xIndex] = (delta[xIndex] - meanDelta[index] - meanDeltaDot[index] * out[xIndex]) / sqrt((std[index] + eta));
					
				}
			}
		}	
		
	}

}


extern "C"
__global__ void normalize_kernel(int N, float *x, float *z, float *out, float *mean, float *variance, float *gama, float *beta,int batch, int filters, int spatial,float eta)
{
    int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (index >= N) return;
    int f = (index/spatial)%filters;
    z[index] = (x[index] - mean[f])/(sqrtf(variance[f] + eta));
    out[index] = z[index] * gama[f] + beta[f];
}


extern "C"
__global__ void dgama_kernel(float *xhat, float *delta, int batch ,int c, int size, float *dgama)
{
    __shared__ float part[BLOCK];
    int i,b;
    int filter = blockIdx.x;
    int p = threadIdx.x;
    float sum = 0;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < size; i += BLOCK){
            int index = p + i + size*(filter + c*b);
            sum += (p+i < size) ? delta[index] * xhat[index] : 0;
        }
    }
    part[p] = sum;
    __syncthreads();
    if (p == 0) {
    	dgama[filter] = 0;
        for(i = 0; i < BLOCK; ++i) dgama[filter] += part[i];
    }
}


extern "C"
__global__ void dgama_kernel2(float *xhat, float *delta, int batch ,int filters, int spatial, float *dgama)
{
    const int threads = BLOCK;
    __shared__ float local[threads];

    int id = threadIdx.x;
    local[id] = 0;

    int filter = blockIdx.x;

    int i, j;
    for(j = 0; j < batch; ++j){
        for(i = 0; i < spatial; i += threads){
            int index = j*spatial*filters + filter*spatial + i + id;
            local[id] += (i+id < spatial) ? delta[index] * xhat[index] : 0;
        }
    }

    __syncthreads();

    if(id == 0){
        dgama[filter] = 0;
        for(i = 0; i < threads; ++i){
            dgama[filter] += local[i];
        }
    }
}


extern "C"
__global__ void dbeta_kernel(float *dbeta, float *delta, int batch, int c, int size)
{
    __shared__ float part[BLOCK];
    int i,b;
    int filter = blockIdx.x;
    int p = threadIdx.x;
    float sum = 0;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < size; i += BLOCK){
            int index = p + i + size*(filter + c*b);
            sum += (p+i < size) ? delta[index] : 0;
        }
    }
    part[p] = sum;
    __syncthreads();
    if (p == 0) {
    	dbeta[filter] = 0;
        for(i = 0; i < BLOCK; ++i) dbeta[filter] += part[i];
    }
}

extern "C"
__global__ void dgama_dbeta_kernel(float *xhat, float *delta, int batch ,int c, int size, float *dgama,float *dbeta)
{
    __shared__ float part[BLOCK];
    __shared__ float local[BLOCK];
    int i,b;
    int filter = blockIdx.x;
    int p = threadIdx.x;
    float sum = 0;
    float sum_b = 0;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < size; i += BLOCK){
            int index = p + i + size*(filter + c*b);
            float delta_val = delta[index];
            sum += (p+i < size) ? delta_val * xhat[index] : 0;
            sum_b += (p+i < size) ? delta_val : 0;
        }
    }
    part[p] = sum;
    local[p] = sum_b;
    __syncthreads();
    if (p == 0) {
    	float dg = 0;
    	float db = 0;
        for(i = 0; i < BLOCK; ++i) {
        	dg += part[i];
        	db += local[i];
        }
        dgama[filter] = dg;
        dbeta[filter] = db;
    }
}

extern "C"
__global__ void computeDelta_full(float* delta,float* deltaGama,float* deltaBeta,float* z,int number,int width)
{
    
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < width; index += blockDim.x * gridDim.x) {

		float deltaGama_val = 0;
		
		float deltaBeta_val = 0;
		
		for(int n = 0;n<number;n++) {	
			
			float delta_val = delta[n * width + index];
					
			deltaGama_val += delta_val * z[n * width + index];
					
			deltaBeta_val += delta_val;

		}	
		
		deltaGama[index] = deltaGama_val;
		deltaBeta[index] = deltaBeta_val;
		
	}

}

extern "C"
__global__ void dxhat_kernel2(int N, float *delta, float *dz, float *gama, int filters, int spatial)
{	
	int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (index >= N) return;
    int f = (index/spatial)%filters;

    dz[index] = delta[index] * gama[f];
}


extern "C"
__global__ void dxhat_kernel(float *dz, float *gama, int c, int size)
{
    int offset = blockIdx.x * blockDim.x + threadIdx.x;
    int filter = blockIdx.y;
    int batch = blockIdx.z;

    if(offset < size) dz[(batch*c+filter)*size + offset] *= gama[filter];
}

extern "C"
__global__ void full_mean_delta_kernel(float *dxhat, float *variance, int batch, int filters, float *mean_delta)
{
    
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < filters; index += blockDim.x * gridDim.x) {
			
		float val = 0;
		
		for(int n = 0;n<batch;n++) {	
			val += dxhat[n * filters + index];
		}	
		
		mean_delta[index] = val * (-1.f/sqrtf(variance[index] + .00001f));
	}

}


//dmean = (∑ dxhat * -1 / (var + eta)^1/2) + dvar * 1 / N * ∑ -2 * (x - mean)
extern "C"
__global__ void full_mean_delta_ov_kernel(float *dxhat, float *variance, float *mean, float *x, float *dvar, int batch, int filters, float *mean_delta)
{
    
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < filters; index += blockDim.x * gridDim.x) {
			
		float val = 0;
		float sum = 0;
		
		for(int n = 0;n<batch;n++) {	
			val += dxhat[n * filters + index] * (-1.0f / sqrtf(variance[index] + .00001f));
			sum += - 2.0f * (x[n * filters + index] - mean[index]);
		}	
		
		mean_delta[index] = val + dvar[index] * sum / batch;
	}

}

//dmean = (∑ dxhat * -1 / (var + eta)^1/2) + dvar * 1 / N * ∑ -2 * (x - mean)
extern "C"
__global__ void fast_mean_delta_kernel(float *dxhat, float *variance, int batch, int filters, int spatial, float *mean_delta)
{
    const int threads = BLOCK;
    __shared__ float local[threads];

    int id = threadIdx.x;
    local[id] = 0;

    int filter = blockIdx.x;

    int i, j;
    for(j = 0; j < batch; ++j){
        for(i = 0; i < spatial; i += threads){
            int index = j*spatial*filters + filter*spatial + i + id;
            local[id] += (i+id < spatial) ? dxhat[index] : 0;
        }
    }

    __syncthreads();

    if(id == 0){
        mean_delta[filter] = 0;
        for(i = 0; i < threads; ++i){
            mean_delta[filter] += local[i];
        }
        mean_delta[filter] *= (-1.f/sqrtf(variance[filter] + .00001f));
    }
}

//dL_davg = (-1.0 / torch.sqrt(variance + eps) * dL_dxi_hat).sum((0, 2, 3), keepdim=True) + (dL_dvar * (-2.0 * (input - mean)).sum((0, 2, 3), keepdim=True) / B)
//dmean = (∑ dxhat * -1 / (var + eta)^1/2) + dvar * 1 / N * ∑ -2 * (x - mean)
extern "C"
__global__ void fast_mean_delta_ov_kernel(float *dxhat, float *variance, float *mean, float *x, float *dvar, int batch, int filters, int spatial, float *mean_delta)
{
    const int threads = BLOCK;
    __shared__ float local[threads];
    __shared__ float local2[threads];

    int id = threadIdx.x;
    local[id] = 0;
    local2[id] = 0;

    int filter = blockIdx.x;

    int i, j;
    for(j = 0; j < batch; ++j){
        for(i = 0; i < spatial; i += threads){
            int index = j*spatial*filters + filter*spatial + i + id;
            local[id] += (i+id < spatial) ? dxhat[index] : 0;
            local2[id] += (i+id < spatial) ? -2.f * (x[index] - mean[filter])  : 0;
        }
    }

    __syncthreads();

    if(id == 0){
        float tmp = 0;
        float sum = 0;
        for(i = 0; i < threads; ++i){
           tmp += local[i];
           sum += local2[i];
        }
        tmp *= (-1.f/sqrtf(variance[filter] + .00001f));
        mean_delta[filter] = tmp + dvar[filter] * sum / batch / spatial;
    }
}

//dL_dvar = (-0.5 * dL_dxi_hat * (input - mean)).sum((0, 2, 3), keepdim=True)  * ((variance + eps) ** -1.5) # edit
extern "C"
__global__ void full_var_delta_kernel(float *x, float *dxhat, float *mean, float *variance, int batch, int filters, float *variance_delta)
{
    
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < filters; index += blockDim.x * gridDim.x) {
			
		float val = 0;
		
		for(int n = 0;n<batch;n++) {	
			val += dxhat[n * filters + index] * (x[n * filters + index] - mean[index]);
		}	
		
		variance_delta[index] = val * -0.5f * powf(variance[index] + .00001f, -1.5f);
	}

}

//dvar = ∑ dxhat * (xi - mean) * -1/2 * (var + eta)^-3/2
extern "C"
__global__ void fast_variance_delta_kernel(float *x, float *dxhat, float *mean, float *variance, int batch, int filters, int spatial, float *variance_delta)
{
    const int threads = BLOCK;
    __shared__ float local[threads];

    int id = threadIdx.x;
    local[id] = 0;

    int filter = blockIdx.x;

    int i, j;
    for(j = 0; j < batch; ++j){
        for(i = 0; i < spatial; i += threads){
            int index = j*spatial*filters + filter*spatial + i + id;

            local[id] += (i+id < spatial) ? dxhat[index]*(x[index] - mean[filter]) : 0;
        }
    }

    __syncthreads();

    if(id == 0){
        variance_delta[filter] = 0;
        for(i = 0; i < threads; ++i){
            variance_delta[filter] += local[i];
        }
        variance_delta[filter] *= -0.5f * powf(variance[filter] + .00001f, -1.5f);
    }
}


//dx = dxhat * 1 / (var + eta)^1/2 + dvar * 2(x - mean) / N + dmean * 1/N
extern "C"
__global__ void dx_kernel(int N, float *x, float *mean, float *variance, float *mean_delta, float *variance_delta, int batch, int filters, int spatial, float *diff)
{
    int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (index >= N) return;
    int f = (index/spatial)%filters;

    diff[index] = diff[index] / (sqrtf(variance[f] + .00001f)) + variance_delta[f] * 2.f * (x[index] - mean[f]) / (spatial * batch) + mean_delta[f] / (spatial * batch);
}


//  dL_dxi = (dL_dxi_hat / torch.sqrt(variance + eps)) + (2.0 * dL_dvar * (input - mean) / B) + (dL_davg / B)
extern "C"
__global__ void dx_kernel_full(float *x, float *mean, float *variance, float *mean_delta, float *variance_delta, int batch, int filters, float *diff)
{
    
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < filters; index += blockDim.x * gridDim.x) {
		
		float dvar_val = variance_delta[index];
		float dmu_val = mean_delta[index];
		float std_val = sqrtf(variance[index] + .00001f);
		float mean_val = mean[index];
		
		for(int n = 0;n<batch;n++) {	
			int xIndex = n * filters + index;
			diff[xIndex] = (diff[xIndex] / std_val) + (2.0f * dvar_val * (x[xIndex] - mean_val) / batch) + (dmu_val / batch);
		}	
		
	}

}
