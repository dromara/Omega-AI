package com.omega.example.dit.models;

import static jcuda.driver.JCudaDriver.cuLaunchKernel;

import com.omega.engine.gpu.CUDAKernel;
import com.omega.engine.gpu.CUDAManager;
import com.omega.engine.tensor.Tensor;

import jcuda.Pointer;
import jcuda.driver.CUfunction;
import jcuda.runtime.cudaError;

public class IDDPMKernel extends CUDAKernel {
	
    private CUfunction add_function;
    private CUfunction sub_function;
    private CUfunction eit_function;
    private CUfunction model_log_variance_function;
    private CUfunction normal_kl_function;
    private CUfunction normal_kl_back_function;
    private CUfunction discretized_gaussian_log_likelihood_function;
    private CUfunction discretized_gaussian_log_likelihood_back_function;
    private CUfunction where_function;
    private CUfunction dvar_back_function;
    private CUfunction mean_function;
    private CUfunction get_score_from_velocity_function;
    private CUfunction q_sample_function;
    private CUfunction q_sample_no_target_function;
    private CUfunction p_sample_function;
    private CUfunction p_sample_last_function;
    
    private int CAFFE_CUDA_NUM_THREADS = 1024;
    
    private Pointer kernelParameters;

    public IDDPMKernel(CUDAManager cudaManager) {
        super(cudaManager);
        init();
    }

    public static void main(String args[]) {
        
    }

    public void initFunction() {
        try {
            if (add_function == null) {
            	add_function = getCudaManager().getLocalFunctionByModule("iddpm.cu", "add_kernel");
            }
            if (sub_function == null) {
            	sub_function = getCudaManager().getLocalFunctionByModule("iddpm.cu", "sub_kernel");
            }
            if (eit_function == null) {
            	eit_function = getCudaManager().getLocalFunctionByModule("iddpm.cu", "extract_into_tensor");
            }
            if (model_log_variance_function == null) {
            	model_log_variance_function = getCudaManager().getLocalFunctionByModule("iddpm.cu", "model_log_variance");
            }
            if (normal_kl_function == null) {
            	normal_kl_function = getCudaManager().getLocalFunctionByModule("iddpm.cu", "normal_kl");
            }
            if(normal_kl_back_function == null) {
            	normal_kl_back_function = getCudaManager().getLocalFunctionByModule("iddpm.cu", "normal_kl_back");
            }
            if (discretized_gaussian_log_likelihood_function == null) {
            	discretized_gaussian_log_likelihood_function = getCudaManager().getLocalFunctionByModule("iddpm.cu", "discretized_gaussian_log_likelihood");
            }
            if(discretized_gaussian_log_likelihood_back_function == null) {
            	discretized_gaussian_log_likelihood_back_function = getCudaManager().getLocalFunctionByModule("iddpm.cu", "discretized_gaussian_log_likelihood_back");
            }
            if(where_function == null) {
            	where_function = getCudaManager().getLocalFunctionByModule("iddpm.cu", "where_kernel");
            }
            if(mean_function ==  null) {
            	mean_function = getCudaManager().getLocalFunctionByModule("iddpm.cu", "mean_kernel");
            }
            if(dvar_back_function ==  null) {
            	dvar_back_function = getCudaManager().getLocalFunctionByModule("iddpm.cu", "var_back");
            }
            if(get_score_from_velocity_function == null) {
            	get_score_from_velocity_function = getCudaManager().getLocalFunctionByModule("iddpm.cu", "get_score_from_velocity"); 
            }
            if(q_sample_function == null) {
            	q_sample_function = getCudaManager().getLocalFunctionByModule("iddpm.cu", "q_sample"); 
            }
            if(q_sample_no_target_function == null) {
            	q_sample_no_target_function = getCudaManager().getLocalFunctionByModule("iddpm.cu", "q_sample_no_target"); 
            }
            if(p_sample_function == null) {
            	p_sample_function = getCudaManager().getLocalFunctionByModule("iddpm.cu", "p_sample"); 
            }
            if(p_sample_last_function == null) {
            	p_sample_last_function = getCudaManager().getLocalFunctionByModule("iddpm.cu", "p_sample_last"); 
            }
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void init() {
        /**
         * 初始化cuda函数

         */
        initFunction();
    }

    public int CAFFE_GET_BLOCKS(int N) {
        return (N + CAFFE_CUDA_NUM_THREADS - 1) / CAFFE_CUDA_NUM_THREADS;
    }

    public void mean(Tensor x,Tensor output) {
        try {
            /**
             * 设置入参
             *  float* x,
			    float* output,
			    int N, int W
             */
            kernelParameters = Pointer.to(Pointer.to(x.getGpuData()), Pointer.to(output.getGpuData()),
            		Pointer.to(new int[]{x.number}), Pointer.to(new int[]{x.getOnceSize()}));
            cuLaunchKernel(mean_function, this.CAFFE_GET_BLOCKS(x.number), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            );
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
    
    public void add_mul(Tensor a, Tensor b,Tensor x,Tensor y,Tensor t,Tensor output) {
        try {
            /**
             * 设置入参
             *  float* x_start,
			    float* xt,
			    float* t,
			    float* output,
			    float* a,
			    float* b,
			    int N, int W
             */
            kernelParameters = Pointer.to(Pointer.to(x.getGpuData()), Pointer.to(y.getGpuData()), Pointer.to(t.getGpuData()), Pointer.to(output.getGpuData()),
            		Pointer.to(a.getGpuData()),Pointer.to(b.getGpuData()),
            		Pointer.to(new int[]{x.dataLength}), Pointer.to(new int[]{x.getOnceSize()}));
            cuLaunchKernel(add_function, this.CAFFE_GET_BLOCKS(x.dataLength), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            );
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
    
    public void sub_mul(Tensor a, Tensor b,Tensor x,Tensor y,Tensor t,Tensor output) {
        try {
            /**
             * 设置入参
             *  float* x_start,
			    float* xt,
			    float* t,
			    float* output,
			    float* a,
			    float* b,
			    int N, int W
             */
            kernelParameters = Pointer.to(Pointer.to(x.getGpuData()), Pointer.to(y.getGpuData()), Pointer.to(t.getGpuData()), Pointer.to(output.getGpuData()),
            		Pointer.to(a.getGpuData()),Pointer.to(b.getGpuData()),
            		Pointer.to(new int[]{x.dataLength}), Pointer.to(new int[]{x.getOnceSize()}));
            cuLaunchKernel(sub_function, this.CAFFE_GET_BLOCKS(x.dataLength), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            );
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
    
    public void where(Tensor a, Tensor b,Tensor t,Tensor output) {
        try {
            /**
             * 设置入参
             *  float* a,
			    float* b,
			    float* t,
			    float* output,
			    int N, int W
             */
            kernelParameters = Pointer.to(Pointer.to(a.getGpuData()), Pointer.to(b.getGpuData()), Pointer.to(t.getGpuData()), Pointer.to(output.getGpuData()),
            		Pointer.to(new int[]{a.dataLength}), Pointer.to(new int[]{a.getOnceSize()}));
            cuLaunchKernel(where_function, this.CAFFE_GET_BLOCKS(a.dataLength), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            );
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
    
    public void extract_into(Tensor a, Tensor t,Tensor output) {
        try {
            /**
             * 设置入参
             *  float* a,
			    float* t,
			    float* output,
			    int N, int W
             */
        	Pointer kernelParameters = Pointer.to(Pointer.to(a.getGpuData()), Pointer.to(t.getGpuData()), Pointer.to(output.getGpuData()),
            		Pointer.to(new int[]{output.dataLength}), Pointer.to(new int[]{output.getOnceSize()}));
            cuLaunchKernel(eit_function, this.CAFFE_GET_BLOCKS(output.dataLength), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            );
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
    
    public void model_log_variance(Tensor var,Tensor max_log,Tensor min_log,Tensor output) {
        try {
            /**
             * 设置入参
             *  float* var,
			    float* max_log,
			    float* min_log,
			    float* output,
			    int N
             */
            kernelParameters = Pointer.to(Pointer.to(var.getGpuData()), Pointer.to(max_log.getGpuData()), Pointer.to(min_log.getGpuData()), Pointer.to(output.getGpuData()),
            		Pointer.to(new int[]{output.dataLength}));
            cuLaunchKernel(model_log_variance_function, this.CAFFE_GET_BLOCKS(output.dataLength), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            );
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
    
    public void normal_kl(Tensor mean1, Tensor logvar1, Tensor mean2,Tensor logvar2,Tensor kl) {
        try {
            /**
             * 设置入参
             *  float* mean1,
			    float* logvar1,
			    float* mean2,
			    float* logvar2,
			    float* output,
			    int N
             */
            kernelParameters = Pointer.to(Pointer.to(mean1.getGpuData()), Pointer.to(logvar1.getGpuData()), Pointer.to(mean2.getGpuData()), Pointer.to(logvar2.getGpuData()),
            		Pointer.to(kl.getGpuData()), Pointer.to(new int[]{mean1.dataLength}));
            cuLaunchKernel(normal_kl_function, this.CAFFE_GET_BLOCKS(mean1.dataLength), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            );
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
    
    public void normal_kl_back(Tensor mean1, Tensor logvar1, Tensor mean2,Tensor logvar2,Tensor dlogvar2) {
        try {
            /**
             * 设置入参
             *  float* mean1,
			    float* logvar1,
			    float* mean2,
			    float* logvar2,
			    float* dlogvar2,
			    int N
             */
            kernelParameters = Pointer.to(Pointer.to(mean1.getGpuData()), Pointer.to(logvar1.getGpuData()), Pointer.to(mean2.getGpuData()), Pointer.to(logvar2.getGpuData()),
            		Pointer.to(dlogvar2.getGpuData()), Pointer.to(new int[]{mean1.dataLength}));
            cuLaunchKernel(normal_kl_back_function, this.CAFFE_GET_BLOCKS(mean1.dataLength), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            );
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
    
    public void discretized_gaussian_log_likelihood(Tensor x, Tensor means, Tensor log_scales,Tensor decoder_nll) {
        try {
            /**
             * 设置入参
             *  float* x,
			    float* means,
			    float* log_scales,
			    float* output,
			    int N
             */
            kernelParameters = Pointer.to(Pointer.to(x.getGpuData()), Pointer.to(means.getGpuData()), Pointer.to(log_scales.getGpuData()),
            		Pointer.to(decoder_nll.getGpuData()), Pointer.to(new int[]{x.dataLength}));
            cuLaunchKernel(discretized_gaussian_log_likelihood_function, this.CAFFE_GET_BLOCKS(x.dataLength), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            );
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
    
    public void discretized_gaussian_log_likelihood_back(Tensor x, Tensor means, Tensor log_scales,Tensor dlogvar) {
        try {
            /**
             * 设置入参
             *  float* x,
			    float* means,
			    float* log_scales,
			    float* dlogvar,
			    int N
             */
            kernelParameters = Pointer.to(Pointer.to(x.getGpuData()), Pointer.to(means.getGpuData()), Pointer.to(log_scales.getGpuData()),
            		Pointer.to(dlogvar.getGpuData()), Pointer.to(new int[]{x.dataLength}));
            cuLaunchKernel(discretized_gaussian_log_likelihood_back_function, this.CAFFE_GET_BLOCKS(x.dataLength), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            );
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
    
    public void dvar_back(Tensor maxLog, Tensor minLog, Tensor delta,Tensor diff) {
        try {
            /**
             * 设置入参
             *  float* maxLog,
			    float* minLog,
			    float* delta,
			    float* diff,
			    int N
             */
            kernelParameters = Pointer.to(Pointer.to(maxLog.getGpuData()), Pointer.to(minLog.getGpuData()), Pointer.to(delta.getGpuData()),
            		Pointer.to(diff.getGpuData()), Pointer.to(new int[]{delta.dataLength}));
            cuLaunchKernel(dvar_back_function, this.CAFFE_GET_BLOCKS(delta.dataLength), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            );
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
    
    public void get_score_from_velocity(Tensor vt,Tensor xt,float t,Tensor score) {
        try {
            /**
             * 设置入参
             *  float* vt,
			    float* xt,
			    float* t,
			    float* score,
			    int N, int W
             */
            kernelParameters = Pointer.to(Pointer.to(vt.getGpuData()), Pointer.to(xt.getGpuData()), Pointer.to(new float[]{t}), Pointer.to(score.getGpuData()),
            		Pointer.to(new int[]{vt.dataLength}), Pointer.to(new int[]{vt.getOnceSize()}));
            cuLaunchKernel(get_score_from_velocity_function, this.CAFFE_GET_BLOCKS(vt.dataLength), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            );
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
    
    public void q_sample(Tensor latend,Tensor noise,Tensor t,Tensor output, Tensor target) {
        try {
            /**
             * 设置入参
             *  float* latend,
			    float* noise,
			    float* t,
			    float output,
			    float target,
			    int N, int W
             */
            kernelParameters = Pointer.to(Pointer.to(latend.getGpuData()),Pointer.to(noise.getGpuData()),Pointer.to(t.getGpuData()),Pointer.to(output.getGpuData()),Pointer.to(target.getGpuData()),
            		Pointer.to(new int[]{latend.dataLength}), Pointer.to(new int[]{latend.getOnceSize()}));
            cuLaunchKernel(q_sample_function, this.CAFFE_GET_BLOCKS(latend.dataLength), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            );
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
    
    public void q_sample(Tensor latend,Tensor noise,Tensor t,Tensor output) {
        try {
            /**
             * 设置入参
             *  float* latend,
			    float* noise,
			    float* t,
			    float output,
			    int N, int W
             */
            kernelParameters = Pointer.to(Pointer.to(latend.getGpuData()),Pointer.to(noise.getGpuData()),Pointer.to(t.getGpuData()),Pointer.to(output.getGpuData()),
            		Pointer.to(new int[]{latend.dataLength}), Pointer.to(new int[]{latend.getOnceSize()}));
            cuLaunchKernel(q_sample_no_target_function, this.CAFFE_GET_BLOCKS(latend.dataLength), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            );
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
    
    public void p_sample(Tensor v_cur,Tensor x_cur,Tensor s_cur,Tensor deps,float diffusion,float dt,Tensor x_next) {
        try {
            /**
             * 设置入参
             *  float* v_cur,
			    float* x_cur,
			    float* s_cur,
			    float* deps,
			    float* diffusion,
			    float* dt,
			    float* x_next,
			    int N, int W
             */
            kernelParameters = Pointer.to(Pointer.to(v_cur.getGpuData()), Pointer.to(x_cur.getGpuData()), Pointer.to(s_cur.getGpuData()), Pointer.to(deps.getGpuData()),
            		Pointer.to(new float[]{diffusion}), Pointer.to(new float[]{dt}), Pointer.to(x_next.getGpuData()),
            		Pointer.to(new int[]{v_cur.dataLength}), Pointer.to(new int[]{v_cur.getOnceSize()}));
            cuLaunchKernel(p_sample_function, this.CAFFE_GET_BLOCKS(v_cur.dataLength), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            );
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
    
    public void p_sample_last(Tensor v_cur,Tensor x_cur,Tensor s_cur,float diffusion,float dt,Tensor x_next) {
        try {
            /**
             * 设置入参
             *  float* v_cur,
			    float* x_cur,
			    float* s_cur,
			    float* diffusion,
			    float* dt,
			    float* x_next,
			    int N, int W
             */
            kernelParameters = Pointer.to(Pointer.to(v_cur.getGpuData()), Pointer.to(x_cur.getGpuData()), Pointer.to(s_cur.getGpuData()),
            		Pointer.to(new float[]{diffusion}), Pointer.to(new float[]{dt}), Pointer.to(x_next.getGpuData()),
            		Pointer.to(new int[]{v_cur.dataLength}), Pointer.to(new int[]{v_cur.getOnceSize()}));
            cuLaunchKernel(p_sample_last_function, this.CAFFE_GET_BLOCKS(v_cur.dataLength), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            );
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
    
    public void checkCUDA(int code) {
        if (code != cudaError.cudaSuccess) {
            System.err.println("Error code " + code + ":" + cudaError.stringFor(code));
        }
    }
    
}

