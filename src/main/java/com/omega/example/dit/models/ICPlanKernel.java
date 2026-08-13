package com.omega.example.dit.models;

import static jcuda.driver.JCudaDriver.cuLaunchKernel;

import com.omega.engine.gpu.CUDAKernel;
import com.omega.engine.gpu.CUDAManager;
import com.omega.engine.tensor.Tensor;

import jcuda.Pointer;
import jcuda.driver.CUfunction;
import jcuda.runtime.cudaError;

public class ICPlanKernel extends CUDAKernel {
	
    private CUfunction compute_xt_function;
    private CUfunction compute_ut_function;
    private CUfunction compute_v_function;
    private CUfunction compute_dv_function;
    private CUfunction compute_z_next_function;
    
    private CUfunction compute_xt_ft_function;
    private CUfunction compute_xt_ft_offset_function;
    private CUfunction compute_ut_ft_function;
    
    private CUfunction cosine_similarity_loss_function;
    private CUfunction cosine_similarity_loss_dim1_function;
    private CUfunction cosine_similarity_loss_back1_function;
    private CUfunction cosine_similarity_loss_back2_function;
    private CUfunction cosine_similarity_function;
    private CUfunction cosine_similarity_back_function;
    
    private CUfunction latend_norm_function;
    private CUfunction latend_un_norm_function;
    
    private CUfunction expand_mask_function;
    private CUfunction expand_mask_skip_text_function;
    private CUfunction expand_function;
    
    private int CAFFE_CUDA_NUM_THREADS = 1024;
    
    private Pointer kernelParameters;

    public ICPlanKernel(CUDAManager cudaManager) {
        super(cudaManager);
        init();
    }

    public static void main(String args[]) {
        
    }

    public void initFunction() {
        try {
            if (compute_xt_function == null) {
            	compute_xt_function = getCudaManager().getLocalFunctionByModule("icplan.cu", "compute_xt");
            }
            if (compute_ut_function == null) {
            	compute_ut_function = getCudaManager().getLocalFunctionByModule("icplan.cu", "compute_ut");
            }
            if (compute_xt_ft_function == null) {
            	compute_xt_ft_function = getCudaManager().getLocalFunctionByModule("icplan.cu", "compute_xt_ft");
            }
            if (compute_xt_ft_offset_function == null) {
            	compute_xt_ft_offset_function = getCudaManager().getLocalFunctionByModule("icplan.cu", "compute_xt_ft_offset");
            }
            if (compute_ut_ft_function == null) {
            	compute_ut_ft_function = getCudaManager().getLocalFunctionByModule("icplan.cu", "compute_ut_ft");
            }
            if (compute_v_function == null) {
            	compute_v_function = getCudaManager().getLocalFunctionByModule("icplan.cu", "compute_v");
            }
            if (compute_dv_function == null) {
            	compute_dv_function = getCudaManager().getLocalFunctionByModule("icplan.cu", "compute_dv");
            }
            if (compute_z_next_function == null) {
            	compute_z_next_function = getCudaManager().getLocalFunctionByModule("icplan.cu", "compute_z_next");
            }
            if (cosine_similarity_loss_function == null) {
            	cosine_similarity_loss_function = getCudaManager().getLocalFunctionByModule("icplan.cu", "cosine_similarity_loss");
            }
            if(cosine_similarity_loss_dim1_function == null) {
            	cosine_similarity_loss_dim1_function = getCudaManager().getLocalFunctionByModule("icplan.cu", "cosine_similarity_loss_dim1");
            }
            if (cosine_similarity_loss_back1_function == null) {
            	cosine_similarity_loss_back1_function = getCudaManager().getLocalFunctionByModule("icplan.cu", "cosine_similarity_loss_back1");
            }
            if (cosine_similarity_loss_back2_function == null) {
            	cosine_similarity_loss_back2_function = getCudaManager().getLocalFunctionByModule("icplan.cu", "cosine_similarity_loss_back2");
            }
            if (cosine_similarity_function == null) {
            	cosine_similarity_function = getCudaManager().getLocalFunctionByModule("icplan.cu", "cosine_similarity");
            }
            if (cosine_similarity_back_function == null) {
            	cosine_similarity_back_function = getCudaManager().getLocalFunctionByModule("icplan.cu", "cosine_similarity_back");
            }
            if(latend_norm_function == null) {
            	latend_norm_function = getCudaManager().getLocalFunctionByModule("icplan.cu", "latend_norm"); 
            }
            if(latend_un_norm_function == null) {
            	latend_un_norm_function = getCudaManager().getLocalFunctionByModule("icplan.cu", "latend_un_norm");
            }
            if(expand_mask_function == null) {
            	expand_mask_function = getCudaManager().getLocalFunctionByModule("icplan.cu", "expand_mask"); 
            }
            if(expand_mask_skip_text_function == null) {
                expand_mask_skip_text_function = getCudaManager().getLocalFunctionByModule("icplan.cu", "expand_mask_skip_text");
            }
            if(expand_function == null) {
            	expand_function = getCudaManager().getLocalFunctionByModule("icplan.cu", "expand_"); 
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
    
    public void latend_norm(Tensor x,Tensor mean,Tensor std) {
        try {
            /**
             * 设置入参
			    float* x1,
			    float* mean,
			    float* std
			    int N,
			    int C
             */
            kernelParameters = Pointer.to(Pointer.to(x.getGpuData()), Pointer.to(mean.getGpuData()), Pointer.to(std.getGpuData()),
            		Pointer.to(new int[]{x.dataLength}), Pointer.to(new int[]{mean.dataLength}), Pointer.to(new int[]{x.height * x.width}));
            cuLaunchKernel(latend_norm_function, this.CAFFE_GET_BLOCKS(x.dataLength), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            );
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void latend_un_norm(Tensor x,Tensor mean,Tensor std) {
        try {
            /**
             * 设置入参
			    float* x1,
			    float* mean,
			    float* std
			    int N,
			    int C
             */
            kernelParameters = Pointer.to(Pointer.to(x.getGpuData()), Pointer.to(mean.getGpuData()), Pointer.to(std.getGpuData()),
            		Pointer.to(new int[]{x.dataLength}), Pointer.to(new int[]{mean.dataLength}), Pointer.to(new int[]{x.height * x.width}));
            cuLaunchKernel(latend_un_norm_function, this.CAFFE_GET_BLOCKS(x.dataLength), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            );
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
    
    public void latend3d_norm(Tensor x,Tensor mean,Tensor std, int thw) {
        try {
            /**
             * 设置入参
			    float* x1,
			    float* mean,
			    float* std
			    int N,
			    int C
             */
            kernelParameters = Pointer.to(Pointer.to(x.getGpuData()), Pointer.to(mean.getGpuData()), Pointer.to(std.getGpuData()),
            		Pointer.to(new int[]{x.dataLength}), Pointer.to(new int[]{mean.dataLength}), Pointer.to(new int[]{thw}));
            cuLaunchKernel(latend_norm_function, this.CAFFE_GET_BLOCKS(x.dataLength), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            );
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void latend3d_un_norm(Tensor x,Tensor mean,Tensor std, int thw) {
        try {
            /**
             * 设置入参
			    float* x1,
			    float* mean,
			    float* std
			    int N,
			    int C
             */
            kernelParameters = Pointer.to(Pointer.to(x.getGpuData()), Pointer.to(mean.getGpuData()), Pointer.to(std.getGpuData()),
            		Pointer.to(new int[]{x.dataLength}), Pointer.to(new int[]{mean.dataLength}), Pointer.to(new int[]{thw}));
            cuLaunchKernel(latend_un_norm_function, this.CAFFE_GET_BLOCKS(x.dataLength), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            );
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
    
    public void compute_xt(Tensor latend,Tensor noise,Tensor t,Tensor xt) {
        try {
            /**
             * 设置入参
             *  float* latend,
			    float* noise,
			    float* t,
			    float* output,
			    int N, int W
             */
            kernelParameters = Pointer.to(Pointer.to(latend.getGpuData()), Pointer.to(noise.getGpuData()), Pointer.to(t.getGpuData()), Pointer.to(xt.getGpuData()),
            		Pointer.to(new int[]{latend.dataLength}), Pointer.to(new int[]{latend.getOnceSize()}));
            cuLaunchKernel(compute_xt_function, this.CAFFE_GET_BLOCKS(latend.dataLength), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            );
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
    
    public void compute_xt(Tensor latend,Tensor noise,Tensor t,Tensor xt, int C, int gh, int gw, int ps) {
        try {
            /**
             * 设置入参
			    float* latend,
			    float* noise,
			    float* t,
			    float* output,
			    int N, int C, int gridH, int gridW, int patchSize
             */
        	Pointer kernelParameters = Pointer.to(Pointer.to(latend.getGpuData()), Pointer.to(noise.getGpuData()), Pointer.to(t.getGpuData()), Pointer.to(xt.getGpuData()),
            		Pointer.to(new int[]{latend.dataLength}), Pointer.to(new int[]{C}), Pointer.to(new int[]{gh}), Pointer.to(new int[]{gw}), Pointer.to(new int[]{ps}));
            cuLaunchKernel(compute_xt_ft_function, this.CAFFE_GET_BLOCKS(latend.dataLength), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            );
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
    
    public void compute_xt(Tensor latend,Tensor noise,Tensor t,Tensor xt, int C, int gh, int gw, int ps, int offset) {
        try {
            /**
             * 设置入参
			    float* latend,
			    float* noise,
			    float* t,
			    float* output,
			    int N, int C, int gridH, int gridW, int patchSize
             */
        	Pointer kernelParameters = Pointer.to(Pointer.to(latend.getGpuData()), Pointer.to(noise.getGpuData()), Pointer.to(t.getGpuData()), Pointer.to(xt.getGpuData()),
            		Pointer.to(new int[]{latend.dataLength}), Pointer.to(new int[]{C}), Pointer.to(new int[]{gh}), Pointer.to(new int[]{gw}), Pointer.to(new int[]{ps}), Pointer.to(new int[]{offset}));
            cuLaunchKernel(compute_xt_ft_offset_function, this.CAFFE_GET_BLOCKS(latend.dataLength), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            );
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
    
    public void compute_ut(Tensor latend,Tensor noise,Tensor t,Tensor ut) {
        try {
        	/**
             * 设置入参
             *  float* latend,
			    float* noise,
			    float* t,
			    float* output,
			    int N, int W
             */
            kernelParameters = Pointer.to(Pointer.to(latend.getGpuData()), Pointer.to(noise.getGpuData()), Pointer.to(t.getGpuData()), Pointer.to(ut.getGpuData()),
            		Pointer.to(new int[]{latend.dataLength}), Pointer.to(new int[]{latend.getOnceSize()}));
            cuLaunchKernel(compute_ut_function, this.CAFFE_GET_BLOCKS(latend.dataLength), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            );
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
    
    public void compute_ut(Tensor latend,Tensor noise,Tensor t,Tensor ut, int C, int gh, int gw, int ps) {
        try {
        	/**
             * 设置入参
             *  float* latend,
			    float* noise,
			    float* t,
			    float* output,
			    int N, int W
             */
        	Pointer kernelParameters = Pointer.to(Pointer.to(latend.getGpuData()), Pointer.to(noise.getGpuData()), Pointer.to(t.getGpuData()), Pointer.to(ut.getGpuData()),
            		Pointer.to(new int[]{latend.dataLength}), Pointer.to(new int[]{C}), Pointer.to(new int[]{gh}), Pointer.to(new int[]{gw}), Pointer.to(new int[]{ps}));
            cuLaunchKernel(compute_ut_ft_function, this.CAFFE_GET_BLOCKS(latend.dataLength), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            );
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
    
    public void compute_v(Tensor x,Tensor z,Tensor t,Tensor v, float t_eps) {
        try {
            /**
             * 设置入参
             *  float* x,
			    float* z,
			    float* t,
			    float* output,
			    float t_eps,
			    int N, int W
             */
            kernelParameters = Pointer.to(Pointer.to(x.getGpuData()), Pointer.to(z.getGpuData()), Pointer.to(t.getGpuData()), Pointer.to(v.getGpuData()),Pointer.to(new float[]{t_eps}),
            		Pointer.to(new int[]{x.dataLength}), Pointer.to(new int[]{x.getOnceSize()}));
            cuLaunchKernel(compute_v_function, this.CAFFE_GET_BLOCKS(x.dataLength), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            );
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
    
    public void compute_dv(Tensor delta,Tensor t,Tensor dx, float t_eps) {
        try {
            /**
             * 设置入参
             *  float* delta,
			    float* t,
			    float* dx,
			    float t_eps,
			    int N, int W
             */
            kernelParameters = Pointer.to(Pointer.to(delta.getGpuData()), Pointer.to(t.getGpuData()), Pointer.to(dx.getGpuData()),Pointer.to(new float[]{t_eps}),
            		Pointer.to(new int[]{delta.dataLength}), Pointer.to(new int[]{delta.getOnceSize()}));
            cuLaunchKernel(compute_dv_function, this.CAFFE_GET_BLOCKS(delta.dataLength), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            );
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
    
    public void compute_z_next(Tensor v_pred,Tensor z,float t, float t_next, Tensor z_next) {
        try {
            /**
             * 设置入参
             *  float* v_pred,
			    float* z,
			    float t,
			    float t_next,
			    float* output,
			    int N
             */
            kernelParameters = Pointer.to(Pointer.to(v_pred.getGpuData()), Pointer.to(z.getGpuData()),Pointer.to(new float[]{t}), Pointer.to(new float[]{t_next}), Pointer.to(z_next.getGpuData()),
            		Pointer.to(new int[]{v_pred.dataLength}));
            cuLaunchKernel(compute_z_next_function, this.CAFFE_GET_BLOCKS(v_pred.dataLength), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            );
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
    
    public void cosine_similarity_loss(Tensor x1,Tensor norm1,Tensor x2,Tensor norm2,Tensor loss) {
    	try {
        	/**
             * 设置入参
             *  float* x1,
			    float* norm1,
			    float* x2,
			    float* norm2,
			    float* out,
			    int N,
			    int C,
			    int W
             */
            kernelParameters = Pointer.to(Pointer.to(x1.getGpuData()), Pointer.to(norm1.getGpuData()), Pointer.to(x2.getGpuData()), Pointer.to(norm2.getGpuData()),
            		Pointer.to(loss.getGpuData()),Pointer.to(new int[]{x1.dataLength}), Pointer.to(new int[]{x1.channel}),Pointer.to(new int[]{x1.height * x1.width}));
            cuLaunchKernel(cosine_similarity_loss_function, this.CAFFE_GET_BLOCKS(x1.dataLength), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            );
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
    
    public void cosine_similarity_loss_dim1(Tensor x1,Tensor norm1,Tensor x2,Tensor norm2,Tensor loss) {
    	try {
        	/**
             * 设置入参
             *  float* x1,
			    float* norm1,
			    float* x2,
			    float* norm2,
			    float* out,
			    int N,
			    int C,
			    int W
             */
            kernelParameters = Pointer.to(Pointer.to(x1.getGpuData()), Pointer.to(norm1.getGpuData()), Pointer.to(x2.getGpuData()), Pointer.to(norm2.getGpuData()),
            		Pointer.to(loss.getGpuData()),Pointer.to(new int[]{loss.dataLength}), Pointer.to(new int[]{x1.channel}),Pointer.to(new int[]{x1.height * x1.width}));
            cuLaunchKernel(cosine_similarity_loss_dim1_function, this.CAFFE_GET_BLOCKS(loss.dataLength), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            );
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
    
    public void cosine_similarity_loss_back1(float delta,Tensor x1,Tensor norm1,Tensor x2,Tensor norm2,Tensor dx1) {
    	try {
        	/**
             * 设置入参
             *  float delta,
			    float* x1,
			    float* norm1,
			    float* x2,
			    float* norm2,
			    float* dx1,
			    int N,
			    int C,
			    int W
             */
            kernelParameters = Pointer.to(Pointer.to(new float[]{delta}),Pointer.to(x1.getGpuData()), Pointer.to(norm1.getGpuData()), Pointer.to(x2.getGpuData()), Pointer.to(norm2.getGpuData()),
            		Pointer.to(dx1.getGpuData()),Pointer.to(new int[]{norm1.dataLength}), Pointer.to(new int[]{x1.channel}),Pointer.to(new int[]{x1.height * x1.width}));
            cuLaunchKernel(cosine_similarity_loss_back1_function, this.CAFFE_GET_BLOCKS(norm1.dataLength), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            );
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
    
    public void cosine_similarity_loss_back2(float delta,Tensor x1,Tensor norm1,Tensor x2,Tensor norm2,Tensor dnorm1) {
    	try {
        	/**
             * 设置入参
             *  float delta,
			    float* x1,
			    float* norm1,
			    float* x2,
			    float* norm2,
			    float* dnorm1,
			    int N,
			    int C,
			    int W
             */
            kernelParameters = Pointer.to(Pointer.to(new float[]{delta}),Pointer.to(x1.getGpuData()), Pointer.to(norm1.getGpuData()), Pointer.to(x2.getGpuData()), Pointer.to(norm2.getGpuData()),
            		Pointer.to(dnorm1.getGpuData()),Pointer.to(new int[]{norm1.dataLength}), Pointer.to(new int[]{x1.channel}),Pointer.to(new int[]{x1.height * x1.width}));
            cuLaunchKernel(cosine_similarity_loss_back2_function, this.CAFFE_GET_BLOCKS(norm1.dataLength), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            );
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void cosine_similarity(Tensor x1, Tensor x2, Tensor output, float eps) {
    	cosine_similarity(x1, x2, output, 3, eps);
    }

    public void cosine_similarity(Tensor x1, Tensor x2, Tensor output, int dim, float eps) {
    	try {
    		if (dim < 0) {
    			dim += 4;
    		}
    		if (dim < 0 || dim > 3) {
    			throw new IllegalArgumentException("cosine_similarity dim must be in [-4, 3], got: " + dim);
    		}
    		int outer = output.dataLength;
            kernelParameters = Pointer.to(Pointer.to(x1.getGpuData()), Pointer.to(x2.getGpuData()), Pointer.to(output.getGpuData()),
            		Pointer.to(new int[]{outer}), Pointer.to(new int[]{x1.number}), Pointer.to(new int[]{x1.channel}),
            		Pointer.to(new int[]{x1.height}), Pointer.to(new int[]{x1.width}), Pointer.to(new int[]{dim}),
            		Pointer.to(new float[]{eps}));
            cuLaunchKernel(cosine_similarity_function, this.CAFFE_GET_BLOCKS(outer), 1, 1,
                    CAFFE_CUDA_NUM_THREADS, 1, 1,
                    0, null,
                    kernelParameters, null
            );
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
    
    public void cosine_similarity_back(Tensor gradOut, Tensor x1, Tensor x2, Tensor dx1, float eps) {
    	cosine_similarity_back(gradOut, x1, x2, dx1, 3, eps);
    }

    public void cosine_similarity_back(Tensor gradOut, Tensor x1, Tensor x2, Tensor dx1, int dim, float eps) {
    	try {
    		if (dim < 0) {
    			dim += 4;
    		}
    		if (dim < 0 || dim > 3) {
    			throw new IllegalArgumentException("cosine_similarity dim must be in [-4, 3], got: " + dim);
    		}
    		int outer = gradOut.dataLength;
            kernelParameters = Pointer.to(Pointer.to(gradOut.getGpuData()), Pointer.to(x1.getGpuData()), Pointer.to(x2.getGpuData()), Pointer.to(dx1.getGpuData()),
            		Pointer.to(new int[]{outer}), Pointer.to(new int[]{x1.number}), Pointer.to(new int[]{x1.channel}),
            		Pointer.to(new int[]{x1.height}), Pointer.to(new int[]{x1.width}), Pointer.to(new int[]{dim}),
            		Pointer.to(new float[]{eps}));
            cuLaunchKernel(cosine_similarity_back_function, this.CAFFE_GET_BLOCKS(outer), 1, 1,
                    CAFFE_CUDA_NUM_THREADS, 1, 1,
                    0, null,
                    kernelParameters, null
            );
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
    
    public void expand_mask(Tensor a,Tensor b,Tensor mask,Tensor out, int W, float maskRatio) {
        try {
            /**
             * 设置入参
			    float* a,
			    float* b,
			    float* mask,
			    float* out,
			    int N,
			    int W,
			    float maskRatio
             */
        	Pointer kernelParameters = Pointer.to(Pointer.to(a.getGpuData()), Pointer.to(b.getGpuData()), Pointer.to(mask.getGpuData()), Pointer.to(out.getGpuData()),
            		Pointer.to(new int[]{out.dataLength}), Pointer.to(new int[]{W}), Pointer.to(new float[]{maskRatio}));
            cuLaunchKernel(expand_mask_function, this.CAFFE_GET_BLOCKS(out.dataLength), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            );
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void expand_mask_skip_text(Tensor a, Tensor b, Tensor mask, Tensor out, int W, int textTokenCount, float maskRatio) {
        if (textTokenCount < 0 || textTokenCount > W) {
            throw new IllegalArgumentException("textTokenCount must be in [0, W], got: " + textTokenCount);
        }
        try {
            Pointer parameters = Pointer.to(Pointer.to(a.getGpuData()), Pointer.to(b.getGpuData()),
                    Pointer.to(mask.getGpuData()), Pointer.to(out.getGpuData()),
                    Pointer.to(new int[]{out.dataLength}), Pointer.to(new int[]{W}),
                    Pointer.to(new int[]{textTokenCount}), Pointer.to(new float[]{maskRatio}));
            cuLaunchKernel(expand_mask_skip_text_function, this.CAFFE_GET_BLOCKS(out.dataLength), 1, 1,
                    CAFFE_CUDA_NUM_THREADS, 1, 1,
                    0, null,
                    parameters, null);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
    
    public void expand_(Tensor a,Tensor out, int W) {
        try {
            /**
             * 设置入参
			    float* a,
			    float* out,
			    int N,
			    int W
             */
        	Pointer kernelParameters = Pointer.to(Pointer.to(a.getGpuData()), Pointer.to(out.getGpuData()),
            		Pointer.to(new int[]{out.dataLength}), Pointer.to(new int[]{W}));
            cuLaunchKernel(expand_function, this.CAFFE_GET_BLOCKS(out.dataLength), 1, 1,      // Grid dimension
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

