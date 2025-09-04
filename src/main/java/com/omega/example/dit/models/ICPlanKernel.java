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
    
    private CUfunction cosine_similarity_loss_function;
    private CUfunction cosine_similarity_loss_back1_function;
    private CUfunction cosine_similarity_loss_back2_function;
    
    private CUfunction latend_norm_function;
    private CUfunction latend_un_norm_function;
    
    
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
            if (cosine_similarity_loss_function == null) {
            	cosine_similarity_loss_function = getCudaManager().getLocalFunctionByModule("icplan.cu", "cosine_similarity_loss");
            }
            if (cosine_similarity_loss_back1_function == null) {
            	cosine_similarity_loss_back1_function = getCudaManager().getLocalFunctionByModule("icplan.cu", "cosine_similarity_loss_back1");
            }
            if (cosine_similarity_loss_back2_function == null) {
            	cosine_similarity_loss_back2_function = getCudaManager().getLocalFunctionByModule("icplan.cu", "cosine_similarity_loss_back2");
            }
            if(latend_norm_function == null) {
            	latend_norm_function = getCudaManager().getLocalFunctionByModule("icplan.cu", "latend_norm"); 
            }
            if(latend_un_norm_function == null) {
            	latend_un_norm_function = getCudaManager().getLocalFunctionByModule("icplan.cu", "latend_un_norm");
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
    
    public void checkCUDA(int code) {
        if (code != cudaError.cudaSuccess) {
            System.err.println("Error code " + code + ":" + cudaError.stringFor(code));
        }
    }
    
}

