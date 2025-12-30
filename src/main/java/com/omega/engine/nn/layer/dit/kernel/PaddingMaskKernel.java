package com.omega.engine.nn.layer.dit.kernel;

import static jcuda.driver.JCudaDriver.cuLaunchKernel;

import com.omega.common.utils.JsonUtils;
import com.omega.common.utils.RandomUtils;
import com.omega.engine.gpu.CUDAKernel;
import com.omega.engine.gpu.CUDAManager;
import com.omega.engine.tensor.Tensor;

import jcuda.Pointer;
import jcuda.driver.CUfunction;
import jcuda.runtime.cudaError;

public class PaddingMaskKernel extends CUDAKernel {
    private CUfunction function;
    private CUfunction function2;
    private CUfunction igone_function;
    private CUfunction igone_function2;
    private CUfunction mask_function;
    private CUfunction mask_back_function;
    private CUfunction mask_igone_function;
    private CUfunction mask_diff_function;
    private CUfunction mask_igone_diff_function;
    private int CAFFE_CUDA_NUM_THREADS = 1024;
    private Pointer kernelParameters;

    public PaddingMaskKernel(CUDAManager cudaManager) {
        super(cudaManager);
        init();
    }

    public void initFunction() {
        try {
            if (function == null) {
                function = getCudaManager().getLocalFunctionByModule("PaddingMaskKernel.cu", "set_ids");
            }
            if (function2 == null) {
            	function2 = getCudaManager().getLocalFunctionByModule("PaddingMaskKernel.cu", "set_ids_back");
            }
            if (igone_function == null) {
            	igone_function = getCudaManager().getLocalFunctionByModule("PaddingMaskKernel.cu", "set_ids_igone");
            }
            if (igone_function2 == null) {
            	igone_function2 = getCudaManager().getLocalFunctionByModule("PaddingMaskKernel.cu", "set_ids_igone_back");
            }
            if (mask_function == null) {
            	mask_function = getCudaManager().getLocalFunctionByModule("PaddingMaskKernel.cu", "set_mask");
            }
            if (mask_igone_function == null) {
            	mask_igone_function = getCudaManager().getLocalFunctionByModule("PaddingMaskKernel.cu", "set_mask_igone");
            }
            if (mask_back_function == null) {
            	mask_back_function = getCudaManager().getLocalFunctionByModule("PaddingMaskKernel.cu", "set_mask_back_igone");
            }
            if (mask_diff_function == null) {
            	mask_diff_function = getCudaManager().getLocalFunctionByModule("PaddingMaskKernel.cu", "mask_diff");
            }
            if (mask_igone_diff_function == null) {
            	mask_igone_diff_function = getCudaManager().getLocalFunctionByModule("PaddingMaskKernel.cu", "mask_igone_diff");
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
    
    public static void main(String args[]) {
    	
    	int M = 9990;
    	int N = 768;
    	
    	Tensor x = new Tensor(M, 1, 1, N, RandomUtils.gaussianRandom(M*N, 1.0f), true);
    	Tensor gpu = new Tensor(1, 1, 1, N, true);
    	
    	float[] cpu = new float[N];
    	
    	for(int n = 0;n<N;n++) {
    		for(int m = 0;m<M;m++) {
    			cpu[n] += x.data[m * N + n];
    		}
    	}
    	
    	CUDAManager cudaManager = new CUDAManager(0);
    	PaddingMaskKernel kernel = new PaddingMaskKernel(cudaManager);
    	
    	kernel.mask_diff(x, gpu, M, N);
    	
    	gpu.showDM();
    	System.err.println(JsonUtils.toJson(cpu));
    	
//    	int N = 3;
//    	int C = 3;
//    	int H = 4;
//    	int W = 4;
//    	
//    	Tensor x = new Tensor(N, C, H, W, RandomUtils.order(N * C * H * W, 1, 0), true);
//    	Tensor out = new Tensor(N, C, H, W, true);
//    	
//    	CUDAManager cudaManager = new CUDAManager(0);
//    	OPKernel op = new OPKernel(cudaManager);
//    	op.roll_dims0(x, out, 1);
//    	
//    	out.showDM();
    }

    public int CAFFE_GET_BLOCKS(int N) {
        return (N + CAFFE_CUDA_NUM_THREADS - 1) / CAFFE_CUDA_NUM_THREADS;
    }
    
    public void set_mask(Tensor mask, Tensor out, int W) {
    	 try {
             /**
              * 设置入参
              * const size_t size, const float *mask, float *out, const int W
              */
             kernelParameters = Pointer.to(Pointer.to(new long[]{out.dataLength}), Pointer.to(mask.getGpuData()), Pointer.to(out.getGpuData()), Pointer.to(new int[]{W}));
             cuLaunchKernel(mask_function, this.CAFFE_GET_BLOCKS(out.dataLength), 1, 1,      // Grid dimension
                     CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                     0, null,               // Shared memory size and stream
                     kernelParameters, null // Kernel- and extra parameters
             );
         } catch (Exception e) {
             // TODO: handle exception
             e.printStackTrace();
         }
    }
    
    public void set_mask_igone(Tensor mask, Tensor out, int T, int igoneT, int W) {
    	try {
            /**
             * 设置入参
             * const size_t size, const float *mask, float *out, const int W
             */
            kernelParameters = Pointer.to(Pointer.to(new long[]{out.dataLength}), Pointer.to(mask.getGpuData()), Pointer.to(out.getGpuData()), Pointer.to(new int[]{T}), Pointer.to(new int[]{igoneT}), Pointer.to(new int[]{W}));
            cuLaunchKernel(mask_igone_function, this.CAFFE_GET_BLOCKS(out.dataLength), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            );
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
   }
   
   public void set_mask_back_igone(Tensor dout, int T, int igoneT, int W) {
    	try {
            /**
             * 设置入参
             * const size_t size, float *dout, const int T, const int igoneT, const int W
             */
            kernelParameters = Pointer.to(Pointer.to(new long[]{dout.dataLength}), Pointer.to(dout.getGpuData()), Pointer.to(new int[]{T}), Pointer.to(new int[]{igoneT}), Pointer.to(new int[]{W}));
            cuLaunchKernel(mask_back_function, this.CAFFE_GET_BLOCKS(dout.dataLength), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            );
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
   }
    
   public void forward(Tensor x, Tensor mask, Tensor idskeep, Tensor output, int FT, int T, int W) {
        try {
        	/**
        	 * set mask value
        	 */
        	set_mask(mask, output, W);
            /**
             * 设置入参
             * const size_t size, const float *x, const float *idskeep, float *out, const int FT, const int T, const int W
             */
            kernelParameters = Pointer.to(Pointer.to(new long[]{x.dataLength}), Pointer.to(x.getGpuData()), Pointer.to(idskeep.getGpuData()), Pointer.to(output.getGpuData()), Pointer.to(new int[]{FT}), Pointer.to(new int[]{T}), Pointer.to(new int[]{W}));
            cuLaunchKernel(function, this.CAFFE_GET_BLOCKS(x.dataLength), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            );
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
    
    public void forward(Tensor x, Tensor mask, Tensor idskeep, Tensor output, int FT, int T, int igoneT, int W) {
        try {
        	/**
        	 * set mask value
        	 */
        	set_mask(mask, output, W);
            /**
             * 设置入参
             * const size_t size, const float *x, const float *idskeep, float *out, const int FT, const int T, const int W
             */
            kernelParameters = Pointer.to(Pointer.to(new long[]{x.dataLength}), Pointer.to(x.getGpuData()), Pointer.to(idskeep.getGpuData()), Pointer.to(output.getGpuData()), Pointer.to(new int[]{FT}), Pointer.to(new int[]{T}), Pointer.to(new int[]{igoneT}), Pointer.to(new int[]{W}));
            cuLaunchKernel(igone_function, this.CAFFE_GET_BLOCKS(x.dataLength), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            );
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
    
    public void backward(Tensor delta, Tensor idskeep, Tensor dx, Tensor dw, int FT, int T, int W) {
    	set_ids_back(dx, delta, idskeep, FT, T, W);
    	mask_diff(delta, dw, delta.number, W);
    }
    
    public void backward(Tensor delta, Tensor idskeep, Tensor dx, Tensor dw, int FT, int T, int igoneT, int W) {
//    	System.err.println("T:"+T+",FT:"+FT+",TT:"+igoneT);
//    	idskeep.showDMByOffsetRed(0, 64, "idskeep");
//    	int idx = (int) idskeep.syncHost()[0] + 1;
//    	System.err.println(idx);
//    	delta.showDMByOffsetRed((idx + igoneT) * W, W, "delta1");
    	set_ids_back(dx, delta, idskeep, FT, T, igoneT, W);
//    	delta.showDMByOffsetRed((idx + igoneT) * W, W, "delta2");
//    	System.err.println("-----");
    	mask_diff(delta, dw, delta.number, W);
    }
    
    public void set_ids_back(Tensor dx, Tensor delta, Tensor idskeep, int FT, int T, int W) {
        try {
            /**
             * 设置入参
             * const size_t size, float *dx, const float *idskeep, const float *dout, const int FT, const int T, const int W
             */
            kernelParameters = Pointer.to(Pointer.to(new long[]{dx.dataLength}), Pointer.to(dx.getGpuData()), Pointer.to(idskeep.getGpuData()), Pointer.to(delta.getGpuData()), Pointer.to(new int[]{FT}), Pointer.to(new int[]{T}), Pointer.to(new int[]{W}));
            cuLaunchKernel(function2, this.CAFFE_GET_BLOCKS(dx.dataLength), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            );
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
    
    public void set_ids_back(Tensor dx, Tensor delta, Tensor idskeep, int FT, int T, int igoneT, int W) {
        try {
            /**
             * 设置入参
             * const size_t size, float *dx, const float *idskeep, const float *dout, const int FT, const int T, const int W
             */
            kernelParameters = Pointer.to(Pointer.to(new long[]{dx.dataLength}), Pointer.to(dx.getGpuData()), Pointer.to(idskeep.getGpuData()), Pointer.to(delta.getGpuData()), Pointer.to(new int[]{FT}), Pointer.to(new int[]{T}), Pointer.to(new int[]{igoneT}), Pointer.to(new int[]{W}));
            cuLaunchKernel(igone_function2, this.CAFFE_GET_BLOCKS(dx.dataLength), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            );
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
    
    public void mask_diff(Tensor delta, Tensor dw, int rows, int cols) {
        try {
        	int gridSize = cols;
            int BLOCK_SIZE = 256;
            int block_size = Math.min(BLOCK_SIZE, rows);
        	/**
             * 设置入参
             * const float *dout, float *dw, const int rows, conts int cols
             */
            kernelParameters = Pointer.to(Pointer.to(delta.getGpuData()), Pointer.to(dw.getGpuData()), Pointer.to(new int[]{rows}), Pointer.to(new int[]{cols}));
            cuLaunchKernel(mask_diff_function, gridSize, 1, 1,      // Grid dimension
            		block_size, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            );
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
    
    public void mask_igone_diff(Tensor delta, Tensor dw, int rows, int cols, int T, int igoneT) {
        try {
            int block_size = 256;
            int grid_dim =  (cols + block_size - 1) / block_size;
        	/**
             * 设置入参
             * const float* __restrict__ inp,float* __restrict__ out, int M, int N 
             */
            kernelParameters = Pointer.to(Pointer.to(delta.getGpuData()), Pointer.to(dw.getGpuData()), Pointer.to(new int[]{rows}), Pointer.to(new int[]{cols}), Pointer.to(new int[]{T}), Pointer.to(new int[]{igoneT}));
            cuLaunchKernel(mask_igone_diff_function, grid_dim, 1, 1,      // Grid dimension
            		block_size, 1, 1,      // Block dimension
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

