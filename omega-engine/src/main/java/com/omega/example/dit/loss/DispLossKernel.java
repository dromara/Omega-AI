package com.omega.example.dit.loss;

import static jcuda.driver.JCudaDriver.cuLaunchKernel;

import java.util.Map;

import com.omega.engine.gpu.CUDAKernel;
import com.omega.engine.gpu.CUDAManager;
import com.omega.engine.tensor.Tensor;
import com.omega.example.common.ModeLoaderlUtils;
import com.omega.example.transformer.utils.LagJsonReader;

import jcuda.Pointer;
import jcuda.driver.CUfunction;
import jcuda.runtime.cudaError;

public class DispLossKernel extends CUDAKernel {
	
    private CUfunction p_dist_function;
    private CUfunction p_dist_back_function;
    private CUfunction p_dist_back_function2;
    private CUfunction p_dist_back_add_function;
    private CUfunction log_exp_function;
    private CUfunction log_exp_back_function;
    
    private int CAFFE_CUDA_NUM_THREADS = 1024;
    
    private Pointer kernelParameters;

    public DispLossKernel(CUDAManager cudaManager) {
        super(cudaManager);
        init();
    }
    
    public static int get_pdist_dim(int N) {
    	return (int) (0.5 * N * (N - 1));
    }
    
    public static void main(String args[]) {
    	 int B = 8;
         int C = 256;
         int headNum = 768;
         
         int N = B;
         
 	     String inputPath = "D:\\models\\x2.json";
 	     Map<String, Object> datas = LagJsonReader.readJsonFileSmallWeight(inputPath);
 	     Tensor input = new Tensor(N, C, 1, headNum, true);
 	     ModeLoaderlUtils.loadData(input, datas, "x2", 3);
 	     
 	     int OW = (int) (0.5 * N * (N - 1));
 	     System.err.println(OW);
 	     Tensor pdist = new Tensor(1, 1, 1, OW, true);
 	     
         CUDAManager cudaManager = new CUDAManager(0);
         DispLossKernel kernel = new DispLossKernel(cudaManager);
         
         input.showDM();
         kernel.p_dist(input, pdist, 2);
         pdist.showDM();
         
         Tensor tmp = new Tensor(1, 1, 1, 1, true);
         Tensor loss = new Tensor(1, 1, 1, 1, true);
         kernel.log_exp(pdist, loss, tmp, N, input.getOnceSize());
         loss.showDM("loss");
         tmp.showDM("tmp");
         
         Tensor dy = new Tensor(1, 1, 1, OW, true);
         
         kernel.log_exp_back(pdist, dy, tmp, N, input.getOnceSize());
         
         dy.showDM("dy");

         Tensor buffer = null;
         Tensor dx = new Tensor(N, 1, C, headNum, true);
         kernel.p_dist_back(input, pdist, dy, buffer, dx, 2);
         dx.showDM();

    }

    public void initFunction() {
        try {
            if (p_dist_function == null) {
            	p_dist_function = getCudaManager().getLocalFunctionByModule("DispLoss.cu", "PDist_Other");
            }
            if (p_dist_back_function == null) {
            	p_dist_back_function = getCudaManager().getLocalFunctionByModule("DispLoss.cu", "PDist_Grad_P");
            }
            if (p_dist_back_function2 == null) {
            	p_dist_back_function2 = getCudaManager().getLocalFunctionByModule("DispLoss.cu", "pdist_backward_kernel_cuda_impl");
            }
            if (p_dist_back_add_function == null) {
            	p_dist_back_add_function = getCudaManager().getLocalFunctionByModule("DispLoss.cu", "AddBuffer");
            }
            if(log_exp_function == null) {
            	log_exp_function = getCudaManager().getLocalFunctionByModule("DispLoss.cu", "log_exp_kernel");
            }
            if(log_exp_back_function == null) {
            	log_exp_back_function = getCudaManager().getLocalFunctionByModule("DispLoss.cu", "log_exp_back_kernel");
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
    
    public void log_exp(Tensor pdist, Tensor loss, Tensor tmp, int N, int xDim) {
    	/**
         * 设置入参
         *  float* x,
		    float* output,
		    float tmp,
		    int N, int x_dim, int W
         */
        kernelParameters = Pointer.to(Pointer.to(pdist.getGpuData()), Pointer.to(loss.getGpuData()),Pointer.to(tmp.getGpuData()),
        		Pointer.to(new int[]{N}),Pointer.to(new int[]{xDim}), Pointer.to(new int[]{pdist.dataLength}));
        cuLaunchKernel(log_exp_function, 1, 1, 1,      // Grid dimension
                1, 1, 1,      // Block dimension
                0, null,               // Shared memory size and stream
                kernelParameters, null // Kernel- and extra parameters
        );
    }
    
    public void log_exp_back(Tensor pdist, Tensor dx, Tensor tmp, int N, int xDim) {
    	/**
         * 设置入参
	    float* x,
	    float* dx,
	    float tmp,
	    int N,
	    int x_dim,
	    int W
         */
        kernelParameters = Pointer.to(Pointer.to(pdist.getGpuData()), Pointer.to(dx.getGpuData()),Pointer.to(tmp.getGpuData()),
        		Pointer.to(new int[]{N}),Pointer.to(new int[]{xDim}), Pointer.to(new int[]{pdist.dataLength}));
        cuLaunchKernel(log_exp_back_function, this.CAFFE_GET_BLOCKS(pdist.dataLength), 1, 1,      // Grid dimension
        		CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                0, null,               // Shared memory size and stream
                kernelParameters, null // Kernel- and extra parameters
        );
    }
    
    public void p_dist(Tensor x,Tensor output, float p) {
        try {
            /**
             * 设置入参
             *  const float *x, float *y, const float p, const int64_t n, const int64_t m, const float n1,const float n2
             */
        	int n = x.number;
        	int m = x.getOnceSize();
        	float n1 = n - 0.5f;
        	float n2 = n1 * n1 - 1;

        	int threads = 256;

            kernelParameters = Pointer.to(Pointer.to(x.getGpuData()), Pointer.to(output.getGpuData()), Pointer.to(new float[]{p}),
            		Pointer.to(new int[]{n}), Pointer.to(new int[]{m}), Pointer.to(new float[]{n1}), Pointer.to(new float[]{n2}));
            cuLaunchKernel(p_dist_function, output.dataLength, 1, 1,      // Grid dimension
            		threads, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            );
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
    
    public void p_dist_back(Tensor x, Tensor y, Tensor dy, Tensor buffer, Tensor dx, float p) {
        try {
        	int n = x.number;
        	int m = x.getOnceSize();
//        	System.err.println(n);
//        	System.err.println(m);
        	float n1 = n - 0.5f;
        	float n2 = n1 * n1 - 1;
        	int max_block = getCudaManager().props.multiProcessorCount;
//        	System.err.println(max_block);
        	int block_x = Math.min(8, max_block);
        	int block_y = Math.min(128, max_block);
//           	System.err.println(block_x);
//        	System.err.println(block_y);
        	int grid_x = (dy.dataLength + block_x - 1) / block_x;
        	int grid_y = (m + block_y * 8 - 1) / (block_y * 8);
//        	System.err.println("grid_x:"+grid_x);
//        	System.err.println("grid_y:"+grid_y);
        	
//        	if(buffer == null) {
//        		buffer = Tensor.createGPUTensor(buffer, n - 1, n, 1, m, true);
//        	}
        	
            /**
             * 设置入参
             *  const size_t y_size, const float *y_grad, const float *x, const float *y, float *buffer, const int64_t n,
             *  const int64_t m, const float p, const float n1, const float n2
             */
        	Pointer kernelParameters = Pointer.to(Pointer.to(new int[]{dy.dataLength}), Pointer.to(dy.getGpuData()), Pointer.to(x.getGpuData()), Pointer.to(y.getGpuData()), Pointer.to(buffer.getGpuData()),
            		Pointer.to(new int[]{n}), Pointer.to(new int[]{m}), Pointer.to(new float[]{p}),
            		Pointer.to(new float[]{n1}), Pointer.to(new float[]{n2}));
            checkCUDA(cuLaunchKernel(p_dist_back_function, grid_x, grid_y, 1,      // Grid dimension
            		block_x, block_y, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            ));
//            buffer.showDMByOffsetRed(0, 100, "buffer");
            /**
             * float *x_grad, float *buffer, const int64_t n, const size_t size
             */
            int block = Math.min((dx.dataLength - 1) / getCudaManager().props.maxThreadsPerBlock + 1, max_block);

            Pointer kernelParameters2 = Pointer.to(Pointer.to(dx.getGpuData()), Pointer.to(buffer.getGpuData()),
            		Pointer.to(new int[]{n}), Pointer.to(new int[]{dx.dataLength}));
            checkCUDA(cuLaunchKernel(p_dist_back_add_function, block, 1, 1,      // Grid dimension
            		getCudaManager().props.maxThreadsPerBlock, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameters2, null // Kernel- and extra parameters
            ));
            
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
    
    public void p_dist_back2(Tensor x, Tensor y, Tensor dy, Tensor buffer, Tensor dx, float p) {
        try {
        	int n = dx.number;
        	int m = x.getOnceSize();
        	System.err.println(n);
        	System.err.println(m);
        	double n1 = n - 0.5f;
        	double n2 = n1 * n1 - 1;
        	int max_block = getCudaManager().props.multiProcessorCount;

        	int block_x = 16;
        	int block_y = 64;

        	int grid_x = (dy.dataLength + block_x - 1) / block_x;
        	int grid_y = (m + block_y * 8 - 1) / (block_y * 8);
        	System.err.println("grid_x:"+grid_x);
        	System.err.println("grid_y:"+grid_y);
        	System.out.println(n1+":"+n2);
            /**
             * 设置入参
             *  float * buffer, const float * grad, const float * self, const float * dist, const int64_t n, const int64_t m, const int64_t combs, const float p,
                const float n2, const float n2_squared_minus_1
             */
        	Pointer kernelParameters = Pointer.to(Pointer.to(buffer.getGpuData()), Pointer.to(dy.getGpuData()), Pointer.to(x.getGpuData()), Pointer.to(y.getGpuData()),
            		Pointer.to(new int[]{n}), Pointer.to(new int[]{m}), Pointer.to(new int[]{dy.dataLength}), Pointer.to(new float[]{p}),
            		Pointer.to(new double[]{n1}), Pointer.to(new double[]{n2}));
            checkCUDA(cuLaunchKernel(p_dist_back_function2, grid_x, grid_y, 1,      // Grid dimension
            		block_x, block_y, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            ));
            buffer.showDM("buffer");
            /**
             * float *x_grad, float *buffer, const int64_t n, const size_t size
             */
            int block = Math.min((dx.dataLength - 1) / getCudaManager().props.maxThreadsPerBlock + 1, max_block);

            Pointer kernelParameters2 = Pointer.to(Pointer.to(dx.getGpuData()), Pointer.to(buffer.getGpuData()),
            		Pointer.to(new int[]{n}), Pointer.to(new int[]{dx.dataLength}));
            checkCUDA(cuLaunchKernel(p_dist_back_add_function, block, 1, 1,      // Grid dimension
            		getCudaManager().props.maxThreadsPerBlock, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameters2, null // Kernel- and extra parameters
            ));
            
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

