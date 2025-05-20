package com.omega.engine.nn.layer.dc_ae.gpu;

import static jcuda.driver.JCudaDriver.cuLaunchKernel;

import com.omega.common.utils.JsonUtils;
import com.omega.common.utils.RandomUtils;
import com.omega.engine.ad.op.TensorOP;
import com.omega.engine.gpu.BaseKernel;
import com.omega.engine.gpu.CUDAManager;
import com.omega.engine.gpu.CUDAMemoryManager;
import com.omega.engine.tensor.Tensor;

import jcuda.Pointer;
import jcuda.driver.CUfunction;

public class DCAEKernel extends BaseKernel {
	
    private CUfunction function;
    private CUfunction function_back;
    private CUfunction channel_duplicate_function;
    private CUfunction channel_duplicate_back;
    private int CAFFE_CUDA_NUM_THREADS = 1024;
    private Pointer forwardKernelParameters;
    private Pointer backwardKernelParameters;

    public DCAEKernel(CUDAManager cudaManager) {
        super(cudaManager);
        init();
    }

    public static void main(String args[]) {
//        int N = 2;
//        int C = 8;
//        int H = 4;
//        int W = 4;
//        int OC = C*2;
//        float[] xd = RandomUtils.order(N * C * H * W, 0.1f, 0.1f);
//        float[] dyd = RandomUtils.order(N * OC * H/2 * W/2, 0.1f, 0.1f);
//        Tensor input = new Tensor(N, C, H, W, xd, true);
//        Tensor output = new Tensor(N, OC, H/2, W/2, true);
//        Tensor delta = new Tensor(N, OC, H/2, W/2, dyd, true);
//        Tensor diff = new Tensor(N, C, H, W, true);
//
//		int[] x_shape = new int[] {N, C, H/2, 2, W/2, 2};
//		int[] o_shape = new int[] {N, C, 2, 2, H/2, W/2};
		
		CUDAManager cudaManager = new CUDAManager(0);
		
//		TensorOP tensorOP = new TensorOP(cudaManager);
//		
//		Tensor sc_x = new Tensor(N, C * 4, H/2, W/2, true);
//		
//		tensorOP.permute(input, sc_x, x_shape, o_shape, new int[] {0, 1, 3, 5, 2, 4});
//        
//		
//       
//        DCAEKernel k = new DCAEKernel(cudaManager);
//        //	    	output.showDM(new float[N * C * H * W]);
//        input.showDM();
//        k.channel_average(sc_x, output);
//        k.channel_average_backward(delta, sc_x);
//        //	    	output.showDM(new float[N * C * H * W]);
//        output.syncHost();
//        System.out.println(JsonUtils.toJson(output.getData()));
//
//        System.out.println(JsonUtils.toJson(sc_x.syncHost()));
//        
//        tensorOP.permute(sc_x, diff, o_shape, x_shape, new int[] {0, 1, 4, 2, 5, 3});
//        
//        System.out.println(JsonUtils.toJson(diff.syncHost()));
        
		DCAEKernel k = new DCAEKernel(cudaManager);
		
		int N = 2;
        int C = 2;
        int H = 4;
        int W = 4;
        int OC = C*2;
        float[] xd = RandomUtils.order(N * C * H * W, 0.1f, 0.1f);
        float[] dyd = RandomUtils.order(N * OC * H * W, 0.1f, 0.1f);
        Tensor input = new Tensor(N, C, H, W, xd, true);
        Tensor output = new Tensor(N, OC, H, W, true);
        Tensor delta = new Tensor(N, OC, H, W, dyd, true);
        Tensor diff = new Tensor(N, C, H, W, true);

		int[] x_shape = new int[] {N, C, H/2, 2, W/2, 2};
		int[] o_shape = new int[] {N, C, 2, 2, H/2, W/2};
		
		k.channel_duplicate(input, output, 2);
		System.out.println(JsonUtils.toJson(output.syncHost()));
		
		k.channel_duplicate_backward(delta, diff, 2);
		
		System.out.println(JsonUtils.toJson(diff.syncHost()));
		
        CUDAMemoryManager.free();
        
        System.out.println(Math.log(64)/Math.log(2));
        
    }

    public void init() {
        /**
         * 初始化cuda函数

         */
        initFunction();
    }

    public void initFunction() {
        try {
            if (function == null) {
                function = getCudaManager().getLocalFunctionByModule("dc_ae.cu", "channel_average");
            }
            if (function_back == null) {
                function_back = getCudaManager().getLocalFunctionByModule("dc_ae.cu", "channel_average_backward");
            }
            if (channel_duplicate_function == null) {
            	channel_duplicate_function = getCudaManager().getLocalFunctionByModule("dc_ae.cu", "channel_duplicate");
            }
            if (channel_duplicate_back == null) {
            	channel_duplicate_back = getCudaManager().getLocalFunctionByModule("dc_ae.cu", "channel_duplicate_backward");
            }
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public int CAFFE_GET_BLOCKS(int N) {
        return (N + CAFFE_CUDA_NUM_THREADS - 1) / CAFFE_CUDA_NUM_THREADS;
    }

    public void channel_average(Tensor input, Tensor output) {
        try {
            //			if(forwardKernelParameters == null || this.N != output.number) {
            /**
             * 设置入参
             * float *x, float *output, int C, int HxW, int n
             */
            forwardKernelParameters = Pointer.to(Pointer.to(input.getGpuData()), Pointer.to(output.getGpuData()), Pointer.to(new int[]{output.channel}), Pointer.to(new int[]{output.height * output.width}), Pointer.to(new int[]{output.dataLength}));

            cuLaunchKernel(function, this.CAFFE_GET_BLOCKS(output.dataLength), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    forwardKernelParameters, null // Kernel- and extra parameters
            );
            //	        JCudaDriver.cuCtxSynchronize();
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void channel_average_backward(Tensor delta, Tensor diff) {
        try {
            /**
             * 设置入参
             * float *dy, float *diff, int C, int HxW, int n
             */
            backwardKernelParameters = Pointer.to(Pointer.to(delta.getGpuData()), Pointer.to(diff.getGpuData()), Pointer.to(new int[]{delta.channel}), Pointer.to(new int[]{delta.height * delta.width}), Pointer.to(new int[]{delta.dataLength}));
            cuLaunchKernel(function_back, this.CAFFE_GET_BLOCKS(delta.dataLength), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    backwardKernelParameters, null // Kernel- and extra parameters
            );
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
    
    public void channel_duplicate(Tensor input, Tensor output,int repeats) {
        try {

            /**
             * 设置入参
             * float *x, float *output, int C, int HxW, int repeats, int n
             */
            forwardKernelParameters = Pointer.to(Pointer.to(input.getGpuData()), Pointer.to(output.getGpuData()), Pointer.to(new int[]{output.channel}), Pointer.to(new int[]{output.height * output.width}), Pointer.to(new int[]{repeats}), Pointer.to(new int[]{output.dataLength}));
            cuLaunchKernel(channel_duplicate_function, this.CAFFE_GET_BLOCKS(output.dataLength), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    forwardKernelParameters, null // Kernel- and extra parameters
            );
            //	        JCudaDriver.cuCtxSynchronize();
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void channel_duplicate_backward(Tensor delta, Tensor diff,int repeats) {
        try {
            /**
             * 设置入参
             * float *dy, float *diff, int C, int HxW, int repeats, int n
             */
            backwardKernelParameters = Pointer.to(Pointer.to(delta.getGpuData()), Pointer.to(diff.getGpuData()), Pointer.to(new int[]{diff.channel}), Pointer.to(new int[]{diff.height * diff.width}), Pointer.to(new int[]{repeats}), Pointer.to(new int[]{diff.dataLength}));
            cuLaunchKernel(channel_duplicate_back, this.CAFFE_GET_BLOCKS(diff.dataLength), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    backwardKernelParameters, null // Kernel- and extra parameters
            );
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
    
}

