package com.omega.engine.nn.layer.gpu;

import static jcuda.driver.JCudaDriver.cuLaunchKernel;

import com.omega.common.utils.MatrixUtils;
import com.omega.common.utils.PrintUtils;
import com.omega.common.utils.RandomUtils;
import com.omega.engine.gpu.BaseKernel;
import com.omega.engine.gpu.CUDAManager;
import com.omega.engine.tensor.Tensor;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUfunction;
import jcuda.runtime.cudaError;

public class UpSample3DKernel extends BaseKernel {

    private CUfunction forward_function;
    private CUfunction backward_function;
    
    private CUfunction forward_trilinear_function;
    private CUfunction backward_trilinear_function;
    
    private CUfunction forward_trilinear_offset_function;
    private CUfunction backward_trilinear_offset_function;
    
    private Pointer forwardKernelParameters;
    private Pointer backwardKernelParameters;

    public UpSample3DKernel(CUDAManager cudaManager) {
        super(cudaManager);
        init();
    }

    public static void main(String args[]) {
        try {
            int N = 2;
            int C = 3;
            int D = 3;
            int H = 2;
            int W = 2;
            int scale = 2;
            int oDepth = D * scale;
            int oHeight = H * scale;
            int oWidth = W * scale;
            float[] x = MatrixUtils.order(N * C * D * H * W, 1, 1);
            float[] d = RandomUtils.order(N * C * oDepth * oHeight * oWidth, 0.1f, 0.1f);
            Tensor input = new Tensor(N, C * D, H, W, x, true);
            Tensor output = new Tensor(N, C * oDepth, oHeight, oWidth, true);
            Tensor delta = new Tensor(N, C * oDepth, oHeight, oWidth, d, true);
            delta.showShape();
            Tensor diff = new Tensor(N, C * D, H, W, true);
            CUDAManager cudaManager = new CUDAManager(0);
            UpSample3DKernel pooling = new UpSample3DKernel(cudaManager);
            long start = System.nanoTime();
            //        	for(int i = 0;i<2;i++) {
//            pooling.forward(input, output, C, D, H, W, scale);
//            //        	}
//            System.out.println((System.nanoTime() - start) / 1e6 + "ms.");
//            input.showDM();
//            output.showDM();
//            pooling.backward(delta, diff, C, D, H, W, scale);
//            delta.showDM();
//            diff.showDM();
            
//            pooling.upsample3d_trilinear(input, output, N, C, 1, H, W, 1, scale, scale, false);
//            output.showDM();
//            output.showShape();
//            
//            pooling.upsample3d_trilinear_delta(delta, diff, N, C, D, H, W, scale, scale, scale, false);
//            diff.showDM();
//            diff.showShape();
            
            Tensor output2 = new Tensor(N, C * (oDepth - 1), oHeight, oWidth, true);
            
            pooling.upsample3d_trilinear_offset(input, output2, N, C, D, 1, H, W, (oDepth - 1), 1, scale, scale, false, 0);
            output2.showDM();
            output2.showShape();
            PrintUtils.printImage(output2);

            pooling.upsample3d_trilinear_offset(input, output2, N, C, D, D - 1, H, W, (oDepth - 1), scale, scale, scale, false, 1);
            output2.showDM();
            output2.showShape();
            PrintUtils.printImage(output2);
            
            float[] d2 = RandomUtils.order(N * C * (oDepth - 1) * oHeight * oWidth, 0.1f, 0.1f);
            Tensor delta2 = new Tensor(N, C * (oDepth - 1), oHeight, oWidth, d2, true);
            
            pooling.upsample3d_trilinear_delta_offset(delta2, diff, N, H, D, 1, H, W, (oDepth - 1), 1, scale, scale, false, 0);
            diff.showShape();
            diff.showDM();
            PrintUtils.printImage(diff);
            
//            pooling.upsample3d_trilinear_delta_offset(delta2, diff, N, H, D, D-1, H, W, (oDepth - 1), scale, scale, scale, false, 1);
//            diff.showShape();
//            diff.showDM();
//            PrintUtils.printImage(diff);
            
//            pooling.upsample3d_trilinear_offset(input, output2, N, C, D, D - 1, H, W, (oDepth - 1), scale, scale, scale, false, 1);
//            output2.showDM();
//            output2.showShape();
//            PrintUtils.printImage(output2);
            
//            float[] x2 = MatrixUtils.order(N * C * (D - 1) * H * W, 5, 1);
//            Tensor input2 = new Tensor(N, C * (D - 1), H, W, x2, true);
//            Tensor output3 = new Tensor(N, C * (oDepth - 2), oHeight, oWidth, true);
//            pooling.upsample3d_trilinear(input2, output3, N, C, D-1, H, W, scale, scale, scale, false);
//            output3.showDM();
//            output3.showShape();
//            PrintUtils.printImage(output3);
            
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void initFunction() {
        try {
            if (forward_function == null) {
                forward_function = getCudaManager().getLocalFunctionByModule("UpSampleKernel2.cu", "upscale3d");
            }
            if (backward_function == null) {
                backward_function = getCudaManager().getLocalFunctionByModule("UpSampleKernel2.cu", "downscale3d");
            }
            if(forward_trilinear_function == null) {
            	forward_trilinear_function = getCudaManager().getLocalFunctionByModule("UpSampleKernel2.cu", "UpsampleTrilinear3DKernel");
            }
            if(backward_trilinear_function == null){
            	backward_trilinear_function = getCudaManager().getLocalFunctionByModule("UpSampleKernel2.cu", "UpsampleTrilinear3DGradKernel");
            }
            if(forward_trilinear_offset_function == null) {
            	forward_trilinear_offset_function = getCudaManager().getLocalFunctionByModule("UpSampleKernel2.cu", "UpsampleTrilinear3DKernel_offset");
            }
            if(backward_trilinear_offset_function == null){
            	backward_trilinear_offset_function = getCudaManager().getLocalFunctionByModule("UpSampleKernel2.cu", "UpsampleTrilinear3DGradKernel_offset");
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

    public void forward(Tensor input, Tensor output,int channel,int depth,int height,int width, int scale) {
        upsample(input, output, channel, depth, height, width, scale);
    }

    public void backward(Tensor delta, Tensor diff,int channel,int depth,int height,int width, int scale) {
        upsampleDelta(delta, diff, channel, depth, height, width, scale);
    }

    public void upsample(Tensor input, Tensor output,int channel,int depth,int height,int width, int scale) {
        try {
        	int d1 = channel;
        	int d2 = depth * scale;
        	int d3 = height * scale;
        	int d4 = width * scale;
            this.N = input.number;

            /**
             * 设置入参
             * const float *input, float *output, int no_elements,int scale_factor, int d1, int d2, int d3, int d4
             */
            forwardKernelParameters = Pointer.to(Pointer.to(input.getGpuData()), Pointer.to(output.getGpuData()), Pointer.to(new int[]{output.dataLength}), Pointer.to(new int[]{scale}), Pointer.to(new int[]{d1}), Pointer.to(new int[]{d2}), Pointer.to(new int[]{d3}), Pointer.to(new int[]{d4}));
            int nthreads = 256;
            int n_xblocks = Math.min(Math.max((int) Math.ceil((float) output.dataLength / nthreads), 1), 65535);
            int n_yblocks = (int) Math.ceil((float) output.dataLength / (float) (n_xblocks * nthreads));
            int[] blocks = new int[]{n_xblocks, n_yblocks, 1};
            int[] threads = new int[]{nthreads, 1, 1};
            checkCUDA(cuLaunchKernel(forward_function, blocks[0], blocks[1], blocks[2],      // Grid dimension
                    threads[0], threads[1], threads[2],      // Block dimension
                    0, null,               // Shared memory size and stream
                    forwardKernelParameters, null // Kernel- and extra parameters
            ));
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void upsampleDelta(Tensor delta, Tensor diff,int channel,int depth,int height,int width, int scale) {
        try {
            diff.clearGPU();
            int d1 = channel;
        	int d2 = depth;
        	int d3 = height;
        	int d4 = width;
            this.N = delta.number;
            /**
             * 设置入参
             * float *gradInput_data, const float *gradOutput_data, int no_elements, int scale_factor, int d1, int d2, int d3, int d4
             */
            backwardKernelParameters = Pointer.to(Pointer.to(diff.getGpuData()), Pointer.to(delta.getGpuData()), Pointer.to(new int[]{diff.dataLength}), Pointer.to(new int[]{scale}), Pointer.to(new int[]{d1}), Pointer.to(new int[]{d2}), Pointer.to(new int[]{d3}), Pointer.to(new int[]{d4}));
            int nthreads = 256;
            int n_xblocks = Math.min(Math.max((int) Math.ceil((float) diff.dataLength / nthreads), 1), 65535);
            int n_yblocks = (int) Math.ceil((float) diff.dataLength / (float) (n_xblocks * nthreads));
            int[] blocks = new int[]{n_xblocks, n_yblocks, 1};
            int[] threads = new int[]{nthreads, 1, 1};
            checkCUDA(cuLaunchKernel(backward_function, blocks[0], blocks[1], blocks[2],      // Grid dimension
                    threads[0], threads[1], threads[2],      // Block dimension
                    0, null,               // Shared memory size and stream
                    backwardKernelParameters, null // Kernel- and extra parameters
            ));
            //	        System.out.println((System.nanoTime() - start1) / 1e6 + "ms1");
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
    
    public void upsample3d_trilinear(Tensor input, Tensor output,int number,int channel,int depth,int height,int width, int dScale, int hScale, int wScale, boolean align_corners) {
        try {
        	int od = depth * dScale;
        	int oh = height * hScale;
        	int ow = width * wScale;
            this.N = number;
            
            int in_dhw = depth * height * width;
            int out_hw = oh * ow;
            int out_dhw = od * oh * ow;
            int num_kernels = out_dhw;
            int blockSize = Math.min(this.getCudaManager().props.maxThreadsPerBlock, 512);
            int gridSize = (num_kernels + blockSize - 1) / blockSize;
            int ab = 0;
            if(align_corners) {
            	ab = 1;
            }
            
            float ds = areaPixelComputeScale(depth, od, align_corners, dScale);
            float hs = areaPixelComputeScale(height, oh, align_corners, hScale);
            float ws = areaPixelComputeScale(width, ow, align_corners, wScale);
 
            /**
             * 设置入参
             * const int num_kernels, const float *input, float *output, const int batch_size,
              const int channel, const int in_d, const int in_h, const int in_w,
              const int out_d, const int out_h, const int out_w, const float d_scale,
              const float h_scale, const float w_scale, const bool align_corners, const int in_dhw,
              const int out_hw, const int out_dhw
             */
            forwardKernelParameters = Pointer.to(Pointer.to(new int[]{num_kernels}), Pointer.to(input.getGpuData()), Pointer.to(output.getGpuData()),
            		Pointer.to(new int[]{N}), Pointer.to(new int[]{channel}), Pointer.to(new int[]{depth}), Pointer.to(new int[]{height}), Pointer.to(new int[]{width}),
            		Pointer.to(new int[]{od}), Pointer.to(new int[]{oh}), Pointer.to(new int[]{ow}), Pointer.to(new float[]{ds}), Pointer.to(new float[]{hs}), Pointer.to(new float[]{ws}),
            		Pointer.to(new int[]{ab}), Pointer.to(new int[]{in_dhw}), Pointer.to(new int[]{out_hw}), Pointer.to(new int[]{out_dhw}));

            checkCUDA(cuLaunchKernel(forward_trilinear_function, gridSize, 1, 1,      // Grid dimension
            		blockSize, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    forwardKernelParameters, null // Kernel- and extra parameters
            ));
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
    
    public void upsample3d_trilinear_offset(Tensor input, Tensor output,int number,int channel,int org_depth,int tar_depth,int height,int width, int oDepth, int dScale, int hScale, int wScale, boolean align_corners,int offset) {
        try {
        	int od = tar_depth * dScale;
        	int oh = height * hScale;
        	int ow = width * wScale;
            this.N = number;
            
            int in_dhw = org_depth * height * width;
            int out_hw = oh * ow;
            int out_dhw = oDepth * oh * ow;
            int num_kernels = od * oh * ow;
            int blockSize = Math.min(this.getCudaManager().props.maxThreadsPerBlock, 512);
            int gridSize = (num_kernels + blockSize - 1) / blockSize;
            int ab = 0;
            if(align_corners) {
            	ab = 1;
            }
            
            float ds = areaPixelComputeScale(tar_depth, od, align_corners, dScale);
            float hs = areaPixelComputeScale(height, oh, align_corners, hScale);
            float ws = areaPixelComputeScale(width, ow, align_corners, wScale);

            /**
             * 设置入参
             * const int num_kernels, const float *input, float *output, const int batch_size,
              const int channel, const int in_d, const int in_h, const int in_w,
              const int out_d, const int out_h, const int out_w, const float d_scale,
              const float h_scale, const float w_scale, const bool align_corners, const int in_dhw,
              const int out_hw, const int out_dhw
             */
            forwardKernelParameters = Pointer.to(Pointer.to(new int[]{num_kernels}), Pointer.to(input.getGpuData()), Pointer.to(output.getGpuData()),
            		Pointer.to(new int[]{N}), Pointer.to(new int[]{channel}), Pointer.to(new int[]{tar_depth}), Pointer.to(new int[]{height}), Pointer.to(new int[]{width}),
            		Pointer.to(new int[]{od}), Pointer.to(new int[]{oh}), Pointer.to(new int[]{ow}), Pointer.to(new float[]{ds}), Pointer.to(new float[]{hs}), Pointer.to(new float[]{ws}),
            		Pointer.to(new int[]{ab}), Pointer.to(new int[]{in_dhw}), Pointer.to(new int[]{out_hw}), Pointer.to(new int[]{out_dhw}), Pointer.to(new int[]{offset}));

            checkCUDA(cuLaunchKernel(forward_trilinear_offset_function, gridSize, 1, 1,      // Grid dimension
            		blockSize, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    forwardKernelParameters, null // Kernel- and extra parameters
            ));
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
    
    public void upsample3d_trilinear_delta(Tensor delta, Tensor diff,int number,int channel,int depth,int height,int width, int dScale, int hScale, int wScale, boolean align_corners) {
        try {
        	int od = depth * dScale;
        	int oh = height * hScale;
        	int ow = width * wScale;
            this.N = number;
            
            int dinput_dhw = depth * height * width;
            int grad_dhw = od * oh * ow;
            long dinput_size = dinput_dhw * N * channel;
            
            int blockSize = Math.min(this.getCudaManager().props.maxThreadsPerBlock, 256);
            int gridSize = (grad_dhw + blockSize - 1) / blockSize;
            
            int ab = 0;
            if(align_corners) {
            	ab = 1;
            }
            
            float ds = areaPixelComputeScale(depth, od, align_corners, dScale);
            float hs = areaPixelComputeScale(height, oh, align_corners, hScale);
            float ws = areaPixelComputeScale(width, ow, align_corners, wScale);
            
            /**
             * 设置入参
             * const size_t elem_num, const float *grad, const int batchsize,
              const int channels, const int grad_d, const int grad_h, const int grad_w,
              const int grad_dhw, const int dinput_d, const int dinput_h,
              const int dinput_w, const int dinput_dhw, const float d_scale,
              const float h_scale, const float w_scale, const bool align_corner, float *dinput
             */
            backwardKernelParameters = Pointer.to(Pointer.to(new long[]{dinput_size}), Pointer.to(delta.getGpuData()), Pointer.to(new int[]{N}), Pointer.to(new int[]{channel}),
            		Pointer.to(new int[]{od}), Pointer.to(new int[]{oh}), Pointer.to(new int[]{ow}), Pointer.to(new int[]{grad_dhw}),
            		Pointer.to(new int[]{depth}), Pointer.to(new int[]{height}), Pointer.to(new int[]{width}), Pointer.to(new int[]{dinput_dhw}),
            		Pointer.to(new float[]{ds}), Pointer.to(new float[]{hs}), Pointer.to(new float[]{ws}), Pointer.to(new int[]{ab}), Pointer.to(diff.getGpuData()));
            checkCUDA(cuLaunchKernel(backward_trilinear_function, gridSize, 1, 1,      // Grid dimension
            		blockSize, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    backwardKernelParameters, null // Kernel- and extra parameters
            ));
            //	        System.out.println((System.nanoTime() - start1) / 1e6 + "ms1");
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
    
    public void upsample3d_trilinear_delta_offset(Tensor delta, Tensor diff,int number,int channel,int org_depth,int tar_depth,int height,int width, int oDepth, int dScale, int hScale, int wScale, boolean align_corners,int offset) {
        try {
        	int od = tar_depth * dScale;
        	int oh = height * hScale;
        	int ow = width * wScale;
            this.N = number;
            
            int dinput_dhw = org_depth * height * width;
            int grad_dhw = oDepth * oh * ow;
            int tar_grad_dhw = od * oh * ow;
            long dinput_size = tar_depth * height * width * N * channel;
            
            int blockSize = Math.min(this.getCudaManager().props.maxThreadsPerBlock, 256);
            int gridSize = (tar_grad_dhw + blockSize - 1) / blockSize;
            
            int ab = 0;
            if(align_corners) {
            	ab = 1;
            }
            
            float ds = areaPixelComputeScale(tar_depth, od, align_corners, dScale);
            float hs = areaPixelComputeScale(height, oh, align_corners, hScale);
            float ws = areaPixelComputeScale(width, ow, align_corners, wScale);
            
            /**
             * 设置入参
             * const size_t elem_num, const float *grad, const int batchsize,
              const int channels, const int grad_d, const int grad_h, const int grad_w,
              const int grad_dhw, const int dinput_d, const int dinput_h,
              const int dinput_w, const int dinput_dhw, const float d_scale,
              const float h_scale, const float w_scale, const bool align_corner, float *dinput
             */
            backwardKernelParameters = Pointer.to(Pointer.to(new long[]{dinput_size}), Pointer.to(delta.getGpuData()), Pointer.to(new int[]{N}), Pointer.to(new int[]{channel}),
            		Pointer.to(new int[]{od}), Pointer.to(new int[]{oh}), Pointer.to(new int[]{ow}), Pointer.to(new int[]{grad_dhw}), Pointer.to(new int[]{tar_grad_dhw}),
            		Pointer.to(new int[]{tar_depth}), Pointer.to(new int[]{height}), Pointer.to(new int[]{width}), Pointer.to(new int[]{dinput_dhw}),
            		Pointer.to(new float[]{ds}), Pointer.to(new float[]{hs}), Pointer.to(new float[]{ws}), Pointer.to(new int[]{ab}), Pointer.to(diff.getGpuData()), Pointer.to(new int[]{offset}));
            checkCUDA(cuLaunchKernel(backward_trilinear_offset_function, gridSize, 1, 1,      // Grid dimension
            		blockSize, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    backwardKernelParameters, null // Kernel- and extra parameters
            ));
            //	        System.out.println((System.nanoTime() - start1) / 1e6 + "ms1");
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
    
    public float areaPixelComputeScale(int inputSize,int outputSize,boolean align_corners,float scale) {
    	if (align_corners) {
    	    if (outputSize > 1) {
    	      return (inputSize * 1.0f - 1.0f) / (outputSize * 1.0f - 1.0f);
    	    } else {
    	      return 0;
    	    }
    	  } else {
    	    return computeScales(scale, inputSize, outputSize);
    	  }
    }
    
    public float computeScales(float scale, int inputSize,int outputSize) {
    	if (scale > 0.) {
    	    return 1.0f / scale;
    	  } else if (outputSize > 0) {
    	    return inputSize / outputSize;
    	  }
    	  return 0;
    }
    
    public void checkCUDA(int code) {
        if (code != cudaError.cudaSuccess) {
            System.err.println("Error code " + code + ":" + cudaError.stringFor(code));
        }
    }
}

