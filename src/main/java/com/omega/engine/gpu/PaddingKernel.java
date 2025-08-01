package com.omega.engine.gpu;

import com.omega.common.utils.JsonUtils;
import com.omega.common.utils.MatrixUtils;
import com.omega.common.utils.PrintUtils;
import com.omega.engine.tensor.Tensor;

import jcuda.Pointer;
import jcuda.driver.CUfunction;
import jcuda.runtime.cudaError;

import static jcuda.driver.JCudaDriver.cuLaunchKernel;

public class PaddingKernel extends CUDAKernel {
    private CUfunction function;
    private CUfunction gradFunction;
    private CUfunction function2d;
    private CUfunction gradFunction2d;
    private CUfunction function_self;
    private CUfunction gradFunction_self;
    private int CAFFE_CUDA_NUM_THREADS = 1024;
    private Pointer kernelParameters;

    public PaddingKernel(CUDAManager cudaManager) {
        super(cudaManager);
        init();
    }

    public static void main(String args[]) {
        int N = 1;
        int C = 1;
        int D = 2;
        int H = 3;
        int W = 3;
        float[] x = MatrixUtils.order(N * C * D * H * W, 1, 1);
        Tensor input = new Tensor(N, C * D, H, W, x, true);
        CUDAManager cudaManager = new CUDAManager(0);
        int[] padding = new int[]{1, 2, 0, 1, 2, 1};
        PaddingKernel pad = new PaddingKernel(cudaManager);
        Tensor output = pad.createOutput(input, D, padding);
        pad.padding3d(input, output, D, padding, 0.1f);
        input.showDM();
        output.showDM();
        PrintUtils.printImage(output);
        output.setData(MatrixUtils.order(output.dataLength, 1, 1));
        output.hostToDevice();
        output.showDM();
        pad.padding3dGrad(output, input, D, padding);
        input.showDM();
        PrintUtils.printImage(input);
        
        int N2 = 1;
        int C2 = 2;
        int H2 = 3;
        int W2 = 3;
        float[] x2 = MatrixUtils.order(N2 * C2 * H2 * W2, 1, 1);
        Tensor input2 = new Tensor(N2, C2, H2, W2, x2, true);

        int[] padding2 = new int[]{0, 1, 0, 1};

        Tensor output2 = pad.createOutput2D(input2, padding2);
        pad.padding2d(input2, output2, padding2, 0.1f);
        input2.showDM();
        output2.showDM();
        PrintUtils.printImage(output2);
        output2.setData(MatrixUtils.order(output2.dataLength, 1, 1));
        output2.hostToDevice();
        output2.showDM();
        pad.padding2dGrad(output2, input2, padding2);
        input2.showDM();
        PrintUtils.printImage(input2);
    }

    public void initFunction() {
        try {
            if (function == null) {
                function = getCudaManager().getLocalFunctionByModule("PaddingKernel.cu", "constPadding3d");
            }
            if (gradFunction == null) {
                gradFunction = getCudaManager().getLocalFunctionByModule("PaddingKernel.cu", "ConstantPadGrad3d");
            }
            if (function2d == null) {
                function2d = getCudaManager().getLocalFunctionByModule("PaddingKernel.cu", "constPadding2d");
            }
            if (gradFunction2d == null) {
                gradFunction2d = getCudaManager().getLocalFunctionByModule("PaddingKernel.cu", "ConstantPadGrad2d");
            }
            if (function_self == null) {
            	function_self = getCudaManager().getLocalFunctionByModule("PaddingKernel.cu", "constPadding3d_seft");
            }
            if (gradFunction_self == null) {
            	gradFunction_self = getCudaManager().getLocalFunctionByModule("PaddingKernel.cu", "ConstantPadGrad3d_self");
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

    public Tensor createOutput(Tensor x, int xd, int[] padding) {
        if (padding.length != 6) {
            throw new RuntimeException("padding shape must be [dHead,dBack,hTop,hBottom,wLeft,wRight].");
        }
        int C = x.channel / xd;
        int D = xd;
        int H = x.height;
        int W = x.width;
        int oDepth = D + padding[4] + padding[5];
        int oHeight = H + padding[2] + padding[3];
        int oWidth = W + padding[0] + padding[1];
        return new Tensor(x.number, C * oDepth, oHeight, oWidth, true);
    }
    
    public Tensor createOutput2D(Tensor x, int[] padding) {
        if (padding.length != 4) {
            throw new RuntimeException("padding shape must be [hTop,hBottom,wLeft,wRight].");
        }
        int C = x.channel;
        int H = x.height;
        int W = x.width;
        int oHeight = H + padding[2] + padding[3];
        int oWidth = W + padding[0] + padding[1];
        return new Tensor(x.number, C , oHeight, oWidth, true);
    }
    
    public Tensor createOutput(int number,int channel,int height,int width, int xd, int[] padding) {
        if (padding.length != 6) {
            throw new RuntimeException("padding shape must be [dHead,dBack,hTop,hBottom,wLeft,wRight].");
        }
        int C = channel / xd;
        int D = xd;
        int H = height;
        int W = width;
        int oDepth = D + padding[4] + padding[5];
        int oHeight = H + padding[2] + padding[3];
        int oWidth = W + padding[0] + padding[1];
        return new Tensor(number, C * oDepth, oHeight, oWidth, true);
    }

    /**
     * padding shape [wLeft,wRight,hTop,hBottom]
     */
    public void padding2d(Tensor x, Tensor y, int[] padding, float val) {
        try {
            if (padding.length != 4) {
                throw new RuntimeException("padding shape must be [hTop,hBottom,wLeft,wRight].");
            }
            int C = x.channel;
            int H = x.height;
            int W = x.width;
            int oHeight = H + padding[2] + padding[3];
            int oWidth = W + padding[0] + padding[1];
            if ( y.height != oHeight || y.width != oWidth) {
                throw new RuntimeException("the output tensor shape is not same as padded shape.");
            }
            /**
             * 设置入参
             *const size_t size, const float *input, const int64_t num, const int64_t channels,
			  const int64_t old_height, const int64_t old_width,
              const int64_t old_hw,
              const int64_t padded_height, const int64_t padded_width,
              const int64_t padded_hw, const int64_t pad_top,
              const int64_t pad_left, const float pad_value, float *output
             */
            kernelParameters = Pointer.to(Pointer.to(new long[]{y.dataLength}), Pointer.to(x.getGpuData()), Pointer.to(new int[]{x.number}), Pointer.to(new int[]{C}), Pointer.to(new int[]{H}), Pointer.to(new int[]{W}), Pointer.to(new int[]{H * W}), Pointer.to(new int[]{oHeight}), Pointer.to(new int[]{oWidth}), Pointer.to(new int[]{oHeight * oWidth}), Pointer.to(new int[]{padding[2]}), Pointer.to(new int[]{padding[0]}), Pointer.to(new float[]{val}), Pointer.to(y.getGpuData()));
            cuLaunchKernel(function2d, this.CAFFE_GET_BLOCKS(y.dataLength), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            );
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
    
    /**
     * padding shape [wLeft,wRight,hTop,hBottom,dHead,dBack]
     */
    public void padding2dGrad(Tensor dy, Tensor dx, int[] padding) {
        try {
            if (padding.length != 4) {
                throw new RuntimeException("padding shape must be [hTop,hBottom,wLeft,wRight].");
            }
            int C = dx.channel;
            int H = dx.height;
            int W = dx.width;
            int oHeight = H + padding[2] + padding[3];
            int oWidth = W + padding[0] + padding[1];
            /**
             * 设置入参
             *const size_t size, const float *dy, const int64_t num, const int64_t channels,
              const int64_t old_height, const int64_t old_width,
              const int64_t old_hw,
              const int64_t padded_height, const int64_t padded_width,
              const int64_t padded_hw, const int64_t pad_top,
              const int64_t pad_left, float *dx
             */
            kernelParameters = Pointer.to(Pointer.to(new long[]{dx.dataLength}), Pointer.to(dy.getGpuData()), Pointer.to(new int[]{dy.number}), Pointer.to(new int[]{C}), Pointer.to(new int[]{H}), Pointer.to(new int[]{W}), Pointer.to(new int[]{H * W}), Pointer.to(new int[]{oHeight}), Pointer.to(new int[]{oWidth}), Pointer.to(new int[]{oHeight * oWidth}), Pointer.to(new int[]{padding[2]}), Pointer.to(new int[]{padding[0]}), Pointer.to(dx.getGpuData()));
            cuLaunchKernel(gradFunction2d, this.CAFFE_GET_BLOCKS(dx.dataLength), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            );
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
    
    /**
     * padding shape [wLeft,wRight,hTop,hBottom,dHead,dBack]
     */
    public void padding3d(Tensor x, Tensor y, int xd, int[] padding, float val) {
        try {
            if (padding.length != 6) {
                throw new RuntimeException("padding shape must be [dHead,dBack,hTop,hBottom,wLeft,wRight].");
            }
            int C = x.channel / xd;
            int D = xd;
            int H = x.height;
            int W = x.width;
            int oDepth = D + padding[4] + padding[5];
            int oHeight = H + padding[2] + padding[3];
            int oWidth = W + padding[0] + padding[1];
            if (y.channel != oDepth * C || y.height != oHeight || y.width != oWidth) {
                throw new RuntimeException("the output tensor shape is not same as padded shape."+JsonUtils.toJson(y.shape())+"->["+C+","+oDepth+":"+oHeight+":"+oWidth+"]");
            }
            /**
             * 设置入参
             *const long size, const float *input, const int64_t num, const int64_t channels,

             const int64_t old_depth, const int64_t old_height, const int64_t old_width,

             const int64_t old_dhw, const int64_t old_hw, const int64_t padded_depth,

             const int64_t padded_height, const int64_t padded_width, const int64_t padded_dhw,

             const int64_t padded_hw, const int64_t pad_head, const int64_t pad_top,

             const int64_t pad_left, const float *pad_value, float *output

             */
            kernelParameters = Pointer.to(Pointer.to(new long[]{y.dataLength}), Pointer.to(x.getGpuData()), Pointer.to(new int[]{x.number}), Pointer.to(new int[]{C}), Pointer.to(new int[]{D}), Pointer.to(new int[]{H}), Pointer.to(new int[]{W}), Pointer.to(new int[]{D * H * W}), Pointer.to(new int[]{H * W}), Pointer.to(new int[]{oDepth}), Pointer.to(new int[]{oHeight}), Pointer.to(new int[]{oWidth}), Pointer.to(new int[]{oDepth * oHeight * oWidth}), Pointer.to(new int[]{oHeight * oWidth}), Pointer.to(new int[]{padding[4]}), Pointer.to(new int[]{padding[2]}), Pointer.to(new int[]{padding[0]}), Pointer.to(new float[]{val}), Pointer.to(y.getGpuData()));
            cuLaunchKernel(function, this.CAFFE_GET_BLOCKS(y.dataLength), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            );
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
    
    /**
     * padding shape [wLeft,wRight,hTop,hBottom,dHead,dBack]
     */
    public void padding3d_self(Tensor x, Tensor y, int xd, int[] padding) {
        try {
            if (padding.length != 6) {
                throw new RuntimeException("padding shape must be [dHead,dBack,hTop,hBottom,wLeft,wRight].");
            }
            int C = x.channel / xd;
            int D = xd;
            int H = x.height;
            int W = x.width;
            int oDepth = D + padding[4] + padding[5];
            int oHeight = H + padding[2] + padding[3];
            int oWidth = W + padding[0] + padding[1];
            if (y.channel != oDepth * C || y.height != oHeight || y.width != oWidth) {
                throw new RuntimeException("the output tensor shape is not same as padded shape.");
            }
            /**
             * 设置入参
             *const long size, const float *input, const int64_t num, const int64_t channels,

             const int64_t old_depth, const int64_t old_height, const int64_t old_width,

             const int64_t old_dhw, const int64_t old_hw, const int64_t padded_depth,

             const int64_t padded_height, const int64_t padded_width, const int64_t padded_dhw,

             const int64_t padded_hw, const int64_t pad_head, const int64_t pad_top,

             const int64_t pad_left, const float *pad_value, float *output

             */
            kernelParameters = Pointer.to(Pointer.to(new long[]{y.dataLength}), Pointer.to(x.getGpuData()), Pointer.to(new int[]{x.number}), Pointer.to(new int[]{C}), Pointer.to(new int[]{D}), Pointer.to(new int[]{H}), Pointer.to(new int[]{W}), Pointer.to(new int[]{D * H * W}), Pointer.to(new int[]{H * W}), Pointer.to(new int[]{oDepth}), Pointer.to(new int[]{oHeight}), Pointer.to(new int[]{oWidth}), Pointer.to(new int[]{oDepth * oHeight * oWidth}), Pointer.to(new int[]{oHeight * oWidth}), Pointer.to(new int[]{padding[4]}), Pointer.to(new int[]{padding[2]}), Pointer.to(new int[]{padding[0]}), Pointer.to(y.getGpuData()));
            cuLaunchKernel(function_self, this.CAFFE_GET_BLOCKS(y.dataLength), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            );
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
    
    /**
     * padding shape [wLeft,wRight,hTop,hBottom,dHead,dBack]
     */
    public void padding3dGrad(Tensor dy, Tensor dx, int xd, int[] padding) {
        try {
            if (padding.length != 6) {
                throw new RuntimeException("padding shape must be [dHead,dBack,hTop,hBottom,wLeft,wRight].");
            }
            int C = dx.channel / xd;
            int D = xd;
            int H = dx.height;
            int W = dx.width;
            int oDepth = D + padding[4] + padding[5];
            int oHeight = H + padding[2] + padding[3];
            int oWidth = W + padding[0] + padding[1];
            /**
             * 设置入参
             *const size_t size, const float *dy, const int64_t num, const int64_t channels,

             const int64_t old_depth, const int64_t old_height, const int64_t old_width,

             const int64_t old_dhw, const int64_t old_hw, const int64_t padded_depth,

             const int64_t padded_height, const int64_t padded_width, const int64_t padded_dhw,

             const int64_t padded_hw, const int64_t pad_head, const int64_t pad_top,

             const int64_t pad_left, float *dx

             */
            kernelParameters = Pointer.to(Pointer.to(new long[]{dx.dataLength}), Pointer.to(dy.getGpuData()), Pointer.to(new int[]{dy.number}), Pointer.to(new int[]{C}), Pointer.to(new int[]{D}), Pointer.to(new int[]{H}), Pointer.to(new int[]{W}), Pointer.to(new int[]{D * H * W}), Pointer.to(new int[]{H * W}), Pointer.to(new int[]{oDepth}), Pointer.to(new int[]{oHeight}), Pointer.to(new int[]{oWidth}), Pointer.to(new int[]{oDepth * oHeight * oWidth}), Pointer.to(new int[]{oHeight * oWidth}), Pointer.to(new int[]{padding[4]}), Pointer.to(new int[]{padding[2]}), Pointer.to(new int[]{padding[0]}), Pointer.to(dx.getGpuData()));
            cuLaunchKernel(gradFunction, this.CAFFE_GET_BLOCKS(dx.dataLength), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            );
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
    
    /**
     * padding shape [wLeft,wRight,hTop,hBottom,dHead,dBack]
     */
    public void padding3dGrad_self(Tensor dy, Tensor dx, int xd, int[] padding) {
        try {
            if (padding.length != 6) {
                throw new RuntimeException("padding shape must be [dHead,dBack,hTop,hBottom,wLeft,wRight].");
            }
            int C = dx.channel / xd;
            int D = xd;
            int H = dx.height;
            int W = dx.width;
            int oDepth = D + padding[4] + padding[5];
            int oHeight = H + padding[2] + padding[3];
            int oWidth = W + padding[0] + padding[1];
            /**
             * 设置入参
             *const size_t size, const float *dy, const int64_t num, const int64_t channels,
             const int64_t old_depth, const int64_t old_height, const int64_t old_width,
             const int64_t old_dhw, const int64_t old_hw, const int64_t padded_depth,
             const int64_t padded_height, const int64_t padded_width, const int64_t padded_dhw,
             const int64_t padded_hw, const int64_t pad_head, const int64_t pad_top,
             const int64_t pad_left, float *dx
             */
            kernelParameters = Pointer.to(Pointer.to(new long[]{dx.dataLength}), Pointer.to(dy.getGpuData()), Pointer.to(new int[]{dy.number}), Pointer.to(new int[]{C}), Pointer.to(new int[]{D}), Pointer.to(new int[]{H}), Pointer.to(new int[]{W}), Pointer.to(new int[]{D * H * W}), Pointer.to(new int[]{H * W}), Pointer.to(new int[]{oDepth}), Pointer.to(new int[]{oHeight}), Pointer.to(new int[]{oWidth}), Pointer.to(new int[]{oDepth * oHeight * oWidth}), Pointer.to(new int[]{oHeight * oWidth}), Pointer.to(new int[]{padding[4]}), Pointer.to(new int[]{padding[2]}), Pointer.to(new int[]{padding[0]}), Pointer.to(dx.getGpuData()));
            cuLaunchKernel(gradFunction, this.CAFFE_GET_BLOCKS(dx.dataLength), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            );
            
            kernelParameters = Pointer.to(Pointer.to(new long[]{dy.dataLength}), Pointer.to(dy.getGpuData()), Pointer.to(new int[]{dy.number}), Pointer.to(new int[]{C}), Pointer.to(new int[]{D}), Pointer.to(new int[]{H}), Pointer.to(new int[]{W}), Pointer.to(new int[]{D * H * W}), Pointer.to(new int[]{H * W}), Pointer.to(new int[]{oDepth}), Pointer.to(new int[]{oHeight}), Pointer.to(new int[]{oWidth}), Pointer.to(new int[]{oDepth * oHeight * oWidth}), Pointer.to(new int[]{oHeight * oWidth}), Pointer.to(new int[]{padding[4]}), Pointer.to(new int[]{padding[2]}), Pointer.to(new int[]{padding[0]}), Pointer.to(dx.getGpuData()));
            cuLaunchKernel(gradFunction_self, this.CAFFE_GET_BLOCKS(dy.dataLength), 1, 1,      // Grid dimension
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

