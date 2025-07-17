package com.omega.engine.ad.op.gpu;

import static jcuda.driver.JCudaDriver.cuLaunchKernel;

import java.io.Serializable;

import com.omega.engine.gpu.BaseKernel;
import com.omega.engine.gpu.CUDAManager;
import com.omega.engine.tensor.Tensor;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUfunction;
import jcuda.driver.CUstream;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaError;
import jcuda.runtime.cudaMemcpyKind;

public class OPKernel extends BaseKernel implements Serializable {
    /**
     *
     */
    private static final long serialVersionUID = 3345793649705471080L;
//    private static OPKernel kernel = null;
    public int N = 0;
    private int CAFFE_CUDA_NUM_THREADS = 1024;
    private CUfunction fill_gpu_function;
    private CUfunction axpy_gpu_function;
    private CUfunction copy_gpu_function;
    private CUfunction copy_number_gpu_function;
    private CUfunction copy_channel_gpu_function;
    private CUfunction add_gpu_function;
    private CUfunction add_axis_function;
    private CUfunction add_axis_function2;
    private CUfunction add_axis_function3;
    private CUfunction add_axis_back_function;
    private CUfunction add_scalar_gpu_function;
    private CUfunction add_number_gpu_function;
    private CUfunction add_channel_gpu_function;
    private CUfunction sub_gpu_function;
    private CUfunction sub_axis_gpu_function;
    private CUfunction sub_scalar_gpu_function;
    private CUfunction scalar_sub_gpu_function;
    private CUfunction mul_gpu_function;
    private CUfunction mul_scalar_gpu_function;
    private CUfunction mul_plus_gpu_function;
    private CUfunction mul_plus_scalar_gpu_function;
    private CUfunction mul_plus_scalar_axis_gpu_function;
    private CUfunction div_gpu_function;
    private CUfunction div_axis_gpu_function;
    private CUfunction div_scalar_gpu_function;
    private CUfunction scalar_div_gpu_function;
    private CUfunction div_plus_gpu_function;
    private CUfunction div_plus_axis_gpu_function;
    private CUfunction div_plus_scalar_gpu_function;
    private CUfunction div_bGrad_gpu_function;
    private CUfunction div_bGrad_axis_gpu_function;
    private CUfunction div_scalar_bGrad_gpu_function;
    private CUfunction scalar_plus_div_gpu_function;
    private CUfunction pow_gpu_function;
    private CUfunction log_gpu_function;
    private CUfunction exp_gpu_function;
    private CUfunction sin_gpu_function;
    private CUfunction cos_gpu_function;
    private CUfunction tan_gpu_function;
    private CUfunction tan_back_gpu_function;
    private CUfunction atan_gpu_function;
    private CUfunction atan_back_gpu_function;
    private CUfunction sum_gpu_function;
    private CUfunction sum_channel_gpu_function;
    private CUfunction sum_height_gpu_function;
    private CUfunction sum_pow_gpu_function;
    private CUfunction sum_pow_channel_gpu_function;
    private CUfunction sum_pow_height_gpu_function;
    private CUfunction max_gpu_function;
    private CUfunction max_channel_gpu_function;
    private CUfunction max_backward_gpu_function;
    private CUfunction max_channel_backward_gpu_function;
    private CUfunction broadcast_gpu_function;
    private CUfunction broadcast_channel_gpu_function;
    private CUfunction broadcast_plus_gpu_function;
    private CUfunction broadcast_channel_plus_gpu_function;
    private CUfunction clamp_gpu_function;
    private CUfunction clamp_back_gpu_function;
    private CUfunction maximum_gpu_function;
    private CUfunction minimum_gpu_function;
    private CUfunction maximum_back_gpu_function;
    private CUfunction minimum_back_gpu_function;
    private CUfunction transpose_gpu_function;
    private CUfunction permute_gpu_function;
    private CUfunction permute_add_gpu_function;
    private CUfunction sqrt_gpu_function;
    private CUfunction bool_gpu_function;
    private CUfunction expand_function;
    private CUfunction broadcast_row_plus_gpu_function;
    private CUfunction onehot_function;
    private CUfunction mean_function;
    private CUfunction mean_back_function;
    private CUfunction mask_gpu_function;
    private CUfunction abs_function;
    private CUfunction abs_backward_function;
    private CUfunction mul_axis_function;
    private CUfunction mul_axis_back_left_function;
    private CUfunction mul_axis_back_right_function;
    private CUfunction cat_width_function;
    private CUfunction cat_width_back_function;
    private CUfunction update_ema_function;
    
    public OPKernel(CUDAManager cudaManager) {
        super(cudaManager);
        fill_gpu_function = this.getCudaManager().getLocalFunctionByModule("OPKernel.cu", "fill_kernel");
        axpy_gpu_function = this.getCudaManager().getLocalFunctionByModule("OPKernel.cu", "axpy_kernel");
        copy_gpu_function = this.getCudaManager().getLocalFunctionByModule("OPKernel.cu", "copy_kernel");
        copy_number_gpu_function = this.getCudaManager().getLocalFunctionByModule("OPKernel.cu", "copy_number_kernel");
        copy_channel_gpu_function = this.getCudaManager().getLocalFunctionByModule("OPKernel.cu", "copy_channel_kernel");
        add_gpu_function = this.getCudaManager().getLocalFunctionByModule("OPKernel.cu", "add_kernel");
        add_axis_function = this.getCudaManager().getLocalFunctionByModule("OPKernel.cu", "add_axis_kernel");
        add_axis_function2 = this.getCudaManager().getLocalFunctionByModule("OPKernel.cu", "add_axis_kernel2");
        add_axis_function3 = this.getCudaManager().getLocalFunctionByModule("OPKernel.cu", "add_axis_kernel3");
        add_axis_back_function = this.getCudaManager().getLocalFunctionByModule("OPKernel.cu", "add_axis_back_kernel");
        add_scalar_gpu_function = this.getCudaManager().getLocalFunctionByModule("OPKernel.cu", "add_scalar_kernel");
        add_number_gpu_function = this.getCudaManager().getLocalFunctionByModule("OPKernel.cu", "add_number_kernel");
        add_channel_gpu_function = this.getCudaManager().getLocalFunctionByModule("OPKernel.cu", "add_channel_kernel");
        sub_gpu_function = this.getCudaManager().getLocalFunctionByModule("OPKernel.cu", "sub_kernel");
        sub_axis_gpu_function = this.getCudaManager().getLocalFunctionByModule("OPKernel.cu", "sub_axis_kernel");
        sub_scalar_gpu_function = this.getCudaManager().getLocalFunctionByModule("OPKernel.cu", "sub_scalar_kernel");
        scalar_sub_gpu_function = this.getCudaManager().getLocalFunctionByModule("OPKernel.cu", "scalar_sub_kernel");
        mul_gpu_function = this.getCudaManager().getLocalFunctionByModule("OPKernel.cu", "mul_kernel");
        mul_scalar_gpu_function = this.getCudaManager().getLocalFunctionByModule("OPKernel.cu", "mul_scalar_kernel");
        mul_plus_gpu_function = this.getCudaManager().getLocalFunctionByModule("OPKernel.cu", "mul_plus_kernel");
        mul_plus_scalar_gpu_function = this.getCudaManager().getLocalFunctionByModule("OPKernel.cu", "mul_plus_scalar_kernel");
        mul_plus_scalar_axis_gpu_function = this.getCudaManager().getLocalFunctionByModule("OPKernel.cu", "mul_plus_scalar_axis_kernel");
        div_gpu_function = this.getCudaManager().getLocalFunctionByModule("OPKernel.cu", "div_kernel");
        div_axis_gpu_function = this.getCudaManager().getLocalFunctionByModule("OPKernel.cu", "div_axis_kernel");
        div_scalar_gpu_function = this.getCudaManager().getLocalFunctionByModule("OPKernel.cu", "div_scalar_kernel");
        scalar_div_gpu_function = this.getCudaManager().getLocalFunctionByModule("OPKernel.cu", "scalar_div_kernel");
        div_plus_gpu_function = this.getCudaManager().getLocalFunctionByModule("OPKernel.cu", "div_plus_kernel");
        div_plus_axis_gpu_function = this.getCudaManager().getLocalFunctionByModule("OPKernel.cu", "div_plus_axis_kernel");
        div_plus_scalar_gpu_function = this.getCudaManager().getLocalFunctionByModule("OPKernel.cu", "div_plus_scalar_kernel");
        scalar_plus_div_gpu_function = this.getCudaManager().getLocalFunctionByModule("OPKernel.cu", "scalar_plus_div_kernel");
        div_bGrad_gpu_function = this.getCudaManager().getLocalFunctionByModule("OPKernel.cu", "div_bGrad_kernel");
        div_bGrad_axis_gpu_function = this.getCudaManager().getLocalFunctionByModule("OPKernel.cu", "div_bGrad_axis_kernel");
        div_scalar_bGrad_gpu_function = this.getCudaManager().getLocalFunctionByModule("OPKernel.cu", "div_scalar_bGrad_kernel");
        pow_gpu_function = this.getCudaManager().getLocalFunctionByModule("OPKernel.cu", "pow_kernel");
        log_gpu_function = this.getCudaManager().getLocalFunctionByModule("OPKernel.cu", "log_kernel");
        exp_gpu_function = this.getCudaManager().getLocalFunctionByModule("OPKernel.cu", "exp_kernel");
        sin_gpu_function = this.getCudaManager().getLocalFunctionByModule("OPKernel.cu", "sin_kernel");
        cos_gpu_function = this.getCudaManager().getLocalFunctionByModule("OPKernel.cu", "cos_kernel");
        tan_gpu_function = this.getCudaManager().getLocalFunctionByModule("OPKernel.cu", "tan_kernel");
        atan_gpu_function = this.getCudaManager().getLocalFunctionByModule("OPKernel.cu", "atan_kernel");
        tan_back_gpu_function = this.getCudaManager().getLocalFunctionByModule("OPKernel.cu", "tan_back_kernel");
        atan_back_gpu_function = this.getCudaManager().getLocalFunctionByModule("OPKernel.cu", "atan_back_kernel");
        sum_gpu_function = this.getCudaManager().getLocalFunctionByModule("OPKernel.cu", "sum_kernel");
        sum_channel_gpu_function = this.getCudaManager().getLocalFunctionByModule("OPKernel.cu", "sum_channel_kernel");
        sum_height_gpu_function = this.getCudaManager().getLocalFunctionByModule("OPKernel.cu", "sum_height_kernel");
        sum_pow_gpu_function = this.getCudaManager().getLocalFunctionByModule("OPKernel.cu", "sum_pow_kernel");
        sum_pow_channel_gpu_function = this.getCudaManager().getLocalFunctionByModule("OPKernel.cu", "sum_pow_channel_kernel");
        sum_pow_height_gpu_function = this.getCudaManager().getLocalFunctionByModule("OPKernel.cu", "sum_pow_height_kernel");
        max_gpu_function = this.getCudaManager().getLocalFunctionByModule("OPKernel.cu", "max_kernel");
        max_channel_gpu_function = this.getCudaManager().getLocalFunctionByModule("OPKernel.cu", "max_channel_kernel");
        max_backward_gpu_function = this.getCudaManager().getLocalFunctionByModule("OPKernel.cu", "max_backward_kernel");
        max_channel_backward_gpu_function = this.getCudaManager().getLocalFunctionByModule("OPKernel.cu", "max_channel_backward_kernel");
        broadcast_gpu_function = this.getCudaManager().getLocalFunctionByModule("OPKernel.cu", "broadcast_kernel");
        broadcast_channel_gpu_function = this.getCudaManager().getLocalFunctionByModule("OPKernel.cu", "broadcast_number_kernel");
        broadcast_plus_gpu_function = this.getCudaManager().getLocalFunctionByModule("OPKernel.cu", "broadcast_plus_kernel");
        broadcast_channel_plus_gpu_function = this.getCudaManager().getLocalFunctionByModule("OPKernel.cu", "broadcast_number_plus_kernel");
        broadcast_row_plus_gpu_function = this.getCudaManager().getLocalFunctionByModule("OPKernel.cu", "broadcast_row_plus_kernel");
        clamp_gpu_function = this.getCudaManager().getLocalFunctionByModule("OPKernel.cu", "clamp_kernel");
        clamp_back_gpu_function = this.getCudaManager().getLocalFunctionByModule("OPKernel.cu", "clamp_back_kernel");
        maximum_gpu_function = this.getCudaManager().getLocalFunctionByModule("OPKernel.cu", "maximum_kernel");
        minimum_gpu_function = this.getCudaManager().getLocalFunctionByModule("OPKernel.cu", "minimum_kernel");
        maximum_back_gpu_function = this.getCudaManager().getLocalFunctionByModule("OPKernel.cu", "maximum_back_kernel");
        minimum_back_gpu_function = this.getCudaManager().getLocalFunctionByModule("OPKernel.cu", "minimum_back_kernel");
        transpose_gpu_function = this.getCudaManager().getLocalFunctionByModule("OPKernel.cu", "transpose_kernel");
        permute_gpu_function = this.getCudaManager().getLocalFunctionByModule("OPKernel.cu", "permute_kernel");
        sqrt_gpu_function = this.getCudaManager().getLocalFunctionByModule("OPKernel.cu", "sqrt_kernel");
        bool_gpu_function = this.getCudaManager().getLocalFunctionByModule("OPKernel.cu", "bool_kernel");
        expand_function = this.getCudaManager().getLocalFunctionByModule("OPKernel.cu", "expand_kernel");
        permute_add_gpu_function = this.getCudaManager().getLocalFunctionByModule("OPKernel.cu", "permute_add_kernel");
        onehot_function = this.getCudaManager().getLocalFunctionByModule("OPKernel.cu", "one_hot_kernel");
        mean_function = this.getCudaManager().getLocalFunctionByModule("OPKernel.cu", "mean_kernel");
        mean_back_function = this.getCudaManager().getLocalFunctionByModule("OPKernel.cu", "mean_back_kernel");
        mask_gpu_function = this.getCudaManager().getLocalFunctionByModule("OPKernel.cu", "mask_kernel");
        abs_function = this.getCudaManager().getLocalFunctionByModule("OPKernel.cu", "abs_kernel");
        abs_backward_function = this.getCudaManager().getLocalFunctionByModule("OPKernel.cu", "abs_backward_kernel");
        mul_axis_function = this.getCudaManager().getLocalFunctionByModule("OPKernel.cu", "mul_axis_kernel");
        mul_axis_back_left_function = this.getCudaManager().getLocalFunctionByModule("OPKernel.cu", "mul_axis_back_left_kernel");
        mul_axis_back_right_function = this.getCudaManager().getLocalFunctionByModule("OPKernel.cu", "mul_axis_back_right_kernel");
        cat_width_function = this.getCudaManager().getLocalFunctionByModule("OPKernel.cu", "cat_width_kernel");
        cat_width_back_function = this.getCudaManager().getLocalFunctionByModule("OPKernel.cu", "cat_width_back_kernel");
        update_ema_function = this.getCudaManager().getLocalFunctionByModule("OPKernel.cu", "update_ema");
    }

    public void fill_gpu(Tensor x, float val) {
        try {
            /**
             * int N, float ALPHA, float *X

             */
            Pointer kernelParameter = Pointer.to(Pointer.to(new int[]{x.getDataLength()}), Pointer.to(new float[]{val}), Pointer.to(x.getGpuData()));
            checkCUDA(cuLaunchKernel(fill_gpu_function, CAFFE_GET_BLOCKS(x.getDataLength()), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameter, null // Kernel- and extra parameters
            ));
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void copy_gpu(Tensor a, Tensor b, int offset) {
        try {
            /**
             * int N,  float *X, int OFFX, float *Y, int OFFY

             */
            Pointer kernelParameter = Pointer.to(Pointer.to(new int[]{b.getDataLength()}), Pointer.to(a.getGpuData()), Pointer.to(new int[]{offset}), Pointer.to(b.getGpuData()), Pointer.to(new int[]{0}));
            checkCUDA(cuLaunchKernel(copy_gpu_function, CAFFE_GET_BLOCKS(b.getDataLength()), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameter, null // Kernel- and extra parameters
            ));
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void copy_gpu(Tensor a, Tensor b, int offsetX, int offsetY) {
        try {
            /**
             * int N,  float *X, int OFFX, float *Y, int OFFY

             */
            Pointer kernelParameter = Pointer.to(Pointer.to(new int[]{a.getDataLength()}), Pointer.to(a.getGpuData()), Pointer.to(new int[]{offsetX}), Pointer.to(b.getGpuData()), Pointer.to(new int[]{offsetY}));
            checkCUDA(cuLaunchKernel(copy_gpu_function, CAFFE_GET_BLOCKS(b.getDataLength()), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameter, null // Kernel- and extra parameters
            ));
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void copy_number_gpu(Tensor a, Tensor b, int start, int cpy) {
        try {
            /**
             * int N,  float *X, float *Y, int n,int c,int h,int w,int start

             */
            Pointer kernelParameter = Pointer.to(Pointer.to(new int[]{b.getDataLength()}), Pointer.to(a.getGpuData()), Pointer.to(b.getGpuData()), Pointer.to(new int[]{a.number}), Pointer.to(new int[]{a.channel}), Pointer.to(new int[]{a.height}), Pointer.to(new int[]{a.width}), Pointer.to(new int[]{start}), Pointer.to(new int[]{cpy}));
            checkCUDA(cuLaunchKernel(copy_number_gpu_function, CAFFE_GET_BLOCKS(b.getDataLength()), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameter, null // Kernel- and extra parameters
            ));
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void copy_channel_gpu(Tensor a, Tensor b, int start, int cpy) {
        try {
            /**
             * int N,  float *X, float *Y, int n,int c,int h,int w,int start
             */
            Pointer kernelParameter = Pointer.to(Pointer.to(new int[]{b.getDataLength()}), Pointer.to(a.getGpuData()), Pointer.to(b.getGpuData()), Pointer.to(new int[]{a.number}), Pointer.to(new int[]{a.channel}), Pointer.to(new int[]{a.height}), Pointer.to(new int[]{a.width}), Pointer.to(new int[]{start}), Pointer.to(new int[]{cpy}));
            checkCUDA(cuLaunchKernel(copy_channel_gpu_function, CAFFE_GET_BLOCKS(b.getDataLength()), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameter, null // Kernel- and extra parameters
            ));
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
    
    public void copy_channel_gpu(Tensor a, Tensor b, int[] shape, int start, int cpy) {
        try {
            /**
             * int N,  float *X, float *Y, int n,int c,int h,int w,int start
             */
            Pointer kernelParameter = Pointer.to(Pointer.to(new int[]{b.getDataLength()}), Pointer.to(a.getGpuData()), Pointer.to(b.getGpuData()), Pointer.to(new int[]{shape[0]}), Pointer.to(new int[]{shape[1]}), Pointer.to(new int[]{shape[2]}), Pointer.to(new int[]{shape[3]}), Pointer.to(new int[]{start}), Pointer.to(new int[]{cpy}));
            checkCUDA(cuLaunchKernel(copy_channel_gpu_function, CAFFE_GET_BLOCKS(b.getDataLength()), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameter, null // Kernel- and extra parameters
            ));
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
    
    public void add_gpu(Tensor a, Tensor b, Tensor y) {
        try {
            /**
             * int N, float *X, float *Y, float *R

             */
            Pointer kernelParameter = Pointer.to(Pointer.to(new int[]{y.getDataLength()}), Pointer.to(a.getGpuData()), Pointer.to(b.getGpuData()), Pointer.to(y.getGpuData()));
            checkCUDA(cuLaunchKernel(add_gpu_function, CAFFE_GET_BLOCKS(y.getDataLength()), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameter, null // Kernel- and extra parameters
            ));
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void add_gpu(Tensor a, Tensor b, Tensor y, CUstream stream) {
        try {
            /**
             * int N, float *X, float *Y, float *R

             */
            Pointer kernelParameter = Pointer.to(Pointer.to(new int[]{y.getDataLength()}), Pointer.to(a.getGpuData()), Pointer.to(b.getGpuData()), Pointer.to(y.getGpuData()));
            checkCUDA(cuLaunchKernel(add_gpu_function, CAFFE_GET_BLOCKS(y.getDataLength()), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, stream,               // Shared memory size and stream
                    kernelParameter, null // Kernel- and extra parameters
            ));
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void add_gpu(Tensor a, Tensor b, Tensor y, int axis) {
        try {
            /**
             * int N, float *X, float *Y, float *R,int axis

             */
            Pointer kernelParameter = Pointer.to(Pointer.to(new int[]{a.getDataLength()}), Pointer.to(a.getGpuData()), Pointer.to(b.getGpuData()), Pointer.to(y.getGpuData()), Pointer.to(new int[]{axis}));
            checkCUDA(cuLaunchKernel(add_axis_function, CAFFE_GET_BLOCKS(a.getDataLength()), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameter, null // Kernel- and extra parameters
            ));
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void add_axis_gpu(Tensor a, Tensor b, Tensor y, int axis) {
        try {
            /**
             * int N, float *X, float *Y, float *R,int axis

             */
            Pointer kernelParameter = Pointer.to(Pointer.to(new int[]{a.getDataLength()}), Pointer.to(a.getGpuData()), Pointer.to(b.getGpuData()), Pointer.to(y.getGpuData()), Pointer.to(new int[]{axis}));
            checkCUDA(cuLaunchKernel(add_axis_function2, CAFFE_GET_BLOCKS(a.getDataLength()), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameter, null // Kernel- and extra parameters
            ));
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
    
    public void add_axis_gpu(Tensor a, Tensor b, Tensor y, int N, int C, int H, int W, int axis) {
        try {
            /**
             * int N, float *X, float *Y, float *R,int N,int C,int H,int W,int axis
             */
            Pointer kernelParameter = Pointer.to(Pointer.to(new int[]{a.getDataLength()}), Pointer.to(a.getGpuData()), Pointer.to(b.getGpuData()), Pointer.to(y.getGpuData()),Pointer.to(new int[]{N}),Pointer.to(new int[]{C}),Pointer.to(new int[]{H}),Pointer.to(new int[]{W}), Pointer.to(new int[]{axis}));
            checkCUDA(cuLaunchKernel(add_axis_function3, CAFFE_GET_BLOCKS(a.getDataLength()), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameter, null // Kernel- and extra parameters
            ));
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
    
    public void add_axis_back_gpu(Tensor dx, Tensor dy, int N, int C, int H, int W, int axis) {
        try {
            /**
             * int N, float *dX, float *dY, int B, int C, int H, int W, int axis
             */
            Pointer kernelParameter = Pointer.to(Pointer.to(new int[]{dx.getDataLength()}), Pointer.to(dx.getGpuData()), Pointer.to(dy.getGpuData()),Pointer.to(new int[]{N}),Pointer.to(new int[]{C}),Pointer.to(new int[]{H}),Pointer.to(new int[]{W}), Pointer.to(new int[]{axis}));
            checkCUDA(cuLaunchKernel(add_axis_back_function, CAFFE_GET_BLOCKS(dx.getDataLength()), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameter, null // Kernel- and extra parameters
            ));
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void add_gpu(Tensor a, Tensor b, Tensor y, int offset, int N) {
        try {
            /**
             * int N, float *X, float *Y, float *R

             */
            Pointer kernelParameter = Pointer.to(Pointer.to(new int[]{N}), Pointer.to(a.getGpuData().withByteOffset(offset * Sizeof.FLOAT)), Pointer.to(b.getGpuData().withByteOffset(offset * Sizeof.FLOAT)), Pointer.to(y.getGpuData().withByteOffset(offset * Sizeof.FLOAT)));
            checkCUDA(cuLaunchKernel(add_gpu_function, CAFFE_GET_BLOCKS(N), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameter, null // Kernel- and extra parameters
            ));
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void add_gpu(Tensor a, Tensor b, Tensor y, int offsetA, int offsetB, int offsetY, int N) {
        try {
            /**
             * int N, float *X, float *Y, float *R

             */
            Pointer kernelParameter = Pointer.to(Pointer.to(new int[]{N}), Pointer.to(a.getGpuData().withByteOffset(offsetA * Sizeof.FLOAT)), Pointer.to(b.getGpuData().withByteOffset(offsetB * Sizeof.FLOAT)), Pointer.to(y.getGpuData().withByteOffset(offsetY * Sizeof.FLOAT)));
            checkCUDA(cuLaunchKernel(add_gpu_function, CAFFE_GET_BLOCKS(N), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameter, null // Kernel- and extra parameters
            ));
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void axpy_gpu(Tensor a, Tensor b, int offsetX, int offsetY) {
        try {
            /**
             * int N,  float *X, int OFFX, float *Y, int OFFY

             */
            Pointer kernelParameter = Pointer.to(Pointer.to(new int[]{a.getDataLength()}), Pointer.to(a.getGpuData()), Pointer.to(new int[]{offsetX}), Pointer.to(b.getGpuData()), Pointer.to(new int[]{offsetY}));
            checkCUDA(cuLaunchKernel(axpy_gpu_function, CAFFE_GET_BLOCKS(b.getDataLength()), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameter, null // Kernel- and extra parameters
            ));
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void add_scalar_gpu(Tensor a, float b, Tensor y) {
        try {
            /**
             * int N, float *X, float ALPHA, float *R

             */
            Pointer kernelParameter = Pointer.to(Pointer.to(new int[]{y.getDataLength()}), Pointer.to(a.getGpuData()), Pointer.to(new float[]{b}), Pointer.to(y.getGpuData()));
            checkCUDA(cuLaunchKernel(add_scalar_gpu_function, CAFFE_GET_BLOCKS(y.getDataLength()), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameter, null // Kernel- and extra parameters
            ));
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void add_number_gpu(Tensor a, Tensor b, int start) {
        try {
            /**
             * int N,  float *X, float *Y, int n,int c,int h,int w,int start

             */
            Pointer kernelParameter = Pointer.to(Pointer.to(new int[]{b.getDataLength()}), Pointer.to(a.getGpuData()), Pointer.to(b.getGpuData()), Pointer.to(new int[]{a.number}), Pointer.to(new int[]{a.channel}), Pointer.to(new int[]{a.height}), Pointer.to(new int[]{a.width}), Pointer.to(new int[]{start}));
            checkCUDA(cuLaunchKernel(add_number_gpu_function, CAFFE_GET_BLOCKS(b.getDataLength()), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameter, null // Kernel- and extra parameters
            ));
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void add_channel_gpu(Tensor a, Tensor b, int start) {
        try {
            /**
             * int N,  float *X, float *Y, int n,int c,int h,int w,int start

             */
            Pointer kernelParameter = Pointer.to(Pointer.to(new int[]{b.getDataLength()}), Pointer.to(a.getGpuData()), Pointer.to(b.getGpuData()), Pointer.to(new int[]{a.number}), Pointer.to(new int[]{a.channel}), Pointer.to(new int[]{a.height}), Pointer.to(new int[]{a.width}), Pointer.to(new int[]{start}));
            checkCUDA(cuLaunchKernel(add_channel_gpu_function, CAFFE_GET_BLOCKS(b.getDataLength()), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameter, null // Kernel- and extra parameters
            ));
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
    

    public void add_channel_gpu(Tensor a, Tensor b, int[] shape, int start) {
        try {
            /**
             * int N,  float *X, float *Y, int n,int c,int h,int w,int start
             */
            Pointer kernelParameter = Pointer.to(Pointer.to(new int[]{b.getDataLength()}), Pointer.to(a.getGpuData()), Pointer.to(b.getGpuData()), Pointer.to(new int[]{shape[0]}), Pointer.to(new int[]{shape[1]}), Pointer.to(new int[]{shape[2]}), Pointer.to(new int[]{shape[3]}), Pointer.to(new int[]{start}));
            checkCUDA(cuLaunchKernel(add_channel_gpu_function, CAFFE_GET_BLOCKS(b.getDataLength()), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameter, null // Kernel- and extra parameters
            ));
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
    

    public void sub_gpu(Tensor a, Tensor b, Tensor y) {
        try {
            /**
             * int N, float *X, float *Y, float *R

             */
            Pointer kernelParameter = Pointer.to(Pointer.to(new int[]{y.getDataLength()}), Pointer.to(a.getGpuData()), Pointer.to(b.getGpuData()), Pointer.to(y.getGpuData()));
            checkCUDA(cuLaunchKernel(sub_gpu_function, CAFFE_GET_BLOCKS(y.getDataLength()), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameter, null // Kernel- and extra parameters
            ));
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void sub_gpu(Tensor a, Tensor b, Tensor y, int axis) {
        try {
            int axis_len = 0;
            switch (axis) {
                case 0:
                    axis_len = a.channel * a.height * a.width;
                    break;
                case 1:
                    axis_len = a.height * a.width;
                    break;
            }
            /**
             * int N, float *X, float *Y, float *R, int axis

             */
            Pointer kernelParameter = Pointer.to(Pointer.to(new int[]{a.getDataLength()}), Pointer.to(a.getGpuData()), Pointer.to(b.getGpuData()), Pointer.to(y.getGpuData()), Pointer.to(new int[]{axis_len}));
            checkCUDA(cuLaunchKernel(sub_axis_gpu_function, CAFFE_GET_BLOCKS(a.getDataLength()), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameter, null // Kernel- and extra parameters
            ));
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void sub_gpu(Tensor a, Tensor b, Tensor y, int offset, int N) {
        try {
            /**
             * int N, float *X, float *Y, float *R

             */
            Pointer kernelParameter = Pointer.to(Pointer.to(new int[]{N}), Pointer.to(a.getGpuData().withByteOffset(offset * Sizeof.FLOAT)), Pointer.to(b.getGpuData().withByteOffset(offset * Sizeof.FLOAT)), Pointer.to(y.getGpuData().withByteOffset(offset * Sizeof.FLOAT)));
            checkCUDA(cuLaunchKernel(sub_gpu_function, CAFFE_GET_BLOCKS(N), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameter, null // Kernel- and extra parameters
            ));
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void sub_scalar_gpu(Tensor a, float b, Tensor y) {
        try {
            /**
             * int N, float *X, float ALPHA, float *R

             */
            Pointer kernelParameter = Pointer.to(Pointer.to(new int[]{y.getDataLength()}), Pointer.to(a.getGpuData()), Pointer.to(new float[]{b}), Pointer.to(y.getGpuData()));
            checkCUDA(cuLaunchKernel(sub_scalar_gpu_function, CAFFE_GET_BLOCKS(y.getDataLength()), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameter, null // Kernel- and extra parameters
            ));
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void scalar_sub_gpu(float a, Tensor b, Tensor y) {
        try {
            /**
             * int N, float *X, float ALPHA, float *R

             */
            Pointer kernelParameter = Pointer.to(Pointer.to(new int[]{y.getDataLength()}), Pointer.to(new float[]{a}), Pointer.to(b.getGpuData()), Pointer.to(y.getGpuData()));
            checkCUDA(cuLaunchKernel(scalar_sub_gpu_function, CAFFE_GET_BLOCKS(y.getDataLength()), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameter, null // Kernel- and extra parameters
            ));
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void scalar_sub_gpu(float a, Tensor b, Tensor y, int offset, int N) {
        try {
            /**
             * int N, float *X, float ALPHA, float *R

             */
            Pointer kernelParameter = Pointer.to(Pointer.to(new int[]{N}), Pointer.to(new float[]{a}), Pointer.to(b.getGpuData().withByteOffset(offset * Sizeof.FLOAT)), Pointer.to(y.getGpuData().withByteOffset(offset * Sizeof.FLOAT)));
            checkCUDA(cuLaunchKernel(scalar_sub_gpu_function, CAFFE_GET_BLOCKS(N), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameter, null // Kernel- and extra parameters
            ));
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void mul_gpu(Tensor a, Tensor b, Tensor y) {
        try {
            /**
             * int N, float *X, float *Y, float *R

             */
            Pointer kernelParameter = Pointer.to(Pointer.to(new int[]{y.getDataLength()}), Pointer.to(a.getGpuData()), Pointer.to(b.getGpuData()), Pointer.to(y.getGpuData()));
            checkCUDA(cuLaunchKernel(mul_gpu_function, CAFFE_GET_BLOCKS(y.getDataLength()), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameter, null // Kernel- and extra parameters
            ));
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void bool_gpu(Tensor a, Tensor b, Tensor y, float val) {
        try {
            /**
             * int N, float *X, float *Y, float *R

             */
            Pointer kernelParameter = Pointer.to(Pointer.to(new int[]{y.getDataLength()}), Pointer.to(a.getGpuData()), Pointer.to(b.getGpuData()), Pointer.to(y.getGpuData()), Pointer.to(new float[]{val}));
            checkCUDA(cuLaunchKernel(bool_gpu_function, CAFFE_GET_BLOCKS(y.getDataLength()), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameter, null // Kernel- and extra parameters
            ));
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
    
    public void mask_gpu(Tensor a, Tensor b, Tensor y, float val) {
        try {
            /**
             * int N, float *X, float *Y, float *R, int onceSize,float val
             */
            Pointer kernelParameter = Pointer.to(Pointer.to(new int[]{y.getDataLength()}), Pointer.to(a.getGpuData()), Pointer.to(b.getGpuData()), Pointer.to(y.getGpuData()), Pointer.to(new int[]{a.getOnceSize()}), Pointer.to(new float[]{val}));
            checkCUDA(cuLaunchKernel(mask_gpu_function, CAFFE_GET_BLOCKS(y.getDataLength()), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameter, null // Kernel- and extra parameters
            ));
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
    
    public void mask_gpu(Tensor a, Tensor b, Tensor y, float val, int dataLen, int onceSize) {
        try {
            /**
             * int N, float *X, float *Y, float *R, int onceSize,float val
             */
            Pointer kernelParameter = Pointer.to(Pointer.to(new int[]{dataLen}), Pointer.to(a.getGpuData()), Pointer.to(b.getGpuData()), Pointer.to(y.getGpuData()), Pointer.to(new int[]{onceSize}), Pointer.to(new float[]{val}));
            checkCUDA(cuLaunchKernel(mask_gpu_function, CAFFE_GET_BLOCKS(dataLen), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameter, null // Kernel- and extra parameters
            ));
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
    
    public void mul_gpu(Tensor a, Tensor b, Tensor y, int N, int C, int H, int W, int axis) {
        try {
            /**
             * int N, float *X, float *Y, float *R, int B, int C, int H, int W, int axis
             */
            Pointer kernelParameter = Pointer.to(Pointer.to(new int[]{a.getDataLength()}), Pointer.to(a.getGpuData()), Pointer.to(b.getGpuData()), Pointer.to(y.getGpuData()), Pointer.to(new int[]{N}), Pointer.to(new int[]{C}), Pointer.to(new int[]{H}), Pointer.to(new int[]{W}), Pointer.to(new int[]{axis}));
            checkCUDA(cuLaunchKernel(mul_axis_function, CAFFE_GET_BLOCKS(a.getDataLength()), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameter, null // Kernel- and extra parameters
            ));
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
    
    public void mul_back_left_gpu(Tensor b, Tensor delta, Tensor da, int N, int C, int H, int W, int axis) {
        try {
            /**
             * int N, float *Y, float *delta, float *dx,int axis
             */
            Pointer kernelParameter = Pointer.to(Pointer.to(new int[]{da.getDataLength()}), Pointer.to(b.getGpuData()), Pointer.to(delta.getGpuData()), Pointer.to(da.getGpuData()), Pointer.to(new int[]{N}), Pointer.to(new int[]{C}), Pointer.to(new int[]{H}), Pointer.to(new int[]{W}), Pointer.to(new int[]{axis}));
            checkCUDA(cuLaunchKernel(mul_axis_back_left_function, CAFFE_GET_BLOCKS(da.getDataLength()), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameter, null // Kernel- and extra parameters
            ));
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
    
    public void mul_back_right_gpu(Tensor a, Tensor delta, Tensor db, int N, int C, int H, int W, int axis) {
        try {
            /**
             * int N, float *X, float *delta, float *dy,int axis
             */
            Pointer kernelParameter = Pointer.to(Pointer.to(new int[]{db.getDataLength()}), Pointer.to(a.getGpuData()), Pointer.to(delta.getGpuData()), Pointer.to(db.getGpuData()), Pointer.to(new int[]{N}), Pointer.to(new int[]{C}), Pointer.to(new int[]{H}), Pointer.to(new int[]{W}), Pointer.to(new int[]{axis}));
            checkCUDA(cuLaunchKernel(mul_axis_back_right_function, CAFFE_GET_BLOCKS(db.getDataLength()), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameter, null // Kernel- and extra parameters
            ));
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
    
    public void mul_gpu(Tensor a, Tensor b, Tensor y, int offset, int N) {
        try {
            /**
             * int N, float *X, float *Y, float *R

             */
            Pointer kernelParameter = Pointer.to(Pointer.to(new int[]{N}), Pointer.to(a.getGpuData().withByteOffset(offset * Sizeof.FLOAT)), Pointer.to(b.getGpuData().withByteOffset(offset * Sizeof.FLOAT)), Pointer.to(y.getGpuData().withByteOffset(offset * Sizeof.FLOAT)));
            checkCUDA(cuLaunchKernel(mul_gpu_function, CAFFE_GET_BLOCKS(N), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameter, null // Kernel- and extra parameters
            ));
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void mul_gpu(Tensor a, Tensor b, Tensor y, int offsetA, int offsetB, int offsetY, int N) {
        try {
            /**
             * int N, float *X, float *Y, float *R

             */
            Pointer kernelParameter = Pointer.to(Pointer.to(new int[]{N}), Pointer.to(a.getGpuData().withByteOffset(offsetA * Sizeof.FLOAT)), Pointer.to(b.getGpuData().withByteOffset(offsetB * Sizeof.FLOAT)), Pointer.to(y.getGpuData().withByteOffset(offsetY * Sizeof.FLOAT)));
            checkCUDA(cuLaunchKernel(mul_gpu_function, CAFFE_GET_BLOCKS(N), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameter, null // Kernel- and extra parameters
            ));
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void mul_scalar_gpu(Tensor a, float b, Tensor y) {
        try {
            /**
             * int N, float *X, float ALPHA, float *R

             */
            Pointer kernelParameter = Pointer.to(Pointer.to(new int[]{y.getDataLength()}), Pointer.to(a.getGpuData()), Pointer.to(new float[]{b}), Pointer.to(y.getGpuData()));
            checkCUDA(cuLaunchKernel(mul_scalar_gpu_function, CAFFE_GET_BLOCKS(y.getDataLength()), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameter, null // Kernel- and extra parameters
            ));
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void mul_plus_gpu(Tensor a, Tensor b, Tensor y) {
        try {
            /**
             * int N, float *X, float *Y, float *R

             */
            Pointer kernelParameter = Pointer.to(Pointer.to(new int[]{y.getDataLength()}), Pointer.to(a.getGpuData()), Pointer.to(b.getGpuData()), Pointer.to(y.getGpuData()));
            checkCUDA(cuLaunchKernel(mul_plus_gpu_function, CAFFE_GET_BLOCKS(y.getDataLength()), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameter, null // Kernel- and extra parameters
            ));
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void mul_plus_scalar_gpu(Tensor a, float b, Tensor y) {
        try {
            /**
             * int N, float *X, float ALPHA, float *R

             */
            Pointer kernelParameter = Pointer.to(Pointer.to(new int[]{y.getDataLength()}), Pointer.to(a.getGpuData()), Pointer.to(new float[]{b}), Pointer.to(y.getGpuData()));
            checkCUDA(cuLaunchKernel(mul_plus_scalar_gpu_function, CAFFE_GET_BLOCKS(y.getDataLength()), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameter, null // Kernel- and extra parameters
            ));
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void mul_plus_scalar_gpu(Tensor a, float b, Tensor y, int axis) {
        try {
            /**
             * int N, float *X, float ALPHA, float *R, int axis

             */
            Pointer kernelParameter = Pointer.to(Pointer.to(new int[]{y.getDataLength()}), Pointer.to(a.getGpuData()), Pointer.to(new float[]{b}), Pointer.to(y.getGpuData()), Pointer.to(new int[]{axis}));
            checkCUDA(cuLaunchKernel(mul_plus_scalar_axis_gpu_function, CAFFE_GET_BLOCKS(y.getDataLength()), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameter, null // Kernel- and extra parameters
            ));
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void div_gpu(Tensor a, Tensor b, Tensor y) {
        try {
            /**
             * int N, float *X, float *Y, float *R

             */
            Pointer kernelParameter = Pointer.to(Pointer.to(new int[]{y.getDataLength()}), Pointer.to(a.getGpuData()), Pointer.to(b.getGpuData()), Pointer.to(y.getGpuData()));
            checkCUDA(cuLaunchKernel(div_gpu_function, CAFFE_GET_BLOCKS(y.getDataLength()), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameter, null // Kernel- and extra parameters
            ));
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void div_gpu(Tensor a, Tensor b, Tensor y, CUstream stream) {
        try {
            /**
             * int N, float *X, float *Y, float *R

             */
            Pointer kernelParameter = Pointer.to(Pointer.to(new int[]{y.getDataLength()}), Pointer.to(a.getGpuData()), Pointer.to(b.getGpuData()), Pointer.to(y.getGpuData()));
            checkCUDA(cuLaunchKernel(div_gpu_function, CAFFE_GET_BLOCKS(y.getDataLength()), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, stream,               // Shared memory size and stream
                    kernelParameter, null // Kernel- and extra parameters
            ));
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void div_gpu(Tensor a, Tensor b, Tensor y, int axis) {
        try {
            int axis_len = 0;
            switch (axis) {
                case 0:
                    axis_len = a.channel * a.height * a.width;
                    break;
                case 1:
                    axis_len = a.height * a.width;
                    break;
            }
            /**
             * int N, float *X, float *Y, float *R, int axis

             */
            Pointer kernelParameter = Pointer.to(Pointer.to(new int[]{y.getDataLength()}), Pointer.to(a.getGpuData()), Pointer.to(b.getGpuData()), Pointer.to(y.getGpuData()), Pointer.to(new int[]{axis_len}));
            checkCUDA(cuLaunchKernel(div_axis_gpu_function, CAFFE_GET_BLOCKS(y.getDataLength()), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameter, null // Kernel- and extra parameters
            ));
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void div_scalar_gpu(Tensor a, float b, Tensor y) {
        try {
            /**
             * int N, float *X, float ALPHA, float *R

             */
            Pointer kernelParameter = Pointer.to(Pointer.to(new int[]{y.getDataLength()}), Pointer.to(a.getGpuData()), Pointer.to(new float[]{b}), Pointer.to(y.getGpuData()));
            checkCUDA(cuLaunchKernel(div_scalar_gpu_function, CAFFE_GET_BLOCKS(y.getDataLength()), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameter, null // Kernel- and extra parameters
            ));
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void div_scalar_gpu(Tensor a, float b, Tensor y, CUstream stream) {
        try {
            /**
             * int N, float *X, float ALPHA, float *R

             */
            Pointer kernelParameter = Pointer.to(Pointer.to(new int[]{y.getDataLength()}), Pointer.to(a.getGpuData()), Pointer.to(new float[]{b}), Pointer.to(y.getGpuData()));
            checkCUDA(cuLaunchKernel(div_scalar_gpu_function, CAFFE_GET_BLOCKS(y.getDataLength()), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, stream,               // Shared memory size and stream
                    kernelParameter, null // Kernel- and extra parameters
            ));
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void scalar_div_gpu(Tensor a, float b, Tensor y) {
        try {
            /**
             * int N, float *X, float ALPHA, float *R

             */
            Pointer kernelParameter = Pointer.to(Pointer.to(new int[]{y.getDataLength()}), Pointer.to(a.getGpuData()), Pointer.to(new float[]{b}), Pointer.to(y.getGpuData()));
            checkCUDA(cuLaunchKernel(scalar_div_gpu_function, CAFFE_GET_BLOCKS(y.getDataLength()), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameter, null // Kernel- and extra parameters
            ));
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void div_plus_gpu(Tensor a, Tensor b, Tensor y) {
        try {
            /**
             * int N, float *X, float *Y, float *R

             */
            Pointer kernelParameter = Pointer.to(Pointer.to(new int[]{y.getDataLength()}), Pointer.to(a.getGpuData()), Pointer.to(b.getGpuData()), Pointer.to(y.getGpuData()));
            checkCUDA(cuLaunchKernel(div_plus_gpu_function, CAFFE_GET_BLOCKS(y.getDataLength()), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameter, null // Kernel- and extra parameters
            ));
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void div_plus_gpu(Tensor a, Tensor b, Tensor y, int axis) {
        try {
            int axis_len = 0;
            switch (axis) {
                case 0:
                    axis_len = a.channel * a.height * a.width;
                    break;
                case 1:
                    axis_len = a.height * a.width;
                    break;
            }
            /**
             * int N, float *X, float *Y, float *R, int axis

             */
            Pointer kernelParameter = Pointer.to(Pointer.to(new int[]{y.getDataLength()}), Pointer.to(a.getGpuData()), Pointer.to(b.getGpuData()), Pointer.to(y.getGpuData()), Pointer.to(new int[]{axis_len}));
            checkCUDA(cuLaunchKernel(div_plus_axis_gpu_function, CAFFE_GET_BLOCKS(y.getDataLength()), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameter, null // Kernel- and extra parameters
            ));
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void div_plus_scalar_gpu(Tensor a, float b, Tensor y) {
        try {
            /**
             * int N, float *X, float ALPHA, float *R

             */
            Pointer kernelParameter = Pointer.to(Pointer.to(new int[]{y.getDataLength()}), Pointer.to(a.getGpuData()), Pointer.to(new float[]{b}), Pointer.to(y.getGpuData()));
            checkCUDA(cuLaunchKernel(div_plus_scalar_gpu_function, CAFFE_GET_BLOCKS(y.getDataLength()), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameter, null // Kernel- and extra parameters
            ));
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void scalar_plus_div_gpu(Tensor a, float b, Tensor y) {
        try {
            /**
             * int N, float *X, float ALPHA, float *R

             */
            Pointer kernelParameter = Pointer.to(Pointer.to(new int[]{y.getDataLength()}), Pointer.to(a.getGpuData()), Pointer.to(new float[]{b}), Pointer.to(y.getGpuData()));
            checkCUDA(cuLaunchKernel(scalar_plus_div_gpu_function, CAFFE_GET_BLOCKS(y.getDataLength()), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameter, null // Kernel- and extra parameters
            ));
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public int getAxis(Tensor a, Tensor b) {
        if (a.getDataLength() == b.getDataLength()) {
            return -1;
        }
        return 0;
    }

    public void div_bGrad_gpu(Tensor a, Tensor b, Tensor c, Tensor y) {
        int axis = this.getAxis(a, y);
        if (axis >= 0) {
            div_bGrad_gpu(a, b, c, y, axis);
            return;
        }
        try {
            /**
             * int N, float *A, float *B, float *C, float *Y

             */
            Pointer kernelParameter = Pointer.to(Pointer.to(new int[]{y.getDataLength()}), Pointer.to(a.getGpuData()), Pointer.to(b.getGpuData()), Pointer.to(c.getGpuData()), Pointer.to(y.getGpuData()));
            checkCUDA(cuLaunchKernel(div_bGrad_gpu_function, CAFFE_GET_BLOCKS(y.getDataLength()), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameter, null // Kernel- and extra parameters
            ));
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void div_bGrad_gpu(Tensor a, Tensor b, Tensor c, Tensor y, int axis) {
        try {
            int axis_len = 0;
            switch (axis) {
                case 0:
                    axis_len = a.channel * a.height * a.width;
                    break;
                case 1:
                    axis_len = a.height * a.width;
                    break;
            }
            /**
             * int N, float *A, float *B, float *C, float *Y

             */
            Pointer kernelParameter = Pointer.to(Pointer.to(new int[]{y.getDataLength()}), Pointer.to(a.getGpuData()), Pointer.to(b.getGpuData()), Pointer.to(c.getGpuData()), Pointer.to(y.getGpuData()), Pointer.to(new int[]{axis_len}));
            checkCUDA(cuLaunchKernel(div_bGrad_axis_gpu_function, CAFFE_GET_BLOCKS(y.getDataLength()), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameter, null // Kernel- and extra parameters
            ));
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void div_scalar_bGrad_gpu(Tensor a, float b, Tensor c, Tensor y) {
        try {
            /**
             * int N, float *D, float A, float *B, float *Y

             */
            Pointer kernelParameter = Pointer.to(Pointer.to(new int[]{y.getDataLength()}), Pointer.to(a.getGpuData()), Pointer.to(new float[]{b}), Pointer.to(c.getGpuData()), Pointer.to(y.getGpuData()));
            checkCUDA(cuLaunchKernel(div_scalar_bGrad_gpu_function, CAFFE_GET_BLOCKS(y.getDataLength()), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameter, null // Kernel- and extra parameters
            ));
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void pow_gpu(Tensor a, float b, Tensor y) {
        try {
            /**
             * int N, float ALPHA, float *X, float *Y

             */
            Pointer kernelParameter = Pointer.to(Pointer.to(new int[]{y.getDataLength()}), Pointer.to(a.getGpuData()), Pointer.to(new float[]{b}), Pointer.to(y.getGpuData()));
            checkCUDA(cuLaunchKernel(pow_gpu_function, CAFFE_GET_BLOCKS(y.getDataLength()), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameter, null // Kernel- and extra parameters
            ));
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void log_gpu(Tensor a, Tensor y) {
        try {
            /**
             * int N, float *X, float *Y

             */
            Pointer kernelParameter = Pointer.to(Pointer.to(new int[]{y.getDataLength()}), Pointer.to(a.getGpuData()), Pointer.to(y.getGpuData()));
            checkCUDA(cuLaunchKernel(log_gpu_function, CAFFE_GET_BLOCKS(y.getDataLength()), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameter, null // Kernel- and extra parameters
            ));
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void exp_gpu(Tensor a, Tensor y) {
        try {
            /**
             * int N, float *X, float *Y

             */
            Pointer kernelParameter = Pointer.to(Pointer.to(new int[]{y.getDataLength()}), Pointer.to(a.getGpuData()), Pointer.to(y.getGpuData()));
            checkCUDA(cuLaunchKernel(exp_gpu_function, CAFFE_GET_BLOCKS(y.getDataLength()), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameter, null // Kernel- and extra parameters
            ));
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void sum_gpu(Tensor a, Tensor y, int axis) {
        try {
            if (axis == 0) {
                /**
                 * int N, float *X, float *Y

                 */
                Pointer kernelParameter = Pointer.to(Pointer.to(new int[]{a.dataLength}), Pointer.to(a.getGpuData()), Pointer.to(y.getGpuData()));
                checkCUDA(cuLaunchKernel(sum_gpu_function, 1, 1, 1,      // Grid dimension
                        1, 1, 1,      // Block dimension
                        0, null,               // Shared memory size and stream
                        kernelParameter, null // Kernel- and extra parameters
                ));
            } else if (axis == 2) {
                /**
                 * int N, float *X, float *Y,int C,int H,int W
                 */
                Pointer kernelParameter = Pointer.to(Pointer.to(new int[]{y.getDataLength()}), Pointer.to(a.getGpuData()), Pointer.to(y.getGpuData()), Pointer.to(new int[]{a.channel}), Pointer.to(new int[]{a.height}), Pointer.to(new int[]{a.width}));
                checkCUDA(cuLaunchKernel(sum_height_gpu_function, CAFFE_GET_BLOCKS(y.getDataLength()), 1, 1,      // Grid dimension
                        CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                        0, null,               // Shared memory size and stream
                        kernelParameter, null // Kernel- and extra parameters
                ));
            } else {
                /**
                 * int N, float *X, float *Y,int C,int H,int W
                 */
                Pointer kernelParameter = Pointer.to(Pointer.to(new int[]{y.getDataLength()}), Pointer.to(a.getGpuData()), Pointer.to(y.getGpuData()), Pointer.to(new int[]{a.channel}), Pointer.to(new int[]{a.height}), Pointer.to(new int[]{a.width}));
                checkCUDA(cuLaunchKernel(sum_channel_gpu_function, CAFFE_GET_BLOCKS(y.getDataLength()), 1, 1,      // Grid dimension
                        CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                        0, null,               // Shared memory size and stream
                        kernelParameter, null // Kernel- and extra parameters
                ));
            }
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void sum_pow_gpu(Tensor a, Tensor y, double p, int axis) {
        try {
            if (axis == 0) {
                /**
                 * int N,double p, float *X, float *Y

                 */
                Pointer kernelParameter = Pointer.to(Pointer.to(new int[]{a.dataLength}), Pointer.to(new double[]{p}), Pointer.to(a.getGpuData()), Pointer.to(y.getGpuData()));
                checkCUDA(cuLaunchKernel(sum_pow_gpu_function, 1, 1, 1,      // Grid dimension
                        1, 1, 1,      // Block dimension
                        0, null,               // Shared memory size and stream
                        kernelParameter, null // Kernel- and extra parameters
                ));
            } else if (axis == 2) {
                /**
                 * int N, float *X, float *Y,int C,int H,int W

                 */
                Pointer kernelParameter = Pointer.to(Pointer.to(new int[]{y.getDataLength()}), Pointer.to(new double[]{p}), Pointer.to(a.getGpuData()), Pointer.to(y.getGpuData()), Pointer.to(new int[]{a.channel}), Pointer.to(new int[]{a.height}), Pointer.to(new int[]{a.width}));
                checkCUDA(cuLaunchKernel(sum_pow_height_gpu_function, CAFFE_GET_BLOCKS(y.getDataLength()), 1, 1,      // Grid dimension
                        CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                        0, null,               // Shared memory size and stream
                        kernelParameter, null // Kernel- and extra parameters
                ));
            } else {
                /**
                 * int N, float *X, float *Y,int C,int H,int W

                 */
                Pointer kernelParameter = Pointer.to(Pointer.to(new int[]{y.getDataLength()}), Pointer.to(new double[]{p}), Pointer.to(a.getGpuData()), Pointer.to(y.getGpuData()), Pointer.to(new int[]{a.channel}), Pointer.to(new int[]{a.height}), Pointer.to(new int[]{a.width}));
                checkCUDA(cuLaunchKernel(sum_pow_channel_gpu_function, CAFFE_GET_BLOCKS(y.getDataLength()), 1, 1,      // Grid dimension
                        CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                        0, null,               // Shared memory size and stream
                        kernelParameter, null // Kernel- and extra parameters
                ));
            }
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void max_gpu(Tensor a, Tensor y, int axis) {
        try {
            if (axis == 0) {
                /**
                 * int N, float *X, float *Y

                 */
                Pointer kernelParameter = Pointer.to(Pointer.to(new int[]{a.dataLength}), Pointer.to(a.getGpuData()), Pointer.to(y.getGpuData()));
                checkCUDA(cuLaunchKernel(max_gpu_function, 1, 1, 1,      // Grid dimension
                        1, 1, 1,      // Block dimension
                        0, null,               // Shared memory size and stream
                        kernelParameter, null // Kernel- and extra parameters
                ));
            } else {
                /**
                 * int N, float *X, float *Y,int C,int H,int W

                 */
                Pointer kernelParameter = Pointer.to(Pointer.to(new int[]{y.getDataLength()}), Pointer.to(a.getGpuData()), Pointer.to(y.getGpuData()), Pointer.to(new int[]{a.channel}), Pointer.to(new int[]{a.height}), Pointer.to(new int[]{a.width}));
                checkCUDA(cuLaunchKernel(max_channel_gpu_function, CAFFE_GET_BLOCKS(y.getDataLength()), 1, 1,      // Grid dimension
                        CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                        0, null,               // Shared memory size and stream
                        kernelParameter, null // Kernel- and extra parameters
                ));
            }
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void max_backward_gpu(Tensor d, Tensor a, Tensor y, int axis) {
        try {
            if (axis == 0) {
                /**
                 * int N, float *D, float *X, float *Y

                 */
                Pointer kernelParameter = Pointer.to(Pointer.to(new int[]{a.dataLength}), Pointer.to(d.getGpuData()), Pointer.to(a.getGpuData()), Pointer.to(y.getGpuData()));
                checkCUDA(cuLaunchKernel(max_backward_gpu_function, 1, 1, 1,      // Grid dimension
                        1, 1, 1,      // Block dimension
                        0, null,               // Shared memory size and stream
                        kernelParameter, null // Kernel- and extra parameters
                ));
            } else {
                /**
                 * int N, float *D, float *X, float *Y, int C, int H, int W

                 */
                Pointer kernelParameter = Pointer.to(Pointer.to(new int[]{y.getDataLength()}), Pointer.to(d.getGpuData()), Pointer.to(a.getGpuData()), Pointer.to(y.getGpuData()), Pointer.to(new int[]{a.channel}), Pointer.to(new int[]{a.height}), Pointer.to(new int[]{a.width}));
                checkCUDA(cuLaunchKernel(max_channel_backward_gpu_function, CAFFE_GET_BLOCKS(y.getDataLength()), 1, 1,      // Grid dimension
                        CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                        0, null,               // Shared memory size and stream
                        kernelParameter, null // Kernel- and extra parameters
                ));
            }
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void sqrt_gpu(Tensor a, Tensor y) {
        try {
            /**
             * int N, float *X, float *Y

             */
            Pointer kernelParameter = Pointer.to(Pointer.to(new int[]{y.getDataLength()}), Pointer.to(a.getGpuData()), Pointer.to(y.getGpuData()));
            checkCUDA(cuLaunchKernel(sqrt_gpu_function, CAFFE_GET_BLOCKS(y.getDataLength()), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameter, null // Kernel- and extra parameters
            ));
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void sin_gpu(Tensor a, Tensor y) {
        try {
            /**
             * int N, float *X, float *Y

             */
            Pointer kernelParameter = Pointer.to(Pointer.to(new int[]{y.getDataLength()}), Pointer.to(a.getGpuData()), Pointer.to(y.getGpuData()));
            checkCUDA(cuLaunchKernel(sin_gpu_function, CAFFE_GET_BLOCKS(y.getDataLength()), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameter, null // Kernel- and extra parameters
            ));
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void cos_gpu(Tensor a, Tensor y) {
        try {
            /**
             * int N, float *X, float *Y

             */
            Pointer kernelParameter = Pointer.to(Pointer.to(new int[]{y.getDataLength()}), Pointer.to(a.getGpuData()), Pointer.to(y.getGpuData()));
            checkCUDA(cuLaunchKernel(cos_gpu_function, CAFFE_GET_BLOCKS(y.getDataLength()), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameter, null // Kernel- and extra parameters
            ));
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void tan_gpu(Tensor a, Tensor y) {
        try {
            /**
             * int N, float *X, float *Y

             */
            Pointer kernelParameter = Pointer.to(Pointer.to(new int[]{y.getDataLength()}), Pointer.to(a.getGpuData()), Pointer.to(y.getGpuData()));
            checkCUDA(cuLaunchKernel(tan_gpu_function, CAFFE_GET_BLOCKS(y.getDataLength()), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameter, null // Kernel- and extra parameters
            ));
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void tan_back_gpu(Tensor a, Tensor y) {
        try {
            /**
             * int N, float *X, float *Y

             */
            Pointer kernelParameter = Pointer.to(Pointer.to(new int[]{y.getDataLength()}), Pointer.to(a.getGpuData()), Pointer.to(y.getGpuData()));
            checkCUDA(cuLaunchKernel(tan_back_gpu_function, CAFFE_GET_BLOCKS(y.getDataLength()), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameter, null // Kernel- and extra parameters
            ));
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void atan_gpu(Tensor a, Tensor y) {
        try {
            /**
             * int N, float *X, float *Y

             */
            Pointer kernelParameter = Pointer.to(Pointer.to(new int[]{y.getDataLength()}), Pointer.to(a.getGpuData()), Pointer.to(y.getGpuData()));
            checkCUDA(cuLaunchKernel(atan_gpu_function, CAFFE_GET_BLOCKS(y.getDataLength()), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameter, null // Kernel- and extra parameters
            ));
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void atan_back_gpu(Tensor a, Tensor y) {
        try {
            /**
             * int N, float *X, float *Y

             */
            Pointer kernelParameter = Pointer.to(Pointer.to(new int[]{y.getDataLength()}), Pointer.to(a.getGpuData()), Pointer.to(y.getGpuData()));
            checkCUDA(cuLaunchKernel(atan_back_gpu_function, CAFFE_GET_BLOCKS(y.getDataLength()), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameter, null // Kernel- and extra parameters
            ));
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void broadcast_gpu(Tensor a, Tensor y, int axis) {
        try {
            if (axis == 0) {
                /**
                 * int N, float *X, float *Y

                 */
                Pointer kernelParameter = Pointer.to(Pointer.to(new int[]{y.getDataLength()}), Pointer.to(a.getGpuData()), Pointer.to(y.getGpuData()));
                checkCUDA(cuLaunchKernel(broadcast_gpu_function, CAFFE_GET_BLOCKS(y.getDataLength()), 1, 1,      // Grid dimension
                        CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                        0, null,               // Shared memory size and stream
                        kernelParameter, null // Kernel- and extra parameters
                ));
            } else {
                /**
                 * int N, float *X, float *Y,int C,int H,int W
                 */
                Pointer kernelParameter = Pointer.to(Pointer.to(new int[]{y.getDataLength()}), Pointer.to(a.getGpuData()), Pointer.to(y.getGpuData()), Pointer.to(new int[]{y.channel}), Pointer.to(new int[]{y.height}), Pointer.to(new int[]{y.width}));
                checkCUDA(cuLaunchKernel(broadcast_channel_gpu_function, CAFFE_GET_BLOCKS(y.getDataLength()), 1, 1,      // Grid dimension
                        CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                        0, null,               // Shared memory size and stream
                        kernelParameter, null // Kernel- and extra parameters
                ));
            }
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void broadcast_plus_gpu(Tensor a, Tensor y, int axis) {
        try {
            if (axis == 0) {
                /**
                 * int N, float *X, float *Y

                 */
                Pointer kernelParameter = Pointer.to(Pointer.to(new int[]{y.getDataLength()}), Pointer.to(a.getGpuData()), Pointer.to(y.getGpuData()));
                checkCUDA(cuLaunchKernel(broadcast_plus_gpu_function, CAFFE_GET_BLOCKS(y.getDataLength()), 1, 1,      // Grid dimension
                        CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                        0, null,               // Shared memory size and stream
                        kernelParameter, null // Kernel- and extra parameters
                ));
            } else {
                /**
                 * int N, float *X, float *Y,int C,int H,int W

                 */
                Pointer kernelParameter = Pointer.to(Pointer.to(new int[]{y.getDataLength()}), Pointer.to(a.getGpuData()), Pointer.to(y.getGpuData()), Pointer.to(new int[]{y.channel}), Pointer.to(new int[]{y.height}), Pointer.to(new int[]{y.width}));
                checkCUDA(cuLaunchKernel(broadcast_channel_plus_gpu_function, CAFFE_GET_BLOCKS(y.getDataLength()), 1, 1,      // Grid dimension
                        CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                        0, null,               // Shared memory size and stream
                        kernelParameter, null // Kernel- and extra parameters
                ));
            }
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void broadcast_row_plus_gpu(Tensor a, Tensor y) {
        try {
            /**
             * int N, float *X, float *Y,int C,int H,int W

             */
            Pointer kernelParameter = Pointer.to(Pointer.to(new int[]{y.getDataLength()}), Pointer.to(a.getGpuData()), Pointer.to(y.getGpuData()), Pointer.to(new int[]{y.channel}), Pointer.to(new int[]{y.height}), Pointer.to(new int[]{y.width}));
            checkCUDA(cuLaunchKernel(broadcast_row_plus_gpu_function, CAFFE_GET_BLOCKS(y.getDataLength()), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameter, null // Kernel- and extra parameters
            ));
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void clamp_gpu(Tensor a, float b1, float b2, Tensor y) {
        try {
            /**
             * int N, float *X, float ALPHA, float *R

             */
            Pointer kernelParameter = Pointer.to(Pointer.to(new int[]{y.getDataLength()}), Pointer.to(a.getGpuData()), Pointer.to(new float[]{b1}), Pointer.to(new float[]{b2}), Pointer.to(y.getGpuData()));
            checkCUDA(cuLaunchKernel(clamp_gpu_function, CAFFE_GET_BLOCKS(y.getDataLength()), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameter, null // Kernel- and extra parameters
            ));
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void clamp_back_gpu(Tensor a, float b1, float b2, Tensor y) {
        try {
            /**
             * int N, float *X, float ALPHA, float *R

             */
            Pointer kernelParameter = Pointer.to(Pointer.to(new int[]{y.getDataLength()}), Pointer.to(a.getGpuData()), Pointer.to(new float[]{b1}), Pointer.to(new float[]{b2}), Pointer.to(y.getGpuData()));
            checkCUDA(cuLaunchKernel(clamp_back_gpu_function, CAFFE_GET_BLOCKS(y.getDataLength()), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameter, null // Kernel- and extra parameters
            ));
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void maximum_gpu(Tensor a, Tensor b, Tensor y) {
        try {
            /**
             * int N, float *X, float ALPHA, float *R

             */
            Pointer kernelParameter = Pointer.to(Pointer.to(new int[]{y.getDataLength()}), Pointer.to(a.getGpuData()), Pointer.to(b.getGpuData()), Pointer.to(y.getGpuData()));
            checkCUDA(cuLaunchKernel(maximum_gpu_function, CAFFE_GET_BLOCKS(y.getDataLength()), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameter, null // Kernel- and extra parameters
            ));
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void minimum_gpu(Tensor a, Tensor b, Tensor y) {
        try {
            /**
             * int N, float *X, float ALPHA, float *R

             */
            Pointer kernelParameter = Pointer.to(Pointer.to(new int[]{y.getDataLength()}), Pointer.to(a.getGpuData()), Pointer.to(b.getGpuData()), Pointer.to(y.getGpuData()));
            checkCUDA(cuLaunchKernel(minimum_gpu_function, CAFFE_GET_BLOCKS(y.getDataLength()), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameter, null // Kernel- and extra parameters
            ));
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void maximum_back_gpu(Tensor a, Tensor b, Tensor y) {
        try {
            /**
             * int N, float *X, float ALPHA, float *R

             */
            Pointer kernelParameter = Pointer.to(Pointer.to(new int[]{y.getDataLength()}), Pointer.to(a.getGpuData()), Pointer.to(b.getGpuData()), Pointer.to(y.getGpuData()));
            checkCUDA(cuLaunchKernel(maximum_back_gpu_function, CAFFE_GET_BLOCKS(y.getDataLength()), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameter, null // Kernel- and extra parameters
            ));
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void transpose_gpu(Tensor x, Tensor y) {
        try {
            /**
             * int N, float *A, float *B

             */
            Pointer kernelParameter = Pointer.to(Pointer.to(new int[]{y.getDataLength()}), Pointer.to(x.getGpuData()), Pointer.to(y.getGpuData()), Pointer.to(new int[]{x.number}), Pointer.to(new int[]{x.width}));
            checkCUDA(cuLaunchKernel(transpose_gpu_function, CAFFE_GET_BLOCKS(y.getDataLength()), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameter, null // Kernel- and extra parameters
            ));
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void permute_gpu(Tensor x, Tensor y, int[] permutes) {
        try {
            int[] strides_in = getStrides(x.shape());
            int[] strides_out = getStrides(y.shape());
            Pointer permutes_p = this.getCudaManager().getMemoryManager().getCUPointer(permutes.length, Sizeof.INT);
            JCuda.cudaMemcpy(permutes_p, Pointer.to(permutes), permutes.length * Sizeof.INT, cudaMemcpyKind.cudaMemcpyHostToDevice);
            Pointer sip = this.getCudaManager().getMemoryManager().getCUPointer(permutes.length, Sizeof.INT);
            JCuda.cudaMemcpy(sip, Pointer.to(strides_in), permutes.length * Sizeof.INT, cudaMemcpyKind.cudaMemcpyHostToDevice);
            Pointer sop = this.getCudaManager().getMemoryManager().getCUPointer(permutes.length, Sizeof.INT);
            JCuda.cudaMemcpy(sop, Pointer.to(strides_out), permutes.length * Sizeof.INT, cudaMemcpyKind.cudaMemcpyHostToDevice);
            /**
             * int N, float *data_in, float *data_out, int *perms, int *strides_in, int *strides_out, int NUM_AXES
             */
            Pointer kernelParameter = Pointer.to(Pointer.to(new int[]{x.getDataLength()}), Pointer.to(x.getGpuData()), Pointer.to(y.getGpuData()), Pointer.to(permutes_p), Pointer.to(sip), Pointer.to(sop), Pointer.to(new int[]{permutes.length}));
            checkCUDA(cuLaunchKernel(permute_gpu_function, CAFFE_GET_BLOCKS(x.getDataLength()), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameter, null // Kernel- and extra parameters
            ));
            this.getCudaManager().getMemoryManager().freeCUPointer(permutes_p);
            this.getCudaManager().getMemoryManager().freeCUPointer(sip);
            this.getCudaManager().getMemoryManager().freeCUPointer(sop);
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
    
    public void permute_add_gpu(Tensor x, Tensor y, int[] permutes) {
        try {
            int[] strides_in = getStrides(x.shape());
            int[] strides_out = getStrides(y.shape());
            Pointer permutes_p = this.getCudaManager().getMemoryManager().getCUPointer(permutes.length, Sizeof.INT);
            JCuda.cudaMemcpy(permutes_p, Pointer.to(permutes), permutes.length * Sizeof.INT, cudaMemcpyKind.cudaMemcpyHostToDevice);
            Pointer sip = this.getCudaManager().getMemoryManager().getCUPointer(permutes.length, Sizeof.INT);
            JCuda.cudaMemcpy(sip, Pointer.to(strides_in), permutes.length * Sizeof.INT, cudaMemcpyKind.cudaMemcpyHostToDevice);
            Pointer sop = this.getCudaManager().getMemoryManager().getCUPointer(permutes.length, Sizeof.INT);
            JCuda.cudaMemcpy(sop, Pointer.to(strides_out), permutes.length * Sizeof.INT, cudaMemcpyKind.cudaMemcpyHostToDevice);
            /**
             * int N, float *data_in, float *data_out, int *perms, int *strides_in, int *strides_out, int NUM_AXES

             */
            Pointer kernelParameter = Pointer.to(Pointer.to(new int[]{x.getDataLength()}), Pointer.to(x.getGpuData()), Pointer.to(y.getGpuData()), Pointer.to(permutes_p), Pointer.to(sip), Pointer.to(sop), Pointer.to(new int[]{permutes.length}));
            checkCUDA(cuLaunchKernel(permute_add_gpu_function, CAFFE_GET_BLOCKS(x.getDataLength()), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameter, null // Kernel- and extra parameters
            ));
            this.getCudaManager().getMemoryManager().freeCUPointer(permutes_p);
            this.getCudaManager().getMemoryManager().freeCUPointer(sip);
            this.getCudaManager().getMemoryManager().freeCUPointer(sop);
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
    
    public void permute_gpu(Tensor x, Tensor y, int[] xSahpe, int[] ySahpe, int[] permutes) {
        try {
            int[] strides_in = getStrides(xSahpe);
            int[] strides_out = getStrides(ySahpe);
            Pointer permutes_p = this.getCudaManager().getMemoryManager().getCUPointer(permutes.length, Sizeof.INT);
            JCuda.cudaMemcpy(permutes_p, Pointer.to(permutes), permutes.length * Sizeof.INT, cudaMemcpyKind.cudaMemcpyHostToDevice);
            Pointer sip = this.getCudaManager().getMemoryManager().getCUPointer(permutes.length, Sizeof.INT);
            JCuda.cudaMemcpy(sip, Pointer.to(strides_in), permutes.length * Sizeof.INT, cudaMemcpyKind.cudaMemcpyHostToDevice);
            Pointer sop = this.getCudaManager().getMemoryManager().getCUPointer(permutes.length, Sizeof.INT);
            JCuda.cudaMemcpy(sop, Pointer.to(strides_out), permutes.length * Sizeof.INT, cudaMemcpyKind.cudaMemcpyHostToDevice);
            /**
             * int N, float *data_in, float *data_out, int *perms, int *strides_in, int *strides_out, int NUM_AXES
             */
            Pointer kernelParameter = Pointer.to(Pointer.to(new int[]{x.getDataLength()}), Pointer.to(x.getGpuData()), Pointer.to(y.getGpuData()), Pointer.to(permutes_p), Pointer.to(sip), Pointer.to(sop), Pointer.to(new int[]{permutes.length}));
            checkCUDA(cuLaunchKernel(permute_gpu_function, CAFFE_GET_BLOCKS(x.getDataLength()), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameter, null // Kernel- and extra parameters
            ));
            this.getCudaManager().getMemoryManager().freeCUPointer(permutes_p);
            this.getCudaManager().getMemoryManager().freeCUPointer(sip);
            this.getCudaManager().getMemoryManager().freeCUPointer(sop);
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
    
    public void permute_gpu(Tensor x, Tensor y, int[] xSahpe, int[] ySahpe, int[] permutes,int dataLen) {
        try {
            int[] strides_in = getStrides(xSahpe);
            int[] strides_out = getStrides(ySahpe);
            Pointer permutes_p = this.getCudaManager().getMemoryManager().getCUPointer(permutes.length, Sizeof.INT);
            JCuda.cudaMemcpy(permutes_p, Pointer.to(permutes), permutes.length * Sizeof.INT, cudaMemcpyKind.cudaMemcpyHostToDevice);
            Pointer sip = this.getCudaManager().getMemoryManager().getCUPointer(permutes.length, Sizeof.INT);
            JCuda.cudaMemcpy(sip, Pointer.to(strides_in), permutes.length * Sizeof.INT, cudaMemcpyKind.cudaMemcpyHostToDevice);
            Pointer sop = this.getCudaManager().getMemoryManager().getCUPointer(permutes.length, Sizeof.INT);
            JCuda.cudaMemcpy(sop, Pointer.to(strides_out), permutes.length * Sizeof.INT, cudaMemcpyKind.cudaMemcpyHostToDevice);
            /**
             * int N, float *data_in, float *data_out, int *perms, int *strides_in, int *strides_out, int NUM_AXES
             */
            Pointer kernelParameter = Pointer.to(Pointer.to(new int[]{dataLen}), Pointer.to(x.getGpuData()), Pointer.to(y.getGpuData()), Pointer.to(permutes_p), Pointer.to(sip), Pointer.to(sop), Pointer.to(new int[]{permutes.length}));
            checkCUDA(cuLaunchKernel(permute_gpu_function, CAFFE_GET_BLOCKS(dataLen), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameter, null // Kernel- and extra parameters
            ));
            this.getCudaManager().getMemoryManager().freeCUPointer(permutes_p);
            this.getCudaManager().getMemoryManager().freeCUPointer(sip);
            this.getCudaManager().getMemoryManager().freeCUPointer(sop);
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public int[] dim_out(int[] dim_in, int[] permutes) {
        int[] dim_out = new int[dim_in.length];
        for (int i = 0; i < permutes.length; ++i) {
            dim_out[i] = dim_in[permutes[i]];
        }
        return dim_out;
    }

    public int[] getStrides(int[] dims) {
        int[] strides = new int[dims.length];
        for (int i = 0; i < dims.length; i++) {
            strides[i] = 1;
        }
        for (int i = dims.length - 2; i >= 0; --i) {
            //			System.out.println(dims[i + 1]);
            strides[i] = strides[i + 1] * dims[i + 1];
        }
        return strides;
    }

    public void minimum_back_gpu(Tensor a, Tensor b, Tensor y) {
        try {
            /**
             * int N, float *X, float ALPHA, float *R

             */
            Pointer kernelParameter = Pointer.to(Pointer.to(new int[]{y.getDataLength()}), Pointer.to(a.getGpuData()), Pointer.to(b.getGpuData()), Pointer.to(y.getGpuData()));
            checkCUDA(cuLaunchKernel(minimum_back_gpu_function, CAFFE_GET_BLOCKS(y.getDataLength()), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameter, null // Kernel- and extra parameters
            ));
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void mean_2dim_gpu(Tensor x, Tensor y) {
        try {
            /**
             * int N, float *x, float *y, int C
             */
            Pointer kernelParameter = Pointer.to(Pointer.to(new int[]{x.number * x.channel}), Pointer.to(x.getGpuData()), Pointer.to(y.getGpuData()), Pointer.to(new int[]{x.height * x.width}));
            checkCUDA(cuLaunchKernel(mean_function, CAFFE_GET_BLOCKS(x.number * x.channel), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameter, null // Kernel- and extra parameters
            ));
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void mean_2dim_back_gpu(Tensor dy, Tensor dx) {
        try {
            /**
             * int N, float *dy, float *dx, int C
             */
            Pointer kernelParameter = Pointer.to(Pointer.to(new int[]{dx.number * dx.channel}), Pointer.to(dy.getGpuData()), Pointer.to(dx.getGpuData()), Pointer.to(new int[]{dx.height * dx.width}));
            checkCUDA(cuLaunchKernel(mean_back_function, CAFFE_GET_BLOCKS(dx.number * dx.channel), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameter, null // Kernel- and extra parameters
            ));
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void mean_gpu(Tensor a, int dim, Tensor y) {
        try {
            int scalar = a.number;
            if (dim == 1) {
                scalar = a.channel;
            }
            sum_gpu(a, y, dim);
            div_scalar_gpu(y, scalar, y);
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void expand_gpu(Tensor a, Tensor b, int num) {
        // TODO Auto-generated method stub
        try {
            /**
             * int N, float *X, float *Y,int axis

             */
            Pointer kernelParameter = Pointer.to(Pointer.to(new int[]{b.getDataLength()}), Pointer.to(a.getGpuData()), Pointer.to(b.getGpuData()), Pointer.to(new int[]{num}));
            checkCUDA(cuLaunchKernel(expand_function, CAFFE_GET_BLOCKS(b.getDataLength()), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameter, null // Kernel- and extra parameters
            ));
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void cat_gpu(Tensor a, Tensor b, Tensor c) {
        // TODO Auto-generated method stub
        try {
            int offset = 0;
            int part_input_size = a.getOnceSize() / 1;
            for (int n = 0; n < a.number; n++) {
                this.copy_gpu(a, c, part_input_size, n * a.getOnceSize() + part_input_size * 0, 1, offset + n * c.getOnceSize(), 1);
            }
            offset += part_input_size;
            part_input_size = b.getOnceSize() / 1;
            for (int n = 0; n < a.number; n++) {
            	this.copy_gpu(b, c, part_input_size, n * b.getOnceSize() + part_input_size * 0, 1, offset + n * c.getOnceSize(), 1);
            }
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void cat_back_gpu(Tensor c, Tensor a, Tensor b) {
        // TODO Auto-generated method stub
        try {
            int offset = 0;
            int part_input_size = a.getOnceSize() / 1;
            for (int n = 0; n < c.number; n++) {
                this.axpy_gpu(c, a, part_input_size, 1, offset + n * c.getOnceSize(), 1, n * a.getOnceSize() + part_input_size * 0, 1);
            }
            offset += part_input_size;
            part_input_size = b.getOnceSize() / 1;
            for (int n = 0; n < c.number; n++) {
            	this.axpy_gpu(c, b, part_input_size, 1, offset + n * c.getOnceSize(), 1, n * b.getOnceSize() + part_input_size * 0, 1);
            }
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
    
    public void cat_width_gpu(Tensor a, Tensor b, Tensor c) {
        // TODO Auto-generated method stub
        try {
        	/**
             * int N, float *a, float *b, float *y, int W
             */
            Pointer kernelParameter = Pointer.to(Pointer.to(new int[]{a.number * a.channel * a.height}), Pointer.to(a.getGpuData()), Pointer.to(b.getGpuData()), Pointer.to(c.getGpuData()), Pointer.to(new int[]{a.width}));
            checkCUDA(cuLaunchKernel(cat_width_function, CAFFE_GET_BLOCKS(a.number * a.channel * a.height), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameter, null // Kernel- and extra parameters
            ));
        	
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
    
    public void cat_width_back_gpu(Tensor dc, Tensor da, Tensor db) {
        // TODO Auto-generated method stub
        try {
        	/**
             * int N, float *da, float *db, float *dy, int W
             */
            Pointer kernelParameter = Pointer.to(Pointer.to(new int[]{da.number * da.channel * da.height}), Pointer.to(da.getGpuData()), Pointer.to(db.getGpuData()), Pointer.to(dc.getGpuData()), Pointer.to(new int[]{da.width}));
            checkCUDA(cuLaunchKernel(cat_width_back_function, CAFFE_GET_BLOCKS(da.number * da.channel * da.height), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameter, null // Kernel- and extra parameters
            ));
        	
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
    
    public void one_hot(Tensor a, Tensor b) {
        try {
            /**
             * int N, float *X, float *Y, int K
             */
            Pointer kernelParameter = Pointer.to(Pointer.to(new int[]{a.getDataLength()}), Pointer.to(a.getGpuData()), Pointer.to(b.getGpuData()), Pointer.to(new int[]{b.width}));
            checkCUDA(cuLaunchKernel(onehot_function, CAFFE_GET_BLOCKS(a.getDataLength()), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameter, null // Kernel- and extra parameters
            ));
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void copy_gpu(Tensor a, Tensor b) {
        this.copy_gpu(a, b, a.getDataLength(), 1, 1);
    }
    
    public void abs_gpu(Tensor x, Tensor y) {
        try {
            /**
             * int N, float* x, float *y
             */
            Pointer kernelParameter = Pointer.to(Pointer.to(new int[]{x.getDataLength()}), Pointer.to(x.getGpuData()), Pointer.to(y.getGpuData()));
            checkCUDA(cuLaunchKernel(abs_function, CAFFE_GET_BLOCKS(x.getDataLength()), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameter, null // Kernel- and extra parameters
            ));
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
    
    public void abs_backward_gpu(Tensor x,Tensor dx,Tensor dy) {
        try {
            /**
             * int N, float* x, float* dx, float *dy
             */
            Pointer kernelParameter = Pointer.to(Pointer.to(new int[]{x.getDataLength()}), Pointer.to(x.getGpuData()), Pointer.to(dx.getGpuData()), Pointer.to(dy.getGpuData()));
            checkCUDA(cuLaunchKernel(abs_backward_function, CAFFE_GET_BLOCKS(x.getDataLength()), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameter, null // Kernel- and extra parameters
            ));
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
    
    public void update_ema_gpu(Tensor x, Tensor y, float decay) {
        try {
            /**
             * int N, float *ema, float *model,float decay
             */
            Pointer kernelParameter = Pointer.to(Pointer.to(new int[]{x.getDataLength()}), Pointer.to(x.getGpuData()), Pointer.to(y.getGpuData()), Pointer.to(new float[]{decay}));
            checkCUDA(cuLaunchKernel(update_ema_function, CAFFE_GET_BLOCKS(x.getDataLength()), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameter, null // Kernel- and extra parameters
            ));
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
    
    public int CAFFE_GET_BLOCKS(int N) {
        return (N + CAFFE_CUDA_NUM_THREADS - 1) / CAFFE_CUDA_NUM_THREADS;
    }

    public void checkCUDA(int code) {
        if (code != cudaError.cudaSuccess) {
            System.err.println("Error code " + code + ":" + cudaError.stringFor(code));
        }
    }
}

