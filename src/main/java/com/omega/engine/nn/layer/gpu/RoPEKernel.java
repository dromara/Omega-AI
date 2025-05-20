package com.omega.engine.nn.layer.gpu;

import com.omega.common.utils.JsonUtils;
import com.omega.common.utils.MatrixOperation;
import com.omega.common.utils.MatrixUtils;
import com.omega.common.utils.RandomUtils;
import com.omega.engine.gpu.BaseKernel;
import com.omega.engine.gpu.CUDAManager;
import com.omega.engine.gpu.CUDAMemoryManager;
import com.omega.engine.nn.layer.transformer.RoPELayer;
import com.omega.engine.nn.network.Transformer;
import com.omega.engine.tensor.Tensor;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfunction;
import jcuda.driver.JCudaDriver;
import jcuda.runtime.cudaError;

import static jcuda.driver.JCudaDriver.cuLaunchKernel;

/**
 * Root Mean Sqrt Normalization
 * <p>
 * p = x^2
 * <p>
 * mean = 1/n∑x^2
 * <p>
 * rms = rsqrt(1/n∑x^2)
 * <p>
 * rms_norm = x * rms
 * <p>
 * drms = sum(x * diff)
 * <p>
 * dmean = -0.5 * (mean).pow(-1.5)
 * <p>
 * dp = 1/n
 * <p>
 * dx = rms * diff + sum(x * diff) * -0.5 * (mean).pow(-1.5) / n * 2 * x
 */
public class RoPEKernel extends BaseKernel {
    /**
     * 向前方法
     */
    private CUfunction forward_function;
    private CUfunction forward_all_function;
    private CUfunction forward_32_function;
    /**
     * 反向传播方法
     */
    private CUfunction backward_function;
    private CUfunction backward_all_function;
    private CUfunction backward_32_function;
    private CUfunction forward_all_32_function;
    private CUfunction backward_all_32_function;
    private int CAFFE_CUDA_NUM_THREADS = 1024;
    /**
     * 前向方法参数
     */
    private Pointer forwardParameters;
    private Pointer backwardParameters;

    public RoPEKernel(CUDAManager cudaManager) {
        super(cudaManager);
        init();
    }

    public static void main(String[] args) {
        try {
            int N = 3;
            int T = 5;
            int HN = 2;
            int W = 4;
            float[] data = RandomUtils.order(N * T * HN * W, 0.1f, 0.1f);
            Tensor input = new Tensor(N, T, HN, W, data, true);
            Tensor[] cs = getCosAndSin(T, HN * W, HN);
            Tensor cos = cs[0];
            Tensor sin = cs[1];
            Tensor delta = new Tensor(N, T, HN, W, RandomUtils.order(N * T * HN * W, 0.1f, 0.1f), true);
            //    	    	Tensor delta = new Tensor(N * T, 1, 1, W, MatrixUtils.order(N * T * W, 0.1f, 0.1f), true);
            Transformer tf = new Transformer();
            RoPELayer rope = new RoPELayer(tf);
            //	    	input.showDM();
            cos.showDM();
            sin.showDM();
            Tensor op1 = new Tensor(N, T, HN, W, true);
            for (int b = 0; b < N; b++) {
                for (int t = 0; t < T; t++) {
                    for (int h = 0; h < HN; h++) {
                        for (int d = 0; d < W / 2; d++) {
                            int index = b * T * HN * W + t * HN * W + h * W + d * 2;
                            float xr = input.data[index];
                            float xi = input.data[index + 1];
                            float cos_th = cos.data[t * W / 2 + d];
                            float sin_th = sin.data[t * W / 2 + d];
                            op1.data[index] = xr * cos_th - xi * sin_th;
                            op1.data[index + 1] = xr * sin_th + xi * cos_th;
                        }
                    }
                }
            }
            System.out.println("cpu-output:" + JsonUtils.toJson(op1.data));
            Tensor op2 = new Tensor(N, T, HN, W, true);
            for (int b = 0; b < N; b++) {
                for (int t = 0; t < T; t++) {
                    for (int h = 0; h < HN; h++) {
                        for (int d = 0; d < W / 2; d++) {
                            int index = b * T * HN * W + t * HN * W + h * W + d * 2;
                            float dr = delta.data[index];
                            float di = delta.data[index + 1];
                            float cos_th = cos.data[t * W / 2 + d];
                            float sin_th = sin.data[t * W / 2 + d];
                            op2.data[index] = dr * cos_th + di * sin_th;
                            op2.data[index + 1] = di * cos_th - dr * sin_th;
                        }
                    }
                }
            }
            System.out.println("cpu-diff:" + JsonUtils.toJson(op2.data));
            for (int i = 0; i < 10; i++) {
                rope.forward(cos, sin, input);
                rope.getOutput().showDM();
                rope.back(cos, sin, delta);
                rope.diff.showDM();
            }
            RoPEKernel kernel = new RoPEKernel(tf.cudaManager);
            float[] data2 = RandomUtils.order(N * T * HN * W, 0.01f, 0.01f);
            Tensor k2 = new Tensor(N, T, HN, W, data, true);
            System.err.println("-----------------------");
            input.showDM();
            kernel.forward(cos, sin, input, input);
            input.showDM();
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        } finally {
            // TODO: handle finally clause
            CUDAMemoryManager.free();
        }
    }

    public static float[] outer(float[] a, float[] b) {
        float[] o = new float[a.length * b.length];
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < b.length; j++) {
                o[i * b.length + j] = a[i] * b[j];
            }
        }
        return o;
    }

    public static float[] freqs(int start, int dim, int step) {
        double theta = 10000.0d;
        float[] r = new float[(dim - start) / step];
        for (int i = 0; i < r.length; i++) {
            r[i] = (float) (1.0d / Math.pow(theta, (i * step + start) * 1.0f / dim));
        }
        return r;
    }

    public static Tensor[] getCosAndSin(int time, int dim, int headNum) {
        int headSize = dim / headNum;
        float[] freqs = freqs(0, headSize, 2);
        float[] t = MatrixUtils.order(time, 0, 1);
        System.err.println(JsonUtils.toJson(t));
        float[] o = outer(t, freqs);
        System.err.println(JsonUtils.toJson(o));
        float[] cos = MatrixOperation.cos(o);
        float[] sin = MatrixOperation.sin(o);
        Tensor cos_t = new Tensor(1, 1, t.length, freqs.length, cos, true);
        Tensor sin_t = new Tensor(1, 1, t.length, freqs.length, sin, true);
        return new Tensor[]{cos_t, sin_t};
    }

    public static void getCosAndSin(int time, int dim, int headNum, Tensor[] pos) {
        if (pos == null) {
            pos = new Tensor[2];
        }
        int headSize = dim / headNum;
        float[] freqs = freqs(0, headSize, 2);
        float[] t = MatrixUtils.order(time, 0, 1);
        float[] o = outer(t, freqs);
        float[] cos = MatrixOperation.cos(o);
        float[] sin = MatrixOperation.sin(o);
        Tensor cos_t = pos[0];
        Tensor sin_t = pos[1];
        cos_t = Tensor.createTensor(cos_t, 1, 1, t.length, freqs.length, cos, true);
        sin_t = Tensor.createTensor(sin_t, 1, 1, t.length, freqs.length, sin, true);
    }

    public void initFunction() {
        try {
            if (forward_function == null) {
                forward_function = getCudaManager().getLocalFunctionByModule("RoPEKernel.cu", "rope_norm");
            }
            if (backward_function == null) {
                backward_function = getCudaManager().getLocalFunctionByModule("RoPEKernel.cu", "rope_backward");
            }
            if (forward_all_function == null) {
                forward_all_function = getCudaManager().getLocalFunctionByModule("RoPEKernel.cu", "rope_all_norm");
            }
            if (backward_all_function == null) {
                backward_all_function = getCudaManager().getLocalFunctionByModule("RoPEKernel.cu", "rope_all_backward");
            }
            if (forward_32_function == null) {
                forward_32_function = getCudaManager().getLocalFunctionByModule("RoPEKernel.cu", "rope_f32");
            }
            if (backward_32_function == null) {
                backward_32_function = getCudaManager().getLocalFunctionByModule("RoPEKernel.cu", "rope_backward_f32");
            }
            if (forward_all_32_function == null) {
                forward_all_32_function = getCudaManager().getLocalFunctionByModule("RoPEKernel.cu", "rope_all_f32");
            }
            if (backward_all_32_function == null) {
                backward_all_32_function = getCudaManager().getLocalFunctionByModule("RoPEKernel.cu", "rope_all_backward_f32");
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

    public void initBackward(Tensor input, Tensor delta, Tensor diff, Tensor gamma, Tensor dgamma, Tensor dbeta) {
    }

    public int CAFFE_GET_BLOCKS(int N) {
        return (N + CAFFE_CUDA_NUM_THREADS - 1) / CAFFE_CUDA_NUM_THREADS;
    }

    public void forward(Tensor cos, Tensor sin, Tensor input, Tensor output) {
        try {
            int nrow = input.number * input.channel;
            int ncol = input.height * input.width;
            /**
             * const float* x, float* dst,float* c_cos,float* c_sin, int ncols
             * const float* x, float* dst,float* c_cos,float* c_sin, int ncols, int T,int headSize

             */
            forwardParameters = Pointer.to(Pointer.to(input.getGpuData()), Pointer.to(output.getGpuData()), Pointer.to(cos.getGpuData()), Pointer.to(sin.getGpuData()), Pointer.to(new int[]{ncol}), Pointer.to(new int[]{input.channel}), Pointer.to(new int[]{input.width}));
            int[] block_dims = new int[]{1, 256, 1};
            int num_blocks_x = (ncol + 2 * 256 - 1) / (2 * 256);
            int[] block_nums = new int[]{nrow, num_blocks_x, 1};
            checkCUDA(cuLaunchKernel(forward_function, block_nums[0], block_nums[1], block_nums[2],      // Grid dimension
                    block_dims[0], block_dims[1], block_dims[2],      // Block dimension
                    0, null,               // Shared memory size and stream
                    forwardParameters, null // Kernel- and extra parameters
            ));
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void forward(Tensor cos, Tensor sin, Tensor q, Tensor k, Tensor qo, Tensor ko) {
        try {
            int nrow = q.number * q.channel;
            int ncol = q.height * q.width;
            /**
             * const float* q,const float* k, float* qo, float* ko,float* c_cos,float* c_sin, int ncols, int T,int headSize

             */
            forwardParameters = Pointer.to(Pointer.to(q.getGpuData()), Pointer.to(k.getGpuData()), Pointer.to(qo.getGpuData()), Pointer.to(ko.getGpuData()), Pointer.to(cos.getGpuData()), Pointer.to(sin.getGpuData()), Pointer.to(new int[]{ncol}), Pointer.to(new int[]{q.channel}), Pointer.to(new int[]{q.width}));
            int[] block_dims = new int[]{1, 256, 1};
            int num_blocks_x = (ncol + 2 * 256 - 1) / (2 * 256);
            int[] block_nums = new int[]{nrow, num_blocks_x, 1};
            checkCUDA(cuLaunchKernel(forward_all_function, block_nums[0], block_nums[1], block_nums[2],      // Grid dimension
                    block_dims[0], block_dims[1], block_dims[2],      // Block dimension
                    0, null,               // Shared memory size and stream
                    forwardParameters, null // Kernel- and extra parameters
            ));
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void forward32(Tensor input, Tensor output) {
        try {
            int nrow = input.number * input.channel;
            int ncol = input.height * input.width;
            float theta_scale = (float) Math.pow(10000.0d, -2.0d / ncol);
            /**
             * const float * x, float * dst, const int ncols, const int T, const float theta_scale

             */
            forwardParameters = Pointer.to(Pointer.to(input.getGpuData()), Pointer.to(output.getGpuData()), Pointer.to(new int[]{ncol}), Pointer.to(new int[]{input.channel}), Pointer.to(new float[]{theta_scale}));
            int[] block_dims = new int[]{1, 2 * 256, 1};
            int num_blocks_x = (ncol + 2 * 256 - 1) / (2 * 256);
            int[] block_nums = new int[]{nrow, num_blocks_x, 1};
            checkCUDA(cuLaunchKernel(forward_32_function, block_nums[0], block_nums[1], block_nums[2],      // Grid dimension
                    block_dims[0], block_dims[1], block_dims[2],      // Block dimension
                    0, null,               // Shared memory size and stream
                    forwardParameters, null // Kernel- and extra parameters
            ));
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void forwardAll32(Tensor q, Tensor k, Tensor qo, Tensor ko) {
        try {
            int nrow = q.number * q.channel;
            int ncol = q.height * q.width;
            float theta_scale = (float) Math.pow(10000.0d, -2.0d / ncol);
            /**
             * const float * q,const float * k, float * rq, float * rk, const int ncols, const int T, const float theta_scale

             */
            forwardParameters = Pointer.to(Pointer.to(q.getGpuData()), Pointer.to(k.getGpuData()), Pointer.to(qo.getGpuData()), Pointer.to(ko.getGpuData()), Pointer.to(new int[]{ncol}), Pointer.to(new int[]{q.channel}), Pointer.to(new float[]{theta_scale}));
            int[] block_dims = new int[]{1, 2 * 256, 1};
            int num_blocks_x = (ncol + 2 * 256 - 1) / (2 * 256);
            int[] block_nums = new int[]{nrow, num_blocks_x, 1};
            checkCUDA(cuLaunchKernel(forward_all_32_function, block_nums[0], block_nums[1], block_nums[2],      // Grid dimension
                    block_dims[0], block_dims[1], block_dims[2],      // Block dimension
                    0, null,               // Shared memory size and stream
                    forwardParameters, null // Kernel- and extra parameters
            ));
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void backward(Tensor cos, Tensor sin, Tensor delta, Tensor diff) {
        try {
            int nrow = delta.number * delta.channel;
            int ncol = delta.height * delta.width;
            /**
             * const float* x, float* dst,float* c_cos,float* c_sin, int ncols
             * const float* x, float* dst,float* c_cos,float* c_sin, int ncols, int T,int headSize
             */
            backwardParameters = Pointer.to(Pointer.to(delta.getGpuData()), Pointer.to(diff.getGpuData()), Pointer.to(cos.getGpuData()), Pointer.to(sin.getGpuData()), Pointer.to(new int[]{ncol}), Pointer.to(new int[]{delta.channel}), Pointer.to(new int[]{delta.width}));
            int[] block_dims = new int[]{1, 256, 1};
            int num_blocks_x = (ncol + 2 * 256 - 1) / (2 * 256);
            int[] block_nums = new int[]{nrow, num_blocks_x, 1};
            checkCUDA(cuLaunchKernel(backward_function, block_nums[0], block_nums[1], block_nums[2],      // Grid dimension
                    block_dims[0], block_dims[1], block_dims[2],      // Block dimension
                    0, null,               // Shared memory size and stream
                    backwardParameters, null // Kernel- and extra parameters
            ));
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void backward(Tensor cos, Tensor sin, Tensor deltaQ, Tensor deltaK, Tensor diffQ, Tensor diffK) {
        try {
            int nrow = deltaQ.number * deltaQ.channel;
            int ncol = deltaQ.height * deltaQ.width;
            /**
             * float* deltaQ,float* deltaK, float* diffQ, float* diffK,float* c_cos,float* c_sin, int ncols, int T,int headSize
             */
            backwardParameters = Pointer.to(Pointer.to(deltaQ.getGpuData()), Pointer.to(deltaK.getGpuData()), Pointer.to(diffQ.getGpuData()), Pointer.to(diffK.getGpuData()), Pointer.to(cos.getGpuData()), Pointer.to(sin.getGpuData()), Pointer.to(new int[]{ncol}), Pointer.to(new int[]{deltaQ.channel}), Pointer.to(new int[]{deltaQ.width}));
            int[] block_dims = new int[]{1, 256, 1};
            int num_blocks_x = (ncol + 2 * 256 - 1) / (2 * 256);
            int[] block_nums = new int[]{nrow, num_blocks_x, 1};
            checkCUDA(cuLaunchKernel(backward_all_function, block_nums[0], block_nums[1], block_nums[2],      // Grid dimension
                    block_dims[0], block_dims[1], block_dims[2],      // Block dimension
                    0, null,               // Shared memory size and stream
                    backwardParameters, null // Kernel- and extra parameters
            ));
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void backward32(Tensor delta, Tensor diff) {
        try {
            int nrow = delta.number * delta.channel;
            int ncol = delta.height * delta.width;
            float theta_scale = (float) Math.pow(10000.0d, -2.0d / ncol);
            /**
             * const float * x, float * dst, const int ncols, const int T, const float theta_scale

             */
            backwardParameters = Pointer.to(Pointer.to(delta.getGpuData()), Pointer.to(diff.getGpuData()), Pointer.to(new int[]{ncol}), Pointer.to(new int[]{delta.channel}), Pointer.to(new float[]{theta_scale}));
            int[] block_dims = new int[]{1, 2 * 256, 1};
            int num_blocks_x = (ncol + 2 * 256 - 1) / (2 * 256);
            int[] block_nums = new int[]{nrow, num_blocks_x, 1};
            checkCUDA(cuLaunchKernel(backward_32_function, block_nums[0], block_nums[1], block_nums[2],      // Grid dimension
                    block_dims[0], block_dims[1], block_dims[2],      // Block dimension
                    0, null,               // Shared memory size and stream
                    backwardParameters, null // Kernel- and extra parameters
            ));
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void backwardAll32(Tensor deltaQ, Tensor deltaK, Tensor diffQ, Tensor diffK) {
        try {
            int nrow = deltaQ.number * deltaQ.channel;
            int ncol = deltaQ.height * deltaQ.width;
            float theta_scale = (float) Math.pow(10000.0d, -2.0d / ncol);
            /**
             * float* deltaQ,float* deltaK, float* diffQ, float* diffK, const int ncols, const int T, const float theta_scale

             */
            backwardParameters = Pointer.to(Pointer.to(deltaQ.getGpuData()), Pointer.to(deltaK.getGpuData()), Pointer.to(diffQ.getGpuData()), Pointer.to(diffK.getGpuData()), Pointer.to(new int[]{ncol}), Pointer.to(new int[]{deltaQ.channel}), Pointer.to(new float[]{theta_scale}));
            int[] block_dims = new int[]{1, 2 * 256, 1};
            int num_blocks_x = (ncol + 2 * 256 - 1) / (2 * 256);
            int[] block_nums = new int[]{nrow, num_blocks_x, 1};
            checkCUDA(cuLaunchKernel(backward_all_32_function, block_nums[0], block_nums[1], block_nums[2],      // Grid dimension
                    block_dims[0], block_dims[1], block_dims[2],      // Block dimension
                    0, null,               // Shared memory size and stream
                    backwardParameters, null // Kernel- and extra parameters
            ));
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void checkCUDA(int code) {
        if (code != cudaError.cudaSuccess) {
            System.err.println("Error code " + code + ":" + cudaError.stringFor(code));
            throw new RuntimeException("Error code " + code + ":" + cudaError.stringFor(code));
        }
    }

    public void showDM(String id, CUdeviceptr d, float[] data) {
        JCudaDriver.cuMemcpyDtoH(Pointer.to(data), d, data.length * Sizeof.FLOAT);
        System.out.println(id + ":" + JsonUtils.toJson(data));
    }
}

