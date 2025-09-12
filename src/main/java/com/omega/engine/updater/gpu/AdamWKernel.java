package com.omega.engine.updater.gpu;

import com.omega.common.utils.RandomUtils;
import com.omega.engine.gpu.BaseKernel;
import com.omega.engine.gpu.CUDAManager;
import com.omega.engine.gpu.CUDAMemoryManager;
import com.omega.engine.loss.SoftmaxWithCrossEntropyLoss;
import com.omega.engine.nn.network.BPNetwork;
import com.omega.engine.nn.network.Network;
import com.omega.engine.tensor.Tensor;

import jcuda.Pointer;
import jcuda.driver.CUfunction;
import jcuda.driver.CUstream;

import java.util.Map;

import static jcuda.driver.JCudaDriver.cuLaunchKernel;

public class AdamWKernel extends BaseKernel {
    public Tensor mw;
    public Tensor vw;
    public Tensor mb;
    public Tensor vb;
    private float beta1 = 0.9f;
    private float beta2 = 0.95f;
    private CUfunction function;
    private CUfunction r_function;
    private CUfunction bn_function;
    private CUfunction adamw_function;
    private int CAFFE_CUDA_NUM_THREADS = 1024;
    private Pointer kernelParameters;
    private Pointer kernelBiasParameters;
    private float weight_decay = 0.0f;

    public AdamWKernel(int weightLength, float weight_decay, CUDAManager cudaManager) {
        super(cudaManager);
        this.mw = Tensor.createGPUTensor(mb, 1, 1, 1, weightLength, true);
        this.vw = Tensor.createGPUTensor(vw, 1, 1, 1, weightLength, true);
        this.weight_decay = weight_decay;
        init();
    }

    public AdamWKernel(int weightLength, int biasLength, float weight_decay, CUDAManager cudaManager) {
        super(cudaManager);
        this.mw = Tensor.createGPUTensor(mb, 1, 1, 1, weightLength, true);
        this.vw = Tensor.createGPUTensor(vw, 1, 1, 1, weightLength, true);
        this.mb = Tensor.createGPUTensor(mb, 1, 1, 1, biasLength, true);
        this.vb = Tensor.createGPUTensor(vb, 1, 1, 1, biasLength, true);
        this.weight_decay = weight_decay;
        init();
    }

    public AdamWKernel(int weightLength, int biasLength, float beta1, float beta2, float weight_decay, CUDAManager cudaManager) {
        super(cudaManager);
        this.mw = Tensor.createGPUTensor(mb, 1, 1, 1, weightLength, true);
        this.vw = Tensor.createGPUTensor(vw, 1, 1, 1, weightLength, true);
        this.mb = Tensor.createGPUTensor(mb, 1, 1, 1, biasLength, true);
        this.vb = Tensor.createGPUTensor(vb, 1, 1, 1, biasLength, true);
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.weight_decay = weight_decay;
        init();
    }

    public static void main(String args[]) {
        CUDAManager cudaManager = new CUDAManager(0);
        int N = 2;
        int C = 1;
        int H = 1;
        int W = 8;
        float[] test = new float[]{0.0075240037f, 0.022312285f, 0.037100658f, 0.05188888f, 0.06667703f, 0.08146531f, 0.09625361f, 0.111041375f, 0.12582973f, 0.14061777f, 0.15540561f, 0.17019409f, 0.18498187f, 0.19977006f, 0.21455756f, 0.22934535f, 0.24413382f, 0.25892144f, 0.27370873f, 0.2884969f, 0.303285f, 0.3180732f, 0.33286023f, 0.347648f, 0.36243534f, 0.37722275f, 0.39201018f, 0.40679908f, 0.42158592f, 0.43637308f, 0.45116156f, 0.46594855f, 0.48073593f, 0.4955238f, 0.5103103f, 0.52509815f, 0.53988534f, 0.55467236f};
        float[] x1 = new float[test.length];
        float[] bias1 = RandomUtils.order(N * C * H * W, 0.00001f, 0.00001f);
        Tensor w = new Tensor(1, 1, 1, test.length, x1, true);
        Tensor delta = new Tensor(1, 1, 1, test.length, test, true);
        BPNetwork net = new BPNetwork(new SoftmaxWithCrossEntropyLoss());
        net.train_time = 1;
        net.number = N;
        AdamWKernel k = new AdamWKernel(bias1.length, 0.001f, cudaManager);
        k.updateGamma(delta, w, net, 0.0001f);
        delta.showDM();
        w.showDM();
        CUDAMemoryManager.free();
    }

    public void setParams(Map<String, Float> params) {
        if (params != null) {
            if (params.get("beta1") != null) {
                this.beta1 = params.get("beta1");
            }
            if (params.get("beta2") != null) {
                this.beta2 = params.get("beta2");
            }
            if (params.get("weight_decay") != null) {
                this.weight_decay = params.get("weight_decay");
            }
        }
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
                function = getCudaManager().getLocalFunctionByModule("updater.cu", "adamw");
            }
            if (r_function == null) {
                r_function = getCudaManager().getLocalFunctionByModule("updater.cu", "adamwr");
            }
            if (bn_function == null) {
                bn_function = getCudaManager().getLocalFunctionByModule("updater.cu", "adamw_bn");
            }
            if (adamw_function == null) {
                adamw_function = getCudaManager().getLocalFunctionByModule("updater.cu", "adamw_kernel");
            }
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    //	public void updateW(Tensor diffW,Tensor weight,Network net,float lr) {
    //
    //		try {
    //
    //	        /**
    //	         * 设置入参
    //	         * float *diffW, float *weight,float *mw,float *vw,float beta1,float beta2,float learnRate, int n
    //	         */
    //			kernelParameters = Pointer.to(
    //					Pointer.to(diffW.getGpuData()),
    //	        		Pointer.to(weight.getGpuData()),
    //	                Pointer.to(mw.getGpuData()),
    //	                Pointer.to(vw.getGpuData()),
    //	                Pointer.to(new float[]{beta1}),
    //	                Pointer.to(new float[]{beta2}),
    //	                Pointer.to(new float[]{lr}),
    //	                Pointer.to(new float[]{weight_decay}),
    //	                Pointer.to(new int[]{diffW.dataLength}),
    //	                Pointer.to(new int[]{net.number}),
    //	                Pointer.to(new int[]{net.train_time})
    //	            );
    //
    //			cuLaunchKernel(function,
    //		            this.CAFFE_GET_BLOCKS(diffW.dataLength),  1, 1,      // Grid dimension
    //		            CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
    //		            0, null,               // Shared memory size and stream
    //		            kernelParameters, null // Kernel- and extra parameters
    //		        );
    //
    ////	        JCudaDriver.cuCtxSynchronize();
    //
    //		} catch (Exception e) {
    //			// TODO: handle exception
    //			e.printStackTrace();
    //		}
    //
    //	}
    public int CAFFE_GET_BLOCKS(int N) {
        return (N + CAFFE_CUDA_NUM_THREADS - 1) / CAFFE_CUDA_NUM_THREADS;
    }

    public void updateW(Tensor diffW, Tensor weight, Network net, float lr) {
        try {
            //			System.out.println(net.train_time);
            /**
             * 设置入参
             * float* weight, const float* diffW, float* mw, float* vw, int n,

             float learnRate, float beta1, float beta2, float beta1_correction, float beta2_correction, float eps, float weight_decay

             */
            kernelParameters = Pointer.to(Pointer.to(weight.getGpuData()), Pointer.to(diffW.getGpuData()), Pointer.to(mw.getGpuData()), Pointer.to(vw.getGpuData()), Pointer.to(new int[]{diffW.dataLength}), Pointer.to(new float[]{lr}), Pointer.to(new float[]{beta1}), Pointer.to(new float[]{beta2}), Pointer.to(new float[]{(float) (1.0f - Math.pow(beta1, net.train_time))}), Pointer.to(new float[]{(float) (1.0f - Math.pow(beta2, net.train_time))}), Pointer.to(new float[]{1e-8f}), Pointer.to(new float[]{weight_decay}));
            cuLaunchKernel(adamw_function, this.CAFFE_GET_BLOCKS(diffW.dataLength), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            );
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void updateW(Tensor diffW, Tensor weight, Network net, float lr, CUstream stream) {
        try {
            /**
             * 设置入参
             * float* weight, const float* diffW, float* mw, float* vw, int n,

             float learnRate, float beta1, float beta2, float beta1_correction, float beta2_correction, float eps, float weight_decay

             */
            kernelParameters = Pointer.to(Pointer.to(weight.getGpuData()), Pointer.to(diffW.getGpuData()), Pointer.to(mw.getGpuData()), Pointer.to(vw.getGpuData()), Pointer.to(new int[]{diffW.dataLength}), Pointer.to(new float[]{lr}), Pointer.to(new float[]{beta1}), Pointer.to(new float[]{beta2}), Pointer.to(new float[]{(float) (1.0f - Math.pow(beta1, net.train_time))}), Pointer.to(new float[]{(float) (1.0f - Math.pow(beta2, net.train_time))}), Pointer.to(new float[]{1e-8f}), Pointer.to(new float[]{weight_decay}));
            cuLaunchKernel(adamw_function, this.CAFFE_GET_BLOCKS(diffW.dataLength), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, stream,               // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            );
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    //	public void updateGamma(Tensor diffW,Tensor weight,Network net,float lr) {
    //
    //		try {
    //
    //	        /**
    //	         * 设置入参
    //	         * float *diffW, float *weight,float *mw,float *vw,float beta1,float beta2,float learnRate, float weight_decay, int n, int batch, int t
    //	         */
    //			kernelParameters = Pointer.to(
    //					Pointer.to(diffW.getGpuData()),
    //	        		Pointer.to(weight.getGpuData()),
    //	                Pointer.to(mw.getGpuData()),
    //	                Pointer.to(vw.getGpuData()),
    //	                Pointer.to(new float[]{beta1}),
    //	                Pointer.to(new float[]{beta2}),
    //	                Pointer.to(new float[]{lr}),
    //	                Pointer.to(new float[]{weight_decay}),
    //	                Pointer.to(new int[]{diffW.dataLength}),
    //	                Pointer.to(new int[]{net.number}),
    //	                Pointer.to(new int[]{net.train_time})
    //	            );
    //
    //			cuLaunchKernel(bn_function,
    //		            this.CAFFE_GET_BLOCKS(diffW.dataLength),  1, 1,      // Grid dimension
    //		            CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
    //		            0, null,               // Shared memory size and stream
    //		            kernelParameters, null // Kernel- and extra parameters
    //		        );
    //
    ////			System.out.println("diffW:"+net.train_time);
    ////			diffW.showDM();
    //
    ////	        JCudaDriver.cuCtxSynchronize();
    //
    //		} catch (Exception e) {
    //			// TODO: handle exception
    //			e.printStackTrace();
    //		}
    //
    //	}
    public void updateW(Tensor diffW, Tensor weight, Network net, float lr, int batchSize) {
        try {
            /**
             * 设置入参
             * float *diffW, float *weight,float *mw,float *vw,float beta1,float beta2,float learnRate, int n

             */
            kernelParameters = Pointer.to(Pointer.to(diffW.getGpuData()), Pointer.to(weight.getGpuData()), Pointer.to(mw.getGpuData()), Pointer.to(vw.getGpuData()), Pointer.to(new float[]{beta1}), Pointer.to(new float[]{beta2}), Pointer.to(new float[]{lr}), Pointer.to(new float[]{weight_decay}), Pointer.to(new int[]{diffW.dataLength}), Pointer.to(new int[]{batchSize}), Pointer.to(new int[]{net.train_time}));
            cuLaunchKernel(function, this.CAFFE_GET_BLOCKS(diffW.dataLength), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            );
            //			System.out.println("diffW:"+net.train_time);
            //			diffW.showDM();
            //	        JCudaDriver.cuCtxSynchronize();
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void updateGamma(Tensor diffW, Tensor weight, Network net, float lr) {
        try {
            /**
             * 设置入参
             * float* weight, const float* diffW, float* mw, float* vw, int n,

             float learnRate, float beta1, float beta2, float beta1_correction, float beta2_correction, float eps, float weight_decay

             */
            kernelParameters = Pointer.to(Pointer.to(weight.getGpuData()), Pointer.to(diffW.getGpuData()), Pointer.to(mw.getGpuData()), Pointer.to(vw.getGpuData()), Pointer.to(new int[]{diffW.dataLength}), Pointer.to(new float[]{lr}), Pointer.to(new float[]{beta1}), Pointer.to(new float[]{beta2}), Pointer.to(new float[]{(float) (1.0f - Math.pow(beta1, net.train_time))}), Pointer.to(new float[]{(float) (1.0f - Math.pow(beta2, net.train_time))}), Pointer.to(new float[]{1e-8f}), Pointer.to(new float[]{weight_decay}));
            cuLaunchKernel(adamw_function, this.CAFFE_GET_BLOCKS(diffW.dataLength), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            );
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    //	public void updateB(Tensor diffB,Tensor bias,Network net,float lr) {
    //
    //		try {
    //
    //	        /**
    //	         * 设置入参
    //	         * float *diffW, float *weight,float *mw,float *vw,float beta1,float beta2,float learnRate, int n
    //	         */
    //			kernelBiasParameters = Pointer.to(
    //					Pointer.to(diffB.getGpuData()),
    //	        		Pointer.to(bias.getGpuData()),
    //	                Pointer.to(mb.getGpuData()),
    //	                Pointer.to(vb.getGpuData()),
    //	                Pointer.to(new float[]{beta1}),
    //	                Pointer.to(new float[]{beta2}),
    //	                Pointer.to(new float[]{lr}),
    //	                Pointer.to(new float[]{weight_decay}),
    //	                Pointer.to(new int[]{diffB.dataLength}),
    //	                Pointer.to(new int[]{net.number}),
    //	                Pointer.to(new int[]{net.train_time})
    //	            );
    //
    //			cuLaunchKernel(function,
    //		            this.CAFFE_GET_BLOCKS(diffB.dataLength),  1, 1,      // Grid dimension
    //		            CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
    //		            0, null,               // Shared memory size and stream
    //		            kernelBiasParameters, null // Kernel- and extra parameters
    //		        );
    //
    ////	        JCudaDriver.cuCtxSynchronize();
    //
    //		} catch (Exception e) {
    //			// TODO: handle exception
    //			e.printStackTrace();
    //		}
    //
    //	}
    public void updateGamma(Tensor diffW, Tensor weight, Network net, float lr, CUstream stream) {
        try {
            /**
             * 设置入参
             * float* weight, const float* diffW, float* mw, float* vw, int n,

             float learnRate, float beta1, float beta2, float beta1_correction, float beta2_correction, float eps, float weight_decay

             */
            kernelParameters = Pointer.to(Pointer.to(weight.getGpuData()), Pointer.to(diffW.getGpuData()), Pointer.to(mw.getGpuData()), Pointer.to(vw.getGpuData()), Pointer.to(new int[]{diffW.dataLength}), Pointer.to(new float[]{lr}), Pointer.to(new float[]{beta1}), Pointer.to(new float[]{beta2}), Pointer.to(new float[]{(float) (1.0f - Math.pow(beta1, net.train_time))}), Pointer.to(new float[]{(float) (1.0f - Math.pow(beta2, net.train_time))}), Pointer.to(new float[]{1e-8f}), Pointer.to(new float[]{weight_decay}));
            cuLaunchKernel(adamw_function, this.CAFFE_GET_BLOCKS(diffW.dataLength), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, stream,               // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            );
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void updateB(Tensor diffB, Tensor bias, Network net, float lr) {
        try {
            /**
             * 设置入参
             * float* weight, const float* diffW, float* mw, float* vw, int n,

             float learnRate, float beta1, float beta2, float beta1_correction, float beta2_correction, float eps, float weight_decay

             */
            kernelParameters = Pointer.to(Pointer.to(bias.getGpuData()), Pointer.to(diffB.getGpuData()), Pointer.to(mb.getGpuData()), Pointer.to(vb.getGpuData()), Pointer.to(new int[]{diffB.dataLength}), Pointer.to(new float[]{lr}), Pointer.to(new float[]{beta1}), Pointer.to(new float[]{beta2}), Pointer.to(new float[]{(float) (1.0f - Math.pow(beta1, net.train_time))}), Pointer.to(new float[]{(float) (1.0f - Math.pow(beta2, net.train_time))}), Pointer.to(new float[]{1e-8f}), Pointer.to(new float[]{weight_decay}));
            cuLaunchKernel(adamw_function, this.CAFFE_GET_BLOCKS(diffB.dataLength), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            );
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void updateB(Tensor diffB, Tensor bias, Network net, float lr, CUstream stream) {
        try {
            /**
             * 设置入参
             * float* weight, const float* diffW, float* mw, float* vw, int n,

             float learnRate, float beta1, float beta2, float beta1_correction, float beta2_correction, float eps, float weight_decay

             */
            kernelParameters = Pointer.to(Pointer.to(bias.getGpuData()), Pointer.to(diffB.getGpuData()), Pointer.to(mb.getGpuData()), Pointer.to(vb.getGpuData()), Pointer.to(new int[]{diffB.dataLength}), Pointer.to(new float[]{lr}), Pointer.to(new float[]{beta1}), Pointer.to(new float[]{beta2}), Pointer.to(new float[]{(float) (1.0f - Math.pow(beta1, net.train_time))}), Pointer.to(new float[]{(float) (1.0f - Math.pow(beta2, net.train_time))}), Pointer.to(new float[]{1e-8f}), Pointer.to(new float[]{weight_decay}));
            cuLaunchKernel(adamw_function, this.CAFFE_GET_BLOCKS(diffB.dataLength), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, stream,               // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            );
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void updateB(Tensor diffB, Tensor bias, Network net, float lr, int batchSize) {
        try {
            /**
             * 设置入参
             * float *diffW, float *weight,float *mw,float *vw,float beta1,float beta2,float learnRate, int n

             */
            kernelBiasParameters = Pointer.to(Pointer.to(diffB.getGpuData()), Pointer.to(bias.getGpuData()), Pointer.to(mb.getGpuData()), Pointer.to(vb.getGpuData()), Pointer.to(new float[]{beta1}), Pointer.to(new float[]{beta2}), Pointer.to(new float[]{lr}), Pointer.to(new float[]{0.0f}), Pointer.to(new int[]{diffB.dataLength}), Pointer.to(new int[]{batchSize}), Pointer.to(new int[]{net.train_time}));
            cuLaunchKernel(function, this.CAFFE_GET_BLOCKS(diffB.dataLength), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelBiasParameters, null // Kernel- and extra parameters
            );
            //	        JCudaDriver.cuCtxSynchronize();
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void updateBeta(Tensor diffB, Tensor bias, Network net, float lr) {
        try {
            /**
             * 设置入参
             * float* weight, const float* diffW, float* mw, float* vw, int n,

             float learnRate, float beta1, float beta2, float beta1_correction, float beta2_correction, float eps, float weight_decay

             */
            kernelParameters = Pointer.to(Pointer.to(bias.getGpuData()), Pointer.to(diffB.getGpuData()), Pointer.to(mb.getGpuData()), Pointer.to(vb.getGpuData()), Pointer.to(new int[]{diffB.dataLength}), Pointer.to(new float[]{lr}), Pointer.to(new float[]{beta1}), Pointer.to(new float[]{beta2}), Pointer.to(new float[]{(float) (1.0f - Math.pow(beta1, net.train_time))}), Pointer.to(new float[]{(float) (1.0f - Math.pow(beta2, net.train_time))}), Pointer.to(new float[]{1e-8f}), Pointer.to(new float[]{weight_decay}));
            cuLaunchKernel(adamw_function, this.CAFFE_GET_BLOCKS(diffB.dataLength), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            );
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    //	public void updateBeta(Tensor diffB,Tensor bias,Network net,float lr) {
    //
    //		try {
    //
    //	        /**
    //	         * 设置入参
    //	         * float *diffW, float *weight,float *mw,float *vw,float beta1,float beta2,float learnRate, int n
    //	         */
    //			kernelBiasParameters = Pointer.to(
    //					Pointer.to(diffB.getGpuData()),
    //	        		Pointer.to(bias.getGpuData()),
    //	                Pointer.to(mb.getGpuData()),
    //	                Pointer.to(vb.getGpuData()),
    //	                Pointer.to(new float[]{beta1}),
    //	                Pointer.to(new float[]{beta2}),
    //	                Pointer.to(new float[]{lr}),
    //	                Pointer.to(new float[]{weight_decay}),
    //	                Pointer.to(new int[]{diffB.dataLength}),
    //	                Pointer.to(new int[]{net.number}),
    //	                Pointer.to(new int[]{net.train_time})
    //	            );
    //
    //			cuLaunchKernel(bn_function,
    //		            this.CAFFE_GET_BLOCKS(diffB.dataLength),  1, 1,      // Grid dimension
    //		            CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
    //		            0, null,               // Shared memory size and stream
    //		            kernelBiasParameters, null // Kernel- and extra parameters
    //		        );
    //
    ////	        JCudaDriver.cuCtxSynchronize();
    //
    //		} catch (Exception e) {
    //			// TODO: handle exception
    //			e.printStackTrace();
    //		}
    //
    //	}
    public void updateBeta(Tensor diffB, Tensor bias, Network net, float lr, CUstream stream) {
        try {
            /**
             * 设置入参
             * float* weight, const float* diffW, float* mw, float* vw, int n,

             float learnRate, float beta1, float beta2, float beta1_correction, float beta2_correction, float eps, float weight_decay

             */
            kernelParameters = Pointer.to(Pointer.to(bias.getGpuData()), Pointer.to(diffB.getGpuData()), Pointer.to(mb.getGpuData()), Pointer.to(vb.getGpuData()), Pointer.to(new int[]{diffB.dataLength}), Pointer.to(new float[]{lr}), Pointer.to(new float[]{beta1}), Pointer.to(new float[]{beta2}), Pointer.to(new float[]{(float) (1.0f - Math.pow(beta1, net.train_time))}), Pointer.to(new float[]{(float) (1.0f - Math.pow(beta2, net.train_time))}), Pointer.to(new float[]{1e-8f}), Pointer.to(new float[]{weight_decay}));
            cuLaunchKernel(adamw_function, this.CAFFE_GET_BLOCKS(diffB.dataLength), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, stream,               // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            );
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
}

