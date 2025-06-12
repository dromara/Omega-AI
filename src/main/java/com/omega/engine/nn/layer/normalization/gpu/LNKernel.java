package com.omega.engine.nn.layer.normalization.gpu;

import com.omega.common.utils.JsonUtils;
import com.omega.common.utils.RandomUtils;
import com.omega.engine.gpu.BaseKernel;
import com.omega.engine.gpu.CUDAManager;
import com.omega.engine.gpu.CUDAMemoryManager;
import com.omega.engine.gpu.GPUOP;
import com.omega.engine.nn.layer.normalization.BNType;
import com.omega.engine.nn.layer.normalization.LNLayer;
import com.omega.engine.nn.network.Transformer;
import com.omega.engine.tensor.Tensor;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfunction;
import jcuda.driver.JCudaDriver;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaError;

import static jcuda.driver.JCudaDriver.cuLaunchKernel;

/**
 * mean = batch均值    : 1/n∑xi
 * <p>
 * var = batch方差    : 1/n∑(xi - mean)^2
 * <p>
 * std = sqrt(var + eta)
 * <p>
 * xhati = (xi - mean) / std
 * <p>
 * yi = gama * xhati + beta
 * <p>
 * dgama = ∑delta * xhat
 * <p>
 * dbeta = ∑delta
 * <p>
 * dxhati = gama * deltai
 * <p>
 * dxi = 1 / std * (dxhati - mean(dxhat) - xhati * mean(dxhat * xhat))
 */
public class LNKernel extends BaseKernel {
    public BNType bnType = null;
    private int B;
    private int W;
    /**
     * 向前方法
     */
    private CUfunction forward_test_function;
    private CUfunction forward_aten_function;
    private CUfunction mean_var_function;
    private CUfunction fused_params_function;
    private CUfunction forward_fused_function;
    //	private CUfunction inter_grad_function;
    private CUfunction backward_fused_function;
    private CUfunction ln_backward_function;
    private CUfunction forward_llm_function;
    private CUfunction backward_llm_function;
    
    private CUfunction forward_llmc_function;
    /**
     * 反向传播方法
     */
    private CUfunction backward_aten_function;
    private CUfunction backward_aten_gamma_function2;
    private CUfunction backward_ig_function;
    private CUfunction backward_fp_function;
    private CUfunction backward_input_function;
    private CUfunction backward_gamma_function;
    private CUfunction backward_gamma_simple_function;
    private int CAFFE_CUDA_NUM_THREADS = 1024;
    private float eta = 1e-5f;
    private int kCUDABlockReduceNumThreads = 512;
    private int kCUDANumThreads = 256;
    private int kColwiseReduceTileSize = 32;
    /**
     * 前向方法参数
     */
    private Pointer forwardTestParameters;
    private Pointer forwardAtenParameters;
    private Pointer meanVarParameters;
    private Pointer fusedParameters;
    private Pointer forwardFusedParams;
    //	private Pointer interGradParameters;
    private Pointer backwardFusedParameters;
    private Pointer lnBKParameters;
    private Pointer forwardLLMParameters;
    private Pointer backwardLLMParameters;
    /**
     * 反向方法参数
     */
    private Pointer backwardAtenParameters;
    private Pointer backwardAtenGammaParameters2;
    private Pointer backwardIGParameters;
    private Pointer backwardFGParameters;
    private Pointer backwardInputParameters;
    private Pointer backwardGammaParameters;
    private Pointer backwardGammaSampleParameters;
    private CUdeviceptr d_mean;
    private CUdeviceptr d_var;
    private CUdeviceptr scratch;
    private CUdeviceptr d_s;
    private CUdeviceptr d_b;
    private CUdeviceptr d_scale;
    private CUdeviceptr d_bias;
    private Pointer aten_mean;
    private Pointer aten_var;
    private Tensor mean;
    private Tensor simga;
    private Tensor scale;
    private Tensor bias;
    private Tensor ds;
    private Tensor db;
    private Tensor rstd;
    private Tensor g_scale;
    private Tensor X_scale;
    private Tensor ones;

    public LNKernel(int W, BNType bnType, CUDAManager cudaManager) {
        super(cudaManager);
        this.W = W;
        this.bnType = bnType;
        init();
    }

    public static void main(String[] args) {
        int N = 1024;
        int T = 1;
        int W = 128;
        float[] data = RandomUtils.gaussianRandom(N * T * W, 0.5f);
        //    	float[] data = RandomUtils.order(N * T * W, 0.1f, 0.1f);
        //    	float[] gammaData = RandomUtils.order(W, 0.1f, 0.1f);
        //    	float[] betaData = RandomUtils.order(W, 0.1f, 0.0f);
        //    	float[] gammaData = RandomUtils.order(W, 0.1f, 0.1f);
        //    	float[] betaData = RandomUtils.order(W, 0.1f, 0.0f);
        //    	float[] gammaData = MatrixUtils.one(W);
        //    	float[] betaData = new float[W];
        //
        //    	Tensor gamma = new Tensor(1, 1, 1, W, gammaData, true);
        //    	Tensor beta = new Tensor(1, 1, 1, W, betaData, true);
        //
        Tensor input = new Tensor(N * T, 1, 1, W, data, true);
        //
        //    	Tensor output = new Tensor(N, T, 1, W, true);
        //    	Tensor output2 = new Tensor(N, T, 1, W, true);
        //
        //    	Tensor delta = new Tensor(N * T, 1, 1, W, MatrixUtils.order(N * T * W, 0.1f, 0.1f), true);
        Tensor delta = new Tensor(N * T, 1, 1, W, RandomUtils.gaussianRandom(N * T * W, 0.5f), true);
        //
        //    	LNKernel kernel = new LNKernel(W, BNType.fully_bn);
        //
        //    	Tensor diff = new Tensor(N, T, 1, W, true);
        //    	Tensor diff2 = new Tensor(N, T, 1, W, true);
        //
        //    	Tensor dgamma = new Tensor(1, 1, 1, W, true);
        //    	Tensor dbeta = new Tensor(1, 1, 1, W, true);
        //
        //    	Tensor dgamma2 = new Tensor(1, 1, 1, W, true);
        //    	Tensor dbeta2 = new Tensor(1, 1, 1, W, true);
        //
        ////    	output.showShape();
        ////    	output.showDM();
        //    	for(int i = 0;i<10;i++) {
        //
        //        	System.out.println("output:");
        ////        	float[] rn = RandomUtils.gaussianRandom(N * T * W, 0.5f);
        ////        	input.setData(rn);
        //
        ////        	kernel.forward(gamma, beta, input, output);
        ////        	output.showDM(0);
        //        	kernel.forwardAten(gamma, beta, input, output2);
        ////        	kernel.forward_aten(gamma, beta, input, output2);
        //        	output2.showDM(0);
        //
        ////        	dgamma2.showDM(0);
        ////        	dbeta2.showDM(0);
        ////
        ////        	System.err.println("---------------------------");
        //
        ////    		kernel.backward(input, delta, diff, gamma, dgamma, dbeta);
        ////        	diff.showDMByNumber(0);
        //        	kernel.backwardAten(input, delta, diff2, gamma, dgamma2, dbeta2);
        //        	diff2.showDMByNumber(0);
        //
        //////        	kernel.backward_aten(input, delta, diff2, gamma, dgamma2, dbeta2);
        //////        	diff2.showDM(0);
        ////        	dgamma.showDM();
        ////        	dbeta.showDM();
        //
        ////        	kernel.backward3_cpu(input, delta, diff2, gamma, dgamma2, dbeta2);
        ////
        ////        	kernel.backward2(input, delta, diff2, gamma, dgamma2, dbeta2);
        ////        	kernel.backward_apex(input, delta, diff2, gamma, dgamma2, dbeta2);
        ////        	diff2.showDM();
        //        	dgamma2.showDM();
        //        	dbeta2.showDM();
        //
        ////        	float[] rmd = RandomUtils.gaussianRandom(N * T * W, 0.1f);
        ////        	Tensor rm = new Tensor(N, T, 1, W, rmd, true);
        ////        	TensorOP.add(input, rm, input);
        ////        	input.showDM();
        //        	System.out.println("========================");
        //    	}
        //
        //    	float[] data2 = RandomUtils.order(128 * W, 0.1f, 0.1f);
        //
        //    	Tensor input2 = new Tensor(128, 1, 1, W, data2, true);
        //
        //    	Tensor output3 = new Tensor(1, T, 1, W, true);
        //
        //    	kernel.forwardAten(gamma, beta, input2, output3);
        //
        //    	output3.showDM();
        Transformer tf = new Transformer();
        tf.number = N * T;
        LNLayer ln = new LNLayer(tf, true);
        for (int i = 0; i < 10; i++) {
            ln.forward(input);
            ln.getOutput().showDM();
            //    		ln.forward(input2);
            //        	ln.getOutput().showDM();
            ln.back(delta);
            ln.diff.showDM();
        }
        //    	tf.number = 1 * T;
        //    	ln.forward(input2);
        //    	ln.getOutput().showDM();
        //    	Tensor dgamma = new Tensor(1, 1, 1, W, true);
        //    	Tensor dbeta = new Tensor(1, 1, 1, W, true);
        //    	Tensor diff3 = new Tensor(N, T, 1, W, true);
        //    	kernel.backward(input, delta, diff, gamma, dgamma, dbeta);
        //
        //    	diff.showShape();
        //    	diff.showDM();
        //    	dgamma.showShape();
        //    	dgamma.showDM();
        //    	dbeta.showShape();
        //    	dbeta.showDM();
        //    	kernel.backward3(input, delta, diff3, gamma, dgamma, dbeta);
        //
        //
        //    	diff3.showDM();
    }

    private void initKernel() {
        //		if(aten_mean == null || aten_mean.number != B) {
        /**
         * 申请向前传播参数显存

         */
        if (this.aten_mean != null) {
            //				CUDAMemoryManager.free(this.d_mean);
            //				CUDAMemoryManager.free(this.d_var);
            CUDAMemoryManager.free(this.aten_mean);
            CUDAMemoryManager.free(this.aten_var);
        }
        //			this.d_mean = CUDAMemoryManager.getDevice(B);
        //			this.d_var = CUDAMemoryManager.getDevice(B);
        //			this.aten_mean = Tensor.createTensor(this.aten_mean, B, 1, 1, 1, true);
        //			this.aten_var = Tensor.createTensor(this.aten_var, B, 1, 1, 1, true);
        this.aten_mean = CUDAMemoryManager.getPointer(B);
        this.aten_var = CUDAMemoryManager.getPointer(B);
        this.scratch = CUDAMemoryManager.getDevice(W * 2 + 1);
        //			this.d_s = CUDAMemoryManager.getDevice(B);
        //			this.d_b = CUDAMemoryManager.getDevice(B);
        //			this.d_scale = CUDAMemoryManager.getDevice(B);
        //			this.d_bias = CUDAMemoryManager.getDevice(B);
        //			this.mean = new Tensor(B, 1, 1, 1, true);
        //			this.simga = new Tensor(B, 1, 1, 1, true);
        //			this.rstd = new Tensor(B, 1, 1, 1, true);
        //			this.scale = new Tensor(B, 1, 1, 1, true);
        //			this.bias = new Tensor(B, 1, 1, 1, true);
        //			this.ds = new Tensor(B, 1, 1, 1, true);
        //			this.db = new Tensor(B, 1, 1, 1, true);
        //			this.g_scale = new Tensor(B, 1, 1, W, true);
        //			this.X_scale = new Tensor(B, 1, 1, 1, true);
        //			this.ones = new Tensor(B, 1, 1, 1, MatrixUtils.one(B), true);
        //		}
    }

    public void initFunction() {
        try {
            //			if(forward_test_function == null) {
            //				forward_test_function = CUDAModules.getLocalFunctionByModule("LNKernel3.cu", "LayerNormFusedForwardKernel");
            //			}
            //
            //			if(backward_ig_function == null) {
            //				backward_ig_function = CUDAModules.getLocalFunctionByModule("LNKernel3.cu", "ComputeInternalGradientsCUDAKernel");
            //			}
            //
            //			if(backward_fp_function == null) {
            //				backward_fp_function = CUDAModules.getLocalFunctionByModule("LNKernel3.cu", "ComputeGradientFusedParamsCUDAKernel");
            //			}
            //
            //			if(backward_input_function == null) {
            //				backward_input_function = CUDAModules.getLocalFunctionByModule("LNKernel3.cu", "layer_norm_grad_input_kernel");
            //			}
            //
            //			if(backward_gamma_function == null) {
            //				backward_gamma_function = CUDAModules.getLocalFunctionByModule("LNKernel3.cu", "GammaBetaBackwardCUDAKernel");
            //			}
            //
            //			if(backward_gamma_simple_function == null) {
            //				backward_gamma_simple_function = CUDAModules.getLocalFunctionByModule("LNKernel3.cu", "GammaBetaBackwardSimpleCUDAKernel");
            //			}
            //
            //			if(mean_var_function == null) {
            //				mean_var_function = CUDAModules.getLocalFunctionByModule("LNKernel_aten.cu", "RowwiseMomentsCUDAKernel");
            //			}
            //
            //			if(fused_params_function == null) {
            //				fused_params_function = CUDAModules.getLocalFunctionByModule("LNKernel_aten.cu", "ComputeSigmaAndFusedParamsCUDAKernel");
            //			}
            //
            //			if(forward_fused_function == null) {
            //				forward_fused_function = CUDAModules.getLocalFunctionByModule("LNKernel_aten.cu", "LayerNormForwardCUDAKernel");
            //			}
            //
            //			if(inter_grad_function == null) {
            //				inter_grad_function = CUDAModules.getLocalFunctionByModule("LNKernel_aten.cu", "ComputeInternalGradientsCUDAKernel");
            //			}
            //
            //			if(backward_fused_function == null) {
            //				backward_fused_function = CUDAModules.getLocalFunctionByModule("LNKernel_aten.cu", "ComputeFusedParamsCUDAKernel");
            //			}
            //
            //			if(ln_backward_function == null) {
            //				ln_backward_function = CUDAModules.getLocalFunctionByModule("LNKernel_aten.cu", "LayerNormBackwardCUDAKernel");
            //			}
            //
            //			if(forward_aten_function == null) {
            //				forward_aten_function = CUDAModules.getLocalFunctionByModule("LNKernel_aten2.cu", "vectorized_layer_norm_kernel");
            //			}
            //
            //			if(backward_aten_function == null) {
            //				backward_aten_function = CUDAModules.getLocalFunctionByModule("LNKernel_aten2.cu", "layer_norm_grad_input_kernel_vectorized");
            //			}
            //
            //			if(backward_aten_function2 == null) {
            //				backward_aten_function2 = CUDAModules.getLocalFunctionByModule("LNKernel_aten2.cu", "layer_norm_grad_input_kernel");
            //			}
            //
            //			if(backward_aten_gamma_function2 == null) {
            //				backward_aten_gamma_function2 = CUDAModules.getLocalFunctionByModule("LNKernel_aten2.cu", "GammaBetaBackwardCUDAKernel");
            //			}
            if (forward_llm_function == null) {
                forward_llm_function = getCudaManager().getLocalFunctionByModule("LNKernel.cu", "layernorm_forward_kernel5");
            }
            if (backward_llm_function == null) {
                backward_llm_function = getCudaManager().getLocalFunctionByModule("LNKernel.cu", "layernorm_backward_kernel7");
            }
            if(forward_llmc_function == null) {
//            	forward_llmc_function = getCudaManager().getLocalFunctionByModule("LNKernel.cu", "layernorm_forward_kernel3_llm");
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

    public boolean checkBatch(Tensor input) {
        int batchSize = 0;
        switch (bnType) {
            case fully_bn:
                batchSize = input.number * input.channel * input.height;
                break;
            case conv_bn:
                batchSize = input.number * input.channel;
                break;
        }
        if (B != batchSize) {
            this.B = batchSize;
            return false;
        }
        return true;
    }

    public void initBackward(Tensor input, Tensor delta, Tensor diff, Tensor gamma, Tensor dgamma, Tensor dbeta) {
        if (backwardInputParameters == null) {
            /**
             * int N, float const *dY, float const *X, float const *gamma, float *ds, float *db

             */
            backwardIGParameters = Pointer.to(Pointer.to(new int[]{W}), Pointer.to(delta.getGpuData()), Pointer.to(input.getGpuData()), Pointer.to(gamma.getGpuData()), Pointer.to(d_s), Pointer.to(d_b));
            /**
             * int M,int N,float const *mean,float const *rstd,float const *ds,float const *db,float *c1,float *c2

             */
            backwardFGParameters = Pointer.to(Pointer.to(new int[]{B}), Pointer.to(new int[]{W}), Pointer.to(d_mean), Pointer.to(d_var), Pointer.to(d_s), Pointer.to(d_b), Pointer.to(d_scale), Pointer.to(d_bias));
            /**
             * float const *__restrict__ dY,

             float const *__restrict__ X,

             float const *__restrict__ mean,

             float const *__restrict__ rstd,

             float const *__restrict__ gamma,

             float *dX,

             int const N

             */
            backwardInputParameters = Pointer.to(Pointer.to(delta.getGpuData()), Pointer.to(input.getGpuData()), Pointer.to(d_mean), Pointer.to(d_var), Pointer.to(gamma.getGpuData()), Pointer.to(diff.getGpuData()), Pointer.to(new int[]{W}));
            /**
             * int M,

             int N,

             float const *dY,

             float const *X,

             float const *mean,

             float const *rstd,

             float *dg,

             float *db

             */
            backwardGammaSampleParameters = Pointer.to(Pointer.to(new int[]{B}), Pointer.to(new int[]{W}), Pointer.to(delta.getGpuData()), Pointer.to(input.getGpuData()), Pointer.to(d_mean), Pointer.to(d_var), Pointer.to(dgamma.getGpuData()), Pointer.to(dbeta.getGpuData()));
            /**
             * int M,

             int N,

             float const *dY,

             float const *X,

             float const *mean,

             float const *rstd,

             float *dg,

             float *db

             */
            backwardGammaParameters = Pointer.to(Pointer.to(new int[]{B}), Pointer.to(new int[]{W}), Pointer.to(delta.getGpuData()), Pointer.to(input.getGpuData()), Pointer.to(d_mean), Pointer.to(d_var), Pointer.to(dgamma.getGpuData()), Pointer.to(dbeta.getGpuData()));
        }
    }

    public int CAFFE_GET_BLOCKS(int N) {
        return (N + CAFFE_CUDA_NUM_THREADS - 1) / CAFFE_CUDA_NUM_THREADS;
    }

    public void forward(Tensor gamma, Tensor beta, Tensor input, Tensor output) {
        try {
            boolean check = checkBatch(input);
            if (!check) {
                initKernel();
                /**
                 * int N,

                 float eps,

                 float const *X,

                 float *mean,

                 float *rstd,

                 float const *gamma,

                 float const *beta,

                 float *Y

                 */
                forwardTestParameters = Pointer.to(Pointer.to(new int[]{W}), Pointer.to(new float[]{eta}), Pointer.to(input.getGpuData()), Pointer.to(d_mean), Pointer.to(d_var), Pointer.to(gamma.getGpuData()), Pointer.to(beta.getGpuData()), Pointer.to(output.getGpuData()));
            }
            int[] kernel1_parallelism = new int[]{B, 512};
            int[] kernel2_parallelism = new int[]{B, 256};
            int num_blocks = Math.max(kernel1_parallelism[0], kernel2_parallelism[0]);
            int num_threads = Math.max(kernel1_parallelism[1], kernel2_parallelism[1]);
            cuLaunchKernel(forward_test_function, num_blocks, 1, 1,      // Grid dimension
                    num_threads, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    forwardTestParameters, null // Kernel- and extra parameters
            );
            //			System.err.println("mean2:");
            //			mean2.setGpuData(d_mean);
            //			mean2.showDM(0);
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void forwardAten(Tensor gamma, Tensor beta, Tensor input, Tensor output) {
        try {
            boolean check = checkBatch(input);
            //			input.showShape();
            //			output.showShape();
            if (!check) {
                initKernel();
                //				System.out.println("in");
                Pointer bias = null;
                if (beta != null) {
                    bias = beta.getGpuData();
                }
                /**
                 * const int N,

                 float eps,

                 const  float* __restrict__ X,

                 const  float* gamma,

                 const  float* beta,

                 float* mean,

                 float* rstd,

                 float* Y

                 */
                forwardAtenParameters = Pointer.to(Pointer.to(new int[]{W}), Pointer.to(new float[]{eta}), Pointer.to(input.getGpuData()), Pointer.to(gamma.getGpuData()), Pointer.to(bias), Pointer.to(aten_mean), Pointer.to(aten_var), Pointer.to(output.getGpuData()));
            }
            int warp_size = 32;
            int[] threads = new int[]{warp_size, 256 / warp_size, 1};
            int[] blocks = new int[]{B, 1, 1};
            int nshared = threads[1] > 1 ? threads[1] * 3 / 2 * Sizeof.FLOAT : 0;
            checkCUDA(cuLaunchKernel(forward_aten_function, blocks[0], blocks[1], blocks[2],      // Grid dimension
                    threads[0], threads[1], threads[2],      // Block dimension
                    nshared, null,               // Shared memory size and stream
                    forwardAtenParameters, null // Kernel- and extra parameters
            ));
            //			System.err.println("mean2:");
            //			mean2.setGpuData(d_mean);
            //			mean2.showDM(0);
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void meanVar(Tensor input) {
        try {
            boolean check = checkBatch(input);
            if (!check) {
                initKernel();
                //const int cols, const float* X, float* mean, float* var
                meanVarParameters = Pointer.to(Pointer.to(new int[]{W}), Pointer.to(input.getGpuData()), Pointer.to(mean.getGpuData()), Pointer.to(simga.getGpuData()));
            }
            cuLaunchKernel(mean_var_function, B, 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    meanVarParameters, null // Kernel- and extra parameters
            );
            System.err.println("mean:");
            mean.showDM(0);
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void fusedParams() {
        try {
            if (fusedParameters == null) {
                /**
                 * const int N,

                 const float eps,

                 const float* mean,

                 const float* var,

                 float* sigma,

                 float* scale,

                 float* bias

                 */
                fusedParameters = Pointer.to(Pointer.to(new int[]{B}), Pointer.to(new float[]{eta}), Pointer.to(mean.getGpuData()), Pointer.to(simga.getGpuData()), Pointer.to(simga.getGpuData()), Pointer.to(scale.getGpuData()), Pointer.to(bias.getGpuData()));
            }
            cuLaunchKernel(fused_params_function, this.CAFFE_GET_BLOCKS(B), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    fusedParameters, null // Kernel- and extra parameters
            );
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void forwardFused(Tensor input, Tensor gamma, Tensor beta, Tensor output) {
        try {
            if (forwardFusedParams == null) {
                /**
                 * const int M,

                 const int N,

                 const float* X,

                 const float* scale,

                 const float* bias,

                 const float* gamma,

                 const float* beta,

                 float* Y

                 */
                forwardFusedParams = Pointer.to(Pointer.to(new int[]{B}), Pointer.to(new int[]{W}), Pointer.to(input.getGpuData()), Pointer.to(scale.getGpuData()), Pointer.to(bias.getGpuData()), Pointer.to(gamma.getGpuData()), Pointer.to(beta.getGpuData()), Pointer.to(output.getGpuData()));
            }
            cuLaunchKernel(forward_fused_function, this.CAFFE_GET_BLOCKS(B * W), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    forwardFusedParams, null // Kernel- and extra parameters
            );
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void forward_aten(Tensor gamma, Tensor beta, Tensor input, Tensor output) {
        try {
            meanVar(input);
            fusedParams();
            forwardFused(input, gamma, beta, output);
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void forward_llm(Tensor gamma, Tensor beta, Tensor input, Tensor output) {
        try {
            boolean check = checkBatch(input);
            if (!check) {
                initKernel();
            }
            /**
             * float* __restrict__ out, float* __restrict__ mean, float* __restrict__ rstd,
             const float*  __restrict__ inp, const float*  __restrict__ weight,
             const float* __restrict__ bias, int N, int C
             */
            forwardLLMParameters = Pointer.to(Pointer.to(output.getGpuData()), Pointer.to(aten_mean), Pointer.to(aten_var), Pointer.to(input.getGpuData()), Pointer.to(gamma.getGpuData()), Pointer.to(beta.getGpuData()), Pointer.to(new int[]{B}), Pointer.to(new int[]{W}));
            checkCUDA(cuLaunchKernel(forward_llm_function, B, 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    forwardLLMParameters, null // Kernel- and extra parameters
            ));
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
    
    public void forward_llmc(Tensor gamma, Tensor beta, Tensor input, Tensor output) {
        try {
            boolean check = checkBatch(input);
            if (!check) {
                initKernel();
            }
            /**
             * float* __restrict__ out, float* __restrict__ mean, float* __restrict__ rstd,
               const float*  __restrict__ inp, const float*  __restrict__ weight,
               const float* __restrict__ bias, int N, int C
             */
            forwardLLMParameters = Pointer.to(Pointer.to(output.getGpuData()), Pointer.to(aten_mean), Pointer.to(aten_var), Pointer.to(input.getGpuData()), Pointer.to(gamma.getGpuData()), Pointer.to(beta.getGpuData()), Pointer.to(new int[]{B}), Pointer.to(new int[]{W}));
            checkCUDA(cuLaunchKernel(forward_llm_function, B, 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    forwardLLMParameters, null // Kernel- and extra parameters
            ));
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void backward_llm(Tensor input, Tensor delta, Tensor diff, Tensor gamma, Tensor dgamma, Tensor dbeta) {
        try {
            //			scratch.showDM();
            diff.clearGPU();
            //			dgamma.clearGPU();
            //			dbeta.clearGPU();
            checkCUDA(JCuda.cudaMemset(scratch, 0, (1 + 2 * W) * Sizeof.FLOAT));
            //			if(backwardLLMParameters == null) {
            /**
             * floatX* dinp, floatX* dweight, floatX* dbias, float* scratch,

             const floatX* dout, const floatX* inp, const floatX* weight, const floatX* mean, const floatX* rstd,

             int B, int T, int C

             */
            /**
             * float* dinp, float* dweight, float* dbias,

             const float* dout, const float* inp, const float* weight, const float* mean, const float* rstd,

             int B, int T, int C

             */
            backwardLLMParameters = Pointer.to(Pointer.to(diff.getGpuData()), Pointer.to(dgamma.getGpuData()), Pointer.to(dbeta.getGpuData()), Pointer.to(scratch), Pointer.to(delta.getGpuData()), Pointer.to(input.getGpuData()), Pointer.to(gamma.getGpuData()), Pointer.to(aten_mean), Pointer.to(aten_var), Pointer.to(new int[]{B}), Pointer.to(new int[]{1}), Pointer.to(new int[]{W}));
            //			}
            int grid_size = (1024 / CAFFE_CUDA_NUM_THREADS) * getCudaManager().props.multiProcessorCount;
            int shared_mem_size = (2 * W + 1) * Sizeof.FLOAT;
            //			int grid_size = this.CAFFE_GET_BLOCKS(B);
            cuLaunchKernel(backward_llm_function, grid_size, 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    shared_mem_size, null,               // Shared memory size and stream
                    backwardLLMParameters, null // Kernel- and extra parameters
            );
            //			diff.showDMByNumber(0);
            //			dgamma.showDM();
            //			dbeta.showDM();
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void backward(Tensor input, Tensor delta, Tensor diff, Tensor gamma, Tensor dgamma, Tensor dbeta) {
        try {
            initBackward(input, delta, diff, gamma, dgamma, dbeta);
            int M = B;
            int N = W;
            cuLaunchKernel(backward_ig_function, M, 1, 1,      // Grid dimension
                    kCUDABlockReduceNumThreads, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    backwardIGParameters, null // Kernel- and extra parameters
            );
            int B2 = (M + kCUDANumThreads - 1) / kCUDANumThreads;
            cuLaunchKernel(backward_fp_function, B2, 1, 1,      // Grid dimension
                    kCUDANumThreads, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    backwardFGParameters, null // Kernel- and extra parameters
            );
            int warp_size = 32;
            int num_threads = 128;
            int nshared = (num_threads / warp_size) * Sizeof.FLOAT;
            cuLaunchKernel(backward_input_function, M, 1, 1,      // Grid dimension
                    num_threads, 1, 1,      // Block dimension
                    nshared, null,               // Shared memory size and stream
                    backwardInputParameters, null // Kernel- and extra parameters
            );
            if (M < 512) {
                int B3 = (N + kCUDANumThreads - 1) / kCUDANumThreads;
                cuLaunchKernel(backward_gamma_simple_function, B3, 1, 1,      // Grid dimension
                        kCUDANumThreads, 1, 1,      // Block dimension
                        0, null,               // Shared memory size and stream
                        backwardGammaSampleParameters, null // Kernel- and extra parameters
                );
            } else {
                int B4 = (N + kColwiseReduceTileSize - 1) / kColwiseReduceTileSize;
                int kThreadX = kColwiseReduceTileSize;
                int kThreadY = kColwiseReduceTileSize / 2;
                cuLaunchKernel(backward_gamma_function, B4, 1, 1,      // Grid dimension
                        kThreadX, kThreadY, 1,      // Block dimension
                        0, null,               // Shared memory size and stream
                        backwardGammaParameters, null // Kernel- and extra parameters
                );
            }
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void backwardAtenGamma(Tensor input, Tensor delta, Tensor dgamma, Tensor dbeta) {
        try {
            if (backwardAtenGammaParameters2 == null) {
                Pointer db = null;
                if (dbeta != null) {
                    db = dbeta.getGpuData();
                }
                /**
                 * int M,

                 int N,

                 const float* dY,

                 const float* X,

                 const float* mean,

                 const float* rstd,

                 float* dg,

                 float* db

                 */
                backwardAtenGammaParameters2 = Pointer.to(Pointer.to(new int[]{B}), Pointer.to(new int[]{W}), Pointer.to(delta.getGpuData()), Pointer.to(input.getGpuData()), Pointer.to(aten_mean), Pointer.to(aten_var), Pointer.to(dgamma.getGpuData()), Pointer.to(db));
            }
            int[] threads = new int[]{16, 32, 1};
            int blocks = (W + threads[0] - 1) / threads[0];
            int shmem_sz = 2 * Sizeof.FLOAT * threads[0] * threads[1];
            cuLaunchKernel(backward_aten_gamma_function2, blocks, 1, 1,      // Grid dimension
                    threads[0], threads[1], threads[2],      // Block dimension
                    shmem_sz, null,               // Shared memory size and stream
                    backwardAtenGammaParameters2, null // Kernel- and extra parameters
            );
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    //
    //	public void interGrad(Tensor delta,Tensor input,Tensor gamma,Tensor dYxX) {
    //
    //		try {
    //
    //			TensorOP.mul(delta, input, dYxX);
    //
    //			if(interGradParameters == null) {
    //				/**
    //				 * const int N,
    //			    const float *const dYxX,
    //			    const float *const dY,
    //			    const float *const gamma,
    //			    float *const ds,
    //			    float *const db
    //				 */
    //				interGradParameters = Pointer.to(
    //						Pointer.to(new int[] {W}),
    //						Pointer.to(dYxX.getGpuData()),
    //						Pointer.to(delta.getGpuData()),
    //						Pointer.to(gamma.getGpuData()),
    //						Pointer.to(ds.getGpuData()),
    //		                Pointer.to(db.getGpuData())
    //		            );
    //			}
    //
    //			cuLaunchKernel(inter_grad_function,
    //					B,  1, 1,      // Grid dimension
    //					CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
    //					0, null,               // Shared memory size and stream
    //					interGradParameters, null // Kernel- and extra parameters
    //		        );
    //
    //		} catch (Exception e) {
    //			// TODO: handle exception
    //			e.printStackTrace();
    //		}
    //
    //	}
    public void backwardAten(Tensor input, Tensor delta, Tensor diff, Tensor gamma, Tensor dgamma, Tensor dbeta) {
        try {
            backwardAtenGamma(input, delta, dgamma, dbeta);
            //			dgamma.showDM();
            //			diff.valueGPU(0);
            if (backwardAtenParameters == null) {
                //				Tensor test = new Tensor(1, 1, 1, B, true);
                //				test.setGpuData(aten_var);
                //				PrintUtils.printImage(test);
                /**
                 *const float* __restrict__ dY,

                 const float* __restrict__ X,

                 const float* __restrict__ mean,

                 const float* __restrict__ rstd,

                 const float* __restrict__ gamma,

                 float* dX,

                 const int N

                 */
                backwardAtenParameters = Pointer.to(Pointer.to(delta.getGpuData()), Pointer.to(input.getGpuData()), Pointer.to(aten_mean), Pointer.to(aten_var), Pointer.to(gamma.getGpuData()), Pointer.to(diff.getGpuData()), Pointer.to(new int[]{W}));
            }
            int[] blocks = new int[]{B, 1, 1};
            int nshared = (256 / 32) * Sizeof.FLOAT;
            cuLaunchKernel(backward_aten_function, blocks[0], blocks[1], blocks[2],      // Grid dimension
                    256, 1, 1,      // Block dimension
                    nshared, null,               // Shared memory size and stream
                    backwardAtenParameters, null // Kernel- and extra parameters
            );
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
        //		try {
        //
        //			/**
        //			 *const float* __restrict__ dY,
        //			  const float* __restrict__ X,
        //			  const float* __restrict__ mean,
        //			  const float* __restrict__ rstd,
        //			  const float* __restrict__ gamma,
        //			  float*  dX,
        //			  const int N
        //			 */
        //			backwardAtenParameters2 = Pointer.to(
        //					Pointer.to(delta.getGpuData()),
        //					Pointer.to(input.getGpuData()),
        //					Pointer.to(aten_mean),
        //					Pointer.to(aten_var),
        //					Pointer.to(gamma.getGpuData()),
        //					Pointer.to(diff.getGpuData()),
        //					Pointer.to(new int[] {W})
        //	            );
        //
        //		    int[] blocks = new int[] {B, 1, 1};
        //		    int nshared = (256 / 32) * Sizeof.FLOAT;
        //
        //			cuLaunchKernel(backward_aten_function2,
        //					blocks[0],  blocks[1], blocks[2],      // Grid dimension
        //					256, 1, 1,      // Block dimension
        //					nshared, null,               // Shared memory size and stream
        //		    		backwardAtenParameters2, null // Kernel- and extra parameters
        //		        );
        //
        //		} catch (Exception e) {
        //			// TODO: handle exception
        //			e.printStackTrace();
        //		}
    }

    public void backwardFusedParams() {
        try {
            if (backwardFusedParameters == null) {
                /**
                 * const int M,

                 const int N,

                 const float* mean,

                 const float* sigma,

                 const float* ds,

                 const float* db,

                 float* rstd,

                 float* X_scale,

                 float* bias,

                 float* g_scale

                 */
                backwardFusedParameters = Pointer.to(Pointer.to(new int[]{B}), Pointer.to(new int[]{W}), Pointer.to(mean.getGpuData()), Pointer.to(simga.getGpuData()), Pointer.to(ds.getGpuData()), Pointer.to(db.getGpuData()), Pointer.to(rstd.getGpuData()), Pointer.to(X_scale.getGpuData()), Pointer.to(bias.getGpuData()), Pointer.to(g_scale.getGpuData()));
            }
            cuLaunchKernel(backward_fused_function, this.CAFFE_GET_BLOCKS(B), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    backwardFusedParameters, null // Kernel- and extra parameters
            );
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void gammaBetaBackward(Tensor dYxX, Tensor delta, Tensor dgamma, Tensor dbeta) {
        try {
            GPUOP.getInstance().gemv(0, B, W, dYxX, rstd, dgamma, 1.0f, 0.0f);
            GPUOP.getInstance().gemv(0, B, W, delta, g_scale, dgamma, 1.0f, 1.0f);
            GPUOP.getInstance().gemv(0, B, W, delta, ones, dbeta, 1.0f, 0.0f);
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    //	public void backward_aten(Tensor input,Tensor delta,Tensor diff,Tensor gamma,Tensor dgamma,Tensor dbeta) {
    //
    //		interGrad(delta, input, gamma, diff);
    //
    //		backwardFusedParams();
    //
    //		gammaBetaBackward(diff, delta, dgamma, dbeta);
    //
    //		lnBackward(delta, input, gamma, diff);
    //
    //	}
    public void lnBackward(Tensor delta, Tensor input, Tensor gamma, Tensor diff) {
        try {
            if (lnBKParameters == null) {
                /**
                 * const int M,

                 const int N,

                 const float* dY,

                 const float* X,

                 const float* gamma,

                 const float* dY_scale,

                 const float* X_scale,

                 const float* bias,

                 float* dX

                 */
                lnBKParameters = Pointer.to(Pointer.to(new int[]{B}), Pointer.to(new int[]{W}), Pointer.to(delta.getGpuData()), Pointer.to(input.getGpuData()), Pointer.to(gamma.getGpuData()), Pointer.to(rstd.getGpuData()), Pointer.to(X_scale.getGpuData()), Pointer.to(bias.getGpuData()), Pointer.to(diff.getGpuData()));
            }
            cuLaunchKernel(ln_backward_function, this.CAFFE_GET_BLOCKS(B), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    lnBKParameters, null // Kernel- and extra parameters
            );
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

