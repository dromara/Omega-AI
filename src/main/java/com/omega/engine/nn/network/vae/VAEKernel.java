package com.omega.engine.nn.network.vae;

import com.omega.engine.gpu.BaseKernel;
import com.omega.engine.gpu.CUDAManager;
import com.omega.engine.tensor.Tensor;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUfunction;

import static jcuda.driver.JCudaDriver.cuLaunchKernel;

public class VAEKernel extends BaseKernel {
    private CUfunction function;
    private CUfunction function_back;
    private CUfunction function_back_org;
    private CUfunction kl_loss_function;
    private CUfunction kl_loss_function_back;
    private CUfunction kl_loss_function2;
    private CUfunction kl_loss_function_back2;
    private CUfunction l1_loss_function;
    private CUfunction l1_loss_back_function;
    private CUfunction l1_loss_allBack_function;
    private CUfunction cdist_function;
    private CUfunction cdist2_function;
    private CUfunction argmin_function;
    private CUfunction mean_function;
    private CUfunction mse_loss_function;
    private CUfunction mse_loss_only_c_function;
    private CUfunction mse_loss_back_function;
    private CUfunction mse_sum_loss_function;
    private CUfunction mse_sum_loss_back_function;
    private CUfunction mse_sum_c_loss_function;
    private CUfunction mse_sum_c_loss_back_function;
    private CUfunction ema_count_function;
    private CUfunction move_ema_count_function;
    private CUfunction move_ema_count2_function;
    private CUfunction update_emb_weight_function;
    private int CAFFE_CUDA_NUM_THREADS = 1024;
    private Pointer forwardKernelParameters;
    private Pointer backwardKernelParameters;

    public VAEKernel(CUDAManager cudaManager) {
        super(cudaManager);
        init();
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
                function = getCudaManager().getLocalFunctionByModule("VAE.cu", "reparameterize_forward");
            }
            if (function_back == null) {
                function_back = getCudaManager().getLocalFunctionByModule("VAE.cu", "reparameterize_backward");
            }
            if (function_back_org == null) {
            	function_back_org = getCudaManager().getLocalFunctionByModule("VAE.cu", "reparameterize_backward_org");
            }
            if (kl_loss_function == null) {
                kl_loss_function = getCudaManager().getLocalFunctionByModule("VAE.cu", "kl_loss");
            }
            if (kl_loss_function_back == null) {
                kl_loss_function_back = getCudaManager().getLocalFunctionByModule("VAE.cu", "kl_loss_back");
            }
            if(l1_loss_function == null) {
            	l1_loss_function = getCudaManager().getLocalFunctionByModule("VAE.cu", "l1_loss");
            }
            if(l1_loss_back_function == null) {
            	l1_loss_back_function = getCudaManager().getLocalFunctionByModule("VAE.cu", "l1_loss_back");
            }
            if(l1_loss_allBack_function == null) {
            	l1_loss_allBack_function = getCudaManager().getLocalFunctionByModule("VAE.cu", "l1_loss_allBack");
            }
            if (cdist_function == null) {
                cdist_function = getCudaManager().getLocalFunctionByModule("VAE.cu", "CdistP");
            }
            if (cdist2_function == null) {
                cdist2_function = getCudaManager().getLocalFunctionByModule("VAE.cu", "calcDistKernel");
            }
            if (argmin_function == null) {
                argmin_function = getCudaManager().getLocalFunctionByModule("VAE.cu", "argmin");
            }
            if (mean_function == null) {
                mean_function = getCudaManager().getLocalFunctionByModule("VAE.cu", "mean_kernel");
            }
            if (mse_loss_function == null) {
                mse_loss_function = getCudaManager().getLocalFunctionByModule("VAE.cu", "mse_loss_kernel");
            }
            if (mse_loss_only_c_function == null) {
                mse_loss_only_c_function = getCudaManager().getLocalFunctionByModule("VAE.cu", "mse_loss_kernel_only_c");
            }
            if (mse_loss_back_function == null) {
                mse_loss_back_function = getCudaManager().getLocalFunctionByModule("VAE.cu", "mse_loss_back");
            }
            if (mse_sum_loss_function == null) {
                mse_sum_loss_function = getCudaManager().getLocalFunctionByModule("VAE.cu", "mse_sum_loss_kernel");
            }
            if (mse_sum_loss_back_function == null) {
                mse_sum_loss_back_function = getCudaManager().getLocalFunctionByModule("VAE.cu", "mse_sum_loss_back");
            }
            if (mse_sum_c_loss_function == null) {
                mse_sum_c_loss_function = getCudaManager().getLocalFunctionByModule("VAE.cu", "mse_sum_only_c_loss_kernel");
            }
            if (mse_sum_c_loss_back_function == null) {
                mse_sum_c_loss_back_function = getCudaManager().getLocalFunctionByModule("VAE.cu", "mse_sum_only_c_loss_back");
            }
            if (ema_count_function == null) {
                ema_count_function = getCudaManager().getLocalFunctionByModule("VAE.cu", "ema_count");
            }
            if (move_ema_count_function == null) {
                move_ema_count_function = getCudaManager().getLocalFunctionByModule("VAE.cu", "move_ema_count");
            }
            if (move_ema_count2_function == null) {
                move_ema_count2_function = getCudaManager().getLocalFunctionByModule("VAE.cu", "move_ema_count2");
            }
            if (update_emb_weight_function == null) {
                update_emb_weight_function = getCudaManager().getLocalFunctionByModule("VAE.cu", "update_emb_weight");
            }
            if(kl_loss_function2 == null) {
            	kl_loss_function2 = getCudaManager().getLocalFunctionByModule("VAE.cu", "kl_loss2");
            }
            if(kl_loss_function_back2 == null) {
            	kl_loss_function_back2 = getCudaManager().getLocalFunctionByModule("VAE.cu", "kl_loss_back2");
            }
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public int CAFFE_GET_BLOCKS(int N) {
        return (N + CAFFE_CUDA_NUM_THREADS - 1) / CAFFE_CUDA_NUM_THREADS;
    }

    public void forward(Tensor mu, Tensor logvar, Tensor eps, Tensor output) {
        try {
            /**
             * 设置入参
             * float *mu,float *logvar,float *eps, float *output, int n

             */
            forwardKernelParameters = Pointer.to(Pointer.to(mu.getGpuData()), Pointer.to(logvar.getGpuData()), Pointer.to(eps.getGpuData()), Pointer.to(output.getGpuData()), Pointer.to(new int[]{output.dataLength}));
            this.N = output.number;
            cuLaunchKernel(function, this.CAFFE_GET_BLOCKS(output.dataLength), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    forwardKernelParameters, null // Kernel- and extra parameters
            );
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void backward(Tensor delta, Tensor eps, Tensor logvar, Tensor dmu, Tensor dlogvar) {
        try {
            /**
             * 设置入参
             * float *dmu,float *dlogvar,float *eps,float *logvar, float *delta, int n

             */
            backwardKernelParameters = Pointer.to(Pointer.to(dmu.getGpuData()), Pointer.to(dlogvar.getGpuData()), Pointer.to(eps.getGpuData()), Pointer.to(logvar.getGpuData()), Pointer.to(delta.getGpuData()), Pointer.to(new int[]{delta.dataLength}));
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
    
    public void backward_org(Tensor delta, Tensor eps, Tensor logvar, Tensor dmu, Tensor dlogvar) {
        try {
            /**
             * 设置入参
             * float *dmu,float *dlogvar,float *eps,float *logvar, float *delta, int n

             */
            backwardKernelParameters = Pointer.to(Pointer.to(dmu.getGpuData()), Pointer.to(dlogvar.getGpuData()), Pointer.to(eps.getGpuData()), Pointer.to(logvar.getGpuData()), Pointer.to(delta.getGpuData()), Pointer.to(new int[]{delta.dataLength}));
            cuLaunchKernel(function_back_org, this.CAFFE_GET_BLOCKS(delta.dataLength), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    backwardKernelParameters, null // Kernel- and extra parameters
            );
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void kl(Tensor mu, Tensor logvar, float kl_weight, Tensor output) {
        try {
            /**
             * 设置入参
             * float *mu,float *logvar,float kl_weight, float *klLoss, int n

             */
            forwardKernelParameters = Pointer.to(Pointer.to(mu.getGpuData()), Pointer.to(logvar.getGpuData()), Pointer.to(new float[]{kl_weight}), Pointer.to(output.getGpuData()), Pointer.to(new int[]{output.dataLength}));
            this.N = output.number;
            cuLaunchKernel(kl_loss_function, this.CAFFE_GET_BLOCKS(output.dataLength), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    forwardKernelParameters, null // Kernel- and extra parameters
            );
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void kl_back(Tensor mu, Tensor logvar, float kl_weight, Tensor dmu, Tensor dlogvar) {
        try {
            /**
             * 设置入参
             * float *mu,float *logvar,float kl_weight, float *dmu, float * dlogvar,int batch, int n

             */
            backwardKernelParameters = Pointer.to(Pointer.to(mu.getGpuData()), Pointer.to(logvar.getGpuData()), Pointer.to(new float[]{kl_weight}), Pointer.to(dmu.getGpuData()), Pointer.to(dlogvar.getGpuData()), Pointer.to(new int[]{mu.number}), Pointer.to(new int[]{mu.dataLength}));
            cuLaunchKernel(kl_loss_function_back, this.CAFFE_GET_BLOCKS(mu.dataLength), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    backwardKernelParameters, null // Kernel- and extra parameters
            );
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
    
    public void kl2(Tensor mu, Tensor logvar, float kl_weight, Tensor output) {
        try {
            /**
             * 设置入参
             * float *mu,float *logvar,float kl_weight, float *klLoss, int n

             */
            forwardKernelParameters = Pointer.to(Pointer.to(mu.getGpuData()), Pointer.to(logvar.getGpuData()), Pointer.to(new float[]{kl_weight}), Pointer.to(output.getGpuData()), Pointer.to(new int[]{output.dataLength}));
            this.N = output.number;
            cuLaunchKernel(kl_loss_function2, this.CAFFE_GET_BLOCKS(output.dataLength), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    forwardKernelParameters, null // Kernel- and extra parameters
            );
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void kl_back2(Tensor mu, Tensor logvar, float kl_weight, Tensor dmu, Tensor dlogvar) {
        try {
            /**
             * 设置入参
             * float *mu,float *logvar,float kl_weight, float *dmu, float * dlogvar,int batch, int n

             */
            backwardKernelParameters = Pointer.to(Pointer.to(mu.getGpuData()), Pointer.to(logvar.getGpuData()), Pointer.to(new float[]{kl_weight}), Pointer.to(dmu.getGpuData()), Pointer.to(dlogvar.getGpuData()), Pointer.to(new int[]{mu.number}), Pointer.to(new int[]{mu.dataLength}));
            cuLaunchKernel(kl_loss_function_back2, this.CAFFE_GET_BLOCKS(mu.dataLength), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    backwardKernelParameters, null // Kernel- and extra parameters
            );
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
    
    public void l1_loss(Tensor x, Tensor y, Tensor output) {
        try {
            /**
             * 设置入参
             * const float* output, const float* target, float* loss, int num_elem
             */
            backwardKernelParameters = Pointer.to(Pointer.to(x.getGpuData()), Pointer.to(y.getGpuData()), Pointer.to(output.getGpuData()), Pointer.to(new int[]{x.dataLength}));
            cuLaunchKernel(l1_loss_function, this.CAFFE_GET_BLOCKS(x.dataLength), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    backwardKernelParameters, null // Kernel- and extra parameters
            );
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
    
    public void l1_loss_back(Tensor x, Tensor y, Tensor diff, int bs) {
        try {
            /**
             * 设置入参
             * const float* output, const float* target, float* diff, int num_elem, int bs
             */
            backwardKernelParameters = Pointer.to(Pointer.to(x.getGpuData()), Pointer.to(y.getGpuData()), Pointer.to(diff.getGpuData()), Pointer.to(new int[]{x.dataLength}), Pointer.to(new int[]{bs}));
            cuLaunchKernel(l1_loss_back_function, this.CAFFE_GET_BLOCKS(x.dataLength), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    backwardKernelParameters, null // Kernel- and extra parameters
            );
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
    
    public void l1_loss_allBack(Tensor x, Tensor y, Tensor diffX, Tensor diffY, int bs) {
        try {
            /**
             * 设置入参
             * const float* output, const float* target, float* diff, int num_elem, int bs
             */
            backwardKernelParameters = Pointer.to(Pointer.to(x.getGpuData()), Pointer.to(y.getGpuData()), Pointer.to(diffX.getGpuData()), Pointer.to(diffY.getGpuData()), Pointer.to(new int[]{x.dataLength}), Pointer.to(new int[]{bs}));
            cuLaunchKernel(l1_loss_allBack_function, this.CAFFE_GET_BLOCKS(x.dataLength), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    backwardKernelParameters, null // Kernel- and extra parameters
            );
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void cdistP(Tensor x, Tensor y, Tensor output, double p) {
        try {
            int r_size = x.number * y.height;
            int l1_size = x.number * x.width;
            int l2_size = y.height * y.width;
            /**
             * 设置入参
             * float *x1, float *x2, float *result, double p, const int64_t r2, const int64_t m, const int64_t r_size, const int64_t l1_size, const int64_t l2_size

             */
            forwardKernelParameters = Pointer.to(Pointer.to(x.getGpuData()), Pointer.to(y.getGpuData()), Pointer.to(output.getGpuData()), Pointer.to(new double[]{p}), Pointer.to(new int[]{y.height}), Pointer.to(new int[]{x.width}), Pointer.to(new int[]{r_size}), Pointer.to(new int[]{l1_size}), Pointer.to(new int[]{l2_size}));
            cuLaunchKernel(cdist_function, output.getDataLength(), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    forwardKernelParameters, null // Kernel- and extra parameters
            );
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void cdist2(Tensor x, Tensor y, Tensor output) {
        try {
            /**
             * 设置入参
             * float* _res, const float * _A, const float * _B,uint _Arows, uint _Brows, uint _dim

             */
            forwardKernelParameters = Pointer.to(Pointer.to(output.getGpuData()), Pointer.to(x.getGpuData()), Pointer.to(y.getGpuData()), Pointer.to(new int[]{x.number}), Pointer.to(new int[]{y.height}), Pointer.to(new int[]{x.width}));
            int blockSize = 16;
            int shmSize = (blockSize * blockSize * 3) * Sizeof.FLOAT;
            cuLaunchKernel(cdist2_function, output.getDataLength(), 1, 1,      // Grid dimension
                    blockSize, blockSize, 1,      // Block dimension
                    shmSize, null,               // Shared memory size and stream
                    forwardKernelParameters, null // Kernel- and extra parameters
            );
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void argmin(Tensor x, Tensor y) {
        try {
            /**
             * 设置入参
             * float *x,float *y,int batch, int n

             */
            forwardKernelParameters = Pointer.to(Pointer.to(x.getGpuData()), Pointer.to(y.getGpuData()), Pointer.to(new int[]{x.number}), Pointer.to(new int[]{x.width}));
            cuLaunchKernel(argmin_function, this.CAFFE_GET_BLOCKS(x.number), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    forwardKernelParameters, null // Kernel- and extra parameters
            );
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void mean(Tensor x, Tensor y) {
        try {
            /**
             * 设置入参
             * float *x,float *y,int batch

             */
            forwardKernelParameters = Pointer.to(Pointer.to(x.getGpuData()), Pointer.to(y.getGpuData()), Pointer.to(new int[]{x.number}));
            cuLaunchKernel(mean_function, this.CAFFE_GET_BLOCKS(x.number), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    forwardKernelParameters, null // Kernel- and extra parameters
            );
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void MSE(Tensor x, Tensor y, Tensor loss, float beta) {
        try {
            /**
             * 设置入参
             * const float* output, const float* target, float* loss, float beta, int num_elem

             */
            forwardKernelParameters = Pointer.to(Pointer.to(x.getGpuData()), Pointer.to(y.getGpuData()), Pointer.to(loss.getGpuData()), Pointer.to(new float[]{beta}), Pointer.to(new int[]{x.dataLength}));
            cuLaunchKernel(mse_loss_function, this.CAFFE_GET_BLOCKS(x.dataLength), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    forwardKernelParameters, null // Kernel- and extra parameters
            );
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void MSE_C(Tensor x, Tensor y, Tensor loss, float beta) {
        try {
            /**
             * 设置入参
             * const float* output, const float* target, float* loss, float beta, int num_elem

             */
            forwardKernelParameters = Pointer.to(Pointer.to(x.getGpuData()), Pointer.to(y.getGpuData()), Pointer.to(loss.getGpuData()), Pointer.to(new float[]{beta}), Pointer.to(new int[]{x.dataLength}));
            cuLaunchKernel(mse_loss_only_c_function, this.CAFFE_GET_BLOCKS(x.dataLength), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    forwardKernelParameters, null // Kernel- and extra parameters
            );
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void MSE_BACK(Tensor x, Tensor y, Tensor diffX, Tensor diffY, float beta) {
        try {
            /**
             * 设置入参
             * float *x, float *y, float beta, float *diffX,float *diffY, int n, int batch

             */
            backwardKernelParameters = Pointer.to(Pointer.to(x.getGpuData()), Pointer.to(y.getGpuData()), Pointer.to(new float[]{beta}), Pointer.to(diffX.getGpuData()), Pointer.to(diffY.getGpuData()), Pointer.to(new int[]{x.dataLength}));
            cuLaunchKernel(mse_loss_back_function, this.CAFFE_GET_BLOCKS(x.dataLength), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    backwardKernelParameters, null // Kernel- and extra parameters
            );
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void MSE_SUM(Tensor x, Tensor y, Tensor loss, float beta) {
        try {
            /**
             * 设置入参
             * const float* output, const float* target, float* loss, float beta, int num_elem

             */
            forwardKernelParameters = Pointer.to(Pointer.to(x.getGpuData()), Pointer.to(y.getGpuData()), Pointer.to(loss.getGpuData()), Pointer.to(new float[]{beta}), Pointer.to(new int[]{x.dataLength}));
            cuLaunchKernel(mse_sum_loss_function, this.CAFFE_GET_BLOCKS(x.dataLength), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    forwardKernelParameters, null // Kernel- and extra parameters
            );
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void MSE_C_SUM(Tensor x, Tensor y, Tensor loss, float beta) {
        try {
            /**
             * 设置入参
             * const float* output, const float* target, float* loss, float beta, int num_elem

             */
            forwardKernelParameters = Pointer.to(Pointer.to(x.getGpuData()), Pointer.to(y.getGpuData()), Pointer.to(loss.getGpuData()), Pointer.to(new float[]{beta}), Pointer.to(new int[]{x.dataLength}));
            cuLaunchKernel(mse_sum_c_loss_function, this.CAFFE_GET_BLOCKS(x.dataLength), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    forwardKernelParameters, null // Kernel- and extra parameters
            );
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void MSE_SUM_BACK(Tensor x, Tensor y, Tensor diffX, Tensor diffY, float beta) {
        try {
            /**
             * 设置入参
             * float *x, float *y, float beta, float *diffX,float *diffY, int n, int batch

             */
            backwardKernelParameters = Pointer.to(Pointer.to(x.getGpuData()), Pointer.to(y.getGpuData()), Pointer.to(new float[]{beta}), Pointer.to(diffX.getGpuData()), Pointer.to(diffY.getGpuData()), Pointer.to(new int[]{x.dataLength}));
            cuLaunchKernel(mse_sum_loss_back_function, this.CAFFE_GET_BLOCKS(x.dataLength), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    backwardKernelParameters, null // Kernel- and extra parameters
            );
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void MSE_SUM_C_BACK(Tensor x, Tensor y, Tensor diffY, float beta) {
        try {
            /**
             * 设置入参
             * float *x, float *y, float beta,float *diffY, int n, int batch

             */
            backwardKernelParameters = Pointer.to(Pointer.to(x.getGpuData()), Pointer.to(y.getGpuData()), Pointer.to(new float[]{beta}), Pointer.to(diffY.getGpuData()), Pointer.to(new int[]{x.dataLength}));
            cuLaunchKernel(mse_sum_c_loss_back_function, this.CAFFE_GET_BLOCKS(x.dataLength), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    backwardKernelParameters, null // Kernel- and extra parameters
            );
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void ema_count(Tensor x, Tensor y) {
        try {
            /**
             * 设置入参
             * int n,float *x, float *y

             */
            forwardKernelParameters = Pointer.to(Pointer.to(new int[]{x.dataLength}), Pointer.to(x.getGpuData()), Pointer.to(y.getGpuData()));
            cuLaunchKernel(ema_count_function, 1, 1, 1,      // Grid dimension
                    1, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    forwardKernelParameters, null // Kernel- and extra parameters
            );
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void move_ema_count(Tensor x, Tensor y, float decay) {
        try {
            /**
             * 设置入参
             * float *x,float *y, float decay, int n

             */
            forwardKernelParameters = Pointer.to(Pointer.to(x.getGpuData()), Pointer.to(y.getGpuData()), Pointer.to(new float[]{decay}), Pointer.to(new int[]{x.dataLength}));
            cuLaunchKernel(move_ema_count_function, this.CAFFE_GET_BLOCKS(x.dataLength), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    forwardKernelParameters, null // Kernel- and extra parameters
            );
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void move_ema_count2(Tensor x, Tensor y, float eps, int D) {
        try {
            /**
             * 设置入参
             * float *x, float *sumec,float eps, int D, int n

             */
            forwardKernelParameters = Pointer.to(Pointer.to(x.getGpuData()), Pointer.to(y.getGpuData()), Pointer.to(new float[]{eps}), Pointer.to(new int[]{D}), Pointer.to(new int[]{x.dataLength}));
            cuLaunchKernel(move_ema_count2_function, this.CAFFE_GET_BLOCKS(x.dataLength), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    forwardKernelParameters, null // Kernel- and extra parameters
            );
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void update_emb_weight(Tensor dw, Tensor weight, Tensor emb_weight, Tensor ema_count, float decay) {
        try {
            /**
             * 设置入参
             * float *dw,float *weight, float *emb_weight,float *ema_count, float decay,int batch, int n

             */
            forwardKernelParameters = Pointer.to(Pointer.to(dw.getGpuData()), Pointer.to(weight.getGpuData()), Pointer.to(emb_weight.getGpuData()), Pointer.to(ema_count.getGpuData()), Pointer.to(new float[]{decay}), Pointer.to(new int[]{dw.height}), Pointer.to(new int[]{dw.width}));
            cuLaunchKernel(update_emb_weight_function, this.CAFFE_GET_BLOCKS(dw.height), 1, 1,      // Grid dimension
                    CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    forwardKernelParameters, null // Kernel- and extra parameters
            );
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
    //	public static void main(String args[]){
    //	    	int N = 2;
    //	    	int C = 4;
    //	    	int H = 3;
    //	    	int W = 3;
    //
    //	    	int K = 6;
    //
    //	    	float[] x1 = RandomUtils.order(N * C * H * W, 1f, 1f);
    //
    //	    	float[] y1 = RandomUtils.order(K * C, 1f, 1f);
    //
    //	    	Tensor x = new Tensor(N, C, H, W, x1, true);
    //
    //	    	Tensor y = new Tensor(1, 1, K, C, y1, true);
    //
    //	    	Tensor z = new Tensor(N, H, W, C, true);
    //
    //	    	Tensor idx = new Tensor(N * H * W, 1, 1, 1, true);
    //
    //	    	TensorOP.permute(x, z, new int[] {0, 2, 3, 1});
    //
    //	    	z.showDM();
    //
    //	    	z = z.view(N * H * W, 1, 1, C);
    //
    ////	    	Tensor output = new Tensor(N * H * W, 1, 1, K, true);
    //
    //	    	VAEKernel k = new VAEKernel();
    //
    ////	    	k.cdistP(z, y, output, 2);
    //
    ////	    	output.showDM();
    //
    ////	    	output.showShape();
    //
    //	    	Tensor zc = new Tensor(z.number, 1, 1, 1, true);
    //	    	Tensor ec = new Tensor(K, 1, 1, 1, true);
    //	    	Tensor ie = new Tensor(z.number, 1, 1, K, true);
    //
    //	    	TensorOP.sum_pow(z, zc, 2, 1);
    //    		TensorOP.sum_pow(y.view(K, 1, 1, C), ec, 2, 1);
    //
    //    		TensorOP.broadcast(zc, ie, 1);
    //
    //    		ie.showDM();
    //
    //    		TensorOP.broadcast_row(ec, ie);
    //
    //    		ie.showDM();
    //
    //    		GPUOP.getInstance().multiplyFloat(z.number, y.number, y.width, z.getGpuData(), y.getGpuData(), ie.getGpuData(),
    //    				cublasOperation.CUBLAS_OP_N, cublasOperation.CUBLAS_OP_T, -2.0f, 1.0f);
    //
    //    		ie.showDM();
    //
    //	    	k.argmin(ie, idx);
    //
    //	    	idx.showDM();
    //
    //
    //	    	Transformer network = new Transformer();
    //
    //	    	EmbeddingIDLayer emd = new EmbeddingIDLayer(K, C, network);
    //	    	emd.weight = new Tensor(1, 1, K, C, MatrixUtils.order(K * C, 1, 1), true);
    //	    	emd.forward(idx);
    //
    //	    	emd.getOutput().showDM();
    //
    //	    	Tensor loss = new Tensor(1, 1, 1, 1, true);
    //
    //	    	k.MSE(emd.getOutput(), z, loss, 0.2f);
    //
    //	    	loss.showDM();
    //
    //	    	Tensor diffX = new Tensor(emd.getOutput().number, emd.getOutput().channel, emd.getOutput().height, emd.getOutput().width, true);
    //
    //	    	Tensor diffY = new Tensor(emd.getOutput().number, emd.getOutput().channel, emd.getOutput().height, emd.getOutput().width, true);
    //
    //	    	k.MSE_BACK(emd.getOutput(), z, diffX, diffY, 0.2f);
    //
    //	    	diffX.showDM();
    //
    //	    	diffY.showDM();
    //
    //	    	emd.back(diffX);
    //
    //	    	emd.diffW.showDM();
    //
    //			CUDAMemoryManager.free();
    //
    //	    }
}

