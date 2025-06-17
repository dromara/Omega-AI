package com.omega.engine.nn.layer.gpu;

import com.omega.engine.gpu.BaseKernel;
import com.omega.engine.gpu.CUDAManager;
import com.omega.engine.tensor.Tensor;

public abstract class PoolingBaseKernel extends BaseKernel {
    public PoolingBaseKernel(CUDAManager cudaManager) {
        super(cudaManager);
        // TODO Auto-generated constructor stub
    }

    public abstract void forward(Tensor input, Tensor output);

    public abstract void backward(Tensor input, Tensor output, Tensor delta, Tensor diff);
}

