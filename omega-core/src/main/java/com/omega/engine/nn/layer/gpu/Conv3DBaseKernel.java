package com.omega.engine.nn.layer.gpu;

import com.omega.engine.gpu.BaseKernel;
import com.omega.engine.gpu.CUDAManager;
import com.omega.engine.tensor.Tensor;

/**
 * ConvBaseKernel
 *
 * @author Administrator
 */
public abstract class Conv3DBaseKernel extends BaseKernel {
    public Conv3DBaseKernel(CUDAManager cudaManager) {
        super(cudaManager);
        // TODO Auto-generated constructor stub
    }

    public abstract void conv(Tensor input, Tensor kernel, Tensor output);

    public abstract void convTranspose(Tensor input, Tensor kernel, Tensor output);

    public abstract void dw(Tensor input, Tensor delta, Tensor diffW);

    public abstract void dx(Tensor delta, Tensor kernel, Tensor diff);
}

