package com.omega.engine.nn.layer.gpu;

import com.omega.engine.gpu.BaseKernel;
import com.omega.engine.gpu.CUDAManager;
import com.omega.engine.nn.network.RunModel;
import com.omega.engine.tensor.Tensor;

public abstract class BNBaseKernel extends BaseKernel {
    public Tensor runingMean;
    public Tensor runingVar;

    public BNBaseKernel(CUDAManager cudaManager) {
        super(cudaManager);
        // TODO Auto-generated constructor stub
    }

    public abstract void forward(RunModel RUN_MODEL, Tensor gama, Tensor beta, Tensor input, Tensor output);

    public abstract void backward(Tensor input, Tensor delta, Tensor diff, Tensor gama, Tensor dgama, Tensor dbeta);
}

