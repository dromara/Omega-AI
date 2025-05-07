package com.omega.engine.loss;

import com.omega.common.data.Tensor;
import com.omega.common.utils.JsonUtils;
import com.omega.common.utils.MatrixOperation;
import com.omega.engine.gpu.CUDAManager;
import com.omega.engine.nn.network.Network;

public abstract class LossFunction {
    public LossType lossType;
    public float eta = 0.00001f;
    public float[] params;
    private Network net;
    private CUDAManager cudaManager;

    public abstract Tensor loss(Tensor x, Tensor label);

    public abstract Tensor loss(Tensor x, Tensor label, int igonre);

    public abstract Tensor loss(Tensor x, Tensor label, Tensor loss);

    public abstract Tensor[] loss(Tensor[] x, Tensor label);

    public abstract Tensor diff(Tensor x, Tensor label);

    public abstract Tensor diff(Tensor x, Tensor label, int igonre);

    public abstract Tensor diff(Tensor x, Tensor label, int igonre, int count);

    public abstract Tensor diff(Tensor x, Tensor label, Tensor diff);

    public abstract Tensor[] diff(Tensor[] x, Tensor label);

    public abstract LossType getLossType();

    public float gradientCheck(Tensor x, Tensor label) {
        Tensor f1 = this.loss(new Tensor(x.number, x.channel, x.height, x.width, MatrixOperation.add(x.data, eta), true), label);
        Tensor f2 = this.loss(new Tensor(x.number, x.channel, x.height, x.width, MatrixOperation.subtraction(x.data, eta), true), label);
        Tensor diff = this.diff(x, label);
        float[] temp = MatrixOperation.subtraction(f1.data, f2.data);
        temp = MatrixOperation.division(temp, 2 * eta);
        System.out.println("diff:" + JsonUtils.toJson(diff.syncHost()));
        System.out.println("gradientCheck:" + JsonUtils.toJson(temp));
        float[] error = MatrixOperation.subtraction(diff.data, temp);
        return MatrixOperation.sum(error);
    }

    public Network getNet() {
        return net;
    }

    public void setNet(Network net) {
        this.net = net;
    }

    public CUDAManager getCudaManager() {
        return cudaManager;
    }

    public void setCudaManager(CUDAManager cudaManager) {
        this.cudaManager = cudaManager;
    }
}

