package com.omega.engine.loss;

import com.omega.common.tensor.Tensor;
import com.omega.engine.ad.Graph;
import com.omega.engine.ad.op.TensorOP;
import com.omega.engine.gpu.CUDAManager;
import com.omega.engine.nn.network.Network;

public class MultiLabelSoftMargin extends LossFunction {
    private static MultiLabelSoftMargin instance;
    public final LossType lossType = LossType.multiLabel_soft_margin;
    private Graph g;

    public MultiLabelSoftMargin(Network network) {
        setNet(network);
        g = new Graph(getNet().tensorOP);
    }

    public MultiLabelSoftMargin(CUDAManager cudaManager) {
        g = new Graph(new TensorOP(cudaManager));
    }

    public static MultiLabelSoftMargin operation(CUDAManager cudaManager) {
        if (instance == null) {
            instance = new MultiLabelSoftMargin(cudaManager);
        }
        return instance;
    }

    public static Tensor sigmoid(Tensor x) {
        return x.mul(-1).exp().add(1).scalarDiv(1);
    }

    public void initGraph(Tensor x, Tensor label) {
        if (x.getG() == null) {
            x.setG(g);
        }
        if (label.getG() == null) {
            label.setG(g);
        }
    }

    @Override
    public Tensor loss(Tensor x, Tensor label) {
        // TODO Auto-generated method stub
        initGraph(x, label);
        x.getG().start();
        x.setRequiresGrad(true);
        int C = x.channel * x.height * x.width;
        Tensor x0 = sigmoid(x).log();
        Tensor x1 = sigmoid(x.mul(-1.0f)).log().mul(label.scalarSub(1.0f));
        Tensor loss = label.mul(x0).add(x1).mul(-1.0f);
        loss = loss.sum(1).div(C).sum(0).div(x.number);
        return loss;
    }

    @Override
    public Tensor diff(Tensor x, Tensor label) {
        // TODO Auto-generated method stub
        x.getG().clearGrad();
        x.getG().backward();
        return x.getGrad();
    }

    @Override
    public Tensor[] loss(Tensor[] x, Tensor label) {
        // TODO Auto-generated method stub
        return null;
    }

    @Override
    public Tensor[] diff(Tensor[] x, Tensor label) {
        // TODO Auto-generated method stub
        return null;
    }

    @Override
    public LossType getLossType() {
        // TODO Auto-generated method stub
        return LossType.multiLabel_soft_margin;
    }

    @Override
    public Tensor loss(Tensor x, Tensor label, Tensor loss) {
        // TODO Auto-generated method stub
        initGraph(x, label);
        x.getG().start();
        x.setRequiresGrad(true);
        int C = x.channel * x.height * x.width;
        Tensor x0 = sigmoid(x).log();
        Tensor x1 = sigmoid(x.mul(-1.0f)).log().mul(label.scalarSub(1.0f));
        loss = label.mul(x0).add(x1).mul(-1.0f);
        loss = loss.sum(1).div(C).sum(0).div(x.number);
        return loss;
    }

    @Override
    public Tensor diff(Tensor x, Tensor label, Tensor diff) {
        // TODO Auto-generated method stub
        x.getG().clearGrad();
        x.getG().backward();
        diff = x.getGrad();
        return diff;
    }

    @Override
    public Tensor loss(Tensor x, Tensor label, int igonre) {
        // TODO Auto-generated method stub
        return null;
    }

    @Override
    public Tensor diff(Tensor x, Tensor label, int igonre) {
        // TODO Auto-generated method stub
        return null;
    }

    @Override
    public Tensor diff(Tensor x, Tensor label, int igonre, int count) {
        // TODO Auto-generated method stub
        return null;
    }
}

