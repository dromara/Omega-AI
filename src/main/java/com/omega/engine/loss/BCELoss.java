package com.omega.engine.loss;

import com.omega.engine.ad.Graph;
import com.omega.engine.gpu.CUDAManager;
import com.omega.engine.loss.gpu.BCELossKernel;
import com.omega.engine.nn.network.Network;
import com.omega.engine.tensor.Tensor;
import com.omega.engine.tensor.Tensors;

/**
 * 二分类loss
 *
 * @author Administrator
 */
public class BCELoss extends LossFunction {
    private static BCELoss instance;
    public final LossType lossType = LossType.BCE;
    private BCELossKernel kernel;
    private Tensor loss;
    private Tensor diff;
    private Graph g;

    public BCELoss(CUDAManager cudaManager) {
        kernel = new BCELossKernel(cudaManager);
        g = new Graph(getNet().tensorOP);
    }

    public BCELoss(Network network) {
        setNet(network);
        kernel = new BCELossKernel(network.cudaManager);
        g = new Graph(getNet().tensorOP);
    }

    public static BCELoss operation(CUDAManager cudaManager) {
        if (instance == null) {
            instance = new BCELoss(cudaManager);
        }
        return instance;
    }

    public static void main(String[] args) {
        CUDAManager cudaManager = new CUDAManager(0);
        float[] x = new float[]{0.99952507f, 0.9999833f, 1.0f, 1.0f, 1.0f, 1.27869424E-20f, 1.0f, 3.8254528E-26f};
        Tensor xt = Tensors.tensor(8, 1, 1, 1, x, true);
        float[] label = new float[]{1, 1, 1, 1, 1, 1, 1, 1, 1};
        Tensor labelt = Tensors.tensor(8, 1, 1, 1, label, true);
        //		Tensor a = sigmoid(xt);
        //		a.showDM();
        Tensor loss = BCELoss.operation(cudaManager).loss(xt, labelt);
        loss.showDM();
        Tensor diff = BCELoss.operation(cudaManager).diff(xt, labelt);
        diff.showDM();
        //		Graph.clearGrad();
        //		Graph.backward();
        //		xt.getGrad().showDM();
        //		float error = BCELoss.operation().gradientCheck(xt,labelt);
        //		System.out.println("error:"+error);
    }

    public void init(Tensor input) {
        if (loss == null || loss.number != input.number) {
            this.loss = new Tensor(input.number, 1, 1, 1, true);
            this.diff = new Tensor(input.number, input.channel, input.height, input.width, true);
        }
    }

    //	@Override
    //	public Tensor loss(Tensor x, Tensor label) {
    //		// TODO Auto-generated method stub
    ////		x.showDM();
    //		initGraph(x, label);
    //		x.setRequiresGrad(true);
    ////		x.getG().start();
    //		Tensor loss1 = label.scalarSub(1.0f).mul(x.scalarSub(1.0f).log());
    //		Tensor loss = loss1.add(label.mul(x.log()));
    //		return loss.sum(0).div(-x.number * x.width);
    //	}
    public void initGraph(Tensor x, Tensor label) {
        if (x.getG() == null) {
            x.setG(g);
        }
        if (label.getG() == null) {
            label.setG(g);
        }
    }

    //	@Override
    //	public Tensor diff(Tensor x, Tensor label) {
    //		// TODO Auto-generated method stub
    //		x.getG().clearGrad();
    //		x.getG().backward();
    //		return x.getGrad();
    //	}
    @Override
    public Tensor loss(Tensor x, Tensor label) {
        // TODO Auto-generated method stub
        init(x);
        kernel.forward(x, label, loss);
        return loss;
    }

    @Override
    public Tensor diff(Tensor x, Tensor label) {
        // TODO Auto-generated method stub
        kernel.backward(x, label, diff);
        return diff;
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
        return LossType.BCE;
    }

    @Override
    public Tensor loss(Tensor x, Tensor label, Tensor loss) {
        // TODO Auto-generated method stub
        init(x);
        kernel.forward(x, label, loss);
        return loss;
    }

    @Override
    public Tensor diff(Tensor x, Tensor label, Tensor diff) {
        // TODO Auto-generated method stub
        kernel.backward(x, label, diff);
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

