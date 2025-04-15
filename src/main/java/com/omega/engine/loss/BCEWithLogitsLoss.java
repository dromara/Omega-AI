package com.omega.engine.loss;

import com.omega.common.config.Tensor;
import com.omega.common.config.Tensors;
import com.omega.engine.loss.gpu.BCEWithLogitsLossKernel;

/**
 * 二分类loss
 *
 * @author Administrator
 */
public class BCEWithLogitsLoss extends LossFunction {
    private static BCEWithLogitsLoss instance;
    public final LossType lossType = LossType.BCEWithLogits;
    private BCEWithLogitsLossKernel kernel;
    private Tensor loss;
    private Tensor diff;

    public BCEWithLogitsLoss() {
        kernel = new BCEWithLogitsLossKernel(getNet().cudaManager);
    }

    public static BCEWithLogitsLoss operation() {
        if (instance == null) {
            instance = new BCEWithLogitsLoss();
        }
        return instance;
    }

    public static void main(String[] args) {
        float[] x = new float[]{0.5f, 0.833f, 1.0f, 1.0f, 1.0f, 1.2E-3f, 1.0f, 3.8E-26f};
        Tensor xt = Tensors.tensor(8, 1, 1, 1, x, true);
        float[] label = new float[]{1, 1, 1, 0, 0, 1, 1, 0};
        Tensor labelt = Tensors.tensor(8, 1, 1, 1, label, true);
        //		Tensor a = sigmoid(xt);
        //		a.showDM();
        Tensor loss = BCEWithLogitsLoss.operation().loss(xt, labelt);
        loss.showDM();
        Tensor diff = BCEWithLogitsLoss.operation().diff(xt, labelt);
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

