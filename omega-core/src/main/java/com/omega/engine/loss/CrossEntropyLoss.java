package com.omega.engine.loss;

import com.omega.common.utils.JsonUtils;
import com.omega.engine.nn.network.Network;
import com.omega.engine.tensor.Tensor;
import com.omega.engine.tensor.Tensors;

/**
 * Cross Entropy loss function
 *
 * @author Administrator
 * @loss: - ∑ y * ln(f(x))
 * @diff: - ∑ y * (1 / f(x))
 */
public class CrossEntropyLoss extends LossFunction {
    private static CrossEntropyLoss instance;
    public final LossType lossType = LossType.cross_entropy;
    private final float eta = 0.0000000001f;

    public CrossEntropyLoss() {
    }

    public CrossEntropyLoss(Network network) {
        setNet(network);
    }

    public static CrossEntropyLoss operation() {
        if (instance == null) {
            instance = new CrossEntropyLoss();
        }
        return instance;
    }

    public static void main(String[] args) {
        float[] x = new float[]{0.2f, 0.3f, 0.5f, 0.1f, 0.1f, 0.8f, 0.3f, 0.1f, 0.6f, 0.9f, 0.01f, 0.09f};
        Tensor xt = Tensors.tensor(4, 1, 1, 3, x);
        float[] label = new float[]{0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1};
        Tensor labelt = Tensors.tensor(4, 1, 1, 3, label);
        float error = CrossEntropyLoss.operation().gradientCheck(xt, labelt);
        System.out.println("error:" + error);
    }

    @Override
    public LossType getLossType() {
        // TODO Auto-generated method stub
        return LossType.cross_entropy;
    }

    @Override
    public Tensor loss(Tensor x, Tensor label) {
        // TODO Auto-generated method stub
        Tensor temp = Tensors.tensor(x.getShape()[0], x.getShape()[1], x.getShape()[2], x.getShape()[3]);
        System.out.println(JsonUtils.toJson(label.getData()));
        for (int i = 0; i < x.getDataLength(); i++) {
            if (x.getData()[i] == 0) {
                temp.getData()[i] = (float) (label.getData()[i] * Math.log(eta) + (1.0d - label.getData()[i]) * Math.log(1.0d - eta));
            } else {
                temp.getData()[i] = (float) (label.getData()[i] * Math.log(x.getData()[i]) + (1.0d - label.getData()[i]) * Math.log(1.0d - x.getData()[i]));
            }
        }
        //		System.out.println(JsonUtils.toJson(temp.maxtir));
        return temp;
    }

    @Override
    public Tensor diff(Tensor x, Tensor label) {
        // TODO Auto-generated method stub
        Tensor temp = Tensors.tensor(x.getShape()[0], x.getShape()[1], x.getShape()[2], x.getShape()[3]);
        for (int i = 0; i < x.getDataLength(); i++) {
            temp.getData()[i] = label.getData()[i] / x.getData()[i] - (1.0f - label.getData()[i]) / (1.0f - x.getData()[i]);
        }
        return temp;
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
    public Tensor loss(Tensor x, Tensor label, Tensor loss) {
        // TODO Auto-generated method stub
        return null;
    }

    @Override
    public Tensor diff(Tensor x, Tensor label, Tensor diff) {
        // TODO Auto-generated method stub
        return null;
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

