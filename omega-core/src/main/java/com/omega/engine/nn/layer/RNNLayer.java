package com.omega.engine.nn.layer;

import com.omega.common.utils.RandomUtils;
import com.omega.engine.active.ActiveType;
import com.omega.engine.nn.layer.active.*;
import com.omega.engine.nn.network.Network;
import com.omega.engine.nn.network.RNN;
import com.omega.engine.tensor.Tensor;

/**
 * Recurrent Layer
 *
 * @author Administrator
 */
public class RNNLayer extends Layer {
    private int time = 0;
    private int inputSize;
    private int hiddenSize;
    private boolean bias = false;
    private ActiveType activeType;
    private FullyLayer inputLayer;
    private FullyLayer selfLayer;
    private ActiveFunctionLayer outputActive;
    private Tensor h;
    private Tensor h_0;

    public RNNLayer(int inputNum, int hiddenNum, int time, ActiveType activeType, boolean bias) {
        this.time = time;
        this.inputSize = inputNum;
        this.hiddenSize = hiddenNum;
        this.activeType = activeType;
        this.bias = bias;
        this.initLayers();
    }

    public RNNLayer(int inputNum, int hiddenNum, int time, ActiveType activeType, boolean bias, Network network) {
        this.network = network;
        this.time = time;
        this.inputSize = inputNum;
        this.hiddenSize = hiddenNum;
        this.activeType = activeType;
        this.bias = bias;
        this.initLayers();
    }

    public void initLayers() {
        //		float stdv = (float) (1.0f / Math.sqrt(hiddenSize));
        //		float stdv = (float) (2.0f / Math.sqrt(hiddenSize + inputSize));
        this.inputLayer = new FullyLayer(inputSize, hiddenSize, bias, this.network);
        //		this.inputLayer.weight = new Tensor(1, 1, inputSize, hiddenSize, RandomUtils.uniform(this.inputSize * this.hiddenSize, 0, stdv), true);
        //		this.inputLayer.bias = new Tensor(1, 1, 1, hiddenSize, RandomUtils.uniform(this.hiddenSize, 0, stdv), true);
        this.inputLayer.weight = new Tensor(1, 1, inputSize, hiddenSize, RandomUtils.uniformFloat(this.inputSize * this.hiddenSize, inputSize), true);
        this.inputLayer.bias = new Tensor(1, 1, 1, hiddenSize, RandomUtils.uniformFloat(this.hiddenSize, hiddenSize), true);
        //		this.inputLayer.weight = new Tensor(1, 1, inputSize, hiddenSize, RandomUtils.order(this.inputSize * this.hiddenSize, 0.1f, 0.0f), true);
        //		this.inputLayer.bias = new Tensor(1, 1, 1, hiddenSize, RandomUtils.val(this.hiddenSize, 0.1f), true);
        this.selfLayer = new FullyLayer(hiddenSize, hiddenSize, bias, this.network);
        //		this.selfLayer.weight = new Tensor(1, 1, hiddenSize, hiddenSize, RandomUtils.uniform(this.hiddenSize * this.hiddenSize, 0, stdv), true);
        //		this.selfLayer.bias = new Tensor(1, 1, 1, hiddenSize, RandomUtils.uniform(this.hiddenSize, 0, stdv), true);
        this.selfLayer.weight = new Tensor(1, 1, hiddenSize, hiddenSize, RandomUtils.uniformFloat(this.hiddenSize * this.hiddenSize, hiddenSize), true);
        this.selfLayer.bias = new Tensor(1, 1, 1, hiddenSize, RandomUtils.uniformFloat(this.hiddenSize, hiddenSize), true);
        //		this.selfLayer.weight = new Tensor(1, 1, hiddenSize, hiddenSize, RandomUtils.order(this.hiddenSize * this.hiddenSize, 0.2f, 0.0f), true);
        //		this.selfLayer.weight.fill(0.2f);
        //		this.selfLayer.bias = new Tensor(1, 1, 1, hiddenSize, RandomUtils.val(this.hiddenSize, 0.2f), true);
        this.outputActive = createActiveLayer(activeType, selfLayer);
        //		System.out.println(JsonUtils.toJson(this.inputLayer.weight.syncHost()));
        //		System.out.println(JsonUtils.toJson(this.inputLayer.bias.syncHost()));
        //		System.out.println(JsonUtils.toJson(this.selfLayer.weight.syncHost()));
        //		System.out.println(JsonUtils.toJson(this.selfLayer.bias.syncHost()));
    }

    public ActiveFunctionLayer createActiveLayer(ActiveType activeType, Layer preLayer) {
        switch (activeType) {
            case sigmoid:
                return new SigmodLayer(preLayer);
            case relu:
                return new ReluLayer(preLayer);
            case leaky_relu:
                return new LeakyReluLayer(preLayer);
            case tanh:
                return new TanhLayer(preLayer);
            default:
                throw new RuntimeException("The rnn layer is not support the [" + activeType + "] active function.");
        }
    }

    @Override
    public void init() {
        // TODO Auto-generated method stub
        this.number = this.network.number;
        RNN network = (RNN) this.network;
        this.time = network.time;
        if (this.h == null || this.h.getShape()[0] != this.number) {
            this.h = Tensor.createTensor(this.h, this.number, 1, 1, hiddenSize, true);
        }
    }

    public void init(int time, int number) {
        // TODO Auto-generated method stub
        this.number = number;
        this.time = time;
        if (this.h == null || this.h.getShape()[0] != this.number) {
            this.h = Tensor.createTensor(this.h, this.number, 1, 1, hiddenSize, true);
        }
    }

    @Override
    public void initBack() {
        // TODO Auto-generated method stub
    }

    @Override
    public void initParam() {
        // TODO Auto-generated method stub
    }

    @Override
    public void output() {
        // TODO Auto-generated method stub
        int batch = this.number / this.time;
        int onceSize = batch * this.h.getOnceSize();
        /**
         * ht = f(W * xt + bx + U * ht-1 + bh)

         */
        //		input.showDM();
        //		this.h.clearGPU();
        if (this.input != null) {
            for (int t = 0; t < time; t++) {
                inputLayer.forward(this.input, batch, t);
                if (t == 0 && h_0 != null) {
                    selfLayer.forward(h_0, batch);
                } else {
                    selfLayer.forward(this.h, batch, t - 1, t);
                }
                Tensor_OP().add(inputLayer.getOutput(), selfLayer.getOutput(), this.h, t * onceSize, onceSize);
                outputActive.forward(this.h, batch, t);
                baseKernel().copy_gpu(outputActive.getOutput(), this.h, onceSize, t * onceSize, 1, t * onceSize, 1);
                //				this.h = outputActive.output;
            }
        }
        this.output = outputActive.output;
    }

    @Override
    public Tensor getOutput() {
        // TODO Auto-generated method stub
        return output;
    }

    @Override
    public void diff() {
        // TODO Auto-generated method stub
        int batch = this.number / time;
        int onceSize = batch * selfLayer.input.getOnceSize();
        inputLayer.clear();
        selfLayer.clear();
        for (int t = time - 1; t >= 0; t--) {
            if (t < time - 1) {
                baseKernel().axpy_gpu(selfLayer.diff, this.delta, onceSize, 1, t * onceSize, 1, t * onceSize, 1);
            }
            outputActive.back(this.delta, batch, t);
            if (t == 0 && h_0 != null) {
                selfLayer.back(outputActive.diff, h_0, h_0.getGrad(), batch, t);
            } else {
                selfLayer.back(outputActive.diff, batch, t, t, t - 1);
            }
            inputLayer.back(outputActive.diff, batch, t);
        }
        this.diff = inputLayer.diff;
        //		this.diff.showDM(0);
        //		inputLayer.diffW.showDM(0);
        //		selfLayer.diffW.showDM(0);
        //		inputLayer.diffB.showDM(0);
        //		selfLayer.diffB.showDM(0);
    }

    @Override
    public void forward() {
        // TODO Auto-generated method stub
        /**
         * 参数初始化

         */
        this.init();
        /**
         * 设置输入

         */
        this.setInput();
        /**
         * 计算输出

         */
        this.output();
    }

    public void forward(int time, int number) {
        // TODO Auto-generated method stub
        /**
         * 参数初始化

         */
        this.init(time, number);
        /**
         * 设置输入

         */
        this.setInput();
        /**
         * 计算输出

         */
        this.output();
    }

    @Override
    public void back() {
        // TODO Auto-generated method stub
        this.initBack();
        /**
         * 设置梯度

         */
        this.setDelta();
        /**
         * 计算梯度

         */
        this.diff();
        if (this.network.GRADIENT_CHECK) {
            this.gradientCheck();
        }
    }

    @Override
    public void forward(Tensor inpnut) {
        // TODO Auto-generated method stub
        /**
         * 参数初始化

         */
        this.init();
        /**
         * 设置输入

         */
        this.setInput(inpnut);
        /**
         * 计算输出

         */
        this.output();
    }

    public void forward(Tensor input, Tensor hidden, int time) {
        // TODO Auto-generated method stub
        this.h_0 = hidden;
        /**
         * 参数初始化

         */
        this.init(time, input.getShape()[0]);
        /**
         * 设置输入

         */
        this.setInput(input);
        /**
         * 计算输出

         */
        this.output();
    }

    public void forwardHidden(Tensor hidden) {
        // TODO Auto-generated method stub
        this.h = hidden;
        /**
         * 参数初始化

         */
        this.init();
        /**
         * 设置输入

         */
        this.setInput();
        /**
         * 计算输出

         */
        this.output();
    }

    @Override
    public void back(Tensor delta) {
        // TODO Auto-generated method stub
        this.initBack();
        /**
         * 设置梯度

         */
        this.setDelta(delta);
        /**
         * 计算梯度

         */
        this.diff();
        if (this.network.GRADIENT_CHECK) {
            this.gradientCheck();
        }
    }

    @Override
    public void update() {
        // TODO Auto-generated method stub
        inputLayer.update(number / time);
        selfLayer.update(number / time);
    }

    @Override
    public void showDiff() {
        // TODO Auto-generated method stub
    }

    @Override
    public LayerType getLayerType() {
        // TODO Auto-generated method stub
        return LayerType.rnn;
    }

    @Override
    public float[][][][] output(float[][][][] input) {
        // TODO Auto-generated method stub
        return null;
    }

    @Override
    public void initCache() {
        // TODO Auto-generated method stub
    }

    @Override
    public void backTemp() {
        // TODO Auto-generated method stub
    }

    public Tensor getH() {
        return h_0;
    }

    @Override
    public void accGrad(float scale) {
        // TODO Auto-generated method stub
    }
}

