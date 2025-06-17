package com.omega.engine.nn.layer;

import com.omega.engine.gpu.cudnn.RNNCudnnKernelV8;
import com.omega.engine.nn.layer.gpu.RNNBaseKernel;
import com.omega.engine.nn.network.Network;
import com.omega.engine.nn.network.RNN;
import com.omega.engine.tensor.Tensor;

import jcuda.Sizeof;

/**
 * Recurrent Layer
 *
 * @author Administrator
 */
public class RNNBlockLayer extends BaseRNNLayer {
    private int time = 0;
    private int inputSize;
    private int hiddenSize;
    private int layerNum = 1;
    private RNNBaseKernel kernel;
    private int rnnMode = 0;
    private boolean bidirectional = false;
    private float dropout = 0;
    private Tensor hx;
    private Tensor cx;
    private Tensor hy;
    private Tensor cy;
    private Tensor dhx;
    private Tensor dcx;
    private Tensor dhy;
    private Tensor dcy;
    private int hidden_len = 0;

    //	private int numLinearLayers = 2;
    public RNNBlockLayer(int time, int inputSize, int hiddenSize, int rnnMode, boolean bidirectional, boolean hasBias, float dropout) {
        this.hasBias = hasBias;
        this.time = time;
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;
        this.rnnMode = rnnMode;
        this.bidirectional = bidirectional;
        this.dropout = dropout;
        this.oChannel = 1;
        this.oHeight = 1;
        this.oWidth = hiddenSize;
        this.initKernel();
    }

    public RNNBlockLayer(int time, int layerNum, int inputSize, int hiddenSize, int rnnMode, boolean bidirectional, boolean hasBias, float dropout) {
        this.hasBias = hasBias;
        this.layerNum = layerNum;
        this.time = time;
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;
        this.rnnMode = rnnMode;
        this.bidirectional = bidirectional;
        this.dropout = dropout;
        this.oChannel = 1;
        this.oHeight = 1;
        this.oWidth = hiddenSize;
        this.initKernel();
    }

    public RNNBlockLayer(int time, int layerNum, int inputSize, int hiddenSize, int rnnMode, boolean bidirectional, boolean hasBias, float dropout, Network network) {
        this.hasBias = hasBias;
        this.layerNum = layerNum;
        this.network = network;
        this.time = time;
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;
        this.oChannel = 1;
        this.oHeight = 1;
        this.oWidth = hiddenSize;
        this.rnnMode = rnnMode;
        this.bidirectional = bidirectional;
        this.dropout = dropout;
        this.initKernel();
    }

    public void initKernel() {
        if (kernel == null) {
            kernel = new RNNCudnnKernelV8(time, layerNum, inputSize, hiddenSize, bidirectional, rnnMode, dropout, hasBias, cuda());
            //			kernel = new RNNCudnnKernel(time, layerNum, inputSize, hiddenSize, bidirectional, rnnMode, dropout);
        }
    }

    @Override
    public void init() {
        // TODO Auto-generated method stub
        this.number = this.network.number;
        RNN network = (RNN) this.network;
        this.time = network.time;
        if (this.time != kernel.seqLength) {
            kernel.seqLength = this.time;
        }
        //h,c * layernum
        hidden_len = number / time * layerNum;
        if (this.getHx() == null || this.getHx().getShape()[0] != hidden_len) {
            this.setHx(Tensor.createTensor(this.getHx(), hidden_len, 1, 1, hiddenSize, true));
        }
        if (this.getCx() == null || this.getCx().getShape()[0] != hidden_len) {
            this.setCx(Tensor.createTensor(this.getCx(), hidden_len, 1, 1, hiddenSize, true));
        }
        if (this.getHy() == null || this.getHy().getShape()[0] != hidden_len) {
            this.setHy(Tensor.createTensor(this.getHy(), hidden_len, 1, 1, hiddenSize, true));
        }
        if (this.getCy() == null || this.getCy().getShape()[0] != hidden_len) {
            this.setCy(Tensor.createTensor(this.getCy(), hidden_len, 1, 1, hiddenSize, true));
        }
        if (this.output == null || this.number != this.output.getShape()[0]) {
            this.output = Tensor.createTensor(this.output, number, 1, 1, hiddenSize, true);
        }
    }

    public void init(int time, int number) {
        // TODO Auto-generated method stub
        this.number = number;
        this.time = time;
        if (this.time != kernel.seqLength) {
            kernel.seqLength = this.time;
        }
        hidden_len = number / time * layerNum;
        if (this.getHx() == null || this.getHx().getShape()[0] != hidden_len) {
            this.setHx(Tensor.createTensor(this.getHx(), hidden_len, 1, 1, hiddenSize, true));
        }
        if (this.getCx() == null || this.getCx().getShape()[0] != hidden_len) {
            this.setCx(Tensor.createTensor(this.getCx(), hidden_len, 1, 1, hiddenSize, true));
        }
        if (this.getHy() == null || this.getHy().getShape()[0] != hidden_len) {
            this.setHy(Tensor.createTensor(this.getHy(), hidden_len, 1, 1, hiddenSize, true));
        }
        if (this.getCy() == null || this.getCy().getShape()[0] != hidden_len) {
            this.setCy(Tensor.createTensor(this.getCy(), hidden_len, 1, 1, hiddenSize, true));
        }
        if (this.output == null || this.number != this.output.getShape()[0]) {
            this.output = Tensor.createTensor(this.output, number, 1, 1, hiddenSize, true);
        }
    }

    @Override
    public void initBack() {
        // TODO Auto-generated method stub
        if (this.dhx == null || this.dhx.getShape()[0] != hidden_len) {
            this.dhx = Tensor.createTensor(this.dhx, hidden_len, 1, 1, hiddenSize, true);
        }
        if (this.dhy == null || this.dhy.getShape()[0] != hidden_len) {
            this.dhy = Tensor.createTensor(this.dhy, hidden_len, 1, 1, hiddenSize, true);
        }
        if (this.dcx == null || this.dcx.getShape()[0] != hidden_len) {
            this.dcx = Tensor.createTensor(this.dcx, hidden_len, 1, 1, hiddenSize, true);
        }
        if (this.dcy == null || this.dcy.getShape()[0] != hidden_len) {
            this.dcy = Tensor.createTensor(this.dcy, hidden_len, 1, 1, hiddenSize, true);
        }
        if (this.diff == null || this.number != this.diff.getShape()[0]) {
            this.diff = new Tensor(number, 1, 1, inputSize, true);
        }
    }

    @Override
    public void initParam() {
        // TODO Auto-generated method stub
        if (this.weight == null) {
            int weightSize = (int) (kernel.weightSize() / Sizeof.FLOAT);
            this.weight = new Tensor(1, 1, 1, weightSize, true);
            this.diffW = new Tensor(1, 1, 1, weightSize, true);
            //			System.out.println(weightSize+":"+this.inputSize * this.hiddenSize);
            //			this.weight = new Tensor(1, 1, 1, weightSize, RandomUtils.uniformFloat(weightSize, inputSize), true);
            kernel.initWeights(weight);
            //			weight.showDM();
        }
    }

    @Override
    public void output() {
        // TODO Auto-generated method stub
        kernel.init(input.getShape()[0], time);
        initParam();
        kernel.forward(this.network.RUN_MODEL, input, getHx(), getCx(), weight, output, getHy(), getCy());
    }

    public void output(Tensor hx, Tensor cx) {
        // TODO Auto-generated method stub
        kernel.init(input.getShape()[0], time);
        initParam();
        kernel.forward(this.network.RUN_MODEL, input, hx, cx, weight, output, getHy(), getCy());
    }

    @Override
    public Tensor getOutput() {
        // TODO Auto-generated method stub
        return output;
    }

    @Override
    public void diff() {
        // TODO Auto-generated method stub
        //		System.out.println("in-rnn");
        kernel.dx(delta, dhy, dcy, output, hx, cx, weight, diff, dhx, dcx);
        //		GradClipping.gradClipping(diff, 1e-7f);
        kernel.dw(delta, output, input, hx, diffW);
    }

    public void diff(Tensor hx, Tensor cx, Tensor dhy, Tensor dcy) {
        // TODO Auto-generated method stub
        //		System.out.println("in-rnn");
        kernel.dx(delta, dhy, dcy, output, hx, cx, weight, diff, dhx, dcx);
        //		GradClipping.gradClipping(diff, 1e-7f);
        kernel.dw(delta, output, input, hx, diffW);
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

    public void back(Tensor delta, Tensor hx, Tensor cx, Tensor dhy, Tensor dcy) {
        // TODO Auto-generated method stub
        this.initBack();
        /**
         * 设置梯度

         */
        this.setDelta(delta);
        /**
         * 计算梯度

         */
        this.diff(hx, cx, dhy, dcy);
        if (this.network.GRADIENT_CHECK) {
            this.gradientCheck();
        }
    }

    @Override
    public void update() {
        // TODO Auto-generated method stub
        if (!this.freeze) {
            if (this.updater != null) {
                this.updater.update(this);
            } else {
                for (int i = 0; i < this.weight.getDataLength(); i++) {
                    this.weight.getData()[i] -= this.learnRate * this.diffW.getData()[i];
                }
                if (hasBias) {
                    for (int i = 0; i < this.bias.getDataLength(); i++) {
                        this.bias.getData()[i] -= this.learnRate * this.diffB.getData()[i];
                    }
                }
            }
        }
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

    @Override
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
    public void forward(Tensor input, Tensor hx, Tensor cx, int time) {
        // TODO Auto-generated method stub
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
        this.output(hx, cx);
    }

    public Tensor getHx() {
        return hx;
    }

    public void setHx(Tensor hx) {
        this.hx = hx;
    }

    public Tensor getCx() {
        return cx;
    }

    public void setCx(Tensor cx) {
        this.cx = cx;
    }

    public Tensor getHy() {
        return hy;
    }

    public void setHy(Tensor hy) {
        this.hy = hy;
    }

    public Tensor getCy() {
        return cy;
    }

    public void setCy(Tensor cy) {
        this.cy = cy;
    }

    public Tensor getDhx() {
        return dhx;
    }

    public Tensor getDcx() {
        return dcx;
    }

    public Tensor getDhy() {
        return dhy;
    }

    public Tensor getDcy() {
        return dcy;
    }

    @Override
    public void accGrad(float scale) {
        // TODO Auto-generated method stub
    }
}

