package com.omega.engine.nn.layer;

import com.omega.engine.active.ActiveType;
import com.omega.engine.gpu.BaseKernel;
import com.omega.engine.nn.layer.active.*;
import com.omega.engine.nn.network.Network;
import com.omega.engine.nn.network.RNN;
import com.omega.engine.tensor.Tensor;

/**
 * LSTM
 *
 * @author Administrator
 * <p>
 * forgot gate
 * <p>
 * ft = sigmoid(Wf * ht-1 + Uf * xt + bf)
 * <p>
 * input gate
 * <p>
 * it = sigmoid(Wi * ht-1 + Ui * xt + bi)
 * <p>
 * candidate memory
 * <p>
 * gt = tanh(Wg * ht-1 + Ug * xt + bg)
 * <p>
 * cell status
 * <p>
 * ct = ct-1 ⊙ ft + it ⊙ gt
 * <p>
 * output gate
 * <p>
 * ot = sigmoid(Wo * ht-1 + Uo * xt + bo)
 * <p>
 * hidden status
 * <p>
 * ht = ot ⊙ tanh(ct)
 */
public class LSTMLayer extends Layer {
    private int time = 0;
    private int inputSize;
    private int hiddenSize;
    private boolean bias = false;
    private FullyLayer fxl;
    private FullyLayer ixl;
    private FullyLayer gxl;
    private FullyLayer oxl;
    private FullyLayer fhl;
    private FullyLayer ihl;
    private FullyLayer ghl;
    private FullyLayer ohl;
    private ActiveFunctionLayer fa;
    private ActiveFunctionLayer ia;
    private ActiveFunctionLayer ga;
    private ActiveFunctionLayer oa;
    private ActiveFunctionLayer ha;
    /**
     * forgot gate
     * <p>
     * ft = sigmoid(Wf * ht-1 + Uf * xt + bf)
     */
    private Tensor f;
    /**
     * input gate
     * <p>
     * it = sigmoid(Wi * ht-1 + Ui * xt + bi)
     */
    private Tensor i;
    /**
     * candidate memory
     * <p>
     * gt = tanh(Wg * ht-1 + Ug * xt + bg)
     */
    private Tensor g;
    /**
     * cell status
     * <p>
     * ct = ct-1 ⊙ ft + it ⊙ gt
     */
    private Tensor c;
    /**
     * output gate
     * <p>
     * ot = sigmoid(Wo * ht-1 + Uo * xt + bo)
     */
    private Tensor o;
    /**
     * hidden status
     * <p>
     * ht = ot ⊙ tanh(ct)
     */
    private Tensor h;
    private Tensor temp;
    private Tensor h_diff;
    private Tensor c_diff;
    private Tensor detlaXo;
    private Tensor d_tanhc;
    private BaseKernel baseKernel;

    public LSTMLayer(int inputNum, int hiddenNum, int time, boolean bias) {
        this.time = time;
        this.inputSize = inputNum;
        this.hiddenSize = hiddenNum;
        this.bias = bias;
        this.initLayers();
    }

    public LSTMLayer(int inputNum, int hiddenNum, int time, boolean bias, Network network) {
        this.network = network;
        this.time = time;
        this.inputSize = inputNum;
        this.hiddenSize = hiddenNum;
        this.bias = bias;
        this.initLayers();
    }

    public void initLayers() {
        this.fxl = FullyLayer.createRNNCell(inputSize, hiddenSize, time, bias, network);
        this.ixl = FullyLayer.createRNNCell(inputSize, hiddenSize, time, bias, network);
        this.gxl = FullyLayer.createRNNCell(inputSize, hiddenSize, time, bias, network);
        this.oxl = FullyLayer.createRNNCell(inputSize, hiddenSize, time, bias, network);
        this.fhl = FullyLayer.createRNNCell(hiddenSize, hiddenSize, time, false, network);
        this.ihl = FullyLayer.createRNNCell(hiddenSize, hiddenSize, time, false, network);
        this.ghl = FullyLayer.createRNNCell(hiddenSize, hiddenSize, time, false, network);
        this.ohl = FullyLayer.createRNNCell(hiddenSize, hiddenSize, time, false, network);
        this.fa = createActiveLayer(ActiveType.sigmoid, fhl);
        this.ia = createActiveLayer(ActiveType.sigmoid, ihl);
        this.ga = createActiveLayer(ActiveType.tanh, ghl);
        this.oa = createActiveLayer(ActiveType.sigmoid, ohl);
        this.ha = createActiveLayer(ActiveType.tanh, fhl);
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
        if (this.h == null || this.h.number != this.number) {
            this.f = Tensor.createTensor(this.f, number, 1, 1, hiddenSize, true);
            this.i = Tensor.createTensor(this.i, number, 1, 1, hiddenSize, true);
            this.g = Tensor.createTensor(this.g, number, 1, 1, hiddenSize, true);
            this.c = Tensor.createTensor(this.c, number, 1, 1, hiddenSize, true);
            this.o = Tensor.createTensor(this.o, number, 1, 1, hiddenSize, true);
            this.h = Tensor.createTensor(this.h, number, 1, 1, hiddenSize, true);
            this.temp = Tensor.createTensor(this.temp, number, 1, 1, hiddenSize, true);
        }
    }

    @Override
    public void initBack() {
        // TODO Auto-generated method stub
        int batch = this.number / this.time;
        if (this.detlaXo == null || this.detlaXo.number != batch) {
            this.detlaXo = Tensor.createTensor(this.detlaXo, batch, 1, 1, hiddenSize, true);
            this.d_tanhc = Tensor.createTensor(this.d_tanhc, batch, 1, 1, hiddenSize, true);
        }
        if (this.h_diff == null || this.h_diff.number != this.number) {
            this.h_diff = Tensor.createTensor(this.h_diff, this.number, 1, 1, hiddenSize, true);
            this.c_diff = Tensor.createTensor(this.c_diff, this.number, 1, 1, hiddenSize, true);
        }
        if (this.diff == null || this.diff.number != this.number) {
            this.diff = Tensor.createTensor(this.diff, this.number, 1, 1, inputSize, true);
        }
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
        if (this.input != null) {
            c.clearGPU();
            //			h.clearGPU();
            for (int t = 0; t < time; t++) {
                fxl.forward(this.input, batch, t);
                ixl.forward(this.input, batch, t);
                gxl.forward(this.input, batch, t);
                oxl.forward(this.input, batch, t);
                fhl.forward(this.h, batch, t - 1, t);
                ihl.forward(this.h, batch, t - 1, t);
                ghl.forward(this.h, batch, t - 1, t);
                ohl.forward(this.h, batch, t - 1, t);
                /**
                 * ft = sigmoid(Wf * ht-1 + Uf * xt + bf)
                 * it = sigmoid(Wi * ht-1 + Ui * xt + bi)
                 * gt = tanh(Wg * ht-1 + Ug * xt + bg)
                 * ot = sigmoid(Wo * ht-1 + Uo * xt + bo)

                 */
                Tensor_OP().add(fxl.getOutput(), fhl.getOutput(), this.f, t * onceSize, onceSize);
                Tensor_OP().add(ixl.getOutput(), ihl.getOutput(), this.i, t * onceSize, onceSize);
                Tensor_OP().add(gxl.getOutput(), ghl.getOutput(), this.g, t * onceSize, onceSize);
                Tensor_OP().add(oxl.getOutput(), ohl.getOutput(), this.o, t * onceSize, onceSize);
                fa.forward(this.f, batch, t);
                ia.forward(this.i, batch, t);
                ga.forward(this.g, batch, t);
                oa.forward(this.o, batch, t);
                /**
                 * ct = ct-1 ⊙ ft + it ⊙ gt

                 */
                Tensor_OP().mul(ia.getOutput(), ga.getOutput(), temp, t * onceSize, onceSize);
                if (t > 0) {
                    Tensor_OP().mul(c, fa.getOutput(), c, (t - 1) * onceSize, t * onceSize, t * onceSize, onceSize);
                }
                Tensor_OP().add(temp, c, c, t * onceSize, onceSize);
                /**
                 * ht = ot ⊙  tanh(ct)

                 */
                ha.forward(c, batch, t);
                Tensor_OP().mul(oa.getOutput(), ha.getOutput(), this.h, t * onceSize, onceSize);
                //				baseKernel.copy_gpu(ha.getOutput(), this.h, onceSize, t * onceSize, 1, t * onceSize, 1);
            }
        }
        this.output = this.h;
        //		this.input.showDMByNumber(0);
        //		this.output.showDMByNumber(0);
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
        int onceSize = batch * hiddenSize;
        fxl.clear();
        ixl.clear();
        gxl.clear();
        oxl.clear();
        fhl.clear();
        ihl.clear();
        ghl.clear();
        ohl.clear();
        this.h_diff.clearGPU();
        this.c_diff.clearGPU();
        for (int t = time - 1; t >= 0; t--) {
            if (t < time - 1) {
                baseKernel.axpy_gpu(this.h_diff, this.delta, onceSize, 1, t * onceSize, 1, t * onceSize, 1);
            }
            // detlaXo = delta_t * o_t
            Tensor_OP().mul(delta, oa.getOutput(), this.detlaXo, t * onceSize, t * onceSize, 0, onceSize);
            // d_tanh(ct) = 1 - tanh_c * tanh_c
            Tensor_OP().mul(ha.getOutput(), ha.getOutput(), d_tanhc, t * onceSize, t * onceSize, 0, onceSize);
            Tensor_OP().sub(1.0f, d_tanhc, d_tanhc, 0, onceSize);
            Tensor_OP().mul(this.detlaXo, d_tanhc, this.detlaXo, 0, onceSize);
            /**
             * delta_ct = delta_ct + delta_t * o_t * d_tanh(ct)

             */
            if (t < time - 1) {
                /**
                 * delta_ct-1 = delta_t * o_t * d_tanh(ct) * ft

                 */
                Tensor_OP().mul(detlaXo, fa.getOutput(), this.c_diff, 0, t * onceSize, (t - 1) * onceSize, onceSize);
                Tensor_OP().add(this.detlaXo, this.c_diff, this.detlaXo, 0, t * onceSize, 0, onceSize);
            }
            /**
             * delta_o = delta_t * tanh_c * d_sigmoid(o)

             */
            Tensor_OP().mul(delta, ha.getOutput(), temp, t * onceSize, onceSize);
            oa.back(temp, batch, t);
            /**
             * delta_f = delta_t * o_t * d_tanh(ct) * c_t-1 * d_sigmoid(f)

             */
            Tensor_OP().mul(detlaXo, c, temp, 0, (t - 1) * onceSize, t * onceSize, onceSize);
            fa.back(temp, batch, t);
            /**
             * delta_i = delta_t * o_t * d_tanh(ct) * c_t * d_sigmoid(i)

             */
            Tensor_OP().mul(detlaXo, c, temp, 0, t * onceSize, t * onceSize, onceSize);
            ia.back(temp, batch, t);
            /**
             * delta_g = delta_t * o_t * d_tanh(ct) * i_t * d_sigmoid(g)

             */
            Tensor_OP().mul(detlaXo, ia.getOutput(), temp, 0, t * onceSize, t * onceSize, onceSize);
            ga.back(temp, batch, t);
            fxl.back(fa.diff, batch, t);
            ixl.back(ia.diff, batch, t);
            gxl.back(ga.diff, batch, t);
            oxl.back(oa.diff, batch, t);
            fhl.back(fa.diff, batch, t, t, t - 1);
            ihl.back(ia.diff, batch, t, t, t - 1);
            ghl.back(ga.diff, batch, t, t, t - 1);
            ohl.back(oa.diff, batch, t, t, t - 1);
            Tensor_OP().add(fhl.diff, ihl.diff, h_diff, (t - 1) * onceSize, onceSize);
            Tensor_OP().add(h_diff, ghl.diff, h_diff, (t - 1) * onceSize, onceSize);
            Tensor_OP().add(h_diff, ohl.diff, h_diff, (t - 1) * onceSize, onceSize);
            Tensor_OP().add(fxl.diff, ixl.diff, this.diff, t * onceSize, onceSize);
            Tensor_OP().add(this.diff, gxl.diff, this.diff, t * onceSize, onceSize);
            Tensor_OP().add(this.diff, oxl.diff, this.diff, t * onceSize, onceSize);
        }
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

    @Override
    public void update() {
        // TODO Auto-generated method stub
        fxl.update(number / time);
        ixl.update(number / time);
        gxl.update(number / time);
        oxl.update(number / time);
        fhl.update(number / time);
        ihl.update(number / time);
        ghl.update(number / time);
        ohl.update(number / time);
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
    public void accGrad(float scale) {
        // TODO Auto-generated method stub
    }
}

