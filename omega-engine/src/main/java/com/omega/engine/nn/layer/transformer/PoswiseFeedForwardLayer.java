package com.omega.engine.nn.layer.transformer;

import com.omega.common.data.Tensor;
import com.omega.common.utils.MatrixUtils;
import com.omega.common.utils.RandomUtils;
import com.omega.engine.nn.layer.ConvolutionLayer;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.active.ReluLayer;
import com.omega.engine.nn.layer.normalization.LNLayer;
import com.omega.engine.nn.network.CNN;
import com.omega.engine.nn.network.Network;

/**
 * PoswiseFeedForward Layer
 *
 * @author Administrator
 */
public class PoswiseFeedForwardLayer extends Layer {
    private int time;
    private int embedDim = 0;
    private int nChannel = 1;
    private boolean bias = false;
    private boolean layer_norm = false;
    private ConvolutionLayer conv1;
    private ReluLayer relu1;
    private ConvolutionLayer conv2;
    private LNLayer lnLayer;
    private Tensor it;
    private Tensor ro;

    public PoswiseFeedForwardLayer(int time, int embedDim, int nChannel, boolean bias, boolean layer_norm) {
        this.time = time;
        this.embedDim = embedDim;
        this.nChannel = nChannel;
        this.bias = bias;
        this.layer_norm = layer_norm;
        this.initLayers();
    }

    public PoswiseFeedForwardLayer(int time, int embedDim, int nChannel, boolean bias, boolean layer_norm, Network network) {
        this.network = network;
        this.time = time;
        this.embedDim = embedDim;
        this.nChannel = nChannel;
        this.bias = bias;
        this.layer_norm = layer_norm;
        this.initLayers();
    }

    public static void main(String[] args) {
        int batchSize = 5;
        int time = 10;
        int embedDim = 8;
        int nChannel = 4;
        CNN tf = new CNN(null);
        tf.CUDNN = true;
        tf.number = batchSize;
        float[] data = RandomUtils.order(batchSize * time * embedDim, 0.1f, 0.1f);
        Tensor input = new Tensor(batchSize, time, 1, embedDim, data, true);
        input.showShape();
        input.showDM();
        float[] delta_data = MatrixUtils.val(batchSize * time * embedDim, 1.0f);
        Tensor delta = new Tensor(batchSize, time, 1, embedDim, delta_data, true);
        PoswiseFeedForwardLayer mal = new PoswiseFeedForwardLayer(time, embedDim, nChannel, false, true, tf);
        //		mal.forward(input);
        mal.forward(input);
        //		input.showDM();
        //		mal.getWeights().showDM();
        //		mal.getOutput().showShape();
        mal.getOutput().showDM();
        mal.back(delta);
        //		mal.diff.showShape();
        mal.diff.showDM();
    }

    public void initLayers() {
        this.conv1 = new ConvolutionLayer(embedDim, nChannel, time, 1, 1, 1, 0, 1, bias, this.network);
        //		this.conv1.weight = new Tensor(nChannel, embedDim, 1, 1, RandomUtils.order(this.nChannel * this.embedDim, 0.1f, 0.1f), true);
        //		this.conv1.bias = new Tensor(1, 1, 1, nChannel, RandomUtils.order(this.nChannel, 0.1f, 0.0f), true);
        this.relu1 = new ReluLayer(conv1);
        this.conv2 = new ConvolutionLayer(nChannel, embedDim, time, 1, 1, 1, 0, 1, bias, this.network);
        //		this.conv2.weight = new Tensor(embedDim, nChannel, 1, 1, RandomUtils.order(this.nChannel * this.embedDim, 0.1f, 0.1f), true);
        //		this.conv2.bias = new Tensor(1, 1, 1, embedDim, RandomUtils.order(this.embedDim, 0.1f, 0.0f), true);
        if (this.layer_norm) {
            this.lnLayer = new LNLayer(this.conv2);
        }
    }

    @Override
    public void init() {
        // TODO Auto-generated method stub
        this.number = this.input.number;
        if (this.ro == null || this.ro.number != this.number) {
            //			System.out.println(number);
            this.it = Tensor.createTensor(this.it, number, embedDim, 1, time, true);
            this.ro = Tensor.createTensor(this.ro, number, time, 1, embedDim, true);
        }
        resize();
    }

    public void resize() {
        this.ro.viewOrg();
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
        Tensor_OP().permute(input, it, new int[]{0, 3, 2, 1});
        it.showShape();
        //		input.view(input.number, nChannel, 1, 1);
        conv1.forward(input);
        relu1.forward(conv1.getOutput());
        conv2.forward(relu1.getOutput());
        Tensor_OP().permute(conv2.getOutput(), this.ro, new int[]{0, 3, 2, 1});
        Tensor_OP().add(this.ro, this.input, this.ro);
        if (this.layer_norm) {
            this.lnLayer.forward(ro);
            this.output = this.lnLayer.getOutput();
        } else {
            this.output = ro;
        }
    }

    @Override
    public Tensor getOutput() {
        // TODO Auto-generated method stub
        return output;
    }

    @Override
    public void diff() {
        // TODO Auto-generated method stub
        this.ro.view(number, embedDim, 1, time);
        if (this.layer_norm) {
            this.lnLayer.back(delta);
            Tensor_OP().permute(this.lnLayer.diff, this.ro, new int[]{0, 3, 2, 1});
        } else {
            Tensor_OP().permute(delta, this.ro, new int[]{0, 3, 2, 1});
        }
        this.conv2.back(this.ro);
        relu1.back(this.conv2.diff);
        conv1.back(relu1.diff);
        this.ro.view(number, time, 1, embedDim);
        Tensor_OP().permute(conv1.diff, this.ro, new int[]{0, 3, 2, 1});
        Tensor_OP().add(this.ro, delta, this.ro);
        this.diff = this.ro;
    }

    @Override
    public void forward() {
        // TODO Auto-generated method stub
        /**
         * 设置输入

         */
        this.setInput();
        /**
         * 参数初始化

         */
        this.init();
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
    public void forward(Tensor input) {
        // TODO Auto-generated method stub
        /**
         * 设置输入

         */
        this.setInput(input);
        /**
         * 参数初始化

         */
        this.init();
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
        conv1.update();
        conv2.update();
        lnLayer.update();
    }

    @Override
    public void showDiff() {
        // TODO Auto-generated method stub
    }

    @Override
    public LayerType getLayerType() {
        // TODO Auto-generated method stub
        return LayerType.poswise_feed_forward;
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
        conv1.accGrad(scale);
        conv2.accGrad(scale);
        lnLayer.accGrad(scale);
    }
}

