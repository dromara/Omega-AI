package com.omega.engine.nn.layer.transformer;

import com.omega.common.tensor.Tensor;
import com.omega.utils.MatrixUtils;
import com.omega.utils.RandomUtils;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.normalization.LNLayer;
import com.omega.engine.nn.network.Network;
import com.omega.engine.nn.network.Transformer;
import com.omega.engine.updater.UpdaterFactory;

/**
 * Transformer Decoder Layer
 *
 * @author Administrator
 */
public class TransformerDecoderLayer extends Layer {
    private int time;
    private int embedDim = 0;
    private int nChannel = 1;
    private boolean bias = false;
    private boolean layer_norm = false;
    private int headNum = 12;
    private MultiHeadAttentionLayer attn;
    private PoswiseFeedForwardLinearLayer feed_forward;
    //	private PoswiseFeedForwardLayer feed_forward;
    private LNLayer ln1;
    private LNLayer ln2;
    private Tensor ln1i;
    private Tensor ln2i;

    public TransformerDecoderLayer(int time, int embedDim, int nChannel, boolean bias, boolean layer_norm) {
        this.time = time;
        this.embedDim = embedDim;
        this.nChannel = nChannel;
        this.bias = bias;
        this.layer_norm = layer_norm;
        this.oChannel = 1;
        this.oHeight = 1;
        this.oWidth = embedDim;
        this.initLayers();
    }

    public TransformerDecoderLayer(int time, int embedDim, int nChannel, boolean bias, boolean layer_norm, Network network) {
        this.network = network;
        if (this.updater == null) {
            this.setUpdater(UpdaterFactory.create(network));
        }
        this.time = time;
        this.embedDim = embedDim;
        this.nChannel = nChannel;
        this.bias = bias;
        this.layer_norm = layer_norm;
        this.oChannel = 1;
        this.oHeight = 1;
        this.oWidth = embedDim;
        this.initLayers();
    }

    public static void main(String[] args) {
        int batchSize = 5;
        int time = 10;
        int embedDim = 8;
        int nChannel = 4;
        Transformer tf = new Transformer();
        //		tf.CUDNN = true;
        tf.number = batchSize * time;
        TransformerDecoderLayer mal = new TransformerDecoderLayer(time, embedDim, nChannel, false, true, tf);
        float[] data = RandomUtils.order(batchSize * time * embedDim, 0.1f, 0.1f);
        Tensor input = new Tensor(batchSize, time, 1, embedDim, data, true);
        input.showShape();
        input.showDM();
        float[] delta_data = MatrixUtils.val(batchSize * time * embedDim, 1.0f);
        Tensor delta = new Tensor(batchSize, time, 1, embedDim, delta_data, true);
        tf.tensorOP.add(input, delta, delta);
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
        this.attn = new MultiHeadAttentionLayer(embedDim, headNum, time, bias, layer_norm, network);
        this.feed_forward = new PoswiseFeedForwardLinearLayer(embedDim, nChannel, bias, layer_norm, network);
        //		this.feed_forward = new PoswiseFeedForwardLayer(time, embedDim, nChannel, bias, layer_norm, network);
        this.ln1 = new LNLayer(attn);
        this.ln2 = new LNLayer(feed_forward);
    }

    @Override
    public void init() {
        // TODO Auto-generated method stub
        this.number = this.input.number;
        if (ln1i == null || this.number != ln1i.number) {
            ln1i = Tensor.createTensor(ln1i, number, input.channel, input.height, input.width, true);
            //			System.out.println("-----------------");
            //			ln1i.showShape();
        }
        if (ln2i == null || this.number != ln2i.number) {
            ln2i = Tensor.createTensor(ln2i, number, input.channel, input.height, input.width, true);
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
        this.attn.forward(input);
        this.feed_forward.forward(this.attn.getOutput());
        this.output = this.feed_forward.getOutput();
        //		this.attn.forward(input);
        //
        //		TensorOP.add(this.attn.getOutput(), input, ro);
        //
        //		this.ln1.forward(ro);
        //
        //		this.feed_forward.forward(this.ln1.getOutput());
        //
        //		TensorOP.add(this.feed_forward.getOutput(), this.ln1.getOutput(), ro);
        //
        //		this.ln2.forward(ro);
        //
        //		this.output = this.ln2.getOutput();
    }

    public void output(Tensor mask) {
        // TODO Auto-generated method stub
        //
        //		this.attn.forward(input, mask);
        //
        //		this.feed_forward.forward(this.attn.getOutput());
        //
        //		this.output = this.feed_forward.getOutput();
        this.attn.forward(input, mask);
        Tensor_OP().add(this.attn.getOutput(), input, ln1i);
        this.ln1.forward(ln1i);
        this.feed_forward.forward(this.ln1.getOutput());
        Tensor_OP().add(this.feed_forward.getOutput(), this.ln1.getOutput(), ln2i);
        this.ln2.forward(ln2i);
        this.output = this.ln2.getOutput();
    }

    @Override
    public Tensor getOutput() {
        // TODO Auto-generated method stub
        return output;
    }

    @Override
    public void diff() {
        // TODO Auto-generated method stub
        //		this.feed_forward.back(delta);
        //
        //		this.attn.back(feed_forward.diff);
        //
        //		this.diff = this.attn.diff;
        this.ln2.back(delta);
        baseKernel().copy_gpu(this.ln2.diff, this.ln2i, this.ln2.diff.getDataLength(), 1, 1);
        this.feed_forward.back(this.ln2.diff);
        Tensor_OP().add(this.feed_forward.diff, ln2i, ln2.getOutput());
        this.ln1.back(ln2.getOutput());
        baseKernel().copy_gpu(this.ln1.diff, this.ln1i, this.ln1.diff.getDataLength(), 1, 1);
        this.attn.back(ln1.diff);
        Tensor_OP().add(this.attn.diff, ln1i, ln1i);
        this.diff = ln1i;
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

    public void forward(Tensor input, Tensor mask) {
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
        this.output(mask);
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
        attn.update();
        feed_forward.update();
        ln1.update();
        ln2.update();
    }

    @Override
    public void showDiff() {
        // TODO Auto-generated method stub
    }

    @Override
    public LayerType getLayerType() {
        // TODO Auto-generated method stub
        return LayerType.transformer_decoder;
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
        attn.accGrad(scale);
        feed_forward.accGrad(scale);
        ln1.accGrad(scale);
        ln2.accGrad(scale);
    }
}

