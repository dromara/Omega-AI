package com.omega.engine.nn.layer.unet;

import com.omega.common.data.Tensor;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.network.Network;

/**
 * UNetTransformerBlockLayer
 *
 * @author Administrator
 */
public class UNetTransformerBlockLayer extends Layer {
    public UNetAttentionLayer2 attn;
    public UNetAttentionLayer2 crossAttn;
    public UNetFFNBlockLayer affn;
    public UNetFFNBlockLayer caffn;
    private boolean seft_attn = false;
    private boolean cross_attn = false;
    private int time;
    private int embedDim;
    private int contextDim;
    private int kvTime;
    private int headNum;

    // input shape must be [B, H*W, C]
    public UNetTransformerBlockLayer(int embedDim, int headNum, int time, int kvTime, int contextDim, boolean seft_attn, boolean cross_attn, Network network) {
        this.network = network;
        this.seft_attn = seft_attn;
        this.cross_attn = cross_attn;
        this.headNum = headNum;
        this.contextDim = contextDim;
        this.embedDim = embedDim;
        this.time = time;
        this.kvTime = kvTime;
        this.channel = time;
        this.oChannel = time;
        this.height = 1;
        this.width = embedDim;
        this.oHeight = height;
        this.oWidth = width;
        initLayers();
    }

    public void initLayers() {
        if (seft_attn) {
            attn = new UNetAttentionLayer2(embedDim, headNum, time, true, false, true, network);
            affn = new UNetFFNBlockLayer(time, 1, width, 4, network);
        }
        if (cross_attn) {
            crossAttn = new UNetAttentionLayer2(embedDim, contextDim, contextDim, headNum, time, kvTime, true, false, true, network);
            caffn = new UNetFFNBlockLayer(time, 1, width, 4, network);
        }
    }

    @Override
    public void init() {
        // TODO Auto-generated method stub
        this.number = this.network.number;
    }

    public void init(Tensor input) {
        // TODO Auto-generated method stub
        this.number = input.number;
        if (this.output == null || this.output.number != this.number) {
            this.output = Tensor.createGPUTensor(this.output, number, oChannel, oHeight, oWidth, true);
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
        Tensor x = input;
        if (seft_attn) {
            attn.forward(x);
            affn.forward(attn.getOutput());
            x = affn.getOutput();
        }
        this.output = x;
    }

    public void output(Tensor context) {
        // TODO Auto-generated method stub
        Tensor x = input;
        if (seft_attn) {
            attn.forward(x);
            affn.forward(attn.getOutput());
            x = affn.getOutput();
        }
        //		x.showDM("attn");
        if (cross_attn) {
            crossAttn.forward(x, context, context);
            //			crossAttn.getOutput().showDM("crossAttn");
            caffn.forward(crossAttn.getOutput());
            x = caffn.getOutput();
        }
        //		x.showDM("st-out");
        this.output = x;
    }

    @Override
    public Tensor getOutput() {
        // TODO Auto-generated method stub
        return this.output;
    }

    @Override
    public void diff() {
        // TODO Auto-generated method stub
        Tensor d = delta;
        if (seft_attn) {
            affn.back(d);
            attn.back(affn.diff);
            d = attn.diff;
        }
        this.diff = d;
    }

    public void diff(Tensor kvDiff) {
        // TODO Auto-generated method stub
        Tensor d = delta;
        if (seft_attn) {
            affn.back(d);
            attn.back(affn.diff);
            d = attn.diff;
        }
        if (cross_attn) {
            caffn.back(d);
            crossAttn.back(caffn.diff, kvDiff);
            d = crossAttn.diff;
        }
        this.diff = d;
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
        initBack();
        /**
         * 设置梯度

         */
        this.setDelta();
        /**
         * 计算梯度

         */
        this.diff();
    }

    @Override
    public void backTemp() {
        // TODO Auto-generated method stub
    }

    @Override
    public void forward(Tensor input) {
        // TODO Auto-generated method stub
        /**
         * 参数初始化

         */
        this.init(input);
        /**
         * 设置输入

         */
        this.setInput(input);
        /**
         * 计算输出

         */
        this.output();
    }

    public void forward(Tensor input, Tensor context) {
        // TODO Auto-generated method stub
        /**
         * 参数初始化

         */
        this.init(input);
        /**
         * 设置输入

         */
        this.setInput(input);
        /**
         * 计算输出

         */
        this.output(context);
    }

    @Override
    public void back(Tensor delta) {
        // TODO Auto-generated method stub
        initBack();
        /**
         * 设置梯度

         */
        this.setDelta(delta);
        /**
         * 计算梯度

         */
        this.diff();
    }

    public void back(Tensor delta, Tensor kvDiff) {
        // TODO Auto-generated method stub
        initBack();
        /**
         * 设置梯度

         */
        this.setDelta(delta);
        /**
         * 计算梯度

         */
        this.diff(kvDiff);
    }

    @Override
    public void update() {
        // TODO Auto-generated method stub
        if (seft_attn) {
            attn.update();
            affn.update();
        }
        if (cross_attn) {
            crossAttn.update();
            caffn.update();
        }
    }

    @Override
    public void showDiff() {
        // TODO Auto-generated method stub
    }

    @Override
    public LayerType getLayerType() {
        // TODO Auto-generated method stub
        return LayerType.unet_down;
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
    public void accGrad(float scale) {
        // TODO Auto-generated method stub
        if (seft_attn) {
            attn.accGrad(scale);
            affn.accGrad(scale);
        }
        if (cross_attn) {
            crossAttn.accGrad(scale);
            caffn.accGrad(scale);
        }
    }
}

