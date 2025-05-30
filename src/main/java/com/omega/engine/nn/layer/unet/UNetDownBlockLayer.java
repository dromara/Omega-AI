package com.omega.engine.nn.layer.unet;

import com.omega.engine.nn.layer.*;
import com.omega.engine.nn.network.Network;
import com.omega.engine.tensor.Tensor;
import com.omega.engine.updater.UpdaterFactory;

import java.util.ArrayList;
import java.util.List;

/**
 * UNetDownBlockLayer
 *
 * @author Administrator
 */
public class UNetDownBlockLayer extends Layer {
    public List<UNetResnetBlockLayer> resnetFirst;
    public List<UNetTEmbLayer> tEmbLayers;
    public List<UNetResnetBlockLayer> resnetSecond;
    public List<UNetAttentionLayer> attns;
    public List<UNetAttentionLayer> crossAttns;
    public List<FullyLayer> contextProjs;
    public List<ConvolutionLayer> residualInputs;
    public ConvolutionLayer downSampleConv;
    private boolean attn = false;
    private boolean crossAttn = false;
    private int tEmbDim;
    private int contextDim = 0;
    private int maxContextLen = 0;
    private boolean downSample;
    private int numHeads;
    private int numLayers = 1;
    private int groups = 32;
    private Tensor[] t_out;
    private Tensor[] res_out;
    private Tensor kvDiff;
    private Tensor dt;

    public UNetDownBlockLayer(int channel, int oChannel, int height, int width, int tEmbDim, boolean downSample, int numHeads, int numLayers, int groups, int contextDim, int maxContextLen, boolean attn, boolean crossAttn, Network network) {
        this.network = network;
        this.channel = channel;
        this.oChannel = oChannel;
        this.height = height;
        this.width = width;
        this.oHeight = height;
        this.oWidth = width;
        this.tEmbDim = tEmbDim;
        this.contextDim = contextDim;
        this.maxContextLen = maxContextLen;
        this.downSample = downSample;
        this.numHeads = numHeads;
        this.numLayers = numLayers;
        this.groups = groups;
        this.attn = attn;
        this.crossAttn = crossAttn;
        initLayers();
    }

    public void initLayers() {
        resnetFirst = new ArrayList<UNetResnetBlockLayer>();
        resnetSecond = new ArrayList<UNetResnetBlockLayer>();
        for (int i = 0; i < numLayers; i++) {
            int ic = oChannel;
            if (i == 0) {
                ic = channel;
            }
            UNetResnetBlockLayer b = new UNetResnetBlockLayer(ic, oChannel, height, width, groups, network);
            resnetFirst.add(b);
        }
        if (tEmbDim > 0) {
            tEmbLayers = new ArrayList<UNetTEmbLayer>();
            for (int i = 0; i < numLayers; i++) {
                UNetTEmbLayer t = new UNetTEmbLayer(tEmbDim, oChannel, network);
                tEmbLayers.add(t);
            }
        }
        for (int i = 0; i < numLayers; i++) {
            UNetResnetBlockLayer b = new UNetResnetBlockLayer(oChannel, oChannel, height, width, groups, network);
            resnetSecond.add(b);
        }
        if (attn) {
            attns = new ArrayList<UNetAttentionLayer>();
            for (int i = 0; i < numLayers; i++) {
                UNetAttentionLayer a = new UNetAttentionLayer(oChannel, numHeads, oChannel, height, width, groups, false, false, true, network);
                attns.add(a);
            }
        }
        if (crossAttn) {
            contextProjs = new ArrayList<FullyLayer>();
            for (int i = 0; i < numLayers; i++) {
                FullyLayer f = new FullyLayer(contextDim, oChannel, true, network);
                contextProjs.add(f);
            }
            crossAttns = new ArrayList<UNetAttentionLayer>();
            for (int i = 0; i < numLayers; i++) {
                UNetAttentionLayer a = new UNetAttentionLayer(oChannel, oChannel, oChannel, numHeads, maxContextLen, oChannel, height, width, groups, false, false, true, network);
                crossAttns.add(a);
            }
        }
        residualInputs = new ArrayList<ConvolutionLayer>();
        for (int i = 0; i < numLayers; i++) {
            int ic = oChannel;
            if (i == 0) {
                ic = channel;
            }
            ConvolutionLayer c = new ConvolutionLayer(ic, oChannel, width, height, 1, 1, 0, 1, false, network);
            c.setUpdater(UpdaterFactory.create(this.network));
            c.paramsInit = ParamsInit.silu;
            residualInputs.add(c);
        }
        if (downSample) {
            downSampleConv = new ConvolutionLayer(oChannel, oChannel, width, height, 4, 4, 1, 2, false, network);
            downSampleConv.setUpdater(UpdaterFactory.create(this.network));
            downSampleConv.paramsInit = ParamsInit.silu;
            this.oHeight = downSampleConv.oHeight;
            this.oWidth = downSampleConv.oWidth;
        }
    }

    @Override
    public void init() {
        // TODO Auto-generated method stub
        this.number = this.network.number;
        if (res_out == null || res_out[0].number != this.number) {
            if (res_out == null) {
                res_out = new Tensor[numLayers];
            }
            for (int i = 0; i < numLayers; i++) {
                res_out[i] = Tensor.createGPUTensor(res_out[i], this.number, oChannel, height, width, true);
            }
        }
        if (tEmbDim > 0 && (t_out == null || t_out[0].number != this.number)) {
            if (t_out == null) {
                t_out = new Tensor[numLayers];
            }
            for (int i = 0; i < numLayers; i++) {
                t_out[i] = Tensor.createGPUTensor(t_out[i], this.number, oChannel, height, width, true);
            }
        }
    }

    public void init(Tensor input) {
        // TODO Auto-generated method stub
        this.number = input.number;
        if (res_out == null || res_out[0].number != this.number) {
            if (res_out == null) {
                res_out = new Tensor[numLayers];
            }
            for (int i = 0; i < numLayers; i++) {
                res_out[i] = Tensor.createGPUTensor(res_out[i], this.number, oChannel, height, width, true);
            }
        }
        if (tEmbDim > 0 && (t_out == null || t_out[0].number != this.number)) {
            if (t_out == null) {
                t_out = new Tensor[numLayers];
            }
            for (int i = 0; i < numLayers; i++) {
                t_out[i] = Tensor.createGPUTensor(t_out[i], this.number, oChannel, height, width, true);
            }
        }
    }

    @Override
    public void initBack() {
        // TODO Auto-generated method stub
        if (crossAttn && (kvDiff == null || kvDiff.number * maxContextLen != this.number * maxContextLen)) {
            kvDiff = Tensor.createGPUTensor(kvDiff, this.number * maxContextLen, 1, 1, oChannel, true);
        }
        if (dt == null || dt.number != this.number) {
            dt = Tensor.createGPUTensor(dt, this.number, 1, 1, oChannel, true);
        }
    }

    @Override
    public void initParam() {
        // TODO Auto-generated method stub
    }

    @Override
    public void output() {
        // TODO Auto-generated method stub
        Tensor x = input;
        for (int i = 0; i < numLayers; i++) {
            Tensor res_i = x;
            resnetFirst.get(i).forward(x);
            resnetSecond.get(i).forward(resnetFirst.get(i).getOutput());
            residualInputs.get(i).forward(res_i);
            Tensor_OP().add(resnetSecond.get(i).getOutput(), residualInputs.get(i).getOutput(), res_out[i]);
            x = res_out[i];
            if (attn) {
                attns.get(i).forward(x);
                x = attns.get(i).getOutput();
            }
        }
        downSampleConv.forward(x);
        this.output = downSampleConv.getOutput();
    }

    public void output(Tensor tembd) {
        // TODO Auto-generated method stub
        Tensor x = input;
        for (int i = 0; i < numLayers; i++) {
            Tensor res_i = x;
            //			System.err.println("----------------");
            resnetFirst.get(i).forward(x);
            //			tembd.showDM();
            tEmbLayers.get(i).forward(tembd);
            Tensor r1 = resnetFirst.get(i).getOutput();
            //			r1.showDM("r1");
            //			tEmbLayers.get(i).linear.weight.showDM("emb-weight");
            //			tEmbLayers.get(i).getOutput().showDM("temb");
            Tensor_OP().add(r1, tEmbLayers.get(i).getOutput(), t_out[i], r1.height * r1.width);
            //			t_out[i].showDM("t_out");
            resnetSecond.get(i).forward(t_out[i]);
            residualInputs.get(i).forward(res_i);
            Tensor_OP().add(resnetSecond.get(i).getOutput(), residualInputs.get(i).getOutput(), res_out[i]);
            x = res_out[i];
            //			x.showDM("attn_in");
            if (attn) {
                attns.get(i).forward(x);
                x = attns.get(i).getOutput();
            }
            //			x.showDM("cx");
        }
        //		x.showDM("down-x");
        downSampleConv.forward(x);
        //		downSampleConv.getOutput().showDM("downSampleConv-x");
        this.output = downSampleConv.getOutput();
    }

    public void output(Tensor tembd, Tensor context) {
        // TODO Auto-generated method stub
        Tensor x = input;
        for (int i = 0; i < numLayers; i++) {
            Tensor res_i = x;
            //			System.err.println("----------------");
            resnetFirst.get(i).forward(x);
            //			tembd.showDM();
            tEmbLayers.get(i).forward(tembd);
            Tensor r1 = resnetFirst.get(i).getOutput();
            //			r1.showDM("r1");
            //			tEmbLayers.get(i).linear.weight.showDM("emb-weight");
            //			tEmbLayers.get(i).getOutput().showDM("temb");
            Tensor_OP().add(r1, tEmbLayers.get(i).getOutput(), t_out[i], r1.height * r1.width);
            //			t_out[i].showDM("t_out");
            resnetSecond.get(i).forward(t_out[i]);
            residualInputs.get(i).forward(res_i);
            Tensor_OP().add(resnetSecond.get(i).getOutput(), residualInputs.get(i).getOutput(), res_out[i]);
            x = res_out[i];
            //			x.showDM("attn_in");
            if (attn) {
                attns.get(i).forward(x);
                x = attns.get(i).getOutput();
            }
            //			x.showDM("ax");
            if (crossAttn) {
                contextProjs.get(i).forward(context);
                Tensor cp = contextProjs.get(i).getOutput();
                crossAttns.get(i).forward(x, cp, cp);
                x = crossAttns.get(i).getOutput();
            }
            //			x.showDM("cx");
        }
        //		x.showDM("down-x");
        downSampleConv.forward(x);
        //		downSampleConv.getOutput().showDM("downSampleConv-x");
        this.output = downSampleConv.getOutput();
    }

    @Override
    public Tensor getOutput() {
        // TODO Auto-generated method stub
        return this.output;
    }

    @Override
    public void diff() {
        // TODO Auto-generated method stub
    }

    public void diff(Tensor tDiff) {
        // TODO Auto-generated method stub
        //		delta.showDM("c+d");
        downSampleConv.back(delta);
        Tensor d = downSampleConv.diff;
        //		d.showDM("downSampleConv.diff");
        for (int i = numLayers - 1; i >= 0; i--) {
            if (crossAttn) {
                crossAttns.get(i).back(d, kvDiff);
                contextProjs.get(i).back(kvDiff);
                d = crossAttns.get(i).diff;
            }
            if (attn) {
                attns.get(i).back(d);
                d = attns.get(i).diff;
            }
            //			d.showDM("attn-down");
            residualInputs.get(i).back(d);
            resnetSecond.get(i).back(d);
            d = resnetSecond.get(i).diff;
            //			d.showDM("tout");
            if (tEmbDim > 0) {
                dt.clearGPU();
                Tensor_OP().sum(d, dt, 2);
                tEmbLayers.get(i).back(dt);
                Tensor_OP().add(tDiff, tEmbLayers.get(i).diff, tDiff);
            }
            resnetFirst.get(i).back(d);
            //			resnetFirst.get(i).conv.diff.showDM("afout");
            d = resnetFirst.get(i).diff;
            //			d.showDM("resnetFirst");
            //			residualInputs.get(i).diff.showDM("residualInputs");
            Tensor_OP().add(d, residualInputs.get(i).diff, d);
            //			d.showDM("d-["+i+"]");
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

    public void forward(Tensor input, Tensor tembd) {
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
        this.output(tembd);
    }

    public void forward(Tensor input, Tensor tembd, Tensor context) {
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
        this.output(tembd, context);
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

    public void back(Tensor delta, Tensor tDiff) {
        // TODO Auto-generated method stub
        initBack();
        /**
         * 设置梯度

         */
        this.setDelta(delta);
        /**
         * 计算梯度

         */
        this.diff(tDiff);
    }

    @Override
    public void update() {
        // TODO Auto-generated method stub
        for (int i = 0; i < numLayers; i++) {
            resnetFirst.get(i).update();
            if (tEmbDim > 0) {
                tEmbLayers.get(i).update();
            }
            resnetSecond.get(i).update();
            residualInputs.get(i).update();
            if (attn) {
                attns.get(i).update();
            }
            if (crossAttn) {
                contextProjs.get(i).update();
                crossAttns.get(i).update();
            }
        }
        if (downSample) {
            downSampleConv.update();
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
        for (int i = 0; i < numLayers; i++) {
            resnetFirst.get(i).accGrad(scale);
            if (tEmbDim > 0) {
                tEmbLayers.get(i).accGrad(scale);
            }
            resnetSecond.get(i).accGrad(scale);
            residualInputs.get(i).accGrad(scale);
            if (attn) {
                attns.get(i).accGrad(scale);
            }
            if (crossAttn) {
                contextProjs.get(i).accGrad(scale);
                crossAttns.get(i).accGrad(scale);
            }
        }
        downSampleConv.accGrad(scale);
    }
}

