package com.omega.engine.nn.layer.vqvae;

import com.omega.common.data.Tensor;
import com.omega.engine.nn.layer.ConvolutionLayer;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.ParamsInit;
import com.omega.engine.nn.layer.active.SiLULayer;
import com.omega.engine.nn.layer.active.TanhLayer;
import com.omega.engine.nn.layer.normalization.BNType;
import com.omega.engine.nn.layer.normalization.GNLayer;
import com.omega.engine.nn.network.Network;
import com.omega.engine.updater.UpdaterFactory;

import java.util.ArrayList;
import java.util.List;

/**
 * VQVAEDecoder
 *
 * @author Administrator
 */
public class VQVAEDecoder extends Layer {
    private int numLayers;
    private int groups = 32;
    private int headNum;
    private int[] downChannels;
    private boolean[] downSample;
    private int[] midChannels;
    private ConvolutionLayer convIn;
    private List<VQVAEMidBlock> midBlock;
    private List<VQVAEUpBlock> upBlock;
    private GNLayer convNormOut;
    private SiLULayer convAct;
    private ConvolutionLayer convOut;
    private TanhLayer outAct;

    public VQVAEDecoder(int channel, int oChannel, int height, int width, int numLayers, int groups, int headNum, int[] downChannels, boolean[] downSample, int[] midChannels, Network network) {
        this.network = network;
        this.channel = channel;
        this.oChannel = oChannel;
        this.height = height;
        this.width = width;
        this.groups = groups;
        this.headNum = headNum;
        this.downChannels = downChannels;
        this.midChannels = midChannels;
        this.downSample = downSample;
        this.numLayers = numLayers;
        initLayers();
    }

    public void initLayers() {
        convIn = new ConvolutionLayer(channel, midChannels[midChannels.length - 1], width, height, 3, 3, 1, 1, true, this.network);
        convIn.setUpdater(UpdaterFactory.create(this.network));
        convIn.paramsInit = ParamsInit.silu;
        // mid
        int outc = midChannels[midChannels.length - 1];
        int ih = convIn.oHeight;
        int iw = convIn.oWidth;
        midBlock = new ArrayList<VQVAEMidBlock>(midChannels.length - 1);
        for (int i = midChannels.length - 1; i >= 1; i--) {
            outc = midChannels[i - 1];
            VQVAEMidBlock mid = new VQVAEMidBlock(midChannels[i], outc, ih, iw, numLayers, groups, headNum, 1.0f, true, network);
            ih = mid.oHeight;
            iw = mid.oWidth;
            midBlock.add(mid);
        }
        // up
        upBlock = new ArrayList<VQVAEUpBlock>(downChannels.length - 1);
        for (int i = downChannels.length - 1; i >= 1; i--) {
            outc = downChannels[i - 1];
            VQVAEUpBlock up = new VQVAEUpBlock(downChannels[i], outc, ih, iw, numLayers, groups, downSample[i - 1], network);
            ih = up.oHeight;
            iw = up.oWidth;
            upBlock.add(up);
        }
        //out
        convNormOut = new GNLayer(groups, upBlock.get(upBlock.size() - 1), BNType.conv_bn);
        convAct = new SiLULayer(convNormOut);
        convOut = new ConvolutionLayer(outc, oChannel, iw, ih, 3, 3, 1, 1, true, this.network);
        convOut.setUpdater(UpdaterFactory.create(this.network));
        convOut.paramsInit = ParamsInit.silu;
        outAct = new TanhLayer(convOut);
        this.oHeight = convOut.oHeight;
        this.oWidth = convOut.oWidth;
    }

    @Override
    public void init() {
        this.number = this.network.number;
    }

    @Override
    public void initBack() {
    }

    @Override
    public void initParam() {
        // TODO Auto-generated method stub
    }

    @Override
    public void output() {
        // TODO Auto-generated method stub
        convIn.forward(this.input);
        Tensor x = convIn.getOutput();
        for (int i = 0; i < midBlock.size(); i++) {
            VQVAEMidBlock mid = midBlock.get(i);
            mid.forward(x);
            x = mid.getOutput();
        }
        for (int i = 0; i < upBlock.size(); i++) {
            VQVAEUpBlock up = upBlock.get(i);
            up.forward(x);
            x = up.getOutput();
        }
        convNormOut.forward(x);
        convAct.forward(convNormOut.getOutput());
        convOut.forward(convAct.getOutput());
        outAct.forward(convOut.getOutput());
        this.output = outAct.getOutput();
    }

    @Override
    public Tensor getOutput() {
        // TODO Auto-generated method stub
        return this.output;
    }

    @Override
    public void diff() {
        // TODO Auto-generated method stub
        outAct.back(delta);
        convOut.back(outAct.diff);
        convAct.back(convOut.diff);
        convNormOut.back(convAct.diff);
        Tensor d = convNormOut.diff;
        for (int i = upBlock.size() - 1; i >= 0; i--) {
            VQVAEUpBlock up = upBlock.get(i);
            up.back(d);
            d = up.diff;
        }
        for (int i = midBlock.size() - 1; i >= 0; i--) {
            VQVAEMidBlock mid = midBlock.get(i);
            mid.back(d);
            d = mid.diff;
        }
        convIn.back(d);
        this.diff = convIn.diff;
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
    public void update() {
        // TODO Auto-generated method stub
        convIn.update();
        for (int i = 0; i < midBlock.size(); i++) {
            midBlock.get(i).update();
        }
        for (int i = 0; i < upBlock.size(); i++) {
            upBlock.get(i).update();
        }
        convNormOut.update();
        convOut.update();
    }

    @Override
    public void showDiff() {
        // TODO Auto-generated method stub
    }

    @Override
    public LayerType getLayerType() {
        // TODO Auto-generated method stub
        return LayerType.block;
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
    public void forward(Tensor input) {
        // TODO Auto-generated method stub
        /**
         * 参数初始化

         */
        this.init();
        /**
         * 设置输入

         */
        this.setInput(input);
        /**
         * 计算输出

         */
        this.output();
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

    @Override
    public void backTemp() {
        // TODO Auto-generated method stub
    }

    @Override
    public void accGrad(float scale) {
        // TODO Auto-generated method stub
    }
}

