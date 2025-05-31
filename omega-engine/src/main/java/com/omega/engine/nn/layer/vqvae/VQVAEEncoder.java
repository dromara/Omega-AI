package com.omega.engine.nn.layer.vqvae;

import com.omega.engine.nn.layer.ConvolutionLayer;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.ParamsInit;
import com.omega.engine.nn.layer.active.SiLULayer;
import com.omega.engine.nn.layer.normalization.BNType;
import com.omega.engine.nn.layer.normalization.GNLayer;
import com.omega.engine.nn.network.Network;
import com.omega.engine.tensor.Tensor;
import com.omega.engine.updater.UpdaterFactory;

import java.util.ArrayList;
import java.util.List;

/**
 * VQVAEEncoder
 *
 * @author Administrator
 */
public class VQVAEEncoder extends Layer {
    private int numLayers;
    private int groups = 32;
    private int headNum;
    private int[] downChannels;
    private boolean[] downSample;
    private int[] midChannels;
    private ConvolutionLayer convIn;
    private List<VQVAEDownBlock> downBlock;
    private List<VQVAEMidBlock> midBlock;
    private GNLayer convNormOut;
    private SiLULayer convAct;
    private ConvolutionLayer convOut;

    public VQVAEEncoder(int channel, int oChannel, int height, int width, int numLayers, int groups, int headNum, int[] downChannels, boolean[] downSample, int[] midChannels, Network network) {
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
        convIn = new ConvolutionLayer(channel, downChannels[0], width, height, 3, 3, 1, 1, true, this.network);
        convIn.setUpdater(UpdaterFactory.create(this.network));
        convIn.paramsInit = ParamsInit.silu;
        downBlock = new ArrayList<VQVAEDownBlock>(downChannels.length - 1);
        int outc = downChannels[0];
        int ih = convIn.oHeight;
        int iw = convIn.oWidth;
        for (int i = 0; i < downChannels.length - 1; i++) {
            int inc = outc;
            outc = downChannels[i + 1];
            VQVAEDownBlock down = new VQVAEDownBlock(inc, outc, ih, iw, numLayers, groups, downSample[i], network);
            ih = down.oHeight;
            iw = down.oWidth;
            downBlock.add(down);
        }
        // mid
        midBlock = new ArrayList<VQVAEMidBlock>(midChannels.length - 1);
        for (int i = 0; i < midChannels.length - 1; i++) {
            outc = midChannels[i + 1];
            VQVAEMidBlock mid = new VQVAEMidBlock(midChannels[i], midChannels[i + 1], ih, iw, numLayers, groups, headNum, 1.0f, true, network);
            ih = mid.oHeight;
            iw = mid.oWidth;
            midBlock.add(mid);
        }
        //out
        convNormOut = new GNLayer(groups, midBlock.get(midBlock.size() - 1), BNType.conv_bn);
        convAct = new SiLULayer(convNormOut);
        convOut = new ConvolutionLayer(outc, oChannel, iw, ih, 3, 3, 1, 1, true, this.network);
        convOut.setUpdater(UpdaterFactory.create(this.network));
        convOut.paramsInit = ParamsInit.silu;
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
        for (int i = 0; i < downBlock.size(); i++) {
            VQVAEDownBlock down = downBlock.get(i);
            down.forward(x);
            x = down.getOutput();
        }
        for (int i = 0; i < midBlock.size(); i++) {
            VQVAEMidBlock mid = midBlock.get(i);
            mid.forward(x);
            x = mid.getOutput();
        }
        convNormOut.forward(x);
        convAct.forward(convNormOut.getOutput());
        convOut.forward(convAct.getOutput());
        this.output = convOut.getOutput();
    }

    @Override
    public Tensor getOutput() {
        // TODO Auto-generated method stub
        return this.output;
    }

    @Override
    public void diff() {
        // TODO Auto-generated method stub
        convOut.back(delta);
        convAct.back(convOut.diff);
        convNormOut.back(convAct.diff);
        Tensor d = convNormOut.diff;
        for (int i = midBlock.size() - 1; i >= 0; i--) {
            VQVAEMidBlock mid = midBlock.get(i);
            mid.back(d);
            d = mid.diff;
        }
        for (int i = downBlock.size() - 1; i >= 0; i--) {
            VQVAEDownBlock down = downBlock.get(i);
            down.back(d);
            d = down.diff;
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
        for (int i = 0; i < downBlock.size(); i++) {
            downBlock.get(i).update();
        }
        for (int i = 0; i < midBlock.size(); i++) {
            midBlock.get(i).update();
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

