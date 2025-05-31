package com.omega.engine.nn.layer.vqvae.tiny;

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

import java.io.IOException;
import java.io.RandomAccessFile;
import java.util.ArrayList;
import java.util.List;

/**
 * VQVAEDecoder
 *
 * @author Administrator
 */
public class TinyVQVAEEncoder extends Layer {
    private int num_res_blocks;
    private int groups = 32;
    private int headNum;
    private int[] channels;
    private boolean[] attn_resolutions;
    private ConvolutionLayer convIn;
    private List<Layer> down;
    private GNLayer convNormOut;
    private SiLULayer convAct;
    private ConvolutionLayer convOut;

    public TinyVQVAEEncoder(int channel, int oChannel, int height, int width, int num_res_blocks, int groups, int headNum, int[] channels, boolean[] attn_resolutions, Network network) {
        this.network = network;
        this.channel = channel;
        this.oChannel = oChannel;
        this.height = height;
        this.width = width;
        this.groups = groups;
        this.headNum = headNum;
        this.channels = channels;
        this.attn_resolutions = attn_resolutions;
        this.num_res_blocks = num_res_blocks;
        initLayers();
    }

    public void initLayers() {
        down = new ArrayList<Layer>();
        convIn = new ConvolutionLayer(channel, channels[0], width, height, 3, 3, 1, 1, true, this.network);
        convIn.setUpdater(UpdaterFactory.create(this.network));
        convIn.paramsInit = ParamsInit.silu;
        int ih = convIn.oHeight;
        int iw = convIn.oWidth;
        for (int i = 0; i < channels.length - 1; i++) {
            int c_in = channels[i];
            int c_out = channels[i + 1];
            for (int ri = 0; ri < num_res_blocks; ri++) {
                VQVAEResidual res = new VQVAEResidual(c_in, c_out, ih, iw, this.groups, network);
                down.add(res);
                c_in = c_out;
                ih = res.oHeight;
                iw = res.oWidth;
            }
            if (attn_resolutions[i]) {
                VQVAEAttentionLayer2 attn = new VQVAEAttentionLayer2(c_out, headNum, ih, iw, groups, false, network);
                down.add(attn);
            }
            ConvolutionLayer downConv = new ConvolutionLayer(c_out, c_out, iw, ih, 3, 3, 1, 2, true, this.network);
            downConv.setUpdater(UpdaterFactory.create(this.network));
            downConv.paramsInit = ParamsInit.silu;
            down.add(downConv);
            ih = downConv.oHeight;
            iw = downConv.oWidth;
        }
        //Bottleneck
        for (int i = 0; i < num_res_blocks; i++) {
            VQVAEResidual res = new VQVAEResidual(channels[channels.length - 1], channels[channels.length - 1], ih, iw, this.groups, network);
            down.add(res);
            ih = res.oHeight;
            iw = res.oWidth;
        }
        VQVAEAttentionLayer2 attn = new VQVAEAttentionLayer2(channels[channels.length - 1], headNum, ih, iw, groups, false, network);
        down.add(attn);
        Layer lastLayer = null;
        for (int i = 0; i < num_res_blocks; i++) {
            VQVAEResidual res = new VQVAEResidual(channels[channels.length - 1], channels[channels.length - 1], ih, iw, this.groups, network);
            down.add(res);
            ih = res.oHeight;
            iw = res.oWidth;
            lastLayer = res;
        }
        convNormOut = new GNLayer(groups, lastLayer, BNType.conv_bn);
        convAct = new SiLULayer(convNormOut);
        convOut = new ConvolutionLayer(channels[channels.length - 1], oChannel, iw, ih, 3, 3, 1, 1, true, this.network);
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
        //		x.showDMByOffset(0, 10, "convIn");
        for (int i = 0; i < down.size(); i++) {
            Layer l = down.get(i);
            l.forward(x);
            x = l.getOutput();
            //			x.showDMByOffset(0, 10, i+"");
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
        for (int i = down.size() - 1; i >= 0; i--) {
            Layer l = down.get(i);
            l.back(d);
            d = l.diff;
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
        for (int i = 0; i < down.size(); i++) {
            down.get(i).update();
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

    public void saveModel(RandomAccessFile outputStream) throws IOException {
        convIn.saveModel(outputStream);
        for (int i = 0; i < down.size(); i++) {
            Layer l = down.get(i);
            if (l instanceof VQVAEResidual) {
                VQVAEResidual r = (VQVAEResidual) l;
                r.saveModel(outputStream);
            }
            if (l instanceof VQVAEAttentionLayer2) {
                VQVAEAttentionLayer2 a = (VQVAEAttentionLayer2) l;
                a.saveModel(outputStream);
            }
            if (l instanceof ConvolutionLayer) {
                ConvolutionLayer c = (ConvolutionLayer) l;
                c.saveModel(outputStream);
            }
        }
        convNormOut.saveModel(outputStream);
        convOut.saveModel(outputStream);
    }

    public void loadModel(RandomAccessFile inputStream) throws IOException {
        convIn.loadModel(inputStream);
        for (int i = 0; i < down.size(); i++) {
            Layer l = down.get(i);
            if (l instanceof VQVAEResidual) {
                VQVAEResidual r = (VQVAEResidual) l;
                r.loadModel(inputStream);
            }
            if (l instanceof VQVAEAttentionLayer2) {
                VQVAEAttentionLayer2 a = (VQVAEAttentionLayer2) l;
                a.loadModel(inputStream);
            }
            if (l instanceof ConvolutionLayer) {
                ConvolutionLayer c = (ConvolutionLayer) l;
                c.loadModel(inputStream);
            }
        }
        convNormOut.loadModel(inputStream);
        convOut.loadModel(inputStream);
    }
}

