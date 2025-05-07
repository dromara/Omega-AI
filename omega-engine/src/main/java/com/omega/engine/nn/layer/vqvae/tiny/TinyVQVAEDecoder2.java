package com.omega.engine.nn.layer.vqvae.tiny;

import com.omega.common.data.Tensor;
import com.omega.engine.nn.layer.ConvolutionLayer;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.active.SiLULayer;
import com.omega.engine.nn.layer.normalization.BNType;
import com.omega.engine.nn.layer.normalization.GNLayer;
import com.omega.engine.nn.network.Network;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.util.ArrayList;
import java.util.List;

/**
 * VQVAEDecoder
 *
 * @author Administrator
 */
public class TinyVQVAEDecoder2 extends Layer {
    private int num_res_blocks = 2;
    private int groups = 32;
    private int headNum;
    private int[] ch_mult;
    private int ch;
    private ConvolutionLayer convIn;
    private List<Layer> up;
    private GNLayer convNormOut;
    private SiLULayer convAct;
    private ConvolutionLayer convOut;

    public TinyVQVAEDecoder2(int channel, int oChannel, int height, int width, int num_res_blocks, int groups, int headNum, int[] ch_mult, int ch, Network network) {
        this.network = network;
        this.channel = channel;
        this.oChannel = oChannel;
        this.height = height;
        this.width = width;
        this.groups = groups;
        this.headNum = headNum;
        this.ch_mult = ch_mult;
        this.ch = ch;
        this.num_res_blocks = num_res_blocks;
        initLayers();
    }

    public void initLayers() {
        up = new ArrayList<Layer>();
        int c_in = ch * ch_mult[ch_mult.length - 1];
        convIn = new ConvolutionLayer(channel, c_in, width, height, 3, 3, 1, 1, true, this.network);
        int ih = convIn.oHeight;
        int iw = convIn.oWidth;
        //middle
        VQVAEResidual res1 = new VQVAEResidual(c_in, c_in, ih, iw, this.groups, network);
        up.add(res1);
        VQVAEAttentionLayer2 attn = new VQVAEAttentionLayer2(c_in, headNum, ih, iw, groups, false, network);
        up.add(attn);
        VQVAEResidual res2 = new VQVAEResidual(c_in, c_in, ih, iw, this.groups, network);
        up.add(res2);
        // up
        int c_out = 0;
        for (int i = ch_mult.length - 1; i >= 0; i--) {
            c_out = ch_mult[i] * ch;
            for (int ri = 0; ri < num_res_blocks; ri++) {
                VQVAEResidual res = new VQVAEResidual(c_in, c_out, ih, iw, this.groups, network);
                up.add(res);
                c_in = c_out;
                ih = res.oHeight;
                iw = res.oWidth;
                if (i == ch_mult.length - 1) {
                    VQVAEAttentionLayer2 rattn = new VQVAEAttentionLayer2(c_out, headNum, ih, iw, groups, false, network);
                    up.add(rattn);
                }
            }
            if (i != 0) {
                VQVAEUpsample upsample = new VQVAEUpsample(c_out, ih, iw, network);
                up.add(upsample);
                ih = upsample.oHeight;
                iw = upsample.oWidth;
            }
        }
        Layer lastLayer = up.get(up.size() - 1);
        convNormOut = new GNLayer(groups, c_out, ih, iw, BNType.conv_bn, lastLayer);
        convAct = new SiLULayer(convNormOut);
        convOut = new ConvolutionLayer(c_out, oChannel, iw, ih, 3, 3, 1, 1, true, this.network);
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
        for (int i = 0; i < up.size(); i++) {
            Layer l = up.get(i);
            l.forward(x);
            x = l.getOutput();
            //			System.err.println(l);
            //			x.showDMByOffsetRed(0, 100, "x"+i);
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
        for (int i = up.size() - 1; i >= 0; i--) {
            Layer l = up.get(i);
            if(l instanceof ConvolutionLayer) {
            	ConvolutionLayer conv = (ConvolutionLayer) l;
            	conv.back(d, conv.input);
            }else {
            	l.back(d);
            }
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
        for (int i = 0; i < up.size(); i++) {
            up.get(i).update();
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
        for (int i = 0; i < up.size(); i++) {
            Layer l = up.get(i);
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
            if (l instanceof VQVAEUpsample) {
                VQVAEUpsample u = (VQVAEUpsample) l;
                u.saveModel(outputStream);
            }
        }
        convNormOut.saveModel(outputStream);
        convOut.saveModel(outputStream);
    }

    public void loadModel(RandomAccessFile inputStream) throws IOException {
        convIn.loadModel(inputStream);
        for (int i = 0; i < up.size(); i++) {
            Layer l = up.get(i);
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
            if (l instanceof VQVAEUpsample) {
                VQVAEUpsample u = (VQVAEUpsample) l;
                u.loadModel(inputStream);
            }
        }
        convNormOut.loadModel(inputStream);
        convOut.loadModel(inputStream);
    }
}

