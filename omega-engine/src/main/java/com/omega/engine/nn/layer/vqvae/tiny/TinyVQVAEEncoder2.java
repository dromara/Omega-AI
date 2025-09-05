package com.omega.engine.nn.layer.vqvae.tiny;

import com.omega.engine.nn.layer.ConvolutionLayer;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.active.SiLULayer;
import com.omega.engine.nn.layer.normalization.BNType;
import com.omega.engine.nn.layer.normalization.GNLayer;
import com.omega.engine.nn.network.Network;
import com.omega.engine.tensor.Tensor;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.util.ArrayList;
import java.util.List;

/**
 * VQVAEDecoder
 *
 * @author Administrator
 */
public class TinyVQVAEEncoder2 extends Layer {
    private int num_res_blocks = 2;
    private int groups = 32;
    private int headNum;
    private int ch;
    private int[] ch_mult;
    public ConvolutionLayer convIn;
    public List<Layer> down;
    public GNLayer convNormOut;
    public SiLULayer convAct;
    public ConvolutionLayer convOut;
    private boolean hasAttn = true;

    public TinyVQVAEEncoder2(int channel, int oChannel, int height, int width, int num_res_blocks, int groups, int headNum, int[] ch_mult, int ch, Network network) {
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
    
    public TinyVQVAEEncoder2(int channel, int oChannel, int height, int width, int num_res_blocks, int groups, int headNum, int[] ch_mult, int ch, boolean hasAttn, Network network) {
        this.network = network;
        this.channel = channel;
        this.oChannel = oChannel;
        this.height = height;
        this.width = width;
        this.groups = groups;
        this.headNum = headNum;
        this.ch_mult = ch_mult;
        this.ch = ch;
        this.hasAttn = hasAttn;
        this.num_res_blocks = num_res_blocks;
        initLayers();
    }

    public void initLayers() {
        down = new ArrayList<Layer>();
        convIn = new ConvolutionLayer(channel, ch, width, height, 3, 3, 1, 1, true, this.network);
        int[] in_ch_mult = new int[ch_mult.length + 1];
        in_ch_mult[0] = 1;
        System.arraycopy(ch_mult, 0, in_ch_mult, 1, ch_mult.length);
        int c_out = 0;
        int ih = convIn.oHeight;
        int iw = convIn.oWidth;
        for (int i = 0; i < ch_mult.length; i++) {
            int c_in = in_ch_mult[i] * ch;
            c_out = in_ch_mult[i + 1] * ch;
            for (int ri = 0; ri < num_res_blocks; ri++) {
                VQVAEResidual res = new VQVAEResidual(c_in, c_out, ih, iw, this.groups, network);
                down.add(res);
                c_in = c_out;
                ih = res.oHeight;
                iw = res.oWidth;
                if (hasAttn && i == ch_mult.length - 1) {
                    VQVAEAttentionLayer2 attn = new VQVAEAttentionLayer2(c_out, headNum, ih, iw, groups, false, network);
                    down.add(attn);
                }
            }
            if (i != ch_mult.length - 1) {
                ConvolutionLayer downConv = new ConvolutionLayer(c_out, c_out, iw, ih, 3, 3, 1, 2, true, this.network);
                down.add(downConv);
                ih = downConv.oHeight;
                iw = downConv.oWidth;
            }
        }
        //middle
        VQVAEResidual res1 = new VQVAEResidual(c_out, c_out, ih, iw, this.groups, network);
        down.add(res1);
        VQVAEAttentionLayer2 attn = new VQVAEAttentionLayer2(c_out, headNum, ih, iw, groups, false, network);
        down.add(attn);
        VQVAEResidual res2 = new VQVAEResidual(c_out, c_out, ih, iw, this.groups, network);
        down.add(res2);
        convNormOut = new GNLayer(groups, c_out, ih, iw, BNType.conv_bn, res2);
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
//        		x.showShape("en-x");
        for (int i = 0; i < down.size(); i++) {
            Layer l = down.get(i);
            l.forward(x);
            x = l.getOutput();
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
            if(l instanceof ConvolutionLayer) {
            	ConvolutionLayer conv = (ConvolutionLayer) l;
            	conv.back(d, conv.input);
            }else {
            	l.back(d);
            }
            d = l.diff;
        }
        //		d.showDM("d");
        convIn.back(d);
        //		convIn.diff.showDMByOffset(0, 100, "encode_conv_in");
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
        //		convIn.weight.showDM("convIn.weight");
        //		convIn.bias.showDM("convIn.bias");
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

