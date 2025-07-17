package com.omega.engine.nn.layer.sd_vae.moudles;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.util.ArrayList;
import java.util.List;

import com.omega.engine.nn.layer.ConvolutionLayer;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.active.SiLULayer;
import com.omega.engine.nn.layer.normalization.BNType;
import com.omega.engine.nn.layer.normalization.GNLayer;
import com.omega.engine.nn.network.Network;
import com.omega.engine.tensor.Tensor;

/**
 * VQVAEDecoder
 *
 * @author Administrator
 */
public class SDVAEDecoder extends Layer {
    private int num_res_blocks = 2;
    private int groups = 32;
    private int headNum;
    private int[] ch_mult;
    private int ch;
    public ConvolutionLayer convIn;
    public List<Layer> up;
    public GNLayer convNormOut;
    private SiLULayer convAct;
    public ConvolutionLayer convOut;
    private boolean hasAttn;
    
    public SDVAEDecoder(int channel, int oChannel, int height, int width, int num_res_blocks, int groups, int headNum, int[] ch_mult, int ch, Network network) {
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
    
    public SDVAEDecoder(int channel, int oChannel, int height, int width, int num_res_blocks, int groups, int headNum, int[] ch_mult, int ch, boolean hasAttn, Network network) {
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
        up = new ArrayList<Layer>();
        int c_in = ch * ch_mult[ch_mult.length - 1];
        convIn = new ConvolutionLayer(channel, c_in, width, height, 3, 3, 1, 1, true, this.network);
        int ih = convIn.oHeight;
        int iw = convIn.oWidth;
        //middle
        SDVAEResidual res1 = new SDVAEResidual(c_in, c_in, ih, iw, this.groups, network);
        up.add(res1);
        SDVAEAttentionLayer attn = new SDVAEAttentionLayer(c_in, headNum, ih, iw, groups, true, network);
        up.add(attn);
        SDVAEResidual res2 = new SDVAEResidual(c_in, c_in, ih, iw, this.groups, network);
        up.add(res2);
        // up
        int c_out = 0;
        for (int i = ch_mult.length - 1; i >= 0; i--) {
            c_out = ch_mult[i] * ch;
            for (int ri = 0; ri < num_res_blocks + 1; ri++) {
                SDVAEResidual res = new SDVAEResidual(c_in, c_out, ih, iw, this.groups, network);
                up.add(res);
                c_in = c_out;
                ih = res.oHeight;
                iw = res.oWidth;
                if (hasAttn && i == ch_mult.length - 1) {
                    SDVAEAttentionLayer rattn = new SDVAEAttentionLayer(c_out, headNum, ih, iw, groups, true, network);
                    up.add(rattn);
                }
            }
            if (i != 0) {
                SDVAEUpsample upsample = new SDVAEUpsample(c_out, ih, iw, network);
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
            if (l instanceof SDVAEResidual) {
                SDVAEResidual r = (SDVAEResidual) l;
                r.saveModel(outputStream);
            }
            if (l instanceof SDVAEAttentionLayer) {
                SDVAEAttentionLayer a = (SDVAEAttentionLayer) l;
                a.saveModel(outputStream);
            }
            if (l instanceof ConvolutionLayer) {
                ConvolutionLayer c = (ConvolutionLayer) l;
                c.saveModel(outputStream);
            }
            if (l instanceof SDVAEUpsample) {
                SDVAEUpsample u = (SDVAEUpsample) l;
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
            if (l instanceof SDVAEResidual) {
                SDVAEResidual r = (SDVAEResidual) l;
                r.loadModel(inputStream);
            }
            if (l instanceof SDVAEAttentionLayer) {
                SDVAEAttentionLayer a = (SDVAEAttentionLayer) l;
                a.loadModel(inputStream);
            }
            if (l instanceof ConvolutionLayer) {
                ConvolutionLayer c = (ConvolutionLayer) l;
                c.loadModel(inputStream);
            }
            if (l instanceof SDVAEUpsample) {
                SDVAEUpsample u = (SDVAEUpsample) l;
                u.loadModel(inputStream);
            }
        }
        convNormOut.loadModel(inputStream);
        convOut.loadModel(inputStream);
    }
}

