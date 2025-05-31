package com.omega.engine.nn.layer.diffusion;

import com.omega.common.utils.MatrixUtils;
import com.omega.engine.gpu.CUDAMemoryManager;
import com.omega.engine.gpu.CUDAModules;
import com.omega.engine.nn.layer.ConvolutionLayer;
import com.omega.engine.nn.layer.FullyLayer;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.active.SiLULayer;
import com.omega.engine.nn.layer.normalization.BNType;
import com.omega.engine.nn.layer.normalization.GNLayer;
import com.omega.engine.nn.network.Network;
import com.omega.engine.nn.network.Transformer;
import com.omega.engine.tensor.Tensor;

import java.io.IOException;
import java.io.RandomAccessFile;

public class ResidualBlockLayer extends Layer {
    public Layer[] block1;
    public Layer[] temb_proj;
    public Layer[] block2;
    public ConvolutionLayer shortcut;
    public DiffusionAttentionBlockLayer attn;
    private boolean hasAttn = true;
    private boolean bias = true;
    private int t_dim;
    private Tensor h;
    private Tensor g;
    private Tensor dt;

    public ResidualBlockLayer(int channel, int oChannel, int height, int width, int t_dim, boolean hasAttn, boolean bias, Network network) {
        this.network = network;
        this.bias = bias;
        this.hasAttn = hasAttn;
        this.t_dim = t_dim;
        this.channel = channel;
        this.oChannel = oChannel;
        this.height = height;
        this.width = width;
        this.oHeight = height;
        this.oWidth = width;
        initLayers();
    }

    public static void main(String[] args) {
        try {
            CUDAModules.initContext();
            int N = 2;
            int T = 1000;
            int d_model = 4;
            int dim = d_model * 4;
            int H = 4;
            int W = 4;
            int ic = 64;
            int oc = 32;
            float[] data = new float[]{100, 200};
            Tensor input = new Tensor(N, 1, 1, 1, data, true);
            float[] data2 = MatrixUtils.order(N * ic * H * W, 0.1f, 0.1f);
            Tensor input2 = new Tensor(N, ic, H, W, data2, true);
            float[] data_d = MatrixUtils.order(N * dim, 0.01f, 0.01f);
            Tensor delta = new Tensor(N, 1, 1, dim, data_d, true);
            Transformer tf = new Transformer();
            tf.CUDNN = true;
            tf.number = 2;
            ResidualBlockLayer rbl = new ResidualBlockLayer(ic, oc, H, W, dim, true, false, tf);
            TimeEmbeddingLayer mal = new TimeEmbeddingLayer(T, d_model, dim, false, tf);
            mal.forward(input);
            mal.getOutput().showShape();
            mal.getOutput().showDM();
            rbl.forward(input2, mal.getOutput());
            rbl.getOutput().showShape();
            rbl.getOutput().showDM();
            mal.back(delta);
            //
            //	  		mal.diff.showDM();
            float[] data_d2 = MatrixUtils.order(N * oc * H * W, 0.01f, 0.01f);
            Tensor delta2 = new Tensor(N, oc, H, W, data_d2, true);
            rbl.back(delta2);
            rbl.diff.showDM();
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        } finally {
            // TODO: handle finally clause
            CUDAMemoryManager.free();
        }
    }

    public void initLayers() {
        block1 = new Layer[3];
        block1[0] = new GNLayer(32, network, BNType.conv_bn);
        block1[1] = new SiLULayer(block1[0]);
        block1[2] = new ConvolutionLayer(channel, oChannel, width, height, 3, 3, 1, 1, bias, network);
        //		block1[2].weight = new Tensor(oChannel, channel, 3, 3, MatrixUtils.order(block1[2].weight.dataLength, 0.01f, 0.01f), true);
        temb_proj = new Layer[2];
        temb_proj[0] = new SiLULayer(network);
        temb_proj[1] = new FullyLayer(t_dim, oChannel, bias, network);
        //		temb_proj[1].weight = new Tensor(1, 1, oChannel, t_dim, MatrixUtils.order(oChannel * t_dim, 0.01f, 0.01f), true);
        block2 = new Layer[3];
        block2[0] = new GNLayer(32, network, BNType.conv_bn);
        block2[1] = new SiLULayer(block2[0]);
        block2[2] = new ConvolutionLayer(oChannel, oChannel, width, height, 3, 3, 1, 1, bias, network);
        //		block2[2].weight = new Tensor(oChannel, oChannel, 3, 3, MatrixUtils.order(block2[2].weight.dataLength, 0.01f, 0.01f), true);
        if (channel != oChannel) {
            shortcut = new ConvolutionLayer(channel, oChannel, width, height, 1, 1, 0, 1, bias, network);
            //			shortcut.weight = new Tensor(oChannel, channel, 3, 3, MatrixUtils.order(shortcut.weight.dataLength, 0.01f, 0.01f), true);
        }
        if (hasAttn) {
            attn = new DiffusionAttentionBlockLayer(oChannel, width, height, bias, false, network);
        }
    }

    @Override
    public void init() {
        // TODO Auto-generated method stub
        this.number = this.network.number;
        initParam();
    }

    @Override
    public void initBack() {
        // TODO Auto-generated method stub
        if (dt == null || dt.number != this.number) {
            dt = Tensor.createGPUTensor(dt, this.number, 1, 1, oChannel, true);
        }
    }

    @Override
    public void initParam() {
        // TODO Auto-generated method stub
        if (h == null || h.number != this.number) {
            h = Tensor.createGPUTensor(h, this.number, oChannel, height, width, true);
            g = Tensor.createGPUTensor(g, this.number, oChannel, height, width, true);
        }
    }

    @Override
    public void output() {
        // TODO Auto-generated method stub
    }

    public void output(Tensor t) {
        // TODO Auto-generated method stub
        //		System.err.println("x:");
        //		input.showDM();
        /**
         * block1

         */
        //		System.out.println("input:"+MatrixOperation.isNaN(input.syncHost()));
        block1[0].forward(input);
        //		System.err.println("gn:");
        //		block1[0].getOutput().showDM();
        //		System.out.println(MatrixOperation.isNaN(block1[0].getOutput().syncHost()));
        block1[1].forward(block1[0].getOutput());
        //		System.out.println(MatrixOperation.isNaN(block1[1].getOutput().syncHost()));
        block1[2].forward(block1[1].getOutput());
        //		System.out.println("block1[2]:"+MatrixOperation.isNaN(block1[2].getOutput().syncHost()));
        /**
         * temb_proj

         */
        temb_proj[0].forward(t);
        temb_proj[1].forward(temb_proj[0].getOutput());
        //		System.out.println(MatrixOperation.isNaN(temb_proj[1].getOutput().syncHost()));
        //		block1[2].getOutput().showShape();
        //		temb_proj[1].getOutput().showShape();
        //		temb_proj[1].getOutput().showDM();
        //		block1[2].getOutput().showDM();
        /**
         * block1 + temb_proj

         */
        Tensor_OP().add(block1[2].getOutput(), temb_proj[1].getOutput(), h, block1[2].getOutput().height * block1[2].getOutput().width);
        //		System.err.println("h:");
        //		h.showDM();
        /**
         * block2

         */
        block2[0].forward(h);
        //		System.out.println("block2[0]:"+MatrixOperation.isNaN(block2[0].getOutput().syncHost()));
        block2[1].forward(block2[0].getOutput());
        //		System.out.println("block2[1]:"+MatrixOperation.isNaN(block2[1].getOutput().syncHost()));
        block2[2].forward(block2[1].getOutput());
        //		System.out.println("block2[2]:"+MatrixOperation.isNaN(block2[2].getOutput().syncHost()));
        /**
         * shortcut

         */
        Tensor tmp = input;
        if (channel != oChannel) {
            shortcut.forward(input);
            tmp = shortcut.getOutput();
        }
        Tensor_OP().add(block2[2].getOutput(), tmp, g);
        /**
         * attn

         */
        if (hasAttn) {
            attn.forward(g);
            this.output = attn.getOutput();
        } else {
            this.output = g;
        }
    }

    @Override
    public Tensor getOutput() {
        // TODO Auto-generated method stub
        return this.output;
    }

    @Override
    public void diff() {
        // TODO Auto-generated method stub
        //		System.out.println("index:["+index+"]("+oChannel+")"+this.delta);
        /**
         * attn backward

         */
        Tensor tmpDelta = delta;
        if (hasAttn) {
            attn.back(delta);
            tmpDelta = attn.diff;
        }
        //		System.err.println("sh:");
        //		tmpDelta.showDM();
        /**
         * block2 backward

         */
        block2[2].back(tmpDelta);
        block2[1].back(block2[2].diff);
        block2[0].back(block2[1].diff);
        //		System.err.println("ht:");
        //		block2[0].diff.showDM();
        /**
         * temb_proj backward

         */
        Tensor_OP().sum(block2[0].diff, dt, 2);
        temb_proj[1].back(dt);
        temb_proj[0].back(temb_proj[1].diff);
        /**
         * block1 backward

         */
        block1[2].back(block2[0].diff);
        block1[1].back(block1[2].diff);
        block1[0].back(block1[1].diff);
        //		System.err.println("gn:");
        //		block1[2].diff.showDM();
        Tensor tmp = tmpDelta;
        if (channel != oChannel) {
            shortcut.back(tmpDelta);
            tmp = shortcut.diff;
        }
        Tensor_OP().add(block1[0].diff, tmp, block1[0].diff);
        this.diff = block1[0].diff;
    }

    public void diff(Tensor t_diff) {
        // TODO Auto-generated method stub
        //		System.out.println("index:["+index+"]("+oChannel+")"+this.delta);
        /**
         * attn backward

         */
        Tensor tmpDelta = delta;
        if (hasAttn) {
            attn.back(delta);
            tmpDelta = attn.diff;
        }
        /**
         * block2 backward

         */
        block2[2].back(tmpDelta);
        block2[1].back(block2[2].diff);
        block2[0].back(block2[1].diff);
        /**
         * temb_proj backward

         */
        Tensor_OP().sum(block2[0].diff, dt, 2);
        //		dt.showDMByOffset(0, 10);
        temb_proj[1].back(dt);
        temb_proj[0].back(temb_proj[1].diff);
        Tensor_OP().add(t_diff, temb_proj[0].diff, t_diff);
        /**
         * block1 backward

         */
        block1[2].back(block2[0].diff);
        block1[1].back(block1[2].diff);
        block1[0].back(block1[1].diff);
        Tensor tmp = tmpDelta;
        if (channel != oChannel) {
            shortcut.back(tmpDelta);
            tmp = shortcut.diff;
        }
        Tensor_OP().add(block1[0].diff, tmp, block1[0].diff);
        this.diff = block1[0].diff;
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

    public void forward(Tensor input, Tensor t) {
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
        this.output(t);
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

    public void back(Tensor delta, Tensor t_diff) {
        // TODO Auto-generated method stub
        initBack();
        /**
         * 设置梯度

         */
        this.setDelta(delta);
        /**
         * 计算梯度

         */
        this.diff(t_diff);
    }

    @Override
    public void update() {
        // TODO Auto-generated method stub
        block1[0].update();
        block1[2].update();
        temb_proj[1].update();
        block2[0].update();
        block2[2].update();
        if (channel != oChannel) {
            shortcut.update();
        }
        if (hasAttn) {
            attn.update();
        }
    }

    @Override
    public void showDiff() {
        // TODO Auto-generated method stub
    }

    @Override
    public LayerType getLayerType() {
        // TODO Auto-generated method stub
        return LayerType.duffsion_res_block;
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

    public void saveModel(RandomAccessFile outputStream) throws IOException {
        GNLayer bgn1 = (GNLayer) block1[0];
        bgn1.saveModel(outputStream);
        ConvolutionLayer bconv1 = (ConvolutionLayer) block1[2];
        bconv1.saveModel(outputStream);
        FullyLayer tl = (FullyLayer) temb_proj[1];
        tl.saveModel(outputStream);
        GNLayer bgn2 = (GNLayer) block2[0];
        bgn2.saveModel(outputStream);
        ConvolutionLayer bconv2 = (ConvolutionLayer) block2[2];
        bconv2.saveModel(outputStream);
        if (shortcut != null) {
            shortcut.saveModel(outputStream);
        }
        attn.saveModel(outputStream);
    }

    public void loadModel(RandomAccessFile inputStream) throws IOException {
        GNLayer bgn1 = (GNLayer) block1[0];
        bgn1.loadModel(inputStream);
        ConvolutionLayer bconv1 = (ConvolutionLayer) block1[2];
        bconv1.loadModel(inputStream);
        FullyLayer tl = (FullyLayer) temb_proj[1];
        tl.loadModel(inputStream);
        GNLayer bgn2 = (GNLayer) block2[0];
        bgn2.loadModel(inputStream);
        ConvolutionLayer bconv2 = (ConvolutionLayer) block2[2];
        bconv2.loadModel(inputStream);
        if (shortcut != null) {
            shortcut.loadModel(inputStream);
        }
        attn.loadModel(inputStream);
    }

    @Override
    public void accGrad(float scale) {
        // TODO Auto-generated method stub
        block1[0].accGrad(scale);
        block1[2].accGrad(scale);
        temb_proj[1].accGrad(scale);
        block2[0].accGrad(scale);
        block2[2].accGrad(scale);
        if (channel != oChannel) {
            shortcut.accGrad(scale);
        }
        if (hasAttn) {
            attn.accGrad(scale);
        }
    }
}

