package com.omega.engine.nn.layer.opensora.dit.moudles;

import java.io.IOException;
import java.io.RandomAccessFile;

import com.omega.common.utils.RandomUtils;
import com.omega.engine.nn.layer.ConvolutionLayer;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.network.Network;
import com.omega.engine.nn.network.Transformer;
import com.omega.engine.tensor.Tensor;
import com.omega.engine.updater.UpdaterFactory;

/**
 * DiT_PatchEmbeddingLayer
 *
 * @author Administrator
 */
public class DiTPatchEmbed2D extends Layer {
	
    private int embedDim = 0;

    public ConvolutionLayer patchEmbedding;
    
    private int[] shape;
    private int[] t_shape;

    public DiTPatchEmbed2D(int channel, int imageSize, int embedDim, int patchSize, boolean bias, Network network) {
        this.network = network;
        if (this.updater == null) {
            this.setUpdater(UpdaterFactory.create(network));
        }
        this.channel = channel;
        this.height = imageSize;
        this.width = imageSize;
        this.embedDim = embedDim;
        initLayers(channel, imageSize, imageSize, patchSize, bias);

    }

    public static void main(String[] args) {
        int batchSize = 2;
        int embedDim = 6;
        int imgSize = 224;
        int patchSize = 32;
        Transformer tf = new Transformer();
        tf.number = batchSize;
        DiTPatchEmbed2D layer = new DiTPatchEmbed2D(3, embedDim, imgSize, patchSize, false, tf);
        float[] data = RandomUtils.order(batchSize * 3 * imgSize * imgSize, 1f, 0f);
        Tensor input = new Tensor(batchSize, 3, imgSize, imgSize, data, true);
        layer.forward(input);
        layer.getOutput().showShape();
        layer.getOutput().showDM();
    }

    @Override
    public void init() {
        // TODO Auto-generated method stub
        this.number = network.number;
        if (output == null || output.number != this.number) {
            int pChannel = this.getPatchEmbedding().oHeight * this.getPatchEmbedding().oWidth;
            output = Tensor.createGPUTensor(output, this.number, pChannel, 1, embedDim, true);
        }
    }

    public void init(Tensor input) {
        // TODO Auto-generated method stub
        this.number = input.number;
        if (output == null || output.number != this.number) {
            int pChannel = this.getPatchEmbedding().oHeight * this.getPatchEmbedding().oWidth;
            output = Tensor.createGPUTensor(output, this.number, pChannel, 1, embedDim, true);
            shape = new int[] {number, this.getPatchEmbedding().oChannel, 1, pChannel};
            t_shape = new int[] {number, pChannel, 1, this.getPatchEmbedding().oChannel};
        }
    }

    public void initLayers(int inChannel, int height, int width, int patchSize, boolean bias) {
        this.patchEmbedding = new ConvolutionLayer(inChannel, embedDim, height, width, patchSize, patchSize, 0, patchSize, bias, network);
        this.patchEmbedding.weight.setData(RandomUtils.xavierUniform(this.patchEmbedding.weight.dataLength, inChannel * patchSize * patchSize, embedDim * patchSize * patchSize, 1));
        if(this.patchEmbedding.bias != null) {
        	this.patchEmbedding.bias.clearGPU();
        }
        int pChannel = this.getPatchEmbedding().oHeight * this.getPatchEmbedding().oWidth;
        this.oChannel = pChannel;
        this.oHeight = 1;
        this.oWidth = embedDim;
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
        getPatchEmbedding().forward(this.input);
        Tensor_OP().permute(getPatchEmbedding().getOutput(), output, shape, t_shape, new int[]{0, 3, 2, 1});
    }

    @Override
    public Tensor getOutput() {
        // TODO Auto-generated method stub
        return output;
    }

    @Override
    public void diff() {
        // TODO Auto-generated method stub
    	Tensor_OP().permute(delta, getPatchEmbedding().getOutput(), t_shape, shape, new int[]{0, 3, 2, 1});
    	getPatchEmbedding().back(getPatchEmbedding().getOutput());
    	this.diff =  getPatchEmbedding().diff;
//    	diff.showDM("dx");
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
    	getPatchEmbedding().update();
    }

    @Override
    public void showDiff() {
        // TODO Auto-generated method stub
    }

    @Override
    public LayerType getLayerType() {
        // TODO Auto-generated method stub
        return LayerType.clip_vision_embedding;
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

    public void saveModel(RandomAccessFile outputStream) throws IOException {
    	getPatchEmbedding().saveModel(outputStream);
    }

    public void loadModel(RandomAccessFile inputStream) throws IOException {
    	getPatchEmbedding().loadModel(inputStream);
    }

    @Override
    public void accGrad(float scale) {
        // TODO Auto-generated method stub
    	getPatchEmbedding().accGrad(scale);
    }

    public ConvolutionLayer getPatchEmbedding() {
        return patchEmbedding;
    }

}

