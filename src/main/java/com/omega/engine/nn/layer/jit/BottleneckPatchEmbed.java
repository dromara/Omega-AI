package com.omega.engine.nn.layer.jit;

import java.io.IOException;
import java.io.RandomAccessFile;

import com.omega.common.utils.RandomUtils;
import com.omega.engine.nn.layer.ConvolutionLayer;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.network.Network;
import com.omega.engine.tensor.Tensor;
import com.omega.engine.updater.UpdaterFactory;

/**
 * BottleneckPatchEmbed
 *
 * @author Administrator
 */
public class BottleneckPatchEmbed extends Layer {
	
    private int embedDim = 0;
    private int pca_dim;

    public ConvolutionLayer proj1;
    public ConvolutionLayer proj2;
    
    private int[] shape;
    private int[] t_shape;

    public BottleneckPatchEmbed(int channel, int imageSize, int pca_dim, int embedDim, int patchSize, boolean bias, Network network) {
        this.network = network;
        if (this.updater == null) {
            this.setUpdater(UpdaterFactory.create(network));
        }
        this.channel = channel;
        this.height = imageSize;
        this.width = imageSize;
        this.pca_dim = pca_dim;
        this.embedDim = embedDim;
        initLayers(channel, imageSize, imageSize, patchSize, bias);
    }

    public static void main(String[] args) {
       
    }

    @Override
    public void init() {
        // TODO Auto-generated method stub
        this.number = network.number;
        if (output == null || output.number != this.number) {
            int pChannel = this.proj2.oHeight * this.proj2.oWidth;
            output = Tensor.createGPUTensor(output, this.number, pChannel, 1, embedDim, true);
        }
    }

    public void init(Tensor input) {
        // TODO Auto-generated method stub
        this.number = input.number;
        if (output == null || output.number != this.number) {
            int pChannel = this.proj2.oHeight * this.proj2.oWidth;
            output = Tensor.createGPUTensor(output, this.number, pChannel, 1, embedDim, true);
            shape = new int[] {number, this.proj2.oChannel, 1, pChannel};
            t_shape = new int[] {number, pChannel, 1, this.proj2.oChannel};
        }
    }

    public void initLayers(int inChannel, int height, int width, int patchSize, boolean bias) {
        this.proj1 = new ConvolutionLayer(inChannel, pca_dim, width, height, patchSize, patchSize, 0, patchSize, false, network);
//        proj1.PROPAGATE_DOWN = false;
        RandomUtils.xavier_uniform(proj1.weight, 1, inChannel * patchSize * patchSize, pca_dim * patchSize * patchSize);

        this.proj2 = new ConvolutionLayer(pca_dim, embedDim, proj1.oWidth, proj1.oHeight, 1, 1, 0, 1, bias, network);
        RandomUtils.xavier_uniform(proj2.weight, 1, pca_dim, embedDim);
        if(this.proj2.bias != null) {
        	this.proj2.bias.clearGPU();
        }
        int pChannel = this.proj2.oHeight * this.proj2.oWidth;
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
    	proj1.forward(input);
        proj2.forward(proj1.getOutput());
        Tensor_OP().permute(proj2.getOutput(), output, shape, t_shape, new int[]{0, 3, 2, 1});
    }

    @Override
    public Tensor getOutput() {
        // TODO Auto-generated method stub
        return output;
    }

    @Override
    public void diff() {
        // TODO Auto-generated method stub
    	Tensor_OP().permute(delta, proj2.getOutput(), t_shape, shape, new int[]{0, 3, 2, 1});
    	proj2.back(proj2.getOutput());
    	proj1.back(proj2.diff);
    	this.diff =  proj1.diff;
//    	diff.showDM("diff");
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
    	proj1.update();
    	proj2.update();
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
    	proj1.saveModel(outputStream);
    	proj2.saveModel(outputStream);
    }

    public void loadModel(RandomAccessFile inputStream) throws IOException {
    	proj1.loadModel(inputStream);
    	proj2.loadModel(inputStream);
    }

    @Override
    public void accGrad(float scale) {
        // TODO Auto-generated method stub
    	proj1.accGrad(scale);
    	proj2.accGrad(scale);
    }

}

