package com.omega.engine.nn.layer.opensora.wfvae;

import java.io.IOException;
import java.io.RandomAccessFile;

import com.omega.engine.nn.layer.ConvolutionLayer;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.ParamsInit;
import com.omega.engine.nn.layer.active.SiLULayer;
import com.omega.engine.nn.layer.gpu.BasicBlockKernel;
import com.omega.engine.nn.layer.normalization.BNType;
import com.omega.engine.nn.layer.normalization.GNLayer;
import com.omega.engine.nn.network.Network;
import com.omega.engine.tensor.Tensor;
import com.omega.engine.updater.UpdaterFactory;

/**
 * resnet block layer
 *
 * @author Administrator
 */
public class WFResnetBlock2D extends Layer {
	private int depth;
    private int group = 32;
    private BasicBlockKernel kernel;
    private GNLayer norm1;
    private SiLULayer a1;
    private ConvolutionLayer conv1;
    private GNLayer norm2;
    private SiLULayer a2;
    private ConvolutionLayer conv2;
    private ConvolutionLayer conv_shortcut;
    private boolean shortcut = false;
    
    private Tensor inputT;

    public WFResnetBlock2D(int channel, int oChannel, int depth, int height, int width, int group, float outputScale, Network network) {
        this.network = network;
        this.depth = depth;
        this.channel = channel;
        this.oChannel = oChannel;
        this.height = height;
        this.width = width;
        this.group = group;
        if (channel != oChannel) {
            shortcut = true;
        }
        kernel = new BasicBlockKernel(cuda());
        initLayers();
        this.oHeight = conv2.oHeight;
        this.oWidth = conv2.oWidth;
    }

    public void initLayers() {
        norm1 = new GNLayer(group, this, BNType.conv_bn);
        a1 = new SiLULayer(norm1);
        conv1 = new ConvolutionLayer(channel, oChannel, width, height, 3, 3, 1, 1, false, this.network);
        conv1.setUpdater(UpdaterFactory.create(this.network));
        conv1.paramsInit = ParamsInit.silu;
        norm2 = new GNLayer(group, conv1);
        a2 = new SiLULayer(norm2);
        conv2 = new ConvolutionLayer(conv1.oChannel, oChannel, conv1.oWidth, conv1.oHeight, 3, 3, 1, 1, false, this.network);
        conv2.setUpdater(UpdaterFactory.create(this.network));
        conv2.paramsInit = ParamsInit.silu;
        if (shortcut) {
            conv_shortcut = new ConvolutionLayer(channel, oChannel, width, height, 1, 1, 0, 1, false, this.network);
            conv_shortcut.setUpdater(UpdaterFactory.create(this.network));
            conv_shortcut.paramsInit = ParamsInit.silu;
        }
    }

    @Override
    public void init() {
        this.number = this.network.number;
        if (this.output == null || this.output.number != this.network.number) {
        	this.inputT = Tensor.createGPUTensor(inputT, number * depth, channel, height, width, true);
            this.output = Tensor.createGPUTensor(this.output, number, oChannel * depth, oHeight, oWidth, true);
        }
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
    	/**
    	 * video to image
    	 */
    	Tensor_OP().permute(input, inputT, new int[] {number, channel, depth, height, width}, new int[] {number, depth, channel, height, width}, new int[]{0, 2, 1, 3, 4});
        norm1.forward(inputT);
        a1.forward(norm1.getOutput());
        conv1.forward(a1.getOutput());
        norm2.forward(conv1.getOutput());
        a2.forward(norm2.getOutput());
        conv2.forward(a2.getOutput());
        if (shortcut) {
            conv_shortcut.forward(inputT);
            kernel.add(conv_shortcut.getOutput(), conv2.getOutput(), conv2.getOutput());
        } else {
            kernel.add(inputT, conv2.getOutput(), conv2.getOutput());
        }
        /**
    	 * image to video
    	 */
        Tensor_OP().permute(conv2.getOutput(), output, new int[] {number, depth, conv2.oChannel, conv2.oHeight, conv2.oWidth}, new int[] {number, conv2.oChannel, depth, conv2.oHeight, conv2.oWidth}, new int[]{0, 2, 1, 3, 4});
    }

    @Override
    public Tensor getOutput() {
        // TODO Auto-generated method stub
        return this.output;
    }

    @Override
    public void diff() {
        // TODO Auto-generated method stub
    	
    	Tensor_OP().permute(delta, conv2.getOutput(), new int[] {number, conv2.oChannel, depth, conv2.oHeight, conv2.oWidth}, new int[] {number, depth, conv2.oChannel, conv2.oHeight, conv2.oWidth}, new int[]{0, 2, 1, 3, 4});
    	
        conv2.back(conv2.getOutput(), conv2.input);
        a2.back(conv2.diff);
        norm2.back(a2.diff);
        conv1.back(norm2.diff, conv1.input);
        a1.back(conv1.diff);
        norm1.back(a1.diff);
        if (shortcut) {
            conv_shortcut.back(conv2.getOutput(), conv_shortcut.input);
            kernel.add(norm1.diff, conv_shortcut.diff, norm1.diff);
        } else {
            kernel.add(norm1.diff, conv2.getOutput(), norm1.diff);
        }
        
        Tensor_OP().permute(norm1.diff, input, new int[] {number, depth, channel, height, width}, new int[] {number, channel, depth, height, width}, new int[]{0, 2, 1, 3, 4});
        
        this.diff = input;
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
        norm1.update();
        conv1.update();
        norm2.update();
        conv2.update();
        if (shortcut) {
            conv_shortcut.update();
        }
    }
    
    public void saveModel(RandomAccessFile outputStream) throws IOException {
    	norm1.saveModel(outputStream);
    	conv1.saveModel(outputStream);
    	norm2.saveModel(outputStream);
    	conv2.saveModel(outputStream);
    	if (shortcut) {
            conv_shortcut.saveModel(outputStream);
        }
    }

    public void loadModel(RandomAccessFile inputStream) throws IOException {
        norm1.loadModel(inputStream);
    	conv1.loadModel(inputStream);
    	norm2.loadModel(inputStream);
    	conv2.loadModel(inputStream);
    	if (shortcut) {
            conv_shortcut.loadModel(inputStream);
        }
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

