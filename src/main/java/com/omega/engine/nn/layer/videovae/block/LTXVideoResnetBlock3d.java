package com.omega.engine.nn.layer.videovae.block;

import java.io.IOException;
import java.io.RandomAccessFile;

import com.omega.common.utils.MatrixUtils;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.ParamsInit;
import com.omega.engine.nn.layer.active.ActiveFunctionLayer;
import com.omega.engine.nn.layer.active.SiLULayer;
import com.omega.engine.nn.layer.gpu.BasicBlockKernel;
import com.omega.engine.nn.layer.normalization.BNType;
import com.omega.engine.nn.layer.normalization.LNLayer;
import com.omega.engine.nn.layer.normalization.RMSLayer;
import com.omega.engine.nn.network.Network;
import com.omega.engine.tensor.Tensor;
import com.omega.engine.updater.UpdaterFactory;

/**
 * LTXVideoResnetBlock3d
 *
 * @author Administrator
 */
public class LTXVideoResnetBlock3d extends Layer {

	public int depth;
	public int oDepth;
	
	private boolean is_causal;
	
    private BasicBlockKernel kernel;
    
    public RMSLayer norm1;
    private ActiveFunctionLayer a1;
    public LTXVideoCausalConv3d conv1;
    
    public RMSLayer norm2;
    private ActiveFunctionLayer a2;
    public LTXVideoCausalConv3d conv2;
    
    public LNLayer norm3;
    public LTXVideoCausalConv3d conv_shortcut;
    
    public boolean shortcut = false;
    
    private Tensor normInput1;
    private Tensor normInput2;
    
    public LTXVideoResnetBlock3d(int channel, int oChannel, int depth, int height, int width, boolean is_causal, Network network) {
        this.network = network;
        this.channel = channel;
        this.depth = depth;
        this.oChannel = oChannel;
        this.height = height;
        this.width = width;
        this.is_causal = is_causal;
        if (channel != oChannel) {
            shortcut = true;
        }
        kernel = new BasicBlockKernel(cuda());
        initLayers();
        this.oDepth = conv2.oDepth;
        this.oHeight = conv2.oHeight;
        this.oWidth = conv2.oWidth;
    }

    public void initLayers() {
        norm1 = new RMSLayer(1, 1, channel, false, BNType.fully_bn, network);
//        norm1.gamma = new Tensor(1, 1, 1, channel, MatrixUtils.one(channel), true);
        a1 = new SiLULayer(norm1);
        conv1 = new LTXVideoCausalConv3d(channel, oChannel, depth, width, height, 3, 3, 3, 1, true, is_causal, network);
//        conv1.weight.showShape("conv1");
        conv1.setUpdater(UpdaterFactory.create(this.network));
        conv1.paramsInit = ParamsInit.silu;
        norm2 = new RMSLayer(1, 1, oChannel, false, BNType.fully_bn, network);
//        norm2.gamma = new Tensor(1, 1, 1, oChannel, MatrixUtils.one(oChannel), true);
        a2 = new SiLULayer(norm2);
        conv2 = new LTXVideoCausalConv3d(oChannel, oChannel, conv1.oDepth, conv1.oWidth, conv1.oHeight, 3, 3, 3, 1, true, is_causal, network);
        conv2.setUpdater(UpdaterFactory.create(this.network));
        conv2.paramsInit = ParamsInit.silu;
        if (shortcut) {
        	norm3 = new LNLayer(1, 1, channel, true, true, BNType.fully_bn, network);
        	norm3.gamma = new Tensor(1, 1, 1, channel, MatrixUtils.one(channel), true);
        	norm3.beta = new Tensor(1, 1, 1, channel, MatrixUtils.zero(channel), true);
            conv_shortcut = new LTXVideoCausalConv3d(channel, oChannel, depth, width, height, 1, 1, 1, 1, true, is_causal, network);
//            conv_shortcut.weight.showShape("conv_shortcut");
            conv_shortcut.setUpdater(UpdaterFactory.create(this.network));
            conv_shortcut.paramsInit = ParamsInit.silu;
        }
    }

    @Override
    public void init() {
        this.number = this.network.number;
        if(normInput1 == null || this.normInput1.number != this.number) {
        	int hw = height * width;
        	this.normInput1 = Tensor.createGPUTensor(normInput1, number, depth, hw, channel, true);
        	this.normInput2 = Tensor.createGPUTensor(normInput2, number, depth, hw, oChannel, true);
        }else {
        	normInput1.viewOrg();
        	normInput2.viewOrg();
        }
        if (this.output == null || this.output.number != this.network.number) {
            this.output = Tensor.createGPUTensor(this.output, number, oChannel * oDepth, oHeight, oWidth, true);
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
//    	input.showDMByOffsetRed((3 * depth + 2) * input.height * input.width, input.height * input.width, "input");
    	Tensor_OP().permute(input, normInput1, new int[] {number, channel, depth, height, width}, new int[] {number, depth, height, width, channel}, new int[] {0, 2, 3, 4, 1});
        norm1.forward(normInput1);
        Tensor_OP().permute(norm1.getOutput(), normInput1, new int[] {number, depth, height, width, channel}, new int[] {number, channel, depth, height, width}, new int[] {0, 4, 1, 2, 3});
        normInput1.view(number, channel * depth, height, width);
        
//        normInput1.showDMByOffsetRed((3 * depth + 2) * normInput1.height * normInput1.width, normInput1.height * normInput1.width, "norm1");

        a1.forward(normInput1);
        conv1.forward(a1.getOutput());
//        conv1.getOutput().showDMByOffsetRed((3 * conv1.oDepth + 2) * conv1.getOutput().height * conv1.getOutput().width, conv1.getOutput().height * conv1.getOutput().width, "conv1.getOutput()");
        
        Tensor_OP().permute(conv1.getOutput(), normInput2, new int[] {number, oChannel, depth, height, width}, new int[] {number, depth, height, width, oChannel}, new int[] {0, 2, 3, 4, 1});
        norm2.forward(normInput2);
        Tensor_OP().permute(norm2.getOutput(), normInput2, new int[] {number, depth, height, width, oChannel}, new int[] {number, oChannel, depth, height, width}, new int[] {0, 4, 1, 2, 3});
        normInput2.view(number, oChannel * depth, height, width);
        a2.forward(normInput2);
        conv2.forward(a2.getOutput());
        
        if (shortcut) {
        	normInput1.viewOrg();
        	Tensor_OP().permute(input, normInput1, new int[] {number, channel, depth, height, width}, new int[] {number, depth, height, width, channel}, new int[] {0, 2, 3, 4, 1});
            norm3.forward(normInput1);
            Tensor_OP().permute(norm3.getOutput(), normInput1, new int[] {number, depth, height, width, channel}, new int[] {number, channel, depth, height, width}, new int[] {0, 4, 1, 2, 3});
            normInput1.view(number, channel * depth, height, width);
            conv_shortcut.forward(normInput1);
            kernel.add(conv_shortcut.getOutput(), conv2.getOutput(), output);
        } else {
            kernel.add(input, conv2.getOutput(), output);
        }
//        output.showDMByOffsetRed((3 * 17 + 2) * height * width, height * width, "output");

    }

    @Override
    public Tensor getOutput() {
        // TODO Auto-generated method stub
        return this.output;
    }

    @Override
    public void diff() {
        // TODO Auto-generated method stub
       
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
        	norm3.update();
            conv_shortcut.update();
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
    
    public void saveModel(RandomAccessFile outputStream) throws IOException {
//        norm1.saveModel(outputStream);
        conv1.saveModel(outputStream);
//        norm2.saveModel(outputStream);
        conv2.saveModel(outputStream);
        if(shortcut) {
        	norm3.saveModel(outputStream);
        	conv_shortcut.saveModel(outputStream);
        }
    }

    public void loadModel(RandomAccessFile inputStream) throws IOException {
//        norm1.loadModel(inputStream);
        conv1.loadModel(inputStream);
//        norm2.loadModel(inputStream);
        conv2.loadModel(inputStream);
        if(shortcut) {
        	norm3.loadModel(inputStream);
        	conv_shortcut.loadModel(inputStream);
        }
    }
}

