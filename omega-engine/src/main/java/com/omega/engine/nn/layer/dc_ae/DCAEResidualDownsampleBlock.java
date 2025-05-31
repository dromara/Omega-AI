package com.omega.engine.nn.layer.dc_ae;

import java.io.IOException;
import java.io.RandomAccessFile;

import com.omega.engine.nn.layer.ConvolutionLayer;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.ParamsInit;
import com.omega.engine.nn.layer.active.SiLULayer;
import com.omega.engine.nn.layer.dc_ae.gpu.DCAEKernel;
import com.omega.engine.nn.layer.normalization.BNType;
import com.omega.engine.nn.layer.normalization.GNLayer;
import com.omega.engine.nn.network.Network;
import com.omega.engine.tensor.Tensor;
import com.omega.engine.updater.UpdaterFactory;

/**
 * DCAE_ResidualDownsampleBlock
 *
 * @author Administrator
 */
public class DCAEResidualDownsampleBlock extends Layer {
	
    private int group = 32;
    private DCAEKernel kernel;
    private GNLayer norm;
    private SiLULayer a1;
    private ConvolutionLayer conv1;
    private SiLULayer a2;
    private ConvolutionLayer conv2;

    private Tensor sc_x;
    private Tensor ca_x;
    
    private int[] x_shape;
    private int[] o_shape;
    
    public DCAEResidualDownsampleBlock(int channel, int oChannel, int height, int width, int group, Network network) {
        this.network = network;
        this.channel = channel;
        this.oChannel = channel;
        this.height = height;
        this.width = width;
        this.group = group;
        kernel = new DCAEKernel(cuda());
        initLayers(oChannel);
        this.oHeight = conv2.oHeight;
        this.oWidth = conv2.oWidth;
        this.oChannel = conv2.oChannel;
    }

    public void initLayers(int oChannel) {
    	
    	conv1 = new ConvolutionLayer(channel, oChannel, width, height, 3, 3, 1, 2, true, this.network);
        conv1.setUpdater(UpdaterFactory.create(this.network));
        conv1.paramsInit = ParamsInit.silu;
        a1 = new SiLULayer(conv1);
        
        conv2 = new ConvolutionLayer(conv1.oChannel, oChannel, conv1.oWidth, conv1.oHeight, 3, 3, 1, 1, true, this.network);
        conv2.setUpdater(UpdaterFactory.create(this.network));
        conv2.paramsInit = ParamsInit.silu;
        norm = new GNLayer(group, conv2.oChannel, conv2.oHeight, conv2.oWidth, BNType.conv_bn, conv2);
        a2 = new SiLULayer(norm);
       
    }

    @Override
    public void init() {
        this.number = this.network.number;
//        if (network.RUN_MODEL == RunModel.EVAL) {
//            this.cache = CUDAMemoryManager.getCache("VQVAEResidual_cache", number, oChannel, oHeight, oWidth);
//        }
    }
    
    public void init(Tensor input) {
        this.number = input.number;
        if(sc_x == null || sc_x.number != number) {
    		x_shape = new int[] {number, channel, height/2, 2, width/2, 2};
    		o_shape = new int[] {number, channel, 2, 2, height/2, width/2};
        	sc_x = Tensor.createGPUTensor(sc_x, number, channel * 4, height / 2, width / 2, true);
        }
        if(ca_x == null || ca_x.number != number) {
        	ca_x = Tensor.createGPUTensor(ca_x, number, sc_x.channel / 2, sc_x.height, sc_x.width, true);
        }
//        if(output == null || output.number != number) {
//        	output = Tensor.createGPUTensor(output, number, oChannel, oHeight, oWidth, true);
//        }
//        if (network.RUN_MODEL == RunModel.EVAL) {
//            this.cache = CUDAMemoryManager.getCache("VQVAEResidual_cache", number, oChannel, oHeight, oWidth);
//        }
    }

    @Override
    public void initBack() {
//    	if (network.gradCacheMode) {
////    		conv1.diff = network.cudaManager.getMemoryManager().getPrivateCaches("vqvae-resblock-conv1", conv1.input.number, conv1.input.channel, conv1.input.height, conv1.input.width);
//            if(conv2.diff == null || !conv2.diff.checkShape(conv2.input)) {
//            	conv2.diff = network.cudaManager.getMemoryManager().getPrivateCaches("vqvae-resblock-conv2", conv2.input.number, conv2.input.channel, conv2.input.height, conv2.input.width);
//            }
//    	}
    }

    @Override
    public void initParam() {
        // TODO Auto-generated method stub
    }

    @Override
    public void output() {
        // TODO Auto-generated method stub
//        if (network.RUN_MODEL == RunModel.EVAL) {
//            output_eval();
//        } else {
    		
    		/**
    		 * space_to_channel
    		 */
    		Tensor_OP().permute(input, sc_x, x_shape, o_shape, new int[] {0, 1, 3, 5, 2, 4});
    		
    		/**
    		 * channel_average
    		 */
    		kernel.channel_average(sc_x, ca_x);
    		
        	conv1.forward(input);
            a1.forward(conv1.getOutput());
            
            conv2.forward(a1.getOutput());
            norm.forward(conv2.getOutput());
            a2.forward(norm.getOutput());
            
            Tensor_OP().add(ca_x, a2.getOutput(), a2.getOutput());
            
            this.output = a2.getOutput();
//        }
    }

//    public void output_eval() {
//        // TODO Auto-generated method stub
//        Tensor norm_out = CUDAMemoryManager.getCache("DCAEResidual_norm1_cache", input.number, input.channel, input.height, input.width);
//        norm1.forward(this.input, norm_out);
//        a1.forward(norm1.getOutput(), norm1.getOutput());
//        conv1.forward(a1.getOutput(), cache);
//        norm2.forward(conv1.getOutput(), cache);
//        a2.forward(norm2.getOutput(), cache);
//        conv2.forward(a2.getOutput());
//        if (shortcut) {
//            conv_shortcut.forward(this.input, cache);
//            kernel.add(conv_shortcut.getOutput(), conv2.getOutput(), conv2.getOutput());
//        } else {
//            kernel.add(input, conv2.getOutput(), conv2.getOutput());
//        }
//        this.output = conv2.getOutput();
//    }

    @Override
    public Tensor getOutput() {
        // TODO Auto-generated method stub
        return this.output;
    }

    @Override
    public void diff() {
        // TODO Auto-generated method stub
        //		System.out.println(index);
    	kernel.channel_average_backward(delta, sc_x);
    	
        a2.back(delta);
        norm.back(a2.diff);
        conv2.back(norm.diff, conv2.input);
        
        a1.back(conv2.diff);
        conv1.back(a1.diff);

        Tensor_OP().permute(sc_x, input, o_shape, x_shape, new int[] {0, 1, 4, 2, 5, 3});

        Tensor_OP().add(conv1.diff, input, conv1.diff);
        
        this.diff = conv1.diff;
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
        conv1.update();
        conv2.update();
        norm.update();
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
        conv1.saveModel(outputStream);
        conv2.saveModel(outputStream);
        norm.saveModel(outputStream);
    }

    public void loadModel(RandomAccessFile inputStream) throws IOException {
    	conv1.loadModel(inputStream);
        conv2.loadModel(inputStream);
        norm.loadModel(inputStream);
    }
}

