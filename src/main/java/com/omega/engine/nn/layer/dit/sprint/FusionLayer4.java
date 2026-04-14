package com.omega.engine.nn.layer.dit.sprint;

import java.io.IOException;
import java.io.RandomAccessFile;

import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.dit.kernel.PaddingMaskKernel;
import com.omega.engine.nn.network.Network;
import com.omega.engine.tensor.Tensor;
import com.omega.engine.updater.UpdaterFactory;

/**
 * FusionLayer
 *
 * @author Administrator
 */
public class FusionLayer4 extends Layer {
	
	private int batchSize;
    private int embedDim = 0;
    private int T;
    private int FT;
    
    private PaddingMaskKernel pmKernel;
    
    public FusionLayer4(int embedDim, int FT, int T, Network network) {
        this.network = network;
        if (this.updater == null) {
            this.setUpdater(UpdaterFactory.create(network));
        }
        this.T = T;
        this.FT = FT;
        this.embedDim = embedDim;
        this.oChannel = 1;
        this.oHeight = 1;
        this.oWidth = embedDim;
        this.initLayers();
    }
    
    public static void main(String[] args) {
    	
    }

    public void initLayers() {
    	this.hasBias = false;
//    	
        if(pmKernel == null) {
        	pmKernel = new PaddingMaskKernel(cuda());
        }
        
    }

    @Override
    public void init() {
        // TODO Auto-generated method stub
        this.number = this.input.number;
        this.batchSize = number / T;
        if(number != batchSize * FT && (output == null || output.number != batchSize * FT)) {
        	this.output = Tensor.createGPUTensor(output, batchSize * FT, 1, 1, embedDim, true);
        }
    }

    @Override
    public void initBack() {
        // TODO Auto-generated method stub
    	if(diff == null || diff.number != number) {
    		diff = Tensor.createGPUTensor(diff, input.shape(), true);
    	}else {
    		diff.clearGPU();
    	}
    }

    @Override
    public void initParam() {
        // TODO Auto-generated method stub
    }

    @Override
    public void output() {
        // TODO Auto-generated method stub

    }
    
    public void output(Tensor encoder, Tensor idskeep) {
    	pmKernel.forward_allmask(input, encoder, idskeep, output, FT, T, 0, embedDim);
    }
    
    @Override
    public Tensor getOutput() {
        // TODO Auto-generated method stub
        return output;
    }

    @Override
    public void diff() {
        // TODO Auto-generated method stub
    	
    }
    
    public void diff(Tensor delta_encoder, Tensor idskeep) {
        // TODO Auto-generated method stub
		pmKernel.backward_allmask(delta, idskeep, diff, FT, T, 0, embedDim);
		delta_encoder = delta;
    }

    @Override
    public void forward() {
        // TODO Auto-generated method stub
        /**
         * 设置输入
         */
        this.setInput();
        /**
         * 参数初始化
         */
        this.init();
        /**
         * 计算输出
         */
        this.output();
    }

    @Override
    public void back() {
        // TODO Auto-generated method stub
        this.initBack();
        /**
         * 设置梯度
         */
        this.setDelta();
        /**
         * 计算梯度
         */
        this.diff();
        if (this.network.GRADIENT_CHECK) {
            this.gradientCheck();
        }
    }

    @Override
    public void forward(Tensor input) {
        // TODO Auto-generated method stub
        /**
         * 设置输入
         */
        this.setInput(input);
        /**
         * 参数初始化
         */
        this.init();
        /**
         * 计算输出
         */
        this.output();
    }
    
    public void forward(Tensor input, Tensor encoder, Tensor idskeep) {
        // TODO Auto-generated method stub
        /**
         * 设置输入
         */
        this.setInput(input);
        /**
         * 参数初始化
         */
        this.init();
        /**
         * 计算输出
         */
        this.output(encoder, idskeep);
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
    
    public void back(Tensor delta, Tensor delta_encoder, Tensor idskeep) {
        // TODO Auto-generated method stub
        this.initBack();
        /**
         * 设置梯度
         */
        this.setDelta(delta);
        /**
         * 计算梯度
         */
        this.diff(delta_encoder, idskeep);
    }
    
    @Override
    public void update() {
        // TODO Auto-generated method stub

    }

    @Override
    public void showDiff() {
        // TODO Auto-generated method stub
    }

    @Override
    public LayerType getLayerType() {
        // TODO Auto-generated method stub
        return LayerType.mlp;
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

    }

    public void loadModel(RandomAccessFile inputStream) throws IOException {

    }

    @Override
    public void accGrad(float scale) {
        // TODO Auto-generated method stub

    }
}

