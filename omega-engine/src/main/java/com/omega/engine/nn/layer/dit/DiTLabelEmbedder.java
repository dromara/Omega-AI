package com.omega.engine.nn.layer.dit;

import java.io.IOException;
import java.io.RandomAccessFile;

import com.omega.engine.gpu.GPUOP;
import com.omega.engine.nn.layer.EmbeddingIDLayer;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.dit.kernel.TokenDropKernel;
import com.omega.engine.nn.network.Network;
import com.omega.engine.nn.network.RunModel;
import com.omega.engine.tensor.Tensor;

/**
 * diffsion model LabelEmbedder
 *
 * @author Administrator
 */
public class DiTLabelEmbedder extends Layer {
	
	public EmbeddingIDLayer embedding_table;

    private int inChannel;
    private int outChannel;
    private float uncond_prob = 0.0f;
    
    private TokenDropKernel kernel;

    private Tensor mask;
    

    public DiTLabelEmbedder(int inChannel, int outChannel, Network network) {
        this.network = network;
        this.inChannel = inChannel;
        this.outChannel = outChannel;
        this.height = 1;
        this.width = inChannel;
        this.oHeight = 1;
        this.oWidth = outChannel;
        initLayers();
    }
    
    public DiTLabelEmbedder(int inChannel, int outChannel, float uncond_prob, Network network) {
        this.network = network;
        this.uncond_prob = uncond_prob;
        this.inChannel = inChannel;
        this.outChannel = outChannel;
        this.height = 1;
        this.width = inChannel;
        this.oHeight = 1;
        this.oWidth = outChannel;
        initLayers();
    }

    public void initLayers() {
    	int no_class = 0;
    	if(uncond_prob > 0) {
    		no_class = 1;
    	}
    	embedding_table = new EmbeddingIDLayer(inChannel + no_class, outChannel, network);
        
        if(kernel == null) {
        	kernel = new TokenDropKernel(cuda());
        }

    }

    @Override
    public void init() {
        // TODO Auto-generated method stub
        this.number = this.network.number;
    }
    
    public void init(Tensor input) {
        // TODO Auto-generated method stub
        this.number = input.number;
        if(network.RUN_MODEL == RunModel.TRAIN && (mask == null || mask.number != number)) {
        	mask = Tensor.createGPUTensor(mask, number, 1, 1, 1, true);
        }
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
    	
    	if(network.RUN_MODEL == RunModel.TRAIN && uncond_prob > 0) {
    		GPUOP.getInstance().cudaRandom(this.mask);
    		kernel.tokenDrop(input, inChannel, mask, input, uncond_prob);
    	}
    	
    	embedding_table.forward(input);

        this.output = embedding_table.getOutput();
    }

    @Override
    public Tensor getOutput() {
        // TODO Auto-generated method stub
        return this.output;
    }

    @Override
    public void diff() {
        // TODO Auto-generated method stub

    	embedding_table.back(delta);

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
    public void update() {
        // TODO Auto-generated method stub
    	embedding_table.update();
    }

    @Override
    public void showDiff() {
        // TODO Auto-generated method stub
    }

    @Override
    public LayerType getLayerType() {
        // TODO Auto-generated method stub
        return LayerType.time_embedding;
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
    	embedding_table.saveModel(outputStream);
    }

    public void loadModel(RandomAccessFile inputStream) throws IOException {
    	embedding_table.loadModel(inputStream);
    }

    @Override
    public void accGrad(float scale) {
        // TODO Auto-generated method stub
    	embedding_table.accGrad(scale);
    }
}

