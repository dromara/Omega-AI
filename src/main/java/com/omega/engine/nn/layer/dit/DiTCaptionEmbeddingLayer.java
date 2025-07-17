package com.omega.engine.nn.layer.dit;

import java.io.IOException;
import java.io.RandomAccessFile;

import com.omega.common.utils.MatrixOperation;
import com.omega.common.utils.RandomUtils;
import com.omega.engine.gpu.GPUOP;
import com.omega.engine.nn.layer.FullyLayer;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.active.GeluLayer;
import com.omega.engine.nn.layer.dit.kernel.TokenDropKernel;
import com.omega.engine.nn.network.Network;
import com.omega.engine.nn.network.RunModel;
import com.omega.engine.tensor.Tensor;

/**
 * diffsion model CaptionEmbeddingLayer
 *
 * @author Administrator
 */
public class DiTCaptionEmbeddingLayer extends Layer {
	
	private int token_num = 77;
	
    public FullyLayer linear1;
    public GeluLayer act;
    public FullyLayer linear2;
    private boolean bias = true;
    private int inChannel;
    private int outChannel;
    private float uncond_prob = 0.0f;
    
    private TokenDropKernel kernel;
    
    private Tensor y_embedding;
    private Tensor mask;
    

    public DiTCaptionEmbeddingLayer(int inChannel, int outChannel, int token_num, boolean bias, Network network) {
        this.network = network;
        this.bias = bias;
        this.token_num = token_num;
        this.inChannel = inChannel;
        this.outChannel = outChannel;
        this.height = 1;
        this.width = inChannel;
        this.oHeight = 1;
        this.oWidth = outChannel;
        initLayers();
    }
    
    public DiTCaptionEmbeddingLayer(int inChannel, int outChannel, int token_num, float uncond_prob, boolean bias, Network network) {
        this.network = network;
        this.bias = bias;
        this.token_num = token_num;
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
        linear1 = new FullyLayer(inChannel, outChannel, bias, network);
        linear1.weight.setData(RandomUtils.normal_(inChannel * outChannel, 0.0f, 0.02f));
        if(linear1.bias != null) {
        	linear1.bias.clearGPU();
        }
        act = new GeluLayer(linear1);
        linear2 = new FullyLayer(outChannel, outChannel, bias, network);
        linear2.weight.setData(RandomUtils.normal_(outChannel * outChannel, 0.0f, 0.02f));
        if(linear2.bias != null) {
        	linear2.bias.clearGPU();
        }
        if(kernel == null) {
        	kernel = new TokenDropKernel(cuda());
        }
//        linear2.weight = new Tensor(1, 1, dim, dim, MatrixUtils.order(dim * dim, 0.01f, 0.01f), true);
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
        if(network.RUN_MODEL == RunModel.TRAIN && y_embedding == null) {
        	float[] data = RandomUtils.gaussianRandom(token_num * inChannel, 0.0f, 1.0f);
        	data = MatrixOperation.multiplication(data, (float) Math.pow(inChannel, 0.5));
        	y_embedding = new Tensor(1, 1, token_num, inChannel, data, true);
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
    		kernel.tokenDrop(input, y_embedding, mask, input, uncond_prob);
    	}
    	
        linear1.forward(input);

        act.forward(linear1.getOutput());

        linear2.forward(act.getOutput());

        this.output = linear2.getOutput();
    }

    @Override
    public Tensor getOutput() {
        // TODO Auto-generated method stub
        return this.output;
    }

    @Override
    public void diff() {
        // TODO Auto-generated method stub

        linear2.back(delta);
        act.back(linear2.diff);
        linear1.back(act.diff);

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
        linear1.update();
        linear2.update();
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
        linear1.saveModel(outputStream);
        linear2.saveModel(outputStream);
    }

    public void loadModel(RandomAccessFile inputStream) throws IOException {
        linear1.loadModel(inputStream);
        linear2.loadModel(inputStream);
    }

    @Override
    public void accGrad(float scale) {
        // TODO Auto-generated method stub
        linear1.accGrad(scale);
        linear2.accGrad(scale);
    }
}

