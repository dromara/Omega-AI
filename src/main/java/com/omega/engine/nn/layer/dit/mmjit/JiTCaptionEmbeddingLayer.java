package com.omega.engine.nn.layer.dit.mmjit;

import java.io.IOException;
import java.io.RandomAccessFile;

import com.omega.common.utils.RandomUtils;
import com.omega.engine.gpu.GPUOP;
import com.omega.engine.nn.layer.FullyLayer;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.dit.kernel.TokenDropKernel;
import com.omega.engine.nn.network.Network;
import com.omega.engine.nn.network.RunModel;
import com.omega.engine.nn.network.utils.ModelUtils;
import com.omega.engine.tensor.Tensor;

/**
 * diffsion model CaptionEmbeddingLayer
 *
 * @author Administrator
 */
public class JiTCaptionEmbeddingLayer extends Layer {
	
	private int token_num = 77;
	
    public FullyLayer linear1;
    
    private boolean bias = true;
    private int inChannel;
    private int outChannel;
    private float uncond_prob = 0.0f;
    
    private TokenDropKernel kernel;

    private Tensor mask;
    

    public JiTCaptionEmbeddingLayer(int inChannel, int outChannel, int token_num, boolean bias, Network network) {
        this.network = network;
        this.bias = bias;
        this.hasBias = false;
        this.token_num = token_num;
        this.inChannel = inChannel;
        this.outChannel = outChannel;
        this.height = 1;
        this.width = inChannel;
        this.oHeight = 1;
        this.oWidth = outChannel;
        initLayers();
    }
    
    public JiTCaptionEmbeddingLayer(int inChannel, int outChannel, int token_num, float uncond_prob, boolean bias, Network network) {
        this.network = network;
        this.bias = bias;
        this.hasBias = false;
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
        int batchSize = number / token_num;
        if(network.RUN_MODEL == RunModel.TRAIN && uncond_prob > 0 && (mask == null || mask.number != batchSize)) {
        	mask = Tensor.createGPUTensor(mask, batchSize, 1, 1, 1, true);
        }
        if(network.RUN_MODEL == RunModel.TRAIN && uncond_prob > 0 && getY_embedding() == null) {
        	float[] data = RandomUtils.gaussianRandom(inChannel, 0.0f, 0.02f);
        	weight = new Tensor(1, 1, 1, inChannel, data, true);
        }
    }

    @Override
    public void initBack() {
        // TODO Auto-generated method stub
    	if(uncond_prob > 0 && diffW == null) {
    		diffW = new Tensor(1, 1, 1, inChannel, true);
    	}
    }

    @Override
    public void initParam() {
        // TODO Auto-generated method stub
    }

    @Override
    public void output() {
        // TODO Auto-generated method stub
    	
    	if(network.RUN_MODEL == RunModel.TRAIN && uncond_prob > 0) {
    		GPUOP.getInstance().cudaRandom(this.mask);//0-1
    		kernel.tokenDrop(input, weight, mask, input, token_num, weight.dataLength, uncond_prob);
    	}
    	
        linear1.forward(input);
        this.output = linear1.getOutput();
    }

    @Override
    public Tensor getOutput() {
        // TODO Auto-generated method stub
        return this.output;
    }

    @Override
    public void diff() {
        // TODO Auto-generated method stub
        linear1.back(delta);
        if(network.RUN_MODEL == RunModel.TRAIN && uncond_prob > 0) {
        	kernel.tokenDropBack(linear1.diff, diffW, mask, token_num, diffW.dataLength, uncond_prob);
        }
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
        if (!this.freeze) {
            if (accDW != null) {
                this.accDW.copy(diffW);
            }
            if (this.updater != null) {
                this.updater.update(this);
            } else {
                for (int i = 0; i < this.weight.getDataLength(); i++) {
                    this.weight.data[i] -= this.learnRate * this.diffW.data[i];
                }
            }
            this.clearAccGrad();
        }
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
        if(uncond_prob > 0) {
        	ModelUtils.saveParams(outputStream, getY_embedding());
        }
    }

    public void loadModel(RandomAccessFile inputStream) throws IOException {
        linear1.loadModel(inputStream);
        if(uncond_prob > 0) {
        	weight = new Tensor(1, 1, 1, inChannel, true);
        	ModelUtils.loadParams(inputStream, weight);
        }
    }

    @Override
    public void accGrad(float scale) {
        // TODO Auto-generated method stub
        linear1.accGrad(scale);
    }

	public Tensor getY_embedding() {
		return weight;
	}
	
	public void setY_embedding(Tensor y_embedding) {
		this.weight = y_embedding;
	}
	
	public Tensor init_y_embedding() {
		if(weight == null) {
			weight = new Tensor(1, 1, 1, inChannel, true);
		}
		return weight;
	}
}

