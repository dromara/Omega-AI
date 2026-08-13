package com.omega.engine.nn.layer.dit;

import java.io.IOException;
import java.io.RandomAccessFile;

import com.omega.common.utils.RandomUtils;
import com.omega.engine.gpu.GPUOP;
import com.omega.engine.nn.layer.FullyLayer;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.active.GeluLayer;
import com.omega.engine.nn.layer.active.GeluType;
import com.omega.engine.nn.layer.dit.kernel.ContextMaskKernel;
import com.omega.engine.nn.network.Network;
import com.omega.engine.nn.network.RunModel;
import com.omega.engine.nn.network.utils.ModelUtils;
import com.omega.engine.tensor.Tensor;
import com.omega.engine.updater.UpdaterFactory;

/**
 * diffsion model CaptionEmbeddingLayer
 *
 * @author Administrator
 */
public class DiTMaskCaptionEmbeddingLayer extends Layer {
	
	private int token_num = 77;
	
    public FullyLayer linear1;
    public GeluLayer act;
    public FullyLayer linear2;
    private boolean bias = true;
    private int inChannel;
    private int outChannel;
    private float uncond_prob = 0.0f;
    
    private ContextMaskKernel maskKernel;

    private Tensor mask;
    private Tensor attnMask;
    private Tensor fusedAttnMask;
    private Tensor activeAttnMask;
    private Tensor maskedInput;

    public DiTMaskCaptionEmbeddingLayer(int inChannel, int outChannel, int token_num, boolean bias, Network network) {
     this.hasBias = false;
        this.network = network;
        this.setUpdater(UpdaterFactory.create(network));
        this.bias = bias;
        this.token_num = token_num;
        this.inChannel = inChannel;
        this.outChannel = outChannel;
        this.height = 1;
        this.width = inChannel;
        this.oHeight = 1;
        this.oWidth = outChannel;
        network.paramLayers.add(this);
        initLayers();
    }
    
    public DiTMaskCaptionEmbeddingLayer(int inChannel, int outChannel, int token_num, float uncond_prob, boolean bias, Network network) {
     this.hasBias = false;
        this.network = network;
        this.setUpdater(UpdaterFactory.create(network));
        this.bias = bias;
        this.token_num = token_num;
        this.uncond_prob = uncond_prob;
        this.inChannel = inChannel;
        this.outChannel = outChannel;
        this.height = 1;
        this.width = inChannel;
        this.oHeight = 1;
        this.oWidth = outChannel;
        network.paramLayers.add(this);
        initLayers();
    }

    public void initLayers() {
    	int hiddenSize = outChannel;
        linear1 = new FullyLayer(inChannel, hiddenSize, bias, network);
        // Required for propagating the projector gradient into mask_token.
        linear1.PROPAGATE_DOWN = true;
//        RandomUtils.xavier_uniform(linear1.weight, 1, inChannel, outChannel);
        linear1.weight.setData(RandomUtils.normal_(inChannel * hiddenSize, 0.0f, 0.02f));
        if(linear1.bias != null) {
        	linear1.bias.clearGPU();
        }
        act = new GeluLayer(linear1, GeluType.TANH);
        linear2 = new FullyLayer(hiddenSize, outChannel, bias, network);
//        RandomUtils.xavier_uniform(linear2.weight, 1, outChannel, outChannel);
        linear2.weight.setData(RandomUtils.normal_(hiddenSize * outChannel, 0.0f, 0.02f));
        if(linear2.bias != null) {
        	linear2.bias.clearGPU();
        }
        if(maskKernel == null) {
         maskKernel = new ContextMaskKernel(cuda());
        }
    	this.weight = new Tensor(1, 1, 1, inChannel, true);
//        linear2.weight = new Tensor(1, 1, dim, dim, MatrixUtils.order(dim * dim, 0.01f, 0.01f), true);
    }

    @Override
    public void init() {
        // TODO Auto-generated method stub
        this.number = this.network.number;
    }
    
    public void init(Tensor input, Tensor attnMask) {
        // TODO Auto-generated method stub
        this.number = input.number;
        if(number % token_num != 0) {
            throw new IllegalArgumentException("input.number must be divisible by token_num.");
        }
        int batchSize = number / token_num;
        if(attnMask == null || attnMask.dataLength != batchSize * token_num) {
            throw new IllegalArgumentException(
                    "attnMask must contain batchSize * token_num elements.");
        }
     this.attnMask = attnMask;
        maskedInput = Tensor.createGPUTensor(maskedInput, input.shape(), true);
        if(network.RUN_MODEL == RunModel.TRAIN && uncond_prob > 0 && (mask == null || mask.number != batchSize)) {
         mask = Tensor.createGPUTensor(mask, batchSize, 1, 1, 1, true);
        }
        if(network.RUN_MODEL == RunModel.TRAIN && uncond_prob > 0) {
            fusedAttnMask = Tensor.createGPUTensor(fusedAttnMask, attnMask.shape(), true);
        }
    }

    @Override
    public void initBack() {
        // TODO Auto-generated method stub
        if(diffW == null) {
            diffW = Tensor.createGPUTensor(diffW, weight.shape(), true);
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
            maskKernel.fuseAttnMask(attnMask, mask, fusedAttnMask, token_num, uncond_prob);
            activeAttnMask = fusedAttnMask;
    		/**
    		 * 实现mask与attnMask融合
    		 */
    	}else {
            activeAttnMask = attnMask;
        }
    	maskKernel.forward(input, activeAttnMask, weight, maskedInput);
//    	kernel.tokenDrop(input, y_embedding, mask, input, y_embedding.dataLength, uncond_prob);
//    	input.showDM("input");
        linear1.forward(maskedInput);
//        linear1.getOutput().showDM("linear1");
        act.forward(linear1.getOutput());
//        act.getOutput().showDM("act");
        linear2.forward(act.getOutput());

        this.output = linear2.getOutput();
    }
    
    public void output_eval() {
        // TODO Auto-generated method stub

        activeAttnMask = attnMask;
    	maskKernel.forward(input, activeAttnMask, weight, maskedInput);
    	
        linear1.forward(maskedInput);

        act.forward(linear1.getOutput(), linear1.getOutput());

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
        maskKernel.backwardMaskToken(linear1.diff, activeAttnMask, diffW);
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
    }
    
    public void forward(Tensor input, Tensor attnMask) {
        // TODO Auto-generated method stub
        /**
         * 参数初始化
         */
        this.init(input, attnMask);
        /**
         * 设置输入
         */
        this.setInput(input);
        /**
         * 计算输出
         */
        this.output();
    }

    public void forward_eval(Tensor input, Tensor attnMask) {
        // TODO Auto-generated method stub
        /**
         * 参数初始化
         */
        this.init(input, attnMask);
        /**
         * 设置输入
         */
        this.setInput(input);
        /**
         * 计算输出
         */
        this.output_eval();
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
        ModelUtils.saveParams(outputStream, weight);
        linear1.saveModel(outputStream);
        linear2.saveModel(outputStream);
    }

    public void loadModel(RandomAccessFile inputStream) throws IOException {
    	ModelUtils.loadParams(inputStream, weight);
        linear1.loadModel(inputStream);
        linear2.loadModel(inputStream);
    }

    @Override
    public void accGrad(float scale) {
        // TODO Auto-generated method stub
        if(accDW == null) {
            accDW = diffW.copyGPU();
        }else {
            network.baseKernel.axpy_gpu(diffW, accDW, accDW.dataLength, scale, 1, 1);
        }
        linear1.accGrad(scale);
        linear2.accGrad(scale);
    }

	public Tensor getY_embedding() {
		return this.weight;
	}
	
}

