package com.omega.engine.nn.layer.dit.mmjit;

import java.io.IOException;
import java.io.RandomAccessFile;

import com.omega.common.utils.RandomUtils;
import com.omega.engine.nn.layer.FullyLayer;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.dit.kernel.PaddingMaskKernel;
import com.omega.engine.nn.network.Network;
import com.omega.engine.nn.network.RunModel;
import com.omega.engine.nn.network.utils.ModelUtils;
import com.omega.engine.tensor.Tensor;
import com.omega.engine.updater.UpdaterFactory;

/**
 * MMJiTFusionLayer
 *
 * @author Administrator
 */
public class MMJiTFusionLayer extends Layer {
	
	private int batchSize;
    private int embedDim = 0;
    private int FT;
    
    private float path_drop_prob = 0.0f;

    public FullyLayer fusion_proj;
    
    private Tensor g_pad;
    private Tensor e_m;
    
    private PaddingMaskKernel pmKernel;
    
    private float pdp = 0.0f;

    public MMJiTFusionLayer(int embedDim, int FT, Network network) {
        this.network = network;
        if (this.updater == null) {
            this.setUpdater(UpdaterFactory.create(network));
        }
        this.FT = FT;
        this.embedDim = embedDim;
        this.oChannel = 1;
        this.oHeight = 1;
        this.oWidth = embedDim;
        this.initLayers();
    }
    
    public MMJiTFusionLayer(int embedDim, int FT, float path_drop_prob, Network network) {
        this.network = network;
        if (this.updater == null) {
            this.setUpdater(UpdaterFactory.create(network));
        }
        this.FT = FT;
        this.path_drop_prob = path_drop_prob;
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
    	this.weight = new Tensor(1, 1, 1, embedDim, true);
    	
        this.fusion_proj = new FullyLayer(embedDim * 2, embedDim, true, network);
        RandomUtils.xavier_uniform(this.fusion_proj.weight, 1, embedDim * 2, embedDim);
        if(this.fusion_proj.bias != null) {
        	this.fusion_proj.bias.clearGPU();
        }
        
        if(pmKernel == null) {
        	pmKernel = new PaddingMaskKernel(cuda());
        }
        
    }

    @Override
    public void init() {
        // TODO Auto-generated method stub
        this.number = this.input.number;
        this.batchSize = number / FT;
        if(number != batchSize * FT && (g_pad == null || g_pad.number != batchSize * FT)) {
        	this.g_pad = Tensor.createGPUTensor(g_pad, batchSize * FT, 1, 1, embedDim, true);
        }
        if(e_m == null || e_m.number != batchSize * FT) {
        	this.e_m = Tensor.createGPUTensor(e_m, batchSize * FT, 1, 1, embedDim * 2, true);
        }
    }

    @Override
    public void initBack() {
        // TODO Auto-generated method stub
    	if(diffW == null) {
    		diffW = Tensor.createGPUTensor(diffW, weight.shape(), true);
    	}else {
    		diffW.clearGPU();
    	}
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
    
    public void output(Tensor encoder) {
    	pdp = RandomUtils.randomFloat();
//    	pdp = 0.00001f;
    	if(network.RUN_MODEL == RunModel.TRAIN && path_drop_prob > 0 && pdp < path_drop_prob) {
    		pmKernel.set_mask_igone(weight, input, FT, 0, embedDim);
    	}
    	Tensor_OP().cat_width(encoder, input, e_m, embedDim, embedDim);
    	fusion_proj.forward(e_m);
    	this.output = fusion_proj.getOutput();
    }
    
    public void output_uncond(Tensor encoder) {
    	pmKernel.set_mask_igone(weight, input, FT, 0, embedDim);
    	Tensor_OP().cat_width(encoder, input, e_m, embedDim, embedDim);
    	fusion_proj.forward(e_m);
    	this.output = fusion_proj.getOutput();
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
    
    public void diff(Tensor dencoder) {
        // TODO Auto-generated method stub
    	fusion_proj.back(delta);
    	Tensor_OP().cat_width_back(dencoder, diff, fusion_proj.diff, embedDim, embedDim);
    	if(path_drop_prob > 0 && pdp < path_drop_prob) {
    		pmKernel.mask_igone_diff2(diff, diffW, batchSize, FT, embedDim, 0);
    		diff.clearGPU();
    	}
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
    
    public void forward(Tensor input, Tensor encoder) {
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
        this.output(encoder);
    }
    
    public void forward_uncond(Tensor input, Tensor encoder) {
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
        this.output_uncond(encoder);
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
    
    public void back(Tensor delta, Tensor dencoder) {
        // TODO Auto-generated method stub
        this.initBack();
        /**
         * 设置梯度
         */
        this.setDelta(delta);
        /**
         * 计算梯度
         */
        this.diff(dencoder);
    }
    
    @Override
    public void update() {
        // TODO Auto-generated method stub
    	if (!this.freeze) {
            if (accDW != null) {
                this.accDW.copy(diffW);
                if (hasBias) {
                    this.accDB.copy(diffB);
                }
            }
            if (this.updater != null) {
                this.updater.update(this);
            } else {
                for (int i = 0; i < this.weight.getDataLength(); i++) {
                    this.weight.data[i] -= this.learnRate * this.diffW.data[i];
                }
                for (int i = 0; i < this.bias.getDataLength(); i++) {
                    this.bias.data[i] -= this.learnRate * this.diffB.data[i];
                }
            }
            this.clearAccGrad();
        }
    	fusion_proj.update();
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
    	ModelUtils.saveParams(outputStream, weight);
    	fusion_proj.saveModel(outputStream);
    }

    public void loadModel(RandomAccessFile inputStream) throws IOException {
    	ModelUtils.loadParams(inputStream, weight);
    	fusion_proj.loadModel(inputStream);
    }

    @Override
    public void accGrad(float scale) {
        // TODO Auto-generated method stub
    	fusion_proj.accGrad(scale);
    }
}

