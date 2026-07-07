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
public class MMJiTFusionTXTLayer extends Layer {
	
	private int batchSize;
    private int embedDim = 0;
    private int FT;
    private int TT;
    
    private float path_drop_prob = 0.0f;

    public FullyLayer fusion_proj;
    public FullyLayer fusion_txt_proj;

    private Tensor e_m;
    private Tensor t_e_m;
    
    private PaddingMaskKernel pmKernel;
    
    private float pdp = 0.0f;
    
    private Tensor diffWTxt;
    public Tensor diffTxt;

    public MMJiTFusionTXTLayer(int embedDim, int FT, int TT, Network network) {
        this.network = network;
        if (this.updater == null) {
            this.setUpdater(UpdaterFactory.create(network));
        }
        this.FT = FT;
        this.TT = TT;
        this.embedDim = embedDim;
        this.oChannel = 1;
        this.oHeight = 1;
        this.oWidth = embedDim;
        this.initLayers();
    }
    
    public MMJiTFusionTXTLayer(int embedDim, int FT, int TT, float path_drop_prob, Network network) {
        this.network = network;
        if (this.updater == null) {
            this.setUpdater(UpdaterFactory.create(network));
        }
        this.FT = FT;
        this.TT = TT;
        this.path_drop_prob = path_drop_prob;
        this.embedDim = embedDim;
        this.oChannel = 1;
        this.oHeight = 1;
        this.oWidth = embedDim;
        network.paramLayers.add(this);
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
        
        this.fusion_txt_proj = new FullyLayer(embedDim * 2, embedDim, true, network);
        RandomUtils.xavier_uniform(this.fusion_txt_proj.weight, 1, embedDim * 2, embedDim);
        if(this.fusion_txt_proj.bias != null) {
        	this.fusion_txt_proj.bias.clearGPU();
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

        if(e_m == null || e_m.number != batchSize * FT) {
        	this.e_m = Tensor.createGPUTensor(e_m, batchSize * FT, 1, 1, embedDim * 2, true);
        	this.t_e_m = Tensor.createGPUTensor(t_e_m, batchSize * TT, 1, 1, embedDim * 2, true);
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
    	if(diffWTxt == null) {
    		diffWTxt = Tensor.createGPUTensor(diffWTxt, weight.shape(), true);
    	}else {
    		diffWTxt.clearGPU();
    	}
    	if(diff == null || diff.number != number) {
    		diff = Tensor.createGPUTensor(diff, input.shape(), true);
    	}else {
    		diff.clearGPU();
    	}
    	if(diffTxt == null || diffTxt.number != batchSize * TT) {
    		diffTxt = Tensor.createGPUTensor(diffTxt, batchSize * TT, 1, 1, embedDim, true);
    	}else {
    		diffTxt.clearGPU();
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
    
    public void output(Tensor encoder, Tensor txt, Tensor e_txt) {
    	pdp = RandomUtils.randomFloat();
//    	pdp = 0.00001f;
    	if(network.RUN_MODEL == RunModel.TRAIN && path_drop_prob > 0 && pdp < path_drop_prob) {
    		pmKernel.set_mask_igone(weight, input, FT, 0, embedDim);
    		pmKernel.set_mask_igone(weight, txt, TT, 0, embedDim);
    	}
    	Tensor_OP().cat_width(encoder, input, e_m, embedDim, embedDim);
       	Tensor_OP().cat_width(e_txt, txt, t_e_m, embedDim, embedDim);
    	fusion_proj.forward(e_m);
    	fusion_txt_proj.forward(t_e_m);
    	this.output = fusion_proj.getOutput();
    }
    
    public void output_uncond(Tensor encoder, Tensor txt, Tensor e_txt) {
    	pmKernel.set_mask_igone(weight, input, FT, 0, embedDim);
		pmKernel.set_mask_igone(weight, txt, TT, 0, embedDim);
    	Tensor_OP().cat_width(encoder, input, e_m, embedDim, embedDim);
       	Tensor_OP().cat_width(e_txt, txt, t_e_m, embedDim, embedDim);
    	fusion_proj.forward(e_m);
    	fusion_txt_proj.forward(t_e_m);
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
    
    public void diff(Tensor dencoder, Tensor dtxt, Tensor de_txt) {
        // TODO Auto-generated method stub
    	fusion_proj.back(delta);
    	fusion_txt_proj.back(dtxt);
    	Tensor_OP().cat_width_back(dencoder, diff, fusion_proj.diff, embedDim, embedDim);
    	Tensor_OP().cat_width_back(de_txt, diffTxt, fusion_txt_proj.diff, embedDim, embedDim);
    	if(path_drop_prob > 0 && pdp < path_drop_prob) {
    		pmKernel.mask_igone_diff2(diff, diffW, batchSize, FT, embedDim, 0);
    		pmKernel.mask_igone_diff2(diffTxt, diffWTxt, batchSize, TT, embedDim, 0);
    		Tensor_OP().add(diffW, diffWTxt, diffW);
    		diff.clearGPU();
    		diffTxt.clearGPU();
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
    
    public void forward(Tensor input, Tensor encoder, Tensor txt, Tensor encoder_txt) {
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
        this.output(encoder, txt, encoder_txt);
    }
    
    public void forward_uncond(Tensor input, Tensor encoder, Tensor txt, Tensor encoder_txt) {
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
        this.output_uncond(encoder, txt, encoder_txt);
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
    
    public void back(Tensor delta, Tensor dencoder, Tensor txt_delta, Tensor txt_dencoder) {
        // TODO Auto-generated method stub
        this.initBack();
        /**
         * 设置梯度
         */
        this.setDelta(delta);
        /**
         * 计算梯度
         */
        this.diff(dencoder, txt_delta, txt_dencoder);
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
    	fusion_txt_proj.update();
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
    	fusion_txt_proj.saveModel(outputStream);
    }

    public void loadModel(RandomAccessFile inputStream) throws IOException {
    	ModelUtils.loadParams(inputStream, weight);
    	fusion_proj.loadModel(inputStream);
    	fusion_txt_proj.loadModel(inputStream);
    }

    @Override
    public void accGrad(float scale) {
        // TODO Auto-generated method stub
    	fusion_proj.accGrad(scale);
    	fusion_txt_proj.accGrad(scale);
    }
}

