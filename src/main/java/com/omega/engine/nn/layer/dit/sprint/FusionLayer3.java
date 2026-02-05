package com.omega.engine.nn.layer.dit.sprint;

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
 * FusionLayer
 *
 * @author Administrator
 */
public class FusionLayer3 extends Layer {
	
	private int batchSize;
    private int embedDim = 0;
    private int T;
    private int FT;
    private int TT = 0;
    
    private float path_drop_prob = 0.0f;

    public FullyLayer fusion_proj;
    
    public FullyLayer fusion_proj_t;
    
    private Tensor g_pad;
    private Tensor e_m;
    private Tensor c_m;
    
    private PaddingMaskKernel pmKernel;
    
    private float pdp = 0.0f;

    public FusionLayer3(int embedDim, int FT, int T, Network network) {
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
    
    public FusionLayer3(int embedDim, int FT, int T, int TT, Network network) {
        this.network = network;
        if (this.updater == null) {
            this.setUpdater(UpdaterFactory.create(network));
        }
        this.T = T;
        this.FT = FT;
        this.TT = TT;
        this.embedDim = embedDim;
        this.oChannel = 1;
        this.oHeight = 1;
        this.oWidth = embedDim;
        this.initLayers();
    }
    
    public FusionLayer3(int embedDim, int FT, int T, int TT, float path_drop_prob, Network network) {
        this.network = network;
        if (this.updater == null) {
            this.setUpdater(UpdaterFactory.create(network));
        }
        this.T = T;
        this.FT = FT;
        this.TT = TT;
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
//    	this.freeze = true;
    	this.weight = new Tensor(1, 1, 1, embedDim, true);
    	
        this.fusion_proj = new FullyLayer(embedDim * 2, embedDim, true, network);
        RandomUtils.xavier_uniform(this.fusion_proj.weight, 1, embedDim * 2, embedDim);
        if(this.fusion_proj.bias != null) {
        	this.fusion_proj.bias.clearGPU();
        }
        
        this.fusion_proj_t = new FullyLayer(embedDim * 2, embedDim, true, network);
        RandomUtils.xavier_uniform(this.fusion_proj_t.weight, 1, embedDim * 2, embedDim);
        if(this.fusion_proj_t.bias != null) {
        	this.fusion_proj_t.bias.clearGPU();
        }
        
        if(pmKernel == null) {
        	pmKernel = new PaddingMaskKernel(cuda());
        }
        
    }

    @Override
    public void init() {
        // TODO Auto-generated method stub
        this.number = this.input.number;
        this.batchSize = number / T;
        if(number != batchSize * FT && (g_pad == null || g_pad.number != batchSize * FT)) {
        	this.g_pad = Tensor.createGPUTensor(g_pad, batchSize * FT, 1, 1, embedDim, true);
        }
        if(e_m == null || e_m.number != batchSize * FT) {
        	this.e_m = Tensor.createGPUTensor(e_m, batchSize * FT, 1, 1, embedDim * 2, true);
        	this.c_m = Tensor.createGPUTensor(c_m, batchSize * TT, 1, 1, embedDim * 2, true);
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
    
    public void output(Tensor encoder, Tensor h_c, Tensor e_c) {
    	pdp = RandomUtils.randomFloat();
//    	pdp = 0.0001f;
    	if(network.RUN_MODEL == RunModel.TRAIN && path_drop_prob > 0 && pdp < path_drop_prob) {
    		pmKernel.set_mask_igone(weight, input, FT, 0, embedDim);
    		pmKernel.set_mask_igone(weight, h_c, TT, 0, embedDim);
    	}
    	Tensor_OP().cat_width(encoder, input, e_m, embedDim, embedDim);
    	Tensor_OP().cat_width(e_c, h_c, c_m, embedDim, embedDim);
    	fusion_proj.forward(e_m);
       	fusion_proj_t.forward(c_m);
    	this.output = fusion_proj.getOutput();
    }
    
    public Tensor getCondOutput() {
    	return this.fusion_proj_t.getOutput();
    }
    
    public void output_uncond(Tensor encoder, Tensor h_c, Tensor e_c) {
    	pmKernel.set_mask_igone(weight, input, FT, 0, embedDim);
    	pmKernel.set_mask_igone(weight, h_c, TT, 0, embedDim);
    	Tensor_OP().cat_width(encoder, input, e_m, embedDim, embedDim);
    	Tensor_OP().cat_width(e_c, h_c, c_m, embedDim, embedDim);
    	fusion_proj.forward(e_m);
       	fusion_proj_t.forward(c_m);
    	this.output = fusion_proj.getOutput();
    }
    
    public void output(Tensor encoder, Tensor h_c, Tensor e_c, Tensor idskeep) {
    	pmKernel.forward(input, weight, idskeep, g_pad, FT, T, 0, embedDim);
    	if(network.RUN_MODEL == RunModel.TRAIN) {
        	pdp = RandomUtils.randomFloat();
//        	pdp = 0.001f;
        	if(path_drop_prob > 0 && pdp < path_drop_prob) {
//        		System.err.println("in:"+pdp);
        		pmKernel.set_mask_igone(weight, g_pad, FT, 0, embedDim);
        		pmKernel.set_mask_igone(weight, h_c, TT, 0, embedDim);
        	}
    	}
    	Tensor_OP().cat_width(encoder, g_pad, e_m, embedDim, embedDim);
    	Tensor_OP().cat_width(e_c, h_c, c_m, embedDim, embedDim);
    	fusion_proj.forward(e_m);
    	fusion_proj_t.forward(c_m);
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
    
    public void diff(Tensor delta_c, Tensor dencoder, Tensor d_hc, Tensor d_ec, Tensor idskeep) {
        // TODO Auto-generated method stub
    	fusion_proj.back(delta);
    	fusion_proj_t.back(delta_c);
    	Tensor_OP().cat_width_back(dencoder, g_pad, fusion_proj.diff, embedDim, embedDim);
    	Tensor_OP().cat_width_back(d_ec, d_hc, fusion_proj_t.diff, embedDim, embedDim);
    	if(path_drop_prob > 0  && pdp < path_drop_prob) {
    		pmKernel.mask_igone_diff(g_pad, diffW, g_pad.number, embedDim, FT, 0);
       		pmKernel.mask_igone_diff_cond(d_hc, diffW, d_hc.number, embedDim, TT);
    		diff.clearGPU();
    		d_hc.clearGPU();
    	}else {
    		pmKernel.backward(g_pad, idskeep, diff, diffW, FT, T, 0, embedDim);
    	}
//    	diff.showDM("diff");
//    	diffW.showDM("diffW");
    }
    
    public void diff(Tensor delta_c, Tensor dencoder, Tensor d_hc, Tensor d_ec) {
        // TODO Auto-generated method stub
    	fusion_proj.back(delta);
    	fusion_proj_t.back(delta_c);
    	Tensor_OP().cat_width_back(dencoder, diff, fusion_proj.diff, embedDim, embedDim);
    	Tensor_OP().cat_width_back(d_ec, d_hc, fusion_proj_t.diff, embedDim, embedDim);
    	if(path_drop_prob > 0 && pdp < path_drop_prob) {
    		pmKernel.mask_igone_diff(diff, diffW, diff.number, embedDim, FT, 0);
    		pmKernel.mask_igone_diff_cond(d_hc, diffW, d_hc.number, embedDim, TT);
    		diff.clearGPU();
    		d_hc.clearGPU();
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
    
    public void forward(Tensor input, Tensor encoder, Tensor h_c, Tensor e_c) {
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
        this.output(encoder, h_c, e_c);
    }
    
    public void forward_uncond(Tensor input, Tensor encoder, Tensor h_c, Tensor e_c) {
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
        this.output_uncond(encoder, h_c, e_c);
    }
    
    public void forward(Tensor input, Tensor encoder, Tensor h_c, Tensor e_c, Tensor idskeep) {
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
        this.output(encoder, h_c, e_c, idskeep);
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
    
    public void back(Tensor delta, Tensor delta_c, Tensor dencoder, Tensor d_hc, Tensor d_ec) {
        // TODO Auto-generated method stub
        this.initBack();
        /**
         * 设置梯度
         */
        this.setDelta(delta);
        /**
         * 计算梯度
         */
        this.diff(delta_c, dencoder, d_hc, d_ec);
    }
    
    public void back(Tensor delta, Tensor delta_c, Tensor dencoder, Tensor d_hc, Tensor d_ec, Tensor idskeep) {
        // TODO Auto-generated method stub
        this.initBack();
        /**
         * 设置梯度
         */
        this.setDelta(delta);
        /**
         * 计算梯度
         */
        this.diff(delta_c, dencoder, d_hc, d_ec, idskeep);
    }
    
    @Override
    public void update() {
        // TODO Auto-generated method stub
    	if (!this.freeze) {
//    		diffW.showDM("diffW");
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
    	fusion_proj_t.update();
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
    	fusion_proj_t.saveModel(outputStream);
    }

    public void loadModel(RandomAccessFile inputStream) throws IOException {
    	ModelUtils.loadParams(inputStream, weight);
    	fusion_proj.loadModel(inputStream);
    	fusion_proj_t.loadModel(inputStream);
    }

    @Override
    public void accGrad(float scale) {
        // TODO Auto-generated method stub
    	fusion_proj.accGrad(scale);
    	fusion_proj_t.accGrad(scale);
    }
}

