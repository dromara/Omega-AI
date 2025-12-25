package com.omega.engine.nn.layer.dit.flux;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import com.omega.common.utils.MatrixOperation;
import com.omega.engine.gpu.BaseKernel;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.dit.DiTCaptionEmbeddingLayer;
import com.omega.engine.nn.layer.dit.DiTOrgTimeEmbeddingLayer;
import com.omega.engine.nn.layer.dit.DiTPatchEmbeddingLayer;
import com.omega.engine.nn.layer.dit.txt.DiT_TXTFinalLayer;
import com.omega.engine.nn.network.Network;
import com.omega.engine.nn.network.RunModel;
import com.omega.engine.tensor.Tensor;
import com.omega.engine.updater.UpdaterFactory;

/**
 * DiT_Block
 * @author Administrator
 */
public class FluxDiTMainMoudue_REPA extends Layer {
	
	public int inChannel;
    public int width;
    public int height;
    public int patchSize;
    private int hiddenSize;
    private int depth;
    private int timeSteps;
    private int headNum;
    private int textEmbedDim;
    private int maxContextLen;
    private int mlpRatio = 4;
    
    private int z_idx = 8;
    private int z_dim = 768;
    private int projector_dim = 2048;
    
    private boolean learnSigma = true;
    
    public DiTPatchEmbeddingLayer patchEmbd;
    public DiTOrgTimeEmbeddingLayer timeEmbd;
    public DiTCaptionEmbeddingLayer labelEmbd;
    public List<FluxDiTBlock> blocks;
    public DiT_TXTFinalLayer finalLayer;
    
    public REPAMLPLayer z_mlp;
    
    private int hw;
    
    private Tensor posEmbd;
    
    private Tensor cat_x;
    private Tensor img_x;
    
    private Tensor z_img_x;
    
    private Tensor dtc;
    private Tensor d_o;
    
    private float y_drop_prob = 0.0f;
    
    private int[] xShape;
    private int[] yShape;
    
    private BaseKernel baseKernel;
    
    public FluxDiTMainMoudue_REPA(int inChannel, int width, int height, int patchSize, int hiddenSize, int headNum, int depth, int timeSteps, int textEmbedDim, int maxContextLen, int mlpRatio, int z_idx, int z_dim, boolean learnSigma, float y_drop_prob, Network network) {
		this.network = network;
        if (this.updater == null) {
            this.setUpdater(UpdaterFactory.create(network));
        }
        this.y_drop_prob = y_drop_prob;
    	this.inChannel = inChannel;
		this.width = width;
		this.height = height;
		this.patchSize = patchSize;
		this.headNum = headNum;
		this.hiddenSize = hiddenSize;
		this.depth = depth;
		this.timeSteps = timeSteps;
		this.textEmbedDim = textEmbedDim;
		this.maxContextLen = maxContextLen;
		this.mlpRatio = mlpRatio;
		this.learnSigma = learnSigma;
		this.headNum = headNum;
		this.z_idx = z_idx;
		this.z_dim = z_dim;
//		this.bias = bias;
		this.initLayers();
		this.oHeight = height;
		this.oWidth = width;
    }

    public void initLayers() {
    	
    	patchEmbd = new DiTPatchEmbeddingLayer(inChannel, width, hiddenSize, patchSize, true, network);
        
    	hw = patchEmbd.oChannel;
    	
        timeEmbd = new DiTOrgTimeEmbeddingLayer(timeSteps, 256, hiddenSize, true, network);
        
        labelEmbd = new DiTCaptionEmbeddingLayer(textEmbedDim, hiddenSize, maxContextLen, y_drop_prob, true, network);
        
        blocks = new ArrayList<FluxDiTBlock>();
         
        for(int i = 0;i<depth;i++) {
        	FluxDiTBlock block = new FluxDiTBlock(hiddenSize, hiddenSize, patchEmbd.oChannel + maxContextLen, mlpRatio * hiddenSize, headNum, maxContextLen, true, false, network);
	        blocks.add(block);
        }
        int os = inChannel;
        if(learnSigma) {
        	os = inChannel * 2;
        }
        this.oChannel = os;
        finalLayer = new DiT_TXTFinalLayer(patchSize, hiddenSize, os, patchEmbd.oChannel, true, true, network);
        
        z_mlp = new REPAMLPLayer(hiddenSize, projector_dim, z_dim, true, network);
        
        if(baseKernel == null) {
        	baseKernel = new BaseKernel(cuda());
        }
        
    }

    @Override
    public void init() {
        // TODO Auto-generated method stub
        this.number = this.input.number;
    }
    
    public static float[] outer(float[] a, float[] b) {
        float[] o = new float[a.length * b.length];
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < b.length; j++) {
                o[i * b.length + j] = a[i] * b[j];
            }
        }
        return o;
    }
    
    public static float[] cat(float[] a, float[] b,int dims) {
        float[] o = new float[a.length + b.length];
        for (int i = 0; i < a.length; i++) {
        	int n = i / dims;
        	int w = i % dims;
            o[n * 2 * dims + 0 * dims + w] = a[i];
            o[n * 2 * dims + 1 * dims + w] = b[i];
        }
        return o;
    }
    
    public static float[] get_1d_sincos_pos_embed_from_grid(int embed_dim, float[] pos){
    	float[] omega = new float[embed_dim/2];
    	for(int i = 0;i<embed_dim/2;i++) {
    		float v = i * 1.0f / (embed_dim / 2.0f);
    		omega[i] = (float) (1.0f / Math.pow(10000, v));
    	}
    	float[] o = outer(pos, omega);
    	float[] cos = MatrixOperation.cos(o);
        float[] sin = MatrixOperation.sin(o);
        return cat(sin, cos, embed_dim/2);
    }
    
    public static float[] get_2d_cossin_pos_embed(int embed_dim,int grid_size) {
    	float[] grid_h = new float[grid_size * grid_size];
    	float[] grid_w = new float[grid_size * grid_size];
    	for(int i = 0;i<grid_size * grid_size;i++) {
    		int w = i % grid_size;
    		int h = i / grid_size;
    		grid_h[i] = w;
       		grid_w[i] = h;
    	}
//    	System.err.println("grid_h:"+JsonUtils.toJson(grid_h));
//    	System.err.println("grid_w:"+JsonUtils.toJson(grid_w));
    	float[] emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim/2, grid_h);
    	float[] emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim/2, grid_w);
    	
    	float[] emb = cat(emb_h, emb_w, embed_dim/2);
//    	System.err.println("emb:"+JsonUtils.toJson(emb));
    	return emb;
    }
    
    public void init(Tensor input) {
        // TODO Auto-generated method stub
        this.number = input.number;
        if(this.output == null || this.output.number != number) {
        	cat_x = Tensor.createGPUTensor(cat_x, number * (patchEmbd.oChannel + maxContextLen), 1, 1, patchEmbd.oWidth, true);
        	img_x = Tensor.createGPUTensor(img_x, number * patchEmbd.oChannel, 1, 1, patchEmbd.oWidth, true);
        	z_img_x = Tensor.createGPUTensor(z_img_x, number * patchEmbd.oChannel, 1, 1, patchEmbd.oWidth, true);
        	output = Tensor.createGPUTensor(output, number, oChannel, oHeight, oWidth, true);
        }
        if(posEmbd == null) {
        	posEmbd = new Tensor(1, patchEmbd.oChannel, 1, hiddenSize, get_2d_cossin_pos_embed(hiddenSize, width/patchSize), true);
        }
        if(patchEmbd.getOutput() != null){
        	patchEmbd.getOutput().viewOrg();
        }
    }
    
    @Override
    public void initBack() {
        // TODO Auto-generated method stub
    	if(dtc == null || dtc.number != timeEmbd.getOutput().number) {
    		dtc = Tensor.createGPUTensor(dtc, timeEmbd.getOutput().shape(), true);
    		d_o = Tensor.createGPUTensor(d_o, input.number * (maxContextLen + hw), 1, 1, patchEmbd.getOutput().width, true);
    	}else {
    		dtc.clearGPU();
    		d_o.clearGPU();
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
    
    public void output(Tensor tc,Tensor label) {
    	
    	patchEmbd.forward(input);

    	Tensor_OP().addAxis(patchEmbd.getOutput(), posEmbd, patchEmbd.getOutput(), posEmbd.channel * posEmbd.width);

    	timeEmbd.forward(tc);
    	
    	labelEmbd.forward(label);

    	Tensor x = patchEmbd.getOutput().view(patchEmbd.getOutput().number * patchEmbd.getOutput().channel, 1, 1, patchEmbd.getOutput().width);
    	
    	Tensor t = timeEmbd.getOutput();
    	
    	Tensor cond = labelEmbd.getOutput();
    	
    	baseKernel.concat_channel_forward(cond, x, cat_x, input.number, maxContextLen, hw, 1, patchEmbd.getOutput().width);
    	
    	Tensor bx = cat_x;

    	for(int i = 0;i<depth;i++) {
    		FluxDiTBlock block = blocks.get(i);
    		block.forward(bx, t);
    		bx = block.getOutput();
    		if(network.RUN_MODEL == RunModel.TRAIN && i == z_idx - 1) {
    			Tensor_OP().getByChannel(bx, z_img_x, new int[] {input.number, maxContextLen + hw, 1, patchEmbd.getOutput().width}, maxContextLen, hw);
    			z_mlp.forward(z_img_x);
    		}
    	}
    	
    	Tensor_OP().getByChannel(bx, img_x, new int[] {input.number, maxContextLen + hw, 1, patchEmbd.getOutput().width}, maxContextLen, hw);

    	finalLayer.forward(img_x, t);
    	
    	/**
    	 * unpatchify
    	 * x: (N, T, patch_size**2 * C)
         * imgs: (N, C, H, W)
    	 */
    	if(xShape == null) {
    		int h = height/patchSize;
        	int w = width/patchSize;
        	xShape = new int[] {number, h, w, patchSize, patchSize, oChannel};
        	yShape = new int[] {number, oChannel, h, patchSize, w, patchSize};
    	}
    	
    	Tensor_OP().permute(finalLayer.getOutput(), this.output, xShape, yShape, new int[] {0, 5, 1, 3, 2, 4});
    	
    }
    
    public void output(Tensor tc,Tensor label,Tensor cos,Tensor sin) {

    	patchEmbd.forward(input);

    	Tensor_OP().addAxis(patchEmbd.getOutput(), posEmbd, patchEmbd.getOutput(), posEmbd.channel * posEmbd.width);
    	
    	timeEmbd.forward(tc);
    	
    	labelEmbd.forward(label);
    	
    	Tensor x = patchEmbd.getOutput().view(patchEmbd.getOutput().number * patchEmbd.getOutput().channel, 1, 1, patchEmbd.getOutput().width);
    	
    	Tensor t = timeEmbd.getOutput();
    	
    	Tensor cond = labelEmbd.getOutput();
    	
    	baseKernel.concat_channel_forward(cond, x, cat_x, input.number, maxContextLen, hw, 1, patchEmbd.getOutput().width);
    	
    	Tensor bx = cat_x;

    	for(int i = 0;i<depth;i++) {
    		FluxDiTBlock block = blocks.get(i);
    		block.forward(bx, t, cos, sin);
    		bx = block.getOutput();
    		if(network.RUN_MODEL == RunModel.TRAIN && i == z_idx - 1) {
    			Tensor_OP().getByChannel(bx, z_img_x, new int[] {input.number, maxContextLen + hw, 1, patchEmbd.getOutput().width}, maxContextLen, hw);
    			z_mlp.forward(z_img_x);
    		}
    	}

    	//img_o = x[:, txt.shape[1]:, ...]
    	Tensor_OP().getByChannel(bx, img_x, new int[] {input.number, maxContextLen + hw, 1, patchEmbd.getOutput().width}, maxContextLen, hw);
    	
    	finalLayer.forward(img_x, t);
    	
    	/**
    	 * unpatchify
    	 * x: (N, T, patch_size**2 * C)
         * imgs: (N, C, H, W)
    	 */
    	if(xShape == null) {
    		int h = height/patchSize;
        	int w = width/patchSize;
        	xShape = new int[] {number, h, w, patchSize, patchSize, oChannel};
        	yShape = new int[] {number, oChannel, h, patchSize, w, patchSize};
    	}
    	Tensor_OP().permute(finalLayer.getOutput(), this.output, xShape, yShape, new int[] {0, 5, 1, 3, 2, 4});

    }

    @Override
    public Tensor getOutput() {
        // TODO Auto-generated method stub
        return output;
    }

    @Override
    public void diff() {

    	/**
    	 * unpatchify back
    	 */
//    	int h = height/patchSize;
//    	int w = width/patchSize;
//    	int[] yShape = new int[] {number, oChannel, h, patchSize, w, patchSize};
//    	int[] xShape = new int[] {number, h, w, patchSize, patchSize, oChannel};
    	Tensor_OP().permute(delta, finalLayer.getOutput(), yShape, xShape, new int[] {0, 2, 4, 3, 5, 1});
    	
    	finalLayer.back(finalLayer.getOutput(), dtc);
    	
    	Tensor dy = d_o;
    	
    	Tensor_OP().getByChannel_back(dy, finalLayer.diff, new int[] {input.number, maxContextLen + hw, 1, patchEmbd.getOutput().width}, maxContextLen, hw);
    	
     	for(int i = depth - 1;i>=0;i--) {
     		if(i == z_idx - 1) {
     	    	Tensor_OP().getByChannel_add_back(dy, z_mlp.diff, new int[] {input.number, maxContextLen + hw, 1, patchEmbd.getOutput().width}, maxContextLen, hw);
     		}
     		FluxDiTBlock block = blocks.get(i);
    		block.back(dy, dtc);
    		dy = block.diff;
    	}
     	
     	baseKernel.concat_channel_backward(dy, labelEmbd.getOutput(), img_x, input.number, maxContextLen, hw, 1, patchEmbd.getOutput().width);

//     	dtxt.showDM("dtxt");
     	labelEmbd.back(labelEmbd.getOutput());
     	
//     	dtc.showDM("dtc");
     	timeEmbd.back(dtc);

     	patchEmbd.back(img_x);
    }
    
    public void diff(Tensor cos,Tensor sin) {
        // TODO Auto-generated method stub

    	/**
    	 * unpatchify back
    	 */
//    	int h = height/patchSize;
//    	int w = width/patchSize;
//    	int[] yShape = new int[] {number, oChannel, h, patchSize, w, patchSize};
//    	int[] xShape = new int[] {number, h, w, patchSize, patchSize, oChannel};
    	Tensor_OP().permute(delta, finalLayer.getOutput(), yShape, xShape, new int[] {0, 2, 4, 3, 5, 1});
    	
    	finalLayer.back(finalLayer.getOutput(), dtc);

    	Tensor dy = d_o;
    	dy.clearGPU();

    	Tensor_OP().getByChannel_back(dy, finalLayer.diff, new int[] {input.number, maxContextLen + hw, 1, patchEmbd.getOutput().width}, maxContextLen, hw);

     	for(int i = depth - 1;i>=0;i--) {
     		if(i == z_idx - 1) {
     			Tensor_OP().getByChannel_add_back(dy, z_mlp.diff, new int[] {input.number, maxContextLen + hw, 1, patchEmbd.getOutput().width}, maxContextLen, hw);
     		}
     		FluxDiTBlock block = blocks.get(i);
    		block.back(dy, dtc, cos, sin);
    		dy = block.diff;
    	}

     	baseKernel.concat_channel_backward(dy, labelEmbd.getOutput(), img_x, input.number, maxContextLen, hw, 1, patchEmbd.getOutput().width);

     	labelEmbd.back(labelEmbd.getOutput());
     	
     	timeEmbd.back(dtc);

     	patchEmbd.back(img_x);
     	
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
    
    /**
     * 
     * @param input
     * @param tc time cond
     * @param text
     */
    public void forward(Tensor input,Tensor tc,Tensor text) {
        // TODO Auto-generated method stub
        /**
         * 设置输入
         */
        this.setInput(input);
        /**
         * 参数初始化
         */
        this.init(input);
        /**
         * 计算输出
         */
        this.output(tc, text);
    }
    
    /**
     * 
     * @param input
     * @param tc time cond
     * @param text
     */
    public void forward(Tensor input,Tensor tc,Tensor text, Tensor cos, Tensor sin) {
        // TODO Auto-generated method stub
        /**
         * 设置输入
         */
        this.setInput(input);
        /**
         * 参数初始化
         */
        this.init(input);
        /**
         * 计算输出
         */
        this.output(tc, text, cos, sin);
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
    
    public void back(Tensor delta,Tensor cos,Tensor sin) {
        // TODO Auto-generated method stub
        this.initBack();
        /**
         * 设置梯度

         */
        this.setDelta(delta);
        /**
         * 计算梯度
         */
        this.diff(cos, sin);
//        if (this.network.GRADIENT_CHECK) {
//            this.gradientCheck();
//        }
    }
    
    @Override
    public void update() {
        // TODO Auto-generated method stub
    	patchEmbd.update();

    	timeEmbd.update();
    	
    	labelEmbd.update();
    	
    	for(int i = 0;i<depth;i++) {
    		blocks.get(i).update();
    	}
    	
    	z_mlp.update();
    	
    	finalLayer.update();
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
    	patchEmbd.saveModel(outputStream);

    	timeEmbd.saveModel(outputStream);

    	labelEmbd.saveModel(outputStream);
    	
    	for(int i = 0;i<depth;i++) {
    		blocks.get(i).saveModel(outputStream);
    	}

    	z_mlp.saveModel(outputStream);
    	
    	finalLayer.saveModel(outputStream);
    }

    public void loadModel(RandomAccessFile inputStream) throws IOException {
    	patchEmbd.loadModel(inputStream);

    	timeEmbd.loadModel(inputStream);
    	
    	labelEmbd.loadModel(inputStream);
    	
    	for(int i = 0;i<depth;i++) {
    		blocks.get(i).loadModel(inputStream);
    	}
    	
    	z_mlp.loadModel(inputStream);
    	
    	finalLayer.loadModel(inputStream);
    }

    @Override
    public void accGrad(float scale) {
        // TODO Auto-generated method stub
    	patchEmbd.accGrad(scale);

    	timeEmbd.accGrad(scale);

    	labelEmbd.accGrad(scale);
    	
    	for(int i = 0;i<depth;i++) {
    		blocks.get(i).accGrad(scale);
    	}
    	
    	z_mlp.accGrad(scale);
    	
    	finalLayer.accGrad(scale);
    }
    
    public static void loadWeight(Map<String, Object> weightMap, FluxDiTMainMoudue_REPA block, boolean showLayers) {
        if (showLayers) {
            for (String key : weightMap.keySet()) {
                System.out.println(key);
            }
        }

    }
    
    public static void main(String[] args) {
    	
    }
    
    public Tensor getZ() {
    	return z_mlp.getOutput();
    }
    
    public void setZGrad(Tensor delta) {
    	z_mlp.back(delta);
    }
    
}

