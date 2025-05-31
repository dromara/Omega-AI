package com.omega.engine.nn.layer.dit;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import com.omega.common.utils.MatrixOperation;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.dit.modules.DiTSimpleHeadLayer;
import com.omega.engine.nn.network.Network;
import com.omega.engine.nn.network.RunModel;
import com.omega.engine.tensor.Tensor;
import com.omega.engine.updater.UpdaterFactory;

/**
 * DiT_Block
 * @author Administrator
 */
public class DiTOrgMoudue_SRA extends Layer {
	
	public int inChannel;
    public int width;
    public int height;
    public int patchSize;
    public int maxContextLen;
    private int hiddenSize;
    private int depth;
    private int timeSteps;
    private int headNum;
    private int textEmbedDim;
    private int mlpRatio = 4;
    private boolean learnSigma = true;
    private boolean hasBias = true;
    private boolean qkNorm = false;
    
    private int ad = 0;
    
    public DiTPatchEmbeddingLayer patchEmbd;
    public DiTTimeEmbeddingLayer timeEmbd;
    public DiTCaptionEmbeddingLayer labelEmbd;
    public List<DiTOrgBlock> blocks;
    public DiTSimpleHeadLayer ap_head;
    public DiTFinalLayer finalLayer;
    
    private Tensor posEmbd;
    
    private Tensor dtc;
    private Tensor dtext;
    
    public Tensor xr;
    
    public DiTOrgMoudue_SRA(int inChannel, int width, int height, int patchSize, int hiddenSize, int headNum, int depth, int timeSteps, int maxContextLen, int textEmbedDim, int mlpRatio, int ad, boolean learnSigma, boolean hasBias, boolean qkNorm, Network network) {
		this.network = network;
        if (this.updater == null) {
            this.setUpdater(UpdaterFactory.create(network));
        }
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
		this.qkNorm = qkNorm;
		this.hasBias = hasBias;
		this.ad = ad;
		this.initLayers();
		this.oHeight = height;
		this.oWidth = width;
    }

    public void initLayers() {
    	
    	patchEmbd = new DiTPatchEmbeddingLayer(inChannel, width, hiddenSize, patchSize, hasBias, network);
         
        timeEmbd = new DiTTimeEmbeddingLayer(timeSteps, 256, hiddenSize, hasBias, network);
        
        labelEmbd = new DiTCaptionEmbeddingLayer(textEmbedDim, hiddenSize, hasBias, network);
        
        blocks = new ArrayList<DiTOrgBlock>();
         
        for(int i = 0;i<depth;i++) {
        	DiTOrgBlock block = new DiTOrgBlock(hiddenSize, hiddenSize, hiddenSize, patchEmbd.oChannel, maxContextLen, mlpRatio * hiddenSize, headNum, hasBias, qkNorm, network);
	        blocks.add(block);
        }
        int os = inChannel;
        if(learnSigma) {
        	os = inChannel * 2;
        }
        
        this.ap_head = new DiTSimpleHeadLayer(hiddenSize, hiddenSize, true, network);
        
        this.oChannel = os;
        finalLayer = new DiTFinalLayer(patchSize, hiddenSize, os, patchEmbd.oChannel, hasBias, network);

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
    		dtext = Tensor.createGPUTensor(dtext, timeEmbd.getOutput().number * maxContextLen, 1, 1, hiddenSize, true);
    	}else {
    		dtext.clearGPU();
    		dtc.clearGPU();
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
    
    public void output(Tensor tc,Tensor text) {
    	
    	patchEmbd.forward(input);
    	Tensor_OP().addAxis(patchEmbd.getOutput(), posEmbd, patchEmbd.getOutput(), patchEmbd.getOutput().number, patchEmbd.getOutput().channel, 1, patchEmbd.getOutput().getWidth(), 1);

    	timeEmbd.forward(tc);
    	
    	labelEmbd.forward(text);

    	Tensor x = patchEmbd.getOutput().view(patchEmbd.getOutput().number * patchEmbd.getOutput().channel, 1, 1, patchEmbd.getOutput().width);
    	
    	for(int i = 0;i<depth;i++) {
    		DiTOrgBlock block = blocks.get(i);
    		block.forward(x, timeEmbd.getOutput(), labelEmbd.getOutput());
    		x = block.getOutput();
    		if(i + 1 == ad){
    			if(network.RUN_MODEL == RunModel.TRAIN) {
    				ap_head.forward(x);
        			xr = ap_head.getOutput();
    			}else {
        			xr = x;
    			}
    		}
    	}

    	finalLayer.forward(x, timeEmbd.getOutput());
    	
    	/**
    	 * unpatchify
    	 * x: (N, T, patch_size**2 * C)
         * imgs: (N, C, H, W)
    	 */
    	int h = height/patchSize;
    	int w = width/patchSize;
    	int[] xShape = new int[] {number, h, w, patchSize, patchSize, oChannel};
    	int[] yShape = new int[] {number, oChannel, h, patchSize, w, patchSize};
    	Tensor_OP().permute(finalLayer.getOutput(), this.output, xShape, yShape, new int[] {0, 5, 1, 3, 2, 4});
    	
    }
    
    public void output(Tensor tc,Tensor text,Tensor cos,Tensor sin) {
    	
    	patchEmbd.forward(input);

    	timeEmbd.forward(tc);
    	
    	labelEmbd.forward(text);
    	
    	Tensor x = patchEmbd.getOutput().view(patchEmbd.getOutput().number * patchEmbd.getOutput().channel, 1, 1, patchEmbd.getOutput().width);

    	for(int i = 0;i<depth;i++) {
    		DiTOrgBlock block = blocks.get(i);
    		block.forward(x, timeEmbd.getOutput(), labelEmbd.getOutput(), cos, sin);
    		x = block.getOutput();
    		if(i + 1 == ad){
    			if(network.RUN_MODEL == RunModel.TRAIN) {
    				ap_head.forward(x);
        			xr = ap_head.getOutput();
    			}else {
        			xr = x;
    			}
    		}
//    		x.showDM("x["+i+"]");
    	}
    	
    	finalLayer.forward(x, timeEmbd.getOutput());
    	
    	/**
    	 * unpatchify
    	 * x: (N, T, patch_size**2 * C)
         * imgs: (N, C, H, W)
    	 */
    	int h = height/patchSize;
    	int w = width/patchSize;

    	int[] xShape = new int[] {number, h, w, patchSize, patchSize, oChannel};
    	int[] yShape = new int[] {number, oChannel, h, patchSize, w, patchSize};
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
    	int h = height/patchSize;
    	int w = width/patchSize;
    	int[] yShape = new int[] {number, oChannel, h, patchSize, w, patchSize};
    	int[] xShape = new int[] {number, h, w, patchSize, patchSize, oChannel};
    	Tensor_OP().permute(delta, finalLayer.getOutput(), yShape, xShape, new int[] {0, 2, 4, 3, 5, 1});
    	
    	finalLayer.back(finalLayer.getOutput(), dtc);

    	Tensor dy = finalLayer.diff;

     	for(int i = depth - 1;i>=0;i--) {
     		if(i + 1 == ad) {
     			Tensor_OP().add(dy, ap_head.diff, dy);
     		}
     		DiTOrgBlock block = blocks.get(i);
    		block.back(dy, dtc, dtext);
    		dy = block.diff;
    	}
     	
     	labelEmbd.back(dtext);
     	
//     	dtc.showDM("dtc");
     	timeEmbd.back(dtc);

     	patchEmbd.back(dy);
    }
    
    public void diff(Tensor cos,Tensor sin) {
        // TODO Auto-generated method stub
//    	delta.showDM("total-delta");
    	/**
    	 * unpatchify back
    	 */
    	int h = height/patchSize;
    	int w = width/patchSize;
    	int[] yShape = new int[] {number, oChannel, h, patchSize, w, patchSize};
    	int[] xShape = new int[] {number, h, w, patchSize, patchSize, oChannel};
    	Tensor_OP().permute(delta, finalLayer.getOutput(), yShape, xShape, new int[] {0, 2, 4, 3, 5, 1});
    	
    	finalLayer.back(finalLayer.getOutput(), dtc);
    	
    	Tensor dy = finalLayer.diff;
//    	dy.showDM("in-block-diff");
     	for(int i = depth - 1;i>=0;i--) {
     		if(i + 1 == ad) {
     			Tensor_OP().add(dy, ap_head.diff, dy);
     		}
     		DiTOrgBlock block = blocks.get(i);
    		block.back(dy, dtc, dtext, cos, sin);
    		dy = block.diff;
//    		dy.showDM("in-block-diff");
    	}
     	
     	labelEmbd.back(dtext);
     	
     	timeEmbd.back(dtc);
//     	dy.showDM("block-diff");
     	patchEmbd.back(dy);
     	
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
        if (this.network.GRADIENT_CHECK) {
            this.gradientCheck();
        }
    }
    
    @Override
    public void update() {
        // TODO Auto-generated method stub
    	patchEmbd.update();

    	timeEmbd.update();
    	
    	labelEmbd.update();
    	
    	ap_head.update();
    	
    	for(int i = 0;i<depth;i++) {
    		blocks.get(i).update();
    	}
    	
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
    	
    	ap_head.saveModel(outputStream);
    	
    	for(int i = 0;i<depth;i++) {
    		blocks.get(i).saveModel(outputStream);
    	}
    	
    	finalLayer.saveModel(outputStream);
    }

    public void loadModel(RandomAccessFile inputStream) throws IOException {
    	patchEmbd.loadModel(inputStream);

    	timeEmbd.loadModel(inputStream);
    	
    	labelEmbd.loadModel(inputStream);
    	
    	ap_head.loadModel(inputStream);
    	
    	for(int i = 0;i<depth;i++) {
    		blocks.get(i).loadModel(inputStream);
    	}
    	
    	finalLayer.loadModel(inputStream);
    }

    @Override
    public void accGrad(float scale) {
        // TODO Auto-generated method stub
    	patchEmbd.accGrad(scale);

    	timeEmbd.accGrad(scale);

    	labelEmbd.accGrad(scale);
    	
    	ap_head.accGrad(scale);
    	
    	for(int i = 0;i<depth;i++) {
    		blocks.get(i).accGrad(scale);
    	}
    	
    	finalLayer.accGrad(scale);
    }
    
    public static void loadWeight(Map<String, Object> weightMap, DiTOrgMoudue_SRA block, boolean showLayers) {
        if (showLayers) {
            for (String key : weightMap.keySet()) {
                System.out.println(key);
            }
        }
        
    }
    
    public static void main(String[] args) {
    	int embed_dim = 8;
    	int grid_size = 4;
    	get_2d_cossin_pos_embed(embed_dim, grid_size);
    	
    }
}

