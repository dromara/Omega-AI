package com.omega.engine.nn.layer.dit.txt;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import com.omega.common.utils.MatrixOperation;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.dit.DiTCaptionEmbeddingLayer;
import com.omega.engine.nn.layer.dit.DiTPatchEmbeddingLayer;
import com.omega.engine.nn.layer.dit.DiTTimeEmbeddingLayer;
import com.omega.engine.nn.layer.gpu.RoPEKernel;
import com.omega.engine.nn.network.CNN;
import com.omega.engine.nn.network.Network;
import com.omega.engine.tensor.Tensor;
import com.omega.engine.updater.UpdaterFactory;
import com.omega.example.common.ModeLoaderlUtils;
import com.omega.example.transformer.utils.LagJsonReader;

/**
 * DiT_Block
 * @author Administrator
 */
public class DiT_TXTMainMoudue extends Layer {
	
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
    private boolean learnSigma = true;
    
    public DiTPatchEmbeddingLayer patchEmbd;
    public DiTTimeEmbeddingLayer timeEmbd;
    public DiTCaptionEmbeddingLayer labelEmbd;
    public List<DiT_TXTBlock> blocks;
    public DiT_TXTFinalLayer finalLayer;
    
    private Tensor posEmbd;
    
    private Tensor dtc;
    private Tensor dtxt;
    
    private float y_drop_prob = 0.0f;
    
    public DiT_TXTMainMoudue(int inChannel, int width, int height, int patchSize, int hiddenSize, int headNum, int depth, int timeSteps, int textEmbedDim, int maxContextLen, int mlpRatio, boolean learnSigma, float y_drop_prob, Network network) {
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
//		this.bias = bias;
		this.initLayers();
		this.oHeight = height;
		this.oWidth = width;
    }

    public void initLayers() {
    	
    	patchEmbd = new DiTPatchEmbeddingLayer(inChannel, width, hiddenSize, patchSize, true, network);
         
        timeEmbd = new DiTTimeEmbeddingLayer(timeSteps, 256, hiddenSize, true, network);
        
        labelEmbd = new DiTCaptionEmbeddingLayer(textEmbedDim, hiddenSize, maxContextLen, y_drop_prob, true, network);
        
        blocks = new ArrayList<DiT_TXTBlock>();
         
        for(int i = 0;i<depth;i++) {
        	DiT_TXTBlock block = new DiT_TXTBlock(hiddenSize, hiddenSize, hiddenSize, patchEmbd.oChannel, maxContextLen, mlpRatio * hiddenSize, headNum, true, false, network);
	        blocks.add(block);
        }
        int os = inChannel;
        if(learnSigma) {
        	os = inChannel * 2;
        }
        this.oChannel = os;
        finalLayer = new DiT_TXTFinalLayer(patchSize, hiddenSize, os, patchEmbd.oChannel, true, true, network);

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
    		dtxt = Tensor.createGPUTensor(dtxt, labelEmbd.getOutput().shape(), true);
    	}else {
    		dtc.clearGPU();
    		dtxt.clearGPU();
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
    	
//    	cond.showDM("cond");
    	
    	for(int i = 0;i<depth;i++) {
    		DiT_TXTBlock block = blocks.get(i);
    		block.forward(x, t, cond);
    		x = block.getOutput();
    	}
    	
    	finalLayer.forward(x, t);
    	
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
    
    public void output(Tensor tc,Tensor label,Tensor cos,Tensor sin) {
    	
    	patchEmbd.forward(input);
    	
    	Tensor_OP().addAxis(patchEmbd.getOutput(), posEmbd, patchEmbd.getOutput(), posEmbd.channel * posEmbd.width);
    	
    	timeEmbd.forward(tc);
    	
    	labelEmbd.forward(label);
    	
    	Tensor x = patchEmbd.getOutput().view(patchEmbd.getOutput().number * patchEmbd.getOutput().channel, 1, 1, patchEmbd.getOutput().width);
    	
    	Tensor t = timeEmbd.getOutput();
    	
    	Tensor cond = labelEmbd.getOutput();

    	for(int i = 0;i<depth;i++) {
    		DiT_TXTBlock block = blocks.get(i);
    		block.forward(x, t, cond, cos, sin);
    		x = block.getOutput();
    	}
    	
    	finalLayer.forward(x, t);
    	
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
     		DiT_TXTBlock block = blocks.get(i);
    		block.back(dy, dtc, dtxt);
    		dy = block.diff;
    	}
//     	dtxt.showDM("dtxt");
     	labelEmbd.back(dtxt);
     	
//     	dtc.showDM("dtc");
     	timeEmbd.back(dtc);

     	patchEmbd.back(dy);
    }
    
    public void diff(Tensor cos,Tensor sin) {
        // TODO Auto-generated method stub
//    	delta.showDM("total-delta");
//    	delta.showDMByOffsetRed(0,10, "delta");
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
//    	dy.showDMByOffsetRed(0,10, "dy");
//    	dy.showDM("in-block-diff");
     	for(int i = depth - 1;i>=0;i--) {
     		DiT_TXTBlock block = blocks.get(i);
    		block.back(dy, dtc, dtxt, cos, sin);
    		dy = block.diff;
//    		dy.showDM("in-block-diff");
//    		dy.showDMByOffsetRed(0,10, i+"");
    	}
     	
     	labelEmbd.back(dtxt);
     	
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
    	
    	for(int i = 0;i<depth;i++) {
    		blocks.get(i).saveModel(outputStream);
    	}
    	
    	finalLayer.saveModel(outputStream);
    }

    public void loadModel(RandomAccessFile inputStream) throws IOException {
    	patchEmbd.loadModel(inputStream);

    	timeEmbd.loadModel(inputStream);
    	
    	labelEmbd.loadModel(inputStream);
    	
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
    	
    	for(int i = 0;i<depth;i++) {
    		blocks.get(i).accGrad(scale);
    	}
    	
    	finalLayer.accGrad(scale);
    }
    
    public static void loadWeight(Map<String, Object> weightMap, DiT_TXTMainMoudue block, boolean showLayers) {
        if (showLayers) {
            for (String key : weightMap.keySet()) {
                System.out.println(key);
            }
        }
     
        ModeLoaderlUtils.loadData(block.patchEmbd.patchEmbedding.weight, weightMap, "x_embedder.proj.weight");
        ModeLoaderlUtils.loadData(block.patchEmbd.patchEmbedding.bias, weightMap, "x_embedder.proj.bias");
        
        ModeLoaderlUtils.loadData(block.timeEmbd.linear1.weight, weightMap, "t_embedder.mlp.0.weight");
        ModeLoaderlUtils.loadData(block.timeEmbd.linear1.bias, weightMap, "t_embedder.mlp.0.bias");
        ModeLoaderlUtils.loadData(block.timeEmbd.linear2.weight, weightMap, "t_embedder.mlp.2.weight");
        ModeLoaderlUtils.loadData(block.timeEmbd.linear2.bias, weightMap, "t_embedder.mlp.2.bias");
        
        ModeLoaderlUtils.loadData(block.labelEmbd.linear1.weight, weightMap, "y_embedder.y_proj.fc1.weight");
        ModeLoaderlUtils.loadData(block.labelEmbd.linear1.bias, weightMap, "y_embedder.y_proj.fc1.bias");
        ModeLoaderlUtils.loadData(block.labelEmbd.linear2.weight, weightMap, "y_embedder.y_proj.fc2.weight");
        ModeLoaderlUtils.loadData(block.labelEmbd.linear2.bias, weightMap, "y_embedder.y_proj.fc2.bias");
        
        for(int i = 0;i<block.depth;i++) {
        	
        	DiT_TXTBlock jb = block.blocks.get(i);
        	
//        	jb.norm1.gamma = ModeLoaderlUtils.loadData(jb.norm1.gamma, weightMap, 1, "blocks."+i+".norm1.weight"); 
//        	jb.norm2.gamma = ModeLoaderlUtils.loadData(jb.norm2.gamma, weightMap, 1, "blocks."+i+".norm2.weight"); 
//        	jb.norm3.gamma = ModeLoaderlUtils.loadData(jb.norm3.gamma, weightMap, 1, "blocks."+i+".norm3.weight"); 
        	
        	ModeLoaderlUtils.loadData(jb.attn.qLinerLayer.weight, weightMap, "blocks."+i+".attn.q.weight");
            ModeLoaderlUtils.loadData(jb.attn.qLinerLayer.bias, weightMap, "blocks."+i+".attn.q.bias");
        	ModeLoaderlUtils.loadData(jb.attn.kLinerLayer.weight, weightMap, "blocks."+i+".attn.k.weight");
            ModeLoaderlUtils.loadData(jb.attn.kLinerLayer.bias, weightMap, "blocks."+i+".attn.k.bias");
        	ModeLoaderlUtils.loadData(jb.attn.vLinerLayer.weight, weightMap, "blocks."+i+".attn.v.weight");
            ModeLoaderlUtils.loadData(jb.attn.vLinerLayer.bias, weightMap, "blocks."+i+".attn.v.bias");
        	ModeLoaderlUtils.loadData(jb.attn.oLinerLayer.weight, weightMap, "blocks."+i+".attn.proj.weight");
            ModeLoaderlUtils.loadData(jb.attn.oLinerLayer.bias, weightMap, "blocks."+i+".attn.proj.bias");
        
        	ModeLoaderlUtils.loadData(jb.cross_attn.qLinerLayer.weight, weightMap, "blocks."+i+".cross_attn.q.weight");
            ModeLoaderlUtils.loadData(jb.cross_attn.qLinerLayer.bias, weightMap, "blocks."+i+".cross_attn.q.bias");
        	ModeLoaderlUtils.loadData(jb.cross_attn.kLinerLayer.weight, weightMap, "blocks."+i+".cross_attn.k.weight");
            ModeLoaderlUtils.loadData(jb.cross_attn.kLinerLayer.bias, weightMap, "blocks."+i+".cross_attn.k.bias");
        	ModeLoaderlUtils.loadData(jb.cross_attn.vLinerLayer.weight, weightMap, "blocks."+i+".cross_attn.v.weight");
            ModeLoaderlUtils.loadData(jb.cross_attn.vLinerLayer.bias, weightMap, "blocks."+i+".cross_attn.v.bias");
        	ModeLoaderlUtils.loadData(jb.cross_attn.oLinerLayer.weight, weightMap, "blocks."+i+".cross_attn.proj.weight");
            ModeLoaderlUtils.loadData(jb.cross_attn.oLinerLayer.bias, weightMap, "blocks."+i+".cross_attn.proj.bias");
            
            ModeLoaderlUtils.loadData(jb.mlp.w12.weight, weightMap, "blocks."+i+".mlp.w12.weight");
            ModeLoaderlUtils.loadData(jb.mlp.w12.bias, weightMap, "blocks."+i+".mlp.w12.bias");
            ModeLoaderlUtils.loadData(jb.mlp.w3.weight, weightMap, "blocks."+i+".mlp.w3.weight");
            ModeLoaderlUtils.loadData(jb.mlp.w3.bias, weightMap, "blocks."+i+".mlp.w3.bias");
            
            ModeLoaderlUtils.loadData(jb.adaLN_modulation.weight, weightMap, "blocks."+i+".adaLN_modulation.1.weight");
            ModeLoaderlUtils.loadData(jb.adaLN_modulation.bias, weightMap, "blocks."+i+".adaLN_modulation.1.bias");
        }
        
//        block.finalLayer.finalNorm.gamma = ModeLoaderlUtils.loadData(block.finalLayer.finalNorm.gamma, weightMap, 1, "final_layer.norm_final.weight"); 
        ModeLoaderlUtils.loadData(block.finalLayer.finalLinear.weight, weightMap, "final_layer.linear.weight");
        ModeLoaderlUtils.loadData(block.finalLayer.finalLinear.bias, weightMap, "final_layer.linear.bias");
        ModeLoaderlUtils.loadData(block.finalLayer.m_linear1.weight, weightMap, "final_layer.adaLN_modulation_l1.weight");
        ModeLoaderlUtils.loadData(block.finalLayer.m_linear1.bias, weightMap, "final_layer.adaLN_modulation_l1.bias");
        ModeLoaderlUtils.loadData(block.finalLayer.m_linear2.weight, weightMap, "final_layer.adaLN_modulation_l2.weight");
        ModeLoaderlUtils.loadData(block.finalLayer.m_linear2.bias, weightMap, "final_layer.adaLN_modulation_l2.bias");
        
    }
    
    public static void main(String[] args) {
    	int N = 2;
    	int C = 32;
    	int H = 16;
    	int W = 16;
    	
    	int TT = 1;
    	int TEM = 768;
    	
    	int patchSize = 2;
    	int hiddenSize = 384;
    	int headNum = 6;
    	int depth = 6;
    	
    	CNN nn = new CNN(null);
        nn.CUDNN = true;
        nn.number = N;
    	
        DiT_TXTMainMoudue jb = new DiT_TXTMainMoudue(C, W, H, patchSize, hiddenSize, headNum, depth, 1000, TEM, TT, 4, false, 0.0f, nn);
    	
        String weight = "D:\\models\\dit_s2.json";
        loadWeight(LagJsonReader.readJsonFileBigWeightIterator(weight), jb, true);
        
	    String inputPath = "D:\\models\\c__temp_dit_x.json";
	    Map<String, Object> datas = LagJsonReader.readJsonFileSmallWeight(inputPath);
	    Tensor input = new Tensor(N, C, H, W, true);
	    ModeLoaderlUtils.loadData(input, datas, "x");
    	
	    String cyPath = "D:\\models\\c__temp_dit_cy.json";
	    Map<String, Object> cydatas = LagJsonReader.readJsonFileSmallWeight(cyPath);
	    Tensor cy = new Tensor(N, 1, 1, TEM, true);
	    ModeLoaderlUtils.loadData(cy, cydatas, "cy", 2);
	    
	    Tensor t = new Tensor(N, 1, 1, 1, new float[] {0.1f, 0.8f}, true);
	    int time = (W / patchSize) * (H / patchSize);
	    Tensor[] cs = RoPEKernel.getCosAndSin2D(time, hiddenSize, headNum);
        Tensor cos = cs[0];
        Tensor sin = cs[1];
	    
	    jb.forward(input, t, cy, cos, sin);
	    
	    jb.getOutput().showDM();
	    
	    jb.back(input, cos, sin);
	    
    }
}

