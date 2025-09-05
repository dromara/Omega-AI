package com.omega.engine.nn.layer.dit.mmdit;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import com.omega.common.utils.MatrixOperation;
import com.omega.engine.nn.layer.FullyLayer;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.dit.DiTFinalLayer;
import com.omega.engine.nn.layer.dit.DiTPatchEmbeddingLayer;
import com.omega.engine.nn.layer.dit.DiTTimeEmbeddingLayer;
import com.omega.engine.nn.network.CNN;
import com.omega.engine.nn.network.Network;
import com.omega.engine.tensor.Tensor;
import com.omega.engine.updater.UpdaterFactory;
import com.omega.example.clip.utils.ClipModelUtils;
import com.omega.example.transformer.utils.LagJsonReader;

/**
 * DiT_Block
 * @author Administrator
 */
public class MMDiTMoudue extends Layer {
	
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
    private boolean normParams = true;
    
    public DiTPatchEmbeddingLayer patchEmbd;
    public DiTTimeEmbeddingLayer timeEmbd;
    public FullyLayer labelEmbd;
    public List<DiTJoinBlock> blocks;
    public DiTFinalLayer finalLayer;
    
    private Tensor posEmbd;
    
    private Tensor dtc;

    public MMDiTMoudue(int inChannel, int width, int height, int patchSize, int hiddenSize, int headNum, int depth, int timeSteps, int maxContextLen, int textEmbedDim, int mlpRatio, boolean learnSigma, boolean normParams, Network network) {
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
		this.normParams = normParams;
		this.initLayers();
		this.oHeight = height;
		this.oWidth = width;
    }
    
    public void initLayers() {
    	
    	patchEmbd = new DiTPatchEmbeddingLayer(inChannel, width, hiddenSize, patchSize, true, network);
         
        timeEmbd = new DiTTimeEmbeddingLayer(timeSteps, 256, hiddenSize, true, network);
        
        labelEmbd = new FullyLayer(textEmbedDim, hiddenSize, true, network);
        
        blocks = new ArrayList<DiTJoinBlock>();
         
        for(int i = 0;i<depth;i++) {
        	boolean pre_only = false;
        	if(i == depth - 1) {
        		pre_only = true;
        	}
        	DiTJoinBlock block = new DiTJoinBlock(hiddenSize, hiddenSize, mlpRatio, headNum, patchEmbd.oChannel, maxContextLen, false, false, pre_only, normParams, network);
	        blocks.add(block);
        }
        int os = inChannel;
        if(learnSigma) {
        	os = inChannel * 2;
        }
        this.oChannel = os;
        finalLayer = new DiTFinalLayer(patchSize, hiddenSize, os, patchEmbd.oChannel, true, normParams, network);

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
    	}else {
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
    	Tensor_OP().addAxis(patchEmbd.getOutput(), posEmbd, patchEmbd.getOutput(), posEmbd.channel * posEmbd.width);

    	timeEmbd.forward(tc);
    	
    	labelEmbd.forward(text);
    	
    	Tensor x = patchEmbd.getOutput().view(patchEmbd.getOutput().number * patchEmbd.getOutput().channel, 1, 1, patchEmbd.getOutput().width);
    	
    	Tensor context = labelEmbd.getOutput();
    	
//    	x.showDM("x");

//    	timeEmbd.getOutput().showDM("t_out");

//    	context.showDM("context");
    	
    	for(int i = 0;i<depth;i++) {
    		DiTJoinBlock block = blocks.get(i);
    		block.forward(x, context, timeEmbd.getOutput());
    		x = block.getOutput();
    		context = block.context_block.getOutput();
//    		x.showDM("x:"+i);
//    		if(context != null) {
//    			context.showDM("context:"+i);
//    		}
    		
//    		System.out.println(context);
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
//    	delta.showDM("delta");
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
    	Tensor dcx = null;

     	for(int i = depth - 1;i>=0;i--) {
     		DiTJoinBlock block = blocks.get(i);
    		block.back(dy, dcx, dtc);
    		dy = block.diff;
    		dcx = block.context_block.diff;
//    		dy.showDM("dx_"+i);
//    		dy.showDMByOffsetRed(10 * 384, 384, "dx_"+i);
    	}

     	labelEmbd.back(dcx);

     	timeEmbd.back(dtc);
//     	dy.showDM("dx");
     	patchEmbd.back(dy);
//     	dy.showDM("dx");
     	this.diff = patchEmbd.patchEmbedding.diff;
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
    
    public static void loadWeight(Map<String, Object> weightMap, MMDiTMoudue block, boolean showLayers) {
        if (showLayers) {
            for (String key : weightMap.keySet()) {
                System.out.println(key);
            }
        }
        
        ClipModelUtils.loadData(block.patchEmbd.patchEmbedding.weight, weightMap, "x_embedder.proj.weight");
        ClipModelUtils.loadData(block.patchEmbd.patchEmbedding.bias, weightMap, "x_embedder.proj.bias");
        
        ClipModelUtils.loadData(block.timeEmbd.linear1.weight, weightMap, "t_embedder.mlp.0.weight");
        ClipModelUtils.loadData(block.timeEmbd.linear1.bias, weightMap, "t_embedder.mlp.0.bias");
        ClipModelUtils.loadData(block.timeEmbd.linear2.weight, weightMap, "t_embedder.mlp.2.weight");
        ClipModelUtils.loadData(block.timeEmbd.linear2.bias, weightMap, "t_embedder.mlp.2.bias");
        
        ClipModelUtils.loadData(block.labelEmbd.weight, weightMap, "context_embedder.weight");
        ClipModelUtils.loadData(block.labelEmbd.bias, weightMap, "context_embedder.bias");
        
        for(int i = 0;i<block.depth;i++) {
        	DiTJoinBlock jb = block.blocks.get(i);
        	
        	jb.context_block.norm1.gamma = ClipModelUtils.loadData(jb.context_block.norm1.gamma, weightMap, 1, "joint_blocks."+i+".context_block.norm1.weight"); 
        	jb.context_block.norm1.beta = ClipModelUtils.loadData(jb.context_block.norm1.beta, weightMap, 1, "joint_blocks."+i+".context_block.norm1.bias"); 
        	
        	ClipModelUtils.loadData(jb.context_block.qLinerLayer.weight, weightMap, "joint_blocks."+i+".context_block.attn.ql.weight");
            ClipModelUtils.loadData(jb.context_block.kLinerLayer.weight, weightMap, "joint_blocks."+i+".context_block.attn.kl.weight");
            ClipModelUtils.loadData(jb.context_block.vLinerLayer.weight, weightMap, "joint_blocks."+i+".context_block.attn.vl.weight");
             
            if(jb.context_block.oLinerLayer != null) {
            	ClipModelUtils.loadData(jb.context_block.oLinerLayer.weight, weightMap, "joint_blocks."+i+".context_block.attn.proj.weight");
                ClipModelUtils.loadData(jb.context_block.oLinerLayer.bias, weightMap, "joint_blocks."+i+".context_block.attn.proj.bias");
            }
            
            if(jb.context_block.mlp != null) {
            	jb.context_block.norm2.gamma = ClipModelUtils.loadData(jb.context_block.norm2.gamma, weightMap, 1, "joint_blocks."+i+".context_block.norm2.weight"); 
            	jb.context_block.norm2.beta = ClipModelUtils.loadData(jb.context_block.norm2.beta, weightMap, 1, "joint_blocks."+i+".context_block.norm2.bias");
           	 	ClipModelUtils.loadData(jb.context_block.mlp.linear1.weight, weightMap, "joint_blocks."+i+".context_block.mlp.fc1.weight");
                ClipModelUtils.loadData(jb.context_block.mlp.linear1.bias, weightMap, "joint_blocks."+i+".context_block.mlp.fc1.bias");
                ClipModelUtils.loadData(jb.context_block.mlp.linear2.weight, weightMap, "joint_blocks."+i+".context_block.mlp.fc2.weight");
                ClipModelUtils.loadData(jb.context_block.mlp.linear2.bias, weightMap, "joint_blocks."+i+".context_block.mlp.fc2.bias");
            }
            ClipModelUtils.loadData(jb.context_block.adaLN_modulation.weight, weightMap, "joint_blocks."+i+".context_block.adaLN_modulation.1.weight");
            ClipModelUtils.loadData(jb.context_block.adaLN_modulation.bias, weightMap, "joint_blocks."+i+".context_block.adaLN_modulation.1.bias");
            
            jb.x_block.norm1.gamma = ClipModelUtils.loadData(jb.x_block.norm1.gamma, weightMap, 1, "joint_blocks."+i+".x_block.norm1.weight"); 
            jb.x_block.norm1.beta = ClipModelUtils.loadData(jb.x_block.norm1.beta, weightMap, 1, "joint_blocks."+i+".x_block.norm1.bias"); 
            ClipModelUtils.loadData(jb.x_block.qLinerLayer.weight, weightMap, "joint_blocks."+i+".x_block.attn.ql.weight");
            ClipModelUtils.loadData(jb.x_block.kLinerLayer.weight, weightMap, "joint_blocks."+i+".x_block.attn.kl.weight");
            ClipModelUtils.loadData(jb.x_block.vLinerLayer.weight, weightMap, "joint_blocks."+i+".x_block.attn.vl.weight");
            ClipModelUtils.loadData(jb.x_block.oLinerLayer.weight, weightMap, "joint_blocks."+i+".x_block.attn.proj.weight");
            ClipModelUtils.loadData(jb.x_block.oLinerLayer.bias, weightMap, "joint_blocks."+i+".x_block.attn.proj.bias");
            jb.x_block.norm2.gamma = ClipModelUtils.loadData(jb.x_block.norm2.gamma, weightMap, 1, "joint_blocks."+i+".x_block.norm2.weight"); 
            jb.x_block.norm2.beta = ClipModelUtils.loadData(jb.x_block.norm2.beta, weightMap, 1, "joint_blocks."+i+".x_block.norm2.bias"); 
            ClipModelUtils.loadData(jb.x_block.mlp.linear1.weight, weightMap, "joint_blocks."+i+".x_block.mlp.fc1.weight");
            ClipModelUtils.loadData(jb.x_block.mlp.linear1.bias, weightMap, "joint_blocks."+i+".x_block.mlp.fc1.bias");
            ClipModelUtils.loadData(jb.x_block.mlp.linear2.weight, weightMap, "joint_blocks."+i+".x_block.mlp.fc2.weight");
            ClipModelUtils.loadData(jb.x_block.mlp.linear2.bias, weightMap, "joint_blocks."+i+".x_block.mlp.fc2.bias");
            ClipModelUtils.loadData(jb.x_block.adaLN_modulation.weight, weightMap, "joint_blocks."+i+".x_block.adaLN_modulation.1.weight");
            ClipModelUtils.loadData(jb.x_block.adaLN_modulation.bias, weightMap, "joint_blocks."+i+".x_block.adaLN_modulation.1.bias");
        }
        
        block.finalLayer.finalNorm.gamma = ClipModelUtils.loadData(block.finalLayer.finalNorm.gamma, weightMap, 1, "final_layer.norm_final.weight"); 
        block.finalLayer.finalNorm.beta = ClipModelUtils.loadData(block.finalLayer.finalNorm.beta, weightMap, 1, "final_layer.norm_final.bias"); 
        ClipModelUtils.loadData(block.finalLayer.finalLinear.weight, weightMap, "final_layer.linear.weight");
        ClipModelUtils.loadData(block.finalLayer.finalLinear.bias, weightMap, "final_layer.linear.bias");
        ClipModelUtils.loadData(block.finalLayer.m_linear1.weight, weightMap, "final_layer.adaLN_modulation_l1.weight");
        ClipModelUtils.loadData(block.finalLayer.m_linear1.bias, weightMap, "final_layer.adaLN_modulation_l1.bias");
        ClipModelUtils.loadData(block.finalLayer.m_linear2.weight, weightMap, "final_layer.adaLN_modulation_l2.weight");
        ClipModelUtils.loadData(block.finalLayer.m_linear2.bias, weightMap, "final_layer.adaLN_modulation_l2.bias");
    }
    
    public static void main(String[] args) {
//    	int embed_dim = 384;
//    	int grid_size = 16;
//    	get_2d_cossin_pos_embed(embed_dim, grid_size);
    	
    	int N = 2;
    	int C = 4;
    	int H = 32;
    	int W = 32;
    	
    	int TT = 1;
    	int TEM = 768;
    	
    	int patchSize = 2;
    	int hiddenSize = 384;
    	int headNum = 6;
    	int depth = 6;
    	
    	CNN nn = new CNN(null);
        nn.CUDNN = true;
        nn.number = N;
    	
        MMDiTMoudue jb = new MMDiTMoudue(C, W, H, patchSize, hiddenSize, headNum, depth, 1000, TT, TEM, 4, false, true, nn);
    	
        String weight = "D:\\models\\mmdit_small.json";
        loadWeight(LagJsonReader.readJsonFileBigWeightIterator(weight), jb, true);
        
	    String inputPath = "D:\\models\\img_x.json";
	    Map<String, Object> datas = LagJsonReader.readJsonFileSmallWeight(inputPath);
	    Tensor input = new Tensor(N, C, H, W, true);
	    ClipModelUtils.loadData(input, datas, "img_x");
      
	    String txtPath = "D:\\models\\txt.json";
	    Map<String, Object> txt_datas = LagJsonReader.readJsonFileSmallWeight(txtPath);
	    Tensor txt = new Tensor(N, TT, 1, TEM, true);
	    ClipModelUtils.loadData(txt, txt_datas, "txt", 3);
	    txt.view(N * TT, 1, 1, TEM);
	    
	    Tensor t = new Tensor(N, 1, 1, 1, new float[] {1, 20}, true);
	    
	    jb.forward(input, t, txt);
	    
	    jb.getOutput().showDM("output");
	    
	    String deltaPath = "D:\\models\\delta.json";
	    Map<String, Object> deltaDatas = LagJsonReader.readJsonFileSmallWeight(deltaPath);
	    Tensor delta = new Tensor(N, C, H, W, true);
	    ClipModelUtils.loadData(delta, deltaDatas, "delta");
	    
	    jb.back(delta);
	    
	    jb.diff.showDM("diff");
	    
	    jb.forward(input, t, txt);
	    
	    jb.getOutput().showDM("output");
	    
	    jb.back(delta);
	    
	    jb.diff.showDM("diff");
	    
    }
}

