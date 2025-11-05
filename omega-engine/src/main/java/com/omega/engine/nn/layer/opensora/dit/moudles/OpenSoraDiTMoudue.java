package com.omega.engine.nn.layer.opensora.dit.moudles;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import com.omega.common.utils.MatrixOperation;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.dit.DiTCaptionEmbeddingLayer;
import com.omega.engine.nn.layer.dit.DiTTimeEmbeddingLayer;
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
public class OpenSoraDiTMoudue extends Layer {
	
	public int inChannel;
	public int time;
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
    
    public OpenSoraDiTPatchEmbed3D patchEmbd;
    public DiTTimeEmbeddingLayer timeEmbd;
    public DiTCaptionEmbeddingLayer labelEmbd;
    public List<OpenSoraDiTBlock> blocks;
    public OpenSoraDiTFinalLayer finalLayer;
    
    private Tensor patchEmbd_tmp;
    
    private Tensor posEmbd_S;
    
    private Tensor posEmbd_T;
    
    private Tensor dtc;
    
    private float y_drop_prob = 0.0f;
    
    private int[] x_shape;
    private int[] xt_shape;
    
    public OpenSoraDiTMoudue(int inChannel, int time, int width, int height, int patchSize, int hiddenSize, int headNum, int depth, int timeSteps, int maxContextLen, int textEmbedDim, int mlpRatio, boolean learnSigma, float y_drop_prob, Network network) {
		this.network = network;
        if (this.updater == null) {
            this.setUpdater(UpdaterFactory.create(network));
        }
        this.y_drop_prob = y_drop_prob;
    	this.inChannel = inChannel;
    	this.time = time;
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
    	
    	patchEmbd = new OpenSoraDiTPatchEmbed3D(inChannel, hiddenSize, time, width, new int[] {1, 2, 2}, true, network);
         
        timeEmbd = new DiTTimeEmbeddingLayer(timeSteps, 256, hiddenSize, true, network);
        
        labelEmbd = new DiTCaptionEmbeddingLayer(textEmbedDim, hiddenSize, maxContextLen, y_drop_prob, true, network);
        
        blocks = new ArrayList<OpenSoraDiTBlock>();
         
        for(int i = 0;i<depth;i++) {
        	OpenSoraDiTBlock block = new OpenSoraDiTBlock(hiddenSize, hiddenSize, patchEmbd.oChannel, mlpRatio * hiddenSize, headNum, true, false, network);
	        blocks.add(block);
        }
        int os = inChannel;
        if(learnSigma) {
        	os = inChannel * 2;
        }
        this.oChannel = os;
        finalLayer = new OpenSoraDiTFinalLayer(patchSize, hiddenSize, os, patchEmbd.oChannel, true, network);

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
    
    public static float[] get_1d_cossin_pos_embed(int embed_dim,int length) {
    	float[] pos = new float[length];
    	for(int i = 0;i<length;i++) {
    		pos[i] = i;
    	}
    	float[] emb_t = get_1d_sincos_pos_embed_from_grid(embed_dim, pos);
    	return emb_t;
    }
    
    public void init(Tensor input) {
        // TODO Auto-generated method stub
        this.number = input.number;
        if(this.output == null || this.output.number != number) {
        	int d = patchEmbd.oChannel / patchEmbd.oDepth;
        	x_shape = new int[] {number, patchEmbd.oDepth, d, patchEmbd.oHeight * patchEmbd.oWidth};
        	xt_shape = new int[] {number, d, patchEmbd.oDepth, patchEmbd.oHeight * patchEmbd.oWidth};
        	output = Tensor.createGPUTensor(output, number, oChannel * time, oHeight, oWidth, true);
        }
        if(posEmbd_S == null) {
        	posEmbd_S = new Tensor(1, patchEmbd.oChannel / patchEmbd.oDepth, 1, hiddenSize, get_2d_cossin_pos_embed(hiddenSize, width/patchSize), true);
        }
        if(posEmbd_T == null) {
        	posEmbd_T = new Tensor(1, patchEmbd.oDepth, 1, hiddenSize, get_1d_cossin_pos_embed(hiddenSize, patchEmbd.oDepth), true);
        }
        if(patchEmbd.getOutput() != null){
        	patchEmbd.getOutput().viewOrg();
        }
        if(patchEmbd_tmp == null) {
        	int d = patchEmbd.oChannel / patchEmbd.oDepth;
        	patchEmbd_tmp = Tensor.createGPUTensor(patchEmbd_tmp, number, d * patchEmbd.oDepth, 1, patchEmbd.oHeight * patchEmbd.oWidth, true);
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
    	Tensor_OP().addAxis(patchEmbd.getOutput(), posEmbd_S, patchEmbd.getOutput(), posEmbd_S.channel * posEmbd_S.width);
    	Tensor_OP().permute(patchEmbd.getOutput(), patchEmbd_tmp, x_shape, xt_shape, new int[] {0, 2, 1, 3});
//    	patchEmbd_tmp.showDM("patchEmbd_tmp1");
    	Tensor_OP().addAxis(patchEmbd_tmp, posEmbd_T, patchEmbd_tmp, posEmbd_T.channel * posEmbd_T.width);
//    	posEmbd_T.showDM("posEmbd_T");
//    	patchEmbd_tmp.showDM("patchEmbd_tmp2");
    	Tensor_OP().permute(patchEmbd_tmp, patchEmbd.getOutput(), xt_shape, x_shape, new int[] {0, 2, 1, 3});

//    	patchEmbd.getOutput().showDM("patchEmbd+t");
    	
    	timeEmbd.forward(tc);
    	
    	labelEmbd.forward(text);
    	
    	Tensor x = patchEmbd.getOutput().view(patchEmbd.getOutput().number * patchEmbd.getOutput().channel, 1, 1, patchEmbd.getOutput().width);
    	
    	Tensor_OP().add(timeEmbd.getOutput(), labelEmbd.getOutput(), timeEmbd.getOutput());
    	
    	for(int i = 0;i<depth;i++) {
    		OpenSoraDiTBlock block = blocks.get(i);
    		block.forward(x, timeEmbd.getOutput());
    		x = block.getOutput();
    	}

    	finalLayer.forward(x, timeEmbd.getOutput());
    	
    	/**
    	 * unpatchify
    	 * x: (N, t, h, w, r, p, q, c)
         * imgs: (N, C, T, H, W)
    	 */
    	int t = time/1;
    	int h = height/patchSize;
    	int w = width/patchSize;
    	int[] xShape = new int[] {number, t, h, w, 1, patchSize, patchSize, oChannel};
    	int[] yShape = new int[] {number, oChannel, t, 1, h, patchSize, w, patchSize};
    	Tensor_OP().permute(finalLayer.getOutput(), this.output, xShape, yShape, new int[] {0, 7, 1, 4, 2, 5, 3, 6});
    	
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
    	int t = time/1;
    	int h = height/patchSize;
    	int w = width/patchSize;
    	int[] yShape = new int[] {number, oChannel, t, 1, h, patchSize, w, patchSize};
    	int[] xShape = new int[] {number, t, h, w, 1, patchSize, patchSize, oChannel};

    	Tensor_OP().permute(delta, finalLayer.getOutput(), yShape, xShape, new int[] {0, 2, 4, 6, 3, 5, 7, 1});
    	
    	finalLayer.back(finalLayer.getOutput(), dtc);

    	Tensor dy = finalLayer.diff;
    	
     	for(int i = depth - 1;i>=0;i--) {
     		OpenSoraDiTBlock block = blocks.get(i);
    		block.back(dy, dtc);
    		dy = block.diff;
    	}

     	labelEmbd.back(dtc);
     	
     	timeEmbd.back(dtc);

     	patchEmbd.back(dy);
     	
     	this.diff = patchEmbd.diff;
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
    
    public static void loadWeight(Map<String, Object> weightMap, OpenSoraDiTMoudue block, boolean showLayers) {
        if (showLayers) {
            for (String key : weightMap.keySet()) {
                System.out.println(key);
            }
        }
        
        ModeLoaderlUtils.loadData(block.patchEmbd.patchEmbedding.weight, weightMap, "x_embedder.proj.weight", 5);
        ModeLoaderlUtils.loadData(block.patchEmbd.patchEmbedding.bias, weightMap, "x_embedder.proj.bias");
        
        ModeLoaderlUtils.loadData(block.labelEmbd.linear1.weight, weightMap, "y_embedder.y_proj.fc1.weight");
        ModeLoaderlUtils.loadData(block.labelEmbd.linear1.bias, weightMap, "y_embedder.y_proj.fc1.bias");
        ModeLoaderlUtils.loadData(block.labelEmbd.linear2.weight, weightMap, "y_embedder.y_proj.fc2.weight");
        ModeLoaderlUtils.loadData(block.labelEmbd.linear2.bias, weightMap, "y_embedder.y_proj.fc2.bias");
        
        ModeLoaderlUtils.loadData(block.timeEmbd.linear1.weight, weightMap, "t_embedder.mlp.0.weight");
        ModeLoaderlUtils.loadData(block.timeEmbd.linear1.bias, weightMap, "t_embedder.mlp.0.bias");
        ModeLoaderlUtils.loadData(block.timeEmbd.linear2.weight, weightMap, "t_embedder.mlp.2.weight");
        ModeLoaderlUtils.loadData(block.timeEmbd.linear2.bias, weightMap, "t_embedder.mlp.2.bias");
        
        for(int i = 0;i<block.depth;i++) {
        	OpenSoraDiTBlock jb = block.blocks.get(i);
        	
        	ModeLoaderlUtils.loadData(jb.attn.qLinerLayer.weight, weightMap, "blocks."+i+".attn.q.weight");
        	ModeLoaderlUtils.loadData(jb.attn.qLinerLayer.bias, weightMap, "blocks."+i+".attn.q.bias");
            ModeLoaderlUtils.loadData(jb.attn.kLinerLayer.weight, weightMap, "blocks."+i+".attn.k.weight");
            ModeLoaderlUtils.loadData(jb.attn.kLinerLayer.bias, weightMap, "blocks."+i+".attn.k.bias");
            ModeLoaderlUtils.loadData(jb.attn.vLinerLayer.weight, weightMap, "blocks."+i+".attn.v.weight");
            ModeLoaderlUtils.loadData(jb.attn.vLinerLayer.bias, weightMap, "blocks."+i+".attn.v.bias");
            ModeLoaderlUtils.loadData(jb.attn.oLinerLayer.weight, weightMap, "blocks."+i+".attn.proj.weight");
            ModeLoaderlUtils.loadData(jb.attn.oLinerLayer.bias, weightMap, "blocks."+i+".attn.proj.bias");
            
            ModeLoaderlUtils.loadData(jb.mlp.linear1.weight, weightMap, "blocks."+i+".mlp.fc1.weight");
            ModeLoaderlUtils.loadData(jb.mlp.linear1.bias, weightMap, "blocks."+i+".mlp.fc1.bias");
            ModeLoaderlUtils.loadData(jb.mlp.linear2.weight, weightMap, "blocks."+i+".mlp.fc2.weight");
            ModeLoaderlUtils.loadData(jb.mlp.linear2.bias, weightMap, "blocks."+i+".mlp.fc2.bias");
            
            ModeLoaderlUtils.loadData(jb.adaLN_modulation.weight, weightMap, "blocks."+i+".adaLN_modulation.1.weight");
            ModeLoaderlUtils.loadData(jb.adaLN_modulation.bias, weightMap, "blocks."+i+".adaLN_modulation.1.bias");
        }
        
        ModeLoaderlUtils.loadData(block.finalLayer.finalLinear.weight, weightMap, "final_layer.linear.weight");
        ModeLoaderlUtils.loadData(block.finalLayer.finalLinear.bias, weightMap, "final_layer.linear.bias");
        ModeLoaderlUtils.loadData(block.finalLayer.m_linear1.weight, weightMap, "final_layer.adaLN_modulation_l1.weight");
        ModeLoaderlUtils.loadData(block.finalLayer.m_linear1.bias, weightMap, "final_layer.adaLN_modulation_l1.bias");
        ModeLoaderlUtils.loadData(block.finalLayer.m_linear2.weight, weightMap, "final_layer.adaLN_modulation_l2.weight");
        ModeLoaderlUtils.loadData(block.finalLayer.m_linear2.bias, weightMap, "final_layer.adaLN_modulation_l2.bias");
        
    }
    
    public static void main(String[] args) {
//    	int embed_dim = 768;
//    	int grid_size = 16;
//    	get_2d_cossin_pos_embed(embed_dim, grid_size);
    	
    	int N = 2;
    	int C = 4;
    	int T = 9;
    	int H = 32;
    	int W = 32;
    	
    	int TT = 1;
    	int TEM = 384;
    	
    	int patchSize = 2;
    	int hiddenSize = 384;
    	int headNum = 6;
    	int depth = 6;
    	
    	CNN nn = new CNN(null);
        nn.CUDNN = true;
        nn.number = N;
    	
        OpenSoraDiTMoudue dit = new OpenSoraDiTMoudue(C, T, W, H, patchSize, hiddenSize, headNum, depth, 1000, 1, TEM, 4, true, 0.0f, nn);
    	
        String weight = "D:\\models\\opensora_model.json";
        loadWeight(LagJsonReader.readJsonFileBigWeightIterator(weight), dit, true);
        
        String inputPath = "D:\\models\\opensora_input.json";
	    Map<String, Object> datas = LagJsonReader.readJsonFileSmallWeight(inputPath);
	    Tensor input = new Tensor(N, C * T, H, W, true);
	    ModeLoaderlUtils.loadData(input, datas, "input", 5);
      
	    String txtPath = "D:\\models\\opensora_txt.json";
	    Map<String, Object> txt_datas = LagJsonReader.readJsonFileSmallWeight(txtPath);
	    Tensor txt = new Tensor(N, TT, 1, TEM, true);
	    ModeLoaderlUtils.loadData(txt, txt_datas, "txt", 3);
	    txt.view(N * TT, 1, 1, TEM);
	    
	    Tensor t = new Tensor(N, 1, 1, 1, new float[] {1, 20}, true);
        
	    String dxPath = "D:\\models\\opensora_dx.json";
	    Map<String, Object> datas3 = LagJsonReader.readJsonFileSmallWeight(dxPath);
	    Tensor dx = new Tensor(N, 2 * C * T, H, W, true);
	    ModeLoaderlUtils.loadData(dx, datas3, "dx", 5);
	    
	    dit.forward(input, t, txt);
	    
	    dit.getOutput().showDM("output");
	    
	    dit.back(dx);
	    
	    dit.diff.showDM("diff");
	    
    }
}

