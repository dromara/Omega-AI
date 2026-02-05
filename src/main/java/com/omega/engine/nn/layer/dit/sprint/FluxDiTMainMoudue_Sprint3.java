package com.omega.engine.nn.layer.dit.sprint;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

import com.omega.common.utils.MatrixOperation;
import com.omega.engine.gpu.BaseKernel;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.dit.DiTCaptionEmbeddingLayer;
import com.omega.engine.nn.layer.dit.DiTOrgTimeEmbeddingLayer;
import com.omega.engine.nn.layer.dit.DiTPatchEmbeddingLayer;
import com.omega.engine.nn.layer.dit.flux.REPAMLPLayer;
import com.omega.engine.nn.layer.dit.kernel.TokenDropKernel;
import com.omega.engine.nn.layer.dit.txt.DiT_TXTFinalLayer;
import com.omega.engine.nn.layer.gpu.RoPEKernel;
import com.omega.engine.nn.network.CNN;
import com.omega.engine.nn.network.Network;
import com.omega.engine.nn.network.RunModel;
import com.omega.engine.tensor.Tensor;
import com.omega.engine.updater.UpdaterFactory;
import com.omega.example.common.ModeLoaderlUtils;
import com.omega.example.transformer.utils.LagJsonReader;

/**
 * DiT_Block
 * @author Administrator
 */
public class FluxDiTMainMoudue_Sprint3 extends Layer {
	
	public int inChannel;
    public int width;
    public int height;
    public int patchSize;
    private int hiddenSize;
    private int depth;
    private int num_f = 2;
    private int num_h = 2;
    private int num_g = 0;
    private int timeSteps;
    private int headNum;
    private int textEmbedDim;
    private int maxContextLen;
    private int mlpRatio = 4;

    private int z_dim = 768;
    private int projector_dim = 2048;
    
    public DiTPatchEmbeddingLayer patchEmbd;
    public DiTOrgTimeEmbeddingLayer timeEmbd;
    public DiTCaptionEmbeddingLayer labelEmbd;
    public List<FluxDiTJoinBlockRoPE> encoders;
    public List<FluxDiTJoinBlockRoPE> mids;
    public FusionLayer3 fusion;
    public List<FluxDiTJoinBlockRoPE> decoders;
    public DiT_TXTFinalLayer finalLayer;
    
    public REPAMLPLayer z_mlp;
    
    private int hw;
    
    private Tensor posEmbd;

    private Tensor dtc;
    private Tensor dencoder;
    private Tensor dencoder_c;
    private Tensor drop_delta;
    
    private float y_drop_prob = 0.0f;
    
    private float token_drop_ratio = 0.75f;
    private int token_t = 0;
    
    private float path_drop_prob = 0.0f;
    
    private Tensor idsKeep;
    private Tensor td_x;
    
    private int[] xShape;
    private int[] yShape;
    
    private BaseKernel baseKernel;
    private TokenDropKernel tokenDropKernel;
    
    private List<Integer> ids;
    
    public boolean uncond = false;
    
    public FluxDiTMainMoudue_Sprint3(int inChannel, int width, int height, int patchSize, int hiddenSize, int headNum, int depth, int timeSteps, int textEmbedDim, int maxContextLen, int mlpRatio, int z_dim, float y_drop_prob, float token_drop_ratio, float path_drop_prob, Network network) {
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
		this.num_g = this.depth - num_f - num_h;
		this.timeSteps = timeSteps;
		this.textEmbedDim = textEmbedDim;
		this.maxContextLen = maxContextLen;
		this.mlpRatio = mlpRatio;
		this.token_drop_ratio = token_drop_ratio;
		this.path_drop_prob = path_drop_prob;
		this.headNum = headNum;
		this.z_dim = z_dim;
		this.initLayers();
		this.oHeight = height;
		this.oWidth = width;
    }

    public void initLayers() {
    	
    	patchEmbd = new DiTPatchEmbeddingLayer(inChannel, width, hiddenSize, patchSize, true, network);
        
    	hw = patchEmbd.oChannel;

		this.token_t = (int) (hw * (1.0f - token_drop_ratio));

        timeEmbd = new DiTOrgTimeEmbeddingLayer(timeSteps, 256, hiddenSize, true, network);
        
        labelEmbd = new DiTCaptionEmbeddingLayer(textEmbedDim, hiddenSize, maxContextLen, y_drop_prob, true, network);
        
        encoders = new ArrayList<FluxDiTJoinBlockRoPE>();
        mids = new ArrayList<FluxDiTJoinBlockRoPE>();
        decoders = new ArrayList<FluxDiTJoinBlockRoPE>();
        
        for(int i = 0;i<num_f;i++) {
        	FluxDiTJoinBlockRoPE block = new FluxDiTJoinBlockRoPE(hiddenSize, hiddenSize, mlpRatio, headNum, patchEmbd.oChannel, maxContextLen, false, false, false, true, network);
        	encoders.add(block);
        }
        
        for(int i = 0;i<num_g;i++) {
        	FluxDiTJoinBlockRoPE block = new FluxDiTJoinBlockRoPE(hiddenSize, hiddenSize, mlpRatio, headNum, token_t, maxContextLen, false, false, false, true, network);
        	mids.add(block);
        }
        
        fusion = new FusionLayer3(hiddenSize, hw, token_t, maxContextLen, path_drop_prob, network);
        
        for(int i = 0;i<num_h;i++) {
        	boolean pre_only = false;
        	if(i == num_h - 1) {
        		pre_only = true;
        	}
        	FluxDiTJoinBlockRoPE block = new FluxDiTJoinBlockRoPE(hiddenSize, hiddenSize, mlpRatio, headNum, patchEmbd.oChannel, maxContextLen, false, false, pre_only, true, network);
        	decoders.add(block);
        }
        
        this.oChannel = inChannel;

        finalLayer = new DiT_TXTFinalLayer(patchSize, hiddenSize, inChannel, patchEmbd.oChannel, true, false, network);
        
        z_mlp = new REPAMLPLayer(hiddenSize, projector_dim, z_dim, true, network);
        
        if(baseKernel == null) {
        	baseKernel = new BaseKernel(cuda());
        }
        
        if(tokenDropKernel == null) {
        	tokenDropKernel = new TokenDropKernel(cuda());
        }
        
    }
    
    public void getRandomIds(Tensor idskeep) {
    	if(ids == null) {
    		ids = new ArrayList<Integer>(hw);
    		for(int i = 0;i<hw;i++) {
    			ids.add(i);
    		}
    		idskeep.syncHost();
    	}
    	for(int b = 0;b<idskeep.number;b++) {
    		Collections.shuffle(ids);
    		for(int w = 0;w<idskeep.width;w++) {
        		idskeep.data[b * idskeep.width + w] = ids.get(w);
    		}
    	}
    	idskeep.hostToDevice();
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
        
        if(token_t < hw && (idsKeep == null || idsKeep.number != number)) {
        	idsKeep = Tensor.createGPUTensor(idsKeep, number, 1, 1, token_t, true);
        	td_x = Tensor.createGPUTensor(td_x, number * token_t, 1, 1, hiddenSize, true);
        }
        
//        if(token_t < hw && td_x == null) {
//        	td_x = Tensor.createGPUTensor(td_x, number * (maxContextLen + token_t), 1, 1, hiddenSize, true);
//        }
        
        if(patchEmbd.getOutput() != null){
        	patchEmbd.getOutput().viewOrg();
        }
        
    }
    
    @Override
    public void initBack() {
        // TODO Auto-generated method stub
    	if(dtc == null || dtc.number != timeEmbd.getOutput().number) {
    		dtc = Tensor.createGPUTensor(dtc, timeEmbd.getOutput().shape(), true);
    		dencoder = Tensor.createGPUTensor(dencoder, number * hw, 1, 1, hiddenSize, true);
    		dencoder_c = Tensor.createGPUTensor(dencoder_c, number * maxContextLen, 1, 1, hiddenSize, true);
    		drop_delta = Tensor.createGPUTensor(drop_delta, number * hw, 1, 1, hiddenSize, true);
    	}else {
    		dtc.clearGPU();
//    		dencoder.clearGPU();
    		drop_delta.clearGPU();
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
    
    public void output(Tensor tc, Tensor label, Tensor cos, Tensor sin) {
    	
    	patchEmbd.forward(input);

    	Tensor_OP().addAxis(patchEmbd.getOutput(), posEmbd, patchEmbd.getOutput(), posEmbd.channel * posEmbd.width);
    	
    	timeEmbd.forward(tc);
    	
    	labelEmbd.forward(label);
    	
    	Tensor x = patchEmbd.getOutput().view(patchEmbd.getOutput().number * patchEmbd.getOutput().channel, 1, 1, patchEmbd.getOutput().width);
    	
    	Tensor t = timeEmbd.getOutput();
    	
    	Tensor cond = labelEmbd.getOutput();
    	
    	/**
    	 * encoder
    	 */
    	Tensor e_x = x;
    	Tensor e_c = cond;
    	for(int i = 0;i<num_f;i++) {
    		FluxDiTJoinBlockRoPE block = encoders.get(i);
    		block.forward(e_x, e_c, t, cos, sin);
    		e_x = block.getOutput();
    		e_c = block.context_block.getOutput();
    	}

    	/**
    	 * repa
    	 */
		if(network.RUN_MODEL == RunModel.TRAIN) {
			z_mlp.forward(e_x);
		}
		
		/**
		 * sprint
		 */
		Tensor h_x = e_x;
		Tensor h_c = e_c;
		if(!uncond) {
			if(idsKeep != null && network.RUN_MODEL == RunModel.TRAIN) {
				tokenDropKernel.idsKeep(idsKeep, number, (hw - 1), token_t);
				tokenDropKernel.imgTokenDrop(e_x, idsKeep, td_x, token_t, hw, 0, hiddenSize);
				h_x = td_x;
			}
			
			/**
			 * mids
			 */
			for(int i = 0;i<num_g;i++) {
				FluxDiTJoinBlockRoPE block = mids.get(i);
				if(idsKeep != null) {
					block.forward(h_x, h_c, t, cos, sin, idsKeep);
				}else {
					block.forward(h_x, h_c, t, cos, sin);
				}
	    		h_x = block.getOutput();
	    		h_c = block.context_block.getOutput();
	    	}
		}else {
			h_x = mids.get(num_g - 1).getOutput();
			h_c = mids.get(num_g - 1).context_block.getOutput();
		}

		/**
		 * pad_mask
		 */
		if(idsKeep != null && network.RUN_MODEL == RunModel.TRAIN) {
			fusion.forward(h_x, e_x, h_c, e_c, idsKeep);
		}else if(uncond){
			fusion.forward_uncond(h_x, e_x, h_c, e_c);
		}else {
			fusion.forward(h_x, e_x, h_c, e_c);
		}
		
		/**
		 * decoders
		 */
		Tensor d_x = fusion.getOutput();
		Tensor d_c = fusion.getCondOutput();
    	for(int i = 0;i<num_h;i++) {
    		FluxDiTJoinBlockRoPE block = decoders.get(i);
    		block.forward(d_x, d_c, t, cos, sin);
    		d_x = block.getOutput();
    		d_c = block.context_block.getOutput();
    	}
    	
    	finalLayer.forward(d_x, t);

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

    }
    
    public void diff(Tensor cos, Tensor sin) {
        // TODO Auto-generated method stub
    	/**
    	 * unpatchify back
    	 */
    	Tensor_OP().permute(delta, finalLayer.getOutput(), yShape, xShape, new int[] {0, 2, 4, 3, 5, 1});
    	
    	finalLayer.back(finalLayer.getOutput(), dtc);
    	
    	Tensor dy = finalLayer.diff;
    	Tensor dcx = null;
    	/**
    	 * decoder backward
    	 */
    	for(int i = num_h - 1;i>=0;i--) {
    		FluxDiTJoinBlockRoPE block = decoders.get(i);
    		block.back(dy, dcx, dtc, cos, sin);
    		dy = block.diff;
    		dcx = block.context_block.diff;
    	}

    	/**
		 * pad_mask backward
		 */
    	if(idsKeep != null) {
			fusion.back(dy, dcx, dencoder, dcx, dencoder_c, idsKeep);
		}else {
			fusion.back(dy, dcx, dencoder, dcx, dencoder_c);
		}

		/**
		 * mids backward
		 */
		Tensor dh = fusion.diff;
		Tensor dhc = dcx;
		for(int i = num_g - 1;i>=0;i--) {
    		FluxDiTJoinBlockRoPE block = mids.get(i);
    		if(idsKeep != null) {
    			block.back(dh, dhc, dtc, cos, sin, idsKeep);
    		}else {
     			block.back(dh, dhc, dtc, cos, sin);
    		}
    		dh = block.diff;
    		dhc = block.context_block.diff;
    	}

//		/**
//		 * sprint backward
//		 */
		Tensor de = dh;
		Tensor dec = dhc;
		if(idsKeep != null) {
			tokenDropKernel.imgTokenDropBack(drop_delta, idsKeep, dh, token_t, hw, 0, hiddenSize);
			de = drop_delta;
		}
		
		/**
		 * repa backward
		 */
		Tensor_OP().add(de, z_mlp.diff, de);
		
    	Tensor_OP().add(de, dencoder, de);
    	Tensor_OP().add(dec, dencoder_c, dec);
    	
		/**
		 * encoder backward
		 */
    	for(int i = num_f - 1;i>=0;i--) {
    		FluxDiTJoinBlockRoPE block = encoders.get(i);
    		block.back(de, dec, dtc, cos, sin);
    		de = block.diff;
    		dec = block.context_block.diff;
    	}

     	labelEmbd.back(dec);
     	
     	timeEmbd.back(dtc);

     	patchEmbd.back(de);
     	
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
    
    public void back(Tensor delta, Tensor cos, Tensor sin) {
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
    }
    
    @Override
    public void update() {
        // TODO Auto-generated method stub
    	patchEmbd.update();

    	timeEmbd.update();
    	
    	labelEmbd.update();
    	
    	for(int i = 0;i<num_f;i++) {
    		encoders.get(i).update();
    	}
    	
    	for(int i = 0;i<num_g;i++) {
    		mids.get(i).update();
    	}
    	
    	fusion.update();
    	
    	for(int i = 0;i<num_h;i++) {
    		decoders.get(i).update();
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
    	
    	for(int i = 0;i<num_f;i++) {
    		encoders.get(i).saveModel(outputStream);
    	}
    	
    	for(int i = 0;i<num_g;i++) {
    		mids.get(i).saveModel(outputStream);
    	}
    	
    	fusion.saveModel(outputStream);
    	
    	for(int i = 0;i<num_h;i++) {
    		decoders.get(i).saveModel(outputStream);
    	}

    	z_mlp.saveModel(outputStream);
    	
    	finalLayer.saveModel(outputStream);
    }

    public void loadModel(RandomAccessFile inputStream) throws IOException {
    	patchEmbd.loadModel(inputStream);

    	timeEmbd.loadModel(inputStream);
    	
    	labelEmbd.loadModel(inputStream);
    	
    	for(int i = 0;i<num_f;i++) {
    		encoders.get(i).loadModel(inputStream);
    	}
    	
    	for(int i = 0;i<num_g;i++) {
    		mids.get(i).loadModel(inputStream);
    	}
    	
    	fusion.loadModel(inputStream);
    	
    	for(int i = 0;i<num_h;i++) {
    		decoders.get(i).loadModel(inputStream);
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
    	
    	for(int i = 0;i<num_f;i++) {
    		encoders.get(i).accGrad(scale);
    	}
    	
    	for(int i = 0;i<num_g;i++) {
    		mids.get(i).accGrad(scale);
    	}
    	
    	fusion.accGrad(scale);
    	
    	for(int i = 0;i<num_h;i++) {
    		decoders.get(i).accGrad(scale);
    	}
    	
    	z_mlp.accGrad(scale);
    	
    	finalLayer.accGrad(scale);
    }
    
    public static void loadWeight(Map<String, Object> weightMap, FluxDiTMainMoudue_Sprint3 block, boolean showLayers) {
        if (showLayers) {
            for (String key : weightMap.keySet()) {
                System.out.println(key);
            }
        }
//        ModeLoaderlUtils.loadData(block.patchEmbd.patchEmbedding.weight, weightMap, "x_embedder.proj.weight", 4);
//        ModeLoaderlUtils.loadData(block.patchEmbd.patchEmbedding.bias, weightMap, "x_embedder.proj.bias");
//        
//        ModeLoaderlUtils.loadData(block.timeEmbd.linear1.weight, weightMap, "t_embedder.mlp.0.weight");
//        ModeLoaderlUtils.loadData(block.timeEmbd.linear1.bias, weightMap, "t_embedder.mlp.0.bias");
//        ModeLoaderlUtils.loadData(block.timeEmbd.linear2.weight, weightMap, "t_embedder.mlp.2.weight");
//        ModeLoaderlUtils.loadData(block.timeEmbd.linear2.bias, weightMap, "t_embedder.mlp.2.bias");
//
//        ModeLoaderlUtils.loadData(block.labelEmbd.init_y_embedding(), weightMap, "y_embedder.y_embedding");
//        
//        ModeLoaderlUtils.loadData(block.labelEmbd.linear1.weight, weightMap, "y_embedder.y_proj.fc1.weight");
//        ModeLoaderlUtils.loadData(block.labelEmbd.linear1.bias, weightMap, "y_embedder.y_proj.fc1.bias");
//        ModeLoaderlUtils.loadData(block.labelEmbd.linear2.weight, weightMap, "y_embedder.y_proj.fc2.weight");
//        ModeLoaderlUtils.loadData(block.labelEmbd.linear2.bias, weightMap, "y_embedder.y_proj.fc2.bias");
//
//        for(int i = 0;i<2;i++) {
//        	
//        	FluxDiTBlock b = block.encoders.get(i);
//        	
//        	b.norm1.gamma = ModeLoaderlUtils.loadData(b.norm1.gamma, weightMap, 1, "blocks."+i+".norm1.weight"); 
//        	b.norm3.gamma = ModeLoaderlUtils.loadData(b.norm3.gamma, weightMap, 1, "blocks."+i+".norm2.weight");  
//        	
//        	ModeLoaderlUtils.loadData(b.attn.qLinerLayer.weight, weightMap, "blocks."+i+".attn.q.weight");
//            ModeLoaderlUtils.loadData(b.attn.qLinerLayer.bias, weightMap, "blocks."+i+".attn.q.bias");
//        	ModeLoaderlUtils.loadData(b.attn.kLinerLayer.weight, weightMap, "blocks."+i+".attn.k.weight");
//            ModeLoaderlUtils.loadData(b.attn.kLinerLayer.bias, weightMap, "blocks."+i+".attn.k.bias");
//        	ModeLoaderlUtils.loadData(b.attn.vLinerLayer.weight, weightMap, "blocks."+i+".attn.v.weight");
//            ModeLoaderlUtils.loadData(b.attn.vLinerLayer.bias, weightMap, "blocks."+i+".attn.v.bias");
//        	ModeLoaderlUtils.loadData(b.attn.oLinerLayer.weight, weightMap, "blocks."+i+".attn.proj.weight");
//            ModeLoaderlUtils.loadData(b.attn.oLinerLayer.bias, weightMap, "blocks."+i+".attn.proj.bias");
//            
//        	ModeLoaderlUtils.loadData(b.mlp.w12.weight, weightMap, "blocks."+i+".mlp.w12.weight");
//        	ModeLoaderlUtils.loadData(b.mlp.w12.bias, weightMap, "blocks."+i+".mlp.w12.bias");
//        	ModeLoaderlUtils.loadData(b.mlp.w3.weight, weightMap, "blocks."+i+".mlp.w3.weight");
//        	ModeLoaderlUtils.loadData(b.mlp.w3.bias, weightMap, "blocks."+i+".mlp.w3.bias");
//        	
//        	ModeLoaderlUtils.loadData(b.adaLN_modulation.weight, weightMap, "blocks."+i+".adaLN_modulation.1.weight");
//        	ModeLoaderlUtils.loadData(b.adaLN_modulation.bias, weightMap, "blocks."+i+".adaLN_modulation.1.bias");
//        }
//        
//        for(int i = 0;i<8;i++) {
//        	int idx = i + 2;
//        	FluxDiTBlock b = block.mids.get(i);
//        	
//        	b.norm1.gamma = ModeLoaderlUtils.loadData(b.norm1.gamma, weightMap, 1, "blocks."+idx+".norm1.weight"); 
//        	b.norm3.gamma = ModeLoaderlUtils.loadData(b.norm3.gamma, weightMap, 1, "blocks."+idx+".norm2.weight");  
//        	
//        	ModeLoaderlUtils.loadData(b.attn.qLinerLayer.weight, weightMap, "blocks."+idx+".attn.q.weight");
//            ModeLoaderlUtils.loadData(b.attn.qLinerLayer.bias, weightMap, "blocks."+idx+".attn.q.bias");
//        	ModeLoaderlUtils.loadData(b.attn.kLinerLayer.weight, weightMap, "blocks."+idx+".attn.k.weight");
//            ModeLoaderlUtils.loadData(b.attn.kLinerLayer.bias, weightMap, "blocks."+idx+".attn.k.bias");
//        	ModeLoaderlUtils.loadData(b.attn.vLinerLayer.weight, weightMap, "blocks."+idx+".attn.v.weight");
//            ModeLoaderlUtils.loadData(b.attn.vLinerLayer.bias, weightMap, "blocks."+idx+".attn.v.bias");
//        	ModeLoaderlUtils.loadData(b.attn.oLinerLayer.weight, weightMap, "blocks."+idx+".attn.proj.weight");
//            ModeLoaderlUtils.loadData(b.attn.oLinerLayer.bias, weightMap, "blocks."+idx+".attn.proj.bias");
//            
//        	ModeLoaderlUtils.loadData(b.mlp.w12.weight, weightMap, "blocks."+idx+".mlp.w12.weight");
//        	ModeLoaderlUtils.loadData(b.mlp.w12.bias, weightMap, "blocks."+idx+".mlp.w12.bias");
//        	ModeLoaderlUtils.loadData(b.mlp.w3.weight, weightMap, "blocks."+idx+".mlp.w3.weight");
//        	ModeLoaderlUtils.loadData(b.mlp.w3.bias, weightMap, "blocks."+idx+".mlp.w3.bias");
//        	
//        	ModeLoaderlUtils.loadData(b.adaLN_modulation.weight, weightMap, "blocks."+idx+".adaLN_modulation.1.weight");
//        	ModeLoaderlUtils.loadData(b.adaLN_modulation.bias, weightMap, "blocks."+idx+".adaLN_modulation.1.bias");
//        }
//
//        block.fusion.weight = ModeLoaderlUtils.loadData(block.fusion.weight, weightMap, 3, "mask_token"); 
//        ModeLoaderlUtils.loadData(block.fusion.fusion_proj.weight, weightMap, "fusion_proj.weight");
//        ModeLoaderlUtils.loadData(block.fusion.fusion_proj.bias, weightMap, "fusion_proj.bias");
//        
//        for(int i = 0;i<2;i++) {
//        	int idx = i + 10;
//        	FluxDiTBlock b = block.decoders.get(i);
//        	
//        	b.norm1.gamma = ModeLoaderlUtils.loadData(b.norm1.gamma, weightMap, 1, "blocks."+idx+".norm1.weight"); 
//        	b.norm3.gamma = ModeLoaderlUtils.loadData(b.norm3.gamma, weightMap, 1, "blocks."+idx+".norm2.weight");  
//        	
//        	ModeLoaderlUtils.loadData(b.attn.qLinerLayer.weight, weightMap, "blocks."+idx+".attn.q.weight");
//            ModeLoaderlUtils.loadData(b.attn.qLinerLayer.bias, weightMap, "blocks."+idx+".attn.q.bias");
//        	ModeLoaderlUtils.loadData(b.attn.kLinerLayer.weight, weightMap, "blocks."+idx+".attn.k.weight");
//            ModeLoaderlUtils.loadData(b.attn.kLinerLayer.bias, weightMap, "blocks."+idx+".attn.k.bias");
//        	ModeLoaderlUtils.loadData(b.attn.vLinerLayer.weight, weightMap, "blocks."+idx+".attn.v.weight");
//            ModeLoaderlUtils.loadData(b.attn.vLinerLayer.bias, weightMap, "blocks."+idx+".attn.v.bias");
//        	ModeLoaderlUtils.loadData(b.attn.oLinerLayer.weight, weightMap, "blocks."+idx+".attn.proj.weight");
//            ModeLoaderlUtils.loadData(b.attn.oLinerLayer.bias, weightMap, "blocks."+idx+".attn.proj.bias");
//            
//        	ModeLoaderlUtils.loadData(b.mlp.w12.weight, weightMap, "blocks."+idx+".mlp.w12.weight");
//        	ModeLoaderlUtils.loadData(b.mlp.w12.bias, weightMap, "blocks."+idx+".mlp.w12.bias");
//        	ModeLoaderlUtils.loadData(b.mlp.w3.weight, weightMap, "blocks."+idx+".mlp.w3.weight");
//        	ModeLoaderlUtils.loadData(b.mlp.w3.bias, weightMap, "blocks."+idx+".mlp.w3.bias");
//        	
//        	ModeLoaderlUtils.loadData(b.adaLN_modulation.weight, weightMap, "blocks."+idx+".adaLN_modulation.1.weight");
//        	ModeLoaderlUtils.loadData(b.adaLN_modulation.bias, weightMap, "blocks."+idx+".adaLN_modulation.1.bias");
//        }
//        
//        block.finalLayer.finalNorm.gamma = ModeLoaderlUtils.loadData(block.finalLayer.finalNorm.gamma, weightMap, 1, "final_layer.norm_final.weight"); 
//        ModeLoaderlUtils.loadData(block.finalLayer.finalLinear.weight, weightMap, "final_layer.linear.weight");
//        ModeLoaderlUtils.loadData(block.finalLayer.finalLinear.bias, weightMap, "final_layer.linear.bias");
//        ModeLoaderlUtils.loadData(block.finalLayer.m_linear1.weight, weightMap, "final_layer.adaLN_modulation_linear1.weight");
//        ModeLoaderlUtils.loadData(block.finalLayer.m_linear1.bias, weightMap, "final_layer.adaLN_modulation_linear1.bias");
//        ModeLoaderlUtils.loadData(block.finalLayer.m_linear2.weight, weightMap, "final_layer.adaLN_modulation_linear2.weight");
//        ModeLoaderlUtils.loadData(block.finalLayer.m_linear2.bias, weightMap, "final_layer.adaLN_modulation_linear2.bias");
    }
    
    public static void main(String[] args) {
    	int N = 2;
    	int C = 32;
    	int H = 16;
    	int W = 16;
    	
    	int TT = 77;
    	int TEM = 768;
    	
    	int patchSize = 1;
    	int hiddenSize = 768;
    	int headNum = 12;
    	int depth = 12;
    	
    	CNN nn = new CNN(null);
        nn.CUDNN = true;
        nn.number = N;
    	
        FluxDiTMainMoudue_Sprint3 jb = new FluxDiTMainMoudue_Sprint3(C, W, H, patchSize, hiddenSize, headNum, depth, 1000, TEM, TT, 4, 768, 0.0f, 0.75f, 0.8f, nn);
    	
        String weight = "D:\\models\\dit_weight.json";
        loadWeight(LagJsonReader.readJsonFileBigWeightIterator(weight), jb, true);
        
	    String inputPath = "D:\\models\\dit_x.json";
	    Map<String, Object> datas = LagJsonReader.readJsonFileSmallWeight(inputPath);
	    Tensor input = new Tensor(N, C, H, W, true);
	    ModeLoaderlUtils.loadData(input, datas, "x");
    	
	    String cyPath = "D:\\models\\dit_context.json";
	    Map<String, Object> cydatas = LagJsonReader.readJsonFileSmallWeight(cyPath);
	    Tensor cy = new Tensor(N, TT, 1, TEM, true);
	    ModeLoaderlUtils.loadData(cy, cydatas, "context", 3);
	    
	    String idskeepPath = "D:\\models\\dit_ids_keep.json";
	    Map<String, Object> idskeepDatas = LagJsonReader.readJsonFileSmallWeight(idskeepPath);
	    Tensor idskeep = new Tensor(N, 1, 1, 64, true);
	    ModeLoaderlUtils.loadData(idskeep, idskeepDatas, "ids_keep", 2);
	    
	    Tensor t = new Tensor(N, 1, 1, 1, new float[] {0.1f, 0.8f}, true);
	    int time = (W / patchSize) * (H / patchSize);
	    Tensor[] cs = RoPEKernel.getCosAndSin2D(time, hiddenSize, headNum);
        Tensor cos = cs[0];
        Tensor sin = cs[1];
	    
        cy.view(N * TT, 1, 1, TEM);
        jb.setIdsKeep(idskeep);
	    jb.forward(input, t, cy, cos, sin);
	    
	    jb.getOutput().showDM("output");
	    
	    String deltaPath = "D:\\models\\dit_delta.json";
	    Map<String, Object> deltaDatas = LagJsonReader.readJsonFileSmallWeight(deltaPath);
	    Tensor delta = new Tensor(N, C, H, W, true);
	    ModeLoaderlUtils.loadData(delta, deltaDatas, "delta");
	    
	    jb.back(delta, cos, sin);
	    
    }
    
    public Tensor getZ() {
    	return z_mlp.getOutput();
    }
    
    public void setZGrad(Tensor delta) {
    	z_mlp.back(delta);
    }

	public void setIdsKeep(Tensor idsKeep) {
		this.idsKeep = idsKeep;
	}
    
}

