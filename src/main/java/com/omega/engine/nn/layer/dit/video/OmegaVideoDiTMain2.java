package com.omega.engine.nn.layer.dit.video;

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
import com.omega.engine.nn.layer.dit.kernel.TokenDropKernel;
import com.omega.engine.nn.layer.dit.sprint.FusionLayer2;
import com.omega.engine.nn.layer.dit.video.block.VideoDiTConvBlock;
import com.omega.engine.nn.layer.dit.video.rope.RoPE3DKernel;
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
public class OmegaVideoDiTMain2 extends Layer {
	
	public int inChannel;
	public int num_frames;
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
    
    public OmegaVideoPatchEmbed patchEmbd;
    public DiTOrgTimeEmbeddingLayer timeEmbd;
    public DiTCaptionEmbeddingLayer labelEmbd;
    public List<VideoDiTConvBlock> encoders;
    public List<VideoDiTConvBlock> mids;
    public FusionLayer2 fusion;
    public List<VideoDiTConvBlock> decoders;
    public OmegaVideoFinalLayer finalLayer;
    
    private int thw;
    
    private Tensor cat_x;
    private Tensor img_x;
    
    private Tensor d_o;
    
    private Tensor dtc;
    private Tensor dencoder;
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
    
//    private List<Integer> ids;
    
    public boolean uncond = false;
    
    public OmegaVideoDiTMain2(int inChannel, int num_frames, int height, int width, int patchSize, int hiddenSize, int headNum, int depth, int timeSteps, int textEmbedDim, int maxContextLen, int mlpRatio, float y_drop_prob, float token_drop_ratio, float path_drop_prob, Network network) {
		this.network = network;
        if (this.updater == null) {
            this.setUpdater(UpdaterFactory.create(network));
        }
        this.y_drop_prob = y_drop_prob;
    	this.inChannel = inChannel;
    	this.num_frames = num_frames;
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
		this.initLayers();
		this.oHeight = height;
		this.oWidth = width;
    }

    public void initLayers() {
    	
    	patchEmbd = new OmegaVideoPatchEmbed(inChannel, num_frames, height, width, hiddenSize, patchSize, true, network);
        
    	thw = patchEmbd.oChannel * patchEmbd.oDepth;

		this.token_t = (int) (thw * (1.0f - token_drop_ratio));

        timeEmbd = new DiTOrgTimeEmbeddingLayer(timeSteps, 256, hiddenSize, true, network);

        labelEmbd = new DiTCaptionEmbeddingLayer(textEmbedDim, hiddenSize, maxContextLen, y_drop_prob, true, network);
        
        encoders = new ArrayList<VideoDiTConvBlock>();
        mids = new ArrayList<VideoDiTConvBlock>();
        decoders = new ArrayList<VideoDiTConvBlock>();
        
        for(int i = 0;i<num_f;i++) {
        	VideoDiTConvBlock block = new VideoDiTConvBlock(hiddenSize, hiddenSize, thw + maxContextLen, mlpRatio * hiddenSize, headNum, maxContextLen, num_frames, height, width, true, false, network);
        	encoders.add(block);
        }
        
        for(int i = 0;i<num_g;i++) {
        	VideoDiTConvBlock block = new VideoDiTConvBlock(hiddenSize, hiddenSize, token_t + maxContextLen, mlpRatio * hiddenSize, headNum, maxContextLen, num_frames, height, width, true, false, network);
        	mids.add(block);
        }

        fusion = new FusionLayer2(hiddenSize, thw, token_t, maxContextLen, path_drop_prob, network);
        
        for(int i = 0;i<num_h;i++) {
        	VideoDiTConvBlock block = new VideoDiTConvBlock(hiddenSize, hiddenSize, thw + maxContextLen, mlpRatio * hiddenSize, headNum, maxContextLen, num_frames, height, width, true, false, network);
        	decoders.add(block);
        }
        
        this.oChannel = inChannel;

        finalLayer = new OmegaVideoFinalLayer(patchSize, hiddenSize, inChannel, thw, true, true, network);
        
        if(baseKernel == null) {
        	baseKernel = new BaseKernel(cuda());
        }
        
        if(tokenDropKernel == null) {
        	tokenDropKernel = new TokenDropKernel(cuda());
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
    	float[] emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim/2, grid_h);
    	float[] emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim/2, grid_w);
    	
    	float[] emb = cat(emb_h, emb_w, embed_dim/2);
    	return emb;
    }
    
    public void init(Tensor input) {
        // TODO Auto-generated method stub
        this.number = input.number;
        if(this.output == null || this.output.number != number) {
        	cat_x = Tensor.createGPUTensor(cat_x, number * (thw + maxContextLen), 1, 1, patchEmbd.oWidth, true);
        	img_x = Tensor.createGPUTensor(img_x, number * thw, 1, 1, patchEmbd.oWidth, true);
        	output = Tensor.createGPUTensor(output, number, oChannel * num_frames, oHeight, oWidth, true);
        }
        
        if(token_t < thw && (idsKeep == null || idsKeep.number != number)) {
        	idsKeep = Tensor.createGPUTensor(idsKeep, number, 1, 1, token_t, true);
        	td_x = Tensor.createGPUTensor(td_x, number * (maxContextLen + token_t), 1, 1, hiddenSize, true);
        }
        
        if(patchEmbd.getOutput() != null){
        	patchEmbd.getOutput().viewOrg();
        }
        
    }
    
    @Override
    public void initBack() {
        // TODO Auto-generated method stub
    	if(dtc == null || dtc.number != timeEmbd.getOutput().number) {
    		d_o = Tensor.createGPUTensor(d_o, input.number * (maxContextLen + thw), 1, 1, patchEmbd.getOutput().width, true);
    		dtc = Tensor.createGPUTensor(dtc, timeEmbd.getOutput().shape(), true);
    		dencoder = Tensor.createGPUTensor(dencoder, number * (maxContextLen + thw), 1, 1, hiddenSize, true);
//    		drop_delta = Tensor.createGPUTensor(drop_delta, number * (maxContextLen + thw), 1, 1, hiddenSize, true);
    	}else {
    		dtc.clearGPU();
    		d_o.clearGPU();
//    		dencoder.clearGPU();
//    		drop_delta.clearGPU();
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
    
    public void output(Tensor tc, Tensor label, Tensor[] cos, Tensor[] sin) {
    	
    	patchEmbd.forward(input);

    	timeEmbd.forward(tc);
    	
    	labelEmbd.forward(label);
    	
    	Tensor x = patchEmbd.getOutput().view(patchEmbd.getOutput().number * patchEmbd.getOutput().channel, 1, 1, patchEmbd.getOutput().width);

    	Tensor t = timeEmbd.getOutput();
    	
    	Tensor cond = labelEmbd.getOutput();

     	baseKernel.concat_channel_forward(cond, x, cat_x, input.number, maxContextLen, thw, 1, patchEmbd.getOutput().width);
     	
    	/**
    	 * encoder
    	 */
    	Tensor e_x = cat_x;
    	for(int i = 0;i<num_f;i++) {
    		VideoDiTConvBlock block = encoders.get(i);
    		block.forward(e_x, t, cos, sin);
    		e_x = block.getOutput();
//    		e_x.showDMByOffsetRed(150 * 1152, 1152, "e_x:"+i);
    	}
//    	e_x.showDM("e_x");
		/**
		 * sprint
		 */
		Tensor h_x = e_x;
		if(!uncond) {
			if(idsKeep != null && network.RUN_MODEL == RunModel.TRAIN) {
				tokenDropKernel.idsKeep(idsKeep, number, (thw - 1), token_t);
				tokenDropKernel.imgTokenDrop(e_x, idsKeep, td_x, token_t, thw, maxContextLen, hiddenSize);
				h_x = td_x;
			}
			
			/**
			 * mids
			 */
			for(int i = 0;i<num_g;i++) {
				VideoDiTConvBlock block = mids.get(i);
				block.forward(h_x, t, cos, sin);
	    		h_x = block.getOutput();
	    	}
		}else {
			h_x = mids.get(num_g - 1).getOutput();
		}
//		h_x.showDMByOffsetRed(150 * 1152, 1152, "h_x");
		/**
		 * pad_mask
		 */
		if(idsKeep != null && network.RUN_MODEL == RunModel.TRAIN) {
			fusion.forward(h_x, e_x, idsKeep);
		}else if(uncond){
			fusion.forward_uncond(h_x, e_x);
		}else {
			fusion.forward(h_x, e_x);
		}

		/**
		 * decoders
		 */
		Tensor d_x = fusion.getOutput();
    	for(int i = 0;i<num_h;i++) {
    		VideoDiTConvBlock block = decoders.get(i);
    		block.forward(d_x, t, cos, sin);
    		d_x = block.getOutput();
    	}
    	
    	Tensor_OP().getByChannel(d_x, img_x, new int[] {input.number, maxContextLen + thw, 1, patchEmbd.getOutput().width}, maxContextLen, thw);
//    	img_x.showShape("img_x");
    	finalLayer.forward(img_x, t);
//    	finalLayer.getOutput().showShape("finalLayer");
    	/**
    	 * unpatchify
    	 * x: (N, T, patch_size**2 * C)
         * imgs: (N, C, F, H, W)
    	 */
    	if(xShape == null) {
    		int f = num_frames / patchSize;
    		int h = height/patchSize;
        	int w = width/patchSize;
        	xShape = new int[] {number, f, h, w, patchSize, patchSize, patchSize, oChannel};
        	yShape = new int[] {number, oChannel, f, patchSize, h, patchSize, w, patchSize};
    	}
    	Tensor_OP().permute(finalLayer.getOutput(), this.output, xShape, yShape, new int[] {0, 7, 1, 4, 2, 5, 3, 6});
//    	output.showDM("out");
    }
    
    @Override
    public Tensor getOutput() {
        // TODO Auto-generated method stub
        return output;
    }

    @Override
    public void diff() {

    }
    
    public void diff(Tensor[] cos, Tensor[] sin) {
        // TODO Auto-generated method stub
    	/**
    	 * unpatchify back
    	 */
    	Tensor_OP().permute(delta, finalLayer.getOutput(), yShape, xShape, new int[] {0, 2, 4, 6, 3, 5, 7, 1});
    	
    	finalLayer.back(finalLayer.getOutput(), dtc);
    	
    	Tensor dy = d_o;
    	dy.clearGPU();

    	Tensor_OP().getByChannel_back(dy, finalLayer.diff, new int[] {input.number, maxContextLen + thw, 1, patchEmbd.getOutput().width}, maxContextLen, thw);

    	/**
    	 * decoder backward
    	 */
    	for(int i = num_h - 1;i>=0;i--) {
    		VideoDiTConvBlock block = decoders.get(i);
    		block.back(dy, dtc, cos, sin);
    		dy = block.diff;
    	}
    	
//     	dy.showDMByOffsetRed(150 * 1152, 1152, "fusion_delta");
    	
    	/**
		 * pad_mask backward
		 */
		if(idsKeep != null) {
			fusion.back(dy, dencoder, idsKeep);
		}else {
			fusion.back(dy, dencoder);
		}
    	
		/**
		 * mids backward
		 */
		Tensor dh = fusion.diff;
		for(int i = num_g - 1;i>=0;i--) {
			VideoDiTConvBlock block = mids.get(i);
			block.back(dh, dtc, cos, sin);
    		dh = block.diff;
    	}
		
		/**
		 * sprint backward
		 */
		Tensor de = dh;
		if(idsKeep != null) {
			tokenDropKernel.imgTokenDropBack(drop_delta, idsKeep, dh, token_t, thw, maxContextLen, hiddenSize);
			de = drop_delta;
		}
		
		/**
		 * repa backward
		 */
    	Tensor_OP().add(dencoder, de, de);
 
		/**
		 * encoder backward
		 */
    	for(int i = num_f - 1;i>=0;i--) {
    		VideoDiTConvBlock block = encoders.get(i);
    		block.back(de, dtc, cos, sin);
    		de = block.diff;
    	}

    	baseKernel.concat_channel_backward(de, labelEmbd.getOutput(), img_x, input.number, maxContextLen, thw, 1, patchEmbd.getOutput().width);

     	labelEmbd.back(labelEmbd.getOutput());
     	
     	timeEmbd.back(dtc);
//     	img_x.showDMByOffsetRed(150 * 1152, 1152, "img_x");
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
     * @param input
     * @param tc time cond
     * @param text
     */
    public void forward(Tensor input,Tensor tc,Tensor text, Tensor[] cos, Tensor[] sin) {
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
    
    public void back(Tensor delta, Tensor[] cos, Tensor[] sin) {
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
    	
    	finalLayer.accGrad(scale);
    }
    
    public static void loadWeight(Map<String, Object> weightMap, OmegaVideoDiTMain2 block, boolean showLayers) {
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
//        	VideoDiTBlock b = block.encoders.get(i);
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
//        for(int i = 0;i<2;i++) {
//        	int idx = i + 2;
//        	VideoDiTBlock b = block.mids.get(i);
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
//        	int idx = i + 4;
//        	VideoDiTBlock b = block.decoders.get(i);
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
    	int C = 128;
    	int num_frames = 3;
    	int H = 11;
    	int W = 20;
    	
    	int TT = 77;
    	int TEM = 768;
    	
    	int patchSize = 1;
    	int hiddenSize = 1152;
    	int headNum = 16;
    	int depth = 6;
    	
    	CNN nn = new CNN(null);
        nn.CUDNN = true;
        nn.number = N;
//        nn.RUN_MODEL = RunModel.EVAL;
    	
        OmegaVideoDiTMain2 jb = new OmegaVideoDiTMain2(C, num_frames, H, W, patchSize, hiddenSize, headNum, depth, 1000, TEM, TT, 4, 0.1f, 0.0f, 0.05f, nn);
    	
        String weight = "D:\\models\\ltx_vae\\sit.json";
        loadWeight(LagJsonReader.readJsonFileBigWeightIterator(weight), jb, true);
        
	    String inputPath = "D:\\models\\ltx_vae\\input_x.json";
	    Map<String, Object> datas = LagJsonReader.readJsonFileSmallWeight(inputPath);
	    Tensor input = new Tensor(N, C * num_frames, H, W, true);
	    ModeLoaderlUtils.loadData(input, datas, "input", 5);
    	
	    String cyPath = "D:\\models\\ltx_vae\\input_x.json";
	    Map<String, Object> cydatas = LagJsonReader.readJsonFileSmallWeight(cyPath);
	    Tensor cy = new Tensor(N, TT, 1, TEM, true);
	    ModeLoaderlUtils.loadData(cy, cydatas, "txt", 3);
        
	    Tensor t = new Tensor(N, 1, 1, 1, new float[] {0.1f, 0.8f}, true);
	    
        Tensor[][] cs = RoPE3DKernel.init3DRoPE(num_frames, H, W, hiddenSize, headNum, 1.0f, 1.4375f, 2.5f);
        Tensor[] cos = cs[0];
        Tensor[] sin = cs[1];
        
        cy.view(N * TT, 1, 1, TEM);
//        input.showDM("input");
        
	    String deltaPath = "D:\\models\\ltx_vae\\delta.json";
	    Map<String, Object> deltaDatas = LagJsonReader.readJsonFileSmallWeight(deltaPath);
	    Tensor delta = new Tensor(N, C * num_frames, H, W, true);
	    ModeLoaderlUtils.loadData(delta, deltaDatas, "delta", 5);
        
        for(int i = 0;i<1;i++) {
        	 
            jb.forward(input, t, cy, cos, sin);
    	    
    	    jb.getOutput().showDM("output");
    	    delta.showDM("delta");
    	    jb.back(delta, cos, sin);
    	    
        }
       
	    
    }
    
	public void setIdsKeep(Tensor idsKeep) {
		this.idsKeep = idsKeep;
	}
    
}

