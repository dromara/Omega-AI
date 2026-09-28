package com.omega.engine.nn.layer.dit.sprint;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import com.omega.common.utils.MatrixOperation;
import com.omega.engine.gpu.BaseKernel;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.dit.DiTMaskCaptionEmbeddingLayer;
import com.omega.engine.nn.layer.dit.DiTOrgTimeEmbeddingLayer;
import com.omega.engine.nn.layer.dit.DiTPatchEmbeddingLayer;
import com.omega.engine.nn.layer.dit.flux.FluxDiTBlock;
import com.omega.engine.nn.layer.dit.flux.REPAMLPLayer;
import com.omega.engine.nn.layer.dit.kernel.TokenDropKernel;
import com.omega.engine.nn.layer.dit.mmjit.PlainTextBlock;
import com.omega.engine.nn.layer.dit.txt.DiT_TXTFinalLayer;
import com.omega.engine.nn.network.Network;
import com.omega.engine.nn.network.RunModel;
import com.omega.engine.tensor.Tensor;
import com.omega.engine.updater.UpdaterFactory;

/**
 * DiT_Block
 * @author Administrator
 */
public class OmegaDiTMainMoudue_Sprint_T5 extends Layer {
	
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
    private int txt_depth = 2;
    private int t5EmbedDim;
    private int maxContextLen;
    private int mlpRatio = 4;

    private int z_dim = 768;
    private int projector_dim = 2048;
    
    public DiTPatchEmbeddingLayer patchEmbd;
    public DiTOrgTimeEmbeddingLayer timeEmbd;
    public DiTMaskCaptionEmbeddingLayer t5Embd;
    public List<PlainTextBlock> txt_blocks;
    public List<FluxDiTBlock> encoders;
    public List<FluxDiTBlock> mids;
    public FusionLayer2 fusion;
    public List<FluxDiTBlock> decoders;
    public DiT_TXTFinalLayer finalLayer;
    
    public REPAMLPLayer z_mlp;
    
    private int hw;
    
    private Tensor posEmbd;
    
    private Tensor cat_x;
    private Tensor img_x;
    
    private Tensor z_img_x;
    
    private Tensor d_o;
    
    private Tensor dvec;
    private Tensor dencoder;
    private Tensor drop_delta;
    private Tensor tmp_cond;
    
    private float y_drop_prob = 0.0f;
    
    private float path_drop_prob = 0.0f;

    private int[] xShape;
    private int[] yShape;
    
    private BaseKernel baseKernel;
    private TokenDropKernel tokenDropKernel;
    
    public boolean uncond = false;
    
    public OmegaDiTMainMoudue_Sprint_T5(int inChannel, int width, int height, int patchSize, int hiddenSize, int headNum, int txt_depth, int depth, int num_f, int num_h, int timeSteps, int t5EmbedDim, int maxContextLen, int mlpRatio, int z_dim, float y_drop_prob, float path_drop_prob, Network network) {
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
		this.num_f = num_f;
		this.num_h = num_h;
		this.num_g = this.depth - num_f - num_h;
		this.timeSteps = timeSteps;
		this.txt_depth = txt_depth;
		this.t5EmbedDim = t5EmbedDim;
		this.maxContextLen = maxContextLen;
		this.mlpRatio = mlpRatio;
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

        timeEmbd = new DiTOrgTimeEmbeddingLayer(timeSteps, 256, hiddenSize, true, network);

        t5Embd = new DiTMaskCaptionEmbeddingLayer(t5EmbedDim, hiddenSize, maxContextLen, y_drop_prob, true, network);
        
        encoders = new ArrayList<FluxDiTBlock>();
        mids = new ArrayList<FluxDiTBlock>();
        decoders = new ArrayList<FluxDiTBlock>();
        
        txt_blocks = new ArrayList<PlainTextBlock>();
        
        for(int i = 0;i<txt_depth;i++) {
        	PlainTextBlock block = new PlainTextBlock(hiddenSize, maxContextLen, headNum, true, true, network);
        	txt_blocks.add(block);
        }
        
        for(int i = 0;i<num_f;i++) {
        	FluxDiTBlock block = new FluxDiTBlock(hiddenSize, hiddenSize, patchEmbd.oChannel + maxContextLen, mlpRatio * hiddenSize, headNum, maxContextLen, true, true, network);
        	encoders.add(block);
        }
        
        for(int i = 0;i<num_g;i++) {
        	FluxDiTBlock block = new FluxDiTBlock(hiddenSize, hiddenSize, patchEmbd.oChannel + maxContextLen, mlpRatio * hiddenSize, headNum, maxContextLen, true, true, network);
        	mids.add(block);
        }
        
        fusion = new FusionLayer2(hiddenSize, hw, patchEmbd.oChannel, maxContextLen, path_drop_prob, network);
        
        for(int i = 0;i<num_h;i++) {
        	FluxDiTBlock block = new FluxDiTBlock(hiddenSize, hiddenSize, patchEmbd.oChannel + maxContextLen, mlpRatio * hiddenSize, headNum, maxContextLen, true, true, network);
        	decoders.add(block);
        }
        
        this.oChannel = inChannel;

        finalLayer = new DiT_TXTFinalLayer(patchSize, hiddenSize, inChannel, patchEmbd.oChannel, true, true, network);
        
        z_mlp = new REPAMLPLayer(hiddenSize, projector_dim, z_dim, true, network);
        
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
    	if(dvec == null || dvec.number != timeEmbd.getOutput().number) {
    		d_o = Tensor.createGPUTensor(d_o, input.number * (maxContextLen + hw), 1, 1, patchEmbd.getOutput().width, true);
    		dvec = Tensor.createGPUTensor(dvec, timeEmbd.getOutput().shape(), true);
    		dencoder = Tensor.createGPUTensor(dencoder, number * (maxContextLen + hw), 1, 1, hiddenSize, true);
    		drop_delta = Tensor.createGPUTensor(drop_delta, number * (maxContextLen + hw), 1, 1, hiddenSize, true);
    		tmp_cond = Tensor.createGPUTensor(tmp_cond, number * maxContextLen, 1, 1, hiddenSize, true);
    	}else {
    		dvec.clearGPU();
    		d_o.clearGPU();
    		tmp_cond.clearGPU();
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
    
    public void output(Tensor tc, Tensor t5Label, Tensor attnMask, Tensor cos1d,Tensor sin1d, Tensor cos2d, Tensor sin2d) {
    	
    	patchEmbd.forward(input);

    	Tensor_OP().addAxis(patchEmbd.getOutput(), posEmbd, patchEmbd.getOutput(), posEmbd.channel * posEmbd.width);
    	
    	timeEmbd.forward(tc);
    	
    	t5Embd.forward(t5Label, attnMask);
    	
    	Tensor x = patchEmbd.getOutput().view(patchEmbd.getOutput().number * patchEmbd.getOutput().channel, 1, 1, patchEmbd.getOutput().width);
    	
    	Tensor vec = timeEmbd.getOutput();
    	
    	Tensor cond = t5Embd.getOutput();
    	
     	/**
     	 * txt attention
     	 */
     	Tensor bc = cond;
     	for(int i = 0;i<txt_depth;i++) {
        	PlainTextBlock block = txt_blocks.get(i);
        	block.forward(bc, cos1d, sin1d);
        	bc = block.getOutput();
        }

     	baseKernel.concat_channel_forward(bc, x, cat_x, input.number, maxContextLen, hw, 1, patchEmbd.getOutput().width);
     	
    	/**
    	 * encoder
    	 */
    	Tensor e_x = cat_x;
    	for(int i = 0;i<num_f;i++) {
    		FluxDiTBlock block = encoders.get(i);
    		block.forward(e_x, vec, cos2d, sin2d);
    		e_x = block.getOutput();
    	}

    	/**
    	 * repa
    	 */
		if(network.RUN_MODEL == RunModel.TRAIN) {
			Tensor_OP().getByChannel(e_x, z_img_x, new int[] {input.number, maxContextLen + hw, 1, patchEmbd.getOutput().width}, maxContextLen, hw);
			z_mlp.forward(z_img_x);
		}
		
		/**
		 * sprint
		 */
		Tensor h_x = e_x;
		if(!uncond) {
			/**
			 * mids
			 */
			for(int i = 0;i<num_g;i++) {
				FluxDiTBlock block = mids.get(i);
				block.forward(h_x, vec, cos2d, sin2d);
	    		h_x = block.getOutput();
	    	}
		}else {
			h_x = mids.get(num_g - 1).getOutput();
		}

		/**
		 * pad_mask
		 */
		if(uncond){
			fusion.forward_uncond(h_x, e_x);
		}else {
			fusion.forward(h_x, e_x);
		}

		/**
		 * decoders
		 */
		Tensor d_x = fusion.getOutput();
    	for(int i = 0;i<num_h;i++) {
    		FluxDiTBlock block = decoders.get(i);
    		block.forward(d_x, vec, cos2d, sin2d);
    		d_x = block.getOutput();
    	}
    	
    	Tensor_OP().getByChannel(d_x, img_x, new int[] {input.number, maxContextLen + hw, 1, patchEmbd.getOutput().width}, maxContextLen, hw);

    	finalLayer.forward(img_x, vec);

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
    
    public void diff(Tensor cos1d, Tensor sin1d, Tensor cos2d, Tensor sin2d) {
        // TODO Auto-generated method stub
    	/**
    	 * unpatchify back
    	 */
    	Tensor_OP().permute(delta, finalLayer.getOutput(), yShape, xShape, new int[] {0, 2, 4, 3, 5, 1});
    	
    	finalLayer.back(finalLayer.getOutput(), dvec);

    	Tensor dy = d_o;
    	dy.clearGPU();

    	Tensor_OP().getByChannel_back(dy, finalLayer.diff, new int[] {input.number, maxContextLen + hw, 1, patchEmbd.getOutput().width}, maxContextLen, hw);

    	/**
    	 * decoder backward
    	 */
    	for(int i = num_h - 1;i>=0;i--) {
    		FluxDiTBlock block = decoders.get(i);
    		block.back(dy, dvec, cos2d, sin2d);
    		dy = block.diff;
    	}
    	
    	/**
		 * pad_mask backward
		 */
    	fusion.back(dy, dencoder);
    	
		/**
		 * mids backward
		 */
		Tensor dh = fusion.diff;
		for(int i = num_g - 1;i>=0;i--) {
			FluxDiTBlock block = mids.get(i);
			block.back(dh, dvec, cos2d, sin2d);
    		dh = block.diff;
    	}
		
		/**
		 * sprint backward
		 */
		Tensor de = dh;
		
		/**
		 * repa backward
		 */
		Tensor_OP().getByChannel_add_back(de, z_mlp.diff, new int[] {input.number, maxContextLen + hw, 1, patchEmbd.getOutput().width}, maxContextLen, hw);
    	Tensor_OP().add(dencoder, de, de);
 
		/**
		 * encoder backward
		 */
    	for(int i = num_f - 1;i>=0;i--) {
    		FluxDiTBlock block = encoders.get(i);
    		block.back(de, dvec, cos2d, sin2d);
    		de = block.diff;
    	}

    	baseKernel.concat_channel_backward(de, tmp_cond, img_x, input.number, maxContextLen, hw, 1, patchEmbd.getOutput().width);
    	
		/**
		 * txt attention backward
		 */
    	Tensor dtc = tmp_cond;
    	for(int i = txt_depth - 1;i>=0;i--) {
    		PlainTextBlock block = txt_blocks.get(i);
    		block.back(dtc, cos1d, sin1d);
    		dtc = block.diff;
    	}
    	
    	t5Embd.back(dtc);
     	
     	timeEmbd.back(dvec);

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
    public void forward(Tensor input,Tensor tc, Tensor t5Label, Tensor attnMask, Tensor cos1d, Tensor sin1d, Tensor cos2d, Tensor sin2d) {
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
        this.output(tc, t5Label, attnMask, cos1d, sin1d, cos2d, sin2d);
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
    
    public void back(Tensor delta, Tensor cos1d, Tensor sin1d, Tensor cos2d, Tensor sin2d) {
        // TODO Auto-generated method stub
        this.initBack();
        /**
         * 设置梯度
         */
        this.setDelta(delta);
        /**
         * 计算梯度
         */
        this.diff(cos1d, sin1d, cos2d, sin2d);
    }
    
    @Override
    public void update() {
        // TODO Auto-generated method stub
    	patchEmbd.update();

    	timeEmbd.update();
    	
    	t5Embd.update();
    	
    	for(int i = 0;i<txt_depth;i++) {
    		txt_blocks.get(i).update();
    	}
    	
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

    	t5Embd.saveModel(outputStream);
    	
    	for(int i = 0;i<txt_depth;i++) {
    		txt_blocks.get(i).saveModel(outputStream);
    	}
    	
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
    	
    	t5Embd.loadModel(inputStream);
    	
    	for(int i = 0;i<txt_depth;i++) {
    		txt_blocks.get(i).loadModel(inputStream);
    	}
    	
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
    	
    	t5Embd.accGrad(scale);
    	
    	for(int i = 0;i<txt_depth;i++) {
    		txt_blocks.get(i).accGrad(scale);
    	}
    	
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
    
    public static void loadWeight(Map<String, Object> weightMap, OmegaDiTMainMoudue_Sprint_T5 block, boolean showLayers) {
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

