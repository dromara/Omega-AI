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
import com.omega.engine.nn.layer.dit.DiTFullTimeEmbeddingLayer;
import com.omega.engine.nn.layer.dit.DiTPatchEmbeddingLayer;
import com.omega.engine.nn.layer.dit.flux.FluxDiTBlockFullTime;
import com.omega.engine.nn.layer.dit.flux.SimpleHeadMLPLayer;
import com.omega.engine.nn.layer.dit.kernel.TokenDropKernel;
import com.omega.engine.nn.layer.dit.txt.PerTokenFinalLayer;
import com.omega.engine.nn.network.Network;
import com.omega.engine.nn.network.RunModel;
import com.omega.engine.tensor.Tensor;
import com.omega.engine.updater.UpdaterFactory;

/**
 * DiT_Block with self flow
 * @author Administrator
 */
public class OmegaDiTMainMoudue_Self_Flow extends Layer {
	
	public boolean teacher = false;
	
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

    private int projector_dim = 2048;
    
    public DiTPatchEmbeddingLayer patchEmbd;
    public DiTFullTimeEmbeddingLayer timeEmbd;
    public DiTCaptionEmbeddingLayer labelEmbd;
    public List<FluxDiTBlockFullTime> encoders;
    public List<FluxDiTBlockFullTime> mids;
    public FusionLayer2 fusion;
    public List<FluxDiTBlockFullTime> decoders;
    public PerTokenFinalLayer finalLayer;
    
    public SimpleHeadMLPLayer z_mlp;
    
    public int hw;
    
    private Tensor posEmbd;
    
    private Tensor cat_x;
    private Tensor img_x;
    private Tensor img_t;
    
    public Tensor z_img_x;
    
    private Tensor d_o;
    
    private Tensor dtc;
    private Tensor dimg_tc;
    private Tensor dencoder;
    private Tensor drop_delta;
    
    private float y_drop_prob = 0.0f;
    
    private float path_drop_prob = 0.0f;

    private int[] xShape;
    private int[] yShape;
    
    private BaseKernel baseKernel;
    private TokenDropKernel tokenDropKernel;
    
    private List<Integer> ids;
    
    public boolean uncond = false;
    
    public OmegaDiTMainMoudue_Self_Flow(int inChannel, int width, int height, int patchSize, int hiddenSize, int headNum, int depth, int timeSteps, int textEmbedDim, int maxContextLen, int mlpRatio, float y_drop_prob, float path_drop_prob, Network network) {
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
		this.path_drop_prob = path_drop_prob;
		this.headNum = headNum;
		this.initLayers();
		this.oHeight = height;
		this.oWidth = width;
    }

    public void initLayers() {
    	
    	patchEmbd = new DiTPatchEmbeddingLayer(inChannel, width, hiddenSize, patchSize, true, network);
        
    	hw = patchEmbd.oChannel;

        timeEmbd = new DiTFullTimeEmbeddingLayer(timeSteps, 256, hiddenSize, true, network);

        labelEmbd = new DiTCaptionEmbeddingLayer(textEmbedDim, hiddenSize, maxContextLen, y_drop_prob, true, network);
        
        encoders = new ArrayList<FluxDiTBlockFullTime>();
        mids = new ArrayList<FluxDiTBlockFullTime>();
        decoders = new ArrayList<FluxDiTBlockFullTime>();
        
        for(int i = 0;i<num_f;i++) {
        	FluxDiTBlockFullTime block = new FluxDiTBlockFullTime(hiddenSize, hiddenSize, patchEmbd.oChannel + maxContextLen, mlpRatio * hiddenSize, headNum, maxContextLen, true, false, network);
        	encoders.add(block);
        }
        
        for(int i = 0;i<num_g;i++) {
        	FluxDiTBlockFullTime block = new FluxDiTBlockFullTime(hiddenSize, hiddenSize, patchEmbd.oChannel + maxContextLen, mlpRatio * hiddenSize, headNum, maxContextLen, true, false, network);
        	mids.add(block);
        }
        
        fusion = new FusionLayer2(hiddenSize, hw, patchEmbd.oChannel, maxContextLen, path_drop_prob, network);
        
        for(int i = 0;i<num_h;i++) {
        	FluxDiTBlockFullTime block = new FluxDiTBlockFullTime(hiddenSize, hiddenSize, patchEmbd.oChannel + maxContextLen, mlpRatio * hiddenSize, headNum, maxContextLen, true, false, network);
        	decoders.add(block);
        }
        
        this.oChannel = inChannel;

        finalLayer = new PerTokenFinalLayer(patchSize, hiddenSize, inChannel, true, true, network);
        
        z_mlp = new SimpleHeadMLPLayer(hiddenSize, projector_dim, hiddenSize, true, network);
        
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
//    	idskeep.showDM("idskeep");
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
        	img_t = Tensor.createGPUTensor(img_t, number * patchEmbd.oChannel, 1, 1, hiddenSize, true);
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
    		d_o = Tensor.createGPUTensor(d_o, input.number * (maxContextLen + hw), 1, 1, patchEmbd.getOutput().width, true);
    		dtc = Tensor.createGPUTensor(dtc, timeEmbd.getOutput().shape(), true);
    		dimg_tc = Tensor.createGPUTensor(dimg_tc, number * patchEmbd.oChannel, 1, 1, hiddenSize, true);
    		dencoder = Tensor.createGPUTensor(dencoder, number * (maxContextLen + hw), 1, 1, hiddenSize, true);
    		drop_delta = Tensor.createGPUTensor(drop_delta, number * (maxContextLen + hw), 1, 1, hiddenSize, true);
    	}else {
    		dimg_tc.clearGPU();
    		dtc.clearGPU();
    		d_o.clearGPU();
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
    
    public void output_teacher(Tensor tc, Tensor label, Tensor cos, Tensor sin) {
    	
    	patchEmbd.forward(input);

    	Tensor_OP().addAxis(patchEmbd.getOutput(), posEmbd, patchEmbd.getOutput(), posEmbd.channel * posEmbd.width);
    	
    	timeEmbd.forward(tc);
    	
    	labelEmbd.forward_eval(label);
    	
    	Tensor x = patchEmbd.getOutput().view(patchEmbd.getOutput().number * patchEmbd.getOutput().channel, 1, 1, patchEmbd.getOutput().width);
    	
    	Tensor t = timeEmbd.getOutput();
    	
    	Tensor cond = labelEmbd.getOutput();
    	
     	baseKernel.concat_channel_forward(cond, x, cat_x, input.number, maxContextLen, hw, 1, patchEmbd.getOutput().width);
     	
    	/**
    	 * encoder
    	 */
    	Tensor e_x = cat_x;
    	for(int i = 0;i<num_f;i++) {
    		FluxDiTBlockFullTime block = encoders.get(i);
    		block.forward(e_x, t, cos, sin);
    		e_x = block.getOutput();
    	}
		
		/**
		 * sprint
		 */
		Tensor h_x = e_x;
		/**
		 * mids
		 */
		for(int i = 0;i<num_g;i++) {
			FluxDiTBlockFullTime block = mids.get(i);
 			block.forward(h_x, t, cos, sin);
    		h_x = block.getOutput();
    	}
		
		z_img_x = h_x;
		
    }
    
    public void output(Tensor tc, Tensor label, Tensor cos, Tensor sin) {
    	
    	patchEmbd.forward(input);

    	Tensor_OP().addAxis(patchEmbd.getOutput(), posEmbd, patchEmbd.getOutput(), posEmbd.channel * posEmbd.width);
    	
    	timeEmbd.forward(tc);
    	
    	labelEmbd.forward(label);
    	
    	Tensor x = patchEmbd.getOutput().view(patchEmbd.getOutput().number * patchEmbd.getOutput().channel, 1, 1, patchEmbd.getOutput().width);
    	
    	Tensor t = timeEmbd.getOutput();
    	
    	Tensor cond = labelEmbd.getOutput();
    	
     	baseKernel.concat_channel_forward(cond, x, cat_x, input.number, maxContextLen, hw, 1, patchEmbd.getOutput().width);
     	
    	/**
    	 * encoder
    	 */
    	Tensor e_x = cat_x;
    	for(int i = 0;i<num_f;i++) {
    		FluxDiTBlockFullTime block = encoders.get(i);
    		block.forward(e_x, t, cos, sin);
    		e_x = block.getOutput();
    	}

    	/**
    	 * repa
    	 */
		if(network.RUN_MODEL == RunModel.TRAIN && !teacher) {
			z_mlp.forward(e_x);
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
				FluxDiTBlockFullTime block = mids.get(i);
     			block.forward(h_x, t, cos, sin);
	    		h_x = block.getOutput();
	    	}
		}else {
			h_x = mids.get(num_g - 1).getOutput();
		}

    	/**
    	 * teacher repa
    	 */
		if(network.RUN_MODEL == RunModel.TRAIN && teacher) {
			z_img_x = h_x;
			return;
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
    		FluxDiTBlockFullTime block = decoders.get(i);
    		block.forward(d_x, t, cos, sin);
    		d_x = block.getOutput();
    	}
    	
    	Tensor_OP().getByChannel(d_x, img_x, new int[] {input.number, maxContextLen + hw, 1, patchEmbd.getOutput().width}, maxContextLen, hw);
    	Tensor_OP().getByChannel(t, img_t, new int[] {input.number, maxContextLen + hw, 1, hiddenSize}, maxContextLen, hw);
//    	img_x.showShape("img_x");
    	finalLayer.forward(img_x, img_t);
//    	finalLayer.getOutput().showShape("finalLayer");
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
    	
    	finalLayer.back(finalLayer.getOutput(), dimg_tc);

    	Tensor dy = d_o;
    	dy.clearGPU();

    	Tensor_OP().getByChannel_back(dy, finalLayer.diff, new int[] {input.number, maxContextLen + hw, 1, patchEmbd.getOutput().width}, maxContextLen, hw);
    	Tensor_OP().getByChannel_back(dtc, dimg_tc, new int[] {input.number, maxContextLen + hw, 1, hiddenSize}, maxContextLen, hw);
    	
    	/**
    	 * decoder backward
    	 */
    	for(int i = num_h - 1;i>=0;i--) {
    		FluxDiTBlockFullTime block = decoders.get(i);
    		block.back(dy, dtc, cos, sin);
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
			FluxDiTBlockFullTime block = mids.get(i);
 			block.back(dh, dtc, cos, sin);
    		dh = block.diff;
    	}
		
		/**
		 * sprint backward
		 */
		Tensor de = dh;
		
		/**
		 * repa backward
		 */
		Tensor_OP().add(de, z_mlp.diff, de);
    	Tensor_OP().add(dencoder, de, de);
 
		/**
		 * encoder backward
		 */
    	for(int i = num_f - 1;i>=0;i--) {
    		FluxDiTBlockFullTime block = encoders.get(i);
    		block.back(de, dtc, cos, sin);
    		de = block.diff;
    	}

    	baseKernel.concat_channel_backward(de, labelEmbd.getOutput(), img_x, input.number, maxContextLen, hw, 1, patchEmbd.getOutput().width);

//    	img_x.showDM("d_img_x");
    	
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
     * @param input
     * @param tc time cond
     * @param text
     */
    public void forward(Tensor input, Tensor tc, Tensor text, Tensor cos, Tensor sin) {
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
    
    /**
     * @param input
     * @param tc time cond
     * @param text
     */
    public void forward_teacher(Tensor input, Tensor tc, Tensor text, Tensor cos, Tensor sin) {
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
        this.output_teacher(tc, text, cos, sin);
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
    
    public static void loadWeight(Map<String, Object> weightMap, OmegaDiTMainMoudue_Self_Flow block, boolean showLayers) {
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

