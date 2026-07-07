package com.omega.engine.nn.layer.jit;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import com.omega.engine.gpu.BaseKernel;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.dit.DiTCaptionEmbeddingLayer;
import com.omega.engine.nn.layer.dit.DiTOrgTimeEmbeddingLayer;
import com.omega.engine.nn.layer.dit.flux.FluxDiTBlock;
import com.omega.engine.nn.layer.dit.flux.REPAMLPLayer;
import com.omega.engine.nn.layer.dit.sprint.FusionLayer2;
import com.omega.engine.nn.layer.dit.txt.DiT_TXTFinalLayer;
import com.omega.engine.nn.network.Network;
import com.omega.engine.nn.network.RunModel;
import com.omega.engine.tensor.Tensor;
import com.omega.engine.updater.UpdaterFactory;

/**
 * MMJiTMainMoudue_REPA
 * @author Administrator
 */
public class JiTMainMoudue_Sprint2 extends Layer {
	
	public int inChannel;
    public int width;
    public int height;
    public int patchSize;
    private int bottleneck_dim;
    private int hiddenSize;
    private int depth;
    private int headNum;
    private int textEmbedDim;
    private int maxContextLen;
    
    private int mlpRatio = 4;
    
    private int projector_dim = 2048;
    private int z_dim = 768;
    
    private int num_f = 2;
    private int num_h = 2;
    private int num_g = 0;
    
    public BottleneckPatchEmbed patchEmbd;
    public DiTOrgTimeEmbeddingLayer timeEmbd;
    public DiTCaptionEmbeddingLayer labelEmbd;
    public List<FluxDiTBlock> encoders;
    public List<FluxDiTBlock> mids;
    public List<FluxDiTBlock> decoders;
    public FusionLayer2 fusion;
    public DiT_TXTFinalLayer finalLayer;
    
    public REPAMLPLayer z_mlp;
    
    private int hw;
    
    private Tensor posEmbd;
    
    private Tensor dencoder;
    
    private float y_drop_prob = 0.0f;
    
    private float path_drop_prob = 0;
    
    private int[] xShape;
    private int[] yShape;
    
    private BaseKernel baseKernel;
    
    public boolean uncond = false;
    
    private int token_t;
    
    private Tensor cat_x;
    private Tensor img_x;
    
    private Tensor z_img_x;
    
    private Tensor dtc;
    private Tensor d_o;
    
    public JiTMainMoudue_Sprint2(int inChannel, int width, int height, int patchSize, int bottleneck_dim, int hiddenSize, int headNum, int z_dim, int depth, int textEmbedDim, int maxContextLen, float y_drop_prob, float path_drop_prob, Network network) {
		this.network = network;
        if (this.updater == null) {
            this.setUpdater(UpdaterFactory.create(network));
        }
        this.y_drop_prob = y_drop_prob;
        this.path_drop_prob = path_drop_prob;
    	this.inChannel = inChannel;
		this.width = width;
		this.height = height;
		this.patchSize = patchSize;
		this.bottleneck_dim = bottleneck_dim;
		this.headNum = headNum;
		this.hiddenSize = hiddenSize;
		this.depth = depth;
		this.textEmbedDim = textEmbedDim;
		this.maxContextLen = maxContextLen;
		this.z_dim = z_dim;
		this.headNum = headNum;
		this.num_g = this.depth - num_f - num_h;
		this.initLayers();
		this.oHeight = height;
		this.oWidth = width;
    }

    public void initLayers() {
    	
    	patchEmbd = new BottleneckPatchEmbed(inChannel, width, bottleneck_dim, hiddenSize, patchSize, true, network);
        
    	hw = patchEmbd.oChannel;

		this.token_t = (int) (hw * (1.0f - 0));
    	
        timeEmbd = new DiTOrgTimeEmbeddingLayer(1000, 256, hiddenSize, true, network);
		
        labelEmbd = new DiTCaptionEmbeddingLayer(textEmbedDim, hiddenSize, maxContextLen, y_drop_prob, true, network);
    	
        encoders = new ArrayList<FluxDiTBlock>();
        mids = new ArrayList<FluxDiTBlock>();
        decoders = new ArrayList<FluxDiTBlock>();
        
        for(int i = 0;i<num_f;i++) {
        	//int embedDim, int time, int mlpHiddenDim, int headNum, int maxContext, boolean bias, boolean qkNorm, Network network
        	FluxDiTBlock block = new FluxDiTBlock(hiddenSize, hiddenSize, patchEmbd.oChannel + maxContextLen, mlpRatio * hiddenSize, headNum, maxContextLen, true, false, network);
	        encoders.add(block);
        }
        
        for(int i = 0;i<num_g;i++) {
        	FluxDiTBlock block = new FluxDiTBlock(hiddenSize, hiddenSize, patchEmbd.oChannel + maxContextLen, mlpRatio * hiddenSize, headNum, maxContextLen, true, false, network);
	        mids.add(block);
        }

        fusion = new FusionLayer2(hiddenSize, hw, token_t, maxContextLen, path_drop_prob, network);
        
        for(int i = 0;i<num_h;i++) {
        	FluxDiTBlock block = new FluxDiTBlock(hiddenSize, hiddenSize, patchEmbd.oChannel + maxContextLen, mlpRatio * hiddenSize, headNum, maxContextLen, true, false, network);
	        decoders.add(block);
        }
        
        this.oChannel = inChannel;

        finalLayer = new DiT_TXTFinalLayer(patchSize, hiddenSize, inChannel, patchEmbd.oChannel, true, true, network);
        
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
    
    public static float[] sincos2d(int embedDim, int grid) {
        int quarter = embedDim / 4;
        int n = grid * grid;

        float[] omega = new float[quarter];

        for (int i = 0; i < quarter; i++) {
            float power = (float) i / quarter;
            omega[i] = (float) (1.0 / Math.pow(10000.0, power));
        }

        float[] result = new float[n * embedDim];

        for (int row = 0; row < grid; row++) {
            for (int col = 0; col < grid; col++) {
                int idx = row * grid + col;
                int base = idx * embedDim;
                for (int d = 0; d < quarter; d++) {
                    float outX = col * omega[d];
                    float outY = row * omega[d];
                    result[base + d] = (float) Math.sin(outX);
                    result[base + quarter + d] = (float) Math.cos(outX);
                    result[base + 2 * quarter + d] = (float) Math.sin(outY);
                    result[base + 3 * quarter + d] = (float) Math.cos(outY);
                }
            }
        }

        return result;
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
        	posEmbd = new Tensor(1, patchEmbd.oChannel, 1, hiddenSize, sincos2d(hiddenSize, width/patchSize), true);
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
    		dencoder = Tensor.createGPUTensor(dencoder, number * (maxContextLen + hw), 1, 1, hiddenSize, true);
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
    
    public void output(Tensor label) {
    	
    }
    
    public void output(Tensor tc, Tensor label, Tensor cos,Tensor sin) {

    	patchEmbd.forward(input);

    	Tensor_OP().addAxis(patchEmbd.getOutput(), posEmbd, patchEmbd.getOutput(), posEmbd.channel * posEmbd.width);

    	timeEmbd.forward(tc);
    	
    	labelEmbd.forward(label);
    	
    	Tensor x = patchEmbd.getOutput().view(patchEmbd.getOutput().number * patchEmbd.getOutput().channel, 1, 1, patchEmbd.getOutput().width);
    	
    	Tensor cond = labelEmbd.getOutput();
    	
    	Tensor t = timeEmbd.getOutput();
    	
     	baseKernel.concat_channel_forward(cond, x, cat_x, input.number, maxContextLen, hw, 1, patchEmbd.getOutput().width);
    	
    	Tensor e_x = cat_x;
    	for(int i = 0;i<num_f;i++) {
    		FluxDiTBlock block = encoders.get(i);
    		block.forward(e_x, t, cos, sin);
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
		 * mids
		 */
		Tensor h_x = e_x;
		if(!uncond) {
			/**
			 * mids
			 */
			for(int i = 0;i<num_g;i++) {
				FluxDiTBlock block = mids.get(i);
				block.forward(h_x, t, cos, sin);
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
		
		Tensor d_x = fusion.getOutput();
		for(int i = 0;i<num_h;i++) {
			FluxDiTBlock block = decoders.get(i);
    		block.forward(d_x, t, cos, sin);
    		d_x = block.getOutput();
    	}
		
    	Tensor_OP().getByChannel(d_x, img_x, new int[] {input.number, maxContextLen + hw, 1, patchEmbd.getOutput().width}, maxContextLen, hw);

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

    }
    
    public void diff(Tensor cos, Tensor sin) {
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

    	for(int i = num_h - 1;i>=0;i--) {
    		FluxDiTBlock block = decoders.get(i);
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
			FluxDiTBlock block = mids.get(i);
    		block.back(dh, dtc, cos, sin);
    		dh = block.diff;
    	}
    	
		Tensor de = dh;

		/**
		 * repa backward
		 */
		Tensor_OP().getByChannel_add_back(de, z_mlp.diff, new int[] {input.number, maxContextLen + hw, 1, patchEmbd.getOutput().width}, maxContextLen, hw);
    	Tensor_OP().add(dencoder, de, de);
    	
    	for(int i = num_f - 1;i>=0;i--) {
    		FluxDiTBlock block = encoders.get(i);
    		block.back(de, dtc, cos, sin);
    		de = block.diff;
    	}
		
    	baseKernel.concat_channel_backward(de, labelEmbd.getOutput(), img_x, input.number, maxContextLen, hw, 1, patchEmbd.getOutput().width);
    	
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

    }
    
    /**
     * 
     * @param input
     * @param tc time cond
     * @param text
     */
    public void forward(Tensor input,Tensor tc,Tensor text) {
        // TODO Auto-generated method stub

    }
    
    /**
     * 
     * @param input
     * @param tc time cond
     * @param text
     */
    public void forward(Tensor input, Tensor t, Tensor text, Tensor cos, Tensor sin) {
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
        this.output(t, text, cos, sin);
    }

    @Override
    public void back(Tensor delta) {
        // TODO Auto-generated method stub

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
    
    public static void loadWeight(Map<String, Object> weightMap, JiTMainMoudue_Sprint2 block, boolean showLayers) {
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

