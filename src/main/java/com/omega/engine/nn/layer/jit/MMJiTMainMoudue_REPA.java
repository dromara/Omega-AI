package com.omega.engine.nn.layer.jit;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import com.omega.engine.gpu.BaseKernel;
import com.omega.engine.nn.layer.FullyLayer;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.dit.flux.REPAMLPLayer;
import com.omega.engine.nn.layer.dit.mmjit.JiTCaptionEmbeddingLayer;
import com.omega.engine.nn.layer.dit.mmjit.MMJiTBlock;
import com.omega.engine.nn.layer.dit.mmjit.PlainTextBlock;
import com.omega.engine.nn.layer.normalization.BNType;
import com.omega.engine.nn.layer.normalization.RMSLayer;
import com.omega.engine.nn.network.Network;
import com.omega.engine.nn.network.RunModel;
import com.omega.engine.tensor.Tensor;
import com.omega.engine.updater.UpdaterFactory;

/**
 * MMJiTMainMoudue_REPA
 * @author Administrator
 */
public class MMJiTMainMoudue_REPA extends Layer {
	
	public int inChannel;
    public int width;
    public int height;
    public int patchSize;
    private int bottleneck_dim;
    private int hiddenSize;
    private int txt_depth;
    private int depth;
    private int headNum;
    private int textEmbedDim;
    private int maxContextLen;
    
    private int align‌_idx = 1;
    private int projector_dim = 2048;
    private int z_dim = 768;
    
    public BottleneckPatchEmbed patchEmbd;
    public JiTCaptionEmbeddingLayer labelEmbd;
//    public DiTCaptionEmbeddingLayer labelEmbd;
    public List<PlainTextBlock> txt_blocks;
    public List<MMJiTBlock> blocks;
    public RMSLayer finalNorm;
    public FullyLayer finalLayer;
    
    public REPAMLPLayer z_mlp;
    
    private Tensor posEmbd;
    
    private Tensor tmp_cond;

    private float y_drop_prob = 0.0f;
    
    private int[] xShape;
    private int[] yShape;
    
    private BaseKernel baseKernel;
    
    public MMJiTMainMoudue_REPA(int inChannel, int width, int height, int patchSize, int bottleneck_dim, int hiddenSize, int headNum, int z_dim, int txt_depth, int depth, int textEmbedDim, int maxContextLen, float y_drop_prob, Network network) {
		this.network = network;
        if (this.updater == null) {
            this.setUpdater(UpdaterFactory.create(network));
        }
        this.y_drop_prob = y_drop_prob;
    	this.inChannel = inChannel;
		this.width = width;
		this.height = height;
		this.patchSize = patchSize;
		this.bottleneck_dim = bottleneck_dim;
		this.headNum = headNum;
		this.hiddenSize = hiddenSize;
		this.txt_depth = txt_depth;
		this.depth = depth;
		this.textEmbedDim = textEmbedDim;
		this.maxContextLen = maxContextLen;
		this.z_dim = z_dim;
		this.headNum = headNum;
		this.initLayers();
		this.oHeight = height;
		this.oWidth = width;
    }

    public void initLayers() {
    	
    	patchEmbd = new BottleneckPatchEmbed(inChannel, width, bottleneck_dim, hiddenSize, patchSize, true, network);
        
    	labelEmbd = new JiTCaptionEmbeddingLayer(textEmbedDim, hiddenSize, maxContextLen, y_drop_prob, false, network);

//        labelEmbd = new DiTCaptionEmbeddingLayer(textEmbedDim, hiddenSize, maxContextLen, y_drop_prob, true, network);
    	
        txt_blocks = new ArrayList<PlainTextBlock>();
        
        blocks = new ArrayList<MMJiTBlock>();
         
        for(int i = 0;i<txt_depth;i++) {
        	PlainTextBlock block = new PlainTextBlock(hiddenSize, maxContextLen, headNum, true, false, network);
        	txt_blocks.add(block);
        }
        
        for(int i = 0;i<depth;i++) {
        	MMJiTBlock block = new MMJiTBlock(hiddenSize, headNum, patchEmbd.oChannel, maxContextLen, true, false, true, network);
	        blocks.add(block);
        }
        
        z_mlp = new REPAMLPLayer(hiddenSize, projector_dim, z_dim, true, network);
        
        this.oChannel = inChannel;
        finalNorm = new RMSLayer(1, 1, hiddenSize, true, BNType.fully_bn, network);
        finalLayer = new FullyLayer(hiddenSize, patchSize * patchSize * oChannel, true, network);
        this.finalLayer.weight.clearGPU();
        if(this.finalLayer.bias != null) {
        	this.finalLayer.bias.clearGPU();
        }
        
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
    	if(tmp_cond == null || tmp_cond.number != number * maxContextLen) {
    		tmp_cond = Tensor.createGPUTensor(tmp_cond, number * maxContextLen, 1, 1, hiddenSize, true);
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
    
    public void output(Tensor label, Tensor cos1d,Tensor sin1d, Tensor cos2d, Tensor sin2d) {

    	patchEmbd.forward(input);

    	Tensor_OP().addAxis(patchEmbd.getOutput(), posEmbd, patchEmbd.getOutput(), posEmbd.channel * posEmbd.width);

    	labelEmbd.forward(label);
    	
    	Tensor x = patchEmbd.getOutput().view(patchEmbd.getOutput().number * patchEmbd.getOutput().channel, 1, 1, patchEmbd.getOutput().width);
    	
    	Tensor cond = labelEmbd.getOutput();
    	
    	Tensor bc = cond;
    	for(int i = 0;i<txt_depth;i++) {
    		PlainTextBlock block = txt_blocks.get(i);
    		block.forward(bc, cos1d, sin1d);
    		bc = block.getOutput();
    	}
    	
    	Tensor bx = x;
    	for(int i = 0;i<depth;i++) {
    		MMJiTBlock block = blocks.get(i);
    		block.forward(bx, bc, cos1d, sin1d, cos2d, sin2d);
    		bx = block.getOutput();
    		bc = block.context_block.getOutput();
    		
    		if(network.RUN_MODEL == RunModel.TRAIN && i == align‌_idx) {
    			z_mlp.forward(bx);
    		}
    		
    	}

    	finalNorm.forward(bx);
    	finalLayer.forward(finalNorm.getOutput());
    	
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
//    	int h = height/patchSize;
//    	int w = width/patchSize;
//    	int[] yShape = new int[] {number, oChannel, h, patchSize, w, patchSize};
//    	int[] xShape = new int[] {number, h, w, patchSize, patchSize, oChannel};
    	Tensor_OP().permute(delta, finalLayer.getOutput(), yShape, xShape, new int[] {0, 2, 4, 3, 5, 1});
    	
    	finalLayer.back(finalLayer.getOutput());
    	finalNorm.back(finalLayer.diff);
    	
    	Tensor dy = finalNorm.diff;
    	Tensor dc = tmp_cond;
    	dc.clearGPU();
     	for(int i = depth - 1;i>=0;i--) {
     		MMJiTBlock block = blocks.get(i);
     		
     		if(i == align‌_idx) {
     			Tensor_OP().add(dy, z_mlp.diff, dy);
     		}
     		
    		block.back(dy, dc, cos1d, sin1d, cos2d, sin2d);
    		dy = block.diff;
    		dc = block.context_block.diff;
    	}
     	
     	for(int i = txt_depth - 1;i>=0;i--) {
     		PlainTextBlock block = txt_blocks.get(i);
    		block.back(dc, cos1d, sin1d);
    		dc = block.diff;
    	}
     	
     	labelEmbd.back(dc);
     	
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
    public void forward(Tensor input,Tensor text, Tensor cos1d, Tensor sin1d, Tensor cos2d, Tensor sin2d) {
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
        this.output(text, cos1d, sin1d, cos2d, sin2d);
    }

    @Override
    public void back(Tensor delta) {
        // TODO Auto-generated method stub

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

    	labelEmbd.update();
    	
    	for(int i = 0;i<txt_depth;i++) {
    		txt_blocks.get(i).update();
    	}
    	
    	for(int i = 0;i<depth;i++) {
    		blocks.get(i).update();
    	}
    	
    	z_mlp.update();
    	
    	finalNorm.update();
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

    	labelEmbd.saveModel(outputStream);
    	
    	for(int i = 0;i<txt_depth;i++) {
    		txt_blocks.get(i).saveModel(outputStream);
    	}
    	
    	for(int i = 0;i<depth;i++) {
    		blocks.get(i).saveModel(outputStream);
    	}
    	
    	z_mlp.saveModel(outputStream);
    	
    	finalNorm.saveModel(outputStream);
    	finalLayer.saveModel(outputStream);
    }

    public void loadModel(RandomAccessFile inputStream) throws IOException {
    	patchEmbd.loadModel(inputStream);

    	labelEmbd.loadModel(inputStream);
    	
    	for(int i = 0;i<txt_depth;i++) {
    		txt_blocks.get(i).loadModel(inputStream);
    	}
    	
    	for(int i = 0;i<depth;i++) {
    		blocks.get(i).loadModel(inputStream);
    	}
    	
    	z_mlp.loadModel(inputStream);
    	
    	finalNorm.loadModel(inputStream, 1, 1, hiddenSize, BNType.fully_bn);
    	finalLayer.loadModel(inputStream);
    }

    @Override
    public void accGrad(float scale) {
        // TODO Auto-generated method stub
    	patchEmbd.accGrad(scale);

    	labelEmbd.accGrad(scale);
    	
    	for(int i = 0;i<txt_depth;i++) {
    		txt_blocks.get(i).accGrad(scale);
    	}
    	
    	for(int i = 0;i<depth;i++) {
    		blocks.get(i).accGrad(scale);
    	}
    	
    	z_mlp.accGrad(scale);
    	
    	finalNorm.accGrad(scale);
    	finalLayer.accGrad(scale);
    }
    
    public static void loadWeight(Map<String, Object> weightMap, MMJiTMainMoudue_REPA block, boolean showLayers) {
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

