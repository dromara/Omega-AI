package com.omega.engine.nn.layer.dit.mmdit;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.dit.DiTCaptionEmbeddingLayer;
import com.omega.engine.nn.layer.dit.DiTFinalLayer;
import com.omega.engine.nn.layer.dit.DiTPatchEmbeddingLayer;
import com.omega.engine.nn.layer.dit.DiTTimeEmbeddingLayer;
import com.omega.engine.nn.network.Network;
import com.omega.engine.tensor.Tensor;
import com.omega.engine.updater.UpdaterFactory;

/**
 * DiT_Block
 * @author Administrator
 */
public class MMDiTMoudueRoPE extends Layer {
	
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
    
    public DiTPatchEmbeddingLayer patchEmbd;
    public DiTTimeEmbeddingLayer timeEmbd;
    public DiTCaptionEmbeddingLayer labelEmbd;
    public List<DiTJoinBlockRoPE> blocks;
    public DiTFinalLayer finalLayer;
    
    private Tensor dtc;

    private float y_drop_prob = 0.0f;
    
    public MMDiTMoudueRoPE(int inChannel, int width, int height, int patchSize, int hiddenSize, int headNum, int depth, int timeSteps, int maxContextLen, int textEmbedDim, int mlpRatio, boolean learnSigma, float y_drop_prob, Network network) {
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
        
        blocks = new ArrayList<DiTJoinBlockRoPE>();
         
        for(int i = 0;i<depth;i++) {
        	boolean pre_only = false;
        	if(i == depth - 1) {
        		pre_only = true;
        	}
        	DiTJoinBlockRoPE block = new DiTJoinBlockRoPE(hiddenSize, hiddenSize, mlpRatio, headNum, patchEmbd.oChannel, maxContextLen, false, false, pre_only, network);
	        blocks.add(block);
        }
        int os = inChannel;
        if(learnSigma) {
        	os = inChannel * 2;
        }
        this.oChannel = os;
        finalLayer = new DiTFinalLayer(patchSize, hiddenSize, os, patchEmbd.oChannel, true, network);

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
    
    public void init(Tensor input) {
        // TODO Auto-generated method stub
        this.number = input.number;
        if(this.output == null || this.output.number != number) {
        	output = Tensor.createGPUTensor(output, number, oChannel, oHeight, oWidth, true);
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
    
    public void output(Tensor tc,Tensor text, Tensor cos, Tensor sin) {
    	
    	patchEmbd.forward(input);

    	timeEmbd.forward(tc);
    	
    	labelEmbd.forward(text);

    	Tensor x = patchEmbd.getOutput().view(patchEmbd.getOutput().number * patchEmbd.getOutput().channel, 1, 1, patchEmbd.getOutput().width);
    	
    	Tensor context = labelEmbd.getOutput();
    	
    	for(int i = 0;i<depth;i++) {
    		DiTJoinBlockRoPE block = blocks.get(i);
    		block.forward(x, context, timeEmbd.getOutput(), cos, sin);
    		x = block.getOutput();
    		context = block.context_block.getOutput();
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
    	
    }
    
    public void diff(Tensor cos, Tensor sin) {
    	/**
    	 * unpatchify back
    	 */
    	int h = height/patchSize;
    	int w = width/patchSize;
    	int[] yShape = new int[] {number, oChannel, h, patchSize, w, patchSize};
    	int[] xShape = new int[] {number, h, w, patchSize, patchSize, oChannel};
    	Tensor_OP().permute(delta, finalLayer.getOutput(), yShape, xShape, new int[] {0, 2, 4, 3, 5, 1});
//    	delta.showDMByOffsetRed(0, 10, "delta");
    	finalLayer.back(finalLayer.getOutput(), dtc);

    	Tensor dy = finalLayer.diff;
    	Tensor dcx = null;
//    	dy.showDMByOffsetRed(0, 10, "dy");
     	for(int i = depth - 1;i>=0;i--) {
     		DiTJoinBlockRoPE block = blocks.get(i);
    		block.back(dy, dcx, dtc, cos, sin);
    		dy = block.diff;
    		dcx = block.context_block.diff;
//        	dy.showDMByOffsetRed(0, 10, "dy["+i+"]");
//        	dy.showShape("dy["+i+"]");
//        	dcx.showDMByOffsetRed(0, 10, "dcx["+i+"]");
//        	dcx.showShape("dcx["+i+"]");
    	}
     	
     	labelEmbd.back(dcx);
     	
//     	dtc.showDM("dtc");
     	timeEmbd.back(dtc);

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
    
    public static void loadWeight(Map<String, Object> weightMap, MMDiTMoudueRoPE block, boolean showLayers) {
        if (showLayers) {
            for (String key : weightMap.keySet()) {
                System.out.println(key);
            }
        }
        
    }
    
    public static void main(String[] args) {

    }
}

