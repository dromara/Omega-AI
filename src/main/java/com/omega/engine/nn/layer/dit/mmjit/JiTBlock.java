package com.omega.engine.nn.layer.dit.mmjit;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.util.Map;

import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.dit.modules.DiTAttentionLayer2;
import com.omega.engine.nn.layer.dit.org.DiTSwiGLUFFN;
import com.omega.engine.nn.layer.normalization.BNType;
//import com.omega.engine.nn.layer.normalization.LNLayer;
import com.omega.engine.nn.layer.normalization.RMSLayer;
import com.omega.engine.nn.network.Network;
import com.omega.engine.tensor.Tensor;
import com.omega.engine.updater.UpdaterFactory;

/**
 * JiTBlock
 * @author Administrator
 */
public class JiTBlock extends Layer {
	
    private int embedDim = 0;
    private int headNum;
    private int time;

    private int maxContext;
    
    private int mlpHiddenDim = 1;
    private boolean bias = false;
    private boolean qkNorm = false;
    
    public RMSLayer norm1;
    public DiTAttentionLayer2 attn;

    public RMSLayer norm3;

    public DiTSwiGLUFFN mlp;

    public JiTBlock(int embedDim, int time, int mlpHiddenDim, int headNum, int maxContext, boolean bias, boolean qkNorm, Network network) {
        this.network = network;
        if (this.updater == null) {
            this.setUpdater(UpdaterFactory.create(network));
        }
        this.embedDim = embedDim;
        this.headNum = headNum;
        this.time = time;
        this.mlpHiddenDim = mlpHiddenDim;
        this.bias = bias;
        this.channel = 1;
        this.height = 1;
        this.width = embedDim;
        this.oChannel = 1;
        this.oHeight = 1;
        this.oWidth = embedDim;
        this.qkNorm = qkNorm;
        this.maxContext = maxContext;
        this.initLayers();
    }

    public void initLayers() {
    	
        this.norm1 = new RMSLayer(1, 1, embedDim, true, BNType.fully_bn, network);
        
        this.attn = new DiTAttentionLayer2(embedDim, headNum, time, bias, qkNorm, network);
        this.norm3 = new RMSLayer(1, 1, embedDim, true, BNType.fully_bn, network);
        
        int swiNum = (int)(2.0f/3.0f * mlpHiddenDim);
        this.mlp = new DiTSwiGLUFFN(embedDim, swiNum, embedDim, bias, network);
    }

    @Override
    public void init() {
        // TODO Auto-generated method stub
        this.number = this.input.number;
    }

    public void init(Tensor input) {
        // TODO Auto-generated method stub
    	this.number = input.number;

    }
    
    @Override
    public void initBack() {
        // TODO Auto-generated method stub

    }

    @Override
    public void initParam() {
        // TODO Auto-generated method stub
    }

    @Override
    public void output() {
        // TODO Auto-generated method stub

    }
    
    public void output(Tensor cos,Tensor sin) {
    	
    	/**
    	 *  x1 = x + self.attn(self.norm1(x))
    	 */
    	norm1.forward(input);
    	attn.forward(norm1.getOutput(), cos, sin, maxContext);
    	Tensor_OP().add(input, attn.getOutput(), attn.getOutput());

    	/**
    	 * x3 = x1 + self.mlp(self.norm3(x1))
    	 */
    	norm3.forward(attn.getOutput());
    	mlp.forward(norm3.getOutput());
    	Tensor_OP().add(attn.getOutput(), mlp.getOutput(), mlp.getOutput());
    	
    	this.output = mlp.getOutput();

    }
    
    @Override
    public Tensor getOutput() {
        // TODO Auto-generated method stub
        return output;
    }

    @Override
    public void diff() {
        // TODO Auto-generated method stub
    	
    }

    public void diff(Tensor cos,Tensor sin) {
//    	delta.showDM("x3");
    	/**
    	 * x3 = x2 + self.mlp(self.norm3(x2))
    	 */
    	mlp.back(delta);
    	norm3.back(mlp.diff, norm3.getOutput());
    	Tensor_OP().add(norm3.diff, delta, norm3.diff);

    	/**
    	 *  x1 = x + self.attn(self.norm1(x))
    	 */
    	attn.back(norm3.diff, cos, sin, maxContext);
    	norm1.back(attn.diff, norm1.getOutput());
    	Tensor_OP().add(norm1.diff, norm3.diff, norm1.diff);
    	
    	this.diff = norm1.diff;

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
    public void forward(Tensor input,Tensor cos,Tensor sin) {
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
        this.output(cos, sin);
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
    	norm1.update();

    	attn.update();

    	norm3.update();
    	
    	mlp.update();
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
    	norm1.saveModel(outputStream);

    	attn.saveModel(outputStream);

    	norm3.saveModel(outputStream);
    	mlp.saveModel(outputStream);
    }

    public void loadModel(RandomAccessFile inputStream) throws IOException {
    	norm1.loadModel(inputStream, channel, height, width, BNType.fully_bn);

    	attn.loadModel(inputStream);

    	norm3.loadModel(inputStream, 1, 1, attn.oWidth, BNType.fully_bn);
    	mlp.loadModel(inputStream);
    }

    @Override
    public void accGrad(float scale) {
    	 // TODO Auto-generated method stub
    	norm1.accGrad(scale);

    	attn.accGrad(scale);

    	norm3.accGrad(scale);
    	
    	mlp.accGrad(scale);
    }
    
    public static void loadWeight(Map<String, Object> weightMap, JiTBlock block, boolean showLayers) {
        if (showLayers) {
            for (String key : weightMap.keySet()) {
                System.out.println(key);
            }
        }
       
    }
    
    public static void main(String[] args) {
    	
    }
    
}

