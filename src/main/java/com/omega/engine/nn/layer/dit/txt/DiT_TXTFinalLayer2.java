package com.omega.engine.nn.layer.dit.txt;

import java.io.IOException;
import java.io.RandomAccessFile;

import com.omega.engine.nn.layer.FullyLayer;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.active.SiLULayer;
import com.omega.engine.nn.layer.normalization.BNType;
import com.omega.engine.nn.layer.normalization.RMSLayer;
import com.omega.engine.nn.network.Network;
import com.omega.engine.tensor.Tensor;
import com.omega.engine.updater.UpdaterFactory;

/**
 * DiT FinalLayer
 *
 * @author Administrator
 */
public class DiT_TXTFinalLayer2 extends Layer {
	
	private int batchSize;
	private int time;
    private int hidden_size = 1;
    private boolean bias = false;
    
    private boolean normParams = true;

    public RMSLayer finalNorm;
    public FullyLayer finalLinear;
    
    private SiLULayer m_active;
    public FullyLayer m_linear;
    
    private Tensor linearInput;
    
    private Tensor dShift;
    private Tensor dScale;
    
    private Tensor m1;
    private Tensor m2;

    public DiT_TXTFinalLayer2(int patch_size, int hidden_size,int out_channels, int time, boolean bias) {
        this.hidden_size = hidden_size;
        this.bias = bias;
        this.oChannel = 1;
        this.oHeight = 1;
        this.oWidth = patch_size * patch_size * out_channels;
        this.time = time;
        this.initLayers();
    }

    public DiT_TXTFinalLayer2(int patch_size, int hidden_size,int out_channels, int time, boolean bias, boolean normParams, Network network) {
        this.network = network;
        if (this.updater == null) {
            this.setUpdater(UpdaterFactory.create(network));
        }
        this.hidden_size = hidden_size;
        this.bias = bias;
        this.oChannel = 1;
        this.oHeight = 1;
        this.oWidth = patch_size * patch_size * out_channels;
        this.time = time;
        this.normParams = normParams;
        this.initLayers();
    }

    public void initLayers() {
    	this.finalNorm = new RMSLayer(1, 1, hidden_size, normParams, BNType.fully_bn, network);
        this.finalLinear = new FullyLayer(hidden_size, oWidth, bias, network);
        this.finalLinear.weight.clearGPU();
        if(this.finalLinear.bias != null) {
        	this.finalLinear.bias.clearGPU();
        }
        this.m_active = new SiLULayer(network);
        this.m_linear = new FullyLayer(hidden_size, hidden_size * 2, bias, network);
    }

    @Override
    public void init() {
        // TODO Auto-generated method stub
        this.number = this.input.number;
    }
    
    public void init(Tensor input) {
    	this.number = input.number;
    	this.batchSize = number / time;
    	if(linearInput == null || linearInput.number != number) {
    		linearInput = Tensor.createGPUTensor(linearInput, number, input.channel, input.height, input.width, true);
    		m1 = Tensor.createGPUTensor(m1, number, 1, 1, hidden_size, true);
    		m2 = Tensor.createGPUTensor(m2, number, 1, 1, hidden_size, true);
    	}
    }
    
    @Override
    public void initBack() {
        // TODO Auto-generated method stub
    	if(dShift == null || dShift.number != batchSize) {
    		dShift = Tensor.createGPUTensor(dShift, batchSize, 1, 1, hidden_size, true);
    	}
    	if(dScale == null || dScale.number != batchSize) {
    		dScale = Tensor.createGPUTensor(dScale, batchSize, 1, 1, hidden_size, true);
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
    
    public void output(Tensor tc) {

    	m_active.forward(tc);
    	m_linear.forward(m_active.getOutput());
    	
    	Tensor_OP().getByChannel(m_linear.getOutput(), m1, new int[] {batchSize, 2, 1, hidden_size}, 0);
    	Tensor_OP().getByChannel(m_linear.getOutput(), m2, new int[] {batchSize, 2, 1, hidden_size}, 1);
    	
    	finalNorm.forward(input);
    	
    	/**
    	 * modulate
    	 * x = x * (1 + scale) + shift
    	 */
    	Tensor_OP().add(m2, 1, m2);
    	Tensor_OP().mul(finalNorm.getOutput(), m2, linearInput, batchSize, time, 1, finalNorm.getOutput().width, 1);
    	Tensor_OP().addAxis(linearInput, m1, linearInput, batchSize, time, 1, finalNorm.getOutput().width, 1);
    	
    	finalLinear.forward(linearInput);
    	this.output = finalLinear.getOutput();

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
    
    public void diff(Tensor dtc) {
        // TODO Auto-generated method stub
    	
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
        this.init(input);
        /**
         * 计算输出
         */
        this.output();
    }
    
    public void forward(Tensor input,Tensor tc) {
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
        this.output(tc);
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
    
    public void back(Tensor delta,Tensor dtc) {
        // TODO Auto-generated method stub
        this.initBack();
        /**
         * 设置梯度
         */
        this.setDelta(delta);
        /**
         * 计算梯度
         */
        this.diff(dtc);
        if (this.network.GRADIENT_CHECK) {
            this.gradientCheck();
        }
    }

    @Override
    public void update() {
        // TODO Auto-generated method stub
//    	finalNorm.update();
//    	finalLinear.update();
//    	m_linear1.update();
//    	m_linear2.update();
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
    	finalNorm.saveModel(outputStream);
    	finalLinear.saveModel(outputStream);
    	m_linear.saveModel(outputStream);
    }

    public void loadModel(RandomAccessFile inputStream) throws IOException {
    	finalNorm.loadModel(inputStream, 1, 1, hidden_size, BNType.fully_bn);
    	finalLinear.loadModel(inputStream);
    	m_linear.loadModel(inputStream);
    }

    @Override
    public void accGrad(float scale) {
        // TODO Auto-generated method stub
    	finalNorm.accGrad(scale);
        finalLinear.accGrad(scale);
        m_linear.accGrad(scale);
    }
    
}

