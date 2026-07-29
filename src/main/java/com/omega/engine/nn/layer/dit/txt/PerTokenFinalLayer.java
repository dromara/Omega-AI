package com.omega.engine.nn.layer.dit.txt;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.util.Map;

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
 * PerTokenFinalLayer
 *
 * @author Administrator
 */
public class PerTokenFinalLayer extends Layer {

    private int hidden_size = 1;
    private boolean bias = false;
    
    private boolean normParams = true;

    public RMSLayer finalNorm;
    public FullyLayer finalLinear;
    
    private SiLULayer m_active;
    public FullyLayer m_linear1;
    public FullyLayer m_linear2;
    
    private Tensor linearInput;
    
    private Tensor dShift;
    private Tensor dScale;

    public PerTokenFinalLayer(int patch_size, int hidden_size,int out_channels, boolean bias) {
        this.hidden_size = hidden_size;
        this.bias = bias;
        this.oChannel = 1;
        this.oHeight = 1;
        this.oWidth = patch_size * patch_size * out_channels;
        this.initLayers();
    }

    public PerTokenFinalLayer(int patch_size, int hidden_size, int out_channels, boolean bias, boolean normParams, Network network) {
        this.network = network;
        if (this.updater == null) {
            this.setUpdater(UpdaterFactory.create(network));
        }
        this.hidden_size = hidden_size;
        this.bias = bias;
        this.oChannel = 1;
        this.oHeight = 1;
        this.oWidth = patch_size * patch_size * out_channels;
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
        this.m_linear1 = new FullyLayer(hidden_size, hidden_size, bias, network);
        this.m_linear2 = new FullyLayer(hidden_size, hidden_size, bias, network);
        this.m_linear1.weight.clearGPU();
        if(this.m_linear1.bias != null) {
        	this.m_linear1.bias.clearGPU();
        }
        this.m_linear2.weight.clearGPU();
        if(this.m_linear2.bias != null) {
        	this.m_linear2.bias.clearGPU();
        }
    }

    @Override
    public void init() {
        // TODO Auto-generated method stub
        this.number = this.input.number;
    }
    
    public void init(Tensor input) {
    	this.number = input.number;
    	if(linearInput == null || linearInput.number != number) {
    		linearInput = Tensor.createGPUTensor(linearInput, number, input.channel, input.height, input.width, true);
    	}
    }
    
    @Override
    public void initBack() {
        // TODO Auto-generated method stub
    	if(dShift == null || dShift.number != number) {
    		dShift = Tensor.createGPUTensor(dShift, number, 1, 1, hidden_size, true);
    	}
    	if(dScale == null || dScale.number != number) {
    		dScale = Tensor.createGPUTensor(dScale, number, 1, 1, hidden_size, true);
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
    	m_linear1.forward(m_active.getOutput());
    	m_linear2.forward(m_active.getOutput());
    	
    	finalNorm.forward(input);
    	
    	/**
    	 * modulate
    	 * x = x * (1 + scale) + shift
    	 */
    	Tensor_OP().add(m_linear2.getOutput(), 1, m_linear2.getOutput());
    	Tensor_OP().mul(finalNorm.getOutput(), m_linear2.getOutput(), linearInput);
    	Tensor_OP().add(linearInput, m_linear1.getOutput(), linearInput);

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
    	finalLinear.back(this.delta);
//    	finalLinear.diff.showDM("l1");
    	Tensor_OP().copyGPU(finalLinear.diff, dShift);

    	Tensor_OP().mul(finalNorm.getOutput(), finalLinear.diff, dScale);
    	Tensor_OP().mul(m_linear2.getOutput(), finalLinear.diff, linearInput);

    	finalNorm.back(linearInput);

    	m_linear1.back(dShift);
    	m_linear2.back(dScale);
    	
    	Tensor_OP().add(m_linear1.diff, m_linear2.diff, m_linear1.diff);
    	
    	m_active.back(m_linear1.diff);
    	
    	Tensor_OP().add(dtc, m_active.diff, dtc);
    	
        this.diff = finalNorm.diff;
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
    	finalNorm.update();
    	finalLinear.update();
    	m_linear1.update();
    	m_linear2.update();
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
    	m_linear1.saveModel(outputStream);
    	m_linear2.saveModel(outputStream);
    }

    public void loadModel(RandomAccessFile inputStream) throws IOException {
    	finalNorm.loadModel(inputStream, 1, 1, hidden_size, BNType.fully_bn);
    	finalLinear.loadModel(inputStream);
    	m_linear1.loadModel(inputStream);
    	m_linear2.loadModel(inputStream);
    }

    @Override
    public void accGrad(float scale) {
        // TODO Auto-generated method stub
    	finalNorm.accGrad(scale);
        finalLinear.accGrad(scale);
        m_linear1.accGrad(scale);
    	m_linear2.accGrad(scale);
    }
    
    public static void loadWeight(Map<String, Object> weightMap, PerTokenFinalLayer block, boolean showLayers) {
        if (showLayers) {
            for (String key : weightMap.keySet()) {
                System.out.println(key);
            }
        }
        
    }
    
    public static void main(String[] args) {

    }
}

