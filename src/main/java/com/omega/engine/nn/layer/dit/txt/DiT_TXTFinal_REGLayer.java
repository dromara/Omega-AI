package com.omega.engine.nn.layer.dit.txt;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.util.Map;

import com.omega.engine.gpu.BaseKernel;
import com.omega.engine.nn.layer.FullyLayer;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.active.SiLULayer;
import com.omega.engine.nn.layer.normalization.BNType;
import com.omega.engine.nn.layer.normalization.RMSLayer;
import com.omega.engine.nn.network.Network;
import com.omega.engine.tensor.Tensor;
import com.omega.engine.updater.UpdaterFactory;
import com.omega.example.common.ModeLoaderlUtils;

/**
 * DiT FinalLayer
 *
 * @author Administrator
 */
public class DiT_TXTFinal_REGLayer extends Layer {
	
	private int batchSize;
	private int time;
    private int hidden_size = 1;
    private int cls_token_dim;
    private boolean bias = false;
    
    private boolean normParams = true;

    public RMSLayer finalNorm;
    public FullyLayer finalLinear;
    
    public FullyLayer linear_cls;
    
    private SiLULayer m_active;
    public FullyLayer m_linear1;
    public FullyLayer m_linear2;
    
    private Tensor linearInput;
    
    private Tensor dShift;
    private Tensor dScale;
    private Tensor dcls;
    
    private Tensor cls_token;
    private Tensor img;
    
    private BaseKernel baseKernel;

    public DiT_TXTFinal_REGLayer(int patch_size, int hidden_size,int out_channels, int time, int cls_token_dim, boolean bias) {
        this.hidden_size = hidden_size;
        this.bias = bias;
        this.oChannel = 1;
        this.oHeight = 1;
        this.oWidth = patch_size * patch_size * out_channels;
        this.time = time;
        this.cls_token_dim = cls_token_dim;
        this.initLayers();
    }

    public DiT_TXTFinal_REGLayer(int patch_size, int hidden_size,int out_channels, int time, int cls_token_dim, boolean bias, boolean normParams, Network network) {
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
        this.cls_token_dim = cls_token_dim;
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
        this.linear_cls = new FullyLayer(hidden_size, cls_token_dim, bias, network);
        this.linear_cls.weight.clearGPU();
        if(this.linear_cls.bias != null) {
        	this.linear_cls.bias.clearGPU();
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
    
    public void init(Tensor input) {
    	this.number = input.number;
    	this.batchSize = number / (time + 1);
    	if(linearInput == null || linearInput.number != number) {
    		linearInput = Tensor.createGPUTensor(linearInput, number, input.channel, input.height, input.width, true);
    		cls_token = Tensor.createGPUTensor(cls_token, batchSize, 1, 1, cls_token_dim, true);
    		img = Tensor.createGPUTensor(img, batchSize * time, 1, 1, hidden_size, true);
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
    	m_linear1.forward(m_active.getOutput());
    	m_linear2.forward(m_active.getOutput());
    	
    	finalNorm.forward(input);
    	
    	/**
    	 * modulate
    	 * x = x * (1 + scale) + shift
    	 */
    	Tensor_OP().add(m_linear2.getOutput(), 1, m_linear2.getOutput());
    	Tensor_OP().mul(finalNorm.getOutput(), m_linear2.getOutput(), linearInput, batchSize, time + 1, 1, finalNorm.getOutput().width, 1);
    	Tensor_OP().addAxis(linearInput, m_linear1.getOutput(), linearInput, batchSize, time + 1, 1, finalNorm.getOutput().width, 1);

    	Tensor_OP().getByChannel(linearInput, cls_token, new int[] {batchSize, time + 1, 1, hidden_size}, 0, 1);
    	Tensor_OP().getByChannel(linearInput, img, new int[] {batchSize, time + 1, 1, hidden_size}, 1, time);
    	
    	linear_cls.forward(cls_token);
    	
    	finalLinear.forward(img);
    	this.output = finalLinear.getOutput();

    }
    
    public Tensor getCLS() {
    	return linear_cls.getOutput();
    }
    
    public void setDCLS(Tensor dcls) {
    	this.dcls = dcls;
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
    	
//    	finalLinear.diff.showDM("img_ddd");
    	
    	linear_cls.back(dcls);
    	
    	baseKernel.concat_channel_forward(linear_cls.diff, finalLinear.diff, linearInput, batchSize,  1, time, 1, hidden_size);

        Tensor_OP().addAxisBack(dShift, linearInput, batchSize, time + 1, 1, finalNorm.getOutput().width, 1);

    	Tensor_OP().mul_right_back(finalNorm.getOutput(), linearInput, dScale, batchSize, time + 1, 1, finalNorm.getOutput().width, 1);
    	Tensor_OP().mul_left_back(m_linear2.getOutput(), linearInput, linearInput,  batchSize, time + 1, 1, finalNorm.getOutput().width, 1);
    	
//    	linearInput.showDM("linearInput");
    	
    	finalNorm.back(linearInput);

    	m_linear1.back(dShift);
    	m_linear2.back(dScale);
    	
    	Tensor_OP().add(m_linear1.diff, m_linear2.diff, m_linear1.diff);
    	
    	m_active.back(m_linear1.diff);
    	
    	Tensor_OP().add(dtc, m_active.diff, dtc);
    	
        this.diff = finalNorm.diff;
//        diff.showDM("diff");
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
    
    public void back(Tensor delta, Tensor cls_detla, Tensor dtc) {
        // TODO Auto-generated method stub
        this.initBack();
        /**
         * 设置梯度
         */
        this.setDelta(delta);
        /**
         * 计算梯度
         */
        setDCLS(cls_detla);
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
    	linear_cls.update();
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
    	linear_cls.saveModel(outputStream);
    }

    public void loadModel(RandomAccessFile inputStream) throws IOException {
    	finalNorm.loadModel(inputStream, 1, 1, hidden_size, BNType.fully_bn);
    	finalLinear.loadModel(inputStream);
    	m_linear1.loadModel(inputStream);
    	m_linear2.loadModel(inputStream);
    	linear_cls.loadModel(inputStream);
    }

    @Override
    public void accGrad(float scale) {
        // TODO Auto-generated method stub
    	finalNorm.accGrad(scale);
        finalLinear.accGrad(scale);
        m_linear1.accGrad(scale);
    	m_linear2.accGrad(scale);
    	linear_cls.accGrad(scale);
    }
    
    public static void loadWeight(Map<String, Object> weightMap, DiT_TXTFinal_REGLayer block, boolean showLayers) {
        if (showLayers) {
            for (String key : weightMap.keySet()) {
                System.out.println(key);
            }
        }
        
        block.finalNorm.gamma = ModeLoaderlUtils.loadData(block.finalNorm.gamma, weightMap, 1, "norm_final.weight");
        
        ModeLoaderlUtils.loadData(block.finalLinear.weight, weightMap, "linear.weight");
        ModeLoaderlUtils.loadData(block.finalLinear.bias, weightMap, "linear.bias");
        
        ModeLoaderlUtils.loadData(block.m_linear1.weight, weightMap, "adaLN_modulation1.weight");
        ModeLoaderlUtils.loadData(block.m_linear1.bias, weightMap, "adaLN_modulation1.bias");
        
        ModeLoaderlUtils.loadData(block.m_linear2.weight, weightMap, "adaLN_modulation2.weight");
        ModeLoaderlUtils.loadData(block.m_linear2.bias, weightMap, "adaLN_modulation2.bias");
    }
    
    public static void main(String[] args) {
    	
//    	String inputPath = "H:\\model\\dit_final.json";
//    	Map<String, Object> datas = LagJsonReader.readJsonFileSmallWeight(inputPath);
//    	
//        int batchSize = 2;
//        int patch_size = 2;
//        int time = 64;
//        int embedDim = 16;
//        int outChannel = 3;
//        
//        Transformer tf = new Transformer();
//        tf.number = batchSize * time;
//        tf.time = time;
//        
//        float[] data = RandomUtils.order(batchSize * time * embedDim, 0.1f, 0.1f);
//        Tensor input = new Tensor(batchSize * time, 1, 1, embedDim, data, true);
//        
//        float[] cData = RandomUtils.order(batchSize * embedDim, 0.1f, 0.1f); 
//        Tensor cond = new Tensor(batchSize , 1, 1, embedDim, cData, true);
//        
//        int ow = patch_size * patch_size * outChannel;
//        
//        float[] delta_data = RandomUtils.order(batchSize * time * ow, 0.01f, 0.01f);
//        Tensor delta = new Tensor(batchSize * time, 1, 1, ow, delta_data, true);
//        
//        Tensor dcond = new Tensor(batchSize, 1, 1, embedDim, true);
//
//        DiT_TXTFinal_REGLayer finalLayer = new DiT_TXTFinal_REGLayer(patch_size, embedDim, outChannel, time, true, true, tf);
//        
//        loadWeight(datas, finalLayer, true);
//        
//        for (int i = 0; i < 10; i++) {
//            //			input.showDM();
//        	dcond.clearGPU();
//        	finalLayer.forward(input, cond);
//        	finalLayer.getOutput().showShape();
//        	finalLayer.getOutput().showDM();
//        	finalLayer.back(delta, dcond);
//////            //			delta.showDM();
//        	finalLayer.diff.showDM("dx");
//        	dcond.showDM("dcond");
//            //			delta.copyData(tmp);
//        }
    }
}

