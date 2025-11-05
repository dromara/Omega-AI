package com.omega.engine.nn.layer.dbnet;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.util.ArrayList;
import java.util.List;

import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.network.Network;
import com.omega.engine.tensor.Tensor;

/**
 * MobileNetV3
 *
 * @author Administrator
 */
public class MobileNetV3 extends Layer {

    private boolean bias = false;
    
    private float scale = 0.5f;
    
    private int inplanes = 16;
    
    private String model_type = "large";
    
    private int[][] cfg;
    
    private int cls_ch_squeeze;
    private int cls_ch_expand;
    
    public List<ResidualUnit> stages;
    
    public ConvBNACT conv1;
    
    public ConvBNACT conv2;
    
    public MobileNetV3(String model_type, float scale, int channel, int height, int width, boolean bias, Network network) {
        this.channel = channel;
        this.height = height;
        this.width = width;
        this.bias = bias;
        this.model_type = model_type;
        this.scale = scale;
        modelSetting(model_type);
        this.initLayers();
    }
    
    public void modelSetting(String model_type) {
    	
    	if(model_type.equals("large")) {
    		cfg = new int[][] {
    			new int[] {3, 16, 16, 0, 0, 1},
    			new int[] {3, 64, 24, 0, 0, 2},
    			new int[] {3, 72, 24, 0, 0, 1},
    			new int[] {5, 72, 40, 1, 0, 2},
    			new int[] {5, 120, 40, 1, 0, 1},
    			new int[] {5, 120, 40, 1, 0, 1},
    			new int[] {3, 240, 80, 0, 1, 2},
    			new int[] {3, 200, 80, 0, 1, 1},
    			new int[] {3, 184, 80, 0, 1, 1},
    			new int[] {3, 184, 80, 0, 1, 1},
    			new int[] {3, 480, 112, 1, 1, 1},
    			new int[] {3, 672, 112, 1, 1, 1},
    			new int[] {5, 672, 160, 1, 1, 2},
    			new int[] {5, 960, 160, 1, 1, 1},
    			new int[] {5, 960, 160, 1, 1, 1}
    		};
    		this.cls_ch_squeeze = 960;
    		this.cls_ch_expand = 1280;
    	}else {
    		cfg = new int[][] {
    			new int[] {3, 16, 16, 1, 0, 2},
    			new int[] {3, 72, 24, 0, 0, 2},
    			new int[] {3, 88, 24, 0, 0, 1},
    			new int[] {5, 96, 40, 1, 1, 2},
    			new int[] {5, 240, 40, 1, 1, 1},
    			new int[] {5, 240, 40, 1, 1, 1},
    			new int[] {5, 120, 48, 1, 1, 1},
    			new int[] {5, 144, 48, 1, 1, 1},
    			new int[] {5, 288, 96, 1, 1, 2},
    			new int[] {5, 576, 96, 1, 1, 1},
    			new int[] {5, 576, 96, 1, 1, 1}
    		};
    		this.cls_ch_squeeze = 576;
    		this.cls_ch_expand = 1280;
    	}
    	
    }
    
    public int make_divisible(int v) {
    	int divisor = 8;
    	int new_v = Math.max(divisor, v + divisor/ 2 / (divisor * divisor));
    	if(new_v < 0.9 * v) {
    		new_v += divisor;
    	}
    	return new_v;
    }

    public void initLayers() {

        this.conv1 = new ConvBNACT(channel, make_divisible((int) (inplanes * scale)), height, width, 3, 2, 1, bias, "hard_swish", network);
        int inplanes = make_divisible((int) (this.inplanes * scale));
        
        stages = new ArrayList<ResidualUnit>();
        
        int ih = conv1.oHeight;
        int iw = conv1.oWidth;
        for(int i = 0;i<cfg.length;i++) {
        	int[] once_cfg = cfg[i];
        	int midChannel = make_divisible((int) (once_cfg[1] * scale));
        	int outChannel = make_divisible((int) (once_cfg[2] * scale));
        	String act = "";
        	switch (once_cfg[4]) {
			case 0:
				act = "relu";
				break;
			default:
				act = "hard_swish";
				break;
			}
        	boolean use_se = false;
        	if(once_cfg[3] == 1) {
        		use_se = true;
        	}
        	ResidualUnit unit = new ResidualUnit(inplanes, midChannel, outChannel, ih, iw, once_cfg[0], once_cfg[5], act, use_se, bias, network);
        	stages.add(unit);
        	ih = unit.oHeight;
        	iw = unit.oWidth;
        	inplanes = outChannel;
        }
        
        this.conv2 = new ConvBNACT(inplanes, make_divisible((int) (cls_ch_squeeze * scale)), ih, iw, 1, 1, 0, bias, "hard_swish", network);
        
        this.oChannel = conv2.oChannel;
        this.oHeight = conv2.oHeight;
        this.oWidth = conv2.oWidth;
    }

    @Override
    public void init() {
        // TODO Auto-generated method stub
        this.number = this.input.number;
    }
    
    public void init(Tensor input) {
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
    	conv1.forward(input);
    	Tensor x = conv1.getOutput();
    	for(int i = 0;i<cfg.length;i++) {
    		ResidualUnit unit = stages.get(i);
    		unit.forward(x);
    		x = unit.getOutput();
    	}
    	conv2.forward(x);
    	this.output = conv2.getOutput();
    }
    
    @Override
    public Tensor getOutput() {
        // TODO Auto-generated method stub
        return output;
    }

    @Override
    public void diff() {
        // TODO Auto-generated method stub
    	conv2.back(delta);
    	Tensor d = conv2.diff;
    	for(int i = cfg.length - 1;i>=0;i--) {
    		ResidualUnit unit = stages.get(i);
    		unit.back(d);
    		d = unit.diff;
    	}
    	conv1.back(d);
    	this.diff = conv1.diff;
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
    
    @Override
    public void update() {
        // TODO Auto-generated method stub
    	conv1.update();
    	for(int i = 0;i<cfg.length;i++) {
    		ResidualUnit unit = stages.get(i);
    		unit.update();
    	}
    	conv2.update();
    }

    @Override
    public void showDiff() {
        // TODO Auto-generated method stub
    }

    @Override
    public LayerType getLayerType() {
        // TODO Auto-generated method stub
        return LayerType.mobile_net;
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
    	conv1.saveModel(outputStream);
    	for(int i = 0;i<cfg.length;i++) {
    		ResidualUnit unit = stages.get(i);
    		unit.saveModel(outputStream);
    	}
    	conv2.saveModel(outputStream);
    }

    public void loadModel(RandomAccessFile inputStream) throws IOException {
    	conv1.loadModel(inputStream);
    	for(int i = 0;i<cfg.length;i++) {
    		ResidualUnit unit = stages.get(i);
    		unit.loadModel(inputStream);
    	}
    	conv2.loadModel(inputStream);
    }

    @Override
    public void accGrad(float scale) {
        // TODO Auto-generated method stub
    	conv1.accGrad(scale);
    	for(int i = 0;i<cfg.length;i++) {
    		ResidualUnit unit = stages.get(i);
    		unit.accGrad(scale);
    	}
    	conv2.accGrad(scale);
    }
//    
//    public static void loadWeight(Map<String, Object> weightMap, SEBlock block, boolean showLayers) {
//        if (showLayers) {
//            for (String key : weightMap.keySet()) {
//                System.out.println(key);
//            }
//        }
//        
//        block.finalNorm.gamma = ModeLoaderlUtils.loadData(block.finalNorm.gamma, weightMap, 1, "norm_final.weight");
//        
//        ModeLoaderlUtils.loadData(block.finalLinear.weight, weightMap, "linear.weight");
//        ModeLoaderlUtils.loadData(block.finalLinear.bias, weightMap, "linear.bias");
//        
//        ModeLoaderlUtils.loadData(block.m_linear1.weight, weightMap, "adaLN_modulation1.weight");
//        ModeLoaderlUtils.loadData(block.m_linear1.bias, weightMap, "adaLN_modulation1.bias");
//        
//        ModeLoaderlUtils.loadData(block.m_linear2.weight, weightMap, "adaLN_modulation2.weight");
//        ModeLoaderlUtils.loadData(block.m_linear2.bias, weightMap, "adaLN_modulation2.bias");
//    }
    
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
//        SEBlock finalLayer = new SEBlock(patch_size, embedDim, outChannel, time, true, true, tf);
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

