package com.omega.engine.nn.layer.dit.deco;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.util.Map;

import com.omega.common.utils.RandomUtils;
import com.omega.engine.nn.layer.FullyLayer;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.active.SiLULayer;
import com.omega.engine.nn.layer.normalization.BNType;
import com.omega.engine.nn.layer.normalization.LNLayer;
import com.omega.engine.nn.network.Network;
import com.omega.engine.nn.network.Transformer;
import com.omega.engine.tensor.Tensor;
import com.omega.engine.updater.UpdaterFactory;
import com.omega.example.common.ModeLoaderlUtils;
import com.omega.example.transformer.utils.LagJsonReader;

/**
 * ResBlock
 * @author Administrator
 */
public class ResBlock extends Layer {
	
    private int channels = 0;

    private boolean bias = false;
    
    private SiLULayer modulationAct;
    public FullyLayer adaLN_modulation;

    public LNLayer in_ln;
    
    public FullyLayer mlp_liner1;
    public SiLULayer mlp_act;
    public FullyLayer mlp_liner2;

    private Tensor mlpInput;
    
    public Tensor shift_mlp;
    public Tensor scale_mlp;
    public Tensor gate_mlp;
    
    private int[] shape;
    
    public ResBlock(int channels, boolean bias, Network network) {
        this.network = network;
        if (this.updater == null) {
            this.setUpdater(UpdaterFactory.create(network));
        }
        this.channels = channels;
        this.bias = bias;
        this.channel = 1;
        this.height = 1;
        this.width = channels;
        this.oChannel = 1;
        this.oHeight = 1;
        this.oWidth = channels;
        this.initLayers();
    }

    public void initLayers() {
    	
        this.in_ln = new LNLayer(1, 1, channels, true, BNType.fully_bn, network);
        
        this.modulationAct = new SiLULayer(network);
        
        this.adaLN_modulation = new FullyLayer(channels, channels * 3, bias, network);
        adaLN_modulation.weight.clearGPU();
        if(adaLN_modulation.bias != null) {
        	adaLN_modulation.bias.clearGPU();
        }
        
        mlp_liner1 = new FullyLayer(channels, channels, bias, network);
        RandomUtils.xavier_uniform(this.mlp_liner1.weight, 1, channels, channels);
        if(this.mlp_liner1.bias != null) {
        	this.mlp_liner1.bias.clearGPU();
        }
        
        mlp_liner2 = new FullyLayer(channels, channels, bias, network);
        RandomUtils.xavier_uniform(this.mlp_liner2.weight, 1, channels, channels);
        if(this.mlp_liner2.bias != null) {
        	this.mlp_liner2.bias.clearGPU();
        }
        
        mlp_act = new SiLULayer(mlp_liner1);
    }

    @Override
    public void init() {
        // TODO Auto-generated method stub
        this.number = this.input.number;
    }

    public void init(Tensor input) {
        // TODO Auto-generated method stub
        this.number = input.number;
        if(shape == null) {
        	shape= new int[] {number, 3, 1, channels};
        }
        if(shift_mlp == null || shift_mlp.number != number) {
    		shift_mlp = Tensor.createGPUTensor(this.shift_mlp, number, 1, 1, channels, true);
        	scale_mlp = Tensor.createGPUTensor(this.scale_mlp, number, 1, 1, channels, true);
        	gate_mlp = Tensor.createGPUTensor(this.gate_mlp, number, 1, 1, channels, true);
        }
        if(mlpInput == null || mlpInput.number != number) {
        	mlpInput = Tensor.createGPUTensor(mlpInput, input.number, input.channel, input.height, input.width, true);
        }
        if(output == null || output.number != number) {
        	output = Tensor.createGPUTensor(output, input.number, oChannel, oHeight, oWidth, true);
        }

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
    
    public void modulate(Tensor x,Tensor shift,Tensor scale,Tensor output) {
    	/**
    	 * modulate
    	 * x = x * (1 + scale) + shift
    	 */
    	Tensor_OP().add(scale, 1.0f, scale);
    	Tensor_OP().mul(x, scale, output);
    	Tensor_OP().add(output, shift, output);
    }
    
    public void output(Tensor tc) {
    	
    	/**
    	 * shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(3, dim=1)
    	 */
    	modulationAct.forward(tc);
    	adaLN_modulation.forward(modulationAct.getOutput());
    	
    	Tensor_OP().getByChannel(adaLN_modulation.getOutput(), shift_mlp, shape, 0);
    	Tensor_OP().getByChannel(adaLN_modulation.getOutput(), scale_mlp, shape, 1);
    	Tensor_OP().getByChannel(adaLN_modulation.getOutput(), gate_mlp, shape, 2);
    	
    	/**
    	 *  h = modulate(self.in_ln(x), shift_mlp, scale_mlp)
    	 *  x + gate_mlp * self.mlp(h)
    	 */
    	in_ln.forward_llmc(input);
    	modulate(in_ln.getOutput(), shift_mlp, scale_mlp, mlpInput);

    	mlp_liner1.forward(mlpInput);
    	mlp_act.forward(mlp_liner1.getOutput());
    	mlp_liner2.forward(mlp_act.getOutput());
    	
    	Tensor_OP().mul(mlp_liner2.getOutput(), gate_mlp, output);
    	Tensor_OP().add(input, output, output);
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

    public void modulate_back(Tensor dShift,Tensor dScale,Tensor dx,Tensor x,Tensor scale,Tensor delta) {
    	delta.copyGPU(dShift);
    	Tensor_OP().mul(x, delta, dScale);
    	Tensor_OP().mul(scale, delta, dx);
    }
    
    public void diff(Tensor dtc) {
    	Tensor_OP().mul(gate_mlp, delta, output);
    	mlp_liner2.back(output);
    	mlp_act.back(mlp_liner2.diff);
    	mlp_liner1.back(mlp_act.diff);
    	Tensor_OP().mul(mlp_liner2.getOutput(), delta, gate_mlp);
    	Tensor_OP().setByChannel(adaLN_modulation.getOutput(), gate_mlp, shape, 2);
    	
    	Tensor dShift = shift_mlp;
    	Tensor dScale = gate_mlp;
    	Tensor x = in_ln.getOutput();
    	Tensor scale = scale_mlp;
    	modulate_back(dShift, dScale, output, x, scale, mlp_liner1.diff);
    	Tensor_OP().setByChannel(adaLN_modulation.getOutput(), dShift, shape, 0);
    	Tensor_OP().setByChannel(adaLN_modulation.getOutput(), dScale, shape, 1);
    	
    	in_ln.back(output);

    	Tensor_OP().add(in_ln.diff, delta, in_ln.diff);
    	
    	adaLN_modulation.back(adaLN_modulation.getOutput());

        modulationAct.back(adaLN_modulation.diff);
        
    	Tensor_OP().add(dtc, modulationAct.diff, dtc);
    	
    	this.diff = in_ln.diff;
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
    	in_ln.update();
    	adaLN_modulation.update();
    	
    	mlp_liner1.update();
    	mlp_liner2.update();
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
    	in_ln.saveModel(outputStream);
    	adaLN_modulation.saveModel(outputStream);

    	mlp_liner1.saveModel(outputStream);
    	mlp_liner2.saveModel(outputStream);
    }

    public void loadModel(RandomAccessFile inputStream) throws IOException {
    	in_ln.loadModel(inputStream, channel, height, width, BNType.fully_bn);
    	adaLN_modulation.loadModel(inputStream);
    	
    	mlp_liner1.loadModel(inputStream);
    	mlp_liner2.loadModel(inputStream);
    }

    @Override
    public void accGrad(float scale) {
    	 // TODO Auto-generated method stub
    	in_ln.accGrad(scale);
    	adaLN_modulation.accGrad(scale);
    	
    	mlp_liner1.accGrad(scale);
    	mlp_liner2.accGrad(scale);
    }
    
    public static void loadWeight(Map<String, Object> weightMap, ResBlock block, boolean showLayers) {
        if (showLayers) {
            for (String key : weightMap.keySet()) {
                System.out.println(key);
            }
        }
        
        block.in_ln.gamma = ModeLoaderlUtils.loadData(block.in_ln.gamma, weightMap, 1, "in_ln.weight");
        block.in_ln.beta = ModeLoaderlUtils.loadData(block.in_ln.beta, weightMap, 1, "in_ln.bias");
        
        ModeLoaderlUtils.loadData(block.mlp_liner1.weight, weightMap, "mlp.0.weight");
        ModeLoaderlUtils.loadData(block.mlp_liner1.bias, weightMap, "mlp.0.bias");
        ModeLoaderlUtils.loadData(block.mlp_liner2.weight, weightMap, "mlp.2.weight");
        ModeLoaderlUtils.loadData(block.mlp_liner2.bias, weightMap, "mlp.2.bias");

        ModeLoaderlUtils.loadData(block.adaLN_modulation.weight, weightMap, "adaLN_modulation.1.weight");
        ModeLoaderlUtils.loadData(block.adaLN_modulation.bias, weightMap, "adaLN_modulation.1.bias");

    }
    
    public static void main(String[] args) {
    	
    	int batchSize = 2;
        int time = 256;
        int embedDim = 32;
         
    	Transformer tf = new Transformer();
        tf.number = batchSize * time;
        tf.time = time;
        
        ResBlock block = new ResBlock(embedDim, true, tf);

        String weight = "D:\\models\\resblock_weight.json";
        loadWeight(LagJsonReader.readJsonFileSmallWeight(weight), block, true);
         
     	String inputPath = "D:\\models\\resblock_x.json";
        Map<String, Object> datas = LagJsonReader.readJsonFileSmallWeight(inputPath);
        Tensor input = new Tensor(batchSize, time, 1, embedDim, true);
        ModeLoaderlUtils.loadData(input, datas, "x", 3);

        Map<String, Object> cydatas = LagJsonReader.readJsonFileSmallWeight(inputPath);
        Tensor c = new Tensor(batchSize, time, 1, embedDim, true);
        ModeLoaderlUtils.loadData(c, cydatas, "c", 3);

        input.view(batchSize * time, 1, 1, embedDim);
        c.view(batchSize * time, 1, 1, embedDim);
        
        block.forward(input, c);
        
        block.getOutput().showDM();
        
     	String deltaPath = "D:\\models\\resblock_delta.json";
        Map<String, Object> ddatas = LagJsonReader.readJsonFileSmallWeight(deltaPath);
        Tensor delta = new Tensor(batchSize, time, 1, embedDim, true);
        ModeLoaderlUtils.loadData(delta, ddatas, "delta", 3);
        
        
        Tensor dt = new Tensor(batchSize * time, 1, 1, embedDim, true);
        
        block.back(delta, dt);
        
        block.diff.showDM("dx");
        dt.showDM("di");
    	
    }
    
}

