package com.omega.engine.nn.layer.opensora.dit.moudles;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.util.Map;

import com.omega.engine.nn.layer.FullyLayer;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.active.SiLULayer;
import com.omega.engine.nn.layer.dit.modules.DiTAttentionLayer2;
import com.omega.engine.nn.layer.dit.modules.DiTMLPLayer;
import com.omega.engine.nn.layer.normalization.BNType;
import com.omega.engine.nn.layer.normalization.LNLayer;
import com.omega.engine.nn.network.Network;
import com.omega.engine.tensor.Tensor;
import com.omega.engine.updater.UpdaterFactory;

/**
 * DiT_Block
 * @author Administrator
 */
public class OpenSoraDiTBlock extends Layer {
	
	private int batchSize;
	
    private int embedDim = 0;
    private int headNum;
    private int time;
    private int cEmbedDim = 0;

    private int mlpHiddenDim = 1;
    private boolean bias = false;
    private boolean qkNorm = false;
    
    private SiLULayer modulationAct;
    public FullyLayer adaLN_modulation;
    
    public LNLayer norm1;

    public DiTAttentionLayer2 attn;
    public LNLayer norm2;
    public DiTMLPLayer mlp;

    public Tensor shift_msa;
    public Tensor scale_msa;
    public Tensor gate_msa;
    public Tensor shift_mlp;
    public Tensor scale_mlp;
    public Tensor gate_mlp;
    
    private Tensor attnInput;
    private Tensor crossAttnInput;
    private Tensor mlpInput;
    
    private int[] shape;

    public OpenSoraDiTBlock(int embedDim, int cEmbedDim, int time, int mlpHiddenDim, int headNum, boolean bias, boolean qkNorm) {
        this.embedDim = embedDim;
        this.cEmbedDim = cEmbedDim;
        this.headNum = headNum;
        this.time = time;
        this.mlpHiddenDim = mlpHiddenDim;
        this.bias = bias;
        this.qkNorm = qkNorm;
        this.channel = 1;
        this.height = 1;
        this.width = embedDim;
        this.oChannel = 1;
        this.oHeight = 1;
        this.oWidth = embedDim;
        this.initLayers();
    }

    public OpenSoraDiTBlock(int embedDim, int cEmbedDim, int time, int mlpHiddenDim, int headNum, boolean bias, boolean qkNorm, Network network) {
        this.network = network;
        if (this.updater == null) {
            this.setUpdater(UpdaterFactory.create(network));
        }
        this.embedDim = embedDim;
        this.cEmbedDim = cEmbedDim;
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
        this.initLayers();
    }

    public void initLayers() {
    	
        this.norm1 = new LNLayer(1, 1, embedDim, false, BNType.fully_bn, network);
        
        this.modulationAct = new SiLULayer(network);

        this.adaLN_modulation = new FullyLayer(cEmbedDim, 6 * embedDim, bias, network);
        this.adaLN_modulation.weight.clearGPU();
        this.adaLN_modulation.bias.clearGPU();
        
        this.attn = new DiTAttentionLayer2(embedDim, headNum, time, bias, qkNorm, network);
        this.norm2 = new LNLayer(1, 1, embedDim, false, BNType.fully_bn, network);
        this.mlp = new DiTMLPLayer(embedDim, mlpHiddenDim, bias, network);
    }

    @Override
    public void init() {
        // TODO Auto-generated method stub
        this.number = this.input.number;
    }

    public void init(Tensor input) {
        // TODO Auto-generated method stub
        this.number = input.number;
        this.batchSize = number / time;
        if(attnInput == null || attnInput.number != number) {
        	attnInput = Tensor.createGPUTensor(attnInput, input.number, input.channel, input.height, input.width, true);
        }
        if(crossAttnInput == null || crossAttnInput.number != number) {
        	crossAttnInput = Tensor.createGPUTensor(crossAttnInput, input.number, input.channel, input.height, input.width, true);
        }
        if(mlpInput == null || mlpInput.number != number) {
        	mlpInput = Tensor.createGPUTensor(mlpInput, input.number, input.channel, input.height, input.width, true);
        }
        if(output == null || output.number != number) {
        	shape = new int[] {batchSize, 6, 1, embedDim};
        	shift_msa = Tensor.createGPUTensor(this.shift_msa, batchSize, 1, 1, embedDim, true);
        	scale_msa = Tensor.createGPUTensor(this.scale_msa, batchSize, 1, 1, embedDim, true);
        	gate_msa = Tensor.createGPUTensor(this.gate_msa, batchSize, 1, 1, embedDim, true);
    		shift_mlp = Tensor.createGPUTensor(this.shift_mlp, batchSize, 1, 1, embedDim, true);
        	scale_mlp = Tensor.createGPUTensor(this.scale_mlp, batchSize, 1, 1, embedDim, true);
        	gate_mlp = Tensor.createGPUTensor(this.gate_mlp, batchSize, 1, 1, embedDim, true);
        	output = Tensor.createGPUTensor(output, input.number, oChannel, oHeight, oWidth, true);
        }

    }
    
    public void init(Tensor input,Tensor tc) {
        // TODO Auto-generated method stub
        this.number = input.number;
        this.batchSize = number / time;
        if(attnInput == null || attnInput.number != number) {
        	attnInput = Tensor.createGPUTensor(attnInput, input.number, input.channel, input.height, input.width, true);
        }
        if(crossAttnInput == null || crossAttnInput.number != number) {
        	crossAttnInput = Tensor.createGPUTensor(crossAttnInput, input.number, input.channel, input.height, input.width, true);
        }
        if(mlpInput == null || mlpInput.number != number) {
        	mlpInput = Tensor.createGPUTensor(mlpInput, input.number, input.channel, input.height, input.width, true);
        }
        
        if(output == null || output.number != number) {
        	shape = new int[] {batchSize, 6, 1, embedDim};
        	shift_msa = Tensor.createGPUTensor(this.shift_msa, batchSize, 1, 1, embedDim, true);
        	scale_msa = Tensor.createGPUTensor(this.scale_msa, batchSize, 1, 1, embedDim, true);
        	gate_msa = Tensor.createGPUTensor(this.gate_msa, batchSize, 1, 1, embedDim, true);
    		shift_mlp = Tensor.createGPUTensor(this.shift_mlp, batchSize, 1, 1, embedDim, true);
        	scale_mlp = Tensor.createGPUTensor(this.scale_mlp, batchSize, 1, 1, embedDim, true);
        	gate_mlp = Tensor.createGPUTensor(this.gate_mlp, batchSize, 1, 1, embedDim, true);
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
    	Tensor_OP().mul(x, scale, output, batchSize, time, 1, output.width, 1);
    	Tensor_OP().addAxis(output, shift, output, batchSize, time, 1, output.width, 1);
    }
    
    public void output(Tensor tc) {
    	
    	modulationAct.forward(tc);
    	
    	adaLN_modulation.forward(modulationAct.getOutput());
    	
    	int[] shape = new int[] {batchSize, 6, 1, embedDim};

    	Tensor_OP().getByChannel(adaLN_modulation.getOutput(), shift_msa, shape, 0);
    	Tensor_OP().getByChannel(adaLN_modulation.getOutput(), scale_msa, shape, 1);
    	Tensor_OP().getByChannel(adaLN_modulation.getOutput(), gate_msa, shape, 2);
    	Tensor_OP().getByChannel(adaLN_modulation.getOutput(), shift_mlp, shape, 3);
    	Tensor_OP().getByChannel(adaLN_modulation.getOutput(), scale_mlp, shape, 4);
    	Tensor_OP().getByChannel(adaLN_modulation.getOutput(), gate_mlp, shape, 5);
    	
    	norm1.forward(input);

    	modulate(norm1.getOutput(), shift_msa, scale_msa, attnInput);

    	attn.forward(attnInput);
    	
    	Tensor_OP().mul(attn.getOutput(), gate_msa, crossAttnInput, batchSize, time, 1, crossAttnInput.width, 1);
    	Tensor_OP().add(input, crossAttnInput, crossAttnInput);
    	
    	norm2.forward(crossAttnInput);
    	
    	modulate(norm2.getOutput(), shift_mlp, scale_mlp, mlpInput);
    	
    	mlp.forward(mlpInput);
    	
    	Tensor_OP().mul(mlp.getOutput(), gate_mlp, output, batchSize, time, 1, output.width, 1);
    	Tensor_OP().add(crossAttnInput, output, output);
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

    public void modulate_back(Tensor tmp,Tensor dx,Tensor x,Tensor scale,Tensor delta, int idx1, int idx2) {
    	Tensor_OP().addAxisBack(tmp, delta, batchSize, time, 1, delta.width, 1);
    	Tensor_OP().setByChannel(adaLN_modulation.getOutput(), tmp, shape, idx1);
    	Tensor_OP().mul_right_back(x, delta, tmp, batchSize, time, 1, delta.width, 1);
    	Tensor_OP().setByChannel(adaLN_modulation.getOutput(), tmp, shape, idx2);
    	Tensor_OP().mul_left_back(scale, delta, dx,  batchSize, time, 1, delta.width, 1);
    }
    
    public void diff(Tensor dtc) {
    	
    	Tensor_OP().mul_left_back(gate_mlp, delta, output,  batchSize, time, 1, output.width, 1);
    	mlp.back(output);
    	Tensor_OP().mul_right_back(mlp.getOutput(), delta, gate_mlp, batchSize, time, 1, output.width, 1);
    	Tensor_OP().setByChannel(adaLN_modulation.getOutput(), gate_mlp, shape, 5);
    	
    	Tensor tmp = shift_mlp;
    	Tensor x = norm2.getOutput();
    	Tensor scale = scale_mlp;
    	modulate_back(tmp, output, x, scale, mlp.diff, 3, 4);
    	
    	norm2.back(output);
    	Tensor_OP().add(norm2.diff, delta, norm2.diff);
    	
    	Tensor_OP().mul_left_back(gate_msa, norm2.diff, output,  batchSize, time, 1, output.width, 1);
    	attn.back(output);
    	Tensor_OP().mul_right_back(attn.getOutput(), norm2.diff, gate_msa, batchSize, time, 1, output.width, 1);
    	Tensor_OP().setByChannel(adaLN_modulation.getOutput(), gate_msa, shape, 2);
    	
    	tmp = shift_msa;
    	x = norm1.getOutput();
    	scale = scale_msa;
    	modulate_back(tmp, norm1.getOutput(), x, scale, attn.diff, 0, 1);
    	norm1.back(norm1.getOutput());
    	Tensor_OP().add(norm1.diff, norm2.diff, norm1.diff);
    	
    	adaLN_modulation.back(adaLN_modulation.getOutput());
    	modulationAct.back(adaLN_modulation.diff);
    	Tensor_OP().add(dtc, modulationAct.diff, dtc);
    	
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
    public void forward(Tensor input,Tensor tc) {
        // TODO Auto-generated method stub
        /**
         * 设置输入
         */
        this.setInput(input);
        /**
         * 参数初始化
         */
        this.init(input, tc);
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
    	norm1.update();
    	adaLN_modulation.update();
    	
    	attn.update();
    	norm2.update();
    	
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
    	adaLN_modulation.saveModel(outputStream);
    	
    	attn.saveModel(outputStream);
    	norm2.saveModel(outputStream);
    	
    	mlp.saveModel(outputStream);
    }

    public void loadModel(RandomAccessFile inputStream) throws IOException {
    	
    	norm1.loadModel(inputStream, channel, height, width, BNType.fully_bn);
    	adaLN_modulation.loadModel(inputStream);
    	
    	attn.loadModel(inputStream);
    	norm2.loadModel(inputStream, 1, 1, attn.oWidth, BNType.fully_bn);
    	
    	mlp.loadModel(inputStream);
    }

    @Override
    public void accGrad(float scale) {
        // TODO Auto-generated method stub
    	norm1.accGrad(scale);
    	adaLN_modulation.accGrad(scale);
    	
    	attn.accGrad(scale);
    	norm2.accGrad(scale);
    	
    	mlp.accGrad(scale);
    }
    
    public static void loadWeight(Map<String, Object> weightMap, OpenSoraDiTBlock block, boolean showLayers) {
        if (showLayers) {
            for (String key : weightMap.keySet()) {
                System.out.println(key);
            }
        }
        
//        block.norm1.gamma = ClipModelUtils.loadData(block.norm1.gamma, weightMap, 1, "norm1.weight");
//        block.norm1.beta = ClipModelUtils.loadData(block.norm1.beta, weightMap, 1, "norm1.bias");
//        
//        ClipModelUtils.loadData(block.attn.qLinerLayer.weight, weightMap, "attn1.qL.weight");
//        ClipModelUtils.loadData(block.attn.qLinerLayer.bias, weightMap, "attn1.qL.bias");
//        ClipModelUtils.loadData(block.attn.kLinerLayer.weight, weightMap, "attn1.kL.weight");
//        ClipModelUtils.loadData(block.attn.kLinerLayer.bias, weightMap, "attn1.kL.bias");
//        ClipModelUtils.loadData(block.attn.vLinerLayer.weight, weightMap, "attn1.vL.weight");
//        ClipModelUtils.loadData(block.attn.vLinerLayer.bias, weightMap, "attn1.vL.bias");
//        ClipModelUtils.loadData(block.attn.oLinerLayer.weight, weightMap, "attn1.proj.weight");
//        ClipModelUtils.loadData(block.attn.oLinerLayer.bias, weightMap, "attn1.proj.bias");
//        
//        block.norm2.gamma = ClipModelUtils.loadData(block.norm2.gamma, weightMap, 1, "norm2.weight");
//        block.norm2.beta = ClipModelUtils.loadData(block.norm2.beta, weightMap, 1, "norm2.bias");
//        
//        ClipModelUtils.loadData(block.cross_attn.qLinerLayer.weight, weightMap, "attn2.query.weight");
//        ClipModelUtils.loadData(block.cross_attn.qLinerLayer.bias, weightMap, "attn2.query.bias");
//        ClipModelUtils.loadData(block.cross_attn.kLinerLayer.weight, weightMap, "attn2.key.weight");
//        ClipModelUtils.loadData(block.cross_attn.kLinerLayer.bias, weightMap, "attn2.key.bias");
//        ClipModelUtils.loadData(block.cross_attn.vLinerLayer.weight, weightMap, "attn2.value.weight");
//        ClipModelUtils.loadData(block.cross_attn.vLinerLayer.bias, weightMap, "attn2.value.bias");
//        ClipModelUtils.loadData(block.cross_attn.oLinerLayer.weight, weightMap, "attn2.out_proj.weight");
//        ClipModelUtils.loadData(block.cross_attn.oLinerLayer.bias, weightMap, "attn2.out_proj.bias");
//        
//        block.norm3.gamma = ClipModelUtils.loadData(block.norm3.gamma, weightMap, 1, "norm3.weight");
//        block.norm3.beta = ClipModelUtils.loadData(block.norm3.beta, weightMap, 1, "norm3.bias");
//        
//        ClipModelUtils.loadData(block.mlp.linear1.weight, weightMap, "mlp.fc1.weight");
//        ClipModelUtils.loadData(block.mlp.linear1.bias, weightMap, "mlp.fc1.bias");
//        ClipModelUtils.loadData(block.mlp.linear2.weight, weightMap, "mlp.fc2.weight");
//        ClipModelUtils.loadData(block.mlp.linear2.bias, weightMap, "mlp.fc2.bias");
//        
//        ClipModelUtils.loadData(block.modulation_shift_msa.weight, weightMap, "adaLN_modulation_1.weight");
//        ClipModelUtils.loadData(block.modulation_shift_msa.bias, weightMap, "adaLN_modulation_1.bias");
//        ClipModelUtils.loadData(block.modulation_scale_msa.weight, weightMap, "adaLN_modulation_2.weight");
//        ClipModelUtils.loadData(block.modulation_scale_msa.bias, weightMap, "adaLN_modulation_2.bias");
//        ClipModelUtils.loadData(block.modulation_gate_msa.weight, weightMap, "adaLN_modulation_3.weight");
//        ClipModelUtils.loadData(block.modulation_gate_msa.bias, weightMap, "adaLN_modulation_3.bias");
//        ClipModelUtils.loadData(block.modulation_shift_mlp.weight, weightMap, "adaLN_modulation_4.weight");
//        ClipModelUtils.loadData(block.modulation_shift_mlp.bias, weightMap, "adaLN_modulation_4.bias");
//        ClipModelUtils.loadData(block.modulation_scale_mlp.weight, weightMap, "adaLN_modulation_5.weight");
//        ClipModelUtils.loadData(block.modulation_scale_mlp.bias, weightMap, "adaLN_modulation_5.bias");
//        ClipModelUtils.loadData(block.modulation_gate_mlp.weight, weightMap, "adaLN_modulation_6.weight");
//        ClipModelUtils.loadData(block.modulation_gate_mlp.bias, weightMap, "adaLN_modulation_6.bias");
    }
    
    public static void main(String[] args) {
//    	
//    	String inputPath = "H:\\model\\dit_block_org.json";
//    	Map<String, Object> datas = LagJsonReader.readJsonFileSmallWeight(inputPath);
//    	
//    	int batchSize = 2;
//        int time = 256;
//        int embedDim = 384;
//        int headNum = 6;
//         
//        float[] data = RandomUtils.order(batchSize * time * embedDim, 0.01f, 0.01f);
//        Tensor input = new Tensor(batchSize * time, 1, 1, embedDim, data, true);
//        
//        int textTime = 77;
//        float[] cData = RandomUtils.order(batchSize * textTime * embedDim, 0.001f, 0.001f); 
//        Tensor cond = new Tensor(batchSize * textTime, 1, 1, embedDim, cData, true);
//        
//        float[] tData = RandomUtils.order(batchSize * embedDim, 0.1f, 0.1f); 
//        Tensor t = new Tensor(batchSize, 1, 1, embedDim, tData, true);
//        
//    	Transformer tf = new Transformer();
//        tf.number = batchSize * time;
//        tf.time = time;
//        
//        DiTBlock block = new DiTBlock(embedDim, embedDim, embedDim, time, textTime, 4 * embedDim, headNum, true, false, tf);
//        
//        loadWeight(datas, block, true);
//        
//        Tensor[] cs = RoPEKernel.getCosAndSin2D(time, embedDim, headNum);
//        Tensor cos = cs[0];
//        Tensor sin = cs[1];
//        
//        block.forward(input, t, cond, cos, sin);
//        
//        block.getOutput().showDM();
//        
//        float[] dd = RandomUtils.order(batchSize * time * embedDim, 0.01f, 0.01f);
//        Tensor delta = new Tensor(batchSize * time, 1, 1, embedDim, dd, true);
//        
//        Tensor dt = new Tensor(batchSize, 1, 1, embedDim, true);
//        Tensor dc = new Tensor(batchSize * textTime, 1, 1, embedDim, true);
//        block.back(delta, dt, dc, cos, sin);
//        block.diff.showDM();
    }
    
}

