package com.omega.engine.nn.layer.dit;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.util.Map;

import com.omega.common.utils.RandomUtils;
import com.omega.engine.nn.layer.FullyLayer;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.active.SiLULayer;
import com.omega.engine.nn.layer.dit.modules.DiTAttentionLayer2;
import com.omega.engine.nn.layer.dit.modules.DiTCrossAttentionLayer2;
import com.omega.engine.nn.layer.dit.modules.DiTMLPLayer;
import com.omega.engine.nn.layer.gpu.RoPEKernel;
import com.omega.engine.nn.layer.normalization.RMSLayer;
import com.omega.engine.nn.network.Network;
import com.omega.engine.nn.network.Transformer;
import com.omega.engine.tensor.Tensor;
import com.omega.engine.updater.UpdaterFactory;
import com.omega.example.clip.utils.ClipModelUtils;
import com.omega.example.transformer.utils.LagJsonReader;

/**
 * DiT_Block
 * @author Administrator
 */
public class DiTOrgBlock extends Layer {
	
	private int batchSize;
	
    private int embedDim = 0;
    private int headNum;
    private int time;
    private int cEmbedDim = 0;
    private int textStateDim;
    private int textTime;
    private int mlpHiddenDim = 1;
    private boolean bias = false;
    private boolean qkNorm = false;
    
    private SiLULayer modulationAct;
    public FullyLayer modulation_shift_msa;
    public FullyLayer modulation_scale_msa;
    public FullyLayer modulation_gate_msa;
    public FullyLayer modulation_shift_mlp;
    public FullyLayer modulation_scale_mlp;
    public FullyLayer modulation_gate_mlp;
    
//    private LNLayer norm1;
    public RMSLayer norm1;
    public DiTAttentionLayer2 attn;
    public RMSLayer norm2;
    public DiTCrossAttentionLayer2 cross_attn;
    public RMSLayer norm3;
    public DiTMLPLayer mlp;
    
    private Tensor attnInput;
    private Tensor crossAttnInput;
    private Tensor crossAttnOut;
    private Tensor mlpInput;
    

    public DiTOrgBlock(int embedDim, int cEmbedDim, int textStateDim, int time, int textTime, int mlpHiddenDim, int headNum, boolean bias, boolean qkNorm) {
        this.embedDim = embedDim;
        this.cEmbedDim = cEmbedDim;
        this.headNum = headNum;
        this.textStateDim = textStateDim;
        this.time = time;
        this.textTime = textTime;
        this.mlpHiddenDim = mlpHiddenDim;
        this.bias = bias;
        this.qkNorm = qkNorm;
        this.oChannel = 1;
        this.oHeight = 1;
        this.oWidth = embedDim;
        this.initLayers();
    }

    public DiTOrgBlock(int embedDim, int cEmbedDim, int textStateDim, int time, int textTime, int mlpHiddenDim, int headNum, boolean bias, boolean qkNorm, Network network) {
        this.network = network;
        if (this.updater == null) {
            this.setUpdater(UpdaterFactory.create(network));
        }
        this.embedDim = embedDim;
        this.cEmbedDim = cEmbedDim;
        this.headNum = headNum;
        this.textStateDim = textStateDim;
        this.time = time;
        this.textTime = textTime;
        this.mlpHiddenDim = mlpHiddenDim;
        this.bias = bias;
        this.oChannel = 1;
        this.oHeight = 1;
        this.oWidth = embedDim;
        this.qkNorm = qkNorm;
        this.initLayers();
    }

    public void initLayers() {
    	
        this.norm1 = new RMSLayer(network);
        
        this.modulationAct = new SiLULayer(network);

        this.modulation_shift_msa = new FullyLayer(cEmbedDim, embedDim, bias, network);
        this.modulation_shift_msa.weight.clearGPU();
        this.modulation_shift_msa.bias.clearGPU();
        this.modulation_scale_msa = new FullyLayer(cEmbedDim, embedDim, bias, network);
        this.modulation_scale_msa.weight.clearGPU();
        this.modulation_scale_msa.bias.clearGPU();
        this.modulation_gate_msa = new FullyLayer(cEmbedDim, embedDim, bias, network);
        this.modulation_gate_msa.weight.clearGPU();
        this.modulation_gate_msa.bias.clearGPU();
        this.modulation_shift_mlp = new FullyLayer(cEmbedDim, embedDim, bias, network);
        this.modulation_shift_mlp.weight.clearGPU();
        this.modulation_shift_mlp.bias.clearGPU();
        this.modulation_scale_mlp = new FullyLayer(cEmbedDim, embedDim, bias, network);
        this.modulation_scale_mlp.weight.clearGPU();
        this.modulation_scale_mlp.bias.clearGPU();
        this.modulation_gate_mlp = new FullyLayer(cEmbedDim, embedDim, bias, network);
        this.modulation_gate_mlp.weight.clearGPU();
        this.modulation_gate_mlp.bias.clearGPU();
        
        this.attn = new DiTAttentionLayer2(embedDim, headNum, time, bias, qkNorm, network);
        this.norm2 = new RMSLayer(network);
        this.cross_attn = new DiTCrossAttentionLayer2(embedDim, textStateDim, headNum, time, textTime, bias, qkNorm, network);
        this.norm3 = new RMSLayer(network);
       
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
        	crossAttnOut = Tensor.createGPUTensor(crossAttnOut, input.number, input.channel, input.height, input.width, true);
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
    	Tensor_OP().mul(x, scale, output, batchSize, time, 1, output.width, 1);
    	Tensor_OP().addAxis(output, shift, output, batchSize, time, 1, output.width, 1);
    }
    
    public void output(Tensor tc,Tensor text) {
    	
    	norm1.forward(input);

    	modulationAct.forward(tc);
    	modulation_shift_msa.forward(modulationAct.getOutput());
    	modulation_scale_msa.forward(modulationAct.getOutput());
    	modulation_gate_msa.forward(modulationAct.getOutput());
    	modulation_shift_mlp.forward(modulationAct.getOutput());
    	modulation_scale_mlp.forward(modulationAct.getOutput());
    	modulation_gate_mlp.forward(modulationAct.getOutput());

    	modulate(norm1.getOutput(), modulation_shift_msa.getOutput(), modulation_scale_msa.getOutput(), attnInput);

    	attn.forward(attnInput);
    	Tensor_OP().mul(attn.getOutput(), modulation_gate_msa.getOutput(), crossAttnInput, batchSize, time, 1, crossAttnInput.width, 1);
    	Tensor_OP().add(input, crossAttnInput, crossAttnInput);
    	
    	norm2.forward(crossAttnInput);

    	cross_attn.forward(norm2.getOutput(), text);

    	Tensor_OP().add(crossAttnInput, cross_attn.getOutput(), crossAttnOut);
    	
    	norm3.forward(crossAttnOut);
    	
    	modulate(norm3.getOutput(), modulation_shift_mlp.getOutput(), modulation_scale_mlp.getOutput(), mlpInput);
    	
    	mlp.forward(mlpInput);
    	
    	Tensor_OP().mul(mlp.getOutput(), modulation_gate_mlp.getOutput(), output, batchSize, time, 1, output.width, 1);
    	
    	Tensor_OP().add(crossAttnOut, output, output);
    }
    
    public void output(Tensor tc,Tensor text,Tensor cos,Tensor sin) {

    	/**
    	 * shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
    	 */
    	modulationAct.forward(tc);
    	modulation_shift_msa.forward(modulationAct.getOutput());
    	modulation_scale_msa.forward(modulationAct.getOutput());
    	modulation_gate_msa.forward(modulationAct.getOutput());
    	modulation_shift_mlp.forward(modulationAct.getOutput());
    	modulation_scale_mlp.forward(modulationAct.getOutput());
    	modulation_gate_mlp.forward(modulationAct.getOutput());
    	
    	/**
    	 *  x1 = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
    	 */
    	norm1.forward(input);
    	modulate(norm1.getOutput(), modulation_shift_msa.getOutput(), modulation_scale_msa.getOutput(), attnInput);
    	attn.forward(attnInput, cos , sin);
    	Tensor_OP().mul(attn.getOutput(), modulation_gate_msa.getOutput(), crossAttnInput, batchSize, time, 1, crossAttnInput.width, 1);
    	Tensor_OP().add(input, crossAttnInput, crossAttnInput);

    	/**
    	 * x2 = x1 + self.crossAttn(self.norm2(x1), text)
    	 */
    	norm2.forward(crossAttnInput);
    	cross_attn.forward(norm2.getOutput(), text, cos , sin);
    	Tensor_OP().add(crossAttnInput, cross_attn.getOutput(), crossAttnOut);

    	/**
    	 * x3 = x2 + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm3(x2), shift_mlp, scale_mlp))
    	 */
    	norm3.forward(crossAttnOut);
    	modulate(norm3.getOutput(), modulation_shift_mlp.getOutput(), modulation_scale_mlp.getOutput(), mlpInput);
    	mlp.forward(mlpInput);
    	Tensor_OP().mul(mlp.getOutput(), modulation_gate_mlp.getOutput(), output, batchSize, time, 1, output.width, 1);
    	Tensor_OP().add(crossAttnOut, output, output);
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
    	Tensor_OP().addAxisBack(dShift, delta, batchSize, time, 1, delta.width, 1);
    	Tensor_OP().mul_right_back(x, delta, dScale, batchSize, time, 1, delta.width, 1);
    	Tensor_OP().mul_left_back(scale, delta, dx,  batchSize, time, 1, delta.width, 1);
    }
    
    public void diff(Tensor dtc,Tensor dText) {
    	
    	Tensor_OP().mul_left_back(modulation_gate_mlp.getOutput(), delta, output,  batchSize, time, 1, output.width, 1);
    	mlp.back(output);
    	Tensor_OP().mul_right_back(mlp.getOutput(), delta, modulation_gate_mlp.getOutput(), batchSize, time, 1, output.width, 1);
    	modulation_gate_mlp.back(modulation_gate_mlp.getOutput());
    	
    	Tensor dShift = modulation_shift_mlp.getOutput();
    	Tensor dScale = modulation_gate_mlp.getOutput();
    	Tensor x = norm3.getOutput();
    	Tensor scale = modulation_scale_mlp.getOutput();
    	modulate_back(dShift, dScale, output, x, scale, mlp.diff);
    	modulation_shift_mlp.back(dShift);
    	modulation_scale_mlp.back(dScale);
    	
    	norm3.back(output);

    	Tensor_OP().add(norm3.diff, delta, norm3.diff);
    	
    	cross_attn.back(norm3.diff);

    	norm2.back(cross_attn.diff);
    	Tensor_OP().add(norm2.diff, norm3.diff, norm2.diff);
    	
    	Tensor_OP().mul_left_back(modulation_gate_msa.getOutput(), norm2.diff, output,  batchSize, time, 1, output.width, 1);
    	attn.back(output);
    	Tensor_OP().mul_right_back(attn.getOutput(), norm2.diff, modulation_gate_msa.getOutput(), batchSize, time, 1, output.width, 1);
    	modulation_gate_msa.back(modulation_gate_msa.getOutput());
    	
    	dShift = modulation_shift_msa.getOutput();
    	dScale = modulation_gate_msa.getOutput();
    	x = norm1.getOutput();
    	scale = modulation_scale_msa.getOutput();
    	modulate_back(dShift, dScale, output, x, scale, attn.diff);
    	modulation_shift_msa.back(dShift);
    	modulation_scale_msa.back(dScale);

    	norm1.back(output);

    	Tensor_OP().add(norm1.diff, norm2.diff, norm1.diff);
    	
    	Tensor dtc_tmp = modulation_scale_msa.diff;
    	
    	Tensor_OP().add(dtc_tmp, modulation_shift_msa.diff, dtc_tmp);
    	Tensor_OP().add(dtc_tmp, modulation_gate_msa.diff, dtc_tmp);
    	Tensor_OP().add(dtc_tmp, modulation_scale_mlp.diff, dtc_tmp);
    	Tensor_OP().add(dtc_tmp, modulation_shift_mlp.diff, dtc_tmp);
    	Tensor_OP().add(dtc_tmp, modulation_gate_mlp.diff, dtc_tmp);

        modulationAct.back(dtc_tmp);
        
    	Tensor_OP().add(dtc, modulationAct.diff, dtc);
    	Tensor_OP().add(dText, cross_attn.kLinerLayer.diff, dText);
    	
    	this.diff = norm1.diff;
    }

    public void diff(Tensor dtc,Tensor dText,Tensor cos,Tensor sin) {
//    	delta.showDM("x3");
    	/**
    	 * x3 = x2 + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm3(x2), shift_mlp, scale_mlp))
    	 */
    	Tensor_OP().mul_left_back(modulation_gate_mlp.getOutput(), delta, output,  batchSize, time, 1, output.width, 1);
    	mlp.back(output);
    	Tensor_OP().mul_right_back(mlp.getOutput(), delta, modulation_gate_mlp.getOutput(), batchSize, time, 1, output.width, 1);
    	modulation_gate_mlp.back(modulation_gate_mlp.getOutput());
    	
    	Tensor dShift = modulation_shift_mlp.getOutput();
    	Tensor dScale = modulation_gate_mlp.getOutput();
    	Tensor x = norm3.getOutput();
    	Tensor scale = modulation_scale_mlp.getOutput();
    	modulate_back(dShift, dScale, output, x, scale, mlp.diff);
    	modulation_shift_mlp.back(dShift);
    	modulation_scale_mlp.back(dScale);
    	
    	norm3.back(output);
    	Tensor_OP().add(norm3.diff, delta, norm3.diff);

    	/**
    	 * x2 = x1 + self.crossAttn(self.norm2(x1), text)
    	 */
    	cross_attn.back(norm3.diff, cos, sin);

    	norm2.back(cross_attn.diff);
    	Tensor_OP().add(norm2.diff, norm3.diff, norm2.diff);
    	
//    	norm2.diff.showDM("x1");
    	/**
    	 *  x1 = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
    	 */
    	Tensor_OP().mul_left_back(modulation_gate_msa.getOutput(), norm2.diff, output,  batchSize, time, 1, output.width, 1);
    	attn.back(output, cos, sin);
    	Tensor_OP().mul_right_back(attn.getOutput(), norm2.diff, modulation_gate_msa.getOutput(), batchSize, time, 1, output.width, 1);
    	modulation_gate_msa.back(modulation_gate_msa.getOutput());
    	
    	dShift = modulation_shift_msa.getOutput();
    	dScale = modulation_gate_msa.getOutput();
    	x = norm1.getOutput();
    	scale = modulation_scale_msa.getOutput();
//    	attn.diff.showDM("m1");
    	modulate_back(dShift, dScale, output, x, scale, attn.diff);
    	modulation_shift_msa.back(dShift);
    	modulation_scale_msa.back(dScale);

    	norm1.back(output);

    	Tensor_OP().add(norm1.diff, norm2.diff, norm1.diff);
    	
    	Tensor dtc_tmp = modulation_scale_msa.diff;
    	
    	Tensor_OP().add(dtc_tmp, modulation_shift_msa.diff, dtc_tmp);
    	Tensor_OP().add(dtc_tmp, modulation_gate_msa.diff, dtc_tmp);
    	Tensor_OP().add(dtc_tmp, modulation_scale_mlp.diff, dtc_tmp);
    	Tensor_OP().add(dtc_tmp, modulation_shift_mlp.diff, dtc_tmp);
    	Tensor_OP().add(dtc_tmp, modulation_gate_mlp.diff, dtc_tmp);

        modulationAct.back(dtc_tmp);
        
    	Tensor_OP().add(dtc, modulationAct.diff, dtc);
    	Tensor_OP().add(dText, cross_attn.kLinerLayer.diff, dText);
    	
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
    public void forward(Tensor input,Tensor tc,Tensor text) {
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
        this.output(tc, text);
    }
    
    /**
     * 
     * @param input
     * @param tc time cond
     * @param text
     */
    public void forward(Tensor input,Tensor tc,Tensor text,Tensor cos,Tensor sin) {
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
    
    public void back(Tensor delta,Tensor dtc,Tensor dtext) {
        // TODO Auto-generated method stub
        this.initBack();
        /**
         * 设置梯度
         */
        this.setDelta(delta);
        /**
         * 计算梯度
         */
        this.diff(dtc, dtext);
        if (this.network.GRADIENT_CHECK) {
            this.gradientCheck();
        }
    }
    
    public void back(Tensor delta,Tensor dtc,Tensor dtext,Tensor cos,Tensor sin) {
        // TODO Auto-generated method stub
        this.initBack();
        /**
         * 设置梯度
         */
        this.setDelta(delta);
        /**
         * 计算梯度
         */
        this.diff(dtc, dtext, cos, sin);
        if (this.network.GRADIENT_CHECK) {
            this.gradientCheck();
        }
    }
    
    @Override
    public void update() {
        // TODO Auto-generated method stub
    	norm1.update();
    	modulation_shift_msa.update();
    	modulation_scale_msa.update();
    	modulation_gate_msa.update();
    	modulation_shift_mlp.update();
    	modulation_scale_mlp.update();
    	modulation_gate_mlp.update();
    	
    	attn.update();
    	norm2.update();
    	
    	cross_attn.update();
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
    	modulation_shift_msa.saveModel(outputStream);
    	modulation_scale_msa.saveModel(outputStream);
    	modulation_gate_msa.saveModel(outputStream);
    	modulation_shift_mlp.saveModel(outputStream);
    	modulation_scale_mlp.saveModel(outputStream);
    	modulation_gate_mlp.saveModel(outputStream);
    	
    	attn.saveModel(outputStream);
    	norm2.saveModel(outputStream);
    	
    	cross_attn.saveModel(outputStream);
    	norm3.saveModel(outputStream);
    	
    	mlp.saveModel(outputStream);
    }

    public void loadModel(RandomAccessFile inputStream) throws IOException {
    	norm1.loadModel(inputStream);
    	modulation_shift_msa.loadModel(inputStream);
    	modulation_scale_msa.loadModel(inputStream);
    	modulation_gate_msa.loadModel(inputStream);
    	modulation_shift_mlp.loadModel(inputStream);
    	modulation_scale_mlp.loadModel(inputStream);
    	modulation_gate_mlp.loadModel(inputStream);
    	
    	attn.loadModel(inputStream);
    	norm2.loadModel(inputStream);
    	
    	cross_attn.loadModel(inputStream);
    	norm3.loadModel(inputStream);
    	
    	mlp.loadModel(inputStream);
    }

    @Override
    public void accGrad(float scale) {
        // TODO Auto-generated method stub
    	norm1.accGrad(scale);
    	modulation_shift_msa.accGrad(scale);
    	modulation_scale_msa.accGrad(scale);
    	modulation_gate_msa.accGrad(scale);
    	modulation_shift_mlp.accGrad(scale);
    	modulation_scale_mlp.accGrad(scale);
    	modulation_gate_mlp.accGrad(scale);
    	
    	attn.accGrad(scale);
    	norm2.accGrad(scale);
    	
    	cross_attn.accGrad(scale);
    	norm3.accGrad(scale);
    	
    	mlp.accGrad(scale);
    }
    
    public static void loadWeight(Map<String, Object> weightMap, DiTOrgBlock block, boolean showLayers) {
        if (showLayers) {
            for (String key : weightMap.keySet()) {
                System.out.println(key);
            }
        }
        
        block.norm1.gamma = ClipModelUtils.loadData(block.norm1.gamma, weightMap, 1, "norm1.weight");
//        block.norm1.beta = ClipModelUtils.loadData(block.norm1.beta, weightMap, 1, "norm1.bias");
        
        ClipModelUtils.loadData(block.attn.qLinerLayer.weight, weightMap, "attn1.qL.weight");
        ClipModelUtils.loadData(block.attn.qLinerLayer.bias, weightMap, "attn1.qL.bias");
        ClipModelUtils.loadData(block.attn.kLinerLayer.weight, weightMap, "attn1.kL.weight");
        ClipModelUtils.loadData(block.attn.kLinerLayer.bias, weightMap, "attn1.kL.bias");
        ClipModelUtils.loadData(block.attn.vLinerLayer.weight, weightMap, "attn1.vL.weight");
        ClipModelUtils.loadData(block.attn.vLinerLayer.bias, weightMap, "attn1.vL.bias");
        ClipModelUtils.loadData(block.attn.oLinerLayer.weight, weightMap, "attn1.proj.weight");
        ClipModelUtils.loadData(block.attn.oLinerLayer.bias, weightMap, "attn1.proj.bias");
        
        block.norm2.gamma = ClipModelUtils.loadData(block.norm2.gamma, weightMap, 1, "norm2.weight");
//        block.norm2.beta = ClipModelUtils.loadData(block.norm2.beta, weightMap, 1, "norm2.bias");
        
        ClipModelUtils.loadData(block.cross_attn.qLinerLayer.weight, weightMap, "attn2.query.weight");
        ClipModelUtils.loadData(block.cross_attn.qLinerLayer.bias, weightMap, "attn2.query.bias");
        ClipModelUtils.loadData(block.cross_attn.kLinerLayer.weight, weightMap, "attn2.key.weight");
        ClipModelUtils.loadData(block.cross_attn.kLinerLayer.bias, weightMap, "attn2.key.bias");
        ClipModelUtils.loadData(block.cross_attn.vLinerLayer.weight, weightMap, "attn2.value.weight");
        ClipModelUtils.loadData(block.cross_attn.vLinerLayer.bias, weightMap, "attn2.value.bias");
        ClipModelUtils.loadData(block.cross_attn.oLinerLayer.weight, weightMap, "attn2.out_proj.weight");
        ClipModelUtils.loadData(block.cross_attn.oLinerLayer.bias, weightMap, "attn2.out_proj.bias");
        
        block.norm3.gamma = ClipModelUtils.loadData(block.norm3.gamma, weightMap, 1, "norm3.weight");
//        block.norm3.beta = ClipModelUtils.loadData(block.norm3.beta, weightMap, 1, "norm3.bias");
        
        ClipModelUtils.loadData(block.mlp.linear1.weight, weightMap, "mlp.fc1.weight");
        ClipModelUtils.loadData(block.mlp.linear1.bias, weightMap, "mlp.fc1.bias");
        ClipModelUtils.loadData(block.mlp.linear2.weight, weightMap, "mlp.fc2.weight");
        ClipModelUtils.loadData(block.mlp.linear2.bias, weightMap, "mlp.fc2.bias");
        
        ClipModelUtils.loadData(block.modulation_shift_msa.weight, weightMap, "adaLN_modulation_1.weight");
        ClipModelUtils.loadData(block.modulation_shift_msa.bias, weightMap, "adaLN_modulation_1.bias");
        ClipModelUtils.loadData(block.modulation_scale_msa.weight, weightMap, "adaLN_modulation_2.weight");
        ClipModelUtils.loadData(block.modulation_scale_msa.bias, weightMap, "adaLN_modulation_2.bias");
        ClipModelUtils.loadData(block.modulation_gate_msa.weight, weightMap, "adaLN_modulation_3.weight");
        ClipModelUtils.loadData(block.modulation_gate_msa.bias, weightMap, "adaLN_modulation_3.bias");
        ClipModelUtils.loadData(block.modulation_shift_mlp.weight, weightMap, "adaLN_modulation_4.weight");
        ClipModelUtils.loadData(block.modulation_shift_mlp.bias, weightMap, "adaLN_modulation_4.bias");
        ClipModelUtils.loadData(block.modulation_scale_mlp.weight, weightMap, "adaLN_modulation_5.weight");
        ClipModelUtils.loadData(block.modulation_scale_mlp.bias, weightMap, "adaLN_modulation_5.bias");
        ClipModelUtils.loadData(block.modulation_gate_mlp.weight, weightMap, "adaLN_modulation_6.weight");
        ClipModelUtils.loadData(block.modulation_gate_mlp.bias, weightMap, "adaLN_modulation_6.bias");
    }
    
    public static void main(String[] args) {
    	
    	String inputPath = "H:\\model\\dit_block_org.json";
    	Map<String, Object> datas = LagJsonReader.readJsonFileSmallWeight(inputPath);
    	
    	int batchSize = 2;
        int time = 256;
        int embedDim = 384;
        int headNum = 6;
         
        float[] data = RandomUtils.order(batchSize * time * embedDim, 0.01f, 0.01f);
        Tensor input = new Tensor(batchSize * time, 1, 1, embedDim, data, true);
        
        int textTime = 77;
        float[] cData = RandomUtils.order(batchSize * textTime * embedDim, 0.001f, 0.001f); 
        Tensor cond = new Tensor(batchSize * textTime, 1, 1, embedDim, cData, true);
        
        float[] tData = RandomUtils.order(batchSize * embedDim, 0.1f, 0.1f); 
        Tensor t = new Tensor(batchSize, 1, 1, embedDim, tData, true);
        
    	Transformer tf = new Transformer();
        tf.number = batchSize * time;
        tf.time = time;
        
        DiTOrgBlock block = new DiTOrgBlock(embedDim, embedDim, embedDim, time, textTime, 4 * embedDim, headNum, true, false, tf);
        
        loadWeight(datas, block, true);
        
        Tensor[] cs = RoPEKernel.getCosAndSin2D(time, embedDim, headNum);
        Tensor cos = cs[0];
        Tensor sin = cs[1];
        
        block.forward(input, t, cond, cos, sin);
        
        block.getOutput().showDM();
        
        float[] dd = RandomUtils.order(batchSize * time * embedDim, 0.01f, 0.01f);
        Tensor delta = new Tensor(batchSize * time, 1, 1, embedDim, dd, true);
        
        Tensor dt = new Tensor(batchSize, 1, 1, embedDim, true);
        Tensor dc = new Tensor(batchSize * textTime, 1, 1, embedDim, true);
        block.back(delta, dt, dc, cos, sin);
        block.diff.showDM();
    }
    
}

