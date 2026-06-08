package com.omega.engine.nn.layer.dit.video.block;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.util.Map;

import com.omega.engine.gpu.BaseKernel;
import com.omega.engine.gpu.CUDAMemoryManager;
import com.omega.engine.nn.layer.FullyLayer;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.active.SiLULayer;
import com.omega.engine.nn.layer.dit.org.DiTSwiGLUFFN;
import com.omega.engine.nn.layer.normalization.BNType;
import com.omega.engine.nn.layer.normalization.RMSLayer;
import com.omega.engine.nn.network.Network;
import com.omega.engine.tensor.Tensor;
import com.omega.engine.updater.UpdaterFactory;

/**
 * VideoDiTBlock
 * @author Administrator
 */
public class VideoDiTConvBlock extends Layer {
	
	private int batchSize;
	
	private int F;
	private int H;
	private int W;
	
    private int embedDim = 0;
    private int headNum;
    private int time;
    private int cEmbedDim = 0;

    private int maxContext;
    
    private int mlpHiddenDim = 1;
    private boolean bias = false;
    private boolean qkNorm = false;
    
    private SiLULayer modulationAct;
    public FullyLayer adaLN_modulation;

    public RMSLayer norm1;
    public VideoDiTAttentionLayer attn;

    public RMSLayer norm3;

    public DiTSwiGLUFFN txt_mlp;
    
    public GLUMBConv img_mlp;
    
    private Tensor attnInput;
    private Tensor crossAttnInput;
    private Tensor mlpInput;
    
    private Tensor txt_mlp_x;
    private Tensor img_mlp_x;
    
    public Tensor shift_msa;
    public Tensor scale_msa;
    public Tensor gate_msa;
    public Tensor shift_mlp;
    public Tensor scale_mlp;
    public Tensor gate_mlp;
    
    private int[] shape;
    
    private BaseKernel baseKernel;
    
    private Tensor txt_mlp_delta;
    private Tensor img_mlp_delta;
    
    public VideoDiTConvBlock(int embedDim, int cEmbedDim, int time, int mlpHiddenDim, int headNum, int maxContext, int F, int H, int W, boolean bias, boolean qkNorm, Network network) {
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
        this.F = F;
        this.H = H;
        this.W = W;
        this.oChannel = 1;
        this.oHeight = 1;
        this.oWidth = embedDim;
        this.qkNorm = qkNorm;
        this.maxContext = maxContext;
        this.initLayers();
    }

    public void initLayers() {
    	
        this.norm1 = new RMSLayer(1, 1, embedDim, true, BNType.fully_bn, network);
        
        this.modulationAct = new SiLULayer(network);
        
        this.adaLN_modulation = new FullyLayer(cEmbedDim, embedDim * 6, bias, network);
        adaLN_modulation.weight.clearGPU();
        if(adaLN_modulation.bias != null) {
        	adaLN_modulation.bias.clearGPU();
        }

        this.attn = new VideoDiTAttentionLayer(embedDim, headNum, time, bias, qkNorm, network);
        this.norm3 = new RMSLayer(1, 1, embedDim, true, BNType.fully_bn, network);
        
        int swiNum = (int)(2.0f/3.0f * mlpHiddenDim);
        this.txt_mlp = new DiTSwiGLUFFN(embedDim, swiNum, embedDim, bias, network);
        
        this.img_mlp = new GLUMBConv(embedDim, F, H, W, embedDim, 4, network);
        
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
        // TODO Auto-generated method stub
        this.number = input.number;
        this.batchSize = number / time;
        if(shape == null) {
        	shape= new int[] {batchSize, 6, 1, embedDim};
        }
        if(attnInput == null || attnInput.number != number) {
        	attnInput = Tensor.createGPUTensor(attnInput, input.number, input.channel, input.height, input.width, true);
        	shift_msa = Tensor.createGPUTensor(this.shift_msa, batchSize, 1, 1, embedDim, true);
        	scale_msa = Tensor.createGPUTensor(this.scale_msa, batchSize, 1, 1, embedDim, true);
        	gate_msa = Tensor.createGPUTensor(this.gate_msa, batchSize, 1, 1, embedDim, true);
    		shift_mlp = Tensor.createGPUTensor(this.shift_mlp, batchSize, 1, 1, embedDim, true);
        	scale_mlp = Tensor.createGPUTensor(this.scale_mlp, batchSize, 1, 1, embedDim, true);
        	gate_mlp = Tensor.createGPUTensor(this.gate_mlp, batchSize, 1, 1, embedDim, true);
        }
        if(crossAttnInput == null || crossAttnInput.number != number) {
        	crossAttnInput = Tensor.createGPUTensor(crossAttnInput, input.number, input.channel, input.height, input.width, true);
        }
        if(mlpInput == null || mlpInput.number != number) {
        	mlpInput = Tensor.createGPUTensor(mlpInput, input.number, input.channel, input.height, input.width, true);
        }
        if(txt_mlp_x == null || txt_mlp_x.number != batchSize * maxContext) {
        	txt_mlp_x = Tensor.createGPUTensor(txt_mlp_x, batchSize * maxContext, 1, 1, embedDim, true);
        	img_mlp_x = Tensor.createGPUTensor(img_mlp_x, batchSize * F * H * W, 1, 1, embedDim, true);
        }

        if(output == null || output.number != number) {
        	output = Tensor.createGPUTensor(output, input.number, oChannel, oHeight, oWidth, true);
        }

    }
    
    public void init(Tensor input,Tensor tc) {
        // TODO Auto-generated method stub
        this.number = input.number;
        this.batchSize = number / time;
        if(shape == null) {
        	shape= new int[] {batchSize, 6, 1, embedDim};
        }
        if(attnInput == null || attnInput.number != number) {
        	attnInput = Tensor.createGPUTensor(attnInput, input.number, input.channel, input.height, input.width, true);
        	shift_msa = Tensor.createGPUTensor(this.shift_msa, batchSize, 1, 1, embedDim, true);
        	scale_msa = Tensor.createGPUTensor(this.scale_msa, batchSize, 1, 1, embedDim, true);
        	gate_msa = Tensor.createGPUTensor(this.gate_msa, batchSize, 1, 1, embedDim, true);
    		shift_mlp = Tensor.createGPUTensor(this.shift_mlp, batchSize, 1, 1, embedDim, true);
        	scale_mlp = Tensor.createGPUTensor(this.scale_mlp, batchSize, 1, 1, embedDim, true);
        	gate_mlp = Tensor.createGPUTensor(this.gate_mlp, batchSize, 1, 1, embedDim, true);
        }
        if(crossAttnInput == null || crossAttnInput.number != number) {
        	crossAttnInput = Tensor.createGPUTensor(crossAttnInput, input.number, input.channel, input.height, input.width, true);
        }
        if(mlpInput == null || mlpInput.number != number) {
        	mlpInput = Tensor.createGPUTensor(mlpInput, input.number, input.channel, input.height, input.width, true);
        }
        if(txt_mlp_x == null || txt_mlp_x.number != batchSize * maxContext) {
        	txt_mlp_x = Tensor.createGPUTensor(txt_mlp_x, batchSize * maxContext, 1, 1, embedDim, true);
        	img_mlp_x = Tensor.createGPUTensor(img_mlp_x, batchSize, F * H * W, 1, embedDim, true);
        }
        if(output == null || output.number != number) {
        	output = Tensor.createGPUTensor(output, input.number, oChannel, oHeight, oWidth, true);
        }
    }
    
    @Override
    public void initBack() {
        // TODO Auto-generated method stub
        if(txt_mlp_delta == null || txt_mlp_delta.number != batchSize * maxContext) {
        	this.txt_mlp_delta = CUDAMemoryManager.getCache("txt_mlp_delta", batchSize * maxContext, 1, 1, embedDim);
        	this.img_mlp_delta = CUDAMemoryManager.getCache("img_mlp_delta", batchSize, F * H * W, 1, embedDim);
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
    
    public void modulate(Tensor x,Tensor shift,Tensor scale,Tensor output) {
    	/**
    	 * modulate
    	 * x = x * (1 + scale) + shift
    	 */
    	Tensor_OP().add(scale, 1.0f, scale);
    	Tensor_OP().mul(x, scale, output, batchSize, time, 1, output.width, 1);
    	Tensor_OP().addAxis(output, shift, output, batchSize, time, 1, output.width, 1);
    }
    
    public void output(Tensor tc, Tensor[] cos, Tensor[] sin) {
    	/**
    	 * shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
    	 */
    	modulationAct.forward(tc);
    	
    	adaLN_modulation.forward(modulationAct.getOutput());
    	
    	Tensor_OP().getByChannel(adaLN_modulation.getOutput(), shift_msa, shape, 0);
    	Tensor_OP().getByChannel(adaLN_modulation.getOutput(), scale_msa, shape, 1);
    	Tensor_OP().getByChannel(adaLN_modulation.getOutput(), gate_msa, shape, 2);
    	Tensor_OP().getByChannel(adaLN_modulation.getOutput(), shift_mlp, shape, 3);
    	Tensor_OP().getByChannel(adaLN_modulation.getOutput(), scale_mlp, shape, 4);
    	Tensor_OP().getByChannel(adaLN_modulation.getOutput(), gate_mlp, shape, 5);
    	
    	/**
    	 *  x1 = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
    	 */
    	norm1.forward(input);
//    	norm1.getOutput().showDMByOffsetRed(0, 100, "norm");
    	modulate(norm1.getOutput(), shift_msa, scale_msa, attnInput);
    	attn.forward(attnInput, cos, sin, maxContext);
//    	attn.getOutput().showDMByOffsetRed(0, 100, "attn");
//    	attn.getOutput().showDM("attn");
    	Tensor_OP().mul(attn.getOutput(), gate_msa, crossAttnInput, batchSize, time, 1, crossAttnInput.width, 1);
    	Tensor_OP().add(input, crossAttnInput, crossAttnInput);

    	/**
    	 * x3 = x1 + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm3(x1), shift_mlp, scale_mlp))
    	 */
    	norm3.forward(crossAttnInput);
    	modulate(norm3.getOutput(), shift_mlp, scale_mlp, mlpInput);
    	
    	Tensor_OP().getByChannel(mlpInput, txt_mlp_x, new int[] {batchSize, time, 1, embedDim}, 0, maxContext);
    	txt_mlp.forward(txt_mlp_x);
    	
    	Tensor_OP().getByChannel(mlpInput, img_mlp_x, new int[] {batchSize, time, 1, embedDim}, maxContext, F*H*W);
    	img_mlp.forward(img_mlp_x);
    	
    	baseKernel.concat_channel_forward(txt_mlp.getOutput(), img_mlp.getOutput(), mlpInput, batchSize, maxContext, F*H*W, 1, embedDim);
    	
    	Tensor_OP().mul(mlpInput, gate_mlp, output, batchSize, time, 1, output.width, 1);
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

    public void modulate_back(Tensor dShift,Tensor dScale,Tensor dx,Tensor x,Tensor scale,Tensor delta) {
    	Tensor_OP().addAxisBack(dShift, delta, batchSize, time, 1, delta.width, 1);
    	Tensor_OP().mul_right_back(x, delta, dScale, batchSize, time, 1, delta.width, 1);
    	Tensor_OP().mul_left_back(scale, delta, dx,  batchSize, time, 1, delta.width, 1);
    }
    
    public void diff(Tensor dtc, Tensor[] cos, Tensor[] sin) {
//    	delta.showDM("x3");
    	/**
    	 * x3 = x2 + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm3(x2), shift_mlp, scale_mlp))
    	 */
    	Tensor_OP().mul_left_back(gate_mlp, delta, output, batchSize, time, 1, output.width, 1);
    	
    	Tensor_OP().getByChannel(output, txt_mlp_delta, new int[] {batchSize, time, 1, embedDim}, 0, maxContext);
    	txt_mlp.back(txt_mlp_delta);
    	
    	Tensor_OP().getByChannel(output, img_mlp_delta, new int[] {batchSize, time, 1, embedDim}, maxContext, F*H*W);
    	img_mlp.back(img_mlp_delta);

//    	mlp.diff.showDM("mlp");
    	Tensor_OP().mul_right_back(mlpInput, delta, gate_mlp, batchSize, time, 1, output.width, 1);
    	Tensor_OP().setByChannel(adaLN_modulation.getOutput(), gate_mlp, shape, 5);
    	
    	baseKernel.concat_channel_forward(txt_mlp.diff, img_mlp.diff, mlpInput, batchSize, maxContext, F*H*W, 1, embedDim);
    	
    	Tensor dShift = shift_mlp;
    	Tensor dScale = gate_mlp;
    	Tensor x = norm3.getOutput();
    	Tensor scale = scale_mlp;
    	modulate_back(dShift, dScale, output, x, scale, mlpInput);
    	Tensor_OP().setByChannel(adaLN_modulation.getOutput(), dShift, shape, 3);
    	Tensor_OP().setByChannel(adaLN_modulation.getOutput(), dScale, shape, 4);
    	
    	norm3.back(output, norm3.getOutput());

    	Tensor_OP().add(norm3.diff, delta, norm3.diff);

    	/**
    	 *  x1 = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
    	 */
    	Tensor_OP().mul_left_back(gate_msa, norm3.diff, output,  batchSize, time, 1, output.width, 1);

//    	output.showDM("attn_delta");
    	
    	attn.back(output, cos, sin, maxContext);

    	Tensor_OP().mul_right_back(attn.getOutput(), norm3.diff, gate_msa, batchSize, time, 1, output.width, 1);
    	Tensor_OP().setByChannel(adaLN_modulation.getOutput(), gate_msa, shape, 2);
    	
    	dShift = shift_msa;
    	dScale = gate_msa;
    	x = norm1.getOutput();
    	scale = scale_msa;
    	modulate_back(dShift, dScale, output, x, scale, attn.diff);
    	Tensor_OP().setByChannel(adaLN_modulation.getOutput(), dShift, shape, 0);
    	Tensor_OP().setByChannel(adaLN_modulation.getOutput(), dScale, shape, 1);

    	norm1.back(output, norm1.getOutput());

    	Tensor_OP().add(norm1.diff, norm3.diff, norm1.diff);

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
    public void forward(Tensor input, Tensor tc, Tensor[] cos, Tensor[] sin) {
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
        this.output(tc, cos, sin);
    }
    
    @Override
    public void back(Tensor delta) {
        // TODO Auto-generated method stub
    	
    }
    
    public void back(Tensor delta,Tensor dtc) {
        // TODO Auto-generated method stub
        
    }
    
    public void back(Tensor delta, Tensor dtc, Tensor[] cos, Tensor[] sin) {
        // TODO Auto-generated method stub
        this.initBack();
        /**
         * 设置梯度
         */
        this.setDelta(delta);
        /**
         * 计算梯度
         */
        this.diff(dtc, cos, sin);
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
    	norm3.update();
    	txt_mlp.update();
    	img_mlp.update();
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

    	norm3.saveModel(outputStream);
    	txt_mlp.saveModel(outputStream);
    	img_mlp.saveModel(outputStream);
    }

    public void loadModel(RandomAccessFile inputStream) throws IOException {
    	norm1.loadModel(inputStream, channel, height, width, BNType.fully_bn);
    	adaLN_modulation.loadModel(inputStream);
    	
    	attn.loadModel(inputStream);

    	norm3.loadModel(inputStream, 1, 1, attn.oWidth, BNType.fully_bn);
    	txt_mlp.loadModel(inputStream);
    	img_mlp.loadModel(inputStream);
    }

    @Override
    public void accGrad(float scale) {
    	 // TODO Auto-generated method stub
    	norm1.accGrad(scale);
    	adaLN_modulation.accGrad(scale);
    	
    	attn.accGrad(scale);

    	norm3.accGrad(scale);
    	
    	txt_mlp.accGrad(scale);
    	img_mlp.accGrad(scale);
    }
    
    public static void loadWeight(Map<String, Object> weightMap, VideoDiTConvBlock block, boolean showLayers) {
        if (showLayers) {
            for (String key : weightMap.keySet()) {
                System.out.println(key);
            }
        }
        
//        block.norm1.gamma = ModeLoaderlUtils.loadData(block.norm1.gamma, weightMap, 1, "norm1.weight");
//        block.norm3.gamma = ModeLoaderlUtils.loadData(block.norm3.gamma, weightMap, 1, "norm2.weight");
//        
//        ModeLoaderlUtils.loadData(block.attn.qLinerLayer.weight, weightMap, "attn.q.weight");
//        ModeLoaderlUtils.loadData(block.attn.qLinerLayer.bias, weightMap, "attn.q.bias");
//        ModeLoaderlUtils.loadData(block.attn.kLinerLayer.weight, weightMap, "attn.k.weight");
//        ModeLoaderlUtils.loadData(block.attn.kLinerLayer.bias, weightMap, "attn.k.bias");
//        ModeLoaderlUtils.loadData(block.attn.vLinerLayer.weight, weightMap, "attn.v.weight");
//        ModeLoaderlUtils.loadData(block.attn.vLinerLayer.bias, weightMap, "attn.v.bias");
//        ModeLoaderlUtils.loadData(block.attn.oLinerLayer.weight, weightMap, "attn.proj.weight");
//        ModeLoaderlUtils.loadData(block.attn.oLinerLayer.bias, weightMap, "attn.proj.bias");
//
//        ModeLoaderlUtils.loadData(block.mlp.w12.weight, weightMap, "mlp.w12.weight");
//        ModeLoaderlUtils.loadData(block.mlp.w12.bias, weightMap, "mlp.w12.bias");
//        ModeLoaderlUtils.loadData(block.mlp.w3.weight, weightMap, "mlp.w3.weight");
//        ModeLoaderlUtils.loadData(block.mlp.w3.bias, weightMap, "mlp.w3.bias");
//        
//        ModeLoaderlUtils.loadData(block.adaLN_modulation.weight, weightMap, "adaLN_modulation.1.weight");
//        ModeLoaderlUtils.loadData(block.adaLN_modulation.bias, weightMap, "adaLN_modulation.1.bias");

    }
    
    public static void main(String[] args) {
    	
    }
    
}

