package com.omega.engine.nn.layer.dit;

import java.io.IOException;
import java.io.RandomAccessFile;

import com.omega.engine.nn.layer.FullyLayer;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.active.SiLULayer;
import com.omega.engine.nn.layer.dit.modules.DiTAttentionLayer2;
import com.omega.engine.nn.layer.dit.modules.DiTCrossAttentionLayer2;
import com.omega.engine.nn.layer.dit.modules.DiTMLPLayer;
import com.omega.engine.nn.layer.normalization.LNLayer;
import com.omega.engine.nn.network.Network;
import com.omega.engine.tensor.Tensor;
import com.omega.engine.updater.UpdaterFactory;

/**
 * DiT_Block
 * @author Administrator
 */
public class DiTBlock extends Layer {
	
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
    public FullyLayer modulation;
    
//    private LNLayer norm1;
    public LNLayer norm1;
    public DiTAttentionLayer2 attn;
    public LNLayer norm2;
    public DiTCrossAttentionLayer2 cross_attn;
    public LNLayer norm3;
    public DiTMLPLayer mlp;
    
    private Tensor attnInput;
    private Tensor crossAttnInput;
    private Tensor mlpInput;
    
    private Tensor dtc_cache;

    public DiTBlock(int embedDim, int cEmbedDim, int textStateDim, int time, int textTime, int mlpHiddenDim, int headNum, boolean bias, boolean qkNorm) {
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

    public DiTBlock(int embedDim, int cEmbedDim, int textStateDim, int time, int textTime, int mlpHiddenDim, int headNum, boolean bias, boolean qkNorm, Network network) {
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
    	
        this.norm1 = new LNLayer(network);
        
        this.modulationAct = new SiLULayer(network);
        this.modulation = new FullyLayer(cEmbedDim, embedDim, bias, network);
        this.modulation.weight.clearGPU();
        this.modulation.bias.clearGPU();
        this.attn = new DiTAttentionLayer2(embedDim, headNum, time, bias, qkNorm, network);
        this.norm2 = new LNLayer(network);
        this.cross_attn = new DiTCrossAttentionLayer2(embedDim, textStateDim, headNum, time, textTime, bias, qkNorm, network);
        this.norm3 = new LNLayer(network);
       
        this.mlp = new DiTMLPLayer(embedDim, mlpHiddenDim, bias, network);
    }

    @Override
    public void init() {
        // TODO Auto-generated method stub
        this.number = this.input.getShape()[0];
    }

    public void init(Tensor input) {
        // TODO Auto-generated method stub
        this.number = input.getShape()[0];
        this.batchSize = number / time;
        if(attnInput == null || attnInput.getShape()[0] != number) {
        	attnInput = Tensor.createGPUTensor(attnInput, input.getShape()[0], input.getShape()[1], input.getShape()[2], input.getShape()[3], true);
        }
        if(crossAttnInput == null || crossAttnInput.getShape()[0] != number) {
        	crossAttnInput = Tensor.createGPUTensor(crossAttnInput, input.getShape()[0], input.getShape()[1], input.getShape()[2], input.getShape()[3], true);
        }
        if(mlpInput == null || mlpInput.getShape()[0] != number) {
        	mlpInput = Tensor.createGPUTensor(mlpInput, input.getShape()[0], input.getShape()[1], input.getShape()[2], input.getShape()[3], true);
        }
        if(output == null || output.getShape()[0] != number) {
        	output = Tensor.createGPUTensor(output, input.getShape()[0], oChannel, oHeight, oWidth, true);
        }
    }
    
    @Override
    public void initBack() {
        // TODO Auto-generated method stub
    	int batchSize = number / time;
    	if(dtc_cache == null || dtc_cache.getShape()[0] != batchSize) {
    		dtc_cache = Tensor.createGPUTensor(dtc_cache, batchSize, 1, 1, cEmbedDim, true);
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
    
    public void output(Tensor tc,Tensor text) {
    	
    	norm1.forward(input);

    	modulationAct.forward(tc);
    	modulation.forward(modulationAct.getOutput());

    	Tensor_OP().addAxis(norm1.getOutput(), modulation.getOutput(), attnInput, batchSize, time, 1, embedDim, 1);

    	attn.forward(attnInput);
//    	attn.getOutput().showDM("attn");
    	Tensor_OP().add(input, attn.getOutput(), crossAttnInput);
    	
    	norm2.forward(crossAttnInput);

    	cross_attn.forward(norm2.getOutput(), text);

    	Tensor_OP().add(crossAttnInput, cross_attn.getOutput(), mlpInput);
    	
    	norm3.forward(mlpInput);
    	
    	mlp.forward(norm3.getOutput());
    	
    	Tensor_OP().add(mlpInput, mlp.getOutput(), output);
    }
    
    public void output(Tensor tc,Tensor text,Tensor cos,Tensor sin) {
    	
    	norm1.forward(input);
    	
    	modulationAct.forward(tc);
    	modulation.forward(modulationAct.getOutput());
    	
    	Tensor_OP().addAxis(norm1.getOutput(), modulation.getOutput(), attnInput, batchSize, time, 1, embedDim, 1);

    	attn.forward(attnInput, cos , sin);

    	Tensor_OP().add(input, attn.getOutput(), crossAttnInput);
    	
    	norm2.forward(crossAttnInput);

    	cross_attn.forward(norm2.getOutput(), text, cos , sin);

    	Tensor_OP().add(crossAttnInput, cross_attn.getOutput(), mlpInput);
    	
    	norm3.forward(mlpInput);
    	
    	mlp.forward(norm3.getOutput());
    	
    	Tensor_OP().add(mlpInput, mlp.getOutput(), output);
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

    public void diff(Tensor dtc,Tensor dText) {
    	
    	mlp.back(delta);
    	norm3.back(mlp.diff);
    	Tensor_OP().add(norm3.diff, delta, norm3.diff);
    	
    	cross_attn.back(norm3.diff);
    	
    	norm2.back(cross_attn.diff);
    	Tensor_OP().add(norm2.diff, norm3.diff, norm2.diff);
    	
    	attn.back(norm2.diff);
    	
    	norm1.back(attn.diff);
    	Tensor_OP().add(norm1.diff, norm2.diff, norm1.diff);
    	
    	dtc_cache.clearGPU();
        Tensor_OP().addAxisBack(dtc_cache, attn.diff, batchSize, time, 1, embedDim, 1);
        
        modulation.back(dtc_cache);
        modulationAct.back(modulation.diff);
        
    	Tensor_OP().add(dtc, modulationAct.diff, dtc);
    	Tensor_OP().add(dText, cross_attn.kLinerLayer.diff, dText);
    	
    	this.diff = norm1.diff;
    }

    public void diff(Tensor dtc,Tensor dText,Tensor cos,Tensor sin) {
    	
    	mlp.back(delta);
    	norm3.back(mlp.diff);
    	Tensor_OP().add(norm3.diff, delta, norm3.diff);

    	cross_attn.back(norm3.diff, cos, sin);
    	
//    	cross_attn.diff.showDMByOffsetRed(5 * cross_attn.diff.height * cross_attn.diff.width, cross_attn.diff.height * cross_attn.diff.width, "cross_attn.diff");
    	norm2.back(cross_attn.diff);
    	Tensor_OP().add(norm2.diff, norm3.diff, norm2.diff);
    	
    	attn.back(norm2.diff, cos, sin);

    	norm1.back(attn.diff);
    	Tensor_OP().add(norm1.diff, norm2.diff, norm1.diff);
    	
    	dtc_cache.clearGPU();
        Tensor_OP().addAxisBack(dtc_cache, attn.diff, batchSize, time, 1, embedDim, 1);
        
        modulation.back(dtc_cache);
        modulationAct.back(modulation.diff);
        
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
    	modulation.update();
    	
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
    	modulation.saveModel(outputStream);
    	
    	attn.saveModel(outputStream);
    	norm2.saveModel(outputStream);
    	
    	cross_attn.saveModel(outputStream);
    	norm3.saveModel(outputStream);
    	
    	mlp.saveModel(outputStream);
    }

    public void loadModel(RandomAccessFile inputStream) throws IOException {
    	norm1.loadModel(inputStream);
    	modulation.loadModel(inputStream);
    	
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
    	modulation.accGrad(scale);
    	
    	attn.accGrad(scale);
    	norm2.accGrad(scale);
    	
    	cross_attn.accGrad(scale);
    	norm3.accGrad(scale);
    	
    	mlp.accGrad(scale);
    }
    
//    public static void loadWeight(Map<String, Object> weightMap, DiTBlock block, boolean showLayers) {
//        if (showLayers) {
//            for (String key : weightMap.keySet()) {
//                System.out.println(key);
//            }
//        }
//        
//        block.norm1.gamma = ClipModelUtils.loadData(block.norm1.gamma, weightMap, 1, "norm1.weight");
////        block.norm1.beta = ClipModelUtils.loadData(block.norm1.beta, weightMap, 1, "norm1.bias");
//        
////        ClipModelUtils.loadData(block.attn.qLinerLayer.weight, weightMap, "attn1.qL.weight");
////        ClipModelUtils.loadData(block.attn.qLinerLayer.bias, weightMap, "attn1.qL.bias");
////        ClipModelUtils.loadData(block.attn.kLinerLayer.weight, weightMap, "attn1.kL.weight");
////        ClipModelUtils.loadData(block.attn.kLinerLayer.bias, weightMap, "attn1.kL.bias");
////        ClipModelUtils.loadData(block.attn.vLinerLayer.weight, weightMap, "attn1.vL.weight");
////        ClipModelUtils.loadData(block.attn.vLinerLayer.bias, weightMap, "attn1.vL.bias");
//        
//        ClipModelUtils.loadData(block.attn.qkvLinerLayer.weight, weightMap, "attn1.qkv.weight");
//        ClipModelUtils.loadData(block.attn.qkvLinerLayer.bias, weightMap, "attn1.qkv.bias");
//        ClipModelUtils.loadData(block.attn.oLinerLayer.weight, weightMap, "attn1.proj.weight");
//        ClipModelUtils.loadData(block.attn.oLinerLayer.bias, weightMap, "attn1.proj.bias");
//        
//        block.norm2.gamma = ClipModelUtils.loadData(block.norm2.gamma, weightMap, 1, "norm2.weight");
////        block.norm2.beta = ClipModelUtils.loadData(block.norm2.beta, weightMap, 1, "norm2.bias");
//        
//        ClipModelUtils.loadData(block.modulation.weight, weightMap, "default_modulation.1.weight");
//        ClipModelUtils.loadData(block.modulation.bias, weightMap, "default_modulation.1.bias");
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
////        block.norm3.beta = ClipModelUtils.loadData(block.norm3.beta, weightMap, 1, "norm3.bias");
//        
//        ClipModelUtils.loadData(block.mlp.linear1.weight, weightMap, "mlp.fc1.weight");
//        ClipModelUtils.loadData(block.mlp.linear1.bias, weightMap, "mlp.fc1.bias");
//        ClipModelUtils.loadData(block.mlp.linear2.weight, weightMap, "mlp.fc2.weight");
//        ClipModelUtils.loadData(block.mlp.linear2.bias, weightMap, "mlp.fc2.bias");
//    }
//    
//    public static void main(String[] args) {
//    	
//    	String inputPath = "H:\\model\\dit_block.json";
//    	Map<String, Object> datas = LagJsonReader.readJsonFileSmallWeight(inputPath);
//    	
//        int batchSize = 2;
//        int time = 64;
//        int embedDim = 16;
//
//        int headNum = 4;
//
//        int textTime = 4;
//        int textEmbedDim = 12;
//        
//        int mlpHiddenDim = embedDim * 4;
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
//        float[] textData = RandomUtils.order(batchSize * textTime * textEmbedDim, 0.1f, 0.1f);
//        Tensor text = new Tensor(batchSize * textTime, 1, 1, textEmbedDim, textData, true);
//        
//        float[] delta_data = RandomUtils.order(batchSize * time * embedDim, 0.01f, 0.01f);
//        Tensor delta = new Tensor(batchSize * time, 1, 1, embedDim, delta_data, true);
//
//        Tensor dcond = new Tensor(batchSize, 1, 1, embedDim, true);
//
//        Tensor dtext = new Tensor(batchSize * textTime, 1, 1, textEmbedDim, true);
//        
//        DiTBlock block = new DiTBlock(embedDim, embedDim, textEmbedDim, time, textTime, mlpHiddenDim, headNum, true, tf);
//        
//        loadWeight(datas, block, true);
//        
//        for (int i = 0; i < 10; i++) {
//            //			input.showDM();
//        	dcond.clearGPU();
//        	dtext.clearGPU();
//        	block.forward(input, cond, text);
//        	block.getOutput().showShape();
//        	block.getOutput().showDM();
//        	block.back(delta, dcond, dtext);
////            //			delta.showDM();
//        	block.diff.showDM("dx");
//        	dcond.showDM("dcond");
//        	dtext.showDM("dtext");
//            //			delta.copyData(tmp);
//        }
//    }
}

