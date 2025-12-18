package com.omega.engine.nn.layer.dit.deco;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import com.omega.engine.gpu.BaseKernel;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.active.SiLULayer;
import com.omega.engine.nn.layer.dit.DiTCaptionEmbeddingLayer;
import com.omega.engine.nn.layer.dit.DiTOrgTimeEmbeddingLayer;
import com.omega.engine.nn.layer.gpu.ConvKernel;
import com.omega.engine.nn.layer.gpu.RoPEKernel;
import com.omega.engine.nn.network.Network;
import com.omega.engine.tensor.Tensor;
import com.omega.engine.updater.UpdaterFactory;

/**
 * PixNerDiTMainMoudue
 * @author Administrator
 */
public class PixNerDiTMainMoudue extends Layer {
	
	public int inChannel;
    public int width;
    public int height;
    public int patchSize;
    private int hiddenSize;
    private int decoder_hidden_size;
    private int num_encoder_blocks;
    private int num_decoder_blocks;
    private int timeSteps;
    private int headNum;
    private int textEmbedDim;
    private int maxContextLen;
    private int mlpRatio = 4;
    
    public EmbedLayer s_embedder;
    public DiTOrgTimeEmbeddingLayer timeEmbd;
    public DiTCaptionEmbeddingLayer labelEmbd;
    private SiLULayer t_act;
    public List<FlattenDiTBlock> blocks;
    private SiLULayer s_act;
    public NerfEmbedder x_embedder;
    public SimpleMLPAdaLN dec_net;
    
    private ConvKernel convKernel;
    
    private int p_emb;
    private int ph;
    private int pw;
    private int hw;
    
    private float y_drop_prob = 0.0f;
    
    private BaseKernel baseKernel;
    
    private Tensor x;
    private Tensor xt;
    private Tensor dec_net_t;
    
    private Tensor dtc;
    private Tensor dcontext;
    
    private Tensor pos;
    
    public PixNerDiTMainMoudue(int inChannel, int width, int height, int patchSize, int hiddenSize, int decoder_hidden_size, int headNum, int num_encoder_blocks, int num_decoder_blocks, int timeSteps, int textEmbedDim, int maxContextLen, int mlpRatio, float y_drop_prob, Network network) {
		this.network = network;
        if (this.updater == null) {
            this.setUpdater(UpdaterFactory.create(network));
        }
        this.y_drop_prob = y_drop_prob;
    	this.inChannel = inChannel;
		this.width = width;
		this.height = height;
		this.patchSize = patchSize;
		this.headNum = headNum;
		this.hiddenSize = hiddenSize;
		this.decoder_hidden_size = decoder_hidden_size;
		this.num_encoder_blocks = num_encoder_blocks;
		this.num_decoder_blocks = num_decoder_blocks;
		this.timeSteps = timeSteps;
		this.textEmbedDim = textEmbedDim;
		this.maxContextLen = maxContextLen;
		this.mlpRatio = mlpRatio;
		this.headNum = headNum;
		this.initLayers();
		this.oHeight = height;
		this.oWidth = width;
    }

    public void initLayers() {
    	
    	int pw = width / patchSize;
    	int ph = height / patchSize;
    	
    	p_emb = inChannel * patchSize * patchSize;

    	s_embedder = new EmbedLayer(p_emb, hiddenSize, true, "ln", network);
        
    	hw = pw * ph;
    	
        timeEmbd = new DiTOrgTimeEmbeddingLayer(timeSteps, 256, hiddenSize, true, network);
        
        labelEmbd = new DiTCaptionEmbeddingLayer(textEmbedDim, hiddenSize, maxContextLen, y_drop_prob, true, network);
        
        t_act = new SiLULayer(network);
        
        blocks = new ArrayList<FlattenDiTBlock>();
         
        for(int i = 0;i<num_encoder_blocks;i++) {
        	FlattenDiTBlock block = new FlattenDiTBlock(hiddenSize, hiddenSize, hw, mlpRatio * hiddenSize, headNum, maxContextLen, true, false, network);
	        blocks.add(block);
        }
        
        s_act = new SiLULayer(network);
        
        x_embedder = new NerfEmbedder(inChannel, decoder_hidden_size, 8, patchSize, true, network);
        
        dec_net = new SimpleMLPAdaLN(decoder_hidden_size, decoder_hidden_size, inChannel, hiddenSize, num_decoder_blocks, patchSize, true, network);
        
        if(baseKernel == null) {
        	baseKernel = new BaseKernel(cuda());
        }
        
        if(convKernel == null) {
        	convKernel = new ConvKernel(inChannel, height, width, 1, patchSize, patchSize, patchSize, 0, cuda());
        }
        
        pos = RoPEKernel.precompute_freqs_cis_2d_tensor(hiddenSize / headNum, ph, pw, 10000, patchSize);
        
    }

    @Override
    public void init() {
        // TODO Auto-generated method stub
        this.number = this.input.number;
    }
    
    public void init(Tensor input) {
        // TODO Auto-generated method stub
        this.number = input.number;
        if(this.output == null || this.output.number != number) {
        	x = Tensor.createGPUTensor(x, number, p_emb, ph, pw, true);
        	xt = Tensor.createGPUTensor(xt, number * hw, 1, 1, p_emb, true);
        	dec_net_t = Tensor.createGPUTensor(dec_net_t, number * hw, 1, 1, p_emb, true);
        	output = Tensor.createGPUTensor(output, number, oChannel, oHeight, oWidth, true);
        }else {
        	x.viewOrg();
        	xt.viewOrg();
        }
    }
    
    @Override
    public void initBack() {
        // TODO Auto-generated method stub
    	if(dtc == null || dtc.number != timeEmbd.getOutput().number) {
    		dtc = Tensor.createGPUTensor(dtc, timeEmbd.getOutput().shape(), true);
    		dcontext = Tensor.createGPUTensor(dcontext, labelEmbd.getOutput().shape(), true);
    	}else {
    		dtc.clearGPU();
    		dcontext.clearGPU();
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
    
    public void output(Tensor tc,Tensor label) {
    		
    	// x = torch.nn.functional.unfold(x, kernel_size=self.patch_size, stride=self.patch_size).transpose(1, 2)
    	convKernel.im2col(input, x);
    	Tensor_OP().permute(x, xt, new int[] {number, p_emb, 1, hw}, new int[] {number, hw, 1, p_emb}, new int[] {0, 3, 2, 1});

    	timeEmbd.forward(tc);
    	t_act.forward(timeEmbd.getOutput());
    	
    	labelEmbd.forward(label);
    	
    	s_embedder.forward(xt);
    	
    	Tensor t = t_act.getOutput();
    	
    	Tensor cond = labelEmbd.getOutput();

    	Tensor s = s_embedder.getOutput();

    	for(int i = 0;i<num_encoder_blocks;i++) {
    		FlattenDiTBlock block = blocks.get(i);
    		block.forward(s, t, cond, pos);
    		s = block.getOutput();
    	}
    	
    	// s = torch.nn.functional.silu(t + s)
    	Tensor_OP().addAxis(s, t, s, number, hw, 1, hiddenSize, 1);
    	s_act.forward(s);
    	
    	/**
    	 *  x = x.reshape(batch_size * length, self.in_channels, self.patch_size ** 2)
         *  x = x.transpose(1, 2)
    	 */
    	x.view(number * hw * patchSize * patchSize, 1, 1, inChannel);
    	Tensor_OP().permute(xt, x, new int[] {number * hw, inChannel, 1, patchSize * patchSize}, new int[] {number * hw, patchSize * patchSize, 1, inChannel}, new int[] {0, 3, 2, 1});
    	
    	x_embedder.forward(x);

    	dec_net.forward(x_embedder.getOutput(), s_act.getOutput());
    	/**
    	 *  x = x.transpose(1, 2)
         *  x = x.reshape(batch_size, length, -1)
    	 */
    	Tensor_OP().permute(dec_net.getOutput(), dec_net_t, new int[] {number, hw, patchSize * patchSize, inChannel}, new int[] {number, inChannel, patchSize * patchSize, hw}, new int[] {0, 3, 2, 1});
    	// x = torch.nn.functional.fold(x.transpose(1, 2).contiguous(), (H, W), kernel_size=self.patch_size, stride=self.patch_size)
    	convKernel.col2im(dec_net_t, output);

    }

    @Override
    public Tensor getOutput() {
        // TODO Auto-generated method stub
        return output;
    }

    @Override
    public void diff() {

    	convKernel.im2col(delta, dec_net_t);
    	Tensor_OP().permute(dec_net_t, dec_net.getOutput(), new int[] {number, inChannel, patchSize * patchSize, hw}, new int[] {number, hw, patchSize * patchSize, inChannel}, new int[] {0, 3, 2, 1});
    	
    	dec_net.back(dec_net.getOutput());
    	
    	x_embedder.back(dec_net.diff);
    	
    	s_act.back(dec_net.getCDiff());
    	
    	Tensor ds = s_act.diff;
    	Tensor_OP().addAxisBack(dtc, ds, number, hw, 1, hiddenSize, 1);
    	
    	for(int i = num_encoder_blocks - 1;i>=0;i--) {
    		FlattenDiTBlock block = blocks.get(i);
    		block.back(ds, dtc, dcontext, pos);
    		ds = block.diff;
    	}
    	
    	s_embedder.back(ds);
    	
//     	dtxt.showDM("dtxt");
     	labelEmbd.back(dcontext);

     	t_act.back(dtc);
     	timeEmbd.back(t_act.diff);

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
    	s_embedder.update();
    	timeEmbd.update();
    	labelEmbd.update();
    	for(int i = 0;i<num_encoder_blocks;i++) {
    		blocks.get(i).update();
    	}
    	x_embedder.update();
    	dec_net.update();
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
    	s_embedder.saveModel(outputStream);
    	timeEmbd.saveModel(outputStream);
    	labelEmbd.saveModel(outputStream);
    	for(int i = 0;i<num_encoder_blocks;i++) {
    		blocks.get(i).saveModel(outputStream);
    	}
    	x_embedder.saveModel(outputStream);
    	dec_net.saveModel(outputStream);
    }

    public void loadModel(RandomAccessFile inputStream) throws IOException {
    	s_embedder.loadModel(inputStream);
    	timeEmbd.loadModel(inputStream);
    	labelEmbd.loadModel(inputStream);
    	for(int i = 0;i<num_encoder_blocks;i++) {
    		blocks.get(i).loadModel(inputStream);
    	}
    	x_embedder.loadModel(inputStream);
    	dec_net.loadModel(inputStream);
    }

    @Override
    public void accGrad(float scale) {
        // TODO Auto-generated method stub
    	s_embedder.accGrad(scale);
    	timeEmbd.accGrad(scale);
    	labelEmbd.accGrad(scale);
    	for(int i = 0;i<num_encoder_blocks;i++) {
    		blocks.get(i).accGrad(scale);
    	}
    	x_embedder.accGrad(scale);
    	dec_net.accGrad(scale);
    }
    
    public static void loadWeight(Map<String, Object> weightMap, PixNerDiTMainMoudue block, boolean showLayers) {
        if (showLayers) {
            for (String key : weightMap.keySet()) {
                System.out.println(key);
            }
        }
     
//        ModeLoaderlUtils.loadData(block.patchEmbd.patchEmbedding.weight, weightMap, "x_embedder.proj.weight");
//        ModeLoaderlUtils.loadData(block.patchEmbd.patchEmbedding.bias, weightMap, "x_embedder.proj.bias");
//        
//        ModeLoaderlUtils.loadData(block.timeEmbd.linear1.weight, weightMap, "t_embedder.mlp.0.weight");
//        ModeLoaderlUtils.loadData(block.timeEmbd.linear1.bias, weightMap, "t_embedder.mlp.0.bias");
//        ModeLoaderlUtils.loadData(block.timeEmbd.linear2.weight, weightMap, "t_embedder.mlp.2.weight");
//        ModeLoaderlUtils.loadData(block.timeEmbd.linear2.bias, weightMap, "t_embedder.mlp.2.bias");
//        
//        ModeLoaderlUtils.loadData(block.labelEmbd.linear1.weight, weightMap, "y_embedder.y_proj.fc1.weight");
//        ModeLoaderlUtils.loadData(block.labelEmbd.linear1.bias, weightMap, "y_embedder.y_proj.fc1.bias");
//        ModeLoaderlUtils.loadData(block.labelEmbd.linear2.weight, weightMap, "y_embedder.y_proj.fc2.weight");
//        ModeLoaderlUtils.loadData(block.labelEmbd.linear2.bias, weightMap, "y_embedder.y_proj.fc2.bias");
//        
//        for(int i = 0;i<block.depth;i++) {
//        	
//        	DiT_TXTBlock jb = block.blocks.get(i);
//        	
////        	jb.norm1.gamma = ModeLoaderlUtils.loadData(jb.norm1.gamma, weightMap, 1, "blocks."+i+".norm1.weight"); 
////        	jb.norm2.gamma = ModeLoaderlUtils.loadData(jb.norm2.gamma, weightMap, 1, "blocks."+i+".norm2.weight"); 
////        	jb.norm3.gamma = ModeLoaderlUtils.loadData(jb.norm3.gamma, weightMap, 1, "blocks."+i+".norm3.weight"); 
//        	
//        	ModeLoaderlUtils.loadData(jb.attn.qLinerLayer.weight, weightMap, "blocks."+i+".attn.q.weight");
//            ModeLoaderlUtils.loadData(jb.attn.qLinerLayer.bias, weightMap, "blocks."+i+".attn.q.bias");
//        	ModeLoaderlUtils.loadData(jb.attn.kLinerLayer.weight, weightMap, "blocks."+i+".attn.k.weight");
//            ModeLoaderlUtils.loadData(jb.attn.kLinerLayer.bias, weightMap, "blocks."+i+".attn.k.bias");
//        	ModeLoaderlUtils.loadData(jb.attn.vLinerLayer.weight, weightMap, "blocks."+i+".attn.v.weight");
//            ModeLoaderlUtils.loadData(jb.attn.vLinerLayer.bias, weightMap, "blocks."+i+".attn.v.bias");
//        	ModeLoaderlUtils.loadData(jb.attn.oLinerLayer.weight, weightMap, "blocks."+i+".attn.proj.weight");
//            ModeLoaderlUtils.loadData(jb.attn.oLinerLayer.bias, weightMap, "blocks."+i+".attn.proj.bias");
//        
//        	ModeLoaderlUtils.loadData(jb.cross_attn.qLinerLayer.weight, weightMap, "blocks."+i+".cross_attn.q.weight");
//            ModeLoaderlUtils.loadData(jb.cross_attn.qLinerLayer.bias, weightMap, "blocks."+i+".cross_attn.q.bias");
//        	ModeLoaderlUtils.loadData(jb.cross_attn.kLinerLayer.weight, weightMap, "blocks."+i+".cross_attn.k.weight");
//            ModeLoaderlUtils.loadData(jb.cross_attn.kLinerLayer.bias, weightMap, "blocks."+i+".cross_attn.k.bias");
//        	ModeLoaderlUtils.loadData(jb.cross_attn.vLinerLayer.weight, weightMap, "blocks."+i+".cross_attn.v.weight");
//            ModeLoaderlUtils.loadData(jb.cross_attn.vLinerLayer.bias, weightMap, "blocks."+i+".cross_attn.v.bias");
//        	ModeLoaderlUtils.loadData(jb.cross_attn.oLinerLayer.weight, weightMap, "blocks."+i+".cross_attn.proj.weight");
//            ModeLoaderlUtils.loadData(jb.cross_attn.oLinerLayer.bias, weightMap, "blocks."+i+".cross_attn.proj.bias");
//            
//            ModeLoaderlUtils.loadData(jb.mlp.w12.weight, weightMap, "blocks."+i+".mlp.w12.weight");
//            ModeLoaderlUtils.loadData(jb.mlp.w12.bias, weightMap, "blocks."+i+".mlp.w12.bias");
//            ModeLoaderlUtils.loadData(jb.mlp.w3.weight, weightMap, "blocks."+i+".mlp.w3.weight");
//            ModeLoaderlUtils.loadData(jb.mlp.w3.bias, weightMap, "blocks."+i+".mlp.w3.bias");
//            
//            ModeLoaderlUtils.loadData(jb.adaLN_modulation.weight, weightMap, "blocks."+i+".adaLN_modulation.1.weight");
//            ModeLoaderlUtils.loadData(jb.adaLN_modulation.bias, weightMap, "blocks."+i+".adaLN_modulation.1.bias");
//        }
//        
////        block.finalLayer.finalNorm.gamma = ModeLoaderlUtils.loadData(block.finalLayer.finalNorm.gamma, weightMap, 1, "final_layer.norm_final.weight"); 
//        ModeLoaderlUtils.loadData(block.finalLayer.finalLinear.weight, weightMap, "final_layer.linear.weight");
//        ModeLoaderlUtils.loadData(block.finalLayer.finalLinear.bias, weightMap, "final_layer.linear.bias");
//        ModeLoaderlUtils.loadData(block.finalLayer.m_linear1.weight, weightMap, "final_layer.adaLN_modulation_l1.weight");
//        ModeLoaderlUtils.loadData(block.finalLayer.m_linear1.bias, weightMap, "final_layer.adaLN_modulation_l1.bias");
//        ModeLoaderlUtils.loadData(block.finalLayer.m_linear2.weight, weightMap, "final_layer.adaLN_modulation_l2.weight");
//        ModeLoaderlUtils.loadData(block.finalLayer.m_linear2.bias, weightMap, "final_layer.adaLN_modulation_l2.bias");
        
    }
    
    public static void main(String[] args) {
//    	int N = 2;
//    	int C = 32;
//    	int H = 16;
//    	int W = 16;
//    	
//    	int TT = 1;
//    	int TEM = 768;
//    	
//    	int patchSize = 2;
//    	int hiddenSize = 384;
//    	int headNum = 6;
//    	int depth = 6;
//    	
//    	CNN nn = new CNN(null);
//        nn.CUDNN = true;
//        nn.number = N;
//    	
//        PixNerDiTMainMoudue jb = new PixNerDiTMainMoudue(C, W, H, patchSize, hiddenSize, headNum, depth, 1000, TEM, TT, 4, false, 0.0f, nn);
//    	
//        String weight = "D:\\models\\dit_s2.json";
//        loadWeight(LagJsonReader.readJsonFileBigWeightIterator(weight), jb, true);
//        
//	    String inputPath = "D:\\models\\c__temp_dit_x.json";
//	    Map<String, Object> datas = LagJsonReader.readJsonFileSmallWeight(inputPath);
//	    Tensor input = new Tensor(N, C, H, W, true);
//	    ModeLoaderlUtils.loadData(input, datas, "x");
//    	
//	    String cyPath = "D:\\models\\c__temp_dit_cy.json";
//	    Map<String, Object> cydatas = LagJsonReader.readJsonFileSmallWeight(cyPath);
//	    Tensor cy = new Tensor(N, 1, 1, TEM, true);
//	    ModeLoaderlUtils.loadData(cy, cydatas, "cy", 2);
//	    
//	    Tensor t = new Tensor(N, 1, 1, 1, new float[] {0.1f, 0.8f}, true);
//	    int time = (W / patchSize) * (H / patchSize);
//	    Tensor[] cs = RoPEKernel.getCosAndSin2D(time, hiddenSize, headNum);
//        Tensor cos = cs[0];
//        Tensor sin = cs[1];
//	    
//	    jb.forward(input, t, cy, cos, sin);
//	    
//	    jb.getOutput().showDM();
//	    
//	    jb.back(input, cos, sin);
	    
    }
}

