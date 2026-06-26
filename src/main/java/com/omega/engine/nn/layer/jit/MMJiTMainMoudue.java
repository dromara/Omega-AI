package com.omega.engine.nn.layer.jit;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import com.omega.common.utils.JsonUtils;
import com.omega.common.utils.MatrixUtils;
import com.omega.engine.gpu.BaseKernel;
import com.omega.engine.loss.MSELoss;
import com.omega.engine.nn.layer.FullyLayer;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.dit.mmjit.JiTCaptionEmbeddingLayer;
import com.omega.engine.nn.layer.dit.mmjit.MMJiTBlock;
import com.omega.engine.nn.layer.dit.mmjit.PlainTextBlock;
import com.omega.engine.nn.layer.gpu.RoPEKernel;
import com.omega.engine.nn.layer.normalization.BNType;
import com.omega.engine.nn.layer.normalization.RMSLayer;
import com.omega.engine.nn.network.CNN;
import com.omega.engine.nn.network.Network;
import com.omega.engine.nn.network.RunModel;
import com.omega.engine.tensor.Tensor;
import com.omega.engine.updater.UpdaterFactory;
import com.omega.example.common.ModeLoaderlUtils;
import com.omega.example.dit.models.ICPlan;
import com.omega.example.transformer.utils.LagJsonReader;

/**
 * MMJiTMainMoudue
 * @author Administrator
 */
public class MMJiTMainMoudue extends Layer {
	
	public int inChannel;
    public int width;
    public int height;
    public int patchSize;
    private int bottleneck_dim;
    private int hiddenSize;
    private int txt_depth;
    private int depth;
    private int headNum;
    private int textEmbedDim;
    private int maxContextLen;

    public BottleneckPatchEmbed patchEmbd;
    public JiTCaptionEmbeddingLayer labelEmbd;
    public List<PlainTextBlock> txt_blocks;
    public List<MMJiTBlock> blocks;
    public RMSLayer finalNorm;
    public FullyLayer finalLayer;
    
    private Tensor posEmbd;
    
    private Tensor tmp_cond;

    private float y_drop_prob = 0.0f;
    
    private int[] xShape;
    private int[] yShape;
    
    private BaseKernel baseKernel;
    
    public MMJiTMainMoudue(int inChannel, int width, int height, int patchSize, int bottleneck_dim, int hiddenSize, int headNum, int txt_depth, int depth, int textEmbedDim, int maxContextLen, float y_drop_prob, Network network) {
		this.network = network;
        if (this.updater == null) {
            this.setUpdater(UpdaterFactory.create(network));
        }
        this.y_drop_prob = y_drop_prob;
    	this.inChannel = inChannel;
		this.width = width;
		this.height = height;
		this.patchSize = patchSize;
		this.bottleneck_dim = bottleneck_dim;
		this.headNum = headNum;
		this.hiddenSize = hiddenSize;
		this.txt_depth = txt_depth;
		this.depth = depth;
		this.textEmbedDim = textEmbedDim;
		this.maxContextLen = maxContextLen;
		this.headNum = headNum;
		this.initLayers();
		this.oHeight = height;
		this.oWidth = width;
    }

    public void initLayers() {
    	
    	patchEmbd = new BottleneckPatchEmbed(inChannel, width, bottleneck_dim, hiddenSize, patchSize, true, network);
        
    	labelEmbd = new JiTCaptionEmbeddingLayer(textEmbedDim, hiddenSize, maxContextLen, y_drop_prob, false, network);

        txt_blocks = new ArrayList<PlainTextBlock>();
        
        blocks = new ArrayList<MMJiTBlock>();
         
        for(int i = 0;i<txt_depth;i++) {
        	PlainTextBlock block = new PlainTextBlock(hiddenSize, maxContextLen, headNum, true, false, network);
        	txt_blocks.add(block);
        }
        
        for(int i = 0;i<depth;i++) {
        	MMJiTBlock block = new MMJiTBlock(hiddenSize, headNum, patchEmbd.oChannel, maxContextLen, true, false, true, network);
	        blocks.add(block);
        }
        this.oChannel = inChannel;
        finalNorm = new RMSLayer(1, 1, hiddenSize, true, BNType.fully_bn, network);
        finalLayer = new FullyLayer(hiddenSize, patchSize * patchSize * oChannel, true, network);
        this.finalLayer.weight.clearGPU();
        if(this.finalLayer.bias != null) {
        	this.finalLayer.bias.clearGPU();
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
    
    public static float[] sincos2d(int embedDim, int grid) {
        int quarter = embedDim / 4;
        int n = grid * grid;

        float[] omega = new float[quarter];

        for (int i = 0; i < quarter; i++) {
            float power = (float) i / quarter;
            omega[i] = (float) (1.0 / Math.pow(10000.0, power));
        }

        float[] result = new float[n * embedDim];

        for (int row = 0; row < grid; row++) {
            for (int col = 0; col < grid; col++) {
                int idx = row * grid + col;
                int base = idx * embedDim;

                for (int d = 0; d < quarter; d++) {
                    float outX = col * omega[d];
                    float outY = row * omega[d];

                    result[base + d] = (float) Math.sin(outX);
                    result[base + quarter + d] = (float) Math.cos(outX);
                    result[base + 2 * quarter + d] = (float) Math.sin(outY);
                    result[base + 3 * quarter + d] = (float) Math.cos(outY);
                }
            }
        }

        return result;
    }
    
    public void init(Tensor input) {
        // TODO Auto-generated method stub
        this.number = input.number;
        if(this.output == null || this.output.number != number) {
        	output = Tensor.createGPUTensor(output, number, oChannel, oHeight, oWidth, true);
        }
        if(posEmbd == null) {
        	posEmbd = new Tensor(1, patchEmbd.oChannel, 1, hiddenSize, sincos2d(hiddenSize, width/patchSize), true);
        }
        if(patchEmbd.getOutput() != null){
        	patchEmbd.getOutput().viewOrg();
        }
    }
    
    @Override
    public void initBack() {
        // TODO Auto-generated method stub
    	if(tmp_cond == null || tmp_cond.number != number) {
    		tmp_cond = Tensor.createGPUTensor(output, number * maxContextLen, 1, 1, hiddenSize, true);
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
    
    public void output(Tensor label) {
    	
    }
    
    public void output(Tensor label, Tensor cos1d,Tensor sin1d, Tensor cos2d, Tensor sin2d) {

    	patchEmbd.forward(input);

    	Tensor_OP().addAxis(patchEmbd.getOutput(), posEmbd, patchEmbd.getOutput(), posEmbd.channel * posEmbd.width);

    	labelEmbd.forward(label);
    	
    	Tensor x = patchEmbd.getOutput().view(patchEmbd.getOutput().number * patchEmbd.getOutput().channel, 1, 1, patchEmbd.getOutput().width);
    	
    	Tensor cond = labelEmbd.getOutput();
    	
    	Tensor bc = cond;
    	for(int i = 0;i<txt_depth;i++) {
    		PlainTextBlock block = txt_blocks.get(i);
    		block.forward(bc, cos1d, sin1d);
    		bc = block.getOutput();
    	}
    	
    	Tensor bx = x;
    	for(int i = 0;i<depth;i++) {
    		MMJiTBlock block = blocks.get(i);
    		block.forward(bx, bc, cos1d, sin1d, cos2d, sin2d);
    		bx = block.getOutput();
    		bc = block.context_block.getOutput();
    	}

    	finalNorm.forward(bx);
    	finalLayer.forward(finalNorm.getOutput());
    	
    	/**
    	 * unpatchify
    	 * x: (N, T, patch_size**2 * C)
         * imgs: (N, C, H, W)
    	 */
    	if(xShape == null) {
    		int h = height/patchSize;
        	int w = width/patchSize;
        	xShape = new int[] {number, h, w, patchSize, patchSize, oChannel};
        	yShape = new int[] {number, oChannel, h, patchSize, w, patchSize};
    	}
    	Tensor_OP().permute(finalLayer.getOutput(), this.output, xShape, yShape, new int[] {0, 5, 1, 3, 2, 4});

    }

    @Override
    public Tensor getOutput() {
        // TODO Auto-generated method stub
        return output;
    }

    @Override
    public void diff() {

    }
    
    public void diff(Tensor cos1d, Tensor sin1d, Tensor cos2d, Tensor sin2d) {
        // TODO Auto-generated method stub
    	/**
    	 * unpatchify back
    	 */
//    	int h = height/patchSize;
//    	int w = width/patchSize;
//    	int[] yShape = new int[] {number, oChannel, h, patchSize, w, patchSize};
//    	int[] xShape = new int[] {number, h, w, patchSize, patchSize, oChannel};
    	Tensor_OP().permute(delta, finalLayer.getOutput(), yShape, xShape, new int[] {0, 2, 4, 3, 5, 1});
    	
    	finalLayer.back(finalLayer.getOutput());
    	finalNorm.back(finalLayer.diff);
    	
    	Tensor dy = finalNorm.diff;
    	Tensor dc = tmp_cond;
    	dc.clearGPU();
     	for(int i = depth - 1;i>=0;i--) {
     		MMJiTBlock block = blocks.get(i);
    		block.back(dy, dc, cos1d, sin1d, cos2d, sin2d);
    		dy = block.diff;
    		dc = block.context_block.diff;
    	}
     	
     	for(int i = txt_depth - 1;i>=0;i--) {
     		PlainTextBlock block = txt_blocks.get(i);
    		block.back(dc, cos1d, sin1d);
    		dc = block.diff;
    	}
     	
     	labelEmbd.back(dc);
     	
     	patchEmbd.back(dy);
     	
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

    }
    
    /**
     * 
     * @param input
     * @param tc time cond
     * @param text
     */
    public void forward(Tensor input,Tensor tc,Tensor text) {
        // TODO Auto-generated method stub

    }
    
    /**
     * 
     * @param input
     * @param tc time cond
     * @param text
     */
    public void forward(Tensor input,Tensor text, Tensor cos1d, Tensor sin1d, Tensor cos2d, Tensor sin2d) {
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
        this.output(text, cos1d, sin1d, cos2d, sin2d);
    }

    @Override
    public void back(Tensor delta) {
        // TODO Auto-generated method stub

    }
    
    public void back(Tensor delta, Tensor cos1d, Tensor sin1d, Tensor cos2d, Tensor sin2d) {
        // TODO Auto-generated method stub
        this.initBack();
        /**
         * 设置梯度

         */
        this.setDelta(delta);
        /**
         * 计算梯度
         */
        this.diff(cos1d, sin1d, cos2d, sin2d);
    }
    
    @Override
    public void update() {
        // TODO Auto-generated method stub
    	patchEmbd.update();

    	labelEmbd.update();
    	
    	for(int i = 0;i<txt_depth;i++) {
    		txt_blocks.get(i).update();
    	}
    	
    	for(int i = 0;i<depth;i++) {
    		blocks.get(i).update();
    	}
    	
    	finalNorm.update();
    	finalLayer.update();
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
    	patchEmbd.saveModel(outputStream);

    	labelEmbd.saveModel(outputStream);
    	
    	for(int i = 0;i<txt_depth;i++) {
    		txt_blocks.get(i).saveModel(outputStream);
    	}
    	
    	for(int i = 0;i<depth;i++) {
    		blocks.get(i).saveModel(outputStream);
    	}
    	
    	finalNorm.saveModel(outputStream);
    	finalLayer.saveModel(outputStream);
    }

    public void loadModel(RandomAccessFile inputStream) throws IOException {
    	patchEmbd.loadModel(inputStream);

    	labelEmbd.loadModel(inputStream);
    	
    	for(int i = 0;i<txt_depth;i++) {
    		txt_blocks.get(i).loadModel(inputStream);
    	}
    	
    	for(int i = 0;i<depth;i++) {
    		blocks.get(i).loadModel(inputStream);
    	}
    	
    	finalNorm.loadModel(inputStream);
    	finalLayer.loadModel(inputStream);
    }

    @Override
    public void accGrad(float scale) {
        // TODO Auto-generated method stub
    	patchEmbd.accGrad(scale);

    	labelEmbd.accGrad(scale);
    	
    	for(int i = 0;i<txt_depth;i++) {
    		txt_blocks.get(i).accGrad(scale);
    	}
    	
    	for(int i = 0;i<depth;i++) {
    		blocks.get(i).accGrad(scale);
    	}
    	
    	finalLayer.accGrad(scale);
    }
    
    public static void loadWeight(Map<String, Object> weightMap, MMJiTMainMoudue block, boolean showLayers) {
        if (showLayers) {
            for (String key : weightMap.keySet()) {
                System.out.println(key);
            }
        }
     
	    ModeLoaderlUtils.loadData(block.patchEmbd.proj1.weight, weightMap, "img_embed.proj1.weight");
	    ModeLoaderlUtils.loadData(block.patchEmbd.proj2.weight, weightMap, "img_embed.proj2.weight", 4);
	    ModeLoaderlUtils.loadData(block.patchEmbd.proj2.bias, weightMap, "img_embed.proj2.bias");
        
	    ModeLoaderlUtils.loadData(block.labelEmbd.linear1.weight, weightMap, "txt_embed.weight");
//	    block.labelEmbd.setY_embedding(new Tensor(1, 1, 30, 768, true));
//	    ModeLoaderlUtils.loadData(block.labelEmbd.getY_embedding(), weightMap, "mask_token", 4);
	    
	    for(int i = 0;i<block.txt_depth;i++) {
	    	PlainTextBlock bl = block.txt_blocks.get(i);
	    	
	    	bl.norm1.gamma = ModeLoaderlUtils.loadData(bl.norm1.gamma, weightMap, 1, "txt_blocks."+i+".norm1.weight"); 
	    	bl.norm2.gamma = ModeLoaderlUtils.loadData(bl.norm2.gamma, weightMap, 1, "txt_blocks."+i+".norm2.weight");
	    	
	    	ModeLoaderlUtils.loadData(bl.attn.qLinerLayer.weight, weightMap, "txt_blocks."+i+".q.weight");
	    	ModeLoaderlUtils.loadData(bl.attn.qLinerLayer.bias, weightMap, "txt_blocks."+i+".q.bias");
	    	ModeLoaderlUtils.loadData(bl.attn.kLinerLayer.weight, weightMap, "txt_blocks."+i+".k.weight");
	    	ModeLoaderlUtils.loadData(bl.attn.kLinerLayer.bias, weightMap, "txt_blocks."+i+".k.bias");
	    	ModeLoaderlUtils.loadData(bl.attn.vLinerLayer.weight, weightMap, "txt_blocks."+i+".v.weight");
	    	ModeLoaderlUtils.loadData(bl.attn.vLinerLayer.bias, weightMap, "txt_blocks."+i+".v.bias");
	    	ModeLoaderlUtils.loadData(bl.attn.oLinerLayer.weight, weightMap, "txt_blocks."+i+".proj.weight");
	    	ModeLoaderlUtils.loadData(bl.attn.oLinerLayer.bias, weightMap, "txt_blocks."+i+".proj.bias");
	    	
	    	ModeLoaderlUtils.loadData(bl.mlp.w12.weight, weightMap, "txt_blocks."+i+".mlp.w12.weight");
	    	ModeLoaderlUtils.loadData(bl.mlp.w3.weight, weightMap, "txt_blocks."+i+".mlp.w3.weight");
	    }
	    
	    for(int i = 0;i<block.depth;i++) {
	    	MMJiTBlock bl = block.blocks.get(i);
	    	
	    	bl.x_block.norm1.gamma = ModeLoaderlUtils.loadData(bl.x_block.norm1.gamma, weightMap, 1, "blocks."+i+".img_norm1.weight"); 
	    	bl.x_block.norm2.gamma = ModeLoaderlUtils.loadData(bl.x_block.norm2.gamma, weightMap, 1, "blocks."+i+".img_norm2.weight"); 
	    	bl.context_block.norm1.gamma = ModeLoaderlUtils.loadData(bl.context_block.norm1.gamma, weightMap, 1, "blocks."+i+".txt_norm1.weight"); 
	    	bl.context_block.norm2.gamma = ModeLoaderlUtils.loadData(bl.context_block.norm2.gamma, weightMap, 1, "blocks."+i+".txt_norm2.weight"); 
	    	
	    	ModeLoaderlUtils.loadData(bl.x_block.qLinerLayer.weight, weightMap, "blocks."+i+".img_q.weight");
	    	ModeLoaderlUtils.loadData(bl.x_block.qLinerLayer.bias, weightMap, "blocks."+i+".img_q.bias");
	    	ModeLoaderlUtils.loadData(bl.x_block.kLinerLayer.weight, weightMap, "blocks."+i+".img_k.weight");
	    	ModeLoaderlUtils.loadData(bl.x_block.kLinerLayer.bias, weightMap, "blocks."+i+".img_k.bias");
	    	ModeLoaderlUtils.loadData(bl.x_block.vLinerLayer.weight, weightMap, "blocks."+i+".img_v.weight");
	    	ModeLoaderlUtils.loadData(bl.x_block.vLinerLayer.bias, weightMap, "blocks."+i+".img_v.bias");
	    	ModeLoaderlUtils.loadData(bl.context_block.qLinerLayer.weight, weightMap, "blocks."+i+".txt_q.weight");
	    	ModeLoaderlUtils.loadData(bl.context_block.qLinerLayer.bias, weightMap, "blocks."+i+".txt_q.bias");
	    	ModeLoaderlUtils.loadData(bl.context_block.kLinerLayer.weight, weightMap, "blocks."+i+".txt_k.weight");
	    	ModeLoaderlUtils.loadData(bl.context_block.kLinerLayer.bias, weightMap, "blocks."+i+".txt_k.bias");
	    	ModeLoaderlUtils.loadData(bl.context_block.vLinerLayer.weight, weightMap, "blocks."+i+".txt_v.weight");
	    	ModeLoaderlUtils.loadData(bl.context_block.vLinerLayer.bias, weightMap, "blocks."+i+".txt_v.bias");
	    	
	    	ModeLoaderlUtils.loadData(bl.x_block.oLinerLayer.weight, weightMap, "blocks."+i+".img_proj.weight");
	    	ModeLoaderlUtils.loadData(bl.x_block.oLinerLayer.bias, weightMap, "blocks."+i+".img_proj.bias");
	    	ModeLoaderlUtils.loadData(bl.context_block.oLinerLayer.weight, weightMap, "blocks."+i+".txt_proj.weight");
	    	ModeLoaderlUtils.loadData(bl.context_block.oLinerLayer.bias, weightMap, "blocks."+i+".txt_proj.bias");
	    	
	    	ModeLoaderlUtils.loadData(bl.x_block.mlp.w12.weight, weightMap, "blocks."+i+".img_mlp.w12.weight");
	    	ModeLoaderlUtils.loadData(bl.x_block.mlp.w3.weight, weightMap, "blocks."+i+".img_mlp.w3.weight");
	    	ModeLoaderlUtils.loadData(bl.context_block.mlp.w12.weight, weightMap, "blocks."+i+".txt_mlp.w12.weight");
	    	ModeLoaderlUtils.loadData(bl.context_block.mlp.w3.weight, weightMap, "blocks."+i+".txt_mlp.w3.weight");
	    }
	    
	    block.finalNorm.gamma = ModeLoaderlUtils.loadData(block.finalNorm.gamma, weightMap, 1, "final_norm.weight"); 
    	ModeLoaderlUtils.loadData(block.finalLayer.weight, weightMap, "final.weight");
    	ModeLoaderlUtils.loadData(block.finalLayer.bias, weightMap, "final.bias");
    }
    
    public static void main(String[] args) {
    	
    	int N = 2;
    	int C = 3;
    	int H = 256;
    	int W = 256;
    	
    	int TT = 30;
    	int TEM = 768;
    	
    	int bottleneck_dim = 128;
    	int patchSize = 16;
    	int hiddenSize = 768;
    	int headNum = 12;
    	int depth = 3;
    	int text_preamble_depth = 2;
    	
    	CNN nn = new CNN(null);
        nn.CUDNN = true;
        nn.number = N;
        
        MSELoss lossfn = new MSELoss(nn);

        MMJiTMainMoudue jb = new MMJiTMainMoudue(C, W, H, patchSize, bottleneck_dim, hiddenSize, headNum, text_preamble_depth, depth, TEM, TT, 0.01f, nn);
    	
        String weight = "D:\\models\\mmjit.json";
        loadWeight(LagJsonReader.readJsonFileBigWeightIterator(weight), jb, true);
        
        int theta = 10000;

        Tensor[] cs1d = RoPEKernel.create1DRope(TT, 64, 0, theta);
        Tensor cos1d = cs1d[0];
        Tensor sin1d = cs1d[1];
        int time = (W / patchSize) * (W / patchSize);
        Tensor[] cs2d = RoPEKernel.create2DRope(headNum, time, 64, W / patchSize, theta);
        Tensor cos2d = cs2d[0];
        Tensor sin2d = cs2d[1];

//        String inputPath = "D:\\models\\mmjit_x.json";
//	    Map<String, Object> datas = LagJsonReader.readJsonFileSmallWeight(inputPath);
//	    Tensor input = new Tensor(N, C, H, W, true);
//        ModeLoaderlUtils.loadData(input, datas, "x", 4);
        
        String noisePath = "D:\\models\\mmjit_noise.json";
	    Map<String, Object> noiseDatas = LagJsonReader.readJsonFileSmallWeight(noisePath);
	    Tensor noise = new Tensor(N, C, H, W, true);
        ModeLoaderlUtils.loadData(noise, noiseDatas, "noise", 4);
    	
	    String cyPath = "D:\\models\\mmjit_txt.json";
	    Map<String, Object> cydatas = LagJsonReader.readJsonFileSmallWeight(cyPath);
	    Tensor txt = new Tensor(N, TT, 1, TEM, true);
	    ModeLoaderlUtils.loadData(txt, cydatas, "txt", 3);
	    txt.view(N * TT, 1, 1, TEM);
//	    
//	    String dinputPath = "D:\\models\\mmjit_delta.json";
//	    Map<String, Object> d_datas = LagJsonReader.readJsonFileSmallWeight(dinputPath);
//	    Tensor delta = new Tensor(N, C, H, W, true);
//	    ModeLoaderlUtils.loadData(delta, d_datas, "delta", 4);
//	    
//        txt.view(N * TT, 1, 1, TEM);
//        
//        Tensor t = new Tensor(N, 1, 1, 1, true);
//        Tensor rnd = new Tensor(N, 1, 1, 1, new float[] {0.2f, 0.8f}, true);
//        RandomUtils.gaussianRandomLogitNormal(t, rnd, -0.8f, 0.8f);
        
        ICPlan icplan = new ICPlan(nn.tensorOP);
        
//        /**
//         * latend add noise
//         */
//        Tensor x_t = new Tensor(N, C, H, W, true);
//        Tensor target = new Tensor(N, C, H, W, true);
//        Tensor v_pred = new Tensor(N, C, H, W, true);
//        
//        icplan.compute_z(input, t, noise, x_t);
//        icplan.compute_v(input, t, x_t, target, 5e-2f);
//        t.showDM("t");
//        x_t.showDM("x_t");
//        for(int i = 0;i<2;i++) {
//            jb.forward(x_t, txt, cos1d, sin1d, cos2d, sin2d);
//            
//            jb.getOutput().showDM("out");
//            
//            jb.getOutput().showDMByOffset(1 * 256 * 256 + 101 * 256, 256, "out_");
//            
//            icplan.compute_v(jb.getOutput(), t, x_t, v_pred, 5e-2f);
//            
//            /**
//             * loss
//             */
//            Tensor loss = lossfn.loss(v_pred, target);
//            float mse_loss = MatrixOperation.sum(loss.syncHost()) / N;
//            System.err.println(mse_loss);
//            /**
//             * loss diff
//             */
//            Tensor lossDiff = lossfn.diff(v_pred, target);
//            
//            /**
//             * dx_pred = delta / (1 - t).clamp_min(self.t_eps)
//             */
//            icplan.compute_dv(lossDiff, t, lossDiff, 5e-2f);
//            
//            /**
//             * back
//             */
//            jb.back(lossDiff, cos1d, sin1d, cos2d, sin2d);
//            
//        }
        
        int count = 100;
        
        Tensor x0 = new Tensor(N, C, H, W, true);
        Tensor t = new Tensor(N, 1, 1, 1, true);
        
        Tensor v = new Tensor(N, C, H, W, true);
        jb.Tensor_OP().mul(noise, 2, x0);
        float[] T = MatrixUtils.linspace(0, 1, count+1);
        System.err.println(JsonUtils.toJson(T));

        x0.showDM("x0");
        txt.showDM("txt");
        nn.RUN_MODEL = RunModel.EVAL;
		for(int i = 0;i<count;i++) {
			float t0 = T[i];
			float t1 = T[i + 1];
			float dt = t1 - t0;
			MatrixUtils.val(t.data, t0);
			t.hostToDevice();
			jb.forward(x0, txt, cos1d, sin1d, cos2d, sin2d);
		    Tensor pred_x0 = jb.getOutput();
		    icplan.compute_v(pred_x0, t, x0, v, 5e-2f);
		    jb.Tensor_OP().mul(v, dt, v);
		    jb.Tensor_OP().add(x0, v, x0);
		}
        x0.showDM("out");
    }
}

