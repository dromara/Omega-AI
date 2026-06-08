package com.omega.engine.nn.layer.dit.video.block;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.util.Map;

import com.omega.common.utils.RandomUtils;
import com.omega.engine.gpu.CUDAMemoryManager;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.active.SiLULayer;
import com.omega.engine.nn.network.CNN;
import com.omega.engine.nn.network.Network;
import com.omega.engine.tensor.Tensor;
import com.omega.engine.updater.UpdaterFactory;
import com.omega.example.common.ModeLoaderlUtils;
import com.omega.example.transformer.utils.LagJsonReader;

/**
 * GLUMBConv
 *
 * @author Administrator
 */
public class GLUMBConv extends Layer {
	
    private int embedDim = 0;
    private float mlp_ratio = 0;
    private int hidden_features;
    
    public int F;
    public int oDepth;

    public ConvLayer inverted_conv;
    public ConvLayer depth_conv;
    public ConvLayer point_conv;
    
    private SiLULayer glu_act;
    
    private Tensor input_t;
    
    private Tensor x;
    private Tensor gate;
    
    private Tensor wt;
    
    public int S;

    public GLUMBConv(int channel, int F, int H, int W, int embedDim, float mlp_ratio, Network network) {
        this.network = network;
        if (this.updater == null) {
            this.setUpdater(UpdaterFactory.create(network));
        }
        this.channel = channel;
        this.F = F;
        this.height = H;
        this.width = W;
        this.S = H * W;
        this.embedDim = embedDim;
        this.mlp_ratio = mlp_ratio;
        initLayers();
    }

    public static void loadWeight(Map<String, Object> weightMap, GLUMBConv block, boolean showLayers) {
        if (showLayers) {
            for (String key : weightMap.keySet()) {
                System.out.println(key);
            }
        }
        ModeLoaderlUtils.loadData(block.inverted_conv.weight, weightMap, "inverted_conv.conv.weight", 4);
        ModeLoaderlUtils.loadData(block.inverted_conv.bias, weightMap, "inverted_conv.conv.bias");
        ModeLoaderlUtils.loadData(block.depth_conv.weight, weightMap, "depth_conv.conv.weight", 4);
        ModeLoaderlUtils.loadData(block.depth_conv.bias, weightMap, "depth_conv.conv.bias");
        ModeLoaderlUtils.loadData(block.point_conv.weight, weightMap, "point_conv.conv.weight", 4);
    }
    
    public static void main(String[] args) {
    	
    	int N = 2;
        int hidden_size = 1152;
        int F = 3;
        int H = 11;
        int W = 20;
        
        String inputPath = "D:\\models\\ltx_vae\\glum_x.json";
        Map<String, Object> datas = LagJsonReader.readJsonFileSmallWeight(inputPath);
        Tensor input = new Tensor(N, F * H * W, 1, hidden_size, true);
        input.view(N * F, H * W, 1, hidden_size);
        ModeLoaderlUtils.loadData(input, datas, "input", 3);
        input.view(N, F * H * W, 1, hidden_size);
        
        CNN nn = new CNN(null);
        nn.CUDNN = true;
        nn.number = N;

        GLUMBConv conv = new GLUMBConv(hidden_size, F, H, W, hidden_size, 4, nn);

        String weight = "D:\\models\\ltx_vae\\glum_weight.json";
        loadWeight(LagJsonReader.readJsonFileSmallWeight(weight), conv, true);

        conv.forward(input);
        conv.getOutput().showDM();

        String deltaPath = "D:\\models\\ltx_vae\\glum_delta.json";
        Map<String, Object> d_datas = LagJsonReader.readJsonFileSmallWeight(deltaPath);
        Tensor delta = new Tensor(N, F * H * W, 1, hidden_size, true);
        delta.view(N * F, H * W, 1, hidden_size);
        ModeLoaderlUtils.loadData(delta, d_datas, "delta", 3);
        delta.view(N, F * H * W, 1, hidden_size);
        
        conv.back(delta);
        conv.diff.showDM("diff");
    	
    }

    @Override
    public void init() {
        // TODO Auto-generated method stub
        this.number = network.number;
        if (output == null || output.number != this.number) {
            input_t = Tensor.createGPUTensor(input_t, number * F, embedDim, height, width, true);
        	x = CUDAMemoryManager.getCache("clum_x", number * F, hidden_features, height, width);
        	gate = CUDAMemoryManager.getCache("clum_gate", number * F, hidden_features, height, width);
        	wt = CUDAMemoryManager.getCache("clum_wt", number * F, hidden_features, height, width);
//            x = Tensor.createGPUTensor(x, number * F, hidden_features, height, width, true);
//            gate = Tensor.createGPUTensor(gate, number * F, hidden_features, height, width, true);
//            wt = Tensor.createGPUTensor(wt, number * F, hidden_features, height, width, true);
            output = Tensor.createGPUTensor(output, this.number, F * height * width, 1, embedDim, true);
        }
    }

    public void init(Tensor input) {
        // TODO Auto-generated method stub
        this.number = input.number;
        if (output == null || output.number != this.number) {
        	input_t = Tensor.createGPUTensor(input_t, number * F, embedDim, height, width, true);
        	x = CUDAMemoryManager.getCache("clum_x", number * F, hidden_features, height, width);
        	gate = CUDAMemoryManager.getCache("clum_gate", number * F, hidden_features, height, width);
        	wt = CUDAMemoryManager.getCache("clum_wt", number * F, hidden_features, height, width);
//            x = Tensor.createGPUTensor(x, number * F, hidden_features, height, width, true);
//            gate = Tensor.createGPUTensor(gate, number * F, hidden_features, height, width, true);
//            wt = Tensor.createGPUTensor(wt, number * F, hidden_features, height, width, true);
            output = Tensor.createGPUTensor(output, this.number, F * height * width, 1, embedDim, true);
        }
    }

    public void initLayers() {
    	hidden_features = (int) (embedDim * mlp_ratio);
        this.inverted_conv = new ConvLayer(embedDim, hidden_features * 2, height, width, 1, 1, 1, 1, 0, true, true, network);
        RandomUtils.xavier_uniform(inverted_conv.weight, 1, embedDim, hidden_features * 2);
        if(inverted_conv.bias != null) {
        	inverted_conv.bias.clearGPU();
        }
        
        this.depth_conv = new ConvLayer(hidden_features * 2, hidden_features * 2, height, width, 3, 1, 1, hidden_features * 2, 1, false, true, network);
        RandomUtils.xavier_uniform(depth_conv.weight, 1, hidden_features * 2, hidden_features * 2);
        if(depth_conv.bias != null) {
        	depth_conv.bias.clearGPU();
        }
        
        this.glu_act = new SiLULayer(depth_conv);
        
        this.point_conv = new ConvLayer(hidden_features, embedDim, height, width, 1, 1, 1, 1, 0, false, false, network);
        RandomUtils.xavier_uniform(point_conv.weight, 1, hidden_features, embedDim);

        this.oChannel = channel;
        this.oHeight = 1;
        this.oWidth = embedDim;
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
    	
    	input = input.view(number * F, height, width, embedDim);
    	/**
    	 * x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
    	 */
    	Tensor_OP().permute(input, input_t, new int[] {number * F, height, width, embedDim}, new int[] {number * F, embedDim, height, width}, new int[] {0, 3, 1, 2});
    	
    	inverted_conv.forward(input_t);
    	
    	depth_conv.forward(inverted_conv.getOutput());

    	/**
    	 * x, gate = torch.chunk(x, 2, dim=1)
    	 */
    	Tensor_OP().getByChannel(depth_conv.getOutput(), x, new int[] {number * F, hidden_features * 2, height, width}, 0, hidden_features);
    	Tensor_OP().getByChannel(depth_conv.getOutput(), gate, new int[] {number * F, hidden_features * 2, height, width}, hidden_features, hidden_features);

    	glu_act.forward(gate);
    	
    	Tensor_OP().mul(x, glu_act.getOutput(), wt);

    	point_conv.forward(wt);
    	
    	Tensor_OP().permute(point_conv.getOutput(), output, new int[] {number * F, embedDim, height, width}, new int[] {number * F, height, width, embedDim}, new int[] {0, 2, 3, 1});
    	
    	input.viewOrg();
    }

    @Override
    public Tensor getOutput() {
        // TODO Auto-generated method stub
        return output;
    }

    @Override
    public void diff() {
        // TODO Auto-generated method stub
    	
    	/**
    	 * recompute x,gate,wt
    	 * x, gate = torch.chunk(x, 2, dim=1)
    	 */
//    	long src_pitch = hidden_features * 2  * height * width * Sizeof.FLOAT;
//    	JCuda.cudaMemcpy2DAsync(x.getGpuData(), src_pitch, depth_conv.getOutput().getGpuData().withByteOffset(src_pitch), S, embedDim, S, F, null);
    	Tensor_OP().getByChannel(depth_conv.getOutput(), x, new int[] {number * F, hidden_features * 2, height, width}, 0, hidden_features);
    	Tensor_OP().getByChannel(depth_conv.getOutput(), gate, new int[] {number * F, hidden_features * 2, height, width}, hidden_features, hidden_features);
    	Tensor_OP().mul(x, glu_act.getOutput(), wt);
    	
    	delta = delta.view(number * F, height, width, embedDim);
    	Tensor_OP().permute(delta, point_conv.getOutput(), new int[] {number * F, height, width, embedDim}, new int[] {number * F, embedDim, height, width}, new int[] {0, 3, 1, 2});
    	point_conv.back(point_conv.getOutput(), wt);

    	//wt = w2Delta
    	Tensor_OP().mul(point_conv.diff, glu_act.getOutput(), glu_act.getOutput()); 
    	Tensor_OP().getByChannel_back(depth_conv.getOutput(), glu_act.getOutput(), new int[] {number * F, hidden_features * 2, height, width}, 0, hidden_features);
    	
    	//wt = actDelta
    	Tensor_OP().mul(point_conv.diff, x, x); 
    	glu_act.back(x);
    	Tensor_OP().getByChannel_back(depth_conv.getOutput(), glu_act.diff, new int[] {number * F, hidden_features * 2, height, width}, hidden_features, hidden_features);
    	
    	depth_conv.back(depth_conv.getOutput(), inverted_conv.getOutput());
    	
    	inverted_conv.back(depth_conv.diff, input_t);

    	Tensor_OP().permute(inverted_conv.diff, delta, new int[] {number * F, embedDim, height, width}, new int[] {number * F, height, width, embedDim}, new int[] {0, 2, 3, 1});
    	delta.viewOrg();
    	this.diff = delta;
    }

    @Override
    public void forward() {
        // TODO Auto-generated method stub
        /**
         * 参数初始化
         */
        this.init();
        /**
         * 设置输入
         */
        this.setInput();
        /**
         * 计算输出
         */
        this.output();
    }

    @Override
    public void back() {
        // TODO Auto-generated method stub
    }

    @Override
    public void forward(Tensor input) {
        // TODO Auto-generated method stub
        /**
         * 参数初始化
         */
        this.init(input);
        /**
         * 设置输入
         */
        this.setInput(input);
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
    	inverted_conv.update();
    	depth_conv.update();
    	point_conv.update();
    }

    @Override
    public void showDiff() {
        // TODO Auto-generated method stub
    }

    @Override
    public LayerType getLayerType() {
        // TODO Auto-generated method stub
        return LayerType.clip_vision_embedding;
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
    	inverted_conv.saveModel(outputStream);
    	depth_conv.saveModel(outputStream);
    	point_conv.saveModel(outputStream);
    }

    public void loadModel(RandomAccessFile inputStream) throws IOException {
    	inverted_conv.loadModel(inputStream);
    	depth_conv.loadModel(inputStream);
    	point_conv.loadModel(inputStream);
    }

    @Override
    public void accGrad(float scale) {
        // TODO Auto-generated method stub
    }


}

