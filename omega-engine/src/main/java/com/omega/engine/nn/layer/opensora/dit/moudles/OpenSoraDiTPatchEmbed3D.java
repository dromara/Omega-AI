package com.omega.engine.nn.layer.opensora.dit.moudles;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.util.Map;

import com.omega.engine.nn.layer.Convolution3DLayer;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.network.Network;
import com.omega.engine.nn.network.Transformer;
import com.omega.engine.tensor.Tensor;
import com.omega.engine.updater.UpdaterFactory;
import com.omega.example.common.ModeLoaderlUtils;
import com.omega.example.transformer.utils.LagJsonReader;

/**
 * DiTPatchEmbed3D
 *
 * @author Administrator
 */
public class OpenSoraDiTPatchEmbed3D extends Layer {
	
    private int embedDim = 0;
    public int depth = 0;
    public int oDepth = 0;
    public int[] patchSize;
    
    public Convolution3DLayer patchEmbedding;
    
    private int[] shape;
    private int[] t_shape;

    public OpenSoraDiTPatchEmbed3D(int channel, int embedDim, int depth, int imageSize, int[] patchSize, boolean bias, Network network) {
        this.network = network;
        if (this.updater == null) {
            this.setUpdater(UpdaterFactory.create(network));
        }
        this.channel = channel;
        this.depth = depth;
        this.height = imageSize;
        this.width = imageSize;
        this.embedDim = embedDim;
        initLayers(channel, imageSize, imageSize, patchSize, bias);

    }

    public static void loadWeight(Map<String, Object> weightMap, OpenSoraDiTPatchEmbed3D block, boolean showLayers) {
        if (showLayers) {
            for (String key : weightMap.keySet()) {
                System.out.println(key);
            }
        }
        
        ModeLoaderlUtils.loadData(block.patchEmbedding.weight, weightMap, "proj.weight", 5);
        ModeLoaderlUtils.loadData(block.patchEmbedding.bias, weightMap, "proj.bias");
//        block.patchEmbedding.bias.showDM("bias");
    }
    
    public static void main(String[] args) {
        int N = 2;
        int C = 4;
        int T = 16;
        int H = 32;
        int W = 32;
        int embedDim = 64;
        int[] patchSize = new int[] {1, 2, 2};
        Transformer tf = new Transformer();
        tf.number = N;
        tf.CUDNN = true;
        OpenSoraDiTPatchEmbed3D layer = new OpenSoraDiTPatchEmbed3D(C, embedDim, T, W, patchSize, true, tf);
        
        String weight = "D:\\models\\PatchEmbed3D.json";
        loadWeight(LagJsonReader.readJsonFileBigWeightIterator(weight), layer, true);
        
        String inputPath = "D:\\models\\sora_x.json";
        Map<String, Object> datas = LagJsonReader.readJsonFileSmallWeight(inputPath);
        Tensor input = new Tensor(N, C * T, H, W, true);
        ModeLoaderlUtils.loadData(input, datas, "x", 5);
        
        layer.forward(input);
        layer.getOutput().showShape();
        layer.getOutput().showDM();
        
        String dxPath = "D:\\models\\sora_dx.json";
        Map<String, Object> dxDatas = LagJsonReader.readJsonFileSmallWeight(dxPath);
        Tensor dx = new Tensor(N, layer.getOutput().channel, layer.getOutput().height, layer.getOutput().width, true);
        ModeLoaderlUtils.loadData(dx, dxDatas, "dx", 3);
        
        layer.back(dx);
        layer.diff.showShape();
        layer.diff.showDM();
        
    }

    @Override
    public void init() {
        // TODO Auto-generated method stub
        this.number = network.number;
        if (output == null || output.number != this.number) {
        	int pChannel = patchEmbedding.oDepth * patchEmbedding.oHeight * patchEmbedding.oWidth;
            output = Tensor.createGPUTensor(output, this.number, pChannel, 1, embedDim, true);
            shape = new int[] {number, patchEmbedding.oChannel, 1, pChannel};
            t_shape = new int[] {number, pChannel, 1, patchEmbedding.oChannel};
        }
    }

    public void init(Tensor input) {
        // TODO Auto-generated method stub
        this.number = input.number;
        if (output == null || output.number != this.number) {
            int pChannel = patchEmbedding.oDepth * patchEmbedding.oHeight * patchEmbedding.oWidth;
            output = Tensor.createGPUTensor(output, this.number, pChannel, 1, embedDim, true);
            shape = new int[] {number, patchEmbedding.oChannel, 1, pChannel};
            t_shape = new int[] {number, pChannel, 1, patchEmbedding.oChannel};
        }
    }

    public void initLayers(int inChannel, int height, int width, int[] patchSize, boolean bias) {
    	int[] padding = new int[] {0, 0, 0};
        this.patchEmbedding = new Convolution3DLayer(inChannel, embedDim, depth, width, height, patchSize[0], patchSize[1], patchSize[2], padding, patchSize, bias, network);
//        this.patchEmbedding.weight.setData(RandomUtils.xavierUniform(this.patchEmbedding.weight.dataLength, inChannel * patchSize[0] * patchSize[1] * patchSize[2], embedDim * patchSize[0] * patchSize[1] * patchSize[2], 1));
//        if(this.patchEmbedding.bias != null) {
//        	this.patchEmbedding.bias.clearGPU();
//        }
        int pChannel = patchEmbedding.oDepth * patchEmbedding.oHeight * patchEmbedding.oWidth;
        this.oChannel = pChannel;
        this.oDepth = patchEmbedding.oDepth;
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
//    	input.showDM("input");
    	patchEmbedding.forward(this.input);
//    	patchEmbedding.getOutput().showDM("patchEmbedding");
        Tensor_OP().permute(patchEmbedding.getOutput(), output, shape, t_shape, new int[]{0, 3, 2, 1});
    }

    @Override
    public Tensor getOutput() {
        // TODO Auto-generated method stub
        return output;
    }

    @Override
    public void diff() {
        // TODO Auto-generated method stub
    	Tensor_OP().permute(delta, patchEmbedding.getOutput(), t_shape, shape, new int[]{0, 3, 2, 1});
    	patchEmbedding.back(patchEmbedding.getOutput());
    	this.diff =  patchEmbedding.diff;
//    	diff.showDM("dx");
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
    	patchEmbedding.update();
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
    	patchEmbedding.saveModel(outputStream);
    }

    public void loadModel(RandomAccessFile inputStream) throws IOException {
    	patchEmbedding.loadModel(inputStream);
    }

    @Override
    public void accGrad(float scale) {
        // TODO Auto-generated method stub
    	patchEmbedding.accGrad(scale);
    }

}

