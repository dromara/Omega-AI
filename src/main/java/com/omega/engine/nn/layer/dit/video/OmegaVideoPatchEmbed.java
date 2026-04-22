package com.omega.engine.nn.layer.dit.video;

import java.io.IOException;
import java.io.RandomAccessFile;

import com.omega.common.utils.RandomUtils;
import com.omega.engine.nn.layer.ConvolutionLayer;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.network.Network;
import com.omega.engine.tensor.Tensor;
import com.omega.engine.updater.UpdaterFactory;

/**
 * OmegaVideoPatchEmbed
 *
 * @author Administrator
 */
public class OmegaVideoPatchEmbed extends Layer {
	
    private int embedDim = 0;
    
    public int F;
    public int oDepth;

    public ConvolutionLayer patchEmbedding;
    
    private int[] shape;
    private int[] t_shape;
    
    private Tensor input_t;
    
    public int S;

    public OmegaVideoPatchEmbed(int channel, int F, int H, int W, int embedDim, int patchSize, boolean bias, Network network) {
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
        initLayers(channel, F, H, W, patchSize, bias);
    }

    public static void main(String[] args) {
//        int batchSize = 2;
//        int embedDim = 6;
//        int imgSize = 224;
//        int patchSize = 32;
//        Transformer tf = new Transformer();
//        tf.number = batchSize;
//        OmegaVideoPatchEmbed layer = new OmegaVideoPatchEmbed(3, embedDim, imgSize, patchSize, false, tf);
//        float[] data = RandomUtils.order(batchSize * 3 * imgSize * imgSize, 1f, 0f);
//        Tensor input = new Tensor(batchSize, 3, imgSize, imgSize, data, true);
//        layer.forward(input);
//        layer.getOutput().showShape();
//        layer.getOutput().showDM();
    }

    @Override
    public void init() {
        // TODO Auto-generated method stub
        this.number = network.number;
        if (output == null || output.number != this.number) {
            int pChannel = this.getPatchEmbedding().oHeight * this.getPatchEmbedding().oWidth;
            input_t = Tensor.createGPUTensor(input_t, this.number * F, channel, height, width, true);
            output = Tensor.createGPUTensor(output, this.number, F * pChannel, 1, embedDim, true);
        }
    }

    public void init(Tensor input) {
        // TODO Auto-generated method stub
        this.number = input.number;
        if (output == null || output.number != this.number) {
            int pChannel = this.getPatchEmbedding().oHeight * this.getPatchEmbedding().oWidth;
            input_t = Tensor.createGPUTensor(input_t, this.number * F, channel, height, width, true);
            output = Tensor.createGPUTensor(output, this.number, F * pChannel, 1, embedDim, true);
            shape = new int[] {number, F, this.getPatchEmbedding().oChannel, pChannel};
            t_shape = new int[] {number, F, pChannel , this.getPatchEmbedding().oChannel};
        }
    }

    public void initLayers(int inChannel, int F, int height, int width, int patchSize, boolean bias) {
        this.patchEmbedding = new ConvolutionLayer(inChannel, embedDim, height, width, patchSize, patchSize, 0, patchSize, bias, network);
        patchEmbedding.PROPAGATE_DOWN = false;
        RandomUtils.xavier_uniform(patchEmbedding.weight, 1, inChannel * patchSize * patchSize, embedDim * patchSize * patchSize);
//        this.patchEmbedding.weight.setData(RandomUtils.xavierUniform(this.patchEmbedding.weight.dataLength, inChannel * patchSize * patchSize, embedDim * patchSize * patchSize, 1));
        if(this.patchEmbedding.bias != null) {
        	this.patchEmbedding.bias.clearGPU();
        }
        int pChannel = this.getPatchEmbedding().oHeight * this.getPatchEmbedding().oWidth;
        this.oChannel = pChannel;
        this.oDepth = F;
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
    	/**
    	 *  x = rearrange(x, "B C T H W -> (B T) C H W")
        	x = self.x_embedder(x)  # (B, N, D) 1152
        	x = rearrange(x, "(B T) S C -> B (T S) C", B=B, T=T, S=S) # [B, 3 * 11 * 20, 1152] = [B, 660, 1152]
    	 */
    	Tensor_OP().permute(input, input_t, new int[] {number, channel, F, height, width}, new int[] {number, F, channel , height, width}, new int[] {0, 2, 1, 3, 4});
        getPatchEmbedding().forward(input_t);
        Tensor_OP().permute(getPatchEmbedding().getOutput(), output, shape, t_shape, new int[]{0, 1, 3, 2});
    }

    @Override
    public Tensor getOutput() {
        // TODO Auto-generated method stub
        return output;
    }

    @Override
    public void diff() {
        // TODO Auto-generated method stub
    	Tensor_OP().permute(delta, getPatchEmbedding().getOutput(), t_shape, shape, new int[]{0, 1, 3, 2});
    	getPatchEmbedding().back(getPatchEmbedding().getOutput());
    	this.diff =  getPatchEmbedding().diff;
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
    	getPatchEmbedding().update();
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
    	getPatchEmbedding().saveModel(outputStream);
    }

    public void loadModel(RandomAccessFile inputStream) throws IOException {
    	getPatchEmbedding().loadModel(inputStream);
    }

    @Override
    public void accGrad(float scale) {
        // TODO Auto-generated method stub
    	getPatchEmbedding().accGrad(scale);
    }

    public ConvolutionLayer getPatchEmbedding() {
        return patchEmbedding;
    }

}

