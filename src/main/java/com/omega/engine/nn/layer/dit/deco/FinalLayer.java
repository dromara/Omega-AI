package com.omega.engine.nn.layer.dit.deco;

import java.io.IOException;
import java.io.RandomAccessFile;

import com.omega.common.utils.RandomUtils;
import com.omega.engine.nn.layer.FullyLayer;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.normalization.BNType;
import com.omega.engine.nn.layer.normalization.LNLayer;
import com.omega.engine.nn.network.Network;
import com.omega.engine.tensor.Tensor;
import com.omega.engine.updater.UpdaterFactory;

/**
 * FinalLayer
 * 基础嵌入层，将输入通过线性投影映射到嵌入空间，可选归一化。
 * @author Administrator
 */
public class FinalLayer extends Layer {
    
    /**
     * 输入通道数 (in_chans)
     */
    private int inChannels;
    
    /**
     * 嵌入维度 (embed_dim)
     */
    private int embedDim;
    
    /**
     * 线性投影层 (proj)
     */
    public FullyLayer proj;
    
    /**
     * Layer 归一化层 (norm) - 可选
     */
    public LNLayer lnNorm;
    
    /**
     * 构造函数 - 带归一化
     *
     * @param inChannels 输入通道数 (对应最后一个维度 width)
     * @param embedDim 嵌入维度
     * @param bias 是否使用偏置
     * @param network 网络实例
     */
    public FinalLayer(int inChannels, int embedDim, boolean bias, Network network) {
        this.network = network;
        if (this.updater == null) {
            this.setUpdater(UpdaterFactory.create(network));
        }
        this.inChannels = inChannels;
        this.embedDim = embedDim;
        this.hasBias = bias;
        initLayers();
        // 保持 channel 和 height 不变，只改变 width (in_chans -> embed_dim)
        this.oChannel = 1;
        this.oHeight = 1;
        this.oWidth = embedDim;
    }
    
    /**
     * 初始化子层
     */
    private void initLayers() {
        // 线性投影层: in_chans -> embed_dim
        // FullyLayer 的输入维度是 width，输出维度是 oWidth
        this.proj = new FullyLayer(inChannels, embedDim, hasBias, network);
        RandomUtils.xavier_uniform(this.proj.weight, 1, inChannels, embedDim);
        if(this.proj.bias != null) {
        	this.proj.bias.clearGPU();
        }
        // 归一化层 (可选) - 根据 normType 选择不同的归一化层
        this.lnNorm = new LNLayer(1, 1, embedDim, true, BNType.fully_bn, network);
    }

    @Override
    public void init() {
        this.number = network.number;
    }
    
    /**
     * 根据输入初始化
     *
     * @param input 输入张量
     */
    public void init(Tensor input) {
        this.number = input.number;
    }

    @Override
    public void initBack() {

    }

    @Override
    public void initParam() {

    }

    @Override
    public void output() {
        // 归一化
        lnNorm.forward(input);
        
        // 2. 线性投影: [flattenedBatch, 1, 1, in_chans] -> [flattenedBatch, 1, 1, embed_dim]
        proj.forward(lnNorm.getOutput());
        
        // 使用 view 重塑输出: [batch*channel*height, 1, 1, embed_dim] -> [batch, channel, height, embed_dim]
        output = proj.getOutput();
    }

    @Override
    public Tensor getOutput() {
        return output;
    }

    @Override
    public void diff() {

        proj.back(delta);
        
        lnNorm.back(proj.diff);

        this.diff = lnNorm.diff;
    }

    @Override
    public void forward() {
        this.init();
        this.setInput();
        this.output();
    }

    @Override
    public void back() {
        this.initBack();
        this.setDelta();
        this.diff();
        if (this.network.GRADIENT_CHECK) {
            this.gradientCheck();
        }
    }

    @Override
    public void forward(Tensor input) {
        this.init(input);
        this.setInput(input);
        this.output();
    }

    @Override
    public void back(Tensor delta) {
        this.initBack();
        this.setDelta(delta);
        this.diff();
        if (this.network.GRADIENT_CHECK) {
            this.gradientCheck();
        }
    }

    @Override
    public void update() {
        lnNorm.update();
        proj.update();
    }

    @Override
    public void accGrad(float scale) {
        lnNorm.accGrad(scale);
        proj.accGrad(scale);
    }

    @Override
    public void showDiff() {
        // TODO Auto-generated method stub
    }

    @Override
    public LayerType getLayerType() {
        return LayerType.mlp;
    }

    @Override
    public float[][][][] output(float[][][][] input) {
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
    
    /**
     * 保存模型参数
     * 
     * @param outputStream 输出流
     * @throws IOException IO异常
     */
    public void saveModel(RandomAccessFile outputStream) throws IOException {
        lnNorm.saveModel(outputStream);
        proj.saveModel(outputStream);
    }
    
    /**
     * 加载模型参数
     * 
     * @param inputStream 输入流
     * @throws IOException IO异常
     */
    public void loadModel(RandomAccessFile inputStream) throws IOException {
        lnNorm.loadModel(inputStream, 1, 1, embedDim, BNType.fully_bn);
        proj.loadModel(inputStream);
    }
    
    // Getters
    
    public int getInChannels() {
        return inChannels;
    }
    
    public int getEmbedDim() {
        return embedDim;
    }
    
    public FullyLayer getProj() {
        return proj;
    }
    
    public LNLayer getLnNorm() {
        return lnNorm;
    }
}