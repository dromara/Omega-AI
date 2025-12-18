package com.omega.engine.nn.layer.dit.deco;

import java.io.IOException;
import java.io.RandomAccessFile;

import com.omega.engine.nn.layer.FullyLayer;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.normalization.BNType;
import com.omega.engine.nn.layer.normalization.LNLayer;
import com.omega.engine.nn.layer.normalization.RMSLayer;
import com.omega.engine.nn.network.Network;
import com.omega.engine.tensor.Tensor;
import com.omega.engine.updater.UpdaterFactory;

/**
 * Embed Layer
 *
 * 基础嵌入层，将输入通过线性投影映射到嵌入空间，可选归一化。
 *
 * @author Administrator
 */
public class EmbedLayer extends Layer {
    
    /**
     * 输入通道数 (in_chans)
     */
    private int inChannels;
    
    /**
     * 嵌入维度 (embed_dim)
     */
    private int embedDim;
    
    /**
     * 是否使用归一化层
     */
    private boolean useNorm;
    
    /**
     * 归一化类型: "rms", "ln", "layer_norm" 或 null
     */
    private String normType;
    
    /**
     * 线性投影层 (proj)
     */
    public FullyLayer proj;
    
    /**
     * RMS 归一化层 (norm) - 可选
     */
    public RMSLayer rmsNorm;
    
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
     * @param normType 归一化类型:
     *                 - null: 不使用归一化 (Identity)
     *                 - "rms": RMS Normalization
     *                 - "ln" 或 "layer_norm": Layer Normalization
     * @param network 网络实例
     */
    public EmbedLayer(int inChannels, int embedDim, boolean bias, String normType, Network network) {
        this.network = network;
        if (this.updater == null) {
            this.setUpdater(UpdaterFactory.create(network));
        }
        this.inChannels = inChannels;
        this.embedDim = embedDim;
        this.normType = normType;
        this.useNorm = normType != null && !normType.isEmpty();
        this.hasBias = bias;
        initLayers();
        // 设置输出维度 - 只有最后一个维度 (width) 改变
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
        
        // 归一化层 (可选) - 根据 normType 选择不同的归一化层
        // 使用简单构造函数，让归一化层在 forward 时通过 init(Tensor input) 自动获取输入形状
        if (useNorm) {
            if ("rms".equalsIgnoreCase(normType)) {
                // RMS Normalization
                // 不指定 channel, height, width，让 RMSLayer 在 forward 时自动从输入获取
                this.rmsNorm = new RMSLayer(network);
            } else if ("ln".equalsIgnoreCase(normType) || "layer_norm".equalsIgnoreCase(normType)) {
                // Layer Normalization
                // 不指定 channel, height, width，让 LNLayer 在 forward 时自动从输入获取
                this.lnNorm = new LNLayer(network);
            }
        }
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
        // 线性投影: [flattenedBatch, 1, 1, in_chans] -> [flattenedBatch, 1, 1, embed_dim]
        proj.forward(input);

        // 归一化 (可选)
        Tensor normOutput;
        if (useNorm) {
            if (rmsNorm != null) {
                rmsNorm.forward(proj.getOutput());
                normOutput = rmsNorm.getOutput();
            } else if (lnNorm != null) {
                lnNorm.forward(proj.getOutput());
                normOutput = lnNorm.getOutput();
            } else {
                // 没有匹配的归一化类型，使用 Identity
                normOutput = proj.getOutput();
            }
        } else {
            // 不使用归一化 (Identity)
            normOutput = proj.getOutput();
        }

        output = normOutput;
    }

    @Override
    public Tensor getOutput() {
        return output;
    }

    @Override
    public void diff() {

        Tensor backDelta;
        if (useNorm) {
            if (rmsNorm != null) {
                // 反向传播: rmsNorm -> proj
                rmsNorm.back(delta);
                backDelta = rmsNorm.diff;
            } else if (lnNorm != null) {
                // 反向传播: lnNorm -> proj
                lnNorm.back(delta);
                backDelta = lnNorm.diff;
            } else {
                // 没有归一化层 (Identity)
                backDelta = delta;
            }
        } else {
            // 不使用归一化 (Identity)
            backDelta = delta;
        }
        
        // 2. 通过 proj 反向传播
        proj.back(backDelta);

        this.diff = proj.diff;
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
        proj.update();
        if (useNorm) {
            if (rmsNorm != null) {
                rmsNorm.update();
            } else if (lnNorm != null) {
                lnNorm.update();
            }
        }
    }

    @Override
    public void accGrad(float scale) {
        proj.accGrad(scale);
        if (useNorm) {
            if (rmsNorm != null) {
                rmsNorm.accGrad(scale);
            } else if (lnNorm != null) {
                lnNorm.accGrad(scale);
            }
        }
    }

    @Override
    public void showDiff() {
        // TODO Auto-generated method stub
    }

    @Override
    public LayerType getLayerType() {
        return LayerType.embed;
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
        proj.saveModel(outputStream);
        if (useNorm) {
            if (rmsNorm != null) {
                rmsNorm.saveModel(outputStream);
            } else if (lnNorm != null) {
                lnNorm.saveModel(outputStream);
            }
        }
    }
    
    /**
     * 加载模型参数
     * 
     * @param inputStream 输入流
     * @throws IOException IO异常
     */
    public void loadModel(RandomAccessFile inputStream) throws IOException {
        proj.loadModel(inputStream);
        if (useNorm) {
            if (rmsNorm != null) {
                rmsNorm.loadModel(inputStream, 1, 1, embedDim, BNType.fully_bn);
            } else if (lnNorm != null) {
                lnNorm.loadModel(inputStream, 1, 1, embedDim, BNType.fully_bn);
            }
        }
    }
    
    // Getters
    
    public int getInChannels() {
        return inChannels;
    }
    
    public int getEmbedDim() {
        return embedDim;
    }
    
    public boolean isUseNorm() {
        return useNorm;
    }
    
    public String getNormType() {
        return normType;
    }
    
    public FullyLayer getProj() {
        return proj;
    }
    
    public RMSLayer getRmsNorm() {
        return rmsNorm;
    }
    
    public LNLayer getLnNorm() {
        return lnNorm;
    }
}