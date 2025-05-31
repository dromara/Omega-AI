package com.omega.engine.nn.layer.normalization;

import com.omega.common.utils.MatrixUtils;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.normalization.gpu.RMSKernel;
import com.omega.engine.nn.model.LayerInit;
import com.omega.engine.nn.network.Network;
import com.omega.engine.nn.network.utils.ModelUtils;
import com.omega.engine.tensor.Tensor;
import com.omega.engine.updater.UpdaterFactory;

import java.io.IOException;
import java.io.RandomAccessFile;

/**
 * RMS Normalization Layer
 *
 * @author Administrator
 * <p>
 * <p>
 * <p>
 * mean = ∑x / m
 * <p>
 * std = (∑(x - mean)^2 / m)^1/2
 * <p>
 * zi = (xi - mean) / std
 * <p>
 * yi = gama * zi + beta
 */
public class RMSLayer extends NormalizationLayer {
    public BNType bnType = null;
    public RMSKernel kernel;
    /**
     * if prelayer is conv layer meanNum = batchSize * channel
     * <p>
     * mean dims = H * W
     * <p>
     * else if prelayer is fully layer meanNum = batchSize * channel
     * <p>
     * mean dims = W
     */
    private int meanNum = 0;

    public RMSLayer() {
        // initParam();
        this.hasParams = true;
    }

    public RMSLayer(boolean hasBias) {
        // initParam();
        this.hasBias = true;
        this.hasParams = true;
    }

    public RMSLayer(Layer preLayer) {
        this.setPreLayer(preLayer);
        this.hasParams = false;
        this.setUpdater(UpdaterFactory.create(this.network));
    }

    public RMSLayer(Layer preLayer, boolean hasBias) {
        this.setPreLayer(preLayer);
        this.hasBias = false;
        this.hasParams = false;
        this.setUpdater(UpdaterFactory.create(this.network));
    }

    public RMSLayer(Network network) {
        this.network = network;
        network.paramLayers.add(this);
        this.setUpdater(UpdaterFactory.create(this.network));
    }

    public RMSLayer(Network network, boolean hasBias) {
        this.network = network;
        this.hasBias = false;
        network.paramLayers.add(this);
        this.setUpdater(UpdaterFactory.create(this.network));
    }

    @Override
    public void init() {
        this.number = this.network.number;
        if (preLayer == null) {
            preLayer = this.network.getPreLayer(this.index);
        }
        if (this.bnType == null) {
            if (preLayer != null) {
                this.channel = preLayer.oChannel;
                this.height = preLayer.oHeight;
                this.width = preLayer.oWidth;
                this.oChannel = this.channel;
                this.oHeight = this.height;
                this.oWidth = this.width;
                if (this.preLayer.getLayerType() == LayerType.conv) {
                    this.setBnType(BNType.conv_bn);
                    this.meanNum = this.height * this.width;
                } else if (this.preLayer.getLayerType() == LayerType.full) {
                    this.setBnType(BNType.fully_bn);
                    this.meanNum = this.channel * this.height * this.width;
                } else if (this.preLayer.getLayerType() == LayerType.conv_transpose) {
                    this.setBnType(BNType.conv_bn);
                    this.meanNum = this.height * this.width;
                } else {
                    this.setBnType(BNType.fully_bn);
                    this.meanNum = this.channel * this.height * this.width;
                }
            } else {
                this.setBnType(BNType.fully_bn);
                this.meanNum = this.channel * this.height * this.width;
            }
        }
        if (kernel == null) {
            kernel = new RMSKernel(width, bnType, cuda());
        }
        if (this.gamma == null) {
            this.gamma = new Tensor(1, 1, 1, meanNum, MatrixUtils.one(this.meanNum), true);
            this.diffGamma = new Tensor(1, 1, 1, meanNum, true);
        }
        if (this.output == null || this.number != this.output.number) {
            this.output = Tensor.createGPUTensor(this.output, number, oChannel, oHeight, oWidth, true);
        }
    }

    public void init(Tensor input) {
        this.number = input.number;
        if (this.bnType == null) {
            this.channel = input.channel;
            this.height = input.height;
            this.width = input.width;
            this.oChannel = this.channel;
            this.oHeight = this.height;
            this.oWidth = this.width;
            this.setBnType(BNType.fully_bn);
            this.meanNum = this.channel * this.height * this.width;
        }
        if (kernel == null) {
            kernel = new RMSKernel(width, bnType, cuda());
        }
        if (this.gamma == null) {
            this.gamma = new Tensor(1, 1, 1, meanNum, MatrixUtils.one(this.meanNum), true);
            this.diffGamma = new Tensor(1, 1, 1, meanNum, true);
        }
        if (this.output == null || this.number != this.output.number) {
            this.output = Tensor.createGPUTensor(this.output, number, oChannel, oHeight, oWidth, true);
        }
    }

    @Override
    public void initParam() {
        // TODO Auto-generated method stub
    }

    @Override
    public void initBack() {
        if (this.diff == null) {
            this.diff = new Tensor(input.number, input.channel, input.height, input.width, true, true);
        }
        if (this.diffGamma == null) {
            this.diffGamma = new Tensor(1, 1, 1, meanNum, true);
        }
    }

    public void initBack(Tensor diff) {
        if (this.diff == null) {
            this.diff = new Tensor(diff.number, diff.channel, diff.height, diff.width, true, true);
        }
        if (this.diffGamma == null) {
            this.diffGamma = new Tensor(1, 1, 1, meanNum, true);
        }
    }

    @Override
    public void output() {
        // TODO Auto-generated method stub
        // System.out.println(this.index+":"+input.number+":"+input.channel+":"+input.height+":"+input.width);
        // System.out.println(this.index+":"+output.number+":"+output.channel+":"+output.height+":"+output.width);
        // System.out.println(JsonUtils.toJson(gamma.shape()));
        // System.out.println(JsonUtils.toJson(beta.shape()));
        // kernel.forward(gamma, beta, input, output);
        // kernel.forwardAten(gamma, beta, input, output);
        kernel.forward3(gamma, input, output);
        // System.err.println("1:");
        // output.showDMByNumber(0);
        // System.err.println("2:");
        // output2.showDMByNumber(0);
        //
        // System.out.println("bn-output:");
        // output.showDM();
    }

    @Override
    public Tensor getOutput() {
        // TODO Auto-generated method stub
        return this.output;
    }

    @Override
    public void forward() {
        // TODO Auto-generated method stub
        /**
         * 参数初始化
         *
         */
        this.init();
        /**
         * 设置输入
         *
         */
        this.setInput();
        /**
         * 计算输出
         *
         */
        this.output();
    }

    /**
     * 原论文公式
     * <p>
     * deltaGama = ∑ deta * z
     * <p>
     * deltaBeta = ∑ deta
     * <p>
     * dxhat = deta * gama
     * <p>
     * dvar = ∑ dxhat * (xi - mean) * -1/2 * (var + eta)^-3/2
     * <p>
     * dmean = ∑ dxhat * -1 / (var + eta)^1/2 + dvar * (∑ -2 * (x - mean)) / n
     * <p>
     * dx = dxhat * 1 / (var + eta)^1/2 + dvar * 2(x - mean) / n + dmean * 1/n
     * <p>
     * darknet公式
     * <p>
     * dmean = (∑ dxhat * -1 / (var + eta)^1/2)
     */
    @Override
    public void diff() {
        // System.out.println(delta);
        // long start = System.nanoTime();
        // System.out.println(index);
        // kernel.backward(input, delta, diff, gamma, diffGamma, diffBeta);
        // kernel.backwardAten(input, delta, diff, gamma, diffGamma, diffBeta);
        kernel.backward3(input, delta, diff, gamma, diffGamma);
        // diff.showDMByNumber(0);
        // diff2.showDMByNumber(0);
        // System.out.println((System.nanoTime() - start) / 1e6 + "ms.");
    }

    @Override
    public void back() {
        // TODO Auto-generated method stub
        this.initBack();
        /**
         * 设置梯度
         *
         */
        this.setDelta();
        /**
         * 计算梯度
         *
         */
        this.diff();
        if (this.network.GRADIENT_CHECK) {
            this.gradientCheck();
        }
    }

    @Override
    public void update() {
        // TODO Auto-generated method stub
        if (!this.freeze) {
            if (accDW != null) {
                this.accDW.copy(diffGamma);
            }
            if (this.updater != null) {
                this.updater.updateForBN(this);
            } else {
                for (int i = 0; i < this.gamma.dataLength; i++) {
                    this.gamma.data[i] -= this.learnRate * this.diffGamma.data[i];
                }
            }
            this.clearAccGrad();
        }
    }

    @Override
    public void showDiff() {
        // TODO Auto-generated method stub
    }

    @Override
    public LayerType getLayerType() {
        // TODO Auto-generated method stub
        return LayerType.rms_norm;
    }

    @Override
    public LayerInit save() {
        // TODO Auto-generated method stub
        return null;
    }

    @Override
    public float[][][][] output(float[][][][] input) {
        // TODO Auto-generated method stub
        return null;
    }

    public BNType getBnType() {
        return bnType;
    }

    public void setBnType(BNType bnType) {
        this.bnType = bnType;
    }

    @Override
    public void initCache() {
        // TODO Auto-generated method stub
    }

    @Override
    public void forward(Tensor input) {
        // TODO Auto-generated method stub
        /**
         * 参数初始化
         *
         */
        this.init(input);
        /**
         * 设置输入
         *
         */
        this.setInput(input);
        /**
         * 计算输出
         *
         */
        this.output();
    }

    @Override
    public void back(Tensor delta) {
        // TODO Auto-generated method stub
        this.initBack(delta);
        /**
         * 设置梯度
         *
         */
        this.setDelta(delta);
        /**
         * 计算梯度
         *
         */
        this.diff();
        if (this.network.GRADIENT_CHECK) {
            this.gradientCheck();
        }
    }

    @Override
    public void backTemp() {
        // TODO Auto-generated method stub
    }

    public void saveModel(RandomAccessFile outputStream) throws IOException {
        ModelUtils.saveParams(outputStream, gamma);
    }

    public void loadModel(RandomAccessFile inputStream) throws IOException {
        init();
        ModelUtils.loadParams(inputStream, gamma);
    }

    public void putParamters() {
        init();
        this.network.addPamamter(gamma);
    }

    public void putParamterGrads() {
        this.network.addDeltaParamters(diffGamma);
    }

    @Override
    public void accGrad(float scale) {
        // TODO Auto-generated method stub
        if (accDW == null) {
            accDW = diffGamma.copyGPU();
        } else {
            kernel.axpy_gpu(diffGamma, accDW, accDW.dataLength, scale, 1, 1);
        }
    }
}
