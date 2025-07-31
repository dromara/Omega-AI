package com.omega.engine.nn.layer.normalization;

import java.io.IOException;
import java.io.RandomAccessFile;

import com.omega.common.utils.MatrixUtils;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.normalization.gpu.LNKernel;
import com.omega.engine.nn.model.LayerInit;
import com.omega.engine.nn.network.Network;
import com.omega.engine.nn.network.utils.ModelUtils;
import com.omega.engine.tensor.Tensor;
import com.omega.engine.updater.UpdaterFactory;

/**
 * Layer Normalization Layer
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
public class LNLayer extends NormalizationLayer {
    public BNType bnType = null;
    public LNKernel kernel;
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

    public LNLayer() {
        //		initParam();
        this.hasParams = true;
    }

    public LNLayer(boolean hasBias) {
        //		initParam();
        this.hasBias = true;
        this.hasParams = true;
    }

    public LNLayer(Layer preLayer) {
        this.setPreLayer(preLayer);
        this.hasParams = true;
        this.setUpdater(UpdaterFactory.create(this.network));
    }

    public LNLayer(Layer preLayer, BNType bnType) {
        this.setPreLayer(preLayer);
        this.bnType = bnType;
        this.hasParams = true;
        this.setUpdater(UpdaterFactory.create(this.network));
    }

    public LNLayer(Layer preLayer, BNType bnType, int channel, int height, int width) {
        this.setPreLayer(preLayer);
        this.channel = channel;
        this.height = height;
        this.width = width;
        this.oChannel = this.channel;
        this.oHeight = this.height;
        this.oWidth = this.width;
        this.bnType = bnType;
        this.hasParams = true;
        if (bnType == BNType.conv_bn) {
            this.meanNum = this.height * this.width;
        } else {
            this.meanNum = this.channel * this.height * this.width;
        }
        this.setUpdater(UpdaterFactory.create(this.network));
    }
    
    public LNLayer(int channel, int height, int width, BNType bnType, Network network) {
    	this.network = network;
		this.setUpdater(UpdaterFactory.create(this.network));
        this.channel = channel;
        this.height = height;
        this.width = width;
        this.oChannel = this.channel;
        this.oHeight = this.height;
        this.oWidth = this.width;
        this.bnType = bnType;
        this.hasParams = true;
        this.meanNum = width;
    }

    public LNLayer(Layer preLayer, boolean hasBias) {
        this.setPreLayer(preLayer);
        this.hasBias = true;
        this.hasParams = true;
        this.setUpdater(UpdaterFactory.create(this.network));
    }

    public LNLayer(Network network) {
        this.network = network;
        network.paramLayers.add(this);
        this.setUpdater(UpdaterFactory.create(this.network));
    }

    public LNLayer(Network network, boolean hasBias) {
        this.network = network;
        network.paramLayers.add(this);
        this.hasBias = true;
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
            kernel = new LNKernel(width, bnType, cuda());
        }
        if (this.gamma == null) {
            this.gamma = new Tensor(1, 1, 1, meanNum, MatrixUtils.one(this.meanNum), true);
            this.diffGamma = new Tensor(1, 1, 1, meanNum, true);
            //			if(network != null) {
            //				this.diffGamma = this.network.createParamterGrad(1, 1, 1, this.meanNum, true);
            //			}else {
            //				this.diffGamma = new Tensor(1, 1, 1, meanNum, true);
            //			}
        }
        if (this.beta == null && hasBias) {
            this.beta = new Tensor(1, 1, 1, meanNum, true);
            if (network != null) {
                this.diffBeta = this.network.createParamterGrad(1, 1, 1, this.meanNum, true);
            } else {
                this.diffBeta = new Tensor(1, 1, 1, meanNum, true);
            }
        }
        if (this.output == null || this.number != this.output.number) {
            this.output = Tensor.createTensor(this.output, number, oChannel, oHeight, oWidth, true);
        }
    }

    public void init(Tensor input) {
        this.number = input.number;
        if (this.output == null && this.bnType == null) {
            this.channel = input.channel;
            this.height = input.height;
            this.width = input.width;
            this.oChannel = this.channel;
            this.oHeight = this.height;
            this.oWidth = this.width;
            this.setBnType(BNType.fully_bn);
        } else if(this.output == null){
            this.channel = input.channel;
            this.height = input.height;
            this.width = input.width;
            this.oChannel = this.channel;
            this.oHeight = this.height;
            this.oWidth = this.width;
            this.setBnType(bnType);
        }
        if(meanNum <= 0) {
        	 if (bnType == BNType.fully_bn) {
                 this.meanNum = this.channel * this.height * this.width;
             } else {
                 this.meanNum = this.height * this.width;
             }
        }
        if (kernel == null) {
            kernel = new LNKernel(width, bnType, cuda());
        }
        if (this.gamma == null) {
            this.gamma = new Tensor(1, 1, 1, meanNum, MatrixUtils.one(this.meanNum), true);
            if (network != null) {
                this.diffGamma = this.network.createParamterGrad(1, 1, 1, meanNum, true);
            } else {
                this.diffGamma = new Tensor(1, 1, 1, meanNum, true);
            }
        }
        if (this.beta == null && hasBias) {
            this.beta = new Tensor(1, 1, 1, meanNum, true);
            if (network != null) {
                this.diffBeta = this.network.createParamterGrad(1, 1, 1, meanNum, true);
            } else {
                this.diffBeta = new Tensor(1, 1, 1, meanNum, true);
            }
        }
        if (this.output == null || this.number != this.output.number) {
            this.output = input.createGPULike();
        }
        
    }
    
    public void init(int channel,int height,int width, BNType bnType) {
    	this.channel = channel;
        this.height = height;
        this.width = width;
        this.oChannel = this.channel;
        this.oHeight = this.height;
        this.oWidth = this.width;
        this.setBnType(bnType);
        if (bnType == BNType.fully_bn) {
            this.meanNum = this.channel * this.height * this.width;
        } else {
            this.meanNum = this.height * this.width;
        }
        if (kernel == null) {
            kernel = new LNKernel(width, bnType, cuda());
        }
        if (this.gamma == null) {
            this.gamma = new Tensor(1, 1, 1, meanNum, MatrixUtils.one(this.meanNum), true);
            if (network != null) {
                this.diffGamma = this.network.createParamterGrad(1, 1, 1, meanNum, true);
            } else {
                this.diffGamma = new Tensor(1, 1, 1, meanNum, true);
            }
        }
       
        if (this.beta == null && hasBias) {
            this.beta = new Tensor(1, 1, 1, meanNum, true);
            if (network != null) {
                this.diffBeta = this.network.createParamterGrad(1, 1, 1, meanNum, true);
            } else {
                this.diffBeta = new Tensor(1, 1, 1, meanNum, true);
            }
        }
//        gamma.showShape("init-gamma");
//        beta.showShape("init-beta");
    }
    
    @Override
    public void initParam() {
        // TODO Auto-generated method stub
    }

    @Override
    public void initBack() {
        if (this.diff == null) {
            this.diff = new Tensor(input.number, input.channel, input.height, input.width, true);
        }
    }

    public void initBack(Tensor delta) {
        if (this.diff == null) {
            this.diff = new Tensor(delta.number, delta.channel, delta.height, delta.width, true);
        }
        if (this.diffGamma == null) {
            this.diffGamma = new Tensor(1, 1, 1, meanNum, true);
            this.diffBeta = new Tensor(1, 1, 1, meanNum, true);
        }
    }
    
    public void initBack(Tensor delta,Tensor diff) {
        this.diff = diff;
        if (this.diffGamma == null) {
            this.diffGamma = new Tensor(1, 1, 1, meanNum, true);
            this.diffBeta = new Tensor(1, 1, 1, meanNum, true);
        }
    }

    @Override
    public void output() {
        // TODO Auto-generated method stub
        //		System.out.println(this.index+":"+input.number+":"+input.channel+":"+input.height+":"+input.width);
        //		System.out.println(this.index+":"+output.number+":"+output.channel+":"+output.height+":"+output.width);
        //		System.out.println(JsonUtils.toJson(gamma.shape()));
        //		System.out.println(JsonUtils.toJson(beta.shape()));
        //		kernel.forward(gamma, beta, input, output);
        //		kernel.forwardAten(gamma, beta, input, output);
        kernel.forward_llm(gamma, beta, input, output);
        //		System.err.println("1:");
        //		output.showDMByNumber(0);
        //		System.err.println("2:");
        //		output2.showDMByNumber(0);
        //
        //		System.out.println("bn-output:");
        //		output.showDM();
    }
    
    public void output_llmc() {
        // TODO Auto-generated method stub
        //		System.out.println(this.index+":"+input.number+":"+input.channel+":"+input.height+":"+input.width);
        //		System.out.println(this.index+":"+output.number+":"+output.channel+":"+output.height+":"+output.width);
        //		System.out.println(JsonUtils.toJson(gamma.shape()));
        //		System.out.println(JsonUtils.toJson(beta.shape()));
        //		kernel.forward(gamma, beta, input, output);
        //		kernel.forwardAten(gamma, beta, input, output);
//    	gamma.showShape("gamma");
//    	beta.showShape("beta");
//    	System.err.println("-------------------");
//    	input.showDMByOffsetRed(0, 100, "ln-in");
        kernel.forward_llmc(gamma, beta, input, output);
//        output.showDMByOffsetRed(0, 100, "ln");
        //		System.err.println("1:");
        //		output.showDMByNumber(0);
        //		System.err.println("2:");
        //		output2.showDMByNumber(0);
        //
        //		System.out.println("bn-output:");
        //		output.showDM();
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
        //		System.out.println(delta);
        //		long start = System.nanoTime();
        //		System.out.println(index);
        //		kernel.backward(input, delta, diff, gamma, diffGamma, diffBeta);
        //		kernel.backwardAten(input, delta, diff, gamma, diffGamma, diffBeta);
        kernel.backward_llm(input, delta, diff, gamma, diffGamma, diffBeta);
        //		diff.showDMByNumber(0);
        //		diff2.showDMByNumber(0);
        //		System.out.println((System.nanoTime() - start) / 1e6 + "ms.");
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
    public void update() {
        // TODO Auto-generated method stub
        if (!this.freeze) {
            if (accDW != null) {
                this.accDW.copy(diffGamma);
                if (hasBias) {
                    this.accDB.copy(diffBeta);
                }
            }
            if (this.updater != null) {
                this.updater.updateForBN(this);
            } else {
                for (int i = 0; i < this.gamma.dataLength; i++) {
                    this.gamma.data[i] -= this.learnRate * this.diffGamma.data[i];
                }
                for (int i = 0; i < this.beta.dataLength; i++) {
                    this.beta.data[i] -= this.learnRate * this.diffBeta.data[i];
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
        return LayerType.layer_norm;
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
    
    public void forward_llmc(Tensor input) {
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
        this.output_llmc();
    }
    
    public void forward_llmc(Tensor input, Tensor output) {
        // TODO Auto-generated method stub
        this.output = output;
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
        this.output_llmc();
    }

    public void forward(Tensor input, Tensor output) {
        // TODO Auto-generated method stub
        this.output = output;
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
        this.initBack(delta);
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
    
    public void back(Tensor delta,Tensor diff) {
        // TODO Auto-generated method stub
        this.initBack(delta, diff);
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
    public void backTemp() {
        // TODO Auto-generated method stub
    }

    public void saveModel(RandomAccessFile outputStream) throws IOException {
        ModelUtils.saveParams(outputStream, gamma);
        if (hasBias) {
            ModelUtils.saveParams(outputStream, beta);
        }
    }

    public void loadModel(RandomAccessFile inputStream) throws IOException {
        init();
        ModelUtils.loadParams(inputStream, gamma);
        if (hasBias) {
            ModelUtils.loadParams(inputStream, beta);
        }
    }
    
    public void loadModel(RandomAccessFile inputStream, int channel, int height, int width, BNType bnType) throws IOException {
        init(channel, height, width, bnType);
        ModelUtils.loadParams(inputStream, gamma);
        if (hasBias) {
            ModelUtils.loadParams(inputStream, beta);
        }
    }

    @Override
    public void accGrad(float scale) {
        // TODO Auto-generated method stub
        if (accDW == null) {
            accDW = diffGamma.copyGPU();
        } else {
            kernel.axpy_gpu(diffGamma, accDW, accDW.dataLength, scale, 1, 1);
        }
        if (hasBias) {
            if (accDB == null) {
                accDB = diffBeta.copyGPU();
            } else {
                kernel.axpy_gpu(diffBeta, accDB, accDB.dataLength, scale, 1, 1);
            }
        }
    }
}

