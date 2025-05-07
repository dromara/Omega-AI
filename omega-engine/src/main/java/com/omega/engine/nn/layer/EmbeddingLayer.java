package com.omega.engine.nn.layer;

import com.omega.common.data.Tensor;
import com.omega.common.utils.RandomUtils;
import com.omega.engine.nn.layer.gpu.EmbeddingKernel;
import com.omega.engine.nn.network.Network;
import com.omega.engine.updater.UpdaterFactory;
import jcuda.jcublas.cublasOperation;

/**
 * FullyLayer
 *
 * @author Administrator
 */
public class EmbeddingLayer extends Layer {
    private EmbeddingKernel kernel;

    public EmbeddingLayer(int inputNum, int outputNum) {
        this.channel = 1;
        this.height = 1;
        this.width = inputNum;
        this.oChannel = channel;
        this.oHeight = height;
        this.oWidth = outputNum;
        this.hasParams = true;
        this.hasBias = false;
        this.initParam();
    }

    public EmbeddingLayer(int inputNum, int outputNum, Network network) {
        this.network = network;
        network.paramLayers.add(this);
        if (this.updater == null) {
            this.setUpdater(UpdaterFactory.create(network));
        }
        this.channel = 1;
        this.height = 1;
        this.width = inputNum;
        this.oChannel = channel;
        this.oHeight = height;
        this.oWidth = outputNum;
        this.hasParams = true;
        this.hasBias = false;
        this.initParam();
    }

    @Override
    public void initBack() {
        // TODO Auto-generated method stub
        if (this.diff == null || this.number != this.diff.number) {
            this.diff = new Tensor(number, channel, height, width, true);
        }
    }

    @Override
    public void init() {
        // TODO Auto-generated method stub
        this.number = this.network.number;
        if (this.output == null || this.number != this.output.number) {
            this.output = Tensor.createTensor(this.output, number, oChannel, oHeight, oWidth, true);
        }
    }

    public void init(Tensor input) {
        // TODO Auto-generated method stub
        this.number = input.number;
        if (this.output == null || this.number != this.output.number) {
            this.output = Tensor.createTensor(this.output, number, oChannel, oHeight, oWidth, true);
        }
    }

    @Override
    public void initParam() {
        // TODO Auto-generated method stub
        if (kernel == null) {
            kernel = new EmbeddingKernel(cuda());
        }
        this.weight = new Tensor(1, 1, width, oWidth, RandomUtils.kaiming_uniform(this.width * this.oWidth, this.width, this.paramsInit), true);
        if (this.network != null) {
            this.diffW = this.network.createParamterGrad(1, 1, width, oWidth, true);
        } else {
            this.diffW = new Tensor(1, 1, width, oWidth, true);
        }
    }

    @Override
    public void output() {
        // TODO Auto-generated method stub
        if (this.input != null) {
            GPU_OP().multiplyFloat(number, oWidth, width, input.getGpuData(), weight.getGpuData(), output.getGpuData(), cublasOperation.CUBLAS_OP_N, cublasOperation.CUBLAS_OP_N, 1.0f, 0.0f);
        }
    }

    @Override
    public void diff() {
        // TODO Auto-generated method stub
        /**
         * deltaW = inputT * delta
         * int m,int n,int k, float A[],float B[], float C[],int CUBLAS_OP_A,int CUBLAS_OP_B,float alpha,float beta
         * number * w
         * number * ow
         * m = w,k = number,n = ow

         */
        GPU_OP().multiplyFloat(this.width, this.oWidth, this.number, input.getGpuData(), delta.getGpuData(), diffW.getGpuData(), cublasOperation.CUBLAS_OP_T, cublasOperation.CUBLAS_OP_N, 1.0f, 0.0f);
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

    /**
     * w(t) = w(t-1) + θ * deltaW
     * <p>
     * b(t) = b(t-1) + θ * deltaB
     * <p>
     * θ : learningRate
     */
    @Override
    public void update() {
        // TODO Auto-generated method stub
        if (!this.freeze) {
            if (accDW != null) {
                this.accDW.copy(diffW);
            }
            if (this.updater != null) {
                this.updater.update(this);
            } else {
                for (int i = 0; i < this.weight.getDataLength(); i++) {
                    this.weight.data[i] -= this.learnRate * this.diffW.data[i];
                }
            }
            this.clearAccGrad();
        }
    }

    @Override
    public void accGrad(float scale) {
        // TODO Auto-generated method stub
        if (accDW == null) {
            accDW = diffW.copyGPU();
        } else {
            kernel.axpy_gpu(diffW, accDW, accDW.dataLength, scale, 1, 1);
        }
    }

    @Override
    public Tensor getOutput() {
        // TODO Auto-generated method stub
        return this.output;
    }

    @Override
    public void showDiff() {
        // TODO Auto-generated method stub
    }

    @Override
    public LayerType getLayerType() {
        // TODO Auto-generated method stub
        return LayerType.embedding;
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

    public void clear() {
        //		this.output.clear();
        //		this.diffW.clear();
        //		this.diff.clear();
        //		this.diffW.clearGPU();
    }

    @Override
    public void backTemp() {
        // TODO Auto-generated method stub
    }
}

