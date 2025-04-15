package com.omega.engine.nn.layer;

import com.omega.common.config.Tensor;
import com.omega.common.utils.RandomUtils;
import com.omega.engine.gpu.CUDAManager;
import com.omega.engine.gpu.GPUOP;
import com.omega.engine.nn.layer.gpu.DropoutKernel;
import com.omega.engine.nn.network.Network;
import com.omega.engine.nn.network.RunModel;

/**
 * Dropout Layer
 *
 * @author Administrator
 */
public class DropoutLayer extends Layer {
    public Layer preLayer;
    private float probability = 0.1f;
    private Tensor mask;
    private float scale = 0.0f;
    private DropoutKernel kernel;

    public DropoutLayer(float probability) {
        this.probability = probability;
        this.scale = 1.0f / (1.0f - probability);
    }

    public DropoutLayer(float probability, Layer preLayer) {
        this.setPreLayer(preLayer);
        this.probability = probability;
        this.scale = 1.0f / (1.0f - probability);
    }

    public DropoutLayer(float probability, Network network) {
        this.network = network;
        this.probability = probability;
        this.scale = 1.0f / (1.0f - probability);
    }

    public static void main(String[] args) {
        int N = 512;
        int W = 2048;
        float[] data = RandomUtils.order(N * W, 0.1f, 0.1f);
        Tensor input = new Tensor(N, 1, 1, W, data, true);
        Tensor mask = new Tensor(N, 1, 1, W, true);
        Tensor output = new Tensor(N, 1, 1, W, true);
        float[] diff_data = RandomUtils.order(N * W, 0.2f, 0.3f);
        Tensor delta = new Tensor(N, 1, 1, W, diff_data, true);
        Tensor diff = new Tensor(N, 1, 1, W, true);
        CUDAManager cudaManager = new CUDAManager(0);
        DropoutKernel kernel = new DropoutKernel(0.2f, 1.0f / (1.0f - 0.2f), cudaManager);
        for (int i = 0; i < 10; i++) {
            GPUOP.getInstance().cudaRandom(mask);
            System.out.println("output:");
            kernel.dropout(input, output, mask);
            output.showDMByNumber(0);
            System.out.println("diff:");
            kernel.dropout(delta, diff, mask);
            diff.showDMByNumber(0);
            System.out.println("========================");
        }
    }

    public void setPreLayer(Layer pre) {
        this.preLayer = pre;
        this.network = pre.network;
        this.channel = preLayer.oChannel;
        this.height = preLayer.oHeight;
        this.width = preLayer.oWidth;
        this.oChannel = this.channel;
        this.oHeight = this.height;
        this.oWidth = this.width;
    }

    @Override
    public void init() {
        // TODO Auto-generated method stub
        if (preLayer == null) {
            preLayer = this.network.getPreLayer(this.index);
            this.channel = preLayer.oChannel;
            this.height = preLayer.oHeight;
            this.width = preLayer.oWidth;
            this.oChannel = this.channel;
            this.oHeight = this.height;
            this.oWidth = this.width;
        }
        if (kernel == null) {
            kernel = new DropoutKernel(this.probability, this.scale, cuda());
        }
        this.number = this.network.number;
        initParam();
    }

    public void init(Tensor input) {
        // TODO Auto-generated method stub
        this.channel = input.channel;
        this.height = input.height;
        this.width = input.width;
        this.oChannel = this.channel;
        this.oHeight = this.height;
        this.oWidth = this.width;
        if (kernel == null) {
            kernel = new DropoutKernel(this.probability, this.scale, cuda());
        }
        this.number = input.number;
        initParam();
    }

    @Override
    public void initParam() {
        // TODO Auto-generated method stub
        /**
         * 训练

         */
        if (this.network.RUN_MODEL == RunModel.TRAIN) {
            if (this.mask == null || this.mask.number != this.number) {
                this.mask = Tensor.createGPUTensor(this.mask, number, oChannel, oHeight, oWidth, true);
            }
            //			JCuda.cudaDeviceSynchronize();
            //			this.mask.clearGPU();
            //			this.mask.uniform(0.0f, 1.0f);
            //			this.mask.showDMByNumber(0);
        }
        if (this.output == null || this.number != this.output.number) {
            this.output = Tensor.createGPUTensor(this.output, number, oChannel, oHeight, oWidth, true);
        }
    }

    @Override
    public void initBack() {
        // TODO Auto-generated method stub
        if (this.diff == null || this.number != this.diff.number) {
            this.diff = Tensor.createGPUTensor(this.diff, number, channel, height, width, true);
        }
    }

    @Override
    public void output() {
        // TODO Auto-generated method stub
        if (this.network.RUN_MODEL == RunModel.TRAIN) {
            //			input.showDMByNumber(0);
            GPUOP.getInstance().cudaRandom(this.mask);
            kernel.dropout(input, output, mask);
            //			output.showDMByNumber(0);
        } else {
            baseKernel().copy_gpu(input, this.output, input.getDataLength(), 1, 1);
        }
    }

    @Override
    public Tensor getOutput() {
        // TODO Auto-generated method stub
        return this.output;
    }

    @Override
    public void diff() {
        // TODO Auto-generated method stub
        kernel.dropout(delta, diff, mask);
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
        initBack();
        /**
         * 设置梯度

         */
        this.setDelta();
        /**
         * 计算梯度

         */
        this.diff();
    }

    @Override
    public void update() {
        // TODO Auto-generated method stub
    }

    @Override
    public void showDiff() {
        // TODO Auto-generated method stub
    }

    @Override
    public LayerType getLayerType() {
        // TODO Auto-generated method stub
        return LayerType.dropout;
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
        //		input.showDMByNumber(0);
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
        //		getOutput().showDMByNumber(0);
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
    public void backTemp() {
        // TODO Auto-generated method stub
    }

    @Override
    public void accGrad(float scale) {
        // TODO Auto-generated method stub
    }
}

