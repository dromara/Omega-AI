package com.omega.engine.nn.layer;

import com.omega.common.utils.MatrixUtils;
import com.omega.common.utils.RandomUtils;
import com.omega.engine.active.ActiveType;
import com.omega.engine.gpu.cudnn.ConvCudnnKernel;
import com.omega.engine.nn.layer.gpu.BiasKernel;
import com.omega.engine.nn.layer.gpu.ConvBaseKernel;
import com.omega.engine.nn.layer.gpu.ConvKernel;
import com.omega.engine.nn.model.ConvLayerInit;
import com.omega.engine.nn.model.LayerInit;
import com.omega.engine.nn.network.CNN;
import com.omega.engine.nn.network.Network;
import com.omega.engine.nn.network.utils.ModelUtils;
import com.omega.engine.tensor.Tensor;
import com.omega.engine.updater.UpdaterFactory;

import java.io.IOException;
import java.io.RandomAccessFile;

/**
 * ConvolutionLayer
 *
 * @author Administrator
 */
public class ConvolutionLayer extends Layer {
    public int kernelNum = 0;
    public int kWidth = 0;
    public int kHeight = 0;
    public int stride = 1;
    public int padding = 0;
    private ConvBaseKernel kernel;
    private BiasKernel biasKernel;

    /**
     * ConvolutionLayer
     *
     * @param channel
     * @param kernelNum
     * @param width
     * @param height
     * @param kWidth
     * @param kHeight
     * @param padding
     * @param stride
     * @param activeFunction
     * @param updater
     */
    public ConvolutionLayer(int channel, int kernelNum, int width, int height, int kWidth, int kHeight, int padding, int stride) {
        this.kernelNum = kernelNum;
        this.channel = channel;
        this.width = width;
        this.height = height;
        this.kWidth = kWidth;
        this.kHeight = kHeight;
        this.padding = padding;
        this.stride = stride;
        this.hasParams = true;
        this.initParam();
    }

    /**
     * ConvolutionLayer
     *
     * @param channel
     * @param kernelNum
     * @param width
     * @param height
     * @param kWidth
     * @param kHeight
     * @param padding
     * @param stride
     * @param activeFunction
     * @param updater
     */
    public ConvolutionLayer(int channel, int kernelNum, int width, int height, int kWidth, int kHeight, int padding, int stride, boolean hasBias) {
        this.kernelNum = kernelNum;
        this.channel = channel;
        this.width = width;
        this.height = height;
        this.kWidth = kWidth;
        this.kHeight = kHeight;
        this.padding = padding;
        this.stride = stride;
        this.hasBias = hasBias;
        this.hasParams = true;
        this.initParam();
    }

    /**
     * ConvolutionLayer
     *
     * @param channel
     * @param kernelNum
     * @param width
     * @param height
     * @param kWidth
     * @param kHeight
     * @param padding
     * @param stride
     * @param activeFunction
     * @param updater
     */
    public ConvolutionLayer(int channel, int kernelNum, int width, int height, int kWidth, int kHeight, int padding, int stride, boolean hasBias, Network network) {
        this.kernelNum = kernelNum;
        this.channel = channel;
        this.width = width;
        this.height = height;
        this.kWidth = kWidth;
        this.kHeight = kHeight;
        this.padding = padding;
        this.stride = stride;
        this.hasBias = hasBias;
        this.network = network;
        this.network.paramLayers.add(this);
        this.setUpdater(UpdaterFactory.create(this.network));
        this.hasParams = true;
        this.initParam();
    }

    /**
     * ConvolutionLayer
     *
     * @param channel
     * @param kernelNum
     * @param width
     * @param height
     * @param kWidth
     * @param kHeight
     * @param padding
     * @param stride
     * @param activeFunction
     * @param updater
     */
    public ConvolutionLayer(int channel, int kernelNum, int width, int height, int kWidth, int kHeight, int padding, int stride, boolean hasBias, boolean freeze, Network network) {
        this.kernelNum = kernelNum;
        this.channel = channel;
        this.width = width;
        this.height = height;
        this.kWidth = kWidth;
        this.kHeight = kHeight;
        this.padding = padding;
        this.stride = stride;
        this.hasBias = hasBias;
        this.network = network;
        this.freeze = freeze;
        this.network.paramLayers.add(this);
        this.setUpdater(UpdaterFactory.create(this.network));
        this.hasParams = true;
        this.initParam();
    }

    public ConvolutionLayer(int channel, int kernelNum, int width, int height, int kWidth, int kHeight, int padding, int stride, boolean hasBias, ParamsInit paramsInit) {
        this.kernelNum = kernelNum;
        this.channel = channel;
        this.width = width;
        this.height = height;
        this.kWidth = kWidth;
        this.kHeight = kHeight;
        this.padding = padding;
        this.stride = stride;
        this.hasBias = hasBias;
        this.hasParams = true;
        this.paramsInit = paramsInit;
        this.initParam();
    }

    public ConvolutionLayer(int channel, int kernelNum, int width, int height, int kWidth, int kHeight, int padding, int stride, boolean hasBias, Network network, ParamsInit paramsInit) {
        this.kernelNum = kernelNum;
        this.channel = channel;
        this.width = width;
        this.height = height;
        this.kWidth = kWidth;
        this.kHeight = kHeight;
        this.padding = padding;
        this.stride = stride;
        this.hasBias = hasBias;
        this.network = network;
        this.network.paramLayers.add(this);
        this.hasParams = true;
        this.paramsInit = paramsInit;
        this.initParam();
    }

    public ConvolutionLayer(int channel, int kernelNum, int width, int height, int kWidth, int kHeight, int padding, int stride, boolean hasBias, Network network, ActiveType activeType) {
        this.kernelNum = kernelNum;
        this.channel = channel;
        this.width = width;
        this.height = height;
        this.kWidth = kWidth;
        this.kHeight = kHeight;
        this.padding = padding;
        this.stride = stride;
        this.hasBias = hasBias;
        this.network = network;
        this.network.paramLayers.add(this);
        this.hasParams = true;
        switch (activeType) {
            case sigmoid:
                this.paramsInit = ParamsInit.sigmoid;
                break;
            case relu:
                this.paramsInit = ParamsInit.relu;
                break;
            case leaky_relu:
                this.paramsInit = ParamsInit.leaky_relu;
                break;
            case tanh:
                this.paramsInit = ParamsInit.tanh;
                break;
            case silu:
                this.paramsInit = ParamsInit.silu;
                break;
            default:
                throw new RuntimeException("The paramsInit is not support the [" + activeType + "] active function.");
        }
        this.initParam();
    }

    public static void main(String[] args) {
        int N = 5;
        int T = 10;
        int E = 8;
        int NC = 4;
        float[] data = RandomUtils.order(N * T * E, 0.1f, 0.1f);
        Tensor input = new Tensor(N, T, 1, E, data, true);
        Tensor it = new Tensor(N, E, 1, T, true);
        float[] delta_data = MatrixUtils.val(N * NC * T, 1.0f);
        Tensor delta = new Tensor(N, NC, 1, T, delta_data, true);
        CNN nn = new CNN(null);
        nn.CUDNN = true;
        nn.number = N;
        nn.tensorOP.permute(input, it, new int[]{0, 3, 2, 1});
        input = it;
        input.showDM();
        ConvolutionLayer conv1 = new ConvolutionLayer(E, NC, T, 1, 1, 1, 0, 1, true, nn);
        conv1.weight = new Tensor(NC, E, 1, 1, RandomUtils.order(NC * E, 0.1f, 0.1f), true);
        conv1.bias = new Tensor(1, 1, 1, NC, RandomUtils.order(NC, 0.1f, 0.0f), true);
        //		mal.forward(input);
        conv1.forward(input);
        //		input.showDM();
        //		mal.getWeights().showDM();
        conv1.getOutput().showShape();
        conv1.getOutput().showDM();
        conv1.back(delta);
        conv1.diff.showDM();
        Tensor difft = new Tensor(N, T, 1, E, true);
        nn.tensorOP.permute(conv1.diff, difft, new int[]{0, 3, 2, 1});
        conv1.diff = difft;
        conv1.diff.showDM();
    }

    @Override
    public void initParam() {
        // TODO Auto-generated method stub
        int dataLength = kernelNum * channel * kHeight * kWidth;
        this.oChannel = this.kernelNum;
        this.oWidth = (this.width + this.padding * 2 - kWidth) / this.stride + 1;
        this.oHeight = (this.height + this.padding * 2 - kHeight) / this.stride + 1;
        //		this.weight = new Tensor(kernelNum, channel, kHeight, kWidth, RandomUtils.xavierUniform(dataLength, channel, kernelNum, 1.0f), true);
        this.weight = new Tensor(kernelNum, channel, kHeight, kWidth, RandomUtils.kaiming_uniform(dataLength, this.channel * kHeight * kWidth, this.paramsInit), true);
        //		this.weight = new Tensor(kernelNum, channel, kHeight, kWidth, RandomUtils.kaiming_normal(dataLength, this.oChannel * this.oHeight * this.oWidth, this.paramsInit), true);
        //		this.weight = new Tensor(kernelNum, channel, kHeight, kWidth, RandomUtils.xavierReluRandom(kernelNum * channel * kHeight * kWidth, this.channel * this.height * this.width, this.oChannel * this.oHeight * this.oWidth), true);
        //		this.weight = new Tensor(kernelNum, channel, kHeight, kWidth, RandomUtils.kaimingNormalRandom(kernelNum * channel * kHeight * kWidth, 0, kernelNum * kHeight * kWidth), true);
        //		this.weight = new Tensor(kernelNum, channel, kHeight, kWidth, RandomUtils.xavierRandom(kernelNum * channel * kHeight * kWidth, this.channel * this.height * this.width, this.oChannel * this.oHeight * this.oWidth));
        //		this.weight = new Tensor(kernelNum, channel, kHeight, kWidth, RandomUtils.xavierRandom(kernelNum * channel * kHeight * kWidth, this.channel * this.height * this.kHeight, this.kWidth * this.kHeight * this.kWidth));
        //		this.weight = new Tensor(kernelNum, channel, kHeight, kWidth, RandomUtils.heRandom(kernelNum * channel * kHeight * kWidth, this.channel * this.oChannel * this.kHeight * this.kWidth));
        //		this.weight = new Tensor(kernelNum, channel, kHeight, kWidth, RandomUtils.val(kernelNum * channel * kHeight * kWidth, 0.1f), true);
        //		this.weight = new Tensor(kernelNum, channel, kHeight, kWidth, RandomUtils.order(kernelNum * channel * kHeight * kWidth, 0.1f, 0.01f), true);
        //		this.bias = new Tensor(1, 1, 1, kernelNum, RandomUtils.kaimingUniformBias(kernelNum, this.channel * kHeight * kWidth), true);
        if (hasBias) {
            this.bias = new Tensor(1, 1, 1, kernelNum, true);
        }
    }

    @Override
    public void init() {
        // TODO Auto-generated method stub
        this.number = this.network.number;
        if (this.output == null || this.number != this.output.number) {
            //			this.output = new Tensor(number, oChannel, oHeight, oWidth, true);
            this.output = Tensor.createTensor(this.output, number, oChannel, oHeight, oWidth, true);
        }
        if (kernel == null) {
            if (this.network.CUDNN) {
                kernel = new ConvCudnnKernel(this.network, channel, height, width, kernelNum, kHeight, kWidth, stride, padding, cuda());
            } else {
                kernel = new ConvKernel(channel, height, width, kernelNum, kHeight, kWidth, stride, padding, cuda());
            }
            if (this.hasBias) {
                biasKernel = new BiasKernel(cuda());
            }
        }
    }

    public void init(Tensor input) {
        // TODO Auto-generated method stub
        this.number = input.number;
        if (this.output == null || this.number != this.output.number) {
            this.output = Tensor.createTensor(this.output, number, oChannel, oHeight, oWidth, true);
        }
        if (kernel == null) {
            if (this.network.CUDNN) {
                kernel = new ConvCudnnKernel(this.network, channel, height, width, kernelNum, kHeight, kWidth, stride, padding, cuda());
            } else {
                kernel = new ConvKernel(channel, height, width, kernelNum, kHeight, kWidth, stride, padding, cuda());
            }
            if (this.hasBias) {
                biasKernel = new BiasKernel(cuda());
            }
        }
    }

    @Override
    public void initBack() {
        // TODO Auto-generated method stub
        if (this.diff == null || this.number != this.diff.number) {
            this.diff = new Tensor(number, channel, height, width, true);
        }
        if (this.diffW == null) {
            if (!freeze) {
                if (this.hasBias) {
                    this.diffB = new Tensor(1, 1, 1, kernelNum, true);
                }
                this.diffW = new Tensor(this.kernelNum, this.channel, this.kHeight, this.kWidth, true);
            }
        }
    }
    
    public void initBack(Tensor diff) {
        // TODO Auto-generated method stub
    	this.diff = diff;
        if (this.diffW == null) {
            if (!freeze) {
                if (this.hasBias) {
                    this.diffB = new Tensor(1, 1, 1, kernelNum, true);
                }
                this.diffW = new Tensor(this.kernelNum, this.channel, this.kHeight, this.kWidth, true);
            }
        }
    }

    @Override
    public void output() {
        // TODO Auto-generated method stub
        kernel.conv(input, weight, output);
        if (this.hasBias) {
            biasKernel.addConvBiasFast(output, bias);
        }
    }

    /**
     * delta = diff(i + 1) * f'(xi)
     * <p>
     * dx = padding(delta) conv r180(kernel)
     * <p>
     * dw = delta * px
     * <p>
     * remark: px is zeropadding x
     */
    @Override
    public void diff() {
        // TODO Auto-generated method stub
        //		long start = System.nanoTime();
        //		if(oWidth == 7) {
        //			System.out.println(JsonUtils.toJson(delta.syncHost()));
        //
        //		}
        if (!freeze) {
            /**
             * 计算deltaW
             * 20220816: dw = diff * im2col(input)T
             * diff[knumber * oh * ow]
             * im2col(input)T[oh * ow * C * kh * kw]

             */
            kernel.dw(input, delta, diffW);
            /**
             * 计算deltaB

             */
            if (this.hasBias) {
                biasKernel.backwardConvBias(diffB, delta);
            }
        }
        //		System.out.println("===========");
        /**
         * 计算diff

         */
        if (PROPAGATE_DOWN || this.network.PROPAGATE_DOWN) {
            /**
             * dx = col2im(a)
             * a = (weight)T * diff
             * a[c * kh * kw * oh * ow]
             * (weight)T[c * kh * kw * ko]
             * diff[ko * oh * ow]

             */
            kernel.dx(delta, weight, diff);
            //			System.out.println(this.index+":"+diff.isZero()+":"+delta.isZero());
        }
        //		System.out.println("back:"+(System.nanoTime() - start) / 1e6 + "ms.");
    }

    public void diff(Tensor diff) {
        // TODO Auto-generated method stub
        //		long start = System.nanoTime();
        //		if(oWidth == 7) {
        //			System.out.println(JsonUtils.toJson(delta.syncHost()));
        //
        //		}
        if (!freeze) {
            /**
             * 计算deltaW
             * 20220816: dw = diff * im2col(input)T
             * diff[knumber * oh * ow]
             * im2col(input)T[oh * ow * C * kh * kw]

             */
            kernel.dw(input, delta, diffW);
        }
        //		diffW.showDM();
        //		System.out.println("===========");
        /**
         * 计算deltaB
         */
        if (this.hasBias) {
            biasKernel.backwardConvBias(diffB, delta);
        }
        /**
         * 计算diff

         */
        if (PROPAGATE_DOWN || this.network.PROPAGATE_DOWN) {
            /**
             * dx = col2im(a)
             * a = (weight)T * diff
             * a[c * kh * kw * oh * ow]
             * (weight)T[c * kh * kw * ko]
             * diff[ko * oh * ow]

             */
            kernel.dx(delta, weight, diff);
            //			System.out.println(this.index+":"+diff.isZero()+":"+delta.isZero());
        }
        //		System.out.println("back:"+(System.nanoTime() - start) / 1e6 + "ms.");
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
        //		long start = System.nanoTime();
        initBack();
        /**
         * 设置梯度

         */
        this.setDelta();
        /**
         * 计算梯度

         */
        this.diff();
        //		System.out.println(JsonUtils.toJson(diffW.data));
        if (this.network.GRADIENT_CHECK) {
            this.gradientCheck();
        }
        //		System.out.println((System.nanoTime() - start) / 1e6+"ms->all back");
    }

    @Override
    public void update() {
        // TODO Auto-generated method stub
        //		long start = System.nanoTime();
        if (!this.freeze) {
            if (accDW != null) {
                this.accDW.copy(diffW);
                if (hasBias) {
                    this.accDB.copy(diffB);
                }
            }
            if (this.updater != null) {
                this.updater.update(this);
            } else {
                for (int i = 0; i < this.weight.getDataLength(); i++) {
                    this.weight.data[i] -= this.learnRate * this.diffW.data[i];
                }
                for (int i = 0; i < this.bias.getDataLength(); i++) {
                    this.bias.data[i] -= this.learnRate * this.diffB.data[i];
                }
            }
            this.clearAccGrad();
        }
        //		System.out.println((System.nanoTime() - start) / 1e6+"ms->all update========>");
    }

    @Override
    public LayerType getLayerType() {
        // TODO Auto-generated method stub
        return LayerType.conv;
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
    public LayerInit save() {
        // TODO Auto-generated method stub
        return new ConvLayerInit(this);
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
        initBack();
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

    public void back(Tensor delta, Tensor diff) {
        // TODO Auto-generated method stub
        this.initBack(diff);
        /**
         * 设置梯度

         */
        this.setDelta(delta);
        /**
         * 计算梯度

         */
        this.diff(diff);
        if (this.network.GRADIENT_CHECK) {
            this.gradientCheck();
        }
    }

    @Override
    public void backTemp() {
        // TODO Auto-generated method stub
    }

    public void saveModel(RandomAccessFile outputStream) throws IOException {
        ModelUtils.saveParams(outputStream, weight);
        if (hasBias) {
            ModelUtils.saveParams(outputStream, bias);
        }
    }

    public void loadModel(RandomAccessFile inputStream) throws IOException {
        ModelUtils.loadParams(inputStream, weight);
        if (hasBias) {
            ModelUtils.loadParams(inputStream, bias);
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
        if (hasBias) {
            if (accDB == null) {
                accDB = diffB.copyGPU();
            } else {
                kernel.axpy_gpu(diffB, accDB, accDB.dataLength, scale, 1, 1);
            }
        }
    }
}

