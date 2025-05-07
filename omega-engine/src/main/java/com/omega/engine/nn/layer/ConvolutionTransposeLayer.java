package com.omega.engine.nn.layer;

import com.omega.common.data.Tensor;
import com.omega.common.utils.MatrixUtils;
import com.omega.common.utils.RandomUtils;
import com.omega.engine.gpu.CUDAModules;
import com.omega.engine.gpu.cudnn.ConvTransposeCudnnKernel;
import com.omega.engine.nn.layer.gpu.BiasKernel;
import com.omega.engine.nn.layer.gpu.ConvBaseKernel;
import com.omega.engine.nn.model.ConvLayerInit;
import com.omega.engine.nn.model.LayerInit;
import com.omega.engine.nn.network.Network;
import com.omega.engine.nn.network.RunModel;
import com.omega.engine.nn.network.Transformer;
import com.omega.engine.nn.network.utils.ModelUtils;

import java.io.IOException;
import java.io.RandomAccessFile;

/**
 * ConvolutionLayer
 *
 * @author Administrator
 */
public class ConvolutionTransposeLayer extends Layer {
    public int kernelNum = 0;
    public int kWidth = 0;
    public int kHeight = 0;
    public int stride = 1;
    public int padding = 0;
    public int dilation = 1;
    public int output_padding = 0;
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
    public ConvolutionTransposeLayer(int channel, int kernelNum, int width, int height, int kWidth, int kHeight, int padding, int stride, int dilation, int output_padding) {
        this.kernelNum = kernelNum;
        this.channel = channel;
        this.width = width;
        this.height = height;
        this.kWidth = kWidth;
        this.kHeight = kHeight;
        this.padding = padding;
        this.stride = stride;
        this.dilation = dilation;
        this.output_padding = output_padding;
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
    public ConvolutionTransposeLayer(int channel, int kernelNum, int width, int height, int kWidth, int kHeight, int padding, int stride, int dilation, int output_padding, boolean hasBias) {
        this.kernelNum = kernelNum;
        this.channel = channel;
        this.width = width;
        this.height = height;
        this.kWidth = kWidth;
        this.kHeight = kHeight;
        this.padding = padding;
        this.stride = stride;
        this.dilation = dilation;
        this.output_padding = output_padding;
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
    public ConvolutionTransposeLayer(int channel, int kernelNum, int width, int height, int kWidth, int kHeight, int padding, int stride, int dilation, int output_padding, boolean hasBias, Network network) {
        this.kernelNum = kernelNum;
        this.channel = channel;
        this.width = width;
        this.height = height;
        this.kWidth = kWidth;
        this.kHeight = kHeight;
        this.padding = padding;
        this.stride = stride;
        this.dilation = dilation;
        this.output_padding = output_padding;
        this.hasBias = hasBias;
        this.network = network;
        this.hasParams = true;
        //		network.paramLayers.add(this);
        this.initParam();
    }

    public static void main(String[] args) {
        try {
            CUDAModules.initContext();
            Transformer tf = new Transformer();
            tf.CUDNN = true;
            tf.RUN_MODEL = RunModel.TRAIN;
            int B = 2;
            int C = 3;
            int H = 4;
            int W = 4;
            int OC = 4;
            int kWidth = 4;
            int kHeight = 4;
            int stride = 2;
            int padding = 1;
            ConvolutionTransposeLayer ct = new ConvolutionTransposeLayer(C, OC, W, H, kWidth, kHeight, padding, stride, 1, 0, false, tf);
            int dataLen = B * C * H * W;
            Tensor im = new Tensor(B, C, H, W, MatrixUtils.order(dataLen, 0.0000001f, 0.0000001f), true);
            ct.weight = new Tensor(C, OC, kHeight, kWidth, MatrixUtils.order(C * OC * kHeight * kWidth, 0.0000001f, 0.0000001f), true);
            ct.weight.showDM();
            tf.number = B;
            ct.forward(im);
            ct.getOutput().showDM();
            Tensor delta = new Tensor(B, OC, 8, 8, MatrixUtils.order(B * OC * 8 * 8, 0.0001f, 0.000001f), true);
            ct.back(delta);
            ct.diff.showDM();
            ct.diffW.showDM();
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    @Override
    public void initParam() {
        // TODO Auto-generated method stub
        int dataLength = kernelNum * channel * kHeight * kWidth;
        this.oChannel = this.kernelNum;
        this.oWidth = (this.width - 1) * this.stride - 2 * this.padding + this.dilation * (this.kWidth - 1) + this.output_padding + 1;
        this.oHeight = (this.height - 1) * this.stride - 2 * this.padding + this.dilation * (this.kHeight - 1) + this.output_padding + 1;
        this.weight = new Tensor(channel, kernelNum, kHeight, kWidth, RandomUtils.kaiming_uniform(dataLength, this.channel * kHeight * kWidth, this.paramsInit), true);
        //		this.weight = new Tensor(kernelNum, channel, kHeight, kWidth, RandomUtils.kaiming_normal(dataLength, this.oChannel * kHeight * kWidth, this.paramsInit), true);
        //		this.weight = new Tensor(kernelNum, channel, kHeight, kWidth, RandomUtils.xavierReluRandom(kernelNum * channel * kHeight * kWidth, this.channel * this.height * this.width, this.oChannel * this.oHeight * this.oWidth), true);
        //		this.weight = new Tensor(kernelNum, channel, kHeight, kWidth, RandomUtils.xavierRandom(kernelNum * channel * kHeight * kWidth, this.channel * this.height * this.width, this.oChannel * this.oHeight * this.oWidth));
        //		this.weight = new Tensor(kernelNum, channel, kHeight, kWidth, RandomUtils.xavierRandom(kernelNum * channel * kHeight * kWidth, this.channel * this.height * this.kHeight, this.kWidth * this.kHeight * this.kWidth));
        //		this.weight = new Tensor(kernelNum, channel, kHeight, kWidth, RandomUtils.heRandom(kernelNum * channel * kHeight * kWidth, this.channel * this.oChannel * this.kHeight * this.kWidth));
        //		this.weight = new Tensor(kernelNum, channel, kHeight, kWidth, RandomUtils.val(kernelNum * channel * kHeight * kWidth, 0.1f), true);
        //		this.weight = new Tensor(kernelNum, channel, kHeight, kWidth, RandomUtils.order(kernelNum * channel * kHeight * kWidth, 0.1f, 0.01f), true);
        //		this.bias = new Tensor(1, 1, 1, kernelNum, true);
        if (this.hasBias) {
            this.bias = new Tensor(1, 1, 1, kernelNum, RandomUtils.kaimingUniformBias(kernelNum, this.channel * kHeight * kWidth), true);
        }
        if (network != null) {
            this.diffB = this.network.createParamterGrad(1, 1, 1, kernelNum, true);
            this.diffW = this.network.createParamterGrad(this.channel, this.kernelNum, this.kHeight, this.kWidth, true);
        } else {
            this.diffB = new Tensor(1, 1, 1, kernelNum, true);
            this.diffW = new Tensor(this.channel, this.kernelNum, this.kHeight, this.kWidth, true);
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
                kernel = new ConvTransposeCudnnKernel(network, channel, height, width, kernelNum, kHeight, kWidth, stride, padding, dilation, output_padding, cuda());
            } else {
                //				kernel = new ConvKernel(channel, height, width, kernelNum, kHeight, kWidth, stride, padding);
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
    }

    @Override
    public void output() {
        // TODO Auto-generated method stub
        //		long start = System.nanoTime();
        //		weight.showDM(weight.dataLength-1);
        kernel.convTranspose(input, weight, output);
        if (this.hasBias) {
            biasKernel.addConvBias(output, bias);
        }
        //		System.out.println(JsonUtils.toJson(output.getByNumberAndChannel(0, 0)));
        //		System.out.println(this.index+":"+(System.nanoTime() - start) / 1e6+"ms.");
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
        /**
         * 计算diff

         */
        if (PROPAGATE_DOWN || this.network.PROPAGATE_DOWN) {
            //			weight.showDM("weight");
            /**
             * dx = col2im(a)
             * a = (weight)T * diff
             * a[c * kh * kw * oh * ow]
             * (weight)T[c * kh * kw * ko]
             * diff[ko * oh * ow]

             */
            kernel.dx(delta, weight, diff);
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

    @Override
    public LayerType getLayerType() {
        // TODO Auto-generated method stub
        return LayerType.conv_transpose;
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
        this.init();
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
}

