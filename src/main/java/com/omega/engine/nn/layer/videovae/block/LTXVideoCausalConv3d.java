package com.omega.engine.nn.layer.videovae.block;

import java.io.IOException;
import java.io.RandomAccessFile;

import com.omega.common.utils.JsonUtils;
import com.omega.common.utils.RandomUtils;
import com.omega.engine.active.ActiveType;
import com.omega.engine.gpu.PaddingKernel;
import com.omega.engine.gpu.cudnn.Conv3DCudnnKernel;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.ParamsInit;
import com.omega.engine.nn.layer.gpu.BiasKernel;
import com.omega.engine.nn.layer.gpu.Conv3DBaseKernel;
import com.omega.engine.nn.model.LayerInit;
import com.omega.engine.nn.network.CNN;
import com.omega.engine.nn.network.Network;
import com.omega.engine.nn.network.utils.ModelUtils;
import com.omega.engine.tensor.Tensor;
import com.omega.engine.updater.UpdaterFactory;

/**
 * LTXVideoCausalConv3d
 *
 * @author Administrator
 */
public class LTXVideoCausalConv3d extends Layer {
    public int kernelNum = 0;
    public int depth = 0;
    public int kDepth = 0;
    public int kWidth = 0;
    public int kHeight = 0;
    public int stride = 1;
    public int oDepth = 0;
    private Conv3DBaseKernel kernel;
    private BiasKernel biasKernel;
    private PaddingKernel paddingKernel;

    private int height_pad;
    private int width_pad;
    
    private int pDepth;
    private int pHeight;
    private int pWidth;
    
    private boolean is_causal = false;
    
    private Tensor pOutput;
    
    private int psize;
    
//    private Tensor pDiff;
    
    private int[] padding3d = new int[]{0, 1, 1};
    
    private int[] stride3d = new int[]{1, 1, 1};

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
    public LTXVideoCausalConv3d(int channel, int kernelNum, int depth, int width, int height, int kDepth, int kWidth, int kHeight, int stride, boolean hasBias, boolean is_causal, Network network) {
        this.kernelNum = kernelNum;
        this.channel = channel;
        this.depth = depth;
        this.width = width;
        this.height = height;
        this.kDepth = kDepth;
        this.kWidth = kWidth;
        this.kHeight = kHeight;
        this.stride = stride;
        this.height_pad = kHeight / 2;
        this.width_pad = kWidth / 2;
        this.hasBias = hasBias;
        this.network = network;
        //		network.paramLayers.add(this);
        this.setUpdater(UpdaterFactory.create(this.network));
        this.hasParams = true;
        this.is_causal = is_causal;
        this.initParam();
    }

    public LTXVideoCausalConv3d(int channel, int kernelNum, int depth, int width, int height, int kDepth, int kWidth, int kHeight, int stride, boolean hasBias, boolean is_causal, Network network, ParamsInit paramsInit) {
        this.kernelNum = kernelNum;
        this.channel = channel;
        this.depth = depth;
        this.width = width;
        this.height = height;
        this.kDepth = kDepth;
        this.kWidth = kWidth;
        this.kHeight = kHeight;
        this.stride = stride;
        this.height_pad = kHeight / 2;
        this.width_pad = kWidth / 2;
        this.hasBias = hasBias;
        this.network = network;
        //		network.paramLayers.add(this);
        this.hasParams = true;
        this.paramsInit = paramsInit;
        this.is_causal = is_causal;
        this.initParam();
    }

    public LTXVideoCausalConv3d(int channel, int kernelNum, int depth, int width, int height, int kDepth, int kWidth, int kHeight, int stride, boolean hasBias, boolean is_causal, Network network, ActiveType activeType) {
        this.kernelNum = kernelNum;
        this.channel = channel;
        this.depth = depth;
        this.width = width;
        this.height = height;
        this.kDepth = kDepth;
        this.kWidth = kWidth;
        this.kHeight = kHeight;
        this.stride = stride;
        this.height_pad = kHeight / 2;
        this.width_pad = kWidth / 2;
        this.hasBias = hasBias;
        this.is_causal = is_causal;
        this.network = network;
        //		network.paramLayers.add(this);
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
        int N = 4;
        int C = 128;
        int F = 17;
        int H = 4;
        int W = 4;
        
        int KC = 129;
        
        int KF = 3;
        int KH = 3;
        int KW = 3;

        int stride = 1;
        float[] data = RandomUtils.order(N * C * F * H * W, 0.1f, 0.1f);
        Tensor input = new Tensor(N, C * F, H, W, data, true);
        CNN nn = new CNN(null);
        nn.CUDNN = true;
        nn.number = N;
        //nt channel,int kernelNum,int depth,int width,int height,int kDepth,int kWidth,int kHeight,int padding,int stride
        LTXVideoCausalConv3d conv1 = new LTXVideoCausalConv3d(C, KC, F, W, H, KF, KW, KH, stride, true, true, nn);

        conv1.weight = new Tensor(KC, C * KF, KH, KW, RandomUtils.order(KC * C * KF * KH * KW, 0.1f, 0.1f), true);
        conv1.bias = new Tensor(1, 1, 1, KC, RandomUtils.order(KC, 0.1f, 0.1f), true);
        conv1.forward(input);
       
        conv1.getOutput().showShape();
        conv1.getOutput().showDM();

    }

    @Override
    public void initParam() {
        // TODO Auto-generated method stub
        int dataLength = kernelNum * channel * kDepth * kHeight * kWidth;
        this.psize = kDepth - 1;
        this.oChannel = this.kernelNum;
        this.padding3d = new int[] {0, height_pad, width_pad};
        this.pDepth = this.depth + padding3d[0] + padding3d[0] + psize;
        this.pHeight = this.height + padding3d[1] + padding3d[1];
        this.pWidth = this.width + padding3d[2] + padding3d[2];
        this.oDepth = (this.pDepth - kDepth) / this.stride + 1;
        this.oWidth = (this.pWidth - kWidth) / this.stride + 1;
        this.oHeight = (this.pHeight - kHeight) / this.stride + 1;
        this.stride3d = new int[] {stride, stride, stride};
        this.weight = new Tensor(kernelNum, channel * kDepth, kHeight, kWidth, RandomUtils.kaiming_uniform(dataLength, this.channel * kDepth * kHeight * kWidth, this.paramsInit), true);
        
        if (hasBias) {
            this.bias = new Tensor(1, 1, 1, kernelNum, true);
        }

    }

    @Override
    public void init() {
        // TODO Auto-generated method stub
        this.number = this.network.number;
        if (kernel == null) {
            if (this.network.CUDNN) {
                kernel = new Conv3DCudnnKernel(this.network, channel, pDepth, height, width, kernelNum, kDepth, kHeight, kWidth, stride, 0, cuda());
            } else {
                //				kernel = new ConvKernel(channel, height, width, kernelNum, kHeight, kWidth, stride, padding, cuda());
            }
            if (this.hasBias) {
                biasKernel = new BiasKernel(cuda());
            }
            paddingKernel = new PaddingKernel(cuda());
        }
        if((this.pOutput == null || this.number != this.pOutput.number) && psize > 0) {
        	this.pOutput = Tensor.createTensor(this.pOutput, number, channel * (psize + depth), height, width, true);
        }
        if (this.output == null || this.number != this.output.number) {
            this.output = Tensor.createTensor(this.output, number, oChannel * oDepth, oHeight, oWidth, true);
        }
        
    }

    public void init(Tensor input) {
        // TODO Auto-generated method stub
        this.number = input.number;
        if (kernel == null) {
            if (this.network.CUDNN) {
            	 kernel = new Conv3DCudnnKernel(this.network, channel, pDepth, height, width, kernelNum, kDepth, kHeight, kWidth, stride, 0, cuda());
            } else {
                //				kernel = new ConvKernel(channel, height, width, kernelNum, kHeight, kWidth, stride, padding, cuda());
            }
            if (this.hasBias) {
                biasKernel = new BiasKernel(cuda());
            }
            paddingKernel = new PaddingKernel(cuda());
        }
        if((this.pOutput == null || this.number != this.pOutput.number) && psize > 0) {
        	this.pOutput = Tensor.createTensor(this.pOutput, number, channel * (psize + depth), height, width, true);
        }
        if (this.output == null || this.number != this.output.number) {
            this.output = Tensor.createTensor(this.output, number, oChannel * oDepth, oHeight, oWidth, true);
        }
    }

    @Override
    public void initBack() {
        // TODO Auto-generated method stub

    }

    @Override
    public void output() {
        // TODO Auto-generated method stub
    	if(pOutput != null) {
	    	if(is_causal) {
	    		paddingKernel.padding_time_head(input, pOutput, channel, pDepth, height, width, psize);
	    	}else {
	    		paddingKernel.padding_time_head_fair(input, pOutput, channel, pDepth, height, width);
	    	}
//	    	pOutput.showDMByOffsetRed((3 * 5 + 0) * pOutput.height * pOutput.width, pOutput.height * pOutput.width, "pOutput");
//	    	pOutput.showDMByOffsetRed((3 * 5 + 4) * pOutput.height * pOutput.width, pOutput.height * pOutput.width, "pOutput");
	    	kernel.conv(pOutput, weight, output, padding3d, stride3d);
    	}else {
    		kernel.conv(input, weight, output, padding3d, stride3d);
    	}
//    	System.err.println(JsonUtils.toJson(padding3d));
//    	System.err.println(JsonUtils.toJson(stride3d));
        if (this.hasBias) {
//        	output.showShape("conv_out");
//        	bias.showDM("bias");
//        	output.showDMByOffsetRed((3 * oDepth + 2) * output.height * output.width, output.height * output.width, "output");
            biasKernel.addConvBiasFast(output, bias, oChannel, oDepth);
//            output.showDMByOffsetRed((3 * oDepth + 2) * output.height * output.width, output.height * output.width, "output");
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

    }

    public void diff(Tensor diff) {
        // TODO Auto-generated method stub
       
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
        return null;
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
        this.initBack();
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

