package com.omega.engine.nn.layer.opensora.wfvae.modules;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.util.Map;

import com.omega.common.utils.MatrixUtils;
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
import com.omega.example.clip.utils.ClipModelUtils;
import com.omega.example.transformer.utils.LagJsonReader;

/**
 * ConvolutionLayer
 *
 * @author Administrator
 */
public class WFCausalConv3D extends Layer {
    public int depth = 0;
    public int kernelNum = 0;
    public int kernelSize = 0;
    public int stride = 1;
    public int padding = 0;
    public int oDepth = 0;
    public Conv3DBaseKernel kernel;
    private BiasKernel biasKernel;
    private PaddingKernel paddingKernel;
    
    private int pDepth;
    private int pHeight;
    private int pWidth;
    
    private Tensor pOutput;
    
    private Tensor pDiff;
    
    private int[] padding3d = new int[]{1, 1, 1, 1, 2, 0};
    
    private int[] pad = new int[] {0, 0, 0};
    
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
    public WFCausalConv3D(int channel, int kernelNum, int depth, int width, int height, int kernelSize, int stride) {
        this.kernelNum = kernelNum;
        this.channel = channel;
        this.depth = depth;
        this.width = width;
        this.height = height;
        this.kernelSize = kernelSize;
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
    public WFCausalConv3D(int channel, int kernelNum, int depth, int width, int height, int kernelSize, int stride, boolean hasBias) {
        this.kernelNum = kernelNum;
        this.channel = channel;
        this.depth = depth;
        this.width = width;
        this.height = height;
        this.kernelSize = kernelSize;
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
    public WFCausalConv3D(int channel, int kernelNum, int depth, int width, int height, int kernelSize, int stride, boolean hasBias, Network network) {
        this.kernelNum = kernelNum;
        this.channel = channel;
        this.depth = depth;
        this.width = width;
        this.height = height;
        this.kernelSize = kernelSize;
        this.stride = stride;
        this.hasBias = hasBias;
        this.network = network;
        //		network.paramLayers.add(this);
        this.setUpdater(UpdaterFactory.create(this.network));
        this.hasParams = true;
        this.initParam();
    }
    
    public WFCausalConv3D(int channel, int kernelNum, int depth, int width, int height, int kernelSize, int stride, int padding, boolean hasBias, Network network) {
        this.kernelNum = kernelNum;
        this.channel = channel;
        this.depth = depth;
        this.width = width;
        this.height = height;
        this.kernelSize = kernelSize;
        this.stride = stride;
        this.padding = padding;
        if(padding > 0) {
        	pad = new int[] {0, padding, padding};
        }
        this.hasBias = hasBias;
        this.network = network;
        //		network.paramLayers.add(this);
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
    public WFCausalConv3D(int channel, int kernelNum, int depth, int width, int height, int kernelSize, int stride, boolean hasBias, boolean freeze, Network network) {
        this.kernelNum = kernelNum;
        this.channel = channel;
        this.depth = depth;
        this.width = width;
        this.height = height;
        this.kernelSize = kernelSize;
        this.stride = stride;
        this.hasBias = hasBias;
        this.network = network;
        this.freeze = freeze;
        //		network.paramLayers.add(this);
        this.setUpdater(UpdaterFactory.create(this.network));
        this.hasParams = true;
        this.initParam();
    }

    public WFCausalConv3D(int channel, int kernelNum, int depth, int width, int height, int kernelSize, int stride, boolean hasBias, ParamsInit paramsInit) {
        this.kernelNum = kernelNum;
        this.channel = channel;
        this.depth = depth;
        this.width = width;
        this.height = height;
        this.kernelSize = kernelSize;
        this.stride = stride;
        this.hasBias = hasBias;
        this.hasParams = true;
        this.paramsInit = paramsInit;
        this.initParam();
    }

    public WFCausalConv3D(int channel, int kernelNum, int depth, int width, int height, int kernelSize, int stride, boolean hasBias, Network network, ParamsInit paramsInit) {
        this.kernelNum = kernelNum;
        this.channel = channel;
        this.depth = depth;
        this.width = width;
        this.height = height;
        this.kernelSize = kernelSize;
        this.stride = stride;
        this.hasBias = hasBias;
        this.network = network;
        //		network.paramLayers.add(this);
        this.hasParams = true;
        this.paramsInit = paramsInit;
        this.initParam();
    }

    public WFCausalConv3D(int channel, int kernelNum, int depth, int width, int height, int kernelSize, int stride, boolean hasBias, Network network, ActiveType activeType) {
        this.kernelNum = kernelNum;
        this.channel = channel;
        this.depth = depth;
        this.width = width;
        this.height = height;
        this.kernelSize = kernelSize;
        this.stride = stride;
        this.hasBias = hasBias;
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
        int N = 2;
        int C = 3;
        int F = 17;
        int H = 32;
        int W = 32;
        
        int KC = 4;
        int KS = 3;
        int stride = 1;
        
//        float[] data = RandomUtils.order(N * C * F * H * W, 0.1f, 0.1f);
//        Tensor input = new Tensor(N, C * F, H, W, data, true);
        
        String inputPath = "D:\\models\\input_wf.json";
    	Map<String, Object> datas = LagJsonReader.readJsonFileSmallWeight(inputPath);
    	Tensor input = new Tensor(N, C * F, H, W, true);
    	ClipModelUtils.loadData(input, datas, "x", 5);
        
        CNN nn = new CNN(null);
        nn.CUDNN = true;
        nn.number = N;
        //nt channel,int kernelNum,int depth,int width,int height,int kDepth,int kWidth,int kHeight,int padding,int stride
        WFCausalConv3D conv1 = new WFCausalConv3D(C, KC, F, W, H, KS, stride, 0, false, nn);
        
//        float[] w = new float[] {0.3536f, 0.3536f, 0.3536f, 0.3536f, 0.3536f, 0.3536f, 0.3536f, 0.3536f};
        
        conv1.weight = new Tensor(4, 3 * 3, 3, 3, MatrixUtils.order(4 * 3 * 3 * 3 * 3, 0.01f, 0.01f), true);
//        conv1.bias = new Tensor(1, 1, 1, KC, RandomUtils.order(KC, 0.1f, 0.1f), true);
        conv1.forward(input);
//        float[] delta_data = MatrixUtils.val(conv1.getOutput().dataLength, 1.0f);
//        Tensor delta = new Tensor(N, conv1.oChannel * conv1.oDepth, conv1.oHeight, conv1.oWidth, delta_data, true);
//        conv1.back(delta);
        conv1.getOutput().showShape();
        conv1.getOutput().showDM();
        
//        String deltaPath = "D:\\models\\input_wf_delta.json";
//    	Map<String, Object> datas2 = LagJsonReader.readJsonFileSmallWeight(deltaPath);
//    	Tensor delta = new Tensor(N, 4 * 9, 16, 16, true);
//    	ClipModelUtils.loadData(delta, datas2, "dx", 5);
//        
//    	conv1.back(delta);
//    	
//        conv1.diff.showDM();
//        conv1.diff.showDMByOffsetRed(0, conv1.diff.getOnceSize(), "");
//        conv1.diffW.showDM();
//        conv1.diffB.showDM();
    }

    @Override
    public void initParam() {
        // TODO Auto-generated method stub
        int dataLength = kernelNum * channel * kernelSize * kernelSize * kernelSize;
        this.oChannel = this.kernelNum;
        int time_pad = (kernelSize - 1) + Math.max(1 - stride, 0);
//        int height_pad = kernelSize / 2;
//        int width_pad = kernelSize / 2;
        this.padding3d = new int[] {0, 0, 0, 0, time_pad, 0};
//        System.err.println(JsonUtils.toJson(this.padding3d));
        this.pDepth = this.depth + padding3d[4] + padding3d[5];
        this.pHeight = this.height + padding3d[2] + padding3d[3];
        this.pWidth = this.width + padding3d[0] + padding3d[1];
        this.oDepth = (this.pDepth - kernelSize) / this.stride + 1;
        this.oWidth = (this.pWidth + this.padding * 2 - kernelSize) / this.stride + 1;
        this.oHeight = (this.pHeight + this.padding * 2 - kernelSize) / this.stride + 1;
        
        this.weight = new Tensor(kernelNum, channel * kernelSize, kernelSize, kernelSize, RandomUtils.kaiming_uniform(dataLength, this.channel * kernelSize * kernelSize * kernelSize, this.paramsInit), true);
//        this.weight = new Tensor(kernelNum, channel * kernelSize, kernelSize, kernelSize, RandomUtils.order(dataLength, 0.1f, 0.1f), true);
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
                kernel = new Conv3DCudnnKernel(this.network, channel, pDepth, pHeight, pWidth, kernelNum, kernelSize, kernelSize, kernelSize, stride, 0, cuda());
            } else {
                //				kernel = new ConvKernel(channel, height, width, kernelNum, kHeight, kWidth, stride, padding, cuda());
            }
            if (this.hasBias) {
                biasKernel = new BiasKernel(cuda());
            }
            paddingKernel = new PaddingKernel(cuda());
        }
        if(this.pOutput == null || this.number != this.pOutput.number) {
        	this.pOutput = paddingKernel.createOutput(number, channel, height, width, depth, padding3d);
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
            	 kernel = new Conv3DCudnnKernel(this.network, channel, pDepth, pHeight, pWidth, kernelNum, kernelSize, kernelSize, kernelSize, stride, 0, cuda());
            } else {
                //				kernel = new ConvKernel(channel, height, width, kernelNum, kHeight, kWidth, stride, padding, cuda());
            }
            if (this.hasBias) {
                biasKernel = new BiasKernel(cuda());
            }
            paddingKernel = new PaddingKernel(cuda());
        }
        if(this.pOutput == null || this.number != this.pOutput.number) {
        	this.pOutput = paddingKernel.createOutput(input, depth, padding3d);
        }
        if (this.output == null || this.number != this.output.number) {
            this.output = Tensor.createTensor(this.output, number, oChannel * oDepth, oHeight, oWidth, true);
        }
    }

    @Override
    public void initBack() {
        // TODO Auto-generated method stub
        if (this.diff == null || this.number != this.diff.number) {
            this.diff = new Tensor(number, channel * depth, height, width, true);
            if (this.diffW == null) {
                if (!freeze) {
                    if (this.hasBias) {
                        this.diffB = new Tensor(1, 1, 1, kernelNum, true);
                    }
                    this.diffW = new Tensor(this.kernelNum, this.channel * kernelSize, this.kernelSize, this.kernelSize, true);
                }
            }
        }
        if(this.pDiff == null || this.number != this.pDiff.number) {
        	this.pDiff = new Tensor(number, channel * pDepth, pHeight, pWidth, true);
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
                this.diffW = new Tensor(this.kernelNum, this.channel * kernelSize, this.kernelSize, this.kernelSize, true);
            }
        }
        if(this.pDiff == null || this.number != this.pDiff.number) {
        	this.pDiff = new Tensor(number, channel * pDepth, pHeight, pWidth, true);
        }
    }

    @Override
    public void output() {
        // TODO Auto-generated method stub
    	paddingKernel.padding3d_self(input, pOutput, depth, padding3d);
//    	System.err.println(pDepth + ":" + pHeight + ":" + pWidth);
//    	pOutput.showDMByOffsetRed(2 * pHeight * pWidth, pHeight * pWidth, "pOutput");
    	if(padding > 0) {
    		kernel.conv(pOutput, weight, output, pad);
    	}else {
//    		System.err.println("in");
    		kernel.conv(pOutput, weight, output);
    	}

//        System.err.println("weight:"+MatrixOperation.sum(weight.syncHost()));
//        System.err.println("output:"+MatrixOperation.sum(output.syncHost()));
        if (this.hasBias) {
            biasKernel.addConvBiasFast(output, bias, oChannel, oDepth);
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
            kernel.dw(pOutput, delta, diffW);
            /**
             * 计算deltaB

             */
            if (this.hasBias) {
                biasKernel.backwardConv3DBias(oDepth, diffB, delta);
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
            kernel.dx(delta, weight, pDiff);

            paddingKernel.padding3dGrad_self(pDiff, diff, depth, padding3d);

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
            kernel.dw(pOutput, delta, diffW);
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
            kernel.dx(delta, weight, pDiff);
//            pDiff.showDM("pDiff");
            paddingKernel.padding3dGrad_self(pDiff, diff, depth, padding3d);
            
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

