package com.omega.engine.nn.layer.causalvae;

import java.io.IOException;
import java.io.RandomAccessFile;

import com.omega.common.utils.MatrixUtils;
import com.omega.common.utils.RandomUtils;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.UPSample3DLayer;
import com.omega.engine.nn.model.LayerInit;
import com.omega.engine.nn.network.CNN;
import com.omega.engine.nn.network.Network;
import com.omega.engine.nn.network.utils.ModelUtils;
import com.omega.engine.tensor.Tensor;
import com.omega.engine.updater.UpdaterFactory;

/**
 * ConvolutionLayer
 *
 * @author Administrator
 */
public class InterpolateConvUpSampleLayer extends Layer {
	
    public int kernelNum = 0;
    public int kernelSize = 0;
    
    public int scale;
    
    public int depth = 0;
    public int oDepth = 0;
    
    private UPSample3DLayer upsample;
    
    private CausalConv3DLayer conv;

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
    public InterpolateConvUpSampleLayer(int channel, int kernelNum, int depth, int width, int height, int kernelSize,int scale, boolean hasBias, Network network) {
        this.kernelNum = kernelNum;
        this.channel = channel;
        this.depth = depth;
        this.scale = scale;
        this.width = width;
        this.height = height;
        this.kernelSize = kernelSize;
        this.hasBias = hasBias;
        this.network = network;
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
    public InterpolateConvUpSampleLayer(int channel, int kernelNum, int depth, int width, int height, int kernelSize,int scale, boolean hasBias, boolean freeze, Network network) {
    	this.kernelNum = kernelNum;
        this.channel = channel;
        this.depth = depth;
        this.scale = scale;
        this.width = width;
        this.height = height;
        this.kernelSize = kernelSize;
		this.hasBias = hasBias;
		this.network = network;
		this.freeze = freeze;
		this.setUpdater(UpdaterFactory.create(this.network));
		this.hasParams = true;
		this.initParam();
    }

    public static void main(String[] args) {
        int N = 4;
        int C = 3;
        int F = 8;
        int H = 4;
        int W = 4;
        
        int KC = 4;
        int KS = 3;
        
        int scale = 2;
        
        float[] data = RandomUtils.order(N * C * F * H * W, 0.1f, 0.1f);
        Tensor input = new Tensor(N, C * F, H, W, data, true);
        CNN nn = new CNN(null);
        nn.CUDNN = true;
        nn.number = N;
        //int channel, int kernelNum, int depth, int width, int height, int kernelSize, boolean hasBias, Network network
        InterpolateConvUpSampleLayer conv1 = new InterpolateConvUpSampleLayer(C, KC, F, W, H, KS, scale, true, nn);

        conv1.conv.weight = new Tensor(KC, C * KS, KS, KS, RandomUtils.order(KC * C * KS * KS * KS, 0.1f, 0.1f), true);
        conv1.conv.bias = new Tensor(1, 1, 1, KC, RandomUtils.order(KC, 0.1f, 0.1f), true);
        conv1.forward(input);
        float[] delta_data = MatrixUtils.val(conv1.getOutput().dataLength, 1.0f);
        Tensor delta = new Tensor(N, conv1.oChannel * conv1.oDepth, conv1.oHeight, conv1.oWidth, delta_data, true);
        conv1.back(delta);
        conv1.getOutput().showShape();
        conv1.getOutput().showDM();
        conv1.diff.showDM();
        conv1.conv.diffW.showDM();
        conv1.conv.diffB.showDM();
    }

    @Override
    public void initParam() {
        // TODO Auto-generated method stub

    	upsample = new UPSample3DLayer(channel, depth, height, width, scale, network);
    	
    	int ud = upsample.oDepth;
    	int uh = upsample.oHeight;
    	int uw = upsample.oWidth;
    	
    	conv = new CausalConv3DLayer(channel, kernelNum, ud, uw, uh, kernelSize, kernelSize, kernelSize, 1, false, hasBias, network);
    	
        this.oChannel = this.kernelNum;
       
        this.oDepth = conv.oDepth;
        this.oWidth = conv.oWidth;
        this.oHeight = conv.oHeight;
    }

    @Override
    public void init() {
        // TODO Auto-generated method stub
        this.number = this.network.number;
    }

    public void init(Tensor input) {
        // TODO Auto-generated method stub
        this.number = input.number;
    }

    @Override
    public void initBack() {
        // TODO Auto-generated method stub
       
    }

    @Override
    public void output() {
        // TODO Auto-generated method stub
    	upsample.forward(input);
    	
    	conv.forward(upsample.getOutput());
    	
    	this.output = conv.getOutput();
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
    	conv.back(delta);
    	
    	upsample.back(conv.diff);
    	
    	this.diff = upsample.diff;
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
       
    }
    
}

