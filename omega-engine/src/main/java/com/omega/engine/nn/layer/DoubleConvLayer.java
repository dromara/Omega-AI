package com.omega.engine.nn.layer;

import com.omega.common.data.Tensor;
import com.omega.engine.active.ActiveType;
import com.omega.engine.nn.layer.active.*;
import com.omega.engine.nn.layer.normalization.BNLayer;
import com.omega.engine.nn.network.Network;
import com.omega.engine.updater.UpdaterFactory;

/**
 * conv + bn + activefunction * 2
 *
 * @author Administrator
 */
public class DoubleConvLayer extends Layer {
    private int kHeight = 3;
    private int kWidth = 3;
    private int padding = 1;
    private int stride = 2;
    private int midChannel = 0;
    private ConvolutionLayer convLayer;
    private BNLayer bnLayer;
    private ActiveFunctionLayer activeLayer;
    private ConvolutionLayer convLayer2;
    private BNLayer bnLayer2;
    private ActiveFunctionLayer activeLayer2;
    private ActiveType activeType;

    public DoubleConvLayer(int channel, int oChannel, int height, int width, ActiveType activeType, Network network) {
        this.network = network;
        this.activeType = activeType;
        this.channel = channel;
        this.oChannel = oChannel;
        this.height = height;
        this.width = width;
        this.kHeight = 3;
        this.kWidth = 3;
        this.stride = 1;
        this.padding = 1;
        this.oHeight = (height + padding * 2 - kHeight) / stride + 1;
        this.oWidth = (width + padding * 2 - kWidth) / stride + 1;
        initLayers();
    }

    public DoubleConvLayer(int channel, int oChannel, int midChannel, int height, int width, ActiveType activeType, Network network) {
        this.network = network;
        this.activeType = activeType;
        this.channel = channel;
        this.oChannel = oChannel;
        this.midChannel = midChannel;
        this.height = height;
        this.width = width;
        this.kHeight = 3;
        this.kWidth = 3;
        this.stride = 1;
        this.padding = 1;
        this.oHeight = (height + padding * 2 - kHeight) / stride + 1;
        this.oWidth = (width + padding * 2 - kWidth) / stride + 1;
        initLayers();
    }

    public DoubleConvLayer(int channel, int oChannel, int height, int width, String activeType, Network network) {
        this.network = network;
        this.activeType = ActiveType.valueOf(activeType);
        this.channel = channel;
        this.oChannel = oChannel;
        this.height = height;
        this.width = width;
        this.kHeight = 3;
        this.kWidth = 3;
        this.stride = 1;
        this.padding = 1;
        this.oHeight = (height + padding * 2 - kHeight) / stride + 1;
        this.oWidth = (width + padding * 2 - kWidth) / stride + 1;
        initLayers();
    }

    public void initLayers() {
        if (midChannel == 0) {
            midChannel = oChannel;
        }
        convLayer = new ConvolutionLayer(channel, midChannel, width, height, 3, 3, 1, 1, false, this.network, activeType);
        convLayer.setUpdater(UpdaterFactory.create(this.network));
        convLayer2 = new ConvolutionLayer(midChannel, oChannel, width, height, 3, 3, 1, 1, false, this.network, activeType);
        convLayer2.setUpdater(UpdaterFactory.create(this.network));
        bnLayer = new BNLayer(convLayer);
        bnLayer2 = new BNLayer(convLayer2);
        //		System.out.println("bnLayer.preLayer:"+bnLayer.preLayer);
        switch (activeType) {
            case sigmoid:
                activeLayer = new SigmodLayer(bnLayer);
                activeLayer2 = new SigmodLayer(bnLayer2);
                break;
            case relu:
                activeLayer = new ReluLayer(bnLayer);
                activeLayer2 = new ReluLayer(bnLayer2);
                break;
            case leaky_relu:
                activeLayer = new LeakyReluLayer(bnLayer);
                activeLayer2 = new LeakyReluLayer(bnLayer2);
                break;
            case tanh:
                activeLayer = new TanhLayer(bnLayer);
                activeLayer2 = new TanhLayer(bnLayer2);
                break;
            case silu:
                activeLayer = new SiLULayer(bnLayer);
                activeLayer2 = new SiLULayer(bnLayer2);
                break;
            default:
                throw new RuntimeException("The doubleConv layer is not support the [" + activeType + "] active function.");
        }
        //		System.out.println("activeLayer.preLayer:"+activeLayer.preLayer);
    }

    @Override
    public void init() {
        // TODO Auto-generated method stub
        this.number = this.network.number;
    }

    @Override
    public void initBack() {
        // TODO Auto-generated method stub
        //		if(this.org_delta == null || output.number != org_delta.number){
        //			this.org_delta = Tensor.createTensor(org_delta, number, output.channel, output.height, output.width, true);
        //		}
    }

    @Override
    public void initParam() {
        // TODO Auto-generated method stub
    }

    @Override
    public void output() {
        // TODO Auto-generated method stub
        //		this.input.showShape();
        convLayer.forward(this.input);
        bnLayer.forward(convLayer.output);
        activeLayer.forward(bnLayer.output);
        convLayer2.forward(activeLayer.output);
        bnLayer2.forward(convLayer2.output);
        activeLayer2.forward(bnLayer2.output);
        this.output = activeLayer2.output;
    }

    @Override
    public Tensor getOutput() {
        // TODO Auto-generated method stub
        return this.output;
    }

    @Override
    public void diff() {
        // TODO Auto-generated method stub
        //		System.out.println("index:["+index+"]("+oChannel+")"+this.delta);
        //		baseKernel.copy_gpu(delta, this.org_delta, delta.getDataLength(), 1, 1);
        activeLayer2.back(delta);
        bnLayer2.back(activeLayer2.diff);
        convLayer2.back(bnLayer2.diff);
        activeLayer.back(convLayer2.diff);
        bnLayer.back(activeLayer.diff);
        convLayer.back(bnLayer.diff);
        this.diff = convLayer.diff;
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
    public void backTemp() {
        // TODO Auto-generated method stub
    }

    @Override
    public void forward(Tensor inpnut) {
        // TODO Auto-generated method stub
        /**
         * 参数初始化

         */
        this.init();
        /**
         * 设置输入

         */
        this.setInput(inpnut);
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
    }

    @Override
    public void update() {
        // TODO Auto-generated method stub
        convLayer.update();
        bnLayer.update();
        convLayer2.update();
        bnLayer2.update();
    }

    @Override
    public void showDiff() {
        // TODO Auto-generated method stub
    }

    @Override
    public LayerType getLayerType() {
        // TODO Auto-generated method stub
        return LayerType.double_conv;
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

    public ConvolutionLayer getConvLayer() {
        return convLayer;
    }

    public BNLayer getBnLayer() {
        return bnLayer;
    }

    @Override
    public void accGrad(float scale) {
        // TODO Auto-generated method stub
        convLayer.accGrad(scale);
        bnLayer.accGrad(scale);
        convLayer2.accGrad(scale);
        bnLayer2.accGrad(scale);
    }
}

