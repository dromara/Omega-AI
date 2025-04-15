package com.omega.engine.nn.network;

import com.omega.common.config.Tensor;
import com.omega.engine.active.ActiveType;
import com.omega.engine.gpu.CUDAMemoryManager;
import com.omega.engine.loss.LossFactory;
import com.omega.engine.loss.LossType;
import com.omega.engine.nn.layer.*;
import com.omega.engine.nn.layer.active.SigmodLayer;
import com.omega.engine.pooling.PoolingType;
import com.omega.engine.updater.UpdaterType;
import jcuda.Sizeof;
import jcuda.runtime.JCuda;

/**
 * Recurrent Neural Networks
 *
 * @author Administrator
 */
public class SimpleUNet extends Network {
    public int inChannel;
    public int outChannel;
    public int width;
    public int height;
    private boolean bias = true;
    private InputLayer inputLayer;
    private DoubleConvLayer conv1;
    private PoolingLayer pool1;
    private DoubleConvLayer conv2;
    private PoolingLayer pool2;
    private DoubleConvLayer conv3;
    private PoolingLayer pool3;
    private DoubleConvLayer conv4;
    private PoolingLayer pool4;
    private DoubleConvLayer conv5;
    private ConvolutionTransposeLayer up6;
    private DoubleConvLayer conv6;
    private ConvolutionTransposeLayer up7;
    private DoubleConvLayer conv7;
    private ConvolutionTransposeLayer up8;
    private DoubleConvLayer conv8;
    private ConvolutionTransposeLayer up9;
    private DoubleConvLayer conv9;
    private RouteLayer cat1;
    private RouteLayer cat2;
    private RouteLayer cat3;
    private RouteLayer cat4;
    private ConvolutionLayer outc;
    private SigmodLayer act;

    public SimpleUNet(LossType lossType, UpdaterType updater, int inChannel, int outChannel, int width, int height, boolean bias) {
        this.lossFunction = LossFactory.create(lossType, this);
        this.bias = bias;
        this.updater = updater;
        this.setWidth(width);
        this.height = height;
        this.inChannel = inChannel;
        this.outChannel = outChannel;
        initLayers();
    }

    public void initLayers() {
        this.inputLayer = new InputLayer(inChannel, height, getWidth());
        this.conv1 = new DoubleConvLayer(inChannel, 32, height, getWidth(), ActiveType.relu, this);
        this.pool1 = new PoolingLayer(conv1.oChannel, conv1.oWidth, conv1.oHeight, 2, 2, 2, PoolingType.MAX_POOLING);
        this.conv2 = new DoubleConvLayer(32, 64, pool1.oHeight, pool1.oWidth, ActiveType.relu, this);
        this.pool2 = new PoolingLayer(conv2.oChannel, conv2.oWidth, conv2.oHeight, 2, 2, 2, PoolingType.MAX_POOLING);
        this.conv3 = new DoubleConvLayer(64, 128, pool2.oHeight, pool2.oWidth, ActiveType.relu, this);
        this.pool3 = new PoolingLayer(conv3.oChannel, conv3.oWidth, conv3.oHeight, 2, 2, 2, PoolingType.MAX_POOLING);
        this.conv4 = new DoubleConvLayer(128, 256, pool3.oHeight, pool3.oWidth, ActiveType.relu, this);
        this.pool4 = new PoolingLayer(conv4.oChannel, conv4.oWidth, conv4.oHeight, 2, 2, 2, PoolingType.MAX_POOLING);
        this.conv5 = new DoubleConvLayer(256, 512, pool4.oHeight, pool4.oWidth, ActiveType.relu, this);
        this.up6 = new ConvolutionTransposeLayer(512, 256, conv5.oWidth, conv5.oHeight, 2, 2, 0, 2, 1, 0);
        this.cat1 = new RouteLayer(new Layer[]{up6, conv4});
        this.conv6 = new DoubleConvLayer(512, 256, cat1.oHeight, cat1.oWidth, ActiveType.relu, this);
        this.up7 = new ConvolutionTransposeLayer(256, 128, up6.oWidth, up6.oHeight, 2, 2, 0, 2, 1, 0);
        this.cat2 = new RouteLayer(new Layer[]{up7, conv3});
        this.conv7 = new DoubleConvLayer(256, 128, cat2.oHeight, cat2.oWidth, ActiveType.relu, this);
        this.up8 = new ConvolutionTransposeLayer(128, 64, up7.oWidth, up7.oHeight, 2, 2, 0, 2, 1, 0);
        this.cat3 = new RouteLayer(new Layer[]{up8, conv2});
        this.conv8 = new DoubleConvLayer(128, 64, cat3.oHeight, cat3.oWidth, ActiveType.relu, this);
        this.up9 = new ConvolutionTransposeLayer(64, 32, up8.oWidth, up8.oHeight, 2, 2, 0, 2, 1, 0);
        this.cat4 = new RouteLayer(new Layer[]{up9, conv1});
        this.conv9 = new DoubleConvLayer(64, 32, cat4.oHeight, cat4.oWidth, ActiveType.relu, this);
        this.outc = new ConvolutionLayer(32, outChannel, conv9.oHeight, conv9.oWidth, 1, 1, 0, 1, bias, this);
        this.act = new SigmodLayer(outc);
        this.addLayer(inputLayer);
        this.addLayer(conv1);
        this.addLayer(pool1);
        this.addLayer(conv2);
        this.addLayer(pool2);
        this.addLayer(conv3);
        this.addLayer(pool3);
        this.addLayer(conv4);
        this.addLayer(pool4);
        this.addLayer(conv5);
        this.addLayer(up6);
        this.addLayer(cat1);
        this.addLayer(conv6);
        this.addLayer(up7);
        this.addLayer(cat2);
        this.addLayer(conv7);
        this.addLayer(up8);
        this.addLayer(cat3);
        this.addLayer(conv8);
        this.addLayer(up9);
        this.addLayer(cat4);
        this.addLayer(conv9);
        this.addLayer(outc);
        this.addLayer(act);
    }

    @Override
    public void init() throws Exception {
        // TODO Auto-generated method stub
        if (layerList.size() <= 0) {
            throw new Exception("layer size must greater than 2.");
        }
        this.layerCount = layerList.size();
        this.setChannel(layerList.get(0).channel);
        this.setHeight(layerList.get(0).height);
        this.setWidth(layerList.get(0).width);
        this.oChannel = this.getLastLayer().oChannel;
        this.oHeight = this.getLastLayer().oHeight;
        this.oWidth = this.getLastLayer().oWidth;
        if (layerList.get(0).getLayerType() != LayerType.input) {
            throw new Exception("first layer must be input layer.");
        }
        if ((layerList.get(layerList.size() - 1).getLayerType() == LayerType.softmax || layerList.get(layerList.size() - 1).getLayerType() == LayerType.softmax_cross_entropy) && this.lossFunction.getLossType() != LossType.cross_entropy) {
            throw new Exception("The softmax function support only cross entropy loss function now.");
        }
        System.out.println("the network is ready.");
    }

    @Override
    public NetworkType getNetworkType() {
        // TODO Auto-generated method stub
        return NetworkType.UNET;
    }

    @Override
    public Tensor predict(Tensor input) {
        // TODO Auto-generated method stub
        this.RUN_MODEL = RunModel.TEST;
        this.forward(input);
        return this.getOutput();
    }

    @Override
    public Tensor forward(Tensor input) {
        //		System.out.println("en_time:"+en_time+",de_time:"+de_time);
        /**
         * 设置输入数据

         */
        this.setInputData(input);
        for (int i = 0; i < layerList.size(); i++) {
            //			System.out.println(i);
            layerList.get(i).forward();
        }
        return this.act.getOutput();
    }

    @Override
    public void back(Tensor lossDiff) {
        // TODO Auto-generated method stub
        //		lossDiff.showDMByNumber(0);
        /**
         * 设置误差
         * 将误差值输入到最后一层

         */
        this.setLossDiff(lossDiff);
        for (int i = layerList.size() - 1; i >= 0; i--) {
            layerList.get(i).back();
        }
    }

    @Override
    public Tensor loss(Tensor output, Tensor label) {
        // TODO Auto-generated method stub
        switch (this.getLastLayer().getLayerType()) {
            case softmax:
                //			SoftmaxLayer softmaxLayer = (SoftmaxLayer)this.getLastLayer();
                //			softmaxLayer.setCurrentLabel(label);
                break;
            case softmax_cross_entropy:
                SoftmaxWithCrossEntropyLayer softmaxWithCrossEntropyLayer = (SoftmaxWithCrossEntropyLayer) this.getLastLayer();
                softmaxWithCrossEntropyLayer.setCurrentLabel(label);
                break;
            default:
                break;
        }
        return this.lossFunction.loss(output, label);
    }

    @Override
    public Tensor lossDiff(Tensor output, Tensor label) {
        // TODO Auto-generated method stub
        this.clearGrad();
        Tensor t = this.lossFunction.diff(output, label);
        //		PrintUtils.printImage(t.data);
        return t;
    }

    @Override
    public void clearGrad() {
        // TODO Auto-generated method stub
        /**
         * forward

         */
        JCuda.cudaMemset(CUDAMemoryManager.workspace.getPointer(), 0, CUDAMemoryManager.workspace.getSize() * Sizeof.FLOAT);
        for (int i = 0; i < layerCount; i++) {
            Layer layer = layerList.get(i);
            if (layer.cache_delta != null) {
                layer.cache_delta.clearGPU();
            }
        }
        JCuda.cudaDeviceSynchronize();
    }

    @Override
    public Tensor loss(Tensor output, Tensor label, Tensor loss) {
        // TODO Auto-generated method stub
        return this.lossFunction.loss(output, label, loss);
    }

    @Override
    public Tensor lossDiff(Tensor output, Tensor label, Tensor diff) {
        // TODO Auto-generated method stub
        this.clearGrad();
        return this.lossFunction.diff(output, label, diff);
    }

    public Tensor loss(Tensor output, Tensor label, int igonre) {
        // TODO Auto-generated method stub
        return this.lossFunction.loss(output, label, igonre);
    }

    public Tensor lossDiff(Tensor output, Tensor label, int igonre) {
        // TODO Auto-generated method stub
        return this.lossFunction.diff(output, label, igonre);
    }

    @Override
    public void putParamters() {
        // TODO Auto-generated method stub
    }

    @Override
    public void putParamterGrads() {
        // TODO Auto-generated method stub
    }
    //	public void saveModel(RandomAccessFile outputStream) throws IOException {
    //		decoder.saveModel(outputStream);
    //		System.out.println("decoder save success...");
    //		fullyLayer.saveModel(outputStream);
    //		System.out.println("fullyLayer save success...");
    //	}
    //
    //	public void loadModel(RandomAccessFile inputStream) throws IOException {
    //		decoder.loadModel(inputStream);
    //		fullyLayer.loadModel(inputStream);
    //	}
}

