package com.omega.engine.nn.network;

import com.omega.engine.gpu.CUDAMemoryManager;
import com.omega.engine.loss.LossFactory;
import com.omega.engine.loss.LossType;
import com.omega.engine.nn.layer.InputLayer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.SoftmaxWithCrossEntropyLayer;
import com.omega.engine.nn.layer.diffusion.unet.UNetCond2;
import com.omega.engine.tensor.Tensor;
import com.omega.engine.updater.UpdaterType;
import jcuda.Sizeof;
import jcuda.runtime.JCuda;

import java.io.IOException;
import java.io.RandomAccessFile;

/**
 * Duffsion-UNet
 *
 * @author Administrator
 */
public class DiffusionUNetCond2 extends Network {
    public int inChannel;
    public int maxContextLen;
    public int width;
    public int height;
    private int[] downChannels;
    private int numLayer;
    private int timeSteps;
    private int tEmbDim;
    private int groupNum = 32;
    private int headNum;
    private int textEmbedDim;
    private InputLayer inputLayer;
    private UNetCond2 unet;

    public DiffusionUNetCond2(LossType lossType, UpdaterType updater, int inChannel, int width, int height, int[] downChannels, int headNum, int numLayer, int timeSteps, int tEmbDim, int maxContextLen, int textEmbedDim, int groupNum) {
        this.lossFunction = LossFactory.create(lossType, this);
        this.updater = updater;
        this.inChannel = inChannel;
        this.width = width;
        this.height = height;
        this.headNum = headNum;
        this.downChannels = downChannels;
        this.numLayer = numLayer;
        this.timeSteps = timeSteps;
        this.tEmbDim = tEmbDim;
        this.textEmbedDim = textEmbedDim;
        this.maxContextLen = maxContextLen;
        this.groupNum = groupNum;
        initLayers();
    }

    public void initLayers() {
        this.inputLayer = new InputLayer(inChannel, height, width);
        unet = new UNetCond2(inChannel, height, width, downChannels, headNum, tEmbDim, timeSteps, maxContextLen, textEmbedDim, numLayer, groupNum, this);
        this.addLayer(inputLayer);
        this.addLayer(unet);
    }

    @Override
    public void init() throws Exception {
        // TODO Auto-generated method stub
        if (layerList.size() <= 1) {
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
        return NetworkType.DUFFSION_UNET_COND;
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
        return null;
    }

    public Tensor forward(Tensor input, Tensor t) {
        /**
         * 设置输入数据

         */
        this.setInputData(input);
        this.unet.forward(input, t);
        return this.unet.getOutput();
    }

    public Tensor forward(Tensor input, Tensor t, Tensor context) {
        /**
         * 设置输入数据

         */
        this.setInputData(input);
        this.unet.forward(input, t, context);
        return this.unet.getOutput();
    }

    public void initBack() {
    }

    @Override
    public void back(Tensor lossDiff) {
        // TODO Auto-generated method stub
        //		lossDiff.showDMByNumber(0);
        initBack();
        /**
         * 设置误差
         * 将误差值输入到最后一层

         */
        //		lossDiff.showDMByOffset(0, 100, "lossDiff");
        this.setLossDiff(lossDiff);
        //		lossDiff.showDM("lossDiff");
        this.unet.back(lossDiff);
        //		this.unet.diff.showDMByOffset(0, 100, "unet.diff");
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

    public void update() {
        this.train_time += 1;
        this.unet.update();
    }

    @Override
    public void clearGrad() {
        // TODO Auto-generated method stub
        /**
         * forward

         */
        JCuda.cudaMemset(CUDAMemoryManager.workspace.getPointer(), 0, CUDAMemoryManager.workspace.getSize() * Sizeof.FLOAT);
        this.clearCacheDelta();
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

    public void saveModel(RandomAccessFile outputStream) throws IOException {
        unet.saveModel(outputStream);
        System.out.println("tail save success...");
    }

    public void loadModel(RandomAccessFile inputStream) throws IOException {
        unet.loadModel(inputStream);
        System.out.println("tail load success...");
    }

    @Override
    public void putParamters() {
        // TODO Auto-generated method stub
    }

    @Override
    public void putParamterGrads() {
        // TODO Auto-generated method stub
    }
}

