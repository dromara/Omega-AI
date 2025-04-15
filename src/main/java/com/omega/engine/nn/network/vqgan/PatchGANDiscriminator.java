package com.omega.engine.nn.network.vqgan;

import com.omega.common.config.Tensor;
import com.omega.engine.loss.LossFactory;
import com.omega.engine.loss.LossType;
import com.omega.engine.loss.gpu.HingeLossKernel;
import com.omega.engine.nn.layer.InputLayer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.patchgan.PatchGANDiscriminatorBlock;
import com.omega.engine.nn.network.Network;
import com.omega.engine.nn.network.NetworkType;
import com.omega.engine.nn.network.RunModel;
import com.omega.engine.updater.UpdaterType;

import java.io.IOException;
import java.io.RandomAccessFile;

/**
 * PatchGANDiscriminator
 *
 * @author Administrator
 */
public class PatchGANDiscriminator extends Network {
    public PatchGANDiscriminatorBlock disc;
    private int imageSize;
    private int[] convChannels;
    private int[] kernels;
    private int[] strides;
    private int[] paddings;
    private InputLayer inputLayer;
    private HingeLossKernel hingeLossKernel;

    public PatchGANDiscriminator(LossType lossType, UpdaterType updater, int imageSize, int[] convChannels, int[] kernels, int[] strides, int[] paddings) {
        this.lossFunction = LossFactory.create(lossType, this);
        this.imageSize = imageSize;
        this.convChannels = convChannels;
        this.kernels = kernels;
        this.strides = strides;
        this.paddings = paddings;
        this.updater = updater;
        initLayers();
    }

    public void initLayers() {
        hingeLossKernel = new HingeLossKernel(cudaManager);
        this.inputLayer = new InputLayer(3, imageSize, imageSize);
        this.disc = new PatchGANDiscriminatorBlock(3, imageSize, imageSize, convChannels, kernels, strides, paddings, this);
        this.addLayer(inputLayer);
        this.addLayer(disc);
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
        return NetworkType.PATCH_GAN_DISC;
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
        /**
         * 设置输入数据

         */
        this.setInputData(input);
        inputLayer.forward();
        disc.forward(input);
        return disc.getOutput();
    }

    public void initBack() {
    }

    @Override
    public void back(Tensor lossDiff) {
        // TODO Auto-generated method stub
        //		lossDiff.showDMByNumber(0);
        /**
         * 设置误差
         * 将误差值输入到最后一层

         */
        this.setLossDiff(lossDiff);  //only decoder delta
        initBack();
        disc.back(lossDiff);
    }

    @Override
    public Tensor loss(Tensor output, Tensor label) {
        // TODO Auto-generated method stub
        return this.lossFunction.loss(output, label);
    }

    public void hingeGLoss(Tensor output, Tensor loss) {
        tensorOP.mean(output, 0, loss);
    }

    public void hingeDLoss(Tensor real, Tensor fake, Tensor loss) {
        hingeLossKernel.hingeLoss(real, fake, loss);
    }

    public void hingeDLossBack(Tensor real, Tensor fake, Tensor dReal, Tensor dFake) {
        hingeLossKernel.hingeLossBackward(real, fake, dReal, dFake);
    }

    public void hingeDRealLoss(Tensor real, Tensor loss) {
        hingeLossKernel.hingeRealLoss(real, loss);
    }

    public void hingeDFakeLoss(Tensor fake, Tensor loss) {
        hingeLossKernel.hingeFakeLoss(fake, loss);
    }

    public void hingeDRealLossBack(Tensor real, Tensor dReal, float weight) {
        hingeLossKernel.hingeLossRealBackward(real, dReal, weight);
    }

    public void hingeDFakeLossBack(Tensor fake, Tensor dFake, float weight) {
        hingeLossKernel.hingeLossFakeBackward(fake, dFake, weight);
    }

    @Override
    public Tensor lossDiff(Tensor output, Tensor label) {
        // TODO Auto-generated method stub
        Tensor t = this.lossFunction.diff(output, label);
        return t;
    }

    @Override
    public void clearGrad() {
        // TODO Auto-generated method stub
    }

    @Override
    public Tensor loss(Tensor output, Tensor label, Tensor loss) {
        // TODO Auto-generated method stub
        return this.lossFunction.loss(output, label, loss);
    }

    @Override
    public Tensor lossDiff(Tensor output, Tensor label, Tensor diff) {
        // TODO Auto-generated method stub
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
    }

    public void loadModel(RandomAccessFile inputStream) throws IOException {
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

