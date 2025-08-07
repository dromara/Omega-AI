package com.omega.engine.nn.network;

import java.io.IOException;
import java.io.RandomAccessFile;

import com.omega.engine.gpu.CUDAMemoryManager;
import com.omega.engine.loss.LossFactory;
import com.omega.engine.loss.LossType;
import com.omega.engine.loss.gpu.SmoothL1Kernel;
import com.omega.engine.nn.layer.InputLayer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.SoftmaxWithCrossEntropyLayer;
import com.omega.engine.nn.layer.dit.mmdit.MMDiTMoudueRoPE;
import com.omega.engine.tensor.Tensor;
import com.omega.engine.updater.UpdaterType;

import jcuda.Sizeof;
import jcuda.runtime.JCuda;

/**
 * Duffsion Transformer
 *
 * @author Administrator
 */
public class MMDiT_RoPE extends Network {
	
    public int inChannel;
    public int width;
    public int height;
    public int patchSize;
    public int maxContextLen;
    public int hiddenSize;
    private int depth;
    private int timeSteps;
    public int headNum;
    private int textEmbedDim;
    private int mlpRatio = 4;
    private boolean learnSigma = true;
    
    private float y_drop_prob = 0.0f;
    
    private InputLayer inputLayer;
    public MMDiTMoudueRoPE main;
    
    private SmoothL1Kernel smoothL1Kernel;
    
    public MMDiT_RoPE(LossType lossType, UpdaterType updater, int inChannel, int width, int height, int patchSize, int hiddenSize, int headNum, int depth, int timeSteps, int maxContextLen, int textEmbedDim, int mlpRatio, boolean learnSigma, float y_drop_prob) {
        this.lossFunction = LossFactory.create(lossType, this);
        this.updater = updater;
        this.inChannel = inChannel;
        this.width = width;
        this.height = height;
        this.patchSize = patchSize;
        this.headNum = headNum;
        this.hiddenSize = hiddenSize;
        this.depth = depth;
        this.timeSteps = timeSteps;
        this.textEmbedDim = textEmbedDim;
        this.maxContextLen = maxContextLen;
        this.mlpRatio = mlpRatio;
        this.learnSigma = learnSigma;
        this.y_drop_prob = y_drop_prob;
        this.time = (width / patchSize) * (height / patchSize);
        initLayers();
    }

    public void initLayers() {
    	
        this.inputLayer = new InputLayer(inChannel, height, width);
        
        main = new MMDiTMoudueRoPE(inChannel, width, height, patchSize, hiddenSize, headNum, depth, timeSteps, maxContextLen, textEmbedDim, mlpRatio, learnSigma, y_drop_prob, this);
        
        this.addLayer(inputLayer);
        this.addLayer(main);
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
        if(smoothL1Kernel == null) {
        	smoothL1Kernel = new SmoothL1Kernel(this.cudaManager);
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

    public Tensor forward(Tensor input, Tensor t, Tensor context, Tensor cos, Tensor sin) {
        /**
         * 设置输入数据
         */
        this.setInputData(input);
        this.main.forward(input, t, context, cos, sin);
        return this.main.getOutput();
    }

    public void initBack() {
    }

    @Override
    public void back(Tensor lossDiff) {
        // TODO Auto-generated method stub
        initBack();
        /**
         * 设置误差
         * 将误差值输入到最后一层
         */
        this.setLossDiff(lossDiff);
        this.main.back(lossDiff);
    }
    
    public void back(Tensor lossDiff, Tensor cos, Tensor sin) {
        // TODO Auto-generated method stub
        initBack();
        /**
         * 设置误差
         * 将误差值输入到最后一层
         */
        this.setLossDiff(lossDiff);
        this.main.back(lossDiff, cos, sin);
    }
    
    @Override
    public Tensor loss(Tensor output, Tensor label) {
        // TODO Auto-generated method stub
        switch (this.getLastLayer().getLayerType()) {
            case softmax:
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
        this.main.update();
    }

    @Override
    public void clearGrad() {
        // TODO Auto-generated method stub
        /**
         * forward
         */
        JCuda.cudaMemset(CUDAMemoryManager.workspace.getPointer(), 0, CUDAMemoryManager.workspace.getSize() * Sizeof.FLOAT);
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
    	main.saveModel(outputStream);
        System.out.println("tail save success...");
    }

    public void loadModel(RandomAccessFile inputStream) throws IOException {
    	main.loadModel(inputStream);
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
    
    public Tensor smoothL1(Tensor s,Tensor t,Tensor alignLoss,float beta) {
    	smoothL1Kernel.forward(s, t, alignLoss, beta);
    	return alignLoss;
    }
    
    public Tensor smoothL1Back(Tensor s,Tensor t,Tensor sxrDiff,float beta) {
    	smoothL1Kernel.backward(s, t, sxrDiff, beta);
    	return sxrDiff;
    }
    
}

