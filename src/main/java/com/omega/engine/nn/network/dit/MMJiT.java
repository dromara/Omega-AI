package com.omega.engine.nn.network.dit;

import java.io.IOException;
import java.io.RandomAccessFile;

import com.omega.engine.gpu.CUDAMemoryManager;
import com.omega.engine.loss.LossFactory;
import com.omega.engine.loss.LossType;
import com.omega.engine.nn.layer.InputLayer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.SoftmaxWithCrossEntropyLayer;
import com.omega.engine.nn.layer.jit.MMJiTMainMoudue;
import com.omega.engine.nn.network.Network;
import com.omega.engine.nn.network.NetworkType;
import com.omega.engine.nn.network.RunModel;
import com.omega.engine.tensor.Tensor;
import com.omega.engine.updater.UpdaterType;
import com.omega.example.dit.models.ICPlan;

import jcuda.Sizeof;
import jcuda.runtime.JCuda;

/**
 * Duffsion Transformer
 *
 * @author Administrator
 */
public class MMJiT extends Network {
	
    public int inChannel;
    public int width;
    public int height;
    public int patchSize;
    public int bottleneck_dim;
    public int textEmbedDim;
    public int maxContextLen;
    public int hiddenSize;
    private int txt_depth;
    private int depth;
    public int headNum;
    
    private float y_drop_prob = 0.0f;
    
    private InputLayer inputLayer;
    public MMJiTMainMoudue main;
    
    private Tensor input_null;
    private Tensor uncond_eps;
    
    public MMJiT(LossType lossType, UpdaterType updater, int inChannel, int width, int height, int patchSize, int bottleneck_dim, int hiddenSize, int headNum, int txt_depth, int depth, int textEmbedDim, int maxContextLen, float y_drop_prob) {
        this.lossFunction = LossFactory.create(lossType, this);
//        this.weight_decay = 0.1f;
        this.updater = updater;
        this.inChannel = inChannel;
        this.width = width;
        this.height = height;
        this.patchSize = patchSize;
        this.bottleneck_dim = bottleneck_dim;
        this.headNum = headNum;
        this.hiddenSize = hiddenSize;
        this.depth = depth;
        this.textEmbedDim = textEmbedDim;
        this.maxContextLen = maxContextLen;
        this.txt_depth = txt_depth;
        this.y_drop_prob = y_drop_prob;
        this.time = (width / patchSize) * (height / patchSize);
        initLayers();
    }

    public void initLayers() {
    	
        this.inputLayer = new InputLayer(inChannel, height, width);
        
        main = new MMJiTMainMoudue(inChannel, width, height, patchSize, bottleneck_dim, hiddenSize, headNum, txt_depth, depth, textEmbedDim, maxContextLen, y_drop_prob, this);
        
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
        System.out.println("the network is ready.");
    }

    @Override
    public NetworkType getNetworkType() {
        // TODO Auto-generated method stub
        return NetworkType.DiT;
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

    public Tensor forward(Tensor input, Tensor t, Tensor context) {
        /**
         * 设置输入数据
         */
        this.setInputData(input);
        this.main.forward(input, t, context);
        return this.main.getOutput();
    }
    
    public Tensor forward(Tensor input, Tensor context, Tensor cos1d, Tensor sin1d, Tensor cos2d, Tensor sin2d) {
        /**
         * 设置输入数据
         */
        this.setInputData(input);
        this.main.forward(input, context, cos1d, sin1d, cos2d, sin2d);
        return this.main.getOutput();
    }
    
    public Tensor forward_with_cfg(ICPlan icplan, Tensor input, Tensor t, Tensor context, Tensor null_context, Tensor cos1d, Tensor sin1d, Tensor cos2d, Tensor sin2d, Tensor eps, float cfg_scale) {
        /**
         * 设置输入数据
         */
        if(input_null == null || input_null.number != input.number) {
    		input_null = Tensor.createGPUTensor(input_null, input.number, input.channel, input.height, input.width, true);
    	}
        input.copyGPU(input_null);
        this.main.forward(input, context, cos1d, sin1d, cos2d, sin2d);
        this.main.getOutput().copyGPU(eps);
        this.main.forward(input_null, null_context, cos1d, sin1d, cos2d, sin2d);
        uncond_eps = this.main.getOutput();
        
        /**
         *  v_cond = (out_cond - z) / (1.0 - t).clamp_min(self.t_eps)
         *  v_uncond = (out_uncond - z) / (1.0 - t).clamp_min(self.t_eps)
         */
        icplan.compute_v(eps, t, input, eps, 5e-2f);
        icplan.compute_v(uncond_eps, t, input, uncond_eps, 5e-2f);
        
        /**
         * out = uncond_eps + cfg_scale * (eps - uncond_eps)
         */
        tensorOP.sub(eps, uncond_eps, eps);
        tensorOP.mul(eps, cfg_scale, eps);
        tensorOP.add(uncond_eps, eps, eps);
        return eps;
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
        this.main.back(lossDiff);
        //		this.unet.diff.showDMByOffset(0, 100, "unet.diff");
    }
    
    public void back(Tensor lossDiff, Tensor cos1d, Tensor sin1d, Tensor cos2d, Tensor sin2d) {
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
        this.main.back(lossDiff, cos1d, sin1d, cos2d, sin2d);
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
    
    public void accGrad(int steps) {
        float scale = 1.0f / steps;
        main.accGrad(scale);
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
    
}

