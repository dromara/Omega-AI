package com.omega.engine.nn.network;

import com.omega.engine.gpu.CUDAMemoryManager;
import com.omega.engine.loss.LossFactory;
import com.omega.engine.loss.LossFunction;
import com.omega.engine.loss.LossType;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.tensor.Tensor;
import com.omega.engine.updater.UpdaterType;
import jcuda.Sizeof;
import jcuda.runtime.JCuda;

/**
 * yolo model
 *
 * @author Administrator
 */
public class Yolo extends OutputsNetwork {
    private LossFunction[] losses;
    private LossType lossType;
    private Tensor[] loss;
    private Tensor[] lossDiff;
    private int class_num = 1;

    public Yolo(LossFunction lossFunction) {
        this.lossFunction = lossFunction;
    }

    public Yolo(LossType lossType, UpdaterType updater) {
        this.lossType = lossType;
        this.updater = updater;
    }

    public void initLayer() {
        for (int i = 0; i < layerCount; i++) {
            Layer layer = layerList.get(i);
            layer.init();
        }
    }

    @Override
    public void init() throws Exception {
        // TODO Auto-generated method stub
        if (layerList.size() <= 0) {
            throw new Exception("layer size must greater than 2.");
        }
        this.layerCount = layerList.size();
        if (layerList.get(0).getLayerType() != LayerType.input) {
            throw new Exception("first layer must be input layer.");
        }
        if ((layerList.get(layerList.size() - 1).getLayerType() == LayerType.softmax || layerList.get(layerList.size() - 1).getLayerType() == LayerType.softmax_cross_entropy) && this.lossFunction.getLossType() != LossType.cross_entropy) {
            throw new Exception("The softmax function support only cross entropy loss function now.");
        }
        Layer inputLayer = layerList.get(0);
        this.setChannel(inputLayer.channel);
        this.setHeight(inputLayer.height);
        this.setWidth(inputLayer.width);
        if (this.outputNum > 1) {
            if (class_num != 1 && class_num != 0) {
                this.losses = LossFactory.create(lossType, outputLayers, class_num, this);
            } else {
                this.losses = LossFactory.create(lossType, outputLayers, this);
            }
            if (this.loss == null) {
                this.loss = new Tensor[this.outputNum];
            }
            if (this.lossDiff == null) {
                this.lossDiff = new Tensor[this.outputNum];
            }
        } else {
            if (class_num != 1 && class_num != 0) {
                this.lossFunction = LossFactory.create(lossType, class_num, this);
            } else {
                this.lossFunction = LossFactory.create(lossType, this);
            }
        }
        System.out.println("the network is ready.");
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
        // TODO Auto-generated method stub
        /**
         * 设置输入数据

         */
        this.setInputData(input);
        /**
         * forward

         */
        for (int i = 0; i < layerCount; i++) {
            Layer layer = layerList.get(i);
            layer.forward();
        }
        return this.getOutput();
    }

    @Override
    public void back(Tensor lossDiff) {
        // TODO Auto-generated method stub
        /**
         * 设置误差
         * 将误差值输入到最后一层

         */
        this.setLossDiff(lossDiff);
        for (int i = layerCount - 1; i >= 0; i--) {
            Layer layer = layerList.get(i);
            layer.learnRate = this.learnRate;
            layer.back();
        }
    }

    @Override
    public void back(Tensor[] lossDiffs) {
        /**
         * 设置误差
         * 将误差值输入到最后一层

         */
        this.setLossDiff(lossDiffs);
        for (int i = layerCount - 1; i >= 0; i--) {
            Layer layer = layerList.get(i);
            layer.learnRate = this.learnRate;
            layer.back();
        }
    }

    @Override
    public Tensor loss(Tensor output, Tensor label) {
        // TODO Auto-generated method stub
        return this.lossFunction.loss(output, label);
    }

    @Override
    public Tensor[] loss(Tensor label) {
        // TODO Auto-generated method stub
        for (int i = 0; i < losses.length; i++) {
            this.loss[i] = losses[i].loss(getOutputs()[i], label);
        }
        return this.loss;
    }

    public Tensor[] loss(Tensor[] outputs, Tensor label) {
        // TODO Auto-generated method stub
        for (int i = 0; i < losses.length; i++) {
            this.loss[i] = losses[i].loss(outputs[i], label);
        }
        return this.loss;
    }

    @Override
    public Tensor lossDiff(Tensor output, Tensor label) {
        // TODO Auto-generated method stub
        /**
         * 清除梯度

         */
        this.clearGrad();
        return this.lossFunction.diff(output, label);
    }

    @Override
    public Tensor[] lossDiff(Tensor label) {
        // TODO Auto-generated method stub
        /**
         * 清除梯度

         */
        this.clearGrad();
        for (int i = 0; i < losses.length; i++) {
            this.lossDiff[i] = losses[i].diff(getOutputs()[i], label);
        }
        return this.lossDiff;
    }

    @Override
    public NetworkType getNetworkType() {
        // TODO Auto-generated method stub
        return NetworkType.YOLO;
    }

    @Override
    public Tensor[] predicts(Tensor input) {
        // TODO Auto-generated method stub
        /**
         * 设置输入数据

         */
        this.setInputData(input);
        /**
         * forward

         */
        for (int i = 0; i < layerCount; i++) {
            Layer layer = layerList.get(i);
            layer.forward();
        }
        return getOutputs();
    }

    public int getClass_num() {
        return class_num;
    }

    public void setClass_num(int class_num) {
        this.class_num = class_num;
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
        return this.lossFunction.diff(output, label, diff);
    }

    @Override
    public void putParamters() {
        // TODO Auto-generated method stub
    }

    @Override
    public void putParamterGrads() {
        // TODO Auto-generated method stub
    }
    //	public void clearGrad() {
    //		// TODO Auto-generated method stub
    //
    //		cudaStream_t stream = new cudaStream_t();
    //		JCuda.cudaStreamCreate(stream);
    //
    //		/**
    //		 * forward
    //		 */
    //		for(int i = 0;i<layerCount;i++) {
    //
    //			Layer layer = layerList.get(i);
    //
    //			if(layer.delta != null) {
    //				layer.delta.clearGPU(stream);
    //				if(layer.cache_delta != null) {
    //					layer.cache_delta.clearGPU(stream);
    //				}
    //			}
    //
    //		}
    //
    //		JCuda.cudaStreamSynchronize(stream);
    //		JCuda.cudaStreamDestroy(stream);
    //
    //	}
}

