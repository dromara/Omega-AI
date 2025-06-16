package com.omega.engine.updater;

import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.normalization.NormalizationLayer;
import com.omega.engine.nn.network.Network;
import com.omega.engine.updater.gpu.AdamWKernel;
import jcuda.driver.CUstream;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaStream_t;

/**
 * Adam Updater
 *
 * @author Administrator
 */
public class AdamW extends Updater {
    private AdamWKernel kernel;
    private CUstream stream;
    private cudaStream_t streamt;
    private float weight_decay = 1e-4f;

    public AdamW(Network network) {
        this.net = network;
        this.weight_decay = this.net.weight_decay;
        this.params = network.updaterParams;
    }

    @Override
    public void update(Layer layer) {
        // TODO Auto-generated method stub
        layer.learnRate = layer.network.learnRate;
        /**
         * init
         */
        if (kernel == null) {
            if (layer.hasBias) {
                kernel = new AdamWKernel(layer.weight.dataLength, layer.bias.dataLength, weight_decay, net.cudaManager);
            } else {
                kernel = new AdamWKernel(layer.weight.dataLength, weight_decay, net.cudaManager);
            }
            streamt = new cudaStream_t();
            JCuda.cudaStreamCreate(streamt);
            stream = new CUstream(streamt);
            kernel.setParams(params);
        }
        kernel.updateW(layer.diffW, layer.weight, layer.network, layer.learnRate, stream);
        if (layer.hasBias) {
            kernel.updateB(layer.diffB, layer.bias, layer.network, layer.learnRate, stream);
        }
    }

    @Override
    public void updateForMatrix(Layer layer) {
        // TODO Auto-generated method stub
    }

    @Override
    public void updateForBN(NormalizationLayer layer) {
        // TODO Auto-generated method stub
        layer.learnRate = layer.network.learnRate;
        //		System.out.println(layer.learnRate);
        /**
         * init

         */
        if (kernel == null) {
            if (layer.beta != null) {
                kernel = new AdamWKernel(layer.gamma.dataLength, layer.beta.dataLength, weight_decay, net.cudaManager);
            } else {
                kernel = new AdamWKernel(layer.gamma.dataLength, 0, weight_decay, net.cudaManager);
            }
            streamt = new cudaStream_t();
            JCuda.cudaStreamCreate(streamt);
            stream = new CUstream(streamt);
        }
        kernel.setParams(params);
        kernel.updateGamma(layer.diffGamma, layer.gamma, layer.network, layer.learnRate, stream);
        layer.diffGamma.clearGPU(streamt);
        if (layer.beta != null) {
            kernel.updateBeta(layer.diffBeta, layer.beta, layer.network, layer.learnRate, stream);
            layer.diffBeta.clearGPU(streamt);
        }
    }

    @Override
    public UpdaterType getUpdaterType() {
        // TODO Auto-generated method stub
        return UpdaterType.adamw;
    }

    @Override
    public void update(Layer layer, int batchSize) {
        // TODO Auto-generated method stub
        layer.learnRate = layer.network.learnRate;
        /**
         * init

         */
        if (kernel == null) {
            if (layer.hasBias) {
                kernel = new AdamWKernel(layer.weight.dataLength, layer.bias.dataLength, weight_decay, net.cudaManager);
            } else {
                kernel = new AdamWKernel(layer.weight.dataLength, weight_decay, net.cudaManager);
            }
            kernel.setParams(params);
        }
        kernel.updateW(layer.diffW, layer.weight, layer.network, layer.learnRate, batchSize);
        //		layer.diffW.clearGPU();
        //
        //		System.out.print(layer.getLayerType().toString()+layer.index+":");
        //		layer.weight.showDM();
        if (layer.hasBias) {
            kernel.updateB(layer.diffB, layer.bias, layer.network, layer.learnRate, batchSize);
            //			layer.diffB.clearGPU();
        }
    }
}

