package com.omega.engine.updater;

import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.normalization.NormalizationLayer;
import com.omega.engine.nn.network.Network;
import com.omega.engine.updater.gpu.SGDKernel;

public class SGDM extends Updater {
    private SGDKernel kernel;

    public SGDM(Network network) {
        this.net = network;
    }

    @Override
    public void update(Layer layer) {
        // TODO Auto-generated method stub
        /**
         * init

         */
        if (kernel == null) {
            if (layer.hasBias) {
                kernel = new SGDKernel(layer.weight.dataLength, layer.bias.dataLength, net.cudaManager);
            } else {
                kernel = new SGDKernel(layer.weight.dataLength, net.cudaManager);
            }
        }
        kernel.updateW(layer.diffW, layer.weight, layer.network, layer.learnRate);
        if (layer.hasBias) {
            kernel.updateB(layer.diffB, layer.bias, layer.network, layer.learnRate);
        }
    }

    @Override
    public void updateForMatrix(Layer layer) {
        // TODO Auto-generated method stub
    }

    @Override
    public void updateForBN(NormalizationLayer layer) {
        // TODO Auto-generated method stub
        /**
         * init

         */
        if (kernel == null) {
            kernel = new SGDKernel(layer.gamma.dataLength, layer.beta.dataLength, net.cudaManager);
            //kernel.weight_decay = 0.0f;
        }
        kernel.updateW(layer.diffGamma, layer.gamma, layer.network, layer.learnRate);
        kernel.updateB(layer.diffBeta, layer.beta, layer.network, layer.learnRate);
    }

    @Override
    public UpdaterType getUpdaterType() {
        // TODO Auto-generated method stub
        return UpdaterType.sgd;
    }

    @Override
    public void update(Layer layer, int batchSize) {
        // TODO Auto-generated method stub
        /**
         * init

         */
        if (kernel == null) {
            if (layer.hasBias) {
                kernel = new SGDKernel(layer.weight.dataLength, layer.bias.dataLength, net.cudaManager);
            } else {
                kernel = new SGDKernel(layer.weight.dataLength, net.cudaManager);
            }
        }
        kernel.updateW(layer.diffW, layer.weight, layer.network, layer.learnRate, batchSize);
        if (layer.hasBias) {
            kernel.updateB(layer.diffB, layer.bias, layer.network, layer.learnRate, batchSize);
        }
    }
}

