package com.omega.engine.updater;

import com.omega.common.utils.MatrixOperation;
import com.omega.common.utils.MatrixUtils;
import com.omega.engine.nn.layer.ConvolutionLayer;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.normalization.NormalizationLayer;
import com.omega.engine.nn.network.Network;

/**
 * Momentum
 * <p>
 * <p>
 * <p>
 * Vw = mu * Vdwi-1 - learning_rate * dwi
 * <p>
 * Vb = mu * Vdbi-1 - learning_rate * dbi
 * <p>
 * w = w + vw
 * <p>
 * b = b + vb
 *
 * @author Administrator
 */
public class Momentum extends Updater {
    public Momentum(Network network) {
        this.net = network;
    }

    @Override
    public void update(Layer layer) {
        // TODO Auto-generated method stub
        /**
         * init

         */
        if (this.vdw == null) {
            this.vdw = MatrixUtils.zero(layer.width * layer.oWidth);
            if (layer.hasBias) {
                this.vdb = MatrixUtils.zero(layer.oWidth);
            }
        }
        /**
         * Vdw = beta * Vdwi-1 - learning_rate * dwi

         */
        this.vdw = MomentumUtils.momentum(layer.diffW.getData(), this.vdw, layer.learnRate);
        if (layer.hasBias) {
            /**
             * Vdb = beta * Vdbi-1 - learning_rate * dbi

             */
            this.vdb = MomentumUtils.momentum(layer.diffB.getData(), this.vdb, layer.learnRate);
        }
        /**
         * W = W + vdw

         */
        layer.weight.setData(MatrixOperation.add(layer.weight.getData(), this.vdw));
        if (layer.hasBias) {
            /**
             * b = b + Vdb

             */
            layer.bias.setData(MatrixOperation.add(layer.bias.getData(), this.vdb));
        }
    }

    @Override
    public void updateForMatrix(Layer layer) {
        // TODO Auto-generated method stub
        if (!layer.getLayerType().equals(LayerType.conv)) {
            throw new RuntimeException("this function param must be conv layer.");
        }
        ConvolutionLayer conv = (ConvolutionLayer) layer;
        /**
         * init

         */
        if (this.vdmw == null) {
            this.vdmw = MatrixUtils.zero(conv.kernelNum * conv.channel * conv.kHeight * conv.kWidth);
            if (layer.hasBias) {
                this.vdmb = MatrixUtils.zero(conv.kernelNum);
            }
        }
        /**
         * Vdw = beta * Vdwi-1 - learning_rate * dwi

         */
        this.vdmw = MomentumUtils.momentum(conv.diffW.getData(), this.vdmw, conv.learnRate);
        if (layer.hasBias) {
            /**
             * Vdb = beta * Vdbi-1 - learning_rate * dbi

             */
            this.vdmb = MomentumUtils.momentum(conv.diffB.getData(), this.vdmb, conv.learnRate);
        }
        /**
         * W = W + vdw

         */
        conv.weight.setData(MatrixOperation.add(conv.weight.getData(), this.vdmw));
        if (layer.hasBias) {
            /**
             * b = b + Vdb

             */
            conv.bias.setData(MatrixOperation.add(conv.bias.getData(), this.vdmb));
        }
    }

    @Override
    public void updateForBN(NormalizationLayer layer) {
        //		// TODO Auto-generated method stub
        //		/**
        //		 * init
        //		 */
        //		if(this.vdgama == null || this.vdb == null) {
        //			this.vdgama = MatrixUtils.zero(layer.deltaGama.length);
        //			this.vdb = MatrixUtils.zero(layer.deltaBeta.length);
        //		}
        //
        //		/**
        //		 * Vdgama = beta * Vdgamai-1 - learning_rate * dgama
        //		 */
        //		this.vdgama = MomentumUtils.momentum(layer.deltaGama, this.vdgama, layer.learnRate);
        //
        //		/**
        //		 * Vdb = beta * Vdbi-1 - learning_rate * dbi
        //		 */
        //		this.vdb = MomentumUtils.momentum(layer.deltaBeta, this.vdb, layer.learnRate);
        //
        //		/**
        //		 * gama = gama + Vdgama
        //		 */
        //		layer.gama = MatrixOperation.add(layer.gama, this.vdgama);
        //
        //		/**
        //		 * b = b + Vdb
        //		 */
        //		layer.beta = MatrixOperation.add(layer.beta, this.vdb);
    }

    @Override
    public UpdaterType getUpdaterType() {
        // TODO Auto-generated method stub
        return UpdaterType.momentum;
    }

    @Override
    public void update(Layer layer, int batchSize) {
        // TODO Auto-generated method stub
    }
}

