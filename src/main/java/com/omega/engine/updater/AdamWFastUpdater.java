package com.omega.engine.updater;

import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.List;

import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.normalization.NormalizationLayer;
import com.omega.engine.nn.network.Network;
import com.omega.engine.tensor.Tensor;
import com.omega.engine.updater.gpu.AdamWFastKernel;

public class AdamWFastUpdater {

    private final Network net;
    private final AdamWFastKernel kernel;
    private final IdentityHashMap<Tensor, Tensor> mStates = new IdentityHashMap<Tensor, Tensor>();
    private final IdentityHashMap<Tensor, Tensor> vStates = new IdentityHashMap<Tensor, Tensor>();

    private float beta1 = 0.9f;
    private float beta2 = 0.95f;
    private float weightDecay;

    public AdamWFastUpdater(Network net) {
        this.net = net;
        this.weightDecay = net.weight_decay;
        if (net.updaterParams != null) {
            if (net.updaterParams.get("beta1") != null) {
                beta1 = net.updaterParams.get("beta1");
            }
            if (net.updaterParams.get("beta2") != null) {
                beta2 = net.updaterParams.get("beta2");
            }
            if (net.updaterParams.get("weight_decay") != null) {
                weightDecay = net.updaterParams.get("weight_decay");
            }
        }
        this.kernel = new AdamWFastKernel(net.cudaManager);
    }

    public void update() {
        List<Tensor> weights = new ArrayList<Tensor>();
        List<Tensor> grads = new ArrayList<Tensor>();
        List<Integer> lengths = new ArrayList<Integer>();
        List<NormalizationLayer> normLayers = new ArrayList<NormalizationLayer>();
        IdentityHashMap<Tensor, Boolean> visited = new IdentityHashMap<Tensor, Boolean>();

        for (Layer layer : net.paramLayers) {
            if (layer == null || layer.freeze) {
                continue;
            }
            layer.learnRate = net.learnRate;
            if (layer instanceof NormalizationLayer) {
                collectNorm((NormalizationLayer) layer, weights, grads, lengths, normLayers, visited);
            } else {
                collectLayer(layer, weights, grads, lengths, visited);
            }
        }

        int tensorCount = weights.size();
        if (tensorCount > 0) {
            Tensor[] weightArray = new Tensor[tensorCount];
            Tensor[] gradArray = new Tensor[tensorCount];
            Tensor[] mArray = new Tensor[tensorCount];
            Tensor[] vArray = new Tensor[tensorCount];
            int[] lengthArray = new int[tensorCount];
            int totalLength = 0;
            for (int i = 0; i < tensorCount; i++) {
                Tensor weight = weights.get(i);
                weightArray[i] = weight;
                gradArray[i] = grads.get(i);
                mArray[i] = stateFor(mStates, weight);
                vArray[i] = stateFor(vStates, weight);
                lengthArray[i] = lengths.get(i);
                totalLength += lengthArray[i];
            }
            kernel.update(weightArray, gradArray, mArray, vArray, lengthArray, totalLength, tensorCount, net.learnRate,
                    beta1, beta2,
                    (float) (1.0f - Math.pow(beta1, net.train_time)),
                    (float) (1.0f - Math.pow(beta2, net.train_time)),
                    1e-8f, weightDecay);
        }

        for (Layer layer : net.paramLayers) {
            if (layer != null && !layer.freeze) {
                layer.clearAccGrad();
            }
        }
        for (NormalizationLayer layer : normLayers) {
            if (layer.diffGamma != null) {
                layer.diffGamma.clearGPU();
            }
            if (layer.diffBeta != null) {
                layer.diffBeta.clearGPU();
            }
        }
    }

    private void collectLayer(Layer layer, List<Tensor> weights, List<Tensor> grads, List<Integer> lengths,
            IdentityHashMap<Tensor, Boolean> visited) {
        if (layer.diffW == null || layer.weight == null) {
            return;
        }
        if (layer.accDW != null) {
            layer.accDW.copy(layer.diffW);
            if (layer.hasBias && layer.accDB != null && layer.diffB != null) {
                layer.accDB.copy(layer.diffB);
            }
        }
        addTensor(layer.weight, layer.diffW, weights, grads, lengths, visited);
        if (layer.hasBias && layer.bias != null && layer.diffB != null) {
            addTensor(layer.bias, layer.diffB, weights, grads, lengths, visited);
        }
    }

    private void collectNorm(NormalizationLayer layer, List<Tensor> weights, List<Tensor> grads, List<Integer> lengths,
            List<NormalizationLayer> normLayers, IdentityHashMap<Tensor, Boolean> visited) {
        if (layer.diffGamma == null || layer.gamma == null) {
            return;
        }
        if (layer.accDW != null) {
            layer.accDW.copy(layer.diffGamma);
            if (layer.hasBias && layer.accDB != null && layer.diffBeta != null) {
                layer.accDB.copy(layer.diffBeta);
            }
        }
        addTensor(layer.gamma, layer.diffGamma, weights, grads, lengths, visited);
        if (layer.beta != null && layer.diffBeta != null) {
            addTensor(layer.beta, layer.diffBeta, weights, grads, lengths, visited);
        }
        normLayers.add(layer);
    }

    private void addTensor(Tensor weight, Tensor grad, List<Tensor> weights, List<Tensor> grads, List<Integer> lengths,
            IdentityHashMap<Tensor, Boolean> visited) {
        if (weight == null || grad == null || weight.dataLength <= 0 || grad.dataLength != weight.dataLength || visited.containsKey(weight)) {
            return;
        }
        visited.put(weight, Boolean.TRUE);
        weights.add(weight);
        grads.add(grad);
        lengths.add(weight.dataLength);
    }

    private Tensor stateFor(IdentityHashMap<Tensor, Tensor> states, Tensor weight) {
        Tensor state = states.get(weight);
        if (state == null || state.dataLength != weight.dataLength) {
            state = new Tensor(1, 1, 1, weight.dataLength, true, true);
            states.put(weight, state);
        }
        return state;
    }
}
