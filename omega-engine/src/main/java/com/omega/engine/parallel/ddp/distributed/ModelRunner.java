package com.omega.engine.parallel.ddp.distributed;

import com.omega.engine.gpu.CUDAModules;
import com.omega.engine.nn.network.Llama3;
import com.omega.engine.nn.network.Network;
import com.omega.engine.nn.network.NetworkType;
import com.omega.engine.parallel.params.Llama3Parameters;
import com.omega.engine.parallel.params.Parameters;

public class ModelRunner {
    private int id;
    private int deviceId;
    private NetworkType networkType;
    private Parameters parameters;
    private Network instance;

    public ModelRunner(int id, int deviceId, NetworkType networkType, Parameters parameters) {
        this.setId(id);
        this.deviceId = deviceId;
        this.networkType = networkType;
        this.parameters = parameters;
        init();
    }

    public void init() {
        /**
         * init cuda environment

         */
        CUDAModules.getContext(deviceId);
        CUDAModules.initCUDAFunctions();
        /**
         * create network instance

         */
        this.setInstance(createNetwork());
    }

    public Network createNetwork() {
        Network network = null;
        switch (networkType) {
            case LLAMA3:
                Llama3Parameters params = (Llama3Parameters) parameters;
                network = new Llama3(params.lossType, params.updater, params.getHeadNum(), params.getnKVHeadNum(), params.getDecoderNum(), params.getVocabSize(), params.getTime(), params.getEmbedDim(), params.isBias(), params.isDropout(), params.isFlashAttention());
                break;
            default:
                break;
        }
        return network;
    }

    public Network getInstance() {
        return instance;
    }

    public void setInstance(Network instance) {
        this.instance = instance;
    }

    public int getId() {
        return id;
    }

    public void setId(int id) {
        this.id = id;
    }
}

