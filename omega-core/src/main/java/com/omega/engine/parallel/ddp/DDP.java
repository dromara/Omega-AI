package com.omega.engine.parallel.ddp;

import com.omega.engine.loss.LossType;
import com.omega.engine.nn.network.NetworkType;
import com.omega.engine.parallel.ddp.distributed.DDPServer;
import com.omega.engine.parallel.ddp.distributed.ProcessManager;
import com.omega.engine.parallel.params.Llama3Parameters;
import com.omega.engine.parallel.params.Parameters;
import com.omega.engine.updater.UpdaterType;

public class DDP {
    private int port;
    private int devices = 1;
    private int[] deviceIds;
    private NetworkType networkType;
    private Parameters parameters;
    private DDPServer masterServer;
    private Thread masterThread;
    private Thread[] subThreads;

    public DDP(int devices, int port, NetworkType networkType, Parameters parameters) {
        this.port = port;
        this.setDevices(devices);
        this.deviceIds = new int[devices];
        this.setNetworkType(networkType);
        this.setParameters(parameters);
        createMasterServer();
        createDevicesProcess();
    }

    public static void main(String[] args) {
        int devices = 1;
        int port = 7800;
        NetworkType networkType = NetworkType.LLAMA3;
        int max_len = 512;
        int embedDim = 512;
        int headNum = 16;
        int nKVHeadNum = 8;
        int decoderNum = 8;
        int vocabSize = 6400;
        Llama3Parameters parameters = new Llama3Parameters(LossType.softmax_with_cross_entropy_idx, UpdaterType.adamw, headNum, nKVHeadNum, decoderNum, vocabSize, max_len, embedDim, false, false, false, 0.0001f);
        DDP ddp = new DDP(devices, port, networkType, parameters);
        //		ProcessManager.createProcess("127.0.0.1", 7800, 0);
    }

    public void createMasterServer() {
        final int p = port;
        masterThread = new Thread(() -> {
            masterServer = new DDPServer(p, this);
            masterServer.startServer();
        });
        masterThread.start();
    }

    public void createDevicesProcess() {
        subThreads = new Thread[getDevices()];
        for (int i = 0; i < getDevices(); i++) {
            final int id = i;
            subThreads[i] = new Thread(() -> {
                ProcessManager.createProcess("127.0.0.1", 7800, id);
            });
            subThreads[i].start();
        }
    }

    public NetworkType getNetworkType() {
        return networkType;
    }

    public void setNetworkType(NetworkType networkType) {
        this.networkType = networkType;
    }

    public Parameters getParameters() {
        return parameters;
    }

    public void setParameters(Parameters parameters) {
        this.parameters = parameters;
    }

    public int getDevices() {
        return devices;
    }

    public void setDevices(int devices) {
        this.devices = devices;
    }
}

