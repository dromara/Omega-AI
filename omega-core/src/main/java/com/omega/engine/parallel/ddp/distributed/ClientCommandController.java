package com.omega.engine.parallel.ddp.distributed;

import com.omega.common.utils.JsonUtils;
import com.omega.common.utils.MatrixUtils;
import com.omega.engine.nn.network.NetworkType;
import com.omega.engine.parallel.params.Llama3Parameters;
import com.omega.engine.tensor.Tensor;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaMemcpyKind;

import java.util.Map;

public class ClientCommandController {
    private static ClientCommandController controller;

    private static ClientCommandController getInstance() {
        if (controller == null) {
            controller = new ClientCommandController();
        }
        return controller;
    }

    public static void exec(DDPClient client, Command command) {
        ClientCommandController controller = getInstance();
        switch (command.getCode()) {
            case "initModel":
                controller.createModelRunner(client, command.getData());
                break;
            default:
                break;
        }
    }

    public void createModelRunner(DDPClient client, Object data) {
        //int id, int deviceId, NetworkType networkType, Parameters parameters
        Map<String, Object> dataMap = (Map<String, Object>) data;
        int id = Integer.parseInt(dataMap.get("id").toString());
        int deviceId = Integer.parseInt(dataMap.get("deviceId").toString());
        NetworkType networkType = Enum.valueOf(NetworkType.class, dataMap.get("networkType").toString());
        Llama3Parameters parameters = (Llama3Parameters) dataMap.get("parameters");
        ;
        ModelRunner modelRunner = new ModelRunner(id, deviceId, networkType, parameters);
        client.setModelRunner(modelRunner);
        Tensor tmp = new Tensor(1, 1, 4, 4, MatrixUtils.order(16, 0.1f, 0.1f), true, true);
        Command command = new Command("shareGPUPointer", tmp.getShareGPU().getHexString());
        System.out.println("org:" + tmp.getShareGPU());
        float[] data2 = new float[16];
        JCuda.cudaMemcpy(Pointer.to(data2), tmp.getShareGPU(), tmp.getDataLength() * (long) Sizeof.FLOAT, cudaMemcpyKind.cudaMemcpyDeviceToHost);
        JCuda.cudaDeviceSynchronize();
        System.out.println(JsonUtils.toJson(data2));
        client.send(command);
    }
}

