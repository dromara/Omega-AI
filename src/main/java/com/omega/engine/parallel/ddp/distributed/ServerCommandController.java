package com.omega.engine.parallel.ddp.distributed;

import java.util.HashMap;
import java.util.Map;

import com.omega.common.data.Tensor;

import jcuda.Pointer;

public class ServerCommandController {
	
	private static ServerCommandController controller;
	
	private static ServerCommandController getInstance() {
		if(controller == null) {
			controller = new ServerCommandController();
		}
		return controller;
	}
	
	public static void exec(DDPServer server,Command command) {
		ServerCommandController controller = getInstance();
		switch (command.getCode()) {
		case "client_ready":
			System.out.println("----client_ready----->");
			controller.clientReady(server);
			break;
		case "shareGPUPointer":
			controller.shareGPUPointer(server, command.getData());
			break;
		default:
			break;
		}
	}
	
	public void clientReady(DDPServer server) {
		if(server.checkReadyProcess()){
			Map<String,Object> data = new HashMap<String, Object>();
			data.put("networkType", server.getDdp().getNetworkType().toString());
			data.put("parameters", server.getDdp().getParameters());
			server.broadcastInitModel(data);
		}
	}
	
	public void shareGPUPointer(DDPServer server,Object data) {
		String hexStr = (String) data;
		server.showGPUPonterData(hexStr);
	}
	
}
