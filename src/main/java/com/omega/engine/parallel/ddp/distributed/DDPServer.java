package com.omega.engine.parallel.ddp.distributed;

import java.net.ServerSocket;
import java.net.Socket;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import com.omega.common.utils.JsonUtils;
import com.omega.engine.gpu.CUDAModules;
import com.omega.engine.parallel.ddp.DDP;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaMemcpyKind;

public class DDPServer {
	
	private final int port;
	
	private ServerSocket serverSocket;
	
	private List<ClientThread> clients = new ArrayList<ClientThread>();

	private DDP ddp;
	
	private int readyProcess = 0;
	
	public DDPServer(int port,DDP ddp) {
		this.port = port;
		this.setDdp(ddp);
	}
	
	public void startServer() {
		try {
			serverSocket = new ServerSocket(port);
			System.out.println("network instance distributor started on port: " + port);
			while (true) {
				Socket clientSocket = serverSocket.accept();
				clients.add(new ClientThread(this, clientSocket));
			}
			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
	}
	
	public synchronized boolean checkReadyProcess() {
		readyProcess+=1;
		if(readyProcess>=getDdp().getDevices()) {
			return true;
		}
		return false;
	}
	
	public void broadcastInitModel(Map<String,Object> data) {
		for(int i = 0;i<clients.size();i++) {
			System.out.println(i);
			ClientThread client = clients.get(i);
			data.put("id", i+"");
			data.put("deviceId", i+"");
			Command command = new Command("initModel", data);
			client.send(command);
		}
	}
	
	public void broadcast(Command command) {
		for(ClientThread client:clients) {
			client.send(command);
		}
	}
	
	public void showGPUPonterData(String hexString) {
		CUDAModules.getContext(0);
		float[] data = new float[16];
		long nativePointerValue = Long.parseLong(hexString, 16);
		SerializablePointer p = new SerializablePointer(nativePointerValue);
		System.out.println(p);
		JCuda.cudaMemcpy(Pointer.to(data), p, data.length * (long)Sizeof.FLOAT, cudaMemcpyKind.cudaMemcpyDeviceToHost);
		JCuda.cudaDeviceSynchronize();
		System.out.println(JsonUtils.toJson(data));
	}
	
	public DDP getDdp() {
		return ddp;
	}

	public void setDdp(DDP ddp) {
		this.ddp = ddp;
	}
	
}
