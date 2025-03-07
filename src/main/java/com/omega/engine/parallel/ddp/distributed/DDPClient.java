package com.omega.engine.parallel.ddp.distributed;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.OutputStream;
import java.net.Socket;
import java.nio.ByteBuffer;
import java.util.Arrays;

public class DDPClient {
	
	private final String host;
	
	private final int masterPort;
	
	private final int deviceId;
	
	private Socket client;
	
	private InputStream in;
	
	private OutputStream out;
	
	private ModelRunner modelRunner;
	
	public DDPClient(String host,int masterPort, int deviceId) {
		this.host = host;
		this.masterPort = masterPort;
		this.deviceId = deviceId;
	}
	
	public void start() {
		try {
			client = new Socket(host, masterPort);
			System.out.println("network instance distributor contect on masterPort: " + masterPort);
			in = client.getInputStream();
			out = client.getOutputStream();
			/**
			 * 同步准备状态
			 */
			ready();
			System.out.println("ready finish.");
			while (true) {
            	byte[] bs = new byte[4];
            	in.read(bs, 0, 4);
            	ByteBuffer buffer = ByteBuffer.wrap(bs);
				int len = buffer.getInt();
				byte[] bytes = new byte[len];
				in.read(bytes, 0, len);
				System.out.println("client reiver["+bytes.length+"]:"+Arrays.toString(bytes));
				ByteArrayInputStream byteArrayInputStream = new ByteArrayInputStream(bytes);
				ObjectInputStream objectInputStream = new ObjectInputStream(byteArrayInputStream);
				Command command = (Command) objectInputStream.readObject();
				ClientCommandController.exec(this, command);
			}

		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
			System.out.println("close1");
		}
	}
	
    public boolean send(Command command) {
    	try {
    		ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();
    		ObjectOutputStream objectOutputStream = new ObjectOutputStream(byteArrayOutputStream);
    		ByteBuffer buffer = ByteBuffer.allocate(4);
    		objectOutputStream.writeObject(command);
    		byte[] bytes = byteArrayOutputStream.toByteArray();
    		System.out.println("client send["+bytes.length+"]:"+Arrays.toString(bytes));
    		buffer.putInt(bytes.length);
    		out.write(buffer.array());
    		out.write(bytes);
    		out.flush();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return true;
    }
	
	public void ready() {
		try {
			System.out.println("client_ready");
			Command command = new Command("client_ready", deviceId);
			send(command);
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
	}
	
	public static void main(String[] args) {
		String host = args[0];
		int masterPort = Integer.parseInt(args[1]);
		int deviceId = Integer.parseInt(args[2]);
		System.out.println("init host:"+host+":"+masterPort + "["+deviceId+"]");
		DDPClient client = new DDPClient(host, masterPort, deviceId);
		client.start();
	}

	public ModelRunner getModelRunner() {
		return modelRunner;
	}

	public void setModelRunner(ModelRunner modelRunner) {
		this.modelRunner = modelRunner;
	}

	public int getDeviceId() {
		return deviceId;
	}
	
}
