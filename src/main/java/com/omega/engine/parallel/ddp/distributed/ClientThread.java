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

public class ClientThread extends Thread {
	
	private DDPServer server;
	
	private Socket client;
	
	private InputStream in;
	
    private OutputStream out;
    
    public ClientThread(DDPServer server,Socket client) {
    	this.server = server;
    	this.client = client;
    	init();
        start();
    }
    
    public void init() {
    	try {
        	in = client.getInputStream();
            out = client.getOutputStream();
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
    }
    
    public void run(){
        try{

            while (true){
            	byte[] bs = new byte[4];
            	in.read(bs, 0, 4);
            	ByteBuffer buffer = ByteBuffer.wrap(bs);
				int len = buffer.getInt();
				byte[] bytes = new byte[len];
				in.read(bytes, 0, len);
				System.out.println("server reiver["+bytes.length+"]:"+Arrays.toString(bytes));
				ByteArrayInputStream byteArrayInputStream = new ByteArrayInputStream(bytes);
				ObjectInputStream objectInputStream = new ObjectInputStream(byteArrayInputStream);
				Command command = (Command) objectInputStream.readObject();
				ServerCommandController.exec(server, command);
            }
//            out.println("--- See you, bye! ---");
//            client.close();
        }catch (Exception e){
            e.printStackTrace();
        }
    }
    
    public boolean send(Command command) {
    	try {
    		ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();
    		ObjectOutputStream objectOutputStream = new ObjectOutputStream(byteArrayOutputStream);
    		ByteBuffer buffer = ByteBuffer.allocate(4);
    		objectOutputStream.writeObject(command);
    		byte[] bs = byteArrayOutputStream.toByteArray();
    		System.out.println("server send["+bs.length+"]:"+Arrays.toString(bs));
    		buffer.putInt(bs.length);
    		out.write(buffer.array());
    		out.write(bs);
    		out.flush();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return true;
    }
    
}
