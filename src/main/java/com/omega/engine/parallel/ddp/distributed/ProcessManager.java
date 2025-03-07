package com.omega.engine.parallel.ddp.distributed;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;

public class ProcessManager {
	
	private static List<Process> processGroup = new ArrayList<Process>();
	
	public static Process createProcess(String host,int masterPort,int deviceId) {
		Process process = null;
		try {

			ProcessBuilder builder = new ProcessBuilder().command("java", "-Xms1480m", "-Xmx1480m", "-cp" ,"H:\\omega\\20240716\\omega-ai\\target\\omega-engine-v4-gpu-win-cu11.7-v1.0-beta.jar", "com.omega.engine.ddp.distributed.DDPClient", host, masterPort+"", deviceId+"", "&");

			process = builder.start();
			
			// 使用 BufferedReader 来读取进程的标准输出
			final Process p = process;
			new Thread(()->{
                try {
                    BufferedReader bufferedReader = new BufferedReader(new InputStreamReader(p.getErrorStream()));
                    String line;
                    while (!"-1".equals(line=bufferedReader.readLine())) {
                        System.out.println(line);
                    }
                } catch (Exception e) {
                    e.printStackTrace();
                }
                System.out.println("结束接收对话");
            }).start();
			
			// 使用 BufferedReader 来读取进程的标准输出
			new Thread(()->{
                try {
                    BufferedReader bufferedReader = new BufferedReader(new InputStreamReader(p.getInputStream()));
                    String line;
                    while (!"-1".equals(line=bufferedReader.readLine())) {
                        System.out.println(line);
                    }
                } catch (Exception e) {
                    e.printStackTrace();
                }
                System.out.println("结束接收对话");
            }).start();
			
			processGroup.add(process);
			
			process.waitFor();
			
			System.err.println("in4");
			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
		return process;
	}
	
	public static void main(String[] args) {
		
		ProcessManager.createProcess("127.0.0.1", 7800, 0);
		
	}
	
}
