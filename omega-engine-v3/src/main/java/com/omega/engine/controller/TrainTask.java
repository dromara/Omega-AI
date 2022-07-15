package com.omega.engine.controller;

import java.util.HashMap;
import java.util.Map;

import org.springframework.scheduling.annotation.Async;

import com.omega.engine.optimizer.Optimizer;

public class TrainTask {
	
	private static Map<String,Optimizer> ops = new HashMap<String,Optimizer>();
	
	public static void addTask(String sid,Optimizer op) {
		ops.put(sid, op);
	}
	
	public static void remove(String sid) {
		ops.remove(sid);
	}
	
	public static void updateLR(String sid,float lr) {
		if(ops.get(sid) != null) {
			ops.get(sid).network.learnRate = lr;
		}
	}
	
	@Async
	public static void sendMsg(String sid,String msg) {
//		System.out.println("in===>");
		try {

			WebSocketServer.push(sid, msg);
			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
}
