package com.omega.engine.controller;

import java.io.IOException;
import java.util.concurrent.ConcurrentHashMap;
import javax.websocket.OnClose;
import javax.websocket.OnError;
import javax.websocket.OnMessage;
import javax.websocket.OnOpen;
import javax.websocket.Session;
import javax.websocket.server.PathParam;
import javax.websocket.server.ServerEndpoint;
import org.springframework.stereotype.Component;

@Component
@ServerEndpoint("/api/pushMessage/{sid}")
public class WebSocketServer {
	
	 /**concurrent包的线程安全Set，用来存放每个客户端对应的WebSocket对象。*/
    private static ConcurrentHashMap<String,WebSocketServer> webSocketMap = new ConcurrentHashMap<String,WebSocketServer>();
    
    /**与某个客户端的连接会话，需要通过它来给客户端发送数据*/
    private Session session;
    
    private String sid;
    
    
    /**
     * 连接建立成
     * 功调用的方法
     */
    @OnOpen
    public void onOpen(Session session,@PathParam("sid") String sid) {
//    	System.out.println("in===>");
        this.session = session;
        this.sid = sid;
        if(webSocketMap.containsKey(sid)){
            webSocketMap.remove(sid);
            //加入set中
            webSocketMap.put(sid,this);
        }else{
            //加入set中
            webSocketMap.put(sid,this);

        }
        sendMessage("连接成功");
    }

	
    /**
     * 连接关闭
     * 调用的方法
     */
    @OnClose
    public void onClose() {
        if(webSocketMap.containsKey(sid)){
            webSocketMap.remove(sid);
        }
    }
    
    /**
     * 收到客户端消
     * 息后调用的方法
     * @param message
     * 客户端发送过来的消息
     **/
    @OnMessage
    public void onMessage(String message, Session session) {
        //可以群发消息
        //消息保存到数据库、redis
        if(message != null && !message.equals("")){
            try {
                
            }catch (Exception e){
                e.printStackTrace();
            }
        }
    }


    /**
     * @param session
     * @param error
     */
    @OnError
    public void onError(Session session, Throwable error) {
        error.printStackTrace();
    }

    
    /**
     * 实现服务
     * 器主动推送
     */
    public void sendMessage(String message) {
        try {
            this.session.getBasicRemote().sendText(message);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
    
    public static void push(String sid,String msg) {
    	
    	try {
//    		System.out.println("in===>");
    		if(sid != null && webSocketMap.containsKey(sid)){
                webSocketMap.get(sid).sendMessage(msg);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    	
    }
    
}
