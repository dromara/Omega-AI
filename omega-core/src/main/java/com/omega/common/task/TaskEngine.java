package com.omega.common.task;

import java.util.Vector;
import java.util.concurrent.*;

/**
 * TaskEngine
 *
 * @author Administrator
 */
public class TaskEngine {
    private static TaskEngine instance;
    private int threadNum = 4;
    private ThreadPoolExecutor threadPool;

    public TaskEngine(int threadNum) {
        if (threadNum < 1) {
            this.threadNum = 4;
            threadPool = new ThreadPoolExecutor(4, 4, 1000, TimeUnit.SECONDS, new LinkedBlockingDeque<Runnable>());
        } else {
            this.threadNum = threadNum;
            threadPool = new ThreadPoolExecutor(threadNum, threadNum * 2, 1000, TimeUnit.SECONDS, new LinkedBlockingDeque<Runnable>());
        }
    }

    public static TaskEngine getInstance(int num) {
        synchronized (TaskEngine.class) {
            if (instance == null) {
                instance = new TaskEngine(num);
            } else if (num != instance.threadNum) {
                //线程数不相等则重新创建
                instance = new TaskEngine(num);
            }
        }
        return instance;
    }

    public void dispatchTask(Vector<Task<Object>> workers) {
        //接收多线程响应结果
        Vector<Future<Object>> results = new Vector<Future<Object>>();
        for (Task<Object> c : workers) {
            Future<Object> f = threadPool.submit(c);
            results.add(f);
        }
        for (int i = 0; i < results.size(); i++) {
            try {
                results.get(i).get();
            } catch (InterruptedException e) {
                // TODO Auto-generated catch block
                e.printStackTrace();
            } catch (ExecutionException e) {
                // TODO Auto-generated catch block
                e.printStackTrace();
            }
        }
    }
}

