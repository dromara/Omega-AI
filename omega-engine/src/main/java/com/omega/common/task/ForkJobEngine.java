package com.omega.common.task;

import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.RecursiveAction;

/**
 * 任务分拆执行引擎
 *
 * @author Administrator
 */
public class ForkJobEngine {
    private static ForkJoinPool forkJoinPool;

    private static ForkJoinPool getPool() {
        if (forkJoinPool == null) {
            forkJoinPool = new ForkJoinPool();
        }
        return forkJoinPool;
    }

    public static void run(RecursiveAction action) {
        getPool().submit(action).join();
    }
}

