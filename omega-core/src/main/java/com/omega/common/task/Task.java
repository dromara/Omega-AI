package com.omega.common.task;

import java.util.concurrent.Callable;

/**
 * Task
 *
 * @param <V>
 * @author Administrator
 */
public class Task<V> implements Callable<V> {
    protected int n = 0;
    protected Object o = 0;

    public Task(int n) {
        this.n = n;
    }

    public Task(int n, Object o) {
        this.n = n;
        this.o = o;
    }

    @Override
    public V call() throws Exception {
        return null;
    }
}

