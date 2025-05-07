package com.omega.engine.gpu;

import jcuda.Pointer;

public class GPUWorkspace {
    private Pointer pointer = new Pointer();
    private long size;

    public Pointer getPointer() {
        return pointer;
    }

    public void setPointer(Pointer pointer) {
        this.pointer = pointer;
    }

    public long getSize() {
        return size;
    }

    public void setSize(long size) {
        this.size = size;
    }
}

