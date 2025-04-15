package com.omega.data.yolo;

public class MemeryBlock {
    private int size = 0;
    private boolean status = false;
    private float[] data;

    public MemeryBlock(int size) {
        this.size = size;
        this.data = new float[size];
    }

    public int getSize() {
        return size;
    }

    public void setSize(int size) {
        this.size = size;
    }

    public boolean isStatus() {
        return status;
    }

    public void setStatus(boolean status) {
        this.status = status;
    }

    public float[] getData() {
        return data;
    }

    public void setData(float[] data) {
        this.data = data;
    }

    public void free() {
        this.status = false;
    }
}

