package com.omega.engine.parallel.ddp.distributed;

import jcuda.Pointer;

import java.io.Serializable;

public class SerializablePointer extends Pointer implements Serializable {
    /**
     *
     */
    private static final long serialVersionUID = -7911915261316902988L;

    public SerializablePointer() {
    }

    public SerializablePointer(long nativePointerValue) {
        super(nativePointerValue);
    }

    public static void main(String[] args) {
        String hex = "b12e00000";
        long longValue = Long.parseLong(hex, 16);
        System.out.println(longValue);
    }

    public String getHexString() {
        return Long.toHexString(getNativePointer());
    }
}

