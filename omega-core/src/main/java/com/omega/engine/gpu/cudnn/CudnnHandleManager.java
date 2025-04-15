package com.omega.engine.gpu.cudnn;

import jcuda.jcudnn.JCudnn;
import jcuda.jcudnn.cudnnHandle;
import jcuda.runtime.JCuda;

public class CudnnHandleManager {
    private static cudnnHandle cudnnHandle;

    public static cudnnHandle getHandle() {
        try {
            if (cudnnHandle == null) {
                GpuHandle(0);
                int version = (int) JCudnn.cudnnGetVersion();
                System.out.printf("cudnnGetVersion() : %d , " + "CUDNN_VERSION from cudnn.h : %d\n", version, JCudnn.CUDNN_VERSION);
            }
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
        return cudnnHandle;
    }

    /**
     * Instantiates a new Cudnn.
     *
     * @param deviceNumber the device number
     */
    public static void GpuHandle(int deviceNumber) {
        if (0 <= deviceNumber) {
            initThread();
            cudnnHandle = new cudnnHandle();
            JCudnn.cudnnCreate(cudnnHandle);
        } else {
            cudnnHandle = null;
        }
        //cudaSetDevice();
    }

    /**
     * Init thread.
     */
    public static void initThread() {
        setDevice(0);
    }

    public static void setDevice(final int cudaDeviceId) {
        if (cudaDeviceId < 0)
            throw new IllegalArgumentException("cudaDeviceId=" + cudaDeviceId);
        if (!isThreadDeviceId(cudaDeviceId)) {
            final int result = JCuda.cudaSetDevice(cudaDeviceId);
            System.out.println(result);
        }
    }

    public static boolean isThreadDeviceId(int deviceId) {
        Integer integer = getThreadDeviceId();
        return integer != null && (deviceId == integer);
    }

    public static Integer getThreadDeviceId() {
        return 0;
    }

    public static void handle(final int returnCode) {
        if (returnCode != jcuda.jcudnn.cudnnStatus.CUDNN_STATUS_SUCCESS) {
            CudnnError cudnnError = new CudnnError(jcuda.jcudnn.cudnnStatus.stringFor(returnCode));
            throw cudnnError;
        }
    }
}

