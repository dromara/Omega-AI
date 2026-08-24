package com.omega.engine.nn.layer.gpu;

import com.omega.engine.tensor.Tensor;

import jcuda.Pointer;

/**
 * Thin JNI wrapper around NVIDIA cuDNN Frontend SDPA.
 *
 * The Java tensors remain FP32. The demo native library converts Q/K/V to
 * BF16 at the SDPA boundary and converts the result and gradients back to
 * FP32. No local FlashAttention implementation is used.
 */
public final class CudnnFlashAttentionKernel implements AutoCloseable {

    private static final String LIBRARY_PROPERTY = "omega.cudnn.sdpa.library";
    
    private static final String libPath = "/omega/native/lib/linux-x86_64/cu11-cudnn8/libomega_cudnn_sdpa.so";

    private static final boolean LOADED;
    private static final Throwable LOAD_ERROR;

    static {
        boolean loaded = false;
        Throwable error = null;
        try {
//            String libraryPath = System.getProperty(LIBRARY_PROPERTY);
//            if (libraryPath == null || libraryPath.trim().isEmpty()) {
//                System.loadLibrary("omega_cudnn_sdpa");
//            } else {
//                System.load(libraryPath);
//            }
            System.load(libPath);
            loaded = true;
        } catch (Throwable e) {
            error = e;
        }
        LOADED = loaded;
        LOAD_ERROR = error;
    }

    private final int batchSize;
    private final int headNum;
    private final int time;
    private final int headDim;

    private long plan;

    public CudnnFlashAttentionKernel(int batchSize, int headNum, int time, int headDim) {
        ensureLoaded();
        if (batchSize <= 0 || headNum <= 0 || time <= 0 || headDim <= 0) {
            throw new IllegalArgumentException("SDPA dimensions must all be positive.");
        }
        this.batchSize = batchSize;
        this.headNum = headNum;
        this.time = time;
        this.headDim = headDim;
        this.plan = nativeCreate(batchSize, headNum, time, headDim, false);
        if (this.plan == 0L) {
            throw new IllegalStateException("cuDNN SDPA returned a null plan.");
        }
    }

    public static boolean isLoaded() {
        return LOADED;
    }

    public static String getLoadError() {
        return LOAD_ERROR == null ? null : LOAD_ERROR.toString();
    }

    public static long getCudnnVersion() {
        ensureLoaded();
        return nativeGetCudnnVersion();
    }

    public long getAllocatedBytes() {
        ensureOpen();
        return nativeGetAllocatedBytes(plan);
    }

    public void forward(Tensor q, Tensor k, Tensor v, Tensor output) {
        ensureOpen();
        checkBHSD("q", q);
        checkBHSD("k", k);
        checkBHSD("v", v);
        checkBHSD("output", output);
        nativeForward(plan, q.getGpuData(), k.getGpuData(), v.getGpuData(), output.getGpuData());
    }

    public void backward(Tensor dOutput, Tensor dQ, Tensor dK, Tensor dV) {
        ensureOpen();
        checkBHSD("dOutput", dOutput);
        checkBHSD("dQ", dQ);
        checkBHSD("dK", dK);
        checkBHSD("dV", dV);
        nativeBackward(plan, dOutput.getGpuData(), dQ.getGpuData(), dK.getGpuData(), dV.getGpuData());
    }

    private void checkBHSD(String name, Tensor tensor) {
        if (tensor == null || tensor.number != batchSize || tensor.channel != headNum
                || tensor.height != time || tensor.width != headDim) {
            String actual = tensor == null ? "null" : tensor.shape().toString();
            throw new IllegalArgumentException(name + " must be [" + batchSize + ", " + headNum
                    + ", " + time + ", " + headDim + "], actual=" + actual);
        }
    }

    private void ensureOpen() {
        if (plan == 0L) {
            throw new IllegalStateException("cuDNN SDPA plan is closed.");
        }
    }

    private static void ensureLoaded() {
        if (!LOADED) {
            throw new IllegalStateException(
                    "Unable to load cuDNN SDPA JNI library. Set -D" + LIBRARY_PROPERTY
                            + "=/absolute/path/libomega_cudnn_sdpa.so",
                    LOAD_ERROR);
        }
    }

    @Override
    public void close() {
        if (plan != 0L) {
            nativeDestroy(plan);
            plan = 0L;
        }
    }

    private static native long nativeGetCudnnVersion();

    private static native long nativeGetAllocatedBytes(long plan);

    private static native long nativeCreate(int batchSize, int headNum, int time, int headDim,
            boolean deterministic);

    private static native void nativeForward(long plan, Pointer q, Pointer k, Pointer v, Pointer output);

    private static native void nativeBackward(long plan, Pointer dOutput, Pointer dQ, Pointer dK, Pointer dV);

    private static native void nativeDestroy(long plan);
}
