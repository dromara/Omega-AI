package com.omega.engine.gpu;

public class JNITest {
    private static JNITest instance;

    static {
        System.out.println("in");
        System.load("H:\\omega\\omega-ai\\omega-engine-v3\\jni\\test_cuda.dll");
    }

    public static JNITest getInstance() {
        if (JNITest.instance == null) {
            JNITest.instance = new JNITest();
        }
        return instance;
    }

    public native int max(int a, int b);

    public native void im2colV2(float[] x, float[] y, int n, int c, int h, int w, int kh, int kw, int stride);

    public native void conv(float[] im, float[] kernel, float[] out, int n, int c, int h, int w, int ko, int kh, int kw, int stride);
}

