package com.omega.engine.gpu;

public class JNITest {
	
	public native int max(int a,int b);
	
	public native float[] im2col(float[] x,float[] y,int n,int c,int h,int w,int kh,int kw,int stride);
	
}
