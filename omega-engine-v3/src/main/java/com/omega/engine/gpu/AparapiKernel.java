package com.omega.engine.gpu;

import com.aparapi.Kernel;
import com.aparapi.Range;
import com.omega.common.utils.MatrixUtils;
import com.omega.common.utils.PrintUtils;

public class AparapiKernel extends Kernel{
	
	public float[] input;
	
	public float[] k;
	
	public float[] output;
	
	public int[] dims = new int[] {0,0,0,0};
	
	public int[] kdims = new int[] {0,0,0,0};
	
	public int[] odims = new int[] {0,0,0,0};
	
	public int stride = 1;
	
	public AparapiKernel(int N,int C,int H,int W,int kN,int kH,int kW) {
		this.dims[0] = N;
		this.dims[1] = C;
		this.dims[2] = H;
		this.dims[3] = W;
		this.kdims[0] = kN;
		this.kdims[1] = C;
		this.kdims[2] =kH;
		this.kdims[3] = kW;
		this.odims[0] = N;
		this.odims[1] = kN;
		this.odims[2] = (H - kH) / this.stride + 1;
		this.odims[3] = (W - kW) / this.stride + 1;
		this.output = new float[N * kN * this.odims[2] * this.odims[3]];
	}
	
	@Override
	public void run() {
		// TODO Auto-generated method stub
		int w = getGlobalId(0);
		int h = getGlobalId(1);
		int oc = getGlobalId(2);
		int n = getPassId();
		int C = dims[1];
		int H = dims[2];
		int W = dims[3];
		int KC = kdims[0];
		int KH = kdims[2];
		int KW = kdims[3];
		
		for(int c = 0;c<KC;c++) {
			
			for(int kh = 0;kh<KH;kh++) {
				
				for(int kw = 0;n<KW;kw++) {
					
					int hi = h * stride + kh;
					
					int wi = w * stride + kw;
					
					output[n * C * H * W + oc * H * W + h * W + w] += input[n * KC * H * W + c * H * W + hi * W + wi] * k[oc * KC * KH * KW + c * KH * KW + kh * KW + kw];

				}
				
			}

		}
			
		
	}

	public static void main(String args[]) {
		
//		System.out.println(Range.MAX_GROUP_SIZE);
//		
//		System.out.println(Range.MAX_OPENCL_GROUP_SIZE);
//
//		int w = 3;
//		int h = 3;
//		int c = 3;
//		int n = 4;
//		int kn = 4;
//		int kh = 2;
//		int kw = 2;
//		int size = n * c * h * w;
//		int kSize = kn * c * kh * kw;
//		
//		AparapiKernel k = new AparapiKernel(n,c,h,w, kn, kh, kw);
//		
//		k.input = MatrixUtils.val(size, 1);
//		k.k = MatrixUtils.val(kSize, 2);
//
//		Range r = Range.create3D(k.odims[3], k.odims[2], kn);
//		
//		k.execute(r, n);
//		
//		PrintUtils.printImage(k.output);
		
		System.out.println(Math.pow(2, -3.0 / 2));
		
		System.out.println(Math.pow(2, -3.0));
		
	}
	
}
