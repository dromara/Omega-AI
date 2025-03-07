package com.omega.engine.nn.layer.gpu;

import com.omega.common.data.Tensor;
import com.omega.engine.gpu.BaseKernel;
import com.omega.engine.gpu.CUDAManager;
import com.omega.engine.nn.network.RunModel;

public abstract class RNNBaseKernel extends BaseKernel{
	
	public RNNBaseKernel(CUDAManager cudaManager) {
		super(cudaManager);
		// TODO Auto-generated constructor stub
	}

	public int seqLength = 1;
	
	public abstract void init(int number,int time);
	
	public abstract long weightSize();
	
	public abstract void forward(RunModel RUN_MODEL,Tensor input, Tensor hx, Tensor cx, Tensor weight, Tensor output, Tensor hy, Tensor cy);
	
	public abstract void dw(Tensor delta, Tensor output, Tensor input, Tensor hx, Tensor dw);
	
	public abstract void dx(Tensor delta,Tensor dhy,Tensor dcy, Tensor output, Tensor hx, Tensor cx, Tensor weight, Tensor diff, Tensor dhx,Tensor dcx);
	
	public abstract void initWeights(Tensor w);
	
}
