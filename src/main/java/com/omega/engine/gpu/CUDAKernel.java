package com.omega.engine.gpu;

public class CUDAKernel {
	
	private CUDAManager cudaManager;
	
	public CUDAKernel(CUDAManager cudaManager) {
		this.cudaManager = cudaManager;
	}
	
	public CUDAManager getCudaManager() {
		return cudaManager;
	}
	
}
