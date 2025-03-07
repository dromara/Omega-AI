package com.omega.engine.parallel.cuda;

import com.omega.engine.ad.op.TensorOP;
import com.omega.engine.gpu.BaseKernel;
import com.omega.engine.gpu.CUDAManager;

public class CUDACommonManager {
	
	private CUDAManager cudaManager;
	
	private TensorOP tensorOP;
	
	private BaseKernel baseKernel;
	
	public CUDACommonManager(CUDAManager cudaManager,TensorOP tensorOP,BaseKernel baseKernel) {
		this.cudaManager = cudaManager;
		this.tensorOP = tensorOP;
		this.baseKernel = baseKernel;
	}
	
	public CUDAManager getCudaManager() {
		return cudaManager;
	}

	public void setCudaManager(CUDAManager cudaManager) {
		this.cudaManager = cudaManager;
	}

	public TensorOP getTensorOP() {
		return tensorOP;
	}

	public void setTensorOP(TensorOP tensorOP) {
		this.tensorOP = tensorOP;
	}

	public BaseKernel getBaseKernel() {
		return baseKernel;
	}

	public void setBaseKernel(BaseKernel baseKernel) {
		this.baseKernel = baseKernel;
	}
	
	
	
}
