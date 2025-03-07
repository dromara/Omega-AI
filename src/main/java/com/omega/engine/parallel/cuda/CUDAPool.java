package com.omega.engine.parallel.cuda;

import java.util.HashMap;
import java.util.Map;

import com.omega.engine.ad.op.TensorOP;
import com.omega.engine.gpu.BaseKernel;
import com.omega.engine.gpu.CUDAManager;

public class CUDAPool {
	
	public static Map<Integer,CUDACommonManager> GPU_POOL = new HashMap<Integer, CUDACommonManager>();
	
	public synchronized static CUDACommonManager cudaCommonManager(int rankId) {
		
		if(!GPU_POOL.containsKey(rankId)) {
			CUDAManager cudaManager = new CUDAManager(rankId);
			TensorOP tensorOP = new TensorOP(cudaManager);
			BaseKernel baseKernel = new BaseKernel(cudaManager);
			GPU_POOL.put(rankId, new CUDACommonManager(cudaManager, tensorOP, baseKernel));
		}
		
		return GPU_POOL.get(rankId);
	}
	
}
