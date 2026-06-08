package com.omega.engine.gpu;

import static jcuda.jcudnn.cudnnDataType.CUDNN_DATA_FLOAT;
import static jcuda.jcudnn.cudnnTensorFormat.CUDNN_TENSOR_NCHW;

import java.util.HashMap;
import java.util.Map;

import jcuda.jcudnn.JCudnn;
import jcuda.jcudnn.cudnnTensorDescriptor;

public class CUDNNDescriptorManager {
	
	public static Map<String, cudnnTensorDescriptor> descriptor4dMap = new HashMap<String, cudnnTensorDescriptor>();
	
	public static cudnnTensorDescriptor cache_4D(String key, int N, int C, int H, int W) {
		String dimKey = "-" + N + ":" + C + ":" + H + ":" + W;
		String insKey = key + dimKey;
		if (descriptor4dMap.containsKey(insKey)) {
	        return descriptor4dMap.get(insKey);
		}else {
			cudnnTensorDescriptor xDesc = new cudnnTensorDescriptor();
			handle(JCudnn.cudnnCreateTensorDescriptor(xDesc));
			handle(JCudnn.cudnnSetTensor4dDescriptor(xDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C, H, W));
			descriptor4dMap.put(insKey, xDesc);
			return xDesc;
		}
	}
	
    /**
     * Handle.
     *
     * @param returnCode the return run
     */
    public static void handle(final int returnCode) {
        if (returnCode != jcuda.jcudnn.cudnnStatus.CUDNN_STATUS_SUCCESS) {
            System.err.println(jcuda.jcudnn.cudnnStatus.stringFor(returnCode));
            throw new RuntimeException(jcuda.jcudnn.cudnnStatus.stringFor(returnCode));
        }
    }

    public static String checkError(final int returnCode) {
        if (returnCode != jcuda.jcudnn.cudnnStatus.CUDNN_STATUS_SUCCESS) {
            return jcuda.jcudnn.cudnnStatus.stringFor(returnCode);
        } else {
            return "success";
        }
    }
	
}
