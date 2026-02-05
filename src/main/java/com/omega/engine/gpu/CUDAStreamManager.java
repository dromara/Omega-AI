package com.omega.engine.gpu;

import java.util.HashMap;
import java.util.Map;

import jcuda.runtime.JCuda;
import jcuda.runtime.cudaStream_t;

public class CUDAStreamManager {
	
    public static Map<String, cudaStream_t> streamMap = new HashMap<String, cudaStream_t>();
    
    public synchronized static cudaStream_t getStream(String key) {
        if (streamMap.containsKey(key)) {
            return streamMap.get(key);
        }
        cudaStream_t stream = new cudaStream_t();
        CUDAModules.checkCUDA(JCuda.cudaStreamCreate(stream));
        streamMap.put(key, stream);
        return stream;
    }
	
}
