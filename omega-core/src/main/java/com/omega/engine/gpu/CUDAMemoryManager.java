package com.omega.engine.gpu;

import com.omega.engine.parallel.ddp.distributed.SerializablePointer;
import com.omega.engine.tensor.Tensor;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUdeviceptr;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaError;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static jcuda.driver.JCudaDriver.cuMemAlloc;
import static jcuda.runtime.JCuda.cudaMalloc;

public class CUDAMemoryManager {
    public static Map<String, CUdeviceptr> deviceMap = new HashMap<String, CUdeviceptr>();
    public static Map<String, Pointer> pointerMap = new HashMap<String, Pointer>();
    public static List<CUdeviceptr> cu_deviceptrs = new ArrayList<CUdeviceptr>();
    public static List<Pointer> cu_porints = new ArrayList<Pointer>();
    public static GPUWorkspace workspace = new GPUWorkspace();
    public static Tensor globalCache;
    public static Map<String, Tensor> caches = new HashMap<String, Tensor>();
    private List<Pointer> porints = new ArrayList<Pointer>();
    private Map<String, Tensor> privateCaches = new HashMap<String, Tensor>();

    public synchronized static Tensor getCache(String key, int N, int C, int H, int W) {
        Tensor c = null;
        if (caches.containsKey(key)) {
            c = caches.get(key);
            //			System.err.println("["+key+"]"+c.gpuLength+":["+N+":"+C+":"+H+":"+W+"]:"+N * C * H * W);
            if (c.getGpuLength() < N * C * H * W) {
                c = Tensor.createGPUTensor(c, N, C, H, W, true);
            } else {
                c = c.viewOrg(N, C, H, W);
            }
        } else {
            //			System.err.println("["+key+"]:"+N * C * H * W);
            c = Tensor.createGPUTensor(c, N, C, H, W, true);
            caches.put(key, c);
        }
        //		JCuda.cudaDeviceSynchronize();
        return c;
    }

    public synchronized static Tensor getGlobalCache(int N, int C, int H, int W) {
        //		if(globalCache != null) {
        //			System.out.println(globalCache.dataLength+":"+N * C * H * W);
        //		}
        if (globalCache == null || globalCache.getDataLength() < N * C * H * W) {
            globalCache = Tensor.createGPUTensor(globalCache, N, C, H, W, true);
        } else {
            globalCache = globalCache.viewOrg(N, C, H, W);
        }
        return globalCache;
    }

    public synchronized static CUdeviceptr getDevice(int size) {
        CUdeviceptr device = new CUdeviceptr();
        cuMemAlloc(device, size * (long) Sizeof.FLOAT);
        cu_deviceptrs.add(device);
        return device;
    }

    public synchronized static CUdeviceptr getDevice(String key, int size) {
        if (deviceMap.containsKey(key)) {
            return deviceMap.get(key);
        }
        CUdeviceptr device = new CUdeviceptr();
        //		System.out.println(key+":"+size);
        cuMemAlloc(device, size * (long) Sizeof.FLOAT);
        deviceMap.put(key, device);
        return device;
    }

    public synchronized static Pointer getWorkspace(int size) {
        if (workspace.getSize() < size * Sizeof.FLOAT) {
            GPUOP.getInstance().free(workspace.getPointer());
            cudaMalloc(workspace.getPointer(), size * (long) Sizeof.FLOAT);
            workspace.setSize(size * Sizeof.FLOAT);
        }
        return workspace.getPointer();
    }

    public synchronized static Pointer getPointer(int size) {
        Pointer p = new Pointer();
        checkCUDA(cudaMalloc(p, size * (long) Sizeof.FLOAT), p.toString(), size * (long) Sizeof.FLOAT);
        cu_porints.add(p);
        return p;
    }

    public synchronized static SerializablePointer getSharePointer(int size) {
        SerializablePointer p = new SerializablePointer();
        checkCUDA(cudaMalloc(p, size * (long) Sizeof.FLOAT), p.toString(), size * (long) Sizeof.FLOAT);
        cu_porints.add(p);
        return p;
    }

    public synchronized static Pointer getPointer(int size, long type) {
        Pointer p = new Pointer();
        checkCUDA(cudaMalloc(p, size * type));
        cu_porints.add(p);
        return p;
    }

    public synchronized static Pointer getPointer(String key, int size) {
        if (pointerMap.containsKey(key)) {
            return pointerMap.get(key);
        }
        Pointer p = new Pointer();
        cudaMalloc(p, size * (long) Sizeof.FLOAT);
        pointerMap.put(key, p);
        return p;
    }

    public synchronized static void free() {
        for (String key : deviceMap.keySet()) {
            JCuda.cudaFree(deviceMap.get(key));
        }
        for (String key : pointerMap.keySet()) {
            GPUOP.getInstance().free(pointerMap.get(key));
        }
        for (String key : caches.keySet()) {
            GPUOP.getInstance().free(caches.get(key).getGpuData());
        }
    }

    public synchronized static void free(Pointer pointer) {
        checkCUDA(JCuda.cudaFree(pointer), "free" + pointer.toString());
        checkCUDA(JCuda.cudaDeviceSynchronize());
        cu_porints.remove(pointer);
    }

    public synchronized static void freeAll() throws Exception {
        for (CUdeviceptr dec : cu_deviceptrs) {
            JCuda.cudaFree(dec);
        }
        for (Pointer p : cu_porints) {
            GPUOP.getInstance().free(p);
        }
    }

    public static void checkCUDA(int code, String op, long size) {
        if (code != cudaError.cudaSuccess) {
            String error = "[[" + op + "](" + size + ")]Error code " + code + ":" + cudaError.stringFor(code);
            throw new RuntimeException(error);
            //			System.err.println();
        }
    }

    public static void checkCUDA(int code, String op) {
        if (code != cudaError.cudaSuccess) {
            String error = "[" + op + "]Error code " + code + ":" + cudaError.stringFor(code);
            throw new RuntimeException(error);
            //			System.err.println();
        }
    }

    public static void checkCUDA(int code) {
        if (code != cudaError.cudaSuccess) {
            String error = cudaError.stringFor(code);
            throw new RuntimeException(error);
            //			System.err.println();
        }
    }

    public Tensor getPrivateCaches(String key, int N, int C, int H, int W) {
        Tensor c = null;
        if (privateCaches.containsKey(key)) {
            c = privateCaches.get(key);
            //			System.err.println("["+key+"]"+c.gpuLength+":["+N+":"+C+":"+H+":"+W+"]:"+N * C * H * W);
            if (c.getGpuLength() < N * C * H * W) {
                c = Tensor.createGPUTensor(c, N, C, H, W, true);
            } else {
                c = c.viewOrg(N, C, H, W);
            }
        } else {
            //			System.err.println("["+key+"]:"+N * C * H * W);
            c = Tensor.createGPUTensor(c, N, C, H, W, true);
            privateCaches.put(key, c);
        }
        //		JCuda.cudaDeviceSynchronize();
        return c;
    }

    public Pointer getCUPointer(int size, long type) {
        Pointer p = new Pointer();
        checkCUDA(cudaMalloc(p, size * type));
        porints.add(p);
        return p;
    }
    
	public void freeCUPointer(Pointer pointer) {
		checkCUDA(JCuda.cudaFree(pointer), "free" + pointer.toString());
		porints.remove(pointer);
	}
}

