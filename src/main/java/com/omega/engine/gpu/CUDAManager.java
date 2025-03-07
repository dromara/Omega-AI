package com.omega.engine.gpu;

import static jcuda.driver.JCudaDriver.cuModuleGetFunction;
import static jcuda.driver.JCudaDriver.cuModuleLoad;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import java.util.HashMap;
import java.util.Map;

import com.omega.common.lib.LibPaths;
import com.omega.common.utils.JarPathUtils;

import jcuda.driver.CUcontext;
import jcuda.driver.CUdevice;
import jcuda.driver.CUfunction;
import jcuda.driver.JCudaDriver;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaDeviceProp;
import jcuda.runtime.cudaError;

public class CUDAManager {
	
	public Map<String,MyCUDAModule> modules = new HashMap<String,MyCUDAModule>();
	
	private CUDAMemoryManager memoryManager;
	
	private int deviceId;
	
	private CUdevice device;
	
	private CUcontext context;
	
	private CUDAUtils instance;
	
	public int maxThreads;
	
	public int threadsPerDimension;
	
	public cudaDeviceProp props;
	
	private final String CU_PATH = "cu/";
	
	private final String TMP_PATH = "/tmp/";
	
	private GPUOP op;
	
	public CUDAManager(int deviceId) {
		this.deviceId = deviceId;
		initContext();
		this.memoryManager = new CUDAMemoryManager();
		this.op = new GPUOP(deviceId);
	}
	
	public Map<String,String> functions = new HashMap<String, String>(){
		/**
		 * 
		 */
		private static final long serialVersionUID = 3230964896540863511L;
		
		{
			put("col2im_gpu_kernelV2", "Col2imKernel.cu");
			put("im2col_gpu_kernelV2", "Im2colKernel.cu");
			put("pooling_diff", "PoolingKernel.cu");
			put("max_pooling", "PoolingKernel.cu");
			put("mean_pooling", "PoolingKernel.cu");
			put("mean_cov", "MathKernel.cu");
			put("fast_mean_kernel", "MathKernel.cu");
			put("var_cov", "MathKernel.cu");
			put("fast_variance_kernel", "MathKernel.cu");
			put("normalize_kernel", "BNKernel.cu");
			put("std_fn", "MathKernel.cu");
			put("mwa", "MathKernel.cu");
			put("culOutput_cov", "BNKernel.cu");
			put("computeDelta", "BNKernel.cu");
			put("computeDelta_full", "BNKernel.cu");
			put("meanDzSum", "BNKernel.cu");
			put("computeDiff", "BNKernel.cu");
			put("dgama_kernel", "BNKernel.cu");
			put("dbeta_kernel", "BNKernel.cu");
			put("dxhat_kernel2", "BNKernel.cu");
			put("full_mean_delta_kernel", "BNKernel.cu");
			put("full_var_delta_kernel", "BNKernel.cu");
			put("fast_variance_delta_kernel", "BNKernel.cu");
			put("dx_kernel", "BNKernel.cu");
			put("dx_kernel_full", "BNKernel.cu");

			put("copy_kernel", "OPKernel.cu");
			put("copy_number_kernel", "OPKernel.cu");
			put("copy_channel_kernel", "OPKernel.cu");
			put("add_kernel", "OPKernel.cu");
			put("add_scalar_kernel", "OPKernel.cu");
			put("add_number_kernel", "OPKernel.cu");
			put("add_channel_kernel", "OPKernel.cu");
			put("sub_kernel", "OPKernel.cu");
			put("sub_scalar_kernel", "OPKernel.cu");
			put("mul_kernel", "OPKernel.cu");
			put("mul_scalar_kernel", "OPKernel.cu");
			put("mul_plus_kernel", "OPKernel.cu");
			put("mul_plus_scalar_kernel", "OPKernel.cu");
			put("div_kernel", "OPKernel.cu");
			put("div_scalar_kernel", "OPKernel.cu");
			put("scalar_div_kernel", "OPKernel.cu");
			put("div_plus_kernel", "OPKernel.cu");
			put("div_plus_scalar_kernel", "OPKernel.cu");
			put("scalar_plus_div_kernel", "OPKernel.cu");
			put("div_bGrad_kernel", "OPKernel.cu");
			put("div_scalar_bGrad_kernel", "OPKernel.cu");
			put("pow_kernel", "OPKernel.cu");
			put("log_kernel", "OPKernel.cu");
			put("exp_kernel", "OPKernel.cu");
			put("sin_kernel", "OPKernel.cu");
			put("cos_kernel", "OPKernel.cu");
			
		}
		};

	public CUfunction getLocalFunctionByModule(String fileName,String functionName) {

		String rootPath = LibPaths.LIB_PATH;
		
		fileName = rootPath + fileName;
		
		File file = new File(fileName);

		if(!file.exists()) {
			try {
				URL url = CUDAManager.class.getProtectionDomain().getCodeSource().getLocation();
				JarPathUtils.copyJarResources(url.getPath(), CU_PATH, rootPath, CUDAManager.class);
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}

		MyCUDAModule m = this.getModule(fileName);

		if(m.getFunctions().containsKey(functionName)) {
			return m.getFunctions().get(functionName);
		}
		
		CUfunction function = new CUfunction();
		
		checkCUDA(cuModuleGetFunction(function, m, functionName));
        m.getFunctions().put(functionName, function);
        
		return function;
	}
	
	public CUfunction getEXFunctionByModule(String fileName,String functionName) {
		
		MyCUDAModule m = this.getModule(fileName);
		
		if(m.getFunctions().containsKey(functionName)) {
			return m.getFunctions().get(functionName);
		}
		
		CUfunction function = new CUfunction();
		checkCUDA(cuModuleGetFunction(function, m, functionName));
        m.getFunctions().put(functionName, function);
        
		return function;
	}
	
	public MyCUDAModule getModule(String fileName) {
		
		// Create the PTX file by calling the NVCC
        try {
        	
			String ptxFileName = preparePtxFile(fileName);
			
			if(this.modules.containsKey(ptxFileName)) {
				return this.modules.get(ptxFileName);
			}

        	setContext(getContext());
        	
        	maxThreads = instance.getMaxThreads(device);
        	
            threadsPerDimension = (int) Math.sqrt(maxThreads);
             
	        // Load the ptx file.
	        MyCUDAModule module = new MyCUDAModule();
	        
	        cuModuleLoad(module, ptxFileName);
	        
	        this.modules.put(ptxFileName, module);
			
			return module;
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
        return null;
	}
	
    /**
     * The extension of the given file name is replaced with "ptx".
     * If the file with the resulting name does not exist, it is
     * compiled from the given file using NVCC. The name of the
     * PTX file is returned.
     *
     * @param cuFileName The name of the .CU file
     * @return The name of the PTX file
     * @throws IOException If an I/O error occurs
     */
    private String preparePtxFile(String cuFileName) throws IOException{
        int endIndex = cuFileName.lastIndexOf('.');
        if (endIndex == -1)
        {
            endIndex = cuFileName.length()-1;
        }
        String ptxFileName = cuFileName.substring(0, endIndex+1)+"ptx";
        File ptxFile = new File(ptxFileName);
        if (ptxFile.exists()) {
            return ptxFileName;
        }
        File cuFile = new File(cuFileName);
        if (!cuFile.exists()) {
            throw new IOException("Input file not found: "+cuFileName);
        }
        String modelString = "-m"+System.getProperty("sun.arch.data.model");
        String command =
            "nvcc " + modelString + " -ptx "+
            cuFile.getPath()+" -o "+ptxFileName;

        System.out.println("Executing\n"+command);
        Process process = Runtime.getRuntime().exec(command);

        String errorMessage =
            new String(toByteArray(process.getErrorStream()));
        String outputMessage =
            new String(toByteArray(process.getInputStream()));
        int exitValue = 0;
        try {
            exitValue = process.waitFor();
        }
        catch (InterruptedException e)
        {
            Thread.currentThread().interrupt();
            throw new IOException(
                "Interrupted while waiting for nvcc output", e);
        }

        if (exitValue != 0)
        {
            System.out.println("nvcc process exitValue "+exitValue);
            System.out.println("errorMessage:\n"+errorMessage);
            System.out.println("outputMessage:\n"+outputMessage);
            throw new IOException(
                "Could not create .ptx file: "+errorMessage);
        }

        System.out.println("Finished creating PTX file");
        return ptxFileName;
    }

    /**
     * Fully reads the given InputStream and returns it as a byte array
     *
     * @param inputStream The input stream to read
     * @return The byte array containing the data from the input stream
     * @throws IOException If an I/O error occurs
     */
    private byte[] toByteArray(InputStream inputStream)
        throws IOException
    {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        byte buffer[] = new byte[8192];
        while (true)
        {
            int read = inputStream.read(buffer);
            if (read == -1)
            {
                break;
            }
            baos.write(buffer, 0, read);
        }
        return baos.toByteArray();
    }

	public CUcontext getContext() {
		
		if(context == null) {
			
			JCudaDriver.setExceptionsEnabled(true);
        	
			// Initialize the driver and create a context for the first device.

        	instance = CUDAUtils.getInstance();
        	
        	instance.initCUDA();
        	
        	device = instance.getDevice(deviceId);
        	
        	context = instance.getContext(device);
        	
        	props = new cudaDeviceProp();
        	
        	JCuda.cudaGetDeviceProperties(props, deviceId);
			
        	System.out.println("CUDA["+deviceId+"] context init finish.");
        	
		}
		
		return context;
	}
	
	public void initContext() {
		getContext();
	}
	
	public void setContext(CUcontext context) {
		this.context = context;
	}
	
	public void initCUDAFunctions() {
		for(String key:functions.keySet()) {
			this.getLocalFunctionByModule(functions.get(key), key);
		}
		System.out.println("CUDA functions init finish.");
	}
	
	public static void checkCUDA(int code) {
		if(code != cudaError.cudaSuccess) {
			System.err.println("Error code "+code+":"+cudaError.stringFor(code));
			throw new RuntimeException("Error code "+code+":"+cudaError.stringFor(code));
		}
	}

	public GPUOP getOp() {
		return op;
	}

	public CUDAMemoryManager getMemoryManager() {
		return memoryManager;
	}
	
}
