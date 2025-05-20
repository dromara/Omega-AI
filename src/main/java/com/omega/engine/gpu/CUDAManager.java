package com.omega.engine.gpu;

import static jcuda.driver.JCudaDriver.cuModuleGetFunction;
import static jcuda.driver.JCudaDriver.cuModuleLoad;
import static jcuda.driver.JCudaDriver.cuModuleLoadData;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;
import java.util.jar.JarFile;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import com.omega.common.lib.LibPaths;

import cn.hutool.core.io.FileUtil;
import cn.hutool.core.io.resource.ResourceUtil;
import jcuda.driver.CUcontext;
import jcuda.driver.CUdevice;
import jcuda.driver.CUfunction;
import jcuda.driver.JCudaDriver;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaDeviceProp;
import jcuda.runtime.cudaError;

public class CUDAManager {

    private final static boolean nvcc = false;

    private final String CU_PATH = "cu/";
    public static Map<String, String> ptxList;
    public Map<String, MyCUDAModule> modules = new HashMap<String, MyCUDAModule>();
    public int maxThreads;
    public int threadsPerDimension;
    public cudaDeviceProp props;
//    private static final String CU_SUFFIX = ".cu";
    private static final String PTX_SUFFIX = ".ptx";

    public Map<String, String> functions = new HashMap<String, String>() {
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
    private CUDAMemoryManager memoryManager;
    private int deviceId;
    private CUdevice device;
    private CUcontext context;
    //	private final String TMP_PATH = "/tmp/";
    private CUDAUtils instance;
    private GPUOP op;

    public CUDAManager(int deviceId) {
        this.deviceId = deviceId;
        initContext();
        this.memoryManager = new CUDAMemoryManager();
        this.op = new GPUOP(deviceId);
    }

    public static void checkCUDA(int code) {
        if (code != cudaError.cudaSuccess) {
            System.err.println("Error code " + code + ":" + cudaError.stringFor(code));
            throw new RuntimeException("Error code " + code + ":" + cudaError.stringFor(code));
        }
    }

    public CUfunction getLocalFunctionByModule(String fileName, String functionName) {

		if(!nvcc) {
			Pattern pattern = Pattern.compile("(?<=\\.)[^\\.]+$"); // 正则表达式匹配最后一个点号后的部分（扩展名）
	        Matcher matcher = pattern.matcher(fileName);
	        if (matcher.find()) {
	        	fileName = fileName.replaceFirst(matcher.group(), "ptx"); // 替换匹配到的部分
	        } else {
	            System.out.println("No extension found");
	        }
		}

    	if(ptxList == null) {
    		listCuFiles(CU_PATH);
    	}

    	MyCUDAModule m = null;

    	if(nvcc) {
    		String rootPath = LibPaths.LIB_PATH;
            fileName = rootPath + fileName;
            m = this.getModule(fileName);
    	}else {
    		m = this.getModule(fileName, ptxList.get(fileName));
    	}

    	if (m.getFunctions().containsKey(functionName)) {
            return m.getFunctions().get(functionName);
        }

    	CUfunction function = new CUfunction();
        checkCUDA(cuModuleGetFunction(function, m, functionName));
        m.getFunctions().put(functionName, function);
        return function;
    }

    public static void listCuFiles(String directory) {

        ptxList = new HashMap<>();
        try {
            String path = CUDAModules.class.getProtectionDomain().getCodeSource().getLocation().getPath();
            path = java.net.URLDecoder.decode(path, "UTF-8");

            // In Jar file else in IDE
            if (path.endsWith(".jar")) {
               loadCuFileFromJar(path, directory);
            } else {
               loadCuFileFromDirectory(path, directory);
            }

        } catch (Exception e) {
            e.printStackTrace();
//            log("Exception:" + e.getMessage() + "\r" + ArrayUtil.join(e.getStackTrace(), "\r"));
        }

    }

    private static Map<String, byte[]> loadCuFileFromDirectory(String path, String directory) throws Exception{

        Map<String, byte[]> cuFiles = new HashMap<>();
        String cuPath = path + "/" + directory;
        File file = new java.io.File(cuPath);
        if (file.isDirectory() && null != file.listFiles()) {
            java.util.Arrays.stream(Objects.requireNonNull(file.listFiles()))
                    .filter(entry -> entry.getName().toLowerCase().endsWith(PTX_SUFFIX))
                    .forEach(entry -> {
                    	String fullName = cuPath + "/" + entry.getName();
  	            	  	ptxList.put(entry.getName(), FileUtil.readUtf8String(fullName));
                    });
        }

        return cuFiles;
    }

    public static void loadCuFileFromJar(String path, String directory) throws Exception{

        try (JarFile jarFile = new JarFile(path)) {
            jarFile.stream()
                    .filter(entry -> !entry.isDirectory())
                    .filter(entry -> entry.getName().startsWith(directory))
                    .filter(entry -> entry.getName().toLowerCase().endsWith(PTX_SUFFIX))
                    .forEach(entry -> {
                    	String content = ResourceUtil.readUtf8Str(entry.getName());
  	            	  	ptxList.put(entry.getName().replaceAll(directory, ""), content);
                    });
        }

    }

    public CUfunction getEXFunctionByModule(String fileName, String functionName) {
        MyCUDAModule m = this.getModule(fileName);
        if (m.getFunctions().containsKey(functionName)) {
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
            if (this.modules.containsKey(ptxFileName)) {
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

    public MyCUDAModule getModule(String fileName, byte[] data) {
        if (CUDAModules.modules.containsKey(fileName)) {
		    return CUDAModules.modules.get(fileName);
		}
		setContext(getContext());
		maxThreads = instance.getMaxThreads(device);
		threadsPerDimension = (int) Math.sqrt(maxThreads);
		// Load the ptx file.
		MyCUDAModule module = new MyCUDAModule();
		try {
			cuModuleLoadData(module, data);
			CUDAModules.modules.put(fileName, module);
		} catch (Exception e) {
			// TODO: handle exception
			System.err.println(fileName+" init fail.");
			e.printStackTrace();
		}
		return module;
    }

    public MyCUDAModule getModule(String fileName, String content) {
        if (CUDAModules.modules.containsKey(fileName)) {
		    return CUDAModules.modules.get(fileName);
		}
		setContext(getContext());
		maxThreads = instance.getMaxThreads(device);
		threadsPerDimension = (int) Math.sqrt(maxThreads);
		// Load the ptx file.
		MyCUDAModule module = new MyCUDAModule();
		try {
			cuModuleLoadData(module, content);
			CUDAModules.modules.put(fileName, module);
		} catch (Exception e) {
			// TODO: handle exception
			System.err.println(fileName+" init fail.");
			e.printStackTrace();
		}
		return module;
    }

    public MyCUDAModule getModule(String fileName, String content) {
        if (CUDAModules.modules.containsKey(fileName)) {
            return CUDAModules.modules.get(fileName);
        }
        setContext(getContext());
        maxThreads = instance.getMaxThreads(device);
        threadsPerDimension = (int) Math.sqrt(maxThreads);
        // Load the ptx file.
        MyCUDAModule module = new MyCUDAModule();
        try {
            cuModuleLoadData(module, content);
            CUDAModules.modules.put(fileName, module);
        } catch (Exception e) {
            // TODO: handle exception
            System.err.println(fileName+" init fail.");
            e.printStackTrace();
        }
        return module;
    }



    /**
     * The extension of the given file name is replaced with "ptx".
     * <p>
     * If the file with the resulting name does not exist, it is
     * <p>
     * compiled from the given file using NVCC. The name of the
     * <p>
     * PTX file is returned.
     *
     * @param cuFileName The name of the .CU file
     * @return The name of the PTX file
     * @throws IOException If an I/O error occurs
     */
    private String preparePtxFile(String cuFileName) throws IOException {
        int endIndex = cuFileName.lastIndexOf('.');
        if (endIndex == -1) {
            endIndex = cuFileName.length() - 1;
        }
        String ptxFileName = cuFileName.substring(0, endIndex + 1) + "ptx";
        File ptxFile = new File(ptxFileName);
        if (ptxFile.exists()) {
            return ptxFileName;
        }
        File cuFile = new File(cuFileName);
        if (!cuFile.exists()) {
            throw new IOException("Input file not found: " + cuFileName);
        }
        String modelString = "-m" + System.getProperty("sun.arch.data.model");
        String command = "nvcc " + modelString + " -ptx " + cuFile.getPath() + " -o " + ptxFileName;
        System.out.println("Executing\n" + command);
        Process process = Runtime.getRuntime().exec(command);
        String errorMessage = new String(toByteArray(process.getErrorStream()));
        String outputMessage = new String(toByteArray(process.getInputStream()));
        int exitValue = 0;
        try {
            exitValue = process.waitFor();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IOException("Interrupted while waiting for nvcc output", e);
        }
        if (exitValue != 0) {
            System.out.println("nvcc process exitValue " + exitValue);
            System.out.println("errorMessage:\n" + errorMessage);
            System.out.println("outputMessage:\n" + outputMessage);
            throw new IOException("Could not create .ptx file: " + errorMessage);
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
    private byte[] toByteArray(InputStream inputStream) throws IOException {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        byte buffer[] = new byte[8192];
        while (true) {
            int read = inputStream.read(buffer);
            if (read == -1) {
                break;
            }
            baos.write(buffer, 0, read);
        }
        return baos.toByteArray();
    }

    public CUcontext getContext() {
        if (context == null) {
            JCudaDriver.setExceptionsEnabled(true);
            // Initialize the driver and create a context for the first device.
            instance = CUDAUtils.getInstance();
            instance.initCUDA();
            device = instance.getDevice(deviceId);
            context = instance.getContext(device);
            props = new cudaDeviceProp();
            JCuda.cudaGetDeviceProperties(props, deviceId);
            System.out.println("CUDA[" + deviceId + "] context init finish.");
        }
        return context;
    }

    public void setContext(CUcontext context) {
        this.context = context;
    }

    public void initContext() {
        getContext();
    }

    public void initCUDAFunctions() {
        for (String key : functions.keySet()) {
            this.getLocalFunctionByModule(functions.get(key), key);
        }
        System.out.println("CUDA functions init finish.");
    }

    public GPUOP getOp() {
        return op;
    }

    public CUDAMemoryManager getMemoryManager() {
        return memoryManager;
    }
}

