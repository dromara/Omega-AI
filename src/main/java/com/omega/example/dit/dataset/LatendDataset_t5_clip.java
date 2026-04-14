package com.omega.example.dit.dataset;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.util.concurrent.CompletableFuture;

import com.omega.common.utils.MathUtils;
import com.omega.engine.ad.op.TensorOP;
import com.omega.engine.gpu.BaseKernel;
import com.omega.engine.gpu.CUDAManager;
import com.omega.engine.gpu.GPUOP;
import com.omega.engine.nn.network.utils.ModelUtils;
import com.omega.engine.tensor.Tensor;
import com.omega.example.dit.models.ICPlanKernel;
import com.omega.example.transformer.utils.BaseTokenizer;
import com.omega.example.transformer.utils.bpe.BinDataType;
import com.omega.example.transformer.utils.tokenizers.Tokenizer;

public class LatendDataset_t5_clip extends BaseTokenizer {
	
	private String clipDataPath;
	private String t5DataPath;
	
	public int channel;
	public int height;
	public int width;
	public int clipEmbd;
	public int clipMaxTime;
	
	public int t5Embd;
	public int t5MaxTime;
	
    public int number = 0;
    public int count_it = 0;
    public Tokenizer tokenizer;
    public int max_len = 256;
    public int vocab_size;
    public String[] vocab;
    public Tensor testInput;
    private int batchSize = 1;
    private String dataPath;
    private RandomAccessFile file;
    private RandomAccessFile clipFile;
    private RandomAccessFile t5File;
    private int index = 0;
    private float[] cache = null;
    private float[] clip_cache = null;
    private float[] t5_cache = null;
    private CompletableFuture<Boolean> cf;
    private BinDataType dataType = BinDataType.float32;
    private int byteUnit = 4;

    private Tensor mean;
    private Tensor logvar;
    private Tensor std;
    private Tensor z;
    
    private BaseKernel kernel;
    
	private ICPlanKernel icplan;

    public LatendDataset_t5_clip(String dataPath, String clipDataPath, String t5DataPath, int batchSize, int channel, int height, int width, int clipMaxTime, int clipEmbd, int t5MaxTime, int t5Embd, BinDataType dataType) {
        this.dataType = dataType;
        if (dataType == BinDataType.unint16) {
            byteUnit = 2;
        }
        this.dataPath = dataPath;
        this.clipDataPath = clipDataPath;
        this.t5DataPath = t5DataPath;
        this.channel = channel;
        this.height = height;
        this.width = width;
        this.clipEmbd = clipEmbd;
        this.clipMaxTime = clipMaxTime;
        this.t5Embd = t5Embd;
        this.t5MaxTime = t5MaxTime;
        this.max_len = channel * height * width;
        this.batchSize = batchSize;
        loadBinCount();
        this.count_it = this.number / batchSize;
        System.out.println("dataCount:" + this.number);
        System.out.println("count_it:" + this.count_it);
    }

    public static void main(String[] args) {

    }

    public int loadBinCount() {
        try {
            file = new RandomAccessFile(dataPath, "r");
            clipFile = new RandomAccessFile(clipDataPath, "r");
            t5File = new RandomAccessFile(t5DataPath, "r");
            number = (int) (file.length() / max_len / byteUnit);
            cache = new float[max_len];
            clip_cache = new float[clipMaxTime * clipEmbd];
            t5_cache = new float[t5MaxTime * t5Embd];
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
        return number;
    }

    public void initBinReader() {
        try {
        	index = 0;
            file.seek(0);
            clipFile.seek(0);
            t5File.seek(0);
            System.out.println("dataset is ready.");
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public float[] loadData() {
        try {
            if ((index + 1) * max_len * byteUnit <= file.length()) {
                //				System.out.println(index);
                if (dataType == BinDataType.float32) {
                	 ModelUtils.readFloatArray(file, cache);
                     ModelUtils.readFloatArray(clipFile, clip_cache);
                     ModelUtils.readFloatArray(t5File, t5_cache);
                }
                file.seek(file.getFilePointer());
                clipFile.seek(clipFile.getFilePointer());
                t5File.seek(t5File.getFilePointer());
                index++;
            } else {
                initBinReader();
                return loadData();
            }
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
        return cache;
    }
    
    public float[] loadData(long idx) {
        try {

        	long fi = idx * max_len * byteUnit;
        	long cfi = idx * clipMaxTime * clipEmbd * byteUnit;
        	long tfi = idx * t5MaxTime * t5Embd * byteUnit;
//            	System.err.println(fi);
        	if(fi < file.length()) {
        		file.seek(fi);
                clipFile.seek(cfi);
                t5File.seek(tfi);
                if (dataType == BinDataType.float32) {
                    ModelUtils.readFloatArray(file, cache);
                    ModelUtils.readFloatArray(clipFile, clip_cache);
                    ModelUtils.readFloatArray(t5File, t5_cache);
                }
        	}else {
        		System.err.println("dataset index["+idx+"] is out.");
        	}

        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
        return cache;
    }

    public void loadData(int[] index,Tensor input, Tensor clipLabel, Tensor t5clipLabel) {
    	input.hostToDevice();
    	clipLabel.hostToDevice();
    	t5clipLabel.hostToDevice();
        cf = loadAsyncData(index, input, clipLabel, t5clipLabel);
    }
    
    public void loadData(int[] index,Tensor input, Tensor clipLabel, Tensor t5Label, int it) {
        try {
            //			System.out.println(it);
            if (cf != null) {
                boolean success = cf.get();
                if(success){
                	cf = null;
                	loadData(index, input, clipLabel, t5Label);
                }
            } else {
                cf = loadAsyncData(index, input, clipLabel, t5Label);
                boolean success = cf.get();
                if(success){
                	cf = null;
                	loadData(index, input, clipLabel, t5Label);
                }
            }
//            System.out.println("load cost:"+(System.nanoTime() - start)/1e6+"ms.");
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
    
    public void loadData(int[] index, int[] next,Tensor input, Tensor clipLabel, Tensor t5Label, int it) {
        try {
            //			System.out.println(it);
        	if(it == 0) {
        		cf = null;
        	}
            if (cf != null) {
                boolean success = cf.get();//等待数据从文件加载完毕
                if(success){
                	cf = null;
                	/**
                	 *  input.hostToDevice(); //把当前内存的数据加载到显存上
				     *  label.hostToDevice(); //把当前内存的数据加载到显存上
				     *  cf = loadAsyncData(index, input, label); //开启下一轮文件数据的读取
                	 */
                	loadData(next, input, clipLabel, t5Label);
                }
            } else {
            	/**
            	 * 首轮数据加载
            	 */
                cf = loadAsyncData(index, input, clipLabel, t5Label);
                boolean success = cf.get();
                if(success){
                	cf = null;
                	loadData(next, input, clipLabel, t5Label);
                }
            }
//            System.out.println("load cost:"+(System.nanoTime() - start)/1e6+"ms.");
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
    
    public Tensor sample(TensorOP tensorOP,Tensor en_out) {
    	
    	if(z == null || z.number != en_out.number) {
    		mean = Tensor.createGPUTensor(mean, en_out.number, channel/2, en_out.height, en_out.width, true);
    		logvar = Tensor.createGPUTensor(logvar, mean.shape(), true);
    		std = Tensor.createGPUTensor(std, mean.shape(), true);
    		z = Tensor.createGPUTensor(z, mean.shape(), true);
    	}
    	
    	GPUOP.getInstance().cudaRandn(z);

    	tensorOP.getByChannel(en_out, mean, 0, channel/2);
    	tensorOP.getByChannel(en_out, logvar, channel/2, channel/2);
    	
    	tensorOP.clamp(logvar, -30, 20, logvar);
    	
    	tensorOP.mul(logvar, 0.5f, std);
    	tensorOP.exp(std, std);
    	
    	tensorOP.mul(z, std, z);
    	tensorOP.add(z, mean, z);
    	
    	return z;
    }
    
    public float[] readIdxData() throws IOException {
    	 return loadData();
    }
    
    public float[] readIdxData(int idx) throws IOException {
   	 return loadData(idx);
   }
    
    public CompletableFuture<Boolean> loadAsyncData(int[] index, Tensor input, Tensor clipLabel, Tensor t5clipLabel) {
        CompletableFuture<Boolean> cf = CompletableFuture.supplyAsync(() -> {
            try {
//            	long start = System.nanoTime();
                for (int b = 0; b < batchSize; b++) {
                	int idx = index[b];
                    float[] onceToken = readIdxData(idx);
                    float[] clipToken = clip_cache;
                    float[] t5Token = t5_cache;
                    System.arraycopy(onceToken, 0, input.data, b * onceToken.length, onceToken.length);
                    System.arraycopy(clipToken, 0, clipLabel.data, b * clipToken.length, clipToken.length);
                    System.arraycopy(t5Token, 0, t5clipLabel.data, b * t5Token.length, t5Token.length);
                }
//                System.out.println("load cost:"+(System.nanoTime() - start)/1e6+"ms.");
            } catch (Exception e) {
                // TODO: handle exception
                e.printStackTrace();
            }
            return true;
        });
        return cf;
    }

    public CompletableFuture<Boolean> loadAsyncData(float[] input, float[] label) {
        CompletableFuture<Boolean> cf = CompletableFuture.supplyAsync(() -> {
            try {
                for (int b = 0; b < batchSize; b++) {
                	float[] onceToken = readIdxData();
                	float[] clipToken = clip_cache;
                	for (int t = 0; t < max_len; t++) {
                		formatNotHeadToIdx(b, t, onceToken, input);
                    }
                	for(int t = 0;t < clipMaxTime * clipEmbd;t++) {
                    	formatNotHeadToIdx(b, t, clipMaxTime * clipEmbd, clipToken, label);
                    }
                }
            } catch (Exception e) {
                // TODO: handle exception
                e.printStackTrace();
            }
            return true;
        });
        return cf;
    }
    
    public void formatNotHeadToIdx(int b, int t, float[] onceToken, Tensor input) {
        if (t < onceToken.length) {
            input.data[b * max_len + t] = onceToken[t];
        }
    }
    
    public void formatNotHeadToIdx(int b, int t, int max_len, float[] onceToken, Tensor input) {
        if (t < onceToken.length) {
            input.data[b * max_len + t] = onceToken[t];
        }
    }
    
    public void formatNotHeadToIdx(int b, int t, float[] onceToken, float[] input) {
        if (t < onceToken.length) {
            input[b * max_len + t] = onceToken[t];
        }
    }
    
    public void formatNotHeadToIdx(int b, int t, int max_len, float[] onceToken, float[] input) {
        if (t < onceToken.length) {
            input[b * max_len + t] = onceToken[t];
        }
    }
    
    public int[][] shuffle() {
        // TODO Auto-generated method stub
        return MathUtils.randomInts(this.number, this.batchSize);
    }
    
    public int[][] order() {
        // TODO Auto-generated method stub
        return MathUtils.orderInts(this.number, this.batchSize);
    }
    
    public void addNoise(Tensor a, Tensor b, Tensor input, Tensor noise, CUDAManager cudaManager) {
        if (kernel == null) {
            kernel = new BaseKernel(cudaManager);
        }
        kernel.add_mul(a, b, input, noise, input);
    }
    
	public void latend_norm(Tensor x,Tensor mean,Tensor std, CUDAManager cudaManager) {
		 if(icplan == null) {
			 icplan = new ICPlanKernel(cudaManager);
		 }
		 icplan.latend_norm(x, mean, std);
	}
    
	public void latend_un_norm(Tensor x,Tensor mean,Tensor std, CUDAManager cudaManager) {
		 if(icplan == null) {
			 icplan = new ICPlanKernel(cudaManager);
		 }
		 icplan.latend_un_norm(x, mean, std);
	}
	
}
