package com.omega.example.vae.dataset;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import com.omega.common.utils.JsonUtils;
import com.omega.common.utils.MathUtils;
import com.omega.common.utils.RandomUtils;
import com.omega.engine.gpu.BaseKernel;
import com.omega.engine.gpu.CUDAManager;
import com.omega.engine.tensor.Tensor;
import com.omega.example.transformer.utils.bpe.BPETokenizerEN;
import com.omega.example.vae.utils.VideoDataLoader;
import com.omega.example.yolo.data.BaseDataLoader;

import cn.hutool.core.io.FileUtil;
import cn.hutool.core.text.csv.CsvData;
import cn.hutool.core.text.csv.CsvReader;
import cn.hutool.core.text.csv.CsvRow;
import cn.hutool.core.text.csv.CsvUtil;

/**
 * VideoDataLoaderEN
 *
 * @author Administrator
 */
public class VideoDataLoaderEN extends BaseDataLoader {
	public int num_frames = 17;
    public int img_w;
    public int img_h;
    public boolean normalization = true;
    public float[] mean;
    public float[] std;
    public int count;
    public int count_it;
    private String csvPath;
    private String imgDirPath;
    private String extName = ".png";
    private int maxContextLen;
    private List<Map<String, String>> datas;
    private BPETokenizerEN tokenizer;
    private BaseKernel kernel;

    public VideoDataLoaderEN(BPETokenizerEN tokenizer, String csvPath, String imgDirPath, int num_frames, int img_w, int img_h, int maxContextLen, int batchSize) {
        this.maxContextLen = maxContextLen;
        this.tokenizer = tokenizer;
        this.csvPath = csvPath;
        this.imgDirPath = imgDirPath;
        this.num_frames = num_frames;
        this.img_w = img_w;
        this.img_h = img_h;
        this.batchSize = batchSize;
        init();
    }

    public VideoDataLoaderEN(BPETokenizerEN tokenizer, String csvPath, String imgDirPath, int num_frames, int img_w, int img_h, int maxContextLen, int batchSize, float[] mean, float[] std) {
        this.csvPath = csvPath;
        this.imgDirPath = imgDirPath;
        this.maxContextLen = maxContextLen;
        this.tokenizer = tokenizer;
        this.num_frames = num_frames;
        this.img_w = img_w;
        this.img_h = img_h;
        this.batchSize = batchSize;
        this.mean = mean;
        this.std = std;
        init();
    }
    
    public VideoDataLoaderEN(BPETokenizerEN tokenizer, String csvPath, String imgDirPath, String extName,int num_frames, int img_w, int img_h, int maxContextLen, int batchSize, float[] mean, float[] std) {
        this.csvPath = csvPath;
        this.imgDirPath = imgDirPath;
        this.maxContextLen = maxContextLen;
        this.tokenizer = tokenizer;
        this.num_frames = num_frames;
        this.img_w = img_w;
        this.img_h = img_h;
        this.batchSize = batchSize;
        this.mean = mean;
        this.std = std;
        this.extName = extName;
        init();
    }

    public void init() {
        loadFileCount();
    }
    
    public void loadCSVData(String path) {
    	
    	CsvReader reader = CsvUtil.getReader();
        CsvData data = reader.read(FileUtil.file(path));
        
        List<CsvRow> rows = data.getRows();
        
        datas = new ArrayList<Map<String,String>>();
        
        for(int i = 1;i<rows.size();i++) {
        	CsvRow row = rows.get(i);
        	Map<String,String> once = new HashMap<String, String>();
        	once.put("filename", row.get(8));
        	once.put("path", row.get(0));
        	once.put("text", row.get(1));
        	once.put("num_frames", row.get(2));
        	once.put("height", row.get(3));
        	once.put("width", row.get(4));
        	datas.add(once);
        }
    	
    }
    
    public void loadFileCount() {
        try {
        	loadCSVData(csvPath);
            this.number = datas.size();
            count = datas.size();
            count_it = datas.size() / batchSize;
            System.err.println("data count[" + count + "].");
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    @Override
    public int[][] shuffle() {
        // TODO Auto-generated method stub
        return MathUtils.randomInts(this.number, this.batchSize);
    }

    public int[][] order() {
        // TODO Auto-generated method stub
        return MathUtils.orderInts(this.number, this.batchSize);
    }

    @Override
    public void loadData(int pageIndex, int batchSize, Tensor input, Tensor label) {
        // TODO Auto-generated method stub
    }

    @Override
    public float[] loadData(int index) {
        // TODO Auto-generated method stub
    	return null;
    }

    @Override
    public void loadData(int[] indexs, Tensor input, Tensor label) {
        // TODO Auto-generated method stub
        /**
         * 加载input数据
         *
         */
        if (mean != null) {
        	VideoDataLoader.load(imgDirPath, extName, datas, indexs, batchSize, num_frames, input, true, mean, std);
        } else {
        	VideoDataLoader.load(imgDirPath, extName, datas, indexs, batchSize, num_frames, input, true);
        }
        loadLabels(indexs, label);
        /**
         * copy data to gpu.
         *
         */
        input.hostToDevice();
        label.hostToDevice();
    }

    public void loadData(int[] indexs, Tensor input, Tensor label, Tensor noise) {
        // TODO Auto-generated method stub
        /**
         * 加载input数据
         *
         */
    	if (mean != null) {
        	VideoDataLoader.load(imgDirPath, extName, datas, indexs, batchSize, num_frames, input, true, mean, std);
        } else {
        	VideoDataLoader.load(imgDirPath, extName, datas, indexs, batchSize, num_frames, input, true);
        }
        loadLabels(indexs, label);
        RandomUtils.gaussianRandom(noise, 0, 1);
        /**
         * copy data to gpu.
         *
         */
        input.hostToDevice();
        label.hostToDevice();
    }

    public void loadData(int[] indexs, Tensor input, Tensor label, Tensor noise, String[] labels) {
        // TODO Auto-generated method stub
        /**
         * 加载input数据
         */
    	if (mean != null) {
        	VideoDataLoader.load(imgDirPath, extName, datas, indexs, batchSize, num_frames, input, true, mean, std);
        } else {
        	VideoDataLoader.load(imgDirPath, extName, datas, indexs, batchSize, num_frames, input, true);
        }
        loadLabels(indexs, label, labels);
        RandomUtils.gaussianRandom(noise, 0, 1);
        /**
         * copy data to gpu.
         */
        input.hostToDevice();
        label.hostToDevice();
    }
    
    public void loadData(int[] indexs, Tensor input, Tensor label, String[] labels) {
        // TODO Auto-generated method stub
        /**
         * 加载input数据
         *
         */
    	if (mean != null) {
        	VideoDataLoader.load(imgDirPath, extName, datas, indexs, batchSize, num_frames, input, true, mean, std);
        } else {
        	VideoDataLoader.load(imgDirPath, extName, datas, indexs, batchSize, num_frames, input, true);
        }
        loadLabels(indexs, label, labels);
        /**
         * copy data to gpu.
         *
         */
        input.hostToDevice();
        label.hostToDevice();
    }

    /**
     * 加载数据集
     *
     * @param indexs 数据集数据索引
     * @param input 保存数据的Tensor
     * @param label 保存label的Tensor
     * @param labels labels数组
     * @param eosIdx 每条数据第一个结束符位置
     */
    public void loadData(int[] indexs, Tensor input, Tensor label, String[] labels, Tensor eosIdx) {
        if (mean != null) {
            VideoDataLoader.load(imgDirPath, extName, datas, indexs, batchSize, num_frames, input, true, mean, std);
        } else {
            VideoDataLoader.load(imgDirPath, extName, datas, indexs, batchSize, num_frames, input, true);
        }
        loadLabels(indexs, label, labels, eosIdx);
        input.hostToDevice();
        label.hostToDevice();
        eosIdx.hostToDevice();
    }


    public void loadData_uncond(int[] indexs, Tensor input, Tensor noise) {
        // TODO Auto-generated method stub
        /**
         * 加载input数据
         *
         */
    	if (mean != null) {
        	VideoDataLoader.load(imgDirPath, extName, datas, indexs, batchSize, num_frames, input, true, mean, std);
        } else {
        	VideoDataLoader.load(imgDirPath, extName, datas, indexs, batchSize, num_frames, input, true);
        }
        RandomUtils.gaussianRandom(noise, 0, 1);
        /**
         * copy data to gpu.
         *
         */
        input.hostToDevice();
    }

    public void loadLabel(int[] indexs, Tensor label) {
        // TODO Auto-generated method stub
        loadLabels(indexs, label);
        /**
         * copy data to gpu.
         *
         */
        label.hostToDevice();
    }

    public void addNoise(Tensor a, Tensor b, Tensor input, Tensor noise, CUDAManager cudaManager) {
        if (kernel == null) {
            kernel = new BaseKernel(cudaManager);
        }
        kernel.add_mul(a, b, input, noise, input);
    }

    public void addNoise(Tensor a, Tensor b, Tensor input, Tensor noise, Tensor output, CUDAManager cudaManager) {
        if (kernel == null) {
            kernel = new BaseKernel(cudaManager);
        }
        kernel.add_mul(a, b, input, noise, output);
    }

    public void unMulGrad(Tensor a, Tensor b, Tensor delta, Tensor noise, Tensor diff, CUDAManager cudaManager) {
        if (kernel == null) {
            kernel = new BaseKernel(cudaManager);
        }
        kernel.un_mul_grad(a, b, delta, noise, diff);
    }

    public void unNoise(Tensor a, Tensor b, Tensor input, Tensor noise, CUDAManager cudaManager) {
        if (kernel == null) {
            kernel = new BaseKernel(cudaManager);
        }
        kernel.un_mul(a, b, input, noise, input);
    }

    public void unNoise(Tensor a, Tensor b, Tensor input, Tensor noise, Tensor output, CUDAManager cudaManager) {
        if (kernel == null) {
            kernel = new BaseKernel(cudaManager);
        }
        kernel.un_mul(a, b, input, noise, output);
    }

    public void loadLabels(int[] indexs, Tensor label) {
        for (int i = 0; i < indexs.length; i++) {
            int idx = indexs[i];
            String text = datas.get(idx).get("text").toString();
            int[] ids = tokenizer.encodeInt(text, maxContextLen);
            for (int j = 0; j < maxContextLen; j++) {
                if (j < ids.length) {
                    label.data[i * maxContextLen + j] = ids[j];
                } else {
                    label.data[i * maxContextLen + j] = 0;
                }
            }
        }
    }
    
    public void loadLabel_offset(Tensor label, int index, String labelStr) {
    	int[] ids = tokenizer.encodeInt(labelStr, maxContextLen);
        for (int j = 0; j < maxContextLen; j++) {
            if (j < ids.length) {
                label.data[index * maxContextLen + j] = ids[j];
            } else {
                label.data[index * maxContextLen + j] = 0;
            }
        }
        label.hostToDevice();
    }
    
    public void loadLabels(int[] indexs, Tensor label, String[] labels) {
        for (int i = 0; i < indexs.length; i++) {
            int idx = indexs[i];
            String text = datas.get(idx).get("text").toString();
            labels[i] = text;
            //			System.out.println(text);
            int[] ids = tokenizer.encodeInt(text, maxContextLen);
            for (int j = 0; j < maxContextLen; j++) {
                if (j < ids.length) {
                    label.data[i * maxContextLen + j] = ids[j];
                } else {
                    label.data[i * maxContextLen + j] = 0;
                }
            }
        }
        //		System.out.println(JsonUtils.toJson(label.data));
    }

    public void loadLabels(int[] indexs, Tensor label, String[] labels, Tensor eosIdx) {
        for (int i = 0; i < indexs.length; i++) {
            int idx = indexs[i];
            String text = datas.get(idx).get("text").toString();
            labels[i] = text;
            int[] ids = tokenizer.encodeInt(text, maxContextLen);
            float eos_id = 0;
            for (int j = 0; j < maxContextLen; j++) {
                if (j < ids.length) {
                    label.data[i * maxContextLen + j] = ids[j];
                } else {
                    label.data[i * maxContextLen + j] = 0;
                }
                //获取第一个结束符位置
                if(label.data[i * maxContextLen + j] == tokenizer.eos() && eos_id == 0) {
                    eos_id = j;
                }
            }
            eosIdx.data[i] = eos_id;
        }
    }

    public void loadLabels(int[] indexs, Tensor label, Tensor mask, String[] labels) {
        for (int i = 0; i < indexs.length; i++) {
            int idx = indexs[i];
            String text = datas.get(idx).get("text").toString();
            labels[i] = text;
            //			System.out.println(text);
            int[] ids = tokenizer.encodeInt(text);
            int[] ids_n = new int[ids.length + 2];
            System.arraycopy(ids, 0, ids_n, 1, ids.length);
            ids_n[0] = tokenizer.sos;
            ids_n[ids_n.length - 1] = tokenizer.eos;
            for (int j = 0; j < maxContextLen; j++) {
                if (j < ids_n.length) {
                    label.data[i * maxContextLen + j] = ids_n[j];
                    mask.data[i * maxContextLen + j] = 0;
                } else {
                    label.data[i * maxContextLen + j] = 0;
                    mask.data[i * maxContextLen + j] = -10000.0f;
                }
            }
        }
        //		System.out.println(JsonUtils.toJson(label.data));
        //		System.out.println(JsonUtils.toJson(mask.data));
    }

    public void loadData(int[] indexs, float[] a, float[] b, Tensor input, Tensor noise) {
//        // TODO Auto-generated method stub
//        RandomUtils.gaussianRandom(noise, 0, 1);
//        /**
//         * 加载input数据
//         *
//         */
//        DiffusionImageLoader.load(imgDirPath, extName, idxSet, indexs, input.number, input, a, b, noise.data, true, horizontalFilp);
//        /**
//         * copy data to gpu.
//         *
//         */
//        input.hostToDevice();
    }

    @Override
    public void loadData(int[] indexs, Tensor input) {
        // TODO Auto-generated method stub
        /**
         * 加载input数据
         */
    	if (mean != null) {
        	VideoDataLoader.load(imgDirPath, extName, datas, indexs, batchSize, num_frames, input, true, mean, std);
        } else {
        	VideoDataLoader.load(imgDirPath, extName, datas, indexs, batchSize, num_frames, input, true);
        }
        /**
         * copy data to gpu.
         *
         */
        input.hostToDevice();
    }
    
    public void normalization(Tensor input) {
        for (int i = 0; i < input.dataLength; i++) {
            int f = (i / input.width / input.height) % input.channel;
            input.data[i] = (input.data[i] - mean[f]) / std[f];
        }
    }

    public int[] getIndexsByAsc(int pageIndex, int batchSize) {
        int start = pageIndex * batchSize;
        int end = pageIndex * batchSize + batchSize;
        if (end > number) {
            start = start - (end - number);
        }
        int[] indexs = new int[batchSize];
        for (int i = 0; i < batchSize; i++) {
            indexs[i] = start + i;
        }
        return indexs;
    }

    @Override
    public Tensor initLabelTensor() {
        // TODO Auto-generated method stub
        return null;
    }
}
