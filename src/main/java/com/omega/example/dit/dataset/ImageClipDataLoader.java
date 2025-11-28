package com.omega.example.dit.dataset;

import java.io.File;
import java.io.RandomAccessFile;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;

import com.omega.common.utils.MathUtils;
import com.omega.common.utils.RandomUtils;
import com.omega.engine.nn.network.utils.ModelUtils;
import com.omega.engine.tensor.Tensor;
import com.omega.example.diffusion.utils.DiffusionImageLoader;
import com.omega.example.transformer.utils.LagJsonReader;
import com.omega.example.transformer.utils.bpe.BPETokenizerEN;
import com.omega.example.unet.utils.SegImageLoader;
import com.omega.example.yolo.data.BaseDataLoader;
import com.omega.example.yolo.data.ImageLoader;
import com.omega.example.yolo.utils.YoloImageUtils;

/**
 * ImageClipDataLoader
 *
 * @author Administrator
 */
public class ImageClipDataLoader extends BaseDataLoader {
	
    public int img_w;
    public int img_h;
    public boolean normalization = true;
    public float[] mean;
    public float[] std;
    public int count;
    public int count_it;
    private String labelPath;
    private String imgDirPath;
    private String clipDataPath;
    private String extName = ".png";
    public int maxContextLen;
    public int yDim;
    private boolean horizontalFilp;
    private List<Map<String, Object>> datas;
    private String[] idxSet;

    private CompletableFuture<Boolean> cf;
    private RandomAccessFile clipFile;
    private float[] clip_cache = null;
    
    private int byteUnit = 4;

    public ImageClipDataLoader(String labelPath, String imgDirPath, String clipDataPath, String extName, int img_w, int img_h, int maxContextLen, int yDim, int batchSize, boolean horizontalFilp, float[] mean, float[] std) {
        this.horizontalFilp = horizontalFilp;
        this.imgDirPath = imgDirPath;
        this.labelPath = labelPath;
        this.clipDataPath = clipDataPath;
        this.maxContextLen = maxContextLen;
        this.yDim = yDim;
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

    public void loadFileCount() {
        try {
            File file = new File(imgDirPath);
            if (file.exists()) {
                datas = LagJsonReader.readJsonDataSamll(labelPath);
                idxSet = new String[datas.size()];
                for (int i = 0; i < datas.size(); i++) {
                    idxSet[i] = datas.get(i).get("id").toString() + extName;
                }
            }
            clipFile = new RandomAccessFile(clipDataPath, "r");
            clip_cache = new float[maxContextLen * yDim];
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

    public float[] loadLabelData(long idx) {
        try {

        	long cfi = idx * maxContextLen * yDim * byteUnit;
        	if(cfi < clipFile.length()) {
                clipFile.seek(cfi);
                ModelUtils.readFloatArray(clipFile, clip_cache);
        	}else {
        		System.err.println("dataset index["+idx+"] is out.");
        	}

        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
        return clip_cache;
    }
    
    public void loadData(int pageIndex, int batchSize, Tensor input) {
        // TODO Auto-generated method stub
        int[] indexs = getIndexsByAsc(pageIndex, batchSize);
        /**
         * 加载input数据
         *
         */
        SegImageLoader.load(imgDirPath, extName, idxSet, indexs, input.number, input, false, false);
        if (normalization) {
            this.normalization(input);
        }
        /**
         * copy data to gpu.
         *
         */
        input.hostToDevice();
    }

    @Override
    public float[] loadData(int index) {
        // TODO Auto-generated method stub
        String filePath = imgDirPath + "/" + idxSet[index];
        if (!filePath.contains(".")) {
            filePath += ".png";
        }
        return ImageLoader.resized(filePath, this.img_w, this.img_h);
    }

    @Override
    public void loadData(int[] indexs, Tensor input, Tensor label) {
        // TODO Auto-generated method stub
    	try {
            if (cf != null) {
                boolean success = cf.get();
                if(success){
                	cf = null;
                	loadDataToGPU(indexs, input, label);
                }
            } else {
                cf = loadAsyncData(indexs, input, label);
                boolean success = cf.get();
                if(success){
                	cf = null;
                	loadDataToGPU(indexs, input, label);
                }
            }
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void loadDataToGPU(int[] index,Tensor input, Tensor label) {
    	input.hostToDevice();
        label.hostToDevice();
        cf = loadAsyncData(index, input, label);
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
    
    public CompletableFuture<Boolean> loadAsyncData(int[] indexs,Tensor input, Tensor label) {
        CompletableFuture<Boolean> cf = CompletableFuture.supplyAsync(() -> {
            try {
            	/**
                 * 加载input数据
                 */
                if (mean != null) {
                    SegImageLoader.load(imgDirPath, extName, idxSet, indexs, input.number, input, false, true, mean, std);
                } else {
                    SegImageLoader.load(imgDirPath, extName, idxSet, indexs, input.number, input, false, true);
                }
                loadLabels(indexs, label);
            } catch (Exception e) {
                // TODO: handle exception
                e.printStackTrace();
            }
            return true;
        });
        return cf;
    }
    
    public void loadLabels(int[] indexs, Tensor label) {
        for (int i = 0; i < indexs.length; i++) {
            int idx = indexs[i];
            float[] data = loadLabelData(idx);
            for (int j = 0; j < maxContextLen; j++) {
            	label.data[i * maxContextLen + j] = data[j];
            }
        }
    }
    
    public void loadData(int[] indexs, float[] a, float[] b, Tensor input, Tensor noise) {
        // TODO Auto-generated method stub
        RandomUtils.gaussianRandom(noise, 0, 1);
        /**
         * 加载input数据
         *
         */
        DiffusionImageLoader.load(imgDirPath, extName, idxSet, indexs, input.number, input, a, b, noise.data, true, horizontalFilp);
        /**
         * copy data to gpu.
         *
         */
        input.hostToDevice();
    }

    @Override
    public void loadData(int[] indexs, Tensor input) {
        // TODO Auto-generated method stub
        /**
         * 加载input数据
         *
         */
        if (mean != null) {
            SegImageLoader.load(imgDirPath, extName, idxSet, indexs, input.number, input, false, true, mean, std);
        } else {
            SegImageLoader.load(imgDirPath, extName, idxSet, indexs, input.number, input, false, true);
        }
        /**
         * copy data to gpu.
         *
         */
        input.hostToDevice();
    }

    public void loadData(String filePath, Tensor input) {
        // TODO Auto-generated method stub
        /**
         * 加载input数据
         *
         */
        float[] data = YoloImageUtils.loadImgDataToArray(filePath, true);
        System.arraycopy(data, 0, input.data, 0, input.channel * input.height * input.width);
        /**
         * copy data to gpu.
         *
         */
        input.hostToDevice();
    }

    public void loadLabelData(String filePath, Tensor label) {
        // TODO Auto-generated method stub
        /**
         * 加载input数据
         *
         */
        float[] data = YoloImageUtils.loadImgDataToGrayArray(filePath, true);
        System.arraycopy(data, 0, label.data, 0, label.channel * label.height * label.width);
        /**
         * copy data to gpu.
         *
         */
        label.hostToDevice();
    }

    public void loadLabel_offset(BPETokenizerEN tokenizer, Tensor label, int index, String labelStr) {
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
