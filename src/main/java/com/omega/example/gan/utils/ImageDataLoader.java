package com.omega.example.gan.utils;

import com.omega.common.config.Tensor;
import com.omega.common.utils.MathUtils;
import com.omega.example.yolo.data.BaseDataLoader;
import com.omega.example.yolo.data.FileDataLoader;
import com.omega.example.yolo.data.ImageLoader;
import com.omega.example.yolo.utils.YoloImageUtils;

import java.io.File;

/**
 * DetectionDataLoader
 *
 * @author Administrator
 */
public class ImageDataLoader extends BaseDataLoader {
    public static float[] mean = new float[]{0.491f, 0.482f, 0.446f};
    public static float[] std = new float[]{0.247f, 0.243f, 0.261f};
    public String[] idxSet;
    public boolean normalization = false;
    private String imgDirPath;
    private boolean dataEnhance = false;
    private int img_w;
    private int img_h;
    private String extName;

    public ImageDataLoader(String imgDirPath, int img_w, int img_h, int batchSize) {
        this.imgDirPath = imgDirPath;
        this.img_w = img_w;
        this.img_h = img_h;
        this.batchSize = batchSize;
        init();
    }

    public ImageDataLoader(String imgDirPath, int img_w, int img_h, int batchSize, boolean normalization, float[] mean, float[] std) {
        this.imgDirPath = imgDirPath;
        this.img_w = img_w;
        this.img_h = img_h;
        this.batchSize = batchSize;
        this.normalization = normalization;
        this.mean = mean;
        this.std = std;
        init();
    }

    public void init() {
        loadFileCount();
    }

    public void loadFileCount() {
        try {
            File file = new File(imgDirPath);
            if (file.exists() && file.isDirectory()) {
                String[] filenames = file.list();
                this.number = filenames.length;
                this.idxSet = new String[number];
                this.extName = filenames[0].split("\\.")[1];
                for (int i = 0; i < number; i++) {
                    this.idxSet[i] = filenames[i];
                }
            }
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void showImg(String outputPath, int[] indexs, Tensor input) {
    }

    @Override
    public int[][] shuffle() {
        // TODO Auto-generated method stub
        return MathUtils.randomInts(this.number, this.batchSize);
    }

    @Override
    public void loadData(int pageIndex, int batchSize, Tensor input, Tensor label) {
        // TODO Auto-generated method stub
        int[] indexs = getIndexsByAsc(pageIndex, batchSize);
        /**
         * 加载input数据

         */
        FileDataLoader.load(imgDirPath, extName, idxSet, indexs, input.number, input, false);
        if (normalization) {
            this.normalization(input);
        }
        /**
         * copy data to gpu.

         */
        input.hostToDevice();
    }

    public void loadData(int pageIndex, int batchSize, Tensor input) {
        // TODO Auto-generated method stub
        int[] indexs = getIndexsByAsc(pageIndex, batchSize);
        /**
         * 加载input数据

         */
        FileDataLoader.load(imgDirPath, extName, idxSet, indexs, input.number, input, false);
        if (normalization) {
            this.normalization(input);
        }
        /**
         * copy data to gpu.

         */
        input.hostToDevice();
    }

    @Override
    public float[] loadData(int index) {
        // TODO Auto-generated method stub
        String filePath = imgDirPath + "/" + idxSet[index];
        if (!filePath.contains(".")) {
            filePath += ".jpg";
        }
        return ImageLoader.resized(filePath, this.img_w, this.img_h);
    }

    @Override
    public void loadData(int[] indexs, Tensor input, Tensor label) {
        // TODO Auto-generated method stub
        /**
         * 加载input数据

         */
        FileDataLoader.load(imgDirPath, extName, idxSet, indexs, input.number, input, dataEnhance);
        if (normalization) {
            this.normalization(input);
        }
        /**
         * copy data to gpu.

         */
        input.hostToDevice();
    }

    public void normalization(Tensor input) {
        for (int i = 0; i < input.dataLength; i++) {
            int f = (i / input.width / input.height) % input.channel;
            input.data[i] = (input.data[i] - mean[f]) / std[f];
        }
    }

    public void fileDataLoader(int[] indexs, Tensor input) {
        for (int i = 0; i < batchSize; i++) {
            String filePath = imgDirPath + "/" + idxSet[indexs[i]] + "." + extName;
            YoloImageUtils.loadImgDataToTensor(filePath, input, i);
            //			System.out.println(filePath);
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

    @Override
    public void loadData(int[] indexs, Tensor input) {
        // TODO Auto-generated method stub
        /**
         * 加载input数据

         */
        FileDataLoader.load(imgDirPath, extName, idxSet, indexs, input.number, input, dataEnhance);
        if (normalization) {
            this.normalization(input);
        }
        /**
         * copy data to gpu.

         */
        input.hostToDevice();
    }
}

