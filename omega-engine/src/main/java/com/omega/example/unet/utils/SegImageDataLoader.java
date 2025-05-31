package com.omega.example.unet.utils;

import com.omega.common.utils.MathUtils;
import com.omega.engine.tensor.Tensor;
import com.omega.example.yolo.data.BaseDataLoader;
import com.omega.example.yolo.data.ImageLoader;
import com.omega.example.yolo.utils.YoloImageUtils;

import java.io.File;

/**
 * DetectionDataLoader
 *
 * @author Administrator
 */
public class SegImageDataLoader extends BaseDataLoader {
    public static float[] mean = new float[]{0.5f, 0.5f, 0.5f};
    public static float[] std = new float[]{0.5f, 0.5f, 0.5f};
    public String[] idxSet;
    public boolean normalization = true;
    private String imgDirPath;
    private String maskDirPath;
    private int img_w;
    private int img_h;
    private String extName;

    public SegImageDataLoader(String imgDirPath, String maskDirPath, int img_w, int img_h, int batchSize) {
        this.imgDirPath = imgDirPath;
        this.maskDirPath = maskDirPath;
        this.img_w = img_w;
        this.img_h = img_h;
        this.batchSize = batchSize;
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
            System.err.println("data count[" + this.idxSet.length + "].");
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

    @Override
    public void loadData(int pageIndex, int batchSize, Tensor input, Tensor label) {
        // TODO Auto-generated method stub
        int[] indexs = getIndexsByAsc(pageIndex, batchSize);
        /**
         * 加载input数据

         */
        SegImageLoader.load(imgDirPath, extName, idxSet, indexs, input.number, input, false, false);
        SegImageLoader.load(maskDirPath, extName, idxSet, indexs, label.number, label, true, false);
        if (normalization) {
            this.normalization(input);
        }
        /**
         * copy data to gpu.

         */
        input.hostToDevice();
        label.hostToDevice();
    }

    public void loadData(int pageIndex, int batchSize, Tensor input) {
        // TODO Auto-generated method stub
        int[] indexs = getIndexsByAsc(pageIndex, batchSize);
        /**
         * 加载input数据

         */
        SegImageLoader.load(imgDirPath, extName, idxSet, indexs, input.number, input, false, false);
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
        SegImageLoader.load(imgDirPath, extName, idxSet, indexs, input.number, input, false, true);
        SegGrayImageLoader.load(maskDirPath, extName, idxSet, indexs, input.number, label, true, true);
        /**
         * copy data to gpu.

         */
        input.hostToDevice();
        label.hostToDevice();
    }

    @Override
    public void loadData(int[] indexs, Tensor input) {
        // TODO Auto-generated method stub
        /**
         * 加载input数据

         */
        SegImageLoader.load(imgDirPath, extName, idxSet, indexs, input.number, input, false, true);
        /**
         * copy data to gpu.

         */
        input.hostToDevice();
    }

    public void loadData(String filePath, Tensor input) {
        // TODO Auto-generated method stub
        /**
         * 加载input数据

         */
        float[] data = YoloImageUtils.loadImgDataToArray(filePath, true);
        System.arraycopy(data, 0, input.data, 0, input.channel * input.height * input.width);
        /**
         * copy data to gpu.

         */
        input.hostToDevice();
    }

    public void loadLabelData(String filePath, Tensor label) {
        // TODO Auto-generated method stub
        /**
         * 加载input数据

         */
        float[] data = YoloImageUtils.loadImgDataToGrayArray(filePath, true);
        System.arraycopy(data, 0, label.data, 0, label.channel * label.height * label.width);
        /**
         * copy data to gpu.

         */
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

