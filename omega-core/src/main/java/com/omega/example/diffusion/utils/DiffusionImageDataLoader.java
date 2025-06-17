package com.omega.example.diffusion.utils;

import com.omega.common.utils.MathUtils;
import com.omega.common.utils.RandomUtils;
import com.omega.engine.tensor.Tensor;
import com.omega.example.unet.utils.SegImageLoader;
import com.omega.example.yolo.data.BaseDataLoader;
import com.omega.example.yolo.data.ImageLoader;
import com.omega.example.yolo.utils.YoloImageUtils;

import java.io.File;
import java.util.Arrays;
import java.util.Comparator;

/**
 * DetectionDataLoader
 *
 * @author Administrator
 */
public class DiffusionImageDataLoader extends BaseDataLoader {
    public String[] idxSet;
    public boolean normalization = true;
    public float[] mean;
    public float[] std;
    public int count;
    public int count_it;
    public boolean sort = false;
    private String imgDirPath;
    private int img_w;
    private int img_h;
    private String extName;
    private boolean horizontalFilp;
    
    private int maxNumber = 0;

    public DiffusionImageDataLoader(String imgDirPath, int img_w, int img_h, int batchSize, boolean horizontalFilp) {
        this.horizontalFilp = horizontalFilp;
        this.imgDirPath = imgDirPath;
        this.img_w = img_w;
        this.img_h = img_h;
        this.batchSize = batchSize;
        init();
    }

    public DiffusionImageDataLoader(String imgDirPath, int img_w, int img_h, int batchSize, boolean horizontalFilp, float[] mean, float[] std) {
        this.horizontalFilp = horizontalFilp;
        this.imgDirPath = imgDirPath;
        this.img_w = img_w;
        this.img_h = img_h;
        this.batchSize = batchSize;
        this.mean = mean;
        this.std = std;
        init();
    }

    public DiffusionImageDataLoader(String imgDirPath, int img_w, int img_h, int batchSize, boolean horizontalFilp, boolean sort, float[] mean, float[] std) {
        this.horizontalFilp = horizontalFilp;
        this.sort = sort;
        this.imgDirPath = imgDirPath;
        this.img_w = img_w;
        this.img_h = img_h;
        this.batchSize = batchSize;
        this.mean = mean;
        this.std = std;
        init();
    }
    
    public DiffusionImageDataLoader(String imgDirPath, int img_w, int img_h, int batchSize, boolean horizontalFilp, boolean sort, float[] mean, float[] std,int maxNumber) {
        this.horizontalFilp = horizontalFilp;
        this.sort = sort;
        this.imgDirPath = imgDirPath;
        this.img_w = img_w;
        this.img_h = img_h;
        this.batchSize = batchSize;
        this.mean = mean;
        this.std = std;
        this.maxNumber = maxNumber;
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
                this.number = maxNumber > 0 ? maxNumber: filenames.length;
                this.idxSet = new String[number];
                this.extName = filenames[0].split("\\.")[1];
                for (int i = 0; i < number; i++) {
                    this.idxSet[i] = filenames[i];
                }
                if (sort) {
                    Arrays.sort(idxSet, new Comparator<String>() {
                        @Override
                        public int compare(String o1, String o2) {
                            // TODO Auto-generated method stub
                            int r = 0;
                            int o1i = Integer.parseInt(o1.split("\\.")[0]);
                            int o2i = Integer.parseInt(o2.split("\\.")[0]);
                            if (o1i == o2i) {
                                r = 0;
                            } else if (o1i > o2i) {
                                r = 1;
                            } else {
                                r = -1;
                            }
                            return r;
                        }
                    });
                }
//                System.err.println(JsonUtils.toJson(this.idxSet));
            }
            count = this.idxSet.length;
            count_it = this.idxSet.length / batchSize;
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

    public int[][] order() {
        // TODO Auto-generated method stub
        return MathUtils.orderInts(this.number, this.batchSize);
    }

    @Override
    public void loadData(int pageIndex, int batchSize, Tensor input, Tensor label) {
        // TODO Auto-generated method stub
    }

    public void loadData(int pageIndex, int batchSize, Tensor input) {
        // TODO Auto-generated method stub
        int[] indexs = getIndexsByAsc(pageIndex, batchSize);
        /**
         * 加载input数据

         */
        SegImageLoader.load(imgDirPath, extName, idxSet, indexs, input.getShape()[0], input, false, false);
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
        SegImageLoader.load(imgDirPath, extName, idxSet, indexs, input.getShape()[0], input, false, true);
        /**
         * copy data to gpu.

         */
        input.hostToDevice();
    }

    public void loadData(int[] indexs, float[] a, float[] b, Tensor input, Tensor noise) {
        // TODO Auto-generated method stub
        RandomUtils.gaussianRandom(noise, 0, 1);
        //		RandomUtils.gaussianRandom2(noise, 0, 1);
        //		noise.setData(MatrixUtils.order(noise.dataLength, 0.001f, 0.001f));
        /**
         * 加载input数据

         */
        DiffusionImageLoader.load(imgDirPath, extName, idxSet, indexs, input.getShape()[0], input, a, b, noise.getData(), true, horizontalFilp);
        /**
         * copy data to gpu.

         */
        input.hostToDevice();
    }

    @Override
    public void loadData(int[] indexs, Tensor input) {
        // TODO Auto-generated method stub
        /**
         * 加载input数据

         */
        if (mean != null) {
            SegImageLoader.load(imgDirPath, extName, idxSet, indexs, input.getShape()[0], input, false, true, mean, std);
        } else {
            SegImageLoader.load(imgDirPath, extName, idxSet, indexs, input.getShape()[0], input, false, true);
        }
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
        System.arraycopy(data, 0, input.getData(), 0, input.getShape()[1] * input.getShape()[2] * input.getShape()[3]);
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
        System.arraycopy(data, 0, label.getData(), 0, label.getShape()[1] * label.getShape()[2] * label.getShape()[3]);
        /**
         * copy data to gpu.

         */
        label.hostToDevice();
    }

    public void normalization(Tensor input) {
        for (int i = 0; i < input.getDataLength(); i++) {
            int f = (i / input.getShape()[3] / input.getShape()[2]) % input.getShape()[1];
            input.getData()[i] = (input.getData()[i] - mean[f]) / std[f];
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

