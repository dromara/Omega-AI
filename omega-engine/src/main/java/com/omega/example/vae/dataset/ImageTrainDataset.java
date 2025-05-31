package com.omega.example.vae.dataset;

import com.omega.engine.tensor.Tensor;
import com.omega.example.yolo.data.ImageLoader;
import jcuda.runtime.JCuda;

import java.io.File;
import java.io.IOException;
import java.util.*;
import java.util.concurrent.CompletableFuture;

public class ImageTrainDataset {
    public final static float[] mean = new float[]{0.48145466f, 0.4578275f, 0.40821073f};
    public final static float[] std = new float[]{0.26862954f, 0.26130258f, 0.27577711f};
    public boolean shuffle = false;
    public int number = 0;
    public int channel = 3;
    public int imageSize = 224;
    public int onceImgSize = 0;
    public int count_it = 0;
    public Tensor testInput;
    public String[] idxSet;
    private int batchSize = 1;
    private String imagePath;
    private List<Map<String, Object>> dataList;
    private CompletableFuture<Boolean> cf;
    private int current = 0;
    private float[] tmpImageInput;
    private boolean sort = false;

    public ImageTrainDataset(String imagePath, int imageSize, int batchSize, boolean shuffle) {
        this.shuffle = shuffle;
        this.imagePath = imagePath;
        this.imageSize = imageSize;
        this.onceImgSize = 3 * this.imageSize * this.imageSize;
        this.batchSize = batchSize;
        this.number = loadCount();
        this.count_it = this.number / batchSize;
        System.out.println("dataCount:" + this.number);
        System.out.println("count_it:" + this.count_it);
    }

    public int loadCount() {
        try {
            File file = new File(imagePath);
            if (file.exists() && file.isDirectory()) {
                String[] filenames = file.list();
                this.number = filenames.length;
                this.idxSet = new String[number];
                //				this.extName = filenames[0].split("\\.")[1];
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
            }
            this.count_it = this.idxSet.length / batchSize;
            System.err.println("data count[" + this.idxSet.length + "].");
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
        return this.idxSet.length;
    }

    public void initReader() {
        current = 1;
    }

    public void loadData(Tensor imageInput) {
        try {
            //			System.out.println(it);
            if (tmpImageInput == null) {
                tmpImageInput = new float[imageInput.dataLength];
            }
            if (cf != null) {
                boolean success = cf.get();
                if (success) {
                    imageInput.hostToDevice(tmpImageInput);
                    JCuda.cudaDeviceSynchronize();
                }
                cf = loadAsyncData(tmpImageInput);
            } else {
                cf = loadAsyncData(tmpImageInput);
            }
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void readQA(int b) throws IOException {
        String filename = idxSet[current];
        if (current >= number - 1) {
            if (shuffle) {
                Collections.shuffle(dataList);
            }
            current = 0;
        }
        String imgPath = imagePath + filename;
        imageProcessor(imgPath, b);
        current++;
    }

    public void imageProcessor(String imgPath, int b) {
        ImageLoader.loadImage(tmpImageInput, onceImgSize, b, imgPath, imageSize, imageSize, mean, std, true);
    }

    public CompletableFuture<Boolean> loadAsyncData(float[] imageData) {
        CompletableFuture<Boolean> cf = CompletableFuture.supplyAsync(() -> {
            try {
                for (int b = 0; b < batchSize; b++) {
                    readQA(b);
                }
            } catch (Exception e) {
                // TODO: handle exception
                e.printStackTrace();
            }
            return true;
        });
        return cf;
    }
}

