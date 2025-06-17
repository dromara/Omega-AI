package com.omega.example.diffusion.utils;

import com.omega.common.utils.ImageUtils;
import com.omega.common.utils.JsonUtils;
import com.omega.engine.tensor.Tensor;
import com.omega.example.yolo.utils.YoloImageUtils;

import java.io.File;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

public class GenImageUtils {
    public static float[] mean = new float[]{0.5f, 0.5f, 0.5f};
    public static float[] std = new float[]{0.5f, 0.5f, 0.5f};
    public static int gif_index = 0;

    public static void showGif(String outputPath, Tensor input, int[][][] rgbs, int it, int count, int ow, int oh, int ogw, int ogh) {
        ImageUtils utils = new ImageUtils();
        int gifw = ogw;
        int gifh = ogh;
        int imgw = ow;
        int imgh = oh;
        int gifWidth = gifw * imgw;
        int gifHeight = gifh * imgh;
        float[] gif = new float[imgw * imgh * 64];
        for (int b = 0; b < gifw * gifh; b++) {
            int gh = b / gifw;
            int gw = b % gifh;
            float[] once = input.getByNumber(b);
            for (int i = 0; i < imgh; i++) {
                int startH = gh * imgh + i;
                for (int j = 0; j < imgw; j++) {
                    int startW = gw * imgw + j;
                    gif[startH * imgw * gifw + startW] = once[i * imgw + j];
                }
            }
        }
        int[][] rgb = ImageUtils.color2rgb2(gif, input.getShape()[1], gifHeight, gifWidth, true, mean, std);
        rgbs[gif_index] = rgb;
        if (gif_index == count - 1) {
            utils.createRGBGIF(outputPath + it + ".gif", "gif", rgbs, gifWidth, gifHeight);
            gif_index = 0;
        } else {
            gif_index++;
        }
    }

    public static String[] loadFileCount(String imgDirPath) {
        try {
            File file = new File(imgDirPath);
            if (file.exists() && file.isDirectory()) {
                String[] filenames = file.list();
                int number = filenames.length;
                String[] idxSet = new String[number];
                //				String extName = filenames[0].split("\\.")[1];
                for (int i = 0; i < number; i++) {
                    idxSet[i] = filenames[i];
                }
                return idxSet;
            }
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
        return null;
    }

    public static void createGIF(String inputPath, String outPath, int imgw, int imgh, int gifw, int gifh) {
        try {
            ImageUtils utils = new ImageUtils();
            int gifWidth = gifw * imgw;
            int gifHeight = gifh * imgh;
            int oCount = gifw * gifh;
            float[] gif = new float[imgw * imgh * oCount * 3];
            String[] filenames = loadFileCount(inputPath);
            List<String> names = Arrays.asList(filenames);
            Collections.sort(names, new Comparator<String>() {
                @Override
                public int compare(String o1, String o2) {
                    // TODO Auto-generated method stub
                    int head1 = Integer.parseInt(o1.split("_")[0]);
                    int head2 = Integer.parseInt(o2.split("_")[0]);
                    int body1 = Integer.parseInt(o1.split("_")[1]);
                    int body2 = Integer.parseInt(o2.split("_")[1]);
                    if (head1 > head2) {
                        return 1;
                    } else if (head1 < head2) {
                        return -1;
                    } else {
                        if (body1 > body2) {
                            return 1;
                        } else if (body1 < body2) {
                            return -1;
                        } else {
                            return 0;
                        }
                    }
                }
            });
            filenames = names.toArray(filenames);
            System.out.println(JsonUtils.toJson(filenames));
            int count = filenames.length;
            int batchCount = count / oCount;
            int[][][] rgbs = new int[batchCount][gifWidth][gifHeight];
            for (int fi = 0; fi < batchCount; fi++) {
                for (int b = 0; b < gifw * gifh; b++) {
                    int gh = b / gifw;
                    int gw = b % gifh;
                    String filePath = inputPath + "\\" + filenames[fi * gifw * gifh + b];
                    float[] once = YoloImageUtils.loadImgDataToArray(filePath, false);
                    System.out.println(filePath);
                    for (int c = 0; c < 3; c++) {
                        int c_idx = c * oCount * imgh * imgw;
                        int c_o_idx = c * imgh * imgw;
                        for (int i = 0; i < imgh; i++) {
                            int startH = gh * imgh + i;
                            for (int j = 0; j < imgw; j++) {
                                int startW = gw * imgw + j;
                                gif[c_idx + startH * imgw * gifw + startW] = once[c_o_idx + i * imgw + j];
                            }
                        }
                    }
                }
                int[][] rgb = ImageUtils.color2rgb2(gif, 3, gifHeight, gifWidth, false);
                rgbs[fi] = rgb;
            }
            utils.createRGBGIF(outPath + "diffusion_anime" + ".gif", "gif", rgbs, gifWidth, gifHeight);
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public static void main(String args[]) {
        String in = "H:\\voc\\gan_anime\\duffsion_test";
        String out = "H:\\voc\\gan_anime\\";
        int imgw = 96;
        int imgh = 96;
        int gifw = 4;
        int gifh = 4;
        createGIF(in, out, imgw, imgh, gifw, gifh);
    }
}

