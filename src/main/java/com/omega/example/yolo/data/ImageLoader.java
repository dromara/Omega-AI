package com.omega.example.yolo.data;

import com.omega.common.config.Tensor;
import com.omega.common.task.ForkJobEngine;
import com.omega.common.utils.*;
import com.omega.example.yolo.utils.OMImage;
import com.omega.example.yolo.utils.YoloImageUtils;

import java.io.*;
import java.util.*;

/**
 * 图片加载组器
 *
 * @author Administrator
 */
public class ImageLoader {
    public static void main(String[] args) {
        //		String imagePath = "H:\\voc\\mask\\data\\face1.jpg";
        //
        //		String outputPath = "H:\\voc\\mask\\data\\";
        //
        ////		float[] labelBoxs = new float[] {1,42,16,83,43,1,129,88,152,108,1,208,94,243,118};
        //
        //		float[] labelBoxs = new float[] {0,81,163,268,309};
        //
        //		int number = 1;
        //		int channel = 3;
        //		int height = 416;
        //		int width = 416;
        //
        //		int boxes = 90;
        //		int classes = 2;
        //		float jitter = 0.1f;
        //		float hue = 0.1f;
        //		float saturation = 1.5f;
        //		float exposure = 1.5f;
        //
        //		Tensor x = new Tensor(number, channel, height, width);
        //		Tensor y = new Tensor(number, 1, 1, boxes * 5);
        //
        //		OMImage orig = loadImage(imagePath);
        //
        //		float[] labelXYWH = formatXYWH(labelBoxs, orig.getWidth(), orig.getHeight());
        //
        //		loadVailDataDetection(x, y, 0, orig, labelXYWH, width, height, boxes, classes);
        //
        ////		loadDataDetection(x, y, 0, orig, labelXYWH, width, height, boxes, classes, jitter, hue, saturation, exposure);
        //
        ////		System.out.println(JsonUtils.toJson(x));
        //
        //		System.out.println(JsonUtils.toJson(y));
        //
        //		showImg(x, y, classes, outputPath);
        //		formatImage();
        resizeFormatImage();
        //		testFormatImage();
    }

    public static void resizeFormatImage() {
        try {
            String imgDirPath = "H:\\voc\\yz\\seal\\test";
            String outputDirPath = "H:\\voc\\yz\\seal\\resized\\test\\";
            int width = 416;
            int height = 416;
            File file = new File(imgDirPath);
            if (file.exists() && file.isDirectory()) {
                for (File img : file.listFiles()) {
                    resizeImage(img, width, height, outputDirPath + img.getName());
                }
            }
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public static void testFormatImage() {
        //		String imagePath = "H:\\voc\\helmet\\resized\\train\\00000.jpg";
        String imagePath = "H:\\voc\\test\\JPEGImages\\000390.jpg";
        String outputPath = "H:\\voc\\test\\";
        //		float[] labelBoxs = new float[] {1,42,16,83,43,1,129,88,152,108,1,208,94,243,118};
        //		float[] labelBoxs = new float[] {1, 56, 159, 115, 168, 1, 186, 159, 243, 169, 1, 0, 164, 45, 174, 1, 222, 165, 276, 174,
        //				1, 332, 154, 389, 165, 1, 155, 159, 211, 167, 1, 23, 173, 74, 183, 1, 353, 147, 416, 163, 1, 291, 142, 353, 153,
        //				1, 117, 155, 174, 164, 1, 261, 157, 323, 167, 1, 88, 170, 145, 179, 1, 320, 146, 369, 155};
        float[] labelBoxs = new float[]{};
        int number = 1;
        int channel = 3;
        int height = 416;
        int width = 416;
        int boxes = 90;
        int classes = 20;
        float jitter = 0.1f;
        float hue = 0.1f;
        float saturation = 1.5f;
        float exposure = 1.5f;
        Tensor x = new Tensor(number, channel, height, width);
        Tensor y = new Tensor(number, 1, 1, boxes * 5);
        OMImage orig = loadImage(imagePath);
        float[] labelXYWH = formatXYWH(labelBoxs, orig.getWidth(), orig.getHeight());
        loadVailDataDetection(x, y, 0, orig, labelXYWH, width, height, boxes, classes);
        //		loadDataDetection(x, y, 0, orig, labelXYWH, width, height, boxes, classes, jitter, hue, saturation, exposure);
        //		System.out.println(JsonUtils.toJson(x));
        System.out.println(JsonUtils.toJson(y));
        showImg(x, y, classes, outputPath);
    }

    public static void formatImage() {
        try {
            //			String imgDirPath = "H:\\voc\\sm\\VOC\\JPEGImages";
            //			String labelPath = "H:\\voc\\sm\\VOC\\bbox.txt";
            //			String outputDirPath = "H:\\voc\\sm\\resized\\imgs\\";
            //			String labelTXTPath = "H:\\voc\\sm\\resized\\rlabels.txt";
            String imgDirPath = "H:\\voc\\yz\\seal\\Images";
            String labelPath = "H:\\voc\\yz\\seal\\bbox.txt";
            String outputDirPath = "H:\\voc\\yz\\seal\\resized\\imgs\\";
            String labelTXTPath = "H:\\voc\\yz\\seal\\resized\\rlabels.txt";
            int width = 416;
            int height = 416;
            Map<String, float[]> labelMap = loadLabelDataForTXT(labelPath);
            Map<String, float[]> rlabelMap = new HashMap<String, float[]>();
            String[] names = new String[labelMap.size()];
            File file = new File(imgDirPath);
            if (file.exists() && file.isDirectory()) {
                int i = 0;
                for (File img : file.listFiles()) {
                    String key = img.getName().split("\\.")[0];
                    float[] rlabel = resizeImage(img, labelMap.get(key), width, height, outputDirPath + img.getName());
                    rlabelMap.put(key, rlabel);
                    names[i] = key;
                    i++;
                }
            }
            File txt = new File(labelTXTPath);
            if (!txt.exists()) {
                txt.createNewFile(); // 创建新文件,有同名的文件的话直接覆盖
            }
            try (FileOutputStream fos = new FileOutputStream(txt);) {
                for (String name : names) {
                    String text = name;
                    for (float val : (float[]) rlabelMap.get(name)) {
                        text += " " + Math.round(val);
                    }
                    text += "\n";
                    //					System.out.println(text);
                    fos.write(text.getBytes());
                }
                fos.flush();
            } catch (Exception e) {
                e.printStackTrace();
            }
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public static Map<String, float[]> loadLabelDataForTXT(String labelPath) {
        Map<String, float[]> orgLabelData = new HashMap<String, float[]>();
        try (FileInputStream fin = new FileInputStream(labelPath); InputStreamReader reader = new InputStreamReader(fin); BufferedReader buffReader = new BufferedReader(reader);) {
            String strTmp = "";
            while ((strTmp = buffReader.readLine()) != null) {
                String[] list = strTmp.split(" ");
                List<float[]> once = new ArrayList<float[]>();
                int page = (list.length - 1) / 5;
                for (int i = 0; i < page; i++) {
                    float[] bbox = new float[5];
                    for (int j = 0; j < 5; j++) {
                        bbox[j] = Float.parseFloat(list[i * 5 + j + 1]);
                    }
                    once.add(bbox);
                }
                /**
                 * 组装bbox

                 */
                float[] r = new float[once.size() * 5];
                for (int i = 0; i < once.size(); i++) {
                    for (int j = 0; j < once.get(i).length; j++) {
                        r[i * 5 + j] = once.get(i)[j];
                    }
                }
                orgLabelData.put(list[0], r);
            }
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
        return orgLabelData;
    }

    public static void load(String path, String extName, Tensor input, Tensor label, String[] idxSet, int[] indexs, Map<String, float[]> orgLabelData, int boxes, int classes) {
        label.clear();
        ImageLoaderJob job = ImageLoaderJob.getInstance(path, extName, input, label, idxSet, indexs, orgLabelData, boxes, classes, 0, input.number - 1);
        ForkJobEngine.run(job);
    }

    public static void loadVail(String path, String extName, Tensor input, Tensor label, String[] idxSet, int[] indexs, Map<String, float[]> orgLabelData, int boxes, int classes) {
        if (label != null) {
            label.clear();
        }
        ImageLoaderVailJob job = ImageLoaderVailJob.getInstance(path, extName, input, label, idxSet, indexs, orgLabelData, boxes, classes, 0, input.number - 1);
        ForkJobEngine.run(job);
    }

    public static void showImg(OMImage img, String outputPath) {
        ImageUtils utils = new ImageUtils();
        img.setData(MatrixOperation.multiplication(img.getData(), 255.0f));
        utils.createRGBImage(outputPath, "png", ImageUtils.color2rgb2(img.getData(), img.getHeight(), img.getWidth()), img.getWidth(), img.getHeight(), null);
    }

    public static void showImg(OMImage img, String outputPath, String filename) {
        ImageUtils utils = new ImageUtils();
        img.setData(MatrixOperation.multiplication(img.getData(), 255.0f));
        utils.createRGBImage(outputPath + filename, "png", ImageUtils.color2rgb2(img.getData(), img.getHeight(), img.getWidth()), img.getWidth(), img.getHeight(), null);
    }

    public static void showImg(Tensor input, Tensor label, int classnum, String outputPath) {
        input.data = MatrixOperation.multiplication(input.data, 255.0f);
        ImageUtils utils = new ImageUtils();
        for (int i = 0; i < input.number; i++) {
            //			System.out.println(i);
            float[] once = input.getByNumber(i);
            int count = 0;
            for (int c = 0; c < 90 / 5; c++) {
                float cx = label.data[i * 90 * 5 + c * 5 + 0];
                float cy = label.data[i * 90 * 5 + c * 5 + 1];
                //				System.out.println(cx+":"+cy);
                if (cx <= 0 && cy <= 0) {
                    break;
                }
                count++;
            }
            int[][] bbox = new int[count][5];
            for (int c = 0; c < count; c++) {
                float cx = label.data[i * 90 * 5 + c * 5 + 0] * input.width;
                float cy = label.data[i * 90 * 5 + c * 5 + 1] * input.height;
                float w = label.data[i * 90 * 5 + c * 5 + 2] * input.width;
                float h = label.data[i * 90 * 5 + c * 5 + 3] * input.height;
                bbox[c][0] = 0;
                bbox[c][1] = (int) (cx - w / 2);
                bbox[c][2] = (int) (cy - h / 2);
                bbox[c][3] = (int) (cx + w / 2);
                bbox[c][4] = (int) (cy + h / 2);
            }
            System.out.println(JsonUtils.toJson(bbox));
            utils.createRGBImage(outputPath + i + ".png", "png", ImageUtils.color2rgb2(once, input.height, input.width), input.width, input.height, bbox);
        }
    }

    public static OMImage resizeImage(OMImage oim, int w, int h) {
        if (oim.getWidth() == w && oim.getHeight() == h) {
            return copyImage(oim);
        }
        OMImage resized = createImage(w, h, oim.getChannel(), 0);
        OMImage part = createImage(w, oim.getHeight(), oim.getChannel(), 0);
        float w_scale = (oim.getWidth() - 1) * 1.0f / (w - 1);
        float h_scale = (oim.getHeight() - 1) * 1.0f / (h - 1);
        for (int k = 0; k < oim.getChannel(); ++k) {
            for (int r = 0; r < oim.getHeight(); ++r) {
                for (int c = 0; c < w; ++c) {
                    float val = 0;
                    if (c == w - 1 || oim.getWidth() == 1) {
                        val = getPixel(oim, oim.getWidth() - 1, r, k);
                    } else {
                        float sx = c * w_scale;
                        int ix = (int) sx;
                        float dx = sx - ix;
                        val = (1 - dx) * getPixel(oim, ix, r, k) + dx * getPixel(oim, ix + 1, r, k);
                    }
                    setPixel(part, c, r, k, val);
                }
            }
        }
        for (int k = 0; k < oim.getChannel(); ++k) {
            for (int r = 0; r < h; ++r) {
                float sy = r * h_scale;
                int iy = (int) sy;
                float dy = sy - iy;
                for (int c = 0; c < w; ++c) {
                    float val = (1 - dy) * getPixel(part, c, iy, k);
                    setPixel(resized, c, r, k, val);
                }
                if (r == h - 1 || oim.getHeight() == 1)
                    continue;
                for (int c = 0; c < w; ++c) {
                    float val = dy * getPixel(part, c, iy + 1, k);
                    addPixel(resized, c, r, k, val);
                }
            }
        }
        return resized;
    }

    public static void addPixel(OMImage m, int x, int y, int c, float val) {
        assert (x < m.getWidth() && y < m.getHeight() && c < m.getChannel());
        m.getData()[c * m.getHeight() * m.getWidth() + y * m.getWidth() + x] += val;
    }

    public static float[] formatXYWH(float[] label, int imw, int imh) {
        float[] output = new float[label.length];
        for (int i = 0; i < label.length / 5; i++) {
            float clazz = label[i * 5 + 0];
            float xmin = label[i * 5 + 1] / imw;
            float ymin = label[i * 5 + 2] / imh;
            float xmax = label[i * 5 + 3] / imw;
            float ymax = label[i * 5 + 4] / imh;
            float cx = (xmax + xmin) / 2;
            float cy = (ymax + ymin) / 2;
            float w = xmax - xmin;
            float h = ymax - ymin;
            output[i * 5 + 0] = clazz;
            output[i * 5 + 1] = cx;
            output[i * 5 + 2] = cy;
            output[i * 5 + 3] = w;
            output[i * 5 + 4] = h;
        }
        return output;
    }

    public static float[] formatXYWHOrg(float[] label, int imw, int imh) {
        float[] output = new float[label.length];
        for (int i = 0; i < label.length / 5; i++) {
            float clazz = label[i * 5 + 0];
            float xmin = label[i * 5 + 1];
            float ymin = label[i * 5 + 2];
            float xmax = label[i * 5 + 3];
            float ymax = label[i * 5 + 4];
            float cx = (xmax + xmin) / 2;
            float cy = (ymax + ymin) / 2;
            float w = xmax - xmin;
            float h = ymax - ymin;
            output[i * 5 + 0] = clazz;
            output[i * 5 + 1] = cx;
            output[i * 5 + 2] = cy;
            output[i * 5 + 3] = w;
            output[i * 5 + 4] = h;
        }
        return output;
    }

    public static float[] resized(String filePath, int w, int h) {
        //		System.out.println(filePath);
        OMImage orig = loadImage(filePath);
        float dw = 0;
        float dh = 0;
        float nw = 0;
        float nh = 0;
        OMImage sized = createImage(w, h, orig.getChannel(), 0.5f);
        float new_ar = (orig.getWidth() + RandomUtils.uniformFloat(-dw, dw)) / (orig.getHeight() + RandomUtils.uniformFloat(-dh, dh));
        float scale = 1;
        if (new_ar < 1) {
            nh = scale * h;
            nw = nh * new_ar;
        } else {
            nw = scale * w;
            nh = nw / new_ar;
        }
        float dx = (w - nw) / 2;
        float dy = (h - nh) / 2;
        placeImage(orig, (int) nw, (int) nh, (int) dx, (int) dy, sized);
        return sized.getData();
    }

    public static float[] resized(float[] data, int oc, int ow, int oh, int c, int w, int h) {
        float dw = 0;
        float dh = 0;
        float nw = 0;
        float nh = 0;
        OMImage orig = new OMImage(oc, oh, ow, data);
        OMImage sized = createImage(w, h, oc, 0.0f);
        float new_ar = (ow + RandomUtils.uniformFloat(-dw, dw)) / (oh + RandomUtils.uniformFloat(-dh, dh));
        float scale = 1;
        if (new_ar < 1) {
            nh = scale * h;
            nw = nh * new_ar;
        } else {
            nw = scale * w;
            nh = nw / new_ar;
        }
        float dx = (w - nw) / 2;
        float dy = (h - nh) / 2;
        placeImage(orig, (int) nw, (int) nh, (int) dx, (int) dy, sized);
        return sized.getData();
    }

    public static void loadVailDataDetection(Tensor x, Tensor y, int index, OMImage orig, float[] labelBoxs, int w, int h, int boxes, int classes) {
        float dw = 0;
        float dh = 0;
        float nw = 0;
        float nh = 0;
        OMImage sized = createImage(w, h, orig.getChannel(), 0.5f);
        float new_ar = (orig.getWidth() + RandomUtils.uniformFloat(-dw, dw)) / (orig.getHeight() + RandomUtils.uniformFloat(-dh, dh));
        float scale = 1;
        if (new_ar < 1) {
            nh = scale * h;
            nw = nh * new_ar;
        } else {
            nw = scale * w;
            nh = nw / new_ar;
        }
        //		System.out.println(nw+":"+nh);
        float dx = (w - nw) / 2;
        float dy = (h - nh) / 2;
        placeImage(orig, (int) nw, (int) nh, (int) dx, (int) dy, sized);
        //		System.out.println(JsonUtils.toJson(labelBoxs));
        setData(x, sized.getData(), index);
        if (y != null) {
            fillTruthDetection(y, index, labelBoxs, boxes, classes, 0, -dx / w, -dy / h, nw / w, nh / h);
        }
    }

    public static float[] resizeBBox(int padw, int padh, int w, int h, float[] bbox, int tw, int th) {
        float[] result = new float[bbox.length];
        for (int i = 0; i < bbox.length / 5; i++) {
            if (padw != 0) {
                result[i * 5 + 1] = (bbox[i * 5 + 1] + padw) * tw / h;
                result[i * 5 + 2] = bbox[i * 5 + 2] * th / h;
                result[i * 5 + 3] = bbox[i * 5 + 3] * tw / h;
                result[i * 5 + 4] = bbox[i * 5 + 4] * th / h;
            }
            if (padh != 0) {
                result[i * 5 + 1] = bbox[i * 5 + 1] * tw / w;
                result[i * 5 + 2] = (bbox[i * 5 + 2] + padh) * th / w;
                result[i * 5 + 3] = bbox[i * 5 + 3] * tw / w;
                result[i * 5 + 4] = bbox[i * 5 + 4] * th / w;
            }
            if (padw == 0 && padh == 0) {
                result[i * 5 + 1] = bbox[i * 5 + 1] * tw / w;
                result[i * 5 + 2] = bbox[i * 5 + 2] * th / w;
                result[i * 5 + 3] = bbox[i * 5 + 3] * tw / w;
                result[i * 5 + 4] = bbox[i * 5 + 4] * th / w;
            }
        }
        return result;
    }

    public static void loadDataDetection2(Tensor x, Tensor y, int index, OMImage orig, float[] labelBoxs, int w, int h, int boxes, int classes, float jitter, float hue, float saturation, float exposure, float resize, int letter_box) {
        int ow = orig.getWidth();
        int oh = orig.getHeight();
        int dw = (int) (jitter * ow);
        int dh = (int) (jitter * oh);
        //		int pleft  = RandomUtils.uniformInt(-dw, dw);
        //	    int pright = RandomUtils.uniformInt(-dw, dw);
        //	    int ptop   = RandomUtils.uniformInt(-dh, dh);
        //	    int pbot   = RandomUtils.uniformInt(-dh, dh);
        float resize_down = resize, resize_up = resize;
        if (resize_down > 1.0)
            resize_down = 1 / resize_down;
        int min_rdw = (int) (ow * (1 - (1 / resize_down)) / 2);
        int min_rdh = (int) (oh * (1 - (1 / resize_down)) / 2);
        if (resize_up < 1.0)
            resize_up = 1 / resize_up;
        int max_rdw = (int) (ow * (1 - (1 / resize_up)) / 2);
        int max_rdh = (int) (oh * (1 - (1 / resize_up)) / 2);
        float r1 = 0, r2 = 0, r3 = 0, r4 = 0;
        float resize_r1 = 0, resize_r2 = 0;
        int flip = 0;
        resize_r1 = RandomUtils.randomFloat();
        resize_r2 = RandomUtils.randomFloat();
        r1 = RandomUtils.randomFloat();
        r2 = RandomUtils.randomFloat();
        r3 = RandomUtils.randomFloat();
        r4 = RandomUtils.randomFloat();
        int pleft = randPrecalcRandom(-dw, dw, r1);
        int pright = randPrecalcRandom(-dw, dw, r2);
        int ptop = randPrecalcRandom(-dh, dh, r3);
        int pbot = randPrecalcRandom(-dh, dh, r4);
        if (resize < 1) {
            pleft += randPrecalcRandom(min_rdw, 0, resize_r1);
            pright += randPrecalcRandom(min_rdw, 0, resize_r2);
            ptop += randPrecalcRandom(min_rdh, 0, resize_r1);
            pbot += randPrecalcRandom(min_rdh, 0, resize_r2);
        } else {
            pleft += randPrecalcRandom(min_rdw, max_rdw, resize_r1);
            pright += randPrecalcRandom(min_rdw, max_rdw, resize_r2);
            ptop += randPrecalcRandom(min_rdh, max_rdh, resize_r1);
            pbot += randPrecalcRandom(min_rdh, max_rdh, resize_r2);
        }
        if (letter_box == 1) {
            float img_ar = (float) ow / (float) oh;
            float net_ar = (float) w / (float) h;
            float result_ar = img_ar / net_ar;
            if (result_ar > 1)  // sheight - should be increased
            {
                float oh_tmp = ow / net_ar;
                float delta_h = (oh_tmp - oh) / 2;
                ptop = (int) (ptop - delta_h);
                pbot = (int) (pbot - delta_h);
            } else  // swidth - should be increased
            {
                float ow_tmp = oh * net_ar;
                float delta_w = (ow_tmp - ow) / 2;
                pleft = (int) (pleft - delta_w);
                pright = (int) (pright - delta_w);
            }
        }
        int swidth = ow - pleft - pright;
        int sheight = oh - ptop - pbot;
        float sx = swidth * 1.0f / ow;
        float sy = sheight * 1.0f / oh;
        OMImage cropped = cropImage(orig, pleft, ptop, swidth, sheight);
        float dx = (pleft * 1.0f / ow) / sx;
        float dy = (ptop * 1.0f / oh) / sy;
        OMImage sized = resizeImage(cropped, w, h);
        flip = Math.abs(RandomUtils.rand() % 2);
        if (flip == 1)
            flipImage(sized);
        YoloDataTransform.randomDistortImage(sized.getData(), sized.getWidth(), sized.getHeight(), sized.getChannel(), hue, saturation, exposure);
        setData(x, sized.getData(), index);
        fillTruthDetection(y, index, labelBoxs, boxes, classes, flip, dx, dy, 1.0f / sx, 1.0f / sy);
    }

    public static void loadDataDetection(Tensor x, Tensor y, int index, OMImage orig, float[] labelBoxs, int w, int h, int boxes, int classes, float jitter, float hue, float saturation, float exposure) {
        OMImage sized = createImage(w, h, orig.getChannel(), 0.5f);
        float dw = jitter * orig.getWidth();
        float dh = jitter * orig.getHeight();
        float nw = 0.0f;
        float nh = 0.0f;
        float new_ar = (orig.getWidth() + RandomUtils.uniformFloat(-dw, dw)) / (orig.getHeight() + RandomUtils.uniformFloat(-dh, dh));
        float scale = 1;
        if (new_ar < 1) {
            nh = scale * h;
            nw = nh * new_ar;
        } else {
            nw = scale * w;
            nh = nw / new_ar;
        }
        float dx = RandomUtils.uniformFloat(0, w - nw);
        float dy = RandomUtils.uniformFloat(0, h - nh);
        //        System.out.println(dy);
        placeImage(orig, (int) nw, (int) nh, (int) dx, (int) dy, sized);
        YoloDataTransform.randomDistortImage(sized.getData(), sized.getWidth(), sized.getHeight(), sized.getChannel(), hue, saturation, exposure);
        int flip = Math.abs(RandomUtils.rand() % 2);
        if (flip == 1)
            flipImage(sized);
        setData(x, sized.getData(), index);
        fillTruthDetection(y, index, labelBoxs, boxes, classes, flip, -dx / w, -dy / h, nw / w, nh / h);
    }

    public static void fillTruthDetection(Tensor output, int index, float[] label, int num_boxes, int classes, int flip, float dx, float dy, float sx, float sy) {
        BoxLabel[] boxes = formatBox(label);
        int count = boxes.length;
        randomizeBoxes(boxes, count);
        //		System.out.println(JsonUtils.toJson(boxes));
        correctBoxes(boxes, count, dx, dy, sx, sy, flip);
        //		System.out.println(JsonUtils.toJson(boxes));
        if (count > num_boxes)
            count = num_boxes;
        float x, y, w, h;
        int ignore = 0;
        for (int i = 0; i < count; ++i) {
            x = boxes[i].x;
            y = boxes[i].y;
            w = boxes[i].w;
            h = boxes[i].h;
            //	        System.out.println(x+":"+y+":"+w+":"+y);
            if ((w < .001 || h < .001)) {
                ++ignore;
                continue;
            }
            int currentC = i - ignore;
            output.data[index * num_boxes * 5 + currentC * 5 + 0] = x;
            output.data[index * num_boxes * 5 + currentC * 5 + 1] = y;
            output.data[index * num_boxes * 5 + currentC * 5 + 2] = w;
            output.data[index * num_boxes * 5 + currentC * 5 + 3] = h;
            output.data[index * num_boxes * 5 + currentC * 5 + 4] = boxes[i].clazz;
        }
    }

    public static float fillTruthDetection(Tensor output, int index, float[] label, int num_boxes, int classes, int flip, float dx, float dy, float sx, float sy, int net_w, int net_h) {
        BoxLabel[] boxes = formatBox(label);
        int count = boxes.length;
        int min_w_h = 0;
        float lowest_w = 1.0f / net_w;
        float lowest_h = 1.0f / net_h;
        randomizeBoxes(boxes, count);
        //		System.out.println(JsonUtils.toJson(boxes));
        correctBoxes(boxes, count, dx, dy, sx, sy, flip);
        //		System.out.println(JsonUtils.toJson(boxes));
        if (count > num_boxes)
            count = num_boxes;
        float x, y, w, h;
        int ignore = 0;
        for (int i = 0; i < count; ++i) {
            x = boxes[i].x;
            y = boxes[i].y;
            w = boxes[i].w;
            h = boxes[i].h;
            //	        System.out.println(x+":"+y+":"+w+":"+y);
            if ((w < .001 || h < .001)) {
                ++ignore;
                continue;
            }
            if ((w < lowest_w || h < lowest_h)) {
                ++ignore;
                continue;
            }
            if (x == 999999 || y == 999999) {
                ++ignore;
                continue;
            }
            if (x <= 0 || x > 1 || y <= 0 || y > 1) {
                ++ignore;
                continue;
            }
            if (w > 1) {
                w = 1;
            }
            if (h > 1) {
                h = 1;
            }
            if (x == 0)
                x += lowest_w;
            if (y == 0)
                y += lowest_h;
            int currentC = i - ignore;
            output.data[index * num_boxes * 5 + currentC * 5 + 0] = x;
            output.data[index * num_boxes * 5 + currentC * 5 + 1] = y;
            output.data[index * num_boxes * 5 + currentC * 5 + 2] = w;
            output.data[index * num_boxes * 5 + currentC * 5 + 3] = h;
            output.data[index * num_boxes * 5 + currentC * 5 + 4] = boxes[i].clazz;
            if (min_w_h == 0)
                min_w_h = (int) (w * net_w);
            if (min_w_h > w * net_w)
                min_w_h = (int) (w * net_w);
            if (min_w_h > h * net_h)
                min_w_h = (int) (h * net_h);
        }
        return min_w_h;
    }

    public static float[] fillTruth(float[] label, float dx, float dy, float sx, float sy, int tw, int th) {
        float[] truth = new float[label.length];
        BoxLabel[] boxes = formatBox(label);
        int count = boxes.length;
        randomizeBoxes(boxes, count);
        //		System.out.println(JsonUtils.toJson(boxes));
        correctBoxes(boxes, count, dx, dy, sx, sy, 0);
        //		System.out.println(JsonUtils.toJson(boxes));
        for (int i = 0; i < count; ++i) {
            truth[i * 5 + 0] = boxes[i].clazz;
            truth[i * 5 + 1] = boxes[i].left * tw;
            truth[i * 5 + 2] = boxes[i].top * th;
            truth[i * 5 + 3] = boxes[i].right * tw;
            truth[i * 5 + 4] = boxes[i].bottom * th;
        }
        return truth;
    }

    public static void correctBoxes(BoxLabel[] boxes, int n, float dx, float dy, float sx, float sy, int flip) {
        for (int i = 0; i < n; ++i) {
            if (boxes[i].x == 0 && boxes[i].y == 0) {
                boxes[i].x = 999999;
                boxes[i].y = 999999;
                boxes[i].w = 999999;
                boxes[i].h = 999999;
                continue;
            }
            //	        System.out.println(boxes[i].left+"*"+sx+"-"+dx);
            boxes[i].left = boxes[i].left * sx - dx;
            boxes[i].right = boxes[i].right * sx - dx;
            boxes[i].top = boxes[i].top * sy - dy;
            boxes[i].bottom = boxes[i].bottom * sy - dy;
            if (flip == 1) {
                float swap = boxes[i].left;
                boxes[i].left = 1.0f - boxes[i].right;
                boxes[i].right = 1.0f - swap;
            }
            //	        System.out.println(boxes[i].left);
            //	        System.out.println("====================");
            boxes[i].left = constrain(0, 1, boxes[i].left);
            boxes[i].right = constrain(0, 1, boxes[i].right);
            boxes[i].top = constrain(0, 1, boxes[i].top);
            boxes[i].bottom = constrain(0, 1, boxes[i].bottom);
            boxes[i].x = (boxes[i].left + boxes[i].right) / 2;
            boxes[i].y = (boxes[i].top + boxes[i].bottom) / 2;
            boxes[i].w = (boxes[i].right - boxes[i].left);
            boxes[i].h = (boxes[i].bottom - boxes[i].top);
            boxes[i].w = constrain(0, 1, boxes[i].w);
            boxes[i].h = constrain(0, 1, boxes[i].h);
        }
    }

    public static float constrain(float min, float max, float a) {
        if (a < min)
            return min;
        if (a > max)
            return max;
        return a;
    }

    public static void randomizeBoxes(BoxLabel[] b, int n) {
        for (int i = 0; i < n; ++i) {
            BoxLabel swap = b[i];
            int index = Math.abs(RandomUtils.rand() % n);
            b[i] = b[index];
            b[index] = swap;
        }
    }

    public static float[] resizeImage(File file, float[] labelBoxs, int w, int h, String outputPath) {
        float dw = 0;
        float dh = 0;
        float nw = 0;
        float nh = 0;
        OMImage orig = loadImage(file);
        OMImage sized = createImage(w, h, orig.getChannel(), 0.5f);
        float new_ar = (orig.getWidth() + RandomUtils.uniformFloat(-dw, dw)) / (orig.getHeight() + RandomUtils.uniformFloat(-dh, dh));
        float scale = 1;
        if (new_ar < 1) {
            nh = scale * h;
            nw = nh * new_ar;
        } else {
            nw = scale * w;
            nh = nw / new_ar;
        }
        float dx = (w - nw) / 2;
        float dy = (h - nh) / 2;
        placeImage(orig, (int) nw, (int) nh, (int) dx, (int) dy, sized);
        float[] label_xywh = formatXYWH(labelBoxs, orig.getWidth(), orig.getHeight());
        float[] labels = fillTruth(label_xywh, -dx / w, -dy / h, nw / w, nh / h, w, h);
        showImg(sized, outputPath);
        return labels;
    }

    public static void resizeImage(File file, int w, int h, String outputPath) {
        float dw = 0;
        float dh = 0;
        float nw = 0;
        float nh = 0;
        OMImage orig = loadImage(file);
        OMImage sized = createImage(w, h, orig.getChannel(), 0.5f);
        float new_ar = (orig.getWidth() + RandomUtils.uniformFloat(-dw, dw)) / (orig.getHeight() + RandomUtils.uniformFloat(-dh, dh));
        float scale = 1;
        if (new_ar < 1) {
            nh = scale * h;
            nw = nh * new_ar;
        } else {
            nw = scale * w;
            nh = nw / new_ar;
        }
        float dx = (w - nw) / 2;
        float dy = (h - nh) / 2;
        placeImage(orig, (int) nw, (int) nh, (int) dx, (int) dy, sized);
        showImg(sized, outputPath);
    }

    public static float[] resizeImage(File file, int w, int h) {
        float nw = 0;
        float nh = 0;
        OMImage orig = loadImage(file);
        OMImage sized = createImage(w, h, orig.getChannel(), 0.0f);
        float new_ar = 1;
        float scale = 1;
        if (new_ar < 1) {
            nh = scale * h;
            nw = nh * new_ar;
        } else {
            nw = scale * w;
            nh = nw / new_ar;
        }
        float dx = (w - nw) / 2;
        float dy = (h - nh) / 2;
        placeImage(orig, (int) nw, (int) nh, (int) dx, (int) dy, sized);
        return sized.getData();
    }

    public static OMImage cropImage(OMImage orig, int dx, int dy, int w, int h) {
        OMImage cropped = createImage(w, h, orig.getChannel(), 0.5f);
        for (int k = 0; k < orig.getChannel(); ++k) {
            for (int j = 0; j < h; ++j) {
                for (int i = 0; i < w; ++i) {
                    int r = j + dy;
                    int c = i + dx;
                    float val = 0;
                    r = constrain_int(r, 0, orig.getHeight() - 1);
                    c = constrain_int(c, 0, orig.getWidth() - 1);
                    if (r >= 0 && r < orig.getHeight() && c >= 0 && c < orig.getWidth()) {
                        val = getPixel(orig, c, r, k);
                    }
                    setPixel(cropped, i, j, k, val);
                }
            }
        }
        return cropped;
    }

    public static int constrain_int(int a, int min, int max) {
        if (a < min)
            return min;
        if (a > max)
            return max;
        return a;
    }

    public static BoxLabel[] formatBox(float[] label) {
        int count = label.length / 5;
        BoxLabel[] boxes = new BoxLabel[count];
        for (int c = 0; c < count; c++) {
            BoxLabel box = new BoxLabel();
            box.setClazz(label[c * 5 + 0]);
            box.setX(label[c * 5 + 1]);
            box.setY(label[c * 5 + 2]);
            box.setW(label[c * 5 + 3]);
            box.setH(label[c * 5 + 4]);
            box.setLeft(box.x - box.w / 2);
            box.setRight(box.x + box.w / 2);
            box.setTop(box.y - box.h / 2);
            box.setBottom(box.y + box.h / 2);
            boxes[c] = box;
        }
        return boxes;
    }

    public static void setData(Tensor x, float[] data, int index) {
        System.arraycopy(data, 0, x.data, index * data.length, data.length);
    }

    public static OMImage loadImage(String filePath, float[] data) {
        OMImage image = null;
        try {
            File file = new File(filePath);
            if (file.exists()) {
                image = YoloImageUtils.IU().loadOMImage(file, data);
            }
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
            System.out.println("=====================>:" + filePath);
        }
        return image;
    }

    public static OMImage loadImage(String filePath) {
        OMImage image = null;
        try {
            File file = new File(filePath);
            if (file.exists()) {
                image = YoloImageUtils.IU().loadOMImage(file);
            }
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
            System.out.println("=====================>:" + filePath);
        }
        return image;
    }

    public static Tensor loadImage(String filePath, int tw, int th, float[] mean, float[] std) {
        Tensor input = null;
        try {
            File file = new File(filePath);
            if (file.exists()) {
                OMImage image = YoloImageUtils.IU().loadOMImgAndResize(file, tw, th, mean, std);
                input = new Tensor(1, image.getChannel(), image.getHeight(), image.getWidth(), image.getData(), true);
            }
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
            System.out.println("=====================>:" + filePath);
        }
        return input;
    }

    public static void loadImage(Tensor input, int index, String filePath, int tw, int th, float[] mean, float[] std, boolean bicubic) {
        try {
            File file = new File(filePath);
            if (file.exists()) {
                OMImage image = null;
                if (bicubic) {
                    image = YoloImageUtils.IU().loadOMImgAndResizeToBicubic(file, tw, th, mean, std);
                } else {
                    image = YoloImageUtils.IU().loadOMImgAndResize(file, tw, th, mean, std);
                }
                System.arraycopy(image.getData(), 0, input.data, index * input.getOnceSize(), input.getOnceSize());
            }
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
            System.out.println("=====================>:" + filePath);
        }
    }

    public static void loadImage(float[] input, int len, int index, String filePath, int tw, int th, float[] mean, float[] std, boolean bicubic) {
        try {
            File file = new File(filePath);
            if (file.exists()) {
                OMImage image = null;
                if (bicubic) {
                    image = YoloImageUtils.IU().loadOMImgAndResizeToBicubic(file, tw, th, mean, std);
                } else {
                    image = YoloImageUtils.IU().loadOMImgAndResize(file, tw, th, mean, std);
                }
                System.arraycopy(image.getData(), 0, input, index * len, len);
            }
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
            System.out.println("=====================>:" + filePath);
        }
    }

    public static OMImage loadImage(File file) {
        OMImage image = null;
        try {
            if (file.exists()) {
                image = YoloImageUtils.IU().loadOMImage(file);
            }
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
        return image;
    }

    public static OMImage createImage(int w, int h, int c, float val) {
        float[] data = new float[c * w * h];
        OMImage image = new OMImage(c, h, w, data);
        if (val != 0) {
            MatrixUtils.val(data, val);
        }
        return image;
    }

    public static OMImage copyImage(OMImage img) {
        float[] data = Arrays.copyOf(img.getData(), img.getData().length);
        OMImage image = new OMImage(img.getChannel(), img.getHeight(), img.getWidth(), data);
        return image;
    }

    public static void flipImage(OMImage a) {
        for (int k = 0; k < a.getChannel(); ++k) {
            for (int i = 0; i < a.getHeight(); ++i) {
                for (int j = 0; j < a.getWidth() / 2; ++j) {
                    int index = j + a.getWidth() * (i + a.getHeight() * (k));
                    int flip = (a.getWidth() - j - 1) + a.getWidth() * (i + a.getHeight() * (k));
                    float swap = a.getData()[flip];
                    a.getData()[flip] = a.getData()[index];
                    a.getData()[index] = swap;
                }
            }
        }
    }

    public static void placeImage(OMImage im, int w, int h, int dx, int dy, OMImage cav) {
        for (int c = 0; c < im.getChannel(); c++) {
            for (int y = 0; y < h; y++) {
                int ry = (int) (((float) y / h) * im.getHeight());
                for (int x = 0; x < w; x++) {
                    int rx = (int) (((float) x / w) * im.getWidth());
                    //	                System.out.println(rx+":"+ry);
                    float val = bilinearInterpolate(im, rx, ry, c);
                    setPixel(cav, x + dx, y + dy, c, val);
                }
            }
        }
    }

    public static float bilinearInterpolate(OMImage im, float x, float y, int c) {
        int ix = (int) Math.floor(x);
        int iy = (int) Math.floor(y);
        float dx = x - ix;
        float dy = y - iy;
        float val = (1 - dy) * (1 - dx) * getPixelExtend(im, ix, iy, c) + dy * (1 - dx) * getPixelExtend(im, ix, iy + 1, c) + (1 - dy) * dx * getPixelExtend(im, ix + 1, iy, c) + dy * dx * getPixelExtend(im, ix + 1, iy + 1, c);
        return val;
    }

    public static float getPixelExtend(OMImage m, int x, int y, int c) {
        if (x < 0 || x >= m.getWidth() || y < 0 || y >= m.getHeight())
            return 0;
        if (c < 0 || c >= m.getChannel())
            return 0;
        return getPixel(m, x, y, c);
    }

    public static float getPixel(OMImage img, int x, int y, int c) {
        assert (x < img.getWidth() && y < img.getHeight() && c < img.getChannel());
        return img.getData()[c * img.getHeight() * img.getWidth() + y * img.getWidth() + x];
    }

    public static void setPixel(OMImage img, int x, int y, int c, float val) {
        if (x < 0 || y < 0 || c < 0 || x >= img.getWidth() || y >= img.getHeight() || c >= img.getChannel())
            return;
        assert (x < img.getWidth() && y < img.getHeight() && c < img.getChannel());
        img.getData()[c * img.getHeight() * img.getWidth() + y * img.getWidth() + x] = val;
    }

    public static int randPrecalcRandom(float min, float max, float random_part) {
        if (max < min) {
            float swap = min;
            min = max;
            max = swap;
        }
        return (int) ((random_part * (max - min)) + min);
    }
}

