package com.omega.example.yolo.utils;

import com.omega.common.config.Tensor;
import com.omega.common.utils.ImageUtils;
import com.omega.engine.nn.data.ImageData;
import com.omega.example.yolo.data.ImageLoader;
import com.omega.example.yolo.model.YoloDataSet;
import com.omega.example.yolo.model.YoloImage;
import org.w3c.dom.*;

import java.io.*;
import java.math.BigDecimal;
import java.util.*;

public class YoloImageUtils {
    public static final String[] GL_CLASSES = new String[]{"person", "bird", "cat", "cow", "dog", "horse", "sheep", "aeroplane", "bicycle", "boat", "bus", "car", "motorbike", "train", "bottle", "chair", "diningtable", "pottedplant", "sofa", "tvmonitor"};
    //	public static final String[] GL_CLASSES = new String[] {"unmask", "mask"};
    public static final int YOLO_IMG_SIZE = 416;
    public static final int GRID_SIZE = 7;
    private static ImageUtils iu;
    public float[] mean = new float[]{0.491f, 0.482f, 0.446f};
    public float[] std = new float[]{0.247f, 0.243f, 0.261f};

    public static ImageUtils IU() {
        if (iu == null) {
            iu = new ImageUtils();
        }
        return iu;
    }

    public static int[] resize(String src, String outPath, int rw, int rh) throws Exception {
        ImageData data = IU().getImageData(src);
        int padw = 0;
        int padh = 0;
        int w = data.getWeight();
        int h = data.getHeight();
        if (h > w) {
            padw = new BigDecimal(h).subtract(new BigDecimal(w)).divide(new BigDecimal(2), BigDecimal.ROUND_DOWN).intValue();
        } else if (h < w) {
            padh = new BigDecimal(w).subtract(new BigDecimal(h)).divide(new BigDecimal(2), BigDecimal.ROUND_DOWN).intValue();
        }
        int ow = w + padw * 2;
        int oh = h + padh * 2;
        int[] square = paddingToSquare(data.getColor(), w, h, padw, padh, ow, oh);
        IU().createRGBImage(outPath, data.getExtName(), ImageUtils.color2rgb(square, ow, oh), rw, rh);
        return new int[]{w, h, padw, padh};
    }

    public static int[] resize(ImageData data, String outPath, int rw, int rh, int padw, int padh, int w, int h) throws Exception {
        if (h > w) {
            padw = new BigDecimal(h).subtract(new BigDecimal(w)).divide(new BigDecimal(2), BigDecimal.ROUND_DOWN).intValue();
        } else if (h < w) {
            padh = new BigDecimal(w).subtract(new BigDecimal(h)).divide(new BigDecimal(2), BigDecimal.ROUND_DOWN).intValue();
        }
        int ow = w + padw * 2;
        int oh = h + padh * 2;
        int[] square = paddingToSquare(data.getColor(), w, h, padw, padh, ow, oh);
        IU().createRGBImage(outPath, data.getExtName(), ImageUtils.color2rgb(square, ow, oh), rw, rh);
        return square;
    }

    public static int[] resize(ImageData data, String outPath, int rw, int rh, int padw, int padh, int w, int h, int[][] bbox) throws Exception {
        if (h > w) {
            padw = new BigDecimal(h).subtract(new BigDecimal(w)).divide(new BigDecimal(2), BigDecimal.ROUND_DOWN).intValue();
        } else if (h < w) {
            padh = new BigDecimal(w).subtract(new BigDecimal(h)).divide(new BigDecimal(2), BigDecimal.ROUND_DOWN).intValue();
        }
        int ow = w + padw * 2;
        int oh = h + padh * 2;
        int[] square = paddingToSquare(data.getColor(), w, h, padw, padh, ow, oh);
        IU().createRGBImage(outPath, data.getExtName(), ImageUtils.color2rgb(square, ow, oh), rw, rh, bbox);
        return square;
    }

    public static int[] paddingToSquare(int[] input, int w, int h, int padw, int padh, int ow, int oh) {
        int[] output = new int[3 * ow * oh];
        for (int i = 0; i < output.length; i++) {
            output[i] = 114;
        }
        //		System.out.println(ow+":"+oh);
        //		System.out.println(padw+":"+padh+":"+output.length);
        for (int x = 0; x < w; x++) {
            for (int y = 0; y < h; y++) {
                output[0 * ow * oh + (x + padw) * oh + (y + padh)] = input[0 * w * h + x * h + y];
                output[1 * ow * oh + (x + padw) * oh + (y + padh)] = input[1 * w * h + x * h + y];
                output[2 * ow * oh + (x + padw) * oh + (y + padh)] = input[2 * w * h + x * h + y];
            }
        }
        return output;
    }

    public static int[][] anno2bbox(Map<String, List<int[]>> labelData, String filename) {
        try {
            //			System.out.println(filename);
            List<int[]> objList = labelData.get(filename);
            int[][] output = new int[objList.size()][5];
            for (int i = 0; i < objList.size(); i++) {
                int[] bbox = objList.get(i);
                output[i][0] = bbox[4];
                output[i][1] = bbox[0];
                output[i][2] = bbox[1];
                output[i][3] = bbox[2];
                output[i][4] = bbox[3];
            }
            return output;
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
        return null;
    }

    public static int[][] anno2bbox(String src) {
        try {
            File file = new File(src);
            //			System.out.println(file.exists());
            if (file.exists()) {
                Document doc = XmlParser.DB().parse(file);
                Element docEle = doc.getDocumentElement();
                NodeList objList = docEle.getElementsByTagName("object");
                if (objList == null || objList.getLength() <= 0) {
                    return null;
                }
                int[][] output = new int[objList.getLength()][5];
                for (int i = 0; i < objList.getLength(); i++) {
                    Node node = objList.item(i);
                    if (node.getNodeType() == Node.ELEMENT_NODE) {
                        Element e = (Element) node;
                        NodeList nodeList = e.getElementsByTagName("name");
                        String name = nodeList.item(0).getChildNodes().item(0).getNodeValue();
                        int nidx = getIndex(name);
                        //						System.out.println(nidx);
                        if (nidx < 0) {
                            continue;
                        }
                        nodeList = e.getElementsByTagName("bndbox");
                        Element boxe = (Element) nodeList.item(0);
                        nodeList = boxe.getElementsByTagName("xmin");
                        int xmin = Integer.parseInt(nodeList.item(0).getChildNodes().item(0).getNodeValue());
                        nodeList = boxe.getElementsByTagName("ymin");
                        int ymin = Integer.parseInt(nodeList.item(0).getChildNodes().item(0).getNodeValue());
                        nodeList = boxe.getElementsByTagName("xmax");
                        int xmax = Integer.parseInt(nodeList.item(0).getChildNodes().item(0).getNodeValue());
                        nodeList = boxe.getElementsByTagName("ymax");
                        int ymax = Integer.parseInt(nodeList.item(0).getChildNodes().item(0).getNodeValue());
                        output[i][0] = nidx;
                        output[i][1] = xmin;
                        output[i][2] = ymin;
                        output[i][3] = xmax;
                        output[i][4] = ymax;
                    }
                }
                return output;
            }
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
        return null;
    }

    public static int[][] anno2bbox(String src, String[] names) {
        try {
            File file = new File(src);
            //			System.out.println(file.exists());
            if (file.exists()) {
                Document doc = XmlParser.DB().parse(file);
                Element docEle = doc.getDocumentElement();
                NodeList objList = docEle.getElementsByTagName("object");
                if (objList == null || objList.getLength() <= 0) {
                    return null;
                }
                int[][] output = new int[objList.getLength()][5];
                for (int i = 0; i < objList.getLength(); i++) {
                    Node node = objList.item(i);
                    if (node.getNodeType() == Node.ELEMENT_NODE) {
                        Element e = (Element) node;
                        NodeList nodeList = e.getElementsByTagName("name");
                        String name = nodeList.item(0).getChildNodes().item(0).getNodeValue();
                        int nidx = getIndex(name, names);
                        //						System.out.println(nidx);
                        if (nidx < 0) {
                            continue;
                        }
                        nodeList = e.getElementsByTagName("bndbox");
                        Element boxe = (Element) nodeList.item(0);
                        nodeList = boxe.getElementsByTagName("xmin");
                        int xmin = Integer.parseInt(nodeList.item(0).getChildNodes().item(0).getNodeValue());
                        nodeList = boxe.getElementsByTagName("ymin");
                        int ymin = Integer.parseInt(nodeList.item(0).getChildNodes().item(0).getNodeValue());
                        nodeList = boxe.getElementsByTagName("xmax");
                        int xmax = Integer.parseInt(nodeList.item(0).getChildNodes().item(0).getNodeValue());
                        nodeList = boxe.getElementsByTagName("ymax");
                        int ymax = Integer.parseInt(nodeList.item(0).getChildNodes().item(0).getNodeValue());
                        output[i][0] = nidx;
                        output[i][1] = xmin;
                        output[i][2] = ymin;
                        output[i][3] = xmax;
                        output[i][4] = ymax;
                    }
                }
                return output;
            }
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
        return null;
    }

    public static int getIndex(String name) {
        for (int i = 0; i < GL_CLASSES.length; i++) {
            if (name.equals(GL_CLASSES[i])) {
                return i;
            }
        }
        return -1;
    }

    public static int getIndex(String name, String[] names) {
        for (int i = 0; i < names.length; i++) {
            if (name.equals(names[i])) {
                return i;
            }
        }
        return -1;
    }

    /**
     * w = wmax - wmin
     * <p>
     * h = hmax - hmin
     * <p>
     * cx = (wmax + wmin) / 2
     * <p>
     * cy = (hmax + hmin) / 2
     *
     * @param label
     * @return
     */
    public static int[][] formatLabel(int[][] label) {
        int[][] bbox = new int[label.length][label[0].length];
        for (int i = 0; i < label.length; i++) {
            bbox[i][0] = label[i][0];
            int xmin = label[i][1];
            int ymin = label[i][2];
            int xmax = label[i][3];
            int ymax = label[i][4];
            int cx = (xmax + xmin) / 2;
            int cy = (ymax + ymin) / 2;
            int w = xmax - xmin;
            int h = ymax - ymin;
            bbox[i][1] = cx;
            bbox[i][2] = cy;
            bbox[i][3] = w;
            bbox[i][4] = h;
        }
        return bbox;
    }

    public static int[][] resizeBBox(int padw, int padh, int w, int h, int[][] bbox) {
        for (int i = 0; i < bbox.length; i++) {
            if (padw != 0) {
                bbox[i][1] = (bbox[i][1] + padw) * YOLO_IMG_SIZE / h;
                bbox[i][2] = (bbox[i][2]) * YOLO_IMG_SIZE / h;
                bbox[i][3] = (bbox[i][3]) * YOLO_IMG_SIZE / h;
                bbox[i][4] = (bbox[i][4]) * YOLO_IMG_SIZE / h;
            }
            if (padh != 0) {
                bbox[i][1] = (bbox[i][1]) * YOLO_IMG_SIZE / w;
                bbox[i][2] = (bbox[i][2] + padh) * YOLO_IMG_SIZE / w;
                bbox[i][3] = (bbox[i][3]) * YOLO_IMG_SIZE / w;
                bbox[i][4] = (bbox[i][4]) * YOLO_IMG_SIZE / w;
            }
            if (padw == 0 && padh == 0) {
                bbox[i][1] = (bbox[i][1]) * YOLO_IMG_SIZE / w;
                bbox[i][2] = (bbox[i][2]) * YOLO_IMG_SIZE / w;
                bbox[i][3] = (bbox[i][3]) * YOLO_IMG_SIZE / w;
                bbox[i][4] = (bbox[i][4]) * YOLO_IMG_SIZE / w;
            }
        }
        return bbox;
    }

    public static YoloImage formatData(String filename, String imgDir, String labelDir, String imgOutDir, boolean boxTest) {
        try {
            String dataName = filename.substring(0, filename.lastIndexOf("."));
            String imgPath = imgDir + "\\" + filename;
            String labelPath = labelDir + "\\" + dataName + ".xml";
            String imgOutPath = imgOutDir + "\\" + filename;
            String imgTestOutPath = imgOutDir + "\\test_" + filename;
            ImageData data = IU().getImageData(imgPath);
            int padw = 0;
            int padh = 0;
            int w = data.getWeight();
            int h = data.getHeight();
            if (h > w) {
                padw = new BigDecimal(h).subtract(new BigDecimal(w)).divide(new BigDecimal(2), BigDecimal.ROUND_DOWN).intValue();
            } else if (h < w) {
                padh = new BigDecimal(w).subtract(new BigDecimal(h)).divide(new BigDecimal(2), BigDecimal.ROUND_DOWN).intValue();
            }
            int[][] label = anno2bbox(labelPath);
            int[][] bbox = formatLabel(label);
            IU().createRGBImage(imgTestOutPath, "jpg", ImageUtils.color2rgb(data.getColor(), w, h), bbox);
            bbox = resizeBBox(padw, padh, w, h, bbox);
            if (boxTest) {
                YoloImageUtils.resize(data, imgOutPath, YOLO_IMG_SIZE, YOLO_IMG_SIZE, padw, padh, w, h, bbox);
            } else {
                YoloImageUtils.resize(data, imgOutPath, YOLO_IMG_SIZE, YOLO_IMG_SIZE, padw, padh, w, h);
            }
            //			System.out.println(JsonUtils.toJson(bbox));
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
        return null;
    }

    public static YoloImage formatData(String filename, String imgDir, String labelDir, String imgOutDir) {
        try {
            YoloImage obj = new YoloImage();
            String dataName = filename.substring(0, filename.lastIndexOf("."));
            String imgPath = imgDir + "\\" + filename;
            String labelPath = labelDir + "\\" + dataName + ".xml";
            String imgOutPath = imgOutDir + "\\" + filename;
            ImageData data = IU().getImageData(imgPath);
            int padw = 0;
            int padh = 0;
            int w = data.getWeight();
            int h = data.getHeight();
            if (h > w) {
                padw = new BigDecimal(h).subtract(new BigDecimal(w)).divide(new BigDecimal(2), BigDecimal.ROUND_DOWN).intValue();
            } else if (h < w) {
                padh = new BigDecimal(w).subtract(new BigDecimal(h)).divide(new BigDecimal(2), BigDecimal.ROUND_DOWN).intValue();
            }
            int[][] label = anno2bbox(labelPath);
            int[][] bbox = formatLabel(label);
            bbox = resizeBBox(padw, padh, w, h, bbox);
            int[] resizeData = YoloImageUtils.resize(data, imgOutPath, YOLO_IMG_SIZE, YOLO_IMG_SIZE, padw, padh, w, h);
            System.out.println(dataName);
            obj.setName(dataName);
            obj.setChannel(3);
            obj.setHeight(YOLO_IMG_SIZE);
            obj.setWidth(YOLO_IMG_SIZE);
            obj.setBbox(bbox);
            obj.setData(resizeData);
            obj.setYoloLabel(LabelUtils.labelToYolo(bbox, GRID_SIZE, YOLO_IMG_SIZE));
            return obj;
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
        return null;
    }

    public static YoloImage formatData(File file, String labelDir, String imgOutDir) {
        try {
            YoloImage obj = new YoloImage();
            String filename = file.getName();
            String dataName = filename.substring(0, filename.lastIndexOf("."));
            String labelPath = labelDir + "\\" + dataName + ".xml";
            String imgOutPath = imgOutDir + "\\" + filename;
            ImageData data = IU().getImageData(file);
            int padw = 0;
            int padh = 0;
            int w = data.getWeight();
            int h = data.getHeight();
            if (h > w) {
                padw = new BigDecimal(h).subtract(new BigDecimal(w)).divide(new BigDecimal(2), BigDecimal.ROUND_DOWN).intValue();
            } else if (h < w) {
                padh = new BigDecimal(w).subtract(new BigDecimal(h)).divide(new BigDecimal(2), BigDecimal.ROUND_DOWN).intValue();
            }
            int[][] label = anno2bbox(labelPath);
            int[][] bbox = formatLabel(label);
            bbox = resizeBBox(padw, padh, w, h, bbox);
            int[] resizeData = YoloImageUtils.resize(data, imgOutPath, YOLO_IMG_SIZE, YOLO_IMG_SIZE, padw, padh, w, h);
            obj.setName(dataName);
            obj.setChannel(3);
            obj.setHeight(YOLO_IMG_SIZE);
            obj.setWidth(YOLO_IMG_SIZE);
            obj.setBbox(bbox);
            obj.setData(resizeData);
            //			System.out.println(dataName+":");
            obj.setYoloLabel(LabelUtils.labelToYolo(bbox, GRID_SIZE, YOLO_IMG_SIZE));
            return obj;
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
        return null;
    }

    public static int[][] xml2xyxy(File file, String labelDir) {
        int[][] label = null;
        try {
            String filename = file.getName();
            String dataName = filename.substring(0, filename.lastIndexOf("."));
            String labelPath = labelDir + "\\" + dataName + ".xml";
            label = anno2bbox(labelPath);
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
        return label;
    }

    public static int[][] xml2xyxy(File file, String labelDir, String[] names) {
        int[][] label = null;
        try {
            String filename = file.getName();
            String dataName = filename.substring(0, filename.lastIndexOf("."));
            String labelPath = labelDir + "\\" + dataName + ".xml";
            label = anno2bbox(labelPath, names);
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
        return label;
    }

    public static YoloImage formatData(File file, String labelDir, String imgOutDir, YoloVersion version) {
        try {
            YoloImage obj = new YoloImage();
            String filename = file.getName();
            String dataName = filename.substring(0, filename.lastIndexOf("."));
            String labelPath = labelDir + "\\" + dataName + ".xml";
            String imgOutPath = imgOutDir + "\\" + filename;
            ImageData data = IU().getImageData(file);
            int padw = 0;
            int padh = 0;
            int w = data.getWeight();
            int h = data.getHeight();
            if (h > w) {
                padw = new BigDecimal(h).subtract(new BigDecimal(w)).divide(new BigDecimal(2), BigDecimal.ROUND_DOWN).intValue();
            } else if (h < w) {
                padh = new BigDecimal(w).subtract(new BigDecimal(h)).divide(new BigDecimal(2), BigDecimal.ROUND_DOWN).intValue();
            }
            int[][] label = anno2bbox(labelPath);
            int[][] bbox = formatLabel(label);
            bbox = resizeBBox(padw, padh, w, h, bbox);
            int[] resizeData = YoloImageUtils.resize(data, imgOutPath, YOLO_IMG_SIZE, YOLO_IMG_SIZE, padw, padh, w, h);
            obj.setName(dataName);
            obj.setChannel(3);
            obj.setHeight(YOLO_IMG_SIZE);
            obj.setWidth(YOLO_IMG_SIZE);
            obj.setBbox(bbox);
            obj.setData(resizeData);
            switch (version) {
                case yolov1:
                    obj.setYoloLabel(LabelUtils.labelToYolo(bbox, GRID_SIZE, YOLO_IMG_SIZE));
                    break;
                case yolov3:
                    obj.setYoloLabel(LabelUtils.labelToYoloV3(bbox, YOLO_IMG_SIZE));
                    break;
                case yolov3_xyxy:
                    obj.setYoloLabel(LabelUtils.labelToYoloV3_xyxy(bbox, YOLO_IMG_SIZE));
                    break;
            }
            return obj;
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
        return null;
    }

    public static YoloImage formatData(File file, String imgOutDir, Map<String, List<int[]>> labelData, YoloVersion version) {
        try {
            YoloImage obj = new YoloImage();
            String filename = file.getName();
            System.out.println(filename);
            String dataName = filename.substring(0, filename.lastIndexOf("."));
            String imgOutPath = imgOutDir + "\\" + filename;
            ImageData data = IU().getImageData(file);
            int padw = 0;
            int padh = 0;
            int w = data.getWeight();
            int h = data.getHeight();
            if (h > w) {
                padw = new BigDecimal(h).subtract(new BigDecimal(w)).divide(new BigDecimal(2), BigDecimal.ROUND_DOWN).intValue();
            } else if (h < w) {
                padh = new BigDecimal(w).subtract(new BigDecimal(h)).divide(new BigDecimal(2), BigDecimal.ROUND_DOWN).intValue();
            }
            int[][] label = anno2bbox(labelData, filename);
            int[][] bbox = formatLabel(label);
            bbox = resizeBBox(padw, padh, w, h, bbox);
            int[] resizeData = YoloImageUtils.resize(data, imgOutPath, YOLO_IMG_SIZE, YOLO_IMG_SIZE, padw, padh, w, h);
            obj.setName(dataName);
            obj.setChannel(3);
            obj.setHeight(YOLO_IMG_SIZE);
            obj.setWidth(YOLO_IMG_SIZE);
            obj.setBbox(bbox);
            obj.setData(resizeData);
            switch (version) {
                case yolov1:
                    obj.setYoloLabel(LabelUtils.labelToYolo(bbox, GRID_SIZE, YOLO_IMG_SIZE));
                    break;
                case yolov3:
                    obj.setYoloLabel(LabelUtils.labelToYoloV3(bbox, YOLO_IMG_SIZE));
                    break;
                case yolov3_xyxy:
                    obj.setYoloLabel(LabelUtils.labelToYoloV3_xyxy(bbox, YOLO_IMG_SIZE));
                    break;
            }
            return obj;
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
        return null;
    }

    public static void image2Yolo(String srcDir, String labelDir, String outDir, String bboxPath, YoloVersion version) {
        try {
            File file = new File(srcDir);
            Map<String, Object> bboxList = new LinkedHashMap<String, Object>();
            if (file.exists() && file.isDirectory()) {
                for (File img : file.listFiles()) {
                    YoloImage yi = formatData(img, labelDir, outDir, version);
                    System.out.println(yi.getName());
                    bboxList.put(yi.getName(), yi.getYoloLabel());
                }
            }
            File txt = new File(bboxPath);
            if (!txt.exists()) {
                txt.createNewFile(); // 创建新文件,有同名的文件的话直接覆盖
            }
            try (FileOutputStream fos = new FileOutputStream(txt);
                 //					OutputStreamWriter osr = new OutputStreamWriter(fos,"utf-8");
                 //					BufferedWriter bufferedWriter = new BufferedWriter(osr);
            ) {
                for (String name : bboxList.keySet()) {
                    String text = name;
                    for (float val : (float[]) bboxList.get(name)) {
                        text += " " + val;
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

    public static void xml2Yolo(String srcDir, String labelDir, String labelPath, String bboxPath) {
        try {
            File file = new File(srcDir);
            Map<String, Object> bboxList = new LinkedHashMap<String, Object>();
            if (file.exists() && file.isDirectory()) {
                String[] names = new String[200];
                try (FileInputStream fin = new FileInputStream(labelPath); InputStreamReader reader = new InputStreamReader(fin); BufferedReader buffReader = new BufferedReader(reader);) {
                    String strTmp = "";
                    int idx = 0;
                    while ((strTmp = buffReader.readLine()) != null) {
                        names[idx] = strTmp;
                        idx++;
                    }
                } catch (Exception e) {
                    // TODO: handle exception
                    e.printStackTrace();
                }
                for (File img : file.listFiles()) {
                    int[][] yi = xml2xyxy(img, labelDir, names);
                    String dataName = img.getName().substring(0, img.getName().lastIndexOf("."));
                    bboxList.put(dataName, yi);
                }
            }
            File txt = new File(bboxPath);
            if (!txt.exists()) {
                txt.createNewFile(); // 创建新文件,有同名的文件的话直接覆盖
            }
            try (FileOutputStream fos = new FileOutputStream(txt);) {
                for (String name : bboxList.keySet()) {
                    String text = name;
                    for (int[] vals : (int[][]) bboxList.get(name)) {
                        for (int val : vals) {
                            text += " " + val;
                        }
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

    public static void xml2Yolo(String srcDir, String labelDir, String bboxPath) {
        try {
            File file = new File(srcDir);
            Map<String, Object> bboxList = new LinkedHashMap<String, Object>();
            if (file.exists() && file.isDirectory()) {
                for (File img : file.listFiles()) {
                    int[][] yi = xml2xyxy(img, labelDir);
                    String dataName = img.getName().substring(0, img.getName().lastIndexOf("."));
                    bboxList.put(dataName, yi);
                }
            }
            File txt = new File(bboxPath);
            if (!txt.exists()) {
                txt.createNewFile(); // 创建新文件,有同名的文件的话直接覆盖
            }
            try (FileOutputStream fos = new FileOutputStream(txt);) {
                for (String name : bboxList.keySet()) {
                    String text = name;
                    for (int[] vals : (int[][]) bboxList.get(name)) {
                        for (int val : vals) {
                            text += " " + val;
                        }
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

    public static void csv2Yolo(String srcDir, String labelPath, String bboxPath, String[] labelSet) {
        try {
            /**
             * 读取CSV label数据

             */
            Map<String, List<int[]>> orgLabelData = new HashMap<String, List<int[]>>();
            loadLabelDataForCSV(labelPath, orgLabelData, labelSet);
            Map<String, Object> bboxList = new LinkedHashMap<String, Object>();
            File file = new File(srcDir);
            if (file.exists() && file.isDirectory()) {
                for (File img : file.listFiles()) {
                    int[][] label = anno2bbox(orgLabelData, img.getName());
                    //					System.out.println(yi.getName());
                    String name = img.getName().split(".jpg")[0];
                    bboxList.put(name, label);
                }
            }
            File txt = new File(bboxPath);
            if (!txt.exists()) {
                txt.createNewFile(); // 创建新文件,有同名的文件的话直接覆盖
            }
            try (FileOutputStream fos = new FileOutputStream(txt);) {
                for (String name : bboxList.keySet()) {
                    String text = name;
                    for (int[] vals : (int[][]) bboxList.get(name)) {
                        for (int val : vals) {
                            text += " " + val;
                        }
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

    public static void loadLabelDataForCSV(String labelPath, Map<String, List<int[]>> orgLabelData, String[] labelSet) {
        try (FileInputStream fin = new FileInputStream(labelPath); InputStreamReader reader = new InputStreamReader(fin); BufferedReader buffReader = new BufferedReader(reader);) {
            String strTmp = "";
            int idx = 0;
            while ((strTmp = buffReader.readLine()) != null) {
                if (idx > 0) {
                    String[] list = strTmp.split(",");
                    List<int[]> once = new ArrayList<int[]>();
                    String[] bboxData = list[1].split(" ");
                    int page = bboxData.length / 5;
                    for (int i = 0; i < page; i++) {
                        int[] bbox = new int[5];
                        for (int j = 0; j < 5; j++) {
                            if (j == 4) {
                                bbox[j] = getIndexByLabelSet(labelSet, bboxData[i * 5 + j]);
                            } else {
                                bbox[j] = Integer.parseInt(bboxData[i * 5 + j]);
                            }
                        }
                        once.add(bbox);
                    }
                    orgLabelData.put(list[0], once);
                }
                idx++;
            }
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public static int getIndexByLabelSet(String[] labelSet, String label) {
        for (int i = 0; i < labelSet.length; i++) {
            if (label.equals(labelSet[i])) {
                return i;
            }
        }
        return 0;
    }

    public static void image2YoloFormCSV(String srcDir, String labelPath, String[] labelSet, String outDir, String bboxPath, YoloVersion version) {
        try {
            /**
             * 读取CSV label数据

             */
            Map<String, List<int[]>> orgLabelData = new HashMap<String, List<int[]>>();
            loadLabelDataForCSV(labelPath, orgLabelData, labelSet);
            File file = new File(srcDir);
            Map<String, Object> bboxList = new LinkedHashMap<String, Object>();
            if (file.exists() && file.isDirectory()) {
                for (File img : file.listFiles()) {
                    YoloImage yi = formatData(img, outDir, orgLabelData, version);
                    //					System.out.println(yi.getName());
                    bboxList.put(yi.getName(), yi.getYoloLabel());
                }
            }
            File txt = new File(bboxPath);
            if (!txt.exists()) {
                txt.createNewFile(); // 创建新文件,有同名的文件的话直接覆盖
            }
            try (FileOutputStream fos = new FileOutputStream(txt);
                 //					OutputStreamWriter osr = new OutputStreamWriter(fos,"utf-8");
                 //					BufferedWriter bufferedWriter = new BufferedWriter(osr);
            ) {
                for (String name : bboxList.keySet()) {
                    String text = name;
                    for (float val : (float[]) bboxList.get(name)) {
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

    public static void image2YoloV3(String srcDir, String labelDir, String outDir, String bboxPath, YoloVersion version) {
        try {
            File file = new File(srcDir);
            Map<String, Object> bboxList = new LinkedHashMap<String, Object>();
            if (file.exists() && file.isDirectory()) {
                for (File img : file.listFiles()) {
                    YoloImage yi = formatData(img, labelDir, outDir, version);
                    System.out.println(yi.getName());
                    bboxList.put(yi.getName(), yi.getYoloLabel());
                }
            }
            File txt = new File(bboxPath);
            if (!txt.exists()) {
                txt.createNewFile(); // 创建新文件,有同名的文件的话直接覆盖
            }
            try (FileOutputStream fos = new FileOutputStream(txt);
                 //					OutputStreamWriter osr = new OutputStreamWriter(fos,"utf-8");
                 //					BufferedWriter bufferedWriter = new BufferedWriter(osr);
            ) {
                for (String name : bboxList.keySet()) {
                    String text = name;
                    for (float val : (float[]) bboxList.get(name)) {
                        text += " " + val;
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

    public static YoloDataSet loadImgData(String dirPath, String labelPath, int channel, int height, int width, int labelSize) {
        try {
            File file = new File(dirPath);
            if (file.exists() && file.isDirectory()) {
                File[] files = file.listFiles();
                YoloDataSet set = new YoloDataSet(files.length, channel, height, width, labelSize);
                for (int i = 0; i < files.length; i++) {
                    File img = files[i];
                    float[] data = IU().getImageData(img, true);
                    System.arraycopy(data, 0, set.input.data, i * channel * height * width, channel * height * width);
                }
                LabelUtils.loadLabel(labelPath, set.label);
                return set;
            }
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
        return null;
    }

    public static void loadImgDataToTensor(String filePath, Tensor out, int idx, boolean normalization) {
        try {
            File file = new File(filePath);
            if (file.exists()) {
                if (normalization) {
                    float[] data = IU().getImageData(file, true, false);
                    System.arraycopy(data, 0, out.data, idx * out.channel * out.height * out.width, out.channel * out.height * out.width);
                } else {
                    float[] data = IU().getImageData(file, false);
                    System.arraycopy(data, 0, out.data, idx * out.channel * out.height * out.width, out.channel * out.height * out.width);
                }
            }
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public static void loadImgDataToTensor(String filePath, Tensor out, int idx) {
        try {
            File file = new File(filePath);
            if (file.exists()) {
                float[] data = IU().getImageData(file, true, true);
                System.arraycopy(data, 0, out.data, idx * out.channel * out.height * out.width, out.channel * out.height * out.width);
            }
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public static void loadImgDataToOrgTensor(String filePath, Tensor out, int idx) {
        try {
            File file = new File(filePath);
            if (file.exists()) {
                float[] data = IU().getImageData(file, true, false);
                System.arraycopy(data, 0, out.data, idx * out.channel * out.height * out.width, out.channel * out.height * out.width);
            }
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public static void loadImgDataToGrayTensor(String filePath, Tensor out, int idx) {
        try {
            File file = new File(filePath);
            if (file.exists()) {
                float[] once = IU().getImageDataToGray(file, true);
                System.arraycopy(once, 0, out.data, idx * out.channel * out.height * out.width, out.channel * out.height * out.width);
            }
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public static float[] loadImgDataToArray(String filePath) {
        try {
            File file = new File(filePath);
            if (file.exists()) {
                float[] data = IU().getImageData(file, false, false);
                return data;
            }
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
        return null;
    }

    public static float[] loadImgDataToArray(String filePath, boolean norm) {
        try {
            File file = new File(filePath);
            if (file.exists()) {
                float[] data = IU().getImageData(file, norm, norm);
                return data;
            }
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
        return null;
    }

    public static float[] loadImgDataToArray(String filePath, boolean norm, float[] mean, float[] std) {
        try {
            File file = new File(filePath);
            if (file.exists()) {
                float[] data = IU().getImageData(file, norm, norm, mean, std);
                return data;
            }
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
        return null;
    }

    public static float[] loadImgDataToGrayArray(String filePath, boolean norm) {
        try {
            File file = new File(filePath);
            if (file.exists()) {
                float[] data = IU().getImageDataToGray(file, norm);
                return data;
            }
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
        return null;
    }

    public static void loadImgDataResize(String filePath, Tensor out, int idx) {
        try {
            float[] sized = ImageLoader.resized(filePath, out.height, out.width);
            System.arraycopy(sized, 0, out.data, idx * out.channel * out.height * out.width, out.channel * out.height * out.width);
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public static void testXML(String path) {
        try {
            File file = new File(path);
            //			System.out.println(file.exists());
            if (file.exists()) {
                Document doc = XmlParser.DB().parse(file);
                Element docEle = doc.getDocumentElement();
                NamedNodeMap am = docEle.getAttributes();
                System.out.println(am.getLength());
                for (int i = 0; i < am.getLength(); i++) {
                    System.out.println(am.item(i).getNodeName());
                }
                //				NodeList objList = docEle.getElementsByTagName("saml2:AttributeStatement");
                //				for(int i = 0;i<objList.getLength();i++) {
                //					Element a = (Element) objList.item(i);
                //					System.out.println(a.getAttributeNode("Name").getChildNodes().getLength());
                ////					System.out.println(a.getAttributeNode("Name").getChildNodes().item(0).getTextContent());
                //				}
                //				docEle.getElementsByTagNameNS(namespaceURI, localName);
                //
                //				for(int i = 0;i<objList.getLength();i++) {
                //					System.out.println(objList.item(i).getChildNodes().item(0).getChildNodes().item(0).getNodeValue());
                //				}
            }
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public static void main(String[] args) {
        //		String testPath = "I:\\000005.jpg";
        //
        //		String outPath = "I:\\_000005.jpg";
        //
        //		try {
        //			YoloImageUtils.resize(testPath, outPath, 448, 448);
        //		} catch (Exception e) {
        //			// TODO Auto-generated catch block
        //			e.printStackTrace();
        //		}
        //
        //		String testXML = "I:\\000005.xml";
        //
        //		int[][] label = anno2bbox(testXML);
        //		int[][] bbox = formatLabel(label);
        //		System.out.println(JsonUtils.toJson(label));
        //		System.out.println(JsonUtils.toJson(bbox));
        //		String filename = "000005.jpg";
        //		formatData(filename, imgDir, labelDir, imgOutDir, true);
        //		String rootPath = "H:\\voc\\mask\\data\\";
        //		String imgDir = rootPath + "\\JPEGImages";
        //		String labelDir = rootPath + "\\Annotations";
        //		String imgOutDir = rootPath + "\\imgs";
        //		String bboxPath = rootPath + "\\yolov3.txt";
        //
        //		image2Yolo(imgDir, labelDir, imgOutDir, bboxPath, YoloVersion.yolov3_xyxy);
        //		String rootPath = "H:\\voc\\helmet_dataset";
        //		String imgDir = rootPath + "\\JPEGImages";
        //		String labelPath = rootPath + "\\train_labels.csv";
        //		String imgOutDir = rootPath + "\\imgs";
        //		String bboxPath = rootPath + "\\labels\\yolov3.txt";
        //		String[] labelSet = new String[] {"none","white","yellow","blue","red"};
        //
        //		image2YoloFormCSV(imgDir, labelPath, labelSet, imgOutDir, bboxPath, YoloVersion.yolov3_xyxy);
        //		String rootPath = "H:\\voc\\test";
        //		String rootPath = "H:\\voc\\yz\\seal";
        //		String imgDir = rootPath + "\\Images";
        //		String labelDir = rootPath + "\\Annotations";
        //		String labelPath = rootPath + "\\labels.txt";
        //		String bboxPath = rootPath + "\\bbox.txt";
        //		String labelPath = rootPath + "\\train_labels.csv";
        //		String[] labelSet = new String[] {"none","white","yellow","blue","red"};
        //		xml2Yolo(imgDir, labelDir, labelPath, bboxPath);
        //		csv2Yolo(imgDir, labelPath, bboxPath, labelSet);
        String xmlPath = "D:\\1.xml";
        testXML(xmlPath);
    }
}

