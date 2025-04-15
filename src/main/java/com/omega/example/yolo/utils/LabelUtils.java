package com.omega.example.yolo.utils;

import com.omega.common.config.Tensor;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.List;

/**
 * yolo label transform to the location
 *
 * @author Administrator
 */
public class LabelUtils {
    public static void loadBoxCSV(String labelPath, Tensor box) {
        try (FileInputStream fin = new FileInputStream(labelPath); InputStreamReader reader = new InputStreamReader(fin); BufferedReader buffReader = new BufferedReader(reader);) {
            String strTmp = "";
            int idx = 0;
            int onceSize = box.channel * box.height * box.width;
            while ((strTmp = buffReader.readLine()) != null) {
                if (idx > 0) {
                    String[] list = strTmp.split(" ");
                    for (int i = 2; i < list.length; i++) {
                        box.data[(idx - 1) * onceSize + i - 2] = Float.parseFloat(list[i]);
                    }
                }
                idx++;
            }
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public static Tensor loadBoxTXT(String labelPath) {
        try (FileInputStream fin = new FileInputStream(labelPath); InputStreamReader reader = new InputStreamReader(fin); BufferedReader buffReader = new BufferedReader(reader);) {
            String strTmp = "";
            int idx = 0;
            int count = 1;
            List<Float> flist = new ArrayList<Float>();
            while ((strTmp = buffReader.readLine()) != null) {
                if (idx > 0) {
                    String[] list = strTmp.split(" ");
                    int page = (list.length - 1) / 5;
                    for (int p = 0; p < page; p++) {
                        for (int i = 0; i < 5; i++) {
                            int index = p * 5 + i + 1;
                            flist.add(Float.parseFloat(list[index]));
                        }
                        count++;
                    }
                }
                idx++;
            }
            float[] r = new float[flist.size()];
            for (int i = 0; i < r.length; i++) {
                r[i] = flist.get(i);
            }
            Tensor result = new Tensor(count, 1, 1, 4, r);
            return result;
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
        return null;
    }

    public static void loadLabelCSV(String labelPath, Tensor label, String[] idxs) {
        try (FileInputStream fin = new FileInputStream(labelPath); InputStreamReader reader = new InputStreamReader(fin); BufferedReader buffReader = new BufferedReader(reader);) {
            String strTmp = "";
            int idx = 0;
            int onceSize = label.channel * label.height * label.width;
            while ((strTmp = buffReader.readLine()) != null) {
                //	        	System.out.println(strTmp);
                if (idx > 0) {
                    String[] list = strTmp.split(",");
                    idxs[idx - 1] = list[0];
                    for (int i = 1; i < list.length; i++) {
                        label.data[(idx - 1) * onceSize + i - 1] = Float.parseFloat(list[i]);
                    }
                }
                idx++;
            }
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public static void loadLabel(String labelPath, Tensor label) {
        try (FileInputStream fin = new FileInputStream(labelPath); InputStreamReader reader = new InputStreamReader(fin); BufferedReader buffReader = new BufferedReader(reader);) {
            String strTmp = "";
            int idx = 0;
            int onceSize = label.channel * label.height * label.width;
            while ((strTmp = buffReader.readLine()) != null) {
                String[] list = strTmp.split(" ");
                for (int i = 1; i < list.length; i++) {
                    label.data[idx * onceSize + i - 1] = Float.parseFloat(list[i]);
                }
                idx++;
            }
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    /**
     * labelToLocation
     *
     * @param wmax
     * @param wmin
     * @param hmax
     * @param hmin
     * @param cla    = 20
     * @param stride = 7
     * @param wimg   = 448
     * @param himg   = 448
     * @return 7 * 7 * (2 + 8 + 20) = 1470
     * <p>
     * target = [cx1,cy1,w1,h1,c1,cx2,cy2,w2,h2,c2,clazz1.....,clazz20]
     * <p>
     * w = wmax - wmin
     * <p>
     * h = hmax - hmin
     * <p>
     * cx = (wmax + wmin) / 2
     * <p>
     * cy = (hmax + hmin) / 2
     * <p>
     * gridx = int(cx / stride)
     * <p>
     * gridy = int(cy / stride)
     * <p>
     * x = (cx - (gridx * cellSize)) / cellSize
     * <p>
     * y = (cy - (gridy * cellSize)) / cellSize
     */
    public static float[] labelToLocation(int wmax, int wmin, int hmax, int hmin, int cla, int stride) {
        float cellSize = 1.0f / stride;
        float[] target = new float[stride * stride * 30];
        float w = wmax - wmin;
        float h = hmax - hmin;
        float cx = (wmax + wmin) / 2;
        float cy = (hmax + hmin) / 2;
        int gridx = new BigDecimal(cx).divide(new BigDecimal(cellSize), BigDecimal.ROUND_CEILING).intValue() - 1;
        int gridy = new BigDecimal(cy).divide(new BigDecimal(cellSize), BigDecimal.ROUND_CEILING).intValue() - 1;
        /**
         * c1

         */
        target[gridx * stride * 30 + gridy * 30 + 0] = 1.0f;
        /**
         * c2

         */
        target[gridx * stride * 30 + gridy * 30 + 5] = 1.0f;
        float x = cx / cellSize - gridx;
        float y = cy / cellSize - gridx;
        /**
         * x1,y1,w1,h1

         */
        target[gridx * stride * 30 + gridy * 30 + 1] = x;
        target[gridx * stride * 30 + gridy * 30 + 2] = y;
        target[gridx * stride * 30 + gridy * 30 + 3] = w;
        target[gridx * stride * 30 + gridy * 30 + 4] = h;
        /**
         * x2,y2,w2,h2

         */
        target[gridx * stride * 30 + gridy * 30 + 6] = x;
        target[gridx * stride * 30 + gridy * 30 + 7] = y;
        target[gridx * stride * 30 + gridy * 30 + 8] = w;
        target[gridx * stride * 30 + gridy * 30 + 9] = h;
        /**
         * class

         */
        target[gridx * stride * 30 + gridy * 30 + cla + 9] = 1.0f;
        return target;
    }

    /**
     * labelToLocation
     *
     * @param cx
     * @param cy
     * @param w
     * @param h
     * @param cla    = 20
     * @param stride = 7
     * @param wimg   = 448
     * @param himg   = 448
     * @return 7 * 7 * (2 + 8 + 20) = 1470
     * <p>
     * target = [px1,py1,w1,h1,c1,px2,py2,w2,h2,c2,clazz1.....,clazz20]
     * <p>
     * gridx = int(cx / stride)
     * <p>
     * gridy = int(cy / stride)
     * <p>
     * px = (cx - (gridx * cellSize)) / cellSize
     * <p>
     * py = (cy - (gridy * cellSize)) / cellSize
     */
    public static float[] labelToYolo(int cx, int cy, int w, int h, int cla, int stride) {
        float cellSize = 1.0f / stride;
        float[] target = new float[stride * stride * 30];
        int gridx = new BigDecimal(cx).divide(new BigDecimal(cellSize), BigDecimal.ROUND_CEILING).intValue() - 1;
        int gridy = new BigDecimal(cy).divide(new BigDecimal(cellSize), BigDecimal.ROUND_CEILING).intValue() - 1;
        /**
         * c1

         */
        target[gridx * stride * 30 + gridy * 30 + 0] = 1.0f;
        /**
         * c2

         */
        target[gridx * stride * 30 + gridy * 30 + 5] = 1.0f;
        float px = cx / cellSize - gridx;
        float py = cy / cellSize - gridy;
        /**
         * x1,y1,w1,h1

         */
        target[gridx * stride * 30 + gridy * 30 + 1] = px;
        target[gridx * stride * 30 + gridy * 30 + 2] = py;
        target[gridx * stride * 30 + gridy * 30 + 3] = w;
        target[gridx * stride * 30 + gridy * 30 + 4] = h;
        /**
         * x2,y2,w2,h2

         */
        target[gridx * stride * 30 + gridy * 30 + 6] = px;
        target[gridx * stride * 30 + gridy * 30 + 7] = py;
        target[gridx * stride * 30 + gridy * 30 + 8] = w;
        target[gridx * stride * 30 + gridy * 30 + 9] = h;
        /**
         * class

         */
        target[gridx * stride * 30 + gridy * 30 + cla + 9] = 1.0f;
        return target;
    }

    /**
     * labelToLocation
     *
     * @param cx
     * @param cy
     * @param w
     * @param h
     * @param cla    = 20
     * @param stride = 7
     * @param wimg   = 448
     * @param himg   = 448
     * @return 7 * 7 * (2 + 8 + 20) = 1470
     * <p>
     * target = [px1,py1,w1,h1,c1,px2,py2,w2,h2,c2,clazz1.....,clazz20]
     * <p>
     * gridx = int(cx / stride)
     * <p>
     * gridy = int(cy / stride)
     * <p>
     * px = (cx - (gridx * cellSize)) / cellSize
     * <p>
     * py = (cy - (gridy * cellSize)) / cellSize
     */
    public static float[] labelToYolo(int[][] bbox, int stride, int im_w) {
        //		float[][] bbox = normalization(data);
        float[] target = new float[stride * stride * 25];
        //		System.out.println(JsonUtils.toJson(bbox));
        for (int i = 0; i < bbox.length; i++) {
            float x1 = bbox[i][1];
            float y1 = bbox[i][2];
            float x2 = bbox[i][3];
            float y2 = bbox[i][4];
            float cx = (x1 + x2) / (2 * im_w);
            float cy = (y1 + y2) / (2 * im_w);
            float w = (x2 - x1) / im_w;
            float h = (y2 - y1) / im_w;
            int gridx = (int) (cx * stride);
            int gridy = (int) (cy * stride);
            float px = cx * stride - gridx;
            float py = cy * stride - gridy;
            int clazz = new Float(bbox[i][0]).intValue();
            /**
             * c1

             */
            target[gridx * stride * 25 + gridy * 25 + 0] = 1.0f;
            /**
             * class

             */
            target[gridx * stride * 25 + gridy * 25 + 1 + clazz] = 1.0f;
            /**
             * x1,y1,w1,h1

             */
            target[gridx * stride * 25 + gridy * 25 + 21 + 0] = px;
            target[gridx * stride * 25 + gridy * 25 + 21 + 1] = py;
            target[gridx * stride * 25 + gridy * 25 + 21 + 2] = w;
            target[gridx * stride * 25 + gridy * 25 + 21 + 3] = h;
        }
        return target;
    }

    public static float[] labelToYoloV3(int[][] bbox, int im_w) {
        float[] target = new float[5 * bbox.length];
        for (int i = 0; i < bbox.length; i++) {
            float clazz = new Float(bbox[i][0]).intValue();
            float cx = bbox[i][1];
            float cy = bbox[i][2];
            float w = bbox[i][3];
            float h = bbox[i][4];
            //			cx = cx / im_w;
            //			cy = cy / im_w;
            //			w = w / im_w;
            //			h = h / im_w;
            target[i * 5 + 0] = clazz;
            target[i * 5 + 1] = cx;
            target[i * 5 + 2] = cy;
            target[i * 5 + 3] = w;
            target[i * 5 + 4] = h;
        }
        return target;
    }

    public static float[] labelToYoloV3_xyxy(int[][] bbox, int im_w) {
        float[] target = new float[5 * bbox.length];
        for (int i = 0; i < bbox.length; i++) {
            float clazz = new Float(bbox[i][0]).intValue();
            float cx = bbox[i][1];
            float cy = bbox[i][2];
            float w = bbox[i][3];
            float h = bbox[i][4];
            //			cx = cx / im_w;
            //			cy = cy / im_w;
            //			w = w / im_w;
            //			h = h / im_w;
            float x1 = cx - w / 2;
            float y1 = cy - h / 2;
            float x2 = cx + w / 2;
            float y2 = cy + h / 2;
            target[i * 5 + 0] = clazz;
            target[i * 5 + 1] = x1;
            target[i * 5 + 2] = y1;
            target[i * 5 + 3] = x2;
            target[i * 5 + 4] = y2;
        }
        return target;
    }

    public static float[][] normalization(int[][] data) {
        float[][] bbox = new float[data.length][data[0].length];
        for (int i = 0; i < bbox.length; i++) {
            for (int j = 0; j < bbox[i].length; j++) {
                bbox[i][j] = data[i][j] * 1.0f / 448.0f;
            }
        }
        return bbox;
    }
}

