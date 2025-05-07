package com.omega.example.yolo.utils;

import com.omega.common.data.Tensor;
import com.omega.common.data.utils.DataTransforms;
import com.omega.engine.nn.data.DataSet;

public class YoloLabelUtils {
    public float[] mean = new float[]{0.491f, 0.482f, 0.446f};
    public float[] std = new float[]{0.247f, 0.243f, 0.261f};
    private DataTransforms t;
    private int classNum = 1;
    private int bboxSize = 4;

    public YoloLabelUtils(int classNum, int bboxSize) {
        this.classNum = classNum;
        this.bboxSize = bboxSize;
    }

    public static void formatToYoloV3(YoloDataLoader dataLoader, int im_w, int im_h, boolean format) {
        if (format) {
            dataLoader.getLabelSet().data = labelToYoloV3(dataLoader.getLabelSet(), dataLoader.getLabelSet().number, im_w, im_h);
        } else {
            dataLoader.getLabelSet().data = labelToYoloV3NotFormat(dataLoader.getLabelSet(), dataLoader.getLabelSet().number, im_w, im_h);
        }
    }

    public static DataSet formatToYoloV3(DataSet dataSet, int im_w, int im_h) {
        dataSet.label.data = labelToYoloV3(dataSet.label, dataSet.label.number, im_w, im_h);
        return dataSet;
    }

    public static DataSet formatToYolo(DataSet dataSet, int im_w, int im_h) {
        dataSet.label.data = labelToYolo(dataSet.label.data, dataSet.label.number, 7, im_w, im_h);
        dataSet.label.width = 7 * 7 * 6;
        dataSet.labelSize = 7 * 7 * 6;
        return dataSet;
    }

    public static Tensor formatToYolo(Tensor label, int im_w, int im_h) {
        label.data = labelToYolo(label.data, label.number, 7, im_w, im_h);
        label.width = 7 * 7 * 6;
        return label;
    }

    public static Tensor formatToYolo(Tensor label, int im_w, int im_h, int classNum) {
        label.data = labelToYolo(label.data, label.number, classNum, 7, im_w, im_h);
        label.width = 7 * 7 * (5 + classNum);
        return label;
    }

    public static DataSet formatToYoloSetClass(DataSet dataSet, int classNum, int im_w, int im_h, boolean format) {
        if (format) {
            dataSet.label.data = labelToYolo(dataSet.label.data, dataSet.label.number, classNum, 7, im_w, im_h);
        } else {
            dataSet.label.data = labelToYoloNotFormat(dataSet.label.data, dataSet.label.number, classNum, 7, im_w, im_h);
        }
        dataSet.label.width = 7 * 7 * (5 + classNum);
        return dataSet;
    }

    public static void formatToYoloSetClass(YoloDataLoader dataLoader, int classNum, int im_w, int im_h, boolean format) {
        if (format) {
            dataLoader.getLabelSet().data = labelToYolo(dataLoader.getLabelSet(), dataLoader.getLabelSet().number, classNum, 7, im_w, im_h);
        } else {
            dataLoader.getLabelSet().data = labelToYoloNotFormat(dataLoader.getLabelSet(), dataLoader.getLabelSet().number, classNum, 7, im_w, im_h);
        }
        dataLoader.getLabelSet().width = 7 * 7 * (5 + classNum);
        dataLoader.labelSize = 7 * 7 * (5 + classNum);
    }

    public static Tensor formatToYoloV3(Tensor label, int im_w, int im_h) {
        label.data = labelToYoloV3(label, label.number, im_w, im_h);
        //		System.out.println(JsonUtils.toJson(label.getByNumber(0)));
        return label;
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
    public static float[] labelToYolo(Tensor label, int number, int stride, int im_w, int im_h) {
        float[] target = new float[number * stride * stride * 6];
        int oneSize = stride * stride * 6;
        for (int i = 0; i < number; i++) {
            float x1 = label.data[i * 5 + 1];
            float y1 = label.data[i * 5 + 2];
            float x2 = label.data[i * 5 + 3];
            float y2 = label.data[i * 5 + 4];
            float cx = (x1 + x2) / (2 * im_w);
            float cy = (y1 + y2) / (2 * im_h);
            float w = (x2 - x1) / im_w;
            float h = (y2 - y1) / im_h;
            int gridx = (int) (cx * stride);
            int gridy = (int) (cy * stride);
            float px = cx * stride - gridx;
            float py = cy * stride - gridy;
            /**
             * c1

             */
            target[i * oneSize + gridy * stride * 6 + gridx * 6 + 0] = 1.0f;
            /**
             * class

             */
            target[i * oneSize + gridy * stride * 6 + gridx * 6 + 1] = 1.0f;
            /**
             * x1,y1,w1,h1

             */
            target[i * oneSize + gridy * stride * 6 + gridx * 6 + 2] = px;
            target[i * oneSize + gridy * stride * 6 + gridx * 6 + 3] = py;
            target[i * oneSize + gridy * stride * 6 + gridx * 6 + 4] = w;
            target[i * oneSize + gridy * stride * 6 + gridx * 6 + 5] = h;
        }
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
    public static float[] labelToYolo(float[] bbox, int number, int stride, int im_w, int im_h) {
        float[] target = new float[number * stride * stride * 6];
        int oneSize = stride * stride * 6;
        for (int i = 0; i < number; i++) {
            float x1 = bbox[i * 5 + 1];
            float y1 = bbox[i * 5 + 2];
            float x2 = bbox[i * 5 + 3];
            float y2 = bbox[i * 5 + 4];
            float cx = (x1 + x2) / (2 * im_w);
            float cy = (y1 + y2) / (2 * im_h);
            float w = (x2 - x1) / im_w;
            float h = (y2 - y1) / im_h;
            int gridx = (int) (cx * stride);
            int gridy = (int) (cy * stride);
            float px = cx * stride - gridx;
            float py = cy * stride - gridy;
            /**
             * c1

             */
            target[i * oneSize + gridy * stride * 6 + gridx * 6 + 0] = 1.0f;
            /**
             * class

             */
            target[i * oneSize + gridy * stride * 6 + gridx * 6 + 1] = 1.0f;
            /**
             * x1,y1,w1,h1

             */
            target[i * oneSize + gridy * stride * 6 + gridx * 6 + 2] = px;
            target[i * oneSize + gridy * stride * 6 + gridx * 6 + 3] = py;
            target[i * oneSize + gridy * stride * 6 + gridx * 6 + 4] = w;
            target[i * oneSize + gridy * stride * 6 + gridx * 6 + 5] = h;
        }
        return target;
    }

    public static float[] labelToYolo(float[] bbox, int number, int classNum, int stride, int im_w, int im_h) {
        int once = (5 + classNum);
        float[] target = new float[number * stride * stride * once];
        int oneSize = stride * stride * once;
        for (int i = 0; i < number; i++) {
            int clazz = (int) bbox[i * 5 + 0] + 1;
            float x1 = bbox[i * 5 + 1];
            float y1 = bbox[i * 5 + 2];
            float x2 = bbox[i * 5 + 3];
            float y2 = bbox[i * 5 + 4];
            float cx = (x1 + x2) / (2 * im_w);
            float cy = (y1 + y2) / (2 * im_h);
            float w = (x2 - x1) / im_w;
            float h = (y2 - y1) / im_h;
            int gridx = (int) (cx * stride);
            int gridy = (int) (cy * stride);
            float px = cx * stride - gridx;
            float py = cy * stride - gridy;
            /**
             * c1

             */
            target[i * oneSize + gridy * stride * once + gridx * once + 0] = 1.0f;
            /**
             * class

             */
            target[i * oneSize + gridy * stride * once + gridx * once + clazz] = 1.0f;
            /**
             * x1,y1,w1,h1

             */
            target[i * oneSize + gridy * stride * once + gridx * once + 2] = px;
            target[i * oneSize + gridy * stride * once + gridx * once + 3] = py;
            target[i * oneSize + gridy * stride * once + gridx * once + 4] = w;
            target[i * oneSize + gridy * stride * once + gridx * once + 5] = h;
        }
        return target;
    }

    public static float[] labelToYolo(Tensor bbox, int number, int classNum, int stride, int im_w, int im_h) {
        int width = bbox.getWidth();  //450
        int boxNum = width / 5; //90
        int once = (5 + classNum);
        float[] target = new float[number * stride * stride * once];
        int oneSize = stride * stride * once;
        for (int b = 0; b < number; b++) {
            for (int n = 0; n < boxNum; n++) {
                int clazz = (int) bbox.data[b * width + n * 5 + 0] + 1;
                float x1 = bbox.data[b * width + n * 5 + 1];
                float y1 = bbox.data[b * width + n * 5 + 2];
                float x2 = bbox.data[b * width + n * 5 + 3];
                float y2 = bbox.data[b * width + n * 5 + 4];
                float cx = (x1 + x2) / (2 * im_w);
                float cy = (y1 + y2) / (2 * im_h);
                float w = (x2 - x1) / im_w;
                float h = (y2 - y1) / im_h;
                if (w == 0 || h == 0) {
                    break;
                }
                int gridx = (int) (cx * stride);
                int gridy = (int) (cy * stride);
                float px = cx * stride - gridx;
                float py = cy * stride - gridy;
                /**
                 * c1

                 */
                target[b * oneSize + gridy * stride * once + gridx * once + 0] = 1.0f;
                /**
                 * class

                 */
                target[b * oneSize + gridy * stride * once + gridx * once + clazz] = 1.0f;
                /**
                 * x1,y1,w1,h1

                 */
                target[b * oneSize + gridy * stride * once + gridx * once + 2] = px;
                target[b * oneSize + gridy * stride * once + gridx * once + 3] = py;
                target[b * oneSize + gridy * stride * once + gridx * once + 4] = w;
                target[b * oneSize + gridy * stride * once + gridx * once + 5] = h;
            }
        }
        return target;
    }

    public static float[] labelToYoloNotFormat(float[] bbox, int number, int classNum, int stride, int im_w, int im_h) {
        int once = (5 + classNum);
        float[] target = new float[number * stride * stride * once];
        int oneSize = stride * stride * once;
        for (int i = 0; i < number; i++) {
            int clazz = (int) bbox[i * 5 + 0] + 1;
            float cx = bbox[i * 5 + 1] / im_w;
            float cy = bbox[i * 5 + 2] / im_h;
            float w = bbox[i * 5 + 3] / im_w;
            float h = bbox[i * 5 + 4] / im_h;
            int gridx = (int) (cx * stride);
            int gridy = (int) (cy * stride);
            float px = cx * stride - gridx;
            float py = cy * stride - gridy;
            /**
             * c1

             */
            target[i * oneSize + gridy * stride * once + gridx * once + 0] = 1.0f;
            /**
             * class

             */
            target[i * oneSize + gridy * stride * once + gridx * once + clazz] = 1.0f;
            /**
             * x1,y1,w1,h1

             */
            target[i * oneSize + gridy * stride * once + gridx * once + classNum + 1] = px;
            target[i * oneSize + gridy * stride * once + gridx * once + classNum + 2] = py;
            target[i * oneSize + gridy * stride * once + gridx * once + classNum + 3] = w;
            target[i * oneSize + gridy * stride * once + gridx * once + classNum + 4] = h;
        }
        return target;
    }

    public static float[] labelToYoloNotFormat(Tensor bbox, int number, int classNum, int stride, int im_w, int im_h) {
        int width = bbox.getWidth();  //450
        int boxNum = width / 5; //90
        int once = (5 + classNum);
        float[] target = new float[number * stride * stride * once];
        int oneSize = stride * stride * once;
        for (int b = 0; b < number; b++) {
            for (int n = 0; n < boxNum; n++) {
                int clazz = (int) bbox.data[b * width + n * 5 + 0] + 1;
                float cx = bbox.data[b * width + n * 5 + 1] / im_w;
                float cy = bbox.data[b * width + n * 5 + 2] / im_h;
                float w = bbox.data[b * width + n * 5 + 3] / im_w;
                float h = bbox.data[b * width + n * 5 + 4] / im_h;
                if (w == 0 || h == 0) {
                    break;
                }
                int gridx = (int) (cx * stride);
                int gridy = (int) (cy * stride);
                float px = cx * stride - gridx;
                float py = cy * stride - gridy;
                /**
                 * c1

                 */
                target[b * oneSize + gridy * stride * once + gridx * once + 0] = 1.0f;
                /**
                 * class

                 */
                target[b * oneSize + gridy * stride * once + gridx * once + clazz] = 1.0f;
                /**
                 * x1,y1,w1,h1

                 */
                target[b * oneSize + gridy * stride * once + gridx * once + classNum + 1] = px;
                target[b * oneSize + gridy * stride * once + gridx * once + classNum + 2] = py;
                target[b * oneSize + gridy * stride * once + gridx * once + classNum + 3] = w;
                target[b * oneSize + gridy * stride * once + gridx * once + classNum + 4] = h;
            }
        }
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
     * @return number * (90 * (4 + class_num))
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
    public static float[] labelToYoloV3(Tensor label, int number, int im_w, int im_h) {
        int channel = label.width / 5;
        float[] target = new float[number * 5 * channel];
        for (int i = 0; i < number; i++) {
            for (int c = 0; c < channel; c++) {
                float clazz = label.data[i * channel * 5 + c * 5 + 0];
                float x1 = label.data[i * channel * 5 + c * 5 + 1];
                float y1 = label.data[i * channel * 5 + c * 5 + 2];
                float x2 = label.data[i * channel * 5 + c * 5 + 3];
                float y2 = label.data[i * channel * 5 + c * 5 + 4];
                float cx = (x1 + x2) / (2 * im_w);
                float cy = (y1 + y2) / (2 * im_h);
                float w = (x2 - x1) / im_w;
                float h = (y2 - y1) / im_h;
                if (cx > 0 && cy > 0 && w > 0 && h > 0) {
                    target[i * channel * 5 + c * 5 + 0] = cx;
                    target[i * channel * 5 + c * 5 + 1] = cy;
                    target[i * channel * 5 + c * 5 + 2] = w;
                    target[i * channel * 5 + c * 5 + 3] = h;
                    target[i * channel * 5 + c * 5 + 4] = clazz;
                }
            }
        }
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
     * @return number * (90 * (4 + class_num))
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
    public static float[] labelToYoloV3NotFormat(Tensor label, int number, int im_w, int im_h) {
        int channel = label.width / 5;
        float[] target = new float[number * 5 * channel];
        for (int i = 0; i < number; i++) {
            for (int c = 0; c < channel; c++) {
                float clazz = label.data[i * channel * 5 + c * 5 + 0];
                float cx = label.data[i * channel * 5 + c * 5 + 1] / im_w;
                float cy = label.data[i * channel * 5 + c * 5 + 2] / im_h;
                float w = label.data[i * channel * 5 + c * 5 + 3] / im_w;
                float h = label.data[i * channel * 5 + c * 5 + 4] / im_h;
                if (cx > 0 && cy > 0 && w > 0 && h > 0) {
                    target[i * channel * 5 + c * 5 + 0] = cx;
                    target[i * channel * 5 + c * 5 + 1] = cy;
                    target[i * channel * 5 + c * 5 + 2] = w;
                    target[i * channel * 5 + c * 5 + 3] = h;
                    target[i * channel * 5 + c * 5 + 4] = clazz;
                }
            }
        }
        return target;
    }

    public static float[] yoloTolabel(float[] target, int number, int stride, int im_w, int im_h) {
        float[] bbox = new float[number * stride * stride * 5];
        int oneSize = 5;
        int oneTargetSize = stride * stride * 6;
        for (int i = 0; i < number; i++) {
            for (int l = 0; l < stride * stride; l++) {
                float c = target[i * oneTargetSize + l * 6];
                if (c <= 0.0f) {
                    continue;
                }
                float px = target[i * oneTargetSize + l * 6 + 2];
                float py = target[i * oneTargetSize + l * 6 + 3];
                float w = target[i * oneTargetSize + l * 6 + 4];
                float h = target[i * oneTargetSize + l * 6 + 5];
                int row = l / stride;
                int col = l % stride;
                float cx = (px + col) / stride;
                float cy = (py + row) / stride;
                bbox[i * oneSize + 0] = 1.0f;
                bbox[i * oneSize + 1] = (cx - w / 2) * im_w;
                bbox[i * oneSize + 2] = (cy - h / 2) * im_h;
                bbox[i * oneSize + 3] = (cx + w / 2) * im_w;
                bbox[i * oneSize + 4] = (cy + h / 2) * im_h;
            }
        }
        return bbox;
    }

    public static void showLabel(float[] bbox, int number, int stride, int im_w, int im_h) {
        float[] target = labelToYolo(bbox, number, stride, im_w, im_h);
        float[] label = yoloTolabel(target, number, stride, im_w, im_h);
        for (int i = 0; i < number; i++) {
            System.out.println("-------" + i + "--------");
            System.out.println(bbox[i * 5 + 0] + ":" + label[i * 5 + 0]);
            System.out.println(bbox[i * 5 + 1] + ":" + label[i * 5 + 1]);
            System.out.println(bbox[i * 5 + 2] + ":" + label[i * 5 + 2]);
            System.out.println(bbox[i * 5 + 3] + ":" + label[i * 5 + 3]);
            System.out.println(bbox[i * 5 + 4] + ":" + label[i * 5 + 4]);
        }
    }

    public DataTransforms getTransformsUtils(int classNum, int bboxSize) {
        if (t == null) {
            t = new DataTransforms(classNum, bboxSize);
        }
        return t;
    }

    public void transforms(Tensor input, Tensor label) {
        /**
         * 随机裁剪

         */
        getTransformsUtils(classNum, bboxSize).randomCropWithLabel(input, label, input.height, input.width, 0.2f);
        /**
         * 随机翻转

         */
        getTransformsUtils(classNum, bboxSize).randomHorizontalFilpWithLabel(input, input, label);
        /**
         * normalize

         */
        DataTransforms.normalize(input, input, 255);
        /**
         * hsv

         */
        //		HsvUtils.hsv(input, 0.1f, 1.5f, 1.5f);
        /**
         * cutout

         */
        DataTransforms.cutout(input, input, 16, 4);
        System.out.println("data transform finish.");
    }
}

