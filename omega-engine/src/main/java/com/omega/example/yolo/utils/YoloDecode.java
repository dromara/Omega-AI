package com.omega.example.yolo.utils;

import com.omega.common.utils.JsonUtils;
import com.omega.engine.tensor.Tensor;

public class YoloDecode {
    public static int grid_size = 7;
    public static int class_number = 20;
    public static int bbox_num = 2;
    public static float thresh = 0.2f;
    public static float iou_thresh = 0.2f;

    public static float[][][] getDetection(Tensor x, int w, int h) {
        int location = grid_size * grid_size;
        int input_num_each = location * (class_number + bbox_num * (1 + 4));
        float[][][] dets = new float[x.number][location][1 + class_number + 4];
        for (int b = 0; b < x.number; b++) {
            int input_index = b * input_num_each;
            float[][] score_bbox = new float[location * 2][class_number + 4 + 1];
            for (int l = 0; l < location; ++l) {
                int row = l / grid_size;
                int col = l % grid_size;
                for (int n = 0; n < bbox_num; ++n) {
                    int confidence_index = input_index + location * class_number + l * bbox_num + n;
                    int class_index = input_index + l * class_number;
                    float scale = x.data[confidence_index];
                    int box_index = input_index + location * (class_number + bbox_num) + (l * bbox_num + n) * 4;
                    float[] bbox = new float[class_number + 1 + 4];
                    bbox[class_number + 1] = (x.data[box_index + 0] + col) / grid_size * w;
                    bbox[class_number + 2] = (x.data[box_index + 1] + row) / grid_size * h;
                    bbox[class_number + 3] = (float) (Math.pow(x.data[box_index + 2], 2.0f) * w);
                    bbox[class_number + 4] = (float) (Math.pow(x.data[box_index + 3], 2.0f) * h);
                    for (int j = 0; j < class_number; ++j) {
                        float prob = scale * x.data[class_index + j];
                        bbox[j] = (prob > thresh) ? prob : 0;
                    }
                    score_bbox[l * bbox_num + n] = bbox;
                }
            }
            /**
             * 使用nms(非极大值抑制)过滤重复和置信度低的bbox

             */
            //			nms(score_bbox);
            dets[b] = score_bbox;
        }
        return dets;
    }

    public static float[][][] getDetectionLabel(Tensor x, int w, int h) {
        int location = grid_size * grid_size;
        int input_num_each = location * (class_number + 1 + 4);
        float[][][] dets = new float[x.number][location][1 + class_number + 4];
        for (int b = 0; b < x.number; b++) {
            int input_index = b * input_num_each;
            float[][] score_bbox = new float[location][class_number + 4 + 1];
            for (int l = 0; l < location; ++l) {
                int row = l / grid_size;
                int col = l % grid_size;
                int confidence_index = input_index + l * (class_number + 1 + 4);
                int class_index = confidence_index + 1;
                float scale = x.data[confidence_index];
                int box_index = confidence_index + class_number + 1;
                float[] bbox = new float[class_number + 1 + 4];
                bbox[class_number + 1] = (x.data[box_index + 0] + col) / grid_size * w;
                bbox[class_number + 2] = (x.data[box_index + 1] + row) / grid_size * h;
                bbox[class_number + 3] = (float) (x.data[box_index + 2] * w);
                bbox[class_number + 4] = (float) (x.data[box_index + 3] * h);
                bbox[0] = scale;
                for (int j = 0; j < class_number; ++j) {
                    float prob = scale * x.data[class_index + j];
                    bbox[j + 1] = (prob > thresh) ? prob : 0;
                }
                if (scale == 1) {
                    //	            	System.out.println(x.data[box_index + 0]+":"+x.data[box_index + 1]+":"+x.data[box_index + 2]+":"+x.data[box_index + 3]);
                    System.out.println(b + ":" + JsonUtils.toJson(bbox));
                }
                score_bbox[l] = bbox;
                //	            System.out.println(JsonUtils.toJson(bbox));
            }
            /**
             * 使用nms(非极大值抑制)过滤重复和置信度低的bbox

             */
            //			nms(score_bbox);
            dets[b] = score_bbox;
        }
        return dets;
    }

    public static float[][][] getDetectionLabelYolov3(Tensor x, int w, int h) {
        int input_num_each = 90 * (1 + 4);
        float[][][] dets = new float[x.number][90][1 + class_number + 4];
        for (int b = 0; b < x.number; b++) {
            System.out.println(JsonUtils.toJson(x.getByNumber(b)));
            int input_index = b * input_num_each;
            float[][] score_bbox = new float[90][class_number + 4 + 1];
            for (int l = 0; l < 90; ++l) {
                int box_index = input_index + l * 5;
                float[] bbox = new float[class_number + 1 + 4];
                if (x.data[box_index + 2] != 0 && x.data[box_index + 3] != 0) {
                    bbox[class_number + 1] = x.data[box_index + 0] * w;
                    bbox[class_number + 2] = x.data[box_index + 1] * h;
                    bbox[class_number + 3] = x.data[box_index + 2] * w;
                    bbox[class_number + 4] = x.data[box_index + 3] * h;
                    bbox[0] = 1;
                    System.out.println(b + ":" + JsonUtils.toJson(bbox));
                }
                score_bbox[l] = bbox;
            }
            /**
             * 使用nms(非极大值抑制)过滤重复和置信度低的bbox

             */
            //			nms(score_bbox);
            dets[b] = score_bbox;
        }
        return dets;
    }

    public static float[][][] getDetection(Tensor x, int w, int h, int class_number) {
        int location = grid_size * grid_size;
        int input_num_each = location * (class_number + bbox_num * (1 + 4));
        float[][][] dets = new float[x.number][location][1 + class_number + 4];
        for (int b = 0; b < x.number; b++) {
            int input_index = b * input_num_each;
            float[][] score_bbox = new float[location * 2][class_number + 4 + 1];
            for (int l = 0; l < location; ++l) {
                int row = l / grid_size;
                int col = l % grid_size;
                for (int n = 0; n < bbox_num; ++n) {
                    int confidence_index = input_index + location * class_number + l * bbox_num + n;
                    int class_index = input_index + l * class_number;
                    float scale = x.data[confidence_index];
                    int box_index = input_index + location * (class_number + bbox_num) + (l * bbox_num + n) * 4;
                    float[] bbox = new float[class_number + 1 + 4];
                    bbox[class_number + 1] = (x.data[box_index + 0] + col) / grid_size * w;
                    bbox[class_number + 2] = (x.data[box_index + 1] + row) / grid_size * h;
                    bbox[class_number + 3] = (float) (Math.pow(x.data[box_index + 2], 2.0f) * w);
                    bbox[class_number + 4] = (float) (Math.pow(x.data[box_index + 3], 2.0f) * h);
                    for (int j = 0; j < class_number; ++j) {
                        float prob = scale * x.data[class_index + j];
                        bbox[j] = (prob > thresh) ? prob : 0;
                    }
                    score_bbox[l * bbox_num + n] = bbox;
                }
            }
            /**
             * 使用nms(非极大值抑制)过滤重复和置信度低的bbox

             */
            //			nms(score_bbox);
            dets[b] = score_bbox;
        }
        return dets;
    }

    public static void quickSort(float[][] arr, int clazz, int start, int end) {
        if (start < end) {
            // 把数组中的首位数字作为基准数
            float stard = arr[start][clazz];
            // 记录需要排序的下标
            int low = start;
            int high = end;
            // 循环找到比基准数大的数和比基准数小的数
            while (low < high) {
                // 右边的数字比基准数大
                while (low < high && arr[high][clazz] >= stard) {
                    high--;
                }
                // 使用右边的数替换左边的数
                arr[low] = arr[high];
                // 左边的数字比基准数小
                while (low < high && arr[low][clazz] <= stard) {
                    low++;
                }
                // 使用左边的数替换右边的数
                arr[high] = arr[low];
            }
            // 把标准值赋给下标重合的位置
            arr[low] = arr[start];
            // 处理所有小的数字
            quickSort(arr, clazz, start, low);
            // 处理所有大的数字
            quickSort(arr, clazz, low + 1, end);
        }
    }

    public static void nms(float[][] score_bbox) {
        int location = grid_size * grid_size;
        for (int c = 0; c < class_number; c++) {
            quickSort(score_bbox, c, 0, score_bbox.length - 1);
            if (score_bbox[0][c] <= 0.0f) {
                continue;
            }
            for (int l = 0; l < location; l++) {
                float[] ba = new float[4];
                ba[0] = score_bbox[l][class_number + 1];
                ba[1] = score_bbox[l][class_number + 2];
                ba[2] = score_bbox[l][class_number + 3];
                ba[3] = score_bbox[l][class_number + 4];
                for (int i = l + 1; i < location; i++) {
                    float[] bb = new float[4];
                    bb[0] = score_bbox[i][class_number + 1];
                    bb[1] = score_bbox[i][class_number + 2];
                    bb[2] = score_bbox[i][class_number + 3];
                    bb[3] = score_bbox[i][class_number + 4];
                    float iou = YoloUtils.box_iou(ba, bb);
                    if (iou >= iou_thresh) {
                        score_bbox[i][c] = 0.0f;
                    }
                }
            }
        }
    }
}

