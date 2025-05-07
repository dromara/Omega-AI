package com.omega.example.yolo.data;

import com.omega.common.data.Tensor;
import com.omega.common.utils.ImageUtils;
import com.omega.common.utils.MatrixOperation;
import com.omega.example.yolo.utils.YoloDecode;

import java.util.Map;

/**
 * @author Administrator
 */
public class YoloDataTransform2 extends DataTransform {
    private int classnum = 1;
    private int numBoxes = 7;
    private DataType dataType;

    public YoloDataTransform2(int classnum, DataType dataType, int numBoxes) {
        this.classnum = classnum;
        this.dataType = dataType;
        this.numBoxes = numBoxes;
    }

    @Override
    public void transform(Tensor input, Tensor label, String[] idxSet, int[] indexs, Map<String, float[]> orgLabelData) {
        // TODO Auto-generated method stub
        label.clear();
        YoloDataTransformJob.transform(input, label, idxSet, indexs, orgLabelData, this.dataType, numBoxes);
    }

    @Override
    public void showTransform(String outputPath, Tensor input, Tensor label, String[] idxSet, int[] indexs, Map<String, float[]> orgLabelData) {
        // TODO Auto-generated method stub
        this.transform(input, label, idxSet, indexs, orgLabelData);
        ImageUtils utils = new ImageUtils();
        input.data = MatrixOperation.multiplication(input.data, 255.0f);
        float[][][] draw_bbox = null;
        if (this.dataType == DataType.yolov1) {
            draw_bbox = YoloDecode.getDetectionLabel(label, input.width, input.height);
        } else {
            draw_bbox = YoloDecode.getDetectionLabelYolov3(label, input.width, input.height);
        }
        for (int b = 0; b < input.number; b++) {
            float[] once = input.getByNumber(b);
            int count = 0;
            for (float[] la : draw_bbox[b]) {
                if (la[0] == 1) {
                    count++;
                }
            }
            int[][] bbox = new int[count][5];
            int index = 0;
            for (float[] la : draw_bbox[b]) {
                if (la[0] == 1) {
                    bbox[index][0] = 0;
                    bbox[index][1] = (int) la[1 + classnum];
                    bbox[index][2] = (int) la[2 + classnum];
                    bbox[index][3] = (int) la[3 + classnum];
                    bbox[index][4] = (int) la[4 + classnum];
                    int cx = bbox[index][1];
                    int cy = bbox[index][2];
                    int w = bbox[index][3];
                    int h = bbox[index][4];
                    bbox[index][1] = cx - w / 2;
                    bbox[index][2] = cy - h / 2;
                    bbox[index][3] = cx + w / 2;
                    bbox[index][4] = cy + h / 2;
                    index++;
                }
            }
            utils.createRGBImage(outputPath + b + ".png", "png", ImageUtils.color2rgb2(once, input.height, input.width), input.width, input.height, bbox);
        }
    }
}

