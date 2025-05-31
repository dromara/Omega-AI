package com.omega.example.yolo.loss;

import com.omega.common.utils.MatrixUtils;
import com.omega.engine.loss.LossFunction;
import com.omega.engine.loss.LossType;
import com.omega.engine.tensor.Tensor;
import com.omega.example.yolo.utils.YoloUtils;

/**
 * YoloLoss
 *
 * @author Administrator
 */
public class YoloLoss extends LossFunction {
    private static YoloLoss instance;
    public final LossType lossType = LossType.yolo;
    private int grid_number = 7;
    private int class_number = 1;
    private int bbox_num = 2;
    private Tensor loss;
    private Tensor diff;
    private float noobject_scale = 0.5f;
    private float coord_scale = 5.0f;
    private float class_scale = 1.0f;
    private float object_scale = 1.0f;

    public YoloLoss(int class_number) {
        this.class_number = class_number;
    }

    public static YoloLoss operation(int class_number) {
        if (instance == null) {
            instance = new YoloLoss(class_number);
        }
        return instance;
    }

    public void init(Tensor input) {
        if (loss == null || input.number != this.diff.number) {
            this.loss = new Tensor(1, 1, 1, 1);
            this.diff = new Tensor(input.number, input.channel, input.height, input.width, true);
        } else {
            MatrixUtils.zero(this.diff.data);
        }
    }

    /**
     * loss = coor_error + iou_error + class_error
     */
    @Override
    public Tensor loss(Tensor x, Tensor label) {
        // TODO Auto-generated method stub
        init(x);
        if (x.isHasGPU()) {
            x.syncHost();
        }
        int location = grid_number * grid_number;
        int input_num_each = location * (class_number + bbox_num * (1 + 4));
        int truth_num_each = location * (1 + class_number + 4);
        int count = 0;
        float avg_iou = 0;
        float avg_cat = 0;
        float avg_allcat = 0;
        float avg_obj = 0;
        float avg_anyobj = 0;
        float cost = 0.0f;
        for (int b = 0; b < x.number; b++) {
            //			System.out.println(JsonUtils.toJson(x.getByNumber(b)));
            int input_index = b * input_num_each;
            for (int l = 0; l < location; l++) {
                for (int n = 0; n < bbox_num; n++) {
                    int confidence_index = input_index + location * class_number + l * bbox_num + n;
                    this.diff.data[confidence_index] = noobject_scale * (x.data[confidence_index]);
                    cost += noobject_scale * Math.pow(x.data[confidence_index], 2.0f);
                    avg_anyobj += x.data[confidence_index];
                }
                int truth_index = (class_number + 4 + 1) * l + b * truth_num_each;
                if (label.data[truth_index] != 1.0f) {
                    continue;
                }
                //				System.out.println(truth_index+":"+label.data[truth_index]);
                //计算loss函数中的第5项，每个预测类型的概率误差
                int class_index = input_index + l * class_number;
                for (int j = 0; j < class_number; ++j) {
                    cost += class_scale * Math.pow(x.data[class_index + j] - label.data[truth_index + 1 + j], 2);
                    this.diff.data[class_index + j] = class_scale * (x.data[class_index + j] - label.data[truth_index + 1 + j]);
                    if (label.data[truth_index + 1 + j] == 1.0f) {
                        avg_cat += x.data[class_index + j];
                    }
                    avg_allcat += x.data[class_index + j];
                }
                //获取gt bbox
                float[] truthCoords = new float[4];
                truthCoords[0] = label.data[truth_index + 1 + class_number + 0] / grid_number;
                truthCoords[1] = label.data[truth_index + 1 + class_number + 1] / grid_number;
                truthCoords[2] = label.data[truth_index + 1 + class_number + 2];
                truthCoords[3] = label.data[truth_index + 1 + class_number + 3];
                //	            System.out.println(JsonUtils.toJson(truthCoords));
                int n_best = -1;   //存储两个候选框最好的，只使用此候选框进行回归计算
                float best_iou = 0;
                float best_square = 20;
                for (int n = 0; n < bbox_num; n++) {
                    int inputCoordsIndex = input_index + (class_number + bbox_num) * location + (l * bbox_num + n) * 4;
                    float[] bbox = new float[4];
                    bbox[0] = x.data[inputCoordsIndex + 0] / grid_number;  //x
                    bbox[1] = x.data[inputCoordsIndex + 1] / grid_number;  //y
                    bbox[2] = x.data[inputCoordsIndex + 2] * x.data[inputCoordsIndex + 2];  //w = w*w
                    bbox[3] = x.data[inputCoordsIndex + 3] * x.data[inputCoordsIndex + 3];  //h = h*h
                    float iou = YoloUtils.box_iou(bbox, truthCoords);
                    float rmse = YoloUtils.box_rmse(bbox, truthCoords);
                    //找到最接近truth标注的框
                    if (iou > 0 || best_iou > 0) {
                        if (iou > best_iou) {
                            n_best = n;
                            best_iou = iou;
                        }
                    } else {
                        if (rmse < best_square) {
                            n_best = n;
                            best_square = rmse;
                        }
                    }
                }
                //计算x,y,w,h的损失，挑选最优的框
                int best_coords = input_index + (class_number + bbox_num) * location + (l * bbox_num + n_best) * 4;
                int t_bbox_index = truth_index + 1 + class_number;
                avg_iou += best_iou;
                cost += coord_scale * Math.pow(x.data[best_coords + 0] - label.data[t_bbox_index + 0], 2);
                cost += coord_scale * Math.pow(x.data[best_coords + 1] - label.data[t_bbox_index + 1], 2);
                cost += coord_scale * Math.pow(x.data[best_coords + 2] - Math.sqrt(label.data[t_bbox_index + 2]), 2);
                cost += coord_scale * Math.pow(x.data[best_coords + 3] - Math.sqrt(label.data[t_bbox_index + 3]), 2);
                //	            cost += Math.pow(1.0f - best_iou, 2.0f);
                this.diff.data[best_coords + 0] = coord_scale * (x.data[best_coords + 0] - label.data[t_bbox_index + 0]);
                this.diff.data[best_coords + 1] = coord_scale * (x.data[best_coords + 1] - label.data[t_bbox_index + 1]);
                this.diff.data[best_coords + 2] = (float) (coord_scale * (x.data[best_coords + 2] - Math.sqrt(label.data[t_bbox_index + 2])));
                this.diff.data[best_coords + 3] = (float) (coord_scale * (x.data[best_coords + 3] - Math.sqrt(label.data[t_bbox_index + 3])));
                //计算loss函数第3项
                //先减去计算第4项lost时多加上的有物体的网格的置信度
                int confidence_index = input_index + location * class_number + l * bbox_num + n_best;
                cost -= noobject_scale * Math.pow(0.0f - x.data[confidence_index], 2);
                cost += object_scale * Math.pow(x.data[confidence_index] - 1.0f, 2);
                this.diff.data[confidence_index] = object_scale * (x.data[confidence_index] - 1.0f);
                //	            this.diff.data[confidence_index] = object_scale*(x.data[confidence_index] - best_iou);
                avg_obj += x.data[confidence_index];
                count++;
            }
        }
        System.out.println("Detection Avg IOU:" + avg_iou / count + ",Pos Cat:" + avg_cat / count + ",All Cat:" + avg_allcat / (class_number * count) + ",Pos Obj:" + avg_obj / count + ",Any Obj:" + avg_anyobj / (location * x.number * bbox_num) + ",count:" + count);
        this.loss.data[0] = cost;
        //		System.out.println(JsonUtils.toJson(x.data));
        //		System.out.println(JsonUtils.toJson(this.diff.data));
        return loss;
    }

    @Override
    public Tensor diff(Tensor x, Tensor label) {
        // TODO Auto-generated method stub
        if (diff.isHasGPU()) {
            diff.hostToDevice();
        }
        //		System.out.println(diff);
        return diff;
    }

    @Override
    public LossType getLossType() {
        // TODO Auto-generated method stub
        return LossType.yolo;
    }

    @Override
    public Tensor[] loss(Tensor[] x, Tensor label) {
        // TODO Auto-generated method stub
        return null;
    }

    @Override
    public Tensor[] diff(Tensor[] x, Tensor label) {
        // TODO Auto-generated method stub
        return null;
    }

    @Override
    public Tensor loss(Tensor x, Tensor label, Tensor loss) {
        // TODO Auto-generated method stub
        return null;
    }

    @Override
    public Tensor diff(Tensor x, Tensor label, Tensor diff) {
        // TODO Auto-generated method stub
        return null;
    }

    @Override
    public Tensor loss(Tensor x, Tensor label, int igonre) {
        // TODO Auto-generated method stub
        return null;
    }

    @Override
    public Tensor diff(Tensor x, Tensor label, int igonre) {
        // TODO Auto-generated method stub
        return null;
    }

    @Override
    public Tensor diff(Tensor x, Tensor label, int igonre, int count) {
        // TODO Auto-generated method stub
        return null;
    }
}

