package com.omega.example.yolo.loss;

import com.omega.common.data.Tensor;
import com.omega.common.utils.MatrixOperation;
import com.omega.common.utils.MatrixUtils;
import com.omega.engine.loss.LossFunction;
import com.omega.engine.loss.LossType;
import com.omega.example.yolo.utils.YoloUtils;

/**
 * YoloLoss
 *
 * @author Administrator
 * <p>
 * <p>
 * <p>
 * label format:
 * <p>
 * [n][maxBox = 90][box = 4][class = 1]
 * <p>
 * <p>
 * <p>
 * output: channel * height * width
 * <p>
 * channel: (tx + ty + tw + th + obj + class[class_num]) * anchor
 * <p>
 * tx,ty:anchor offset(锚框偏移:锚框的锚点[左上角点的偏移值]),结合锚框的锚点可以定位出预测框的中心点
 * <p>
 * tw,th:anchor sacle(锚框的比值)
 * <p>
 * bx = sigmoid(tx)+  cx
 * <p>
 * by = sigmoid(ty) + cy
 * <p>
 * cx,cy:预测所属grid_idx
 * <p>
 * bw = pw * exp(tw)
 * <p>
 * bh = ph * exp(th)
 * <p>
 * pw,ph:锚框的宽高
 */
public class YoloLoss7 extends LossFunction {
    public final LossType lossType = LossType.yolo;
    private int class_number = 1;
    private int bbox_num = 3;
    private int total = 6;
    private int outputs = 0;
    private int truths = 0;
    private Tensor loss;
    private Tensor diff;
    private int[] mask;
    private float[] anchors;
    private int orgW;
    private int orgH;
    private int maxBox = 90;
    private float ignoreThresh = 0.5f;
    private float truthThresh = 1.0f;
    private float iou_normalizer = 0.05f;
    private float max_delta = 2.0f;
    private float cls_normalizer = 0.5f;
    private float obj_normalizer = 1.0f;
    private int objectness_smooth = 0;
    private float iou_thresh = 0.2f;
    private float focal_loss = 1f;

    public YoloLoss7(int class_number, int bbox_num, int[] mask, float[] anchors, int orgH, int orgW, int maxBox, int total, float ignoreThresh, float truthThresh) {
        this.class_number = class_number;
        this.bbox_num = bbox_num;
        this.mask = mask;
        this.anchors = anchors;
        this.orgH = orgH;
        this.orgW = orgW;
        this.maxBox = maxBox;
        this.total = total;
        this.ignoreThresh = ignoreThresh;
        this.truthThresh = truthThresh;
    }

    /**
     * 真实框w,h
     * <p>
     * bh = ph * exp(th)
     * <p>
     * bw = pw * exp(tw)
     * <p>
     * ph,pw:锚框(anchor)
     * <p>
     * th,tw:网络输出(锚框的比值)
     *
     * @param x
     * @param anchors
     * @param n
     * @param index
     * @param i
     * @param j
     * @param lw
     * @param lh
     * @param w
     * @param h
     * @param stride
     * @return
     */
    public static float[] getYoloBox(Tensor x, float[] anchors, int n, int index, int i, int j, int lw, int lh, int w, int h, int stride) {
        //		System.out.println(lw+":"+anchors[2 * n]+":"+anchors[2 * n + 1]);
        float[] box = new float[4];
        box[0] = (i + x.data[index + 0 * stride]) / lw;
        box[1] = (j + x.data[index + 1 * stride]) / lh;
        box[2] = (float) x.data[index + 2 * stride] * x.data[index + 2 * stride] * 4 * anchors[2 * n] / w;
        box[3] = (float) x.data[index + 3 * stride] * x.data[index + 3 * stride] * 4 * anchors[2 * n + 1] / h;
        return box;
    }

    public static float fix_nan_inf(float val) {
        if (Float.isNaN(val) || Float.isInfinite(val)) {
            return 0;
        }
        return val;
    }

    public static void fix_nan_inf(float[] vals) {
        for (int i = 0; i < vals.length; i++) {
            if (Float.isNaN(vals[i]) || Float.isInfinite(vals[i])) {
                vals[i] = 0;
            }
        }
    }

    public static float clip(float val, float max) {
        if (val > max) {
            val = max;
        } else if (val < -max) {
            val = -max;
        }
        return val;
    }

    public static void clip(float[] vals, float max) {
        for (int i = 0; i < vals.length; i++) {
            if (vals[i] > max) {
                vals[i] = max;
            } else if (vals[i] < -max) {
                vals[i] = -max;
            }
        }
    }

    public void init(Tensor input) {
        if (loss == null || input.number != this.diff.number) {
            this.loss = new Tensor(1, 1, 1, 1);
            this.diff = new Tensor(input.number, input.channel, input.height, input.width, true);
            this.outputs = input.height * input.width * bbox_num * (class_number + 4 + 1);
            this.truths = maxBox * (4 + 1);
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
        float avg_iou = 0;
        float recall = 0;
        float recall75 = 0;
        float avg_cat = 0;
        float avg_obj = 0;
        float avg_anyobj = 0;
        int count = 0;
        int class_count = 0;
        int testCount = 0;
        int stride = x.width * x.height;
        for (int b = 0; b < x.number; b++) {
            /**
             * 计算负样本损失

             */
            for (int h = 0; h < x.height; h++) {
                for (int w = 0; w < x.width; w++) {
                    for (int n = 0; n < this.bbox_num; n++) {
                        int n_index = n * x.width * x.height + h * x.width + w;
                        int box_index = entryIndex(b, x.width, x.height, n_index, 0);
                        int obj_index = entryIndex(b, x.width, x.height, n_index, 4);
                        int class_index = entryIndex(b, x.width, x.height, n_index, 4 + 1);
                        float[] pred = getYoloBox(x, anchors, mask[n], box_index, w, h, x.width, x.height, orgW, orgH, stride);
                        float best_match_iou = 0;
                        float bestIOU = 0;
                        for (int t = 0; t < maxBox; t++) {
                            float[] truth = floatToBox(label, b, t, 1);
                            if (truth[0] == 0) {
                                break;
                            }
                            int class_id = (int) label.data[t * (4 + 1) + b * truths + 4];
                            if (class_id >= class_number || class_id < 0) {
                                System.err.println("error class.");
                            }
                            float objectness = x.data[obj_index];
                            if (Float.isNaN(objectness) || Float.isInfinite(objectness)) {
                                x.data[obj_index] = 0;
                            }
                            int class_id_match = compareYoloClass(x, this.class_number, class_index, x.width * x.height, objectness, class_id, 0.25f);
                            float iou = YoloUtils.box_iou(pred, truth);
                            if (iou > best_match_iou && class_id_match == 1) {
                                best_match_iou = iou;
                            }
                            if (iou > bestIOU) {
                                bestIOU = iou;
                            }
                        }
                        avg_anyobj += x.data[obj_index];
                        //						this.diff.data[obj_index] =  this.obj_normalizer * (0 - x.data[obj_index]);
                        this.diff.data[obj_index] = this.obj_normalizer * x.data[obj_index];
                        if (best_match_iou > ignoreThresh) {
                            //							System.out.println(best_match_iou);
                            if (objectness_smooth == 1) {
                                //								float delta_obj = obj_normalizer * (best_match_iou - x.data[obj_index]);
                                float delta_obj = obj_normalizer * (x.data[obj_index] - best_match_iou);
                                if (delta_obj > this.diff.data[obj_index]) {
                                    this.diff.data[obj_index] = delta_obj;
                                }
                            } else {
                                this.diff.data[obj_index] = 0;
                            }
                        }
                        if (bestIOU > truthThresh) {
                            System.out.println(bestIOU);
                        }
                    }
                }
            }
            for (int t = 0; t < maxBox; t++) {
                float[] truth = floatToBox(label, b, t, 1);
                if (truth[0] == 0) {
                    break;
                }
                if (truth[0] < 0 || truth[1] < 0 || truth[0] > 1 || truth[1] > 1 || truth[2] < 0 || truth[3] < 0) {
                    System.err.println("wrong label:[" + truth[0] + ":" + truth[1] + ":" + truth[2] + ":" + truth[3] + "].");
                }
                //				System.out.println(JsonUtils.toJson(truth));
                float bestIOU = 0;
                int bestIndex = 0;
                int i = (int) (truth[0] * x.width);
                int j = (int) (truth[1] * x.height);
                float[] truthShift = new float[]{0, 0, truth[2], truth[3]};
                for (int n = 0; n < this.total; n++) {
                    float[] pred = new float[]{0, 0, anchors[2 * n] / orgW, anchors[2 * n + 1] / orgH};
                    float iou = YoloUtils.box_iou(pred, truthShift);
                    //	            	System.out.println(iou);
                    if (iou > bestIOU) {
                        bestIOU = iou;// 记录最大的IOU
                        bestIndex = n;// 以及记录该bbox的编号n
                    }
                }
                int mask_n = intIndex(mask, bestIndex, bbox_num);
                /**
                 * 计算正样本Lobj,Lcls,Lloct

                 */
                if (mask_n >= 0) {
                    int mask_n_index = mask_n * x.width * x.height + j * x.width + i;
                    int box_index = entryIndex(b, x.width, x.height, mask_n_index, 0);
                    float iou = deltaYoloBox(truth, x, anchors, bestIndex, box_index, i, j, x.width, x.height, stride);
                    int obj_index = entryIndex(b, x.width, x.height, mask_n_index, 4);
                    if (x.data[obj_index] >= 0.7f) {
                        testCount++;
                    }
                    avg_obj += x.data[obj_index];
                    if (objectness_smooth == 1) {
                        if (this.diff.data[obj_index] == 0) {
                            this.diff.data[obj_index] = obj_normalizer * (x.data[obj_index] - 1.0f);
                        }
                    } else {
                        this.diff.data[obj_index] = obj_normalizer * (x.data[obj_index] - 1.0f);
                    }
                    int class_id = (int) label.data[t * (4 + 1) + b * truths + 4];
                    int class_index = entryIndex(b, x.width, x.height, mask_n_index, 4 + 1);
                    avg_cat = deltaYoloClass(x, class_index, class_id, class_number, stride, avg_cat);
                    count++;
                    class_count++;
                    if (iou > .5)
                        recall += 1;
                    if (iou > .75)
                        recall75 += 1;
                    avg_iou += iou;
                }
                // iou_thresh
                for (int n = 0; n < total; ++n) {
                    int mask_n_t = intIndex(mask, n, bbox_num);
                    if (mask_n_t >= 0 && n != bestIndex && this.iou_thresh < 1.0f) {
                        float[] pred = new float[]{0, 0, anchors[2 * n] / orgW, anchors[2 * n + 1] / orgH};
                        float iou = YoloUtils.box_iou(pred, truthShift);
                        //		            	System.out.println("iou:"+iou);
                        if (iou > iou_thresh) {
                            int mask_n_index = mask_n_t * x.width * x.height + j * x.width + i;
                            int class_id = (int) label.data[t * (4 + 1) + b * truths + 4];
                            int box_index = entryIndex(b, x.width, x.height, mask_n_index, 0);
                            float ciou = deltaYoloBox(truth, x, anchors, n, box_index, i, j, x.width, x.height, stride);
                            int obj_index = entryIndex(b, x.width, x.height, mask_n_index, 4);
                            if (x.data[obj_index] >= 0.7f) {
                                testCount++;
                            }
                            avg_obj += x.data[obj_index];
                            if (objectness_smooth == 1) {
                                if (this.diff.data[obj_index] == 0) {
                                    this.diff.data[obj_index] = obj_normalizer * (x.data[obj_index] - 1.0f);
                                }
                            } else {
                                this.diff.data[obj_index] = obj_normalizer * (x.data[obj_index] - 1.0f);
                            }
                            int class_index = entryIndex(b, x.width, x.height, mask_n_index, 4 + 1);
                            avg_cat = deltaYoloClass(x, class_index, class_id, class_number, stride, avg_cat);
                            count++;
                            class_count++;
                            if (ciou > .5)
                                recall += 1;
                            if (ciou > .75)
                                recall75 += 1;
                            avg_iou += ciou;
                        }
                    }
                }
            }
            if (iou_thresh < 1.0f) {
                // averages the deltas obtained by the function: delta_yolo_box()_accumulate
                for (int h = 0; h < x.height; h++) {
                    for (int w = 0; w < x.width; w++) {
                        for (int n = 0; n < this.bbox_num; n++) {
                            int n_index = n * x.width * x.height + h * x.width + w;
                            int box_index = entryIndex(b, x.width, x.height, n_index, 0);
                            int obj_index = entryIndex(b, x.width, x.height, n_index, 4);
                            int class_index = entryIndex(b, x.width, x.height, n_index, 4 + 1);
                            if (this.diff.data[obj_index] != 0) {
                                averagesYoloDeltas(class_index, box_index, stride, class_number, this.diff);
                            }
                        }
                    }
                }
            }
        }
        System.out.println("loss:" + Math.pow(mag_array(this.diff.data), 2.0) / x.number);
        System.out.println("Avg IOU: " + avg_iou / count + ", Class: " + avg_cat / class_count + ", Obj: " + avg_obj / count + "," + " No Obj: " + avg_anyobj / (x.width * x.height * bbox_num * x.number) + ", .5R: " + recall / count + ", .75R: " + recall75 / count + ",  count: " + count + ", testCount:" + testCount);
        return loss;
    }

    public float mag_array(float[] a) {
        int i;
        float sum = 0;
        for (i = 0; i < a.length; ++i) {
            sum += a[i] * a[i];
        }
        return (float) Math.sqrt(sum);
    }

    private float deltaYoloBox(float[] truth, Tensor x, float[] anchors, int n, int index, int i, int j, int lw, int lh, int stride) {
        float[] pred = getYoloBox(x, anchors, n, index, i, j, lw, lh, orgW, orgH, stride);
        float ciou = YoloUtils.box_ciou(pred, truth);
        if (pred[2] == 0) {
            pred[2] = 1.0f;
        }
        if (pred[3] == 0) {
            pred[3] = 1.0f;
        }
        float[] dx = YoloUtils.dx_box_ciou(pred, truth);
        dx = MatrixOperation.multiplication(dx, this.iou_normalizer);
        fix_nan_inf(dx);
        clip(dx, this.max_delta);
        this.diff.data[index + 0 * stride] -= dx[0];
        this.diff.data[index + 1 * stride] -= dx[1];
        this.diff.data[index + 2 * stride] -= dx[2];
        this.diff.data[index + 3 * stride] -= dx[3];
        return ciou;
    }

    private void averagesYoloDeltas(int class_index, int box_index, int stride, int classes, Tensor delta) {
        int classes_in_one_box = 0;
        for (int c = 0; c < classes; ++c) {
            if (delta.data[class_index + stride * c] > 0) {
                classes_in_one_box++;
            }
        }
        //	    System.out.println(classes_in_one_box);
        if (classes_in_one_box > 0) {
            delta.data[box_index + 0 * stride] /= classes_in_one_box;
            delta.data[box_index + 1 * stride] /= classes_in_one_box;
            delta.data[box_index + 2 * stride] /= classes_in_one_box;
            delta.data[box_index + 3 * stride] /= classes_in_one_box;
        }
    }

    private float deltaYoloClass(Tensor output, int index, int class_id, int classes, int stride, float avg_cat) {
        if (this.diff.data[index + stride * class_id] == 1.0f) {
            float y_true = 1;
            //			float result_delta = y_true - output.data[index + stride*class_id];
            float result_delta = output.data[index + stride * class_id] - y_true;
            if (!Float.isNaN(result_delta) && !Float.isInfinite(result_delta)) {
                this.diff.data[index + stride * class_id] = result_delta;
            }
            avg_cat += output.data[index + stride * class_id];
            return avg_cat;
        }
        if (this.focal_loss == 1) {
            // Focal Loss
            float alpha = 0.25f;    // 0.25 or 0.5
            //float gamma = 2;    // hardcoded in many places of the grad-formula
            int ti = index + stride * class_id;
            float pt = output.data[ti] + 0.000000000000001f;
            // http://fooplot.com/#W3sidHlwZSI6MCwiZXEiOiItKDEteCkqKDIqeCpsb2coeCkreC0xKSIsImNvbG9yIjoiIzAwMDAwMCJ9LHsidHlwZSI6MTAwMH1d
            float grad = (float) (-(1 - pt) * (2 * pt * Math.log(pt) + pt - 1));    // http://blog.csdn.net/linmingan/article/details/77885832
            //float grad = (1 - pt) * (2 * pt*Math.log(pt) + pt - 1);    // https://github.com/unsky/focal-loss
            for (int n = 0; n < classes; ++n) {
                float y_true = ((n == class_id) ? 1 : 0);
                this.diff.data[index + stride * n] = output.data[index + stride * n] - y_true;
                this.diff.data[index + stride * n] *= alpha * grad;
                if (n == class_id) {
                    avg_cat += output.data[index + stride * n];
                }
            }
        } else {
            for (int n = 0; n < classes; n++) {
                float y_true = ((n == class_id) ? 1 : 0);
                //				float result_delta = y_true - output.data[index + stride*n];
                float result_delta = output.data[index + stride * n] - y_true;
                //				if(y_true == 1){
                //					System.out.println(output.data[index + stride * n] + ":" + result_delta);
                //				}
                if (!Float.isNaN(result_delta) && !Float.isInfinite(result_delta)) {
                    this.diff.data[index + stride * n] = result_delta;
                }
                //				System.out.println(index+":"+this.diff.data[index + stride * n] + "{"+output.data[index + stride*n]+"}");
                if (n == class_id) {
                    //					this.diff.data[index + stride * n] *= cls_normalizer;
                    avg_cat += output.data[index + stride * n];
                }
            }
        }
        return avg_cat;
    }

    private int compareYoloClass(Tensor output, int classes, int class_index, int stride, float obj, int class_id, float conf_thresh) {
        for (int j = 0; j < classes; ++j) {
            float prob = output.data[class_index + stride * j];
            //	        System.out.println(class_index + ":" + j + ":" + prob);
            if (prob > conf_thresh) {
                return 1;
            }
        }
        return 0;
    }

    private int intIndex(int[] mask, int bestIndex, int bbox_num) {
        for (int i = 0; i < bbox_num; ++i) {
            if (mask[i] == bestIndex)
                return i;
        }
        return -1;
    }

    private float[] floatToBox(Tensor label, int b, int t, int stride) {
        float[] box = new float[4];
        box[0] = label.data[(b * truths + t * 5 + 0) * stride];
        box[1] = label.data[(b * truths + t * 5 + 1) * stride];
        box[2] = label.data[(b * truths + t * 5 + 2) * stride];
        box[3] = label.data[(b * truths + t * 5 + 3) * stride];
        return box;
    }

    private int entryIndex(int batch, int w, int h, int location, int entry) {
        int n = location / (w * h);
        int loc = location % (w * h);
        return batch * this.outputs + n * w * h * (4 + this.class_number + 1) + entry * w * h + loc;
    }

    @Override
    public Tensor diff(Tensor x, Tensor label) {
        // TODO Auto-generated method stub
        if (diff.isHasGPU()) {
            diff.hostToDevice();
        }
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

    public void test(Tensor output, int bbox_num, int b, int index) {
        for (int i = 0; i < output.height * output.width; i++) {
            int row = i / output.width;
            int col = i % output.width;
            for (int n = 0; n < bbox_num; n++) {
                int n_index = n * output.width * output.height + row * output.width + col;
                //	        	System.out.println(n_index);
                int obj_index = entryIndex(b, output.width, output.height, n_index, 4);
                float objectness = output.data[obj_index];
                if (obj_index == index) {
                    System.out.println("test:" + objectness + "=" + output.data[index]);
                }
            }
        }
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

