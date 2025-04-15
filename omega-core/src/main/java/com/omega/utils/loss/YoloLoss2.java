package com.omega.utils.loss;

import com.omega.common.tensor.Tensor;
import com.omega.utils.MatrixUtils;
import com.omega.engine.loss.LossFunction;
import com.omega.engine.loss.LossType;
import com.omega.utils.yolo.YoloUtils;

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
 * channel: tx + ty + tw + th + obj + class[class_num]
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
public class YoloLoss2 extends LossFunction {
    public final LossType lossType = LossType.yolo;
    private int class_number = 1;
    private int bbox_num = 3;
    private int total = 6;
    private int outputs = 0;
    private int truths = 0;
    private Tensor loss;
    private Tensor diff;
    private float[] anchors;
    private int orgW;
    private int orgH;
    private int maxBox = 90;
    private float ignoreThresh = 0.5f;
    private float truthThresh = 1.0f;
    private float noobject_scale = 1.0f;
    private float coord_scale = 5.0f;
    private float class_scale = 1.0f;
    private float object_scale = 5.0f;

    public YoloLoss2(int class_number, int bbox_num, float[] anchors, int orgH, int orgW, int maxBox, int total, float ignoreThresh, float truthThresh) {
        this.class_number = class_number;
        this.bbox_num = bbox_num;
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
        float[] box = new float[4];
        box[0] = (i + x.data[index + 0 * stride]) / lw;
        box[1] = (j + x.data[index + 1 * stride]) / lh;
        box[2] = (float) (Math.exp(x.data[index + 2 * stride]) * anchors[2 * n] / w);
        box[3] = (float) (Math.exp(x.data[index + 3 * stride]) * anchors[2 * n + 1] / h);
        return box;
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
        //		System.out.println(JsonUtils.toJson(label.data));
        //		System.out.println(x.dataLength);
        init(x);
        if (x.isHasGPU()) {
            x.syncHost();
        }
        //		if(x.width == 8) {
        //
        //			x.showDMByNumber(0);
        //
        //		}
        float avg_iou = 0;
        float recall = 0;
        float avg_cat = 0;
        float avg_obj = 0;
        float avg_anyobj = 0;
        int count = 0;
        int class_count = 0;
        int stride = x.width * x.height;
        for (int b = 0; b < x.number; b++) {
            for (int h = 0; h < x.height; h++) {
                for (int w = 0; w < x.width; w++) {
                    for (int n = 0; n < this.bbox_num; n++) {
                        int n_index = n * x.width * x.height + h * x.width + w;
                        int box_index = entryIndex(b, x.width, x.height, n_index, 0);
                        float[] pred = getYoloBox(x, anchors, n, box_index, w, h, x.width, x.height, orgW, orgH, stride);
                        float bestIOU = 0;
                        for (int t = 0; t < maxBox; t++) {
                            float[] truth = floatToBox(label, b, t, 1);
                            if (truth[0] == 0) {
                                break;
                            }
                            float iou = YoloUtils.box_giou(pred, truth);
                            if (iou > bestIOU) {
                                bestIOU = iou;
                            }
                        }
                        int obj_index = entryIndex(b, x.width, x.height, n_index, 4);
                        avg_anyobj += x.data[obj_index];
                        this.diff.data[obj_index] = this.noobject_scale * x.data[obj_index];
                        if (bestIOU > ignoreThresh) {
                            this.diff.data[obj_index] = 0;
                        }
                    }
                }
            }
            for (int t = 0; t < maxBox; t++) {
                float[] truth = floatToBox(label, b, t, 1);
                //				System.out.println(JsonUtils.toJson(truth));
                if (truth[0] == 0) {
                    break;
                }
                //				System.out.println(JsonUtils.toJson(truth));
                float bestIOU = 0;
                int bestIndex = 0;
                int i = (int) (truth[0] * x.width);
                int j = (int) (truth[1] * x.height);
                float[] truthShift = new float[]{0, 0, truth[2], truth[3]};
                for (int n = 0; n < this.total; n++) {
                    float[] pred = new float[]{0, 0, anchors[2 * n] / orgW, anchors[2 * n + 1] / orgH};
                    float iou = YoloUtils.box_giou(pred, truthShift);
                    if (iou > bestIOU) {
                        bestIOU = iou;// 记录最大的IOU
                        bestIndex = n;// 以及记录该bbox的编号n
                    }
                }
                int box_index = entryIndex(b, x.width, x.height, bestIndex * x.width * x.height + j * x.width + i, 0);
                float iou = deltaYoloBox(truth, x, anchors, bestIndex, box_index, i, j, x.width, x.height, coord_scale * (2.0f - truth[2] * truth[3]), stride);
                if (iou > .5)
                    recall += 1;
                avg_iou += iou;
                int obj_index = entryIndex(b, x.width, x.height, bestIndex * x.width * x.height + j * x.width + i, 4);
                avg_obj += x.data[obj_index];
                this.diff.data[obj_index] = object_scale * (x.data[obj_index] - 1.0f);
                //                this.diff.data[obj_index] = object_scale * (x.data[obj_index] - iou);
                int clazz = (int) label.data[t * (4 + 1) + b * truths + 4];
                int class_index = entryIndex(b, x.width, x.height, bestIndex * x.width * x.height + j * x.width + i, 4 + 1);
                avg_cat = deltaYoloClass(x, class_index, clazz, class_number, stride, avg_cat);
                ++count;
                ++class_count;
            }
        }
        System.out.println("loss:" + Math.pow(mag_array(this.diff.data), 2.0) / x.number + ", Avg IOU: " + avg_iou / count + ", Class: " + avg_cat / class_count + ", Obj: " + avg_obj / count + "," + " No Obj: " + avg_anyobj / (x.width * x.height * bbox_num * x.number) + ", Avg Recall: " + recall / count + ", count: " + count);
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

    private float deltaYoloClass(Tensor x, int index, int clazz, int classes, int stride, float avg_cat) {
        if (this.diff.data[index] == 1.0f) {
            this.diff.data[index + stride * clazz] = 1.0f - x.data[index + stride * clazz];
            avg_cat += x.data[index + stride * clazz];
            return avg_cat;
        }
        for (int n = 0; n < classes; n++) {
            this.diff.data[index + stride * n] = x.data[index + stride * n] - ((n == clazz) ? 1 : 0);
            if (n == clazz) {
                avg_cat += x.data[index + stride * n];
            }
        }
        return avg_cat;
    }

    private float deltaYoloBox(float[] truth, Tensor x, float[] anchors, int n, int index, int i, int j, int lw, int lh, float scale, int stride) {
        float[] pred = getYoloBox(x, anchors, n, index, i, j, lw, lh, orgW, orgH, stride);
        float iou = YoloUtils.box_giou(pred, truth);
        float tx = (truth[0] * lw - i);
        float ty = (truth[1] * lh - j);
        float tw = (float) Math.log(truth[2] * orgW / anchors[2 * n]);
        float th = (float) Math.log(truth[3] * orgH / anchors[2 * n + 1]);
        this.diff.data[index + 0 * stride] = scale * (x.data[index + 0 * stride] - tx);
        this.diff.data[index + 1 * stride] = scale * (x.data[index + 1 * stride] - ty);
        this.diff.data[index + 2 * stride] = scale * (x.data[index + 2 * stride] - tw);
        this.diff.data[index + 3 * stride] = scale * (x.data[index + 3 * stride] - th);
        return iou;
    }

    private void deltaYoloMask(float[] truth, Tensor x, int n, int index, int stride, int scale) {
        for (int i = 0; i < n; i++) {
            this.diff.data[index + i * stride] = scale * (truth[i] - x.data[index + i * stride]);
        }
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
        box[0] = label.data[b * truths + t * 5 + 0 * stride];
        box[1] = label.data[b * truths + t * 5 + 1 * stride];
        box[2] = label.data[b * truths + t * 5 + 2 * stride];
        box[3] = label.data[b * truths + t * 5 + 3 * stride];
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

