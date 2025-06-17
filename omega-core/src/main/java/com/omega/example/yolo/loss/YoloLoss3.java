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
public class YoloLoss3 extends LossFunction {
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
    private float eta = 1e-6f;

    public YoloLoss3(int class_number, int bbox_num, int[] mask, float[] anchors, int orgH, int orgW, int maxBox, int total, float ignoreThresh, float truthThresh) {
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
        float[] box = new float[4];
        box[0] = (i + x.getData()[index + 0 * stride]) / lw;
        box[1] = (j + x.getData()[index + 1 * stride]) / lh;
        box[2] = (float) (Math.exp(x.getData()[index + 2 * stride]) * anchors[2 * n] / w);
        box[3] = (float) (Math.exp(x.getData()[index + 3 * stride]) * anchors[2 * n + 1] / h);
        return box;
    }

    public void init(Tensor input) {
        if (loss == null || input.getShape()[0] != this.diff.getShape()[0]) {
            this.loss = new Tensor(1, 1, 1, 1);
            this.diff = new Tensor(input.getShape()[0], input.getShape()[1], input.getShape()[2], input.getShape()[3], true);
            this.outputs = input.getShape()[2] * input.getShape()[3] * bbox_num * (class_number + 4 + 1);
            this.truths = maxBox * (4 + 1);
        } else {
            MatrixUtils.zero(this.diff.getData());
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
        //		System.out.println(x.number + ":" + x.height + ":" + x.getDataLength());
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
        float recall75 = 0;
        float avg_cat = 0;
        float avg_obj = 0;
        float avg_anyobj = 0;
        int count = 0;
        int class_count = 0;
        int testCount = 0;
        int stride = x.getShape()[3] * x.getShape()[2];
        for (int b = 0; b < x.getShape()[0]; b++) {
            /**
             * 计算负样本损失

             */
            for (int h = 0; h < x.getShape()[2]; h++) {
                for (int w = 0; w < x.getShape()[3]; w++) {
                    for (int n = 0; n < this.bbox_num; n++) {
                        int n_index = n * x.getShape()[3] * x.getShape()[2] + h * x.getShape()[3] + w;
                        int box_index = entryIndex(b, x.getShape()[3], x.getShape()[2], n_index, 0);
                        float[] pred = getYoloBox(x, anchors, mask[n], box_index, w, h, x.getShape()[3], x.getShape()[2], orgW, orgH, stride);
                        float bestIOU = 0;
                        //						int bestIndex = 0;
                        for (int t = 0; t < maxBox; t++) {
                            float[] truth = floatToBox(label, b, t, 1);
                            if (truth[0] == 0) {
                                break;
                            }
                            float iou = YoloUtils.box_iou(pred, truth);
                            if (iou > bestIOU) {
                                bestIOU = iou;
                                //								bestIndex = t;
                            }
                        }
                        int obj_index = entryIndex(b, x.getShape()[3], x.getShape()[2], n_index, 4);
                        avg_anyobj += x.getData()[obj_index];
                        //						this.diff.data[obj_index] = 0 - x.data[obj_index];
                        this.diff.getData()[obj_index] = x.getData()[obj_index];
                        if (bestIOU > ignoreThresh) {
                            this.diff.getData()[obj_index] = 0;
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
                //				System.out.println(JsonUtils.toJson(truth));
                float bestIOU = 0;
                int bestIndex = 0;
                int i = (int) (truth[0] * x.getShape()[3]);
                int j = (int) (truth[1] * x.getShape()[2]);
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
                    //					System.out.println(JsonUtils.toJson(truth));
                    int mask_n_index = mask_n * x.getShape()[3] * x.getShape()[2] + j * x.getShape()[3] + i;
                    int box_index = entryIndex(b, x.getShape()[3], x.getShape()[2], mask_n_index, 0);
                    float iou = deltaYoloBox(truth, x, anchors, bestIndex, box_index, i, j, x.getShape()[3], x.getShape()[2], (2.0f - truth[2] * truth[3]), stride);
                    int obj_index = entryIndex(b, x.getShape()[3], x.getShape()[2], mask_n_index, 4);
                    if (x.getData()[obj_index] >= 0.8f) {
                        testCount++;
                    }
                    avg_obj += x.getData()[obj_index];
                    //	            	if(x.width == 8) {
                    //
                    //		            	System.out.println(i+":"+j+"["+x.data[obj_index]+"]");
                    //
                    //	            	}
                    this.diff.getData()[obj_index] = x.getData()[obj_index] - 1.0f;
                    int clazz = (int) label.getData()[t * (4 + 1) + b * truths + 4];
                    int class_index = entryIndex(b, x.getShape()[3], x.getShape()[2], mask_n_index, 4 + 1);
                    avg_cat = deltaYoloClass(x, class_index, clazz, class_number, stride, avg_cat);
                    count++;
                    class_count++;
                    if (iou > .5)
                        recall += 1;
                    if (iou > .75)
                        recall75 += 1;
                    avg_iou += iou;
                    //	                System.out.println("iou:"+iou);
                }
            }
        }
        //		if(net.RUN_MODEL == RunModel.TEST || (avg_obj/count) >= 0.8f){
        //			System.out.println(JsonUtils.toJson(x.getByNumberAndChannel(0, 4)));
        //		}
        //		if(Double.isInfinite(Math.pow(mag_array(this.diff.data), 2.0)/x.number) || avg_iou == 0) {
        //			x.showDM();
        //			label.showDM();
        //		}
        System.out.println("loss:" + Math.pow(mag_array(this.diff.getData()), 2.0) / x.getShape()[0]);
        System.out.println("Avg IOU: " + avg_iou / count + ", Class: " + avg_cat / class_count + ", Obj: " + avg_obj / count + "," + " No Obj: " + avg_anyobj / (x.getShape()[3] * x.getShape()[2] * bbox_num * x.getShape()[0]) + ", .5R: " + recall / count + ", .75R: " + recall75 / count + ",  count: " + count + ", testCount:" + testCount);
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
        if (this.diff.getData()[index] == 1.0f) {
            //			this.diff.data[index + stride * clazz] = 1.0f - x.data[index + stride * clazz];
            this.diff.getData()[index + stride * clazz] = x.getData()[index + stride * clazz] - 1.0f;
            avg_cat += x.getData()[index + stride * clazz];
            return avg_cat;
        }
        for (int n = 0; n < classes; n++) {
            //			this.diff.data[index + stride * n] = ((n == clazz)?1 : 0) - x.data[index + stride * n];
            this.diff.getData()[index + stride * n] = x.getData()[index + stride * n] - ((n == clazz) ? 1 : 0);
            if (n == clazz) {
                avg_cat += x.getData()[index + stride * n];
            }
        }
        return avg_cat;
    }

    private float deltaYoloBox(float[] truth, Tensor x, float[] anchors, int n, int index, int i, int j, int lw, int lh, float scale, int stride) {
        float[] pred = getYoloBox(x, anchors, n, index, i, j, lw, lh, orgW, orgH, stride);
        float iou = YoloUtils.box_iou(pred, truth);
        float tx = (truth[0] * lw - i);
        float ty = (truth[1] * lh - j);
        //	    float tw = (float) Math.log(truth[2] * orgW / anchors[2*n] + eta);
        //	    float th = (float) Math.log(truth[3] * orgH / anchors[2*n + 1] + eta);
        float tw = (float) Math.log(truth[2] * orgW / anchors[2 * n]);
        float th = (float) Math.log(truth[3] * orgH / anchors[2 * n + 1]);
        //	    if(Float.isInfinite(tw) || Float.isInfinite(th) || Float.isNaN(tw) || Float.isNaN(th)) {
        //	    	System.out.println(JsonUtils.toJson(truth));
        //	    	System.out.println(tw+":"+th);
        //	    }
        this.diff.getData()[index + 0 * stride] = scale * (x.getData()[index + 0 * stride] - tx);
        this.diff.getData()[index + 1 * stride] = scale * (x.getData()[index + 1 * stride] - ty);
        this.diff.getData()[index + 2 * stride] = scale * (x.getData()[index + 2 * stride] - tw);
        this.diff.getData()[index + 3 * stride] = scale * (x.getData()[index + 3 * stride] - th);
        return iou;
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
        box[0] = label.getData()[(b * truths + t * 5 + 0) * stride];
        box[1] = label.getData()[(b * truths + t * 5 + 1) * stride];
        box[2] = label.getData()[(b * truths + t * 5 + 2) * stride];
        box[3] = label.getData()[(b * truths + t * 5 + 3) * stride];
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
        for (int i = 0; i < output.getShape()[2] * output.getShape()[3]; i++) {
            int row = i / output.getShape()[3];
            int col = i % output.getShape()[3];
            for (int n = 0; n < bbox_num; n++) {
                int n_index = n * output.getShape()[3] * output.getShape()[2] + row * output.getShape()[3] + col;
                //	        	System.out.println(n_index);
                int obj_index = entryIndex(b, output.getShape()[3], output.getShape()[2], n_index, 4);
                float objectness = output.getData()[obj_index];
                if (obj_index == index) {
                    System.out.println("test:" + objectness + "=" + output.getData()[index]);
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

