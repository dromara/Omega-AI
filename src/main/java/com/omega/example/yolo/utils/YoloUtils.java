package com.omega.example.yolo.utils;

import com.omega.engine.tensor.Tensor;
import com.omega.example.yolo.model.YoloDetection;

import java.util.List;

public class YoloUtils {
    private final static float eps = 1e-6f;

    public static float box_rmse(float[] a, float[] b) {
        return (float) Math.sqrt(Math.pow(a[0] - b[0], 2) + Math.pow(a[1] - b[1], 2) + Math.pow(a[2] - b[2], 2) + Math.pow(a[3] - b[3], 2));
    }

    public static float overlap(float x1, float w1, float x2, float w2) {
        float l1 = x1 - w1 / 2;   //边框1的左边坐标
        float l2 = x2 - w2 / 2;
        float left = l1 > l2 ? l1 : l2;    //重合部分左边（上边）的坐标
        float r1 = x1 + w1 / 2;
        float r2 = x2 + w2 / 2;
        float right = r1 > r2 ? r2 : r1;    //重合部分右边（下边）的坐标
        return (right - left);
    }

    public static float overlapMin(float x1, float w1, float x2, float w2) {
        float l1 = x1 - w1 / 2;   //边框1的左边坐标
        float l2 = x2 - w2 / 2;
        float left = l1 < l2 ? l1 : l2;    //重合部分左边（上边）的坐标
        float r1 = x1 + w1 / 2;
        float r2 = x2 + w2 / 2;
        float right = r1 < r2 ? r2 : r1;    //重合部分右边（下边）的坐标
        return (right - left);
    }

    public static float box_intersection(float[] a, float[] b) {
        float w = overlap(a[0], a[2], b[0], b[2]);
        float h = overlap(a[1], a[3], b[1], b[3]);
        if (w < 0 || h < 0)
            return 0;
        float area = w * h;
        return area;
    }

    public static float box_encolsed(float[] a, float[] b) {
        float w = overlapMin(a[0], a[2], b[0], b[2]);
        float h = overlapMin(a[1], a[3], b[1], b[3]);
        if (w < 0 || h < 0)
            return 0;
        float area = w * h;
        return area;
    }

    public static float box_union(float[] a, float[] b) {
        float i = box_intersection(a, b);
        float u = a[2] * a[3] + b[2] * b[3] - i;
        return u;
    }

    public static float box_iou(float[] a, float[] b) {
        return box_intersection(a, b) / box_union(a, b);  //IOU=交集/并集
    }

    public static float box_ciou(float[] a, float[] b) {
        float[] ba = box_c(a, b);
        float w = ba[3] - ba[2];
        float h = ba[1] - ba[0];
        float c = w * w + h * h;
        float iou = box_iou(a, b);
        if (c == 0) {
            return iou;
        }
        float u = (a[0] - b[0]) * (a[0] - b[0]) + (a[1] - b[1]) * (a[1] - b[1]);
        float d = u / c;
        float ar_gt = b[2] / b[3];
        float ar_pred = a[2] / a[3];
        float ar_loss = (float) (4.0f / (Math.PI * Math.PI) * (Math.atan(ar_gt) - Math.atan(ar_pred)) * (Math.atan(ar_gt) - Math.atan(ar_pred)));
        float alpha = ar_loss / (1 - iou + ar_loss + eps);
        float ciou_term = d + alpha * ar_loss;
        return iou - ciou_term;
    }

    public static float[] to_tblr(float[] a) {
        float[] tblr = new float[4];
        tblr[0] = a[1] - a[3] / 2; //top
        tblr[1] = a[1] + a[3] / 2; //bottom
        tblr[2] = a[0] - a[2] / 2; //left
        tblr[3] = a[0] + a[2] / 2; //right
        return tblr;
    }

    public static float[] dx_box_ciou(float[] pred, float[] truth) {
        float[] dx = new float[4];
        float[] pred_tblr = to_tblr(pred);
        float[] truth_tblr = to_tblr(truth);
        float pred_t = Float.min(pred_tblr[0], pred_tblr[1]);
        float pred_b = Float.max(pred_tblr[0], pred_tblr[1]);
        float pred_l = Float.min(pred_tblr[2], pred_tblr[3]);
        float pred_r = Float.max(pred_tblr[2], pred_tblr[3]);
        float X = (pred_b - pred_t) * (pred_r - pred_l);
        float Xhat = (truth_tblr[1] - truth_tblr[0]) * (truth_tblr[3] - truth_tblr[2]);
        float Ih = Float.min(pred_b, truth_tblr[1]) - Float.max(pred_t, truth_tblr[0]);
        float Iw = Float.min(pred_r, truth_tblr[3]) - Float.max(pred_l, truth_tblr[2]);
        float I = Iw * Ih;
        float U = X + Xhat - I;
        float S = (pred[0] - truth[0]) * (pred[0] - truth[0]) + (pred[1] - truth[1]) * (pred[1] - truth[1]);
        //Partial Derivatives, derivatives
        float dX_wrt_t = -1 * (pred_r - pred_l);
        float dX_wrt_b = pred_r - pred_l;
        float dX_wrt_l = -1 * (pred_b - pred_t);
        float dX_wrt_r = pred_b - pred_t;
        // gradient of I min/max in IoU calc (prediction)
        float dI_wrt_t = pred_t > truth_tblr[0] ? (-1 * Iw) : 0;
        float dI_wrt_b = pred_b < truth_tblr[1] ? Iw : 0;
        float dI_wrt_l = pred_l > truth_tblr[2] ? (-1 * Ih) : 0;
        float dI_wrt_r = pred_r < truth_tblr[3] ? Ih : 0;
        // derivative of U with regard to x
        float dU_wrt_t = dX_wrt_t - dI_wrt_t;
        float dU_wrt_b = dX_wrt_b - dI_wrt_b;
        float dU_wrt_l = dX_wrt_l - dI_wrt_l;
        float dU_wrt_r = dX_wrt_r - dI_wrt_r;
        float p_dt = 0;
        float p_db = 0;
        float p_dl = 0;
        float p_dr = 0;
        if (U > 0) {
            p_dt = ((U * dI_wrt_t) - (I * dU_wrt_t)) / (U * U);
            p_db = ((U * dI_wrt_b) - (I * dU_wrt_b)) / (U * U);
            p_dl = ((U * dI_wrt_l) - (I * dU_wrt_l)) / (U * U);
            p_dr = ((U * dI_wrt_r) - (I * dU_wrt_r)) / (U * U);
        }
        // apply grad from prediction min/max for correct corner selection
        p_dt = pred_tblr[0] < pred_tblr[1] ? p_dt : p_db;
        p_db = pred_tblr[0] < pred_tblr[1] ? p_db : p_dt;
        p_dl = pred_tblr[2] < pred_tblr[3] ? p_dl : p_dr;
        p_dr = pred_tblr[2] < pred_tblr[3] ? p_dr : p_dl;
        float Ct = Float.min(pred[1] - pred[3] / 2, truth[1] - truth[3] / 2);
        float Cb = Float.max(pred[1] + pred[3] / 2, truth[1] + truth[3] / 2);
        float Cl = Float.min(pred[0] - pred[2] / 2, truth[0] - truth[2] / 2);
        float Cr = Float.max(pred[0] + pred[2] / 2, truth[0] + truth[2] / 2);
        float Cw = Cr - Cl;
        float Ch = Cb - Ct;
        float C = Cw * Cw + Ch * Ch;
        float dCt_dx = 0;
        float dCt_dy = pred_t < truth_tblr[0] ? 1 : 0;
        float dCt_dw = 0;
        float dCt_dh = pred_t < truth_tblr[0] ? -0.5f : 0;
        float dCb_dx = 0;
        float dCb_dy = pred_b > truth_tblr[1] ? 1 : 0;
        float dCb_dw = 0;
        float dCb_dh = pred_b > truth_tblr[1] ? 0.5f : 0;
        float dCl_dx = pred_l < truth_tblr[2] ? 1 : 0;
        float dCl_dy = 0;
        float dCl_dw = pred_l < truth_tblr[2] ? -0.5f : 0;
        float dCl_dh = 0;
        float dCr_dx = pred_r > truth_tblr[3] ? 1 : 0;
        float dCr_dy = 0;
        float dCr_dw = pred_r > truth_tblr[3] ? 0.5f : 0;
        float dCr_dh = 0;
        float dCw_dx = dCr_dx - dCl_dx;
        float dCw_dy = dCr_dy - dCl_dy;
        float dCw_dw = dCr_dw - dCl_dw;
        float dCw_dh = dCr_dh - dCl_dh;
        float dCh_dx = dCb_dx - dCt_dx;
        float dCh_dy = dCb_dy - dCt_dy;
        float dCh_dw = dCb_dw - dCt_dw;
        float dCh_dh = dCb_dh - dCt_dh;
        float p_dx = 0;
        float p_dy = 0;
        float p_dw = 0;
        float p_dh = 0;
        p_dx = p_dl + p_dr;           //p_dx, p_dy, p_dw and p_dh are the gradient of IoU or GIoU.
        p_dy = p_dt + p_db;
        p_dw = (p_dr - p_dl);         //For dw and dh, we do not divided by 2.
        p_dh = (p_db - p_dt);
        /**
         * CIOU DELTA

         */
        float ar_gt = truth[2] / truth[3];
        float ar_pred = pred[2] / pred[3];
        float ar_loss = (float) (4.0f / (Math.PI * Math.PI) * (Math.atan(ar_gt) - Math.atan(ar_pred)) * (Math.atan(ar_gt) - Math.atan(ar_pred)));
        float alpha = ar_loss / (1 - I / U + ar_loss + eps);
        float ar_dw = (float) (8 / (Math.PI * Math.PI) * (Math.atan(ar_gt) - Math.atan(ar_pred)) * pred[3]);
        float ar_dh = (float) (-8 / (Math.PI * Math.PI) * (Math.atan(ar_gt) - Math.atan(ar_pred)) * pred[2]);
        if (C > 0) {
            p_dx += (2 * (truth[0] - pred[0]) * C - (2 * Cw * dCw_dx + 2 * Ch * dCh_dx) * S) / (C * C);
            p_dy += (2 * (truth[1] - pred[1]) * C - (2 * Cw * dCw_dy + 2 * Ch * dCh_dy) * S) / (C * C);
            p_dw += (2 * Cw * dCw_dw + 2 * Ch * dCh_dw) * S / (C * C) + alpha * ar_dw;
            p_dh += (2 * Cw * dCw_dh + 2 * Ch * dCh_dh) * S / (C * C) + alpha * ar_dh;
        }
        if (Iw <= 0 || Ih <= 0) {
            p_dx = (2 * (truth[0] - pred[0]) * C - (2 * Cw * dCw_dx + 2 * Ch * dCh_dx) * S) / (C * C);
            p_dy = (2 * (truth[1] - pred[1]) * C - (2 * Cw * dCw_dy + 2 * Ch * dCh_dy) * S) / (C * C);
            p_dw = (2 * Cw * dCw_dw + 2 * Ch * dCh_dw) * S / (C * C) + alpha * ar_dw;
            p_dh = (2 * Cw * dCw_dh + 2 * Ch * dCh_dh) * S / (C * C) + alpha * ar_dh;
        }
        dx[0] = p_dx;
        dx[1] = p_dy;
        dx[2] = p_dw;
        dx[3] = p_dh;
        return dx;
    }

    public static float[] box_c(float[] a, float[] b) {
        float[] ba = new float[4];
        ba[0] = Float.min(a[1] - a[3] / 2, b[1] - b[3] / 2); //top
        ba[1] = Float.max(a[1] + a[3] / 2, b[1] + b[3] / 2); //bottom
        ba[2] = Float.min(a[0] - a[2] / 2, b[0] - b[2] / 2);  //left
        ba[3] = Float.max(a[0] + a[2] / 2, b[0] + b[2] / 2);  //right
        return ba;
    }

    public static float box_giou(float[] a, float[] b) {
        //intersection area
        float interArea = box_intersection(a, b);
        float unionArea = box_union(a, b);
        float iou = interArea / unionArea;
        //enclosed area
        float encolseArea = box_encolsed(a, b);
        //compute giou
        return iou - 1.0f * (encolseArea - unionArea) / encolseArea;
    }

    public static int entryIndex(int batch, int w, int h, int location, int entry, int outputs, int class_number) {
        int n = location / (w * h);
        int loc = location % (w * h);
        return batch * outputs + n * w * h * (4 + class_number + 1) + entry * w * h + loc;
    }

    public static YoloDetection[][] getYoloDetections(Tensor output, float[] anchors, int[] mask, int bbox_num, int outputs, int class_number, int orgW, int orgH, float thresh) {
        YoloDetection[][] dets = new YoloDetection[output.number][output.height * output.width * bbox_num];
        if (output.isHasGPU()) {
            output.syncHost();
        }
        int count = 0;
        for (int b = 0; b < output.number; b++) {
            for (int i = 0; i < output.height * output.width; i++) {
                int row = i / output.width;
                int col = i % output.width;
                int nCount = 0;
                for (int n = 0; n < bbox_num; n++) {
                    int n_index = n * output.width * output.height + row * output.width + col;
                    int obj_index = entryIndex(b, output.width, output.height, n_index, 4, outputs, class_number);
                    float objectness = output.data[obj_index];
                    //		            if(objectness > thresh) {
                    nCount++;
                    //		            System.out.println(objectness);
                    int box_index = entryIndex(b, output.width, output.height, n_index, 0, outputs, class_number);
                    YoloDetection det = new YoloDetection(class_number);
                    det.setBbox(getYoloBox(output, anchors, mask[n], box_index, col, row, output.width, output.height, orgW, orgH, output.height * output.width));
                    det.setObjectness(objectness);
                    //		            det.setClasses(classes);
                    float classes = 0.0f;
                    float max = 0.0f;
                    for (int j = 0; j < class_number; ++j) {
                        int class_index = entryIndex(b, output.width, output.height, n_index, 4 + 1 + j, outputs, class_number);
                        if (output.data[class_index] >= max) {
                            max = output.data[class_index];
                            classes = j;
                        }
                        float prob = objectness * output.data[class_index];
                        det.getProb()[j] = (prob > thresh) ? prob : 0;
                    }
                    det.setClasses(classes);
                    //		            System.out.println(b+":"+det.getBbox()[0] + ":" + det.getBbox()[1] + ":" + det.getBbox()[2] + ":" + det.getBbox()[3]);
                    dets[b][i * bbox_num + n] = det;
                    //		            }
                }
                if (nCount > 0) {
                    count++;
                }
            }
        }
        System.out.println("testCount:" + count);
        return dets;
    }

    public static YoloDetection[][] getYoloDetectionsV7(Tensor output, float[] anchors, int[] mask, int bbox_num, int outputs, int class_number, int orgW, int orgH, float thresh) {
        YoloDetection[][] dets = new YoloDetection[output.number][output.height * output.width * bbox_num];
        if (output.isHasGPU()) {
            output.syncHost();
        }
        //		System.out.println(JsonUtils.toJson(output.getByNumber(0)));
        int count = 0;
        for (int b = 0; b < output.number; b++) {
            for (int i = 0; i < output.height * output.width; i++) {
                int row = i / output.width;
                int col = i % output.width;
                int nCount = 0;
                for (int n = 0; n < bbox_num; n++) {
                    int n_index = n * output.width * output.height + row * output.width + col;
                    int obj_index = entryIndex(b, output.width, output.height, n_index, 4, outputs, class_number);
                    float objectness = output.data[obj_index];
                    //		            if(objectness > thresh) {
                    nCount++;
                    //		            System.out.println(objectness);
                    int box_index = entryIndex(b, output.width, output.height, n_index, 0, outputs, class_number);
                    YoloDetection det = new YoloDetection(class_number);
                    det.setBbox(getYoloBoxV7(output, anchors, mask[n], box_index, col, row, output.width, output.height, orgW, orgH, output.height * output.width));
                    det.setObjectness(objectness);
                    //		            det.setClasses(classes);
                    float classes = 0.0f;
                    float max = 0.0f;
                    for (int j = 0; j < class_number; ++j) {
                        int class_index = entryIndex(b, output.width, output.height, n_index, 4 + 1 + j, outputs, class_number);
                        if (output.data[class_index] >= max) {
                            max = output.data[class_index];
                            classes = j;
                        }
                        float prob = objectness * output.data[class_index];
                        det.getProb()[j] = (prob > thresh) ? prob : 0;
                    }
                    det.setClasses(classes);
                    //			            System.out.println(b+":"+det.getBbox()[0] + ":" + det.getBbox()[1] + ":" + det.getBbox()[2] + ":" + det.getBbox()[3]);
                    dets[b][i * bbox_num + n] = det;
                    //		            }
                }
                if (nCount > 0) {
                    count++;
                }
            }
        }
        System.out.println("testCount:" + count);
        return dets;
    }

    public static void correntYoloBoxes(List<YoloDetection> dets, int n, int w, int h, int orgw, int orgh, boolean relative) {
        int new_w = 0;
        int new_h = 0;
        if (((float) orgw / w) < ((float) orgh / h)) {
            new_w = orgw;
            new_h = (h * orgw) / w;
        } else {
            new_h = orgh;
            new_w = (w * orgh) / h;
        }
        for (int i = 0; i < n; ++i) {
            float[] bbox = dets.get(i).getBbox();
            bbox[0] = (float) ((bbox[0] - (orgw - new_w) / 2. / orgw) / ((float) new_w / orgw));
            bbox[1] = (float) ((bbox[1] - (orgh - new_h) / 2. / orgh) / ((float) new_h / orgh));
            bbox[2] *= (float) orgw / new_w;
            bbox[3] *= (float) orgh / new_h;
            if (!relative) {
                bbox[0] *= w;
                bbox[1] *= w;
                bbox[2] *= h;
                bbox[3] *= h;
            }
            dets.get(i).setBbox(bbox);
        }
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
    public static float[] getYoloBoxV7(Tensor x, float[] anchors, int n, int index, int i, int j, int lw, int lh, int w, int h, int stride) {
        float[] box = new float[4];
        box[0] = (i + x.data[index + 0 * stride]) / lw;
        box[1] = (j + x.data[index + 1 * stride]) / lh;
        box[2] = (float) x.data[index + 2 * stride] * x.data[index + 2 * stride] * 4 * anchors[2 * n] / w;
        box[3] = (float) x.data[index + 3 * stride] * x.data[index + 3 * stride] * 4 * anchors[2 * n + 1] / h;
        return box;
    }
}

