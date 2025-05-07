package com.omega.example.yolo.utils;

import com.omega.common.data.Tensor;
import com.omega.common.utils.JsonUtils;
import com.omega.common.utils.MatrixOperation;
import com.omega.common.utils.RandomUtils;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class AnchorBoxUtils {
    public static Tensor sort(Tensor centers) {
        Tensor result = new Tensor(centers.number, 1, 1, centers.width);
        for (int i = 0; i < centers.number; i++) {
            float w = centers.data[i * 2 + 0];
            float h = centers.data[i * 2 + 1];
            float product = w * h;
            int j;
            for (j = i; j > 0 && product < (centers.data[(j - 1) * 2 + 0] * centers.data[(j - 1) * 2 + 1]); j--) {
                centers.data[j * 2 + 0] = centers.data[(j - 1) * 2 + 0];
                centers.data[j * 2 + 1] = centers.data[(j - 1) * 2 + 1];
            }
            centers.data[j * 2 + 0] = w;
            centers.data[j * 2 + 1] = h;
        }
        return result;
    }

    public static Tensor kmeansAnchors(String trainLabelPath, int k) {
        try {
            Tensor bbox = LabelUtils.loadBoxTXT(trainLabelPath);
            Tensor anchors = AnchorBoxUtils.calcAnchors(bbox, k);
            sort(anchors);
            return anchors;
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
        return null;
    }

    public static Tensor calcAnchors(Tensor bboxs, int num_clusters) {
        int num_labels = bboxs.number;
        Tensor wh = new Tensor(num_labels, 1, 1, 2);
        int[] assig = new int[wh.number];
        for (int n = 0; n < num_labels; n++) {
            wh.data[n * 2 + 0] = bboxs.data[n * 4 + 2];
            wh.data[n * 2 + 1] = bboxs.data[n * 4 + 3];
        }
        Tensor centers = randomCenters(wh, num_clusters);
        for (int i = 0; i < 1000 && kmeans_exp(wh, assig, centers) == 0; i++) {
            kmeans_max(wh, assig, centers);
        }
        return centers;
    }

    public static void kmeans_max(Tensor wh, int[] assig, Tensor centers) {
        float[] old_centers = new float[centers.dataLength];
        int[] counts = new int[centers.number];
        for (int i = 0; i < centers.number; i++) {
            for (int j = 0; j < centers.getOnceSize(); j++) {
                old_centers[i * centers.getOnceSize() + j] = centers.data[i * wh.getOnceSize() + j];
                centers.data[i * centers.getOnceSize() + j] = 0;
            }
        }
        for (int i = 0; i < wh.number; i++) {
            counts[assig[i]]++;
            for (int j = 0; j < wh.getOnceSize(); j++) {
                centers.data[assig[i] * centers.getOnceSize() + j] += wh.data[i * wh.getOnceSize() + j];
            }
        }
        for (int i = 0; i < centers.number; i++) {
            if (counts[i] > 0) {
                for (int j = 0; j < centers.getOnceSize(); j++) {
                    centers.data[i * centers.getOnceSize() + j] /= counts[i];
                }
            }
        }
        for (int i = 0; i < centers.number; i++) {
            for (int j = 0; j < centers.getOnceSize(); j++) {
                if (centers.data[i * centers.getOnceSize() + j] == 0) {
                    centers.data[i * centers.getOnceSize() + j] = old_centers[i * centers.getOnceSize() + j];
                }
            }
        }
    }

    public static int kmeans_exp(Tensor wh, int[] assig, Tensor centers) {
        int converged = 1;
        for (int i = 0; i < wh.number; i++) {
            float[] datum = new float[2];
            datum[0] = wh.data[i * 2 + 0];
            datum[1] = wh.data[i * 2 + 1];
            int closest = closest_center(datum, centers);
            if (closest != assig[i]) {
                converged = 0;
            }
            assig[i] = closest;
        }
        return converged;
    }

    public static int closest_center(float[] datum, Tensor centers) {
        int best = 0;
        int cols = centers.getOnceSize();
        float best_dist = dist(datum, centers.getByNumber(best), cols);
        for (int i = 0; i < centers.number; i++) {
            float new_dist = dist(datum, centers.getByNumber(i), cols);
            if (new_dist < best_dist) {
                best_dist = new_dist;
                best = i;
            }
        }
        return best;
    }

    public static float dist(float[] x, float[] y, int n) {
        float mw = (x[0] < y[0]) ? x[0] : y[0];
        float mh = (x[1] < y[1]) ? x[1] : y[1];
        float inter = mw * mh;
        float sum = x[0] * x[1] + y[0] * y[1];
        float un = sum - inter;
        float iou = inter / un;
        return 1 - iou;
    }

    public static Tensor randomCenters(Tensor wh, int k) {
        Tensor centers = new Tensor(k, 1, 1, 2);
        List<Integer> list = new ArrayList<Integer>();
        for (int i = 0; i < wh.number; i++) {
            list.add(i);
        }
        Collections.shuffle(list);
        for (int n = 0; n < k; n++) {
            System.arraycopy(wh.getByNumber(list.get(n)), 0, centers.data, n * wh.getOnceSize(), wh.getOnceSize());
        }
        return centers;
    }

    public static Tensor getAnchorBox(Tensor bboxs, int num_clusters) {
        Tensor boxs = translateBoxes(bboxs);
        int N = boxs.number;
        float[][] dis = new float[N][num_clusters];
        int[] last_clu = new int[N];
        Tensor clu = RandomUtils.random(boxs, num_clusters);
        while (true) {
            for (int n = 0; n < bboxs.number; n++) {
                dis[n] = MatrixOperation.subtraction(1.0f, iou(boxs.getByNumber(n), clu));
            }
            //获取类聚中心
            int[] nearest_clusters = argmin(dis);
            boolean isAll = true;
            for (int i = 0; i < nearest_clusters.length; i++) {
                if (last_clu[i] != nearest_clusters[i]) {
                    isAll = false;
                }
            }
            if (isAll) {
                break;
            }
            for (int k = 0; k < num_clusters; k++) {
                clu.setByNumber(k, median(boxs, nearest_clusters, k));
            }
            /**
             * copy nearest_clusters to last_clu

             */
            System.arraycopy(nearest_clusters, 0, last_clu, 0, last_clu.length);
        }
        return clu;
    }

    public static float[] median(Tensor boxs, int[] nearest_clu, int cidx) {
        float[] result = new float[2];
        List<Float> w = new ArrayList<Float>();
        List<Float> h = new ArrayList<Float>();
        for (int n = 0; n < nearest_clu.length; n++) {
            if (nearest_clu[n] == cidx) {
                w.add(boxs.getByNumber(n)[0]);
                h.add(boxs.getByNumber(n)[1]);
            }
        }
        result[0] = median(w);
        result[1] = median(h);
        return result;
    }

    private static float median(List<Float> total) {
        float j = 0;
        //集合排序
        Collections.sort(total);
        int size = total.size();
        if (size % 2 == 1) {
            j = total.get((size - 1) / 2);
        } else {
            j = ((total.get(size / 2 - 1) + total.get(size / 2)) / 2.0f);
        }
        return j;
    }

    public static int[] argmin(float[][] dis) {
        int[] arg = new int[dis.length];
        for (int n = 0; n < arg.length; n++) {
            arg[n] = minIndex(dis[n]);
        }
        return arg;
    }

    public static int minIndex(float[] x) {
        int index = 0;
        float temp = x[0];
        for (int i = 1; i < x.length; i++) {
            if (x[i] < temp) {
                temp = x[i];
                index = i;
            }
        }
        return index;
    }

    public static Tensor translateBoxes(Tensor boxs) {
        Tensor wh = new Tensor(boxs.number, 1, 1, boxs.width / 2);
        for (int n = 0; n < boxs.number; n++) {
            wh.data[n * wh.width + 0] = Math.abs(boxs.data[n * boxs.width + 2] - boxs.data[n * boxs.width + 0]);
            wh.data[n * wh.width + 1] = Math.abs(boxs.data[n * boxs.width + 3] - boxs.data[n * boxs.width + 1]);
        }
        return wh;
    }

    public static float[] iou(float[] box, Tensor clusters) {
        float[] iou = new float[clusters.number];
        for (int k = 0; k < clusters.number; k++) {
            float x = clusters.data[k * 2 + 0] < box[0] ? clusters.data[k * 2 + 0] : box[0];
            float y = clusters.data[k * 2 + 1] < box[1] ? clusters.data[k * 2 + 1] : box[1];
            float intersection = x * y;
            float box_area = box[0] * box[1];
            float cluster_area = clusters.data[k * 2 + 0] * clusters.data[k * 2 + 1];
            iou[k] = intersection / (box_area + cluster_area - intersection);
        }
        return iou;
    }

    public static void main(String[] args) {
        int k = 2;
        int n = 4;
        float[] data = new float[]{183, 63, 241, 112, 26, 86, 79, 133, 139, 108, 178, 148, 20, 130, 63, 170};
        Tensor boxs = new Tensor(n, 1, 1, 4, data);
        Tensor anchors = getAnchorBox(boxs, k);
        for (int i = 0; i < k; i++) {
            System.out.println(JsonUtils.toJson(anchors.getByNumber(i)));
        }
        float[] data2 = new float[]{10, 14, 23, 27, 37, 58, 81, 82, 135, 169, 344, 319};
        int[] data3 = new int[data2.length];
        for (int i = 0; i < data2.length; i++) {
            data3[i] = (int) (data2[i] / 416 * 256);
        }
        System.out.println(JsonUtils.toJson(data3));
    }
}

