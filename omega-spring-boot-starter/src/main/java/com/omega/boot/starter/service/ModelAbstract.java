package com.omega.boot.starter.service;

import com.omega.common.utils.ImageUtils;
import com.omega.common.utils.MatrixOperation;
import com.omega.common.utils.MatrixUtils;
import com.omega.common.utils.RandomUtils;
import com.omega.engine.gpu.CUDAManager;
import com.omega.engine.gpu.SoftmaxKernel;
import com.omega.engine.tensor.Tensor;
import com.omega.example.yolo.data.DetectionDataLoader;
import com.omega.example.yolo.model.YoloBox;
import com.omega.example.yolo.model.YoloDetection;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * 模型抽象公共方法类
 *
 * @author haylee
 * @date 2025/05/14 14:33
 */
public class ModelAbstract extends TokenizerAbstract{

    public static Tensor loadByTxtToIdx(Tensor testInput, int[] idxs) {
        testInput = Tensor.createTensor(testInput, idxs.length, 1, 1, 1, true);
        for (int t = 0; t < idxs.length; t++) {
            testInput.data[t] = idxs[t];
        }
        testInput.hostToDevice();
        return testInput;
    }

    public static int output2NextIDXTopN(Tensor output, int nextTokenIdx, int topK, CUDAManager cudaManager) {
        SoftmaxKernel kernel = new SoftmaxKernel(cudaManager);
        Tensor tmp = new Tensor(1, 1, 1, output.width, true);
        Tensor prof = new Tensor(1, 1, 1, output.width, true);
        if (nextTokenIdx < output.number) {
            tmp.hostToDevice(MatrixOperation.multiplication(output.getByNumber(nextTokenIdx), 0.7f));
            kernel.softmax_out(tmp, prof);
            return pickTopN(prof.syncHost(), topK);
        }
        return 0;
    }

    public static int output2NextIDX(Tensor output, int nextTokenIdx, int topK) {
        if (nextTokenIdx < output.number) {
            return pickTopN(output.getByNumber(nextTokenIdx), topK);
        }
        return 0;
    }
    public static int pickTopN(float[] x, int n) {
        float[] sort = Arrays.copyOf(x, x.length);
        Arrays.sort(sort);
        float[] topN = Arrays.copyOfRange(sort, sort.length - n, sort.length);
        float v = topN[RandomUtils.getRandomNumber(topN)];
        for (int i = 0; i < x.length; i++) {
            if (v == x[i]) {
                return i;
            }
        }
        return 0;
    }

    public static List<String> showImg(String outputPath, DetectionDataLoader dataSet, int class_num, List<YoloBox> score_bbox, int batchSize, boolean format, int im_w, int im_h, String[] labelset) {
        List<String> urls = new ArrayList<>();
        ImageUtils utils = new ImageUtils();
        int lastIndex = dataSet.number % batchSize;
        for (int b = 0; b < dataSet.number; b++) {
            float[] once = dataSet.loadData(b);
            once = MatrixOperation.multiplication(once, 255.0f);
            int bbox_index = b;
            if (b >= (dataSet.number - lastIndex)) {
                bbox_index = b + (batchSize - lastIndex);
            }
            YoloBox box = score_bbox.get(bbox_index);
            List<Integer> indexs = new ArrayList<Integer>();
            for (int l = 0; l < box.getDets().size(); l++) {
                if (box.getDets().get(l) != null && box.getDets().get(l).getObjectness() > 0 && !MatrixUtils.isZero(box.getDets().get(l).getProb())) {
                    indexs.add(l);
                }
            }
            int[][] bbox = new int[indexs.size()][5];
            for (int i = 0; i < indexs.size(); i++) {
                Integer index = indexs.get(i);
                YoloDetection det = box.getDets().get(index);
                bbox[i][0] = (int) det.getClasses();
                bbox[i][1] = (int) ((det.getBbox()[0] - det.getBbox()[2] / 2.0f) * im_w);
                bbox[i][2] = (int) ((det.getBbox()[1] - det.getBbox()[3] / 2.0f) * im_h);
                bbox[i][3] = (int) ((det.getBbox()[0] + det.getBbox()[2] / 2.0f) * im_w);
                bbox[i][4] = (int) ((det.getBbox()[1] + det.getBbox()[3] / 2.0f) * im_h);
            }
            utils.createRGBImage(outputPath + "_" + b + ".png", "png", ImageUtils.color2rgb2(once, im_w, im_h, format), im_w, im_h, bbox, labelset);
            urls.add(outputPath + "_" + b + ".png");
        }
        return urls;
    }
}
