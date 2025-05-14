package com.omega.boot.starter.service;

import com.omega.common.data.Tensor;
import com.omega.common.utils.MatrixOperation;
import com.omega.common.utils.RandomUtils;
import com.omega.engine.gpu.CUDAManager;
import com.omega.engine.gpu.SoftmaxKernel;

import java.util.Arrays;

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
}
