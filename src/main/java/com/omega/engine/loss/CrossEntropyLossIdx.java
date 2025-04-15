package com.omega.engine.loss;

import com.omega.common.config.Tensor;
import com.omega.common.utils.JsonUtils;
import com.omega.common.utils.MatrixUtils;
import com.omega.engine.gpu.CUDAManager;
import com.omega.engine.loss.gpu.CrossEntropyKernel;
import com.omega.engine.nn.network.Network;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

/**
 * Cross Entropy loss function
 *
 * @author Administrator
 * @loss: - ∑ y * ln(f(x))
 * @diff: - ∑ y * (1 / f(x))
 */
public class CrossEntropyLossIdx extends LossFunction {
    private static CrossEntropyLossIdx instance;
    public final LossType lossType = LossType.softmax_with_cross_entropy_idx;
    //	private Tensor output;
    private Tensor loss;
    private Tensor probs;
    private Tensor diff;
    //	private SoftmaxKernel softmaxKernel;
    private CrossEntropyKernel crossEntropyKernel;

    public CrossEntropyLossIdx(Network network) {
        setNet(network);
        crossEntropyKernel = new CrossEntropyKernel(network.cudaManager);
    }

    public CrossEntropyLossIdx(CUDAManager cudaManager) {
        crossEntropyKernel = new CrossEntropyKernel(cudaManager);
    }

    public static CrossEntropyLossIdx operation(CUDAManager cudaManager) {
        if (instance == null) {
            instance = new CrossEntropyLossIdx(cudaManager);
        }
        return instance;
    }

    public static List<List<List<Double>>> readJsonFileSmallJson(String path) {
        List<List<List<Double>>> mapList = new ArrayList<List<List<Double>>>();
        try {
            String jsonString = new String(Files.readAllBytes(Paths.get(path)));
            mapList = JsonUtils.gson.fromJson(jsonString, mapList.getClass());
            return mapList;
        } catch (IOException e) {
            e.printStackTrace();
        }
        return null;
    }

    public static int getDim(Tensor x) {
        int dim = 0;
        if (x.number > 1) {
            dim++;
        }
        if (x.channel > 1) {
            dim++;
        }
        if (x.height > 1) {
            dim++;
        }
        if (x.width > 1) {
            dim++;
        }
        return dim;
    }

    public static void loadData(Tensor x, Object meta, String key) {
        if (meta != null) {
            List<List<List<Double>>> dataA = (List<List<List<Double>>>) meta;
            int N = x.number;
            int C = x.channel;
            int W = x.width;
            for (int n = 0; n < N; n++) {
                for (int c = 0; c < C; c++) {
                    for (int w = 0; w < W; w++) {
                        x.data[n * x.getOnceSize() + c * W + w] = dataA.get(n).get(c).get(w).floatValue();
                    }
                }
            }
            x.hostToDevice();
            System.out.println(key + "_finish.");
        }
    }

    public static void main(String[] args) {
        CUDAManager cudaManager = new CUDAManager(0);
        //		List<List<List<Double>>> weightMap = readJsonFileSmallJson("H:\\transformer_dataset\\6400\\tensor.json");
        //		Tensor xp = new Tensor(1, 512, 1, 6400, true);
        //		loadData(xp, weightMap, "");
        float[] label = new float[]{551, 3438, 269, 3347, 1988, 4391, 4541, 270, 1866, 451, 3827, 1988, 2094, 4391, 733, 107, 269, 2669, 286, 201, 265, 120, 3600, 3347, 1988, 4391, 4541, 2559, 1911, 355, 4514, 763, 1486, 270, 622, 110, 872, 315, 622, 110, 3654, 919, 639, 4541, 453, 1122, 4391, 270, 3744, 1775, 3757, 3654, 315, 4375, 345, 4308, 269, 270, 1046, 2026, 2907, 5961, 639, 3811, 229, 3963, 253, 270, 1933, 650, 2907, 294, 102, 453, 3548, 270, 549, 166, 255, 108, 708, 1988, 2262, 286, 201, 265, 120, 3600, 3347, 1988, 4391, 4541, 1008, 355, 201, 19, 16, 223, 845, 1243, 639, 355, 845, 1243, 639, 3597, 1346, 617, 2075, 1346, 617, 2406, 2773, 4075, 507, 270, 453, 2268, 845, 336, 101, 554, 2120, 4424, 6126, 270, 845, 1243, 639, 587, 270, 1988, 4391, 4541, 1343, 1237, 5047, 341, 3548, 341, 4308, 537, 2140, 553, 623, 834, 413, 270, 631, 763, 6236, 1988, 505, 242, 4391, 269, 2503, 1674, 1436, 726, 270, 1343, 549, 2687, 554, 2588, 508, 4424, 270, 650, 1988, 4391, 4541, 378, 117, 233, 4391, 1520, 2818, 1197, 668, 859, 286, 201, 20, 16, 223, 1988, 4391, 1723, 355, 1988, 4391, 1723, 345, 1988, 4391, 4541, 269, 682, 1953, 4985, 270, 168, 234, 115, 1415, 856, 1467, 4985, 3268, 232, 1552, 3951, 605, 508, 3601, 572, 270, 4536, 2053, 508, 647, 1044, 378, 117, 233, 4391, 4541, 270, 4981, 1988, 4391, 4541, 378, 117, 233, 4391, 3053, 286, 0, 2, 1, 741, 1170, 368, 3783, 5338, 413, 2105, 1325, 6234, 269, 1294, 415, 2504, 446, 1492, 333, 2063, 1884, 315, 6210, 1492, 333, 686, 819, 3932, 270, 1751, 368, 1199, 3827, 1409, 2658, 510, 415, 2236, 3123, 115, 1016, 270, 1323, 557, 966, 4808, 2191, 2073, 681, 270, 6234, 451, 3897, 4102, 2259, 269, 3793, 2796, 415, 2236, 5154, 2217, 270, 650, 5154, 1946, 2236, 570, 468, 115, 2111, 1892, 270, 468, 115, 2111, 2259, 270, 4243, 451, 6165, 945, 2259, 269, 2844, 1842, 757, 6234, 269, 4418, 595, 757, 619, 286, 1314, 4808, 2191, 2073, 681, 400, 1121, 553, 2259, 269, 4274, 1409, 286, 4808, 2191, 2073, 681, 345, 2259, 1761, 415, 2236, 1852, 4648, 269, 4463, 4101, 286, 2259, 368, 850, 654, 508, 3788, 2902, 3072, 2658, 451, 468, 110, 474, 122, 1434, 5154, 800, 1325, 270, 3425, 510, 3438, 6088, 1495, 5388, 1154, 415, 2236, 4647, 286, 201, 495, 102, 694, 234, 2610, 3788, 595, 4238, 464, 252, 766, 4564, 4238, 931, 315, 5008, 3670, 740, 824, 3788, 1543, 595, 4238, 3554, 6145, 796, 226, 847, 110, 4238, 931, 4536, 2504, 446, 1425, 544, 6234, 415, 2236, 5154, 1994, 270, 581, 510, 1492, 333, 2063, 1884, 269, 654, 1641, 1409, 2921, 1313, 102, 314, 105, 315, 5432, 4978, 270, 2921, 658, 315, 1199, 953, 967, 4444, 341, 6159, 269, 3072, 270, 4935, 1735, 5912, 270, 650, 6234, 315, 2259, 1761, 4605, 269, 2844, 1730, 2134, 270, 2192, 973, 496, 1149, 931, 1175, 3072, 5945, 286, 201, 833, 1281, 2073, 681, 4418, 595, 5354, 2009, 5571, 774, 270, 345, 6210, 1066, 105, 819, 654, 1122, 829, 4808, 2191};
        Tensor labelt = new Tensor(512, 1, 1, 1, label, true);
        int N = 1;
        int T = 512;
        int W = 6400;
        float[] x = MatrixUtils.order(N * T * W, 0.01f, 0.01f);
        Tensor xp = new Tensor(N, T, 1, W, x, true);
        //		float[] label = new float[] {1, 8};
        //		Tensor labelt = new Tensor(2, 1, 1, 1, label, true);
        //		float max = MatrixOperation.max(x);
        //
        //		float[] tmp = MatrixOperation.subtraction(x, max);
        //
        //		float ln = (float) Math.log(MatrixOperation.sum(MatrixOperation.exp(tmp)));
        //		PrintUtils.printImage(MatrixOperation.subtraction(tmp, ln));
        //		xp.showDM();
        Tensor loss = CrossEntropyLossIdx.operation(cudaManager).loss(xp, labelt, -1);
        loss.showDM();
        //		PrintUtils.printImage(loss.syncHost());
        System.out.println();
        //		System.out.println("loss:"+JsonUtils.toJson(MatrixOperation.sum(loss.syncHost())/1));
        //		Tensor diff = CrossEntropyLossIdx.operation().diff(xt, labelt);
        //
        //		System.out.println("diff:"+JsonUtils.toJson(diff.syncHost()));
        //		System.out.println(Math.log(Math.exp(-1.3470f)/sum));
        //
        //		float d_yhat_k_x = yhat_k * (1 - yhat_k);
        //
        //		float d_l_yhat_k = - 1 / yhat_k;
        //
        //		System.out.println(d_yhat_k_x * d_l_yhat_k);
    }

    public void init(Tensor input) {
        if (loss == null || loss.number != input.number) {
            this.loss = new Tensor(input.number, 1, 1, 1, true);
            this.probs = new Tensor(input.number, input.channel, input.height, input.width, true);
            //			this.output = new Tensor(input.number, input.channel, input.height, input.width, true);
            this.diff = new Tensor(input.number, input.channel, input.height, input.width, true);
        }
    }

    public void initKernel() {
        //		softmaxKernel = new SoftmaxKernel();
        crossEntropyKernel = new CrossEntropyKernel(getNet().cudaManager);
    }

    @Override
    public LossType getLossType() {
        // TODO Auto-generated method stub
        return LossType.cross_entropy;
    }

    @Override
    public Tensor loss(Tensor x, Tensor label) {
        // TODO Auto-generated method stub
        init(x);
        /**
         * q(x) = softmax(x) H(p,q) = - ∑p(x)logq(x) 简化log_softmax:
         * log(exp(xi)/sum(exp(X))) = (xi - max) - log(sum(exp(xi - max)))
         * 该操作为了防止上溢出与下溢出情况导致nan与inf出现.
         *
         */
        //		crossEntropyKernel.softmax(x, probs);
        //
        //		crossEntropyKernel.crossentropy(probs, label, loss);
        //		System.out.println("in");
        crossEntropyKernel.forwardIDX2(x, label, probs, loss, -99999);
        return loss;
    }

    @Override
    public Tensor diff(Tensor x, Tensor label) {
        // TODO Auto-generated method stub
        /**
         * diff(x) = softmax(x) - label
         *
         */
        //		crossEntropyKernel.crossentropy_backward(probs, label, diff);
        crossEntropyKernel.backwardIDX2(probs, label, diff, -99999);
        return diff;
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
        init(x);
        /**
         * q(x) = softmax(x) H(p,q) = - ∑p(x)logq(x) 简化log_softmax:
         * log(exp(xi)/sum(exp(X))) = (xi - max) - log(sum(exp(xi - max)))
         * 该操作为了防止上溢出与下溢出情况导致nan与inf出现.
         *
         */
        //		crossEntropyKernel.softmax(x, probs);
        //
        //		crossEntropyKernel.crossentropy(probs, label, loss);
        crossEntropyKernel.forwardIDX2(x, label, probs, loss, -99999);
        return loss;
    }

    @Override
    public Tensor diff(Tensor x, Tensor label, Tensor diff) {
        // TODO Auto-generated method stub
        /**
         * diff(x) = softmax(x) - label
         *
         */
        //		crossEntropyKernel.crossentropy_backward(probs, label, diff);
        crossEntropyKernel.backwardIDX2(probs, label, diff, -99999);
        return diff;
    }

    @Override
    public Tensor loss(Tensor x, Tensor label, int igonre) {
        // TODO Auto-generated method stub
        init(x);
        /**
         * q(x) = softmax(x) H(p,q) = - ∑p(x)logq(x) 简化log_softmax:
         * log(exp(xi)/sum(exp(X))) = (xi - max) - log(sum(exp(xi - max)))
         * 该操作为了防止上溢出与下溢出情况导致nan与inf出现.
         *
         */
        //		crossEntropyKernel.softmax(x, probs);
        //		probs.showDM(0);
        //		JCuda.cudaDeviceSynchronize();
        ////		probs.showDMByNumber(0);
        //		crossEntropyKernel.crossentropy_igone(probs, label, loss, igonre);
        //		JCuda.cudaDeviceSynchronize();
        //		loss.showDM();
        //		x.showShape();
        //		label.showShape();
        //		x.showDMByOffsetRed(0, 100, "dffsd");
        //		label.showDM();
        crossEntropyKernel.forwardIDX2(x, label, probs, loss, igonre);
        //		loss.showDMByOffsetRed(0, 100, "loss");
        return loss;
    }

    @Override
    public Tensor diff(Tensor x, Tensor label, int igonre) {
        // TODO Auto-generated method stub
        /**
         * diff(x) = softmax(x) - label
         *
         */
        //		probs.showDMByNumber(0);
        //		crossEntropyKernel.crossentropy_backward_igone(probs, label, diff, igonre);
        crossEntropyKernel.backwardIDX2(probs, label, diff, igonre);
        return diff;
    }

    @Override
    public Tensor diff(Tensor x, Tensor label, int igonre, int count) {
        // TODO Auto-generated method stub
        crossEntropyKernel.backwardIDX2(probs, label, diff, igonre, count);
        return diff;
    }
}
