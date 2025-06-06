package com.omega.engine.nn.grad;

import com.omega.common.utils.MatrixOperation;
import com.omega.engine.tensor.Tensor;

/**
 * 梯度裁剪
 *
 * @author Administrator
 */
public class GradClipping {
    public static Tensor gradClipping(Tensor grad, float theta) {
        if (grad.isHasGPU()) {
            grad.syncHost();
            grad_clipping_cpu(grad.data, theta);
            grad.hostToDevice();
        } else {
            grad_clipping_cpu(grad.data, theta);
        }
        return grad;
    }

    public static float[] grad_clipping_cpu(float[] data, float theta) {
        float[] power = MatrixOperation.pow(data, 2);
        float norm = (float) Math.sqrt(MatrixOperation.sum(power));
        System.out.println(norm);
        if (norm > theta) {
            data = MatrixOperation.multiplication(data, theta / norm);
        }
        return data;
    }
}

