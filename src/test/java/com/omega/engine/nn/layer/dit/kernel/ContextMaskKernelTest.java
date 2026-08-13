package com.omega.engine.nn.layer.dit.kernel;

import com.omega.engine.gpu.CUDAManager;
import com.omega.engine.tensor.Tensor;

public class ContextMaskKernelTest {

    private static final float EPSILON = 1.0e-6f;

    public static void main(String[] args) {
        int batch = 2;
        int time = 3;
        int hidden = 4;
        float[] contextData = new float[batch * time * hidden];
        float[] deltaData = new float[contextData.length];
        for (int i = 0; i < contextData.length; i++) {
            contextData[i] = i + 1.0f;
            deltaData[i] = 0.1f * (i + 1.0f);
        }

        Tensor context = new Tensor(batch * time, 1, 1, hidden, contextData, true);
        Tensor mask = new Tensor(batch, 1, 1, time,
                new float[]{1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f}, true);
        Tensor maskToken = new Tensor(1, 1, 1, hidden,
                new float[]{-1.0f, -2.0f, -3.0f, -4.0f}, true);
        Tensor dropMask = new Tensor(batch, 1, 1, 1,
                new float[]{0.05f, 0.8f}, true);
        Tensor fusedMask = new Tensor(batch, 1, 1, time, true);
        Tensor output = new Tensor(batch * time, 1, 1, hidden, true);
        Tensor delta = new Tensor(batch * time, 1, 1, hidden, deltaData, true);
        Tensor diffContext = new Tensor(batch * time, 1, 1, hidden, true);
        Tensor diffMaskToken = new Tensor(1, 1, 1, hidden, true);
        Tensor diffMaskTokenOnly = new Tensor(1, 1, 1, hidden, true);

        ContextMaskKernel kernel = new ContextMaskKernel(new CUDAManager(0));
        kernel.fuseAttnMask(mask, dropMask, fusedMask, time, 0.1f);
        kernel.forward(context, mask, maskToken, output);
        kernel.backward(delta, mask, diffContext, diffMaskToken);
        kernel.backwardMaskToken(delta, mask, diffMaskTokenOnly);

        output.syncHost();
        fusedMask.syncHost();
        diffContext.syncHost();
        diffMaskToken.syncHost();
        diffMaskTokenOnly.syncHost();

        float[] expectedMaskGrad = new float[hidden];
        float[] expectedFusedMask = new float[]{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f};
        for (int i = 0; i < expectedFusedMask.length; i++) {
            assertClose(fusedMask.data[i], expectedFusedMask[i]);
        }
        for (int row = 0; row < batch * time; row++) {
            boolean valid = mask.data[row] == 1.0f;
            for (int h = 0; h < hidden; h++) {
                int index = row * hidden + h;
                assertClose(output.data[index], valid ? contextData[index] : maskToken.data[h]);
                assertClose(diffContext.data[index], valid ? deltaData[index] : 0.0f);
                if (!valid) {
                    expectedMaskGrad[h] += deltaData[index];
                }
            }
        }
        for (int h = 0; h < hidden; h++) {
            assertClose(diffMaskToken.data[h], expectedMaskGrad[h]);
            assertClose(diffMaskTokenOnly.data[h], expectedMaskGrad[h]);
        }

        System.out.println("ContextMaskKernel forward/backward test passed.");
    }

    private static void assertClose(float actual, float expected) {
        if (Math.abs(actual - expected) > EPSILON) {
            throw new AssertionError("Expected " + expected + ", got " + actual);
        }
    }
}
