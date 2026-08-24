package com.omega.example.dit.test;

import java.lang.reflect.Field;
import java.util.Arrays;

import com.omega.common.utils.RandomUtils;
import com.omega.engine.gpu.CUDAMemoryManager;
import com.omega.engine.loss.LossType;
import com.omega.engine.nn.layer.FullyLayer;
import com.omega.engine.nn.layer.dit.modules.DiTAttentionLayer2;
import com.omega.engine.nn.layer.gpu.AttentionKernel;
import com.omega.engine.nn.layer.gpu.CudnnFlashAttentionKernel;
import com.omega.engine.nn.network.BPNetwork;
import com.omega.engine.nn.network.RunModel;
import com.omega.engine.tensor.Tensor;
import com.omega.engine.updater.UpdaterType;

import jcuda.runtime.JCuda;

/**
 * Compares the final output of DiTAttentionLayer2 with NVIDIA cuDNN SDPA.
 *
 * Run this on an SM80+ GPU with the native demo library installed. Both paths
 * use the same Q/K/V/O projection parameters. The cuDNN path receives exactly
 * the same normalized BHSD Q/K and permuted V as the reference path.
 */
public class CudnnFlashAttentionCompareDemo {

    private static final int[] PERMUTE_0213 = new int[] {0, 2, 1, 3};

    public static void main(String[] args) {
        if (!CudnnFlashAttentionKernel.isLoaded()) {
            throw new IllegalStateException("cuDNN SDPA JNI library is unavailable: "
                    + CudnnFlashAttentionKernel.getLoadError());
        }

        System.out.println("cuDNN version: " + CudnnFlashAttentionKernel.getCudnnVersion());
        String qkNormMode = System.getProperty("omega.sdpa.demo.qkNorm", "both");
        if ("false".equalsIgnoreCase(qkNormMode) || "both".equalsIgnoreCase(qkNormMode)) {
            compare(false);
        }
        if ("true".equalsIgnoreCase(qkNormMode) || "both".equalsIgnoreCase(qkNormMode)) {
            compare(true);
        }
        if (!"false".equalsIgnoreCase(qkNormMode)
                && !"true".equalsIgnoreCase(qkNormMode)
                && !"both".equalsIgnoreCase(qkNormMode)) {
            throw new IllegalArgumentException(
                    "omega.sdpa.demo.qkNorm must be false, true, or both.");
        }
        CUDAMemoryManager.free();
    }

    private static void compare(boolean qkNorm) {
        int batchSize = Integer.getInteger("omega.sdpa.demo.batch", 2);
        int time = Integer.getInteger("omega.sdpa.demo.time", 333);
        int headNum = Integer.getInteger("omega.sdpa.demo.heads", 12);
        int headDim = Integer.getInteger("omega.sdpa.demo.headDim", 64);
        int embedDim = headNum * headDim;
        boolean bias = true;

        BPNetwork network = new BPNetwork(LossType.MSE, UpdaterType.none);
        network.CUDNN = true;
        network.RUN_MODEL = RunModel.TRAIN;

        DiTAttentionLayer2 reference = new DiTAttentionLayer2(
                embedDim, headNum, time, bias, qkNorm, network);

        int inputLength = batchSize * time * embedDim;
        Tensor input = new Tensor(batchSize * time, 1, 1, embedDim,
                RandomUtils.gaussianRandom(inputLength, 0.02f), true);

        MemorySnapshot beforeReferenceForward = MemorySnapshot.capture();
        reference.forward(input);
        JCuda.cudaDeviceSynchronize();
        MemorySnapshot afterReferenceForward = MemorySnapshot.capture();
        float[] expected = reference.getOutput().syncHost().clone();

        Tensor q = getQ(reference, network, batchSize, time, headNum, headDim, qkNorm);
        Tensor k = getK(reference, network, batchSize, time, headNum, headDim, qkNorm);
        Tensor v = permuteProjection(reference.vLinerLayer.getOutput(), network,
                batchSize, time, headNum, headDim);

        Tensor attentionOutput = new Tensor(batchSize, headNum, time, headDim, true);
        Tensor unpermuted = new Tensor(batchSize * time, 1, 1, embedDim, true);
        Tensor actualOutput = new Tensor(batchSize * time, 1, 1, embedDim, true);

        FullyLayer outputProjection = new FullyLayer(embedDim, embedDim, bias, network);
        reference.oLinerLayer.weight.copyGPU(outputProjection.weight);
        if (bias) {
            reference.oLinerLayer.bias.copyGPU(outputProjection.bias);
        }

        MemorySnapshot beforeSdpaPlan = MemorySnapshot.capture();
        try (CudnnFlashAttentionKernel sdpa = new CudnnFlashAttentionKernel(
                batchSize, headNum, time, headDim)) {
            MemorySnapshot afterSdpaPlan = MemorySnapshot.capture();
            long referenceCoreBytes = referenceCoreTrainingBytes(
                    batchSize, headNum, time, headDim);
            System.out.printf("qkNorm=%s memory total=%s, process-used=%s, "
                            + "reference-forward-delta=%s, reference-core=%s, sdpa-plan=%s, "
                            + "sdpa-observed-delta=%s%n",
                    qkNorm, formatMiB(afterSdpaPlan.totalBytes),
                    formatMiB(afterSdpaPlan.usedBytes()),
                    formatMiB(afterReferenceForward.usedBytes()
                            - beforeReferenceForward.usedBytes()),
                    formatMiB(referenceCoreBytes),
                    formatMiB(sdpa.getAllocatedBytes()),
                    formatMiB(afterSdpaPlan.usedBytes() - beforeSdpaPlan.usedBytes()));
            sdpa.forward(q, k, v, attentionOutput);

            AttentionKernel layoutKernel = new AttentionKernel(network.cudaManager);
            layoutKernel.unpermute(attentionOutput, unpermuted,
                    batchSize, time, headNum, headDim);
            outputProjection.forward(unpermuted, actualOutput);

            float[] actual = actualOutput.syncHost();
            Metrics metrics = Metrics.of(expected, actual);
            System.out.println("qkNorm=" + qkNorm + " " + metrics);

            float maxAbsTolerance = Float.parseFloat(
                    System.getProperty("omega.sdpa.demo.maxAbs", "0.02"));
            float meanAbsTolerance = Float.parseFloat(
                    System.getProperty("omega.sdpa.demo.meanAbs", "0.002"));
            if (metrics.maxAbs > maxAbsTolerance || metrics.meanAbs > meanAbsTolerance) {
                throw new AssertionError("cuDNN SDPA output mismatch: " + metrics
                        + ", tolerances maxAbs=" + maxAbsTolerance
                        + ", meanAbs=" + meanAbsTolerance);
            }

            compareBackward(reference, sdpa, batchSize, time, headNum, headDim, embedDim,
                    qkNorm, q, k, v, attentionOutput);
        }
    }

    private static void compareBackward(DiTAttentionLayer2 reference,
            CudnnFlashAttentionKernel sdpa, int batchSize, int time, int headNum,
            int headDim, int embedDim, boolean qkNorm, Tensor q, Tensor k, Tensor v,
            Tensor attentionOutput) {
        int outputLength = batchSize * time * embedDim;
        Tensor delta = new Tensor(batchSize * time, 1, 1, embedDim,
                RandomUtils.gaussianRandom(outputLength, 0.02f), true);
        MemorySnapshot beforeReferenceBackward = MemorySnapshot.capture();
        reference.back(delta);
        JCuda.cudaDeviceSynchronize();
        MemorySnapshot afterReferenceBackward = MemorySnapshot.capture();
        System.out.printf("qkNorm=%s reference-backward-delta=%s, process-used=%s%n",
                qkNorm,
                formatMiB(afterReferenceBackward.usedBytes()
                        - beforeReferenceBackward.usedBytes()),
                formatMiB(afterReferenceBackward.usedBytes()));

        int coreLength = batchSize * headNum * time * headDim;
        Tensor referenceDOutput = privateTensor(reference, "temp");
        float[] dOutputData = Arrays.copyOf(referenceDOutput.syncHost(), coreLength);
        Tensor dOutput = new Tensor(batchSize, headNum, time, headDim, dOutputData, true);
        Tensor dQ = new Tensor(batchSize, headNum, time, headDim, true);
        Tensor dK = new Tensor(batchSize, headNum, time, headDim, true);
        Tensor dV = new Tensor(batchSize, headNum, time, headDim, true);

        sdpa.backward(dOutput, dQ, dK, dV);
        assertGradient("dQ", qkNorm, privateTensor(reference, "dqt"), dQ);
        assertGradient("dK", qkNorm, privateTensor(reference, "dkt"), dK);
        assertGradient("dV", qkNorm, privateTensor(reference, "dvt"), dV);
        benchmark(reference, sdpa, q, k, v, attentionOutput, dOutput, dQ, dK, dV,
                qkNorm);
    }

    private static void benchmark(DiTAttentionLayer2 reference,
            CudnnFlashAttentionKernel sdpa, Tensor q, Tensor k, Tensor v,
            Tensor attentionOutput, Tensor dOutput, Tensor dQ, Tensor dK, Tensor dV,
            boolean qkNorm) {
        int warmup = Integer.getInteger("omega.sdpa.demo.warmup", 10);
        int iterations = Integer.getInteger("omega.sdpa.demo.iterations", 50);

        for (int i = 0; i < warmup; i++) {
            reference.scaledDotProductAttention(q, k, v);
            sdpa.forward(q, k, v, attentionOutput);
            reference.scaledDotProductAttentionBackward(q, k);
            sdpa.backward(dOutput, dQ, dK, dV);
        }
        JCuda.cudaDeviceSynchronize();

        runBenchmarkRound(reference, sdpa, q, k, v, attentionOutput,
                dOutput, dQ, dK, dV, iterations);
        BenchmarkMetrics second = runBenchmarkRound(reference, sdpa, q, k, v,
                attentionOutput, dOutput, dQ, dK, dV, iterations);

        System.out.printf("qkNorm=%s benchmark(second loop) core forward "
                        + "%.4f -> %.4f ms (%.2fx), backward %.4f -> %.4f ms (%.2fx)%n",
                qkNorm, second.referenceForwardMs, second.cudnnForwardMs,
                second.referenceForwardMs / second.cudnnForwardMs,
                second.referenceBackwardMs, second.cudnnBackwardMs,
                second.referenceBackwardMs / second.cudnnBackwardMs);
    }

    private static BenchmarkMetrics runBenchmarkRound(DiTAttentionLayer2 reference,
            CudnnFlashAttentionKernel sdpa, Tensor q, Tensor k, Tensor v,
            Tensor attentionOutput, Tensor dOutput, Tensor dQ, Tensor dK, Tensor dV,
            int iterations) {
        long start = System.nanoTime();
        for (int i = 0; i < iterations; i++) {
            reference.scaledDotProductAttention(q, k, v);
        }
        JCuda.cudaDeviceSynchronize();
        double referenceForwardMs = elapsedMs(start, iterations);

        start = System.nanoTime();
        for (int i = 0; i < iterations; i++) {
            sdpa.forward(q, k, v, attentionOutput);
        }
        JCuda.cudaDeviceSynchronize();
        double cudnnForwardMs = elapsedMs(start, iterations);

        start = System.nanoTime();
        for (int i = 0; i < iterations; i++) {
            reference.scaledDotProductAttentionBackward(q, k);
        }
        JCuda.cudaDeviceSynchronize();
        double referenceBackwardMs = elapsedMs(start, iterations);

        start = System.nanoTime();
        for (int i = 0; i < iterations; i++) {
            sdpa.backward(dOutput, dQ, dK, dV);
        }
        JCuda.cudaDeviceSynchronize();
        double cudnnBackwardMs = elapsedMs(start, iterations);

        return new BenchmarkMetrics(referenceForwardMs, cudnnForwardMs,
                referenceBackwardMs, cudnnBackwardMs);
    }

    private static double elapsedMs(long startNanos, int iterations) {
        return (System.nanoTime() - startNanos) / 1_000_000.0 / iterations;
    }

    private static String formatMiB(long bytes) {
        return String.format("%.2f MiB", bytes / 1024.0 / 1024.0);
    }

    private static long referenceCoreTrainingBytes(int batchSize, int headNum,
            int time, int headDim) {
        long attentionMatrixElements = (long) batchSize * headNum * time * time;
        long projectionElements = (long) batchSize * headNum * time * headDim;
        return 4L * (3L * attentionMatrixElements + 3L * projectionElements);
    }

    private static void assertGradient(String name, boolean qkNorm, Tensor expected,
            Tensor actual) {
        Metrics metrics = Metrics.of(expected.syncHost(), actual.syncHost());
        System.out.println("qkNorm=" + qkNorm + " " + name + " " + metrics);
        float maxAbsTolerance = Float.parseFloat(
                System.getProperty("omega.sdpa.demo.gradMaxAbs", "0.02"));
        float meanAbsTolerance = Float.parseFloat(
                System.getProperty("omega.sdpa.demo.gradMeanAbs", "0.002"));
        if (metrics.maxAbs > maxAbsTolerance || metrics.meanAbs > meanAbsTolerance) {
            throw new AssertionError("cuDNN SDPA " + name + " mismatch: " + metrics
                    + ", tolerances maxAbs=" + maxAbsTolerance
                    + ", meanAbs=" + meanAbsTolerance);
        }
    }

    private static Tensor privateTensor(DiTAttentionLayer2 layer, String fieldName) {
        try {
            Field field = DiTAttentionLayer2.class.getDeclaredField(fieldName);
            field.setAccessible(true);
            return (Tensor) field.get(layer);
        } catch (ReflectiveOperationException e) {
            throw new IllegalStateException("Unable to inspect " + fieldName, e);
        }
    }

    private static Tensor getQ(DiTAttentionLayer2 layer, BPNetwork network,
            int batchSize, int time, int headNum, int headDim, boolean qkNorm) {
        if (qkNorm) {
            return layer.qNorm.getOutput();
        }
        return permuteProjection(layer.qLinerLayer.getOutput(), network,
                batchSize, time, headNum, headDim);
    }

    private static Tensor getK(DiTAttentionLayer2 layer, BPNetwork network,
            int batchSize, int time, int headNum, int headDim, boolean qkNorm) {
        if (qkNorm) {
            return layer.kNorm.getOutput();
        }
        return permuteProjection(layer.kLinerLayer.getOutput(), network,
                batchSize, time, headNum, headDim);
    }

    private static Tensor permuteProjection(Tensor projection, BPNetwork network,
            int batchSize, int time, int headNum, int headDim) {
        projection.view(batchSize, time, headNum, headDim);
        Tensor result = new Tensor(batchSize, headNum, time, headDim, true);
        network.tensorOP.permute(projection, result, PERMUTE_0213);
        return result;
    }

    private static final class BenchmarkMetrics {
        private final double referenceForwardMs;
        private final double cudnnForwardMs;
        private final double referenceBackwardMs;
        private final double cudnnBackwardMs;

        private BenchmarkMetrics(double referenceForwardMs, double cudnnForwardMs,
                double referenceBackwardMs, double cudnnBackwardMs) {
            this.referenceForwardMs = referenceForwardMs;
            this.cudnnForwardMs = cudnnForwardMs;
            this.referenceBackwardMs = referenceBackwardMs;
            this.cudnnBackwardMs = cudnnBackwardMs;
        }
    }

    private static final class MemorySnapshot {
        private final long freeBytes;
        private final long totalBytes;

        private MemorySnapshot(long freeBytes, long totalBytes) {
            this.freeBytes = freeBytes;
            this.totalBytes = totalBytes;
        }

        private static MemorySnapshot capture() {
            long[] free = new long[1];
            long[] total = new long[1];
            JCuda.cudaDeviceSynchronize();
            JCuda.cudaMemGetInfo(free, total);
            return new MemorySnapshot(free[0], total[0]);
        }

        private long usedBytes() {
            return totalBytes - freeBytes;
        }
    }

    private static final class Metrics {
        private final float maxAbs;
        private final float meanAbs;
        private final float rms;
        private final float cosine;

        private Metrics(float maxAbs, float meanAbs, float rms, float cosine) {
            this.maxAbs = maxAbs;
            this.meanAbs = meanAbs;
            this.rms = rms;
            this.cosine = cosine;
        }

        private static Metrics of(float[] expected, float[] actual) {
            if (expected.length != actual.length) {
                throw new IllegalArgumentException("Output lengths differ: "
                        + expected.length + " vs " + actual.length);
            }
            double absSum = 0.0;
            double squareSum = 0.0;
            double dot = 0.0;
            double expectedSquare = 0.0;
            double actualSquare = 0.0;
            float maxAbs = 0.0f;
            for (int i = 0; i < expected.length; i++) {
                float error = Math.abs(expected[i] - actual[i]);
                maxAbs = Math.max(maxAbs, error);
                absSum += error;
                squareSum += error * error;
                dot += expected[i] * actual[i];
                expectedSquare += expected[i] * expected[i];
                actualSquare += actual[i] * actual[i];
            }
            float meanAbs = (float) (absSum / expected.length);
            float rms = (float) Math.sqrt(squareSum / expected.length);
            float cosine = (float) (dot / Math.sqrt(expectedSquare * actualSquare));
            return new Metrics(maxAbs, meanAbs, rms, cosine);
        }

        @Override
        public String toString() {
            return "maxAbs=" + maxAbs + ", meanAbs=" + meanAbs
                    + ", rms=" + rms + ", cosine=" + cosine;
        }
    }
}
