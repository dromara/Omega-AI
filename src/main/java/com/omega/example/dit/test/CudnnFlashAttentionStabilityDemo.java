package com.omega.example.dit.test;

import java.io.*;
import java.nio.file.*;
import java.security.MessageDigest;
import java.util.*;

/** Standalone, opt-in SDPA diagnostics. No Omega runtime or CUDA is needed for --self-test. */
public final class CudnnFlashAttentionStabilityDemo {
    private static final int MAGIC = 0x4f464131; // OFA1; big-endian int32 header and float32 payload.
    private static final String[] NAMES = {"O", "dQ", "dK", "dV"};
    private final Config config;
    private final PrintWriter report;
    private int failures;

    private CudnnFlashAttentionStabilityDemo(Config config, PrintWriter report) {
        this.config = config;
        this.report = report;
        report.println("case,tensor,status,maxAbs,rmse,relativeL2,cosine,normRatio,nonFiniteExpected,nonFiniteActual,worstIndex,expected,actual,worstHeadSlot,worstHeadRelativeL2");
    }

    public static void main(String[] args) throws Exception {
        if (args.length == 1 && "--self-test".equals(args[0])) {
            selfTest();
            return;
        }
        if (args.length == 0 || Arrays.asList(args).contains("--help")) {
            System.out.println("Required: --library /absolute/path/libomega_cudnn_sdpa_diagnostic.so\n"
                    + "Options: --batch 12 --heads 12 --time 1101 --dim 64 --seed 1234\n"
                    + "  --scales 1 --do-scale 1 --reference-heads 3 --repeats 3 --layers 3\n"
                    + "  --ref-rel 0.05 --invariant-rel 0.001 --abs 0.00001\n"
                    + "  --qk-norm false --qk-norm-eps 0.000001\n"
                    + "  --reference-only false (skip repeat/scratch/padding stress tests)\n"
                    + "  --deterministic false --physical 0 --padding-step 64 --layer-step 64 --report fa-stability.csv\n"
                    + "  --snapshot captured.ofa (overrides shape; preserves input values)\n"
                    + "  --write-snapshot inputs.ofa (synthetic mode, one scale only)\n"
                    + "Exit: 0=within configured budgets, 2=numerical/nonfinite failure; exceptions=setup/runtime error.\n"
                    + "CPU reference samples complete heads, NOT shortened sequences. Invariants check all heads.");
            return;
        }
        Config c = new Config(args);
        Path library = Paths.get(c.library).toRealPath();
        if (!Paths.get(c.library).isAbsolute()) throw new IllegalArgumentException("Use an absolute library path.");
        System.out.println("Diagnostic library=" + library + " sha256=" + sha256(library));
        System.load(library.toString());
        System.out.println(nativeDescribe(0));
        System.out.println("Budgets: abs=" + c.abs + " referenceRel=" + c.refRel
                + " invariantRel=" + c.invariantRel + " deterministic=" + c.deterministic
                + " seed=" + c.seed + " repeats=" + c.repeats + " layers=" + c.layers);
        Path reportPath = Paths.get(c.report).toAbsolutePath();
        if (Files.exists(reportPath)) throw new IOException("Refusing to overwrite report: " + reportPath);
        try (PrintWriter report = new PrintWriter(Files.newBufferedWriter(reportPath,
                java.nio.charset.StandardCharsets.UTF_8, StandardOpenOption.CREATE_NEW))) {
            CudnnFlashAttentionStabilityDemo test = new CudnnFlashAttentionStabilityDemo(c, report);
            if (c.snapshot != null) {
                Inputs snapshot = readSnapshot(Paths.get(c.snapshot));
                test.run(c.qkNorm ? snapshot.rmsNormalizeQK(c.qkNormEps) : snapshot,
                        c.qkNorm ? "snapshot-qk-norm" : "snapshot", c.qkNorm ? snapshot : null);
            } else {
                for (double scale : c.scales) {
                    Inputs input = Inputs.random(c.shape, c.seed, scale, c.doScale);
                    if (c.writeSnapshot != null) writeSnapshot(Paths.get(c.writeSnapshot), input.shape, input.values);
                    test.run(c.qkNorm ? input.rmsNormalizeQK(c.qkNormEps) : input,
                            "qk-scale-" + scale + (c.qkNorm ? "-qk-norm" : ""), c.qkNorm ? input : null);
                }
            }
            System.out.println("SUMMARY failures=" + test.failures + " report=" + reportPath);
            System.out.println("PASS means these inputs met the configured budgets; it does not prove training stability.");
            if (test.failures != 0) {
                report.flush();
                System.exit(2);
            }
        }
    }

    private void run(Inputs in, String label, Inputs preNorm) {
        Shape s = in.shape;
        int[] selected = selectedHeads(s.b * s.h, config.referenceHeads);
        System.out.println("CASE " + label + " BHSD=" + s + " sampled flattened (batch,head)=" + Arrays.toString(selected));
        if (preNorm != null) {
            System.out.println("QK RMSNorm enabled: eps=" + config.qkNormEps + " gamma=1; "
                    + qkStats(preNorm, "before") + "; " + qkStats(in, "after"));
        }
        for (int i = 0; i < 4; ++i) requireFinite(in.values[i], "input " + i);
        try (Plan base = new Plan(s, config.physical, false, config.deterministic)) {
            base.upload(in);
            System.out.println(nativeDescribe(base.handle));
            Result baseline = base.run(0, 0);
            System.out.println("Computing full-sequence CPU double-accumulation reference on selected heads...");
            Result rawReference = reference(in, selected, false);
            Result roundedReference = reference(in, selected, true);
            Result roundedSavedOutputReference = reference(in, selected, true, true);
            Result sampled = baseline.select(s, selected);
            compare(label + "/quantization-only", rawReference, roundedReference, config.refRel, false);
            compare(label + "/FA-vs-original", rawReference, sampled, config.refRel, true);
            compare(label + "/FA-vs-BF16-input-reference", roundedReference, sampled, config.refRel, true);
            compare(label + "/FA-vs-BF16-input-and-saved-O-reference", roundedSavedOutputReference,
                    sampled, config.refRel, true);
            if (preNorm != null) {
                Result rawBack = rmsNormBackward(rawReference, preNorm, selected, config.qkNormEps);
                Result faBack = rmsNormBackward(sampled, preNorm, selected, config.qkNormEps);
                Result roundedBack = rmsNormBackward(roundedReference, preNorm, selected, config.qkNormEps);
                compareQk(label + "/FA-vs-original-after-QKNorm-backward",
                        rawBack, faBack, config.refRel, true);
                compareQk(label + "/FA-vs-BF16-input-after-QKNorm-backward",
                        roundedBack, faBack, config.refRel, true);
            }
            if (config.referenceOnly) return;
            for (int i = 0; i < config.repeats; ++i) {
                compare(label + "/repeat-" + i, baseline, base.run(0, 0), config.invariantRel, true);
            }
            try (Plan isolated = new Plan(s, config.physical, true, config.deterministic)) {
                isolated.upload(in);
                compare(label + "/private-scratch", baseline, isolated.run(0, 0), config.invariantRel, true);
            }
            // Force a masked graph even on versions that support the original unpadded shape.
            int padded = config.physical != 0 ? config.physical : Math.multiplyExact(Math.addExact(s.s, 63) / 64, 64);
            if (padded == s.s) padded = Math.addExact(padded, 64);
            padding(in, label, baseline, padded);
            padding(in, label, baseline, Math.addExact(padded, config.paddingStep));
            interleaved(in, label, base, baseline);
        }
    }

    private void padding(Inputs in, String label, Result baseline, int physical) {
        try (Plan p = new Plan(in.shape, physical, false, config.deterministic)) {
            p.upload(in);
            System.out.println(nativeDescribe(p.handle));
            Result zero = p.run(0, 0);
            compare(label + "/physical-" + physical, baseline, zero, config.invariantRel, true);
            compare(label + "/physical-" + physical + "-zero-repeat", zero, p.run(0, 0), config.invariantRel, true);
            compare(label + "/physical-" + physical + "-QKV-poison", zero, p.run(16, 0), config.invariantRel, true);
            compare(label + "/physical-" + physical + "-dO-poison", zero, p.run(0, 16), config.invariantRel, true);
        }
    }

    private void interleaved(Inputs original, String label, Plan base, Result baseline) {
        List<Plan> plans = new ArrayList<>();
        List<Result> expected = new ArrayList<>();
        plans.add(base);
        expected.add(baseline);
        try {
            nativeForward(base.handle, 0);
            for (int i = 1; i < config.layers; ++i) {
                // Different shapes also grow shared scratch while earlier plans retain saved state.
                Shape shape = new Shape(original.shape.b, original.shape.h,
                        Math.addExact(original.shape.s, Math.multiplyExact(i, config.layerStep)), original.shape.d);
                Inputs input = Inputs.random(shape, config.seed + i, 1.0 + i * 0.25, config.doScale);
                Plan p = new Plan(shape, 0, false, config.deterministic);
                plans.add(p);
                p.upload(input);
                expected.add(p.run(0, 0));
            }
            nativeBackward(base.handle, 0);
            compare(label + "/scratch-growth-with-pending-backward", baseline, base.read(), config.invariantRel, true);
            for (Plan p : plans) nativeForward(p.handle, 0);
            // Do not download/synchronize between backward launches. All share production scratch.
            for (int i = plans.size() - 1; i >= 0; --i) nativeBackward(plans.get(i).handle, 0);
            for (int i = 0; i < plans.size(); ++i) {
                compare(label + "/interleaved-layer-" + i, expected.get(i), plans.get(i).read(), config.invariantRel, true);
            }
        } finally {
            for (int i = plans.size() - 1; i >= 1; --i) plans.get(i).close();
        }
    }

    private void compare(String label, Result expected, Result actual, double rel, boolean enforce) {
        compareTensors(label, expected, actual, rel, enforce, new int[] {0, 1, 2, 3});
    }

    private void compareQk(String label, Result expected, Result actual, double rel, boolean enforce) {
        compareTensors(label, expected, actual, rel, enforce, new int[] {1, 2});
    }

    private void compareTensors(String label, Result expected, Result actual, double rel,
            boolean enforce, int[] tensors) {
        for (int t : tensors) {
            Metrics m = Metrics.of(expected.values[t], actual.values[t]);
            boolean valid = m.accepts(config.abs, rel);
            String status = enforce ? (valid ? "PASS" : "FAIL") : "INFO";
            if (enforce && !valid) ++failures;
            int index = m.worst;
            // Report the worst complete head so aggregate statistics cannot hide a bad head.
            int headLength = expected.headLength;
            double worstError = -1;
            int worstHead = 0;
            boolean headPass = true;
            for (int offset = 0; offset < expected.values[t].length; offset += headLength) {
                Metrics head = Metrics.of(expected.values[t], actual.values[t], offset, headLength);
                headPass &= head.accepts(config.abs, rel);
                if (!Double.isFinite(head.relative) || head.relative > worstError) {
                    worstError = head.relative;
                    worstHead = offset / headLength;
                }
            }
            if (enforce && valid && !headPass) { ++failures; status = "FAIL_HEAD"; }
            System.out.printf(Locale.ROOT, "%s %s %s maxAbs=%.7g rms=%.7g relL2=%.7g cos=%.9f normRatio=%.7g nonfinite=%d/%d%n",
                    status, label, NAMES[t], m.max, m.rmse, m.relative, m.cosine, m.ratio, m.badExpected, m.badActual);
            System.out.printf(Locale.ROOT, "  worst[%d] reference=%.9g FA/test=%.9g; first=%s / %s%n", index,
                    expected.values[t][index], actual.values[t][index], preview(expected.values[t]), preview(actual.values[t]));
            System.out.println("  worst head slot=" + worstHead + " relL2=" + worstError + " headBudget=" + (headPass ? "PASS" : "FAIL"));
            report.printf(Locale.ROOT, "%s,%s,%s,%.12g,%.12g,%.12g,%.12g,%.12g,%d,%d,%d,%.12g,%.12g,%d,%.12g%n",
                    label, NAMES[t], status, m.max, m.rmse, m.relative, m.cosine, m.ratio,
                    m.badExpected, m.badActual, index, expected.values[t][index], actual.values[t][index], worstHead, worstError);
        }
        report.flush();
    }

    private static String preview(float[] a) { return Arrays.toString(Arrays.copyOf(a, Math.min(4, a.length))); }

    /** Numerically stable mathematical reference, not an emulation of cuDNN's internal rounding. */
    static Result reference(Inputs input, int[] heads, boolean quantize) {
        return reference(input, heads, quantize, false);
    }

    /** Emulates FlashAttention's D_i = dot(dO_i, O_i) using a saved BF16 O. */
    static Result reference(Inputs input, int[] heads, boolean quantize, boolean quantizeSavedOutput) {
        Shape s = input.shape;
        int headLength = s.s * s.d;
        Result result = new Result(heads.length * headLength, headLength);
        double scale = 1.0 / Math.sqrt(s.d);
        for (int hi = 0; hi < heads.length; ++hi) {
            float[][] x = new float[4][headLength];
            for (int t = 0; t < 4; ++t) {
                System.arraycopy(input.values[t], heads[hi] * headLength, x[t], 0, headLength);
                if (quantize) for (int j = 0; j < headLength; ++j) x[t][j] = bf16(x[t][j]);
            }
            double[][] out = new double[4][headLength];
            double[] p = new double[s.s], dp = new double[s.s], outputRow = new double[s.d];
            for (int i = 0; i < s.s; ++i) {
                int row = i * s.d;
                double max = Double.NEGATIVE_INFINITY;
                for (int j = 0; j < s.s; ++j) {
                    double dot = 0, derivative = 0;
                    int col = j * s.d;
                    for (int d = 0; d < s.d; ++d) {
                        dot += (double) x[0][row + d] * x[1][col + d];
                        derivative += (double) x[3][row + d] * x[2][col + d];
                    }
                    p[j] = dot * scale;
                    dp[j] = derivative;
                    max = Math.max(max, p[j]);
                }
                double sum = 0;
                for (int j = 0; j < s.s; ++j) { p[j] = Math.exp(p[j] - max); sum += p[j]; }
                double delta = 0;
                Arrays.fill(outputRow, 0);
                for (int j = 0; j < s.s; ++j) {
                    p[j] /= sum;
                    int col = j * s.d;
                    for (int d = 0; d < s.d; ++d) outputRow[d] += p[j] * x[2][col + d];
                    if (!quantizeSavedOutput) delta += p[j] * dp[j];
                }
                if (quantizeSavedOutput) {
                    for (int d = 0; d < s.d; ++d) {
                        delta += (double) x[3][row + d] * bf16((float) outputRow[d]);
                    }
                }
                for (int j = 0; j < s.s; ++j) {
                    int col = j * s.d;
                    double ds = p[j] * (dp[j] - delta) * scale;
                    for (int d = 0; d < s.d; ++d) {
                        out[0][row + d] = outputRow[d];
                        out[1][row + d] += ds * x[1][col + d];
                        out[2][col + d] += ds * x[0][row + d];
                        out[3][col + d] += p[j] * x[3][row + d];
                    }
                }
            }
            for (int t = 0; t < 4; ++t) for (int j = 0; j < headLength; ++j) {
                result.values[t][hi * headLength + j] = (float) out[t][j];
            }
        }
        return result;
    }

    static float bf16(float value) {
        int bits = Float.floatToRawIntBits(value);
        if ((bits & 0x7fffffff) > 0x7f800000) return Float.NaN;
        return Float.intBitsToFloat((bits + 0x7fff + ((bits >>> 16) & 1)) & 0xffff0000);
    }

    static Result rmsNormBackward(Result gradient, Inputs preNorm, int[] heads, double eps) {
        int headLength = preNorm.shape.s * preNorm.shape.d;
        Result result = new Result(gradient.values[0].length, gradient.headLength);
        for (int t = 0; t < 4; ++t) {
            System.arraycopy(gradient.values[t], 0, result.values[t], 0, gradient.values[t].length);
        }
        for (int tensor = 0; tensor < 2; ++tensor) {
            int gradTensor = tensor + 1;
            for (int hi = 0; hi < heads.length; ++hi) {
                int rawHead = heads[hi] * headLength;
                int gradHead = hi * headLength;
                for (int token = 0; token < preNorm.shape.s; ++token) {
                    int rawOffset = rawHead + token * preNorm.shape.d;
                    int gradOffset = gradHead + token * preNorm.shape.d;
                    rmsNormBackwardRow(preNorm.values[tensor], rawOffset, gradient.values[gradTensor],
                            gradOffset, result.values[gradTensor], gradOffset, preNorm.shape.d, eps);
                }
            }
        }
        return result;
    }

    static void rmsNormBackwardRow(float[] x, int xOffset, float[] gradient, int gradientOffset,
            float[] output, int outputOffset, int d, double eps) {
        double squareSum = 0;
        for (int i = 0; i < d; ++i) squareSum += (double) x[xOffset + i] * x[xOffset + i];
        double rstd = 1.0 / Math.sqrt(squareSum / d + eps);
        double meanGradientTimesNormalized = 0;
        for (int i = 0; i < d; ++i) {
            meanGradientTimesNormalized += gradient[gradientOffset + i] * x[xOffset + i] * rstd;
        }
        meanGradientTimesNormalized /= d;
        for (int i = 0; i < d; ++i) {
            double normalized = x[xOffset + i] * rstd;
            output[outputOffset + i] = (float) (rstd
                    * (gradient[gradientOffset + i] - normalized * meanGradientTimesNormalized));
        }
    }

    private static String qkStats(Inputs input, String label) {
        StringBuilder out = new StringBuilder(label);
        for (int tensor = 0; tensor < 2; ++tensor) {
            double sum = 0;
            double max = 0;
            for (float value : input.values[tensor]) {
                sum += (double) value * value;
                max = Math.max(max, Math.abs(value));
            }
            out.append(tensor == 0 ? " Q" : " K")
                    .append(String.format(Locale.ROOT, "(rms=%.7g,max=%.7g)",
                            Math.sqrt(sum / input.values[tensor].length), max));
        }
        return out.toString();
    }

    static int[] selectedHeads(int total, int count) {
        count = Math.min(total, count);
        int[] indices = new int[count];
        for (int i = 0; i < count; ++i) indices[i] = count == 1 ? 0 : (int) ((long) i * (total - 1) / (count - 1));
        return indices;
    }

    /** Call only after GPU tensors are synchronized; Q/K must already include QK norm and RoPE. */
    public static void writeSnapshot(Path path, int b, int h, int s, int d,
            float[] q, float[] k, float[] v, float[] dO) throws IOException {
        writeSnapshot(path, new Shape(b, h, s, d), new float[][] {q, k, v, dO});
    }

    private static void writeSnapshot(Path path, Shape shape, float[][] values) throws IOException {
        for (float[] v : values) if (v.length != shape.length) throw new IllegalArgumentException("Snapshot array length mismatch.");
        try (DataOutputStream out = new DataOutputStream(new BufferedOutputStream(Files.newOutputStream(path, StandardOpenOption.CREATE_NEW)))) {
            out.writeInt(MAGIC);
            out.writeInt(shape.b); out.writeInt(shape.h); out.writeInt(shape.s); out.writeInt(shape.d);
            for (float[] tensor : values) for (float value : tensor) out.writeFloat(value);
        }
    }

    static Inputs readSnapshot(Path path) throws IOException {
        try (DataInputStream in = new DataInputStream(new BufferedInputStream(Files.newInputStream(path)))) {
            if (in.readInt() != MAGIC) throw new IOException("Not an OFA1 snapshot.");
            Shape s = new Shape(in.readInt(), in.readInt(), in.readInt(), in.readInt());
            if (Files.size(path) != 20L + 16L * s.length) throw new IOException("Truncated or oversized snapshot.");
            float[][] values = new float[4][s.length];
            for (float[] tensor : values) for (int i = 0; i < tensor.length; ++i) tensor[i] = in.readFloat();
            return new Inputs(s, values);
        }
    }

    private static String sha256(Path path) throws Exception {
        MessageDigest digest = MessageDigest.getInstance("SHA-256");
        try (InputStream in = Files.newInputStream(path)) {
            byte[] buffer = new byte[65536];
            for (int n; (n = in.read(buffer)) != -1;) digest.update(buffer, 0, n);
        }
        StringBuilder hex = new StringBuilder();
        for (byte b : digest.digest()) hex.append(String.format(Locale.ROOT, "%02x", b & 255));
        return hex.toString();
    }

    private static void requireFinite(float[] a, String label) {
        for (int i = 0; i < a.length; ++i) if (!Float.isFinite(a[i])) {
            throw new IllegalArgumentException(label + " contains nonfinite value at " + i);
        }
    }

    static final class Shape {
        final int b, h, s, d, length;
        Shape(int b, int h, int s, int d) {
            if (b <= 0 || h <= 0 || s <= 0 || d <= 0) throw new IllegalArgumentException("Shape must be positive.");
            this.b = b; this.h = h; this.s = s; this.d = d;
            length = Math.multiplyExact(Math.multiplyExact(Math.multiplyExact(b, h), s), d);
        }
        public String toString() { return "[" + b + "," + h + "," + s + "," + d + "]"; }
    }

    static final class Inputs {
        final Shape shape;
        final float[][] values;
        Inputs(Shape shape, float[][] values) { this.shape = shape; this.values = values; }
        static Inputs random(Shape s, long seed, double qkScale, double doScale) {
            Random random = new Random(seed);
            float[][] values = new float[4][s.length];
            for (int t = 0; t < 4; ++t) for (int i = 0; i < s.length; ++i) {
                values[t][i] = (float) (random.nextGaussian() * (t < 2 ? qkScale : t == 3 ? doScale : 1));
            }
            return new Inputs(s, values);
        }

        Inputs rmsNormalizeQK(double eps) {
            float[][] normalized = new float[4][];
            for (int i = 0; i < 4; ++i) normalized[i] = values[i].clone();
            for (int tensor = 0; tensor < 2; ++tensor) {
                for (int offset = 0; offset < shape.length; offset += shape.d) {
                    double squareSum = 0;
                    for (int i = 0; i < shape.d; ++i) {
                        squareSum += (double) values[tensor][offset + i] * values[tensor][offset + i];
                    }
                    double rstd = 1.0 / Math.sqrt(squareSum / shape.d + eps);
                    for (int i = 0; i < shape.d; ++i) {
                        normalized[tensor][offset + i] = (float) (values[tensor][offset + i] * rstd);
                    }
                }
            }
            return new Inputs(shape, normalized);
        }
    }

    static final class Result {
        final float[][] values;
        final int headLength;
        Result(int length, int headLength) { this.values = new float[4][length]; this.headLength = headLength; }
        Result select(Shape shape, int[] heads) {
            Result r = new Result(heads.length * headLength, headLength);
            for (int t = 0; t < 4; ++t) for (int i = 0; i < heads.length; ++i) {
                System.arraycopy(values[t], heads[i] * headLength, r.values[t], i * headLength, headLength);
            }
            return r;
        }
    }

    static final class Metrics {
        double max, rmse, relative, cosine, ratio, expectedRms, expectedMax;
        int badExpected, badActual, worst;
        static Metrics of(float[] a, float[] b) { return of(a, b, 0, a.length); }
        static Metrics of(float[] a, float[] b, int offset, int length) {
            if (a.length != b.length || length <= 0) throw new IllegalArgumentException("Metric shape mismatch.");
            Metrics m = new Metrics();
            m.worst = offset;
            double e2 = 0, a2 = 0, b2 = 0, dot = 0;
            for (int i = offset; i < offset + length; ++i) {
                if (!Float.isFinite(a[i])) ++m.badExpected;
                if (!Float.isFinite(b[i])) ++m.badActual;
                if (!Float.isFinite(a[i]) || !Float.isFinite(b[i])) { m.worst = i; continue; }
                double error = (double) a[i] - b[i];
                if (Math.abs(error) > m.max) {
                    m.max = Math.abs(error);
                    if (m.badExpected + m.badActual == 0) m.worst = i;
                }
                m.expectedMax = Math.max(m.expectedMax, Math.abs(a[i]));
                e2 += error * error; a2 += (double) a[i] * a[i]; b2 += (double) b[i] * b[i]; dot += (double) a[i] * b[i];
            }
            m.rmse = Math.sqrt(e2 / length);
            m.expectedRms = Math.sqrt(a2 / length);
            m.relative = a2 == 0 ? (e2 == 0 ? 0 : Double.POSITIVE_INFINITY) : Math.sqrt(e2 / a2);
            m.ratio = a2 == 0 ? (b2 == 0 ? 1 : Double.POSITIVE_INFINITY) : Math.sqrt(b2 / a2);
            m.cosine = a2 == 0 || b2 == 0 ? (a2 == b2 ? 1 : 0) : dot / Math.sqrt(a2 * b2);
            if (m.badExpected + m.badActual != 0) m.max = m.rmse = m.relative = Double.POSITIVE_INFINITY;
            return m;
        }
        boolean accepts(double abs, double rel) {
            return badExpected == 0 && badActual == 0 && rmse <= abs + rel * expectedRms
                    && max <= abs + rel * expectedMax;
        }
    }

    private static final class Plan implements AutoCloseable {
        long handle;
        final Shape shape;
        Plan(Shape s, int padded, boolean independent, boolean deterministic) {
            shape = s;
            handle = nativeCreate(s.b, s.h, s.s, s.d, padded, independent, deterministic);
            if (handle == 0) throw new IllegalStateException("Native plan was not created.");
        }
        void upload(Inputs in) { nativeUpload(handle, in.values[0], in.values[1], in.values[2], in.values[3]); }
        Result run(float qkvPadding, float doPadding) {
            nativeForward(handle, qkvPadding); nativeBackward(handle, doPadding); return read();
        }
        Result read() {
            Result r = new Result(shape.length, shape.s * shape.d);
            for (int i = 0; i < 4; ++i) nativeRead(handle, i, r.values[i]);
            return r;
        }
        public void close() { if (handle != 0) { nativeDestroy(handle); handle = 0; } }
    }

    private static final class Config {
        final String library, report, snapshot, writeSnapshot;
        final Shape shape;
        final long seed;
        final double[] scales;
        final double doScale, refRel, invariantRel, abs, qkNormEps;
        final int referenceHeads, repeats, layers, paddingStep, layerStep, physical;
        final boolean deterministic, qkNorm, referenceOnly;
        Config(String[] args) {
            Map<String, String> p = new HashMap<>();
            Set<String> allowed = new HashSet<>(Arrays.asList("library", "report", "snapshot", "write-snapshot", "batch", "heads",
                    "time", "dim", "seed", "scales", "do-scale", "ref-rel", "invariant-rel", "abs", "reference-heads", "repeats", "layers", "deterministic", "padding-step", "layer-step", "physical", "qk-norm", "qk-norm-eps", "reference-only"));
            for (int i = 0; i < args.length; i += 2) {
                if (!args[i].startsWith("--") || i + 1 == args.length || !allowed.contains(args[i].substring(2))) {
                    throw new IllegalArgumentException("Unknown/missing option: " + args[i]);
                }
                if (p.put(args[i].substring(2), args[i + 1]) != null) throw new IllegalArgumentException("Duplicate option " + args[i]);
            }
            library = p.get("library");
            if (library == null) throw new IllegalArgumentException("--library is required.");
            report = p.getOrDefault("report", "fa-stability.csv");
            snapshot = p.get("snapshot"); writeSnapshot = p.get("write-snapshot");
            shape = new Shape(integer(p, "batch", 12), integer(p, "heads", 12), integer(p, "time", 1101), integer(p, "dim", 64));
            seed = Long.parseLong(p.getOrDefault("seed", "1234"));
            String[] scaleText = p.getOrDefault("scales", "1").split(",");
            scales = new double[scaleText.length];
            for (int i = 0; i < scales.length; ++i) scales[i] = positive(scaleText[i]);
            doScale = positive(p.getOrDefault("do-scale", "1"));
            refRel = positive(p.getOrDefault("ref-rel", "0.05"));
            invariantRel = positive(p.getOrDefault("invariant-rel", "0.001"));
            abs = positive(p.getOrDefault("abs", "0.00001"));
            qkNormEps = positive(p.getOrDefault("qk-norm-eps", "0.000001"));
            referenceHeads = integer(p, "reference-heads", 3);
            repeats = integer(p, "repeats", 3); layers = integer(p, "layers", 3);
            paddingStep = integer(p, "padding-step", 64); layerStep = integer(p, "layer-step", 64);
            physical = integer(p, "physical", 0);
            if (physical < 0 || physical % 64 != 0) throw new IllegalArgumentException("physical must be zero or a positive multiple of 64.");
            if (paddingStep <= 0 || paddingStep % 64 != 0 || layerStep <= 0 || layerStep % 64 != 0) {
                throw new IllegalArgumentException("padding-step/layer-step must be positive multiples of 64.");
            }
            deterministic = bool(p, "deterministic", false);
            qkNorm = bool(p, "qk-norm", false);
            referenceOnly = bool(p, "reference-only", false);
            if (referenceHeads < 1 || repeats < 1 || layers < 2) throw new IllegalArgumentException("Need positive reference-heads/repeats and layers >= 2.");
            if (snapshot != null && (p.containsKey("scales") || writeSnapshot != null)) throw new IllegalArgumentException("Snapshot mode does not rescale or generate inputs.");
            if (writeSnapshot != null && scales.length != 1) throw new IllegalArgumentException("write-snapshot needs one scale.");
        }
        private static int integer(Map<String, String> p, String key, int value) { return Integer.parseInt(p.getOrDefault(key, Integer.toString(value))); }
        private static boolean bool(Map<String, String> p, String key, boolean value) {
            String text = p.getOrDefault(key, Boolean.toString(value));
            if (!"true".equals(text) && !"false".equals(text)) {
                throw new IllegalArgumentException("Invalid boolean option --" + key + ": " + text);
            }
            return Boolean.parseBoolean(text);
        }
        private static double positive(String text) {
            double v = Double.parseDouble(text);
            if (!Double.isFinite(v) || v <= 0) throw new IllegalArgumentException("Expected a positive finite value: " + text);
            return v;
        }
    }

    private static void selfTest() throws Exception {
        check(bf16(1.001f) == 1.0f, "BF16 rounding");
        check(bf16(1.00390625f) == 1.0f && bf16(1.01171875f) == 1.015625f, "BF16 ties-even");
        check(Float.isNaN(bf16(Float.NaN)) && bf16(Float.POSITIVE_INFINITY) == Float.POSITIVE_INFINITY, "BF16 special values");
        check(Float.floatToRawIntBits(bf16(-0.0f)) == 0x80000000, "negative zero");
        check(!Metrics.of(new float[] {1}, new float[] {Float.NaN}).accepts(1, 1), "NaN must fail");
        check(!Metrics.of(new float[] {Float.POSITIVE_INFINITY}, new float[] {1}).accepts(1, 1), "Inf must fail");
        check(Metrics.of(new float[] {0}, new float[] {0}).cosine == 1, "zero metrics");
        check(!Metrics.of(new float[] {1, 2}, new float[] {-1, -2}).accepts(0.00001, 0.05), "opposite gradients");
        float[] sparseExpected = new float[1000], sparseActual = new float[1000];
        Arrays.fill(sparseExpected, 1); Arrays.fill(sparseActual, 1); sparseActual[499] = 1.1f;
        check(!Metrics.of(sparseExpected, sparseActual).accepts(0.00001, 0.05), "max error catches sparse corruption");
        check(Arrays.equals(selectedHeads(144, 3), new int[] {0, 71, 143}), "head coverage");
        float[] normInput = {0.25f, -0.5f, 1.25f, -2.0f};
        float[] normGradient = {0.7f, -0.2f, 0.5f, 0.1f};
        float[] normBackward = new float[4];
        rmsNormBackwardRow(normInput, 0, normGradient, 0, normBackward, 0, 4, 1e-6);
        for (int i = 0; i < normInput.length; ++i) {
            float old = normInput[i], eps = 0.001f;
            normInput[i] = old + eps; double plus = rmsNormObjective(normInput, normGradient, 1e-6);
            normInput[i] = old - eps; double minus = rmsNormObjective(normInput, normGradient, 1e-6);
            normInput[i] = old;
            check(Math.abs((plus - minus) / (2 * eps) - normBackward[i]) < 0.0002,
                    "RMSNorm finite difference index=" + i);
        }
        Inputs input = Inputs.random(new Shape(1, 2, 3, 2), 42, 0.5, 1);
        Result r = reference(input, new int[] {1}, false);
        // Finite differences independently validate all Q/K/V gradient components of one complete head.
        for (int t = 0; t < 3; ++t) for (int i = 6; i < 12; ++i) {
            float old = input.values[t][i], eps = 0.002f;
            input.values[t][i] = old + eps; double plus = objective(input);
            input.values[t][i] = old - eps; double minus = objective(input);
            input.values[t][i] = old;
            double numerical = (plus - minus) / (2 * eps);
            check(Math.abs(numerical - r.values[t + 1][i - 6]) < 0.0002, "finite difference tensor=" + t + " index=" + i);
        }
        Path path = Files.createTempFile("omega-fa-snapshot-", ".ofa");
        Files.delete(path);
        try {
            writeSnapshot(path, input.shape, input.values);
            Inputs restored = readSnapshot(path);
            for (int i = 0; i < 4; ++i) check(Arrays.equals(input.values[i], restored.values[i]), "snapshot roundtrip");
            boolean refused = false;
            try { writeSnapshot(path, input.shape, input.values); } catch (FileAlreadyExistsException e) { refused = true; }
            check(refused, "snapshot overwrite guard");
            Files.write(path, new byte[] {0}, StandardOpenOption.APPEND);
            boolean rejected = false;
            try { readSnapshot(path); } catch (IOException e) { rejected = true; }
            check(rejected, "snapshot trailing bytes");
        } finally { Files.deleteIfExists(path); }
        Inputs one = Inputs.random(new Shape(1, 1, 1, 2), 3, 1, 1);
        Result oneResult = reference(one, new int[] {0}, false);
        check(Arrays.equals(oneResult.values[0], one.values[2]), "single token output");
        check(Arrays.equals(oneResult.values[3], one.values[3]), "single token dV");
        check(oneResult.values[1][0] == 0 && oneResult.values[2][1] == 0, "single token dQ/dK");
        System.out.println("PASS CPU self-tests: BF16, metrics, analytical gradients, snapshot format, one-token attention.");
    }

    private static double rmsNormObjective(float[] input, float[] gradient, double eps) {
        double squareSum = 0;
        for (float value : input) squareSum += (double) value * value;
        double rstd = 1.0 / Math.sqrt(squareSum / input.length + eps);
        double value = 0;
        for (int i = 0; i < input.length; ++i) value += input[i] * rstd * gradient[i];
        return value;
    }

    private static double objective(Inputs in) {
        float[] o = reference(in, new int[] {1}, false).values[0];
        double value = 0;
        for (int i = 0; i < o.length; ++i) value += (double) o[i] * in.values[3][6 + i];
        return value;
    }
    private static void check(boolean condition, String message) { if (!condition) throw new AssertionError(message); }

    private static native long nativeCreate(int b, int h, int s, int d, int physical, boolean isolated, boolean deterministic);
    private static native void nativeUpload(long plan, float[] q, float[] k, float[] v, float[] dO);
    private static native void nativeForward(long plan, float padding);
    private static native void nativeBackward(long plan, float padding);
    private static native void nativeRead(long plan, int index, float[] output);
    private static native String nativeDescribe(long plan);
    private static native void nativeDestroy(long plan);
}
