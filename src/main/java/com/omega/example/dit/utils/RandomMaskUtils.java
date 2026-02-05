package com.omega.example.dit.utils;

import java.util.Arrays;
import java.util.Random;
import java.util.concurrent.*;

import com.omega.engine.tensor.Tensor;

/**
 * RandomMaskUtils
 *
 * Generate random masks for masked modeling (e.g., MAE).
 * Multi-threaded CPU implementation.
 *
 * @author omega-ai
 */
public class RandomMaskUtils {

    private static final int DEFAULT_THREAD_NUM = Runtime.getRuntime().availableProcessors();
    private static ThreadPoolExecutor threadPool;

    /**
     * Result container for random mask generation
     */
    public static class MaskResult {
        public float[] mask;        // [B, N] - 1 is masked, 0 is keep
        public int[][] idsKeep;     // [B, M] - indices of kept tokens
        public int[][] idsRemove;   // [B, N-M] - indices of removed tokens
        public int[][] idsRestore;  // [B, N] - indices to restore original order
        public int[][] idsShuffle;  // [B, N] - shuffled indices (optional, null by default)

        // For structured mask: per-group selected positions
        public int[][][] offsetsRelH;  // [B, Hs, Ws, selection_per_group] - relative H positions
        public int[][][] offsetsRelW;  // [B, Hs, Ws, selection_per_group] - relative W positions

        public int batchSize;
        public int seqLength;
        public int lenKeep;
        public int tokensH;
        public int tokensW;
        public int groupSize;
        public int selectionPerGroup;

        public MaskResult(int B, int N, int lenKeep) {
            this.batchSize = B;
            this.seqLength = N;
            this.lenKeep = lenKeep;
            this.mask = new float[B * N];
            this.idsKeep = new int[B][lenKeep];
            this.idsRemove = new int[B][N - lenKeep];
            this.idsRestore = new int[B][N];
            this.idsShuffle = null;
        }

        public MaskResult(int B, int N, int lenKeep, int tokensH, int tokensW, int groupSize, int selectionPerGroup) {
            this(B, N, lenKeep);
            this.tokensH = tokensH;
            this.tokensW = tokensW;
            this.groupSize = groupSize;
            this.selectionPerGroup = selectionPerGroup;
            int Hs = tokensH / groupSize;
            int Ws = tokensW / groupSize;
            this.offsetsRelH = new int[B][Hs * Ws][selectionPerGroup];
            this.offsetsRelW = new int[B][Hs * Ws][selectionPerGroup];
        }
    }

    /**
     * Initialize thread pool
     */
    private static synchronized ThreadPoolExecutor getThreadPool() {
        if (threadPool == null || threadPool.isShutdown()) {
            threadPool = new ThreadPoolExecutor(
                DEFAULT_THREAD_NUM,
                DEFAULT_THREAD_NUM * 2,
                60,
                TimeUnit.SECONDS,
                new LinkedBlockingDeque<>()
            );
        }
        return threadPool;
    }

    /**
     * Calculate sequence length based on input dimensions
     *
     * @param ndim number of dimensions (3, 4, or 5)
     * @param dims dimension sizes [B, ...]
     *             3D: [B, seq_length, feature]
     *             4D: [B, C, H, W]
     *             5D: [B, C, T, H, W]
     * @return sequence length
     */
    public static int calculateSeqLength(int ndim, int[] dims) {
        if (ndim == 3) {
            // [B, seq_length, feature]
            return dims[1];
        } else if (ndim == 4) {
            // [B, C, H, W] -> seq_length = (H // 2) * (W // 2)
            int H = dims[2];
            int W = dims[3];
            return (H / 2) * (W / 2);
        } else if (ndim == 5) {
            // [B, C, T, H, W] -> seq_length = T * (H // 2) * (W // 2)
            int T = dims[2];
            int H = dims[3];
            int W = dims[4];
            return T * (H / 2) * (W / 2);
        } else {
            throw new IllegalArgumentException("Input must be 3D, 4D or 5D tensor. Got ndim=" + ndim);
        }
    }

    /**
     * Generate random mask for a batch (multi-threaded)
     *
     * @param B batch size
     * @param seqLength sequence length
     * @param maskRatio ratio of tokens to mask (0.0 to 1.0)
     * @return MaskResult containing mask and indices
     */
    public static MaskResult getRandomMask(int B, int seqLength, float maskRatio) {
        return getRandomMask(B, seqLength, maskRatio, null);
    }

    /**
     * Generate random mask for a batch (multi-threaded)
     *
     * @param B batch size
     * @param seqLength sequence length
     * @param maskRatio ratio of tokens to mask (0.0 to 1.0)
     * @param seed random seed (null for random)
     * @return MaskResult containing mask and indices
     */
    public static MaskResult getRandomMask(int B, int seqLength, float maskRatio, Long seed) {
        int lenKeep = (int) (seqLength * (1 - maskRatio));
        MaskResult result = new MaskResult(B, seqLength, lenKeep);

        // Use multi-threading for large batches
        if (B >= 4) {
            generateMaskParallel(result, B, seqLength, lenKeep, seed);
        } else {
            generateMaskSequential(result, B, seqLength, lenKeep, seed);
        }

        return result;
    }

    /**
     * Generate random mask from tensor dimensions
     *
     * @param ndim number of dimensions
     * @param dims dimension array
     * @param maskRatio mask ratio
     * @return MaskResult
     */
    public static MaskResult getRandomMask(int ndim, int[] dims, float maskRatio) {
        int B = dims[0];
        int seqLength = calculateSeqLength(ndim, dims);
        return getRandomMask(B, seqLength, maskRatio);
    }

    /**
     * Sequential mask generation (for small batches)
     */
    private static void generateMaskSequential(MaskResult result, int B, int seqLength, int lenKeep, Long seed) {
        Random random = seed != null ? new Random(seed) : new Random();

        for (int b = 0; b < B; b++) {
            generateMaskForBatch(result, b, seqLength, lenKeep, random.nextLong());
        }
    }

    /**
     * Parallel mask generation (for large batches)
     */
    private static void generateMaskParallel(MaskResult result, int B, int seqLength, int lenKeep, Long seed) {
        ThreadPoolExecutor pool = getThreadPool();
        int threadNum = Math.min(DEFAULT_THREAD_NUM, B);
        int batchPerThread = (B + threadNum - 1) / threadNum;

        Random seedRandom = seed != null ? new Random(seed) : new Random();
        long[] seeds = new long[B];
        for (int i = 0; i < B; i++) {
            seeds[i] = seedRandom.nextLong();
        }

        CountDownLatch latch = new CountDownLatch(threadNum);

        for (int t = 0; t < threadNum; t++) {
            final int threadIdx = t;
            final int startBatch = threadIdx * batchPerThread;
            final int endBatch = Math.min(startBatch + batchPerThread, B);

            pool.execute(() -> {
                try {
                    for (int b = startBatch; b < endBatch; b++) {
                        generateMaskForBatch(result, b, seqLength, lenKeep, seeds[b]);
                    }
                } finally {
                    latch.countDown();
                }
            });
        }

        try {
            latch.await();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new RuntimeException("Mask generation interrupted", e);
        }
    }

    /**
     * Generate mask for a single batch element
     */
    private static void generateMaskForBatch(MaskResult result, int batchIdx, int seqLength, int lenKeep, long seed) {
        Random random = new Random(seed);

        // Generate random noise and create index pairs for sorting
        IndexValue[] indexValues = new IndexValue[seqLength];
        for (int i = 0; i < seqLength; i++) {
            indexValues[i] = new IndexValue(i, random.nextFloat());
        }

        // Sort by noise value (argsort)
        Arrays.sort(indexValues, (a, b) -> Float.compare(a.value, b.value));

        // ids_shuffle: sorted indices
        int[] idsShuffle = new int[seqLength];
        for (int i = 0; i < seqLength; i++) {
            idsShuffle[i] = indexValues[i].index;
        }

        // ids_restore: argsort of ids_shuffle (inverse permutation)
        int[] idsRestore = new int[seqLength];
        for (int i = 0; i < seqLength; i++) {
            idsRestore[idsShuffle[i]] = i;
        }

        // ids_keep: first lenKeep indices from ids_shuffle
        for (int i = 0; i < lenKeep; i++) {
            result.idsKeep[batchIdx][i] = idsShuffle[i];
        }

        // ids_remove: remaining indices from ids_shuffle
        for (int i = lenKeep; i < seqLength; i++) {
            result.idsRemove[batchIdx][i - lenKeep] = idsShuffle[i];
        }

        // Store ids_restore
        result.idsRestore[batchIdx] = idsRestore;

        // Create mask: initially all 1s (masked), then set first lenKeep to 0 (keep)
        // After gather with ids_restore, positions corresponding to kept tokens will be 0
        float[] tempMask = new float[seqLength];
        Arrays.fill(tempMask, 1.0f);
        for (int i = 0; i < lenKeep; i++) {
            tempMask[i] = 0.0f;
        }

        // Gather: mask = torch.gather(mask, dim=1, index=ids_restore)
        // This reorders tempMask according to idsRestore
        int offset = batchIdx * seqLength;
        for (int i = 0; i < seqLength; i++) {
            result.mask[offset + i] = tempMask[idsRestore[i]];
        }
    }

    /**
     * Helper class for sorting indices by value
     */
    private static class IndexValue {
        int index;
        float value;

        IndexValue(int index, float value) {
            this.index = index;
            this.value = value;
        }
    }

    /**
     * Gather operation: output[i] = input[indices[i]]
     *
     * @param input input array
     * @param indices index array
     * @return gathered array
     */
    public static float[] gather(float[] input, int[] indices) {
        float[] output = new float[indices.length];
        for (int i = 0; i < indices.length; i++) {
            output[i] = input[indices[i]];
        }
        return output;
    }

    /**
     * Gather operation for 2D arrays along dim=1
     *
     * @param input [B, N] input array
     * @param indices [B, M] index array
     * @param B batch size
     * @param N input sequence length
     * @param M output sequence length
     * @return [B, M] gathered array
     */
    public static float[] gather2D(float[] input, int[][] indices, int B, int N, int M) {
        float[] output = new float[B * M];

        for (int b = 0; b < B; b++) {
            int inputOffset = b * N;
            int outputOffset = b * M;
            for (int i = 0; i < M; i++) {
                output[outputOffset + i] = input[inputOffset + indices[b][i]];
            }
        }

        return output;
    }

    /**
     * Scatter operation: reverse of gather
     * Places values at specified indices
     *
     * @param values values to scatter
     * @param indices where to place values
     * @param outputLength length of output array
     * @param fillValue value for unfilled positions
     * @return scattered array
     */
    public static float[] scatter(float[] values, int[] indices, int outputLength, float fillValue) {
        float[] output = new float[outputLength];
        Arrays.fill(output, fillValue);
        for (int i = 0; i < indices.length; i++) {
            output[indices[i]] = values[i];
        }
        return output;
    }

    /**
     * Apply mask to select kept tokens from input
     *
     * @param input [B, N, D] input tensor (flattened)
     * @param maskResult mask result from getRandomMask
     * @param D feature dimension
     * @return [B, M, D] kept tokens (flattened)
     */
    public static float[] applyMaskKeep(float[] input, MaskResult maskResult, int D) {
        int B = maskResult.batchSize;
        int N = maskResult.seqLength;
        int M = maskResult.lenKeep;

        float[] output = new float[B * M * D];

        for (int b = 0; b < B; b++) {
            int[] keepIndices = maskResult.idsKeep[b];
            for (int m = 0; m < M; m++) {
                int srcIdx = keepIndices[m];
                int srcOffset = (b * N + srcIdx) * D;
                int dstOffset = (b * M + m) * D;
                System.arraycopy(input, srcOffset, output, dstOffset, D);
            }
        }

        return output;
    }

    /**
     * Restore tokens to original order using ids_restore
     *
     * @param keptTokens [B, M, D] kept tokens
     * @param maskTokens [B, N-M, D] mask tokens (or null to fill with zeros)
     * @param maskResult mask result
     * @param D feature dimension
     * @return [B, N, D] restored tokens
     */
    public static float[] restoreTokens(float[] keptTokens, float[] maskTokens, MaskResult maskResult, int D) {
        int B = maskResult.batchSize;
        int N = maskResult.seqLength;
        int M = maskResult.lenKeep;

        float[] output = new float[B * N * D];

        for (int b = 0; b < B; b++) {
            // Place kept tokens
            for (int m = 0; m < M; m++) {
                int srcOffset = (b * M + m) * D;
                int dstIdx = maskResult.idsRestore[b][maskResult.idsKeep[b][m]];
                // Actually we need to unshuffle:
                // The idsRestore tells us where each position in the shuffled sequence
                // should go in the original sequence
            }

            // Concatenate kept and mask tokens, then unshuffle
            float[] combined = new float[N * D];

            // First M positions: kept tokens
            System.arraycopy(keptTokens, b * M * D, combined, 0, M * D);

            // Remaining positions: mask tokens or zeros
            if (maskTokens != null) {
                System.arraycopy(maskTokens, b * (N - M) * D, combined, M * D, (N - M) * D);
            }
            // else: already zeros

            // Unshuffle using ids_restore
            for (int i = 0; i < N; i++) {
                int srcIdx = maskResult.idsRestore[b][i];
                int srcOffset = srcIdx * D;
                int dstOffset = (b * N + i) * D;
                System.arraycopy(combined, srcOffset, output, dstOffset, D);
            }
        }

        return output;
    }

    // ==================== Structured Mask with Random Offset ====================

    /**
     * Generate structured mask with random offset for block-wise masking.
     * Divides the token grid into groups and randomly selects tokens from each group.
     *
     * @param B batch size
     * @param tokensH height in tokens (H)
     * @param tokensW width in tokens (W)
     * @param groupSize size of each square group (e.g., 2 for 2x2 groups)
     * @param selectionPerGroup number of tokens to select from each group
     * @param seed random seed (null for random)
     * @return MaskResult with structured mask
     */
    public static MaskResult getStructuredMaskWithRandomOffset(
            int B, int tokensH, int tokensW,
            int groupSize, int selectionPerGroup,
            Long seed) {

        if (tokensH % groupSize != 0 || tokensW % groupSize != 0) {
            throw new IllegalArgumentException("tokensH and tokensW must be divisible by groupSize");
        }

        if (selectionPerGroup > groupSize * groupSize) {
            throw new IllegalArgumentException("selectionPerGroup cannot exceed groupSize^2");
        }

        int H = tokensH;
        int W = tokensW;
        int N = H * W;
        int Hs = H / groupSize;  // number of groups along H
        int Ws = W / groupSize;  // number of groups along W
        int numGroups = Hs * Ws;
        int M = numGroups * selectionPerGroup;  // total kept tokens

        MaskResult result = new MaskResult(B, N, M, H, W, groupSize, selectionPerGroup);

        // Use multi-threading for large batches
        if (B >= 4) {
            generateStructuredMaskParallel(result, B, H, W, Hs, Ws, groupSize, selectionPerGroup, seed);
        } else {
            generateStructuredMaskSequential(result, B, H, W, Hs, Ws, groupSize, selectionPerGroup, seed);
        }

        return result;
    }
    
    public static float[] getStructuredMaskWithRandomOffset_idskeep(
            int B, int tokensH, int tokensW,
            int groupSize, int selectionPerGroup,
            Long seed) {

        if (tokensH % groupSize != 0 || tokensW % groupSize != 0) {
            throw new IllegalArgumentException("tokensH and tokensW must be divisible by groupSize");
        }

        if (selectionPerGroup > groupSize * groupSize) {
            throw new IllegalArgumentException("selectionPerGroup cannot exceed groupSize^2");
        }

        int H = tokensH;
        int W = tokensW;
        int N = H * W;
        int Hs = H / groupSize;  // number of groups along H
        int Ws = W / groupSize;  // number of groups along W
        int numGroups = Hs * Ws;
        int M = numGroups * selectionPerGroup;  // total kept tokens

        float[] idskeep = new float[B * M];

        // Use multi-threading for large batches
        generateStructuredMaskSequential(idskeep, B, H, W, Hs, Ws, groupSize, selectionPerGroup, seed);

        return idskeep;
    }
    
    public static Tensor getStructuredMaskWithRandomOffset_idskeep(
            int B, int tokensH, int tokensW,
            int groupSize, int selectionPerGroup,
            Long seed, Tensor idskeep) {

        if (tokensH % groupSize != 0 || tokensW % groupSize != 0) {
            throw new IllegalArgumentException("tokensH and tokensW must be divisible by groupSize");
        }

        if (selectionPerGroup > groupSize * groupSize) {
            throw new IllegalArgumentException("selectionPerGroup cannot exceed groupSize^2");
        }

        int H = tokensH;
        int W = tokensW;
        int Hs = H / groupSize;  // number of groups along H
        int Ws = W / groupSize;  // number of groups along W

        // Use multi-threading for large batches
        generateStructuredMaskSequential(idskeep.data, B, H, W, Hs, Ws, groupSize, selectionPerGroup, seed);
        idskeep.hostToDevice();
        return idskeep;
    }
    
    /**
     * Generate structured mask from tensor dimensions
     *
     * @param ndim number of dimensions (3 or 4)
     * @param dims dimension array
     * @param groupSize group size
     * @param selectionPerGroup selections per group
     * @param seed random seed
     * @return MaskResult
     */
    public static MaskResult getStructuredMaskWithRandomOffset(
            int ndim, int[] dims,
            int groupSize, int selectionPerGroup,
            Long seed) {

        int B = dims[0];
        int H, W;

        if (ndim == 3) {
            // [B, N, D] - assume square grid
            int N = dims[1];
            H = W = (int) Math.sqrt(N);
            if (H * W != N) {
                throw new IllegalArgumentException("Sequence length must be a perfect square for block masking. Got N=" + N);
            }
        } else if (ndim == 4) {
            // [B, C, H, W]
            H = dims[2] / 2;  // TODO: replace hard code for patch size 2x2
            W = dims[3] / 2;
        } else {
            throw new IllegalArgumentException("Input must be 3D or 4D tensor for structured masking");
        }

        return getStructuredMaskWithRandomOffset(B, H, W, groupSize, selectionPerGroup, seed);
    }

    /**
     * Overload without seed
     */
    public static MaskResult getStructuredMaskWithRandomOffset(
            int B, int tokensH, int tokensW,
            int groupSize, int selectionPerGroup) {
        return getStructuredMaskWithRandomOffset(B, tokensH, tokensW, groupSize, selectionPerGroup, null);
    }
    
    public static float[] getStructuredMaskWithRandomOffset_idskeep(
            int B, int tokensH, int tokensW,
            int groupSize, int selectionPerGroup) {
        return getStructuredMaskWithRandomOffset_idskeep(B, tokensH, tokensW, groupSize, selectionPerGroup, null);
    }
    
    public static Tensor getStructuredMaskWithRandomOffset_idskeep_tensor(
            int B, int tokensH, int tokensW,
            int groupSize, int selectionPerGroup, Tensor idskeep) {
        return getStructuredMaskWithRandomOffset_idskeep(B, tokensH, tokensW, groupSize, selectionPerGroup, null, idskeep);
    }
    
    /**
     * Sequential structured mask generation
     */
    private static void generateStructuredMaskSequential(
            MaskResult result, int B, int H, int W, int Hs, int Ws,
            int groupSize, int selectionPerGroup, Long seed) {

        Random random = seed != null ? new Random(seed) : new Random();

        for (int b = 0; b < B; b++) {
            generateStructuredMaskForBatch(result, b, H, W, Hs, Ws, groupSize, selectionPerGroup, random.nextLong());
        }
    }

    private static void generateStructuredMaskSequential(
            float[] idskeep, int B, int H, int W, int Hs, int Ws,
            int groupSize, int selectionPerGroup, Long seed) {

        Random random = seed != null ? new Random(seed) : new Random();

        for (int b = 0; b < B; b++) {
            generateStructuredMaskForBatch(idskeep, b, H, W, Hs, Ws, groupSize, selectionPerGroup, random.nextLong());
        }
    }
    
    /**
     * Parallel structured mask generation
     */
    private static void generateStructuredMaskParallel(
            MaskResult result, int B, int H, int W, int Hs, int Ws,
            int groupSize, int selectionPerGroup, Long seed) {

        ThreadPoolExecutor pool = getThreadPool();
        int threadNum = Math.min(DEFAULT_THREAD_NUM, B);
        int batchPerThread = (B + threadNum - 1) / threadNum;

        Random seedRandom = seed != null ? new Random(seed) : new Random();
        long[] seeds = new long[B];
        for (int i = 0; i < B; i++) {
            seeds[i] = seedRandom.nextLong();
        }

        CountDownLatch latch = new CountDownLatch(threadNum);

        for (int t = 0; t < threadNum; t++) {
            final int threadIdx = t;
            final int startBatch = threadIdx * batchPerThread;
            final int endBatch = Math.min(startBatch + batchPerThread, B);

            pool.execute(() -> {
                try {
                    for (int b = startBatch; b < endBatch; b++) {
                        generateStructuredMaskForBatch(result, b, H, W, Hs, Ws, groupSize, selectionPerGroup, seeds[b]);
                    }
                } finally {
                    latch.countDown();
                }
            });
        }

        try {
            latch.await();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new RuntimeException("Structured mask generation interrupted", e);
        }
    }
    
    private static void generateStructuredMaskForBatch(
            float[] idskeep, int batchIdx, int H, int W, int Hs, int Ws,
            int groupSize, int selectionPerGroup, long seed) {

        Random random = new Random(seed);
//        int N = H * W;
        int numGroups = Hs * Ws;
        int M = numGroups * selectionPerGroup;

        // For each group, generate random values and select top-k
//        int[] idsKeep = new int[M];
        int keepIdx = 0;
        
        for (int gh = 0; gh < Hs; gh++) {
            for (int gw = 0; gw < Ws; gw++) {

                // Generate random values for all positions in this group
                IndexValue[] groupValues = new IndexValue[groupSize * groupSize];
                for (int i = 0; i < groupSize * groupSize; i++) {
                    groupValues[i] = new IndexValue(i, random.nextFloat());
                }

                // Select top selectionPerGroup by sorting
                Arrays.sort(groupValues, (a, b) -> Float.compare(b.value, a.value)); // descending

                // Convert to global indices and store
                for (int s = 0; s < selectionPerGroup; s++) {
                    int relIdx = groupValues[s].index;
                    int relH = relIdx / groupSize;
                    int relW = relIdx % groupSize;

                    // Global position
                    int globalH = gh * groupSize + relH;
                    int globalW = gw * groupSize + relW;
                    int globalIdx = globalH * W + globalW;
                    idskeep[batchIdx * M + keepIdx] = globalIdx;
                    keepIdx++;
                }
            }
        }

    }
    
    /**
     * Generate structured mask for a single batch element
     */
    private static void generateStructuredMaskForBatch(
            MaskResult result, int batchIdx, int H, int W, int Hs, int Ws,
            int groupSize, int selectionPerGroup, long seed) {

        Random random = new Random(seed);
        int N = H * W;
        int numGroups = Hs * Ws;
        int M = numGroups * selectionPerGroup;

        // For each group, generate random values and select top-k
        int[] idsKeep = new int[M];
        int keepIdx = 0;

        for (int gh = 0; gh < Hs; gh++) {
            for (int gw = 0; gw < Ws; gw++) {
                int groupIdx = gh * Ws + gw;

                // Generate random values for all positions in this group
                IndexValue[] groupValues = new IndexValue[groupSize * groupSize];
                for (int i = 0; i < groupSize * groupSize; i++) {
                    groupValues[i] = new IndexValue(i, random.nextFloat());
                }

                // Select top selectionPerGroup by sorting
                Arrays.sort(groupValues, (a, b) -> Float.compare(b.value, a.value)); // descending

                // Convert to global indices and store
                for (int s = 0; s < selectionPerGroup; s++) {
                    int relIdx = groupValues[s].index;
                    int relH = relIdx / groupSize;
                    int relW = relIdx % groupSize;

                    // Global position
                    int globalH = gh * groupSize + relH;
                    int globalW = gw * groupSize + relW;
                    int globalIdx = globalH * W + globalW;

                    idsKeep[keepIdx++] = globalIdx;

                    // Store offset information
                    result.offsetsRelH[batchIdx][groupIdx][s] = relH;
                    result.offsetsRelW[batchIdx][groupIdx][s] = relW;
                }
            }
        }

        // Sort idsKeep for consistency (optional, but Python version doesn't require it)
        // Arrays.sort(idsKeep);  // Keep unsorted to maintain group structure

        // Store idsKeep
        result.idsKeep[batchIdx] = idsKeep;

        // Generate idsRemove: all indices not in idsKeep (ascending order)
        boolean[] isKept = new boolean[N];
        for (int idx : idsKeep) {
            isKept[idx] = true;
        }

        int[] idsRemove = new int[N - M];
        int removeIdx = 0;
        for (int i = 0; i < N; i++) {
            if (!isKept[i]) {
                idsRemove[removeIdx++] = i;
            }
        }
        result.idsRemove[batchIdx] = idsRemove;

        // Generate mask: True (1.0) for masked, False (0.0) for kept
        float[] mask = new float[N];
        Arrays.fill(mask, 1.0f);  // All masked by default
        for (int idx : idsKeep) {
            mask[idx] = 0.0f;  // Keep
        }
        int offset = batchIdx * N;
        System.arraycopy(mask, 0, result.mask, offset, N);

        // Generate idsRestore: inverse permutation of [idsKeep, idsRemove]
        int[] perm = new int[N];
        System.arraycopy(idsKeep, 0, perm, 0, M);
        System.arraycopy(idsRemove, 0, perm, M, N - M);

        int[] idsRestore = new int[N];
        for (int i = 0; i < N; i++) {
            idsRestore[perm[i]] = i;
        }
        result.idsRestore[batchIdx] = idsRestore;
    }

    /**
     * Print structured mask result for debugging
     */
    public static void printStructuredMaskResult(MaskResult result) {
        System.out.println("=== Structured MaskResult ===");
        System.out.println("Batch size: " + result.batchSize);
        System.out.println("Tokens H x W: " + result.tokensH + " x " + result.tokensW);
        System.out.println("Seq length: " + result.seqLength);
        System.out.println("Group size: " + result.groupSize);
        System.out.println("Selection per group: " + result.selectionPerGroup);
        System.out.println("Keep length: " + result.lenKeep);

        for (int b = 0; b < Math.min(result.batchSize, 2); b++) {
            System.out.println("\n--- Batch " + b + " ---");
            System.out.print("ids_keep (first 20): [");
            for (int i = 0; i < Math.min(20, result.idsKeep[b].length); i++) {
                System.out.print(result.idsKeep[b][i]);
                if (i < Math.min(20, result.idsKeep[b].length) - 1) System.out.print(", ");
            }
            System.out.println("]");

            System.out.print("ids_remove (first 20): [");
            for (int i = 0; i < Math.min(20, result.idsRemove[b].length); i++) {
                System.out.print(result.idsRemove[b][i]);
                if (i < Math.min(20, result.idsRemove[b].length) - 1) System.out.print(", ");
            }
            System.out.println("]");

            // Print first few groups' offsets
            int Hs = result.tokensH / result.groupSize;
            int Ws = result.tokensW / result.groupSize;
            System.out.println("\nFirst 4 groups' selected positions (relH, relW):");
            for (int g = 0; g < Math.min(4, Hs * Ws); g++) {
                System.out.print("  Group " + g + ": ");
                for (int s = 0; s < result.selectionPerGroup; s++) {
                    System.out.print("(" + result.offsetsRelH[b][g][s] + "," + result.offsetsRelW[b][g][s] + ")");
                    if (s < result.selectionPerGroup - 1) System.out.print(" ");
                }
                System.out.println();
            }
        }
    }

    /**
     * Shutdown the thread pool
     */
    public static void shutdown() {
        if (threadPool != null && !threadPool.isShutdown()) {
            threadPool.shutdown();
        }
    }

    /**
     * Print mask result for debugging
     */
    public static void printMaskResult(MaskResult result) {
        System.out.println("=== MaskResult ===");
        System.out.println("Batch size: " + result.batchSize);
        System.out.println("Seq length: " + result.seqLength);
        System.out.println("Keep length: " + result.lenKeep);

        for (int b = 0; b < Math.min(result.batchSize, 2); b++) {
            System.out.println("\n--- Batch " + b + " ---");
            System.out.print("ids_keep: [");
            for (int i = 0; i < result.idsKeep[b].length; i++) {
                System.out.print(result.idsKeep[b][i]);
                if (i < result.idsKeep[b].length - 1) System.out.print(", ");
            }
            System.out.println("]");

            System.out.print("ids_remove: [");
            for (int i = 0; i < result.idsRemove[b].length; i++) {
                System.out.print(result.idsRemove[b][i]);
                if (i < result.idsRemove[b].length - 1) System.out.print(", ");
            }
            System.out.println("]");

            System.out.print("ids_restore: [");
            for (int i = 0; i < result.idsRestore[b].length; i++) {
                System.out.print(result.idsRestore[b][i]);
                if (i < result.idsRestore[b].length - 1) System.out.print(", ");
            }
            System.out.println("]");

            System.out.print("mask: [");
            int offset = b * result.seqLength;
            for (int i = 0; i < result.seqLength; i++) {
                System.out.print((int) result.mask[offset + i]);
                if (i < result.seqLength - 1) System.out.print(", ");
            }
            System.out.println("]");
        }
    }

    /**
     * Test main method
     */
    public static void main(String[] args) {
        System.out.println("Testing RandomMaskUtils...\n");

        // ==================== Test Random Mask ====================
        System.out.println("========== Random Mask Tests ==========\n");

        // Test 1: Basic usage
        int B = 4;
        int seqLength = 16;
        float maskRatio = 0.75f;

        System.out.println("Test 1: Basic mask generation");
        System.out.println("B=" + B + ", seqLength=" + seqLength + ", maskRatio=" + maskRatio);

        long startTime = System.currentTimeMillis();
        MaskResult result = getRandomMask(B, seqLength, maskRatio);
        long endTime = System.currentTimeMillis();

        printMaskResult(result);
        System.out.println("\nTime: " + (endTime - startTime) + "ms");

        // Test 2: From 4D tensor dimensions
        System.out.println("\n\nTest 2: From 4D tensor [B, C, H, W]");
        int[] dims4D = {8, 3, 16, 16};  // B=8, C=3, H=16, W=16 -> seqLength = 8*8 = 64
        System.out.println("dims: [" + dims4D[0] + ", " + dims4D[1] + ", " + dims4D[2] + ", " + dims4D[3] + "]");

        startTime = System.currentTimeMillis();
        result = getRandomMask(4, dims4D, 0.75f);
        endTime = System.currentTimeMillis();

        System.out.println("Calculated seqLength: " + result.seqLength);
        System.out.println("Keep length: " + result.lenKeep);
        System.out.println("Time: " + (endTime - startTime) + "ms");

        // Test 3: Performance test with large batch
        System.out.println("\n\nTest 3: Performance test (Random Mask)");
        int[] testBatches = {16, 64, 256};
        int testSeqLen = 256;

        for (int testB : testBatches) {
            startTime = System.currentTimeMillis();
            for (int i = 0; i < 10; i++) {
                result = getRandomMask(testB, testSeqLen, 0.75f);
            }
            endTime = System.currentTimeMillis();
            System.out.println("B=" + testB + ", N=" + testSeqLen + " x10: " + (endTime - startTime) + "ms");
        }

        // ==================== Test Structured Mask ====================
        System.out.println("\n\n========== Structured Mask Tests ==========\n");

        // Test 4: Basic structured mask with 2x2 groups, select 1 per group
        System.out.println("Test 4: Structured mask (2x2 groups, select 1)");
        B = 2;
        int tokensH = 8;
        int tokensW = 8;
        int groupSize = 2;
        int selectionPerGroup = 1;

        System.out.println("B=" + B + ", H=" + tokensH + ", W=" + tokensW);
        System.out.println("groupSize=" + groupSize + ", selectionPerGroup=" + selectionPerGroup);

        startTime = System.currentTimeMillis();
        result = getStructuredMaskWithRandomOffset(B, tokensH, tokensW, groupSize, selectionPerGroup);
        endTime = System.currentTimeMillis();

        printStructuredMaskResult(result);
        System.out.println("\nTime: " + (endTime - startTime) + "ms");

        // Verify mask: print 2D view for first batch
        System.out.println("\nMask 2D view (batch 0):");
        for (int h = 0; h < tokensH; h++) {
            for (int w = 0; w < tokensW; w++) {
                int idx = h * tokensW + w;
                System.out.print((int) result.mask[idx] + " ");
            }
            System.out.println();
        }

        // Test 5: 4x4 groups, select 4 per group
        System.out.println("\n\nTest 5: Structured mask (4x4 groups, select 4)");
        B = 4;
        tokensH = 16;
        tokensW = 16;
        groupSize = 4;
        selectionPerGroup = 4;

        System.out.println("B=" + B + ", H=" + tokensH + ", W=" + tokensW);
        System.out.println("groupSize=" + groupSize + ", selectionPerGroup=" + selectionPerGroup);
        System.out.println("Expected: numGroups=" + (tokensH/groupSize * tokensW/groupSize) +
                ", M=" + (tokensH/groupSize * tokensW/groupSize * selectionPerGroup));

        startTime = System.currentTimeMillis();
        result = getStructuredMaskWithRandomOffset(B, tokensH, tokensW, groupSize, selectionPerGroup);
        endTime = System.currentTimeMillis();

        System.out.println("Actual M (lenKeep): " + result.lenKeep);
        System.out.println("Time: " + (endTime - startTime) + "ms");

        // Test 6: Performance test for structured mask
        System.out.println("\n\nTest 6: Performance test (Structured Mask)");
        tokensH = 32;
        tokensW = 32;
        groupSize = 2;
        selectionPerGroup = 1;

        for (int testB : testBatches) {
            startTime = System.currentTimeMillis();
            for (int i = 0; i < 10; i++) {
                result = getStructuredMaskWithRandomOffset(testB, tokensH, tokensW, groupSize, selectionPerGroup);
            }
            endTime = System.currentTimeMillis();
            System.out.println("B=" + testB + ", H=" + tokensH + ", W=" + tokensW + " x10: " + (endTime - startTime) + "ms");
        }

        // Test 7: Verify ids_restore correctness
        System.out.println("\n\nTest 7: Verify ids_restore correctness");
        B = 1;
        tokensH = 4;
        tokensW = 4;
        groupSize = 2;
        selectionPerGroup = 1;

        result = getStructuredMaskWithRandomOffset(B, tokensH, tokensW, groupSize, selectionPerGroup);

        System.out.println("ids_keep: " + arrayToString(result.idsKeep[0]));
        System.out.println("ids_remove: " + arrayToString(result.idsRemove[0]));
        System.out.println("ids_restore: " + arrayToString(result.idsRestore[0]));

        // Verify: perm[ids_restore[i]] should give original order 0,1,2,...,N-1
        int[] perm = new int[tokensH * tokensW];
        System.arraycopy(result.idsKeep[0], 0, perm, 0, result.lenKeep);
        System.arraycopy(result.idsRemove[0], 0, perm, result.lenKeep, tokensH * tokensW - result.lenKeep);

        System.out.print("Verification (should be 0,1,2,...): ");
        for (int i = 0; i < tokensH * tokensW; i++) {
            System.out.print(perm[result.idsRestore[0][i]] + " ");
        }
        System.out.println();

        // Cleanup
        shutdown();
        System.out.println("\nAll tests completed.");
    }

    private static String arrayToString(int[] arr) {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < arr.length; i++) {
            sb.append(arr[i]);
            if (i < arr.length - 1) sb.append(", ");
        }
        sb.append("]");
        return sb.toString();
    }
}
