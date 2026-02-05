package com.omega.example.dit.utils;

import com.omega.engine.tensor.Tensor;

/**
 * Utility class for computing 2D Rotary Position Embeddings
 */
public class RoPE2DUtils {

    /**
     * Compute 2D rotary position embeddings
     *
     * @param hiddenSize Total hidden dimension (e.g., 768)
     * @param headNum Number of attention heads (e.g., 12)
     * @param ftSeqLen Fine-tuning sequence length
     * @param ptSeqLen Pre-training sequence length
     * @return Array containing [freqs_cos, freqs_sin] tensors
     */
    public static Tensor[] compute2DRoPE(int hiddenSize, int headNum, int ftSeqLen, int ptSeqLen) {
        // dim = 768 // 12 // 2
        int dim = hiddenSize / headNum / 2;

        // freqs = 1. / (10000 ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
        int freqsLen = dim / 2;
        float[] freqs = new float[freqsLen];
        for (int i = 0; i < freqsLen; i++) {
            int arangeVal = i * 2;
            float exponent = (float) arangeVal / dim;
            freqs[i] = (float) (1.0 / Math.pow(10000, exponent));
        }

        // t = torch.arange(ft_seq_len) / ft_seq_len * pt_seq_len
        float[] t = new float[ftSeqLen];
        for (int i = 0; i < ftSeqLen; i++) {
            t[i] = (float) i / ftSeqLen * ptSeqLen;
        }

        // base = torch.einsum('..., f -> ... f', t, freqs)  # (S, dim//2)
        // This is an outer product
        float[] base = outer(t, freqs);

        // base = repeat(base, '... n -> ... (n r)', r=2)  # (S, dim)
        // Repeat each element twice
        float[] baseRepeated = repeatInterleave(base, 2, ftSeqLen, freqsLen);

        // freqs_2d = broadcat((base[:, None, :], base[None, :, :]), dim=-1)  # (S,S,2*dim)
        // base[:, None, :] is (S, 1, dim) -> broadcast to (S, S, dim)
        // base[None, :, :] is (1, S, dim) -> broadcast to (S, S, dim)
        // Concatenate along last dimension
        float[] freqs2d = broadcat(baseRepeated, baseRepeated, ftSeqLen, dim);

        // freqs_cos = freqs_2d.cos().reshape(-1, freqs_2d.shape[-1])  # (HW, 2*dim)
        // freqs_sin = freqs_2d.sin().reshape(-1, freqs_2d.shape[-1])  # (HW, 2*dim)
        int hw = ftSeqLen * ftSeqLen;
        int twoDim = 2 * dim;
        float[] freqsCos = new float[hw * twoDim];
        float[] freqsSin = new float[hw * twoDim];

        for (int i = 0; i < freqs2d.length; i++) {
            freqsCos[i] = (float) Math.cos(freqs2d[i]);
            freqsSin[i] = (float) Math.sin(freqs2d[i]);
        }

        // Create tensors: shape (hw, twoDim)
        Tensor cosT = new Tensor(1, 1, hw, twoDim, freqsCos, true);
        Tensor sinT = new Tensor(1, 1, hw, twoDim, freqsSin, true);

        return new Tensor[]{cosT, sinT};
    }

    /**
     * Compute outer product of two arrays
     * Result: a[i] * b[j] for all i, j
     *
     * @param a First array
     * @param b Second array
     * @return Outer product as flat array (row-major)
     */
    private static float[] outer(float[] a, float[] b) {
        float[] result = new float[a.length * b.length];
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < b.length; j++) {
                result[i * b.length + j] = a[i] * b[j];
            }
        }
        return result;
    }

    /**
     * Repeat each element r times along the last dimension
     * Input shape: (rows, cols)
     * Output shape: (rows, cols * r)
     *
     * @param input Input array
     * @param r Repeat count
     * @param rows Number of rows
     * @param cols Number of columns
     * @return Repeated array
     */
    private static float[] repeatInterleave(float[] input, int r, int rows, int cols) {
        float[] result = new float[rows * cols * r];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                float val = input[i * cols + j];
                for (int k = 0; k < r; k++) {
                    result[i * cols * r + j * r + k] = val;
                }
            }
        }
        return result;
    }

    /**
     * Broadcast and concatenate two arrays along the last dimension
     * base1: (S, dim) -> broadcast to (S, S, dim)
     * base2: (S, dim) -> broadcast to (S, S, dim)
     * Result: concatenate to (S, S, 2*dim)
     *
     * @param base1 First base array (S, dim) flattened
     * @param base2 Second base array (S, dim) flattened
     * @param S Sequence length
     * @param dim Dimension size
     * @return Broadcasted and concatenated array (S, S, 2*dim) flattened
     */
    private static float[] broadcat(float[] base1, float[] base2, int S, int dim) {
        int twoDim = 2 * dim;
        float[] result = new float[S * S * twoDim];

        for (int i = 0; i < S; i++) {
            for (int j = 0; j < S; j++) {
                int outIdx = (i * S + j) * twoDim;
                // Copy base1[i, :] to result[i, j, 0:dim]
                for (int k = 0; k < dim; k++) {
                    result[outIdx + k] = base1[i * dim + k];
                }
                // Copy base2[j, :] to result[i, j, dim:2*dim]
                for (int k = 0; k < dim; k++) {
                    result[outIdx + dim + k] = base2[j * dim + k];
                }
            }
        }

        return result;
    }

    /**
     * Main method for testing
     */
    public static void main(String[] args) {
        // Example from the Python code
        int hiddenSize = 768;
        int headNum = 12;
        int ftSeqLen = 16;
        int ptSeqLen = 16;

        Tensor[] result = compute2DRoPE(hiddenSize, headNum, ftSeqLen, ptSeqLen);
        Tensor freqsCos = result[0];
        Tensor freqsSin = result[1];

        System.out.println("freqs_cos shape: [" + freqsCos.number + ", " +
                          freqsCos.channel + ", " + freqsCos.height + ", " +
                          freqsCos.width + "]");
        System.out.println("freqs_sin shape: [" + freqsSin.number + ", " +
                          freqsSin.channel + ", " + freqsSin.height + ", " +
                          freqsSin.width + "]");

        // Show first few values
        freqsCos.showDM("cos");
        freqsSin.showDM("sin");
    }
}
