package com.omega.example.asr.utils;

import com.omega.common.utils.JsonUtils;
import com.omega.common.utils.MatrixOperation;
import com.omega.common.utils.fft.RealFFT;
import com.omega.engine.nn.network.utils.ModelUtils;
import jcuda.Sizeof;

import javax.sound.sampled.AudioFormat;
import javax.sound.sampled.AudioInputStream;
import javax.sound.sampled.AudioSystem;
import javax.sound.sampled.UnsupportedAudioFileException;
import java.io.File;
import java.io.IOException;
import java.math.BigInteger;

/**
 * 获取fbank
 */
public class FBank {
    public static boolean checkMaxSeqLen(String path, int numBins, int maxLen) {
        try {
            File file = new File(path);
            // 获取音频输入流
            AudioInputStream audioInputStream = AudioSystem.getAudioInputStream(file);
            // 获取音频格式信息
            AudioFormat format = audioInputStream.getFormat();
            int fr = (int) format.getFrameRate();
            int window_size = Math.round(fr * 0.025f);
            int window_shift = Math.round(fr * 0.01f);
            int num = audioInputStream.available() / 2;
            int m = 1 + (num - window_size) / window_shift;
            if (m > maxLen) {
                return false;
            } else {
                return true;
            }
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
        return false;
    }

    public static float[] fbank(String path, int num_bins) {
        File file = new File(path);
        try {
            // 获取音频输入流
            AudioInputStream audioInputStream = AudioSystem.getAudioInputStream(file);
            // 获取音频格式信息
            AudioFormat format = audioInputStream.getFormat();
            float pre_emphasis = 0.97f;
            // 读取音频数据（可选，例如转换成字节数组）
            byte[] bytes = new byte[audioInputStream.available()];
            audioInputStream.read(bytes);
            float[] emphasized_signal = new float[bytes.length / 2];
            byte[] buffer = new byte[Sizeof.SHORT];
            for (int i = 0; i < bytes.length / 2; i++) {
                buffer[0] = bytes[i * Sizeof.SHORT];
                buffer[1] = bytes[i * Sizeof.SHORT + 1];
                emphasized_signal[i] = ModelUtils.readShort(buffer);
            }
            int fr = (int) format.getFrameRate();
            boolean remove_dc_offset = true;
            float[][] strided = get_strided(emphasized_signal, fr, 0.025f, 0.01f, remove_dc_offset, pre_emphasis);
            RealFFT fft = new RealFFT();
            float[][] abs = fft.batchForwardABS(strided);
            float[][] pow = MatrixOperation.pow(abs, 2);
            int window_length_padded = strided[0].length;
            float low_freq = 20;
            float high_freq = 0.0f;
            float vtln_low = 100;
            float vtln_high = -500;
            float vtln_warp = 1.0f;
            float[][] mel_energies = get_mel_banks(num_bins, window_length_padded, fr, low_freq, high_freq, vtln_low, vtln_high, vtln_warp);
            float[][] mel_energies_t = MatrixOperation.transpose(mel_energies);
            float[][] r = MatrixOperation.multiplicationForMatrix(pow, mel_energies_t);
            float eps = 1.1920928955078125e-07f;
            float[] fbank = new float[r.length * r[0].length];
            int cols = r[0].length;
            for (int i = 0; i < fbank.length; i++) {
                fbank[i] = (float) Math.log(Math.max(r[i / cols][i % cols], eps));
            }
            return fbank;
        } catch (UnsupportedAudioFileException e) {
            System.err.println("The specified audio file is not supported.");
            e.printStackTrace();
        } catch (IOException e) {
            System.err.println("Error reading the file.");
            e.printStackTrace();
        }
        return null;
    }

    public static float[] fbank(String path, int num_bins, int maxLen, int idx, float[] len) {
        File file = new File(path);
        try {
            // 获取音频输入流
            AudioInputStream audioInputStream = AudioSystem.getAudioInputStream(file);
            // 获取音频格式信息
            AudioFormat format = audioInputStream.getFormat();
            float pre_emphasis = 0.97f;
            // 读取音频数据（可选，例如转换成字节数组）
            byte[] bytes = new byte[audioInputStream.available()];
            audioInputStream.read(bytes);
            float[] emphasized_signal = new float[bytes.length / 2];
            byte[] buffer = new byte[Sizeof.SHORT];
            for (int i = 0; i < bytes.length / 2; i++) {
                buffer[0] = bytes[i * Sizeof.SHORT];
                buffer[1] = bytes[i * Sizeof.SHORT + 1];
                emphasized_signal[i] = ModelUtils.readShort(buffer);
            }
            int fr = (int) format.getFrameRate();
            boolean remove_dc_offset = true;
            float[][] strided = get_strided(emphasized_signal, fr, 0.025f, 0.01f, remove_dc_offset, pre_emphasis);
            RealFFT fft = new RealFFT();
            float[][] abs = fft.batchForwardABS(strided);
            float[][] pow = MatrixOperation.pow(abs, 2);
            int window_length_padded = strided[0].length;
            float low_freq = 20;
            float high_freq = 0.0f;
            float vtln_low = 100;
            float vtln_high = -500;
            float vtln_warp = 1.0f;
            float[][] mel_energies = get_mel_banks(num_bins, window_length_padded, fr, low_freq, high_freq, vtln_low, vtln_high, vtln_warp);
            float[][] mel_energies_t = MatrixOperation.transpose(mel_energies);
            float[][] r = MatrixOperation.multiplicationForMatrix(pow, mel_energies_t);
            float eps = 1.1920928955078125e-07f;
            float[] fbank = new float[maxLen * r[0].length];
            int cols = r[0].length;
            for (int i = 0; i < r.length * r[0].length; i++) {
                fbank[i] = (float) Math.log(Math.max(r[i / cols][i % cols], eps));
            }
            len[idx] = r.length;
            return fbank;
        } catch (UnsupportedAudioFileException e) {
            System.err.println("The specified audio file is not supported.");
            e.printStackTrace();
        } catch (IOException e) {
            System.err.println("Error reading the file.");
            e.printStackTrace();
        }
        return null;
    }

    public static void main(String[] args) {
        // 文件路径
        String filePath = "I:\\dataset\\asr\\data\\0.wav";
        File file = new File(filePath);
        try {
            // 获取音频输入流
            AudioInputStream audioInputStream = AudioSystem.getAudioInputStream(file);
            // 获取音频格式信息
            AudioFormat format = audioInputStream.getFormat();
            System.out.println("FrameRate:" + format.getFrameRate());
            System.out.println("FrameSize:" + format.getFrameSize());
            System.out.println("Format: " + format);
            float pre_emphasis = 0.97f;
            // 读取音频数据（可选，例如转换成字节数组）
            byte[] bytes = new byte[audioInputStream.available()];
            audioInputStream.read(bytes);
            int[] cache = new int[bytes.length / 2];
            float[] emphasized_signal = new float[cache.length];
            byte[] buffer = new byte[Sizeof.SHORT];
            for (int i = 0; i < bytes.length / 2; i++) {
                buffer[0] = bytes[i * Sizeof.SHORT];
                buffer[1] = bytes[i * Sizeof.SHORT + 1];
                cache[i] = ModelUtils.readShort(buffer);
            }
            for (int i = 0; i < cache.length; i++) {
                emphasized_signal[i] = cache[i];
            }
            System.out.println(cache.length);
            int fr = (int) format.getFrameRate();
            boolean remove_dc_offset = true;
            float[][] strided = get_strided(emphasized_signal, fr, 0.025f, 0.01f, remove_dc_offset, pre_emphasis);
            // System.out.println(JsonUtils.toJson(strided));
            RealFFT fft = new RealFFT();
            float[][] abs = fft.batchForwardABS(strided);
            // System.out.println(JsonUtils.toJson(abs));
            float[][] pow = MatrixOperation.pow(abs, 2);
            // System.out.println(JsonUtils.toJson(pow));
            int num_bins = 80;
            int window_length_padded = strided[0].length;
            float low_freq = 20;
            float high_freq = 0.0f;
            float vtln_low = 100;
            float vtln_high = -500;
            float vtln_warp = 1.0f;
            float[][] mel_energies = get_mel_banks(num_bins, window_length_padded, fr, low_freq, high_freq, vtln_low, vtln_high, vtln_warp);
            // System.out.println(JsonUtils.toJson(mel_energies));
            float[][] mel_energies_t = MatrixOperation.transpose(mel_energies);
            float[][] r = MatrixOperation.multiplicationForMatrix(pow, mel_energies_t);
            float eps = 1.1920928955078125e-07f;
            float[] fbank = new float[r.length * r[0].length];
            int cols = r[0].length;
            for (int i = 0; i < fbank.length; i++) {
                fbank[i] = (float) Math.log(Math.max(r[i / cols][i % cols], eps));
            }
            System.out.println(JsonUtils.toJson(fbank));
            System.out.println("Read " + bytes.length + " bytes of audio data.");
        } catch (UnsupportedAudioFileException e) {
            System.err.println("The specified audio file is not supported.");
            e.printStackTrace();
        } catch (IOException e) {
            System.err.println("Error reading the file.");
            e.printStackTrace();
        }
    }

    public static float mel_scale_scalar(float freq) {
        return (float) (1127.0f * Math.log(1.0 + freq / 700.0));
    }

    public static float inverse_mel_scale(float mel_freq) {
        return (float) (700.0 * (Math.exp(mel_freq / 1127.0) - 1.0));
    }

    public static float[][] get_mel_banks(int num_bins, int window_length_padded, float sample_freq, float low_freq, float high_freq, float vtln_low, float vtln_high, float vtln_warp_factor) {
        // System.out.println(num_bins);
        // System.out.println(window_length_padded);
        // System.out.println(sample_freq);
        // System.out.println(low_freq);
        // System.out.println(high_freq);
        // System.out.println(vtln_low);
        // System.out.println(vtln_high);
        // System.out.println(vtln_warp_factor);
        int num_fft_bins = window_length_padded / 2;
        float nyquist = 0.5f * sample_freq;
        if (high_freq <= 0.0) {
            high_freq += nyquist;
        }
        float fft_bin_width = sample_freq / window_length_padded;
        float mel_low_freq = mel_scale_scalar(low_freq);
        float mel_high_freq = mel_scale_scalar(high_freq);
        float mel_freq_delta = (mel_high_freq - mel_low_freq) / (num_bins + 1);
        if (vtln_high < 0.0) {
            vtln_high += nyquist;
        }
        float[][] left_mel = new float[num_bins][1];
        float[][] center_mel = new float[num_bins][1];
        float[][] right_mel = new float[num_bins][1];
        // float[][] center_freqs = new float[num_bins][1];
        // float[][] mel = new float[1][num_fft_bins];
        // System.err.println(mel_freq_delta);
        for (int i = 0; i < num_bins; i++) {
            left_mel[i][0] = mel_low_freq + i * mel_freq_delta;
            // float cv = mel_low_freq + (i + 1.0f) * mel_freq_delta;
            center_mel[i][0] = mel_low_freq + (i + 1.0f) * mel_freq_delta;
            right_mel[i][0] = mel_low_freq + (i + 2.0f) * mel_freq_delta;
            // center_freqs[i][0] = (float) (700.0 * (Math.exp(cv / 1127.0) - 1.0));
        }
        // System.err.println(JsonUtils.toJson(left_mel));
        /**
         * make up down slope
         */
        // float[][] up_slope = new float[num_bins][num_fft_bins];
        // float[][] down_slope = new float[num_bins][num_fft_bins];
        float[][] bins = new float[num_bins][num_fft_bins + 1];
        for (int j = 0; j < num_fft_bins; j++) {
            float mel_v = (float) (1127.0 * Math.log((1.0 + fft_bin_width * j / 700.0)));
            // System.err.println(mel_v);
            // mel[0][j] = mel_v;
            for (int i = 0; i < num_bins; i++) {
                float up_slope = (mel_v - left_mel[i][0]) / (center_mel[i][0] - left_mel[i][0]);
                float down_slope = (right_mel[i][0] - mel_v) / (right_mel[i][0] - center_mel[i][0]);
                bins[i][j] = Math.max(0, Math.min(up_slope, down_slope));
            }
        }
        return bins;
    }

    public static float[] getWindowFunction(int window_size) {
        /**
         * POVEY = HAMMING.pow(0.85)
         *
         */
        float[] window_function = new float[window_size];
        for (int i = 0; i < window_size; i++) {
            window_function[i] = (float) Math.pow((0.5d - 0.5d * Math.cos(2 * Math.PI * i / (window_size - 1))), 0.85d);
        }
        return window_function;
    }

    public static float[][] get_strided(float[] emphasized_signal, int fs, float flen, float fss, boolean remove_dc_offset, float pre_emphasis) {
        int window_size = Math.round(fs * flen);
        int window_shift = Math.round(fs * fss);
        int num = emphasized_signal.length;
        int m = 1 + (num - window_size) / window_shift;
        /**
         * padding
         *
         */
        int padding = new BigInteger("2").pow(new BigInteger((window_size - 1) + "").bitLength()).intValue();
        float[][] strided = new float[m][padding];
        float[] mean = new float[m];
        for (int i = 0; i < m; i++) {
            float mv = 0.0f;
            for (int j = 0; j < window_size; j++) {
                float val = emphasized_signal[i * window_shift + j];
                strided[i][j] = val;
                mv += val / window_size;
            }
            mean[i] = mv;
        }
        float[] getWindowFunction = getWindowFunction(window_size);
        // System.err.println(JsonUtils.toJson(getWindowFunction));
        float preVal = 0.0f;
        if (remove_dc_offset) {
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < window_size; j++) {
                    float tmp = (strided[i][j] - mean[i]);
                    if (j == 0) {
                        strided[i][j] = (tmp - pre_emphasis * tmp) * getWindowFunction[j];
                    } else {
                        strided[i][j] = ((strided[i][j] - mean[i]) - pre_emphasis * preVal) * getWindowFunction[j];
                    }
                    preVal = tmp;
                }
            }
        }
        // System.err.println(JsonUtils.toJson(strided[1]));
        return strided;
    }
}
