package com.omega.common.utils;

import com.omega.common.config.Tensor;
import com.omega.engine.active.ActiveType;
import com.omega.engine.nn.layer.ParamsInit;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

/**
 * random utils
 *
 * @author Administrator
 */
public class RandomUtils {
    private static Random instance;

    public static void setSeed(long seed) {
        getInstance().setSeed(seed);
    }

    public static Random getInstance() {
        if (instance == null) {
            instance = new Random();
        }
        return instance;
    }

    public static int rand() {
        return getInstance().nextInt();
    }

    public static int randomInt(int min, int max) {
        return min + (int) (Math.random() * (max - min));
    }

    public static int[] randomInt(int min, int max, int count) {
        int[] tmp = new int[count];
        for (int i = 0; i < count; i++) {
            tmp[i] = min + (int) (Math.random() * (max - min));
        }
        return tmp;
    }

    //	public static int[] randomInt2(int min,int max,int count) {
    //		RandomDataGenerator r = new RandomDataGenerator();
    //		int[] tmp = new int[count];
    //		for(int i = 0;i<count;i++) {
    //			tmp[i] = r.nextInt(min, max);
    //		}
    //		return tmp;
    //	}
    public static int uniformInt(int min, int max) {
        if (max < min) {
            int swap = min;
            min = max;
            max = swap;
        }
        return min + (int) (Math.random() * (max - min));
    }

    public static float randomFloat(float min, float max) {
        return min + (float) (Math.random() * (max - min));
    }

    public static float randomFloat() {
        return (float) Math.random();
    }

    public static float randomGaussianFloat() {
        return (float) getInstance().nextGaussian();
    }

    public static float uniformFloat(float min, float max) {
        if (max < min) {
            float swap = min;
            min = max;
            max = swap;
        }
        return min + (float) (Math.random() * (max - min));
    }

    public static float[] uniformFloat(int size, float min, float max) {
        float[] y = new float[size];
        if (max < min) {
            float swap = min;
            min = max;
            max = swap;
        }
        for (int i = 0; i < size; i++) {
            y[i] = min + (float) (Math.random() * (max - min));
        }
        return y;
    }

    public static float[] uniformFloat(int size, int gain) {
        float stdv = (float) (1.0f / Math.sqrt(gain));
        return uniformFloat(size, -stdv, stdv);
    }

    public static float randomScale(float s) {
        float scale = randomFloat(1, s);
        if (Math.random() >= 0.5d) {
            return scale;
        }
        return 1.0f / scale;
    }

    /**
     * 根据激活函数获取增益值
     *
     * @return
     */
    public static float getGain(ActiveType at) {
        float gain = 1.0f;
        switch (at) {
            case relu:
                gain = (float) Math.sqrt(2.0);
                break;
            case tanh:
                gain = 5.0f / 3.0f;
                break;
            default:
                break;
        }
        return gain;
    }

    /**
     * he随机数
     *
     * @param x
     * @return
     */
    public static float[][] heRandom(int x, int y, float n) {
        float[][] temp = new float[x][y];
        float t = (float) Math.sqrt(2.0d / n);
        for (int i = 0; i < x; i++) {
            for (int j = 0; j < y; j++) {
                temp[i][j] = (float) (Math.abs(getInstance().nextGaussian()) * t);
            }
        }
        return temp;
    }

    /**
     * he随机数
     *
     * @param x
     * @return
     */
    public static float[] heRandom(int x, float n) {
        float[] temp = new float[x];
        float t = (float) Math.sqrt(2.0d / n);
        for (int i = 0; i < x; i++) {
            temp[i] = (float) (Math.abs(getInstance().nextGaussian()) * t);
        }
        return temp;
    }

    /**
     * kaiming初始化
     * <p>
     * N～ (0,std)
     * <p>
     * std = sqrt(2/(1+a^2)*fan_in)
     * <p>
     * a 为激活函数的负半轴的斜率，relu 是 0
     *
     * @param x
     * @return
     */
    public static float[] kaimingNormalRandom(int x, float a, float fan) {
        float[] temp = new float[x];
        float gain = (float) (Math.sqrt(2.0d));
        /**
         * std = gain / math.sqrt(fan)

         */
        float std = (float) (gain / Math.sqrt(fan));
        //		float mean = 0;
        for (int i = 0; i < x; i++) {
            //			temp[i] = (float) (Math.sqrt(std) * getInstance().nextGaussian() + mean);
            temp[i] = (float) (getInstance().nextGaussian() * std);
        }
        return temp;
    }
    //	/**
    //	 * kaiming初始化(0,std)
    //	 * N～ (0,std)
    //	 * std = sqrt(2/(1+a^2)*fan_in)
    //	 * a 为激活函数的负半轴的斜率，relu 是 0
    //	 * @param x
    //	 * @return
    //	 */
    //	public static float[] kaimingNormalRandom(int x,float a,float fan){
    //		float[] temp = new float[x];
    //		/**
    //		 * std = gain / math.sqrt(fan)
    //		 */
    //		float std = (float) Math.sqrt(2 / (1 + Math.pow(a, 2)) * fan);
    //		float mean = 0;
    //		for(int i = 0;i<x;i++) {
    //			temp[i] = (float) (std * getInstance().nextGaussian() + mean);
    //		}
    //		return temp;
    //	}

    /**
     * kaiming初始化(均值分布)
     * <p>
     * N～ (0,std)
     * <p>
     * std = sqrt(2/(1+a^2)*fan_in)
     * <p>
     * a 为激活函数的负半轴的斜率，relu 是 0
     *
     * @param x
     * @return
     */
    public static float[] kaimingUniformRandom(int x, float a, float fan) {
        float[] temp = new float[x];
        float bound = (float) Math.sqrt(6 / (1 + Math.pow(a, 2)) * fan);
        float max = bound;
        float min = -bound;
        for (int i = 0; i < x; i++) {
            temp[i] = (float) Math.random() * (max - min) + min;
        }
        return temp;
    }

    /**
     * kaiming初始化
     * <p>
     * N～ (0,std)
     * <p>
     * std = sqrt(2/(1+a^2)*fan_in)
     * <p>
     * a 为激活函数的负半轴的斜率，relu 是 0
     *
     * @param x
     * @return
     */
    public static float[] kaimingUniformBias(int x, float n) {
        float[] temp = new float[x];
        float bound = (float) (1 / Math.sqrt(n));
        float max = bound;
        float min = -bound;
        for (int i = 0; i < x; i++) {
            temp[i] = (float) Math.random() * (max - min) + min;
        }
        return temp;
    }

    /**
     * he随机数
     *
     * @param x
     * @return
     */
    public static float[][][][] heRandom(int c, int n, int x, int y, float nn) {
        float[][][][] temp = new float[c][n][x][y];
        float t = (float) Math.sqrt(2.0d / nn);
        for (int k = 0; k < c; k++) {
            for (int l = 0; l < n; l++) {
                for (int i = 0; i < x; i++) {
                    for (int j = 0; j < y; j++) {
                        temp[k][l][i][j] = (float) (getInstance().nextGaussian() * t);
                    }
                }
            }
        }
        return temp;
    }

    /**
     * 高斯随机数
     *
     * @param x
     * @return
     */
    public static float[] gaussianRandom(int x, float ratio) {
        float[] temp = new float[x];
        for (int i = 0; i < x; i++) {
            temp[i] = (float) (getInstance().nextGaussian() * ratio);
        }
        return temp;
    }

    public static float[] gaussianRandom(int x, float mean, float std) {
        float[] temp = new float[x];
        for (int i = 0; i < x; i++) {
            temp[i] = (float) (getInstance().nextGaussian() * std + mean);
        }
        return temp;
    }

    public static void gaussianRandom(Tensor output) {
        for (int i = 0; i < output.dataLength; i++) {
            output.data[i] = (float) (getInstance().nextGaussian());
        }
        if (output.isHasGPU()) {
            output.hostToDevice();
        }
    }

    public static void gaussianRandom(Tensor output, float mean, float std) {
        if (output.data == null) {
            output.data = new float[output.dataLength];
        }
        for (int i = 0; i < output.dataLength; i++) {
            output.data[i] = (float) (getInstance().nextGaussian() * std + mean);
        }
        if (output.isHasGPU()) {
            output.hostToDevice();
        }
    }
    //	public static void gaussianRandom2(Tensor output,float mean,float std){
    //		RandomDataGenerator r = new RandomDataGenerator();
    ////		NormalDistribution n = new NormalDistribution(mean, std);
    //		for(int i = 0;i<output.dataLength;i++) {
    //			output.data[i] = (float)(r.nextGaussian(mean, std));
    ////			System.out.println(new StringBuilder(String.valueOf(output.data[i])).insert(2,"-"));
    //		}
    //		if(output.isHasGPU()) {
    //			output.hostToDevice();
    ////			output.showDM();
    //
    //		}
    //	}

    /**
     * 高斯随机数
     *
     * @param x
     * @return
     */
    public static void gaussianRandom(float[] temp, float ratio) {
        for (int i = 0; i < temp.length; i++) {
            temp[i] = (float) (getInstance().nextGaussian() * ratio);
        }
    }

    /**
     * 高斯随机数
     *
     * @param x
     * @return
     */
    public static void random(float[] temp, float ratio) {
        for (int i = 0; i < temp.length; i++) {
            temp[i] = (float) (getInstance().nextFloat() * ratio);
        }
    }

    /**
     * 高斯随机数
     *
     * @param x
     * @return
     */
    public static float[][] gaussianRandom(int x, int y, float ratio) {
        float[][] temp = new float[x][y];
        for (int i = 0; i < x; i++) {
            for (int j = 0; j < y; j++) {
                temp[i][j] = (float) (getInstance().nextGaussian() * ratio);
            }
        }
        return temp;
    }

    /**
     * 高斯随机数
     *
     * @param x
     * @return
     */
    public static float[][] random(int x, int y, float ratio) {
        float[][] temp = new float[x][y];
        for (int i = 0; i < x; i++) {
            for (int j = 0; j < y; j++) {
                temp[i][j] = (float) (getInstance().nextFloat() * ratio);
            }
        }
        return temp;
    }

    /**
     * 高斯随机数
     *
     * @param x
     * @return
     */
    public static float[][][][] gaussianRandom(int c, int n, int x, int y, float ratio) {
        float[][][][] temp = new float[c][n][x][y];
        for (int k = 0; k < c; k++) {
            for (int l = 0; l < n; l++) {
                for (int i = 0; i < x; i++) {
                    for (int j = 0; j < y; j++) {
                        temp[k][l][i][j] = (float) (getInstance().nextGaussian() * ratio);
                    }
                }
            }
        }
        return temp;
    }

    public static float[] x2Random(int x) {
        float[] temp = new float[x];
        float scale = (float) Math.sqrt(1.0 / x);
        for (int i = 0; i < x; i++) {
            temp[i] = getInstance().nextFloat() * scale;
        }
        return temp;
    }

    public static float[][] x2Random(int x, int y) {
        float[][] temp = new float[x][y];
        float scale = (float) Math.sqrt(1.0 / x);
        for (int i = 0; i < x; i++) {
            for (int j = 0; j < y; j++) {
                temp[i][j] = getInstance().nextFloat() * scale;
            }
        }
        return temp;
    }

    public static float[][][][] x2Random(int c, int n, int x, int y) {
        float[][][][] temp = new float[c][n][x][y];
        float scale = (float) Math.sqrt(1.0 / x);
        for (int k = 0; k < c; k++) {
            for (int l = 0; l < n; l++) {
                for (int i = 0; i < x; i++) {
                    for (int j = 0; j < y; j++) {
                        temp[k][l][i][j] = getInstance().nextFloat() * scale;
                    }
                }
            }
        }
        return temp;
    }

    /**
     * xavier随机数
     *
     * @param x
     * @return
     */
    public static float[] xavierRandom(int x, float ratio) {
        float[] temp = new float[x];
        for (int i = 0; i < x; i++) {
            temp[i] = (float) (getInstance().nextGaussian() / Math.sqrt(x));
        }
        return temp;
    }

    /**
     * xavier随机数
     *
     * @param x
     * @return
     */
    public static float[][] xavierRandom(int x, int y, int fanIn, int fanOut) {
        float[][] temp = new float[x][y];
        float t = (float) Math.sqrt(2.0f / (fanIn + fanOut));
        for (int i = 0; i < x; i++) {
            for (int j = 0; j < y; j++) {
                temp[i][j] = (float) (getInstance().nextGaussian() * t);
            }
        }
        return temp;
    }

    /**
     * xavier随机数
     *
     * @param x
     * @return
     */
    public static float[] xavierRandom(int x, int fanIn, int fanOut) {
        float[] temp = new float[x];
        float t = (float) Math.sqrt(2.0f / (fanIn + fanOut));
        for (int i = 0; i < x; i++) {
            temp[i] = (float) (getInstance().nextGaussian() * t);
        }
        return temp;
    }

    /**
     * xavier随机数
     *
     * @param x
     * @return
     */
    public static float[] xavierReluRandom(int x, int fanIn, int fanOut) {
        float[] temp = new float[x];
        float t = (float) (Math.sqrt(2.0f / (fanIn + fanOut)) * Math.sqrt(2.0f));
        for (int i = 0; i < x; i++) {
            temp[i] = (float) (getInstance().nextGaussian() * t);
        }
        return temp;
    }

    /**
     * xavier随机数
     *
     * @param x
     * @return
     */
    public static float[] xavierUniform(int x, int fanIn, int fanOut, float gain) {
        float[] temp = new float[x];
        double std = gain * Math.sqrt(2.0 / (fanIn + fanOut));
        float max = (float) (Math.sqrt(3.0) * std);
        float min = (float) -(Math.sqrt(3.0) * std);
        for (int i = 0; i < x; i++) {
            temp[i] = min + getInstance().nextFloat() * (max - min);
        }
        return temp;
    }

    /**
     * xavier随机数
     *
     * @param x
     * @return
     */
    public static float[] xavierLeakyReluRandom(int x, int fanIn, int fanOut) {
        float[] temp = new float[x];
        float t = (float) (Math.sqrt(2.0f / (fanIn + fanOut)) * Math.sqrt(2.0f / (1 + 0.01f * 0.01f)));
        for (int i = 0; i < x; i++) {
            temp[i] = (float) (getInstance().nextGaussian() * t);
        }
        return temp;
    }

    public static float gain(ParamsInit paramsInit) {
        float gain = 1.0f;
        switch (paramsInit) {
            case sigmoid:
                gain = 1.0f;
                break;
            case tanh:
                gain = 5.0f / 3.0f;
                break;
            case relu:
                gain = (float) Math.sqrt(2.0f);
                break;
            case leaky_relu:
                gain = (float) Math.sqrt(2.0f / (1.0f + Math.sqrt(5) * Math.sqrt(5)));
                break;
            default:
                gain = 1.0f;
                break;
        }
        return gain;
    }

    public static float[] kaiming_normal(int x, int fan, ParamsInit paramsInit) {
        float[] temp = new float[x];
        float gain = gain(paramsInit);
        float std = (float) (gain / Math.sqrt(fan));
        for (int i = 0; i < x; i++) {
            temp[i] = (float) (getInstance().nextGaussian() * std);
        }
        return temp;
    }

    public static float[] kaiming_uniform(int x, int fan, ParamsInit paramsInit) {
        if (paramsInit == null) {
            paramsInit = ParamsInit.leaky_relu;
        }
        float[] temp = new float[x];
        float gain = gain(paramsInit);
        float std = (float) (gain / Math.sqrt(fan));
        float bound = (float) (Math.sqrt(3.0f) * std);
        for (int i = 0; i < x; i++) {
            temp[i] = (float) Math.random() * (bound - (-bound)) + (-bound);
        }
        return temp;
    }

    public static float[] uniform(int x, float mean, float std) {
        float[] temp = new float[x];
        for (int i = 0; i < x; i++) {
            temp[i] = (float) (std * Math.random() + mean);
        }
        return temp;
    }

    public static float[] normal_(int x, float mean, float std) {
        float[] temp = new float[x];
        for (int i = 0; i < x; i++) {
            temp[i] = (float) (std * getInstance().nextGaussian() + mean);
        }
        return temp;
    }

    /**
     * xavier随机数
     *
     * @param x
     * @return
     */
    public static float[] val(int x, float val) {
        float[] temp = new float[x];
        for (int i = 0; i < x; i++) {
            temp[i] = val;
        }
        return temp;
    }

    /**
     * xavier随机数
     *
     * @param x
     * @return
     */
    public static float[][] xavierRandomCaffe(int x, int y, int fanIn, int fanOut) {
        float[][] temp = new float[x][y];
        float t = (float) 2.0f / (fanIn + fanOut);
        for (int i = 0; i < x; i++) {
            for (int j = 0; j < y; j++) {
                temp[i][j] = (float) (getInstance().nextGaussian() * t);
            }
        }
        return temp;
    }

    /**
     * xavier随机数
     *
     * @param x
     * @return
     */
    public static float[][][][] xavierRandom(int c, int n, int x, int y, int fanIn, int fanOut) {
        float[][][][] temp = new float[c][n][x][y];
        float t = (float) Math.sqrt(2.0d / (fanIn + fanOut));
        for (int k = 0; k < c; k++) {
            for (int l = 0; l < n; l++) {
                for (int i = 0; i < x; i++) {
                    for (int j = 0; j < y; j++) {
                        temp[k][l][i][j] = (float) (getInstance().nextGaussian() * t);
                    }
                }
            }
        }
        return temp;
    }

    /**
     * xavier随机数
     *
     * @param x
     * @return
     */
    public static float[][][][] heRandom(int c, int n, int x, int y, int fanIn) {
        float[][][][] temp = new float[c][n][x][y];
        float t = (float) Math.sqrt(2.0d / fanIn);
        for (int k = 0; k < c; k++) {
            for (int l = 0; l < n; l++) {
                for (int i = 0; i < x; i++) {
                    for (int j = 0; j < y; j++) {
                        temp[k][l][i][j] = (float) (Math.abs(getInstance().nextGaussian()) * t);
                    }
                }
            }
        }
        return temp;
    }

    /**
     * xavier随机数
     *
     * @param x
     * @return
     */
    public static float[][][][] xavierRandomCaffeIn(int c, int n, int x, int y, int fanIn, int fanOut) {
        float[][][][] temp = new float[c][n][x][y];
        float t = (float) 1.0f / fanIn;
        for (int k = 0; k < c; k++) {
            for (int l = 0; l < n; l++) {
                for (int i = 0; i < x; i++) {
                    for (int j = 0; j < y; j++) {
                        temp[k][l][i][j] = (float) (getInstance().nextGaussian() * t);
                    }
                }
            }
        }
        return temp;
    }

    /**
     * xavier随机数
     *
     * @param x
     * @return
     */
    public static float[][][][] xavierRandomCaffeOut(int c, int n, int x, int y, int fanIn, int fanOut) {
        float[][][][] temp = new float[c][n][x][y];
        float t = (float) 1.0f / fanOut;
        for (int k = 0; k < c; k++) {
            for (int l = 0; l < n; l++) {
                for (int i = 0; i < x; i++) {
                    for (int j = 0; j < y; j++) {
                        temp[k][l][i][j] = (float) (getInstance().nextGaussian() * t);
                    }
                }
            }
        }
        return temp;
    }

    /**
     * xavier随机数
     *
     * @param x
     * @return
     */
    public static float[][][][] xavierRandomCaffe(int c, int n, int x, int y, int fanIn, int fanOut) {
        float[][][][] temp = new float[c][n][x][y];
        float t = (float) 2.0d / (fanIn + fanOut);
        for (int k = 0; k < c; k++) {
            for (int l = 0; l < n; l++) {
                for (int i = 0; i < x; i++) {
                    for (int j = 0; j < y; j++) {
                        temp[k][l][i][j] = (float) (getInstance().nextGaussian() * t);
                    }
                }
            }
        }
        return temp;
    }

    /**
     * xavier随机数
     *
     * @param x
     * @return
     */
    public static float[] order(int x, float a, float b) {
        float[] temp = new float[x];
        for (int i = 0; i < x; i++) {
            temp[i] = i * a + b;
        }
        return temp;
    }

    public static float[] orderAndUnsqueeze(int x, int N, float a, float b) {
        float[] temp = new float[N * x];
        for (int n = 0; n < N; n++) {
            for (int i = 0; i < x; i++) {
                temp[n * x + i] = i * a + b;
            }
        }
        return temp;
    }

    /**
     * 范围随机
     *
     * @param org
     * @param k
     * @return
     */
    public static Tensor random(Tensor org, int k) {
        List<Integer> list = new ArrayList<Integer>();
        for (int i = 0; i < org.number; i++) {
            list.add(i);
        }
        Collections.shuffle(list);
        float[] data = new float[k * org.getOnceSize()];
        for (int kn = 0; kn < k; kn++) {
            System.arraycopy(org.getByNumber(list.get(kn)), 0, data, kn * org.getOnceSize(), org.getOnceSize());
        }
        return new Tensor(k, org.channel, org.height, org.width, data);
    }

    public static int getRandomNumber(float[] pros) {
        float rv = RandomUtils.getInstance().nextFloat();
        float sum = MatrixOperation.sum(pros);
        float cu_proy = 0;
        for (int i = 0; i < pros.length; i++) {
            cu_proy += pros[i] / sum;
            if (rv <= cu_proy) {
                return i;
            }
        }
        return 0;
    }

    public static void main(String[] args) {
        RandomUtils.setSeed(100);
        float[] tmp = RandomUtils.kaimingNormalRandom(100, 10, 10);
        System.out.println(JsonUtils.toJson(tmp));
    }
}

