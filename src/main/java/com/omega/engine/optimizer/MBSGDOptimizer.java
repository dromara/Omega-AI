package com.omega.engine.optimizer;

import com.omega.common.data.utils.DataTransforms;
import com.omega.common.utils.*;
import com.omega.engine.check.BaseCheck;
import com.omega.engine.gpu.CUDAModules;
import com.omega.engine.nn.data.BaseData;
import com.omega.engine.nn.grad.GradClipping;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.gpu.RoPEKernel;
import com.omega.engine.nn.network.*;
import com.omega.engine.nn.network.vae.*;
import com.omega.engine.nn.network.vqgan.LPIPS;
import com.omega.engine.nn.network.vqgan.PatchGANDiscriminator;
import com.omega.engine.optimizer.lr.LearnRateUpdate;
import com.omega.engine.tensor.Tensor;
import com.omega.example.diffusion.utils.DiffusionImageDataLoader;
import com.omega.example.rnn.data.OneHotDataLoader;
import com.omega.example.rnn.data.RNNDataLoader;
import com.omega.example.sd.utils.SDImageDataLoader;
import com.omega.example.sd.utils.SDImageDataLoaderEN;
import com.omega.example.transformer.utils.ModelUtils;
import com.omega.example.yolo.data.BaseDataLoader;
import com.omega.example.yolo.data.DetectionDataLoader;
import com.omega.example.yolo.utils.YoloLabelUtils;
import jcuda.driver.JCudaDriver;
import jcuda.runtime.JCuda;

import java.math.BigDecimal;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

/**
 * Mini Batch Stochastic Gradient Descent
 *
 * @author Administrator
 */
public class MBSGDOptimizer extends Optimizer {
    private YoloLabelUtils u;

    public MBSGDOptimizer(Network network, int trainTime, float error, int batchSize, boolean warmUp) throws Exception {
        super(network, batchSize, trainTime, error, warmUp);
        // TODO Auto-generated constructor stub
        this.batchSize = batchSize;
        this.loss = new Tensor(batchSize, this.network.oChannel, this.network.oHeight, this.network.oWidth);
        this.lossDiff = new Tensor(batchSize, this.network.oChannel, this.network.oHeight, this.network.oWidth);
    }

    public MBSGDOptimizer(String sid, Network network, int trainTime, float error, int batchSize, boolean warmUp) throws Exception {
        super(network, batchSize, trainTime, error, warmUp);
        // TODO Auto-generated constructor stub
        this.setSid(sid);
        this.batchSize = batchSize;
        this.loss = new Tensor(batchSize, this.network.oChannel, this.network.oHeight, this.network.oWidth);
        this.lossDiff = new Tensor(batchSize, this.network.oChannel, this.network.oHeight, this.network.oWidth);
    }

    public MBSGDOptimizer(Network network, int trainTime, float error, int batchSize, LearnRateUpdate learnRateUpdate, boolean warmUp) throws Exception {
        super(network, batchSize, trainTime, error, warmUp);
        // TODO Auto-generated constructor stub
        this.batchSize = batchSize;
        this.loss = new Tensor(batchSize, this.network.oChannel, this.network.oHeight, this.network.oWidth);
        this.lossDiff = new Tensor(batchSize, this.network.oChannel, this.network.oHeight, this.network.oWidth);
        this.learnRateUpdate = learnRateUpdate;
    }

    public MBSGDOptimizer(Network network, int trainTime, float error, int batchSize, LearnRateUpdate learnRateUpdate, boolean warmUp, BaseCheck check) throws Exception {
        super(network, batchSize, trainTime, error, warmUp);
        // TODO Auto-generated constructor stub
        this.batchSize = batchSize;
        this.loss = new Tensor(batchSize, this.network.oChannel, this.network.oHeight, this.network.oWidth);
        this.lossDiff = new Tensor(batchSize, this.network.oChannel, this.network.oHeight, this.network.oWidth);
        this.learnRateUpdate = learnRateUpdate;
        this.check = check;
    }

    public MBSGDOptimizer(String sid, Network network, int trainTime, float error, int batchSize, LearnRateUpdate learnRateUpdate, boolean warmUp) throws Exception {
        super(network, batchSize, trainTime, error, warmUp);
        // TODO Auto-generated constructor stub
        this.setSid(sid);
        this.batchSize = batchSize;
        this.loss = new Tensor(batchSize, this.network.oChannel, this.network.oHeight, this.network.oWidth);
        this.lossDiff = new Tensor(batchSize, this.network.oChannel, this.network.oHeight, this.network.oWidth);
        this.learnRateUpdate = learnRateUpdate;
    }

    public static void q_mean_variance(Tensor x_0, Tensor x_t, Tensor t, float[] posterior_mean_coef1, float[] posterior_mean_coef2, float[] posterior_mean) {
        for (int b = 0; b < x_t.number; b++) {
            for (int i = 0; i < x_t.getOnceSize(); i++) {
                int idx = b * x_t.getOnceSize() + i;
                posterior_mean[idx] = posterior_mean_coef1[b] * x_0.data[idx] - posterior_mean_coef2[b] * x_t.data[idx];
            }
        }
    }

    public static void sample_prev_timestep(DiffusionUNetCond2 network, Tensor condInput, Tensor xt, Tensor t, Tensor x0, int timestep, float[] a, float[] b, float[] betas, float[] alphas, float[] alphas_bar) {
        for (int i = 0; i < xt.number; i++) {
            t.data[i] = timestep;
        }
        t.hostToDevice();
        Tensor noisePred = null;
        if (condInput != null) {
            noisePred = network.forward(xt, t, condInput);
        } else {
            noisePred = network.forward(xt, t);
        }
        noisePred.syncHost();
        JCuda.cudaDeviceSynchronize();
        if (timestep > 0) {
            System.err.println("timestep:" + timestep);
            float var = (1.0f - alphas_bar[timestep - 1]) / (1.0f - alphas_bar[timestep]) * betas[timestep];
            float sigma = (float) Math.pow(var, 0.5);
            float[] noise = RandomUtils.gaussianRandom(noisePred.dataLength, 1.0f);
            for (int i = 0; i < xt.dataLength; i++) {
                xt.data[i] = (float) ((xt.data[i] - (betas[timestep] * noisePred.data[i]) / b[timestep]) / Math.sqrt(alphas[timestep])) + sigma * noise[i];
            }
        } else {
            /**
             * mean

             */
            for (int i = 0; i < xt.dataLength; i++) {
                xt.data[i] = (float) ((xt.data[i] - (betas[timestep] * noisePred.data[i]) / b[timestep]) / Math.sqrt(alphas[timestep]));
            }
        }
        xt.hostToDevice();
    }
    
    public static void sample_prev_timestep(DiT network, Tensor condInput, Tensor xt, Tensor t,Tensor cos,Tensor sin, Tensor x0, int timestep, float[] a, float[] b, float[] betas, float[] alphas, float[] alphas_bar) {
        for (int i = 0; i < xt.number; i++) {
            t.data[i] = timestep;
        }
        t.hostToDevice();
        Tensor noisePred = network.forward(xt, t, condInput, cos, sin);
        noisePred.syncHost();
        JCuda.cudaDeviceSynchronize();
        if (timestep > 0) {
            System.err.println("timestep:" + timestep);
            float var = (1.0f - alphas_bar[timestep - 1]) / (1.0f - alphas_bar[timestep]) * betas[timestep];
            float sigma = (float) Math.pow(var, 0.5);
            float[] noise = RandomUtils.gaussianRandom(noisePred.dataLength, 1.0f);
            for (int i = 0; i < xt.dataLength; i++) {
                xt.data[i] = (float) ((xt.data[i] - (betas[timestep] * noisePred.data[i]) / b[timestep]) / Math.sqrt(alphas[timestep])) + sigma * noise[i];
            }
        } else {
            /**
             * mean
             */
            for (int i = 0; i < xt.dataLength; i++) {
                xt.data[i] = (float) ((xt.data[i] - (betas[timestep] * noisePred.data[i]) / b[timestep]) / Math.sqrt(alphas[timestep]));
            }
        }
        xt.hostToDevice();
    }
    
    public static void sample_prev_timestep(DiT network, Tensor condInput, Tensor xt, Tensor t, Tensor x0, int timestep, float[] a, float[] b, float[] betas, float[] alphas, float[] alphas_bar) {
        for (int i = 0; i < xt.number; i++) {
            t.data[i] = timestep;
        }
        t.hostToDevice();
        Tensor noisePred = network.forward(xt, t, condInput);
        noisePred.syncHost();
        JCuda.cudaDeviceSynchronize();
        if (timestep > 0) {
            System.err.println("timestep:" + timestep);
            float var = (1.0f - alphas_bar[timestep - 1]) / (1.0f - alphas_bar[timestep]) * betas[timestep];
            float sigma = (float) Math.pow(var, 0.5);
            float[] noise = RandomUtils.gaussianRandom(noisePred.dataLength, 1.0f);
            for (int i = 0; i < xt.dataLength; i++) {
                xt.data[i] = (float) ((xt.data[i] - (betas[timestep] * noisePred.data[i]) / b[timestep]) / Math.sqrt(alphas[timestep])) + sigma * noise[i];
            }
        } else {
            /**
             * mean
             */
            for (int i = 0; i < xt.dataLength; i++) {
                xt.data[i] = (float) ((xt.data[i] - (betas[timestep] * noisePred.data[i]) / b[timestep]) / Math.sqrt(alphas[timestep]));
            }
        }
        xt.hostToDevice();
    }

    public static void showImgs(String outputPath, Tensor input) {
        ImageUtils utils = new ImageUtils();
        if (input.isHasGPU()) {
            input.syncHost();
        }
        for (int b = 0; b < input.number; b++) {
            float[] once = input.getByNumber(b);
            //			once = MatrixOperation.add(once, 0.5f);
            utils.createRGBImage(outputPath + b + ".png", "png", ImageUtils.color2rgb2(once, input.channel, input.height, input.width, true), input.height, input.width, null, null);
        }
    }

    public static void showImgs(String outputPath, Tensor input, String it, float[] mean, float[] std) {
        ImageUtils utils = new ImageUtils();
        for (int b = 0; b < input.number; b++) {
            float[] once = input.getByNumber(b);
            utils.createRGBImage(outputPath + it + "_" + b + ".png", "png", ImageUtils.color2rgb2(once, input.channel, input.height, input.width, true, mean, std), input.height, input.width, null, null);
        }
    }

    public static void showImgs(String outputPath, Tensor input, String it, float[] mean, float[] std, String[] labels) {
        ImageUtils utils = new ImageUtils();
        if (labels != null) {
            for (int b = 0; b < input.number; b++) {
                float[] once = input.getByNumber(b);
                String title = labels[b];
                if (title.length() > 30) {
                    title = title.substring(0, 30);
                }
                utils.createRGBImage(outputPath + it + "_[" + title + "]" + b + ".png", "png", ImageUtils.color2rgb2(once, input.channel, input.height, input.width, true, mean, std), input.height, input.width, null, null);
            }
        } else {
            for (int b = 0; b < input.number; b++) {
                float[] once = input.getByNumber(b);
                utils.createRGBImage(outputPath + it + "_" + b + ".png", "png", ImageUtils.color2rgb2(once, input.channel, input.height, input.width, true, mean, std), input.height, input.width, null, null);
            }
        }
    }

    public static void showImgsLabel(String outputPath, Tensor input, String it, float[] mean, float[] std, String[] labels) {
        ImageUtils utils = new ImageUtils();
        for (int b = 0; b < input.number; b++) {
            float[] once = input.getByNumber(b);
            utils.createRGBImage(outputPath + it + "_[" + labels[b] + "]" + b + "_label.png", "png", ImageUtils.color2rgb2(once, input.channel, input.height, input.width, true, mean, std), input.height, input.width, null, null);
        }
    }

    public static void showImgs(String outputPath, Tensor input, String it) {
        ImageUtils utils = new ImageUtils();
        //		if(input.isHasGPU()) {
        //			input.syncHost();
        //		}
        for (int b = 0; b < input.number; b++) {
            float[] once = input.getByNumber(b);
            //			once = MatrixOperation.add(once, 0.5f);
            utils.createRGBImage(outputPath + it + "_" + b + ".png", "png", ImageUtils.color2rgb2(once, input.channel, input.height, input.width, true), input.height, input.width, null, null);
        }
    }

    public static void showImgs(String outputPath, Tensor input, float[] mean, float[] std) {
        ImageUtils utils = new ImageUtils();
        if (input.isHasGPU()) {
            input.syncHost();
        }
        for (int b = 0; b < input.number; b++) {
            float[] once = input.getByNumber(b);
            //			once = MatrixOperation.add(once, 0.5f);
            utils.createRGBImage(outputPath + b + ".png", "png", ImageUtils.color2rgb2(once, input.channel, input.height, input.width, true, mean, std), input.height, input.width, null, null);
        }
    }

    public static void testSD(String it, Tensor noiseInput, Tensor t, Tensor condInput, DiffusionUNetCond2 network, TinyVQVAE2 vae, String[] labels) {
        try {
            float beta_1 = 0.00085f;
            float beta_T = 0.012f;
            int T = 1000;
            //			float scale_factor = 0.143262f;
            float scale_factor = 0.18215f;
            float[] mean = new float[]{0.5f, 0.5f, 0.5f};
            float[] std = new float[]{0.5f, 0.5f, 0.5f};
            RandomUtils.gaussianRandom(noiseInput, 0, 1);
            //			noiseInput.data = RandomUtils.val(noiseInput.dataLength, 1.0f);
            //			noiseInput.hostToDevice();
            float[] betas = MatrixUtils.linspace(beta_1, beta_T, T);
            float[] alphas = MatrixOperation.subtraction(1, betas);
            float[] alphas_bar = MatrixUtils.cumprod(alphas);
            float[] sqrt_alphas_bar = MatrixOperation.sqrt(alphas_bar);
            float[] sqrt_one_minus_alphas_bar = MatrixOperation.sqrt(MatrixOperation.subtraction(1, alphas_bar));
            Tensor xt = noiseInput;
            for (int ts = T - 1; ts >= 0; ts--) {
                sample_prev_timestep(network, condInput, xt, t, null, ts, sqrt_alphas_bar, sqrt_one_minus_alphas_bar, betas, alphas, alphas_bar);
            }
            JCuda.cudaDeviceSynchronize();
            network.tensorOP.mul(xt, 1 / scale_factor, xt);
            //			Tensor result = vae.decodeCode(xt);
            //			xt.showDM("xt");
            Tensor result = vae.decode(xt);
            JCuda.cudaDeviceSynchronize();
            result.data = MatrixOperation.clampSelf(result.syncHost(), -1, 1);
            //			System.err.println("in");
            /**
             * print image

             */
            showImgs("H://vae_dataset//pokemon-blip//vqvae2//sd//", result, it, mean, std, labels);
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public static void testSD(String it, Tensor noiseInput, Tensor t, Tensor condInput, DiffusionUNetCond2 network, VQVAE2 vae, String[] labels) {
        try {
            float beta_1 = 0.00085f;
            float beta_T = 0.012f;
            int T = 1000;
            //			float scale_factor = 0.143262f;
            float scale_factor = 0.18215f;
            float[] mean = new float[]{0.5f, 0.5f, 0.5f};
            float[] std = new float[]{0.5f, 0.5f, 0.5f};
            RandomUtils.gaussianRandom(noiseInput, 0, 1);
            //			noiseInput.data = RandomUtils.val(noiseInput.dataLength, 1.0f);
            //			noiseInput.hostToDevice();
            float[] betas = MatrixUtils.linspace(beta_1, beta_T, T);
            float[] alphas = MatrixOperation.subtraction(1, betas);
            float[] alphas_bar = MatrixUtils.cumprod(alphas);
            float[] sqrt_alphas_bar = MatrixOperation.sqrt(alphas_bar);
            float[] sqrt_one_minus_alphas_bar = MatrixOperation.sqrt(MatrixOperation.subtraction(1, alphas_bar));
            Tensor xt = noiseInput;
            for (int ts = T - 1; ts >= 0; ts--) {
                sample_prev_timestep(network, condInput, xt, t, null, ts, sqrt_alphas_bar, sqrt_one_minus_alphas_bar, betas, alphas, alphas_bar);
            }
            JCuda.cudaDeviceSynchronize();
            network.tensorOP.mul(xt, 1.0f / scale_factor, xt);
            //			Tensor result = vae.decodeCode(xt);
            //			xt.showDM("xt");
            Tensor result = vae.decode(xt);
            JCuda.cudaDeviceSynchronize();
            result.data = MatrixOperation.clampSelf(result.syncHost(), -1, 1);
            //			System.err.println("in");
            /**
             * print image

             */
            showImgs("/omega/test/sd/", result, it, mean, std, labels);
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public static void testSD(String it, Tensor noiseInput, Tensor t, Tensor condInput, DiffusionUNetCond2 network, VQVAE2 vae, String[] labels, String outputPath) {
        try {
            float beta_1 = 0.00085f;
            float beta_T = 0.012f;
            int T = 1000;
            float scale_factor = 0.18215f;
            float[] mean = new float[]{0.5f, 0.5f, 0.5f};
            float[] std = new float[]{0.5f, 0.5f, 0.5f};
            RandomUtils.gaussianRandom(noiseInput, 0, 1);
            float[] betas = MatrixUtils.linspace(beta_1, beta_T, T);
            float[] alphas = MatrixOperation.subtraction(1, betas);
            float[] alphas_bar = MatrixUtils.cumprod(alphas);
            float[] sqrt_alphas_bar = MatrixOperation.sqrt(alphas_bar);
            float[] sqrt_one_minus_alphas_bar = MatrixOperation.sqrt(MatrixOperation.subtraction(1, alphas_bar));
            Tensor xt = noiseInput;
            for (int ts = T - 1; ts >= 0; ts--) {
                sample_prev_timestep(network, condInput, xt, t, null, ts, sqrt_alphas_bar, sqrt_one_minus_alphas_bar, betas, alphas, alphas_bar);
            }
            network.tensorOP.mul(xt, 1.0f / scale_factor, xt);
            Tensor result = vae.decode(xt);
            JCuda.cudaDeviceSynchronize();
            result.data = MatrixOperation.clampSelf(result.syncHost(), -1, 1);
            /**
             * print image

             */
            showImgs(outputPath, result, it, mean, std, labels);
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public static void testDiT(String it, Tensor noiseInput, Tensor t, Tensor condInput,Tensor cos,Tensor sin, DiT network, VQVAE2 vae, String[] labels) {
        try {
            float beta_1 = 0.00085f;
            float beta_T = 0.012f;
            int T = 1000;
            //			float scale_factor = 0.143262f;
            float scale_factor = 0.18215f;
            float[] mean = new float[]{0.5f, 0.5f, 0.5f};
            float[] std = new float[]{0.5f, 0.5f, 0.5f};
            RandomUtils.gaussianRandom(noiseInput, 0, 1);
            //			noiseInput.data = RandomUtils.val(noiseInput.dataLength, 1.0f);
            //			noiseInput.hostToDevice();
            float[] betas = MatrixUtils.linspace(beta_1, beta_T, T);
            float[] alphas = MatrixOperation.subtraction(1, betas);
            float[] alphas_bar = MatrixUtils.cumprod(alphas);
            float[] sqrt_alphas_bar = MatrixOperation.sqrt(alphas_bar);
            float[] sqrt_one_minus_alphas_bar = MatrixOperation.sqrt(MatrixOperation.subtraction(1, alphas_bar));
            Tensor xt = noiseInput;
            for (int ts = T - 1; ts >= 0; ts--) {
                sample_prev_timestep(network, condInput, xt, t, cos, sin, null, ts, sqrt_alphas_bar, sqrt_one_minus_alphas_bar, betas, alphas, alphas_bar);
            }
            JCuda.cudaDeviceSynchronize();
            network.tensorOP.mul(xt, 1.0f / scale_factor, xt);
            //			Tensor result = vae.decodeCode(xt);
            //			xt.showDM("xt");
            Tensor result = vae.decode(xt);
            JCuda.cudaDeviceSynchronize();
            result.data = MatrixOperation.clampSelf(result.syncHost(), -1, 1);
            //			System.err.println("in");
            /**
             * print image
             */
            showImgs("/omega/test/sd/", result, it, mean, std, labels);
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
    
    public static void testDiT(String it, Tensor noiseInput, Tensor t, Tensor condInput, DiT network, VQVAE2 vae, String[] labels) {
        try {
            float beta_1 = 0.00085f;
            float beta_T = 0.012f;
            int T = 1000;
            //			float scale_factor = 0.143262f;
            float scale_factor = 0.18215f;
            float[] mean = new float[]{0.5f, 0.5f, 0.5f};
            float[] std = new float[]{0.5f, 0.5f, 0.5f};
            RandomUtils.gaussianRandom(noiseInput, 0, 1);
            //			noiseInput.data = RandomUtils.val(noiseInput.dataLength, 1.0f);
            //			noiseInput.hostToDevice();
            float[] betas = MatrixUtils.linspace(beta_1, beta_T, T);
            float[] alphas = MatrixOperation.subtraction(1, betas);
            float[] alphas_bar = MatrixUtils.cumprod(alphas);
            float[] sqrt_alphas_bar = MatrixOperation.sqrt(alphas_bar);
            float[] sqrt_one_minus_alphas_bar = MatrixOperation.sqrt(MatrixOperation.subtraction(1, alphas_bar));
            Tensor xt = noiseInput;
            for (int ts = T - 1; ts >= 0; ts--) {
                sample_prev_timestep(network, condInput, xt, t, null, ts, sqrt_alphas_bar, sqrt_one_minus_alphas_bar, betas, alphas, alphas_bar);
            }
            JCuda.cudaDeviceSynchronize();
            network.tensorOP.mul(xt, 1.0f / scale_factor, xt);
            //			Tensor result = vae.decodeCode(xt);
            //			xt.showDM("xt");
            Tensor result = vae.decode(xt);
            JCuda.cudaDeviceSynchronize();
            result.data = MatrixOperation.clampSelf(result.syncHost(), -1, 1);
            //			System.err.println("in");
            /**
             * print image
             */
            showImgs("H:\\vae_dataset\\anime_test256\\dit_test2\\", result, it, mean, std, labels);
//            showImgs("/omega/test/sd/", result, it, mean, std, labels);
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
    
    public static String output2TXT(Tensor output, RNNDataLoader trainData) {
        String txt = "";
        //		output.showDMByNumber(0);
        OneHotDataLoader tr = (OneHotDataLoader) trainData;
        for (int i = 0; i < output.number; i++) {
            int charIndex = pickTopN(output.getByNumber(i), 1);
            char c = tr.dictionaryData[charIndex];
            txt += c;
        }
        return txt;
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

    public YoloLabelUtils dataEnhanceInstance() {
        if (u == null) {
            u = new YoloLabelUtils(1, 4);
        }
        return u;
    }

    @Override
    public void train(BaseData trainingData) {
        // TODO Auto-generated method stub
        try {
            //			CUDAModules.initCUDAFunctions();
            this.dataSize = trainingData.number;
            if (isWarmUp()) {
                this.network.learnRate = (float) (this.lr * Math.pow(batchIndex * 1.0f / burnIn * 1.0f, power));
            }
            Tensor input = new Tensor(batchSize, this.network.getChannel(), this.network.getHeight(), this.network.getWidth(), true);
            Tensor label = new Tensor(batchSize, 1, 1, trainingData.labelSize, true);
            for (int i = 0; i < this.trainTime; i++) {
                if (this.trainIndex >= this.minTrainTime) {
                    break;
                }
                this.trainIndex = i + 1;
                int[][] indexs = MathUtils.randomInts(trainingData.number, this.batchSize);
                /**
                 * 遍历整个训练集

                 */
                for (int it = 0; it < indexs.length; it++) {
                    //				for(int it = 0;it<1;it++) {
                    if (Math.abs(this.currentError) <= this.error) {
                        break;
                    }
                    long start = System.nanoTime();
                    this.loss.clear();
                    this.lossDiff.clear();
                    trainingData.getRandomData(indexs[it], input, label);
                    input.hostToDevice();
                    label.hostToDevice();
                    //					input.showDM();
                    //					long output_start = System.nanoTime();
                    /**
                     * forward

                     */
                    Tensor output = this.network.forward(input);
                    //					System.out.println(JsonUtils.toJson(output.data));
                    //					System.out.println("output1:"+(System.nanoTime() - output_start) / 1e6 + "ms.");
                    //					output.syncHost();
                    //					System.out.println(JsonUtils.toJson(output.data));
                    //					System.out.println("output2:"+(System.nanoTime() - output_start) / 1e6 + "ms.");
                    /**
                     * loss

                     */
                    this.loss = this.network.loss(output, label);
                    /**
                     * loss diff

                     */
                    this.lossDiff = this.network.lossDiff(output, label);
                    //					System.out.println("=========>:"+JsonUtils.toJson(lossDiff.data));
                    /**
                     * current time error

                     */
                    if (this.loss.isHasGPU()) {
                        this.currentError = MatrixOperation.sum(this.loss.syncHost()) / this.batchSize;
                    } else {
                        this.currentError = MatrixOperation.sum(this.loss.data) / this.batchSize;
                    }
                    //					long back_start = System.nanoTime();
                    lossDiff.hostToDevice();
                    /**
                     * back

                     */
                    this.network.back(this.lossDiff);
                    /**
                     * update

                     */
                    this.network.update();
                    output.syncHost();
                    //					System.out.println("back:"+(System.nanoTime() - back_start) / 1e6 + "ms.");
                    float error = this.accuracy(output, label, trainingData.labelSet);
                    String msg = "training[" + this.trainIndex + "]{" + it + "} (lr:" + this.network.learnRate + ") accuracy:{" + error + "%} currentError:" + this.currentError + " [costTime:" + (System.nanoTime() - start) / 1e6 + "ms.]";
                    System.out.println(msg);
                    //					/**
                    //					 * update learning rate
                    //					 */
                    //					this.updateLR();
                    this.batchIndex++;
                }
                /**
                 * update learning rate

                 */
                this.updateLR(this.lr_step);
            }
            /**
             * 停止训练

             */
            System.out.println("training finish. [" + this.trainIndex + "] finalError:" + this.currentError);
            //			System.out.println(JsonUtils.toJson(this.network.layerList));
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    @Override
    public void train(BaseData trainingData, BaseData testData) {
        // TODO Auto-generated method stub
        try {
            CUDAModules.initCUDAFunctions();
            this.dataSize = trainingData.number;
            if (isWarmUp()) {
                this.network.learnRate = (float) (this.lr * Math.pow(batchIndex * 1.0f / burnIn * 1.0f, power));
            }
            Tensor input = new Tensor(batchSize, this.network.getChannel(), this.network.getHeight(), this.network.getWidth(), true);
            Tensor label = new Tensor(batchSize, 1, 1, trainingData.labelSize, true);
            for (int i = 0; i < this.trainTime; i++) {
                if (this.trainIndex >= this.minTrainTime) {
                    break;
                }
                this.trainIndex = i + 1;
                int[][] indexs = MathUtils.randomInts(trainingData.number, this.batchSize);
                this.network.RUN_MODEL = RunModel.TRAIN;
                /**
                 * 遍历整个训练集

                 */
                for (int it = 0; it < indexs.length; it++) {
                    if (Math.abs(this.currentError) <= this.error) {
                        break;
                    }
                    long start = System.nanoTime();
                    //					this.loss.clear();
                    //
                    //					this.lossDiff.clear();
                    trainingData.getRandomData(indexs[it], input, label);
                    input.hostToDevice();
                    label.hostToDevice();
                    //					input.showDM();
                    //					long output_start = System.nanoTime();
                    /**
                     * forward

                     */
                    Tensor output = this.network.forward(input);
                    //					System.out.println(JsonUtils.toJson(output.data));
                    //					System.out.println("output1:"+(System.nanoTime() - output_start) / 1e6 + "ms.");
                    //					System.out.println(JsonUtils.toJson(output.data));
                    //					System.out.println("output2:"+(System.nanoTime() - output_start) / 1e6 + "ms.");
                    /**
                     * loss

                     */
                    this.loss = this.network.loss(output, label);
                    /**
                     * loss diff

                     */
                    this.lossDiff = this.network.lossDiff(output, label);
                    //					System.out.println(JsonUtils.toJson(label.syncHost()));
                    //
                    //					System.out.println(JsonUtils.toJson(output.syncHost()));
                    //
                    //					System.out.println(JsonUtils.toJson(this.lossDiff.syncHost()));
                    //					System.out.println("=========>:"+JsonUtils.toJson(lossDiff.data));
                    //					long back_start = System.nanoTime();
                    //					loss.hostToDevice();
                    //					lossDiff.hostToDevice();
                    /**
                     * back

                     */
                    this.network.back(this.lossDiff);
                    /**
                     * update

                     */
                    this.network.update();
                    JCudaDriver.cuCtxSynchronize();
                    //					System.out.println("back:"+(System.nanoTime() - back_start) / 1e6 + "ms.");
                    output.syncHost();
                    float error = this.accuracy(output, label, trainingData.labelSet);
                    /**
                     * current time error

                     */
                    this.currentError = MatrixOperation.sum(this.loss.syncHost()) / this.batchSize;
                    String msg = "training[" + this.trainIndex + "]{" + it + "} (lr:" + this.network.learnRate + ") accuracy:{" + error + "%} currentError:" + this.currentError + " [costTime:" + (System.nanoTime() - start) / 1e6 + "ms.]";
                    System.out.println(msg);
                    //					/**
                    //					 * update learning rate
                    //					 */
                    //					this.updateLR();
                    this.batchIndex++;
                }
                /**
                 * update learning rate

                 */
                this.updateLR(this.lr_step);
                /**
                 * vail data test

                 */
                this.test(testData, this.batchSize);
            }
            /**
             * 停止训练

             */
            System.out.println("training finish. [" + this.trainIndex + "] finalError:" + this.currentError);
            //			System.out.println(JsonUtils.toJson(this.network.layerList));
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    @Override
    public void train(BaseData trainingData, BaseData validata, BaseData testData) {
        // TODO Auto-generated method stub
    }

    public void train(BaseData trainingData, BaseData validata, float[] mean, float[] std) {
        // TODO Auto-generated method stub
        try {
            //			CUDAModules.initCUDAFunctions();
            this.dataSize = trainingData.number;
            if (isWarmUp()) {
                this.network.learnRate = (float) (this.lr * Math.pow(batchIndex * 1.0f / burnIn * 1.0f, power));
            }
            Tensor input = new Tensor(batchSize, this.network.getChannel(), this.network.getHeight(), this.network.getWidth(), true);
            Tensor label = new Tensor(batchSize, 1, 1, trainingData.labelSize, true);
            Tensor transData = new Tensor(trainingData.number, trainingData.channel, trainingData.height, trainingData.width);
            Tensor vail_input = new Tensor(batchSize, validata.channel, validata.height, validata.width, true);
            Tensor vail_label = new Tensor(batchSize, 1, 1, validata.labelSize, true);
            for (int i = 0; i < this.trainTime; i++) {
                if (this.trainIndex >= this.minTrainTime) {
                    break;
                }
                transforms(trainingData.input, transData, mean, std);
                this.trainIndex = i + 1;
                int[][] indexs = MathUtils.randomInts(trainingData.number, this.batchSize);
                this.network.RUN_MODEL = RunModel.TRAIN;
                float train_loss = 0.0f;
                /**
                 * 遍历整个训练集

                 */
                for (int it = 0; it < indexs.length; it++) {
                    long start = System.nanoTime();
                    if (Math.abs(this.currentError) <= this.error) {
                        break;
                    }
                    trainingData.randomData(indexs[it], transData.data, input, label);
                    input.hostToDevice();
                    label.hostToDevice();
                    /**
                     * forward

                     */
                    Tensor output = this.network.forward(input);
                    /**
                     * loss

                     */
                    this.loss = this.network.loss(output, label);
                    /**
                     * loss diff

                     */
                    this.lossDiff = this.network.lossDiff(output, label);
                    /**
                     * back

                     */
                    this.network.back(this.lossDiff);
                    /**
                     * update

                     */
                    this.network.update();
                    output.syncHost();
                    float error = this.accuracy(output, label, trainingData.labelSet);
                    /**
                     * current time error

                     */
                    if (this.loss.isHasGPU()) {
                        this.currentError = MatrixOperation.sum(this.loss.syncHost()) / this.batchSize;
                    } else {
                        this.currentError = MatrixOperation.sum(this.loss.data) / this.batchSize;
                    }
                    train_loss += this.currentError;
                    String msg = "training[" + this.trainIndex + "]{" + it + "} (lr:" + this.network.learnRate + ") accuracy:{" + error + "%} train_loss:" + this.currentError + " [costTime:" + (System.nanoTime() - start) / 1e6 + "ms.]";
                    System.out.println(msg);
                    this.batchIndex++;
                }
                System.out.println("training[" + this.trainIndex + "] train loss:{" + train_loss / indexs.length + "} ");
                /**
                 * vail data test

                 */
                float vail_loss = this.testAndLoss(validata, vail_input, vail_label, this.batchSize);
                /**
                 * update learning rate

                 */
                this.updateLR(vail_loss);
            }
            /**
             * 停止训练

             */
            System.out.println("training finish. [" + this.trainIndex + "] finalError:" + this.currentError);
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void train(BaseDataLoader trainingData) {
        // TODO Auto-generated method stub
        try {
            CUDAModules.initCUDAFunctions();
            this.dataSize = trainingData.number;
            //			/**
            //			 * normalize vailSet
            //			 */
            //			DataTransforms.normalize(validata.input, mean, std);
            if (isWarmUp()) {
                this.network.learnRate = (float) (this.lr * Math.pow(batchIndex * 1.0f / burnIn * 1.0f, power));
            }
            Tensor input = new Tensor(batchSize, this.network.getChannel(), this.network.getHeight(), this.network.getWidth(), true);
            Tensor label = new Tensor(batchSize, 1, 1, trainingData.labelSize, true);
            for (int i = 0; i < this.trainTime; i++) {
                if (this.trainIndex >= this.minTrainTime) {
                    break;
                }
                this.trainIndex = i + 1;
                int[][] indexs = trainingData.shuffle();
                this.network.RUN_MODEL = RunModel.TRAIN;
                float train_loss = 0.0f;
                /**
                 * 遍历整个训练集

                 */
                for (int it = 0; it < indexs.length; it++) {
                    long start = System.nanoTime();
                    if (Math.abs(this.currentError) <= this.error) {
                        break;
                    }
                    trainingData.loadData(indexs[it], input, label);
                    //					System.out.println(JsonUtils.toJson(label.data));
                    //					input.hostToDevice();
                    //
                    //					label.hostToDevice();
                    /**
                     * forward

                     */
                    Tensor output = this.network.forward(input);
                    //					System.out.println(JsonUtils.toJson(output.syncHost()));
                    /**
                     * loss

                     */
                    this.loss = this.network.loss(output, label);
                    /**
                     * loss diff

                     */
                    this.lossDiff = this.network.lossDiff(output, label);
                    /**
                     * back

                     */
                    this.network.back(this.lossDiff);
                    /**
                     * update

                     */
                    this.network.update();
                    //					output.syncHost();
                    //					float error = this.accuracy(output, label, trainingData.labelSet);
                    JCudaDriver.cuCtxSynchronize();
                    /**
                     * current time error

                     */
                    if (this.loss.isHasGPU()) {
                        this.currentError = MatrixOperation.sum(this.loss.syncHost()) / this.batchSize;
                    } else {
                        this.currentError = MatrixOperation.sum(this.loss.data) / this.batchSize;
                    }
                    train_loss += this.currentError;
                    String msg = "training[" + this.trainIndex + "]{" + it + "} (lr:" + this.network.learnRate + ") accuracy:{" + error + "%} train_loss:" + this.currentError + " [costTime:" + (System.nanoTime() - start) / 1e6 + "ms.]";
                    System.out.println(msg);
                    this.batchIndex++;
                }
                System.out.println("training[" + this.trainIndex + "] train loss:{" + train_loss / indexs.length + "} ");
                //				/**
                //				 * vail data test
                //				 */
                //				float vail_loss = this.testAndLoss(validata, vail_input, vail_label, this.batchSize);
                /**
                 * update learning rate

                 */
                this.updateLR(this.lr_step);
            }
            /**
             * 停止训练

             */
            System.out.println("training finish. [" + this.trainIndex + "] finalError:" + this.currentError);
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void train(BaseDataLoader trainingData, BaseDataLoader valiData, BaseCheck check) {
        // TODO Auto-generated method stub
        try {
            CUDAModules.initCUDAFunctions();
            this.dataSize = trainingData.number;
            if (isWarmUp()) {
                this.network.learnRate = (float) (this.lr * Math.pow(batchIndex * 1.0f / burnIn * 1.0f, power));
            }
            Tensor input = new Tensor(batchSize, this.network.getChannel(), this.network.getHeight(), this.network.getWidth(), true);
            Tensor label = trainingData.initLabelTensor();
            for (int i = 0; i < this.trainTime; i++) {
                if (this.trainIndex >= this.minTrainTime) {
                    break;
                }
                this.network.RUN_MODEL = RunModel.TRAIN;
                this.trainIndex = i + 1;
                int[][] indexs = trainingData.shuffle();
                /**
                 * 遍历整个训练集

                 */
                for (int it = 0; it < indexs.length; it++) {
                    long start = System.nanoTime();
                    this.loss.clear();
                    this.lossDiff.clear();
                    /**
                     * 读取训练数据

                     */
                    trainingData.loadData(indexs[it], input, label);
                    /**
                     * forward

                     */
                    Tensor output = network.forward(input);
                    /**
                     * loss

                     */
                    Tensor loss = this.network.loss(output, label);
                    /**
                     * loss diff

                     */
                    this.lossDiff = network.lossDiff(output, label);
                    /**
                     * back

                     */
                    network.back(lossDiff);
                    /**
                     * update

                     */
                    this.network.update();
                    if (loss.isHasGPU()) {
                        loss.syncHost();
                    }
                    float accuracy = check.check(output, label, trainingData.labelSet, false);
                    String msg = "training[" + this.trainIndex + "]{" + it + "} (lr:" + this.network.learnRate + ") (loss:" + loss.getByIndex(0, 0, 0, 0) + ") (accuracy:" + accuracy / batchSize * 100 + "%) [costTime:" + (System.nanoTime() - start) / 1e6 + "ms.]";
                    System.out.println(msg);
                    this.batchIndex++;
                }
                /**
                 * update learning rate

                 */
                this.updateLR(this.lr_step);
                if (this.trainIndex % 10 == 0) {
                    System.out.println("----------------testing start----------------");
                    this.testAndLoss(valiData, input, label, this.batchSize, check);
                    System.out.println("----------------testing finish---------------");
                }
            }
            /**
             * 停止训练

             */
            System.out.println("training finish. [" + this.trainIndex + "] finalError:" + this.currentError);
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void trainObjectRecognition(BaseData trainingData) {
        // TODO Auto-generated method stub
        try {
            CUDAModules.initCUDAFunctions();
            this.dataSize = trainingData.number;
            if (isWarmUp()) {
                this.network.learnRate = (float) (this.lr * Math.pow(batchIndex * 1.0f / burnIn * 1.0f, power));
            }
            Tensor input = new Tensor(batchSize, this.network.getChannel(), this.network.getHeight(), this.network.getWidth(), true);
            Tensor label = new Tensor(batchSize, 1, 1, trainingData.labelSize, true);
            for (int i = 0; i < this.trainTime; i++) {
                if (this.trainIndex >= this.minTrainTime) {
                    break;
                }
                this.trainIndex = i + 1;
                int[][] indexs = MathUtils.randomInts(trainingData.number, this.batchSize);
                /**
                 * 遍历整个训练集

                 */
                for (int it = 0; it < indexs.length; it++) {
                    long start = System.nanoTime();
                    this.loss.clear();
                    this.lossDiff.clear();
                    trainingData.getRandomData(indexs[it], input, label);
                    input.hostToDevice();
                    label.hostToDevice();
                    //					input.showDM();
                    //					long output_start = System.nanoTime();
                    /**
                     * forward

                     */
                    Tensor output = this.network.forward(input);
                    //					System.out.println(JsonUtils.toJson(output.data));
                    //					System.out.println("output1:"+(System.nanoTime() - output_start) / 1e6 + "ms.");
                    //					output.syncHost();
                    //					System.out.println(JsonUtils.toJson(output.data));
                    //					System.out.println("output2:"+(System.nanoTime() - output_start) / 1e6 + "ms.");
                    /**
                     * loss

                     */
                    this.loss = this.network.loss(output, label);
                    /**
                     * loss diff

                     */
                    this.lossDiff = this.network.lossDiff(output, label);
                    //					System.out.println("=========>:"+JsonUtils.toJson(lossDiff.data));
                    //					long back_start = System.nanoTime();
                    /**
                     * back

                     */
                    this.network.back(this.lossDiff);
                    /**
                     * update

                     */
                    this.network.update();
                    //					JCudaDriver.cuCtxSynchronize();
                    /**
                     * current time error

                     */
                    if (this.loss.isHasGPU()) {
                        this.currentError = MatrixOperation.sum(this.loss.syncHost()) / this.batchSize;
                    } else {
                        this.currentError = MatrixOperation.sum(this.loss.data) / this.batchSize;
                    }
                    //					System.out.println("back:"+(System.nanoTime() - back_start) / 1e6 + "ms.");
                    //					float error = 0.0f;
                    //
                    String msg = "training[" + this.trainIndex + "]{" + it + "} (lr:" + this.network.learnRate + ") train_loss:" + this.currentError + " [costTime:" + (System.nanoTime() - start) / 1e6 + "ms.]";
                    System.out.println(msg);
                    this.batchIndex++;
                }
                /**
                 * update learning rate

                 */
                this.updateLR(this.lr_step);
            }
            /**
             * 停止训练

             */
            System.out.println("training finish. [" + this.trainIndex + "] finalError:" + this.currentError);
            //			System.out.println(JsonUtils.toJson(this.network.layerList));
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void trainObjectRecognition(BaseData trainingData, BaseData validata) {
        // TODO Auto-generated method stub
        try {
            CUDAModules.initCUDAFunctions();
            this.dataSize = trainingData.number;
            if (isWarmUp()) {
                this.network.learnRate = (float) (this.lr * Math.pow(batchIndex * 1.0f / burnIn * 1.0f, power));
            }
            Tensor input = new Tensor(batchSize, this.network.getChannel(), this.network.getHeight(), this.network.getWidth(), true);
            Tensor label = new Tensor(batchSize, 1, 1, trainingData.labelSize, true);
            Tensor vail_input = new Tensor(batchSize, validata.channel, validata.height, validata.width, true);
            Tensor vail_label = new Tensor(batchSize, 1, 1, validata.labelSize, true);
            for (int i = 0; i < this.trainTime; i++) {
                if (this.trainIndex >= this.minTrainTime) {
                    break;
                }
                this.network.RUN_MODEL = RunModel.TRAIN;
                this.trainIndex = i + 1;
                int[][] indexs = MathUtils.randomInts(trainingData.number, this.batchSize);
                /**
                 * 遍历整个训练集

                 */
                for (int it = 0; it < indexs.length; it++) {
                    long start = System.nanoTime();
                    this.loss.clear();
                    this.lossDiff.clear();
                    trainingData.getRandomData(indexs[it], input, label);
                    input.hostToDevice();
                    label.hostToDevice();
                    /**
                     * forward

                     */
                    Tensor output = this.network.forward(input);
                    /**
                     * loss

                     */
                    this.loss = this.network.loss(output, label);
                    /**
                     * loss diff

                     */
                    this.lossDiff = this.network.lossDiff(output, label);
                    /**
                     * back

                     */
                    this.network.back(this.lossDiff);
                    /**
                     * update

                     */
                    this.network.update();
                    /**
                     * current time error

                     */
                    if (this.loss.isHasGPU()) {
                        this.currentError = MatrixOperation.sum(this.loss.syncHost()) / this.batchSize;
                    } else {
                        this.currentError = MatrixOperation.sum(this.loss.data) / this.batchSize;
                    }
                    //
                    String msg = "training[" + this.trainIndex + "]{" + it + "} (lr:" + this.network.learnRate + ") train_loss:" + this.currentError + " [costTime:" + (System.nanoTime() - start) / 1e6 + "ms.]";
                    System.out.println(msg);
                    this.batchIndex++;
                }
                /**
                 * update learning rate

                 */
                this.updateLR(this.lr_step);
                if (this.trainIndex % 100 == 0) {
                    System.out.println("----------------testing start----------------");
                    this.testObjectRecognition(validata, vail_input, vail_label, this.batchSize);
                    System.out.println("----------------testing finish---------------");
                }
            }
            /**
             * 停止训练

             */
            System.out.println("training finish. [" + this.trainIndex + "] finalError:" + this.currentError);
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    //	public static void main(String[] args) {
    //
    //		try {
    //
    //			int N = 2;
    //			int C = 3;
    //			int H = 4;
    //			int W = 4;
    //			int T = 10;
    //			float beta_1 = 1e-4f;
    //			float beta_T = 0.02f;
    //
    //			Tensor noiseInput = new Tensor(N, C, H, W, true);
    //
    //			Tensor t = new Tensor(N, 1, 1, 1, true);
    //
    //			testGaussianDiffusion(noiseInput, t, T, beta_1, beta_T);
    //
    //		} catch (Exception e) {
    //			// TODO: handle exception
    //			e.printStackTrace();
    //		}
    //
    //	}
    //	public void testGaussianDiffusion(String it,int ddim_timesteps,Tensor noiseInput,Tensor noise) {
    //
    //		try {
    //
    //			DuffsionUNet network = (DuffsionUNet) this.network;
    //
    //			float beta_1 = 1e-4f;
    //			float beta_T = 0.02f;
    //			int T = 1000;
    //			float[] mean = new float[] {0.5f, 0.5f, 0.5f};
    //			float[] std = new float[] {0.5f, 0.5f, 0.5f};
    //
    ////			Tensor noiseInput = new Tensor(batchSize, this.network.getChannel(), this.network.getHeight(), this.network.getWidth(), true);
    //
    //			Tensor t = new Tensor(batchSize, 1, 1, 1, true);
    //
    ////			RandomUtils.gaussianRandom(noiseInput);
    //
    //			float[] betas = MatrixUtils.linspace(beta_1, beta_T, T);
    //			float[] alphas = MatrixOperation.subtraction(1, betas);
    //			float[] alphas_bar = MatrixUtils.cumprod(alphas);
    //
    //			int step = T / ddim_timesteps;
    //
    //			float[] ddim_timestep_seq = MatrixUtils.range(0, T, step, 1);
    //
    //			float[] ddim_timestep_prev_seq = new float[ddim_timestep_seq.length];
    //
    //			for(int i = 1;i<ddim_timestep_seq.length;i++) {
    //				ddim_timestep_prev_seq[i] = ddim_timestep_seq[i - 1];
    //			}
    //			int[] t_data = new int[batchSize];
    //			int[] prev_t_data = new int[batchSize];
    //			for(int timestep = ddim_timesteps - 1;timestep>=0;timestep--) {
    //				for(int i = 0;i<batchSize;i++) {
    //					t_data[i] = (int) ddim_timestep_seq[timestep];
    //					prev_t_data[i] = (int) ddim_timestep_prev_seq[timestep];
    //				}
    //				t.setData(t_data);
    //
    //				Tensor eps = noise;
    ////				eps.showDMByOffset(0, 100);
    //				float[] exsa1 = MatrixUtils.gather(alphas_bar, t_data);
    //
    //				float[] exsa2 = MatrixUtils.gather(alphas_bar, prev_t_data);
    //
    //				prev_mean_from_eps(noiseInput, eps, exsa1, exsa2, 1, timestep);
    //
    //				noiseInput.hostToDevice();
    //
    //				if(timestep == 100) {
    //					MatrixOperation.clampSelf(noiseInput.data, -1, 1);
    //
    //					/**
    //					 * print image
    //					 */
    //					showImgs("H:\\voc\\gan_anime\\duffsion_test\\", noiseInput, it+"_100");
    //				}
    //
    //			}
    //
    //			MatrixOperation.clampSelf(noiseInput.data, -1, 1);
    //
    //			/**
    //			 * print image
    //			 */
    //			showImgs("H:\\voc\\gan_anime\\duffsion_test\\", noiseInput, it);
    //
    //		} catch (Exception e) {
    //			// TODO: handle exception
    //			e.printStackTrace();
    //		}
    //
    //	}
    public void trainObjectRecognition(BaseData trainingData, BaseData validata, boolean dataEnhance) {
        // TODO Auto-generated method stub
        try {
            CUDAModules.initCUDAFunctions();
            this.dataSize = trainingData.number;
            if (isWarmUp()) {
                this.network.learnRate = (float) (this.lr * Math.pow(batchIndex * 1.0f / burnIn * 1.0f, power));
            }
            Tensor input = new Tensor(batchSize, this.network.getChannel(), this.network.getHeight(), this.network.getWidth(), true);
            Tensor label = new Tensor(batchSize, 1, 1, trainingData.labelSize, true);
            Tensor vail_input = new Tensor(batchSize, validata.channel, validata.height, validata.width, true);
            for (int i = 0; i < this.trainTime; i++) {
                if (this.trainIndex >= this.minTrainTime) {
                    break;
                }
                this.network.RUN_MODEL = RunModel.TRAIN;
                this.trainIndex = i + 1;
                int[][] indexs = MathUtils.randomInts(trainingData.number, this.batchSize);
                /**
                 * 遍历整个训练集

                 */
                for (int it = 0; it < indexs.length; it++) {
                    long start = System.nanoTime();
                    this.loss.clear();
                    this.lossDiff.clear();
                    trainingData.getRandomData(indexs[it], input, label);
                    /**
                     * 数据增强

                     */
                    if (dataEnhance) {
                        dataEnhanceInstance().transforms(input, label);
                        YoloLabelUtils.formatToYolo(label, input.height, input.width);
                    }
                    input.hostToDevice();
                    label.hostToDevice();
                    /**
                     * forward

                     */
                    Tensor output = this.network.forward(input);
                    /**
                     * loss

                     */
                    this.loss = this.network.loss(output, label);
                    /**
                     * loss diff

                     */
                    this.lossDiff = this.network.lossDiff(output, label);
                    /**
                     * back

                     */
                    this.network.back(this.lossDiff);
                    /**
                     * update

                     */
                    this.network.update();
                    /**
                     * current time error

                     */
                    if (this.loss.isHasGPU()) {
                        this.currentError = MatrixOperation.sum(this.loss.syncHost()) / this.batchSize;
                    } else {
                        this.currentError = MatrixOperation.sum(this.loss.data) / this.batchSize;
                    }
                    //
                    String msg = "training[" + this.trainIndex + "]{" + it + "} (lr:" + this.network.learnRate + ") train_loss:" + this.currentError + " [costTime:" + (System.nanoTime() - start) / 1e6 + "ms.]";
                    System.out.println(msg);
                    this.batchIndex++;
                }
                /**
                 * update learning rate

                 */
                this.updateLR(this.lr_step);
                if (this.trainIndex % 100 == 0) {
                    System.out.println("----------------testing start----------------");
                    this.testObjectRecognition(validata, vail_input, label, this.batchSize);
                    System.out.println("----------------testing finish---------------");
                }
            }
            /**
             * 停止训练

             */
            System.out.println("training finish. [" + this.trainIndex + "] finalError:" + this.currentError);
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void trainObjectRecognition(DetectionDataLoader trainingData, DetectionDataLoader valiData) {
        // TODO Auto-generated method stub
        try {
            CUDAModules.initCUDAFunctions();
            this.dataSize = trainingData.number;
            Yolo network = (Yolo) this.network;
            if (isWarmUp()) {
                this.network.learnRate = (float) (this.lr * Math.pow(batchIndex * 1.0f / burnIn * 1.0f, power));
            }
            Tensor input = new Tensor(batchSize, this.network.getChannel(), this.network.getHeight(), this.network.getWidth(), true);
            Tensor label = trainingData.initLabelTensor();
            for (int i = 0; i < this.trainTime; i++) {
                if (this.trainIndex >= this.minTrainTime) {
                    break;
                }
                this.network.RUN_MODEL = RunModel.TRAIN;
                this.trainIndex = i + 1;
                int[][] indexs = trainingData.shuffle();
                /**
                 * 遍历整个训练集

                 */
                for (int it = 0; it < indexs.length; it++) {
                    long start = System.nanoTime();
                    this.loss.clear();
                    this.lossDiff.clear();
                    /**
                     * 读取训练数据

                     */
                    trainingData.loadData(indexs[it], input, label);
                    /**
                     * forward

                     */
                    Tensor output = network.forward(input);
                    /**
                     * loss

                     */
                    this.network.loss(output, label);
                    /**
                     * loss diff

                     */
                    this.lossDiff = network.lossDiff(output, label);
                    /**
                     * back

                     */
                    network.back(lossDiff);
                    /**
                     * update

                     */
                    this.network.update();
                    String msg = "training[" + this.trainIndex + "]{" + it + "} (lr:" + this.network.learnRate + ") [costTime:" + (System.nanoTime() - start) / 1e6 + "ms.]";
                    System.out.println(msg);
                    this.batchIndex++;
                }
                /**
                 * update learning rate

                 */
                this.updateLR(this.lr_step);
                if (this.trainIndex % 100 == 0) {
                    System.out.println("----------------testing start----------------");
                    this.testObjectRecognition(valiData, input, label, this.batchSize);
                    System.out.println("----------------testing finish---------------");
                }
            }
            /**
             * 停止训练

             */
            System.out.println("training finish. [" + this.trainIndex + "] finalError:" + this.currentError);
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void trainObjectRecognitionOutputs(BaseData trainingData, BaseData valiData, boolean dataEnhance) {
        // TODO Auto-generated method stub
        try {
            CUDAModules.initCUDAFunctions();
            OutputsNetwork network = (OutputsNetwork) this.network;
            this.dataSize = trainingData.number;
            if (isWarmUp()) {
                this.network.learnRate = (float) (this.lr * Math.pow(batchIndex * 1.0f / burnIn * 1.0f, power));
            }
            Tensor input = new Tensor(batchSize, this.network.getChannel(), this.network.getHeight(), this.network.getWidth(), true);
            Tensor label = new Tensor(batchSize, 1, 1, trainingData.labelSize);
            Tensor vail_input = new Tensor(batchSize, this.network.getChannel(), this.network.getHeight(), this.network.getWidth(), true);
            for (int i = 0; i < this.trainTime; i++) {
                if (this.trainIndex >= this.minTrainTime) {
                    break;
                }
                this.network.RUN_MODEL = RunModel.TRAIN;
                this.trainIndex = i + 1;
                int[][] indexs = MathUtils.randomInts(trainingData.number, this.batchSize);
                /**
                 * 遍历整个训练集

                 */
                for (int it = 0; it < indexs.length; it++) {
                    long start = System.nanoTime();
                    this.loss.clear();
                    this.lossDiff.clear();
                    trainingData.getRandomData(indexs[it], input, label);
                    /**
                     * 数据增强

                     */
                    if (dataEnhance) {
                        dataEnhanceInstance().transforms(input, label);
                        YoloLabelUtils.formatToYoloV3(label, input.height, input.width);
                    }
                    input.hostToDevice();
                    label.hostToDevice();
                    /**
                     * forward

                     */
                    network.forward(input);
                    /**
                     * loss

                     */
                    network.loss(label);
                    /**
                     * loss diff

                     */
                    Tensor[] lossDiffs = network.lossDiff(label);
                    /**
                     * back

                     */
                    network.back(lossDiffs);
                    /**
                     * update

                     */
                    this.network.update();
                    String msg = "training[" + this.trainIndex + "]{" + it + "} (lr:" + this.network.learnRate + ") [costTime:" + (System.nanoTime() - start) / 1e6 + "ms.]";
                    System.out.println(msg);
                    this.batchIndex++;
                }
                /**
                 * update learning rate

                 */
                this.updateLR(this.lr_step);
                if (this.trainIndex % 100 == 0) {
                    System.out.println("----------------testing start----------------");
                    this.testObjectRecognitionOutputs(valiData, vail_input, label, this.batchSize);
                    System.out.println("----------------testing finish---------------");
                }
            }
            /**
             * 停止训练

             */
            System.out.println("training finish. [" + this.trainIndex + "] finalError:" + this.currentError);
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void trainObjectRecognitionOutputs(BaseDataLoader trainingData, BaseDataLoader valiData, boolean dataEnhance) {
        // TODO Auto-generated method stub
        try {
            CUDAModules.initCUDAFunctions();
            OutputsNetwork network = (OutputsNetwork) this.network;
            this.dataSize = trainingData.number;
            if (isWarmUp()) {
                this.network.learnRate = (float) (this.lr * Math.pow(batchIndex * 1.0f / burnIn * 1.0f, power));
            }
            Tensor input = new Tensor(batchSize, this.network.getChannel(), this.network.getHeight(), this.network.getWidth(), true);
            Tensor label = new Tensor(batchSize, 1, 1, trainingData.labelSize, true);
            Tensor vail_input = new Tensor(batchSize, this.network.getChannel(), this.network.getHeight(), this.network.getWidth(), true);
            Tensor vail_label = new Tensor(batchSize, 1, 1, valiData.labelSize, true);
            for (int i = 0; i < this.trainTime; i++) {
                if (this.trainIndex >= this.minTrainTime) {
                    break;
                }
                this.network.RUN_MODEL = RunModel.TRAIN;
                this.trainIndex = i + 1;
                int[][] indexs = trainingData.shuffle();
                /**
                 * 遍历整个训练集

                 */
                for (int it = 0; it < indexs.length; it++) {
                    long start = System.nanoTime();
                    this.loss.clear();
                    this.lossDiff.clear();
                    trainingData.loadData(indexs[it], input, label);
                    /**
                     * 数据增强

                     */
                    if (dataEnhance) {
                        dataEnhanceInstance().transforms(input, label);
                        YoloLabelUtils.formatToYolo(label, input.height, input.width);
                    }
                    input.hostToDevice();
                    label.hostToDevice();
                    /**
                     * forward

                     */
                    network.forward(input);
                    /**
                     * loss

                     */
                    network.loss(label);
                    System.out.println("in--------------->");
                    /**
                     * loss diff

                     */
                    Tensor[] lossDiffs = network.lossDiff(label);
                    /**
                     * back

                     */
                    network.back(lossDiffs);
                    /**
                     * update

                     */
                    this.network.update();
                    String msg = "training[" + this.trainIndex + "]{" + it + "} (lr:" + this.network.learnRate + ") [costTime:" + (System.nanoTime() - start) / 1e6 + "ms.]";
                    System.out.println(msg);
                    this.batchIndex++;
                }
                /**
                 * update learning rate

                 */
                this.updateLR(this.lr_step);
                if (this.trainIndex % 100 == 0) {
                    System.out.println("----------------testing start----------------");
                    this.testObjectRecognitionOutputs(valiData, vail_input, vail_label, this.batchSize);
                    System.out.println("----------------testing finish---------------");
                }
            }
            /**
             * 停止训练

             */
            System.out.println("training finish. [" + this.trainIndex + "] finalError:" + this.currentError);
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void trainObjectRecognitionOutputs(DetectionDataLoader trainingData, DetectionDataLoader valiData) {
        // TODO Auto-generated method stub
        try {
            //			CUDAModules.initCUDAFunctions();
            OutputsNetwork network = (OutputsNetwork) this.network;
            this.dataSize = trainingData.number;
            if (isWarmUp()) {
                this.network.learnRate = (float) (this.lr * Math.pow(batchIndex * 1.0f / burnIn * 1.0f, power));
            }
            Tensor input = new Tensor(batchSize, this.network.getChannel(), this.network.getHeight(), this.network.getWidth(), true);
            Tensor label = trainingData.initLabelTensor();
            Tensor vail_input = new Tensor(batchSize, this.network.getChannel(), this.network.getHeight(), this.network.getWidth(), true);
            for (int i = 0; i < this.trainTime; i++) {
                if (this.trainIndex >= this.minTrainTime) {
                    break;
                }
                if (this.trainIndex == 2) {
                    this.network.unfreeze();
                }
                this.network.RUN_MODEL = RunModel.TRAIN;
                this.trainIndex = i + 1;
                int[][] indexs = trainingData.shuffle();
                /**
                 * 遍历整个训练集

                 */
                for (int it = 0; it < indexs.length; it++) {
                    long start = System.nanoTime();
                    this.loss.clear();
                    this.lossDiff.clear();
                    trainingData.loadData(indexs[it], input, label);
                    /**
                     * forward

                     */
                    network.forward(input);
                    /**
                     * loss

                     */
                    network.loss(label);
                    /**
                     * loss diff

                     */
                    Tensor[] lossDiffs = network.lossDiff(label);
                    /**
                     * back

                     */
                    network.back(lossDiffs);
                    /**
                     * update

                     */
                    this.network.update();
                    String msg = "training[" + this.trainIndex + "]{" + it + "} (lr:" + this.network.learnRate + ") [costTime:" + (System.nanoTime() - start) / 1e6 + "ms.]";
                    System.out.println(msg);
                    this.batchIndex++;
                }
                /**
                 * update learning rate

                 */
                this.updateLR(this.lr_step);
                if (this.trainIndex % 100 == 0) {
                    System.out.println("----------------testing start----------------");
                    this.testObjectRecognitionOutputs(valiData, vail_input, label, this.batchSize);
                    System.out.println("----------------testing finish---------------");
                }
            }
            /**
             * 停止训练

             */
            System.out.println("training finish. [" + this.trainIndex + "] finalError:" + this.currentError);
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void testRNN(Tensor input) {
        try {
            CUDAModules.initCUDAFunctions();
            /**
             * forward

             */
            Tensor output = this.network.forward(input);
            output.showDM();
            /**
             * loss diff

             */
            float[] ld = MatrixUtils.one(output.dataLength);
            this.lossDiff = new Tensor(output.number, output.channel, output.height, output.width, ld, true);
            /**
             * back

             */
            this.network.back(this.lossDiff);
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void trainRNN(RNNDataLoader trainingData) {
        // TODO Auto-generated method stub
        try {
            CUDAModules.initCUDAFunctions();
            this.dataSize = trainingData.number;
            if (isWarmUp()) {
                this.network.learnRate = (float) (this.lr * Math.pow(batchIndex * 1.0f / burnIn * 1.0f, power));
            }
            Tensor input = new Tensor(trainingData.time * batchSize, this.network.getChannel(), this.network.getHeight(), this.network.getWidth(), true);
            Tensor label = trainingData.initLabelTensor();
            for (int i = 0; i < this.trainTime; i++) {
                if (this.trainIndex >= this.minTrainTime) {
                    break;
                }
                this.network.RUN_MODEL = RunModel.TRAIN;
                this.trainIndex = i + 1;
                int[][] indexs = trainingData.shuffle();
                /**
                 * 遍历整个训练集

                 */
                for (int it = 0; it < indexs.length; it++) {
                    long start = System.nanoTime();
                    this.loss.clear();
                    this.lossDiff.clear();
                    /**
                     * 读取训练数据

                     */
                    trainingData.loadData(indexs[it], input, label);
                    //					System.out.println(output2TXT(input, trainingData));
                    /**
                     * forward

                     */
                    Tensor output = this.network.forward(input);
                    /**
                     * loss

                     */
                    this.loss = this.network.loss(output, label);
                    /**
                     * loss diff

                     */
                    this.lossDiff = network.lossDiff(output, label);
                    //					System.out.println(JsonUtils.toJson(output.syncHost()));
                    //					GradClipping.gradClipping(this.lossDiff, 1e-7f);
                    /**
                     * back

                     */
                    this.network.back(this.lossDiff);
                    /**
                     * grad clipping

                     */
                    //					this.gradClipping(this.network);
                    /**
                     * update

                     */
                    this.network.update();
                    JCudaDriver.cuCtxSynchronize();
                    /**
                     * current time error

                     */
                    if (this.loss.isHasGPU()) {
                        this.currentError = MatrixOperation.sum(this.loss.syncHost()) / input.number;
                    } else {
                        this.currentError = MatrixOperation.sum(this.loss.data) / input.number;
                    }
                    //					train_loss += this.currentError;
                    output.syncHost();
                    float error = this.accuracy(output, label);
                    //					if(error > 99) {
                    //						break;
                    //					}
                    String msg = "training[" + this.trainIndex + "]{" + it + "} (lr:" + this.network.learnRate + ") accuracy:{" + error + "%} train_loss:" + this.currentError + " [costTime:" + (System.nanoTime() - start) / 1e6 + "ms.]";
                    System.out.println(msg);
                    this.batchIndex++;
                }
                /**
                 * update learning rate

                 */
                this.updateLR(this.lr_step);
            }
            /**
             * 停止训练

             */
            System.out.println("training finish. [" + this.trainIndex + "] finalError:" + this.currentError);
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void trainSeg(BaseDataLoader trainingData) {
        // TODO Auto-generated method stub
        try {
            CUDAModules.initCUDAFunctions();
            this.dataSize = trainingData.number;
            if (isWarmUp()) {
                this.network.learnRate = (float) (this.lr * Math.pow(batchIndex * 1.0f / burnIn * 1.0f, power));
            }
            Tensor input = new Tensor(batchSize, this.network.getChannel(), this.network.getHeight(), this.network.getWidth(), true);
            Tensor label = new Tensor(batchSize, 1, this.network.getHeight(), this.network.getWidth(), true);
            for (int i = 0; i < this.trainTime; i++) {
                if (this.trainIndex >= this.minTrainTime) {
                    break;
                }
                this.trainIndex = i + 1;
                int[][] indexs = trainingData.shuffle();
                this.network.RUN_MODEL = RunModel.TRAIN;
                float train_loss = 0.0f;
                /**
                 * 遍历整个训练集

                 */
                for (int it = 0; it < indexs.length; it++) {
                    long start = System.nanoTime();
                    if (Math.abs(this.currentError) <= this.error) {
                        break;
                    }
                    trainingData.loadData(indexs[it], input, label);
                    /**
                     * forward

                     */
                    Tensor output = this.network.forward(input);
                    /**
                     * loss

                     */
                    this.loss = this.network.loss(output, label);
                    /**
                     * loss diff

                     */
                    this.lossDiff = this.network.lossDiff(output, label);
                    /**
                     * back

                     */
                    this.network.back(this.lossDiff);
                    /**
                     * update

                     */
                    this.network.update();
                    JCudaDriver.cuCtxSynchronize();
                    /**
                     * current time error

                     */
                    if (this.loss.isHasGPU()) {
                        this.currentError = MatrixOperation.sum(this.loss.syncHost()) / this.batchSize;
                    } else {
                        this.currentError = MatrixOperation.sum(this.loss.data) / this.batchSize;
                    }
                    train_loss += this.currentError;
                    String msg = "training[" + this.trainIndex + "]{" + it + "} (lr:" + this.network.learnRate + ") train_loss:" + this.currentError + " [costTime:" + (System.nanoTime() - start) / 1e6 + "ms.]";
                    System.out.println(msg);
                    this.batchIndex++;
                }
                System.out.println("training[" + this.trainIndex + "] train loss:{" + train_loss / indexs.length + "} ");
                /**
                 * update learning rate

                 */
                this.updateLR(this.lr_step);
            }
            /**
             * 停止训练

             */
            System.out.println("training finish. [" + this.trainIndex + "] finalError:" + this.currentError);
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public Map<String, float[]> initGussianDiffusionTest(Tensor x_t, int T, float beta_1, float beta_T) {
        Map<String, float[]> result = new HashMap<String, float[]>();
        float[] betas = MatrixUtils.linspace(beta_1, beta_T, T);
        float[] alphas = MatrixOperation.subtraction(1, betas);
        float[] alphas_bar = MatrixUtils.cumprod(alphas);
        float[] alphas_bar_prev = new float[alphas_bar.length];
        alphas_bar_prev[0] = 1.0f;
        for (int i = 1; i < alphas_bar_prev.length; i++) {
            alphas_bar_prev[i] = alphas_bar[i - 1];
        }
        float[] sqrt_recip_alphas_bar = new float[alphas_bar.length];
        float[] sqrt_recipm1_alphas_bar = new float[alphas_bar.length];
        float[] posterior_var = new float[alphas_bar.length];
        float[] posterior_log_var_clipped = new float[alphas_bar.length];
        float[] posterior_mean_coef1 = new float[alphas_bar.length];
        float[] posterior_mean_coef2 = new float[alphas_bar.length];
        float[] model_log_var = new float[alphas_bar.length];
        for (int i = 0; i < alphas_bar.length; i++) {
            sqrt_recip_alphas_bar[i] = (float) Math.sqrt(1 / alphas_bar[i]);
            sqrt_recipm1_alphas_bar[i] = (float) Math.sqrt(1 / alphas_bar[i] - 1);
            posterior_var[i] = betas[i] * (1 - alphas_bar_prev[i]) / (1 - alphas_bar[i]);
            if (i == 0) {
                posterior_log_var_clipped[i] = (float) Math.log(betas[1] * (1 - alphas_bar_prev[1]) / (1 - alphas_bar[1]));
            } else {
                posterior_log_var_clipped[i] = (float) Math.log(posterior_var[i]);
            }
            posterior_mean_coef1[i] = (float) (Math.sqrt(alphas_bar_prev[i]) * betas[i] / (1 - alphas_bar[i]));
            posterior_mean_coef2[i] = (float) (Math.sqrt(alphas[i]) * (1 - alphas_bar_prev[i]) / (1 - alphas_bar[i]));
            if (i == 0) {
                model_log_var[i] = (float) Math.log(betas[1] * (1 - alphas_bar_prev[1]) / (1 - alphas_bar[1]));
            } else {
                model_log_var[i] = (float) Math.log(betas[i]);
            }
        }
        float[] posterior_mean = new float[x_t.dataLength];
        result.put("model_log_var", model_log_var);
        result.put("sqrt_recip_alphas_bar", sqrt_recip_alphas_bar);
        result.put("sqrt_recipm1_alphas_bar", sqrt_recipm1_alphas_bar);
        result.put("posterior_mean_coef1", posterior_mean_coef1);
        result.put("posterior_mean_coef2", posterior_mean_coef2);
        result.put("posterior_mean", posterior_mean);
        return result;
    }

    public void testGaussianDiffusion(Tensor x_t, Tensor t, int T, float beta_1, float beta_T, Map<String, float[]> params, float[] mean, float[] std) {
        try {
            DiffusionUNet network = (DiffusionUNet) this.network;
            RandomUtils.gaussianRandom(x_t);
            float[] model_log_var = params.get("model_log_var");
            float[] sqrt_recip_alphas_bar = params.get("sqrt_recip_alphas_bar");
            float[] sqrt_recipm1_alphas_bar = params.get("sqrt_recipm1_alphas_bar");
            float[] posterior_mean_coef1 = params.get("posterior_mean_coef1");
            float[] posterior_mean_coef2 = params.get("posterior_mean_coef2");
            float[] posterior_mean = params.get("posterior_mean");
            for (int timestep = T - 1; timestep >= 0; timestep--) {
                int[] t_data = MatrixUtils.valInt(x_t.number, timestep);
                float[] model_log_var_t = MatrixUtils.gather(model_log_var, t_data);
                float[] exsa1 = MatrixUtils.gather(sqrt_recip_alphas_bar, t_data);
                float[] exsa2 = MatrixUtils.gather(sqrt_recipm1_alphas_bar, t_data);
                float[] posterior_mean_coef1_t = MatrixUtils.gather(posterior_mean_coef1, t_data);
                float[] posterior_mean_coef2_t = MatrixUtils.gather(posterior_mean_coef2, t_data);
                //				float[] posterior_log_var_clipped_t = MatrixUtils.gather(posterior_log_var_clipped, t_data);
                t.setData(t_data);
                Tensor eps = network.forward(x_t, t);
                predict_xstart_from_eps(x_t, t, eps, exsa1, exsa2, posterior_mean_coef1_t, posterior_mean_coef2_t, posterior_mean);
                decodeXT(posterior_mean, model_log_var_t, x_t, timestep);
                System.out.println(timestep);
            }
            MatrixOperation.clampSelf(x_t.data, -1, 1);
            /**
             * print image

             */
            showImgs("H:\\voc\\gan_anime\\duffsion_test\\", x_t);
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void decodeXT(float[] mean, float[] log_var, Tensor x_t, int it) {
        for (int b = 0; b < x_t.number; b++) {
            for (int i = 0; i < x_t.getOnceSize(); i++) {
                int idx = b * x_t.getOnceSize() + i;
                if (it == 0) {
                    x_t.data[idx] = mean[idx];
                } else {
                    x_t.data[idx] = (float) (mean[idx] + Math.exp(0.5 * log_var[b]) * RandomUtils.randomGaussianFloat());
                }
            }
        }
        x_t.hostToDevice();
    }

    public void predict_xstart_from_eps(Tensor x_t, Tensor t, Tensor eps, float[] sqrt_recip_alphas_bar, float[] sqrt_recipm1_alphas_bar, float[] posterior_mean_coef1, float[] posterior_mean_coef2, float[] posterior_mean) {
        float[] eps_data = eps.syncHost();
        for (int b = 0; b < x_t.number; b++) {
            for (int i = 0; i < x_t.getOnceSize(); i++) {
                int idx = b * x_t.getOnceSize() + i;
                float x_0 = sqrt_recip_alphas_bar[b] * x_t.data[idx] - sqrt_recipm1_alphas_bar[b] * eps_data[idx];
                posterior_mean[idx] = posterior_mean_coef1[b] * x_0 - posterior_mean_coef2[b] * x_t.data[idx];
            }
        }
    }

    public void sample_prev_timestep(SDImageDataLoader trainingData, TinyVQVAE2 vae, Tensor xt, Tensor t, Tensor x0, int timestep, float[] a, float[] b, float[] betas, float[] alphas, float[] alphas_bar) {
        for (int i = 0; i < xt.number; i++) {
            t.data[i] = timestep;
        }
        t.hostToDevice();
        DiffusionUNet network = (DiffusionUNet) this.network;
        Tensor noisePred = network.forward(xt, t);
        noisePred.syncHost();
        //		if(x0 != null) {
        //
        //			trainingData.unNoise(a, b, xt, noisePred, x0);
        //
        //			TensorOP.clamp(x0, -1, 1, x0);
        //
        //		}
        JCuda.cudaDeviceSynchronize();
        if (timestep > 0) {
            System.err.println("timestep:" + timestep);
            float var = (1.0f - alphas_bar[timestep - 1]) / (1.0f - alphas_bar[timestep]) * betas[timestep];
            float sigma = (float) Math.pow(var, 0.5);
            float[] noise = RandomUtils.gaussianRandom(noisePred.dataLength, 1.0f);
            for (int i = 0; i < xt.dataLength; i++) {
                xt.data[i] = (float) ((xt.data[i] - (betas[timestep] * noisePred.data[i]) / b[timestep]) / Math.sqrt(alphas[timestep])) + sigma * noise[i];
            }
        } else {
            /**
             * mean

             */
            for (int i = 0; i < xt.dataLength; i++) {
                xt.data[i] = (float) ((xt.data[i] - (betas[timestep] * noisePred.data[i]) / b[timestep]) / Math.sqrt(alphas[timestep]));
            }
        }
        xt.hostToDevice();
    }

    public void sample_prev_timestep(DiffusionUNetCond network, SDImageDataLoader trainingData, TinyVQVAE2 vae, Tensor condInput, Tensor xt, Tensor t, Tensor x0, int timestep, float[] a, float[] b, float[] betas, float[] alphas, float[] alphas_bar) {
        for (int i = 0; i < xt.number; i++) {
            t.data[i] = timestep;
        }
        t.hostToDevice();
        Tensor noisePred = network.forward(xt, t, condInput);
        noisePred.syncHost();
        //		if(x0 != null) {
        //
        //			trainingData.unNoise(a, b, xt, noisePred, x0);
        //
        //			TensorOP.clamp(x0, -1, 1, x0);
        //
        //		}
        JCuda.cudaDeviceSynchronize();
        if (timestep > 0) {
            System.err.println("timestep:" + timestep);
            float var = (1.0f - alphas_bar[timestep - 1]) / (1.0f - alphas_bar[timestep]) * betas[timestep];
            float sigma = (float) Math.pow(var, 0.5);
            float[] noise = RandomUtils.gaussianRandom(noisePred.dataLength, 1.0f);
            for (int i = 0; i < xt.dataLength; i++) {
                xt.data[i] = (float) ((xt.data[i] - (betas[timestep] * noisePred.data[i]) / b[timestep]) / Math.sqrt(alphas[timestep])) + sigma * noise[i];
            }
        } else {
            /**
             * mean

             */
            for (int i = 0; i < xt.dataLength; i++) {
                xt.data[i] = (float) ((xt.data[i] - (betas[timestep] * noisePred.data[i]) / b[timestep]) / Math.sqrt(alphas[timestep]));
            }
        }
        xt.hostToDevice();
    }

    public void prev_mean_from_eps(Tensor xt, Tensor t, float[] alphas_bar, float[] alphas_bar_prev, float eta, int timestep) {
        DiffusionUNet network = (DiffusionUNet) this.network;
        //		xt.showDMByOffset(0, 100);
        Tensor eps = network.forward(xt, t);
        float[] eps_data = eps.syncHost();
        float[] noise = RandomUtils.gaussianRandom(eps.dataLength, 1.0f);
        //		System.out.println(JsonUtils.toJson(noise));
        //		xt.syncHost();
        //		eps.showDMByOffset(0, 96);
        for (int b = 0; b < xt.number; b++) {
            float sigma_t = (float) (eta * Math.sqrt((1.0f - alphas_bar_prev[b]) / (1.0f - alphas_bar[b]) * (1.0f - alphas_bar[b] / alphas_bar_prev[b])));
            for (int l = 0; l < xt.getOnceSize(); l++) {
                int i = b * xt.getOnceSize() + l;
                float pred_x0 = (float) ((xt.data[i] - Math.sqrt(1.0f - alphas_bar[b]) * eps_data[i]) / Math.sqrt(alphas_bar[b]));
                if (pred_x0 > 1) {
                    pred_x0 = 1;
                } else if (pred_x0 < -1) {
                    pred_x0 = -1;
                }
                float pred_dir_xt = (float) (Math.sqrt(1.0f - alphas_bar_prev[b] - sigma_t * sigma_t) * eps_data[i]);
                xt.data[i] = (float) Math.sqrt(alphas_bar_prev[b]) * pred_x0 + pred_dir_xt + sigma_t * noise[i];
                //				xt.data[i] = (float) (Math.sqrt(alphas_bar_prev[b]) * pred_x0 + pred_dir_xt + sigma_t);
            }
        }
        xt.hostToDevice();
        //		xt.showDMByOffset(0, 100);
    }

    public void prev_mean_from_eps(DiffusionUNetCond network, Tensor xt, Tensor t, float[] alphas_bar, float[] alphas_bar_prev, float eta, int timestep) {
        //		xt.showDMByOffset(0, 100);
        Tensor eps = network.forward(xt, t);
        float[] eps_data = eps.syncHost();
        float[] noise = RandomUtils.gaussianRandom(eps.dataLength, 1.0f);
        //		System.out.println(JsonUtils.toJson(noise));
        //		xt.syncHost();
        //		eps.showDMByOffset(0, 96);
        for (int b = 0; b < xt.number; b++) {
            float sigma_t = (float) (eta * Math.sqrt((1.0f - alphas_bar_prev[b]) / (1.0f - alphas_bar[b]) * (1.0f - alphas_bar[b] / alphas_bar_prev[b])));
            for (int l = 0; l < xt.getOnceSize(); l++) {
                int i = b * xt.getOnceSize() + l;
                float pred_x0 = (float) ((xt.data[i] - Math.sqrt(1.0f - alphas_bar[b]) * eps_data[i]) / Math.sqrt(alphas_bar[b]));
                if (pred_x0 > 1) {
                    pred_x0 = 1;
                } else if (pred_x0 < -1) {
                    pred_x0 = -1;
                }
                float pred_dir_xt = (float) (Math.sqrt(1.0f - alphas_bar_prev[b] - sigma_t * sigma_t) * eps_data[i]);
                xt.data[i] = (float) Math.sqrt(alphas_bar_prev[b]) * pred_x0 + pred_dir_xt + sigma_t * noise[i];
                //				xt.data[i] = (float) (Math.sqrt(alphas_bar_prev[b]) * pred_x0 + pred_dir_xt + sigma_t);
            }
        }
        xt.hostToDevice();
        //		xt.showDMByOffset(0, 100);
    }

    public void prev_mean_from_eps(DiffusionUNetCond network, Tensor xt, Tensor t, Tensor condInput, float[] alphas_bar, float[] alphas_bar_prev, float eta, int timestep) {
        //		xt.showDMByOffset(0, 100);
        Tensor eps = network.forward(xt, t, condInput);
        float[] eps_data = eps.syncHost();
        float[] noise = RandomUtils.gaussianRandom(eps.dataLength, 1.0f);
        //		System.out.println(JsonUtils.toJson(noise));
        //		xt.syncHost();
        //		eps.showDMByOffset(0, 96);
        for (int b = 0; b < xt.number; b++) {
            float sigma_t = (float) (eta * Math.sqrt((1.0f - alphas_bar_prev[b]) / (1.0f - alphas_bar[b]) * (1.0f - alphas_bar[b] / alphas_bar_prev[b])));
            for (int l = 0; l < xt.getOnceSize(); l++) {
                int i = b * xt.getOnceSize() + l;
                //				System.err.println(b+":"+xt.getOnceSize());
                float pred_x0 = (float) ((xt.data[i] - Math.sqrt(1.0f - alphas_bar[b]) * eps_data[i]) / Math.sqrt(alphas_bar[b]));
                if (pred_x0 > 1) {
                    pred_x0 = 1;
                } else if (pred_x0 < -1) {
                    pred_x0 = -1;
                }
                float pred_dir_xt = (float) (Math.sqrt(1.0f - alphas_bar_prev[b] - sigma_t * sigma_t) * eps_data[i]);
                xt.data[i] = (float) Math.sqrt(alphas_bar_prev[b]) * pred_x0 + pred_dir_xt + sigma_t * noise[i];
                //				xt.data[i] = (float) (Math.sqrt(alphas_bar_prev[b]) * pred_x0 + pred_dir_xt + sigma_t);
            }
        }
        xt.hostToDevice();
        //		xt.showDMByOffset(0, 100);
    }

    public void testGaussianDiffusion(String it, int ddim_timesteps, Tensor noiseInput, Tensor t) {
        try {
            float beta_1 = 1e-4f;
            float beta_T = 0.02f;
            int T = 1000;
            float[] mean = new float[]{0.5f, 0.5f, 0.5f};
            float[] std = new float[]{0.5f, 0.5f, 0.5f};
            //			Tensor noiseInput = new Tensor(batchSize, this.network.getChannel(), this.network.getHeight(), this.network.getWidth(), true);
            //
            //			Tensor t = new Tensor(batchSize, 1, 1, 1, true);
            //			RandomUtils.gaussianRandom2(noiseInput, 0, 1);
            RandomUtils.gaussianRandom(noiseInput, 0, 1);
            float[] betas = MatrixUtils.linspace(beta_1, beta_T, T);
            float[] alphas = MatrixOperation.subtraction(1, betas);
            float[] alphas_bar = MatrixUtils.cumprod(alphas);
            int step = T / ddim_timesteps;
            float[] ddim_timestep_seq = MatrixUtils.range(0, T, step, 1);
            float[] ddim_timestep_prev_seq = new float[ddim_timestep_seq.length];
            for (int i = 1; i < ddim_timestep_seq.length; i++) {
                ddim_timestep_prev_seq[i] = ddim_timestep_seq[i - 1];
            }
            int[] t_data = new int[batchSize];
            int[] prev_t_data = new int[batchSize];
            for (int timestep = ddim_timesteps - 1; timestep >= 0; timestep--) {
                for (int i = 0; i < batchSize; i++) {
                    t.data[i] = ddim_timestep_seq[timestep];
                    t_data[i] = (int) ddim_timestep_seq[timestep];
                    prev_t_data[i] = (int) ddim_timestep_prev_seq[timestep];
                }
                t.hostToDevice();
                float[] exsa1 = MatrixUtils.gather(alphas_bar, t_data);
                float[] exsa2 = MatrixUtils.gather(alphas_bar, prev_t_data);
                prev_mean_from_eps(noiseInput, t, exsa1, exsa2, 1, timestep);
            }
            noiseInput.data = MatrixOperation.clampSelf(noiseInput.data, -1, 1);
            /**
             * print image

             */
            showImgs("H:\\voc\\gan_anime\\duffsion_test\\", noiseInput, it, mean, std);
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void trainGaussianDiffusion(DiffusionImageDataLoader trainingData) {
        // TODO Auto-generated method stub
        try {
            CUDAModules.initCUDAFunctions();
            DiffusionUNet network = (DiffusionUNet) this.network;
            this.dataSize = trainingData.number;
            if (isWarmUp()) {
                this.network.learnRate = (float) (this.lr * Math.pow(batchIndex * 1.0f / burnIn * 1.0f, power));
            }
            float beta_1 = 1e-4f;
            float beta_T = 0.02f;
            int T = 1000;
            Tensor input = new Tensor(batchSize, this.network.getChannel(), this.network.getHeight(), this.network.getWidth(), true);
            //			Tensor x_t = new Tensor(batchSize, this.network.getChannel(), this.network.getHeight(), this.network.getWidth(), true);
            Tensor t = new Tensor(batchSize, 1, 1, 1, true);
            Tensor noise = new Tensor(batchSize, this.network.getChannel(), this.network.getHeight(), this.network.getWidth(), true);
            float[] betas = MatrixUtils.linspace(beta_1, beta_T, T);
            float[] alphas = MatrixOperation.subtraction(1, betas);
            float[] alphas_bar = MatrixUtils.cumprod(alphas);
            float[] sqrt_alphas_bar = MatrixOperation.sqrt(alphas_bar);
            float[] sqrt_one_minus_alphas_bar = MatrixOperation.sqrt(MatrixOperation.subtraction(1, alphas_bar));
            //			Map<String,float[]> testParams = initGussianDiffusionTest(input, T, beta_1, beta_T);
            for (int i = 0; i < this.trainTime; i++) {
                if (this.trainIndex >= this.minTrainTime) {
                    break;
                }
                this.trainIndex = i + 1;
                int[][] indexs = trainingData.shuffle();
                //				int[][] indexs = trainingData.order();
                this.network.RUN_MODEL = RunModel.TRAIN;
                float train_loss = 0.0f;
                /**
                 * 遍历整个训练集

                 */
                for (int it = 0; it < indexs.length; it++) {
                    long start = System.nanoTime();
                    if (Math.abs(this.currentError) <= this.error) {
                        break;
                    }
                    //					int[] t_data = RandomUtils.randomInt2(0, T - 1, batchSize);
                    int[] t_data = RandomUtils.randomInt(0, T - 1, batchSize);
                    //					int[] t_data = new int[] {100, 902, 31, 698};
                    //					System.out.println(JsonUtils.toJson(t_data));
                    t.setData(t_data);
                    //					t.showDM();
                    float[] exsa1 = MatrixUtils.gather(sqrt_alphas_bar, t_data);
                    float[] exsa2 = MatrixUtils.gather(sqrt_one_minus_alphas_bar, t_data);
                    trainingData.loadData(indexs[it], exsa1, exsa2, input, noise);
                    JCudaDriver.cuCtxSynchronize();
                    //					/**
                    //					 * print image
                    //					 */
                    //					if(it > 0 && it % 1 == 0) {
                    //						float[] mean = new float[] {0.5f, 0.5f, 0.5f};
                    //						float[] std = new float[] {0.5f, 0.5f, 0.5f};
                    //						showImgs("E:\\voc\\gan_anime\\duffsion_test_input\\", input, mean, std);
                    //					}
                    /**
                     * forward

                     */
                    Tensor output = network.forward(input, t);
                    /**
                     * loss

                     */
                    this.loss = network.loss(output, noise);
                    /**
                     * loss diff

                     */
                    this.lossDiff = network.lossDiff(output, noise);
                    /**
                     * back

                     */
                    network.back(this.lossDiff);
                    //					System.out.println(JsonUtils.toJson(this.loss.syncHost()));
                    /**
                     * update

                     */
                    network.update();
                    JCudaDriver.cuCtxSynchronize();
                    /**
                     * current time error

                     */
                    if (this.loss.isHasGPU()) {
                        this.currentError = MatrixOperation.sum(this.loss.syncHost()) / this.batchSize;
                        //						System.out.println(JsonUtils.toJson(this.loss.syncHost()));
                    } else {
                        this.currentError = MatrixOperation.sum(this.loss.data) / this.batchSize;
                    }
                    train_loss += this.currentError;
                    String msg = "training[" + this.trainIndex + "]{" + it + "} (lr:" + this.network.learnRate + ") train_loss:" + this.currentError + " [costTime:" + (System.nanoTime() - start) / 1e6 + "ms.]";
                    System.out.println(msg);
                    this.batchIndex++;
                    if (it > 0 && it % 500 == 0) {
                        network.RUN_MODEL = RunModel.TEST;
                        System.out.println("start create test images.");
                        //						testGaussianDiffusion(i + "_" + it, 200, input, noise);
                        testGaussianDiffusion(i + "_" + it, 200, input, t);
                        System.out.println("finish create.");
                        //						testGaussianDiffusion(x_t, t, T, beta_1, beta_T, testParams, trainingData.mean, trainingData.std);
                        network.RUN_MODEL = RunModel.TRAIN;
                        //						this.network.learnRate = this.network.learnRate * 0.1f;
                    }
                }
                System.out.println("training[" + this.trainIndex + "] train loss:{" + train_loss / indexs.length + "} ");
                /**
                 * update learning rate

                 */
                this.updateLR(this.lr_step);
            }
            /**
             * 停止训练

             */
            System.out.println("training finish. [" + this.trainIndex + "] finalError:" + this.currentError);
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void trainTinyVAE(DiffusionImageDataLoader trainingData) {
        // TODO Auto-generated method stub
        try {
            CUDAModules.initCUDAFunctions();
            TinyVAE network = (TinyVAE) this.network;
            this.dataSize = trainingData.number;
            if (isWarmUp()) {
                this.network.learnRate = (float) (this.lr * Math.pow(batchIndex * 1.0f / burnIn * 1.0f, power));
            }
            Tensor input = new Tensor(batchSize, this.network.getChannel(), this.network.getHeight(), this.network.getWidth(), true);
            for (int i = 0; i < this.trainTime; i++) {
                if (this.trainIndex >= this.minTrainTime) {
                    break;
                }
                this.trainIndex = i + 1;
                int[][] indexs = trainingData.shuffle();
                //				int[][] indexs = trainingData.order();
                this.network.RUN_MODEL = RunModel.TRAIN;
                float train_loss = 0.0f;
                /**
                 * 遍历整个训练集

                 */
                for (int it = 0; it < indexs.length; it++) {
                    long start = System.nanoTime();
                    if (Math.abs(this.currentError) <= this.error) {
                        break;
                    }
                    trainingData.loadData(indexs[it], input);
                    JCudaDriver.cuCtxSynchronize();
                    /**
                     * forward

                     */
                    Tensor output = network.forward(input);
                    /**
                     * loss

                     */
                    float loss = network.totalLoss(output, input);
                    /**
                     * loss diff

                     */
                    this.lossDiff = network.lossDiff(output, input);
                    /**
                     * back

                     */
                    network.back(this.lossDiff);
                    //					System.out.println(JsonUtils.toJson(this.loss.syncHost()));
                    /**
                     * update

                     */
                    network.update();
                    JCudaDriver.cuCtxSynchronize();
                    /**
                     * current time error

                     */
                    this.currentError = loss;
                    train_loss += this.currentError;
                    String msg = "training[" + this.trainIndex + "]{" + it + "} (lr:" + this.network.learnRate + ") train_loss:" + this.currentError + " [costTime:" + (System.nanoTime() - start) / 1e6 + "ms.]";
                    System.out.println(msg);
                    this.batchIndex++;
                    /**
                     * update learning rate

                     */
                    //					this.updateLR(this.lr_step);
                    //					updateLRDynamic(i * trainingData.count_it + it, this.trainTime * trainingData.count_it);
                }
                System.out.println("training[" + this.trainIndex + "] train loss:{" + train_loss / indexs.length + "} ");
                /**
                 * update learning rate

                 */
                this.updateLR(this.lr_step);
                if (i % 10 == 0) {
                    /**
                     * showImage

                     */
                    this.network.RUN_MODEL = RunModel.TEST;
                    Tensor output = network.forward(input);
                    output.syncHost();
                    //					output.data = MatrixOperation.clampSelf(output.data, -1, 1);
                    /**
                     * print image

                     */
                    showImgs("H:\\vae_dataset\\pokemon-blip\\test\\", output, i + "", trainingData.mean, trainingData.std);
                }
            }
            /**
             * 停止训练

             */
            System.out.println("training finish. [" + this.trainIndex + "] finalError:" + this.currentError);
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void trainVQVAE(DiffusionImageDataLoader trainingData) {
        // TODO Auto-generated method stub
        try {
            CUDAModules.initCUDAFunctions();
            VQVAE network = (VQVAE) this.network;
            this.dataSize = trainingData.number;
            if (isWarmUp()) {
                this.network.learnRate = (float) (this.lr * Math.pow(batchIndex * 1.0f / burnIn * 1.0f, power));
            }
            Tensor input = new Tensor(batchSize, this.network.getChannel(), this.network.getHeight(), this.network.getWidth(), true);
            for (int i = 0; i < this.trainTime; i++) {
                if (this.trainIndex >= this.minTrainTime) {
                    break;
                }
                this.trainIndex = i + 1;
                int[][] indexs = trainingData.shuffle();
                //				int[][] indexs = trainingData.order();
                this.network.RUN_MODEL = RunModel.TRAIN;
                float train_loss = 0.0f;
                /**
                 * 遍历整个训练集

                 */
                for (int it = 0; it < indexs.length; it++) {
                    long start = System.nanoTime();
                    if (Math.abs(this.currentError) <= this.error) {
                        break;
                    }
                    trainingData.loadData(indexs[it], input);
                    JCudaDriver.cuCtxSynchronize();
                    /**
                     * forward

                     */
                    Tensor output = network.forward(input);
                    /**
                     * loss

                     */
                    float loss = network.totalLoss(output, input);
                    /**
                     * loss diff

                     */
                    this.lossDiff = network.lossDiff(output, input);
                    /**
                     * back

                     */
                    network.back(this.lossDiff);
                    //					System.out.println(JsonUtils.toJson(this.loss.syncHost()));
                    /**
                     * update

                     */
                    network.update();
                    JCudaDriver.cuCtxSynchronize();
                    /**
                     * current time error

                     */
                    this.currentError = loss;
                    train_loss += this.currentError;
                    String msg = "training[" + this.trainIndex + "]{" + it + "} (lr:" + this.network.learnRate + ") train_loss:" + this.currentError + " [costTime:" + (System.nanoTime() - start) / 1e6 + "ms.]";
                    System.out.println(msg);
                    this.batchIndex++;
                    /**
                     * update learning rate

                     */
                    //					this.updateLR(this.lr_step);
                    //					updateLRDynamic(i * trainingData.count_it + it, this.trainTime * trainingData.count_it);
                }
                System.out.println("training[" + this.trainIndex + "] train loss:{" + train_loss / indexs.length + "} ");
                /**
                 * update learning rate

                 */
                this.updateLR(this.lr_step);
                if (i % 1 == 0) {
                    /**
                     * showImage

                     */
                    this.network.RUN_MODEL = RunModel.TEST;
                    Tensor output = network.forward(input);
                    output.syncHost();
                    //					output.data = MatrixOperation.clampSelf(output.data, -1, 1);
                    /**
                     * print image

                     */
                    showImgs("H:\\vae_dataset\\pokemon-blip\\test128\\", output, i + "", trainingData.mean, trainingData.std);
                }
            }
            /**
             * 停止训练

             */
            System.out.println("training finish. [" + this.trainIndex + "] finalError:" + this.currentError);
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void trainTinyVQVAE(DiffusionImageDataLoader trainingData) {
        // TODO Auto-generated method stub
        try {
            CUDAModules.initCUDAFunctions();
            TinyVQVAE network = (TinyVQVAE) this.network;
            this.dataSize = trainingData.number;
            if (isWarmUp()) {
                this.network.learnRate = (float) (this.lr * Math.pow(batchIndex * 1.0f / burnIn * 1.0f, power));
            }
            Tensor input = new Tensor(batchSize, this.network.getChannel(), this.network.getHeight(), this.network.getWidth(), true);
            for (int i = 0; i < this.trainTime; i++) {
                if (this.trainIndex >= this.minTrainTime) {
                    break;
                }
                this.trainIndex = i + 1;
                int[][] indexs = trainingData.shuffle();
                //				int[][] indexs = trainingData.order();
                this.network.RUN_MODEL = RunModel.TRAIN;
                float train_loss = 0.0f;
                /**
                 * 遍历整个训练集

                 */
                for (int it = 0; it < indexs.length; it++) {
                    long start = System.nanoTime();
                    if (Math.abs(this.currentError) <= this.error) {
                        break;
                    }
                    trainingData.loadData(indexs[it], input);
                    JCudaDriver.cuCtxSynchronize();
                    /**
                     * forward

                     */
                    Tensor output = network.forward(input);
                    /**
                     * loss

                     */
                    float loss = network.totalLoss(output, input);
                    /**
                     * loss diff

                     */
                    this.lossDiff = network.lossDiff(output, input);
                    /**
                     * back

                     */
                    network.back(this.lossDiff);
                    //					System.out.println(JsonUtils.toJson(this.loss.syncHost()));
                    /**
                     * update

                     */
                    network.update();
                    JCudaDriver.cuCtxSynchronize();
                    /**
                     * current time error

                     */
                    this.currentError = loss;
                    train_loss += this.currentError;
                    String msg = "training[" + this.trainIndex + "]{" + it + "} (lr:" + this.network.learnRate + ") train_loss:" + this.currentError + " [costTime:" + (System.nanoTime() - start) / 1e6 + "ms.]";
                    System.out.println(msg);
                    this.batchIndex++;
                    /**
                     * update learning rate

                     */
                    this.updateLR(this.lr_step);
                    updateLRDynamic(i * trainingData.count_it + it, this.trainTime * trainingData.count_it, 1e-6f);
                    //					if(it % 100 == 0) {
                    //
                    //						/**
                    //						 * showImage
                    //						 */
                    //						this.network.RUN_MODEL = RunModel.TEST;
                    //
                    //						output = network.forward(input);
                    //						output.syncHost();
                    ////						output.data = MatrixOperation.clampSelf(output.data, -1, 1);
                    //
                    //						/**
                    //						 * print image
                    //						 */
                    //						showImgs("H:\\vae_dataset\\pokemon-blip\\test256\\", output, i + "", trainingData.mean, trainingData.std);
                    //
                    //						this.network.RUN_MODEL = RunModel.TRAIN;
                    //
                    //					}
                }
                System.out.println("training[" + this.trainIndex + "] train loss:{" + train_loss / indexs.length + "} ");
                /**
                 * update learning rate

                 */
                //				this.updateLR(this.lr_step);
                if (i % 10 == 0) {
                    /**
                     * showImage

                     */
                    this.network.RUN_MODEL = RunModel.TEST;
                    Tensor output = network.forward(input);
                    output.syncHost();
                    //					output.data = MatrixOperation.clampSelf(output.data, -1, 1);
                    /**
                     * print image

                     */
                    showImgs("H:\\vae_dataset\\pokemon-blip\\test256\\", output, i + "", trainingData.mean, trainingData.std);
                }
            }
            /**
             * 停止训练

             */
            System.out.println("training finish. [" + this.trainIndex + "] finalError:" + this.currentError);
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void trainTinyVQVAE2(DiffusionImageDataLoader trainingData) {
        // TODO Auto-generated method stub
        try {
            CUDAModules.initCUDAFunctions();
            TinyVQVAE2 network = (TinyVQVAE2) this.network;
            this.dataSize = trainingData.number;
            if (isWarmUp()) {
                this.network.learnRate = (float) (this.lr * Math.pow(batchIndex * 1.0f / burnIn * 1.0f, power));
            }
            Tensor input = new Tensor(batchSize, this.network.getChannel(), this.network.getHeight(), this.network.getWidth(), true);
            for (int i = 0; i < this.trainTime; i++) {
                if (this.trainIndex >= this.minTrainTime) {
                    break;
                }
                this.trainIndex = i + 1;
                int[][] indexs = trainingData.shuffle();
                //				int[][] indexs = trainingData.order();
                this.network.RUN_MODEL = RunModel.TRAIN;
                float train_loss = 0.0f;
                /**
                 * 遍历整个训练集

                 */
                for (int it = 0; it < indexs.length; it++) {
                    long start = System.nanoTime();
                    if (Math.abs(this.currentError) <= this.error) {
                        break;
                    }
                    trainingData.loadData(indexs[it], input);
                    JCudaDriver.cuCtxSynchronize();
                    /**
                     * forward

                     */
                    Tensor output = network.forward(input);
                    /**
                     * loss

                     */
                    float loss = network.totalLoss(output, input);
                    /**
                     * loss diff

                     */
                    this.lossDiff = network.lossDiff(output, input);
                    /**
                     * back

                     */
                    network.back(this.lossDiff);
                    //					System.out.println(JsonUtils.toJson(this.loss.syncHost()));
                    /**
                     * update

                     */
                    network.update();
                    JCudaDriver.cuCtxSynchronize();
                    /**
                     * current time error

                     */
                    this.currentError = loss;
                    train_loss += this.currentError;
                    String msg = "training[" + this.trainIndex + "]{" + it + "} (lr:" + this.network.learnRate + ") train_loss:" + this.currentError + " [costTime:" + (System.nanoTime() - start) / 1e6 + "ms.]";
                    System.out.println(msg);
                    this.batchIndex++;
                    /**
                     * update learning rate

                     */
                    this.updateLR(this.lr_step);
                    updateLRDynamic(i * trainingData.count_it + it, this.trainTime * trainingData.count_it, 1e-6f);
                    //					if(it % 100 == 0) {
                    //
                    //						/**
                    //						 * showImage
                    //						 */
                    //						this.network.RUN_MODEL = RunModel.TEST;
                    //
                    //						output = network.forward(input);
                    //						output.syncHost();
                    ////						output.data = MatrixOperation.clampSelf(output.data, -1, 1);
                    //
                    //						/**
                    //						 * print image
                    //						 */
                    //						showImgs("H:\\vae_dataset\\pokemon-blip\\test256\\", output, i + "", trainingData.mean, trainingData.std);
                    //
                    //						this.network.RUN_MODEL = RunModel.TRAIN;
                    //
                    //					}
                }
                System.out.println("training[" + this.trainIndex + "] train loss:{" + train_loss / indexs.length + "} ");
                /**
                 * update learning rate

                 */
                //				this.updateLR(this.lr_step);
                if (i % 1 == 0) {
                    /**
                     * showImage

                     */
                    this.network.RUN_MODEL = RunModel.TEST;
                    Tensor output = network.forward(input);
                    output.syncHost();
                    output.data = MatrixOperation.clampSelf(output.data, -1, 1);
                    /**
                     * print image

                     */
                    showImgs("H:\\vae_dataset\\pokemon-blip\\test256\\", output, i + "", trainingData.mean, trainingData.std);
                }
            }
            /**
             * 停止训练

             */
            System.out.println("training finish. [" + this.trainIndex + "] finalError:" + this.currentError);
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void trainTinyVQVAE_lpips(DiffusionImageDataLoader trainingData, LPIPS lpips) {
        // TODO Auto-generated method stub
        try {
            float perceptual_weight = 1;
            CUDAModules.initCUDAFunctions();
            TinyVQVAE network = (TinyVQVAE) this.network;
            this.dataSize = trainingData.number;
            if (isWarmUp()) {
                this.network.learnRate = (float) (this.lr * Math.pow(batchIndex * 1.0f / burnIn * 1.0f, power));
            }
            Tensor input = new Tensor(batchSize, this.network.getChannel(), this.network.getHeight(), this.network.getWidth(), true);
            Tensor lpipLoss = new Tensor(1, 1, 1, 1, true);
            Tensor lpipsLossDiff = new Tensor(batchSize, 1, 1, 1, MatrixUtils.val(batchSize, 1.0f / batchSize), true);
            for (int i = 0; i < this.trainTime; i++) {
                if (this.trainIndex >= this.minTrainTime) {
                    break;
                }
                this.trainIndex = i + 1;
                int[][] indexs = trainingData.shuffle();
                //				int[][] indexs = trainingData.order();
                this.network.RUN_MODEL = RunModel.TRAIN;
                float train_loss = 0.0f;
                /**
                 * 遍历整个训练集

                 */
                for (int it = 0; it < indexs.length; it++) {
                    long start = System.nanoTime();
                    //					if(Math.abs(this.currentError) <= this.error) {
                    //						break;
                    //					}
                    trainingData.loadData(indexs[it], input);
                    JCudaDriver.cuCtxSynchronize();
                    /**
                     * forward

                     */
                    Tensor output = network.forward(input);
                    Tensor lpipsOutput = lpips.forward(output, input);
                    /**
                     * current time error

                     */
                    network.tensorOP.mean(lpipsOutput, 0, lpipLoss);
                    /**
                     * loss

                     */
                    float loss = network.totalLoss(output, input);
                    /**
                     * loss diff

                     */
                    this.lossDiff = network.lossDiff(output, input);
                    lpips.back(lpipsLossDiff);
                    network.tensorOP.add(this.lossDiff, lpips.lpips.diff, this.lossDiff);
                    /**
                     * back

                     */
                    network.back(this.lossDiff);
                    //					System.out.println(JsonUtils.toJson(this.loss.syncHost()));
                    /**
                     * update

                     */
                    network.update();
                    JCudaDriver.cuCtxSynchronize();
                    /**
                     * current time error

                     */
                    float ploss = lpipLoss.syncHost()[0];
                    System.out.println("ploss:" + ploss);
                    this.currentError = loss + perceptual_weight * ploss;
                    train_loss += this.currentError;
                    String msg = "training[" + this.trainIndex + "]{" + it + "} (lr:" + this.network.learnRate + ") train_loss:" + this.currentError + " [costTime:" + (System.nanoTime() - start) / 1e6 + "ms.]";
                    System.out.println(msg);
                    this.batchIndex++;
                    /**
                     * update learning rate

                     */
                    this.updateLR(this.lr_step);
                    updateLRDynamic(i * trainingData.count_it + it, this.trainTime * trainingData.count_it, 1e-6f);
                }
                System.out.println("training[" + this.trainIndex + "] train loss:{" + train_loss / indexs.length + "} ");
                /**
                 * update learning rate

                 */
                //				this.updateLR(this.lr_step);
                if (i % 10 == 0) {
                    /**
                     * showImage

                     */
                    this.network.RUN_MODEL = RunModel.TEST;
                    Tensor output = network.forward(input);
                    output.syncHost();
                    output.data = MatrixOperation.clampSelf(output.data, -1, 1);
                    /**
                     * print image

                     */
                    showImgs("H:\\vae_dataset\\pokemon-blip\\test128\\", output, i + "", trainingData.mean, trainingData.std);
                }
            }
            /**
             * 停止训练

             */
            System.out.println("training finish. [" + this.trainIndex + "] finalError:" + this.currentError);
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void trainTinyVQVAE2_lpips(DiffusionImageDataLoader trainingData, LPIPS lpips) {
        // TODO Auto-generated method stub
        try {
            float perceptual_weight = 1;
            CUDAModules.initCUDAFunctions();
            TinyVQVAE2 network = (TinyVQVAE2) this.network;
            this.dataSize = trainingData.number;
            if (isWarmUp()) {
                this.network.learnRate = (float) (this.lr * Math.pow(batchIndex * 1.0f / burnIn * 1.0f, power));
            }
            Tensor input = new Tensor(batchSize, this.network.getChannel(), this.network.getHeight(), this.network.getWidth(), true);
            Tensor lpipLoss = new Tensor(1, 1, 1, 1, true);
            Tensor lpipsLossDiff = new Tensor(batchSize, 1, 1, 1, MatrixUtils.val(batchSize, 1.0f / batchSize), true);
            for (int i = 0; i < this.trainTime; i++) {
                if (this.trainIndex >= this.minTrainTime) {
                    break;
                }
                this.trainIndex = i + 1;
                int[][] indexs = trainingData.shuffle();
                //				int[][] indexs = trainingData.order();
                this.network.RUN_MODEL = RunModel.TRAIN;
                float train_loss = 0.0f;
                /**
                 * 遍历整个训练集

                 */
                for (int it = 0; it < indexs.length; it++) {
                    long start = System.nanoTime();
                    if (Math.abs(this.currentError) <= this.error) {
                        break;
                    }
                    trainingData.loadData(indexs[it], input);
                    JCudaDriver.cuCtxSynchronize();
                    /**
                     * forward

                     */
                    Tensor output = network.forward(input);
                    Tensor lpipsOutput = lpips.forward(output, input);
                    /**
                     * current time error

                     */
                    network.tensorOP.mean(lpipsOutput, 0, lpipLoss);
                    /**
                     * loss

                     */
                    float loss = network.totalLoss(output, input);
                    /**
                     * loss diff

                     */
                    this.lossDiff = network.lossDiff(output, input);
                    lpips.back(lpipsLossDiff);
                    network.tensorOP.add(this.lossDiff, lpips.lpips.diff, this.lossDiff);
                    /**
                     * back
                     */
                    network.back(this.lossDiff);
                    //					System.out.println(JsonUtils.toJson(this.loss.syncHost()));
                    /**
                     * update

                     */
                    network.update();
                    JCudaDriver.cuCtxSynchronize();
                    /**
                     * current time error

                     */
                    float ploss = lpipLoss.syncHost()[0];
                    System.out.println("ploss:" + ploss);
                    this.currentError = loss + perceptual_weight * ploss;
                    train_loss += this.currentError;
                    String msg = "training[" + this.trainIndex + "]{" + it + "} (lr:" + this.network.learnRate + ") train_loss:" + this.currentError + " [costTime:" + (System.nanoTime() - start) / 1e6 + "ms.]";
                    System.out.println(msg);
                    this.batchIndex++;
                    /**
                     * update learning rate

                     */
                    this.updateLR(this.lr_step);
                    updateLRDynamic(i * trainingData.count_it + it, this.trainTime * trainingData.count_it, 1e-6f);
                    //					if(it % 100 == 0) {
                    //
                    //						/**
                    //						 * showImage
                    //						 */
                    //						this.network.RUN_MODEL = RunModel.TEST;
                    //
                    //						output = network.forward(input);
                    //						output.syncHost();
                    ////						output.data = MatrixOperation.clampSelf(output.data, -1, 1);
                    //
                    //						/**
                    //						 * print image
                    //						 */
                    //						showImgs("H:\\vae_dataset\\pokemon-blip\\test256\\", output, i + "", trainingData.mean, trainingData.std);
                    //
                    //						this.network.RUN_MODEL = RunModel.TRAIN;
                    //
                    //					}
                }
                System.out.println("training[" + this.trainIndex + "] train loss:{" + train_loss / indexs.length + "} ");
                /**
                 * update learning rate

                 */
                //				this.updateLR(this.lr_step);
                if (i % 1 == 0) {
                    /**
                     * showImage

                     */
                    this.network.RUN_MODEL = RunModel.TEST;
                    Tensor output = network.forward(input);
                    output.syncHost();
                    output.data = MatrixOperation.clampSelf(output.data, -1, 1);
                    /**
                     * print image

                     */
                    showImgs("H:\\vae_dataset\\pokemon-blip\\test128\\", output, i + "", trainingData.mean, trainingData.std);
                }
            }
            /**
             * 停止训练

             */
            System.out.println("training finish. [" + this.trainIndex + "] finalError:" + this.currentError);
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void trainTinyVQVAE2_lpips_patchGANDisc(DiffusionImageDataLoader trainingData, LPIPS lpips, PatchGANDiscriminator disc, int discStepStart) {
        // TODO Auto-generated method stub
        try {
            float perceptual_weight = 1;
            float disc_weight = 0.5f;
            CUDAModules.initCUDAFunctions();
            TinyVQVAE2 network = (TinyVQVAE2) this.network;
            this.dataSize = trainingData.number;
            if (isWarmUp()) {
                this.network.learnRate = (float) (this.lr * Math.pow(batchIndex * 1.0f / burnIn * 1.0f, power));
            }
            Tensor input = new Tensor(batchSize, this.network.getChannel(), this.network.getHeight(), this.network.getWidth(), true);
            Tensor lpipLoss = new Tensor(1, 1, 1, 1, true);
            Tensor lpipsLossDiff = new Tensor(batchSize, 1, 1, 1, MatrixUtils.val(batchSize, 1.0f / batchSize), true);
            Tensor ones = null;
            Tensor zeros = null;
            int stepCount = 0;
            for (int i = 0; i < this.trainTime; i++) {
                if (this.trainIndex >= this.minTrainTime) {
                    break;
                }
                this.trainIndex = i + 1;
                int[][] indexs = trainingData.shuffle();
                //				int[][] indexs = trainingData.order();
                this.network.RUN_MODEL = RunModel.TRAIN;
                float train_loss = 0.0f;
                /**
                 * 遍历整个训练集

                 */
                for (int it = 0; it < indexs.length; it++) {
                    long start = System.nanoTime();
                    if (Math.abs(this.currentError) <= this.error) {
                        break;
                    }
                    trainingData.loadData(indexs[it], input);
                    JCudaDriver.cuCtxSynchronize();
                    /**
                     * forward

                     */
                    Tensor output = network.forward(input);
                    Tensor lpipsOutput = lpips.forward(output, input);
                    /**
                     * current time error

                     */
                    network.tensorOP.mean(lpipsOutput, 0, lpipLoss);
                    /**
                     * loss

                     */
                    float loss = network.totalLoss(output, input);
                    /**
                     * loss diff
                     */
                    this.lossDiff = network.lossDiff(output, input);
                    lpips.back(lpipsLossDiff);
                    network.tensorOP.add(this.lossDiff, lpips.lpips.diff, this.lossDiff);
                    Tensor discFakePred = null;
                    if (stepCount > discStepStart) {
                        discFakePred = disc.forward(output);
                        if (ones == null) {
                            ones = discFakePred.createLike(1);
                            zeros = discFakePred.createLike(0);
                        }
                        Tensor fakeGLoss = disc.loss(discFakePred, ones);
                        Tensor fakeGDiff = disc.lossDiff(discFakePred, ones);
                        network.tensorOP.mul(fakeGDiff, disc_weight, fakeGDiff);
                        disc.back(fakeGDiff);
                        float fakeGloss = MatrixOperation.sum(fakeGLoss.syncHost()) / batchSize;
                        loss += fakeGloss * disc_weight;
                        network.tensorOP.add(this.lossDiff, disc.disc.diff, this.lossDiff);
                    }
                    /**
                     * back
                     */
                    network.back(this.lossDiff);
                    //					System.out.println(JsonUtils.toJson(this.loss.syncHost()));
                    /**
                     * update
                     */
                    network.update();
                    /**
                     * train discriminator
                     */
                    if (stepCount > discStepStart) {
                        Tensor fakeDLoss = disc.loss(discFakePred, zeros);
                        Tensor fakeDDiff = disc.lossDiff(discFakePred, zeros);
                        network.tensorOP.mul(fakeDDiff, disc_weight, fakeDDiff);
                        disc.back(fakeDDiff);
                        /**
                         * 梯度叠加
                         */
                        disc.accGrad(2);
                        Tensor discRealPred = disc.forward(input);
                        Tensor realDLoss = disc.loss(discRealPred, ones);
                        Tensor realDDiff = disc.lossDiff(discRealPred, ones);
                        network.tensorOP.mul(realDDiff, disc_weight, realDDiff);
                        disc.back(realDDiff);
                        /**
                         * 梯度叠加
                         */
                        disc.accGrad(2);
                        disc.update();
                        float discLoss = (MatrixOperation.sum(fakeDLoss.syncHost()) + MatrixOperation.sum(realDLoss.syncHost())) / this.batchSize * disc_weight;
                        System.out.println("discLoss:" + discLoss);
                    }
                    JCudaDriver.cuCtxSynchronize();
                    /**
                     * current time error

                     */
                    float ploss = lpipLoss.syncHost()[0];
                    System.out.println("ploss:" + ploss);
                    this.currentError = loss + perceptual_weight * ploss;
                    train_loss += this.currentError;
                    String msg = "training[" + this.trainIndex + "]{" + it + "} (lr:" + this.network.learnRate + ") train_loss:" + this.currentError + " [costTime:" + (System.nanoTime() - start) / 1e6 + "ms.]";
                    System.out.println(msg);
                    this.batchIndex++;
                    /**
                     * update learning rate

                     */
                    this.updateLR(this.lr_step);
                    updateLRDynamic(i * trainingData.count_it + it, this.trainTime * trainingData.count_it, 1e-6f);
                    stepCount++;
                }
                System.out.println("training[" + this.trainIndex + "] train loss:{" + train_loss / indexs.length + "} ");
                /**
                 * update learning rate

                 */
                //				this.updateLR(this.lr_step);
                if (i % 1 == 0) {
                    /**
                     * showImage

                     */
                    this.network.RUN_MODEL = RunModel.TEST;
                    Tensor output = network.forward(input);
                    output.syncHost();
                    output.data = MatrixOperation.clampSelf(output.data, -1, 1);
                    /**
                     * print image

                     */
                    showImgs("/omega/test/vqvae/anime/", output, i + "", trainingData.mean, trainingData.std);
                }
                if (i > 0 && i % 10 == 0) {
                    String save_model_path = "/omega/models/anime_vqvae2_256_" + i + ".model";
                    ModelUtils.saveModel(network, save_model_path);
                }
            }
            /**
             * 停止训练

             */
            System.out.println("training finish. [" + this.trainIndex + "] finalError:" + this.currentError);
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void trainTinyVQVAE2_lpips_nogan(DiffusionImageDataLoader trainingData, LPIPS lpips) {
        // TODO Auto-generated method stub
        try {
            float perceptual_weight = 1;
            CUDAModules.initCUDAFunctions();
            TinyVQVAE2 network = (TinyVQVAE2) this.network;
            this.dataSize = trainingData.number;
            if (isWarmUp()) {
                this.network.learnRate = (float) (this.lr * Math.pow(batchIndex * 1.0f / burnIn * 1.0f, power));
            }
            Tensor input = new Tensor(batchSize, this.network.getChannel(), this.network.getHeight(), this.network.getWidth(), true);
            Tensor lpipLoss = new Tensor(1, 1, 1, 1, true);
            Tensor lpipsLossDiff = new Tensor(batchSize, 1, 1, 1, MatrixUtils.val(batchSize, 1.0f / batchSize), true);
            for (int i = 0; i < this.trainTime; i++) {
                if (this.trainIndex >= this.minTrainTime) {
                    break;
                }
                this.trainIndex = i + 1;
                int[][] indexs = trainingData.shuffle();
                //				int[][] indexs = trainingData.order();
                this.network.RUN_MODEL = RunModel.TRAIN;
                float train_loss = 0.0f;
                /**
                 * 遍历整个训练集

                 */
                for (int it = 0; it < indexs.length; it++) {
                    long start = System.nanoTime();
                    if (Math.abs(this.currentError) <= this.error) {
                        break;
                    }
                    trainingData.loadData(indexs[it], input);
                    JCudaDriver.cuCtxSynchronize();
                    /**
                     * forward

                     */
                    Tensor output = network.forward(input);
                    Tensor lpipsOutput = lpips.forward(output, input);
                    /**
                     * current time error

                     */
                    network.tensorOP.mean(lpipsOutput, 0, lpipLoss);
                    /**
                     * loss

                     */
                    float loss = network.totalLoss(output, input);
                    /**
                     * loss diff

                     */
                    this.lossDiff = network.lossDiff(output, input);
                    lpips.back(lpipsLossDiff);
                    network.tensorOP.add(this.lossDiff, lpips.lpips.diff, this.lossDiff);
                    /**
                     * back

                     */
                    network.back(this.lossDiff);
                    //					System.out.println(JsonUtils.toJson(this.loss.syncHost()));
                    /**
                     * update

                     */
                    network.update();
                    JCudaDriver.cuCtxSynchronize();
                    /**
                     * current time error

                     */
                    float ploss = lpipLoss.syncHost()[0];
                    System.out.println("ploss:" + ploss);
                    this.currentError = loss + perceptual_weight * ploss;
                    train_loss += this.currentError;
                    String msg = "training[" + this.trainIndex + "]{" + it + "} (lr:" + this.network.learnRate + ") train_loss:" + this.currentError + " [costTime:" + (System.nanoTime() - start) / 1e6 + "ms.]";
                    System.out.println(msg);
                    this.batchIndex++;
                    /**
                     * update learning rate

                     */
                    this.updateLR(this.lr_step);
                    updateLRDynamic(i * trainingData.count_it + it, this.trainTime * trainingData.count_it, 1e-6f);
                }
                System.out.println("training[" + this.trainIndex + "] train loss:{" + train_loss / indexs.length + "} ");
                /**
                 * update learning rate

                 */
                //				this.updateLR(this.lr_step);
                if (i % 1 == 0) {
                    /**
                     * showImage

                     */
                    this.network.RUN_MODEL = RunModel.TEST;
                    Tensor output = network.forward(input);
                    output.syncHost();
                    output.data = MatrixOperation.clampSelf(output.data, -1, 1);
                    /**
                     * print image

                     */
                    showImgs("/omega/test/vqvae/anime/", output, i + "", trainingData.mean, trainingData.std);
                }
                if (i > 0 && i % 10 == 0) {
                    String save_model_path = "/omega/models/anime_vqvae2_256_" + i + ".model";
                    ModelUtils.saveModel(network, save_model_path);
                }
            }
            /**
             * 停止训练

             */
            System.out.println("training finish. [" + this.trainIndex + "] finalError:" + this.currentError);
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void trainVQVAE2_lpips_nogan(DiffusionImageDataLoader trainingData, LPIPS lpips) {
        // TODO Auto-generated method stub
        try {
            float perceptual_weight = 1;
            VQVAE2 network = (VQVAE2) this.network;
            this.dataSize = trainingData.number;
            if (isWarmUp()) {
                this.network.learnRate = (float) (this.lr * Math.pow(batchIndex * 1.0f / burnIn * 1.0f, power));
            }
            Tensor input = new Tensor(batchSize, this.network.getChannel(), this.network.getHeight(), this.network.getWidth(), true);
            Tensor lpipLoss = new Tensor(1, 1, 1, 1, true);
            Tensor lpipsLossDiff = new Tensor(batchSize, 1, 1, 1, MatrixUtils.val(batchSize, 1.0f / batchSize), true);
            for (int i = 0; i < this.trainTime; i++) {
                if (this.trainIndex >= this.minTrainTime) {
                    break;
                }
                this.trainIndex = i + 1;
                int[][] indexs = trainingData.shuffle();
                //				int[][] indexs = trainingData.order();
                this.network.RUN_MODEL = RunModel.TRAIN;
                float train_loss = 0.0f;
                /**
                 * 遍历整个训练集
                 */
                for (int it = 0; it < indexs.length; it++) {
                    long start = System.nanoTime();
                    if (Math.abs(this.currentError) <= this.error) {
                        break;
                    }
                    trainingData.loadData(indexs[it], input);
                    JCudaDriver.cuCtxSynchronize();
                    /**
                     * forward
                     */
                    Tensor output = network.forward(input);
                    Tensor lpipsOutput = lpips.forward(output, input);
                    /**
                     * current time error
                     */
                    network.tensorOP.mean(lpipsOutput, 0, lpipLoss);
                    /**
                     * loss
                     */
                    float loss = network.totalLoss(output, input);
                    /**
                     * loss diff
                     */
                    this.lossDiff = network.lossDiff(output, input);
                    lpips.back(lpipsLossDiff);
                    //					lossDiff.showDMByOffsetRed(0, 100, "lossDiff");
                    network.tensorOP.add(this.lossDiff, lpips.lpips.diff, this.lossDiff);
                    //					lossDiff.showDMByOffsetRed(0, 100, "lossDiff2");
                    /**
                     * back
                     */
                    network.back(this.lossDiff);
                    //					System.out.println(JsonUtils.toJson(this.loss.syncHost()));
                    /**
                     * update
                     */
                    network.update();
                    JCudaDriver.cuCtxSynchronize();
                    /**
                     * current time error

                     */
                    float ploss = lpipLoss.syncHost()[0];
                    System.out.println("ploss:" + ploss);
                    this.currentError = loss + perceptual_weight * ploss;
                    train_loss += this.currentError;
                    String msg = "training[" + this.trainIndex + "]{" + it + "/" + indexs.length + "} (lr:" + this.network.learnRate + ") train_loss:" + this.currentError + " [costTime:" + (System.nanoTime() - start) / 1e6 + "ms.]";
                    System.out.println(msg);
                    this.batchIndex++;
                    /**
                     * update learning rate
                     */
                    this.updateLR(this.lr_step);
                    updateLRDynamic(i * trainingData.count_it + it, this.trainTime * trainingData.count_it, 1e-6f);
                }
                System.out.println("training[" + this.trainIndex + "] train loss:{" + train_loss / indexs.length + "} ");
                /**
                 * update learning rate
                 */
                //				this.updateLR(this.lr_step);
                if (i % 10 == 0) {
                    /**
                     * showImage

                     */
                    this.network.RUN_MODEL = RunModel.TEST;
                    Tensor output = network.forward(input);
                    output.syncHost();
                    output.data = MatrixOperation.clampSelf(output.data, -1, 1);
                    /**
                     * print image
                     */
                    //					showImgs("/omega/test/vqvae/anime/", output, i + "", trainingData.mean, trainingData.std);
                    showImgs("H:\\vae_dataset\\pokemon-blip\\test256\\", output, i + "", trainingData.mean, trainingData.std);
                }
                if (i > 0 && i % 20 == 0) {
                    String save_model_path = "/omega/models/anime_vqvae2_256_" + i + ".model";
                    ModelUtils.saveModel(network, save_model_path);
                }
            }
            /**
             * 停止训练

             */
            System.out.println("training finish. [" + this.trainIndex + "] finalError:" + this.currentError);
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
    
    public void trainVQVAE2_lpips_patchGANDisc(DiffusionImageDataLoader trainingData, LPIPS lpips, PatchGANDiscriminator disc,int discStepStart) {
        // TODO Auto-generated method stub
        try {
            float perceptual_weight = 1f;
            float disc_weight = 0.5f;
            CUDAModules.initCUDAFunctions();
            VQVAE2 network = (VQVAE2) this.network;
            this.dataSize = trainingData.number;
            if (isWarmUp()) {
                this.network.learnRate = (float) (this.lr * Math.pow(batchIndex * 1.0f / burnIn * 1.0f, power));
            }
            Tensor input = new Tensor(batchSize, this.network.getChannel(), this.network.getHeight(), this.network.getWidth(), true);
            Tensor lpipLoss = new Tensor(1, 1, 1, 1, true);
            Tensor lpipsLossDiff = new Tensor(batchSize, 1, 1, 1, MatrixUtils.val(batchSize, 1.0f / batchSize), true);
            
            Tensor discFakePred = null;
            Tensor fakeGLoss = null;
            Tensor fakeGDiff = null;
            
            Tensor discLoss = null;
            Tensor discReadDiff = null;
            Tensor discFakeDiff = null;
            
            int stepCount = 0;

            for (int i = 0; i < this.trainTime; i++) {
                if (this.trainIndex >= this.minTrainTime) {
                    break;
                }
                this.trainIndex = i + 1;
                int[][] indexs = trainingData.shuffle();
                //				int[][] indexs = trainingData.order();
                this.network.RUN_MODEL = RunModel.TRAIN;
                float train_loss = 0.0f;
                /**
                 * 遍历整个训练集
                 */
                for (int it = 0; it < indexs.length; it++) {
                    long start = System.nanoTime();
                    if (Math.abs(this.currentError) <= this.error) {
                        break;
                    }
                    trainingData.loadData(indexs[it], input);
                    JCudaDriver.cuCtxSynchronize();
                    /**
                     * forward
                     */
                    Tensor output = network.forward(input);
                    Tensor lpipsOutput = lpips.forward(output, input);
              
                    /**
                     * current time error
                     */
                    network.tensorOP.mean(lpipsOutput, 0, lpipLoss);
                    /**
                     * loss
                     */
                    float loss = network.totalLoss(output, input);
                    /**
                     * loss diff
                     */
                    this.lossDiff = network.lossDiff(output, input);
                    
                    lpips.back(lpipsLossDiff);
//                    network.tensorOP.mul(lpips.lpips.diff, perceptual_weight, lpips.lpips.diff);
                    network.tensorOP.add(this.lossDiff, lpips.lpips.diff, this.lossDiff);

                    /**
                     * add discriminator fakePred
                     */
                    if (stepCount > discStepStart) {
                        Tensor discOut = disc.forward(output);
                        if (discFakePred == null) {
                        	discFakePred = discOut.createLike(); 
                            fakeGLoss = new Tensor(1, 1, 1, 1, true);
                            fakeGDiff = discFakePred.createLike(1 / discFakePred.number * disc_weight);
                        }
                        discOut.copyGPU(discFakePred);
                        disc.hingeGLoss(discFakePred, fakeGLoss);
                        disc.back(fakeGDiff);
                        float fakeGloss = MatrixOperation.sum(fakeGLoss.syncHost()) / batchSize;
                        loss += fakeGloss * disc_weight;
                        System.out.println("fakeGloss:"+fakeGloss * disc_weight);
                        network.tensorOP.add(this.lossDiff, disc.disc.diff, this.lossDiff);
                    }
                    /**
                     * back
                     */
                    network.back(this.lossDiff);
                    //					System.out.println(JsonUtils.toJson(this.loss.syncHost()));
                    /**
                     * update
                     */
                    network.update();
                    /**
                     * train discriminator
                     */
                    if (stepCount > discStepStart) {
                    	if(discLoss ==  null) {
                    		discLoss = discFakePred.createLike(0);
                    		discReadDiff = discFakePred.createLike(0); 
                    		discFakeDiff = discFakePred.createLike(0);
                    	}
                    	Tensor discRealPred = disc.forward(input);
                        disc.hingeDLoss(discRealPred, discFakePred, discLoss);
                        disc.hingeDLossBack(discRealPred, discFakePred, discReadDiff, discFakeDiff);
                        network.tensorOP.mul(discReadDiff, disc_weight, discReadDiff);
                        network.tensorOP.mul(discFakeDiff, disc_weight, discFakeDiff);
                        
                        disc.back(discReadDiff);
                        disc.accGrad(2);
                        
                    	Tensor discFakeOut = disc.forward(output);
                        disc.back(discFakeDiff);
                        disc.accGrad(2);

                        disc.update();
                        float discLossVal = MatrixOperation.sum(discLoss.syncHost()) * disc_weight;
                        System.out.println("discLoss:" + discLossVal);
                    }
                    JCudaDriver.cuCtxSynchronize();
                    /**
                     * current time error
                     */
                    float ploss = lpipLoss.syncHost()[0];
                    System.out.println("ploss:" + ploss);
                    this.currentError = loss + perceptual_weight * ploss;
                    train_loss += this.currentError;
                    String msg = "training[" + this.trainIndex + "]{" + it + "/" + indexs.length + "} (lr:" + this.network.learnRate + ") train_loss:" + this.currentError + " [costTime:" + (System.nanoTime() - start) / 1e6 + "ms.]";
                    System.out.println(msg);
                    this.batchIndex++;
                    /**
                     * update learning rate
                     */
                    this.updateLR(this.lr_step);
                    updateLRDynamic(i * trainingData.count_it + it, this.trainTime * trainingData.count_it, 1e-6f);
                    stepCount++;
                }
                System.out.println("training[" + this.trainIndex + "] train loss:{" + train_loss / indexs.length + "} ");
                /**
                 * update learning rate
                 */
                //				this.updateLR(this.lr_step);
                if (i % 1 == 0) {
                    /**
                     * showImage

                     */
                    this.network.RUN_MODEL = RunModel.TEST;
                    Tensor output = network.forward(input);
                    output.syncHost();
                    output.data = MatrixOperation.clampSelf(output.data, -1, 1);
                    /**
                     * print image

                     */
                    //					showImgs("/omega/test/vqvae/anime/", output, i + "", trainingData.mean, trainingData.std);
                    showImgs("/omega/test/", output, i + "", trainingData.mean, trainingData.std);
                }
                if (i > 0 && i % 20 == 0) {
                    String save_model_path = "/omega/models/anime_vqvae2_256_" + i + ".model";
                    ModelUtils.saveModel(network, save_model_path);
                }
            }
            /**
             * 停止训练

             */
            System.out.println("training finish. [" + this.trainIndex + "] finalError:" + this.currentError);
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
    
    public void trainTinyVQVAE_lpips_patchGANDisc(DiffusionImageDataLoader trainingData, LPIPS lpips, PatchGANDiscriminator disc, int discStepStart) {
        // TODO Auto-generated method stub
        try {
            float perceptual_weight = 1;
            float disc_weight = 0.5f;
            CUDAModules.initCUDAFunctions();
            TinyVQVAE network = (TinyVQVAE) this.network;
            this.dataSize = trainingData.number;
            if (isWarmUp()) {
                this.network.learnRate = (float) (this.lr * Math.pow(batchIndex * 1.0f / burnIn * 1.0f, power));
            }
            Tensor input = new Tensor(batchSize, this.network.getChannel(), this.network.getHeight(), this.network.getWidth(), true);
            Tensor lpipLoss = new Tensor(1, 1, 1, 1, true);
            Tensor lpipsLossDiff = new Tensor(batchSize, 1, 1, 1, MatrixUtils.val(batchSize, 1.0f / batchSize), true);
            Tensor ones = null;
            Tensor zeros = null;
            int stepCount = 0;
            for (int i = 0; i < this.trainTime; i++) {
                if (this.trainIndex >= this.minTrainTime) {
                    break;
                }
                this.trainIndex = i + 1;
                int[][] indexs = trainingData.shuffle();
                //				int[][] indexs = trainingData.order();
                this.network.RUN_MODEL = RunModel.TRAIN;
                float train_loss = 0.0f;
                /**
                 * 遍历整个训练集

                 */
                for (int it = 0; it < indexs.length; it++) {
                    long start = System.nanoTime();
                    if (Math.abs(this.currentError) <= this.error) {
                        break;
                    }
                    trainingData.loadData(indexs[it], input);
                    JCudaDriver.cuCtxSynchronize();
                    /**
                     * forward

                     */
                    Tensor output = network.forward(input);
                    Tensor lpipsOutput = lpips.forward(output, input);
                    /**
                     * current time error

                     */
                    network.tensorOP.mean(lpipsOutput, 0, lpipLoss);
                    /**
                     * loss

                     */
                    float loss = network.totalLoss(output, input);
                    /**
                     * loss diff

                     */
                    this.lossDiff = network.lossDiff(output, input);
                    lpips.back(lpipsLossDiff);
                    network.tensorOP.add(this.lossDiff, lpips.lpips.diff, this.lossDiff);
                    Tensor discFakePred = null;
                    if (stepCount > discStepStart) {

                        discFakePred = disc.forward(output);
                        if (ones == null) {
                            ones = discFakePred.createLike(1.0f);
                            zeros = discFakePred.createLike(0.0f);
                        }
                        Tensor discFakeLoss = disc.loss(discFakePred, ones);
                        Tensor discFakeDiff = disc.lossDiff(discFakePred, ones);
                        network.tensorOP.mul(discFakeDiff, disc_weight, discFakeDiff);
                        disc.back(discFakeDiff);
                        /**
                         * 梯度叠加
                         */
                        disc.accGrad(3);
                        loss += MatrixOperation.sum(discFakeLoss.syncHost()) / this.batchSize * disc_weight;
                        network.tensorOP.add(this.lossDiff, disc.disc.diff, this.lossDiff);
                    }
                    /**
                     * back
                     */
                    network.back(this.lossDiff);
                    //					System.out.println(JsonUtils.toJson(this.loss.syncHost()));
                    /**
                     * update
                     */
                    network.update();
                    /**
                     * train discriminator
                     */
                    if (stepCount > discStepStart) {
                        Tensor discFakeLoss = disc.loss(discFakePred, zeros);
                        Tensor discFakeDiff = disc.lossDiff(discFakePred, zeros);
                        disc.back(discFakeDiff);
                        /**
                         * 梯度叠加
                         */
                        disc.accGrad(3);
                        Tensor discRealPred = disc.forward(input);
                        Tensor discRealLoss = disc.loss(discRealPred, ones);
                        Tensor discRealDiff = disc.lossDiff(discRealPred, ones);
                        disc.back(discRealDiff);
                        /**
                         * 梯度叠加
                         */
                        disc.accGrad(3);
                        disc.update();
                        float discLoss = (MatrixOperation.sum(discFakeLoss.syncHost()) + MatrixOperation.sum(discRealLoss.syncHost())) / this.batchSize * disc_weight / 2;
                        System.out.println("discLoss:" + discLoss);
                    }
                    JCudaDriver.cuCtxSynchronize();
                    /**
                     * current time error

                     */
                    float ploss = lpipLoss.syncHost()[0];
                    System.out.println("ploss:" + ploss);
                    this.currentError = loss + perceptual_weight * ploss;
                    train_loss += this.currentError;
                    String msg = "training[" + this.trainIndex + "]{" + it + "} (lr:" + this.network.learnRate + ") train_loss:" + this.currentError + " [costTime:" + (System.nanoTime() - start) / 1e6 + "ms.]";
                    System.out.println(msg);
                    this.batchIndex++;
                    /**
                     * update learning rate
                     */
                    this.updateLR(this.lr_step);
                    updateLRDynamic(i * trainingData.count_it + it, this.trainTime * trainingData.count_it, 1e-6f);
                    stepCount++;
                }
                System.out.println("training[" + this.trainIndex + "] train loss:{" + train_loss / indexs.length + "} ");
                /**
                 * update learning rate

                 */
                //				this.updateLR(this.lr_step);
                if (i % 1 == 0) {
                    /**
                     * showImage

                     */
                    this.network.RUN_MODEL = RunModel.TEST;
                    Tensor output = network.forward(input);
                    output.syncHost();
                    output.data = MatrixOperation.clampSelf(output.data, -1, 1);
                    /**
                     * print image

                     */
                    showImgs("H:\\vae_dataset\\pokemon-blip\\test128\\", output, i + "", trainingData.mean, trainingData.std);
                }
            }
            /**
             * 停止训练

             */
            System.out.println("training finish. [" + this.trainIndex + "] finalError:" + this.currentError);
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void trainDCAE_lpips(DiffusionImageDataLoader trainingData, LPIPS lpips) {
        // TODO Auto-generated method stub
        try {
            DC_AE network = (DC_AE) this.network;
            this.dataSize = trainingData.number;
            if (isWarmUp()) {
                this.network.learnRate = (float) (this.lr * Math.pow(batchIndex * 1.0f / burnIn * 1.0f, power));
            }
            Tensor input = new Tensor(batchSize, this.network.getChannel(), this.network.getHeight(), this.network.getWidth(), true);
            Tensor lpipLoss = new Tensor(1, 1, 1, 1, true);
            Tensor lpipsLossDiff = new Tensor(batchSize, 1, 1, 1, MatrixUtils.val(batchSize, 1.0f / batchSize), true);
            for (int i = 0; i < this.trainTime; i++) {
                if (this.trainIndex >= this.minTrainTime) {
                    break;
                }
                this.trainIndex = i + 1;
                int[][] indexs = trainingData.shuffle();
                //				int[][] indexs = trainingData.order();
                this.network.RUN_MODEL = RunModel.TRAIN;
                float train_loss = 0.0f;
                /**
                 * 遍历整个训练集
                 */
                for (int it = 0; it < indexs.length; it++) {
                    long start = System.nanoTime();
                    if (Math.abs(this.currentError) <= this.error) {
                        break;
                    }
                    trainingData.loadData(indexs[it], input);
                    JCudaDriver.cuCtxSynchronize();
                    /**
                     * forward
                     */
                    Tensor output = network.forward(input);
                    Tensor lpipsOutput = lpips.forward(output, input);
                    /**
                     * current time error
                     */
                    network.tensorOP.mean(lpipsOutput, 0, lpipLoss);
                    /**
                     * loss
                     */
                    this.loss = network.loss(output, input);
                    /**
                     * loss diff
                     */
                    this.lossDiff = network.lossDiff(output, input);
                    lpips.back(lpipsLossDiff);
                    //					lossDiff.showDMByOffsetRed(0, 100, "lossDiff");
                    network.tensorOP.add(this.lossDiff, lpips.lpips.diff, this.lossDiff);
                    //					lossDiff.showDMByOffsetRed(0, 100, "lossDiff2");
                    /**
                     * back
                     */
                    network.back(this.lossDiff);
                    //					System.out.println(JsonUtils.toJson(this.loss.syncHost()));
                    /**
                     * update
                     */
                    network.update();
                    JCudaDriver.cuCtxSynchronize();
                    /**
                     * current time error
                     */
                    float ploss = lpipLoss.syncHost()[0];
                    System.out.println("ploss:" + ploss);
                    this.currentError = MatrixOperation.sum(loss.syncHost())/loss.number + ploss;
                    train_loss += this.currentError;
                    String msg = "training[" + this.trainIndex + "]{" + it + "/" + indexs.length + "} (lr:" + this.network.learnRate + ") train_loss:" + this.currentError + " [costTime:" + (System.nanoTime() - start) / 1e6 + "ms.]";
                    System.out.println(msg);
                    this.batchIndex++;
                    /**
                     * update learning rate
                     */
                    this.updateLR(this.lr_step);
                    updateLRDynamic(i * trainingData.count_it + it, this.trainTime * trainingData.count_it, 1e-6f);
                }
                System.out.println("training[" + this.trainIndex + "] train loss:{" + train_loss / indexs.length + "} ");
                /**
                 * update learning rate
                 */
                //				this.updateLR(this.lr_step);
                if (i % 10 == 0) {
                    /**
                     * showImage
                     */
                    this.network.RUN_MODEL = RunModel.TEST;
                    Tensor output = network.forward(input);
                    output.syncHost();
                    output.data = MatrixOperation.clampSelf(output.data, -1, 1);
                    /**
                     * print image
                     */
                    //					showImgs("/omega/test/vqvae/anime/", output, i + "", trainingData.mean, trainingData.std);
                    showImgs("H:\\vae_dataset\\pokemon-blip\\test128\\", output, i + "", trainingData.mean, trainingData.std);
                }
//                if (i > 0 && i % 20 == 0) {
//                    String save_model_path = "/omega/models/anime_vqvae2_256_" + i + ".model";
//                    ModelUtils.saveModel(network, save_model_path);
//                }
            }
            /**
             * 停止训练

             */
            System.out.println("training finish. [" + this.trainIndex + "] finalError:" + this.currentError);
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
    
    public void trainLPIPIS(DiffusionImageDataLoader trainingData) {
        // TODO Auto-generated method stub
        try {
            CUDAModules.initCUDAFunctions();
            LPIPS network = (LPIPS) this.network;
            this.dataSize = trainingData.number;
            if (isWarmUp()) {
                this.network.learnRate = (float) (this.lr * Math.pow(batchIndex * 1.0f / burnIn * 1.0f, power));
            }
            Tensor input = new Tensor(batchSize, this.network.getChannel(), this.network.getHeight(), this.network.getWidth(), true);
            int len = batchSize * 3 * 256 * 256;
            Tensor label = new Tensor(batchSize, this.network.getChannel(), this.network.getHeight(), this.network.getWidth(), MatrixUtils.order(len, 0.0001f, 0.0001f), true);
            Tensor loss = new Tensor(1, 1, 1, 1, true);
            Tensor lossDiff = new Tensor(batchSize, 1, 1, 1, MatrixUtils.val(batchSize, 1.0f / batchSize), true);
            for (int i = 0; i < this.trainTime; i++) {
                if (this.trainIndex >= this.minTrainTime) {
                    break;
                }
                this.trainIndex = i + 1;
                //				int[][] indexs = trainingData.shuffle();
                int[][] indexs = trainingData.order();
                this.network.RUN_MODEL = RunModel.TRAIN;
                float train_loss = 0.0f;
                /**
                 * 遍历整个训练集

                 */
                for (int it = 0; it < indexs.length; it++) {
                    long start = System.nanoTime();
                    if (Math.abs(this.currentError) <= this.error) {
                        break;
                    }
//                    trainingData.loadData(indexs[it], label);
//                    JCudaDriver.cuCtxSynchronize();
                    network.tensorOP.mul(label, 0.9f, input);
                    /**
                     * forward
                     */
                    Tensor output = network.forward(label, input);
                    System.err.println("output:");
                    output.showDM();
                    /**
                     * current time error
                     */
                    loss.clearGPU();
                    network.tensorOP.mean(output, 0, loss);
                    this.currentError = loss.syncHost()[0];
                    System.err.println(this.currentError);
                    /**
                     * back
                     */
                    network.back(lossDiff);
                    JCudaDriver.cuCtxSynchronize();
                    train_loss += this.currentError;
                    String msg = "training[" + this.trainIndex + "]{" + it + "} (lr:" + this.network.learnRate + ") train_loss:" + this.currentError + " [costTime:" + (System.nanoTime() - start) / 1e6 + "ms.]";
                    System.out.println(msg);
                    this.batchIndex++;
                }
                System.out.println("training[" + this.trainIndex + "] train loss:{" + train_loss / indexs.length + "} ");
            }
            /**
             * 停止训练

             */
            System.out.println("training finish. [" + this.trainIndex + "] finalError:" + this.currentError);
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void trainLPatchGANDisc(DiffusionImageDataLoader trainingData) {
        // TODO Auto-generated method stub
        try {
            CUDAModules.initCUDAFunctions();
            PatchGANDiscriminator network = (PatchGANDiscriminator) this.network;
            this.dataSize = trainingData.number;
            if (isWarmUp()) {
                this.network.learnRate = (float) (this.lr * Math.pow(batchIndex * 1.0f / burnIn * 1.0f, power));
            }
            Tensor input = new Tensor(batchSize, this.network.getChannel(), this.network.getHeight(), this.network.getWidth(), true);
            Tensor label = null;
            for (int i = 0; i < this.trainTime; i++) {
                if (this.trainIndex >= this.minTrainTime) {
                    break;
                }
                this.trainIndex = i + 1;
                int[][] indexs = trainingData.shuffle();
                //				int[][] indexs = trainingData.order();
                this.network.RUN_MODEL = RunModel.TRAIN;
                float train_loss = 0.0f;
                /**
                 * 遍历整个训练集

                 */
                for (int it = 0; it < indexs.length; it++) {
                    long start = System.nanoTime();
                    if (Math.abs(this.currentError) <= this.error) {
                        break;
                    }
                    trainingData.loadData(indexs[it], input);
                    JCudaDriver.cuCtxSynchronize();
                    /**
                     * forward

                     */
                    Tensor output = network.forward(input);
                    //					System.err.println("output:");
                    //					output.showDM();
                    if (label == null) {
                        label = output.createLike(1.0f);
                    }
                    /**
                     * loss

                     */
                    this.loss = network.loss(output, label);
                    /**
                     * loss diff

                     */
                    this.lossDiff = network.lossDiff(output, label);
                    //					this.lossDiff.showDM();
                    /**
                     * back

                     */
                    network.back(lossDiff);
                    /**
                     * update

                     */
                    network.update();
                    JCudaDriver.cuCtxSynchronize();
                    /**
                     * current time error

                     */
                    if (this.loss.isHasGPU()) {
                        this.currentError = MatrixOperation.sum(this.loss.syncHost()) / this.batchSize;
                        //						System.out.println(JsonUtils.toJson(this.loss.syncHost()));
                    } else {
                        this.currentError = MatrixOperation.sum(this.loss.data) / this.batchSize;
                    }
                    train_loss += this.currentError;
                    String msg = "training[" + this.trainIndex + "]{" + it + "} (lr:" + this.network.learnRate + ") train_loss:" + this.currentError + " [costTime:" + (System.nanoTime() - start) / 1e6 + "ms.]";
                    System.out.println(msg);
                    this.batchIndex++;
                }
                System.out.println("training[" + this.trainIndex + "] train loss:{" + train_loss / indexs.length + "} ");
            }
            /**
             * 停止训练

             */
            System.out.println("training finish. [" + this.trainIndex + "] finalError:" + this.currentError);
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void trainSD(SDImageDataLoader trainingData, TinyVQVAE2 vae, ClipText clip) {
        // TODO Auto-generated method stub
        try {
            CUDAModules.initCUDAFunctions();
            DiffusionUNetCond network = (DiffusionUNetCond) this.network;
            this.dataSize = trainingData.number;
            if (isWarmUp()) {
                this.network.learnRate = (float) (this.lr * Math.pow(batchIndex * 1.0f / burnIn * 1.0f, power));
            }
            Tensor input = new Tensor(batchSize, 3, trainingData.img_h, trainingData.img_w, true);
            Tensor label = new Tensor(batchSize * network.maxContextLen, 1, 1, 1, true);
            Tensor mask = new Tensor(batchSize, 1, 1, network.maxContextLen, true);
            String[] labels = new String[batchSize];
            Tensor context = null;
            float beta_1 = 0.00085f;
            float beta_T = 0.012f;
            int T = 1000;
            float scale_factor = 0.18215f;
            //			float scale_factor = 0.143262f;
            Tensor t = new Tensor(batchSize, 1, 1, 1, true);
            Tensor a = new Tensor(batchSize, 1, 1, 1, true);
            Tensor b = new Tensor(batchSize, 1, 1, 1, true);
            Tensor noise = new Tensor(batchSize, network.inChannel, network.height, network.width, true);
            Tensor condInput = null;
            Tensor latend = null;
            float[] betas = MatrixUtils.linspace(beta_1, beta_T, T);
            float[] alphas = MatrixOperation.subtraction(1, betas);
            float[] alphas_bar = MatrixUtils.cumprod(alphas);
            float[] sqrt_alphas_bar = MatrixOperation.sqrt(alphas_bar);
            float[] sqrt_one_minus_alphas_bar = MatrixOperation.sqrt(MatrixOperation.subtraction(1, alphas_bar));
            for (int i = 0; i < this.trainTime; i++) {
                if (this.trainIndex >= this.minTrainTime) {
                    break;
                }
                this.trainIndex = i + 1;
                int[][] indexs = trainingData.shuffle();
                //				int[][] indexs = trainingData.order();
                this.network.RUN_MODEL = RunModel.TRAIN;
                float train_loss = 0.0f;
                /**
                 * 遍历整个训练集

                 */
                for (int it = 0; it < indexs.length; it++) {
                    long start = System.nanoTime();
                    if (Math.abs(this.currentError) <= this.error) {
                        break;
                    }
                    int[] t_data = RandomUtils.randomInt(0, T - 1, batchSize);
                    //					int[] t_data = new int[] {100, 902, 31, 698};
                    //					System.out.println(JsonUtils.toJson(t_data));
                    t.setData(t_data);
                    //					t.showDM();
                    float[] exsa1 = MatrixUtils.gather(sqrt_alphas_bar, t_data);
                    float[] exsa2 = MatrixUtils.gather(sqrt_one_minus_alphas_bar, t_data);
                    a.setData(exsa1);
                    b.setData(exsa2);
                    trainingData.loadData(indexs[it], input, label, mask, noise, labels);
                    JCudaDriver.cuCtxSynchronize();
                    //					System.out.println("in");
                    /**
                     * get latend

                     */
                    //					input.showShape();
                    latend = vae.encode(input);
                    latend.showShape();
                    network.tensorOP.mul(latend, scale_factor, latend);
                    /**
                     * get context embd

                     */
                    condInput = clip.forward(label, mask);
                    //					latend.showDMByOffset(0, 100, "before latend");
                    /**
                     * latend add noise

                     */
                    trainingData.addNoise(a, b, latend, noise, network.cudaManager);
                    if (context == null) {
                        context = condInput.createLike();
                    }
                    network.tensorOP.mul(condInput, condInput.norm(network.tensorOP), context);
                    network.tensorOP.div(condInput, context, context);
                    //					context.showDMByOffset(0, 100, "condInput");
                    //					latend.showDMByOffset(0, 100, "after latend");
                    /**
                     * forward

                     */
                    Tensor output = network.forward(latend, t, context);
                    //					output.showDMByOffset(0, 100, "output");
                    //					output.showDM();
                    /**
                     * loss

                     */
                    this.loss = network.loss(output, noise);
                    /**
                     * loss diff

                     */
                    this.lossDiff = network.lossDiff(output, noise);
                    /**
                     * back

                     */
                    network.back(this.lossDiff);
                    //					System.out.println(JsonUtils.toJson(this.loss.syncHost()));
                    /**
                     * update

                     */
                    network.update();
                    JCudaDriver.cuCtxSynchronize();
                    /**
                     * current time error

                     */
                    if (this.loss.isHasGPU()) {
                        this.currentError = MatrixOperation.sum(this.loss.syncHost()) / this.batchSize;
                        //						System.out.println(JsonUtils.toJson(this.loss.syncHost()));
                    } else {
                        this.currentError = MatrixOperation.sum(this.loss.data) / this.batchSize;
                    }
                    train_loss += this.currentError;
                    String msg = "training[" + this.trainIndex + "]{" + it + "} (lr:" + this.network.learnRate + ") train_loss:" + this.currentError + " [costTime:" + (System.nanoTime() - start) / 1e6 + "ms.]";
                    System.out.println(msg);
                    this.batchIndex++;
                    //					if(it % 10 == 0) {
                    //						network.RUN_MODEL = RunModel.TEST;
                    //						System.out.println("start create test images.");
                    ////						testGaussianDiffusion(i + "_" + it, 200, input, noise);
                    //						testSD(i + "", 200, latend, t, condInput, network, vae);
                    //						System.out.println("finish create.");
                    ////						testGaussianDiffusion(x_t, t, T, beta_1, beta_T, testParams, trainingData.mean, trainingData.std);
                    //						network.RUN_MODEL = RunModel.TRAIN;
                    ////						this.network.learnRate = this.network.learnRate * 0.1f;
                    //					}
                }
                if (i % 1 == 0) {
                    network.RUN_MODEL = RunModel.TEST;
                    System.out.println("start create test images.");
                    //					testGaussianDiffusion(i + "_" + it, 200, input, noise);
                    //String it,Tensor noiseInput,Tensor t,Tensor condInput,Tensor input,DiffusionUNetCond network,TinyVQVAE2 vae,SDImageDataLoader trainingData
                    testSD(i + "", latend, t, context, network, vae, trainingData, labels);
                    //					testSD_DDPM(i + "", latend, t, context, network, vae, trainingData, labels);
                    //					testSD(i + "", 200, latend, t, context, input, network, vae, labels);
                    System.out.println("finish create.");
                    //					testGaussianDiffusion(x_t, t, T, beta_1, beta_T, testParams, trainingData.mean, trainingData.std);
                    network.RUN_MODEL = RunModel.TRAIN;
                    //					this.network.learnRate = this.network.learnRate * 0.1f;
                }
                System.out.println("training[" + this.trainIndex + "] train loss:{" + train_loss / indexs.length + "} ");
                /**
                 * update learning rate

                 */
                this.updateLR(this.lr_step);
            }
            /**
             * 停止训练

             */
            System.out.println("training finish. [" + this.trainIndex + "] finalError:" + this.currentError);
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void trainTinySD(SDImageDataLoader trainingData, TinyVQVAE2 vae, ClipText clip) {
        // TODO Auto-generated method stub
        try {
            CUDAModules.initCUDAFunctions();
            DiffusionUNetCond2 network = (DiffusionUNetCond2) this.network;
            this.dataSize = trainingData.number;
            if (isWarmUp()) {
                this.network.learnRate = (float) (this.lr * Math.pow(batchIndex * 1.0f / burnIn * 1.0f, power));
            }
            Tensor input = new Tensor(batchSize, 3, trainingData.img_h, trainingData.img_w, true);
            Tensor label = new Tensor(batchSize * network.maxContextLen, 1, 1, 1, true);
            Tensor mask = new Tensor(batchSize, 1, 1, network.maxContextLen, true);
            String[] labels = new String[batchSize];
            float beta_1 = 0.00085f;
            float beta_T = 0.012f;
            int T = 1000;
            float scale_factor = 0.18215f;
            //			float scale_factor = 0.143262f;
            Tensor t = new Tensor(batchSize, 1, 1, 1, true);
            Tensor a = new Tensor(batchSize, 1, 1, 1, true);
            Tensor b = new Tensor(batchSize, 1, 1, 1, true);
            Tensor noise = new Tensor(batchSize, network.inChannel, network.height, network.width, true);
            Tensor condInput = null;
            Tensor latend = null;
            float[] betas = MatrixUtils.linspace(beta_1, beta_T, T);
            float[] alphas = MatrixOperation.subtraction(1, betas);
            float[] alphas_bar = MatrixUtils.cumprod(alphas);
            float[] sqrt_alphas_bar = MatrixOperation.sqrt(alphas_bar);
            float[] sqrt_one_minus_alphas_bar = MatrixOperation.sqrt(MatrixOperation.subtraction(1, alphas_bar));
            for (int i = 0; i < this.trainTime; i++) {
                if (this.trainIndex >= this.minTrainTime) {
                    break;
                }
                this.trainIndex = i + 1;
                int[][] indexs = trainingData.shuffle();
                //				int[][] indexs = trainingData.order();
                this.network.RUN_MODEL = RunModel.TRAIN;
                float train_loss = 0.0f;
                /**
                 * 遍历整个训练集

                 */
                for (int it = 0; it < indexs.length; it++) {
                    long start = System.nanoTime();
                    if (Math.abs(this.currentError) <= this.error) {
                        break;
                    }
                    int[] t_data = RandomUtils.randomInt(0, T - 1, batchSize);
                    //					int[] t_data = new int[] {100, 902, 31, 698};
                    //					System.out.println(JsonUtils.toJson(t_data));
                    t.setData(t_data);
                    //					t.showDM();
                    float[] exsa1 = MatrixUtils.gather(sqrt_alphas_bar, t_data);
                    float[] exsa2 = MatrixUtils.gather(sqrt_one_minus_alphas_bar, t_data);
                    a.setData(exsa1);
                    b.setData(exsa2);
                    trainingData.loadData(indexs[it], input, label, mask, noise, labels);
                    JCudaDriver.cuCtxSynchronize();
                    //					System.out.println("in");
                    /**
                     * get latend

                     */
                    //					input.showShape();
                    latend = vae.encode(input);
                    latend.showShape();
                    network.tensorOP.mul(latend, scale_factor, latend);
                    /**
                     * get context embd

                     */
                    condInput = clip.forward(label, mask);
                    //					latend.showDMByOffset(0, 100, "before latend");
                    /**
                     * latend add noise

                     */
                    trainingData.addNoise(a, b, latend, noise, network.cudaManager);
                    //					if(context == null) {
                    //						context = condInput.createLike();
                    //					}
                    //
                    //					TensorOP.mul(condInput, condInput.norm(), context);
                    //
                    //					TensorOP.div(condInput, context, context);
                    //					context.showDMByOffset(0, 100, "condInput");
                    //					latend.showDMByOffset(0, 100, "after latend");
                    /**
                     * forward

                     */
                    Tensor output = network.forward(latend, t, condInput);
                    //					output.showDMByOffset(0, 100, "output");
                    //					output.showDM();
                    /**
                     * loss

                     */
                    this.loss = network.loss(output, noise);
                    /**
                     * loss diff

                     */
                    this.lossDiff = network.lossDiff(output, noise);
                    /**
                     * back

                     */
                    network.back(this.lossDiff);
                    //					System.out.println(JsonUtils.toJson(this.loss.syncHost()));
                    /**
                     * update

                     */
                    network.update();
                    JCudaDriver.cuCtxSynchronize();
                    /**
                     * current time error

                     */
                    if (this.loss.isHasGPU()) {
                        this.currentError = MatrixOperation.sum(this.loss.syncHost()) / this.batchSize;
                        //						System.out.println(JsonUtils.toJson(this.loss.syncHost()));
                    } else {
                        this.currentError = MatrixOperation.sum(this.loss.data) / this.batchSize;
                    }
                    train_loss += this.currentError;
                    String msg = "training[" + this.trainIndex + "]{" + it + "} (lr:" + this.network.learnRate + ") train_loss:" + this.currentError + " [costTime:" + (System.nanoTime() - start) / 1e6 + "ms.]";
                    System.out.println(msg);
                    this.batchIndex++;
                }
                if (i % 1 == 0) {
                    network.RUN_MODEL = RunModel.TEST;
                    System.out.println("start create test images.");
                    //					testGaussianDiffusion(i + "_" + it, 200, input, noise);
                    //String it,Tensor noiseInput,Tensor t,Tensor condInput,Tensor input,DiffusionUNetCond network,TinyVQVAE2 vae,SDImageDataLoader trainingData
                    testSD(i + "", latend, t, condInput, network, vae, labels);
                    //					testSD_DDPM(i + "", latend, t, context, network, vae, trainingData, labels);
                    //					testSD(i + "", 200, latend, t, context, input, network, vae, labels);
                    System.out.println("finish create.");
                    //					testGaussianDiffusion(x_t, t, T, beta_1, beta_T, testParams, trainingData.mean, trainingData.std);
                    network.RUN_MODEL = RunModel.TRAIN;
                    //					this.network.learnRate = this.network.learnRate * 0.1f;
                }
                if (i > 0 && i % 500 == 0) {
                    String save_model_path = "/omega/models/pm_sd_" + i + ".model";
                    ModelUtils.saveModel(network, save_model_path);
                }
                System.out.println("training[" + this.trainIndex + "] train loss:{" + train_loss / indexs.length + "} ");
                /**
                 * update learning rate

                 */
                this.updateLR(this.lr_step);
            }
            /**
             * 停止训练

             */
            System.out.println("training finish. [" + this.trainIndex + "] finalError:" + this.currentError);
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void trainTinySD(SDImageDataLoader trainingData, TinyVQVAE2 vae) {
        // TODO Auto-generated method stub
        try {
            CUDAModules.initCUDAFunctions();
            DiffusionUNetCond2 network = (DiffusionUNetCond2) this.network;
            this.dataSize = trainingData.number;
            if (isWarmUp()) {
                this.network.learnRate = (float) (this.lr * Math.pow(batchIndex * 1.0f / burnIn * 1.0f, power));
            }
            Tensor input = new Tensor(batchSize, 3, trainingData.img_h, trainingData.img_w, true);
            float beta_1 = 0.00085f;
            float beta_T = 0.012f;
            int T = 1000;
            float scale_factor = 0.18215f;
            //			float scale_factor = 0.143262f;
            Tensor t = new Tensor(batchSize, 1, 1, 1, true);
            Tensor a = new Tensor(batchSize, 1, 1, 1, true);
            Tensor b = new Tensor(batchSize, 1, 1, 1, true);
            Tensor noise = new Tensor(batchSize, network.inChannel, network.height, network.width, true);
            Tensor latend = null;
            Tensor noiseLatend = null;
            Tensor latendLoss = null;
            Tensor latendDiff = null;
            float[] betas = MatrixUtils.linspace(beta_1, beta_T, T);
            float[] alphas = MatrixOperation.subtraction(1, betas);
            float[] alphas_bar = MatrixUtils.cumprod(alphas);
            float[] sqrt_alphas_bar = MatrixOperation.sqrt(alphas_bar);
            float[] sqrt_one_minus_alphas_bar = MatrixOperation.sqrt(MatrixOperation.subtraction(1, alphas_bar));
            for (int i = 0; i < this.trainTime; i++) {
                if (this.trainIndex >= this.minTrainTime) {
                    break;
                }
                this.trainIndex = i + 1;
                int[][] indexs = trainingData.shuffle();
                //				int[][] indexs = trainingData.order();
                this.network.RUN_MODEL = RunModel.TRAIN;
                float train_loss = 0.0f;
                /**
                 * 遍历整个训练集

                 */
                for (int it = 0; it < indexs.length; it++) {
                    long start = System.nanoTime();
                    if (Math.abs(this.currentError) <= this.error) {
                        break;
                    }
                    int[] t_data = RandomUtils.randomInt(0, T - 1, batchSize);
                    //					int[] t_data = new int[] {100, 902, 31, 698};
                    //					System.out.println(JsonUtils.toJson(t_data));
                    t.setData(t_data);
                    //					t.showDM();
                    float[] exsa1 = MatrixUtils.gather(sqrt_alphas_bar, t_data);
                    float[] exsa2 = MatrixUtils.gather(sqrt_one_minus_alphas_bar, t_data);
                    a.setData(exsa1);
                    b.setData(exsa2);
                    trainingData.loadData(indexs[it], input);
                    JCudaDriver.cuCtxSynchronize();
                    //					System.out.println("in");
                    /**
                     * get latend

                     */
                    //					input.showShape();
                    latend = vae.encode(input);
                    latend.showShape();
                    network.tensorOP.mul(latend, scale_factor, latend);
                    //					latend.showDMByOffset(0, 100, "before latend");
                    RandomUtils.gaussianRandom(noise, 0, 1);
                    /**
                     * latend add noise

                     */
                    if (noiseLatend == null) {
                        noiseLatend = latend.createLike();
                    }
                    trainingData.addNoise(a, b, latend, noise, noiseLatend, network.cudaManager);
                    /**
                     * forward

                     */
                    Tensor output = network.forward(noiseLatend, t);
                    //					output.showDMByOffset(0, 100, "output");
                    //					output.showDM();
                    /**
                     * loss

                     */
                    this.loss = network.loss(output, noise);
                    /**
                     * loss diff

                     */
                    this.lossDiff = network.lossDiff(output, noise);
                    if (latendLoss == null) {
                        latendLoss = this.loss.createLike();
                        latendDiff = this.lossDiff.createLike();
                    }
                    /**
                     * get noise latend

                     */
                    trainingData.unNoise(a, b, noiseLatend, output, network.cudaManager);
                    network.loss(noiseLatend, latend, latendLoss);
                    network.tensorOP.mul(latendLoss, 0.1f, latendLoss);
                    network.tensorOP.add(this.loss, latendLoss, this.loss);
                    /**
                     * back

                     */
                    network.lossDiff(noiseLatend, latend, latendDiff);
                    trainingData.unMulGrad(a, b, latendDiff, output, latendDiff, network.cudaManager);
                    network.tensorOP.mul(latendDiff, 0.1f, latendDiff);
                    network.tensorOP.add(this.lossDiff, latendDiff, this.lossDiff);
                    /**
                     * back

                     */
                    network.back(this.lossDiff);
                    //					System.out.println(JsonUtils.toJson(this.loss.syncHost()));
                    /**
                     * update

                     */
                    network.update();
                    JCudaDriver.cuCtxSynchronize();
                    /**
                     * current time error

                     */
                    if (this.loss.isHasGPU()) {
                        this.currentError = MatrixOperation.sum(this.loss.syncHost()) / this.batchSize;
                        //						System.out.println(JsonUtils.toJson(this.loss.syncHost()));
                    } else {
                        this.currentError = MatrixOperation.sum(this.loss.data) / this.batchSize;
                    }
                    train_loss += this.currentError;
                    System.out.println("latent loss:" + JsonUtils.toJson(MatrixOperation.sum(latendLoss.syncHost()) / this.batchSize));
                    String msg = "training[" + this.trainIndex + "]{" + it + "} (lr:" + this.network.learnRate + ") train_loss:" + this.currentError + " [costTime:" + (System.nanoTime() - start) / 1e6 + "ms.]";
                    System.out.println(msg);
                    this.batchIndex++;
                }
                if (i % 1 == 0) {
                    network.RUN_MODEL = RunModel.TEST;
                    System.out.println("start create test images.");
                    //					testGaussianDiffusion(i + "_" + it, 200, input, noise);
                    //String it,Tensor noiseInput,Tensor t,Tensor condInput,Tensor input,DiffusionUNetCond network,TinyVQVAE2 vae,SDImageDataLoader trainingData
                    testSD(i + "", latend, t, null, network, vae, null);
                    //					testSD_DDPM(i + "", latend, t, context, network, vae, trainingData, labels);
                    //					testSD(i + "", 200, latend, t, context, input, network, vae, labels);
                    System.out.println("finish create.");
                    //					testGaussianDiffusion(x_t, t, T, beta_1, beta_T, testParams, trainingData.mean, trainingData.std);
                    network.RUN_MODEL = RunModel.TRAIN;
                    //					this.network.learnRate = this.network.learnRate * 0.1f;
                }
                System.out.println("training[" + this.trainIndex + "] train loss:{" + train_loss / indexs.length + "} ");
                /**
                 * update learning rate

                 */
                this.updateLR(this.lr_step);
            }
            /**
             * 停止训练

             */
            System.out.println("training finish. [" + this.trainIndex + "] finalError:" + this.currentError);
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void trainTinySD_Anime(SDImageDataLoaderEN trainingData, VQVAE2 vae, ClipTextModel clip) {
        // TODO Auto-generated method stub
        try {
//            CUDAModules.initCUDAFunctions();
            DiffusionUNetCond2 network = (DiffusionUNetCond2) this.network;
            this.dataSize = trainingData.number;
            if (isWarmUp()) {
                this.network.learnRate = (float) (this.lr * Math.pow(batchIndex * 1.0f / burnIn * 1.0f, power));
            }
            Tensor input = new Tensor(batchSize, 3, trainingData.img_h, trainingData.img_w, true);
            Tensor label = new Tensor(batchSize * network.maxContextLen, 1, 1, 1, true);
            String[] labels = new String[batchSize];
            float beta_1 = 0.00085f;
            float beta_T = 0.012f;
            int T = 1000;
            float scale_factor = 0.18215f;
            //			float scale_factor = 0.143262f;
            Tensor t = new Tensor(batchSize, 1, 1, 1, true);
            Tensor a = new Tensor(batchSize, 1, 1, 1, true);
            Tensor b = new Tensor(batchSize, 1, 1, 1, true);
            Tensor noise = new Tensor(batchSize, network.inChannel, network.height, network.width, true);
            Tensor condInput = null;
            Tensor latend = null;
            float[] betas = MatrixUtils.linspace(beta_1, beta_T, T);
            float[] alphas = MatrixOperation.subtraction(1, betas);
            float[] alphas_bar = MatrixUtils.cumprod(alphas);
            float[] sqrt_alphas_bar = MatrixOperation.sqrt(alphas_bar);
            float[] sqrt_one_minus_alphas_bar = MatrixOperation.sqrt(MatrixOperation.subtraction(1, alphas_bar));
            for (int i = 0; i < this.trainTime; i++) {
                if (this.trainIndex >= this.minTrainTime) {
                    break;
                }
                this.trainIndex = i + 1;
                int[][] indexs = trainingData.shuffle();
                //				int[][] indexs = trainingData.order();
                this.network.RUN_MODEL = RunModel.TRAIN;
                float train_loss = 0.0f;
                /**
                 * 遍历整个训练集

                 */
                for (int it = 0; it < indexs.length; it++) {
                    long start = System.nanoTime();
                    if (Math.abs(this.currentError) <= this.error) {
                        break;
                    }
                    int[] t_data = RandomUtils.randomInt(0, T - 1, batchSize);
                    //					int[] t_data = new int[] {100, 902, 31, 698};
                    //					System.out.println(JsonUtils.toJson(t_data));
                    t.setData(t_data);
                    //					t.showDM();
                    float[] exsa1 = MatrixUtils.gather(sqrt_alphas_bar, t_data);
                    float[] exsa2 = MatrixUtils.gather(sqrt_one_minus_alphas_bar, t_data);
                    a.setData(exsa1);
                    b.setData(exsa2);
                    trainingData.loadData(indexs[it], input, label, noise, labels);
                    JCudaDriver.cuCtxSynchronize();
                    //					System.out.println("in");
                    /**
                     * get latend

                     */
                    //					input.showShape();
                    latend = vae.encode(input);
                    JCudaDriver.cuCtxSynchronize();
                    latend.showShape();
                    network.tensorOP.mul(latend, scale_factor, latend);
                    /**
                     * get context embd
                     */
                    condInput = clip.forward(label);
                    JCudaDriver.cuCtxSynchronize();
                    //					latend.showDMByOffset(0, 100, "before latend");
                    /**
                     * latend add noise
                     */
                    trainingData.addNoise(a, b, latend, noise, network.cudaManager);
                    /**
                     * forward
                     */
                    Tensor output = network.forward(latend, t, condInput);
                    /**
                     * loss
                     */
                    this.loss = network.loss(output, noise);
                    /**
                     * loss diff
                     */
                    this.lossDiff = network.lossDiff(output, noise);
                    /**
                     * back
                     */
                    network.back(this.lossDiff);
                    /**
                     * update
                     */
                    network.update();
                    JCudaDriver.cuCtxSynchronize();
                    /**
                     * current time error
                     */
                    if (this.loss.isHasGPU()) {
                        this.currentError = MatrixOperation.sum(this.loss.syncHost()) / this.batchSize;
                    } else {
                        this.currentError = MatrixOperation.sum(this.loss.data) / this.batchSize;
                    }
                    train_loss += this.currentError;
                    String msg = "training[" + this.trainIndex + "]{" + it + "} (lr:" + this.network.learnRate + ") train_loss:" + this.currentError + " [costTime:" + (System.nanoTime() - start) / 1e6 + "ms.]";
                    System.out.println(msg);
                    this.batchIndex++;
                    /**
                     * update learning rate

                     */
                    this.updateLR(this.lr_step);
                    updateLRDynamic(i * trainingData.count_it + it, this.trainTime * trainingData.count_it, 1e-6f);
                }
                if (i % 10 == 0) {
                    network.RUN_MODEL = RunModel.TEST;
                    System.out.println("start create test images.");
                    //					testGaussianDiffusion(i + "_" + it, 200, input, noise);
                    //String it,Tensor noiseInput,Tensor t,Tensor condInput,Tensor input,DiffusionUNetCond network,TinyVQVAE2 vae,SDImageDataLoader trainingData
                    testSD(i + "", latend, t, condInput, network, vae, labels);
                    //					testSD_DDPM(i + "", latend, t, context, network, vae, trainingData, labels);
                    //					testSD(i + "", 200, latend, t, context, input, network, vae, labels);
                    System.out.println("finish create.");
                    //					testGaussianDiffusion(x_t, t, T, beta_1, beta_T, testParams, trainingData.mean, trainingData.std);
                    network.RUN_MODEL = RunModel.TRAIN;
                    //					this.network.learnRate = this.network.learnRate * 0.1f;
                }
                if (i > 0 && i % 20 == 0) {
                    String save_model_path = "/omega/models/anime_sd_" + i + ".model";
                    ModelUtils.saveModel(network, save_model_path);
                }
                System.out.println("training[" + this.trainIndex + "] train loss:{" + train_loss / indexs.length + "} ");
                /**
                 * update learning rate

                 */
                //				this.updateLR(this.lr_step);
            }
            /**
             * 停止训练

             */
            System.out.println("training finish. [" + this.trainIndex + "] finalError:" + this.currentError);
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void trainSD(SDImageDataLoader trainingData, TinyVQVAE2 vae) {
        // TODO Auto-generated method stub
        try {
            CUDAModules.initCUDAFunctions();
            DiffusionUNetCond network = (DiffusionUNetCond) this.network;
            this.dataSize = trainingData.number;
            if (isWarmUp()) {
                this.network.learnRate = (float) (this.lr * Math.pow(batchIndex * 1.0f / burnIn * 1.0f, power));
            }
            Tensor input = new Tensor(batchSize, 3, trainingData.img_h, trainingData.img_w, true);
            float beta_1 = 0.00085f;
            float beta_T = 0.012f;
            int T = 1000;
            //			float scale_factor = 0.18215f;
            float scale_factor = 0.143262f;
            Tensor t = new Tensor(batchSize, 1, 1, 1, true);
            Tensor a = new Tensor(batchSize, 1, 1, 1, true);
            Tensor b = new Tensor(batchSize, 1, 1, 1, true);
            Tensor noise = new Tensor(batchSize, network.inChannel, network.height, network.width, true);
            Tensor latend = null;
            float[] betas = MatrixUtils.linspace(beta_1, beta_T, T);
            float[] alphas = MatrixOperation.subtraction(1, betas);
            float[] alphas_bar = MatrixUtils.cumprod(alphas);
            float[] sqrt_alphas_bar = MatrixOperation.sqrt(alphas_bar);
            float[] sqrt_one_minus_alphas_bar = MatrixOperation.sqrt(MatrixOperation.subtraction(1, alphas_bar));
            for (int i = 0; i < this.trainTime; i++) {
                if (this.trainIndex >= this.minTrainTime) {
                    break;
                }
                this.trainIndex = i + 1;
                int[][] indexs = trainingData.shuffle();
                //				int[][] indexs = trainingData.order();
                this.network.RUN_MODEL = RunModel.TRAIN;
                float train_loss = 0.0f;
                /**
                 * 遍历整个训练集

                 */
                for (int it = 0; it < indexs.length; it++) {
                    long start = System.nanoTime();
                    if (Math.abs(this.currentError) <= this.error) {
                        break;
                    }
                    int[] t_data = RandomUtils.randomInt(0, T - 1, batchSize);
                    //					int[] t_data = new int[] {100, 902, 31, 698};
                    //					System.out.println(JsonUtils.toJson(t_data));
                    t.setData(t_data);
                    //					t.showDM();
                    float[] exsa1 = MatrixUtils.gather(sqrt_alphas_bar, t_data);
                    float[] exsa2 = MatrixUtils.gather(sqrt_one_minus_alphas_bar, t_data);
                    a.setData(exsa1);
                    b.setData(exsa2);
                    trainingData.loadData(indexs[it], input);
                    JCudaDriver.cuCtxSynchronize();
                    //					System.out.println("in");
                    /**
                     * get latend

                     */
                    //					input.showShape();
                    latend = vae.encode(input);
                    latend.showShape();
                    network.tensorOP.mul(latend, scale_factor, latend);
                    //					latend.showDMByOffset(0, 100, "before latend");
                    RandomUtils.gaussianRandom(noise, 0, 1);
                    /**
                     * latend add noise

                     */
                    trainingData.addNoise(a, b, latend, noise, network.cudaManager);
                    /**
                     * forward

                     */
                    Tensor output = network.forward(latend, t);
                    //					output.showDMByOffset(0, 100, "output");
                    //					output.showDM();
                    /**
                     * loss

                     */
                    this.loss = network.loss(output, noise);
                    /**
                     * loss diff

                     */
                    this.lossDiff = network.lossDiff(output, noise);
                    /**
                     * back

                     */
                    network.back(this.lossDiff);
                    //					System.out.println(JsonUtils.toJson(this.loss.syncHost()));
                    /**
                     * update

                     */
                    network.update();
                    JCudaDriver.cuCtxSynchronize();
                    /**
                     * current time error

                     */
                    if (this.loss.isHasGPU()) {
                        this.currentError = MatrixOperation.sum(this.loss.syncHost()) / this.batchSize;
                        //						System.out.println(JsonUtils.toJson(this.loss.syncHost()));
                    } else {
                        this.currentError = MatrixOperation.sum(this.loss.data) / this.batchSize;
                    }
                    train_loss += this.currentError;
                    String msg = "training[" + this.trainIndex + "]{" + it + "} (lr:" + this.network.learnRate + ") train_loss:" + this.currentError + " [costTime:" + (System.nanoTime() - start) / 1e6 + "ms.]";
                    System.out.println(msg);
                    this.batchIndex++;
                    //					if(it % 10 == 0) {
                    //						network.RUN_MODEL = RunModel.TEST;
                    //						System.out.println("start create test images.");
                    ////						testGaussianDiffusion(i + "_" + it, 200, input, noise);
                    //						testSD(i + "", 200, latend, t, condInput, network, vae);
                    //						System.out.println("finish create.");
                    ////						testGaussianDiffusion(x_t, t, T, beta_1, beta_T, testParams, trainingData.mean, trainingData.std);
                    //						network.RUN_MODEL = RunModel.TRAIN;
                    ////						this.network.learnRate = this.network.learnRate * 0.1f;
                    //					}
                }
                if (i % 1 == 0) {
                    network.RUN_MODEL = RunModel.TEST;
                    System.out.println("start create test images.");
                    //					testGaussianDiffusion(i + "_" + it, 200, input, noise);
                    //String it,Tensor noiseInput,Tensor t,Tensor condInput,Tensor input,DiffusionUNetCond network,TinyVQVAE2 vae,SDImageDataLoader trainingData
                    //					testSD(i + "", latend, t, context, network, vae, trainingData, labels);
                    testSD_DDPM(i + "", latend, t, network, vae, trainingData);
                    //					testSD(i + "", 200, latend, t, context, input, network, vae, labels);
                    System.out.println("finish create.");
                    //					testGaussianDiffusion(x_t, t, T, beta_1, beta_T, testParams, trainingData.mean, trainingData.std);
                    network.RUN_MODEL = RunModel.TRAIN;
                    //					this.network.learnRate = this.network.learnRate * 0.1f;
                }
                System.out.println("training[" + this.trainIndex + "] train loss:{" + train_loss / indexs.length + "} ");
                /**
                 * update learning rate

                 */
                this.updateLR(this.lr_step);
            }
            /**
             * 停止训练

             */
            System.out.println("training finish. [" + this.trainIndex + "] finalError:" + this.currentError);
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void trainSD_uncond(SDImageDataLoader trainingData, TinyVQVAE2 vae) {
        // TODO Auto-generated method stub
        try {
            CUDAModules.initCUDAFunctions();
            DiffusionUNet network = (DiffusionUNet) this.network;
            this.dataSize = trainingData.number;
            if (isWarmUp()) {
                this.network.learnRate = (float) (this.lr * Math.pow(batchIndex * 1.0f / burnIn * 1.0f, power));
            }
            Tensor input = new Tensor(batchSize, 3, trainingData.img_h, trainingData.img_w, true);
            float beta_1 = 1e-4f;
            float beta_T = 0.02f;
            int T = 1000;
            Tensor t = new Tensor(batchSize, 1, 1, 1, true);
            Tensor a = new Tensor(batchSize, 1, 1, 1, true);
            Tensor b = new Tensor(batchSize, 1, 1, 1, true);
            Tensor noise = new Tensor(batchSize, this.network.getChannel(), network.height, network.width, true);
            Tensor noiseTest = new Tensor(batchSize, 3, trainingData.img_h, trainingData.img_w, true);
            Tensor latend = null;
            Tensor noiseLatend = null;
            Tensor latendLoss = null;
            Tensor latendDiff = null;
            float[] betas = MatrixUtils.linspace(beta_1, beta_T, T);
            float[] alphas = MatrixOperation.subtraction(1, betas);
            float[] alphas_bar = MatrixUtils.cumprod(alphas);
            float[] sqrt_alphas_bar = MatrixOperation.sqrt(alphas_bar);
            float[] sqrt_one_minus_alphas_bar = MatrixOperation.sqrt(MatrixOperation.subtraction(1, alphas_bar));
            for (int i = 0; i < this.trainTime; i++) {
                if (this.trainIndex >= this.minTrainTime) {
                    break;
                }
                this.trainIndex = i + 1;
                int[][] indexs = trainingData.shuffle();
                //				int[][] indexs = trainingData.order();
                this.network.RUN_MODEL = RunModel.TRAIN;
                float train_loss = 0.0f;
                /**
                 * 遍历整个训练集

                 */
                for (int it = 0; it < indexs.length; it++) {
                    long start = System.nanoTime();
                    if (Math.abs(this.currentError) <= this.error) {
                        break;
                    }
                    int[] t_data = RandomUtils.randomInt(0, T - 1, batchSize);
                    //					int[] t_data = new int[] {100, 902, 31, 698};
                    //					System.out.println(JsonUtils.toJson(t_data));
                    t.setData(t_data);
                    //					t.showDM();
                    float[] exsa1 = MatrixUtils.gather(sqrt_alphas_bar, t_data);
                    float[] exsa2 = MatrixUtils.gather(sqrt_one_minus_alphas_bar, t_data);
                    a.setData(exsa1);
                    b.setData(exsa2);
                    trainingData.loadData_uncond(indexs[it], input, noise);
                    JCudaDriver.cuCtxSynchronize();
                    //					System.out.println("in");
                    /**
                     * get latend

                     */
                    //					input.showShape();
                    latend = vae.encode(input);
                    latend.showShape();
                    //					latend.showDMByOffset(0, 100, "before latend");
                    /**
                     * latend add noise

                     */
                    if (noiseLatend == null) {
                        noiseLatend = latend.createLike();
                    }
                    trainingData.addNoise(a, b, latend, noise, noiseLatend, network.cudaManager);
                    RandomUtils.gaussianRandom(noiseTest, 0, 1);
                    trainingData.addNoise(a, b, input, noiseTest, network.cudaManager);
                    JCudaDriver.cuCtxSynchronize();
                    /**
                     * forward

                     */
                    Tensor output = network.forward(noiseLatend, t);
                    //					output.showDMByOffset(0, 100, "output");
                    //					output.showDM();
                    /**
                     * loss

                     */
                    this.loss = network.loss(output, noise);
                    /**
                     * loss diff

                     */
                    this.lossDiff = network.lossDiff(output, noise);
                    //					if(latendLoss == null) {
                    //						latendLoss = this.loss.createLike();
                    //						latendDiff = this.lossDiff.createLike();
                    //					}
                    //
                    //					/**
                    //					 * get noise latend
                    //					 */
                    //					trainingData.unNoise(a, b, noiseLatend, output);
                    //					network.loss(noiseLatend, latend, latendLoss);
                    //
                    //					TensorOP.mul(latendLoss, 0.1f, latendLoss);
                    //					TensorOP.add(this.loss, latendLoss, this.loss);
                    //
                    //					/**
                    //					 * back
                    //					 */
                    //					network.lossDiff(noiseLatend, latend, latendDiff);
                    //					trainingData.unMulGrad(a, b, latendDiff, output, latendDiff);
                    //					TensorOP.mul(latendDiff, 0.1f, latendDiff);
                    //					TensorOP.add(this.lossDiff, latendDiff, this.lossDiff);
                    network.back(this.lossDiff);
                    //					System.out.println(JsonUtils.toJson(this.loss.syncHost()));
                    /**
                     * update

                     */
                    network.update();
                    JCudaDriver.cuCtxSynchronize();
                    /**
                     * current time error

                     */
                    if (this.loss.isHasGPU()) {
                        this.currentError = MatrixOperation.sum(this.loss.syncHost()) / this.batchSize;
                    } else {
                        this.currentError = MatrixOperation.sum(this.loss.data) / this.batchSize;
                    }
                    train_loss += this.currentError;
                    String msg = "training[" + this.trainIndex + "]{" + it + "} (lr:" + this.network.learnRate + ") train_loss:" + this.currentError + " [costTime:" + (System.nanoTime() - start) / 1e6 + "ms.]";
                    System.out.println(msg);
                    this.batchIndex++;
                }
                if (i % 1 == 0) {
                    network.RUN_MODEL = RunModel.TEST;
                    System.out.println("start create test images.");
                    //					testGaussianDiffusion(i + "_" + it, 200, input, noise);
                    //String it,Tensor noiseInput,Tensor t,Tensor input,TinyVQVAE2 vae,SDImageDataLoader trainingData,Tensor a,Tensor b
                    testSD(i + "", latend, t, input, vae, trainingData);
                    System.out.println("finish create.");
                    //					testGaussianDiffusion(x_t, t, T, beta_1, beta_T, testParams, trainingData.mean, trainingData.std);
                    network.RUN_MODEL = RunModel.TRAIN;
                    //					this.network.learnRate = this.network.learnRate * 0.1f;
                }
                System.out.println("training[" + this.trainIndex + "] train loss:{" + train_loss / indexs.length + "} ");
                /**
                 * update learning rate

                 */
                this.updateLR(this.lr_step);
            }
            /**
             * 停止训练

             */
            System.out.println("training finish. [" + this.trainIndex + "] finalError:" + this.currentError);
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
    
    public void train_DiT_Anime(SDImageDataLoaderEN trainingData, VQVAE2 vae, ClipTextModel clip) {
        // TODO Auto-generated method stub
        try {
//            CUDAModules.initCUDAFunctions();
            DiT network = (DiT) this.network;
            
            this.dataSize = trainingData.number;
            if (isWarmUp()) {
                this.network.learnRate = (float) (this.lr * Math.pow(batchIndex * 1.0f / burnIn * 1.0f, power));
            }
            Tensor input = new Tensor(batchSize, 3, trainingData.img_h, trainingData.img_w, true);
            Tensor label = new Tensor(batchSize * network.maxContextLen, 1, 1, 1, true);
            
            Tensor[] cs = RoPEKernel.getCosAndSin(network.time, network.hiddenSize * 2, network.headNum);
            Tensor cos = cs[0];
            Tensor sin = cs[1];
            
            String[] labels = new String[batchSize];
            float beta_1 = 0.00085f;
            float beta_T = 0.012f;
            int T = 1000;
            float scale_factor = 0.18215f;
            //			float scale_factor = 0.143262f;
            Tensor t = new Tensor(batchSize, 1, 1, 1, true);
            Tensor a = new Tensor(batchSize, 1, 1, 1, true);
            Tensor b = new Tensor(batchSize, 1, 1, 1, true);
            Tensor noise = new Tensor(batchSize, network.inChannel, network.height, network.width, true);
            Tensor condInput = null;
            Tensor latend = null;
            float[] betas = MatrixUtils.linspace(beta_1, beta_T, T);
            float[] alphas = MatrixOperation.subtraction(1, betas);
            float[] alphas_bar = MatrixUtils.cumprod(alphas);
            float[] sqrt_alphas_bar = MatrixOperation.sqrt(alphas_bar);
            float[] sqrt_one_minus_alphas_bar = MatrixOperation.sqrt(MatrixOperation.subtraction(1, alphas_bar));
            for (int i = 0; i < this.trainTime; i++) {
                if (this.trainIndex >= this.minTrainTime) {
                    break;
                }
                this.trainIndex = i + 1;
                int[][] indexs = trainingData.shuffle();
                //				int[][] indexs = trainingData.order();
                this.network.RUN_MODEL = RunModel.TRAIN;
                float train_loss = 0.0f;
                /**
                 * 遍历整个训练集
                 */
                for (int it = 0; it < indexs.length; it++) {
                    long start = System.nanoTime();
                    if (Math.abs(this.currentError) <= this.error) {
                        break;
                    }
                    int[] t_data = RandomUtils.randomInt(0, T - 1, batchSize);
                    //					int[] t_data = new int[] {100, 902, 31, 698};
                    //					System.out.println(JsonUtils.toJson(t_data));
                    t.setData(t_data);
                    //					t.showDM();
                    float[] exsa1 = MatrixUtils.gather(sqrt_alphas_bar, t_data);
                    float[] exsa2 = MatrixUtils.gather(sqrt_one_minus_alphas_bar, t_data);
                    a.setData(exsa1);
                    b.setData(exsa2);
                    trainingData.loadData(indexs[it], input, label, noise, labels);
                    JCudaDriver.cuCtxSynchronize();
                    //					System.out.println("in");
                    /**
                     * get latend
                     */
                    //					input.showShape();
                    latend = vae.encode(input);
                    JCudaDriver.cuCtxSynchronize();
                    latend.showShape();
                    network.tensorOP.mul(latend, scale_factor, latend);
                    /**
                     * get context embd
                     */
                    condInput = clip.forward(label);
                    condInput.showShape();
                    JCudaDriver.cuCtxSynchronize();
                    //					latend.showDMByOffset(0, 100, "before latend");
                    /**
                     * latend add noise
                     */
                    trainingData.addNoise(a, b, latend, noise, network.cudaManager);
                    /**
                     * forward
                     */
                    Tensor output = network.forward(latend, t, condInput, cos, sin);
                    /**
                     * loss
                     */
                    this.loss = network.loss(output, noise);
                    /**
                     * loss diff
                     */
                    this.lossDiff = network.lossDiff(output, noise);
                    /**
                     * back
                     */
                    network.back(this.lossDiff, cos, sin);
                    /**
                     * update
                     */
                    network.update();
                    JCudaDriver.cuCtxSynchronize();
                    /**
                     * current time error
                     */
                    if (this.loss.isHasGPU()) {
                        this.currentError = MatrixOperation.sum(this.loss.syncHost()) / this.batchSize;
                    } else {
                        this.currentError = MatrixOperation.sum(this.loss.data) / this.batchSize;
                    }
                    train_loss += this.currentError;
                    String msg = "training[" + this.trainIndex + "]{" + it + "} (lr:" + this.network.learnRate + ") train_loss:" + this.currentError + " [costTime:" + (System.nanoTime() - start) / 1e6 + "ms.]";
                    System.out.println(msg);
                    this.batchIndex++;
                    /**
                     * update learning rate
                     */
                    this.updateLR(this.lr_step);
                    updateLRDynamic(i * trainingData.count_it + it, this.trainTime * trainingData.count_it, 1e-6f);
                }
                if (i % 10 == 0) {
                    network.RUN_MODEL = RunModel.TEST;
                    System.out.println("start create test images.");
                    //					testGaussianDiffusion(i + "_" + it, 200, input, noise);
                    //String it,Tensor noiseInput,Tensor t,Tensor condInput,Tensor input,DiffusionUNetCond network,TinyVQVAE2 vae,SDImageDataLoader trainingData
                    testDiT(i + "", latend, t, condInput, cos, sin, network, vae, labels);
                    //					testSD_DDPM(i + "", latend, t, context, network, vae, trainingData, labels);
                    //					testSD(i + "", 200, latend, t, context, input, network, vae, labels);
                    System.out.println("finish create.");
                    //					testGaussianDiffusion(x_t, t, T, beta_1, beta_T, testParams, trainingData.mean, trainingData.std);
                    network.RUN_MODEL = RunModel.TRAIN;
                    //					this.network.learnRate = this.network.learnRate * 0.1f;
                }
//                if (i > 0 && i % 20 == 0) {
//                    String save_model_path = "/omega/models/anime_sd_" + i + ".model";
//                    ModelUtils.saveModel(network, save_model_path);
//                }
                System.out.println("training[" + this.trainIndex + "] train loss:{" + train_loss / indexs.length + "} ");
                /**
                 * update learning rate

                 */
                //				this.updateLR(this.lr_step);
            }
            /**
             * 停止训练

             */
            System.out.println("training finish. [" + this.trainIndex + "] finalError:" + this.currentError);
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
    
    public void train_DiT_Anime2(SDImageDataLoaderEN trainingData, VQVAE2 vae, ClipTextModel clip) {
        // TODO Auto-generated method stub
        try {
//            CUDAModules.initCUDAFunctions();
            DiT network = (DiT) this.network;
            
            this.dataSize = trainingData.number;
            if (isWarmUp()) {
                this.network.learnRate = (float) (this.lr * Math.pow(batchIndex * 1.0f / burnIn * 1.0f, power));
            }
            Tensor input = new Tensor(batchSize, 3, trainingData.img_h, trainingData.img_w, true);
            Tensor label = new Tensor(batchSize * network.maxContextLen, 1, 1, 1, true);
            
            String[] labels = new String[batchSize];
            float beta_1 = 0.00085f;
            float beta_T = 0.012f;
            int T = 1000;
            float scale_factor = 0.18215f;
            //			float scale_factor = 0.143262f;
            Tensor t = new Tensor(batchSize, 1, 1, 1, true);
            Tensor a = new Tensor(batchSize, 1, 1, 1, true);
            Tensor b = new Tensor(batchSize, 1, 1, 1, true);
            Tensor noise = new Tensor(batchSize, network.inChannel, network.height, network.width, true);
            Tensor condInput = null;
            Tensor latend = null;
            float[] betas = MatrixUtils.linspace(beta_1, beta_T, T);
            float[] alphas = MatrixOperation.subtraction(1, betas);
            float[] alphas_bar = MatrixUtils.cumprod(alphas);
            float[] sqrt_alphas_bar = MatrixOperation.sqrt(alphas_bar);
            float[] sqrt_one_minus_alphas_bar = MatrixOperation.sqrt(MatrixOperation.subtraction(1, alphas_bar));
            for (int i = 0; i < this.trainTime; i++) {
                if (this.trainIndex >= this.minTrainTime) {
                    break;
                }
                this.trainIndex = i + 1;
                int[][] indexs = trainingData.shuffle();
                //				int[][] indexs = trainingData.order();
                this.network.RUN_MODEL = RunModel.TRAIN;
                float train_loss = 0.0f;
                /**
                 * 遍历整个训练集
                 */
                for (int it = 0; it < indexs.length; it++) {
                    long start = System.nanoTime();
                    if (Math.abs(this.currentError) <= this.error) {
                        break;
                    }
                    int[] t_data = RandomUtils.randomInt(0, T - 1, batchSize);
                    //					int[] t_data = new int[] {100, 902, 31, 698};
                    //					System.out.println(JsonUtils.toJson(t_data));
                    t.setData(t_data);
                    //					t.showDM();
                    float[] exsa1 = MatrixUtils.gather(sqrt_alphas_bar, t_data);
                    float[] exsa2 = MatrixUtils.gather(sqrt_one_minus_alphas_bar, t_data);
                    a.setData(exsa1);
                    b.setData(exsa2);
                    trainingData.loadData(indexs[it], input, label, noise, labels);
                    JCudaDriver.cuCtxSynchronize();
                    //					System.out.println("in");
                    /**
                     * get latend
                     */
                    //					input.showShape();
                    latend = vae.encode(input);
                    JCudaDriver.cuCtxSynchronize();
                    latend.showShape();
                    network.tensorOP.mul(latend, scale_factor, latend);
                    /**
                     * get context embd
                     */
                    condInput = clip.forward(label);
                    condInput.showShape();
                    JCudaDriver.cuCtxSynchronize();
                    //					latend.showDMByOffset(0, 100, "before latend");
                    /**
                     * latend add noise
                     */
                    trainingData.addNoise(a, b, latend, noise, network.cudaManager);
                    /**
                     * forward
                     */
                    Tensor output = network.forward(latend, t, condInput);
                    /**
                     * loss
                     */
                    this.loss = network.loss(output, noise);
                    /**
                     * loss diff
                     */
                    this.lossDiff = network.lossDiff(output, noise);
                    /**
                     * back
                     */
                    network.back(this.lossDiff);
                    /**
                     * update
                     */
                    network.update();
                    JCudaDriver.cuCtxSynchronize();
                    /**
                     * current time error
                     */
                    if (this.loss.isHasGPU()) {
                        this.currentError = MatrixOperation.sum(this.loss.syncHost()) / this.batchSize;
                    } else {
                        this.currentError = MatrixOperation.sum(this.loss.data) / this.batchSize;
                    }
                    train_loss += this.currentError;
                    String msg = "training[" + this.trainIndex + "]{" + it + "} (lr:" + this.network.learnRate + ") train_loss:" + this.currentError + " [costTime:" + (System.nanoTime() - start) / 1e6 + "ms.]";
                    System.out.println(msg);
                    this.batchIndex++;
                    /**
                     * update learning rate
                     */
                    this.updateLR(this.lr_step);
                    updateLRDynamic(i * trainingData.count_it + it, this.trainTime * trainingData.count_it, 1e-6f);
                }
                if (i % 10 == 0) {
                    network.RUN_MODEL = RunModel.TEST;
                    System.out.println("start create test images.");
                    //					testGaussianDiffusion(i + "_" + it, 200, input, noise);
                    //String it,Tensor noiseInput,Tensor t,Tensor condInput,Tensor input,DiffusionUNetCond network,TinyVQVAE2 vae,SDImageDataLoader trainingData
                    testDiT(i + "", latend, t, condInput, network, vae, labels);
                    //					testSD_DDPM(i + "", latend, t, context, network, vae, trainingData, labels);
                    //					testSD(i + "", 200, latend, t, context, input, network, vae, labels);
                    System.out.println("finish create.");
                    //					testGaussianDiffusion(x_t, t, T, beta_1, beta_T, testParams, trainingData.mean, trainingData.std);
                    network.RUN_MODEL = RunModel.TRAIN;
                    //					this.network.learnRate = this.network.learnRate * 0.1f;
                }
//                if (i > 0 && i % 20 == 0) {
//                    String save_model_path = "/omega/models/anime_sd_" + i + ".model";
//                    ModelUtils.saveModel(network, save_model_path);
//                }
                System.out.println("training[" + this.trainIndex + "] train loss:{" + train_loss / indexs.length + "} ");
                /**
                 * update learning rate

                 */
                //				this.updateLR(this.lr_step);
            }
            /**
             * 停止训练

             */
            System.out.println("training finish. [" + this.trainIndex + "] finalError:" + this.currentError);
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void testSD(String it, int ddim_timesteps, Tensor noiseInput, Tensor t, Tensor context, Tensor input, DiffusionUNetCond network, TinyVQVAE2 vae, String[] labels) {
        try {
            float beta_1 = 1e-4f;
            float beta_T = 0.02f;
            int T = 1000;
            float[] mean = new float[]{0.5f, 0.5f, 0.5f};
            float[] std = new float[]{0.5f, 0.5f, 0.5f};
            //			Tensor noiseInput = new Tensor(batchSize, this.network.getChannel(), this.network.getHeight(), this.network.getWidth(), true);
            //
            //			Tensor t = new Tensor(batchSize, 1, 1, 1, true);
            //			RandomUtils.gaussianRandom2(noiseInput, 0, 1);
            //			noiseInput.showShape();
            RandomUtils.gaussianRandom(noiseInput, 0, 1);
            float[] betas = MatrixUtils.linspace(beta_1, beta_T, T);
            float[] alphas = MatrixOperation.subtraction(1, betas);
            float[] alphas_bar = MatrixUtils.cumprod(alphas);
            int step = T / ddim_timesteps;
            float[] ddim_timestep_seq = MatrixUtils.range(0, T, step, 1);
            float[] ddim_timestep_prev_seq = new float[ddim_timestep_seq.length];
            for (int i = 1; i < ddim_timestep_seq.length; i++) {
                ddim_timestep_prev_seq[i] = ddim_timestep_seq[i - 1];
            }
            int[] t_data = new int[batchSize];
            int[] prev_t_data = new int[batchSize];
            for (int timestep = ddim_timesteps - 1; timestep >= 0; timestep--) {
                for (int i = 0; i < batchSize; i++) {
                    t.data[i] = ddim_timestep_seq[timestep];
                    t_data[i] = (int) ddim_timestep_seq[timestep];
                    prev_t_data[i] = (int) ddim_timestep_prev_seq[timestep];
                }
                t.hostToDevice();
                float[] exsa1 = MatrixUtils.gather(alphas_bar, t_data);
                float[] exsa2 = MatrixUtils.gather(alphas_bar, prev_t_data);
                prev_mean_from_eps(network, noiseInput, t, context, exsa1, exsa2, 1, timestep);
            }
            Tensor result = vae.decode(noiseInput);
            result.data = MatrixOperation.clampSelf(result.syncHost(), -1, 1);
            /**
             * print image

             */
            showImgs("H://vae_dataset//pokemon-blip//vqvae2//sd//", result, it, mean, std, labels);
            Tensor label = vae.encode(input);
            Tensor labelResult = vae.decode(label);
            labelResult.data = MatrixOperation.clampSelf(labelResult.syncHost(), -1, 1);
            showImgsLabel("H://vae_dataset//pokemon-blip//vqvae2//sd//", labelResult, it, mean, std, labels);
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void testSD(String it, Tensor noiseInput, Tensor t, Tensor input, TinyVQVAE2 vae, SDImageDataLoader trainingData) {
        try {
            float beta_1 = 1e-4f;
            float beta_T = 0.02f;
            int T = 1000;
            float[] mean = new float[]{0.5f, 0.5f, 0.5f};
            float[] std = new float[]{0.5f, 0.5f, 0.5f};
            //			Tensor noiseInput = new Tensor(batchSize, this.network.getChannel(), this.network.getHeight(), this.network.getWidth(), true);
            //
            //			Tensor t = new Tensor(batchSize, 1, 1, 1, true);
            //			RandomUtils.gaussianRandom2(noiseInput, 0, 1);
            //			noiseInput.showShape();
            RandomUtils.gaussianRandom(noiseInput, 0, 1);
            float[] betas = MatrixUtils.linspace(beta_1, beta_T, T);
            float[] alphas = MatrixOperation.subtraction(1, betas);
            float[] alphas_bar = MatrixUtils.cumprod(alphas);
            float[] sqrt_alphas_bar = MatrixOperation.sqrt(alphas_bar);
            float[] sqrt_one_minus_alphas_bar = MatrixOperation.sqrt(MatrixOperation.subtraction(1, alphas_bar));
            Tensor xt = noiseInput;
            for (int ts = T - 1; ts >= 0; ts--) {
                sample_prev_timestep(trainingData, vae, xt, t, null, ts, sqrt_alphas_bar, sqrt_one_minus_alphas_bar, betas, alphas, alphas_bar);
            }
            Tensor result = vae.decodeCode(xt);
            //			Tensor result = vae.decode(xt);
            result.data = MatrixOperation.clampSelf(result.syncHost(), -1, 1);
            System.err.println("in");
            /**
             * print image

             */
            showImgs("H://vae_dataset//pokemon-blip//vqvae2//sd//", result, it, mean, std);
            //			input.data = MatrixOperation.clampSelf(input.syncHost(), -1, 1);
            //
            //			/**
            //			 * print image
            //			 */
            //			showImgs("H://vae_dataset//pokemon-blip//vqvae2//sd_test//", input, it, mean, std);
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void testSD(String it, Tensor noiseInput, Tensor t, Tensor condInput, DiffusionUNetCond network, TinyVQVAE2 vae, SDImageDataLoader trainingData, String[] labels) {
        try {
            float beta_1 = 0.00085f;
            float beta_T = 0.012f;
            int T = 1000;
            //			float scale_factor = 0.143262f;
            float scale_factor = 0.18215f;
            float[] mean = new float[]{0.5f, 0.5f, 0.5f};
            float[] std = new float[]{0.5f, 0.5f, 0.5f};
            RandomUtils.gaussianRandom(noiseInput, 0, 1);
            float[] betas = MatrixUtils.linspace(beta_1, beta_T, T);
            float[] alphas = MatrixOperation.subtraction(1, betas);
            float[] alphas_bar = MatrixUtils.cumprod(alphas);
            float[] sqrt_alphas_bar = MatrixOperation.sqrt(alphas_bar);
            float[] sqrt_one_minus_alphas_bar = MatrixOperation.sqrt(MatrixOperation.subtraction(1, alphas_bar));
            Tensor xt = noiseInput;
            for (int ts = T - 1; ts >= 0; ts--) {
                sample_prev_timestep(network, trainingData, vae, condInput, xt, t, null, ts, sqrt_alphas_bar, sqrt_one_minus_alphas_bar, betas, alphas, alphas_bar);
            }
            JCuda.cudaDeviceSynchronize();
            network.tensorOP.mul(xt, 1 / scale_factor, xt);
            //			Tensor result = vae.decodeCode(xt);
            Tensor result = vae.decode(xt);
            JCuda.cudaDeviceSynchronize();
            result.data = MatrixOperation.clampSelf(result.syncHost(), -1, 1);
            System.err.println("in");
            /**
             * print image

             */
            showImgs("H://vae_dataset//pokemon-blip//vqvae2//sd//", result, it, mean, std, labels);
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void testSD_DDPM(String it, Tensor noiseInput, Tensor t, Tensor condInput, DiffusionUNetCond network, TinyVQVAE2 vae, SDImageDataLoader trainingData, String[] labels) {
        try {
            float beta_1 = 0.00085f;
            float beta_T = 0.012f;
            int ddim_timesteps = 200;
            int T = 1000;
            //			float scale_factor = 0.143262f;
            float scale_factor = 0.18215f;
            float[] mean = new float[]{0.5f, 0.5f, 0.5f};
            float[] std = new float[]{0.5f, 0.5f, 0.5f};
            RandomUtils.gaussianRandom(noiseInput, 0, 1);
            float[] betas = MatrixUtils.linspace(beta_1, beta_T, T);
            float[] alphas = MatrixOperation.subtraction(1, betas);
            float[] alphas_bar = MatrixUtils.cumprod(alphas);
            int step = T / ddim_timesteps;
            float[] ddim_timestep_seq = MatrixUtils.range(0, T, step, 1);
            float[] ddim_timestep_prev_seq = new float[ddim_timestep_seq.length];
            for (int i = 1; i < ddim_timestep_seq.length; i++) {
                ddim_timestep_prev_seq[i] = ddim_timestep_seq[i - 1];
            }
            int[] t_data = new int[batchSize];
            int[] prev_t_data = new int[batchSize];
            for (int timestep = ddim_timesteps - 1; timestep >= 0; timestep--) {
                for (int i = 0; i < batchSize; i++) {
                    t.data[i] = ddim_timestep_seq[timestep];
                    t_data[i] = (int) ddim_timestep_seq[timestep];
                    prev_t_data[i] = (int) ddim_timestep_prev_seq[timestep];
                }
                t.hostToDevice();
                float[] exsa1 = MatrixUtils.gather(alphas_bar, t_data);
                float[] exsa2 = MatrixUtils.gather(alphas_bar, prev_t_data);
                prev_mean_from_eps(network, noiseInput, t, condInput, exsa1, exsa2, 1, timestep);
            }
            JCuda.cudaDeviceSynchronize();
            network.tensorOP.mul(noiseInput, 1.0f / scale_factor, noiseInput);
            //			Tensor result = vae.decodeCode(xt);
            Tensor result = vae.decode(noiseInput);
            JCuda.cudaDeviceSynchronize();
            result.data = MatrixOperation.clampSelf(result.syncHost(), -1, 1);
            System.err.println("in");
            /**
             * print image

             */
            showImgs("H://vae_dataset//pokemon-blip//vqvae2//sd//", result, it, mean, std, labels);
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void testSD_DDPM(String it, Tensor noiseInput, Tensor t, DiffusionUNetCond network, TinyVQVAE2 vae, SDImageDataLoader trainingData) {
        try {
            float beta_1 = 0.00085f;
            float beta_T = 0.012f;
            int ddim_timesteps = 200;
            int T = 1000;
            float scale_factor = 0.143262f;
            float[] mean = new float[]{0.5f, 0.5f, 0.5f};
            float[] std = new float[]{0.5f, 0.5f, 0.5f};
            RandomUtils.gaussianRandom(noiseInput, 0, 1);
            float[] betas = MatrixUtils.linspace(beta_1, beta_T, T);
            float[] alphas = MatrixOperation.subtraction(1, betas);
            float[] alphas_bar = MatrixUtils.cumprod(alphas);
            int step = T / ddim_timesteps;
            float[] ddim_timestep_seq = MatrixUtils.range(0, T, step, 1);
            float[] ddim_timestep_prev_seq = new float[ddim_timestep_seq.length];
            for (int i = 1; i < ddim_timestep_seq.length; i++) {
                ddim_timestep_prev_seq[i] = ddim_timestep_seq[i - 1];
            }
            int[] t_data = new int[batchSize];
            int[] prev_t_data = new int[batchSize];
            for (int timestep = ddim_timesteps - 1; timestep >= 0; timestep--) {
                for (int i = 0; i < batchSize; i++) {
                    t.data[i] = ddim_timestep_seq[timestep];
                    t_data[i] = (int) ddim_timestep_seq[timestep];
                    prev_t_data[i] = (int) ddim_timestep_prev_seq[timestep];
                }
                t.hostToDevice();
                float[] exsa1 = MatrixUtils.gather(alphas_bar, t_data);
                float[] exsa2 = MatrixUtils.gather(alphas_bar, prev_t_data);
                prev_mean_from_eps(network, noiseInput, t, exsa1, exsa2, 1, timestep);
            }
            JCuda.cudaDeviceSynchronize();
            network.tensorOP.mul(noiseInput, 1.0f / scale_factor, noiseInput);
            //			Tensor result = vae.decodeCode(xt);
            Tensor result = vae.decode(noiseInput);
            JCuda.cudaDeviceSynchronize();
            result.data = MatrixOperation.clampSelf(result.syncHost(), -1, 1);
            System.err.println("in");
            /**
             * print image

             */
            showImgs("H://vae_dataset//pokemon-blip//vqvae2//sd//", result, it, mean, std);
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void updateLRDynamic(int it, int count, float min) {
        int warmup_iters = 0;
        int lr_decay_iters = count;
        //		System.out.println(this.lr);
        //		System.out.println(lr_decay_iters);
        double min_lr = min;
        if (it < warmup_iters) {
            network.learnRate = this.lr * it / warmup_iters;
            return;
        }
        if (it > lr_decay_iters) {
            network.learnRate = (float) min_lr;
            return;
        }
        BigDecimal decay_ratio = new BigDecimal(0);
        if (it > 0) {
            decay_ratio = new BigDecimal(it - warmup_iters).divide(new BigDecimal(lr_decay_iters - warmup_iters), 24, BigDecimal.ROUND_HALF_DOWN);
        }
        //	    System.out.println(decay_ratio.doubleValue());
        BigDecimal coeff = new BigDecimal(0.5d).multiply(new BigDecimal(1).add(new BigDecimal(Math.cos(new BigDecimal(Math.PI).multiply(decay_ratio).doubleValue()))));
        BigDecimal tlr = new BigDecimal(min_lr).add(coeff.multiply(new BigDecimal((this.lr - min_lr))));
        tlr = tlr.setScale(24, BigDecimal.ROUND_HALF_DOWN);
        network.learnRate = (float) tlr.doubleValue();
    }

    public void gradClipping(Network network) {
        for (Layer layer : network.layerList) {
            if (layer.diffW != null) {
                //				System.out.println(layer.getLayerType()+"-diffW");
                GradClipping.gradClipping(layer.diffW, 1e-7f);
            }
            if (layer.diffB != null) {
                //				System.out.println("diffB");
                GradClipping.gradClipping(layer.diffB, 1e-7f);
            }
        }
    }

    public void transforms(Tensor trainData, Tensor transData, float[] mean, float[] std) {
        /**
         * 随机裁剪

         */
        DataTransforms.randomCrop(trainData, transData, 32, 32, 4);
        /**
         * 随机翻转

         */
        DataTransforms.randomHorizontalFilp(transData, transData);
        /**
         * normalize

         */
        DataTransforms.normalize(transData, transData, mean, std);
        /**
         * cutcout

         */
        DataTransforms.cutout(transData, transData, 16, 1);
        System.out.println("data transform finish.");
    }

    public void transforms2(Tensor trainData, Tensor transData, float[] mean, float[] std) {
        /**
         * normalize

         */
        DataTransforms.normalize(trainData, transData, mean, std);
        System.out.println("data transform finish.");
    }
}

