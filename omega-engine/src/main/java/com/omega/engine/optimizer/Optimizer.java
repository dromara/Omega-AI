package com.omega.engine.optimizer;

import com.omega.common.data.Tensor;
import com.omega.common.task.TaskEngine;
import com.omega.common.utils.JsonUtils;
import com.omega.common.utils.LabelUtils;
import com.omega.common.utils.MatrixOperation;
import com.omega.common.utils.RandomUtils;
import com.omega.engine.check.BaseCheck;
import com.omega.engine.nn.data.BaseData;
import com.omega.engine.nn.layer.YoloLayer;
import com.omega.engine.nn.network.Network;
import com.omega.engine.nn.network.RunModel;
import com.omega.engine.nn.network.Yolo;
import com.omega.engine.optimizer.lr.GDDecay;
import com.omega.engine.optimizer.lr.HalfDecay;
import com.omega.engine.optimizer.lr.LRDecay;
import com.omega.engine.optimizer.lr.LearnRateUpdate;
import com.omega.example.transformer.tokenizer.bertTokenizer.BertTokenizer;
import com.omega.example.transformer.utils.BPETokenizer;
import com.omega.example.transformer.utils.SentencePieceTokenizer;
import com.omega.example.transformer.utils.tokenizers.Tokenizer;
import com.omega.example.yolo.data.BaseDataLoader;
import com.omega.example.yolo.data.DetectionDataLoader;
import com.omega.example.yolo.model.YoloBox;
import com.omega.example.yolo.model.YoloDetection;
import com.omega.example.yolo.utils.YoloDataLoader;
import com.omega.example.yolo.utils.YoloDecode;
import com.omega.example.yolo.utils.YoloUtils;

import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;

/**
 * Optimizer
 *
 * @author Administrator
 */
public abstract class Optimizer {
    public int batchIndex = 1;
    public int trainIndex = 1;
    public int batchSize = 1;
    public int dataSize = 0;
    public Tensor loss;
    public Tensor lossDiff;
    public int trainTime = 10;
    public int minTrainTime = 10000;
    public float currentError = 1.0f;
    public float error = 0.01f;
    public Network network;
    public LearnRateUpdate learnRateUpdate = LearnRateUpdate.NONE;
    public BaseCheck check;
    public int burnIn = 300;
    public int power = 4;
    public float scale = 0.1f;
    public int step = 500;
    public float gama = 0.9999f;
    public float lr = 0.1f;
    public boolean isOnline = false;
    public int lrStartTime = 5;
    public float max_lr = 0.0001f;
    public float min_lr = 0;
    public int trainMax = 100;
    public float min_loss = Float.POSITIVE_INFINITY;
    public int counter = 0;
    public int[] lr_step;
    public int warmUpTime = 0;
    public int lrDecayIters = 0;
    private String sid;
    private TaskEngine trainEngine;
    private BaseData trainingData;
    private BaseData testData;
    private boolean warmUp = false;

    /**
     * @param network
     * @param trainTime
     * @param error
     * @throws Exception
     */
    public Optimizer(Network network, int batchSize, int trainTime, float error, boolean warmUp) throws Exception {
        this.network = network;
        this.trainTime = trainTime;
        this.error = error;
        this.lr = network.learnRate;
        this.max_lr = network.learnRate;
        this.warmUp = warmUp;
        this.network.init();
    }

    /**
     * @param network
     * @param trainTime
     * @param error
     * @throws Exception
     */
    public Optimizer(Network network, int batchSize, int trainTime, int minTrainTime, float error, boolean warmUp) throws Exception {
        this.network = network;
        this.trainTime = trainTime;
        this.minTrainTime = minTrainTime;
        this.error = error;
        this.warmUp = warmUp;
        this.lr = network.learnRate;
        this.max_lr = network.learnRate;
        this.network.init();
    }

    /**
     * @param network
     * @param trainTime
     * @param error
     * @throws Exception
     */
    public Optimizer(Network network, int batchSize, int trainTime, int minTrainTime, float error, boolean warmUp, LearnRateUpdate learnRateUpdate) throws Exception {
        this.network = network;
        this.trainTime = trainTime;
        this.minTrainTime = minTrainTime;
        this.error = error;
        this.warmUp = warmUp;
        this.learnRateUpdate = learnRateUpdate;
        this.lr = network.learnRate;
        this.max_lr = network.learnRate;
        this.network.init();
    }

    public static int entryIndex(int batch, int w, int h, int location, int entry, int outputs, int class_number) {
        int n = location / (w * h);
        int loc = location % (w * h);
        return batch * outputs + n * w * h * (4 + class_number + 1) + entry * w * h + loc;
    }

    public static float testLoss2(float[] output, float[] label) {
        //		System.out.println(JsonUtils.toJson(label));
        float sum = 0.0f;
        float loss = 0.0f;
        /**
         * max
         *
         */
        float max = MatrixOperation.max(output);
        /**
         * sum
         *
         */
        for (int i = 0; i < output.length; i++) {
            sum += Math.exp(output[i] - max);
        }
        /**
         * softmax + log + nlloss
         *
         */
        for (int i = 0; i < output.length; i++) {
            loss += (float) (-((output[i] - max) - Math.log(sum)) * label[i]);
        }
        return loss;
    }

    public static void main(String[] args) {
        float[] x = new float[]{0.6079413f, -1.1546507f, 1.444119f, 1.5811894f, 1.131686f, 1.5374337f, 0.39088273f, -0.19011068f, -0.010914803f, -1.4776193f};
        float[] l = new float[]{0, 0, 0, 1, 0, 0, 0, 0, 0, 0};
        System.out.println(testLoss2(x, l));
    }

    public abstract void train(BaseData trainingData);

    public abstract void train(BaseData trainingData, BaseData testData);

    public abstract void train(BaseData trainingData, BaseData validata, BaseData testData);

    public TaskEngine getTrainEngine() {
        return this.trainEngine;
    }

    public void setTrainEngine(TaskEngine trainEngine) {
        this.trainEngine = trainEngine;
    }

    public BaseData getTrainingData() {
        return trainingData;
    }

    public void setTrainingData(BaseData trainingData) {
        this.trainingData = trainingData;
    }

    public BaseData getTestData() {
        return testData;
    }

    public void setTestData(BaseData testData) {
        this.testData = testData;
    }

    public void updateLR(int[] lr_step) {
        if (warmUp && batchIndex < burnIn) {
            this.network.learnRate = (float) (this.lr * Math.pow(batchIndex * 1.0f / burnIn * 1.0f, power));
        } else {
            switch (this.learnRateUpdate) {
                case LR_DECAY:
                    this.network.learnRate = LRDecay.decayedLR(this.max_lr, this.network.learnRate, this.trainIndex, 5);
                    break;
                case GD_GECAY:
                    this.network.learnRate = GDDecay.decayedLR(this.max_lr, this.trainIndex);
                    break;
                case NONE:
                    break;
                case CONSTANT:
                    break;
                case COSINE:
                    if (this.trainIndex >= lrStartTime) {
                        this.network.learnRate = (float) (0.5d * max_lr * (Math.cos(this.trainIndex * Math.PI / trainTime) + 1.0d)) * this.network.learnRate;
                    } else {
                        this.network.learnRate = this.trainIndex * this.network.learnRate / lrStartTime;
                    }
                    break;
                case COSINE_ANNEALING:
                    this.network.learnRate = (float) (min_lr + 0.5d * (max_lr - min_lr) * (Math.cos(this.trainIndex * Math.PI / this.trainMax) + 1.0d));
                    //				System.out.println(this.trainMax);
                    break;
                case RANDOM:
                    this.network.learnRate = (float) Math.pow(RandomUtils.getInstance().nextFloat(), power) * this.lr;
                    break;
                case POLY:
                    float t = batchIndex * 1.0f / trainTime / dataSize * batchSize;
                    this.network.learnRate = (float) (this.lr * Math.pow((1.0f - t), power));
                    break;
                case STEP:
                    this.network.learnRate = (float) (this.lr * Math.pow(this.scale, batchIndex / step));
                    break;
                case EXP:
                    this.network.learnRate = (float) (this.lr * Math.pow(this.gama, batchIndex));
                    break;
                case SIG:
                    this.network.learnRate = (float) (this.lr / (1.0f + Math.pow(Math.E, this.gama * (batchIndex - step))));
                    break;
                case HALF:
                    if (counter % 10 == 0) {
                        this.network.learnRate = HalfDecay.decayedLR(this.network.learnRate);
                    }
                    break;
                case SMART_HALF:
                    if (this.learnRateUpdate == LearnRateUpdate.SMART_HALF) {
                        if (lr_step != null) {
                            for (int index : lr_step) {
                                if (index == this.trainIndex) {
                                    this.network.learnRate = this.network.learnRate * 0.5f;
                                }
                            }
                        } else if (this.trainIndex % 200 == 0) {
                            this.network.learnRate = this.network.learnRate * 0.5f;
                        }
                    }
                    break;
                default:
                    break;
            }
        }
    }

    public void updateLR(int[] lr_step, int step) {
        int it = (trainIndex - 1) * this.dataSize + step;
        if (warmUp && it < warmUpTime) {
            this.network.learnRate = this.lr * it / warmUpTime;
        } else {
            switch (this.learnRateUpdate) {
                case LR_DECAY:
                    this.network.learnRate = LRDecay.decayedLR(this.max_lr, this.network.learnRate, this.trainIndex, 5);
                    break;
                case GD_GECAY:
                    this.network.learnRate = GDDecay.decayedLR(this.max_lr, this.trainIndex);
                    break;
                case NONE:
                    break;
                case CONSTANT:
                    break;
                case COSINE:
                    if (it > lrDecayIters) {
                        this.network.learnRate = this.min_lr;
                    }
                    double decay_ratio = (it - warmUpTime) * 1.0d / (lrDecayIters - warmUpTime) * 1.0d;
                    double coeff = 0.5d * (1.0d + Math.cos(Math.PI * decay_ratio));
                    this.network.learnRate = (float) (this.min_lr + coeff * (this.lr - this.min_lr));
                    break;
                case COSINE_ANNEALING:
                    this.network.learnRate = (float) (min_lr + 0.5d * (max_lr - min_lr) * (Math.cos(this.trainIndex * Math.PI / this.trainMax) + 1.0d));
                    //				System.out.println(this.trainMax);
                    break;
                case RANDOM:
                    this.network.learnRate = (float) Math.pow(RandomUtils.getInstance().nextFloat(), power) * this.lr;
                    break;
                case POLY:
                    float t = batchIndex * 1.0f / trainTime / dataSize * batchSize;
                    this.network.learnRate = (float) (this.lr * Math.pow((1.0f - t), power));
                    break;
                case STEP:
                    this.network.learnRate = (float) (this.lr * Math.pow(this.scale, batchIndex / step));
                    break;
                case EXP:
                    this.network.learnRate = (float) (this.lr * Math.pow(this.gama, batchIndex));
                    break;
                case SIG:
                    this.network.learnRate = (float) (this.lr / (1.0f + Math.pow(Math.E, this.gama * (batchIndex - step))));
                    break;
                case HALF:
                    if (counter % 10 == 0) {
                        this.network.learnRate = HalfDecay.decayedLR(this.network.learnRate);
                    }
                    break;
                case SMART_HALF:
                    if (this.learnRateUpdate == LearnRateUpdate.SMART_HALF) {
                        if (lr_step != null) {
                            for (int index : lr_step) {
                                if (index == this.trainIndex) {
                                    this.network.learnRate = this.network.learnRate * 0.1f;
                                }
                            }
                        } else if (this.trainIndex % 200 == 0) {
                            this.network.learnRate = this.network.learnRate * 0.5f;
                        }
                    }
                    break;
                default:
                    break;
            }
        }
    }

    public float updateLR(int[] lr_step, float learnRate, float olr) {
        if (warmUp && batchIndex < burnIn) {
            learnRate = (float) (this.lr * Math.pow(batchIndex * 1.0f / burnIn * 1.0f, power));
        } else {
            switch (this.learnRateUpdate) {
                case LR_DECAY:
                    learnRate = LRDecay.decayedLR(this.max_lr, learnRate, this.trainIndex, 5);
                    break;
                case GD_GECAY:
                    learnRate = GDDecay.decayedLR(this.max_lr, this.trainIndex);
                    break;
                case NONE:
                    break;
                case CONSTANT:
                    break;
                case COSINE:
                    if (this.trainIndex >= lrStartTime) {
                        learnRate = (float) (0.5d * max_lr * (Math.cos(this.trainIndex * Math.PI / trainTime) + 1.0d)) * learnRate;
                    } else {
                        learnRate = this.trainIndex * learnRate / lrStartTime;
                    }
                    break;
                case COSINE_ANNEALING:
                    learnRate = (float) (min_lr + 0.5d * (max_lr - min_lr) * (Math.cos(this.trainIndex * Math.PI / this.trainMax) + 1.0d));
                    //				System.out.println(this.trainMax);
                    break;
                case RANDOM:
                    learnRate = (float) Math.pow(RandomUtils.getInstance().nextFloat(), power) * this.lr;
                    break;
                case POLY:
                    float t = batchIndex * 1.0f / trainTime / dataSize * batchSize;
                    learnRate = (float) (olr * Math.pow((1.0f - t), power));
                    break;
                case STEP:
                    learnRate = (float) (olr * Math.pow(this.scale, batchIndex / step));
                    break;
                case EXP:
                    learnRate = (float) (olr * Math.pow(this.gama, batchIndex));
                    break;
                case SIG:
                    learnRate = (float) (olr / (1.0f + Math.pow(Math.E, this.gama * (batchIndex - step))));
                    break;
                case HALF:
                    if (counter % 10 == 0) {
                        learnRate = HalfDecay.decayedLR(learnRate);
                    }
                    break;
                case SMART_HALF:
                    if (this.learnRateUpdate == LearnRateUpdate.SMART_HALF) {
                        if (lr_step != null) {
                            for (int index : lr_step) {
                                if (index == this.trainIndex) {
                                    learnRate = learnRate * 0.1f;
                                }
                            }
                        } else if (this.trainIndex % 200 == 0) {
                            learnRate = learnRate * 0.5f;
                        }
                    }
                    break;
                default:
                    break;
            }
        }
        return learnRate;
    }

    public void updateLR(float loss) {
        if (warmUp && batchIndex < burnIn) {
            this.network.learnRate = (float) (this.lr * Math.pow(batchIndex * 1.0f / burnIn * 1.0f, power));
        } else {
            switch (this.learnRateUpdate) {
                case LR_DECAY:
                    this.network.learnRate = LRDecay.decayedLR(this.max_lr, this.network.learnRate, this.trainIndex, 5);
                    break;
                case GD_GECAY:
                    this.network.learnRate = GDDecay.decayedLR(this.max_lr, this.trainIndex);
                    break;
                case NONE:
                    break;
                case CONSTANT:
                    break;
                case COSINE:
                    if (this.trainIndex >= lrStartTime) {
                        this.network.learnRate = (float) (0.5d * max_lr * (Math.cos(this.trainIndex / trainTime * Math.PI)) + 1.0d) * this.network.learnRate;
                    } else {
                        this.network.learnRate = this.trainIndex * this.network.learnRate / lrStartTime;
                    }
                    break;
                case RANDOM:
                    this.network.learnRate = (float) Math.pow(RandomUtils.getInstance().nextFloat(), power) * this.lr;
                    break;
                case POLY:
                    float t = batchIndex * 1.0f / trainTime / dataSize * batchSize;
                    this.network.learnRate = (float) (this.lr * Math.pow((1.0f - t), power));
                    break;
                case STEP:
                    this.network.learnRate = (float) (this.lr * Math.pow(this.scale, batchIndex / step));
                    break;
                case EXP:
                    this.network.learnRate = (float) (this.lr * Math.pow(this.gama, batchIndex));
                    break;
                case SIG:
                    this.network.learnRate = (float) (this.lr / (1.0f + Math.pow(Math.E, this.gama * (batchIndex - step))));
                    break;
                case HALF:
                    if (counter % 10 == 0) {
                        this.network.learnRate = HalfDecay.decayedLR(this.network.learnRate);
                    }
                    break;
                case SMART_HALF:
                    if (loss <= min_loss) {
                        System.out.println("Validation loss decreased (" + min_loss + " --> " + loss + ")");
                        min_loss = loss;
                        this.counter = 0;
                    } else {
                        this.counter++;
                        System.out.println("Validation loss decreased (" + min_loss + " < " + loss + ") update counter:" + this.counter);
                    }
                    if (this.counter >= 9) {
                        this.network.learnRate = HalfDecay.decayedLR(this.network.learnRate);
                        this.counter = 0;
                    }
                    break;
            }
        }
    }

    public float test(BaseData testData) {
        // TODO Auto-generated method stub
        float error = 0.0f;
        float trueCount = 0;
        Tensor sample = new Tensor(1, testData.channel, testData.height, testData.width, true);
        this.network.RUN_MODEL = RunModel.TEST;
        for (int n = 0; n < testData.number; n++) {
            //			float[] onceError = MatrixOperation.subtraction(output, testData.dataLabel[i]);
            testData.getOnceData(n, sample);
            Tensor output = this.network.predict(sample);
            output.syncHost();
            String label = testData.labels[n];
            //			System.out.println(JsonUtils.toJson(output.data));
            String predictLabel = LabelUtils.vectorTolabel(output.data, testData.labelSet);
            //			System.out.println("final output:"+JsonUtils.toJson(output.data));
            //
            //			System.out.println(label+"="+predictLabel+":"+label.equals(predictLabel));
            if (!label.equals(predictLabel)) {
                //				System.out.println("index:"+n+"::"+JsonUtils.toJson(output)+"==>predictLabel:"+predictLabel+"==label:"+label+":"+label.equals(predictLabel));
            } else {
                trueCount++;
            }
        }
        error = trueCount / testData.number;
        System.out.println("准确率:" + error * 100 + "%");
        return error;
    }

    public float test(BaseData testData, int batchSize) {
        // TODO Auto-generated method stub
        float error = 0.0f;
        float trueCount = 0;
        long startTime = System.nanoTime();
        this.network.RUN_MODEL = RunModel.TEST;
        Tensor input = new Tensor(batchSize, testData.channel, testData.height, testData.width, true);
        Tensor label = new Tensor(batchSize, testData.label.channel, testData.label.height, testData.label.width);
        int itc = new BigDecimal(testData.number).divide(new BigDecimal(batchSize), 0, BigDecimal.ROUND_DOWN).intValue();
        for (int pageIndex = 0; pageIndex < itc; pageIndex++) {
            testData.getBatchData(pageIndex, batchSize, input, label);
            input.hostToDevice();
            Tensor output = this.network.predict(input);
            output.syncHost();
            trueCount += this.accuracyTrueCount(output, label, testData.labelSet);
        }
        error = trueCount / itc / batchSize;
        System.out.println("training[" + this.trainIndex + "] vail accuracy:{" + error * 100 + "%}" + " [costTime:" + (System.nanoTime() - startTime) / 1e6 + "ms.]");
        return error;
    }

    public float testAndLoss(BaseData testData, int batchSize) {
        // TODO Auto-generated method stub
        float error = 0.0f;
        float trueCount = 0;
        float vailLoss = 0.0f;
        long startTime = System.nanoTime();
        this.network.RUN_MODEL = RunModel.TEST;
        Tensor input = new Tensor(batchSize, testData.channel, testData.height, testData.width, true);
        Tensor label = new Tensor(batchSize, testData.label.channel, testData.label.height, testData.label.width);
        int itc = new BigDecimal(testData.number).divide(new BigDecimal(batchSize), 0, BigDecimal.ROUND_DOWN).intValue();
        for (int pageIndex = 0; pageIndex < itc; pageIndex++) {
            testData.getBatchData(pageIndex, batchSize, input, label);
            input.hostToDevice();
            label.hostToDevice();
            Tensor output = this.network.predict(input);
            /**
             * loss
             *
             */
            Tensor loss = this.network.loss(output, label);
            /**
             * current time error
             *
             */
            vailLoss += MatrixOperation.sum(loss.syncHost()) / batchSize;
            output.syncHost();
            trueCount += this.accuracyTrueCount(output, label, testData.labelSet);
        }
        error = trueCount / itc / batchSize;
        vailLoss = vailLoss / itc;
        System.out.println("training[" + this.trainIndex + "] vail accuracy:{" + error * 100 + "%} vail loss:{" + vailLoss + "} " + " [costTime:" + (System.nanoTime() - startTime) / 1e6 + "ms.]");
        return vailLoss;
    }

    public float testAndLoss(BaseData testData, Tensor input, Tensor label, int batchSize) {
        // TODO Auto-generated method stub
        float error = 0.0f;
        float trueCount = 0;
        float vailLoss = 0.0f;
        long startTime = System.nanoTime();
        this.network.RUN_MODEL = RunModel.TEST;
        int itc = new BigDecimal(testData.number).divide(new BigDecimal(batchSize), 0, BigDecimal.ROUND_DOWN).intValue();
        for (int pageIndex = 0; pageIndex < itc; pageIndex++) {
            testData.getBatchData(pageIndex, batchSize, input, label);
            input.hostToDevice();
            label.hostToDevice();
            Tensor output = this.network.predict(input);
            /**
             * loss
             *
             */
            Tensor loss = this.network.loss(output, label);
            /**
             * current time error
             *
             */
            if (loss.isHasGPU()) {
                vailLoss += MatrixOperation.sum(loss.syncHost());
            } else {
                vailLoss += MatrixOperation.sum(loss.data);
            }
            output.syncHost();
            int currentError = this.accuracyTrueCount(output, label, testData.labelSet);
            trueCount += currentError;
            //			System.out.println("vaildating["+pageIndex+"] vail accuracy:{"+ currentError * 100 / batchSize +"%} vail loss:{"+currentLoss / batchSize+"}]");
        }
        error = trueCount / itc / batchSize;
        vailLoss = vailLoss / itc / batchSize;
        System.out.println("training[" + this.trainIndex + "] vail accuracy:{" + error * 100 + "%} vail loss:{" + vailLoss + "} " + " [costTime:" + (System.nanoTime() - startTime) / 1e6 + "ms.]");
        return vailLoss;
    }

    public float testAndLoss(BaseDataLoader testData, Tensor input, Tensor label, int batchSize, BaseCheck check) {
        // TODO Auto-generated method stub
        float vailLoss = 0.0f;
        long startTime = System.nanoTime();
        this.network.RUN_MODEL = RunModel.TEST;
        int itc = new BigDecimal(testData.number).divide(new BigDecimal(batchSize), 0, BigDecimal.ROUND_UP).intValue();
        float accuracy = 0.0f;
        for (int pageIndex = 0; pageIndex < itc; pageIndex++) {
            testData.loadData(pageIndex, batchSize, input, label);
            input.hostToDevice();
            label.hostToDevice();
            Tensor output = this.network.predict(input);
            /**
             * current time error
             *
             */
            Tensor loss = this.network.loss(output, label);
            if (loss.isHasGPU()) {
                vailLoss += MatrixOperation.sum(loss.syncHost());
            } else {
                vailLoss += MatrixOperation.sum(loss.data);
            }
            output.syncHost();
            accuracy += check.check(output, label, testData.labelSet, true);
        }
        vailLoss = vailLoss / itc;
        System.out.println("test[" + this.trainIndex + "] vail loss:{" + vailLoss + "} (accuracy:" + accuracy / testData.number * 100 + "%) [costTime:" + (System.nanoTime() - startTime) / 1e6 + "ms.]");
        return vailLoss;
    }

    public float testObjectRecognition(BaseData testData, Tensor input, Tensor label, int batchSize) {
        // TODO Auto-generated method stub
        float vailLoss = 0.0f;
        long startTime = System.nanoTime();
        this.network.RUN_MODEL = RunModel.TEST;
        int itc = new BigDecimal(testData.number).divide(new BigDecimal(batchSize), 0, BigDecimal.ROUND_UP).intValue();
        for (int pageIndex = 0; pageIndex < itc; pageIndex++) {
            testData.getBatchData(pageIndex, batchSize, input, label);
            input.hostToDevice();
            label.hostToDevice();
            Tensor output = this.network.predict(input);
            /**
             * current time error
             *
             */
            Tensor loss = this.network.loss(output, label);
            if (loss.isHasGPU()) {
                vailLoss += MatrixOperation.sum(loss.syncHost());
            } else {
                vailLoss += MatrixOperation.sum(loss.data);
            }
        }
        vailLoss = vailLoss / itc / batchSize;
        System.out.println("test[" + this.trainIndex + "] vail loss:{" + vailLoss + "} [costTime:" + (System.nanoTime() - startTime) / 1e6 + "ms.]");
        return vailLoss;
    }

    public float testObjectRecognition(BaseDataLoader testData, Tensor input, Tensor label, int batchSize) {
        // TODO Auto-generated method stub
        float vailLoss = 0.0f;
        long startTime = System.nanoTime();
        this.network.RUN_MODEL = RunModel.TEST;
        int itc = new BigDecimal(testData.number).divide(new BigDecimal(batchSize), 0, BigDecimal.ROUND_UP).intValue();
        for (int pageIndex = 0; pageIndex < itc; pageIndex++) {
            testData.loadData(pageIndex, batchSize, input, label);
            input.hostToDevice();
            label.hostToDevice();
            Tensor output = this.network.predict(input);
            /**
             * current time error
             *
             */
            Tensor loss = this.network.loss(output, label);
            if (loss.isHasGPU()) {
                vailLoss += MatrixOperation.sum(loss.syncHost());
            } else {
                vailLoss += MatrixOperation.sum(loss.data);
            }
        }
        vailLoss = vailLoss / itc / batchSize;
        System.out.println("test[" + this.trainIndex + "] vail loss:{" + vailLoss + "} [costTime:" + (System.nanoTime() - startTime) / 1e6 + "ms.]");
        return vailLoss;
    }

    public float testObjectRecognition(DetectionDataLoader testData, Tensor input, Tensor label, int batchSize) {
        // TODO Auto-generated method stub
        float vailLoss = 0.0f;
        long startTime = System.nanoTime();
        this.network.RUN_MODEL = RunModel.TEST;
        int itc = new BigDecimal(testData.number).divide(new BigDecimal(batchSize), 0, BigDecimal.ROUND_UP).intValue();
        for (int pageIndex = 0; pageIndex < itc; pageIndex++) {
            testData.loadData(pageIndex, batchSize, input, label);
            Tensor output = this.network.predict(input);
            /**
             * current time error
             *
             */
            Tensor loss = this.network.loss(output, label);
            if (loss.isHasGPU()) {
                vailLoss += MatrixOperation.sum(loss.syncHost());
            } else {
                vailLoss += MatrixOperation.sum(loss.data);
            }
        }
        vailLoss = vailLoss / itc / batchSize;
        System.out.println("test[" + this.trainIndex + "] vail loss:{" + vailLoss + "} [costTime:" + (System.nanoTime() - startTime) / 1e6 + "ms.]");
        return vailLoss;
    }

    public float testObjectRecognitionOutputs(BaseData testData, Tensor input, Tensor label, int batchSize) {
        // TODO Auto-generated method stub
        long startTime = System.nanoTime();
        this.network.RUN_MODEL = RunModel.TEST;
        int itc = new BigDecimal(testData.number).divide(new BigDecimal(batchSize), 0, BigDecimal.ROUND_UP).intValue();
        Yolo network = (Yolo) this.network;
        for (int pageIndex = 0; pageIndex < itc; pageIndex++) {
            testData.getBatchData(pageIndex, batchSize, input, label);
            input.hostToDevice();
            label.hostToDevice();
            Tensor[] output = network.predicts(input);
            /**
             * current time error
             *
             */
            network.loss(output, label);
        }
        System.out.println("test[" + this.trainIndex + "] [costTime:" + (System.nanoTime() - startTime) / 1e6 + "ms.]");
        return 0.0f;
    }

    public float testObjectRecognitionOutputs(BaseDataLoader testData, Tensor input, Tensor label, int batchSize) {
        // TODO Auto-generated method stub
        long startTime = System.nanoTime();
        this.network.RUN_MODEL = RunModel.TEST;
        int itc = new BigDecimal(testData.number).divide(new BigDecimal(batchSize), 0, BigDecimal.ROUND_UP).intValue();
        Yolo network = (Yolo) this.network;
        for (int pageIndex = 0; pageIndex < itc; pageIndex++) {
            testData.loadData(pageIndex, batchSize, input, label);
            input.hostToDevice();
            label.hostToDevice();
            if (network.outputNum > 1) {
                Tensor[] output = network.predicts(input);
                /**
                 * current time error
                 *
                 */
                network.loss(output, label);
            } else {
                Tensor output = network.predict(input);
                /**
                 * current time error
                 *
                 */
                network.loss(output, label);
            }
        }
        System.out.println("test[" + this.trainIndex + "] [costTime:" + (System.nanoTime() - startTime) / 1e6 + "ms.]");
        return 0.0f;
    }

    public float testObjectRecognitionOutputs(DetectionDataLoader testData, Tensor input, Tensor label, int batchSize) {
        // TODO Auto-generated method stub
        long startTime = System.nanoTime();
        this.network.RUN_MODEL = RunModel.TEST;
        int itc = new BigDecimal(testData.number).divide(new BigDecimal(batchSize), 0, BigDecimal.ROUND_UP).intValue();
        Yolo network = (Yolo) this.network;
        for (int pageIndex = 0; pageIndex < itc; pageIndex++) {
            testData.loadData(pageIndex, batchSize, input, label);
            if (network.outputNum > 1) {
                Tensor[] output = network.predicts(input);
                /**
                 * current time error
                 *
                 */
                network.loss(output, label);
            } else {
                Tensor output = network.predict(input);
                /**
                 * current time error
                 *
                 */
                network.loss(output, label);
            }
        }
        System.out.println("test[" + this.trainIndex + "] [costTime:" + (System.nanoTime() - startTime) / 1e6 + "ms.]");
        return 0.0f;
    }

    public List<Tensor> predictObjectRecognitionOutputs(DetectionDataLoader testData, Tensor input, int batchSize) {
        // TODO Auto-generated method stub
        List<Tensor> results = new ArrayList<Tensor>();
        long startTime = System.nanoTime();
        this.network.RUN_MODEL = RunModel.TEST;
        int itc = new BigDecimal(testData.number).divide(new BigDecimal(batchSize), 0, BigDecimal.ROUND_UP).intValue();
        Yolo network = (Yolo) this.network;
        for (int pageIndex = 0; pageIndex < itc; pageIndex++) {
            testData.loadData(pageIndex, batchSize, input);
            if (network.outputNum > 1) {
                Tensor[] outputs = network.predicts(input);
                for (Tensor output : outputs) {
                    results.add(output);
                }
            } else {
                Tensor output = network.predict(input);
                results.add(output);
            }
        }
        System.out.println("test[" + this.trainIndex + "] [costTime:" + (System.nanoTime() - startTime) / 1e6 + "ms.]");
        return results;
    }

    public float testObjectRecognitionOutputs(BaseData testData, int batchSize) {
        // TODO Auto-generated method stub
        long startTime = System.nanoTime();
        this.network.RUN_MODEL = RunModel.TEST;
        int itc = new BigDecimal(testData.number).divide(new BigDecimal(batchSize), 0, BigDecimal.ROUND_UP).intValue();
        Tensor input = new Tensor(batchSize, testData.channel, testData.height, testData.width, true);
        Yolo net = (Yolo) this.network;
        for (int pageIndex = 0; pageIndex < itc; pageIndex++) {
            testData.getBatchData(pageIndex, batchSize, input);
            input.hostToDevice();
            Tensor[] output = net.predicts(input);
            for (int l = 0; l < net.outputLayers.size(); l++) {
                YoloLayer layer = (YoloLayer) net.outputLayers.get(l);
                for (int b = 0; b < output[l].number; b++) {
                    for (int i = 0; i < output[l].height * output[l].width; i++) {
                        int row = i / output[l].width;
                        int col = i % output[l].width;
                        for (int n = 0; n < layer.bbox_num; n++) {
                            int n_index = n * output[l].width * output[l].height + row * output[l].width + col;
                            int obj_index = entryIndex(b, output[l].width, output[l].height, n_index, 4, layer.outputs, layer.class_number);
                            float objectness = output[l].data[obj_index];
                            if (objectness > 0.1f) {
                                System.out.println(objectness);
                            }
                        }
                    }
                }
            }
        }
        System.out.println("test[" + this.trainIndex + "] [costTime:" + (System.nanoTime() - startTime) / 1e6 + "ms.]");
        return 0.0f;
    }

    public float[][][] showObjectRecognition(BaseData testData, Tensor input, int batchSize) {
        // TODO Auto-generated method stub
        this.network.RUN_MODEL = RunModel.TEST;
        float[][][] bbox = new float[testData.number][YoloDecode.grid_size * YoloDecode.grid_size * YoloDecode.bbox_num][YoloDecode.class_number + 1 + 4];
        int itc = new BigDecimal(testData.number).divide(new BigDecimal(batchSize), 0, BigDecimal.ROUND_UP).intValue();
        for (int pageIndex = 0; pageIndex < itc; pageIndex++) {
            testData.getBatchData(pageIndex, batchSize, input);
            input.hostToDevice();
            Tensor output = this.network.predict(input);
            output.syncHost();
            float[][][] draw_bbox = YoloDecode.getDetection(output, testData.width, testData.height);
            if ((pageIndex + 1) * batchSize > testData.number) {
                System.arraycopy(draw_bbox, 0, bbox, pageIndex * batchSize, (pageIndex + 1) * batchSize - testData.number);
            } else {
                System.arraycopy(draw_bbox, 0, bbox, pageIndex * batchSize, batchSize);
            }
        }
        return bbox;
    }

    public List<YoloBox> showObjectRecognitionYoloV3(BaseData testData, int batchSize) {
        // TODO Auto-generated method stub
        this.network.RUN_MODEL = RunModel.TEST;
        List<YoloBox> list = new ArrayList<YoloBox>();
        int itc = new BigDecimal(testData.number).divide(new BigDecimal(batchSize), 0, BigDecimal.ROUND_UP).intValue();
        Tensor input = new Tensor(batchSize, testData.channel, testData.height, testData.width, true);
        Yolo net = (Yolo) this.network;
        for (int pageIndex = 0; pageIndex < itc; pageIndex++) {
            testData.getBatchData(pageIndex, batchSize, input);
            input.hostToDevice();
            Tensor[] output = net.predicts(input);
            YoloBox[] boxs = new YoloBox[input.number];
            for (int i = 0; i < net.outputLayers.size(); i++) {
                YoloLayer layer = (YoloLayer) net.outputLayers.get(i);
                YoloDetection[][] dets = YoloUtils.getYoloDetections(output[i], layer.anchors, layer.mask, layer.bbox_num, layer.outputs, layer.class_number, testData.width, testData.height, 0.5f);
                for (int j = 0; j < dets.length; j++) {
                    if (boxs[j] != null) {
                        boxs[j].getDets().addAll(new ArrayList<>(Arrays.asList(dets[j])));
                    } else {
                        YoloBox box = new YoloBox(dets[j]);
                        boxs[j] = box;
                    }
                }
            }
            list.addAll(new ArrayList<>(Arrays.asList(boxs)));
        }
        return list;
    }

    public List<YoloBox> showObjectRecognitionYoloV3(DetectionDataLoader testData, int batchSize) {
        // TODO Auto-generated method stub
        this.network.RUN_MODEL = RunModel.TEST;
        List<YoloBox> list = new ArrayList<YoloBox>();
        int itc = new BigDecimal(testData.number).divide(new BigDecimal(batchSize), 0, BigDecimal.ROUND_UP).intValue();
        Tensor input = new Tensor(batchSize, this.network.getChannel(), this.network.getHeight(), this.network.getWidth(), true);
        Yolo net = (Yolo) this.network;
        for (int pageIndex = 0; pageIndex < itc; pageIndex++) {
            testData.loadData(pageIndex, batchSize, input);
            Tensor[] output = net.predicts(input);
            YoloBox[] boxs = new YoloBox[input.number];
            for (int i = 0; i < net.outputLayers.size(); i++) {
                YoloLayer layer = (YoloLayer) net.outputLayers.get(i);
                YoloDetection[][] dets = YoloUtils.getYoloDetections(output[i], layer.anchors, layer.mask, layer.bbox_num, layer.outputs, layer.class_number, this.network.getHeight(), this.network.getWidth(), 0.5f);
                for (int j = 0; j < dets.length; j++) {
                    /**
                     * nms
                     *
                     */
                    nmsSort(dets[j], dets[j].length, layer.class_number, 0.7f);
                    if (boxs[j] != null) {
                        boxs[j].getDets().addAll(new ArrayList<>(Arrays.asList(dets[j])));
                    } else {
                        YoloBox box = new YoloBox(dets[j]);
                        boxs[j] = box;
                    }
                }
            }
            list.addAll(new ArrayList<YoloBox>(Arrays.asList(boxs)));
        }
        return list;
    }

    public List<YoloBox> showObjectRecognitionYoloV7(DetectionDataLoader testData, int batchSize) {
        // TODO Auto-generated method stub
        this.network.RUN_MODEL = RunModel.TEST;
        List<YoloBox> list = new ArrayList<YoloBox>();
        int itc = new BigDecimal(testData.number).divide(new BigDecimal(batchSize), 0, BigDecimal.ROUND_UP).intValue();
        Tensor input = new Tensor(batchSize, this.network.getChannel(), this.network.getHeight(), this.network.getWidth(), true);
        Yolo net = (Yolo) this.network;
        for (int pageIndex = 0; pageIndex < itc; pageIndex++) {
            testData.loadData(pageIndex, batchSize, input);
            Tensor[] output = net.predicts(input);
            YoloBox[] boxs = new YoloBox[input.number];
            for (int i = 0; i < net.outputLayers.size(); i++) {
                YoloLayer layer = (YoloLayer) net.outputLayers.get(i);
                YoloDetection[][] dets = YoloUtils.getYoloDetectionsV7(output[i], layer.anchors, layer.mask, layer.bbox_num, layer.outputs, layer.class_number, this.network.getHeight(), this.network.getWidth(), 0.5f);
                for (int j = 0; j < dets.length; j++) {
                    /**
                     * nms
                     *
                     */
                    nmsSort(dets[j], dets[j].length, layer.class_number, 0.7f);
                    if (boxs[j] != null) {
                        boxs[j].getDets().addAll(new ArrayList<>(Arrays.asList(dets[j])));
                    } else {
                        YoloBox box = new YoloBox(dets[j]);
                        boxs[j] = box;
                    }
                }
            }
            list.addAll(new ArrayList<YoloBox>(Arrays.asList(boxs)));
        }
        return list;
    }

    public void nmsSort(YoloDetection[] dets, int total, int classes, float thresh) {
        int k = total - 1;
        for (int i = 0; i <= k; i++) {
            //			System.out.println(dets[i]);
            if (dets[i] == null || dets[i].getObjectness() == 0) {
                YoloDetection swap = dets[i];
                dets[i] = dets[k];
                dets[k] = swap;
                k--;
                i--;
            }
        }
        total = k + 1;
        //		System.out.println("total:"+total);
        for (int c = 0; c < classes; c++) {
            for (int i = 0; i < total; i++) {
                if (dets[i] != null) {
                    dets[i].setSortClass(c);
                }
            }
            Arrays.sort(dets, new Comparator<YoloDetection>() {
                @Override
                public int compare(YoloDetection a, YoloDetection b) {
                    // TODO Auto-generated method stub
                    float diff = 0;
                    if (b.getSortClass() >= 0) {
                        diff = a.getProb()[b.getSortClass()] - b.getProb()[b.getSortClass()];
                    } else {
                        diff = a.getObjectness() - b.getObjectness();
                    }
                    if (diff < 0)
                        return 1;
                    else if (diff > 0)
                        return -1;
                    return 0;
                }
            });
            for (int i = 0; i < total; ++i) {
                if (dets[i].getProb()[c] == 0)
                    continue;
                float[] a = dets[i].getBbox();
                for (int j = i + 1; j < total; ++j) {
                    float[] b = dets[j].getBbox();
                    if (YoloUtils.box_iou(a, b) > thresh) {
                        dets[j].getProb()[c] = 0;
                    }
                }
            }
        }
    }

    public float[][][] showObjectRecognition(BaseData testData, int batchSize) {
        // TODO Auto-generated method stub
        this.network.RUN_MODEL = RunModel.TEST;
        float[][][] bbox = new float[testData.number][YoloDecode.grid_size * YoloDecode.grid_size * YoloDecode.bbox_num][YoloDecode.class_number + 1 + 4];
        int itc = new BigDecimal(testData.number).divide(new BigDecimal(batchSize), 0, BigDecimal.ROUND_UP).intValue();
        Tensor input = new Tensor(batchSize, testData.channel, testData.height, testData.width, true);
        for (int pageIndex = 0; pageIndex < itc; pageIndex++) {
            testData.getBatchData(pageIndex, batchSize, input);
            input.hostToDevice();
            Tensor output = this.network.predict(input);
            output.syncHost();
            float[][][] draw_bbox = YoloDecode.getDetection(output, testData.width, testData.height);
            if ((pageIndex + 1) * batchSize > testData.number) {
                /**
                 * 处理不能整除数据 由于批量预测的时候是向上补充最后一页的数据 所以在获取bbox的时候需要获取的下标应该是 pageIndex * batchSize -
                 * (batchSize - testData.number % batchSize)
                 *
                 */
                System.arraycopy(draw_bbox, batchSize - testData.number % batchSize, bbox, pageIndex * batchSize, testData.number % batchSize);
            } else {
                System.arraycopy(draw_bbox, 0, bbox, pageIndex * batchSize, batchSize);
            }
        }
        return bbox;
    }

    public float[][][] showObjectRecognition(YoloDataLoader testData, int batchSize, int classNum) {
        // TODO Auto-generated method stub
        System.out.println("start object recognition.");
        long start = System.currentTimeMillis();
        this.network.RUN_MODEL = RunModel.TEST;
        float[][][] bbox = new float[testData.number][YoloDecode.grid_size * YoloDecode.grid_size * YoloDecode.bbox_num][YoloDecode.class_number + 1 + 4];
        int itc = new BigDecimal(testData.number).divide(new BigDecimal(batchSize), 0, BigDecimal.ROUND_UP).intValue();
        Tensor input = new Tensor(batchSize, this.network.getChannel(), this.network.getHeight(), this.network.getWidth(), true);
        for (int pageIndex = 0; pageIndex < itc; pageIndex++) {
            testData.loadData(pageIndex, batchSize, input);
            input.hostToDevice();
            Tensor output = this.network.predict(input);
            output.syncHost();
            float[][][] draw_bbox = YoloDecode.getDetection(output, testData.getDataSet().width, testData.getDataSet().height, classNum);
            //			System.out.println(JsonUtils.toJson(draw_bbox));
            if ((pageIndex + 1) * batchSize > testData.number) {
                /**
                 * 处理不能整除数据 由于批量预测的时候是向上补充最后一页的数据 所以在获取bbox的时候需要获取的下标应该是 pageIndex * batchSize -
                 * (batchSize - testData.number % batchSize)
                 *
                 */
                System.arraycopy(draw_bbox, batchSize - testData.number % batchSize, bbox, pageIndex * batchSize, testData.number % batchSize);
            } else {
                System.arraycopy(draw_bbox, 0, bbox, pageIndex * batchSize, batchSize);
            }
        }
        System.out.println("finish object recognition[" + ((System.currentTimeMillis() - start) / 1000) + "s].");
        return bbox;
    }

    public float accuracy(Tensor output, Tensor labelData, String[] labelSet) {
        float error = 0.0f;
        float trueCount = 0;
        for (int n = 0; n < output.number; n++) {
            String label = LabelUtils.vectorTolabel(labelData.getByNumber(n), labelSet);
            String predictLabel = LabelUtils.vectorTolabel(output.getByNumber(n), labelSet);
            if (label.equals(predictLabel)) {
                trueCount++;
            }
        }
        error = trueCount / output.number * 100;
        return error;
    }

    public float accuracy(Tensor output, Tensor labelData) {
        float error = 0.0f;
        float trueCount = 0;
        //		System.out.println(JsonUtils.toJson(output.getByNumber(0)));
        for (int n = 0; n < output.number; n++) {
            int labelIndex = MatrixOperation.maxIndex(labelData.getByNumber(n));
            int predictIndex = MatrixOperation.maxIndex(output.getByNumber(n));
            //			System.out.println(JsonUtils.toJson(output.getByNumber(n)));
            //			System.out.println(predictIndex+":"+labelIndex);
            if (labelIndex == predictIndex) {
                trueCount++;
            }
        }
        error = trueCount / output.number * 100;
        return error;
    }

    public float accuracy(Tensor output, Tensor labelData, int time, int batchSize) {
        float error = 0.0f;
        float trueCount = 0;
        for (int n = 0; n < batchSize; n++) {
            boolean allRight = true;
            for (int t = 0; t < time; t++) {
                int predictIndex = MatrixOperation.maxIndex(output.getByNumber(t * batchSize + n));
                int labelIndex = MatrixOperation.maxIndex(labelData.getByNumber(t * batchSize + n));
                System.out.println(predictIndex + ":" + labelIndex);
                if (labelIndex != predictIndex) {
                    allRight = false;
                }
            }
            if (allRight) {
                trueCount++;
            }
        }
        error = trueCount / batchSize * 100;
        return error;
    }

    public float accuracyBatchFisrt(Tensor input, Tensor output, Tensor labelData, int time, int batchSize, String[] vocab, int igonre) {
        float error = 0.0f;
        float trueCount = 0;
        int max_score = 0;
        int max_index = 0;
        output.syncHost();
        for (int n = 0; n < batchSize; n++) {
            boolean allRight = true;
            int score = time;
            for (int t = 0; t < time; t++) {
                int predictIndex = MatrixOperation.maxIndex(output.getByNumber(n * time + t));
                int labelIndex = MatrixOperation.maxIndex(labelData.getByNumber(n * time + t));
                if (labelIndex != igonre && labelIndex != predictIndex) {
                    allRight = false;
                    score--;
                }
            }
            if (max_score <= score) {
                max_score = score;
                max_index = n;
            }
            if (allRight) {
                trueCount++;
            }
        }
        String max_itxt = "";
        String max_ptxt = "";
        String max_ltxt = "";
        for (int t = 0; t < time; t++) {
            int predictIndex = MatrixOperation.maxIndex(output.getByNumber(max_index * time + t));
            int labelIndex = MatrixOperation.maxIndex(labelData.getByNumber(max_index * time + t));
            int inputIndex = (int) input.data[max_index * time + t];
            max_ptxt += vocab[predictIndex];
            max_ltxt += vocab[labelIndex];
            max_itxt += vocab[inputIndex];
        }
        System.out.println("max_score:" + max_score);
        System.out.println("itxt:" + max_itxt);
        System.out.println("ptxt:" + max_ptxt);
        System.out.println("ltxt:" + max_ltxt);
        error = trueCount / batchSize * 100;
        return error;
    }

    public float accuracyBatchFisrt(Tensor input, float[] tmpInput, Tensor output, Tensor labelData, float[] tmpLabel, int time, int batchSize, BertTokenizer tokenizer, int igonre) {
        float error = 0.0f;
        float trueCount = 0;
        int max_score = -9999;
        int max_index = 0;
        int[] itxt = new int[time];
        int[] ptxt = new int[time];
        int[] ltxt = new int[time];
        for (int n = 0; n < batchSize; n++) {
            boolean allRight = true;
            int score = time;
            for (int t = 0; t < time; t++) {
                int predictIndex = MatrixOperation.maxIndex(output.getByNumber(n * time + t));
                //				int labelIndex = MatrixOperation.maxIndex(labelData.getByNumber(n * time + t));
                int labelIndex = (int) tmpLabel[n * time + t];
                if (labelIndex != igonre && labelIndex != predictIndex) {
                    allRight = false;
                    score--;
                }
            }
            if (max_score <= score) {
                max_score = score;
                max_index = n;
            }
            if (allRight) {
                trueCount++;
            }
        }
        for (int t = 0; t < time; t++) {
            int predictIndex = MatrixOperation.maxIndex(output.getByNumber(max_index * time + t));
            //			int labelIndex = MatrixOperation.maxIndex(labelData.getByNumber(n * time + t));
            int labelIndex = (int) tmpLabel[max_index * time + t];
            int inputIndex = (int) tmpInput[max_index * time + t];
            itxt[t] = inputIndex;
            ptxt[t] = predictIndex;
            ltxt[t] = labelIndex;
        }
        System.out.println("max_score:" + max_score);
        System.out.println("itxt:" + tokenizer.decode(itxt));
        System.out.println("ptxt:" + tokenizer.decode(ptxt));
        System.out.println("ltxt:" + tokenizer.decode(ltxt));
        error = trueCount / batchSize * 100;
        return error;
    }

    public float accuracyBatchFisrt(Tensor input, float[] tmpInput, Tensor output, float[] tmpLabel, int time, int batchSize, Tokenizer tokenizer, int igonre) {
        float error = 0.0f;
        float trueCount = 0;
        int max_score = -9999;
        int max_index = 0;
        int[] itxt = new int[time];
        int[] ptxt = new int[time];
        int[] ltxt = new int[time];
        for (int n = 0; n < batchSize; n++) {
            boolean allRight = true;
            int score = time;
            for (int t = 0; t < time; t++) {
                int predictIndex = MatrixOperation.maxIndex(output.getByNumber(n * time + t));
                //				int labelIndex = MatrixOperation.maxIndex(labelData.getByNumber(n * time + t));
                int labelIndex = (int) tmpLabel[n * time + t];
                if (labelIndex != igonre && labelIndex != predictIndex) {
                    allRight = false;
                    score--;
                }
            }
            if (max_score <= score) {
                max_score = score;
                max_index = n;
            }
            if (allRight) {
                trueCount++;
            }
        }
        for (int t = 0; t < time; t++) {
            int predictIndex = MatrixOperation.maxIndex(output.getByNumber(max_index * time + t));
            //			int labelIndex = MatrixOperation.maxIndex(labelData.getByNumber(n * time + t));
            int labelIndex = (int) tmpLabel[max_index * time + t];
            int inputIndex = (int) tmpInput[max_index * time + t];
            itxt[t] = inputIndex;
            ptxt[t] = predictIndex;
            ltxt[t] = labelIndex;
        }
        System.out.println("max_score:" + max_score);
        System.out.println("itxt:" + tokenizer.decode(itxt));
        System.out.println("ptxt:" + tokenizer.decode(ptxt));
        System.out.println("ltxt:" + tokenizer.decode(ltxt));
        error = trueCount / batchSize * 100;
        return error;
    }

    public float accuracyBatchFisrt(Tensor input, Tensor output, Tensor label, int time, int batchSize, Tokenizer tokenizer, int igonre) {
        float error = 0.0f;
        float trueCount = 0;
        int max_score = -9999;
        int max_index = 0;
        int[] itxt = new int[time];
        int[] ptxt = new int[time];
        int[] ltxt = new int[time];
        label.syncHost();
        input.syncHost();
        output.syncHost();
        for (int n = 0; n < batchSize; n++) {
            boolean allRight = true;
            int score = time;
            for (int t = 0; t < time; t++) {
                int predictIndex = MatrixOperation.maxIndex(output.getByNumber(n * time + t));
                //					int labelIndex = MatrixOperation.maxIndex(labelData.getByNumber(n * time + t));
                int labelIndex = (int) label.data[n * time + t];
                if (labelIndex != igonre && labelIndex != predictIndex) {
                    allRight = false;
                    score--;
                }
            }
            if (max_score <= score) {
                max_score = score;
                max_index = n;
            }
            if (allRight) {
                trueCount++;
            }
        }
        for (int t = 0; t < time; t++) {
            int predictIndex = MatrixOperation.maxIndex(output.getByNumber(max_index * time + t));
            //			int labelIndex = MatrixOperation.maxIndex(labelData.getByNumber(n * time + t));
            int labelIndex = (int) label.data[max_index * time + t];
            int inputIndex = (int) input.data[max_index * time + t];
            itxt[t] = inputIndex;
            ptxt[t] = predictIndex;
            ltxt[t] = labelIndex;
        }
        System.out.println("max_score:" + max_score);
        System.out.println("itxt:" + tokenizer.decode(itxt));
        System.out.println("ptxt:" + tokenizer.decode(ptxt));
        System.out.println("ltxt:" + tokenizer.decode(ltxt));
        error = trueCount / batchSize * 100;
        return error;
    }

    public float accuracyBatchFisrt(Tensor output, Tensor label, int time, int batchSize, Tokenizer tokenizer, int igonre) {
        float error = 0.0f;
        float trueCount = 0;
        int max_score = -9999;
        int max_index = 0;
        int[] ptxt = new int[time];
        int[] ltxt = new int[time];
        label.syncHost();
        output.syncHost();
        for (int n = 0; n < batchSize; n++) {
            boolean allRight = true;
            int score = time;
            for (int t = 0; t < time; t++) {
                int predictIndex = MatrixOperation.maxIndex(output.getByNumber(n * time + t));
                //					int labelIndex = MatrixOperation.maxIndex(labelData.getByNumber(n * time + t));
                int labelIndex = (int) label.data[n * time + t];
                if (labelIndex != igonre && labelIndex != predictIndex) {
                    allRight = false;
                    score--;
                }
            }
            if (max_score <= score) {
                max_score = score;
                max_index = n;
            }
            if (allRight) {
                trueCount++;
            }
        }
        for (int t = 0; t < time; t++) {
            int predictIndex = MatrixOperation.maxIndex(output.getByNumber(max_index * time + t));
            int labelIndex = (int) label.data[max_index * time + t];
            ptxt[t] = predictIndex;
            ltxt[t] = labelIndex;
        }
        System.out.println("max_score:" + max_score);
        System.out.println("ptxt:" + tokenizer.decode(ptxt));
        System.out.println("ltxt:" + tokenizer.decode(ltxt));
        error = trueCount / batchSize * 100;
        return error;
    }

    public float accuracyBatchFisrt(Tensor input, float[] tmpInput, Tensor output, Tensor labelData, float[] tmpLabel, int time, int batchSize, SentencePieceTokenizer tokenizer, int igonre) {
        float error = 0.0f;
        float trueCount = 0;
        int max_score = -9999;
        int max_index = 0;
        int[] itxt = new int[time];
        int[] ptxt = new int[time];
        int[] ltxt = new int[time];
        for (int n = 0; n < batchSize; n++) {
            boolean allRight = true;
            int score = time;
            for (int t = 0; t < time; t++) {
                int predictIndex = MatrixOperation.maxIndex(output.getByNumber(n * time + t));
                //				int labelIndex = MatrixOperation.maxIndex(labelData.getByNumber(n * time + t));
                int labelIndex = (int) tmpLabel[n * time + t];
                if (labelIndex != igonre && labelIndex != predictIndex) {
                    allRight = false;
                    score--;
                }
            }
            if (max_score <= score) {
                max_score = score;
                max_index = n;
            }
            if (allRight) {
                trueCount++;
            }
        }
        for (int t = 0; t < time; t++) {
            int predictIndex = MatrixOperation.maxIndex(output.getByNumber(max_index * time + t));
            //			int labelIndex = MatrixOperation.maxIndex(labelData.getByNumber(n * time + t));
            int labelIndex = (int) tmpLabel[max_index * time + t];
            int inputIndex = (int) tmpInput[max_index * time + t];
            itxt[t] = inputIndex;
            ptxt[t] = predictIndex;
            ltxt[t] = labelIndex;
        }
        System.out.println("max_score:" + max_score);
        System.out.println("itxt:" + tokenizer.decode(itxt));
        System.out.println("ptxt:" + tokenizer.decode(ptxt));
        System.out.println("ltxt:" + tokenizer.decode(ltxt));
        error = trueCount / batchSize * 100;
        return error;
    }

    public float accuracyBatchFisrt(Tensor input, Tensor output, Tensor labelData, int time, int batchSize, SentencePieceTokenizer tokenizer, int igonre) {
        float error = 0.0f;
        float trueCount = 0;
        int max_score = 0;
        int max_index = 0;
        int[] itxt = new int[time];
        int[] ptxt = new int[time];
        int[] ltxt = new int[time];
        for (int n = 0; n < batchSize; n++) {
            boolean allRight = true;
            int score = time;
            for (int t = 0; t < time; t++) {
                int predictIndex = MatrixOperation.maxIndex(output.getByNumber(n * time + t));
                int labelIndex = MatrixOperation.maxIndex(labelData.getByNumber(n * time + t));
                if (labelIndex != igonre && labelIndex != predictIndex) {
                    allRight = false;
                    score--;
                }
            }
            if (max_score <= score) {
                max_score = score;
                max_index = n;
            }
            if (allRight) {
                trueCount++;
            }
        }
        for (int t = 0; t < time; t++) {
            int predictIndex = MatrixOperation.maxIndex(output.getByNumber(max_index * time + t));
            //			int labelIndex = MatrixOperation.maxIndex(labelData.getByNumber(n * time + t));
            int labelIndex = (int) labelData.data[max_index * time + t];
            int inputIndex = (int) input.data[max_index * time + t];
            itxt[t] = inputIndex;
            ptxt[t] = predictIndex;
            ltxt[t] = labelIndex;
        }
        System.out.println("max_score:" + max_score);
        System.out.println("itxt:" + tokenizer.decode(itxt));
        System.out.println("ptxt:" + tokenizer.decode(ptxt));
        System.out.println("ltxt:" + tokenizer.decode(ltxt));
        error = trueCount / batchSize * 100;
        return error;
    }

    public float accuracyBatchFisrt(Tensor input, Tensor output, Tensor labelData, int time, int batchSize, BertTokenizer tokenizer, int igonre) {
        float error = 0.0f;
        float trueCount = 0;
        int max_score = -9999;
        int max_index = 0;
        int[] itxt = new int[time];
        int[] ptxt = new int[time];
        int[] ltxt = new int[time];
        for (int n = 0; n < batchSize; n++) {
            boolean allRight = true;
            int score = time;
            for (int t = 0; t < time; t++) {
                int predictIndex = MatrixOperation.maxIndex(output.getByNumber(n * time + t));
                //				int labelIndex = MatrixOperation.maxIndex(labelData.getByNumber(n * time + t));
                int labelIndex = (int) labelData.data[n * time + t];
                if (labelIndex != igonre && labelIndex != predictIndex) {
                    allRight = false;
                    score--;
                }
            }
            if (max_score <= score) {
                max_score = score;
                max_index = n;
            }
            if (allRight) {
                trueCount++;
            }
        }
        for (int t = 0; t < time; t++) {
            int predictIndex = MatrixOperation.maxIndex(output.getByNumber(max_index * time + t));
            //			int labelIndex = MatrixOperation.maxIndex(labelData.getByNumber(n * time + t));
            int labelIndex = (int) labelData.data[max_index * time + t];
            int inputIndex = (int) input.data[max_index * time + t];
            itxt[t] = inputIndex;
            ptxt[t] = predictIndex;
            ltxt[t] = labelIndex;
        }
        System.out.println("max_score:" + max_score);
        System.out.println("itxt:" + tokenizer.decode(itxt));
        System.out.println("ptxt:" + tokenizer.decode(ptxt));
        System.out.println("ltxt:" + tokenizer.decode(ltxt));
        error = trueCount / batchSize * 100;
        return error;
    }

    public float accuracyBatchFisrt(Tensor input, Tensor output, Tensor labelData, int time, int batchSize, BPETokenizer tokenizer, int igonre) {
        float error = 0.0f;
        float trueCount = 0;
        int max_score = 0;
        String max_itxt = "";
        String max_ptxt = "";
        String max_ltxt = "";
        for (int n = 0; n < batchSize; n++) {
            boolean allRight = true;
            int score = time;
            List<Integer> i_ids = new ArrayList<Integer>();
            List<Integer> p_ids = new ArrayList<Integer>();
            List<Integer> l_ids = new ArrayList<Integer>();
            for (int t = 0; t < time; t++) {
                int predictIndex = MatrixOperation.maxIndex(output.getByNumber(n * time + t));
                int labelIndex = MatrixOperation.maxIndex(labelData.getByNumber(n * time + t));
                int inputIndex = (int) input.data[n * time + t];
                i_ids.add(inputIndex);
                p_ids.add(predictIndex);
                l_ids.add(labelIndex);
                if (labelIndex != igonre && labelIndex != predictIndex) {
                    allRight = false;
                    score--;
                }
            }
            if (max_score <= score) {
                max_score = score;
                //				max_itxt = tokenizer.decode(i_ids);
                //				max_ptxt = tokenizer.decode(p_ids);
                //				max_ltxt = tokenizer.decode(l_ids);
                max_itxt = tokenizer.toText(i_ids);
                max_ptxt = tokenizer.toText(p_ids);
                max_ltxt = tokenizer.toText(l_ids);
            }
            if (allRight) {
                trueCount++;
            }
        }
        System.out.println("max_score:" + max_score);
        System.out.println("itxt:" + max_itxt);
        System.out.println("ptxt:" + max_ptxt);
        System.out.println("ltxt:" + max_ltxt);
        error = trueCount / batchSize * 100;
        return error;
    }

    public float accuracyBatchFisrt(Tensor input, Tensor output, Tensor labelData, int time, int batchSize, String[] vocab) {
        float error = 0.0f;
        float trueCount = 0;
        int max_score = 0;
        String max_itxt = "";
        String max_ptxt = "";
        String max_ltxt = "";
        for (int n = 0; n < batchSize; n++) {
            boolean allRight = true;
            int score = time;
            String itxt = "";
            String ptxt = "";
            String ltxt = "";
            for (int t = 0; t < time; t++) {
                int predictIndex = MatrixOperation.maxIndex(output.getByNumber(n * time + t));
                int labelIndex = MatrixOperation.maxIndex(labelData.getByNumber(n * time + t));
                int inputIndex = (int) input.data[n * time + t];
                ptxt += vocab[predictIndex];
                ltxt += vocab[labelIndex];
                itxt += vocab[inputIndex];
                if (labelIndex != predictIndex) {
                    allRight = false;
                    score--;
                }
            }
            if (max_score <= score) {
                max_score = score;
                max_itxt = itxt;
                max_ptxt = ptxt;
                max_ltxt = ltxt;
            }
            if (allRight) {
                trueCount++;
            }
        }
        System.out.println("max_score:" + max_score);
        System.out.println("itxt:" + max_itxt);
        System.out.println("ptxt:" + max_ptxt);
        System.out.println("ltxt:" + max_ltxt);
        error = trueCount / batchSize * 100;
        return error;
    }

    public float accuracyIdx(Tensor input, Tensor output, Tensor labelData, int time, int batchSize, String[] vocab) {
        float error = 0.0f;
        float trueCount = 0;
        int max_score = 0;
        int max_index = 0;
        String max_itxt = "";
        String max_ptxt = "";
        String max_ltxt = "";
        for (int n = 0; n < batchSize; n++) {
            boolean allRight = true;
            int score = time;
            for (int t = 0; t < time; t++) {
                int predictIndex = MatrixOperation.maxIndex(output.getByNumber(n * time + t));
                int labelIndex = (int) labelData.data[n * time + t];
                if (labelIndex != predictIndex) {
                    allRight = false;
                    score--;
                }
            }
            if (max_score <= score) {
                max_score = score;
                max_index = n;
            }
            if (allRight) {
                trueCount++;
            }
        }
        for (int t = 0; t < time; t++) {
            int predictIndex = MatrixOperation.maxIndex(output.getByNumber(max_index * time + t));
            int labelIndex = (int) labelData.data[max_index * time + t];
            int inputIndex = (int) input.data[max_index * time + t];
            max_ptxt += vocab[predictIndex];
            max_ltxt += vocab[labelIndex];
            max_itxt += vocab[inputIndex];
        }
        System.out.println("max_score:" + max_score);
        System.out.println("itxt:" + max_itxt);
        System.out.println("ptxt:" + max_ptxt);
        System.out.println("ltxt:" + max_ltxt);
        error = trueCount / batchSize * 100;
        return error;
    }

    public float accuracyIdx(Tensor input, Tensor output, Tensor labelData, int time, int batchSize, String[] vocab, int igonre) {
        float error = 0.0f;
        float trueCount = 0;
        int max_score = 0;
        int max_index = 0;
        for (int n = 0; n < batchSize; n++) {
            boolean allRight = true;
            int score = time;
            for (int t = 0; t < time; t++) {
                int predictIndex = MatrixOperation.maxIndex(output.getByNumber(n * time + t));
                int labelIndex = (int) labelData.data[n * time + t];
                if (labelIndex != igonre && labelIndex != predictIndex) {
                    allRight = false;
                    score--;
                }
            }
            if (max_score <= score) {
                max_score = score;
                max_index = n;
            }
            if (allRight) {
                trueCount++;
            }
        }
        String max_itxt = "";
        String max_ptxt = "";
        String max_ltxt = "";
        for (int t = 0; t < time; t++) {
            int predictIndex = MatrixOperation.maxIndex(output.getByNumber(max_index * time + t));
            int labelIndex = (int) labelData.data[max_index * time + t];
            int inputIndex = (int) input.data[max_index * time + t];
            max_ptxt += vocab[predictIndex];
            max_ltxt += vocab[labelIndex];
            max_itxt += vocab[inputIndex];
        }
        System.out.println("max_score:" + max_score);
        System.out.println("itxt:" + max_itxt);
        System.out.println("ptxt:" + max_ptxt);
        System.out.println("ltxt:" + max_ltxt);
        error = trueCount / batchSize * 100;
        return error;
    }

    public float testLoss(Tensor output, Tensor labelData) {
        float[] data = new float[output.number];
        float loss = 0.0f;
        for (int n = 0; n < output.number; n++) {
            float onceLoss = testLoss(output.getByNumber(n), labelData.getByNumber(n));
            loss += onceLoss;
            data[n] = onceLoss;
        }
        System.out.println("cpu_loss:" + JsonUtils.toJson(data));
        return loss;
    }

    public int accuracyTrueCount(Tensor output, Tensor labelData, String[] labelSet) {
        int trueCount = 0;
        for (int n = 0; n < output.number; n++) {
            String label = LabelUtils.vectorTolabel(labelData.getByNumber(n), labelSet);
            String predictLabel = LabelUtils.vectorTolabel(output.getByNumber(n), labelSet);
            //			System.out.println(label+":"+predictLabel);
            if (label.equals(predictLabel)) {
                trueCount++;
            }
        }
        return trueCount;
    }

    public float testLoss(float[] output, float[] label) {
        //		System.out.println(JsonUtils.toJson(label));
        float sum = 0.0f;
        float loss = 0.0f;
        /**
         * max
         *
         */
        float max = MatrixOperation.max(output);
        /**
         * sum
         *
         */
        for (int i = 0; i < output.length; i++) {
            sum += Math.exp(output[i] - max);
        }
        /**
         * softmax + log + nlloss
         *
         */
        for (int i = 0; i < output.length; i++) {
            loss += (float) (-((output[i] - max) - Math.log(sum)) * label[i]);
        }
        return loss;
    }

    public boolean isWarmUp() {
        return warmUp;
    }

    public void setWarmUp(boolean warmUp) {
        this.warmUp = warmUp;
    }

    public void online(boolean isOnline) {
        this.isOnline = isOnline;
    }

    public String getSid() {
        return sid;
    }

    public void setSid(String sid) {
        this.sid = sid;
    }
}
