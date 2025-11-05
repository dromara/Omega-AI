package com.omega.engine.optimizer;

import com.omega.common.utils.MathUtils;
import com.omega.common.utils.MatrixOperation;
import com.omega.common.utils.MatrixUtils;
import com.omega.common.utils.RandomUtils;
import com.omega.engine.nn.data.BaseData;
import com.omega.engine.nn.grad.GradClipping;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.gpu.RoPEKernel;
import com.omega.engine.nn.network.*;
import com.omega.engine.optimizer.lr.LearnRateUpdate;
import com.omega.engine.tensor.Tensor;
import com.omega.example.asr.dataset.AudioDataset;
import com.omega.example.rnn.data.IndexDataLoader;
import com.omega.example.transformer.dataset.LVMPreTrainDataset;
import com.omega.example.transformer.dataset.PreTrainDataset;
import com.omega.example.transformer.dataset.PreTrainDataset2;
import com.omega.example.transformer.dataset.SFTDataset;
import com.omega.example.transformer.utils.*;
import com.omega.example.transformer.utils.bpe.CNBpeTokenizer;
import jcuda.driver.JCudaDriver;

import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class EDOptimizer extends Optimizer {
    // private float clamp_val = -100;
    public EDOptimizer(Network network, int batchSize, int trainTime, float error, LearnRateUpdate learnRateUpdate, boolean warmUp) throws Exception {
        super(network, batchSize, trainTime, error, warmUp);
        this.lr = network.learnRate;
        this.batchSize = batchSize;
        this.trainTime = trainTime;
        this.learnRateUpdate = learnRateUpdate;
        this.loss = new Tensor(batchSize, this.network.oChannel, this.network.oHeight, this.network.oWidth);
        this.lossDiff = new Tensor(batchSize, this.network.oChannel, this.network.oHeight, this.network.oWidth);
    }

    public EDOptimizer(Network network, int batchSize, int trainTime, float error, boolean warmUp) throws Exception {
        super(network, batchSize, trainTime, error, warmUp);
        // TODO Auto-generated constructor stub
    }

    public static String[] output2TXT(Tensor output, IndexDataLoader trainData, int batchSize) {
        String[] txts = new String[batchSize];
        for (int b = 0; b < batchSize; b++) {
            String txt = "";
            for (int t = 0; t < trainData.max_ch - 1; t++) {
                int charIndex = pickTopN(output.getByNumber(t * batchSize + b), 1);
                String c = trainData.ch_dic[charIndex];
                txt += c;
            }
            txts[b] = txt;
        }
        return txts;
    }

    public static String[] output2TXT(Tensor output, IndexDataLoader trainData, int time, int batchSize) {
        String[] txts = new String[batchSize];
        for (int b = 0; b < batchSize; b++) {
            String txt = "";
            for (int t = 0; t < time; t++) {
                int charIndex = pickTopN(output.getByNumber(t * batchSize + b), 1);
                String c = trainData.ch_dic[charIndex];
                txt += c;
            }
            txts[b] = txt;
        }
        return txts;
    }

    public static String[] input2TXT(Tensor input, IndexDataLoader trainData, int batchSize) {
        String[] txts = new String[batchSize];
        for (int b = 0; b < batchSize; b++) {
            String txt = "";
            for (int t = 0; t < trainData.max_en; t++) {
                int charIndex = pickTopN(input.getByNumber(t * batchSize + b), 1);
                String c = trainData.en_dic[charIndex];
                txt += c;
            }
            txts[b] = txt;
        }
        return txts;
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

    @Override
    public void train(BaseData trainingData) {
        // TODO Auto-generated method stub
    }

    public void trainSeq2SeqRNN(IndexDataLoader trainingData) {
        // TODO Auto-generated method stub
        try {
            // CUDAModules.initCUDAFunctions();
            this.dataSize = trainingData.number;
            if (isWarmUp()) {
                this.network.learnRate = (float) (this.lr * Math.pow(batchIndex * 1.0f / burnIn * 1.0f, power));
            }
            Seq2SeqRNN network = (Seq2SeqRNN) this.network;
            Tensor inputEncoder = new Tensor(network.en_time * batchSize, 1, 1, network.en_len, true);
            Tensor inputDecoder = new Tensor(network.de_time * batchSize, 1, 1, network.de_len, true);
            Tensor label = new Tensor(network.de_time * batchSize, 1, 1, network.de_len, true);
            for (int i = 0; i < this.trainTime; i++) {
                if (this.trainIndex >= this.minTrainTime) {
                    break;
                }
                this.trainIndex = i + 1;
                int[][] indexs = MathUtils.randomInts(trainingData.number, this.batchSize);
                Tensor output = null;
                /**
                 * 遍历整个训练集
                 *
                 */
                for (int it = 0; it < indexs.length; it++) {
                    if (Math.abs(this.currentError) <= this.error) {
                        break;
                    }
                    long start = System.nanoTime();
                    this.loss.clear();
                    this.lossDiff.clear();
                    /**
                     * 读取训练数据
                     *
                     */
                    trainingData.loadData(indexs[it], inputEncoder, inputDecoder, label);
                    /**
                     * forward
                     *
                     */
                    output = network.forward(inputEncoder, inputDecoder);
                    /**
                     * loss
                     *
                     */
                    this.loss = network.loss(output, label);
                    /**
                     * loss diff
                     *
                     */
                    this.lossDiff = network.lossDiff(output, label);
                    // System.out.println(JsonUtils.toJson(output.syncHost()));
                    // GradClipping.gradClipping(this.lossDiff, 1e-7f);
                    /**
                     * back
                     *
                     */
                    this.network.back(this.lossDiff);
                    // /**
                    // * grad clipping
                    // */
                    // this.gradClipping(this.network);
                    /**
                     * update
                     *
                     */
                    this.network.update();
                    JCudaDriver.cuCtxSynchronize();
                    /**
                     * current time error
                     *
                     */
                    if (this.loss.isHasGPU()) {
                        this.currentError = MatrixOperation.sum(this.loss.syncHost()) / inputDecoder.number;
                    } else {
                        this.currentError = MatrixOperation.sum(this.loss.data) / inputDecoder.number;
                    }
                    // train_loss += this.currentError;
                    output.syncHost();
                    // System.out.println(JsonUtils.toJson(inputEncoder.shape()));
                    // System.out.println(JsonUtils.toJson(output.shape()));
                    // System.out.println(JsonUtils.toJson(label.shape()));
                    int time = output.number / batchSize;
                    float error = this.accuracy(output, label, time, batchSize);
                    // if(error > 99) {
                    // break;
                    // }
                    String msg = "training[" + this.trainIndex + "]{" + it + "} (lr:" + this.network.learnRate + ") accuracy:{" + error + "%} train_loss:" + this.currentError + " [costTime:" + (System.nanoTime() - start) / 1e6 + "ms.]";
                    System.out.println(msg);
                    this.batchIndex++;
                }
                // output.showDMByNumber(0);
                // showOutputAndLabel(trainingData, inputEncoder, output, label,
                // this.batchSize);
                /**
                 * update learning rate
                 *
                 */
                this.updateLR(this.lr_step);
            }
            /**
             * 停止训练
             *
             */
            System.out.println("training finish. [" + this.trainIndex + "] finalError:" + this.currentError);
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void trainSeq2Seq(IndexDataLoader trainingData) {
        // TODO Auto-generated method stub
        try {
            // CUDAModules.initCUDAFunctions();
            this.dataSize = trainingData.number;
            if (isWarmUp()) {
                this.network.learnRate = (float) (this.lr * Math.pow(batchIndex * 1.0f / burnIn * 1.0f, power));
            }
            Seq2Seq network = (Seq2Seq) this.network;
            Tensor inputEncoder = new Tensor(network.en_time * batchSize, 1, 1, network.en_len, true);
            Tensor inputDecoder = new Tensor(network.de_time * batchSize, 1, 1, network.de_len, true);
            Tensor label = new Tensor(network.de_time * batchSize, 1, 1, network.de_len, true);
            for (int i = 0; i < this.trainTime; i++) {
                if (this.trainIndex >= this.minTrainTime) {
                    break;
                }
                this.trainIndex = i + 1;
                int[][] indexs = MathUtils.randomInts(trainingData.number, this.batchSize);
                Tensor output = null;
                /**
                 * 遍历整个训练集
                 *
                 */
                for (int it = 0; it < indexs.length; it++) {
                    if (Math.abs(this.currentError) <= this.error) {
                        break;
                    }
                    long start = System.nanoTime();
                    this.loss.clear();
                    this.lossDiff.clear();
                    /**
                     * 读取训练数据
                     *
                     */
                    trainingData.loadData(indexs[it], inputEncoder, inputDecoder, label);
                    /**
                     * forward
                     *
                     */
                    output = network.forward(inputEncoder, inputDecoder);
                    /**
                     * loss
                     *
                     */
                    this.loss = network.loss(output, label);
                    /**
                     * loss diff
                     *
                     */
                    this.lossDiff = network.lossDiff(output, label);
                    // System.out.println(JsonUtils.toJson(output.syncHost()));
                    // GradClipping.gradClipping(this.lossDiff, 1e-7f);
                    /**
                     * back
                     *
                     */
                    this.network.back(this.lossDiff);
                    // /**
                    // * grad clipping
                    // */
                    // this.gradClipping(this.network);
                    /**
                     * update
                     *
                     */
                    this.network.update();
                    JCudaDriver.cuCtxSynchronize();
                    /**
                     * current time error
                     *
                     */
                    if (this.loss.isHasGPU()) {
                        this.currentError = MatrixOperation.sum(this.loss.syncHost()) / inputDecoder.number;
                    } else {
                        this.currentError = MatrixOperation.sum(this.loss.data) / inputDecoder.number;
                    }
                    // train_loss += this.currentError;
                    output.syncHost();
                    // System.out.println(JsonUtils.toJson(inputEncoder.shape()));
                    // System.out.println(JsonUtils.toJson(output.shape()));
                    // System.out.println(JsonUtils.toJson(label.shape()));
                    int time = output.number / batchSize;
                    float error = this.accuracy(output, label, time, batchSize);
                    // if(error > 99) {
                    // break;
                    // }
                    String msg = "training[" + this.trainIndex + "]{" + it + "} (lr:" + this.network.learnRate + ") accuracy:{" + error + "%} train_loss:" + this.currentError + " [costTime:" + (System.nanoTime() - start) / 1e6 + "ms.]";
                    System.out.println(msg);
                    this.batchIndex++;
                }
                // output.showDMByNumber(0);
                // showOutputAndLabel(trainingData, inputEncoder, output, label,
                // this.batchSize);
                /**
                 * update learning rate
                 *
                 */
                this.updateLR(this.lr_step);
            }
            /**
             * 停止训练
             *
             */
            System.out.println("training finish. [" + this.trainIndex + "] finalError:" + this.currentError);
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void trainGPT(ENTokenizer trainingData) {
        // TODO Auto-generated method stub
        try {
            // CUDAModules.initCUDAFunctions();
            this.dataSize = trainingData.number;
            if (isWarmUp()) {
                this.network.learnRate = (float) (this.lr * Math.pow(batchIndex * 1.0f / burnIn * 1.0f, power));
            }
            GPT network = (GPT) this.network;
            Tensor input = new Tensor(batchSize * network.time, 1, 1, network.vocab_size, true);
            Tensor label = new Tensor(batchSize * network.time, 1, 1, network.vocab_size, true);
            Tensor mask = ENTokenizer.triu(batchSize, network.head_num, network.time, network.time, 1);
            Tensor positions = ENTokenizer.getPositions(batchSize, network.time);
            for (int i = 0; i < this.trainTime; i++) {
                if (this.trainIndex >= this.minTrainTime) {
                    break;
                }
                this.trainIndex = i + 1;
                int[][] indexs = MathUtils.randomInts(trainingData.number, this.batchSize);
                Tensor output = null;
                /**
                 * 遍历整个训练集
                 *
                 */
                for (int it = 0; it < indexs.length; it++) {
                    if (Math.abs(this.currentError) <= this.error) {
                        break;
                    }
                    long start = System.nanoTime();
                    this.loss.clear();
                    this.lossDiff.clear();
                    /**
                     * 读取训练数据
                     *
                     */
                    trainingData.loadData(indexs[it], input, label);
                    /**
                     * forward
                     *
                     */
                    output = network.forward(input, positions, mask);
                    // output.showDMByNumber(0);
                    // label.showDMByNumber(0);
                    /**
                     * loss
                     *
                     */
                    this.loss = network.loss(output, label, trainingData.dictionary.get("<pad>"));
                    // this.loss = network.loss(output, label);
                    /**
                     * loss diff
                     *
                     */
                    this.lossDiff = network.lossDiff(output, label, trainingData.dictionary.get("<pad>"));
                    // this.lossDiff = network.lossDiff(output, label);
                    // System.out.println(JsonUtils.toJson(output.syncHost()));
                    // GradClipping.gradClipping(this.lossDiff, 1e-7f);
                    /**
                     * back
                     *
                     */
                    this.network.back(this.lossDiff);
                    // /**
                    // * grad clipping
                    // */
                    // this.gradClipping(this.network);
                    /**
                     * update
                     *
                     */
                    this.network.update();
                    JCudaDriver.cuCtxSynchronize();
                    /**
                     * current time error
                     *
                     */
                    if (this.loss.isHasGPU()) {
                        this.currentError = MatrixOperation.sum(this.loss.syncHost()) / input.number;
                    } else {
                        this.currentError = MatrixOperation.sum(this.loss.data) / input.number;
                    }
                    // train_loss += this.currentError;
                    output.syncHost();
                    // System.out.println(JsonUtils.toJson(inputEncoder.shape()));
                    // System.out.println(JsonUtils.toJson(output.shape()));
                    // System.out.println(JsonUtils.toJson(label.shape()));
                    int time = output.number / batchSize;
                    float error = this.accuracyBatchFisrt(input, output, label, time, batchSize, trainingData.vocab, trainingData.dictionary.get("<pad>"));
                    // if(error > 99) {
                    // break;
                    // }
                    String msg = "training[" + this.trainIndex + "]{" + it + "} (lr:" + this.network.learnRate + ") accuracy:{" + error + "%} train_loss:" + this.currentError + " [costTime:" + (System.nanoTime() - start) / 1e6 + "ms.]";
                    System.out.println(msg);
                    this.batchIndex++;
                }
                // output.showDMByNumber(0);
                // showOutputAndLabel(trainingData, inputEncoder, output, label,
                // this.batchSize);
                /**
                 * update learning rate
                 *
                 */
                this.updateLR(this.lr_step);
            }
            /**
             * 停止训练
             *
             */
            System.out.println("training finish. [" + this.trainIndex + "] finalError:" + this.currentError);
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void trainGPT2(ENTokenizer trainingData) {
        // TODO Auto-generated method stub
        try {
            // CUDAModules.initCUDAFunctions();
            this.dataSize = trainingData.number;
            if (isWarmUp()) {
                this.network.learnRate = (float) (this.lr * Math.pow(batchIndex * 1.0f / burnIn * 1.0f, power));
            }
            GPT2 network = (GPT2) this.network;
            Tensor input = new Tensor(batchSize * network.time, 1, 1, network.vocabSize, true);
            Tensor label = new Tensor(batchSize * network.time, 1, 1, network.vocabSize, true);
            Tensor mask = ENTokenizer.triu(batchSize, network.headNum, network.time, network.time, 1);
            Tensor positions = ENTokenizer.getPositions(batchSize, network.time);
            for (int i = 0; i < this.trainTime; i++) {
                if (this.trainIndex >= this.minTrainTime) {
                    break;
                }
                this.trainIndex = i + 1;
                int[][] indexs = MathUtils.randomInts(trainingData.number, this.batchSize);
                Tensor output = null;
                /**
                 * 遍历整个训练集
                 *
                 */
                for (int it = 0; it < indexs.length; it++) {
                    if (Math.abs(this.currentError) <= this.error) {
                        break;
                    }
                    long start = System.nanoTime();
                    this.loss.clear();
                    this.lossDiff.clear();
                    /**
                     * 读取训练数据
                     *
                     */
                    trainingData.loadData(indexs[it], input, label);
                    /**
                     * forward
                     *
                     */
                    output = network.forward(input, positions, mask);
                    // output.showDMByNumber(0);
                    // label.showDMByNumber(0);
                    /**
                     * loss
                     *
                     */
                    this.loss = network.loss(output, label, trainingData.dictionary.get("<pad>"));
                    // this.loss = network.loss(output, label);
                    /**
                     * loss diff
                     *
                     */
                    this.lossDiff = network.lossDiff(output, label, trainingData.dictionary.get("<pad>"));
                    // this.lossDiff = network.lossDiff(output, label);
                    // System.out.println(JsonUtils.toJson(output.syncHost()));
                    // GradClipping.gradClipping(this.lossDiff, 1e-7f);
                    /**
                     * back
                     *
                     */
                    this.network.back(this.lossDiff);
                    // /**
                    // * grad clipping
                    // */
                    // this.gradClipping(this.network);
                    this.network.clipGradNorm(1.0f);
                    /**
                     * update
                     *
                     */
                    this.network.update();
                    JCudaDriver.cuCtxSynchronize();
                    /**
                     * current time error
                     *
                     */
                    if (this.loss.isHasGPU()) {
                        this.currentError = MatrixOperation.sum(this.loss.syncHost()) / input.number;
                    } else {
                        this.currentError = MatrixOperation.sum(this.loss.data) / input.number;
                    }
                    // train_loss += this.currentError;
                    output.syncHost();
                    // System.out.println(JsonUtils.toJson(inputEncoder.shape()));
                    // System.out.println(JsonUtils.toJson(output.shape()));
                    // System.out.println(JsonUtils.toJson(label.shape()));
                    int time = output.number / batchSize;
                    float error = this.accuracyBatchFisrt(input, output, label, time, batchSize, trainingData.vocab, trainingData.dictionary.get("<pad>"));
                    // if(error > 99) {
                    // break;
                    // }
                    String msg = "training[" + this.trainIndex + "]{" + it + "} (lr:" + this.network.learnRate + ") accuracy:{" + error + "%} train_loss:" + this.currentError + " [costTime:" + (System.nanoTime() - start) / 1e6 + "ms.]";
                    System.out.println(msg);
                    this.batchIndex++;
                }
                // output.showDMByNumber(0);
                // showOutputAndLabel(trainingData, inputEncoder, output, label,
                // this.batchSize);
                /**
                 * update learning rate
                 *
                 */
                this.updateLR(this.lr_step);
            }
            /**
             * 停止训练
             *
             */
            System.out.println("training finish. [" + this.trainIndex + "] finalError:" + this.currentError);
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void trainGPT(CNChatTokenizer trainingData) {
        // TODO Auto-generated method stub
        try {
            // CUDAModules.initCUDAFunctions();
            this.dataSize = trainingData.number;
            if (isWarmUp()) {
                this.network.learnRate = (float) (this.lr * Math.pow(batchIndex * 1.0f / burnIn * 1.0f, power));
            }
            GPT network = (GPT) this.network;
            Tensor input = new Tensor(batchSize * network.time, 1, 1, network.vocab_size, true);
            Tensor label = new Tensor(batchSize * network.time, 1, 1, network.vocab_size, true);
            Tensor mask = ENTokenizer.triu(batchSize, network.head_num, network.time, network.time, 1);
            Tensor positions = ENTokenizer.getPositions(batchSize, network.time);
            for (int i = 0; i < this.trainTime; i++) {
                if (this.trainIndex >= this.minTrainTime) {
                    break;
                }
                this.trainIndex = i + 1;
                int[][] indexs = MathUtils.randomInts(trainingData.number, this.batchSize);
                Tensor output = null;
                /**
                 * 遍历整个训练集
                 *
                 */
                for (int it = 0; it < indexs.length; it++) {
                    if (Math.abs(this.currentError) <= this.error) {
                        break;
                    }
                    long start = System.nanoTime();
                    this.loss.clear();
                    this.lossDiff.clear();
                    /**
                     * 读取训练数据
                     *
                     */
                    trainingData.loadData(indexs[it], input, label);
                    /**
                     * forward
                     *
                     */
                    output = network.forward(input, positions, mask);
                    // output.showDMByNumber(0);
                    // label.showDMByNumber(0);
                    /**
                     * loss
                     *
                     */
                    this.loss = network.loss(output, label, trainingData.dictionary.get("<pad>"));
                    // this.loss = network.loss(output, label);
                    /**
                     * loss diff
                     *
                     */
                    this.lossDiff = network.lossDiff(output, label, trainingData.dictionary.get("<pad>"));
                    // this.lossDiff = network.lossDiff(output, label);
                    // lossDiff.showDMByNumber(0);
                    // System.out.println(JsonUtils.toJson(output.syncHost()));
                    // GradClipping.gradClipping(this.lossDiff, 1e-7f);
                    /**
                     * back
                     *
                     */
                    this.network.back(this.lossDiff);
                    // /**
                    // * grad clipping
                    // */
                    // this.gradClipping(this.network);
                    /**
                     * update
                     *
                     */
                    this.network.update();
                    JCudaDriver.cuCtxSynchronize();
                    /**
                     * current time error
                     *
                     */
                    if (this.loss.isHasGPU()) {
                        this.currentError = MatrixOperation.sum(this.loss.syncHost()) / input.number;
                    } else {
                        this.currentError = MatrixOperation.sum(this.loss.data) / input.number;
                    }
                    // train_loss += this.currentError;
                    output.syncHost();
                    // System.out.println(JsonUtils.toJson(inputEncoder.shape()));
                    // System.out.println(JsonUtils.toJson(output.shape()));
                    // System.out.println(JsonUtils.toJson(label.shape()));
                    int time = output.number / batchSize;
                    float error = this.accuracyBatchFisrt(input, output, label, time, batchSize, trainingData.vocab, trainingData.dictionary.get("<pad>"));
                    // if(error > 99) {
                    // break;
                    // }
                    String msg = "training[" + this.trainIndex + "]{" + it + "} (lr:" + this.network.learnRate + ") accuracy:{" + error + "%} train_loss:" + this.currentError + " [costTime:" + (System.nanoTime() - start) / 1e6 + "ms.]";
                    System.out.println(msg);
                    this.batchIndex++;
                }
                // output.showDMByNumber(0);
                // showOutputAndLabel(trainingData, inputEncoder, output, label,
                // this.batchSize);
                /**
                 * update learning rate
                 *
                 */
                this.updateLR(this.lr_step);
            }
            /**
             * 停止训练
             *
             */
            System.out.println("training finish. [" + this.trainIndex + "] finalError:" + this.currentError);
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void trainGPT2(CNChatTokenizer trainingData) {
        // TODO Auto-generated method stub
        try {
            // CUDAModules.initCUDAFunctions();
            this.dataSize = trainingData.number;
            if (isWarmUp()) {
                this.network.learnRate = (float) (this.lr * Math.pow(batchIndex * 1.0f / burnIn * 1.0f, power));
            }
            GPT2 network = (GPT2) this.network;
            Tensor input = new Tensor(batchSize * network.time, 1, 1, network.vocabSize, true);
            Tensor label = new Tensor(batchSize * network.time, 1, 1, network.vocabSize, true);
            Tensor mask = ENTokenizer.triu(batchSize, network.headNum, network.time, network.time, 1);
            Tensor positions = ENTokenizer.getPositions(batchSize, network.time);
            for (int i = 0; i < this.trainTime; i++) {
                if (this.trainIndex >= this.minTrainTime) {
                    break;
                }
                this.trainIndex = i + 1;
                int[][] indexs = MathUtils.randomInts(trainingData.number, this.batchSize);
                Tensor output = null;
                /**
                 * 遍历整个训练集
                 *
                 */
                for (int it = 0; it < indexs.length; it++) {
                    if (Math.abs(this.currentError) <= this.error) {
                        break;
                    }
                    long start = System.nanoTime();
                    this.loss.clear();
                    this.lossDiff.clear();
                    /**
                     * 读取训练数据
                     *
                     */
                    trainingData.loadData(indexs[it], input, label);
                    /**
                     * forward
                     *
                     */
                    output = network.forward(input, positions, mask);
                    // output.showDMByNumber(0);
                    // label.showDMByNumber(0);
                    /**
                     * loss
                     *
                     */
                    this.loss = network.loss(output, label, trainingData.dictionary.get("<pad>"));
                    // this.loss = network.loss(output, label);
                    /**
                     * loss diff
                     *
                     */
                    this.lossDiff = network.lossDiff(output, label, trainingData.dictionary.get("<pad>"));
                    // this.lossDiff = network.lossDiff(output, label);
                    // lossDiff.showDMByNumber(0);
                    // System.out.println(JsonUtils.toJson(output.syncHost()));
                    // GradClipping.gradClipping(this.lossDiff, 1e-7f);
                    /**
                     * back
                     *
                     */
                    this.network.back(this.lossDiff);
                    // /**
                    // * grad clipping
                    // */
                    // this.gradClipping(this.network);
                    /**
                     * update
                     *
                     */
                    this.network.update();
                    JCudaDriver.cuCtxSynchronize();
                    /**
                     * current time error
                     *
                     */
                    if (this.loss.isHasGPU()) {
                        this.currentError = MatrixOperation.sum(this.loss.syncHost()) / input.number;
                    } else {
                        this.currentError = MatrixOperation.sum(this.loss.data) / input.number;
                    }
                    // train_loss += this.currentError;
                    output.syncHost();
                    // System.out.println(JsonUtils.toJson(inputEncoder.shape()));
                    // System.out.println(JsonUtils.toJson(output.shape()));
                    // System.out.println(JsonUtils.toJson(label.shape()));
                    int time = output.number / batchSize;
                    float error = this.accuracyBatchFisrt(input, output, label, time, batchSize, trainingData.vocab, trainingData.dictionary.get("<pad>"));
                    // if(error > 99) {
                    // break;
                    // }
                    String msg = "training[" + this.trainIndex + "]{" + it + "} (lr:" + this.network.learnRate + ") accuracy:{" + error + "%} train_loss:" + this.currentError + " [costTime:" + (System.nanoTime() - start) / 1e6 + "ms.]";
                    System.out.println(msg);
                    this.batchIndex++;
                }
                // output.showDMByNumber(0);
                // showOutputAndLabel(trainingData, inputEncoder, output, label,
                // this.batchSize);
                /**
                 * update learning rate
                 *
                 */
                this.updateLR(this.lr_step);
            }
            /**
             * 停止训练
             *
             */
            System.out.println("training finish. [" + this.trainIndex + "] finalError:" + this.currentError);
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void trainNanoGPT(ENTokenizer trainingData) {
        // TODO Auto-generated method stub
        try {
            // CUDAModules.initCUDAFunctions();
            this.dataSize = trainingData.number;
            if (isWarmUp()) {
                this.network.learnRate = (float) (this.lr * Math.pow(batchIndex * 1.0f / burnIn * 1.0f, power));
            }
            NanoGPT network = (NanoGPT) this.network;
            Tensor input = new Tensor(batchSize * network.time, 1, 1, network.vocabSize, true);
            Tensor label = new Tensor(batchSize * network.time, 1, 1, network.vocabSize, true);
            // Tensor mask = ENTokenizer.triu(batchSize, network.headNum, network.time,
            // network.time, 1);
            Tensor positions = ENTokenizer.getPositions(batchSize, network.time);
            int itc = new BigDecimal(trainingData.number).divide(new BigDecimal(batchSize), 0, BigDecimal.ROUND_UP).intValue();
            int[][] tmp = new int[itc][batchSize];
            List<Integer> list = new ArrayList<Integer>();
            for (int i = 0; i < this.trainTime; i++) {
                if (this.trainIndex >= this.minTrainTime) {
                    break;
                }
                this.trainIndex = i + 1;
                int[][] indexs = MathUtils.randomInts(trainingData.number, this.batchSize, tmp, list);
                Tensor output = null;
                /**
                 * 遍历整个训练集
                 *
                 */
                for (int it = 0; it < indexs.length; it++) {
                    if (Math.abs(this.currentError) <= this.error) {
                        break;
                    }
                    long start = System.nanoTime();
                    // this.loss.clear();
                    //
                    // this.lossDiff.clear();
                    /**
                     * 读取训练数据
                     *
                     */
                    trainingData.loadData(indexs[it], input, label);
                    /**
                     * forward
                     *
                     */
                    output = network.forward(input, positions);
                    // output.showDMByNumber(0);
                    // label.showDMByNumber(0);
                    /**
                     * loss
                     *
                     */
                    this.loss = network.loss(output, label, trainingData.dictionary.get("<pad>"));
                    // this.loss = network.loss(output, label);
                    /**
                     * loss diff
                     *
                     */
                    this.lossDiff = network.lossDiff(output, label, trainingData.dictionary.get("<pad>"));
                    // this.lossDiff = network.lossDiff(output, label);
                    // System.out.println(JsonUtils.toJson(output.syncHost()));
                    // GradClipping.gradClipping(this.lossDiff, 1e-7f);
                    /**
                     * back
                     *
                     */
                    this.network.back(this.lossDiff);
                    // /**
                    // * grad clipping
                    // */
                    // this.gradClipping(this.network);
                    // this.network.clipGradNorm(1.0f);
                    /**
                     * update
                     *
                     */
                    this.network.update();
                    // JCudaDriver.cuCtxSynchronize();
                    /**
                     * current time error
                     *
                     */
                    if (this.loss.isHasGPU()) {
                        this.currentError = MatrixOperation.sum(this.loss.syncHost()) / input.number;
                    } else {
                        this.currentError = MatrixOperation.sum(this.loss.data) / input.number;
                    }
                    // train_loss += this.currentError;
                    output.syncHost();
                    // System.out.println(JsonUtils.toJson(inputEncoder.shape()));
                    // System.out.println(JsonUtils.toJson(output.shape()));
                    // System.out.println(JsonUtils.toJson(label.shape()));
                    int time = output.number / batchSize;
                    float error = this.accuracyBatchFisrt(input, output, label, time, batchSize, trainingData.vocab, trainingData.dictionary.get("<pad>"));
                    String msg = "training[" + this.trainIndex + "]{" + it + "} (lr:" + this.network.learnRate + ") accuracy:{" + error + "%} train_loss:" + this.currentError + " [costTime:" + (System.nanoTime() - start) / 1e6 + "ms.]";
                    System.out.println(msg);
                    this.batchIndex++;
                }
                // output.showDMByNumber(0);
                // showOutputAndLabel(trainingData, inputEncoder, output, label,
                // this.batchSize);
                /**
                 * update learning rate
                 *
                 */
                this.updateLR(this.lr_step);
            }
            /**
             * 停止训练
             *
             */
            System.out.println("training finish. [" + this.trainIndex + "] finalError:" + this.currentError);
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void trainNanoGPT(CNChatTokenizer trainingData) {
        // TODO Auto-generated method stub
        try {
            // CUDAModules.initCUDAFunctions();
            this.dataSize = trainingData.number;
            if (isWarmUp()) {
                this.network.learnRate = (float) (this.lr * Math.pow(batchIndex * 1.0f / burnIn * 1.0f, power));
            }
            NanoGPT network = (NanoGPT) this.network;
            Tensor input = new Tensor(batchSize * network.time, 1, 1, 1, true);
            Tensor label = new Tensor(batchSize * network.time, 1, 1, 1, true);
            // Tensor mask = CNChatTokenizer.triu(batchSize, network.headNum, network.time,
            // network.time, 1);
            Tensor positions = CNChatTokenizer.getPositions(batchSize, network.time);
            for (int i = 0; i < this.trainTime; i++) {
                if (this.trainIndex >= this.minTrainTime) {
                    break;
                }
                this.trainIndex = i + 1;
                int[][] indexs = MathUtils.randomInts(trainingData.trainData.size(), this.batchSize);
                Tensor output = null;
                /**
                 * 遍历整个训练集
                 *
                 */
                for (int it = 0; it < indexs.length; it++) {
                    if (Math.abs(this.currentError) <= this.error) {
                        break;
                    }
                    long start = System.nanoTime();
                    this.loss.clear();
                    this.lossDiff.clear();
                    /**
                     * 读取训练数据
                     *
                     */
                    trainingData.loadTrainIdx(indexs[it], input, label);
                    /**
                     * forward
                     *
                     */
                    output = network.forward(input, positions);
                    /**
                     * loss
                     *
                     */
                    this.loss = network.loss(output, label, trainingData.dictionary.get("<pad>"));
                    /**
                     * loss diff
                     *
                     */
                    this.lossDiff = network.lossDiff(output, label, trainingData.dictionary.get("<pad>"));
                    /**
                     * back
                     *
                     */
                    this.network.back(this.lossDiff);
                    /**
                     * update
                     *
                     */
                    this.network.update();
                    /**
                     * current time error
                     *
                     */
                    if (this.loss.isHasGPU()) {
                        this.currentError = MatrixOperation.sum(this.loss.syncHost()) / input.number;
                    } else {
                        this.currentError = MatrixOperation.sum(this.loss.data) / input.number;
                    }
                    // train_loss += this.currentError;
                    output.syncHost();
                    // System.out.println(JsonUtils.toJson(inputEncoder.shape()));
                    // System.out.println(JsonUtils.toJson(output.shape()));
                    // System.out.println(JsonUtils.toJson(label.shape()));
                    int time = output.number / batchSize;
                    if (it % 20 == 0) {
                        float error = this.accuracyIdx(input, output, label, time, batchSize, trainingData.vocab, trainingData.dictionary.get("<pad>"));
                    }
                    // if(error > 99) {
                    // break;
                    // }
                    String msg = "training[" + this.trainIndex + "]{" + it + "} (lr:" + this.network.learnRate + ") train_loss:" + this.currentError + " [costTime:" + (System.nanoTime() - start) / 1e6 + "ms.]";
                    System.out.println(msg);
                    this.batchIndex++;
                    // if(it != 0 && it % 200 == 0) {
                    // vail_chat(network, input, output, label, positions, trainingData);
                    // network.RUN_MODEL = RunModel.TRAIN;
                    // }
                    // if(it != 0 && it % 200 == 0) {
                    // break;
                    // }
                }
                // showOutputAndLabel(trainingData, inputEncoder, output, label,
                // this.batchSize);
                // output.showDMByNumber(1);
                //
                // OPKernel.getInstance().copy_number_gpu(input, input1, 0, 0);
                //
                // output1 = network.forward(input1, positions1, mask1);
                // output1.showDMByNumber(1);
                //
                // System.out.println("==============>");
                // GPTTest.output2TXT(input1, trainingData, true);
                // GPTTest.output2TXT(output1, trainingData, true);
                // input1.showDMByNumber(0);
                /**
                 * update learning rate
                 *
                 */
                this.updateLR(this.lr_step);
            }
            /**
             * 停止训练
             *
             */
            System.out.println("training finish. [" + this.trainIndex + "] finalError:" + this.currentError);
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void trainNanoGPT(CNChatTokenizer2 trainingData) {
        // TODO Auto-generated method stub
        try {
            // CUDAModules.initCUDAFunctions();
            this.dataSize = trainingData.number;
            if (isWarmUp()) {
                this.network.learnRate = (float) (this.lr * Math.pow(batchIndex * 1.0f / burnIn * 1.0f, power));
            }
            NanoGPT network = (NanoGPT) this.network;
            Tensor input = new Tensor(batchSize * network.time, 1, 1, 1, true);
            Tensor label = new Tensor(batchSize * network.time, 1, 1, network.vocabSize, true);
            // Tensor mask = CNChatTokenizer.triu(batchSize, network.headNum, network.time,
            // network.time, 1);
            Tensor positions = CNChatTokenizer.getPositions(batchSize, network.time);
            for (int i = 0; i < this.trainTime; i++) {
                if (this.trainIndex >= this.minTrainTime) {
                    break;
                }
                this.trainIndex = i + 1;
                int[][] indexs = MathUtils.randomInts(trainingData.trainData.size(), this.batchSize);
                Tensor output = null;
                /**
                 * 遍历整个训练集
                 *
                 */
                for (int it = 0; it < indexs.length; it++) {
                    if (Math.abs(this.currentError) <= this.error) {
                        break;
                    }
                    long start = System.nanoTime();
                    this.loss.clear();
                    this.lossDiff.clear();
                    /**
                     * 读取训练数据
                     *
                     */
                    trainingData.loadTrainData(indexs[it], input, label);
                    /**
                     * forward
                     *
                     */
                    output = network.forward(input, positions);
                    /**
                     * loss
                     *
                     */
                    this.loss = network.loss(output, label, trainingData.tokenizer.specials.get("<pad>"));
                    /**
                     * loss diff
                     *
                     */
                    this.lossDiff = network.lossDiff(output, label, trainingData.tokenizer.specials.get("<pad>"));
                    /**
                     * back
                     *
                     */
                    this.network.back(this.lossDiff);
                    // /**
                    // * grad clipping
                    // */
                    // this.network.clipGradNorm(1.0f);
                    /**
                     * update
                     *
                     */
                    this.network.update();
                    /**
                     * current time error
                     *
                     */
                    if (this.loss.isHasGPU()) {
                        this.currentError = MatrixOperation.sum(this.loss.syncHost()) / input.number;
                    } else {
                        this.currentError = MatrixOperation.sum(this.loss.data) / input.number;
                    }
                    output.syncHost();
                    int time = output.number / batchSize;
                    float error = this.accuracyBatchFisrt(input, output, label, time, batchSize, trainingData.tokenizer, trainingData.tokenizer.specials.get("<pad>"));
                    String msg = "training[" + this.trainIndex + "]{" + it + "} (lr:" + this.network.learnRate + ") accuracy:{" + error + "%} train_loss:" + this.currentError + " [costTime:" + (System.nanoTime() - start) / 1e6 + "ms.]";
                    System.out.println(msg);
                    this.batchIndex++;
                }
                /**
                 * update learning rate
                 *
                 */
                this.updateLR(this.lr_step);
            }
            /**
             * 停止训练
             *
             */
            System.out.println("training finish. [" + this.trainIndex + "] finalError:" + this.currentError);
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void trainNanoGPT_GEN(CNTokenizer trainingData) {
        // TODO Auto-generated method stub
        try {
            // CUDAModules.initCUDAFunctions();
            this.dataSize = trainingData.number;
            if (isWarmUp()) {
                this.network.learnRate = (float) (this.lr * Math.pow(batchIndex * 1.0f / burnIn * 1.0f, power));
            }
            NanoGPT network = (NanoGPT) this.network;
            Tensor input = new Tensor(batchSize * network.time, 1, 1, 1, true);
            Tensor label = new Tensor(batchSize * network.time, 1, 1, network.vocabSize, true);
            Tensor positions = CNChatTokenizer.getPositions(batchSize, network.time);
            for (int i = 0; i < this.trainTime; i++) {
                if (this.trainIndex >= this.minTrainTime) {
                    break;
                }
                this.trainIndex = i + 1;
                int[][] indexs = MathUtils.randomInts(trainingData.trainData.length - network.time, this.batchSize);
                Tensor output = null;
                /**
                 * 遍历整个训练集
                 *
                 */
                for (int it = 0; it < indexs.length; it++) {
                    if (Math.abs(this.currentError) <= this.error) {
                        break;
                    }
                    long start = System.nanoTime();
                    this.loss.clear();
                    this.lossDiff.clear();
                    /**
                     * 读取训练数据
                     *
                     */
                    trainingData.loadData(indexs[it], input, label);
                    /**
                     * forward
                     *
                     */
                    output = network.forward(input, positions);
                    /**
                     * loss
                     *
                     */
                    // this.loss = network.loss(output, label,
                    // trainingData.dictionary.get("<pad>"));
                    this.loss = network.loss(output, label);
                    /**
                     * loss diff
                     *
                     */
                    // this.lossDiff = network.lossDiff(output, label,
                    // trainingData.dictionary.get("<pad>"));
                    this.lossDiff = network.lossDiff(output, label);
                    // lossDiff.showDMByNumber(0);
                    // System.out.println(JsonUtils.toJson(output.syncHost()));
                    // GradClipping.gradClipping(this.lossDiff, 1e-7f);
                    /**
                     * back
                     *
                     */
                    this.network.back(this.lossDiff);
                    // /**
                    // * grad clipping
                    // */
                    // this.gradClipping(this.network);
                    // this.network.clipGradNorm(1.0f);
                    /**
                     * update
                     *
                     */
                    this.network.update();
                    /**
                     * current time error
                     *
                     */
                    if (this.loss.isHasGPU()) {
                        this.currentError = MatrixOperation.sum(this.loss.syncHost()) / input.number;
                    } else {
                        this.currentError = MatrixOperation.sum(this.loss.data) / input.number;
                    }
                    // train_loss += this.currentError;
                    output.syncHost();
                    // System.out.println(JsonUtils.toJson(inputEncoder.shape()));
                    // System.out.println(JsonUtils.toJson(output.shape()));
                    // System.out.println(JsonUtils.toJson(label.shape()));
                    int time = output.number / batchSize;
                    float error = this.accuracyBatchFisrt(input, output, label, time, batchSize, trainingData.vocab);
                    // if(error > 99) {
                    // break;
                    // }
                    String msg = "training[" + this.trainIndex + "]{" + it + "} (lr:" + this.network.learnRate + ") accuracy:{" + error + "%} train_loss:" + this.currentError + " [costTime:" + (System.nanoTime() - start) / 1e6 + "ms.]";
                    System.out.println(msg);
                    this.batchIndex++;
                    // if(it != 0 && it % 200 == 0) {
                    // vail_gen(network, input, output, label, positions, trainingData);
                    // network.RUN_MODEL = RunModel.TRAIN;
                    // }
                }
                // showOutputAndLabel(trainingData, inputEncoder, output, label,
                // this.batchSize);
                /**
                 * update learning rate
                 *
                 */
                this.updateLR(this.lr_step);
            }
            /**
             * 停止训练
             *
             */
            System.out.println("training finish. [" + this.trainIndex + "] finalError:" + this.currentError);
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void trainLlama2_GEN(CNTokenizer trainingData) {
        // TODO Auto-generated method stub
        try {
            // CUDAModules.initCUDAFunctions();
            this.dataSize = trainingData.number;
            if (isWarmUp()) {
                this.network.learnRate = (float) (this.lr * Math.pow(batchIndex * 1.0f / burnIn * 1.0f, power));
            }
            Llama2 network = (Llama2) this.network;
            Tensor input = new Tensor(batchSize * network.time, 1, 1, 1, true);
            Tensor label = new Tensor(batchSize * network.time, 1, 1, 1, true);
            Tensor[] cs = RoPEKernel.getCosAndSin(network.time, network.embedDim, network.headNum);
            Tensor cos = cs[0];
            Tensor sin = cs[1];
            for (int i = 0; i < this.trainTime; i++) {
                if (this.trainIndex >= this.minTrainTime) {
                    break;
                }
                this.trainIndex = i + 1;
                int[][] indexs = MathUtils.randomInts(trainingData.trainData.length - network.time, this.batchSize);
                Tensor output = null;
                /**
                 * 遍历整个训练集
                 *
                 */
                for (int it = 0; it < indexs.length; it++) {
                    if (Math.abs(this.currentError) <= this.error) {
                        break;
                    }
                    long start = System.nanoTime();
                    this.loss.clear();
                    this.lossDiff.clear();
                    /**
                     * 读取训练数据
                     *
                     */
                    trainingData.loadIDXData(indexs[it], input, label);
                    /**
                     * forward
                     *
                     */
                    output = network.forward(cos, sin, input);
                    /**
                     * loss
                     *
                     */
                    this.loss = network.loss(output, label);
                    /**
                     * loss diff
                     *
                     */
                    this.lossDiff = network.lossDiff(output, label);
                    /**
                     * back
                     *
                     */
                    network.back(cos, sin, this.lossDiff);
                    /**
                     * update
                     *
                     */
                    this.network.update();
                    /**
                     * current time error
                     *
                     */
                    if (this.loss.isHasGPU()) {
                        this.currentError = MatrixOperation.sum(this.loss.syncHost()) / input.number;
                    } else {
                        this.currentError = MatrixOperation.sum(this.loss.data) / input.number;
                    }
                    output.syncHost();
                    int time = output.number / batchSize;
                    float error = this.accuracyIdx(input, output, label, time, batchSize, trainingData.vocab);
                    String msg = "training[" + this.trainIndex + "]{" + it + "} (lr:" + this.network.learnRate + ") accuracy:{" + error + "%} train_loss:" + this.currentError + " [costTime:" + (System.nanoTime() - start) / 1e6 + "ms.]";
                    System.out.println(msg);
                    this.batchIndex++;
                }
                /**
                 * update learning rate
                 *
                 */
                this.updateLR(this.lr_step);
            }
            /**
             * 停止训练
             *
             */
            System.out.println("training finish. [" + this.trainIndex + "] finalError:" + this.currentError);
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void trainLlama2(CNChatTokenizer trainingData) {
        // TODO Auto-generated method stub
        try {
            // CUDAModules.initCUDAFunctions();
            this.dataSize = trainingData.number;
            if (isWarmUp()) {
                this.network.learnRate = (float) (this.lr * Math.pow(batchIndex * 1.0f / burnIn * 1.0f, power));
            }
            Llama2 network = (Llama2) this.network;
            Tensor input = new Tensor(batchSize * network.time, 1, 1, 1, true);
            Tensor label = new Tensor(batchSize * network.time, 1, 1, 1, true);
            Tensor[] cs = RoPEKernel.getCosAndSin(network.time, network.embedDim, network.headNum);
            Tensor cos = cs[0];
            Tensor sin = cs[1];
            for (int i = 0; i < this.trainTime; i++) {
                if (this.trainIndex >= this.minTrainTime) {
                    break;
                }
                this.trainIndex = i + 1;
                int[][] indexs = MathUtils.randomInts(trainingData.trainData.size(), this.batchSize);
                Tensor output = null;
                /**
                 * 遍历整个训练集
                 *
                 */
                for (int it = 0; it < indexs.length; it++) {
                    if (Math.abs(this.currentError) <= this.error) {
                        break;
                    }
                    long start = System.nanoTime();
                    this.loss.clear();
                    this.lossDiff.clear();
                    /**
                     * 读取训练数据
                     *
                     */
                    trainingData.loadTrainIdx(indexs[it], input, label);
                    /**
                     * forward
                     *
                     */
                    output = network.forward(cos, sin, input);
                    /**
                     * loss
                     *
                     */
                    this.loss = network.loss(output, label, -1);
                    /**
                     * loss diff
                     *
                     */
                    this.lossDiff = network.lossDiff(output, label, -1);
                    /**
                     * back
                     *
                     */
                    network.back(cos, sin, this.lossDiff);
                    /**
                     * update
                     *
                     */
                    this.network.update();
                    /**
                     * current time error
                     *
                     */
                    if (this.loss.isHasGPU()) {
                        this.currentError = MatrixOperation.sum(this.loss.syncHost()) / input.number;
                    } else {
                        this.currentError = MatrixOperation.sum(this.loss.data) / input.number;
                    }
                    output.syncHost();
                    int time = output.number / batchSize;
                    if (it % 20 == 0) {
                        float error = this.accuracyIdx(input, output, label, time, batchSize, trainingData.vocab, trainingData.dictionary.get("<pad>"));
                    }
                    String msg = "training[" + this.trainIndex + "]{" + it + "} (lr:" + this.network.learnRate + ") train_loss:" + this.currentError + " [costTime:" + (System.nanoTime() - start) / 1e6 + "ms.]";
                    System.out.println(msg);
                    this.batchIndex++;
                }
                /**
                 * update learning rate
                 *
                 */
                this.updateLR(this.lr_step);
            }
            /**
             * 停止训练
             *
             */
            System.out.println("training finish. [" + this.trainIndex + "] finalError:" + this.currentError);
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void trainLlama2_wiki(CNWikiTokenizer trainingData) {
        // TODO Auto-generated method stub
        try {
            // CUDAModules.initCUDAFunctions();
            this.dataSize = trainingData.number;
            if (isWarmUp()) {
                this.network.learnRate = (float) (this.lr * Math.pow(batchIndex * 1.0f / burnIn * 1.0f, power));
            }
            Llama2 network = (Llama2) this.network;
            Tensor input = new Tensor(batchSize * network.time, 1, 1, 1, true);
            Tensor label = new Tensor(batchSize * network.time, 1, 1, network.vocabSize, true);
            Tensor[] cs = RoPEKernel.getCosAndSin(network.time, network.embedDim, network.headNum);
            Tensor cos = cs[0];
            Tensor sin = cs[1];
            int pad = 0;
            if (trainingData.tokenizer != null) {
                pad = trainingData.tokenizer.unk;
            } else {
                pad = trainingData.dictionary.get("<pad>");
            }
            for (int i = 0; i < this.trainTime; i++) {
                if (this.trainIndex >= this.minTrainTime) {
                    break;
                }
                this.trainIndex = i + 1;
                // int[][] indexs =
                // MathUtils.randomInts(trainingData.trainData.size(),this.batchSize);
                int[][] indexs = MathUtils.sortInts(trainingData.trainData.size(), this.batchSize);
                Tensor output = null;
                /**
                 * 遍历整个训练集
                 *
                 */
                for (int it = 0; it < indexs.length; it++) {
                    if (Math.abs(this.currentError) <= this.error) {
                        break;
                    }
                    long start = System.nanoTime();
                    this.loss.clear();
                    this.lossDiff.clear();
                    /**
                     * 读取训练数据
                     *
                     */
                    trainingData.loadTrainData(indexs[it], input, label);
                    /**
                     * forward
                     *
                     */
                    output = network.forward(cos, sin, input);
                    /**
                     * loss
                     *
                     */
                    this.loss = network.loss(output, label, pad);
                    /**
                     * loss diff
                     *
                     */
                    this.lossDiff = network.lossDiff(output, label, pad);
                    /**
                     * back
                     *
                     */
                    network.back(cos, sin, this.lossDiff);
                    /**
                     * update
                     *
                     */
                    this.network.update();
                    /**
                     * current time error
                     *
                     */
                    if (this.loss.isHasGPU()) {
                        this.currentError = MatrixOperation.sum(this.loss.syncHost()) / input.number;
                    } else {
                        this.currentError = MatrixOperation.sum(this.loss.data) / input.number;
                    }
                    output.syncHost();
                    if (it % 20 == 0) {
                        int time = output.number / batchSize;
                        if (trainingData.tokenizer != null) {
                            float error = this.accuracyBatchFisrt(input, output, label, time, batchSize, trainingData.tokenizer, pad);
                        } else {
                            float error = this.accuracyBatchFisrt(input, output, label, time, batchSize, trainingData.vocab, pad);
                        }
                    }
                    String msg = "training[" + this.trainIndex + "]{" + it + "} (lr:" + this.network.learnRate + ") train_loss:" + this.currentError + " [costTime:" + (System.nanoTime() - start) / 1e6 + "ms.]";
                    System.out.println(msg);
                    this.batchIndex++;
                    // if(it != 0 && it % 200 == 0) {
                    // vail_chat(network, input, output, label, positions, trainingData);
                    // network.RUN_MODEL = RunModel.TRAIN;
                    // }
                    /**
                     * update learning rate
                     *
                     */
                    this.updateLR(this.lr_step, it);
                }
            }
            /**
             * 停止训练
             *
             */
            System.out.println("training finish. [" + this.trainIndex + "] finalError:" + this.currentError);
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void trainLlama2_wiki(CNWikiTokenizer2 trainingData) {
        // TODO Auto-generated method stub
        try {
            // CUDAModules.initCUDAFunctions();
            this.dataSize = trainingData.number;
            if (isWarmUp()) {
                this.network.learnRate = (float) (this.lr * Math.pow(batchIndex * 1.0f / burnIn * 1.0f, power));
            }
            Llama2 network = (Llama2) this.network;
            Tensor input = new Tensor(batchSize * network.time, 1, 1, 1, true);
            Tensor label = new Tensor(batchSize, 1, 1, network.time, true);
            Tensor[] cs = RoPEKernel.getCosAndSin(network.time, network.embedDim, network.headNum);
            Tensor cos = cs[0];
            Tensor sin = cs[1];
            int pad = -1;
            if (trainingData.tokenizer != null) {
                pad = trainingData.tokenizer.pad;
            } else {
                pad = trainingData.dictionary.get("<pad>");
            }
            for (int i = 0; i < this.trainTime; i++) {
                if (this.trainIndex >= this.minTrainTime) {
                    break;
                }
                this.trainIndex = i + 1;
                // int[][] indexs =
                // MathUtils.randomInts(trainingData.trainData.size(),this.batchSize);
                int[][] indexs = MathUtils.sortInts(trainingData.trainData.size(), this.batchSize);
                Tensor output = null;
                /**
                 * 遍历整个训练集
                 *
                 */
                for (int it = 0; it < indexs.length; it++) {
                    this.network.train_time = it + 1;
                    if (Math.abs(this.currentError) <= this.error) {
                        break;
                    }
                    long start = System.nanoTime();
                    this.loss.clear();
                    this.lossDiff.clear();
                    // JCuda.cudaDeviceSynchronize();
                    // long start22 = System.nanoTime();
                    /**
                     * 读取训练数据
                     *
                     */
                    trainingData.loadTrainData(indexs[it], input, label);
                    // JCuda.cudaDeviceSynchronize();
                    // System.out.println("loadTrainData:"+(System.nanoTime() - start22) /
                    // 1e6+"ms.");
                    /**
                     * forward
                     *
                     */
                    output = network.forward(cos, sin, input);
                    // JCuda.cudaDeviceSynchronize();
                    // long start33 = System.nanoTime();
                    /**
                     * loss
                     *
                     */
                    this.loss = network.loss(output, label, pad);
                    // JCuda.cudaDeviceSynchronize();
                    // System.out.println("loss:"+(System.nanoTime() - start33) / 1e6+"ms.");
                    // long start44 = System.nanoTime();
                    /**
                     * loss diff
                     *
                     */
                    this.lossDiff = network.lossDiff(output, label, pad);
                    // JCuda.cudaDeviceSynchronize();
                    // System.out.println("lossDiff:"+(System.nanoTime() - start44) / 1e6+"ms.");
                    /**
                     * back
                     *
                     */
                    network.back(cos, sin, this.lossDiff);
                    // JCuda.cudaDeviceSynchronize();
                    // long start55 = System.nanoTime();
                    /**
                     * update
                     *
                     */
                    this.network.update();
                    // JCuda.cudaDeviceSynchronize();
                    // System.out.println("update:"+(System.nanoTime() - start55) / 1e6+"ms.");
                    /**
                     * current time error
                     *
                     */
                    if (this.loss.isHasGPU()) {
                        this.currentError = MatrixOperation.sum(this.loss.syncHost()) / input.number;
                    } else {
                        this.currentError = MatrixOperation.sum(this.loss.data) / input.number;
                    }
                    output.syncHost();
                    if (it % 20 == 0) {
                        int time = output.number / batchSize;
                        if (trainingData.tokenizer != null) {
                            float error = this.accuracyBatchFisrt(input, output, label, time, batchSize, trainingData.tokenizer, pad);
                        } else {
                            float error = this.accuracyBatchFisrt(input, output, label, time, batchSize, trainingData.vocab, pad);
                        }
                    }
                    String msg = "training[" + this.trainIndex + "]{" + it + "} (lr:" + this.network.learnRate + ") train_loss:" + this.currentError + " [costTime:" + (System.nanoTime() - start) / 1e6 + "ms.]";
                    System.out.println(msg);
                    this.batchIndex++;
                    // if(it != 0 && it % 20 == 0) {
                    // break;
                    // }
                    /**
                     * update learning rate
                     *
                     */
                    this.updateLR(this.lr_step, it);
                }
            }
            /**
             * 停止训练
             *
             */
            System.out.println("training finish. [" + this.trainIndex + "] finalError:" + this.currentError);
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void trainLlama2_chinese(CNWikiTokenizer3 trainingData) {
        // TODO Auto-generated method stub
        try {
            // CUDAModules.initCUDAFunctions();
            this.dataSize = trainingData.number;
            if (isWarmUp()) {
                this.network.learnRate = (float) (this.lr * Math.pow(batchIndex * 1.0f / burnIn * 1.0f, power));
            }
            Llama2 network = (Llama2) this.network;
            Tensor input = new Tensor(batchSize * network.time, 1, 1, 1, true);
            float[] tmpInput = new float[batchSize * network.time];
            Tensor label = new Tensor(batchSize, 1, 1, network.time, true);
            float[] tmpLabel = new float[batchSize * network.time];
            Tensor[] cs = RoPEKernel.getCosAndSin(network.time, network.embedDim, network.headNum);
            Tensor cos = cs[0];
            Tensor sin = cs[1];
            int pad = trainingData.tokenizer.pad;
            trainingData.loadData(input, label, tmpInput, tmpLabel);
            for (int i = 0; i < this.trainTime; i++) {
                if (this.trainIndex >= this.minTrainTime) {
                    break;
                }
                this.trainIndex = i + 1;
                Tensor output = null;
                /**
                 * 遍历整个训练集
                 *
                 */
                // System.out.println(trainingData.count_it);
                for (int it = 0; it < trainingData.count_it; it++) {
                    this.network.train_time = it + 1;
                    if (Math.abs(this.currentError) <= this.error) {
                        break;
                    }
                    long start = System.nanoTime();
                    this.loss.clear();
                    this.lossDiff.clear();
                    // JCuda.cudaDeviceSynchronize();
                    // long start22 = System.nanoTime();
                    /**
                     * 读取训练数据
                     *
                     */
                    trainingData.loadData(input, label, tmpInput, tmpLabel);
                    // trainingData.loadTrainData(indexs[it], input, label);
                    // JCuda.cudaDeviceSynchronize();
                    // System.out.println("loadTrainData:"+(System.nanoTime() - start22) /
                    // 1e6+"ms.");
                    /**
                     * forward
                     *
                     */
                    output = network.forward(cos, sin, input);
                    // JCuda.cudaDeviceSynchronize();
                    // long start33 = System.nanoTime();
                    /**
                     * loss
                     *
                     */
                    this.loss = network.loss(output, label, pad);
                    // JCuda.cudaDeviceSynchronize();
                    // System.out.println("loss:"+(System.nanoTime() - start33) / 1e6+"ms.");
                    // long start44 = System.nanoTime();
                    /**
                     * loss diff
                     *
                     */
                    this.lossDiff = network.lossDiff(output, label, pad);
                    // JCuda.cudaDeviceSynchronize();
                    // System.out.println("lossDiff:"+(System.nanoTime() - start44) / 1e6+"ms.");
                    /**
                     * back
                     *
                     */
                    network.back(cos, sin, this.lossDiff);
                    // JCuda.cudaDeviceSynchronize();
                    // long start55 = System.nanoTime();
                    /**
                     * update
                     *
                     */
                    this.network.update();
                    // JCuda.cudaDeviceSynchronize();
                    // System.out.println("update:"+(System.nanoTime() - start55) / 1e6+"ms.");
                    /**
                     * current time error
                     *
                     */
                    if (this.loss.isHasGPU()) {
                        this.currentError = MatrixOperation.sum(this.loss.syncHost()) / input.number;
                    } else {
                        this.currentError = MatrixOperation.sum(this.loss.data) / input.number;
                    }
                    output.syncHost();
                    if (it % 20 == 0) {
                        int time = output.number / batchSize;
                        if (trainingData.tokenizer != null) {
                            float error = this.accuracyBatchFisrt(input, tmpInput, output, label, tmpLabel, time, batchSize, trainingData.tokenizer, pad);
                        } else {
                            float error = this.accuracyBatchFisrt(input, output, label, time, batchSize, trainingData.vocab, pad);
                        }
                    }
                    String msg = "training[" + this.trainIndex + "]{" + it + "} (lr:" + this.network.learnRate + ") train_loss:" + this.currentError + " [costTime:" + (System.nanoTime() - start) / 1e6 + "ms.]";
                    System.out.println(msg);
                    this.batchIndex++;
                    // if(it != 0 && it % 20 == 0) {
                    // break;
                    // }
                    /**
                     * update learning rate
                     *
                     */
                    this.updateLR(this.lr_step, it);
                }
            }
            /**
             * 停止训练
             *
             */
            System.out.println("training finish. [" + this.trainIndex + "] finalError:" + this.currentError);
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void trainLlama2_chinese(CNWikiTokenizer4 trainingData) {
        // TODO Auto-generated method stub
        try {
            // CUDAModules.initCUDAFunctions();
            this.dataSize = trainingData.number;
            if (isWarmUp()) {
                this.network.learnRate = (float) (this.lr * Math.pow(batchIndex * 1.0f / burnIn * 1.0f, power));
            }
            Llama2 network = (Llama2) this.network;
            Tensor input = new Tensor(batchSize * network.time, 1, 1, 1, true);
            float[] tmpInput = new float[batchSize * network.time];
            Tensor label = new Tensor(batchSize, 1, 1, network.time, true);
            float[] tmpLabel = new float[batchSize * network.time];
            Tensor[] cs = RoPEKernel.getCosAndSin(network.time, network.embedDim, network.headNum);
            Tensor cos = cs[0];
            Tensor sin = cs[1];
            int pad = -1;
            trainingData.loadData(input, label, tmpInput, tmpLabel);
            for (int i = 0; i < this.trainTime; i++) {
                if (this.trainIndex >= this.minTrainTime) {
                    break;
                }
                this.trainIndex = i + 1;
                Tensor output = null;
                /**
                 * 遍历整个训练集
                 *
                 */
                // System.out.println(trainingData.count_it);
                for (int it = 0; it < trainingData.count_it; it++) {
                    this.network.train_time = it + 1;
                    if (Math.abs(this.currentError) <= this.error) {
                        break;
                    }
                    long start = System.nanoTime();
                    this.loss.clear();
                    this.lossDiff.clear();
                    // JCuda.cudaDeviceSynchronize();
                    // long start22 = System.nanoTime();
                    /**
                     * 读取训练数据
                     *
                     */
                    trainingData.loadData(input, label, tmpInput, tmpLabel);
                    // trainingData.loadTrainData(indexs[it], input, label);
                    // JCuda.cudaDeviceSynchronize();
                    // System.out.println("loadTrainData:"+(System.nanoTime() - start22) /
                    // 1e6+"ms.");
                    /**
                     * forward
                     *
                     */
                    output = network.forward(cos, sin, input);
                    // JCuda.cudaDeviceSynchronize();
                    // long start33 = System.nanoTime();
                    /**
                     * loss
                     *
                     */
                    this.loss = network.loss(output, label, pad);
                    // JCuda.cudaDeviceSynchronize();
                    // System.out.println("loss:"+(System.nanoTime() - start33) / 1e6+"ms.");
                    // long start44 = System.nanoTime();
                    /**
                     * loss diff
                     *
                     */
                    this.lossDiff = network.lossDiff(output, label, pad);
                    // JCuda.cudaDeviceSynchronize();
                    // System.out.println("lossDiff:"+(System.nanoTime() - start44) / 1e6+"ms.");
                    /**
                     * back
                     *
                     */
                    network.back(cos, sin, this.lossDiff);
                    // JCuda.cudaDeviceSynchronize();
                    // long start55 = System.nanoTime();
                    /**
                     * update
                     *
                     */
                    this.network.update();
                    // JCuda.cudaDeviceSynchronize();
                    // System.out.println("update:"+(System.nanoTime() - start55) / 1e6+"ms.");
                    /**
                     * current time error
                     *
                     */
                    if (this.loss.isHasGPU()) {
                        this.currentError = MatrixOperation.sum(this.loss.syncHost()) / input.number;
                    } else {
                        this.currentError = MatrixOperation.sum(this.loss.data) / input.number;
                    }
                    output.syncHost();
                    if (it % 20 == 0) {
                        int time = output.number / batchSize;
                        if (trainingData.tokenizer != null) {
                            float error = this.accuracyBatchFisrt(input, tmpInput, output, label, tmpLabel, time, batchSize, trainingData.tokenizer, pad);
                        } else {
                            float error = this.accuracyBatchFisrt(input, output, label, time, batchSize, trainingData.vocab, pad);
                        }
                    }
                    String msg = "training[" + this.trainIndex + "]{" + it + "} (lr:" + this.network.learnRate + ") train_loss:" + this.currentError + " [costTime:" + (System.nanoTime() - start) / 1e6 + "ms.]";
                    System.out.println(msg);
                    this.batchIndex++;
                    /**
                     * update learning rate
                     *
                     */
                    this.updateLR(this.lr_step, it);
                }
            }
            /**
             * 停止训练
             *
             */
            System.out.println("training finish. [" + this.trainIndex + "] finalError:" + this.currentError);
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void trainLlama2_chinese_sft(CNWikiTokenizer4 trainingData) {
        // TODO Auto-generated method stub
        try {
            // CUDAModules.initCUDAFunctions();
            this.dataSize = trainingData.number;
            if (isWarmUp()) {
                this.network.learnRate = (float) (this.lr * Math.pow(batchIndex * 1.0f / burnIn * 1.0f, power));
            }
            Llama2 network = (Llama2) this.network;
            Tensor input = new Tensor(batchSize * network.time, 1, 1, 1, true);
            float[] tmpInput = new float[batchSize * network.time];
            Tensor label = new Tensor(batchSize, 1, 1, network.time, true);
            float[] tmpLabel = new float[batchSize * network.time];
            Tensor[] cs = RoPEKernel.getCosAndSin(network.time, network.embedDim, network.headNum);
            Tensor cos = cs[0];
            Tensor sin = cs[1];
            int pad = trainingData.tokenizer.pad;
            trainingData.loadData(input, label, tmpInput, tmpLabel);
            for (int i = 0; i < this.trainTime; i++) {
                if (this.trainIndex >= this.minTrainTime) {
                    break;
                }
                this.trainIndex = i + 1;
                Tensor output = null;
                /**
                 * 遍历整个训练集
                 *
                 */
                // System.out.println(trainingData.count_it);
                for (int it = 0; it < trainingData.count_it; it++) {
                    this.network.train_time = it + 1;
                    if (Math.abs(this.currentError) <= this.error) {
                        break;
                    }
                    long start = System.nanoTime();
                    this.loss.clear();
                    this.lossDiff.clear();
                    // JCuda.cudaDeviceSynchronize();
                    // long start22 = System.nanoTime();
                    /**
                     * 读取训练数据
                     *
                     */
                    trainingData.loadData(input, label, tmpInput, tmpLabel);
                    // trainingData.loadTrainData(indexs[it], input, label);
                    // JCuda.cudaDeviceSynchronize();
                    // System.out.println("loadTrainData:"+(System.nanoTime() - start22) /
                    // 1e6+"ms.");
                    /**
                     * forward
                     *
                     */
                    output = network.forward(cos, sin, input);
                    // JCuda.cudaDeviceSynchronize();
                    // long start33 = System.nanoTime();
                    /**
                     * loss
                     *
                     */
                    this.loss = network.loss(output, label, pad);
                    // JCuda.cudaDeviceSynchronize();
                    // System.out.println("loss:"+(System.nanoTime() - start33) / 1e6+"ms.");
                    // long start44 = System.nanoTime();
                    /**
                     * loss diff
                     *
                     */
                    this.lossDiff = network.lossDiff(output, label, pad);
                    // JCuda.cudaDeviceSynchronize();
                    // System.out.println("lossDiff:"+(System.nanoTime() - start44) / 1e6+"ms.");
                    /**
                     * back
                     *
                     */
                    network.back(cos, sin, this.lossDiff);
                    // JCuda.cudaDeviceSynchronize();
                    // long start55 = System.nanoTime();
                    /**
                     * update
                     *
                     */
                    this.network.update();
                    // JCuda.cudaDeviceSynchronize();
                    // System.out.println("update:"+(System.nanoTime() - start55) / 1e6+"ms.");
                    /**
                     * current time error
                     *
                     */
                    if (this.loss.isHasGPU()) {
                        this.currentError = MatrixOperation.sum(this.loss.syncHost()) / input.number;
                    } else {
                        this.currentError = MatrixOperation.sum(this.loss.data) / input.number;
                    }
                    output.syncHost();
                    if (it % 20 == 0) {
                        int time = output.number / batchSize;
                        if (trainingData.tokenizer != null) {
                            float error = this.accuracyBatchFisrt(input, tmpInput, output, label, tmpLabel, time, batchSize, trainingData.tokenizer, pad);
                        } else {
                            float error = this.accuracyBatchFisrt(input, output, label, time, batchSize, trainingData.vocab, pad);
                        }
                    }
                    String msg = "training[" + this.trainIndex + "]{" + it + "} (lr:" + this.network.learnRate + ") train_loss:" + this.currentError + " [costTime:" + (System.nanoTime() - start) / 1e6 + "ms.]";
                    System.out.println(msg);
                    this.batchIndex++;
                    /**
                     * update learning rate
                     *
                     */
                    this.updateLR(this.lr_step, it);
                }
            }
            /**
             * 停止训练
             *
             */
            System.out.println("training finish. [" + this.trainIndex + "] finalError:" + this.currentError);
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void trainLlama3_GEN(CNTokenizer trainingData) {
        // TODO Auto-generated method stub
        try {
            // CUDAModules.initCUDAFunctions();
            this.dataSize = trainingData.number;
            if (isWarmUp()) {
                this.network.learnRate = (float) (this.lr * Math.pow(batchIndex * 1.0f / burnIn * 1.0f, power));
            }
            Llama3 network = (Llama3) this.network;
            Tensor input = new Tensor(batchSize * network.time, 1, 1, 1, true);
            Tensor label = new Tensor(batchSize * network.time, 1, 1, 1, true);
            Tensor[] cs = RoPEKernel.getCosAndSin(network.time, network.embedDim, network.headNum);
            Tensor cos = cs[0];
            Tensor sin = cs[1];
            for (int i = 0; i < this.trainTime; i++) {
                if (this.trainIndex >= this.minTrainTime) {
                    break;
                }
                this.trainIndex = i + 1;
                int[][] indexs = MathUtils.randomInts(trainingData.trainData.length - network.time, this.batchSize);
                Tensor output = null;
                /**
                 * 遍历整个训练集
                 *
                 */
                for (int it = 0; it < indexs.length; it++) {
                    if (Math.abs(this.currentError) <= this.error) {
                        break;
                    }
                    long start = System.nanoTime();
                    this.loss.clear();
                    this.lossDiff.clear();
                    /**
                     * 读取训练数据
                     *
                     */
                    trainingData.loadIDXData(indexs[it], input, label);
                    /**
                     * forward
                     *
                     */
                    output = network.forward(cos, sin, input);
                    /**
                     * loss
                     *
                     */
                    this.loss = network.loss(output, label);
                    /**
                     * loss diff
                     *
                     */
                    this.lossDiff = network.lossDiff(output, label);
                    /**
                     * back
                     *
                     */
                    network.back(cos, sin, this.lossDiff);
                    /**
                     * update
                     *
                     */
                    this.network.update();
                    /**
                     * current time error
                     *
                     */
                    if (this.loss.isHasGPU()) {
                        this.currentError = MatrixOperation.sum(this.loss.syncHost()) / input.number;
                    } else {
                        this.currentError = MatrixOperation.sum(this.loss.data) / input.number;
                    }
                    output.syncHost();
                    int time = output.number / batchSize;
                    float error = this.accuracyIdx(input, output, label, time, batchSize, trainingData.vocab);
                    String msg = "training[" + this.trainIndex + "]{" + it + "} (lr:" + this.network.learnRate + ") accuracy:{" + error + "%} train_loss:" + this.currentError + " [costTime:" + (System.nanoTime() - start) / 1e6 + "ms.]";
                    System.out.println(msg);
                    this.batchIndex++;
                }
                /**
                 * update learning rate
                 *
                 */
                this.updateLR(this.lr_step);
            }
            /**
             * 停止训练
             *
             */
            System.out.println("training finish. [" + this.trainIndex + "] finalError:" + this.currentError);
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void trainLlama2_chinese(PreTrainDataset trainingData, int gradAccSteps, boolean saveWeight) {
        // TODO Auto-generated method stub
        try {
            // CUDAModules.initCUDAFunctions();
            this.dataSize = trainingData.number;
            if (isWarmUp()) {
                this.network.learnRate = (float) (this.lr * Math.pow(batchIndex * 1.0f / burnIn * 1.0f, power));
            }
            Llama2 network = (Llama2) this.network;
            Tensor input = new Tensor(batchSize * network.time, 1, 1, 1, true);
            float[] tmpInput = new float[batchSize * network.time];
            Tensor label = new Tensor(batchSize, 1, 1, network.time, true);
            float[] tmpLabel = new float[batchSize * network.time];
            Tensor[] cs = RoPEKernel.getCosAndSin(network.time, network.embedDim, network.headNum);
            Tensor cos = cs[0];
            Tensor sin = cs[1];
            // int pad = trainingData.tokenizer.pad;
            int pad = -1;
            trainingData.loadData2(input, label, tmpInput, tmpLabel, 0);
            for (int i = 0; i < this.trainTime; i++) {
                if (this.trainIndex >= this.minTrainTime) {
                    break;
                }
                this.trainIndex = i + 1;
                Tensor output = null;
                /**
                 * 遍历整个训练集
                 *
                 */
                for (int it = 0; it < trainingData.count_it; it++) {
                    if (Math.abs(this.currentError) <= this.error) {
                        break;
                    }
                    long start = System.nanoTime();
                    this.loss.clear();
                    this.lossDiff.clear();
                    // long start22 = System.nanoTime();
                    /**
                     * 读取训练数据
                     *
                     */
                    trainingData.loadData2(input, label, tmpInput, tmpLabel, it);
                    /**
                     * forward
                     *
                     */
                    output = network.forward(cos, sin, input);
                    // long start24 = System.nanoTime();
                    /**
                     * loss
                     *
                     */
                    this.loss = network.loss(output, label, pad);
                    // long start25 = System.nanoTime();
                    /**
                     * loss diff
                     *
                     */
                    this.lossDiff = network.lossDiff(output, label, pad);
                    // long start26 = System.nanoTime();
                    /**
                     * back
                     *
                     */
                    network.back(cos, sin, this.lossDiff);
                    /**
                     * update
                     *
                     */
                    if (gradAccSteps > 1) {
                        this.network.accGrad(gradAccSteps);
                    }
                    if (it > 1 && it % gradAccSteps == 0) {
                        if (this.network.CLIP_GRAD_NORM) {
                            this.network.clipGradNorm(1.0f);
                        }
                        this.network.update();
                    }
                    /**
                     * current time error
                     *
                     */
                    if (this.loss.isHasGPU()) {
                        this.currentError = MatrixOperation.sum(this.loss.syncHost()) / input.number;
                    } else {
                        this.currentError = MatrixOperation.sum(this.loss.data) / input.number;
                    }
                    if (it % 100 == 0) {
                        int time = output.number / batchSize;
                        if (trainingData.tokenizer != null) {
                            float error = this.accuracyBatchFisrt(input, output, label, time, batchSize, trainingData.tokenizer, pad);
                        } else {
                            float error = this.accuracyBatchFisrt(input, output, label, time, batchSize, trainingData.vocab, pad);
                        }
                    }
                    String msg = "training[" + this.trainIndex + "]{" + it + "/" + trainingData.count_it + "} (lr:" + this.network.learnRate + ") train_loss:" + this.currentError + " [costTime:" + (System.nanoTime() - start) / 1e6 + "ms.]";
                    System.out.println(msg);
                    /**
                     * dynamic update learnRate
                     *
                     */
                    updateLRDynamic(i * trainingData.count_it + it, this.trainTime * trainingData.count_it);
                    this.batchIndex++;
                    if (saveWeight && it > 1 && it % 20000 == 0) {
                        /**
                         * save model
                         *
                         */
                        String model_path = "/omega/models/llama2-92m-chinese_" + trainIndex + "_" + it + ".model";
                        ModelUtils.saveModel(network, model_path);
                    }
                }
                if (saveWeight) {
                    /**
                     * save model
                     *
                     */
                    String model_path = "/omega/models/llama2-92m-chinese_" + trainIndex + ".model";
                    ModelUtils.saveModel(network, model_path);
                }
            }
            /**
             * 停止训练
             *
             */
            System.out.println("training finish. [" + this.trainIndex + "] finalError:" + this.currentError);
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void train_llava_chinese(LVMPreTrainDataset trainingData, ClipVision vision, int gradAccSteps, boolean saveWeight) {
        // TODO Auto-generated method stub
        try {
            // CUDAModules.initCUDAFunctions();
            this.dataSize = trainingData.number;
            if (isWarmUp()) {
                this.network.learnRate = (float) (this.lr * Math.pow(batchIndex * 1.0f / burnIn * 1.0f, power));
            }
            Llava network = (Llava) this.network;
            Tensor imageInput = new Tensor(batchSize, 3, trainingData.imageSize, trainingData.imageSize, true);
            Tensor indice = new Tensor(batchSize, 1, 1, 1, true);
            Tensor input = new Tensor(batchSize * network.time, 1, 1, 1, true);
            Tensor label = new Tensor(batchSize, 1, 1, network.time, true);
            Tensor[] cs = RoPEKernel.getCosAndSin(network.time, network.embedDim, network.headNum);
            Tensor cos = cs[0];
            Tensor sin = cs[1];
            int pad = trainingData.tokenizer.pad();
            trainingData.loadData(input, label, imageInput, indice);
            for (int i = 0; i < this.trainTime; i++) {
                if (this.trainIndex >= this.minTrainTime) {
                    break;
                }
                this.trainIndex = i + 1;
                Tensor output = null;
                /**
                 * 遍历整个训练集
                 *
                 */
                for (int it = 0; it < trainingData.count_it; it++) {
                    if (Math.abs(this.currentError) <= this.error) {
                        break;
                    }
                    long start = System.nanoTime();
                    this.loss.clear();
                    this.lossDiff.clear();
                    /**
                     * 读取训练数据
                     *
                     */
                    trainingData.loadData(input, label, imageInput, indice);
                    /**
                     * vision forward
                     *
                     */
                    vision.forward(imageInput);
                    /**
                     * forward
                     *
                     */
                    output = network.forward(vision.getEncoder().getImageEncoders(), indice, cos, sin, input);
                    // System.out.println("forward:"+(System.nanoTime() - start23) / 1e6+"ms.");
                    // long start24 = System.nanoTime();
                    /**
                     * loss
                     *
                     */
                    this.loss = network.loss(output, label, pad);
                    // System.out.println("loss:"+(System.nanoTime() - start24) / 1e6+"ms.");
                    // label.showDM();
                    // long start25 = System.nanoTime();
                    /**
                     * loss diff
                     *
                     */
                    this.lossDiff = network.lossDiff(output, label, pad);
                    // System.out.println("lossDiff:"+(System.nanoTime() - start25) / 1e6+"ms.");
                    // long start26 = System.nanoTime();
                    /**
                     * back
                     *
                     */
                    network.back(indice, cos, sin, this.lossDiff);
                    // System.out.println("back:"+(System.nanoTime() - start26) / 1e6+"ms.");
                    /**
                     * update
                     *
                     */
                    if (gradAccSteps > 1) {
                        this.network.accGrad(gradAccSteps);
                    }
                    if (it > 1 && it % gradAccSteps == 0) {
                        if (this.network.CLIP_GRAD_NORM) {
                            this.network.clipGradNorm(1.0f);
                        }
                        this.network.update();
                    }
                    /**
                     * current time error
                     *
                     */
                    if (this.loss.isHasGPU()) {
                        int N = input.number;
                        if (pad > -1) {
                            N = input.number - MatrixUtils.countOccurrences(label.data, pad);
                        }
                        this.currentError = MatrixOperation.sum(this.loss.syncHost()) / N;
                    } else {
                        this.currentError = MatrixOperation.sum(this.loss.data) / input.number;
                    }
                    if (it % 100 == 0) {
                        int time = output.number / batchSize;
                        if (trainingData.tokenizer != null) {
                            float error = this.accuracyBatchFisrt(input, output, label, time, batchSize, trainingData.tokenizer, pad);
                        } else {
                            float error = this.accuracyBatchFisrt(input, output, label, time, batchSize, trainingData.vocab, pad);
                        }
                    }
                    String msg = "training[" + this.trainIndex + "]{" + it + "/" + trainingData.count_it + "} (lr:" + this.network.learnRate + ") train_loss:" + this.currentError + " [costTime:" + (System.nanoTime() - start) / 1e6 + "ms.]";
                    System.out.println(msg);
                    /**
                     * dynamic update learnRate
                     *
                     */
                    updateLRDynamic(i * trainingData.count_it + it, this.trainTime * trainingData.count_it);
                    this.batchIndex++;
                }
                if (saveWeight) {
                    /**
                     * save model
                     *
                     */
                    String model_path = "/omega/models/llava-26m-chinese_" + trainIndex + ".model";
                    ModelUtils.saveModel(network, model_path);
                }
            }
            /**
             * 停止训练
             *
             */
            System.out.println("training finish. [" + this.trainIndex + "] finalError:" + this.currentError);
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void trainLlama3_chinese(CNBpeTokenizer trainingData, int gradAccSteps, boolean saveWeight, String model_path) {
        // TODO Auto-generated method stub
        try {
            // CUDAModules.initCUDAFunctions();
            this.dataSize = trainingData.number;
            if (isWarmUp()) {
                this.network.learnRate = (float) (this.lr * Math.pow(batchIndex * 1.0f / burnIn * 1.0f, power));
            }
            Llama3 network = (Llama3) this.network;
            Tensor input = new Tensor(batchSize * network.time, 1, 1, 1, true);
            float[] tmpInput = new float[batchSize * network.time];
            Tensor label = new Tensor(batchSize, 1, 1, network.time, true);
            float[] tmpLabel = new float[batchSize * network.time];
            Tensor[] cs = RoPEKernel.getCosAndSin(network.time, network.embedDim, network.headNum);
            Tensor cos = cs[0];
            Tensor sin = cs[1];
            // int pad = trainingData.tokenizer.pad;
            int pad = -1;
            trainingData.loadData2(input, label, tmpInput, tmpLabel, 0);
            for (int i = 0; i < this.trainTime; i++) {
                if (this.trainIndex >= this.minTrainTime) {
                    break;
                }
                this.trainIndex = i + 1;
                Tensor output = null;
                /**
                 * 遍历整个训练集
                 *
                 */
                for (int it = 0; it < trainingData.count_it; it++) {
                    if (Math.abs(this.currentError) <= this.error) {
                        break;
                    }
                    long start = System.nanoTime();
                    this.loss.clear();
                    this.lossDiff.clear();
                    // JCuda.cudaDeviceSynchronize();
                    // long start22 = System.nanoTime();
                    /**
                     * 读取训练数据
                     *
                     */
                    trainingData.loadData2(input, label, tmpInput, tmpLabel, it);
                    // input.showDM(0);
                    // label.showDM(0);
                    // System.out.println("loadTrainData:"+(System.nanoTime() - start22) /
                    // 1e6+"ms.");
                    // System.out.println(JsonUtils.toJson(tmpInput));
                    // System.out.println(JsonUtils.toJson(tmpLabel));
                    // long start23 = System.nanoTime();
                    /**
                     * forward
                     *
                     */
                    output = network.forward(cos, sin, input);
                    // System.out.println("forward:"+(System.nanoTime() - start23) / 1e6+"ms.");
                    // long start24 = System.nanoTime();
                    /**
                     * loss
                     *
                     */
                    this.loss = network.loss(output, label, pad);
                    // System.out.println("loss:"+(System.nanoTime() - start24) / 1e6+"ms.");
                    // label.showDM();
                    // long start25 = System.nanoTime();
                    /**
                     * loss diff
                     *
                     */
                    this.lossDiff = network.lossDiff(output, label, pad);
                    // System.out.println("lossDiff:"+(System.nanoTime() - start25) / 1e6+"ms.");
                    // long start26 = System.nanoTime();
                    /**
                     * back
                     *
                     */
                    network.back(cos, sin, this.lossDiff);
                    // System.out.println("back:"+(System.nanoTime() - start26) / 1e6+"ms.");
                    /**
                     * update
                     *
                     */
                    if (gradAccSteps > 1) {
                        this.network.accGrad(gradAccSteps);
                    }
                    if (it > 1 && it % gradAccSteps == 0) {
                        if (this.network.CLIP_GRAD_NORM) {
                            this.network.clipGradNorm(1.0f);
                        }
                        this.network.update();
                    }
                    /**
                     * current time error
                     *
                     */
                    if (this.loss.isHasGPU()) {
                        this.currentError = MatrixOperation.sum(this.loss.syncHost()) / input.number;
                    } else {
                        this.currentError = MatrixOperation.sum(this.loss.data) / input.number;
                    }
                    if (it % 100 == 0) {
                        int time = output.number / batchSize;
                        if (trainingData.tokenizer != null) {
                            float error = this.accuracyBatchFisrt(input, output, label, time, batchSize, trainingData.tokenizer, pad);
                        } else {
                            float error = this.accuracyBatchFisrt(input, output, label, time, batchSize, trainingData.vocab, pad);
                        }
                    }
                    String msg = "training[" + this.trainIndex + "]{" + it + "/" + trainingData.count_it + "} (lr:" + this.network.learnRate + ") train_loss:" + this.currentError + " [costTime:" + (System.nanoTime() - start) / 1e6 + "ms.]";
                    System.out.println(msg);
                    /**
                     * dynamic update learnRate
                     *
                     */
                    updateLRDynamic(i * trainingData.count_it + it, this.trainTime * trainingData.count_it);
                    this.batchIndex++;
                    if (saveWeight && it > 1 && it % 20000 == 0) {
                        ModelUtils.saveModel(network, model_path);
                    }
                }
                if (saveWeight) {
                    ModelUtils.saveModel(network, model_path);
                }
            }
            /**
             * 停止训练
             *
             */
            System.out.println("training finish. [" + this.trainIndex + "] finalError:" + this.currentError);
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void trainLlama3_chinese(PreTrainDataset trainingData, int gradAccSteps, boolean saveWeight) {
        // TODO Auto-generated method stub
        try {
            // CUDAModules.initCUDAFunctions();
            this.dataSize = trainingData.number;
            if (isWarmUp()) {
                this.network.learnRate = (float) (this.lr * Math.pow(batchIndex * 1.0f / burnIn * 1.0f, power));
            }
            Llama3 network = (Llama3) this.network;
            Tensor input = new Tensor(batchSize * network.time, 1, 1, 1, true);
            float[] tmpInput = new float[batchSize * network.time];
            Tensor label = new Tensor(batchSize, 1, 1, network.time, true);
            float[] tmpLabel = new float[batchSize * network.time];
            Tensor[] cs = RoPEKernel.getCosAndSin(network.time, network.embedDim, network.headNum);
            Tensor cos = cs[0];
            Tensor sin = cs[1];
            int pad = -1;
            trainingData.loadData2(input, label, tmpInput, tmpLabel, 0);
            for (int i = 0; i < this.trainTime; i++) {
                if (this.trainIndex >= this.minTrainTime) {
                    break;
                }
                this.trainIndex = i + 1;
                Tensor output = null;
                /**
                 * 遍历整个训练集
                 */
                for (int it = 0; it < trainingData.count_it; it++) {
                    if (Math.abs(this.currentError) <= this.error) {
                        break;
                    }
                    long start = System.nanoTime();
                    this.loss.clear();
                    this.lossDiff.clear();
                    /**
                     * 读取训练数据
                     */
                    trainingData.loadData2(input, label, tmpInput, tmpLabel, it);
                    /**
                     * forward
                     */
                    output = network.forward(cos, sin, input);
                    /**
                     * loss
                     */
                    this.loss = network.loss(output, label, pad);
                    /**
                     * loss diff
                     */
                    this.lossDiff = network.lossDiff(output, label, pad);
                    /**
                     * back
                     */
                    network.back(cos, sin, this.lossDiff);
                    /**
                     * update
                     */
                    if (gradAccSteps > 1) {
                        this.network.accGrad(gradAccSteps);
                    }
                    if (it > 1 && it % gradAccSteps == 0) {
                        if (this.network.CLIP_GRAD_NORM) {
                            this.network.clipGradNorm(1.0f);
                        }
                        this.network.update();
                    }
                    /**
                     * current time error
                     */
                    if (this.loss.isHasGPU()) {
                        this.currentError = MatrixOperation.sum(this.loss.syncHost()) / input.number;
                    } else {
                        this.currentError = MatrixOperation.sum(this.loss.data) / input.number;
                    }
                    if (it % 100 == 0) {
                        int time = output.number / batchSize;
                        if (trainingData.tokenizer != null) {
                            float error = this.accuracyBatchFisrt(input, output, label, time, batchSize, trainingData.tokenizer, pad);
                        } else {
                            float error = this.accuracyBatchFisrt(input, output, label, time, batchSize, trainingData.vocab, pad);
                        }
                    }
                    String msg = "training[" + this.trainIndex + "]{" + it + "/" + trainingData.count_it + "} (lr:" + this.network.learnRate + ") train_loss:" + this.currentError + " [costTime:" + (System.nanoTime() - start) / 1e6 + "ms.]";
                    System.out.println(msg);
                    /**
                     * dynamic update learnRate
                     */
                    updateLRDynamic(i * trainingData.count_it + it, this.trainTime * trainingData.count_it);
                    this.batchIndex++;
                    if (saveWeight && it > 1 && it % 20000 == 0) {
                        /**
                         * save model
                         *
                         */
                        String model_path = "/omega/models/llama3-26m-chinese_" + trainIndex + "_" + it + ".model";
                        ModelUtils.saveModel(network, model_path);
                    }
                }
                if (saveWeight) {
                    /**
                     * save model
                     *
                     */
                    String model_path = "/omega/models/llama3-26m-chinese_" + trainIndex + ".model";
                    ModelUtils.saveModel(network, model_path);
                }
            }
            /**
             * 停止训练
             *
             */
            System.out.println("training finish. [" + this.trainIndex + "] finalError:" + this.currentError);
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void trainLlama3_chinese_sft(SFTDataset trainingData, int gradAccSteps, boolean saveWeight) {
        // TODO Auto-generated method stub
        try {
            // CUDAModules.initCUDAFunctions();
            this.dataSize = trainingData.number;
            if (isWarmUp()) {
                this.network.learnRate = (float) (this.lr * Math.pow(batchIndex * 1.0f / burnIn * 1.0f, power));
            }
            Llama3 network = (Llama3) this.network;
            Tensor input = new Tensor(batchSize * network.time, 1, 1, 1, true);
            float[] tmpInput = new float[batchSize * network.time];
            Tensor label = new Tensor(batchSize, 1, 1, network.time, true);
            float[] tmpLabel = new float[batchSize * network.time];
            Tensor[] cs = RoPEKernel.getCosAndSin(network.time, network.embedDim, network.headNum);
            Tensor cos = cs[0];
            Tensor sin = cs[1];
             int pad = trainingData.tokenizer.pad();
//            int pad = -1;
            trainingData.loadData(input, label, tmpInput, tmpLabel, 0);
            for (int i = 0; i < this.trainTime; i++) {
                if (this.trainIndex >= this.minTrainTime) {
                    break;
                }
                this.trainIndex = i + 1;
                Tensor output = null;
                /**
                 * 遍历整个训练集
                 *
                 */
                for (int it = 0; it < trainingData.count_it; it++) {
                    if (Math.abs(this.currentError) <= this.error) {
                        break;
                    }
                    long start = System.nanoTime();
                    this.loss.clear();
                    this.lossDiff.clear();
                    trainingData.loadData(input, label, tmpInput, tmpLabel, it);
                    /**
                     * forward
                     *
                     */
                    output = network.forward(cos, sin, input);
                    /**
                     * loss
                     *
                     */
                    this.loss = network.loss(output, label, pad);
                    /**
                     * loss diff
                     *
                     */
                    this.lossDiff = network.lossDiff(output, label, pad);
                    /**
                     * back
                     *
                     */
                    network.back(cos, sin, this.lossDiff);
                    /**
                     * update
                     *
                     */
                    if (gradAccSteps > 1) {
                        this.network.accGrad(gradAccSteps);
                    }
                    if (it > 1 && it % gradAccSteps == 0) {
                        if (this.network.CLIP_GRAD_NORM) {
                            this.network.clipGradNorm(1.0f);
                        }
                        this.network.update();
                    }
                    /**
                     * current time error
                     *
                     */
                    if (this.loss.isHasGPU()) {
                        int N = input.number;
                        if (pad > -1) {
                            N = input.number - MatrixUtils.countOccurrences(label.data, pad);
                        }
                        this.currentError = MatrixOperation.sum(this.loss.syncHost()) / N;
                    } else {
                        this.currentError = MatrixOperation.sum(this.loss.data) / input.number;
                    }
                    if (it % 100 == 0) {
                        int time = output.number / batchSize;
                        if (trainingData.tokenizer != null) {
                            float error = this.accuracyBatchFisrt(input, output, label, time, batchSize, trainingData.tokenizer, pad);
                        } else {
                            float error = this.accuracyBatchFisrt(input, output, label, time, batchSize, trainingData.vocab, pad);
                        }
                    }
                    String msg = "training[" + this.trainIndex + "]{" + it + "/" + trainingData.count_it + "} (lr:" + this.network.learnRate + ") train_loss:" + this.currentError + " [costTime:" + (System.nanoTime() - start) / 1e6 + "ms.]";
                    System.out.println(msg);
                    /**
                     * dynamic update learnRate
                     *
                     */
                    updateLRDynamic(i * trainingData.count_it + it, this.trainTime * trainingData.count_it);
                    this.batchIndex++;
                    if (saveWeight && it > 1 && it % 20000 == 0) {
                        /**
                         * save model
                         *
                         */
                        String model_path = "/omega/models/llama3-26m-chinese_" + trainIndex + "_" + it + ".model";
                        ModelUtils.saveModel(network, model_path);
                    }
                }
                if (saveWeight) {
                    /**
                     * save model
                     *
                     */
                    String model_path = "/omega/models/llama3-26m-chinese_" + trainIndex + ".model";
                    ModelUtils.saveModel(network, model_path);
                }
            }
            /**
             * 停止训练
             *
             */
            System.out.println("training finish. [" + this.trainIndex + "] finalError:" + this.currentError);
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void trainLlama3_chinese(PreTrainDataset2 trainingData, int gradAccSteps, boolean saveWeight, String model_path) {
        // TODO Auto-generated method stub
        try {
            // CUDAModules.initCUDAFunctions();
            this.dataSize = trainingData.number;
            if (isWarmUp()) {
                this.network.learnRate = (float) (this.lr * Math.pow(batchIndex * 1.0f / burnIn * 1.0f, power));
            }
            Llama3 network = (Llama3) this.network;
            Tensor input = new Tensor(batchSize * network.time, 1, 1, 1, true);
            float[] tmpInput = new float[batchSize * network.time];
            Tensor label = new Tensor(batchSize, 1, 1, network.time, true);
            float[] tmpLabel = new float[batchSize * network.time];
            Tensor[] cs = RoPEKernel.getCosAndSin(network.time, network.embedDim, network.headNum);
            Tensor cos = cs[0];
            Tensor sin = cs[1];
            int[] padCount = new int[1];
            // int pad = trainingData.tokenizer.pad;
            int pad = -1;
            trainingData.loadData(input, label, tmpInput, tmpLabel, padCount, 0);
            for (int i = 0; i < this.trainTime; i++) {
                if (this.trainIndex >= this.minTrainTime) {
                    break;
                }
                this.trainIndex = i + 1;
                Tensor output = null;
                /**
                 * 遍历整个训练集
                 *
                 */
                for (int it = 0; it < trainingData.count_it; it++) {
                    if (Math.abs(this.currentError) <= this.error) {
                        break;
                    }
                    long start = System.nanoTime();
                    this.loss.clear();
                    this.lossDiff.clear();
                    /**
                     * 读取训练数据
                     *
                     */
                    trainingData.loadData(input, label, tmpInput, tmpLabel, padCount, it);
                    /**
                     * forward
                     *
                     */
                    output = network.forward(cos, sin, input);
                    /**
                     * loss
                     *
                     */
                    this.loss = network.loss(output, label, pad);
                    /**
                     * loss diff
                     *
                     */
                    this.lossDiff = network.lossDiff(output, label, pad);
                    /**
                     * back
                     *
                     */
                    network.back(cos, sin, this.lossDiff);
                    /**
                     * update
                     *
                     */
                    if (gradAccSteps > 1) {
                        this.network.accGrad(gradAccSteps);
                    }
                    if (it > 1 && it % gradAccSteps == 0) {
                        if (this.network.CLIP_GRAD_NORM) {
                            this.network.clipGradNorm(1.0f);
                        }
                        this.network.update();
                    }
                    /**
                     * current time error
                     *
                     */
                    if (this.loss.isHasGPU()) {
                        this.currentError = MatrixOperation.sum(this.loss.syncHost()) / input.number;
                    } else {
                        this.currentError = MatrixOperation.sum(this.loss.data) / input.number;
                    }
                    if (it % 100 == 0) {
                        int time = output.number / batchSize;
                        if (trainingData.tokenizer != null) {
                            float error = this.accuracyBatchFisrt(input, output, label, time, batchSize, trainingData.tokenizer, pad);
                        } else {
                            float error = this.accuracyBatchFisrt(input, output, label, time, batchSize, trainingData.vocab, pad);
                        }
                    }
                    String msg = "training[" + this.trainIndex + "]{" + it + "/" + trainingData.count_it + "} (lr:" + this.network.learnRate + ") train_loss:" + this.currentError + " [costTime:" + (System.nanoTime() - start) / 1e6 + "ms.]";
                    System.out.println(msg);
                    /**
                     * dynamic update learnRate
                     *
                     */
                    updateLRDynamic(i * trainingData.count_it + it, this.trainTime * trainingData.count_it);
                    this.batchIndex++;
                    if (saveWeight && it > 1 && it % 20000 == 0) {
                        ModelUtils.saveModel(network, model_path);
                    }
                }
                if (saveWeight) {
                    ModelUtils.saveModel(network, model_path);
                }
            }
            /**
             * 停止训练
             *
             */
            System.out.println("training finish. [" + this.trainIndex + "] finalError:" + this.currentError);
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void trainASR_chinese(AudioDataset trainingData) {
        // TODO Auto-generated method stub
        try {
            this.dataSize = trainingData.number;
            if (isWarmUp()) {
                this.network.learnRate = (float) (this.lr * Math.pow(batchIndex * 1.0f / burnIn * 1.0f, power));
            }
            ASR network = (ASR) this.network;
            Tensor input = new Tensor(batchSize * trainingData.getMaxWavLength(), 1, 1, trainingData.getNumBins(), true);
            Tensor inputLen = new Tensor(batchSize, 1, 1, 1, true);
            Tensor labelInput = new Tensor(batchSize * trainingData.getMaxContextLen(), 1, 1, 1, true);
            Tensor labelLen = new Tensor(batchSize, 1, 1, 1, true);
            Tensor label = new Tensor(batchSize, 1, 1, trainingData.getMaxContextLen(), true);
            int[] count = new int[1];
             int pad = -1;
//            int pad = trainingData.tokenizer.pad();
            Tensor output = null;
            for (int i = 0; i < this.trainTime; i++) {
                if (this.trainIndex >= this.minTrainTime) {
                    break;
                }
                this.trainIndex = i + 1;
                int[][] indexs = trainingData.shuffle();
                //				int[][] indexs = trainingData.order();
                /**
                 * 遍历整个训练集
                 *
                 */
                for (int it = 0; it < trainingData.count_it; it++) {
                    long start = System.nanoTime();
                    this.loss.clear();
                    this.lossDiff.clear();
                    /**
                     * 读取训练数据
                     *
                     */
                    trainingData.loadData(indexs[it], input, inputLen, labelInput, labelLen, label, count);
                    /**
                     * forward
                     *
                     */
                    output = network.forward(input, inputLen, labelInput, labelLen);
                    /**
                     * loss
                     *
                     */
                    this.loss = network.loss(output, label, pad);
                    /**
                     * loss diff
                     *
                     */
                    this.lossDiff = network.lossDiff(output, label, pad, count[0]);
                    // label.showDM("label");
                    // this.lossDiff.showDM("lossDiff");
                    /**
                     * current time error
                     *
                     */
                    if (this.loss.isHasGPU()) {
                        this.currentError = MatrixOperation.sum(this.loss.syncHost()) / input.number;
                    } else {
                        this.currentError = MatrixOperation.sum(this.loss.data) / input.number;
                    }
                    /**
                     * update
                     *
                     */
                    if (this.currentError > 0) {
                        /**
                         * back
                         */
                        network.back(this.lossDiff);
                        this.network.update();
                    }
                    if (it % 100 == 0) {
                        int time = output.number / batchSize;
                        float error = this.accuracyBatchFisrt(output, label, time, batchSize, trainingData.tokenizer, pad);
                    }
                    String msg = "training[" + this.trainIndex + "]{" + it + "/" + trainingData.count_it + "} (lr:" + this.network.learnRate + ") train_loss:" + this.currentError + " [costTime:" + (System.nanoTime() - start) / 1e6 + "ms.]";
                    System.out.println(msg);
                    /**
                     * dynamic update learnRate
                     *
                     */
                    updateLRDynamic(i * trainingData.count_it + it, this.trainTime * trainingData.count_it);
                    this.batchIndex++;
                }
                //				String save_model_path = "H:\\model\\asr2.model";
                //				ModelUtils.saveModel(network, save_model_path);
            }
            /**
             * 停止训练
             *
             */
            System.out.println("training finish. [" + this.trainIndex + "] finalError:" + this.currentError);
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void updateLRDynamic(int it, int count) {
        int warmup_iters = 0;
        int lr_decay_iters = count;
        // System.out.println(this.lr);
        // System.out.println(lr_decay_iters);
        double min_lr = this.lr / 10.0d;
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
        // System.out.println(decay_ratio.doubleValue());
        BigDecimal coeff = new BigDecimal(0.5d).multiply(new BigDecimal(1).add(new BigDecimal(Math.cos(new BigDecimal(Math.PI).multiply(decay_ratio).doubleValue()))));
        BigDecimal tlr = new BigDecimal(min_lr).add(coeff.multiply(new BigDecimal((this.lr - min_lr))));
        tlr = tlr.setScale(24, BigDecimal.ROUND_HALF_DOWN);
        network.learnRate = (float) tlr.doubleValue();
    }

    @Override
    public void train(BaseData trainingData, BaseData testData) {
        // TODO Auto-generated method stub
    }

    @Override
    public void train(BaseData trainingData, BaseData validata, BaseData testData) {
        // TODO Auto-generated method stub
    }

    public void showOutputAndLabel(IndexDataLoader dataLoader, Tensor input, Tensor output, Tensor label, int batchSize) {
        String[] inputTxts = input2TXT(input, dataLoader, batchSize);
        String[] outputTxts = output2TXT(output, dataLoader, batchSize);
        String[] labelTxts = output2TXT(label, dataLoader, batchSize);
        for (int b = 0; b < batchSize; b++) {
            System.out.println("input:" + inputTxts[b]);
            System.out.println("output:" + outputTxts[b]);
            System.out.println("label :" + labelTxts[b]);
        }
    }

    public void gradClipping(Network network) {
        for (Layer layer : network.layerList) {
            if (layer.diffW != null) {
                // System.out.println(layer.getLayerType()+"-diffW");
                GradClipping.gradClipping(layer.diffW, 1e-7f);
            }
            if (layer.diffB != null) {
                // System.out.println("diffB");
                GradClipping.gradClipping(layer.diffB, 1e-7f);
            }
        }
    }

    public String predict(IndexDataLoader trainingData, String input_txt) {
        Seq2Seq network = (Seq2Seq) this.network;
        Tensor input = trainingData.loadByTxt(input_txt);
        Tensor[] outputs = network.encoder(input);
        Tensor de_hx = outputs[1];
        Tensor de_cx = outputs[2];
        float[] data = new float[trainingData.ch_characters];
        data[trainingData.ch_dictionary.get("<BOS>")] = 1.0f;
        Tensor startInput = new Tensor(1, 1, 1, trainingData.ch_characters, data, true);
        String txt = "";
        for (int t = 0; t < network.de_time; t++) {
            Tensor output = network.decoder(de_hx, de_cx, startInput);
            output.syncHost();
            String[] txts = output2TXT(output, trainingData, 1, 1);
            txt += txts[0];
            startInput.clear();
            startInput.data[trainingData.ch_dictionary.get(txts[0])] = 1.0f;
            startInput.hostToDevice();
        }
        System.out.println(txt);
        return txt;
    }

    public String predictRNN(IndexDataLoader trainingData, String input_txt) {
        Seq2SeqRNN network = (Seq2SeqRNN) this.network;
        Tensor input = trainingData.loadByTxt(input_txt);
        Tensor en_hidden = network.encoder(input);
        en_hidden.syncHost();
        Tensor de_hidden = new Tensor(1, 1, 1, en_hidden.width, en_hidden.getByNumber(en_hidden.number - 1), true);
        float[] data = new float[trainingData.ch_characters];
        data[trainingData.ch_dictionary.get("<BOS>")] = 1.0f;
        Tensor startInput = new Tensor(1, 1, 1, trainingData.ch_characters, data, true);
        String txt = "";
        for (int t = 0; t < network.de_time; t++) {
            Tensor output = network.decoder(de_hidden, startInput);
            output.syncHost();
            String[] txts = output2TXT(output, trainingData, 1, 1);
            txt += txts[0];
            startInput.clear();
            startInput.data[trainingData.ch_dictionary.get(txts[0])] = 1.0f;
            startInput.hostToDevice();
        }
        System.out.println(txt);
        return txt;
    }

    public void vail_gen(NanoGPT network, Tensor input, Tensor output, Tensor label, Tensor positions, CNTokenizer trainingData) {
        network.RUN_MODEL = RunModel.TEST;
        int[][] vailIndexs = MathUtils.randomInts(trainingData.vailData.length - network.time, this.batchSize);
        for (int it = 0; it < vailIndexs.length; it++) {
            if (it > 20) {
                break;
            }
            long start = System.nanoTime();
            this.loss.clear();
            /**
             * 读取训练数据
             *
             */
            trainingData.loadDataVail(vailIndexs[it], input, label);
            /**
             * forward
             *
             */
            output = network.forward(input, positions);
            /**
             * loss
             *
             */
            this.loss = network.loss(output, label);
            /**
             * current time error
             *
             */
            if (this.loss.isHasGPU()) {
                this.currentError = MatrixOperation.sum(this.loss.syncHost()) / input.number;
            } else {
                this.currentError = MatrixOperation.sum(this.loss.data) / input.number;
            }
            output.syncHost();
            int time = output.number / batchSize;
            float error = this.accuracyBatchFisrt(input, output, label, time, batchSize, trainingData.vocab);
            String msg = "vail[" + this.trainIndex + "]{" + it + "} (lr:" + this.network.learnRate + ") accuracy:{" + error + "%} vail_loss:" + this.currentError + " [costTime:" + (System.nanoTime() - start) / 1e6 + "ms.]";
            System.out.println(msg);
        }
    }

    public void vail_chat(NanoGPT network, Tensor input, Tensor output, Tensor label, Tensor positions, CNChatTokenizer trainingData) {
        network.RUN_MODEL = RunModel.TEST;
        int[][] vailIndexs = MathUtils.randomInts(trainingData.vailData.size() - network.time, this.batchSize);
        for (int it = 0; it < vailIndexs.length; it++) {
            if (it > 20) {
                break;
            }
            long start = System.nanoTime();
            this.loss.clear();
            /**
             * 读取训练数据
             *
             */
            trainingData.loadVailData(vailIndexs[it], input, label);
            /**
             * forward
             *
             */
            output = network.forward(input, positions);
            /**
             * loss
             *
             */
            this.loss = network.loss(output, label, trainingData.dictionary.get("<pad>"));
            /**
             * current time error
             *
             */
            if (this.loss.isHasGPU()) {
                this.currentError = MatrixOperation.sum(this.loss.syncHost()) / input.number;
            } else {
                this.currentError = MatrixOperation.sum(this.loss.data) / input.number;
            }
            output.syncHost();
            int time = output.number / batchSize;
            float error = this.accuracyBatchFisrt(input, output, label, time, batchSize, trainingData.vocab, trainingData.dictionary.get("<pad>"));
            String msg = "vail[" + this.trainIndex + "]{" + it + "} (lr:" + this.network.learnRate + ") accuracy:{" + error + "%} vail_loss:" + this.currentError + " [costTime:" + (System.nanoTime() - start) / 1e6 + "ms.]";
            System.out.println(msg);
        }
    }
}
