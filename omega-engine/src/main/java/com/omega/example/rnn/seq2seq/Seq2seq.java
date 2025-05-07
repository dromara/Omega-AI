package com.omega.example.rnn.seq2seq;

import com.omega.engine.gpu.CUDAMemoryManager;
import com.omega.engine.loss.LossType;
import com.omega.engine.nn.model.RNNCellType;
import com.omega.engine.nn.network.Seq2Seq;
import com.omega.engine.nn.network.Seq2SeqRNN;
import com.omega.engine.optimizer.EDOptimizer;
import com.omega.engine.optimizer.lr.LearnRateUpdate;
import com.omega.engine.updater.UpdaterType;
import com.omega.example.rnn.data.IndexDataLoader;

import java.util.Scanner;

public class Seq2seq {
    public static void main(String[] args) {
        try {
            //			CUDAModules.initContext();
            Seq2seq t = new Seq2seq();
            t.seq2seq();
            //			t.seq2seqRNN();
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        } finally {
            // TODO: handle finally clause
            CUDAMemoryManager.free();
        }
    }

    public void seq2seq() {
        try {
            int batchSize = 128;
            int en_em = 64;
            int de_em = 128;
            int en_hidden = 512;
            int de_hidden = 512;
            String trainPath = "H:\\rnn_dataset\\translate.csv";
            IndexDataLoader trainData = new IndexDataLoader(trainPath, batchSize);
            Seq2Seq network = new Seq2Seq(RNNCellType.LSTM, LossType.softmax_with_cross_entropy, UpdaterType.adamw, trainData.max_en, trainData.max_ch - 1, en_em, en_hidden, trainData.en_characters, de_em, de_hidden, trainData.ch_characters);
            network.CUDNN = true;
            network.learnRate = 0.01f;
            EDOptimizer optimizer = new EDOptimizer(network, batchSize, 200, 0.001f, LearnRateUpdate.SMART_HALF, false);
            optimizer.lr_step = new int[]{100};
            optimizer.trainSeq2Seq(trainData);
            Scanner scanner = new Scanner(System.in);
            while (true) {
                System.out.println("请输入英文:");
                String input_txt = scanner.nextLine();
                if (input_txt.equals("exit")) {
                    break;
                }
                input_txt = input_txt.toLowerCase();
                System.out.println(input_txt);
                optimizer.predict(trainData, input_txt);
            }
            scanner.close();
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void seq2seqRNN() {
        try {
            int batchSize = 128;
            int en_em = 64;
            int de_em = 128;
            int en_hidden = 512;
            int de_hidden = 512;
            String trainPath = "H:\\rnn_dataset\\translate4000.csv";
            IndexDataLoader trainData = new IndexDataLoader(trainPath, batchSize);
            Seq2SeqRNN network = new Seq2SeqRNN(LossType.softmax_with_cross_entropy, UpdaterType.adamw, trainData.max_en, trainData.max_ch - 1, en_em, en_hidden, trainData.en_characters, de_em, de_hidden, trainData.ch_characters);
            network.CUDNN = true;
            network.learnRate = 0.01f;
            EDOptimizer optimizer = new EDOptimizer(network, batchSize, 100, 0.001f, LearnRateUpdate.SMART_HALF, false);
            //			optimizer.lr_step = new int[] {20,50,80};
            optimizer.trainSeq2SeqRNN(trainData);
            Scanner scanner = new Scanner(System.in);
            while (true) {
                System.out.println("请输入英文:");
                String input_txt = scanner.nextLine();
                if (input_txt.equals("exit")) {
                    break;
                }
                input_txt = input_txt.toLowerCase();
                System.out.println(input_txt);
                optimizer.predictRNN(trainData, input_txt);
            }
            scanner.close();
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
}

