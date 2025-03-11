package com.omega.example.rnn.test;

import java.util.Arrays;
import java.util.Map;

import com.omega.common.data.Tensor;
import com.omega.common.utils.RandomUtils;
import com.omega.engine.active.ActiveType;
import com.omega.engine.gpu.CUDAMemoryManager;
import com.omega.engine.loss.LossType;
import com.omega.engine.nn.layer.EmbeddingLayer;
import com.omega.engine.nn.layer.FullyLayer;
import com.omega.engine.nn.layer.InputLayer;
import com.omega.engine.nn.layer.LSTMLayer;
import com.omega.engine.nn.layer.RNNBlockLayer;
import com.omega.engine.nn.layer.RNNLayer;
import com.omega.engine.nn.layer.active.LeakyReluLayer;
import com.omega.engine.nn.layer.normalization.BNLayer;
import com.omega.engine.nn.layer.normalization.LNLayer;
import com.omega.engine.nn.network.RNN;
import com.omega.engine.nn.network.RunModel;
import com.omega.engine.optimizer.MBSGDOptimizer;
import com.omega.engine.optimizer.lr.LearnRateUpdate;
import com.omega.engine.updater.UpdaterType;
import com.omega.example.rnn.data.OneHotDataLoader;

public class CharRNN {
	
	public void charRNN() {
		
		try {
			
			int time = 256;
			
			int batchSize = 64;
			
			int embedding_dim = 256;
			
			int hiddenSize = 512;
			
			String trainPath = "H:\\rnn_dataset\\dpcc.txt";
			
//			String trainPath = "H:\\rnn_dataset\\shakespeare.txt";
			
			OneHotDataLoader trainData = new OneHotDataLoader(trainPath, time, batchSize);
			
			RNN netWork = new RNN(LossType.softmax_with_cross_entropy, UpdaterType.adamw, time);
			
			InputLayer inputLayer = new InputLayer(1, 1, trainData.characters);
			
			EmbeddingLayer em = new EmbeddingLayer(trainData.characters, embedding_dim);

			RNNLayer l1 = new RNNLayer(embedding_dim, hiddenSize, time, ActiveType.tanh, false, netWork);
			
			RNNLayer l2 = new RNNLayer(hiddenSize, hiddenSize, time, ActiveType.tanh, false, netWork);
			
			RNNLayer l3 = new RNNLayer(hiddenSize, hiddenSize, time, ActiveType.tanh, false, netWork);
			
			FullyLayer f1 = new FullyLayer(hiddenSize, hiddenSize, false);
			BNLayer bn = new BNLayer();
			
//			LNLayer ln4 = new LNLayer();
			
			LeakyReluLayer a1 = new LeakyReluLayer();
			
			FullyLayer f2 = new FullyLayer(hiddenSize, trainData.characters, true);

			netWork.addLayer(inputLayer);
			netWork.addLayer(em);
			netWork.addLayer(l1);
			netWork.addLayer(l2);
			netWork.addLayer(l3);
			netWork.addLayer(f1);
//			netWork.addLayer(ln4);
			netWork.addLayer(bn);
			netWork.addLayer(a1);
			netWork.addLayer(f2);
			
			netWork.CUDNN = true;
			
			netWork.learnRate = 0.01f;
			
			MBSGDOptimizer optimizer = new MBSGDOptimizer(netWork, 2, 0.001f, batchSize, LearnRateUpdate.POLY, false);

//			long start = System.currentTimeMillis();
			
			optimizer.trainRNN(trainData);
			
			int gen_len = 1000;
			int max_len = 256;
			
			String pre_txt = "这个故事所造成的后果，便是造就了大批每天";
			
//			String pre_txt = "All:";
			
			Tensor input = null;
			
			Tensor output = null;
			
			input = createTxtData(input, pre_txt, trainData.characters, trainData.dictionary, max_len);
			
			netWork.RUN_MODEL = RunModel.TEST;
			
			for(int i = 0;i<gen_len;i++) {
				netWork.time = input.number;
				String txt = genTxt(input, output, netWork, trainData, max_len);
				if(netWork.time > 1) {
					pre_txt += txt.substring(input.number - 1, input.number);
				}else {
					pre_txt += txt;
				}
				input = createTxtData(input, pre_txt, trainData.characters, trainData.dictionary, max_len);
			}
			System.out.println(pre_txt);
			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public void charLSTM() {
		
		try {
			
			int time = 256;
			
			int batchSize = 64;
			
			int embedding_dim = 256;
			
			int hiddenSize = 512;
			
			String trainPath = "H:\\rnn_dataset\\dpcc50.txt";
			
//			String trainPath = "H:\\rnn_dataset\\shakespeare.txt";
			
			OneHotDataLoader trainData = new OneHotDataLoader(trainPath, time, batchSize);
			
			RNN netWork = new RNN(LossType.softmax_with_cross_entropy, UpdaterType.adamw, time);
			
			InputLayer inputLayer = new InputLayer(1, 1, trainData.characters);
			
			EmbeddingLayer em = new EmbeddingLayer(trainData.characters, embedding_dim);

			LSTMLayer l1 = new LSTMLayer(embedding_dim, hiddenSize, time, true, netWork);
			
			FullyLayer f1 = new FullyLayer(hiddenSize, hiddenSize, false);
//			BNLayer bn = new BNLayer();
			LNLayer ln4 = new LNLayer();
			LeakyReluLayer a1 = new LeakyReluLayer();
			
			FullyLayer f2 = new FullyLayer(hiddenSize, trainData.characters, true);

			netWork.addLayer(inputLayer);
			netWork.addLayer(em);
			netWork.addLayer(l1);
			netWork.addLayer(f1);
			netWork.addLayer(ln4);
//			netWork.addLayer(bn);
			netWork.addLayer(a1);
			netWork.addLayer(f2);
			
			netWork.CUDNN = true;
			
			netWork.learnRate = 0.01f;
			
			MBSGDOptimizer optimizer = new MBSGDOptimizer(netWork, 5, 0.001f, batchSize, LearnRateUpdate.CONSTANT, false);

//			long start = System.currentTimeMillis();
			
			optimizer.trainRNN(trainData);
			
			int gen_len = 1000;
			int max_len = 100;
			
			String pre_txt = "萧";
			
//			String pre_txt = "All:";
			
			Tensor input = null;
			
			Tensor output = null;
			
			input = createTxtData(input, pre_txt, trainData.characters, trainData.dictionary, max_len);
			
			netWork.RUN_MODEL = RunModel.TEST;
			
			for(int i = 0;i<gen_len;i++) {
				netWork.time = input.number;
				String txt = genTxt(input, output, netWork, trainData, max_len);
				if(netWork.time > 1) {
					pre_txt += txt.substring(input.number - 1, input.number);
				}else {
					pre_txt += txt;
				}
				input = createTxtData(input, pre_txt, trainData.characters, trainData.dictionary, max_len);
//				System.out.println(pre_txt);
			}
			System.out.println(pre_txt);
			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public void RNN() {
		
		try {
			
			int time = 3;
			
			int batchSize = 2;
			
			int inputSize = 3;
			
			int hiddenSize = 5;
			
			int number = time * batchSize;

			float[] xd = RandomUtils.order(time * batchSize * inputSize, 0.1f, 0.0f);
			
			Tensor x = new Tensor(number, 1, 1, inputSize, xd, true);
			
			RNN netWork = new RNN(LossType.softmax_with_cross_entropy, UpdaterType.adamw, time);
			
			InputLayer inputLayer = new InputLayer(1, 1, inputSize);

			RNNLayer l1 = new RNNLayer(inputSize, hiddenSize, time, ActiveType.tanh, true, netWork);

			netWork.addLayer(inputLayer);
			netWork.addLayer(l1);
			
			netWork.CUDNN = true;
			
			netWork.learnRate = 0.002f;
			
			MBSGDOptimizer optimizer = new MBSGDOptimizer(netWork, 500, 0.001f, batchSize, LearnRateUpdate.POLY, false);

			optimizer.testRNN(x);
			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public void LSTM() {
		
		try {
			
			int time = 3;
			
			int batchSize = 2;
			
			int inputSize = 3;
			
			int hiddenSize = 5;
			
			int number = time * batchSize;

			float[] xd = RandomUtils.order(time * batchSize * inputSize, 0.1f, 0.0f);
			
			Tensor x = new Tensor(number, 1, 1, inputSize, xd, true);
			
			RNN netWork = new RNN(LossType.softmax_with_cross_entropy, UpdaterType.adamw, time);
			
			InputLayer inputLayer = new InputLayer(1, 1, inputSize);

			LSTMLayer l1 = new LSTMLayer(inputSize, hiddenSize, time, true, netWork);

			netWork.addLayer(inputLayer);
			netWork.addLayer(l1);
			
			netWork.CUDNN = true;
			
			netWork.learnRate = 0.002f;
			
			MBSGDOptimizer optimizer = new MBSGDOptimizer(netWork, 500, 0.001f, batchSize, LearnRateUpdate.POLY, false);

			optimizer.testRNN(x);
			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public void RNN_CUDNN() {
		
		try {
			
			int time = 3;
			
			int batchSize = 2;
			
			int inputSize = 3;
			
			int hiddenSize = 5;
			
			int number = time * batchSize;

			float[] xd = RandomUtils.order(time * batchSize * inputSize, 0.1f, 0.0f);
			
			Tensor x = new Tensor(number, 1, 1, inputSize, xd, true);
			
			RNN netWork = new RNN(LossType.softmax_with_cross_entropy, UpdaterType.adamw, time);
			
			InputLayer inputLayer = new InputLayer(1, 1, inputSize);
			
			RNNBlockLayer l1 = new RNNBlockLayer(time, 1, inputSize, hiddenSize, 1, false, false, 0.0f, netWork);
			
			netWork.addLayer(inputLayer);
			netWork.addLayer(l1);
			
			netWork.CUDNN = true;
			
			netWork.learnRate = 0.002f;
			
			MBSGDOptimizer optimizer = new MBSGDOptimizer(netWork, 500, 0.001f, batchSize, LearnRateUpdate.POLY, false);

			optimizer.testRNN(x);
			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public void charRNN2() {
		
		try {
			
			int time = 576;
			
			int batchSize = 64;
			
//			int embedding_dim = 256;
			
			int hiddenSize = 1024;
			
			int rnnLayerNum = 1;
			
			float dropout = 0.0f;
			
			boolean bidirectional = false;
			
			int rnnMode = 1;
			
			String trainPath = "H:\\rnn_dataset\\shakespeare.txt";
			
			OneHotDataLoader trainData = new OneHotDataLoader(trainPath, time, batchSize);
			
			RNN netWork = new RNN(LossType.softmax_with_cross_entropy, UpdaterType.adamw, time);
			
			InputLayer inputLayer = new InputLayer(1, 1, trainData.characters);
			
//			EmbeddingLayer em = new EmbeddingLayer(trainData.characters, embedding_dim);
			
			RNNBlockLayer l1 = new RNNBlockLayer(time, rnnLayerNum, trainData.characters, hiddenSize, rnnMode, bidirectional, false, dropout, netWork);
			
//			TanhLayer a1 = new TanhLayer();
			
			FullyLayer f1 = new FullyLayer(hiddenSize, trainData.characters, true);
//			LeakyReluLayer a1 = new LeakyReluLayer();
//			SigmodLayer a1 = new SigmodLayer();
			
			netWork.addLayer(inputLayer);
//			netWork.addLayer(em);
			netWork.addLayer(l1);
//			netWork.addLayer(a1);
			netWork.addLayer(f1);
//			netWork.addLayer(a1);
			
			netWork.CUDNN = true;
			
			netWork.learnRate = 0.001f;
			
			MBSGDOptimizer optimizer = new MBSGDOptimizer(netWork, 3, 0.001f, batchSize, LearnRateUpdate.POLY, false);

//			long start = System.currentTimeMillis();
			
			optimizer.trainRNN(trainData);
			
			int gen_len = 1000;
			int max_len = 100;

			String pre_txt = "All:";
			
			Tensor input = null;
			
			Tensor output = null;
			
			input = createTxtData(input, pre_txt, trainData.characters, trainData.dictionary, max_len);
			
			for(int i = 0;i<gen_len;i++) {
				netWork.time = input.number;
				String txt = genTxt(input, output, netWork, trainData, max_len);
				if(netWork.time > 1) {
					pre_txt += txt.substring(input.number - 1, input.number);
				}else {
					pre_txt += txt;
				}
				input = createTxtData(input, pre_txt, trainData.characters, trainData.dictionary, max_len);
			}
			System.out.println(pre_txt);
			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public void charRNN3() {
		
		try {
			
			int time = 256;
			
			int batchSize = 64;
			
			int embedding_dim = 256;
			
			int hiddenSize = 512;
			
			int rnnLayerNum = 1;
			
			float dropout = 0.5f;
			
			boolean bidirectional = false;
			
			int rnnMode = 2;
			
			String trainPath = "H:\\rnn_dataset\\dpcc50.txt";
			
			OneHotDataLoader trainData = new OneHotDataLoader(trainPath, time, batchSize);
			
			RNN netWork = new RNN(LossType.softmax_with_cross_entropy, UpdaterType.adamw, time);
			
			InputLayer inputLayer = new InputLayer(1, 1, trainData.characters);
			
			EmbeddingLayer em = new EmbeddingLayer(trainData.characters, embedding_dim);
			
			RNNBlockLayer l1 = new RNNBlockLayer(time, rnnLayerNum, embedding_dim, hiddenSize, rnnMode, bidirectional, false, dropout, netWork);
			
			FullyLayer f1 = new FullyLayer(hiddenSize, hiddenSize, false);
			BNLayer bn = new BNLayer();
			LeakyReluLayer a1 = new LeakyReluLayer();
			
			FullyLayer f2 = new FullyLayer(hiddenSize, trainData.characters, true);

			netWork.addLayer(inputLayer);
			netWork.addLayer(em);
			netWork.addLayer(l1);
			netWork.addLayer(f1);
			netWork.addLayer(bn);
			netWork.addLayer(a1);
			netWork.addLayer(f2);
			
			netWork.CUDNN = true;
			
			netWork.learnRate = 0.001f;
			
			MBSGDOptimizer optimizer = new MBSGDOptimizer(netWork, 2, 0.001f, batchSize, LearnRateUpdate.SMART_HALF, false);
			
			optimizer.lr_step = new int[] {5, 8, 10};
			
//			long start = System.currentTimeMillis();
			
			optimizer.trainRNN(trainData);
			
			int gen_len = 1000;
			int max_len = 100;
			
			String pre_txt = "萧";
			
			netWork.RUN_MODEL = RunModel.TEST;
			
			Tensor input = null;
			
			Tensor output = null;

			input = createTxtData(input, pre_txt, trainData.characters, trainData.dictionary, max_len);
			
			for(int i = 0;i<gen_len;i++) {
				netWork.time = input.number;
				String txt = genTxt(input, output, netWork, trainData, max_len);
				if(netWork.time > 1) {
					pre_txt += txt.substring(input.number - 1, input.number);
				}else {
					pre_txt += txt;
				}
				input = createTxtData(input, pre_txt, trainData.characters, trainData.dictionary, max_len);
			}
			System.out.println(pre_txt);
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public static void createTxtData(String txt,int charDim,Map<Character,Integer> dictionary,Tensor input) {
		char[] charset = new char[txt.length()];
		txt.getChars(0, txt.length(), charset, 0);
//		System.out.println(JsonUtils.toJson(charset));
		float[] td = new float[charset.length * charDim];
		
		for(int i = 0;i<txt.length();i++) {
			td[i * charDim + dictionary.get(charset[i])] = 1;
		}
		input.number = charset.length;
		input.data = td;
		input.hostToDevice();
	}
	
	public static Tensor createTxtData(Tensor input,String txt,int charDim,Map<Character,Integer> dictionary,int maxLenght) {
		int charLength = txt.length();
		if(txt.length() > maxLenght) {
			charLength = maxLenght;
		}
		char[] charset = new char[charLength];
		int start = txt.length() - maxLenght;
		if(start <= 0) {
			start = 0;
		}
		txt.getChars(start, txt.length(), charset, 0);
//		System.out.println(JsonUtils.toJson(charset));
		float[] td = new float[charLength * charDim];
		
		for(int i = 0;i<charLength;i++) {
			td[i * charDim + dictionary.get(charset[i])] = 1;
		}
		input = Tensor.createTensor(input, charset.length, 1, 1, charDim, td, true);
		return input;
	}
	
	public static String genTxt(Tensor input,Tensor output,RNN network,OneHotDataLoader trainData,int maxLength) {
		output = network.forward(input);
		output.syncHost();
//		output.showDMByNumber(0);
		return output2TXT(output, trainData);
	}
	
	public static String output2TXT(Tensor output,OneHotDataLoader trainData) {
		String txt = "";
		for(int i = 0;i<output.number;i++) {
			int charIndex = pickTopN(output.getByNumber(i), 3);
			char c = trainData.dictionaryData[charIndex];
			txt += c;
		}
		return txt;
	}
	
	public static int pickTopN(float[] x,int n) {

		float[] sort = Arrays.copyOf(x, x.length);
		
		Arrays.sort(sort);
		
		float[] topN = Arrays.copyOfRange(sort, sort.length - n, sort.length);
		
		float v = topN[RandomUtils.getRandomNumber(topN)];
		
		for(int i = 0;i<x.length;i++) {
			if(v == x[i]) {
				return i;
			}
		}
		
		return 0;
	}
	
	public static void main(String[] args) {
		
		try {

//			CUDAModules.initContext();
			
			CharRNN t = new CharRNN();
			
//			t.LSTM();
			
			t.charLSTM();
			
//			t.RNN();
			
//			t.RNN_CUDNN();
			
//			t.charRNN();
			
//			t.charRNN2();
			
//			t.charRNN3();
			
//			t.createTxtData("这废物真是把家族的脸都给丢光了");
			
//			t.charRNN();
			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		} finally {
			// TODO: handle finally clause
			CUDAMemoryManager.free();
		}
		
	}
	
}
