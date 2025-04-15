package com.omega.test.asr.test;

import java.util.Arrays;
import java.util.Map;
import java.util.Scanner;

import com.omega.common.tensor.Tensor;
import com.omega.utils.MatrixOperation;
import com.omega.utils.MatrixUtils;
import com.omega.utils.RandomUtils;
import com.omega.engine.gpu.CUDAManager;
import com.omega.engine.gpu.CUDAMemoryManager;
import com.omega.engine.gpu.SoftmaxKernel;
import com.omega.engine.loss.LossType;
import com.omega.engine.nn.network.ASR;
import com.omega.engine.optimizer.EDOptimizer;
import com.omega.engine.optimizer.lr.LearnRateUpdate;
import com.omega.engine.updater.UpdaterType;
import com.omega.data.asr.AudioDataset;
import com.omega.utils.asr.FBank;
import com.omega.utils.clip.ClipModelUtils;
import com.omega.models.transformer.LagJsonReader;
import com.omega.models.transformer.ModelUtils;
import com.omega.models.transformer.bpe.BPETokenizer3;

public class ASRTest {
	public static void loadWeight(Map<String, Object> weightMap, ASR network, boolean showLayers) {
		if (showLayers) {
			for (String key : weightMap.keySet()) {
				System.out.println(key);
			}
		}
		/**
		 * encoder
		 */
		ClipModelUtils.loadData(network.transformer.encoder.feature_emb.weight, weightMap, "frontend.linear.0.weight");
		ClipModelUtils.loadData(network.transformer.encoder.feature_emb.bias, weightMap, "frontend.linear.0.bias");
		for (int i = 0; i < 2; i++) {
			network.transformer.encoder.encoders.get(i).ln1.gamma = ClipModelUtils.loadData(
					network.transformer.encoder.encoders.get(i).ln1.gamma, weightMap, 1,
					"encoder.layers." + i + ".norm1.weight");
			network.transformer.encoder.encoders.get(i).ln1.beta = ClipModelUtils.loadData(
					network.transformer.encoder.encoders.get(i).ln1.beta, weightMap, 1,
					"encoder.layers." + i + ".norm1.bias");
			network.transformer.encoder.encoders.get(i).ln2.gamma = ClipModelUtils.loadData(
					network.transformer.encoder.encoders.get(i).ln2.gamma, weightMap, 1,
					"encoder.layers." + i + ".norm2.weight");
			network.transformer.encoder.encoders.get(i).ln2.beta = ClipModelUtils.loadData(
					network.transformer.encoder.encoders.get(i).ln2.beta, weightMap, 1,
					"encoder.layers." + i + ".norm2.bias");
			ClipModelUtils.loadData(network.transformer.encoder.encoders.get(i).attn.qLinerLayer.weight, weightMap,
					"encoder.layers." + i + ".multi_head_attn.wq.weight");
			ClipModelUtils.loadData(network.transformer.encoder.encoders.get(i).attn.qLinerLayer.bias, weightMap,
					"encoder.layers." + i + ".multi_head_attn.wq.bias");
			ClipModelUtils.loadData(network.transformer.encoder.encoders.get(i).attn.kLinerLayer.weight, weightMap,
					"encoder.layers." + i + ".multi_head_attn.wk.weight");
			ClipModelUtils.loadData(network.transformer.encoder.encoders.get(i).attn.kLinerLayer.bias, weightMap,
					"encoder.layers." + i + ".multi_head_attn.wk.bias");
			ClipModelUtils.loadData(network.transformer.encoder.encoders.get(i).attn.vLinerLayer.weight, weightMap,
					"encoder.layers." + i + ".multi_head_attn.wv.weight");
			ClipModelUtils.loadData(network.transformer.encoder.encoders.get(i).attn.vLinerLayer.bias, weightMap,
					"encoder.layers." + i + ".multi_head_attn.wv.bias");
			ClipModelUtils.loadData(network.transformer.encoder.encoders.get(i).attn.oLinerLayer.weight, weightMap,
					"encoder.layers." + i + ".multi_head_attn.W_out.weight");
			ClipModelUtils.loadData(network.transformer.encoder.encoders.get(i).attn.oLinerLayer.bias, weightMap,
					"encoder.layers." + i + ".multi_head_attn.W_out.bias");
			ClipModelUtils.loadData(network.transformer.encoder.encoders.get(i).pos_ffn.linear1.weight, weightMap,
					"encoder.layers." + i + ".poswise_ffn.lin1.weight");
			ClipModelUtils.loadData(network.transformer.encoder.encoders.get(i).pos_ffn.linear1.bias, weightMap,
					"encoder.layers." + i + ".poswise_ffn.lin1.bias");
			ClipModelUtils.loadData(network.transformer.encoder.encoders.get(i).pos_ffn.linear2.weight, weightMap,
					"encoder.layers." + i + ".poswise_ffn.lin2.weight");
			ClipModelUtils.loadData(network.transformer.encoder.encoders.get(i).pos_ffn.linear2.bias, weightMap,
					"encoder.layers." + i + ".poswise_ffn.lin2.bias");
		}
		/**
		 * decoder
		 */
		ClipModelUtils.loadData(network.transformer.decoder.tgt_emb.weight, weightMap, "decoder.tgt_emb.weight");
		ClipModelUtils.loadData(network.transformer.decoder.pos_emb.weight, weightMap, "decoder.pos_emb.weight");
		for (int i = 0; i < 2; i++) {
			network.transformer.decoder.decoders.get(i).ln1.gamma = ClipModelUtils.loadData(
					network.transformer.decoder.decoders.get(i).ln1.gamma, weightMap, 1,
					"decoder.layers." + i + ".norm1.weight");
			network.transformer.decoder.decoders.get(i).ln1.beta = ClipModelUtils.loadData(
					network.transformer.decoder.decoders.get(i).ln1.beta, weightMap, 1,
					"decoder.layers." + i + ".norm1.bias");
			network.transformer.decoder.decoders.get(i).ln2.gamma = ClipModelUtils.loadData(
					network.transformer.decoder.decoders.get(i).ln2.gamma, weightMap, 1,
					"decoder.layers." + i + ".norm2.weight");
			network.transformer.decoder.decoders.get(i).ln2.beta = ClipModelUtils.loadData(
					network.transformer.decoder.decoders.get(i).ln2.beta, weightMap, 1,
					"decoder.layers." + i + ".norm2.bias");
			network.transformer.decoder.decoders.get(i).ln3.gamma = ClipModelUtils.loadData(
					network.transformer.decoder.decoders.get(i).ln3.gamma, weightMap, 1,
					"decoder.layers." + i + ".norm3.weight");
			network.transformer.decoder.decoders.get(i).ln3.beta = ClipModelUtils.loadData(
					network.transformer.decoder.decoders.get(i).ln3.beta, weightMap, 1,
					"decoder.layers." + i + ".norm3.bias");
			ClipModelUtils.loadData(network.transformer.decoder.decoders.get(i).pos_ffn.linear1.weight, weightMap,
					"decoder.layers." + i + ".poswise_ffn.lin1.weight");
			ClipModelUtils.loadData(network.transformer.decoder.decoders.get(i).pos_ffn.linear1.bias, weightMap,
					"decoder.layers." + i + ".poswise_ffn.lin1.bias");
			ClipModelUtils.loadData(network.transformer.decoder.decoders.get(i).pos_ffn.linear2.weight, weightMap,
					"decoder.layers." + i + ".poswise_ffn.lin2.weight");
			ClipModelUtils.loadData(network.transformer.decoder.decoders.get(i).pos_ffn.linear2.bias, weightMap,
					"decoder.layers." + i + ".poswise_ffn.lin2.bias");
			ClipModelUtils.loadData(network.transformer.decoder.decoders.get(i).attn.qLinerLayer.weight, weightMap,
					"decoder.layers." + i + ".dec_attn.wq.weight");
			ClipModelUtils.loadData(network.transformer.decoder.decoders.get(i).attn.qLinerLayer.bias, weightMap,
					"decoder.layers." + i + ".dec_attn.wq.bias");
			ClipModelUtils.loadData(network.transformer.decoder.decoders.get(i).attn.kLinerLayer.weight, weightMap,
					"decoder.layers." + i + ".dec_attn.wk.weight");
			ClipModelUtils.loadData(network.transformer.decoder.decoders.get(i).attn.kLinerLayer.bias, weightMap,
					"decoder.layers." + i + ".dec_attn.wk.bias");
			ClipModelUtils.loadData(network.transformer.decoder.decoders.get(i).attn.vLinerLayer.weight, weightMap,
					"decoder.layers." + i + ".dec_attn.wv.weight");
			ClipModelUtils.loadData(network.transformer.decoder.decoders.get(i).attn.vLinerLayer.bias, weightMap,
					"decoder.layers." + i + ".dec_attn.wv.bias");
			ClipModelUtils.loadData(network.transformer.decoder.decoders.get(i).attn.oLinerLayer.weight, weightMap,
					"decoder.layers." + i + ".dec_attn.W_out.weight");
			ClipModelUtils.loadData(network.transformer.decoder.decoders.get(i).attn.oLinerLayer.bias, weightMap,
					"decoder.layers." + i + ".dec_attn.W_out.bias");
			ClipModelUtils.loadData(network.transformer.decoder.decoders.get(i).cross_attn.qLinerLayer.weight,
					weightMap, "decoder.layers." + i + ".enc_dec_attn.wq.weight");
			ClipModelUtils.loadData(network.transformer.decoder.decoders.get(i).cross_attn.qLinerLayer.bias, weightMap,
					"decoder.layers." + i + ".enc_dec_attn.wq.bias");
			ClipModelUtils.loadData(network.transformer.decoder.decoders.get(i).cross_attn.kLinerLayer.weight,
					weightMap, "decoder.layers." + i + ".enc_dec_attn.wk.weight");
			ClipModelUtils.loadData(network.transformer.decoder.decoders.get(i).cross_attn.kLinerLayer.bias, weightMap,
					"decoder.layers." + i + ".enc_dec_attn.wk.bias");
			ClipModelUtils.loadData(network.transformer.decoder.decoders.get(i).cross_attn.vLinerLayer.weight,
					weightMap, "decoder.layers." + i + ".enc_dec_attn.wv.weight");
			ClipModelUtils.loadData(network.transformer.decoder.decoders.get(i).cross_attn.vLinerLayer.bias, weightMap,
					"decoder.layers." + i + ".enc_dec_attn.wv.bias");
			ClipModelUtils.loadData(network.transformer.decoder.decoders.get(i).cross_attn.oLinerLayer.weight,
					weightMap, "decoder.layers." + i + ".enc_dec_attn.W_out.weight");
			ClipModelUtils.loadData(network.transformer.decoder.decoders.get(i).cross_attn.oLinerLayer.bias, weightMap,
					"decoder.layers." + i + ".enc_dec_attn.W_out.bias");
		}
		network.transformer.decoder.norm.gamma = ClipModelUtils.loadData(network.transformer.decoder.norm.gamma,
				weightMap, 1, "norm.weight");
		ClipModelUtils.loadData(network.fullyLayer.weight, weightMap, "linear.weight");
		ClipModelUtils.loadData(network.fullyLayer.bias, weightMap, "linear.bias");
	}

	public static void testASR() throws Exception {
		boolean bias = true;
		boolean dropout = false;
		int batchSize = 4;
		int maxWavLength = 100;
		int maxContextLen = 100;
		int numBins = 80;
		int wavDim = 512;
		int headNum = 8;
		int n_layers = 2;
		int nChannel = 2048;
		int voc_size = 1000;
		ASR network = new ASR(LossType.softmax_with_cross_entropy_idx, UpdaterType.adamw, numBins, maxWavLength,
				voc_size, maxContextLen, wavDim, nChannel, headNum, n_layers, bias, dropout);
		network.CUDNN = true;
		network.learnRate = 0.0001f;
		String weight = "H:\\model\\asr_transformer.json";
		loadWeight(LagJsonReader.readJsonFileSmallWeight(weight), network, true);
		Tensor fbank_feature = new Tensor(batchSize * maxWavLength, 1, 1, numBins,
				MatrixUtils.order(batchSize * maxWavLength * numBins, 0.01f, 0.01f), true);
		Tensor feat_lens = new Tensor(batchSize, 1, 1, 1, MatrixUtils.order(batchSize, 1, 1), true);
		Tensor label = new Tensor(batchSize * maxContextLen, 1, 1, 1,
				MatrixUtils.order(batchSize * maxContextLen, 1, 1), true);
		Tensor label2 = new Tensor(batchSize, 1, 1, maxContextLen, MatrixUtils.order(batchSize * maxContextLen, 1, 1),
				true);
		Tensor labelLen = new Tensor(batchSize, 1, 1, 1, MatrixUtils.val(batchSize, maxContextLen), true);
		network.init();
		Tensor output = network.forward(fbank_feature, feat_lens, label, labelLen);
		// output.showDM();
		Tensor loss = network.loss(output, label2, 0);
		float currentError = MatrixOperation.sum(loss.syncHost()) / label.number;
		System.out.println(currentError);
		Tensor lossDiff = network.lossDiff(output, label2, 0);
		network.back(lossDiff);
		network.update();
		output = network.forward(fbank_feature, feat_lens, label, labelLen);
		loss = network.loss(output, label2, 0);
		currentError = MatrixOperation.sum(loss.syncHost()) / label.number;
		System.out.println(currentError);
	}

	public static void asr_train() throws Exception {
		boolean bias = false;
		boolean dropout = false;
		int batchSize = 4;
		int maxWavLength = 512;
		int maxContextLen = 32;
		int numBins = 80;
		int wavDim = 512;
		int headNum = 8;
		int n_layers = 8;
		int nChannel = 2048;
		String vocabPath = "H:\\transformer_dataset\\6400\\vocab.json";
		String mergesPath = "H:\\transformer_dataset\\6400\\merges.txt";
		BPETokenizer3 tokenizer = new BPETokenizer3(vocabPath, mergesPath);
		String labelPath = "I:\\dataset\\asr\\data.json";
		String wavDirPath = "I:\\dataset\\asr\\data\\";
		AudioDataset dataLoader = new AudioDataset(tokenizer, labelPath, wavDirPath, numBins, maxWavLength,
				maxContextLen, batchSize);
		ASR network = new ASR(LossType.softmax_with_cross_entropy_idx, UpdaterType.adamw, numBins, maxWavLength,
				tokenizer.voc_size(), maxContextLen, wavDim, nChannel, headNum, n_layers, bias, dropout);
		network.CUDNN = true;
		network.learnRate = 0.0001f;
		// String model_path = "H:\\model\\asr2.model";
		// ModelUtils.loadModel(network, model_path);
		EDOptimizer optimizer = new EDOptimizer(network, batchSize, 20, 0.0001f, LearnRateUpdate.CONSTANT, false);
		optimizer.trainASR_chinese(dataLoader);
		String save_model_path = "H:\\model\\asr.model";
		ModelUtils.saveModel(network, save_model_path);
	}

	public static void asr_predict() throws Exception {
		boolean bias = true;
		boolean dropout = false;
		int maxWavLength = 256;
		int maxContextLen = 64;
		int numBins = 80;
		int wavDim = 256;
		int headNum = 8;
		int n_layers = 8;
		int nChannel = 2048;
		String vocabPath = "H:\\transformer_dataset\\6400\\vocab.json";
		String mergesPath = "H:\\transformer_dataset\\6400\\merges.txt";
		BPETokenizer3 tokenizer = new BPETokenizer3(vocabPath, mergesPath);
		ASR network = new ASR(LossType.softmax_with_cross_entropy_idx, UpdaterType.adamw, numBins, maxWavLength,
				tokenizer.voc_size(), maxContextLen, wavDim, nChannel, headNum, n_layers, bias, dropout);
		network.CUDNN = true;
		network.learnRate = 0.0001f;
		String model_path = "H:\\model\\asr.model";
		ModelUtils.loadModel(network, model_path);
		Tensor labelInput = null;
		Tensor input = new Tensor(maxWavLength, 1, 1, numBins, true);
		Tensor feat_lens = new Tensor(1, 1, 1, 1, true);
		String bos = "<s>";
		Scanner scanner = new Scanner(System.in);
		while (true) {
			System.out.println("请输入音频位置:");
			String path = scanner.nextLine();
			if (path.equals("exit")) {
				break;
			}
			float[] len = new float[1];
			input.data = FBank.fbank(path, numBins, maxWavLength, 0, len);
			input.hostToDevice();
			feat_lens.data[0] = len[0];
			feat_lens.hostToDevice();
			feat_lens.showDM();
			int[] idx = tokenizer.encodeInt(bos);
			int startLen = idx.length;
			float[] labelLenData = new float[] { 1 };
			Tensor labelLen = new Tensor(1, 1, 1, 1, labelLenData, true);
			labelInput = loadByTxtToIdx(labelInput, idx, maxContextLen);
			labelInput.showDM();
			for (int t = 0; t < maxContextLen - startLen; t++) {
				Tensor output = network.forward(input, feat_lens, labelInput, labelLen);
				output.syncHost();
				output.showDM();
				int nextIDX = output2NextIDXTopN(output, idx.length - 1, 3, network.cudaManager);
				idx = Arrays.copyOf(idx, idx.length + 1);
				idx[idx.length - 1] = nextIDX;
				if (nextIDX == tokenizer.eos) {
					break;
				}
				labelInput = loadByTxtToIdx(labelInput, idx, maxContextLen);
				labelLen.data[0] = idx.length;
				labelLen.hostToDevice();
			}
			int[] awIdx = Arrays.copyOfRange(idx, startLen, idx.length);
			System.out.println("search:" + tokenizer.decode(awIdx));
		}
		scanner.close();
	}

	public static int output2NextIDXTopN(Tensor output, int nextTokenIdx, int topK, CUDAManager cudaManager) {
		SoftmaxKernel kernel = new SoftmaxKernel(cudaManager);
		Tensor tmp = new Tensor(1, 1, 1, output.width, true);
		Tensor prof = new Tensor(1, 1, 1, output.width, true);
		if (nextTokenIdx < output.number) {
			tmp.hostToDevice(MatrixOperation.multiplication(output.getByNumber(nextTokenIdx), 0.7f));
			kernel.softmax_out(tmp, prof);
			return pickTopN(prof.syncHost(), topK);
		}
		return 0;
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

	public static Tensor loadByTxtToIdx(Tensor testInput, int[] idxs) {
		// System.out.println(idxs.length);
		testInput = Tensor.createTensor(testInput, idxs.length, 1, 1, 1, true);
		for (int t = 0; t < idxs.length; t++) {
			testInput.data[t] = idxs[t];
		}
		testInput.hostToDevice();
		return testInput;
	}

	public static Tensor loadByTxtToIdx(Tensor testInput, int[] idxs, int maxLen) {
		// System.out.println(idxs.length);
		testInput = Tensor.createTensor(testInput, maxLen, 1, 1, 1, true);
		testInput.clear();
		for (int t = 0; t < idxs.length; t++) {
			testInput.data[t] = idxs[t];
		}
		testInput.hostToDevice();
		return testInput;
	}

	public static void main(String[] args) {
		try {
			// testASR();
			asr_train();
//			asr_predict();
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		} finally {
			// TODO: handle finally clause
			CUDAMemoryManager.free();
		}
	}
}
