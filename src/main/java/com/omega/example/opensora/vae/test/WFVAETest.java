package com.omega.example.opensora.vae.test;

import java.io.File;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

import org.bytedeco.ffmpeg.global.avcodec;
import org.bytedeco.ffmpeg.global.avutil;
import org.bytedeco.javacpp.Loader;

import com.omega.common.utils.MatrixOperation;
import com.omega.common.utils.MatrixShape;
import com.omega.common.utils.VideoReader;
import com.omega.common.utils.VideoWriter;
import com.omega.engine.gpu.CUDAMemoryManager;
import com.omega.engine.loss.LossType;
import com.omega.engine.nn.network.vae.WFVAE;
import com.omega.engine.nn.network.vqgan.LPIPS;
import com.omega.engine.optimizer.MBSGDOptimizer;
import com.omega.engine.optimizer.lr.LearnRateUpdate;
import com.omega.engine.tensor.Tensor;
import com.omega.engine.updater.UpdaterType;
import com.omega.example.clip.utils.ClipModelUtils;
import com.omega.example.opensora.vae.dataset.VideoDataLoaderEN;
import com.omega.example.transformer.utils.LagJsonReader;
import com.omega.example.transformer.utils.ModelUtils;
import com.omega.example.transformer.utils.bpe.BPETokenizerEN;
import com.omega.example.vae.test.LPIPSTest;

import cn.hutool.core.io.FileUtil;
import cn.hutool.core.text.csv.CsvData;
import cn.hutool.core.text.csv.CsvReader;
import cn.hutool.core.text.csv.CsvRow;
import cn.hutool.core.text.csv.CsvUtil;
import cn.hutool.core.text.csv.CsvWriter;

public class WFVAETest {

	/**
	 * 使用自定CSV配置生成CSV文件
	 * @param data 数据来源
	 * @param path CSV写文件配置
	 */
	public static void generateCsvWithConfig(CsvData data, String path){
		// 可以通过设置FileWriter的编码来控制输出文件的编码格式
		try(OutputStreamWriter fileWriter = new OutputStreamWriter(Files.newOutputStream(Paths.get(path)), "UTF-8");
			CsvWriter csvWriter = CsvUtil.getWriter(fileWriter)){
			csvWriter.write(data);
			csvWriter.flush();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}


	public static CsvRow findRow(List<CsvRow> rows, String filename) {
		for(CsvRow row:rows) {
			String fn = row.get(0);
			if(fn.contains(filename)) {
				return row;
			}
		}
		return null;
	}

	public static void createVideoDatasetCSV() {

		String csvPath = "D:\\dataset\\pexels_45k\\test.csv";
		String filePath = "D:\\dataset\\pexels_45k\\pexels_45k\\popular_2\\";
		String dataPath = "/root/gpufree-data/pexels/tar/pexels_45k/popular_2/";
		String outputPath = "D:\\dataset\\pexels_45k\\train_linux.csv";

		CsvReader reader = CsvUtil.getReader();
		CsvData data = reader.read(FileUtil.file(csvPath));

		List<CsvRow> rows = data.getRows();

		File root = new File(filePath);

		if(root.isDirectory()) {

			File[] files = root.listFiles();
			List<CsvRow> hasList = new ArrayList<CsvRow>();
			for(File file:files) {
				String filename = file.getName();
				CsvRow hit = findRow(rows, filename);
				if(hit != null) {
					hit.set(0, dataPath + filename);
					hit.add(filename);
					hasList.add(hit);
				}
			}

			List<String> header = new ArrayList<String>();
			header.add("path");
			header.add("text");
			header.add("num_frames");
			header.add("height");
			header.add("width");
			header.add("aspect_ratio");
			header.add("resolution");
			header.add("fps");
			header.add("filename");
			CsvData outData = new CsvData(header, hasList);

			generateCsvWithConfig(outData, outputPath);
		}

	}

	public static void createVideoDatasetCSV2() {

		String csvPath = "D:\\dataset\\pexels_45k\\test.csv";
		String filePath = "D:\\dataset\\t2v_dataset\\";
		String dataPath = "/root/gpufree-data/pexels/tar/pexels_45k/popular_2/";
		String outputPath = "D:\\dataset\\pexels_45k\\train_linux_set.csv";

		CsvReader reader = CsvUtil.getReader();
		CsvData data = reader.read(FileUtil.file(csvPath));

		List<CsvRow> rows = data.getRows();

		File root = new File(filePath);

		if(root.isDirectory()) {

			File[] files = root.listFiles();
			List<CsvRow> hasList = new ArrayList<CsvRow>();
			for(File file:files) {
				String filename = file.getName();
				CsvRow hit = findRow(rows, filename);
				if(hit != null) {
					hit.set(0, dataPath + filename);
					hit.add(filename);
					hasList.add(hit);
				}
			}

			List<String> header = new ArrayList<String>();
			header.add("path");
			header.add("text");
			header.add("num_frames");
			header.add("height");
			header.add("width");
			header.add("aspect_ratio");
			header.add("resolution");
			header.add("fps");
			header.add("filename");
			CsvData outData = new CsvData(header, hasList);

			generateCsvWithConfig(outData, outputPath);
		}

	}

	public static void checkDataset() {
		String dataPath = "D:\\dataset\\pexels_45k\\train_set.csv";
		String imgDirPath = "D:\\dataset\\t2v_dataset\\";
		CsvReader reader = CsvUtil.getReader();
		CsvData data = reader.read(FileUtil.file(dataPath));

		List<CsvRow> rows = data.getRows();

		File root = new File(imgDirPath);

		if(root.isDirectory()) {

			File[] files = root.listFiles();

			for(File file:files) {
				String filename = file.getName();
				CsvRow hit = findRow(rows, filename);
				if(hit != null) {
					File once = new File(imgDirPath + filename);
					if(once.isDirectory()) {
						Integer num_frames = Integer.parseInt(hit.get(2));
//        				System.err.println(num_frames+":"+once.listFiles().length);
						if(num_frames != once.listFiles().length) {
							System.err.println(filename);
						}
					}
				}
			}
		}
		System.out.println("check finish.");
	}

	public static void checkDataset2() {
		String dataPath = "D:\\dataset\\pexels_45k\\train_set.csv";
		String imgDirPath = "D:\\dataset\\t2v_dataset\\";
		CsvReader reader = CsvUtil.getReader();
		CsvData data = reader.read(FileUtil.file(dataPath));

		List<CsvRow> rows = data.getRows();

		File root = new File(imgDirPath);

		if(root.isDirectory()) {
			File[] files = root.listFiles();
			for(CsvRow row:rows) {
				String fn = row.get(0);
				boolean extis = false;
				for(File file:files) {
					if(fn.contains(file.getName())) {
						extis = true;
					}
				}
				if(!extis) {
					System.err.println(row.get(8));
				}
			}

		}
		System.out.println("check finish.");
	}

	public static void wf_vae_train() throws Exception {

		String dataPath = "D:\\dataset\\pexels_45k\\train_set.csv";
		String imgDirPath = "D:\\dataset\\t2v_dataset\\";
		int imgSize = 256;
		int num_frames = 9;
		int maxContextLen = 77;
		int batchSize = 2;

		float[] mean = new float[]{0.5f, 0.5f, 0.5f};
		float[] std = new float[]{0.5f, 0.5f, 0.5f};
		String vocabPath = "D:\\models\\bpe_tokenizer\\vocab.json";
		String mergesPath = "D:\\models\\bpe_tokenizer\\merges.txt";
		BPETokenizerEN bpe = new BPETokenizerEN(vocabPath, mergesPath, 49406, 49407);

		VideoDataLoaderEN dataLoader = new VideoDataLoaderEN(bpe, dataPath, imgDirPath, ".png", num_frames, imgSize, imgSize, maxContextLen, batchSize, mean, std);

		int latendDim = 8;
		int base_channels = 64;
		int en_energy_flow_hidden_size = 64;
		int de_energy_flow_hidden_size = 128;
		int num_res_blocks = 2;
		int connect_res_layer_num = 1;
		WFVAE network = new WFVAE(LossType.MSE, UpdaterType.adamw, num_frames, latendDim, imgSize, base_channels, en_energy_flow_hidden_size, de_energy_flow_hidden_size, num_res_blocks, connect_res_layer_num);
		network.CUDNN = true;
		network.learnRate = 0.0001f;

		LPIPS lpips = new LPIPS(LossType.MSE, UpdaterType.adamw, imgSize);
		String lpipsWeight = "D:\\models\\lpips.json";
		LPIPSTest.loadLPIPSWeight(LagJsonReader.readJsonFileSmallWeight(lpipsWeight), lpips, false);
		lpips.CUDNN = true;

//		String vaeWeight = "D:\\models\\wfvae-s.json";
//		ClipModelUtils.loadWeight(LagJsonReader.readJsonFileBigWeightIterator(vaeWeight), network, true);

		MBSGDOptimizer optimizer = new MBSGDOptimizer(network, 500, 0.00001f, batchSize, LearnRateUpdate.CONSTANT, false);

		optimizer.train_wfvae(dataLoader, lpips, "D:\\test\\vae\\256\\", "/omega/models/wfvae/");

		String save_model_path = "/omega/models/wfvae_256.model";
		ModelUtils.saveModel(network, save_model_path);

//        String encoder_out_path = "D:\\models\\encoder_out.json";
//        Map<String, Object> datas2 = LagJsonReader.readJsonFileSmallWeight(encoder_out_path);
//        Tensor encoder_out = new Tensor(2, 48, 32, 32, true);
//        ClipModelUtils.loadData(encoder_out, datas2, "encoder_out", 5);
//
//        String decoder_out_path = "D:\\models\\decoder_out.json";
//        Map<String, Object> datas3 = LagJsonReader.readJsonFileSmallWeight(decoder_out_path);
//        Tensor decoder_out = new Tensor(2, 3 * 9, 256, 256, true);
//        ClipModelUtils.loadData(decoder_out, datas3, "decoder_out", 5);
//
//        String target_out_path = "D:\\models\\target_out.json";
//        Map<String, Object> datas4 = LagJsonReader.readJsonFileSmallWeight(target_out_path);
//        Tensor target_out = new Tensor(2, 3 * 9, 256, 256, true);
//        ClipModelUtils.loadData(target_out, datas4, "target_out", 5);
//
//        String posteriors_rn_path = "D:\\models\\posteriors_rn.json";
//        Map<String, Object> rn_data = LagJsonReader.readJsonFileSmallWeight(posteriors_rn_path);
//        Tensor rn = new Tensor(2, 24, 32, 32, true);
//        ClipModelUtils.loadData(rn, rn_data, "posteriors_rn", 5);
//
//        network.sample(encoder_out, rn);
//
//        network.totalLoss(decoder_out, target_out, lpips);
//
//        network.backward(lpips);

	}


	public static void test_weight() throws Exception {
		int imgSize = 256;
		int num_frames = 9;

		String dataPath = "/root/lanyun-tmp/train_linux_set.csv";
		String imgDirPath = "/root/lanyun-tmp/t2v_dataset";
		int maxContextLen = 77;
		int batchSize = 2;

		float[] mean = new float[]{0.5f, 0.5f, 0.5f};
		float[] std = new float[]{0.5f, 0.5f, 0.5f};
		String vocabPath = "/root/lanyun-tmp/vocab.json";
		String mergesPath = "/root/lanyun-tmp/merges.txt";
		BPETokenizerEN bpe = new BPETokenizerEN(vocabPath, mergesPath, 49406, 49407);

		VideoDataLoaderEN dataLoader = new VideoDataLoaderEN(bpe, dataPath, imgDirPath, ".png", num_frames, imgSize, imgSize, maxContextLen, batchSize, mean, std);

		int latendDim = 8;
		int base_channels = 128;
		int en_energy_flow_hidden_size = 64;
		int de_energy_flow_hidden_size = 128;
		int num_res_blocks = 2;
		int connect_res_layer_num = 2;
		WFVAE network = new WFVAE(LossType.MSE, UpdaterType.adamw, num_frames, latendDim, imgSize, base_channels, en_energy_flow_hidden_size, de_energy_flow_hidden_size, num_res_blocks, connect_res_layer_num);
		network.CUDNN = true;
		network.learnRate = 0.0001f;
		network.init();

		LPIPS lpips = new LPIPS(LossType.MSE, UpdaterType.adamw, imgSize);
		String lpipsWeight = "/root/lanyun-tmp/lpips.json";
		LPIPSTest.loadLPIPSWeight(LagJsonReader.readJsonFileSmallWeight(lpipsWeight), lpips, false);
		lpips.CUDNN = true;

		String vaeWeight = "/root/lanyun-tmp/wfvae.json";
		ClipModelUtils.loadWeight(LagJsonReader.readJsonFileBigWeightIterator(vaeWeight), network, true);

		Tensor org_input = new Tensor(batchSize, dataLoader.num_frames * network.getChannel(), network.getHeight(), network.getWidth(), true);
		Tensor input = new Tensor(batchSize, network.getChannel() * dataLoader.num_frames, network.getHeight(), network.getWidth(), true);

		int[] index = new int[] {0, 1};

		dataLoader.loadData(index, org_input);

		/**
		 * [B, T, C, H, W] > B, C, T, H, W
		 */
		network.tensorOP.permute(org_input, input, new int[] {batchSize, num_frames, 3, imgSize, imgSize}, new int[] {batchSize, 3, num_frames, imgSize, imgSize}, new int[] {0, 2, 1, 3, 4});

//        String inputsPath = "D:\\models\\inputs.json";
//	    Map<String, Object> datas2 = LagJsonReader.readJsonFileSmallWeight(inputsPath);
//	    ClipModelUtils.loadData(input, datas2, "inputs", 5);
//
//        String posteriors_rn_path = "D:\\models\\rn.json";
//        Map<String, Object> rn_data = LagJsonReader.readJsonFileSmallWeight(posteriors_rn_path);
//        Tensor rn = new Tensor(1, 24, 32, 32, true);
//        ClipModelUtils.loadData(rn, rn_data, "rn", 5);
//
////        input.showDM("input");
//
//		Tensor lantnd = network.encode(input, rn);
////		lantnd.showDM("lantnd");
//		Tensor output = network.decode(lantnd);
//
//		float loss = network.totalLoss(output, input, lpips);
//
//		System.err.println("loss:"+loss);
//
//		network.backward(lpips);

		Tensor lantnd = network.encode(input);

		Tensor output = network.decode(lantnd);

		/**
		 * [B, C, T, H, W] > B, T, C, H, W
		 */
		network.tensorOP.permute(output, org_input, new int[] {batchSize, 3, num_frames, imgSize, imgSize}, new int[] {batchSize, num_frames, 3, imgSize, imgSize}, new int[] {0, 2, 1, 3, 4});
		org_input.syncHost();
		org_input.data = MatrixOperation.clampSelf(org_input.data, -1, 1);

		String testPath = "/root/lanyun-tmp/test3";

		MBSGDOptimizer.showVideos(testPath, num_frames, org_input, 0+"", mean, std);

	}

	public static List<Integer> getSampledFrames(int totalFrames, int numFrames) {
		List<Integer> sampledFrames = new ArrayList<>();

		if (numFrames <= 0 || totalFrames <= 0 || numFrames > totalFrames) {
			throw new IllegalArgumentException("Invalid input parameters");
		}

		if (numFrames == 1) {
			// 如果只需要采样1帧，默认取中间帧
			sampledFrames.add(totalFrames / 2);
			return sampledFrames;
		}

		// 计算采样间隔
		// double interval = (double)(totalFrames - 1) / (numFrames - 1);
		double interval = 1;
		for (int i = 0; i < numFrames; i++) {
			// 计算每帧位置并四舍五入
//			int frameIndex = (int)Math.round(i * interval);
			// 确保不超过总帧数范围
//			frameIndex = Math.min(frameIndex, totalFrames - 1);
			sampledFrames.add(i);
		}

		return sampledFrames;
	}

	/**
	 * 从视频中采样指定帧数
	 *
	 * @param videoPath 视频路径
	 * @param numFrames 采样帧数
	 * @param width 宽
	 * @param height 高
	 * @param mean 均值
	 * @param std 方差
	 * @param groupSize 每次处理的帧数，VAE的num_frames
	 *
	 * @return
	 */
	public static List<float[][][][]> getSampledFramesGroupData(String videoPath, int numFrames, int width, int height, float[] mean, float[] std, int groupSize) {
		File in = new File(videoPath);
		Loader.load(avutil.class);
		Loader.load(avcodec.class);
		int step = 1;
		List<Integer> sampledFrames = new ArrayList<>();
		try (VideoReader reader = new VideoReader(in, width, height, step, true)) {
			sampledFrames = getSampledFrames(reader.getTotalFrames(), numFrames);
			// 读取采样帧
			float[][][][] batchData = new float[reader.getTotalFrames()][3][width][height];

			int dataIndex = 0;
			while (true) {
				float[][][][] floats = reader.nextBatch4D(1, mean, std);
				if (floats.length == 0) {
					break;
				}
				batchData[dataIndex] = floats[0];
				dataIndex++;
			}

			int totalFrames = sampledFrames.size();
			int numGroups = (int) Math.ceil((double) totalFrames / groupSize);
			List<float[][][][]> frameGroups = new ArrayList<>();
			// 按16帧一组进行分组
			for (int groupIdx = 0; groupIdx < numGroups; groupIdx++) {
				int start = groupIdx * groupSize;
				int end = Math.min(start + groupSize, totalFrames);

				// 创建当前组的容器 [16][C][W][H]
				int framesInGroup = end - start;
				int channels = batchData[0].length;
				float[][][][] group = new float[framesInGroup][channels][width][height];

				// 填充当前组的数据
				for (int i = start; i < end; i++) {
					int frameIdx = sampledFrames.get(i);
					// 复制帧数据
					System.arraycopy(batchData[frameIdx], 0, group[i - start], 0, channels);
				}

				frameGroups.add(group);
			}

			return frameGroups;
		} catch (Exception e) {
			e.printStackTrace();
		}
		return null;
	}

	/**
	 *  vae生成视频
	 *
	 * @param videoPath 原视频地址
	 * @param numFrames 需要的总帧数 17 的倍数
	 * @throws Exception
	 */
	public static void reconVideo(String videoPath, int numFrames) throws Exception {
		int imgSize = 256;
		int num_frames = 17;

		int latendDim = 8;
		int base_channels = 128;
		int en_energy_flow_hidden_size = 64;
		int de_energy_flow_hidden_size = 128;
		int num_res_blocks = 2;
		int connect_res_layer_num = 2;
		WFVAE network = new WFVAE(LossType.MSE, UpdaterType.adamw, num_frames, latendDim, imgSize, base_channels, en_energy_flow_hidden_size, de_energy_flow_hidden_size, num_res_blocks, connect_res_layer_num);
		network.CUDNN = true;
		network.learnRate = 0.0001f;
		network.init();

		String vaeWeight = "/root/lanyun-tmp/wfvae.json";
		ClipModelUtils.loadWeight(LagJsonReader.readJsonFileBigWeightIterator(vaeWeight), network, true);

		// 输入数据
		Tensor org_input = new Tensor(1, num_frames * network.getChannel(), network.getHeight(), network.getWidth(), true);
		Tensor input = new Tensor(1, network.getChannel() * num_frames, network.getHeight(), network.getWidth(), true);

		File outOverwrite = new File("/root/lanyun-tmp/test/out_overwrite.mp4");
		VideoWriter writer = new VideoWriter(outOverwrite, true, imgSize, imgSize, 10, true);

		float[] mean = new float[]{0.5f, 0.5f, 0.5f};
		float[] std = new float[]{0.5f, 0.5f, 0.5f};

		// 采样视频数据并分组
		List<float[][][][]> groupData = getSampledFramesGroupData(videoPath, numFrames, imgSize, imgSize, mean, std, num_frames);
		if (null == groupData) {
			throw new RuntimeException("error get video data");
		}
		for (int i = 0; i < groupData.size(); i++) {
			float[] rowData = MatrixShape.reshape(groupData.get(i));
			org_input.setData(rowData);

			// B T C W H -> B C T H W
			network.tensorOP.permute(org_input, input, new int[] {1, num_frames, 3, imgSize, imgSize}, new int[] {1, 3, num_frames, imgSize, imgSize}, new int[] {0, 2, 1, 4, 3});

			// VAE encode
			Tensor lantnd = network.encode(input);

			// VAE Decode
			Tensor output = network.decode(lantnd);

			// B C T H W -> B T C W H
			network.tensorOP.permute(output, org_input, new int[] {1, 3, num_frames, imgSize, imgSize}, new int[] {1, num_frames, 3, imgSize, imgSize}, new int[] {0, 2, 1, 4, 3});
			org_input.syncHost();
			org_input.setData(MatrixOperation.clampSelf(org_input.data, -1, 1));
			// 写入视频数据
			writer.writeBatch(MatrixShape.toCube(org_input.data, num_frames, 3, imgSize, imgSize), mean, std);

			// 保存中间图片文件
			String testPath = "/root/lanyun-tmp/test";
			// 交换 H W
			network.tensorOP.permute(org_input, input, new int[] {1, num_frames, 3, imgSize, imgSize}, new int[] {1, num_frames, 3, imgSize, imgSize}, new int[] {0, 1, 2, 4, 3});
			MBSGDOptimizer.showVideos(testPath, num_frames, input, i+"", mean, std);

		}
		writer.close();
	}

	public static void main(String[] args) {
		try {

//        	createVideoDatasetCSV();
//        	createVideoDatasetCSV2();

//        	wf_vae_train();

//        	checkDataset();
//        	checkDataset2();

//			test_weight();

			reconVideo(args[0], 17);
//			ImageUtils.createGifFromFolder("C:\\Temp\\test_image", "C:\\Temp\\test\\4.gif", 100, false);
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		} finally {
			// TODO: handle finally clause
			CUDAMemoryManager.free();
		}
	}

}
