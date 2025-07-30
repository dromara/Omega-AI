package com.omega.example.vae.test;

import cn.hutool.core.io.FileUtil;
import cn.hutool.core.text.csv.*;
import com.omega.engine.gpu.CUDAMemoryManager;
import com.omega.engine.loss.LossType;
import com.omega.engine.nn.network.vae.WFVAE;
import com.omega.engine.nn.network.vqgan.LPIPS;
import com.omega.engine.optimizer.MBSGDOptimizer;
import com.omega.engine.optimizer.lr.LearnRateUpdate;
import com.omega.engine.updater.UpdaterType;
import com.omega.example.clip.utils.ClipModelUtils;
import com.omega.example.transformer.utils.LagJsonReader;
import com.omega.example.transformer.utils.ModelUtils;
import com.omega.example.transformer.utils.bpe.BPETokenizerEN;
import com.omega.example.vae.dataset.VideoDataLoaderEN;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

public class WFVAETest {

	/**
	 * 使用自定CSV配置生成CSV文件
	 * @param users 数据来源
	 * @param config CSV写文件配置
	 */
	public static void generateCsvWithConfig(CsvData data, String path){
	    // 可以通过设置FileWriter的编码来控制输出文件的编码格式
	    try(FileWriter fileWriter = new FileWriter(path, StandardCharsets.UTF_8);
	        CsvWriter csvWriter = CsvUtil.getWriter(fileWriter)){
	    	csvWriter.write(data);
	        csvWriter.flush();
	    } catch (IOException e) {
	        e.printStackTrace();
	    }
	}


	public static CsvRow findRow(List<CsvRow> rows,String filename) {
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
		String outputPath = "D:\\dataset\\pexels_45k\\train.csv";

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
					hit.set(0, filePath + filename);
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

	public static void wf_vae_train() throws Exception {

		String dataPath = "D:\\dataset\\pexels_45k\\train.csv";
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

		MBSGDOptimizer optimizer = new MBSGDOptimizer(network, 500, 0.00001f, batchSize, LearnRateUpdate.CONSTANT, false);

		optimizer.train_wfvae(dataLoader, lpips, "D:\\test\\vae\\256\\");
		String save_model_path = "/omega/models/wfvae_256.model";
		ModelUtils.saveModel(network, save_model_path);

	}

	public static void test_weight() {
		int imgSize = 256;
		int num_frames = 9;

		int latendDim = 8;
		int base_channels = 128;
		int en_energy_flow_hidden_size = 64;
		int de_energy_flow_hidden_size = 128;
		int num_res_blocks = 2;
		int connect_res_layer_num = 2;
		WFVAE network = new WFVAE(LossType.MSE, UpdaterType.adamw, num_frames, latendDim, imgSize, base_channels, en_energy_flow_hidden_size, de_energy_flow_hidden_size, num_res_blocks, connect_res_layer_num);
		network.CUDNN = true;
		network.learnRate = 0.0001f;

		String vaeWeight = "c:\\temp\\wfvae.json";
		ClipModelUtils.loadWeight(LagJsonReader.readJsonFileBigWeightIterator(vaeWeight), network, true);
	}

	public static void main(String[] args) {
		try {

//        	createVideoDatasetCSV();

			test_weight();

		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		} finally {
			// TODO: handle finally clause
			CUDAMemoryManager.free();
		}
	}

}
