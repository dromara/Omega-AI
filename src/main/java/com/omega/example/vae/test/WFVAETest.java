package com.omega.example.vae.test;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import com.omega.common.utils.ImageUtils;
import com.omega.common.utils.MatrixOperation;
import com.omega.engine.gpu.CUDAMemoryManager;
import com.omega.engine.loss.LossType;
import com.omega.engine.nn.network.vae.WFVAE;
import com.omega.engine.nn.network.vqgan.LPIPS;
import com.omega.engine.optimizer.MBSGDOptimizer;
import com.omega.engine.optimizer.lr.LearnRateUpdate;
import com.omega.engine.tensor.Tensor;
import com.omega.engine.updater.UpdaterType;
import com.omega.example.clip.utils.ClipModelUtils;
import com.omega.example.transformer.utils.LagJsonReader;
import com.omega.example.transformer.utils.ModelUtils;
import com.omega.example.transformer.utils.bpe.BPETokenizerEN;
import com.omega.example.vae.dataset.VideoDataLoaderEN;

import cn.hutool.core.io.FileUtil;
import cn.hutool.core.text.csv.CsvData;
import cn.hutool.core.text.csv.CsvReader;
import cn.hutool.core.text.csv.CsvRow;
import cn.hutool.core.text.csv.CsvUtil;
import cn.hutool.core.text.csv.CsvWriter;

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

		String dataPath = "D:\\dataset\\pexels_45k\\train_set.csv";
        String imgDirPath = "D:\\dataset\\t2v_dataset\\";
        int maxContextLen = 77;
        int batchSize = 1;
       
        float[] mean = new float[]{0.5f, 0.5f, 0.5f};
        float[] std = new float[]{0.5f, 0.5f, 0.5f};
        String vocabPath = "D:\\models\\bpe_tokenizer\\vocab.json";
        String mergesPath = "D:\\models\\bpe_tokenizer\\merges.txt";
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
        String lpipsWeight = "D:\\models\\lpips.json";
        LPIPSTest.loadLPIPSWeight(LagJsonReader.readJsonFileSmallWeight(lpipsWeight), lpips, false);
        lpips.CUDNN = true;

		String vaeWeight = "D:\\models\\wfvae-s.json";
		ClipModelUtils.loadWeight(LagJsonReader.readJsonFileBigWeightIterator(vaeWeight), network, true);
		
		Tensor org_input = new Tensor(batchSize, dataLoader.num_frames * network.getChannel(), network.getHeight(), network.getWidth(), true);
        Tensor input = new Tensor(batchSize, network.getChannel() * dataLoader.num_frames, network.getHeight(), network.getWidth(), true);
		
//        int[] index = new int[] {13, 11};
//        
//		dataLoader.loadData(index, org_input);
		
		/**
         * [B, T, C, H, W] > B, C, T, H, W
         */
//        network.tensorOP.permute(org_input, input, new int[] {batchSize, num_frames, 3, imgSize, imgSize}, new int[] {batchSize, 3, num_frames, imgSize, imgSize}, new int[] {0, 2, 1, 3, 4});

        String inputsPath = "D:\\models\\inputs.json";
	    Map<String, Object> datas2 = LagJsonReader.readJsonFileSmallWeight(inputsPath);
	    ClipModelUtils.loadData(input, datas2, "inputs", 5);
        
        String posteriors_rn_path = "D:\\models\\rn.json";
        Map<String, Object> rn_data = LagJsonReader.readJsonFileSmallWeight(posteriors_rn_path);
        Tensor rn = new Tensor(1, 24, 32, 32, true);
        ClipModelUtils.loadData(rn, rn_data, "rn", 5);

//        input.showDM("input");

		Tensor lantnd = network.encode(input, rn);
//		lantnd.showDM("lantnd");
		Tensor output = network.decode(lantnd);

		float loss = network.totalLoss(output, input, lpips);

		System.err.println("loss:"+loss);

		network.backward(lpips);

//		/**
//         * [B, C, T, H, W] > B, T, C, H, W
//         */
//        network.tensorOP.permute(output, org_input, new int[] {batchSize, 3, num_frames, imgSize, imgSize}, new int[] {batchSize, num_frames, 3, imgSize, imgSize}, new int[] {0, 2, 1, 3, 4});
//        org_input.syncHost();
//        org_input.data = MatrixOperation.clampSelf(org_input.data, -1, 1);
//
//		String testPath = "D:\\test\\vae\\256\\";
//
//		MBSGDOptimizer.showVideos(testPath, num_frames, org_input, 0+"", mean, std);
		
	}

	public static void main(String[] args) {
        try {
        	
//        	createVideoDatasetCSV();
//        	createVideoDatasetCSV2();

//        	wf_vae_train();

//        	checkDataset();
//        	checkDataset2();
        	
			test_weight();

//			ImageUtils.createGifFromFolder("C:\\Temp\\gif", "C:\\Temp\\gif\\result.gif", 500, false);
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		} finally {
			// TODO: handle finally clause
			CUDAMemoryManager.free();
		}
	}
	
}
