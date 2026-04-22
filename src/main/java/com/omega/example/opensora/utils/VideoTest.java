package com.omega.example.opensora.utils;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import com.omega.common.utils.ImageUtils;
import com.omega.common.utils.MathUtils;
import com.omega.engine.gpu.CUDAMemoryManager;
import com.omega.engine.loss.LossType;
import com.omega.engine.nn.network.ClipTextModel;
import com.omega.engine.nn.network.RunModel;
import com.omega.engine.nn.network.vae.LTXVideo_VAE;
import com.omega.engine.tensor.Tensor;
import com.omega.engine.updater.UpdaterType;
import com.omega.example.common.ModeLoaderlUtils;
import com.omega.example.dit.utils.DatasetCreater;
import com.omega.example.transformer.utils.LagJsonReader;
import com.omega.example.transformer.utils.ModelUtils;
import com.omega.example.transformer.utils.bpe.BPETokenizerEN;

import cn.hutool.core.io.FileUtil;
import cn.hutool.core.text.csv.CsvData;
import cn.hutool.core.text.csv.CsvReader;
import cn.hutool.core.text.csv.CsvRow;
import cn.hutool.core.text.csv.CsvUtil;
import cn.hutool.core.text.csv.CsvWriter;
import jcuda.driver.JCudaDriver;

public class VideoTest {
	
	public static void loadWeight(Map<String, Object> weightMap, LTXVideo_VAE vae, boolean showLayers) {
        if (showLayers) {
            for (String key : weightMap.keySet()) {
                System.out.println(key);
            }
        }
    }

	
	public static void vae() {
		
		try {
			
			int num_frames = 17;
			int height = 352;
			int width = 640;
			int patch_size_t = 1;
			int patch_size = 4;
			int[] block_out_channels = new int[] {128, 256, 512, 512};
			int[] layers_per_block = new int[] {4, 3, 3, 3, 4};
			boolean[] spatio_temporal_scaling = new boolean[] {true, true, true, false};
			
			LTXVideo_VAE vae = new LTXVideo_VAE(LossType.MSE, UpdaterType.adamw, num_frames, height, width, patch_size_t, patch_size, block_out_channels, layers_per_block, spatio_temporal_scaling);
			vae.CUDNN = true;
			
//	        String weight = "D:\\models\\ltx_vae\\ltx_vae.json";
//	        loadWeight(LagJsonReader.readJsonFileBigWeightIterator(weight), vae, true);
			
//	        /**
//	         * encoders
//	         */
//	        // conv_in
//			Map<String, Tensor> weights = new HashMap<String, Tensor>();
//			weights.put("encoder.conv_in.conv.weight", vae.encoder.conv_in.weight);
//			weights.put("encoder.conv_in.conv.bias", vae.encoder.conv_in.bias);
//			
//			//down_blocks
//			for(int i=0;i<4;i++) {
//				//resnets
//				for(int j = 0;j<layers_per_block[i];j++) {
//					weights.put("encoder.down_blocks."+i+".resnets."+j+".conv1.conv.weight", vae.encoder.down_blocks.get(i).resnets.get(j).conv1.weight);
//					weights.put("encoder.down_blocks."+i+".resnets."+j+".conv1.conv.bias", vae.encoder.down_blocks.get(i).resnets.get(j).conv1.bias);
//					weights.put("encoder.down_blocks."+i+".resnets."+j+".conv2.conv.weight", vae.encoder.down_blocks.get(i).resnets.get(j).conv2.weight);
//					weights.put("encoder.down_blocks."+i+".resnets."+j+".conv2.conv.bias", vae.encoder.down_blocks.get(i).resnets.get(j).conv2.bias);
//				}
//				//downsamplers
//				if(vae.encoder.down_blocks.get(i).spatio_temporal_scale) {
//					weights.put("encoder.down_blocks."+i+".downsamplers.0.conv.weight", vae.encoder.down_blocks.get(i).downsampler.weight);
//					weights.put("encoder.down_blocks."+i+".downsamplers.0.conv.bias", vae.encoder.down_blocks.get(i).downsampler.bias);
//				}
//				//conv_out
//				if(vae.encoder.down_blocks.get(i).shortcut) {
////					System.err.println("encoder.down_blocks."+i+".conv_out.conv1.conv.weight---"+vae.encoder.down_blocks.get(i).conv_out.conv1.weight);
//					weights.put("encoder.down_blocks."+i+".conv_out.conv1.conv.weight", vae.encoder.down_blocks.get(i).conv_out.conv1.weight);
//					weights.put("encoder.down_blocks."+i+".conv_out.conv1.conv.bias", vae.encoder.down_blocks.get(i).conv_out.conv1.bias);
//					weights.put("encoder.down_blocks."+i+".conv_out.conv2.conv.weight", vae.encoder.down_blocks.get(i).conv_out.conv2.weight);
//					weights.put("encoder.down_blocks."+i+".conv_out.conv2.conv.bias", vae.encoder.down_blocks.get(i).conv_out.conv2.bias);
//					weights.put("encoder.down_blocks."+i+".conv_out.norm3.weight", vae.encoder.down_blocks.get(i).conv_out.norm3.gamma);
//					weights.put("encoder.down_blocks."+i+".conv_out.norm3.bias", vae.encoder.down_blocks.get(i).conv_out.norm3.beta);
//					weights.put("encoder.down_blocks."+i+".conv_out.conv_shortcut.conv.weight", vae.encoder.down_blocks.get(i).conv_out.conv_shortcut.weight);
//					weights.put("encoder.down_blocks."+i+".conv_out.conv_shortcut.conv.bias", vae.encoder.down_blocks.get(i).conv_out.conv_shortcut.bias);
//				}
//			}
//			
//			//mid_block
//			for(int i=0;i<4;i++) {
//				weights.put("encoder.mid_block.resnets."+i+".conv1.conv.weight", vae.encoder.mid_block.resnets.get(i).conv1.weight);
//				weights.put("encoder.mid_block.resnets."+i+"conv1.conv.bias", vae.encoder.mid_block.resnets.get(i).conv1.bias);
//				weights.put("encoder.mid_block.resnets."+i+".conv2.conv.weight", vae.encoder.mid_block.resnets.get(i).conv2.weight);
//				weights.put("encoder.mid_block.resnets."+i+"conv2.conv.bias", vae.encoder.mid_block.resnets.get(i).conv2.bias);
//			}
//			
//			//conv_out
//			weights.put("encoder.conv_out.conv.weight", vae.encoder.conv_out.weight);
//			weights.put("encoder.conv_out.conv.bias", vae.encoder.conv_out.bias);
//			
//			 /**
//	         * decoders
//	         */
//	        // conv_in
//			weights.put("decoder.conv_in.conv.weight", vae.decoder.conv_in.weight);
//			weights.put("decoder.conv_in.conv.bias", vae.decoder.conv_in.bias);
//			
//			//mid_block
//			for(int i=0;i<4;i++) {
//				weights.put("decoder.mid_block.resnets."+i+".conv1.conv.weight", vae.decoder.mid_block.resnets.get(i).conv1.weight);
//				weights.put("decoder.mid_block.resnets."+i+"conv1.conv.bias", vae.decoder.mid_block.resnets.get(i).conv1.bias);
//				weights.put("decoder.mid_block.resnets."+i+".conv2.conv.weight", vae.decoder.mid_block.resnets.get(i).conv2.weight);
//				weights.put("decoder.mid_block.resnets."+i+"conv2.conv.bias", vae.decoder.mid_block.resnets.get(i).conv2.bias);
//			}
//			
//			//up_blocks
//			for(int i=0;i<4;i++) {
//				//conv_in
//				if(vae.decoder.up_blocks.get(i).shortcut) {
//					weights.put("decoder.up_blocks."+i+".conv_in.conv1.conv.weight", vae.decoder.up_blocks.get(i).conv_in.conv1.weight);
//					weights.put("decoder.up_blocks."+i+".conv_in.conv1.conv.bias", vae.decoder.up_blocks.get(i).conv_in.conv1.bias);
//					weights.put("decoder.up_blocks."+i+".conv_in.conv2.conv.weight", vae.decoder.up_blocks.get(i).conv_in.conv2.weight);
//					weights.put("decoder.up_blocks."+i+".conv_in.conv2.conv.bias", vae.decoder.up_blocks.get(i).conv_in.conv2.bias);
//					weights.put("decoder.up_blocks."+i+".conv_in.norm3.weight", vae.decoder.up_blocks.get(i).conv_in.norm3.gamma);
//					weights.put("decoder.up_blocks."+i+".conv_in.norm3.bias", vae.decoder.up_blocks.get(i).conv_in.norm3.beta);
//					weights.put("decoder.up_blocks."+i+".conv_in.conv_shortcut.conv.weight", vae.decoder.up_blocks.get(i).conv_in.conv_shortcut.weight);
//					weights.put("decoder.up_blocks."+i+".conv_in.conv_shortcut.conv.bias", vae.decoder.up_blocks.get(i).conv_in.conv_shortcut.bias);
//				}
//				//upsamplers
//				if(vae.decoder.up_blocks.get(i).spatio_temporal_scale) {
//					weights.put("decoder.up_blocks."+i+".upsamplers.0.conv.conv.weight", vae.decoder.up_blocks.get(i).upsampler.conv.weight);
//					weights.put("decoder.up_blocks."+i+".upsamplers.0.conv.conv.bias", vae.decoder.up_blocks.get(i).upsampler.conv.bias);
//				}
//				//resnets
//				for(int j = 0;j<layers_per_block[i+1];j++) {
//					weights.put("decoder.up_blocks."+i+".resnets."+j+".conv1.conv.weight", vae.decoder.up_blocks.get(i).resnets.get(j).conv1.weight);
//					weights.put("decoder.up_blocks."+i+".resnets."+j+".conv1.conv.bias", vae.decoder.up_blocks.get(i).resnets.get(j).conv1.bias);
//					weights.put("decoder.up_blocks."+i+".resnets."+j+".conv2.conv.weight", vae.decoder.up_blocks.get(i).resnets.get(j).conv2.weight);
//					weights.put("decoder.up_blocks."+i+".resnets."+j+".conv2.conv.bias", vae.decoder.up_blocks.get(i).resnets.get(j).conv2.bias);
//				}
//			}
//			
//			//conv_out
//			weights.put("decoder.conv_out.conv.weight", vae.decoder.conv_out.weight);
//			weights.put("decoder.conv_out.conv.bias", vae.decoder.conv_out.bias);
//			
//	        String weightPath = "D:\\models\\ltx_vae\\ltx_vae.json";
//	        List<String> igones = new ArrayList<String>();
//	        igones.add("latents_mean");
//	        igones.add("latents_std");
//	        LagJsonReader.readJsonFileBigWeightIterator_5dims(weightPath, weights, igones);
	        
//	        String save_model_path = "D:\\models\\ltx_vae\\ltx_vae.model";
//	        ModelUtils.saveModel(vae, save_model_path);
	        
			String save_model_path = "D:\\models\\ltx_vae\\ltx_vae.model";
	        ModelUtils.loadModel(vae, save_model_path);
	        
	        int N = 1;
	        int C = 3;

//			String inputPath = "D:\\models\\ltx_vae\\video.json";
//		    Map<String, Object> datas = LagJsonReader.readJsonFileSmallWeight(inputPath);
//		    Tensor input = new Tensor(N, C * num_frames, height, width, true);
//		    ModeLoaderlUtils.loadData(input, datas, "video", 5);
	        
//			String samplePath = "D:\\models\\ltx_vae\\sample.json";
//		    Map<String, Object> sampleDatas = LagJsonReader.readJsonFileSmallWeight(samplePath);
//		    Tensor sample = new Tensor(N, 128 * 3, 11, 20, true);
//		    ModeLoaderlUtils.loadData(sample, sampleDatas, "sample", 5);
//		    sample.view(1, 128, 33, 20);
//		    sample.showDM("sample");
//		    Tensor z = vae.encode(input, sample);
//		    z.showDM("z");
//		    System.err.println("----------------");
	        String videoPath = "D:\\test\\ar\\vc-465ca0c9-024d-5677-bf7a-9414540fb5f1.mp4";
	        Tensor input = VideoReaderExample.loadVideo2Tesnro(videoPath, num_frames, height, width);
	        
		    Tensor z = vae.encode(input);
//		    z.showDM("z");
		    z.view(N, 128 * 3, 11, 20);
		    
//			String zPath = "D:\\models\\ltx_vae\\z.json";
//		    Map<String, Object> zDatas = LagJsonReader.readJsonFileSmallWeight(zPath);
//		    Tensor torch_z = new Tensor(N, 128 * 3, 11, 20, true);
//		    ModeLoaderlUtils.loadData(torch_z, zDatas, "z", 5);
//		    torch_z.showDM("torch_z");
		    Tensor decoder = vae.decode(z);
		    Tensor out = new Tensor(N * num_frames, C, height, width, true);
		    vae.tensorOP.permute(decoder, out, new int[] {N, C, num_frames, height, width}, new int[] {N, num_frames, C, height, width}, new int[] {0, 2, 1, 3, 4});
		    out.showDMByOffsetRed((2 * 3 + 3) * out.height * out.width, out.height * out.width, "out");
		    //		    out.showDM("out");
		    tensor2video(out, N, num_frames, C, height, width, "D:\\models\\ltx_vae\\");
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public static void tensor2video(Tensor org, int N, int F, int C, int H, int W, String outputPath) throws Exception {
		ImageUtils utils = new ImageUtils();
        for (int b = 0; b < N; b++) {
        	List<BufferedImage> frames = new ArrayList<BufferedImage>();
        	for(int f = 0;f<F;f++) {
        		int bf = b * F + f;
        		float[] once = org.getByNumber(bf);
        		int[][] rgb = ImageUtils.color2argb(once, C, H, W);
//        		System.err.println(JsonUtils.toJson(rgb));
        		BufferedImage bufferedImage = utils.convertRGBImage(rgb);
        		frames.add(bufferedImage);
        	}
        	VideoReaderExample.writeFramesToVideo(frames, outputPath + b + "_2.mp4");
        }
	}
	
	public static void create10WDataset() {
		
		String videoPath = "/root/gpufree-data/OmegaDiT-master/vid_dataset/vc2_videos_all/";
		String labelPath = "/root/gpufree-data/OmegaDiT-master/vid_dataset/VidProM_unique_1.csv";
		String targetLabelPath = "/root/gpufree-data/OmegaDiT-master/vid_dataset/VidProM_10w.csv";
		
		try {

			CsvReader reader = CsvUtil.getReader();
			reader.setFieldSeparator(','); // 设置分隔符为逗号，默认就是逗号，这里仅为示例
            reader.setTextDelimiter('"'); //
            
	        CsvData data = reader.read(FileUtil.file(labelPath), Charset.forName("UTF-8"));
	        
	        List<CsvRow> rows = data.getRows();
	        
	        Map<String,String> datas = new HashMap<String,String>();
	        
	        for(int i = 1;i<rows.size();i++) {
	        	CsvRow row = rows.get(i);
	        	String uuid = row.get(0);
	        	String prompt = row.get(1);
	        	if(uuid.length() > 100) {
	        		System.err.println(uuid);
	        	}
//	        	if(prompt == null || prompt.equals("")) {
//            		System.err.println(uuid);
//            	}
	        	datas.put("vc-"+row.get(0), prompt);
	        }
			
			File directory = new File(videoPath);
			
			File[] files = directory.listFiles();
			
			List<List<String>> wdata = new ArrayList<List<String>>();
			wdata.add(Arrays.asList("uuid", "prompt"));
			if (files != null) {
	            for (File file : files) {
	                if (file.isFile()) {
	                	
	                	String uuid = file.getName().split("\\.")[0];
//	                	uuid = uuid.replaceFirst("vc-", "");
	                	String prompt = datas.get(uuid);
//	                	if(prompt == null || prompt.equals("")) {
//	                		System.err.println(file.getName()+"===>"+uuid);
//	                	}
	                	wdata.add(Arrays.asList(uuid, prompt));
	                }
	            }
//	            FileWriter fileWriter = new FileWriter(targetLabelPath);
	            // 使用 Hutool 的 CsvWriter 写入数据到文件
	            CsvWriter csvWriter = CsvUtil.getWriter(targetLabelPath, Charset.forName("UTF-8"));
	            csvWriter.write(wdata); // 写入数据
	            csvWriter.close(); // 关闭 writer
	            System.out.println("CSV文件写入成功！");
	        }
			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public static void createVideoLatend() {
		
		try {
			
			String labelPath = "/root/gpufree-data/OmegaDiT-master/vid_dataset/VidProM_10w.csv";
			String videoPath = "/root/gpufree-data/OmegaDiT-master/vid_dataset/vc2_videos_all/";
			String outputPath = "/root/gpufree-data/OmegaDiT-master/vid_dataset/vc2_videos_all/latend.bin";
			
			CsvReader reader = CsvUtil.getReader();
	        CsvData data = reader.read(FileUtil.file(labelPath));
	        
	        List<CsvRow> rows = data.getRows();
	        
	        List<Map<String,String>> datas = new ArrayList<Map<String,String>>();
	        
	        for(int i = 1;i<rows.size();i++) {
	        	CsvRow row = rows.get(i);
	        	Map<String,String> once = new HashMap<String, String>();
	        	once.put("filename", videoPath + row.get(0) + ".mp4");
	        	once.put("prompt", row.get(1));
	        	datas.add(once);
	        }
			
	        int num_frames = 17;
			int height = 352;
			int width = 640;
			int patch_size_t = 1;
			int patch_size = 4;
			int[] block_out_channels = new int[] {128, 256, 512, 512};
			int[] layers_per_block = new int[] {4, 3, 3, 3, 4};
			boolean[] spatio_temporal_scaling = new boolean[] {true, true, true, false};
			
			LTXVideo_VAE vae = new LTXVideo_VAE(LossType.MSE, UpdaterType.adamw, num_frames, height, width, patch_size_t, patch_size, block_out_channels, layers_per_block, spatio_temporal_scaling);
			vae.CUDNN = true;
			vae.RUN_MODEL = RunModel.EVAL;
			
			String save_model_path = "D:\\models\\ltx_vae\\ltx_vae.model";
	        ModelUtils.loadModel(vae, save_model_path);
	        
	        int N = 4;
	        int C = 3;

		    File file = new File(outputPath);
            FileOutputStream writer = new FileOutputStream(file);

	        Tensor input = new Tensor(N, C * num_frames, height, width, true);
	        
	        for(int b = 0;b<25000;b++){
	        	long start = System.nanoTime();
	        	for(int n = 0;n<N;n++) {
	        		Map<String, String> once = datas.get(b * N + n);
	        		String filename = once.get("filename");
	        		VideoReaderExample.loadVideo2Tensor(filename, num_frames, height, width, input, n);
	        	}
	        	input.hostToDevice();
	        	Tensor latend = vae.encode(input);
	        	latend.view(N, 128 * 3, 11, 20);
		        
	            JCudaDriver.cuCtxSynchronize();
	            DatasetCreater.writeTensor(latend, writer);
                System.out.println(b + "/" + 25000 + " cost["+(System.nanoTime() - start)/1e6+"ms] finish.");
	        }
	        
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public static void createVideoClip() {
		
		try {
			
			String labelPath = "D:\\dataset\\video\\VidProM_10w.csv";
			String clipDataPath = "D:\\dataset\\video\\clip.bin";
			
			CsvReader reader = CsvUtil.getReader();
	        CsvData data = reader.read(FileUtil.file(labelPath));
	        
	        List<CsvRow> rows = data.getRows();
	        
	        List<Map<String,Object>> datas = new ArrayList<Map<String,Object>>();
	        
	        for(int i = 1;i<rows.size();i++) {
	        	CsvRow row = rows.get(i);
	        	Map<String,Object> once = new HashMap<String, Object>();
	        	once.put("filename", row.get(0));
	        	once.put("prompt", row.get(1));
	        	datas.add(once);
	        }
			
	        String vocabPath = "D:\\models\\bpe_tokenizer\\vocab.json";
	        String mergesPath = "D:\\models\\bpe_tokenizer\\merges.txt";
            BPETokenizerEN bpe = new BPETokenizerEN(vocabPath, mergesPath, 49406, 49407);
    		
            int maxPositionEmbeddingsSize = 77;
            int vocabSize = 49408;
            int headNum = 12;
            int n_layers = 12;
            int textEmbedDim = 768;
            int intermediateSize = 3072;
            ClipTextModel clip = new ClipTextModel(LossType.MSE, UpdaterType.adamw, headNum, maxPositionEmbeddingsSize, vocabSize, textEmbedDim, maxPositionEmbeddingsSize, intermediateSize, n_layers);
            clip.CUDNN = true;
            clip.time = maxPositionEmbeddingsSize;
            clip.RUN_MODEL = RunModel.EVAL;
            String clipWeight = "D:\\models\\CLIP-GmP-ViT-L-14\\CLIP-GmP-ViT-L-14.json";
            ModeLoaderlUtils.loadWeight(LagJsonReader.readJsonFileBigWeightIterator(clipWeight), clip, "", false);
            
            File clipFile = new File(clipDataPath);
            FileOutputStream clipWriter = new FileOutputStream(clipFile);
            
            int N = 1000;
            
            int[][] indexs = MathUtils.orderInts(100000, N);

            Tensor label = new Tensor(N * 77, 1, 1, 1, true);
            for(int b = 0;b<100;b++){
	        	DatasetCreater.loadLabels(bpe, datas, "prompt", indexs[b], label, maxPositionEmbeddingsSize, N);
//	        	label.showDM("label");
				Tensor condInput = clip.get_full_clip_prompt_embeds(label);
				JCudaDriver.cuCtxSynchronize();
				DatasetCreater.writeTensor(condInput, clipWriter);
				System.out.println(b + "/" + 100 + " finish.");
	        }
	        
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public static void main(String[] args) {
		try {
//			vae();
//			create10WDataset();
//			createVideoLatend();
			createVideoClip();
		} catch (Exception e) {
	        // TODO: handle exception
	        e.printStackTrace();
	    } finally {
	        // TODO: handle finally clause
	        CUDAMemoryManager.free();
	    }
	}
	
}



