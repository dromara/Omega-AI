package com.omega.example.opensora.utils;

import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import com.omega.common.utils.ImageUtils;
import com.omega.common.utils.JsonUtils;
import com.omega.engine.gpu.CUDAMemoryManager;
import com.omega.engine.loss.LossType;
import com.omega.engine.nn.network.vae.LTXVideo_VAE;
import com.omega.engine.tensor.Tensor;
import com.omega.engine.updater.UpdaterType;
import com.omega.example.common.ModeLoaderlUtils;
import com.omega.example.transformer.utils.LagJsonReader;
import com.omega.example.transformer.utils.ModelUtils;

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
	        String videoPath = "D:\\dataset\\wfvae\\4105473_scene-0_cut-border.mp4";
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
	
	public static void main(String[] args) {
		try {
			vae();
		} catch (Exception e) {
	        // TODO: handle exception
	        e.printStackTrace();
	    } finally {
	        // TODO: handle finally clause
	        CUDAMemoryManager.free();
	    }
	}
	
}



