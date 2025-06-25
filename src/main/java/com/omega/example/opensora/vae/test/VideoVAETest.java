package com.omega.example.opensora.vae.test;

import java.util.Map;

import com.omega.common.utils.MatrixUtils;
import com.omega.engine.gpu.CUDAMemoryManager;
import com.omega.engine.loss.LossType;
import com.omega.engine.nn.layer.opensora.vae.modules.AttentionBlock3D;
import com.omega.engine.nn.layer.opensora.vae.modules.Downsample2D;
import com.omega.engine.nn.layer.opensora.vae.modules.Downsample3D;
import com.omega.engine.nn.layer.opensora.vae.modules.Resnet3DBlock;
import com.omega.engine.nn.layer.opensora.vae.modules.Upsample2D;
import com.omega.engine.nn.layer.opensora.vae.modules.Upsample3D;
import com.omega.engine.nn.network.vae.OpenSoraVAE;
import com.omega.engine.nn.network.vqgan.Opensora_LPIPS;
import com.omega.engine.tensor.Tensor;
import com.omega.engine.updater.UpdaterType;
import com.omega.example.clip.utils.ClipModelUtils;
import com.omega.example.transformer.utils.LagJsonReader;
import com.omega.example.vae.test.LPIPSTest;

public class VideoVAETest {
    
	public static void loadWeight(Map<String, Object> weightMap, OpenSoraVAE network, boolean showLayers) {
        if (showLayers) {
            for (String key : weightMap.keySet()) {
                System.out.println(key);
            }
        }
        
        /**
         * encoder block1
         */
        ClipModelUtils.loadData(network.encoder.convIn.weight, weightMap, "encoder.conv_in.conv.weight", 5);
        ClipModelUtils.loadData(network.encoder.convIn.bias, weightMap, "encoder.conv_in.conv.bias");
        
        for(int i = 0;i<4;i++){
        	Resnet3DBlock block = (Resnet3DBlock) network.encoder.downBlock.get(i * 2 + 0);
        	block.norm1.norm.gamma = ClipModelUtils.loadData(block.norm1.norm.gamma, weightMap, 1, "encoder.down."+i+".block.0.norm1.weight");
        	block.norm1.norm.beta = ClipModelUtils.loadData(block.norm1.norm.beta, weightMap, 1, "encoder.down."+i+".block.0.norm1.bias");
        	ClipModelUtils.loadData(block.conv1.weight, weightMap, "encoder.down."+i+".block.0.conv1.conv.weight", 5);
            ClipModelUtils.loadData(block.conv1.bias, weightMap, "encoder.down."+i+".block.0.conv1.conv.bias");
        	block.norm2.norm.gamma = ClipModelUtils.loadData(block.norm2.norm.gamma, weightMap, 1, "encoder.down."+i+".block.0.norm2.weight");
        	block.norm2.norm.beta = ClipModelUtils.loadData(block.norm2.norm.beta, weightMap, 1, "encoder.down."+i+".block.0.norm2.bias");
        	ClipModelUtils.loadData(block.conv2.weight, weightMap, "encoder.down."+i+".block.0.conv2.conv.weight", 5);
            ClipModelUtils.loadData(block.conv2.bias, weightMap, "encoder.down."+i+".block.0.conv2.conv.bias");
            if(block.shortcut != null) {
            	ClipModelUtils.loadData(block.shortcut.weight, weightMap, "encoder.down."+i+".block.0.nin_shortcut.conv.weight", 5);
                ClipModelUtils.loadData(block.shortcut.bias, weightMap, "encoder.down."+i+".block.0.nin_shortcut.conv.bias");
            }
            if(i < 3) {
            	if(network.encoder.downBlock.get(i * 2 + 1) instanceof Downsample2D) {
                	Downsample2D down = (Downsample2D) network.encoder.downBlock.get(i * 2 + 1);
                	ClipModelUtils.loadData(down.conv.weight, weightMap, "encoder.down."+i+".downsample.conv.weight");
                    ClipModelUtils.loadData(down.conv.bias, weightMap, "encoder.down."+i+".downsample.conv.bias");
                }else {
                	Downsample3D down = (Downsample3D) network.encoder.downBlock.get(i * 2 + 1);
                	ClipModelUtils.loadData(down.conv3d.weight, weightMap, "encoder.down."+i+".downsample.conv.conv.weight", 5);
                    ClipModelUtils.loadData(down.conv3d.bias, weightMap, "encoder.down."+i+".downsample.conv.conv.bias");
                }
            }
        }
        
        /**
         * encoder mid
         */
        Resnet3DBlock mb1 = (Resnet3DBlock) network.encoder.midBlock.get(0);
        mb1.norm1.norm.gamma = ClipModelUtils.loadData(mb1.norm1.norm.gamma, weightMap, 1, "encoder.mid.block_1.norm1.weight");
        mb1.norm1.norm.beta = ClipModelUtils.loadData(mb1.norm1.norm.beta, weightMap, 1, "encoder.mid.block_1.norm1.bias");
        ClipModelUtils.loadData(mb1.conv1.weight, weightMap, "encoder.mid.block_1.conv1.conv.weight", 5);
        ClipModelUtils.loadData(mb1.conv1.bias, weightMap, "encoder.mid.block_1.conv1.conv.bias");
        mb1.norm2.norm.gamma = ClipModelUtils.loadData(mb1.norm2.norm.gamma, weightMap, 1, "encoder.mid.block_1.norm2.weight");
        mb1.norm2.norm.beta = ClipModelUtils.loadData(mb1.norm2.norm.beta, weightMap, 1, "encoder.mid.block_1.norm2.bias");
        ClipModelUtils.loadData(mb1.conv2.weight, weightMap, "encoder.mid.block_1.conv2.conv.weight", 5);
        ClipModelUtils.loadData(mb1.conv2.bias, weightMap, "encoder.mid.block_1.conv2.conv.bias");
        
        /**
         * encoder mid attn
         */
        AttentionBlock3D attn = (AttentionBlock3D) network.encoder.midBlock.get(1);
        attn.norm.norm.gamma = ClipModelUtils.loadData(attn.norm.norm.gamma, weightMap, 1, "encoder.mid.attn_1.norm.weight");
        attn.norm.norm.beta = ClipModelUtils.loadData(attn.norm.norm.beta, weightMap, 1, "encoder.mid.attn_1.norm.bias");
        ClipModelUtils.loadData(attn.qLinerLayer.weight, weightMap, "encoder.mid.attn_1.q.conv.weight", 5);
        ClipModelUtils.loadData(attn.qLinerLayer.bias, weightMap, "encoder.mid.attn_1.q.conv.bias");
        ClipModelUtils.loadData(attn.kLinerLayer.weight, weightMap, "encoder.mid.attn_1.k.conv.weight", 5);
        ClipModelUtils.loadData(attn.kLinerLayer.bias, weightMap, "encoder.mid.attn_1.k.conv.bias");
        ClipModelUtils.loadData(attn.vLinerLayer.weight, weightMap, "encoder.mid.attn_1.v.conv.weight", 5);
        ClipModelUtils.loadData(attn.vLinerLayer.bias, weightMap, "encoder.mid.attn_1.v.conv.bias");
        ClipModelUtils.loadData(attn.oLinerLayer.weight, weightMap, "encoder.mid.attn_1.proj_out.conv.weight", 5);
        ClipModelUtils.loadData(attn.oLinerLayer.bias, weightMap, "encoder.mid.attn_1.proj_out.conv.bias");
        
        /**
         * encoder block
         */
        Resnet3DBlock mb2 = (Resnet3DBlock) network.encoder.midBlock.get(2);
        mb2.norm1.norm.gamma = ClipModelUtils.loadData(mb2.norm1.norm.gamma, weightMap, 1, "encoder.mid.block_2.norm1.weight");
        mb2.norm1.norm.beta = ClipModelUtils.loadData(mb2.norm1.norm.beta, weightMap, 1, "encoder.mid.block_2.norm1.bias");
        ClipModelUtils.loadData(mb2.conv1.weight, weightMap, "encoder.mid.block_2.conv1.conv.weight", 5);
        ClipModelUtils.loadData(mb2.conv1.bias, weightMap, "encoder.mid.block_2.conv1.conv.bias");
        mb2.norm2.norm.gamma = ClipModelUtils.loadData(mb2.norm2.norm.gamma, weightMap, 1, "encoder.mid.block_2.norm2.weight");
        mb2.norm2.norm.beta = ClipModelUtils.loadData(mb2.norm2.norm.beta, weightMap, 1, "encoder.mid.block_2.norm2.bias");
        ClipModelUtils.loadData(mb2.conv2.weight, weightMap, "encoder.mid.block_2.conv2.conv.weight", 5);
        ClipModelUtils.loadData(mb2.conv2.bias, weightMap, "encoder.mid.block_2.conv2.conv.bias");
        
        network.encoder.convNormOut.norm.gamma = ClipModelUtils.loadData(network.encoder.convNormOut.norm.gamma, weightMap, 1, "encoder.norm_out.weight");
        network.encoder.convNormOut.norm.beta = ClipModelUtils.loadData(network.encoder.convNormOut.norm.beta, weightMap, 1, "encoder.norm_out.bias");
        ClipModelUtils.loadData(network.encoder.convOut.weight, weightMap, "encoder.conv_out.conv.weight", 5);
        ClipModelUtils.loadData(network.encoder.convOut.bias, weightMap, "encoder.conv_out.conv.bias");
        
        /**
         * decoder convIn
         */
        ClipModelUtils.loadData(network.decoder.convIn.weight, weightMap, "decoder.conv_in.conv.weight", 5);
        ClipModelUtils.loadData(network.decoder.convIn.bias, weightMap, "decoder.conv_in.conv.bias");
        
        Resnet3DBlock dmb1 = (Resnet3DBlock) network.decoder.midBlock.get(0);
        dmb1.norm1.norm.gamma = ClipModelUtils.loadData(dmb1.norm1.norm.gamma, weightMap, 1, "decoder.mid.block_1.norm1.weight");
        dmb1.norm1.norm.beta = ClipModelUtils.loadData(dmb1.norm1.norm.beta, weightMap, 1, "decoder.mid.block_1.norm1.bias");
        ClipModelUtils.loadData(dmb1.conv1.weight, weightMap, "decoder.mid.block_1.conv1.conv.weight", 5);
        ClipModelUtils.loadData(dmb1.conv1.bias, weightMap, "decoder.mid.block_1.conv1.conv.bias");
        dmb1.norm2.norm.gamma = ClipModelUtils.loadData(dmb1.norm2.norm.gamma, weightMap, 1, "decoder.mid.block_1.norm2.weight");
        dmb1.norm2.norm.beta = ClipModelUtils.loadData(dmb1.norm2.norm.beta, weightMap, 1, "decoder.mid.block_1.norm2.bias");
        ClipModelUtils.loadData(dmb1.conv2.weight, weightMap, "decoder.mid.block_1.conv2.conv.weight", 5);
        ClipModelUtils.loadData(dmb1.conv2.bias, weightMap, "decoder.mid.block_1.conv2.conv.bias");
        /**
         * decoder mid attn
         */
        AttentionBlock3D dattn = (AttentionBlock3D) network.decoder.midBlock.get(1);
        dattn.norm.norm.gamma = ClipModelUtils.loadData(dattn.norm.norm.gamma, weightMap, 1, "decoder.mid.attn_1.norm.weight");
        dattn.norm.norm.beta = ClipModelUtils.loadData(dattn.norm.norm.beta, weightMap, 1, "decoder.mid.attn_1.norm.bias");
        ClipModelUtils.loadData(dattn.qLinerLayer.weight, weightMap, "decoder.mid.attn_1.q.conv.weight", 5);
        ClipModelUtils.loadData(dattn.qLinerLayer.bias, weightMap, "decoder.mid.attn_1.q.conv.bias");
        ClipModelUtils.loadData(dattn.kLinerLayer.weight, weightMap, "decoder.mid.attn_1.k.conv.weight", 5);
        ClipModelUtils.loadData(dattn.kLinerLayer.bias, weightMap, "decoder.mid.attn_1.k.conv.bias");
        ClipModelUtils.loadData(dattn.vLinerLayer.weight, weightMap, "decoder.mid.attn_1.v.conv.weight", 5);
        ClipModelUtils.loadData(dattn.vLinerLayer.bias, weightMap, "decoder.mid.attn_1.v.conv.bias");
        ClipModelUtils.loadData(dattn.oLinerLayer.weight, weightMap, "decoder.mid.attn_1.proj_out.conv.weight", 5);
        ClipModelUtils.loadData(dattn.oLinerLayer.bias, weightMap, "decoder.mid.attn_1.proj_out.conv.bias");
        
        Resnet3DBlock dmb2 = (Resnet3DBlock) network.decoder.midBlock.get(2);
        dmb2.norm1.norm.gamma = ClipModelUtils.loadData(dmb2.norm1.norm.gamma, weightMap, 1, "decoder.mid.block_2.norm1.weight");
        dmb2.norm1.norm.beta = ClipModelUtils.loadData(dmb2.norm1.norm.beta, weightMap, 1, "decoder.mid.block_2.norm1.bias");
        ClipModelUtils.loadData(dmb2.conv1.weight, weightMap, "decoder.mid.block_2.conv1.conv.weight", 5);
        ClipModelUtils.loadData(dmb2.conv1.bias, weightMap, "decoder.mid.block_2.conv1.conv.bias");
        dmb2.norm2.norm.gamma = ClipModelUtils.loadData(dmb2.norm2.norm.gamma, weightMap, 1, "decoder.mid.block_2.norm2.weight");
        dmb2.norm2.norm.beta = ClipModelUtils.loadData(dmb2.norm2.norm.beta, weightMap, 1, "decoder.mid.block_2.norm2.bias");
        ClipModelUtils.loadData(dmb2.conv2.weight, weightMap, "decoder.mid.block_2.conv2.conv.weight", 5);
        ClipModelUtils.loadData(dmb2.conv2.bias, weightMap, "decoder.mid.block_2.conv2.conv.bias");
        
        for(int i = 0;i<4;i++) {
        	int idx = 3 - i;
        	Resnet3DBlock block = (Resnet3DBlock) network.decoder.upBlock.get(idx * 3 + 0);
        	block.norm1.norm.gamma = ClipModelUtils.loadData(block.norm1.norm.gamma, weightMap, 1, "decoder.up."+i+".block.0.norm1.weight");
        	block.norm1.norm.beta = ClipModelUtils.loadData(block.norm1.norm.beta, weightMap, 1, "decoder.up."+i+".block.0.norm1.bias");
        	ClipModelUtils.loadData(block.conv1.weight, weightMap, "decoder.up."+i+".block.0.conv1.conv.weight", 5);
            ClipModelUtils.loadData(block.conv1.bias, weightMap, "decoder.up."+i+".block.0.conv1.conv.bias");
        	block.norm2.norm.gamma = ClipModelUtils.loadData(block.norm2.norm.gamma, weightMap, 1, "decoder.up."+i+".block.0.norm2.weight");
        	block.norm2.norm.beta = ClipModelUtils.loadData(block.norm2.norm.beta, weightMap, 1, "decoder.up."+i+".block.0.norm2.bias");
        	ClipModelUtils.loadData(block.conv2.weight, weightMap, "decoder.up."+i+".block.0.conv2.conv.weight", 5);
            ClipModelUtils.loadData(block.conv2.bias, weightMap, "decoder.up."+i+".block.0.conv2.conv.bias");
            if(block.shortcut != null) {
            	ClipModelUtils.loadData(block.shortcut.weight, weightMap, "decoder.up."+i+".block.0.nin_shortcut.conv.weight", 5);
                ClipModelUtils.loadData(block.shortcut.bias, weightMap, "decoder.up."+i+".block.0.nin_shortcut.conv.bias");
            }
            Resnet3DBlock block2 = (Resnet3DBlock) network.decoder.upBlock.get(idx * 3 + 1);
            block2.norm1.norm.gamma = ClipModelUtils.loadData(block2.norm1.norm.gamma, weightMap, 1, "decoder.up."+i+".block.1.norm1.weight");
            block2.norm1.norm.beta = ClipModelUtils.loadData(block2.norm1.norm.beta, weightMap, 1, "decoder.up."+i+".block.1.norm1.bias");
        	ClipModelUtils.loadData(block2.conv1.weight, weightMap, "decoder.up."+i+".block.1.conv1.conv.weight", 5);
            ClipModelUtils.loadData(block2.conv1.bias, weightMap, "decoder.up."+i+".block.1.conv1.conv.bias");
            block2.norm2.norm.gamma = ClipModelUtils.loadData(block2.norm2.norm.gamma, weightMap, 1, "decoder.up."+i+".block.1.norm2.weight");
            block2.norm2.norm.beta = ClipModelUtils.loadData(block2.norm2.norm.beta, weightMap, 1, "decoder.up."+i+".block.1.norm2.bias");
        	ClipModelUtils.loadData(block2.conv2.weight, weightMap, "decoder.up."+i+".block.1.conv2.conv.weight", 5);
            ClipModelUtils.loadData(block2.conv2.bias, weightMap, "decoder.up."+i+".block.1.conv2.conv.bias");
            if(i > 0) {
            	if(network.decoder.upBlock.get(idx * 3 + 2) instanceof Upsample2D) {
            		Upsample2D up = (Upsample2D) network.decoder.upBlock.get(idx * 3 + 2);
                	ClipModelUtils.loadData(up.conv.weight, weightMap, "decoder.up."+i+".upsample.conv.weight");
                    ClipModelUtils.loadData(up.conv.bias, weightMap, "decoder.up."+i+".upsample.conv.bias");
            	}else {
            		Upsample3D up = (Upsample3D) network.decoder.upBlock.get(idx * 3 + 2);
            		ClipModelUtils.loadData(up.conv3d.conv1.weight, weightMap, "decoder.up."+i+".upsample.conv3d.conv1.conv.weight", 5);
                    ClipModelUtils.loadData(up.conv3d.conv1.bias, weightMap, "decoder.up."+i+".upsample.conv3d.conv1.conv.bias");
                    up.conv3d.norm1.norm.gamma = ClipModelUtils.loadData(up.conv3d.norm1.norm.gamma, weightMap, 1, "decoder.up."+i+".upsample.conv3d.norm1.weight");
                    up.conv3d.norm1.norm.beta = ClipModelUtils.loadData(up.conv3d.norm1.norm.beta, weightMap, 1, "decoder.up."+i+".upsample.conv3d.norm1.bias");
            		ClipModelUtils.loadData(up.conv3d.conv2.weight, weightMap, "decoder.up."+i+".upsample.conv3d.conv2.conv.weight", 5);
                    ClipModelUtils.loadData(up.conv3d.conv2.bias, weightMap, "decoder.up."+i+".upsample.conv3d.conv2.conv.bias");
                    up.conv3d.norm2.norm.gamma = ClipModelUtils.loadData(up.conv3d.norm2.norm.gamma, weightMap, 1, "decoder.up."+i+".upsample.conv3d.norm2.weight");
                    up.conv3d.norm2.norm.beta = ClipModelUtils.loadData(up.conv3d.norm2.norm.beta, weightMap, 1, "decoder.up."+i+".upsample.conv3d.norm2.bias");
            	}
            }
            
        }

        network.decoder.convNormOut.norm.gamma = ClipModelUtils.loadData(network.decoder.convNormOut.norm.gamma, weightMap, 1, "decoder.norm_out.weight");
        network.decoder.convNormOut.norm.beta = ClipModelUtils.loadData(network.encoder.convNormOut.norm.beta, weightMap, 1, "decoder.norm_out.bias");
        ClipModelUtils.loadData(network.decoder.convOut.weight, weightMap, "decoder.conv_out.weight", 5);
        ClipModelUtils.loadData(network.decoder.convOut.bias, weightMap, "decoder.conv_out.bias");
	}
	
    public static void video_vae_train() {
        try {
            
        	int batchSize = 2;
        	int channel = 3;
        	int numFrames = 17;
            int imageSize = 32;
            int latendDim = 8;
            int num_res_blocks = 1;
            int[] ch_mult = new int[]{1, 2, 2, 4};
            int ch = 32;
        	
        	OpenSoraVAE network = new OpenSoraVAE(LossType.MSE, UpdaterType.adamw, latendDim, numFrames, imageSize, ch_mult, ch, num_res_blocks);
        	network.CUDNN = true;
        	
        	String inputPath = "H:\\model\\input.json";
        	Map<String, Object> datas = LagJsonReader.readJsonFileSmallWeight(inputPath);
        	Tensor input = new Tensor(batchSize, channel * numFrames, imageSize, imageSize, true);
        	ClipModelUtils.loadData(input, datas, "x", 5);
        	
        	String inputZPath = "H:\\model\\input_z.json";
        	Map<String, Object> zdatas = LagJsonReader.readJsonFileSmallWeight(inputZPath);
        	Tensor input_z = new Tensor(batchSize, 40, 4, 4, true);
        	ClipModelUtils.loadData(input_z, zdatas, "z", 5);
        	
        	input.showDM("x:");
        	
        	String path = "H:\\model\\opensora_vae.json";
        	loadWeight(LagJsonReader.readJsonFileSmallWeight(path), network, true);
        	
        	network.forward(input);
        	network.getOutput().showDM();
        	
        	Opensora_LPIPS lpips = new Opensora_LPIPS(LossType.MSE, UpdaterType.adamw, imageSize);
            String lpipsWeight = "H:\\model\\opensora_lpips.json";
            LPIPSTest.loadLPIPSWeight(LagJsonReader.readJsonFileSmallWeight(lpipsWeight), lpips, true);

        	float totalLoss = network.totalLoss(network.getOutput(), input, lpips);

        	System.err.println("totalLoss:"+totalLoss);
        	network.getOutput().showDM();
        	
        	network.back();
        	
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public static void main(String[] args) {
        try {
            video_vae_train();
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        } finally {
            // TODO: handle finally clause
            CUDAMemoryManager.free();
        }
    }
}
