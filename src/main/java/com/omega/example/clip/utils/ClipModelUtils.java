package com.omega.example.clip.utils;

import java.util.List;
import java.util.Map;

import com.omega.common.utils.JsonUtils;
import com.omega.common.utils.MatrixUtils;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.clip.bert.BertLayer;
import com.omega.engine.nn.layer.opensora.wfvae.decoder.WFDecoder;
import com.omega.engine.nn.layer.opensora.wfvae.encoder.WFEncoder;
import com.omega.engine.nn.layer.opensora.wfvae.modules.WFResnet2DBlock;
import com.omega.engine.nn.layer.opensora.wfvae.modules.WFResnet3DBlock;
import com.omega.engine.nn.layer.sd_vae.moudles.SDVAEAttentionLayer;
import com.omega.engine.nn.layer.sd_vae.moudles.SDVAEDownsample;
import com.omega.engine.nn.layer.sd_vae.moudles.SDVAEResidual;
import com.omega.engine.nn.layer.sd_vae.moudles.SDVAEUpsample;
import com.omega.engine.nn.layer.va_vae.VAVAEAttentionLayer;
import com.omega.engine.nn.network.ClipText;
import com.omega.engine.nn.network.ClipTextModel;
import com.omega.engine.nn.network.ClipVision;
import com.omega.engine.nn.network.vae.SD_VAE;
import com.omega.engine.nn.network.vae.VA_VAE;
import com.omega.engine.nn.network.vae.WFVAE;
import com.omega.engine.tensor.Tensor;

public class ClipModelUtils {
	
    public static void loadWeight(Map<String, Object> weightMap, WFVAE network, boolean showLayers) {
        if (showLayers) {
            for (String key : weightMap.keySet()) {
                System.out.println(key);
            }
        }
        WFEncoder block = network.encoder;
        ClipModelUtils.loadData(block.down1.conv.conv.weight, weightMap, "vae.encoder.down1.0.weight");
        ClipModelUtils.loadData(block.down1.conv.conv.bias, weightMap, "vae.encoder.down1.0.bias");
        for(int i = 0;i<block.getNum_resblocks();i++) {
            WFResnet2DBlock b2d = block.down1.resBlocks.get(i);
            b2d.norm1.norm.gamma = ClipModelUtils.loadData(b2d.norm1.norm.gamma, weightMap, 1, "vae.encoder.down1."+(i+1)+".norm1.norm.weight");
            b2d.norm1.norm.beta = ClipModelUtils.loadData(b2d.norm1.norm.beta, weightMap, 1, "vae.encoder.down1."+(i+1)+".norm1.norm.bias");
            ClipModelUtils.loadData(b2d.conv1.weight, weightMap, "vae.encoder.down1."+(i+1)+".conv1.weight");
            ClipModelUtils.loadData(b2d.conv1.bias, weightMap, "vae.encoder.down1."+(i+1)+".conv1.bias");
            b2d.norm2.norm.gamma = ClipModelUtils.loadData(b2d.norm2.norm.gamma, weightMap, 1, "vae.encoder.down1."+(i+1)+".norm2.norm.weight");
            b2d.norm2.norm.beta = ClipModelUtils.loadData(b2d.norm2.norm.beta, weightMap, 1, "vae.encoder.down1."+(i+1)+".norm2.norm.bias");
            ClipModelUtils.loadData(b2d.conv2.weight, weightMap, "vae.encoder.down1."+(i+1)+".conv2.weight");
            ClipModelUtils.loadData(b2d.conv2.bias, weightMap, "vae.encoder.down1."+(i+1)+".conv2.bias");
        }
        ClipModelUtils.loadData(block.down1.downsample.conv.weight, weightMap, "vae.encoder.down1.3.conv.weight");
        ClipModelUtils.loadData(block.down1.downsample.conv.bias, weightMap, "vae.encoder.down1.3.conv.bias");

        /**
         * encoder down2
         */
        ClipModelUtils.loadData(block.down2.conv.conv.weight, weightMap, "vae.encoder.down2.0.weight");
        ClipModelUtils.loadData(block.down2.conv.conv.bias, weightMap, "vae.encoder.down2.0.bias");
        for(int i = 0;i<block.getNum_resblocks();i++) {
            WFResnet3DBlock b3d = block.down2.resBlocks.get(i);
            b3d.norm1.norm.gamma = ClipModelUtils.loadData(b3d.norm1.norm.gamma, weightMap, 1, "vae.encoder.down2."+(i+1)+".norm1.norm.weight");
            b3d.norm1.norm.beta = ClipModelUtils.loadData(b3d.norm1.norm.beta, weightMap, 1, "vae.encoder.down2."+(i+1)+".norm1.norm.bias");
            ClipModelUtils.loadData(b3d.conv1.weight, weightMap, "vae.encoder.down2."+(i+1)+".conv1.conv.weight", 5);
            ClipModelUtils.loadData(b3d.conv1.bias, weightMap, "vae.encoder.down2."+(i+1)+".conv1.conv.bias");
            b3d.norm2.norm.gamma = ClipModelUtils.loadData(b3d.norm2.norm.gamma, weightMap, 1, "vae.encoder.down2."+(i+1)+".norm2.norm.weight");
            b3d.norm2.norm.beta = ClipModelUtils.loadData(b3d.norm2.norm.beta, weightMap, 1, "vae.encoder.down2."+(i+1)+".norm2.norm.bias");
            ClipModelUtils.loadData(b3d.conv2.weight, weightMap, "vae.encoder.down2."+(i+1)+".conv2.conv.weight", 5);
            ClipModelUtils.loadData(b3d.conv2.bias, weightMap, "vae.encoder.down2."+(i+1)+".conv2.conv.bias");
        }
        ClipModelUtils.loadData(block.down2.downsample3d.conv.weight, weightMap, "vae.encoder.down2.3.conv.conv.weight", 5);
        ClipModelUtils.loadData(block.down2.downsample3d.conv.bias, weightMap, "vae.encoder.down2.3.conv.conv.bias");

        ClipModelUtils.loadData(block.connect_l1.conv.weight, weightMap, "vae.encoder.connect_l1.weight");
        ClipModelUtils.loadData(block.connect_l1.conv.bias, weightMap, "vae.encoder.connect_l1.bias");
        ClipModelUtils.loadData(block.connect_l2.conv.weight, weightMap, "vae.encoder.connect_l2.weight");
        ClipModelUtils.loadData(block.connect_l2.conv.bias, weightMap, "vae.encoder.connect_l2.bias");

        block.mid.block1.norm1.norm.gamma = ClipModelUtils.loadData(block.mid.block1.norm1.norm.gamma, weightMap, 1, "vae.encoder.mid.0.norm1.norm.weight");
        block.mid.block1.norm1.norm.beta = ClipModelUtils.loadData(block.mid.block1.norm1.norm.beta, weightMap, 1, "vae.encoder.mid.0.norm1.norm.bias");
        ClipModelUtils.loadData(block.mid.block1.conv1.weight, weightMap, "vae.encoder.mid.0.conv1.conv.weight", 5);
        ClipModelUtils.loadData(block.mid.block1.conv1.bias, weightMap, "vae.encoder.mid.0.conv1.conv.bias");
        block.mid.block1.norm2.norm.gamma = ClipModelUtils.loadData(block.mid.block1.norm2.norm.gamma, weightMap, 1, "vae.encoder.mid.0.norm2.norm.weight");
        block.mid.block1.norm2.norm.beta = ClipModelUtils.loadData(block.mid.block1.norm2.norm.beta, weightMap, 1, "vae.encoder.mid.0.norm2.norm.bias");
        ClipModelUtils.loadData(block.mid.block1.conv2.weight, weightMap, "vae.encoder.mid.0.conv2.conv.weight", 5);
        ClipModelUtils.loadData(block.mid.block1.conv2.bias, weightMap, "vae.encoder.mid.0.conv2.conv.bias");
        ClipModelUtils.loadData(block.mid.block1.shortcut.weight, weightMap, "vae.encoder.mid.0.nin_shortcut.conv.weight", 5);
        ClipModelUtils.loadData(block.mid.block1.shortcut.bias, weightMap, "vae.encoder.mid.0.nin_shortcut.conv.bias");

        block.mid.attn.norm.norm.gamma = ClipModelUtils.loadData(block.mid.attn.norm.norm.gamma, weightMap, 1, "vae.encoder.mid.1.norm.norm.weight");
        block.mid.attn.norm.norm.beta = ClipModelUtils.loadData(block.mid.attn.norm.norm.beta, weightMap, 1, "vae.encoder.mid.1.norm.norm.bias");
        ClipModelUtils.loadData(block.mid.attn.qLinerLayer.weight, weightMap, "vae.encoder.mid.1.q.conv.weight", 5);
        ClipModelUtils.loadData(block.mid.attn.kLinerLayer.weight, weightMap, "vae.encoder.mid.1.k.conv.weight", 5);
        ClipModelUtils.loadData(block.mid.attn.vLinerLayer.weight, weightMap, "vae.encoder.mid.1.v.conv.weight", 5);
        ClipModelUtils.loadData(block.mid.attn.oLinerLayer.weight, weightMap, "vae.encoder.mid.1.proj_out.conv.weight", 5);

        block.mid.block2.norm1.norm.gamma = ClipModelUtils.loadData(block.mid.block2.norm1.norm.gamma, weightMap, 1, "vae.encoder.mid.2.norm1.norm.weight");
        block.mid.block2.norm1.norm.beta = ClipModelUtils.loadData(block.mid.block2.norm1.norm.beta, weightMap, 1, "vae.encoder.mid.2.norm1.norm.bias");
        ClipModelUtils.loadData(block.mid.block2.conv1.weight, weightMap, "vae.encoder.mid.2.conv1.conv.weight", 5);
        ClipModelUtils.loadData(block.mid.block2.conv1.bias, weightMap, "vae.encoder.mid.2.conv1.conv.bias");
        block.mid.block2.norm2.norm.gamma = ClipModelUtils.loadData(block.mid.block2.norm2.norm.gamma, weightMap, 1, "vae.encoder.mid.2.norm2.norm.weight");
        block.mid.block2.norm2.norm.beta = ClipModelUtils.loadData(block.mid.block2.norm2.norm.beta, weightMap, 1, "vae.encoder.mid.2.norm2.norm.bias");
        ClipModelUtils.loadData(block.mid.block2.conv2.weight, weightMap, "vae.encoder.mid.2.conv2.conv.weight", 5);
        ClipModelUtils.loadData(block.mid.block2.conv2.bias, weightMap, "vae.encoder.mid.2.conv2.conv.bias");

        block.norm_out.norm.gamma = ClipModelUtils.loadData(block.norm_out.norm.gamma, weightMap, 1, "vae.encoder.norm_out.norm.weight");
        block.norm_out.norm.beta = ClipModelUtils.loadData(block.norm_out.norm.beta, weightMap, 1, "vae.encoder.norm_out.norm.bias");
        ClipModelUtils.loadData(block.conv_out.weight, weightMap, "vae.encoder.conv_out.conv.weight", 5);
        ClipModelUtils.loadData(block.conv_out.bias, weightMap, "vae.encoder.conv_out.conv.bias");

        // deocder
        WFDecoder decoder = network.decoder;
        ClipModelUtils.loadData(decoder.conv_in.weight, weightMap, "vae.decoder.conv_in.conv.weight", 5);
        ClipModelUtils.loadData(decoder.conv_in.bias, weightMap, "vae.decoder.conv_in.conv.bias");
        /**
         * mid blocks
         */
        decoder.mid.block1.norm1.norm.gamma = ClipModelUtils.loadData(decoder.mid.block1.norm1.norm.gamma, weightMap, 1, "vae.decoder.mid.0.norm1.norm.weight");
        decoder.mid.block1.norm1.norm.beta = ClipModelUtils.loadData(decoder.mid.block1.norm1.norm.beta, weightMap, 1, "vae.decoder.mid.0.norm1.norm.bias");
        ClipModelUtils.loadData(decoder.mid.block1.conv1.weight, weightMap, "vae.decoder.mid.0.conv1.conv.weight", 5);
        ClipModelUtils.loadData(decoder.mid.block1.conv1.bias, weightMap, "vae.decoder.mid.0.conv1.conv.bias");
        decoder.mid.block1.norm2.norm.gamma = ClipModelUtils.loadData(decoder.mid.block1.norm2.norm.gamma, weightMap, 1, "vae.decoder.mid.0.norm2.norm.weight");
        decoder.mid.block1.norm2.norm.beta = ClipModelUtils.loadData(decoder.mid.block1.norm2.norm.beta, weightMap, 1, "vae.decoder.mid.0.norm2.norm.bias");
        ClipModelUtils.loadData(decoder.mid.block1.conv2.weight, weightMap, "vae.decoder.mid.0.conv2.conv.weight", 5);
        ClipModelUtils.loadData(decoder.mid.block1.conv2.bias, weightMap, "vae.decoder.mid.0.conv2.conv.bias");

        decoder.mid.attn.norm.norm.gamma = ClipModelUtils.loadData(decoder.mid.attn.norm.norm.gamma, weightMap, 1, "vae.decoder.mid.1.norm.norm.weight");
        decoder.mid.attn.norm.norm.beta = ClipModelUtils.loadData(decoder.mid.attn.norm.norm.beta, weightMap, 1, "vae.decoder.mid.1.norm.norm.bias");
        ClipModelUtils.loadData(decoder.mid.attn.qLinerLayer.weight, weightMap, "vae.decoder.mid.1.q.conv.weight", 5);
        ClipModelUtils.loadData(decoder.mid.attn.kLinerLayer.weight, weightMap, "vae.decoder.mid.1.k.conv.weight", 5);
        ClipModelUtils.loadData(decoder.mid.attn.vLinerLayer.weight, weightMap, "vae.decoder.mid.1.v.conv.weight", 5);
        ClipModelUtils.loadData(decoder.mid.attn.oLinerLayer.weight, weightMap, "vae.decoder.mid.1.proj_out.conv.weight", 5);

        decoder.mid.block2.norm1.norm.gamma = ClipModelUtils.loadData(decoder.mid.block2.norm1.norm.gamma, weightMap, 1, "vae.decoder.mid.2.norm1.norm.weight");
        decoder.mid.block2.norm1.norm.beta = ClipModelUtils.loadData(decoder.mid.block2.norm1.norm.beta, weightMap, 1, "vae.decoder.mid.2.norm1.norm.bias");
        ClipModelUtils.loadData(decoder.mid.block2.conv1.weight, weightMap, "vae.decoder.mid.2.conv1.conv.weight", 5);
        ClipModelUtils.loadData(decoder.mid.block2.conv1.bias, weightMap, "vae.decoder.mid.2.conv1.conv.bias");
        decoder.mid.block2.norm2.norm.gamma = ClipModelUtils.loadData(decoder.mid.block2.norm2.norm.gamma, weightMap, 1, "vae.decoder.mid.2.norm2.norm.weight");
        decoder.mid.block2.norm2.norm.beta = ClipModelUtils.loadData(decoder.mid.block2.norm2.norm.beta, weightMap, 1, "vae.decoder.mid.2.norm2.norm.bias");
        ClipModelUtils.loadData(decoder.mid.block2.conv2.weight, weightMap, "vae.decoder.mid.2.conv2.conv.weight", 5);
        ClipModelUtils.loadData(decoder.mid.block2.conv2.bias, weightMap, "vae.decoder.mid.2.conv2.conv.bias");
        ClipModelUtils.loadData(decoder.mid.block2.shortcut.weight, weightMap, "vae.decoder.mid.2.nin_shortcut.conv.weight", 5);
        ClipModelUtils.loadData(decoder.mid.block2.shortcut.bias, weightMap, "vae.decoder.mid.2.nin_shortcut.conv.bias");

        /**
         * up2
         */
        for(int i = 0;i<decoder.getNum_resblocks();i++) {
            WFResnet3DBlock b3d = decoder.up2.resBlocks.get(i);
            b3d.norm1.norm.gamma = ClipModelUtils.loadData(b3d.norm1.norm.gamma, weightMap, 1, "vae.decoder.up2."+i+".norm1.norm.weight");
            b3d.norm1.norm.beta = ClipModelUtils.loadData(b3d.norm1.norm.beta, weightMap, 1, "vae.decoder.up2."+i+".norm1.norm.bias");
            ClipModelUtils.loadData(b3d.conv1.weight, weightMap, "vae.decoder.up2."+i+".conv1.conv.weight", 5);
            ClipModelUtils.loadData(b3d.conv1.bias, weightMap, "vae.decoder.up2."+i+".conv1.conv.bias");
            b3d.norm2.norm.gamma = ClipModelUtils.loadData(b3d.norm2.norm.gamma, weightMap, 1, "vae.decoder.up2."+i+".norm2.norm.weight");
            b3d.norm2.norm.beta = ClipModelUtils.loadData(b3d.norm2.norm.beta, weightMap, 1, "vae.decoder.up2."+i+".norm2.norm.bias");
            ClipModelUtils.loadData(b3d.conv2.weight, weightMap, "vae.decoder.up2."+i+".conv2.conv.weight", 5);
            ClipModelUtils.loadData(b3d.conv2.bias, weightMap, "vae.decoder.up2."+i+".conv2.conv.bias");
            if(b3d.shortcut != null) {
                ClipModelUtils.loadData(b3d.shortcut.weight, weightMap, "vae.decoder.up2."+i+".nin_shortcut.conv.weight", 5);
                ClipModelUtils.loadData(b3d.shortcut.bias, weightMap, "vae.decoder.up2."+i+".nin_shortcut.conv.bias");
            }
        }
        ClipModelUtils.loadData(decoder.up2.up3d.conv.weight, weightMap, "vae.decoder.up2.2.conv.conv.weight", 5);
        ClipModelUtils.loadData(decoder.up2.up3d.conv.bias, weightMap, "vae.decoder.up2.2.conv.conv.bias");
        WFResnet3DBlock up2_b = decoder.up2.block;
        up2_b.norm1.norm.gamma = ClipModelUtils.loadData(up2_b.norm1.norm.gamma, weightMap, 1, "vae.decoder.up2.3.norm1.norm.weight");
        up2_b.norm1.norm.beta = ClipModelUtils.loadData(up2_b.norm1.norm.beta, weightMap, 1, "vae.decoder.up2.3.norm1.norm.bias");
        ClipModelUtils.loadData(up2_b.conv1.weight, weightMap, "vae.decoder.up2.3.conv1.conv.weight", 5);
        ClipModelUtils.loadData(up2_b.conv1.bias, weightMap, "vae.decoder.up2.3.conv1.conv.bias");
        up2_b.norm2.norm.gamma = ClipModelUtils.loadData(up2_b.norm2.norm.gamma, weightMap, 1, "vae.decoder.up2.3.norm2.norm.weight");
        up2_b.norm2.norm.beta = ClipModelUtils.loadData(up2_b.norm2.norm.beta, weightMap, 1, "vae.decoder.up2.3.norm2.norm.bias");
        ClipModelUtils.loadData(up2_b.conv2.weight, weightMap, "vae.decoder.up2.3.conv2.conv.weight", 5);
        ClipModelUtils.loadData(up2_b.conv2.bias, weightMap, "vae.decoder.up2.3.conv2.conv.bias");
        ClipModelUtils.loadData(up2_b.shortcut.weight, weightMap, "vae.decoder.up2.3.nin_shortcut.conv.weight", 5);
        ClipModelUtils.loadData(up2_b.shortcut.bias, weightMap, "vae.decoder.up2.3.nin_shortcut.conv.bias");

        /**
         * up1
         */
        for(int i = 0;i<decoder.getNum_resblocks();i++) {
            WFResnet3DBlock b3d = decoder.up1.resBlocks.get(i);
            b3d.norm1.norm.gamma = ClipModelUtils.loadData(b3d.norm1.norm.gamma, weightMap, 1, "vae.decoder.up1."+i+".norm1.norm.weight");
            b3d.norm1.norm.beta = ClipModelUtils.loadData(b3d.norm1.norm.beta, weightMap, 1, "vae.decoder.up1."+i+".norm1.norm.bias");
            ClipModelUtils.loadData(b3d.conv1.weight, weightMap, "vae.decoder.up1."+i+".conv1.conv.weight", 5);
            ClipModelUtils.loadData(b3d.conv1.bias, weightMap, "vae.decoder.up1."+i+".conv1.conv.bias");
            b3d.norm2.norm.gamma = ClipModelUtils.loadData(b3d.norm2.norm.gamma, weightMap, 1, "vae.decoder.up1."+i+".norm2.norm.weight");
            b3d.norm2.norm.beta = ClipModelUtils.loadData(b3d.norm2.norm.beta, weightMap, 1, "vae.decoder.up1."+i+".norm2.norm.bias");
            ClipModelUtils.loadData(b3d.conv2.weight, weightMap, "vae.decoder.up1."+i+".conv2.conv.weight", 5);
            ClipModelUtils.loadData(b3d.conv2.bias, weightMap, "vae.decoder.up1."+i+".conv2.conv.bias");
            if(b3d.shortcut != null) {
                ClipModelUtils.loadData(b3d.shortcut.weight, weightMap, "vae.decoder.up1."+i+".nin_shortcut.conv.weight", 5);
                ClipModelUtils.loadData(b3d.shortcut.bias, weightMap, "vae.decoder.up1."+i+".nin_shortcut.conv.bias");
            }
        }
        ClipModelUtils.loadData(decoder.up1.up2d.conv.weight, weightMap, "vae.decoder.up1.2.conv.weight");
        ClipModelUtils.loadData(decoder.up1.up2d.conv.bias, weightMap, "vae.decoder.up1.2.conv.bias");
        WFResnet3DBlock up1_b = decoder.up1.block;
        up1_b.norm1.norm.gamma = ClipModelUtils.loadData(up1_b.norm1.norm.gamma, weightMap, 1, "vae.decoder.up1.3.norm1.norm.weight");
        up1_b.norm1.norm.beta = ClipModelUtils.loadData(up1_b.norm1.norm.beta, weightMap, 1, "vae.decoder.up1.3.norm1.norm.bias");
        ClipModelUtils.loadData(up1_b.conv1.weight, weightMap, "vae.decoder.up1.3.conv1.conv.weight", 5);
        ClipModelUtils.loadData(up1_b.conv1.bias, weightMap, "vae.decoder.up1.3.conv1.conv.bias");
        up1_b.norm2.norm.gamma = ClipModelUtils.loadData(up1_b.norm2.norm.gamma, weightMap, 1, "vae.decoder.up1.3.norm2.norm.weight");
        up1_b.norm2.norm.beta = ClipModelUtils.loadData(up1_b.norm2.norm.beta, weightMap, 1, "vae.decoder.up1.3.norm2.norm.bias");
        ClipModelUtils.loadData(up1_b.conv2.weight, weightMap, "vae.decoder.up1.3.conv2.conv.weight", 5);
        ClipModelUtils.loadData(up1_b.conv2.bias, weightMap, "vae.decoder.up1.3.conv2.conv.bias");

        /**
         * layers
         */
        for(int i = 0;i<2;i++) {
            WFResnet3DBlock b3d = decoder.blocks.get(i);
            b3d.norm1.norm.gamma = ClipModelUtils.loadData(b3d.norm1.norm.gamma, weightMap, 1, "vae.decoder.layer."+i+".norm1.norm.weight");
            b3d.norm1.norm.beta = ClipModelUtils.loadData(b3d.norm1.norm.beta, weightMap, 1, "vae.decoder.layer."+i+".norm1.norm.bias");
            ClipModelUtils.loadData(b3d.conv1.weight, weightMap, "vae.decoder.layer."+i+".conv1.conv.weight", 5);
            ClipModelUtils.loadData(b3d.conv1.bias, weightMap, "vae.decoder.layer."+i+".conv1.conv.bias");
            b3d.norm2.norm.gamma = ClipModelUtils.loadData(b3d.norm2.norm.gamma, weightMap, 1, "vae.decoder.layer."+i+".norm2.norm.weight");
            b3d.norm2.norm.beta = ClipModelUtils.loadData(b3d.norm2.norm.beta, weightMap, 1, "vae.decoder.layer."+i+".norm2.norm.bias");
            ClipModelUtils.loadData(b3d.conv2.weight, weightMap, "vae.decoder.layer."+i+".conv2.conv.weight", 5);
            ClipModelUtils.loadData(b3d.conv2.bias, weightMap, "vae.decoder.layer."+i+".conv2.conv.bias");
            if(b3d.shortcut != null) {
                ClipModelUtils.loadData(b3d.shortcut.weight, weightMap, "vae.decoder.layer."+i+".nin_shortcut.conv.weight", 5);
                ClipModelUtils.loadData(b3d.shortcut.bias, weightMap, "vae.decoder.layer."+i+".nin_shortcut.conv.bias");
            }
        }

        /**
         * connect_l1
         */
        WFResnet3DBlock b3d = decoder.connect_l1.resBlocks.get(0);
        b3d.norm1.norm.gamma = ClipModelUtils.loadData(b3d.norm1.norm.gamma, weightMap, 1, "vae.decoder.connect_l1.0.norm1.norm.weight");
        b3d.norm1.norm.beta = ClipModelUtils.loadData(b3d.norm1.norm.beta, weightMap, 1, "vae.decoder.connect_l1.0.norm1.norm.bias");
        ClipModelUtils.loadData(b3d.conv1.weight, weightMap, "vae.decoder.connect_l1.0.conv1.conv.weight", 5);
        ClipModelUtils.loadData(b3d.conv1.bias, weightMap, "vae.decoder.connect_l1.0.conv1.conv.bias");
        b3d.norm2.norm.gamma = ClipModelUtils.loadData(b3d.norm2.norm.gamma, weightMap, 1, "vae.decoder.connect_l1.0.norm2.norm.weight");
        b3d.norm2.norm.beta = ClipModelUtils.loadData(b3d.norm2.norm.beta, weightMap, 1, "vae.decoder.connect_l1.0.norm2.norm.bias");
        ClipModelUtils.loadData(b3d.conv2.weight, weightMap, "vae.decoder.connect_l1.0.conv2.conv.weight", 5);
        ClipModelUtils.loadData(b3d.conv2.bias, weightMap, "vae.decoder.connect_l1.0.conv2.conv.bias");
        WFResnet3DBlock b3d_ = decoder.connect_l1.resBlocks.get(1);
        b3d_.norm1.norm.gamma = ClipModelUtils.loadData(b3d_.norm1.norm.gamma, weightMap, 1, "vae.decoder.connect_l1.1.norm1.norm.weight");
        b3d_.norm1.norm.beta = ClipModelUtils.loadData(b3d_.norm1.norm.beta, weightMap, 1, "vae.decoder.connect_l1.1.norm1.norm.bias");
        ClipModelUtils.loadData(b3d_.conv1.weight, weightMap, "vae.decoder.connect_l1.1.conv1.conv.weight", 5);
        ClipModelUtils.loadData(b3d_.conv1.bias, weightMap, "vae.decoder.connect_l1.1.conv1.conv.bias");
        b3d_.norm2.norm.gamma = ClipModelUtils.loadData(b3d_.norm2.norm.gamma, weightMap, 1, "vae.decoder.connect_l1.1.norm2.norm.weight");
        b3d_.norm2.norm.beta = ClipModelUtils.loadData(b3d_.norm2.norm.beta, weightMap, 1, "vae.decoder.connect_l1.1.norm2.norm.bias");
        ClipModelUtils.loadData(b3d_.conv2.weight, weightMap, "vae.decoder.connect_l1.1.conv2.conv.weight", 5);
        ClipModelUtils.loadData(b3d_.conv2.bias, weightMap, "vae.decoder.connect_l1.1.conv2.conv.bias");
        ClipModelUtils.loadData(decoder.connect_l1.conv.conv.weight, weightMap, "vae.decoder.connect_l1.2.weight");
        ClipModelUtils.loadData(decoder.connect_l1.conv.conv.bias, weightMap, "vae.decoder.connect_l1.2.bias");

        /**
         * connect_l2
         */
        WFResnet3DBlock b3d2 = decoder.connect_l2.resBlocks.get(0);
        b3d2.norm1.norm.gamma = ClipModelUtils.loadData(b3d2.norm1.norm.gamma, weightMap, 1, "vae.decoder.connect_l2.0.norm1.norm.weight");
        b3d2.norm1.norm.beta = ClipModelUtils.loadData(b3d2.norm1.norm.beta, weightMap, 1, "vae.decoder.connect_l2.0.norm1.norm.bias");
        ClipModelUtils.loadData(b3d2.conv1.weight, weightMap, "vae.decoder.connect_l2.0.conv1.conv.weight", 5);
        ClipModelUtils.loadData(b3d2.conv1.bias, weightMap, "vae.decoder.connect_l2.0.conv1.conv.bias");
        b3d2.norm2.norm.gamma = ClipModelUtils.loadData(b3d2.norm2.norm.gamma, weightMap, 1, "vae.decoder.connect_l2.0.norm2.norm.weight");
        b3d2.norm2.norm.beta = ClipModelUtils.loadData(b3d2.norm2.norm.beta, weightMap, 1, "vae.decoder.connect_l2.0.norm2.norm.bias");
        ClipModelUtils.loadData(b3d2.conv2.weight, weightMap, "vae.decoder.connect_l2.0.conv2.conv.weight", 5);
        ClipModelUtils.loadData(b3d2.conv2.bias, weightMap, "vae.decoder.connect_l2.0.conv2.conv.bias");
        WFResnet3DBlock b3d2_ = decoder.connect_l2.resBlocks.get(1);
        b3d2_.norm1.norm.gamma = ClipModelUtils.loadData(b3d2_.norm1.norm.gamma, weightMap, 1, "vae.decoder.connect_l2.1.norm1.norm.weight");
        b3d2_.norm1.norm.beta = ClipModelUtils.loadData(b3d2_.norm1.norm.beta, weightMap, 1, "vae.decoder.connect_l2.1.norm1.norm.bias");
        ClipModelUtils.loadData(b3d2_.conv1.weight, weightMap, "vae.decoder.connect_l2.1.conv1.conv.weight", 5);
        ClipModelUtils.loadData(b3d2_.conv1.bias, weightMap, "vae.decoder.connect_l2.1.conv1.conv.bias");
        b3d2_.norm2.norm.gamma = ClipModelUtils.loadData(b3d2_.norm2.norm.gamma, weightMap, 1, "vae.decoder.connect_l2.1.norm2.norm.weight");
        b3d2_.norm2.norm.beta = ClipModelUtils.loadData(b3d2_.norm2.norm.beta, weightMap, 1, "vae.decoder.connect_l2.1.norm2.norm.bias");
        ClipModelUtils.loadData(b3d2_.conv2.weight, weightMap, "vae.decoder.connect_l2.1.conv2.conv.weight", 5);
        ClipModelUtils.loadData(b3d2_.conv2.bias, weightMap, "vae.decoder.connect_l2.1.conv2.conv.bias");
        ClipModelUtils.loadData(decoder.connect_l2.conv.conv.weight, weightMap, "vae.decoder.connect_l2.2.weight");
        ClipModelUtils.loadData(decoder.connect_l2.conv.conv.bias, weightMap, "vae.decoder.connect_l2.2.bias");
        
        decoder.norm_out.norm.gamma = ClipModelUtils.loadData(decoder.norm_out.norm.gamma, weightMap, 1, "vae.decoder.norm_out.norm.weight");
        decoder.norm_out.norm.beta = ClipModelUtils.loadData(decoder.norm_out.norm.beta, weightMap, 1, "vae.decoder.norm_out.norm.bias");
        ClipModelUtils.loadData(decoder.conv_out.conv.weight, weightMap, "vae.decoder.conv_out.weight");
        ClipModelUtils.loadData(decoder.conv_out.conv.bias, weightMap, "vae.decoder.conv_out.bias");
    }

    public static void loadWeight(Map<String, Object> weightMap, SD_VAE network, boolean showLayers) {
        if (showLayers) {
            for (String key : weightMap.keySet()) {
                System.out.println(key);
            }
        }
        
        /**
         * encoders
         */
        loadData(network.encoder.convIn.weight, weightMap, "encoder.conv_in.weight");
        loadData(network.encoder.convIn.bias, weightMap, "encoder.conv_in.bias");
        
        int idx = 0;
        
        for (int i = 0; i < network.ch_mult.length; i++) {
            for (int ri = 0; ri < network.num_res_blocks; ri++) {
            	Layer layer = network.encoder.down.get(idx);
            	if(layer instanceof SDVAEResidual) {
            		SDVAEResidual vr = (SDVAEResidual) layer;
            		vr.norm1.gamma = loadData(vr.norm1.gamma, weightMap, 1, "encoder.down_blocks."+i+".resnets."+ri+".norm1.weight");
            		vr.norm1.beta = loadData(vr.norm1.beta, weightMap, 1, "encoder.down_blocks."+i+".resnets."+ri+".norm1.bias");
            		loadData(vr.conv1.weight, weightMap, "encoder.down_blocks."+i+".resnets."+ri+".conv1.weight");
                    loadData(vr.conv1.bias, weightMap, "encoder.down_blocks."+i+".resnets."+ri+".conv1.bias");
            		vr.norm2.gamma = loadData(vr.norm2.gamma, weightMap, 1, "encoder.down_blocks."+i+".resnets."+ri+".norm2.weight");
            		vr.norm2.beta = loadData(vr.norm2.beta, weightMap, 1, "encoder.down_blocks."+i+".resnets."+ri+".norm2.bias");
            		loadData(vr.conv2.weight, weightMap, "encoder.down_blocks."+i+".resnets."+ri+".conv2.weight");
                    loadData(vr.conv2.bias, weightMap, "encoder.down_blocks."+i+".resnets."+ri+".conv2.bias");
                    if(vr.conv_shortcut != null) {
                    	vr.conv_shortcut.weight = loadData(vr.conv_shortcut.weight, weightMap, 4, "encoder.down_blocks."+i+".resnets.0.conv_shortcut.weight");
                        loadData(vr.conv_shortcut.bias, weightMap, "encoder.down_blocks."+i+".resnets.0.conv_shortcut.bias");
                    }
                    idx++;
            	}
            	
            }

            if (i != network.ch_mult.length - 1) {
            	Layer layer = network.encoder.down.get(idx);
            	if(layer instanceof SDVAEDownsample) {
            		SDVAEDownsample downsample = (SDVAEDownsample) layer;
            		loadData(downsample.conv.weight, weightMap, "encoder.down_blocks."+i+".downsamplers.0.conv.weight");
                    loadData(downsample.conv.bias, weightMap, "encoder.down_blocks."+i+".downsamplers.0.conv.bias");
            	}
            	idx++;
            }
        }
        
        /**
         * encoder mids
         */
        Layer layer = network.encoder.down.get(idx);
        if(layer instanceof SDVAEResidual) {
        	SDVAEResidual vr = (SDVAEResidual) layer;
    		vr.norm1.gamma = loadData(vr.norm1.gamma, weightMap, 1, "encoder.mid_block.resnets.0.norm1.weight");
    		vr.norm1.beta = loadData(vr.norm1.beta, weightMap, 1, "encoder.mid_block.resnets.0.norm1.bias");
    		loadData(vr.conv1.weight, weightMap, "encoder.mid_block.resnets.0.conv1.weight");
            loadData(vr.conv1.bias, weightMap, "encoder.mid_block.resnets.0.conv1.bias");
    		vr.norm2.gamma = loadData(vr.norm2.gamma, weightMap, 1, "encoder.mid_block.resnets.0.norm2.weight");
    		vr.norm2.beta = loadData(vr.norm2.beta, weightMap, 1, "encoder.mid_block.resnets.0.norm2.bias");
    		loadData(vr.conv2.weight, weightMap, "encoder.mid_block.resnets.0.conv2.weight");
            loadData(vr.conv2.bias, weightMap, "encoder.mid_block.resnets.0.conv2.bias");
            idx++;
        }
        
        Layer layer2 = network.encoder.down.get(idx);
        
        if(layer2 instanceof SDVAEAttentionLayer) {
        	SDVAEAttentionLayer va = (SDVAEAttentionLayer) layer2;
        	va.gn.gamma = loadData(va.gn.gamma, weightMap, 1, "encoder.mid_block.attentions.0.group_norm.weight");
    		va.gn.beta = loadData(va.gn.beta, weightMap, 1, "encoder.mid_block.attentions.0.group_norm.bias");
        	loadData(va.attn.qLinerLayer.weight, weightMap, "encoder.mid_block.attentions.0.to_q.weight");
            loadData(va.attn.qLinerLayer.bias, weightMap, "encoder.mid_block.attentions.0.to_q.bias");
        	loadData(va.attn.kLinerLayer.weight, weightMap, "encoder.mid_block.attentions.0.to_k.weight");
            loadData(va.attn.kLinerLayer.bias, weightMap, "encoder.mid_block.attentions.0.to_k.bias");
        	loadData(va.attn.vLinerLayer.weight, weightMap, "encoder.mid_block.attentions.0.to_v.weight");
            loadData(va.attn.vLinerLayer.bias, weightMap, "encoder.mid_block.attentions.0.to_v.bias");
        	loadData(va.attn.oLinerLayer.weight, weightMap, "encoder.mid_block.attentions.0.to_out.0.weight");
            loadData(va.attn.oLinerLayer.bias, weightMap, "encoder.mid_block.attentions.0.to_out.0.bias");
            idx++;
        }
        
        Layer layer3 = network.encoder.down.get(idx);
        if(layer3 instanceof SDVAEResidual) {
        	SDVAEResidual vr = (SDVAEResidual) layer3;
    		vr.norm1.gamma = loadData(vr.norm1.gamma, weightMap, 1, "encoder.mid_block.resnets.1.norm1.weight");
    		vr.norm1.beta = loadData(vr.norm1.beta, weightMap, 1, "encoder.mid_block.resnets.1.norm1.bias");
    		loadData(vr.conv1.weight, weightMap, "encoder.mid_block.resnets.1.conv1.weight");
            loadData(vr.conv1.bias, weightMap, "encoder.mid_block.resnets.1.conv1.bias");
    		vr.norm2.gamma = loadData(vr.norm2.gamma, weightMap, 1, "encoder.mid_block.resnets.1.norm2.weight");
    		vr.norm2.beta = loadData(vr.norm2.beta, weightMap, 1, "encoder.mid_block.resnets.1.norm2.bias");
    		loadData(vr.conv2.weight, weightMap, "encoder.mid_block.resnets.1.conv2.weight");
            loadData(vr.conv2.bias, weightMap, "encoder.mid_block.resnets.1.conv2.bias");
            idx++;
        }
        
        network.encoder.convNormOut.gamma = loadData(network.encoder.convNormOut.gamma, weightMap, 1, "encoder.conv_norm_out.weight");
        network.encoder.convNormOut.beta = loadData(network.encoder.convNormOut.beta, weightMap, 1, "encoder.conv_norm_out.bias");
        loadData(network.encoder.convOut.weight, weightMap, "encoder.conv_out.weight");
        loadData(network.encoder.convOut.bias, weightMap, "encoder.conv_out.bias");
        
        /**
         * decoder
         */
        loadData(network.decoder.convIn.weight, weightMap, "decoder.conv_in.weight");
        loadData(network.decoder.convIn.bias, weightMap, "decoder.conv_in.bias");
        
        int idx_up = 0;
        
        /**
         * decoder mids
         */
        Layer layer_up1 = network.decoder.up.get(idx_up);
        if(layer_up1 instanceof SDVAEResidual) {
        	SDVAEResidual vr = (SDVAEResidual) layer_up1;
    		vr.norm1.gamma = loadData(vr.norm1.gamma, weightMap, 1, "decoder.mid_block.resnets.0.norm1.weight");
    		vr.norm1.beta = loadData(vr.norm1.beta, weightMap, 1, "decoder.mid_block.resnets.0.norm1.bias");
    		loadData(vr.conv1.weight, weightMap, "decoder.mid_block.resnets.0.conv1.weight");
            loadData(vr.conv1.bias, weightMap, "decoder.mid_block.resnets.0.conv1.bias");
    		vr.norm2.gamma = loadData(vr.norm2.gamma, weightMap, 1, "decoder.mid_block.resnets.0.norm2.weight");
    		vr.norm2.beta = loadData(vr.norm2.beta, weightMap, 1, "decoder.mid_block.resnets.0.norm2.bias");
    		loadData(vr.conv2.weight, weightMap, "decoder.mid_block.resnets.0.conv2.weight");
            loadData(vr.conv2.bias, weightMap, "decoder.mid_block.resnets.0.conv2.bias");
            idx_up++;
        }
        
        Layer layer_up2 = network.decoder.up.get(idx_up);
        if(layer_up2 instanceof SDVAEAttentionLayer) {
        	SDVAEAttentionLayer va = (SDVAEAttentionLayer) layer_up2;
        	va.gn.gamma = loadData(va.gn.gamma, weightMap, 1, "decoder.mid_block.attentions.0.group_norm.weight");
    		va.gn.beta = loadData(va.gn.beta, weightMap, 1, "decoder.mid_block.attentions.0.group_norm.bias");
        	loadData(va.attn.qLinerLayer.weight, weightMap, "decoder.mid_block.attentions.0.to_q.weight");
            loadData(va.attn.qLinerLayer.bias, weightMap, "decoder.mid_block.attentions.0.to_q.bias");
        	loadData(va.attn.kLinerLayer.weight, weightMap, "decoder.mid_block.attentions.0.to_k.weight");
            loadData(va.attn.kLinerLayer.bias, weightMap, "decoder.mid_block.attentions.0.to_k.bias");
        	loadData(va.attn.vLinerLayer.weight, weightMap, "decoder.mid_block.attentions.0.to_v.weight");
            loadData(va.attn.vLinerLayer.bias, weightMap, "decoder.mid_block.attentions.0.to_v.bias");
        	loadData(va.attn.oLinerLayer.weight, weightMap, "decoder.mid_block.attentions.0.to_out.0.weight");
            loadData(va.attn.oLinerLayer.bias, weightMap, "decoder.mid_block.attentions.0.to_out.0.bias");
            idx_up++;
        }
        
        Layer layer_up3 = network.decoder.up.get(idx_up);
        if(layer_up3 instanceof SDVAEResidual) {
        	SDVAEResidual vr = (SDVAEResidual) layer_up3;
    		vr.norm1.gamma = loadData(vr.norm1.gamma, weightMap, 1, "decoder.mid_block.resnets.1.norm1.weight");
    		vr.norm1.beta = loadData(vr.norm1.beta, weightMap, 1, "decoder.mid_block.resnets.1.norm1.bias");
    		loadData(vr.conv1.weight, weightMap, "decoder.mid_block.resnets.1.conv1.weight");
            loadData(vr.conv1.bias, weightMap, "decoder.mid_block.resnets.1.conv1.bias");
    		vr.norm2.gamma = loadData(vr.norm2.gamma, weightMap, 1, "decoder.mid_block.resnets.1.norm2.weight");
    		vr.norm2.beta = loadData(vr.norm2.beta, weightMap, 1, "decoder.mid_block.resnets.1.norm2.bias");
    		loadData(vr.conv2.weight, weightMap, "decoder.mid_block.resnets.1.conv2.weight");
            loadData(vr.conv2.bias, weightMap, "decoder.mid_block.resnets.1.conv2.bias");
            idx_up++;
        }

        for (int i = network.ch_mult.length - 1; i >= 0; i--) {
        	int real_i = (network.ch_mult.length - 1 - i);
//        	System.err.println("real_i:"+real_i);
            for (int ri = 0; ri < network.num_res_blocks + 1; ri++) {
            	Layer layer_up = network.decoder.up.get(idx_up);
            	if(layer_up instanceof SDVAEResidual) {
            		SDVAEResidual vr = (SDVAEResidual) layer_up;
                	vr.norm1.gamma = loadData(vr.norm1.gamma, weightMap, 1, "decoder.up_blocks."+real_i+".resnets."+ri+".norm1.weight");
            		vr.norm1.beta = loadData(vr.norm1.beta, weightMap, 1, "decoder.up_blocks."+real_i+".resnets."+ri+".norm1.bias");
            		loadData(vr.conv1.weight, weightMap, "decoder.up_blocks."+real_i+".resnets."+ri+".conv1.weight");
                    loadData(vr.conv1.bias, weightMap, "decoder.up_blocks."+real_i+".resnets."+ri+".conv1.bias");
            		vr.norm2.gamma = loadData(vr.norm2.gamma, weightMap, 1, "decoder.up_blocks."+real_i+".resnets."+ri+".norm2.weight");
            		vr.norm2.beta = loadData(vr.norm2.beta, weightMap, 1, "decoder.up_blocks."+real_i+".resnets."+ri+".norm2.bias");
            		loadData(vr.conv2.weight, weightMap, "decoder.up_blocks."+real_i+".resnets."+ri+".conv2.weight");
                    loadData(vr.conv2.bias, weightMap, "decoder.up_blocks."+real_i+".resnets."+ri+".conv2.bias");
                    if(vr.conv_shortcut != null) {
                    	vr.conv_shortcut.weight = loadData(vr.conv_shortcut.weight, weightMap, 4, "decoder.up_blocks."+real_i+".resnets."+ri+".conv_shortcut.weight");
                        loadData(vr.conv_shortcut.bias, weightMap, "decoder.up_blocks."+real_i+".resnets."+ri+".conv_shortcut.bias");
                    }
                	idx_up++;
            	}
            	
            }
            if (i != 0) {
            	Layer layer_up = network.decoder.up.get(idx_up);
            	if(layer_up instanceof SDVAEUpsample) {
            		SDVAEUpsample upsample = (SDVAEUpsample) layer_up;
	                loadData(upsample.conv.weight, weightMap, "decoder.up_blocks."+real_i+".upsamplers.0.conv.weight");
	                loadData(upsample.conv.bias, weightMap, "decoder.up_blocks."+real_i+".upsamplers.0.conv.bias");
	                idx_up++;
            	}
            }
        }
        
        network.decoder.convNormOut.gamma = loadData(network.decoder.convNormOut.gamma, weightMap, 1, "decoder.conv_norm_out.weight");
        network.decoder.convNormOut.beta = loadData(network.decoder.convNormOut.beta, weightMap, 1, "decoder.conv_norm_out.bias");
        loadData(network.decoder.convOut.weight, weightMap, "decoder.conv_out.weight");
        loadData(network.decoder.convOut.bias, weightMap, "decoder.conv_out.bias");
        
        network.pre_quant_conv.weight = loadData(network.pre_quant_conv.weight, weightMap, 4, "quant_conv.weight");
        loadData(network.pre_quant_conv.bias, weightMap, "quant_conv.bias");
        
        network.post_quant_conv.weight = loadData(network.post_quant_conv.weight, weightMap, 4, "post_quant_conv.weight");
        loadData(network.post_quant_conv.bias, weightMap, "post_quant_conv.bias");
        
    }
    
    public static void loadWeight(Map<String, Object> weightMap, VA_VAE network, boolean showLayers) {
        if (showLayers) {
            for (String key : weightMap.keySet()) {
                System.out.println(key);
            }
        }
        
        /**
         * encoders
         */
        loadData(network.encoder.convIn.weight, weightMap, "encoder.conv_in.weight");
        loadData(network.encoder.convIn.bias, weightMap, "encoder.conv_in.bias");
        
        int idx = 0;
        
        for (int i = 0; i < network.ch_mult.length; i++) {
            for (int ri = 0; ri < network.num_res_blocks; ri++) {
            	Layer layer = network.encoder.down.get(idx);
            	if(layer instanceof SDVAEResidual) {
            		SDVAEResidual vr = (SDVAEResidual) layer;
            		vr.norm1.gamma = loadData(vr.norm1.gamma, weightMap, 1, "encoder.down."+i+".block."+ri+".norm1.weight");
            		vr.norm1.beta = loadData(vr.norm1.beta, weightMap, 1, "encoder.down."+i+".block."+ri+".norm1.bias");
            		loadData(vr.conv1.weight, weightMap, "encoder.down."+i+".block."+ri+".conv1.weight");
                    loadData(vr.conv1.bias, weightMap, "encoder.down."+i+".block."+ri+".conv1.bias");
            		vr.norm2.gamma = loadData(vr.norm2.gamma, weightMap, 1, "encoder.down."+i+".block."+ri+".norm2.weight");
            		vr.norm2.beta = loadData(vr.norm2.beta, weightMap, 1, "encoder.down."+i+".block."+ri+".norm2.bias");
            		loadData(vr.conv2.weight, weightMap, "encoder.down."+i+".block."+ri+".conv2.weight");
                    loadData(vr.conv2.bias, weightMap, "encoder.down."+i+".block."+ri+".conv2.bias");
                    if(vr.conv_shortcut != null) {
                    	vr.conv_shortcut.weight = loadData(vr.conv_shortcut.weight, weightMap, 4, "encoder.down."+i+".block.0.nin_shortcut.weight");
                        loadData(vr.conv_shortcut.bias, weightMap, "encoder.down."+i+".block.0.nin_shortcut.bias");
                    }
                    idx++;
            	}
            	
            	if (network.encoder.hasAttn && i == network.ch_mult.length - 1) {
            		VAVAEAttentionLayer attn = (VAVAEAttentionLayer) network.encoder.down.get(idx);
            		attn.gn.gamma = loadData(attn.gn.gamma, weightMap, 1, "encoder.down."+i+".attn."+ri+".norm.weight");
            		attn.gn.beta = loadData(attn.gn.beta, weightMap, 1, "encoder.down."+i+".attn."+ri+".norm.bias");
                	loadData(attn.attn.qLinerLayer.weight, weightMap, "encoder.down."+i+".attn."+ri+".q.weight", 4);
                    loadData(attn.attn.qLinerLayer.bias, weightMap, "encoder.down."+i+".attn."+ri+".q.bias");
                	loadData(attn.attn.kLinerLayer.weight, weightMap, "encoder.down."+i+".attn."+ri+".k.weight", 4);
                    loadData(attn.attn.kLinerLayer.bias, weightMap, "encoder.down."+i+".attn."+ri+".k.bias");
                	loadData(attn.attn.vLinerLayer.weight, weightMap, "encoder.down."+i+".attn."+ri+".v.weight", 4);
                    loadData(attn.attn.vLinerLayer.bias, weightMap, "encoder.down."+i+".attn."+ri+".v.bias");
                	loadData(attn.attn.oLinerLayer.weight, weightMap, "encoder.down."+i+".attn."+ri+".proj_out.weight", 4);
                    loadData(attn.attn.oLinerLayer.bias, weightMap, "encoder.down."+i+".attn."+ri+".proj_out.bias");
                    idx++;
                }

            }

            if (i != network.ch_mult.length - 1) {
            	Layer layer = network.encoder.down.get(idx);
            	if(layer instanceof SDVAEDownsample) {
            		SDVAEDownsample downsample = (SDVAEDownsample) layer;
            		loadData(downsample.conv.weight, weightMap, "encoder.down."+i+".downsample.conv.weight");
                    loadData(downsample.conv.bias, weightMap, "encoder.down."+i+".downsample.conv.bias");
            	}
            	idx++;
            }
            
        }
        
        /**
         * encoder mids
         */
        Layer layer = network.encoder.down.get(idx);
        if(layer instanceof SDVAEResidual) {
        	SDVAEResidual vr = (SDVAEResidual) layer;
    		vr.norm1.gamma = loadData(vr.norm1.gamma, weightMap, 1, "encoder.mid.block_1.norm1.weight");
    		vr.norm1.beta = loadData(vr.norm1.beta, weightMap, 1, "encoder.mid.block_1.norm1.bias");
    		loadData(vr.conv1.weight, weightMap, "encoder.mid.block_1.conv1.weight");
            loadData(vr.conv1.bias, weightMap, "encoder.mid.block_1.conv1.bias");
    		vr.norm2.gamma = loadData(vr.norm2.gamma, weightMap, 1, "encoder.mid.block_1.norm2.weight");
    		vr.norm2.beta = loadData(vr.norm2.beta, weightMap, 1, "encoder.mid.block_1.norm2.bias");
    		loadData(vr.conv2.weight, weightMap, "encoder.mid.block_1.conv2.weight");
            loadData(vr.conv2.bias, weightMap, "encoder.mid.block_1.conv2.bias");
            idx++;
        }
        
        Layer layer2 = network.encoder.down.get(idx);
        
        if(layer2 instanceof VAVAEAttentionLayer) {
        	VAVAEAttentionLayer va = (VAVAEAttentionLayer) layer2;
        	va.gn.gamma = loadData(va.gn.gamma, weightMap, 1, "encoder.mid.attn_1.norm.weight");
    		va.gn.beta = loadData(va.gn.beta, weightMap, 1, "encoder.mid.attn_1.norm.bias");
        	loadData(va.attn.qLinerLayer.weight, weightMap, "encoder.mid.attn_1.q.weight", 4);
            loadData(va.attn.qLinerLayer.bias, weightMap, "encoder.mid.attn_1.q.bias");
        	loadData(va.attn.kLinerLayer.weight, weightMap, "encoder.mid.attn_1.k.weight", 4);
            loadData(va.attn.kLinerLayer.bias, weightMap, "encoder.mid.attn_1.k.bias");
        	loadData(va.attn.vLinerLayer.weight, weightMap, "encoder.mid.attn_1.v.weight", 4);
            loadData(va.attn.vLinerLayer.bias, weightMap, "encoder.mid.attn_1.v.bias");
        	loadData(va.attn.oLinerLayer.weight, weightMap, "encoder.mid.attn_1.proj_out.weight", 4);
            loadData(va.attn.oLinerLayer.bias, weightMap, "encoder.mid.attn_1.proj_out.bias");
            idx++;
        }
        
        Layer layer3 = network.encoder.down.get(idx);
        if(layer3 instanceof SDVAEResidual) {
        	SDVAEResidual vr = (SDVAEResidual) layer3;
    		vr.norm1.gamma = loadData(vr.norm1.gamma, weightMap, 1, "encoder.mid.block_2.norm1.weight");
    		vr.norm1.beta = loadData(vr.norm1.beta, weightMap, 1, "encoder.mid.block_2.norm1.bias");
    		loadData(vr.conv1.weight, weightMap, "encoder.mid.block_2.conv1.weight");
            loadData(vr.conv1.bias, weightMap, "encoder.mid.block_2.conv1.bias");
    		vr.norm2.gamma = loadData(vr.norm2.gamma, weightMap, 1, "encoder.mid.block_2.norm2.weight");
    		vr.norm2.beta = loadData(vr.norm2.beta, weightMap, 1, "encoder.mid.block_2.norm2.bias");
    		loadData(vr.conv2.weight, weightMap, "encoder.mid.block_2.conv2.weight");
            loadData(vr.conv2.bias, weightMap, "encoder.mid.block_2.conv2.bias");
            idx++;
        }
        
        network.encoder.convNormOut.gamma = loadData(network.encoder.convNormOut.gamma, weightMap, 1, "encoder.norm_out.weight");
        network.encoder.convNormOut.beta = loadData(network.encoder.convNormOut.beta, weightMap, 1, "encoder.norm_out.bias");
        loadData(network.encoder.convOut.weight, weightMap, "encoder.conv_out.weight");
        loadData(network.encoder.convOut.bias, weightMap, "encoder.conv_out.bias");
        
        /**
         * decoder
         */
        loadData(network.decoder.convIn.weight, weightMap, "decoder.conv_in.weight");
        loadData(network.decoder.convIn.bias, weightMap, "decoder.conv_in.bias");
        
        int idx_up = 0;
        
        /**
         * decoder mids
         */
        Layer layer_up1 = network.decoder.up.get(idx_up);
        if(layer_up1 instanceof SDVAEResidual) {
        	SDVAEResidual vr = (SDVAEResidual) layer_up1;
    		vr.norm1.gamma = loadData(vr.norm1.gamma, weightMap, 1, "decoder.mid.block_1.norm1.weight");
    		vr.norm1.beta = loadData(vr.norm1.beta, weightMap, 1, "decoder.mid.block_1.norm1.bias");
    		loadData(vr.conv1.weight, weightMap, "decoder.mid.block_1.conv1.weight");
            loadData(vr.conv1.bias, weightMap, "decoder.mid.block_1.conv1.bias");
    		vr.norm2.gamma = loadData(vr.norm2.gamma, weightMap, 1, "decoder.mid.block_1.norm2.weight");
    		vr.norm2.beta = loadData(vr.norm2.beta, weightMap, 1, "decoder.mid.block_1.norm2.bias");
    		loadData(vr.conv2.weight, weightMap, "decoder.mid.block_1.conv2.weight");
            loadData(vr.conv2.bias, weightMap, "decoder.mid.block_1.conv2.bias");
            idx_up++;
        }
        
        Layer layer_up2 = network.decoder.up.get(idx_up);
        if(layer_up2 instanceof VAVAEAttentionLayer) {
        	VAVAEAttentionLayer va = (VAVAEAttentionLayer) layer_up2;
        	va.gn.gamma = loadData(va.gn.gamma, weightMap, 1, "decoder.mid.attn_1.norm.weight");
    		va.gn.beta = loadData(va.gn.beta, weightMap, 1, "decoder.mid.attn_1.norm.bias");
        	loadData(va.attn.qLinerLayer.weight, weightMap, "decoder.mid.attn_1.q.weight", 4);
            loadData(va.attn.qLinerLayer.bias, weightMap, "decoder.mid.attn_1.q.bias");
        	loadData(va.attn.kLinerLayer.weight, weightMap, "decoder.mid.attn_1.k.weight", 4);
            loadData(va.attn.kLinerLayer.bias, weightMap, "decoder.mid.attn_1.k.bias");
        	loadData(va.attn.vLinerLayer.weight, weightMap, "decoder.mid.attn_1.v.weight", 4);
            loadData(va.attn.vLinerLayer.bias, weightMap, "decoder.mid.attn_1.v.bias");
        	loadData(va.attn.oLinerLayer.weight, weightMap, "decoder.mid.attn_1.proj_out.weight", 4);
            loadData(va.attn.oLinerLayer.bias, weightMap, "decoder.mid.attn_1.proj_out.bias");
            idx_up++;
        }
        
        Layer layer_up3 = network.decoder.up.get(idx_up);
        if(layer_up3 instanceof SDVAEResidual) {
        	SDVAEResidual vr = (SDVAEResidual) layer_up3;
    		vr.norm1.gamma = loadData(vr.norm1.gamma, weightMap, 1, "decoder.mid.block_2.norm1.weight");
    		vr.norm1.beta = loadData(vr.norm1.beta, weightMap, 1, "decoder.mid.block_2.norm1.bias");
    		loadData(vr.conv1.weight, weightMap, "decoder.mid.block_2.conv1.weight");
            loadData(vr.conv1.bias, weightMap, "decoder.mid.block_2.conv1.bias");
    		vr.norm2.gamma = loadData(vr.norm2.gamma, weightMap, 1, "decoder.mid.block_2.norm2.weight");
    		vr.norm2.beta = loadData(vr.norm2.beta, weightMap, 1, "decoder.mid.block_2.norm2.bias");
    		loadData(vr.conv2.weight, weightMap, "decoder.mid.block_2.conv2.weight");
            loadData(vr.conv2.bias, weightMap, "decoder.mid.block_2.conv2.bias");
            idx_up++;
        }

        for (int i = network.ch_mult.length - 1; i >= 0; i--) {
            for (int ri = 0; ri < network.num_res_blocks + 1; ri++) {
            	Layer layer_up = network.decoder.up.get(idx_up);
            	if(layer_up instanceof SDVAEResidual) {
            		SDVAEResidual vr = (SDVAEResidual) layer_up;
                	vr.norm1.gamma = loadData(vr.norm1.gamma, weightMap, 1, "decoder.up."+i+".block."+ri+".norm1.weight");
            		vr.norm1.beta = loadData(vr.norm1.beta, weightMap, 1, "decoder.up."+i+".block."+ri+".norm1.bias");
            		loadData(vr.conv1.weight, weightMap, "decoder.up."+i+".block."+ri+".conv1.weight");
                    loadData(vr.conv1.bias, weightMap, "decoder.up."+i+".block."+ri+".conv1.bias");
            		vr.norm2.gamma = loadData(vr.norm2.gamma, weightMap, 1, "decoder.up."+i+".block."+ri+".norm2.weight");
            		vr.norm2.beta = loadData(vr.norm2.beta, weightMap, 1, "decoder.up."+i+".block."+ri+".norm2.bias");
            		loadData(vr.conv2.weight, weightMap, "decoder.up."+i+".block."+ri+".conv2.weight");
                    loadData(vr.conv2.bias, weightMap, "decoder.up."+i+".block."+ri+".conv2.bias");
                    if(vr.conv_shortcut != null) {
                    	vr.conv_shortcut.weight = loadData(vr.conv_shortcut.weight, weightMap, 4, "decoder.up."+i+".block."+ri+".nin_shortcut.weight");
                        loadData(vr.conv_shortcut.bias, weightMap, "decoder.up."+i+".block."+ri+".nin_shortcut.bias");
                    }
                	idx_up++;
            	}
            	if (network.decoder.hasAttn && i == network.ch_mult.length - 1) {
            		VAVAEAttentionLayer attn = (VAVAEAttentionLayer) network.decoder.up.get(idx_up);
            		attn.gn.gamma = loadData(attn.gn.gamma, weightMap, 1, "decoder.up."+i+".attn."+ri+".norm.weight");
            		attn.gn.beta = loadData(attn.gn.beta, weightMap, 1, "decoder.up."+i+".attn."+ri+".norm.bias");
                	loadData(attn.attn.qLinerLayer.weight, weightMap, "decoder.up."+i+".attn."+ri+".q.weight", 4);
                    loadData(attn.attn.qLinerLayer.bias, weightMap, "decoder.up."+i+".attn."+ri+".q.bias");
                	loadData(attn.attn.kLinerLayer.weight, weightMap, "decoder.up."+i+".attn."+ri+".k.weight", 4);
                    loadData(attn.attn.kLinerLayer.bias, weightMap, "decoder.up."+i+".attn."+ri+".k.bias");
                	loadData(attn.attn.vLinerLayer.weight, weightMap, "decoder.up."+i+".attn."+ri+".v.weight", 4);
                    loadData(attn.attn.vLinerLayer.bias, weightMap, "decoder.up."+i+".attn."+ri+".v.bias");
                	loadData(attn.attn.oLinerLayer.weight, weightMap, "decoder.up."+i+".attn."+ri+".proj_out.weight", 4);
                    loadData(attn.attn.oLinerLayer.bias, weightMap, "decoder.up."+i+".attn."+ri+".proj_out.bias");
                    idx_up++;
            	}
            }
            if (i != 0) {
            	Layer layer_up = network.decoder.up.get(idx_up);
            	if(layer_up instanceof SDVAEUpsample) {
            		SDVAEUpsample upsample = (SDVAEUpsample) layer_up;
	                loadData(upsample.conv.weight, weightMap, "decoder.up."+i+".upsample.conv.weight");
	                loadData(upsample.conv.bias, weightMap, "decoder.up."+i+".upsample.conv.bias");
	                idx_up++;
            	}
            }
        }
        
        network.decoder.convNormOut.gamma = loadData(network.decoder.convNormOut.gamma, weightMap, 1, "decoder.norm_out.weight");
        network.decoder.convNormOut.beta = loadData(network.decoder.convNormOut.beta, weightMap, 1, "decoder.norm_out.bias");
        loadData(network.decoder.convOut.weight, weightMap, "decoder.conv_out.weight");
        loadData(network.decoder.convOut.bias, weightMap, "decoder.conv_out.bias");
        
        network.pre_quant_conv.weight = loadData(network.pre_quant_conv.weight, weightMap, 4, "quant_conv.weight");
        loadData(network.pre_quant_conv.bias, weightMap, "quant_conv.bias");
        
        network.post_quant_conv.weight = loadData(network.post_quant_conv.weight, weightMap, 4, "post_quant_conv.weight");
        loadData(network.post_quant_conv.bias, weightMap, "post_quant_conv.bias");
        
    }
    
    public static void loadWeight(Map<String, Object> weightMap, ClipTextModel network, boolean showLayers) {
        if (showLayers) {
            for (String key : weightMap.keySet()) {
                System.out.println(key);
            }
        }
        /**
         * embeddings

         */
        loadData(network.clip.getEmbeddings().getTokenEmbedding().weight, weightMap, "text_model.embeddings.token_embedding.weight");
        loadData(network.clip.getEmbeddings().getPositionEmbedding().weight, weightMap, "text_model.embeddings.position_embedding.weight");
        /**
         * encoders

         */
        for (int i = 0; i < 12; i++) {
            /**
             * self attention

             */
            loadData(network.clip.getEncoders().get(i).getAttn().getqLinerLayer().weight, weightMap, "text_model.encoder.layers." + i + ".self_attn.q_proj.weight");
            loadData(network.clip.getEncoders().get(i).getAttn().getqLinerLayer().bias, weightMap, "text_model.encoder.layers." + i + ".self_attn.q_proj.bias");
            loadData(network.clip.getEncoders().get(i).getAttn().getkLinerLayer().weight, weightMap, "text_model.encoder.layers." + i + ".self_attn.k_proj.weight");
            loadData(network.clip.getEncoders().get(i).getAttn().getkLinerLayer().bias, weightMap, "text_model.encoder.layers." + i + ".self_attn.k_proj.bias");
            loadData(network.clip.getEncoders().get(i).getAttn().getvLinerLayer().weight, weightMap, "text_model.encoder.layers." + i + ".self_attn.v_proj.weight");
            loadData(network.clip.getEncoders().get(i).getAttn().getvLinerLayer().bias, weightMap, "text_model.encoder.layers." + i + ".self_attn.v_proj.bias");
            loadData(network.clip.getEncoders().get(i).getAttn().getoLinerLayer().weight, weightMap, "text_model.encoder.layers." + i + ".self_attn.out_proj.weight");
            loadData(network.clip.getEncoders().get(i).getAttn().getoLinerLayer().bias, weightMap, "text_model.encoder.layers." + i + ".self_attn.out_proj.bias");
            /**
             * layer norm1

             */
            network.clip.getEncoders().get(i).getNorm1().gamma = loadData(network.clip.getEncoders().get(i).getNorm1().gamma, weightMap, 1, "text_model.encoder.layers." + i + ".layer_norm1.weight");
            network.clip.getEncoders().get(i).getNorm1().beta = loadData(network.clip.getEncoders().get(i).getNorm1().beta, weightMap, 1, "text_model.encoder.layers." + i + ".layer_norm1.bias");
            /**
             * mlp

             */
            loadData(network.clip.getEncoders().get(i).getMlp().getLinear1().weight, weightMap, "text_model.encoder.layers." + i + ".mlp.fc1.weight");
            loadData(network.clip.getEncoders().get(i).getMlp().getLinear1().bias, weightMap, "text_model.encoder.layers." + i + ".mlp.fc1.bias");
            loadData(network.clip.getEncoders().get(i).getMlp().getLinear2().weight, weightMap, "text_model.encoder.layers." + i + ".mlp.fc2.weight");
            loadData(network.clip.getEncoders().get(i).getMlp().getLinear2().bias, weightMap, "text_model.encoder.layers." + i + ".mlp.fc2.bias");
            /**
             * layer norm2

             */
            network.clip.getEncoders().get(i).getNorm2().gamma = loadData(network.clip.getEncoders().get(i).getNorm2().gamma, weightMap, 1, "text_model.encoder.layers." + i + ".layer_norm2.weight");
            network.clip.getEncoders().get(i).getNorm2().beta = loadData(network.clip.getEncoders().get(i).getNorm2().beta, weightMap, 1, "text_model.encoder.layers." + i + ".layer_norm2.bias");
        }
        /**
         * post_layernorm

         */
        network.clip.getFinalLayerNorm().gamma = loadData(network.clip.getFinalLayerNorm().gamma, weightMap, 1, "text_model.final_layer_norm.weight");
        network.clip.getFinalLayerNorm().beta = loadData(network.clip.getFinalLayerNorm().beta, weightMap, 1, "text_model.final_layer_norm.bias");
    }
    
    public static void loadWeight(Map<String, Object> weightMap, ClipTextModel network, String modelKey, boolean showLayers) {
        if (showLayers) {
            for (String key : weightMap.keySet()) {
                System.out.println(key);
            }
        }
        /**
         * embeddings

         */
        loadData(network.clip.getEmbeddings().getTokenEmbedding().weight, weightMap, "embeddings.token_embedding.weight");
        loadData(network.clip.getEmbeddings().getPositionEmbedding().weight, weightMap, "embeddings.position_embedding.weight");
        /**
         * encoders

         */
        for (int i = 0; i < 12; i++) {
            /**
             * self attention

             */
            loadData(network.clip.getEncoders().get(i).getAttn().getqLinerLayer().weight, weightMap, modelKey + "encoder.layers." + i + ".self_attn.q_proj.weight");
            loadData(network.clip.getEncoders().get(i).getAttn().getqLinerLayer().bias, weightMap, modelKey + "encoder.layers." + i + ".self_attn.q_proj.bias");
            loadData(network.clip.getEncoders().get(i).getAttn().getkLinerLayer().weight, weightMap, modelKey + "encoder.layers." + i + ".self_attn.k_proj.weight");
            loadData(network.clip.getEncoders().get(i).getAttn().getkLinerLayer().bias, weightMap, modelKey + "encoder.layers." + i + ".self_attn.k_proj.bias");
            loadData(network.clip.getEncoders().get(i).getAttn().getvLinerLayer().weight, weightMap, modelKey + "encoder.layers." + i + ".self_attn.v_proj.weight");
            loadData(network.clip.getEncoders().get(i).getAttn().getvLinerLayer().bias, weightMap, modelKey + "encoder.layers." + i + ".self_attn.v_proj.bias");
            loadData(network.clip.getEncoders().get(i).getAttn().getoLinerLayer().weight, weightMap, modelKey + "encoder.layers." + i + ".self_attn.out_proj.weight");
            loadData(network.clip.getEncoders().get(i).getAttn().getoLinerLayer().bias, weightMap, modelKey + "encoder.layers." + i + ".self_attn.out_proj.bias");
            /**
             * layer norm1

             */
            network.clip.getEncoders().get(i).getNorm1().gamma = loadData(network.clip.getEncoders().get(i).getNorm1().gamma, weightMap, 1, modelKey + "encoder.layers." + i + ".layer_norm1.weight");
            network.clip.getEncoders().get(i).getNorm1().beta = loadData(network.clip.getEncoders().get(i).getNorm1().beta, weightMap, 1, modelKey + "encoder.layers." + i + ".layer_norm1.bias");
            /**
             * mlp

             */
            loadData(network.clip.getEncoders().get(i).getMlp().getLinear1().weight, weightMap, modelKey + "encoder.layers." + i + ".mlp.fc1.weight");
            loadData(network.clip.getEncoders().get(i).getMlp().getLinear1().bias, weightMap, modelKey + "encoder.layers." + i + ".mlp.fc1.bias");
            loadData(network.clip.getEncoders().get(i).getMlp().getLinear2().weight, weightMap, modelKey + "encoder.layers." + i + ".mlp.fc2.weight");
            loadData(network.clip.getEncoders().get(i).getMlp().getLinear2().bias, weightMap, modelKey + "encoder.layers." + i + ".mlp.fc2.bias");
            /**
             * layer norm2

             */
            network.clip.getEncoders().get(i).getNorm2().gamma = loadData(network.clip.getEncoders().get(i).getNorm2().gamma, weightMap, 1, modelKey + "encoder.layers." + i + ".layer_norm2.weight");
            network.clip.getEncoders().get(i).getNorm2().beta = loadData(network.clip.getEncoders().get(i).getNorm2().beta, weightMap, 1, modelKey + "encoder.layers." + i + ".layer_norm2.bias");
        }
        /**
         * post_layernorm

         */
        network.clip.getFinalLayerNorm().gamma = loadData(network.clip.getFinalLayerNorm().gamma, weightMap, 1, modelKey + "final_layer_norm.weight");
        network.clip.getFinalLayerNorm().beta = loadData(network.clip.getFinalLayerNorm().beta, weightMap, 1, modelKey + "final_layer_norm.bias");
    }

    public static void loadWeight(Map<String, Object> weightMap, ClipVision network, boolean showLayers) {
        if (showLayers) {
            for (String key : weightMap.keySet()) {
                System.out.println(key);
            }
        }
        /**
         * embeddings

         */
        loadData(network.getEncoder().getEmbeddings().getClassEmbedding(), weightMap, "embeddings.class_embedding");
        loadData(network.getEncoder().getEmbeddings().getPatchEmbedding().weight, weightMap, "embeddings.patch_embedding.weight");
        loadData(network.getEncoder().getEmbeddings().getPositionEmbedding().weight, weightMap, "embeddings.position_embedding.weight");
        /**
         * pre_layernorm

         */
        network.getEncoder().getPreLayrnorm().gamma = loadData(network.getEncoder().getPreLayrnorm().gamma, weightMap, 1, "pre_layrnorm.weight");
        network.getEncoder().getPreLayrnorm().beta = loadData(network.getEncoder().getPreLayrnorm().beta, weightMap, 1, "pre_layrnorm.bias");
        /**
         * encoders

         */
        for (int i = 0; i < 12; i++) {
            /**
             * attn

             */
            loadData(network.getEncoder().getEncoders().get(i).getAttn().getqLinerLayer().weight, weightMap, "encoder.layers." + i + ".self_attn.q_proj.weight");
            loadData(network.getEncoder().getEncoders().get(i).getAttn().getqLinerLayer().bias, weightMap, "encoder.layers." + i + ".self_attn.q_proj.bias");
            loadData(network.getEncoder().getEncoders().get(i).getAttn().getkLinerLayer().weight, weightMap, "encoder.layers." + i + ".self_attn.k_proj.weight");
            loadData(network.getEncoder().getEncoders().get(i).getAttn().getkLinerLayer().bias, weightMap, "encoder.layers." + i + ".self_attn.k_proj.bias");
            loadData(network.getEncoder().getEncoders().get(i).getAttn().getvLinerLayer().weight, weightMap, "encoder.layers." + i + ".self_attn.v_proj.weight");
            loadData(network.getEncoder().getEncoders().get(i).getAttn().getvLinerLayer().bias, weightMap, "encoder.layers." + i + ".self_attn.v_proj.bias");
            loadData(network.getEncoder().getEncoders().get(i).getAttn().getoLinerLayer().weight, weightMap, "encoder.layers." + i + ".self_attn.out_proj.weight");
            loadData(network.getEncoder().getEncoders().get(i).getAttn().getoLinerLayer().bias, weightMap, "encoder.layers." + i + ".self_attn.out_proj.bias");
            /**
             * ln1

             */
            network.getEncoder().getEncoders().get(i).getNorm1().gamma = loadData(network.getEncoder().getEncoders().get(i).getNorm1().gamma, weightMap, 1, "encoder.layers." + i + ".layer_norm1.weight");
            network.getEncoder().getEncoders().get(i).getNorm1().beta = loadData(network.getEncoder().getEncoders().get(i).getNorm1().beta, weightMap, 1, "encoder.layers." + i + ".layer_norm1.bias");
            /**
             * mlp

             */
            loadData(network.getEncoder().getEncoders().get(i).getMlp().getLinear1().weight, weightMap, "encoder.layers." + i + ".mlp.fc1.weight");
            loadData(network.getEncoder().getEncoders().get(i).getMlp().getLinear1().bias, weightMap, "encoder.layers." + i + ".mlp.fc1.bias");
            loadData(network.getEncoder().getEncoders().get(i).getMlp().getLinear2().weight, weightMap, "encoder.layers." + i + ".mlp.fc2.weight");
            loadData(network.getEncoder().getEncoders().get(i).getMlp().getLinear2().bias, weightMap, "encoder.layers." + i + ".mlp.fc2.bias");
            /**
             * ln2

             */
            network.getEncoder().getEncoders().get(i).getNorm2().gamma = loadData(network.getEncoder().getEncoders().get(i).getNorm2().gamma, weightMap, 1, "encoder.layers." + i + ".layer_norm2.weight");
            network.getEncoder().getEncoders().get(i).getNorm2().beta = loadData(network.getEncoder().getEncoders().get(i).getNorm2().beta, weightMap, 1, "encoder.layers." + i + ".layer_norm2.bias");
            //			network.getEncoder().getEncoders().get(i).getNorm2().gamma.showShape();
        }
        /**
         * post_layernorm

         */
        network.getEncoder().getPostLayernorm().gamma = loadData(network.getEncoder().getPostLayernorm().gamma, weightMap, 1, "post_layernorm.weight");
        network.getEncoder().getPostLayernorm().beta = loadData(network.getEncoder().getPostLayernorm().beta, weightMap, 1, "post_layernorm.bias");
    }

    public static void loadWeight(Map<String, Object> weightMap, ClipText network, boolean showLayers) {
        if (showLayers) {
            for (String key : weightMap.keySet()) {
                System.out.println(key);
            }
        }
        /**
         * text_projection

         */
        loadData(network.textProjection, weightMap, "text_projection");
        /**
         * bert.embeddings

         */
        loadData(network.bert.embeddings.wordEmbeddings.weight, weightMap, "bert.embeddings.word_embeddings.weight");
        loadData(network.bert.embeddings.positionEmbeddings.weight, weightMap, "bert.embeddings.position_embeddings.weight");
        loadData(network.bert.embeddings.tokenTypeEmbeddings.weight, weightMap, "bert.embeddings.token_type_embeddings.weight");
        network.bert.embeddings.norm.gamma = loadData(network.bert.embeddings.norm.gamma, weightMap, 1, "bert.embeddings.LayerNorm.weight");
        network.bert.embeddings.norm.beta = loadData(network.bert.embeddings.norm.beta, weightMap, 1, "bert.embeddings.LayerNorm.bias");
        /**
         * bert.encoder

         */
        for (int i = 0; i < 12; i++) {
            BertLayer bl = network.bert.encoder.layers.get(i);
            /**
             * attention

             */
            loadData(bl.attn.attn.getqLinerLayer().weight, weightMap, "bert.encoder.layer." + i + ".attention.self.query.weight");
            loadData(bl.attn.attn.getqLinerLayer().bias, weightMap, "bert.encoder.layer." + i + ".attention.self.query.bias");
            loadData(bl.attn.attn.getkLinerLayer().weight, weightMap, "bert.encoder.layer." + i + ".attention.self.key.weight");
            loadData(bl.attn.attn.getkLinerLayer().bias, weightMap, "bert.encoder.layer." + i + ".attention.self.key.bias");
            loadData(bl.attn.attn.getvLinerLayer().weight, weightMap, "bert.encoder.layer." + i + ".attention.self.value.weight");
            loadData(bl.attn.attn.getvLinerLayer().bias, weightMap, "bert.encoder.layer." + i + ".attention.self.value.bias");
            /**
             * attention output

             */
            loadData(bl.attn.out.linear.weight, weightMap, "bert.encoder.layer." + i + ".attention.output.dense.weight");
            loadData(bl.attn.out.linear.bias, weightMap, "bert.encoder.layer." + i + ".attention.output.dense.bias");
            bl.attn.out.norm.gamma = loadData(bl.attn.out.norm.gamma, weightMap, 1, "bert.encoder.layer." + i + ".attention.output.LayerNorm.weight");
            bl.attn.out.norm.beta = loadData(bl.attn.out.norm.beta, weightMap, 1, "bert.encoder.layer." + i + ".attention.output.LayerNorm.bias");
            /**
             * intermediate

             */
            loadData(bl.inter.linear.weight, weightMap, "bert.encoder.layer." + i + ".intermediate.dense.weight");
            loadData(bl.inter.linear.bias, weightMap, "bert.encoder.layer." + i + ".intermediate.dense.bias");
            /**
             * output

             */
            loadData(bl.out.linear.weight, weightMap, "bert.encoder.layer." + i + ".output.dense.weight");
            loadData(bl.out.linear.bias, weightMap, "bert.encoder.layer." + i + ".output.dense.bias");
            bl.out.norm.gamma = loadData(bl.out.norm.gamma, weightMap, 1, "bert.encoder.layer." + i + ".output.LayerNorm.weight");
            bl.out.norm.beta = loadData(bl.out.norm.beta, weightMap, 1, "bert.encoder.layer." + i + ".output.LayerNorm.bias");
        }
    }

    @SuppressWarnings("unchecked")
    public static void loadData(Tensor x, Map<String, Object> weightMap, String key) {
        Object meta = weightMap.get(key);
        if (meta != null) {
            int dim = getDim(x);
            if (dim == 1) {
                List<Double> dataA = (List<Double>) meta;
                for (int n = 0; n < dataA.size(); n++) {
                    x.data[n] = dataA.get(n).floatValue();
                }
            } else if (dim == 2) {
                List<List<Double>> dataA = (List<List<Double>>) meta;
                //				x.showShape();
                //				System.out.println(dataA.size()+":"+dataA.get(0).size());
                for (int n = 0; n < dataA.size(); n++) {
                    for (int w = 0; w < dataA.get(n).size(); w++) {
                        x.data[n * dataA.get(n).size() + w] = dataA.get(n).get(w).floatValue();
                    }
                }
            } else if (dim == 3) {
                float[][][] data = (float[][][]) meta;
                x.data = MatrixUtils.transform(data);
            }else {
                List<List<List<List<Double>>>> dataA = (List<List<List<List<Double>>>>) meta;
                int N = dataA.size();
                int C = dataA.get(0).size();
                int H = dataA.get(0).get(0).size();
                int W = dataA.get(0).get(0).get(0).size();
                if(!x.checkShape(N, C, H, W)) {
                	throw new RuntimeException("the tensor shape" + JsonUtils.toJson(x.shape())+" is not shape:["+ N + "," + C + "," + H + "," + W +"]。");
                }
                for (int n = 0; n < N; n++) {
                    for (int c = 0; c < C; c++) {
                        for (int h = 0; h < H; h++) {
                            for (int w = 0; w < W; w++) {
                                x.data[n * x.getOnceSize() + c * H * W + h * W + w] = dataA.get(n).get(c).get(h).get(w).floatValue();
                            }
                        }
                    }
                }
            }
            x.hostToDevice();
            System.out.println(key + "_finish.");
        }else {
        	System.err.println(key+" is null.");
        }
    }
    
    public static void loadData(Tensor x, Map<String, Object> weightMap, String key,int dim) {
        Object meta = weightMap.get(key);
        if (meta != null) {
        	if (dim == 1) {
                List<Double> dataA = (List<Double>) meta;
                for (int n = 0; n < dataA.size(); n++) {
                    x.data[n] = dataA.get(n).floatValue();
                }
            } else if (dim == 2) {
                List<List<Double>> dataA = (List<List<Double>>) meta;
                //				x.showShape();
                //				System.out.println(dataA.size()+":"+dataA.get(0).size());
                for (int n = 0; n < dataA.size(); n++) {
                    for (int w = 0; w < dataA.get(n).size(); w++) {
                        x.data[n * dataA.get(n).size() + w] = dataA.get(n).get(w).floatValue();
                    }
                }
            } else if (dim == 3) {
            	 List<List<List<Double>>> dataA = (List<List<List<Double>>>) meta;
            	 int N = dataA.size();
                 int C = dataA.get(0).size();
                 int W = dataA.get(0).get(0).size();
                 if(!x.checkShape(N, C, 1, W)) {
                 	throw new RuntimeException("the tensor shape" + JsonUtils.toJson(x.shape())+" is not shape:["+ N + "," + C + "," + 1 + "," + W +"]。");
                 }
            	 int tw = dataA.get(0).size() * dataA.get(0).get(0).size();
            	 for (int n = 0; n < dataA.size(); n++) {
            		 for(int t = 0;t < dataA.get(n).size();t++) {
            			 for (int w = 0; w < dataA.get(n).get(t).size(); w++) {
                             x.data[n * tw + t * dataA.get(0).get(0).size() + w] = dataA.get(n).get(t).get(w).floatValue();
                         }
            		 }
                 }
            } else if (dim == 5) {
            	List<List<List<List<List<Double>>>>> data = (List<List<List<List<List<Double>>>>>) meta;
                x.data = MatrixUtils.transform(data);
            }  else {
                List<List<List<List<Double>>>> dataA = (List<List<List<List<Double>>>>) meta;
                int N = dataA.size();
                int C = dataA.get(0).size();
                int H = dataA.get(0).get(0).size();
                int W = dataA.get(0).get(0).get(0).size();
                for (int n = 0; n < N; n++) {
                    for (int c = 0; c < C; c++) {
                        for (int h = 0; h < H; h++) {
                            for (int w = 0; w < W; w++) {
                                x.data[n * x.getOnceSize() + c * H * W + h * W + w] = dataA.get(n).get(c).get(h).get(w).floatValue();
                            }
                        }
                    }
                }
            }
            x.hostToDevice();
            System.out.println(key + "_finish.");
        }
    }

    @SuppressWarnings("unchecked")
    public static Tensor loadData(Tensor x, Map<String, Object> weightMap, int dim, String key) {
        Object meta = weightMap.get(key);
        if (meta != null) {
            if (dim == 1) {
                List<Double> dataA = (List<Double>) meta;
                x = new Tensor(1, 1, 1, dataA.size(), true);
                int N = dataA.size();
                if(x != null) {
                	if(!x.checkShape(1, 1, 1, N)) {
                    	throw new RuntimeException("the tensor shape" + JsonUtils.toJson(x.shape())+" is not shape:["+ 1 + "," + 1 + "," + 1 + "," + N +"]。");
                    }
                }
                for (int n = 0; n < dataA.size(); n++) {
                    x.data[n] = dataA.get(n).floatValue();
                }
            } else if (dim == 2) {
                List<List<Double>> dataA = (List<List<Double>>) meta;
                x = new Tensor(dataA.size(), 1, 1, dataA.get(0).size(), true);
                for (int n = 0; n < dataA.size(); n++) {
                    for (int w = 0; w < dataA.get(n).size(); w++) {
                        x.data[n * dataA.get(n).size() + w] = dataA.get(n).get(w).floatValue();
                    }
                }
                //				float[][] data = (float[][]) meta;
                //				x.data = MatrixUtils.transform(data);
            } else if (dim == 3) {
                float[][][] data = (float[][][]) meta;
                x.data = MatrixUtils.transform(data);
            } else {
                List<List<List<List<Double>>>> dataA = (List<List<List<List<Double>>>>) meta;
                int N = dataA.size();
                int C = dataA.get(0).size();
                int H = dataA.get(0).get(0).size();
                int W = dataA.get(0).get(0).get(0).size();
                x = new Tensor(N, C, H, W, true);
                if(x != null) {
                	if(!x.checkShape(N , C, H, W)) {
                    	throw new RuntimeException("the tensor shape" + JsonUtils.toJson(x.shape())+" is not shape:["+ N + "," + C + "," + H + "," + W +"]。");
                    }
                }
                for (int n = 0; n < dataA.size(); n++) {
                    for (int c = 0; c < dataA.get(n).size(); c++) {
                        for (int h = 0; h < dataA.get(n).get(c).size(); h++) {
                            for (int w = 0; w < dataA.get(n).get(c).get(h).size(); w++) {
                                x.data[n * x.getOnceSize() + c * x.height * x.width + h * x.width + w] = dataA.get(n).get(c).get(h).get(w).floatValue();
                            }
                        }
                    }
                }
            }
            x.hostToDevice();
            System.out.println(key + "_finish.");
            return x;
        }
        return null;
    }

    public static int getDim(Tensor x) {
        int dim = 0;
        if (x.number > 1) {
            dim++;
        }
        if (x.channel > 1) {
            dim++;
        }
        if (x.height > 1) {
            dim++;
        }
        if (x.width > 1) {
            dim++;
        }
        return dim;
    }
}

