package com.omega.example.vae.test;

import com.omega.engine.gpu.CUDAMemoryManager;
import com.omega.engine.gpu.CUDAModules;
import com.omega.engine.loss.LossType;
import com.omega.engine.nn.layer.lpips.NetLinLayer;
import com.omega.engine.nn.layer.lpips.VGGBlock;
import com.omega.engine.nn.network.vqgan.LPIPS;
import com.omega.engine.nn.network.vqgan.Opensora_LPIPS;
import com.omega.engine.optimizer.MBSGDOptimizer;
import com.omega.engine.optimizer.lr.LearnRateUpdate;
import com.omega.engine.updater.UpdaterType;
import com.omega.example.clip.utils.ClipModelUtils;
import com.omega.example.diffusion.utils.DiffusionImageDataLoader;
import com.omega.example.transformer.utils.LagJsonReader;

import java.util.Map;

public class LPIPSTest {
	
    public static void loadLPIPSWeight(Map<String, Object> weightMap, LPIPS network, boolean showLayers) {
        if (showLayers) {
            for (String key : weightMap.keySet()) {
                System.out.println(key);
            }
            for (int i = 0; i < network.lpips.vgg.features.size(); i++) {
                System.out.println(network.lpips.vgg.features.get(i) + "===>" + i);
            }
        }
        /**
         * vgg slice
         */
        VGGBlock block0 = (VGGBlock) network.lpips.vgg.features.get(0);
        ClipModelUtils.loadData(block0.conv.weight, weightMap, "net.slice1.0.weight");
        ClipModelUtils.loadData(block0.conv.bias, weightMap, "net.slice1.0.bias");
        VGGBlock block2 = (VGGBlock) network.lpips.vgg.features.get(1);
        ClipModelUtils.loadData(block2.conv.weight, weightMap, "net.slice1.2.weight");
        ClipModelUtils.loadData(block2.conv.bias, weightMap, "net.slice1.2.bias");
        VGGBlock block5 = (VGGBlock) network.lpips.vgg.features.get(3);
        ClipModelUtils.loadData(block5.conv.weight, weightMap, "net.slice2.5.weight");
        ClipModelUtils.loadData(block5.conv.bias, weightMap, "net.slice2.5.bias");
        VGGBlock block7 = (VGGBlock) network.lpips.vgg.features.get(4);
        ClipModelUtils.loadData(block7.conv.weight, weightMap, "net.slice2.7.weight");
        ClipModelUtils.loadData(block7.conv.bias, weightMap, "net.slice2.7.bias");
        VGGBlock block10 = (VGGBlock) network.lpips.vgg.features.get(6);
        ClipModelUtils.loadData(block10.conv.weight, weightMap, "net.slice3.10.weight");
        ClipModelUtils.loadData(block10.conv.bias, weightMap, "net.slice3.10.bias");
        VGGBlock block12 = (VGGBlock) network.lpips.vgg.features.get(7);
        ClipModelUtils.loadData(block12.conv.weight, weightMap, "net.slice3.12.weight");
        ClipModelUtils.loadData(block12.conv.bias, weightMap, "net.slice3.12.bias");
        VGGBlock block14 = (VGGBlock) network.lpips.vgg.features.get(8);
        ClipModelUtils.loadData(block14.conv.weight, weightMap, "net.slice3.14.weight");
        ClipModelUtils.loadData(block14.conv.bias, weightMap, "net.slice3.14.bias");
        VGGBlock block17 = (VGGBlock) network.lpips.vgg.features.get(10);
        ClipModelUtils.loadData(block17.conv.weight, weightMap, "net.slice4.17.weight");
        ClipModelUtils.loadData(block17.conv.bias, weightMap, "net.slice4.17.bias");
        VGGBlock block19 = (VGGBlock) network.lpips.vgg.features.get(11);
        ClipModelUtils.loadData(block19.conv.weight, weightMap, "net.slice4.19.weight");
        ClipModelUtils.loadData(block19.conv.bias, weightMap, "net.slice4.19.bias");
        VGGBlock block21 = (VGGBlock) network.lpips.vgg.features.get(12);
        ClipModelUtils.loadData(block21.conv.weight, weightMap, "net.slice4.21.weight");
        ClipModelUtils.loadData(block21.conv.bias, weightMap, "net.slice4.21.bias");
        VGGBlock block24 = (VGGBlock) network.lpips.vgg.features.get(14);
        ClipModelUtils.loadData(block24.conv.weight, weightMap, "net.slice5.24.weight");
        ClipModelUtils.loadData(block24.conv.bias, weightMap, "net.slice5.24.bias");
        VGGBlock block26 = (VGGBlock) network.lpips.vgg.features.get(15);
        ClipModelUtils.loadData(block26.conv.weight, weightMap, "net.slice5.26.weight");
        ClipModelUtils.loadData(block26.conv.bias, weightMap, "net.slice5.26.bias");
        VGGBlock block28 = (VGGBlock) network.lpips.vgg.features.get(16);
        ClipModelUtils.loadData(block28.conv.weight, weightMap, "net.slice5.28.weight");
        ClipModelUtils.loadData(block28.conv.bias, weightMap, "net.slice5.28.bias");
        for (int i = 0; i < 5; i++) {
            NetLinLayer nl = network.lpips.lins.get(i);
            nl.conv.weight = ClipModelUtils.loadData(nl.conv.weight, weightMap, 4, "lin" + i + ".model.1.weight");
        }
    }
    
    public static void loadLPIPSWeight(Map<String, Object> weightMap, Opensora_LPIPS network, boolean showLayers) {
        if (showLayers) {
            for (String key : weightMap.keySet()) {
                System.out.println(key);
            }
            for (int i = 0; i < network.lpips.vgg.features.size(); i++) {
                System.out.println(network.lpips.vgg.features.get(i) + "===>" + i);
            }
        }
        /**
         * vgg slice

         */
        VGGBlock block0 = (VGGBlock) network.lpips.vgg.features.get(0);
        ClipModelUtils.loadData(block0.conv.weight, weightMap, "net.slice1.0.weight");
        ClipModelUtils.loadData(block0.conv.bias, weightMap, "net.slice1.0.bias");
        VGGBlock block2 = (VGGBlock) network.lpips.vgg.features.get(1);
        ClipModelUtils.loadData(block2.conv.weight, weightMap, "net.slice1.2.weight");
        ClipModelUtils.loadData(block2.conv.bias, weightMap, "net.slice1.2.bias");
        VGGBlock block5 = (VGGBlock) network.lpips.vgg.features.get(3);
        ClipModelUtils.loadData(block5.conv.weight, weightMap, "net.slice2.5.weight");
        ClipModelUtils.loadData(block5.conv.bias, weightMap, "net.slice2.5.bias");
        VGGBlock block7 = (VGGBlock) network.lpips.vgg.features.get(4);
        ClipModelUtils.loadData(block7.conv.weight, weightMap, "net.slice2.7.weight");
        ClipModelUtils.loadData(block7.conv.bias, weightMap, "net.slice2.7.bias");
        VGGBlock block10 = (VGGBlock) network.lpips.vgg.features.get(6);
        ClipModelUtils.loadData(block10.conv.weight, weightMap, "net.slice3.10.weight");
        ClipModelUtils.loadData(block10.conv.bias, weightMap, "net.slice3.10.bias");
        VGGBlock block12 = (VGGBlock) network.lpips.vgg.features.get(7);
        ClipModelUtils.loadData(block12.conv.weight, weightMap, "net.slice3.12.weight");
        ClipModelUtils.loadData(block12.conv.bias, weightMap, "net.slice3.12.bias");
        VGGBlock block14 = (VGGBlock) network.lpips.vgg.features.get(8);
        ClipModelUtils.loadData(block14.conv.weight, weightMap, "net.slice3.14.weight");
        ClipModelUtils.loadData(block14.conv.bias, weightMap, "net.slice3.14.bias");
        VGGBlock block17 = (VGGBlock) network.lpips.vgg.features.get(10);
        ClipModelUtils.loadData(block17.conv.weight, weightMap, "net.slice4.17.weight");
        ClipModelUtils.loadData(block17.conv.bias, weightMap, "net.slice4.17.bias");
        VGGBlock block19 = (VGGBlock) network.lpips.vgg.features.get(11);
        ClipModelUtils.loadData(block19.conv.weight, weightMap, "net.slice4.19.weight");
        ClipModelUtils.loadData(block19.conv.bias, weightMap, "net.slice4.19.bias");
        VGGBlock block21 = (VGGBlock) network.lpips.vgg.features.get(12);
        ClipModelUtils.loadData(block21.conv.weight, weightMap, "net.slice4.21.weight");
        ClipModelUtils.loadData(block21.conv.bias, weightMap, "net.slice4.21.bias");
        VGGBlock block24 = (VGGBlock) network.lpips.vgg.features.get(14);
        ClipModelUtils.loadData(block24.conv.weight, weightMap, "net.slice5.24.weight");
        ClipModelUtils.loadData(block24.conv.bias, weightMap, "net.slice5.24.bias");
        VGGBlock block26 = (VGGBlock) network.lpips.vgg.features.get(15);
        ClipModelUtils.loadData(block26.conv.weight, weightMap, "net.slice5.26.weight");
        ClipModelUtils.loadData(block26.conv.bias, weightMap, "net.slice5.26.bias");
        VGGBlock block28 = (VGGBlock) network.lpips.vgg.features.get(16);
        ClipModelUtils.loadData(block28.conv.weight, weightMap, "net.slice5.28.weight");
        ClipModelUtils.loadData(block28.conv.bias, weightMap, "net.slice5.28.bias");
        for (int i = 0; i < 5; i++) {
            NetLinLayer nl = network.lpips.lins.get(i);
            nl.conv.weight = ClipModelUtils.loadData(nl.conv.weight, weightMap, 4, "lin" + i + ".model.1.weight");
        }
    }


    public static void lpips() {
        try {
            int batchSize = 4;
            int imageSize = 256;
            float[] mean = new float[]{0.5f, 0.5f, 0.5f};
            float[] std = new float[]{0.5f, 0.5f, 0.5f};
            String imgDirPath = "H:\\vae_dataset\\pokemon-blip\\dataset256\\";
            DiffusionImageDataLoader dataLoader = new DiffusionImageDataLoader(imgDirPath, imageSize, imageSize, batchSize, false, true, mean, std);
            LPIPS network = new LPIPS(LossType.MSE, UpdaterType.adamw, imageSize);
            network.CUDNN = true;
            network.learnRate = 0.001f;
            String lpipsWeight = "H:\\model\\lpips.json";
            loadLPIPSWeight(LagJsonReader.readJsonFileSmallWeight(lpipsWeight), network, true);
            MBSGDOptimizer optimizer = new MBSGDOptimizer(network, 500, 0.00001f, batchSize, LearnRateUpdate.CONSTANT, false);
            //			optimizer.lr_step = new int[] {50, 100, 150, 200, 250, 300, 350, 400, 450};
            optimizer.trainLPIPIS(dataLoader);
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public static void main(String[] args) {
        try {
            CUDAModules.initContext();
            lpips();
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        } finally {
            // TODO: handle finally clause
            CUDAMemoryManager.free();
        }
    }
}

