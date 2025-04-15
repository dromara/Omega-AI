package com.omega.example.diffusion.test;

import com.omega.common.config.Tensor;
import com.omega.common.utils.MatrixUtils;
import com.omega.engine.gpu.CUDAMemoryManager;
import com.omega.engine.loss.LossType;
import com.omega.engine.nn.layer.ConvolutionLayer;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.diffusion.ResidualBlockLayer;
import com.omega.engine.nn.layer.diffusion.UpSampleLayer;
import com.omega.engine.nn.layer.normalization.GNLayer;
import com.omega.engine.nn.network.DiffusionUNet;
import com.omega.engine.optimizer.MBSGDOptimizer;
import com.omega.engine.optimizer.lr.LearnRateUpdate;
import com.omega.engine.updater.UpdaterType;
import com.omega.example.diffusion.utils.DiffusionImageDataLoader;

import java.util.List;
import java.util.Map;

public class DiffusionModelTest {
    public static void duffsion_anime() {
        try {
            boolean bias = false;
            int batchSize = 4;
            int imw = 96;
            int imh = 96;
            int mChannel = 64;
            int resBlockNum = 2;
            int T = 1000;
            int[] channelMult = new int[]{1, 2};
            String imgDirPath = "H:\\voc\\gan_anime\\ml2021spring-hw6\\faces\\";
            //			String weightPath = "H:\\voc\\gan_anime\\torch_weights.json";
            DiffusionImageDataLoader dataLoader = new DiffusionImageDataLoader(imgDirPath, imw, imh, batchSize, false);
            DiffusionUNet network = new DiffusionUNet(LossType.MSE, UpdaterType.adamw, T, 3, mChannel, channelMult, resBlockNum, imw, imh, bias);
            network.CUDNN = true;
            network.learnRate = 0.0005f;
            //			Map<String, Object> weightMap = LagJsonReader.readJsonFileSmallWeight(weightPath);
            //
            //			loadWeight(weightMap, network);
            MBSGDOptimizer optimizer = new MBSGDOptimizer(network, 50, 0.00001f, batchSize, LearnRateUpdate.GD_GECAY, false);
            optimizer.trainGaussianDiffusion(dataLoader);
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public static void loadWeight(Map<String, Object> weightMap, DiffusionUNet network) {
        for (String key : weightMap.keySet()) {
            System.out.println(key);
        }
        //time_embedding
        loadData(network.getTemb().emb.weight, weightMap.get("time_embedding.timembedding.0.weight"), "time_embedding.timembedding.0.weight");
        loadData(network.getTemb().linear1.weight, weightMap.get("time_embedding.timembedding.1.weight"), "time_embedding.timembedding.1.weight");
        loadData(network.getTemb().linear2.weight, weightMap.get("time_embedding.timembedding.3.weight"), "time_embedding.timembedding.3.weight");
        //head
        loadData(network.getHead().weight, weightMap.get("head.weight"), "head.weight");
        loadData(network.getHead().bias, weightMap.get("head.bias"), "head.bias");
        //downblocks
        for (int i = 0; i < network.getDownBlocks().size(); i++) {
            Layer layer = network.getDownBlocks().get(i);
            if (layer instanceof ResidualBlockLayer) {
                ResidualBlockLayer rbl = (ResidualBlockLayer) layer;
                //block1.gn
                GNLayer bgn = (GNLayer) rbl.block1[0];
                bgn.gamma = loadData(bgn.gamma, weightMap.get("downblocks." + i + ".block1.0.weight"), 1, "downblocks." + i + ".block1.0.weight");
                bgn.beta = loadData(bgn.beta, weightMap.get("downblocks." + i + ".block1.0.bias"), 1, "downblocks." + i + ".block1.0.bias");
                //block1.conv
                loadData(rbl.block1[2].weight, weightMap.get("downblocks." + i + ".block1.2.weight"), "downblocks." + i + ".block1.2.weight");
                //temb
                loadData(rbl.temb_proj[1].weight, weightMap.get("downblocks." + i + ".temb_proj.1.weight"), "downblocks." + i + ".temb_proj.1.weight");
                //block2.gn
                GNLayer bgn2 = (GNLayer) rbl.block2[0];
                bgn2.gamma = loadData(bgn2.gamma, weightMap.get("downblocks." + i + ".block2.0.weight"), 1, "downblocks." + i + ".block2.0.weight");
                bgn2.beta = loadData(bgn2.beta, weightMap.get("downblocks." + i + ".block2.0.bias"), 1, "downblocks." + i + ".block2.0.bias");
                //block2.conv
                loadData(rbl.block2[2].weight, weightMap.get("downblocks." + i + ".block2.3.weight"), "downblocks." + i + ".block2.3.weight");
                //shortcut
                if (rbl.shortcut != null) {
                    rbl.shortcut.weight = loadData(rbl.shortcut.weight, weightMap.get("downblocks." + i + ".shortcut.weight"), 4, "downblocks." + i + ".shortcut.weight");
                }
            } else if (layer instanceof ConvolutionLayer) {
                //down
                loadData(layer.weight, weightMap.get("downblocks." + i + ".main.weight"), "downblocks." + i + ".main.weight");
            }
        }
        //middleblocks1
        ResidualBlockLayer mid1 = network.getMidResBlock1();
        GNLayer mgn = (GNLayer) mid1.block1[0];
        mgn.gamma = loadData(mgn.gamma, weightMap.get("middleblocks.0.block1.0.weight"), 1, "middleblocks.0.block1.0.weight");
        mgn.beta = loadData(mgn.beta, weightMap.get("middleblocks.0.block1.0.bias"), 1, "middleblocks.0.block1.0.bias");
        //block1.conv
        loadData(mid1.block1[2].weight, weightMap.get("middleblocks.0.block1.2.weight"), "middleblocks.0.block1.2.weight");
        //temb
        loadData(mid1.temb_proj[1].weight, weightMap.get("middleblocks.0.temb_proj.1.weight"), "middleblocks.0.temb_proj.1.weight");
        //block2.gn
        GNLayer mgn2 = (GNLayer) mid1.block2[0];
        mgn2.gamma = loadData(mgn2.gamma, weightMap.get("middleblocks.0.block2.0.weight"), 1, "middleblocks.0.block2.0.weight");
        mgn2.beta = loadData(mgn2.beta, weightMap.get("middleblocks.0.block2.0.bias"), 1, "middleblocks.0.block2.0.bias");
        //block2.conv
        loadData(mid1.block2[2].weight, weightMap.get("middleblocks.0.block2.3.weight"), "middleblocks.0.block2.3.weight");
        //att.gn
        mid1.attn.gn.gamma = loadData(mid1.attn.gn.gamma, weightMap.get("middleblocks.0.attn.group_norm.weight"), 1, "middleblocks.0.attn.group_norm.weight");
        mid1.attn.gn.beta = loadData(mid1.attn.gn.beta, weightMap.get("middleblocks.0.attn.group_norm.bias"), 1, "middleblocks.0.attn.group_norm.bias");
        //att.qkvo
        mid1.attn.qLayer.weight = loadData(mid1.attn.qLayer.weight, weightMap.get("middleblocks.0.attn.proj_q.weight"), 4, "middleblocks.0.attn.proj_q.weight");
        mid1.attn.kLayer.weight = loadData(mid1.attn.kLayer.weight, weightMap.get("middleblocks.0.attn.proj_k.weight"), 4, "middleblocks.0.attn.proj_k.weight");
        mid1.attn.vLayer.weight = loadData(mid1.attn.vLayer.weight, weightMap.get("middleblocks.0.attn.proj_v.weight"), 4, "middleblocks.0.attn.proj_v.weight");
        mid1.attn.oLayer.weight = loadData(mid1.attn.oLayer.weight, weightMap.get("middleblocks.0.attn.proj.weight"), 4, "middleblocks.0.attn.proj.weight");
        //middleblocks2
        ResidualBlockLayer mid2 = network.getMidResBlock2();
        GNLayer mbgn1 = (GNLayer) mid2.block1[0];
        mbgn1.gamma = loadData(mbgn1.gamma, weightMap.get("middleblocks.1.block1.0.weight"), 1, "middleblocks.1.block1.0.weight");
        mbgn1.beta = loadData(mbgn1.beta, weightMap.get("middleblocks.1.block1.0.bias"), 1, "middleblocks.1.block1.0.bias");
        //block1.conv
        loadData(mid2.block1[2].weight, weightMap.get("middleblocks.1.block1.2.weight"), "middleblocks.1.block1.2.weight");
        //temb
        loadData(mid2.temb_proj[1].weight, weightMap.get("middleblocks.1.temb_proj.1.weight"), "middleblocks.1.temb_proj.1.weight");
        //block2.gn
        GNLayer mbgn2 = (GNLayer) mid2.block2[0];
        mbgn2.gamma = loadData(mbgn2.gamma, weightMap.get("middleblocks.1.block2.0.weight"), 1, "middleblocks.1.block2.0.weight");
        mbgn2.beta = loadData(mbgn2.beta, weightMap.get("middleblocks.1.block2.0.bias"), 1, "middleblocks.1.block2.0.bias");
        //block2.conv
        loadData(mid2.block2[2].weight, weightMap.get("middleblocks.1.block2.3.weight"), "middleblocks.1.block2.3.weight");
        //upblocks
        int ri = 0;
        for (int i = 0; i < network.getUpBlocks().size(); i++) {
            Layer layer = network.getUpBlocks().get(i);
            System.err.println(layer);
            if (layer instanceof ResidualBlockLayer) {
                ResidualBlockLayer rbl = (ResidualBlockLayer) layer;
                System.out.println(rbl + ":" + rbl.channel);
                //block1.gn
                GNLayer bgn = (GNLayer) rbl.block1[0];
                //				System.out.println(weightMap.get("upblocks."+ri+".block1.0.weight"));
                bgn.gamma = loadData(bgn.gamma, weightMap.get("upblocks." + ri + ".block1.0.weight"), 1, "upblocks." + ri + ".block1.0.weight");
                bgn.beta = loadData(bgn.beta, weightMap.get("upblocks." + ri + ".block1.0.bias"), 1, "upblocks." + ri + ".block1.0.bias");
                //block1.conv
                loadData(rbl.block1[2].weight, weightMap.get("upblocks." + ri + ".block1.2.weight"), "upblocks." + ri + ".block1.2.weight");
                //temb
                loadData(rbl.temb_proj[1].weight, weightMap.get("upblocks." + ri + ".temb_proj.1.weight"), "upblocks." + ri + ".temb_proj.1.weight");
                //block2.gn
                GNLayer bgn2 = (GNLayer) rbl.block2[0];
                bgn2.gamma = loadData(bgn2.gamma, weightMap.get("upblocks." + ri + ".block2.0.weight"), 1, "upblocks." + ri + ".block2.0.weight");
                bgn2.beta = loadData(bgn2.beta, weightMap.get("upblocks." + ri + ".block2.0.bias"), 1, "upblocks." + ri + ".block2.0.bias");
                //block2.conv
                loadData(rbl.block2[2].weight, weightMap.get("upblocks." + ri + ".block2.3.weight"), "upblocks." + ri + ".block2.3.weight");
                //shortcut
                if (rbl.shortcut != null) {
                    rbl.shortcut.weight = loadData(rbl.shortcut.weight, weightMap.get("upblocks." + ri + ".shortcut.weight"), 4, "upblocks." + ri + ".shortcut.weight");
                }
                System.out.println(ri + "_loaded.");
                ri++;
            } else if (layer instanceof UpSampleLayer) {
                //up
                UpSampleLayer up = (UpSampleLayer) layer;
                loadData(up.conv.weight, weightMap.get("upblocks." + ri + ".main.weight"), "upblocks." + ri + ".main.weight");
                ri++;
            }
        }
        //tail
        network.getGn().gamma = loadData(network.getGn().gamma, weightMap.get("tail.0.weight"), 1, "tail.0.weight");
        network.getGn().beta = loadData(network.getGn().beta, weightMap.get("tail.0.bias"), 1, "tail.0.bias");
        loadData(network.getConv().weight, weightMap.get("tail.2.weight"), "tail.2.weight");
    }

    public static void loadData(Tensor x, Object meta, String key) {
        if (meta != null) {
            int dim = getDim(x);
            if (dim == 1) {
                List<Double> dataA = (List<Double>) meta;
                for (int n = 0; n < dataA.size(); n++) {
                    x.data[n] = dataA.get(n).floatValue();
                }
            } else if (dim == 2) {
                List<List<Double>> dataA = (List<List<Double>>) meta;
                for (int n = 0; n < dataA.size(); n++) {
                    for (int w = 0; w < dataA.get(n).size(); w++) {
                        x.data[n * dataA.get(n).size() + w] = dataA.get(n).get(w).floatValue();
                    }
                }
            } else if (dim == 3) {
                float[][][] data = (float[][][]) meta;
                x.data = MatrixUtils.transform(data);
            } else {
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

    public static Tensor loadData(Tensor x, Object meta, int dim, String key) {
        if (meta != null) {
            if (dim == 1) {
                List<Double> dataA = (List<Double>) meta;
                x = new Tensor(1, 1, 1, dataA.size(), true);
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

    public static void main(String[] args) {
        try {
            //			CUDAModules.initContext();
            duffsion_anime();
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        } finally {
            // TODO: handle finally clause
            CUDAMemoryManager.free();
        }
    }
}

