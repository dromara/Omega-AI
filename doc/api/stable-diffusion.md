# Stable Diffusion 稳定扩散模型

本页只记录 Omega-AI 当前源码中真实存在的实现入口和阅读路径。API 示例必须以源码为准，不使用伪类名或非项目实现的示例代码。

[[toc]]

---

## 一、源码入口

| 类型 | 位置 |
| --- | --- |
| 主要实现 | `com.omega.engine.nn.network.vae.SD_VAE / DiffusionUNetCond` |
| 源码路径 | `src/main/java/com/omega/engine/nn/network/vae/SD_VAE.java` |
| 示例入口 | `src/main/java/com/omega/example/sd/test/SDTest.java` |
| 示例代码（Demo） | `SDTest.sd_train_pokem()` |

## 二、入口代码片段

节选自 `src/main/java/com/omega/engine/nn/network/vae/SD_VAE.java`，仅保留入口签名和关键初始化，完整实现以源码为准：

```java
public class SD_VAE extends Network {
    public float beta = 0.25f;
    public float decay = 0.999f;

    public SD_VAE(LossType lossType, UpdaterType updater, int latendDim,
                  int num_vq_embeddings, int imageSize, int[] ch_mult,
                  int ch, int num_res_blocks, boolean double_z) {
        this.lossFunction = LossFactory.create(lossType, this);
        this.latendDim = latendDim;
        this.num_vq_embeddings = num_vq_embeddings;
    }
}
```

## 三、示例代码（Demo）

节选自 `src/main/java/com/omega/example/sd/test/SDTest.java`，保留 Stable Diffusion 训练入口的关键初始化和训练调用：

```java
public class SDTest {
    public static void sd_train_pokem() throws Exception {
        String tokenizerPath = "H:\\clip\\CLIP\\clip_cn\\vocab.txt";
        String labelPath = "H:\\vae_dataset\\pokemon-blip\\data.json";
        String imgDirPath = "H:\\vae_dataset\\pokemon-blip\\dataset256\\";
        boolean horizontalFilp = true;
        int imgSize = 256;
        int maxContextLen = 64;
        int batchSize = 1;
        float[] mean = new float[]{0.5f, 0.5f, 0.5f};
        float[] std = new float[]{0.5f, 0.5f, 0.5f};
        SDImageDataLoader dataLoader = new SDImageDataLoader(tokenizerPath, labelPath, imgDirPath, imgSize, imgSize, maxContextLen, batchSize, horizontalFilp, mean, std);

        TinyVQVAE2 vae = new TinyVQVAE2(LossType.MSE, UpdaterType.adamw, 32, 4, 512, 256,
                new int[]{64, 128, 256}, new boolean[]{false, false, false}, 2);
        vae.CUDNN = true;
        vae.learnRate = 0.001f;
        vae.RUN_MODEL = RunModel.EVAL;
        ModelUtils.loadModel(vae, "H:\\model\\vqvae2_32_256_500.model");

        ClipText clip = new ClipText(LossType.MSE, UpdaterType.adamw, 12, maxContextLen, 21128, 768, 512, 512, 2, 3072, 12);
        clip.CUDNN = true;
        clip.time = maxContextLen;
        clip.RUN_MODEL = RunModel.EVAL;
        ModeLoaderlUtils.loadWeight(LagJsonReader.readJsonFileSmallWeight("H:\\model\\clip_cn_vit-b-16.json"), clip, true);

        DiffusionUNetCond unet = new DiffusionUNetCond(LossType.MSE, UpdaterType.adamw, 4, 64, 64, 128, 8,
                new int[]{32, 48, 64}, new int[]{64, 48}, new boolean[]{true, true}, 1, 1, 1,
                1000, 512, 512, maxContextLen, true, new boolean[]{true, true});
        unet.CUDNN = true;
        unet.learnRate = 0.001f;

        MBSGDOptimizer optimizer = new MBSGDOptimizer(unet, 500, 0.00001f, batchSize, LearnRateUpdate.GD_GECAY, false);
        optimizer.trainSD(dataLoader, vae, clip);
    }
}
```

## 四、相关组件

`SD_VAE、DiffusionUNetCond、UNetCrossAttentionLayer、TimeEmbeddingLayer、DiffusionImageLoader`

## 五、阅读建议

1. 先打开示例入口，查看 `main` 方法或训练方法中如何准备数据、创建网络和启动训练。
2. 再进入主要实现类，查看构造函数、`init`、`forward`、`back`、`loss`、`update` 等方法。
3. 如果涉及 GPU 加速，继续追踪对应 layer、kernel wrapper 和 `src/main/resources/cu` 下的 CUDA 实现。
4. 文档中的类名、构造参数和调用方式必须与当前源码保持一致；如果源码变更，以源码为准同步更新本文档。

## 六、常见注意事项

- 示例中的数据集路径通常是作者本机路径，运行前需要改成本机目录。
- 训练 batch size、学习率、是否启用 CUDNN/CUDA 需要结合显存和任务规模调整。
- 不同模型的 loss、label 格式和输出 shape 不同，不能直接混用其它模型示例。
- 若需要补充代码片段，应直接从对应示例类中摘取真实代码，并标注来源方法。
