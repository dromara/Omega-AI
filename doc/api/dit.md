# DiT Diffusion Transformer

本页记录 Omega-AI 当前源码中真实存在的 DiT 实现入口、核心构造函数和 Demo 示例代码。代码片段均来自当前 Omega-AI 源码节选，完整实现以源码文件为准。

[[toc]]

---

## 一、源码入口

| 类型 | 位置 |
| --- | --- |
| 主要实现 | `com.omega.engine.nn.network.dit.DiT` |
| 源码路径 | `src/main/java/com/omega/engine/nn/network/dit/DiT.java` |
| 示例入口 | `src/main/java/com/omega/example/dit/test/OmegaDiT2Test.java` |
| 示例代码（Demo） | `OmegaDiT2Test.omega_sprint_b1_clip_train_flux2vae_v_512()` |

## 二、入口代码片段

节选自 `src/main/java/com/omega/engine/nn/network/dit/DiT.java`，仅保留入口签名和关键初始化，完整实现以源码为准：

```java
public class DiT extends Network {

    public int inChannel;
    public int width;
    public int height;
    public int patchSize;
    public int maxContextLen;
    public int hiddenSize;
    public int headNum;
    public DiTMoudue main;

    public DiT(LossType lossType, UpdaterType updater, int inChannel, int width,
               int height, int patchSize, int hiddenSize, int headNum, int depth,
               int timeSteps, int maxContextLen, int textEmbedDim, int mlpRatio,
               boolean learnSigma) {
        this.lossFunction = LossFactory.create(lossType, this);
        this.updater = updater;
        this.inChannel = inChannel;
        this.width = width;
        this.height = height;
        this.patchSize = patchSize;
        this.headNum = headNum;
        this.hiddenSize = hiddenSize;
        this.depth = depth;
        this.timeSteps = timeSteps;
        this.textEmbedDim = textEmbedDim;
        this.maxContextLen = maxContextLen;
        this.mlpRatio = mlpRatio;
        this.learnSigma = learnSigma;
        this.time = (width / patchSize) * (height / patchSize);
        initLayers();
    }

    public void initLayers() {
        this.inputLayer = new InputLayer(inChannel, height, width);
        main = new DiTMoudue(inChannel, width, height, patchSize, hiddenSize,
                headNum, depth, timeSteps, maxContextLen, textEmbedDim,
                mlpRatio, learnSigma, this);
        this.addLayer(inputLayer);
        this.addLayer(main);
    }
}
```

## 三、示例代码（Demo）

节选自 `src/main/java/com/omega/example/dit/test/OmegaDiT2Test.java`，只保留 512 训练入口代码：

```java
public static void omega_sprint_b1_clip_train_flux2vae_v_512() throws Exception {
    String dataPath = "/root/gpufree-data/6m/flux2vae_latend_512.bin";

    int batchSize = 12;
    int latendDim = 128;
    int height = 32;
    int width = 32;
    int textEmbedDim = 768;
    int maxContext = 77;

    LatendDataset dataLoader = new LatendDataset(dataPath, null, batchSize, latendDim, height, width, maxContext, textEmbedDim, BinDataType.float32);

    String vocabPath = "/root/gpufree-data/models/CLIP-GmP-ViT-L-14/vocab.json";
    String mergesPath = "/root/gpufree-data/models/CLIP-GmP-ViT-L-14/merges.txt";
    BPETokenizerEN bpe = new BPETokenizerEN(vocabPath, mergesPath, 49406, 49407);

    int maxPositionEmbeddingsSize = 77;
    int vocabSize = 49408;
    int headNum = 12;
    int n_layers = 12;
    int intermediateSize = 3072;
    ClipTextModel clip = new ClipTextModel(LossType.MSE, UpdaterType.adamw, headNum, maxContext, vocabSize, textEmbedDim, maxPositionEmbeddingsSize, intermediateSize, n_layers);
    clip.CUDNN = true;
    clip.time = maxContext;
    clip.RUN_MODEL = RunModel.EVAL;
    String clipWeight = "/root/gpufree-data/models/CLIP-GmP-ViT-L-14/CLIP-GmP-ViT-L-14.json";
    ModeLoaderlUtils.loadWeight(LagJsonReader.readJsonFileBigWeightIterator(clipWeight), clip, "", false);

    String labelPath = "/root/gpufree-data/6m/metadata_utf8.json";
    String imgDirPath = "/root/gpufree-data/6m/448/";
    boolean horizontalFilp = false;
    int imgSize = 448;

    float[] mean = new float[]{0.485f, 0.456f, 0.406f};
    float[] std = new float[]{0.229f, 0.224f, 0.225f};
    SDImageLoader dataLoader2 = new SDImageLoader(labelPath, imgDirPath, ".jpg", "path", "en", bpe, maxContext, imgSize, imgSize, batchSize, horizontalFilp, mean, std);

    int dinov_patchSize = 14;
    int dinov_hiddenSize = 768;
    int dinov_headNum = 12;
    int dinov_depth = 12;
    int dinov_mlpRatio = 4;
    Dinov2 dinov = new Dinov2(LossType.MSE, UpdaterType.adamw, 3, imgSize, imgSize, dinov_patchSize, dinov_hiddenSize, dinov_headNum, dinov_depth, dinov_mlpRatio);
    dinov.CUDNN = true;
    dinov.RUN_MODEL = RunModel.EVAL;

    String repa_model_path = "/root/gpufree-data/models/dionv2-14-b-512.model";
    ModelUtils.loadModel(dinov, repa_model_path);

    int ditHeadNum = 12;
    int latendSize = 32;
    int depth = 12;
    int timeSteps = 1000;
    int mlpRatio = 4;
    int patchSize = 1;
    int hiddenSize = 768;

    float y_prob = 0.1f;
    float token_drop = 0.0f;
    float path_drop_prob = 0.05f;

    OmegaDiT dit = new OmegaDiT(LossType.MSE, UpdaterType.adamw, latendDim, latendSize, latendSize, patchSize, hiddenSize, ditHeadNum, depth, timeSteps, textEmbedDim, maxContext, mlpRatio, dinov_hiddenSize, token_drop, path_drop_prob, y_prob);
    dit.CUDNN = true;
    dit.CUDNN_SDPA = true;
    dit.learnRate = 1e-4f;

    ICPlan icplan = new ICPlan(dit.tensorOP);

    String model_path = "/root/gpufree-data/omega/flux_sprint_b1_base.model";
    ModelUtils.loadModel(dit, model_path);

    MBSGDOptimizer optimizer = new MBSGDOptimizer(dit, 10, 0.00001f, batchSize, LearnRateUpdate.NONE, false);

    optimizer.train_Flux_Sprint_ICPlan_V(clip, dinov, dataLoader2, dataLoader, icplan, "/root/gpufree-data/omega/dit_512/", 1);
    String save_model_path = "/root/gpufree-data/omega/dit_512/fluxvae_sprint_b1.model";
    ModelUtils.saveModel(dit, save_model_path);
}
```

## 四、相关组件

`DiT、DiTMoudue、DiTBlock、DiTAttentionLayer、DiTSwiGLUFFN、ClipTextModel、VQVAE2、SDImageDataLoaderEN、MBSGDOptimizer`

## 五、阅读建议

1. 先阅读 `OmegaDiT2Test.omega_sprint_b1_clip_train_flux2vae_v_512()`，确认数据、CLIP、VAE、DiT 和 optimizer 如何串起来。
2. 再进入 `DiT.java` 查看构造函数、`initLayers()`、`forward(input, t, context)` 和 `back()`。
3. 继续追踪 `DiTMoudue`、DiT block、attention、MLP、AdaLN 等模块。
4. 若开启 CUDA 或 cuDNN 优化，继续检查对应 kernel wrapper 和 `src/main/resources/cu` 下的实现。

## 六、常见注意事项

- 示例中的数据路径、CLIP 权重路径、VAE 权重路径需要按本机环境调整。
- `latendSize` 需要和 VAE 输出 latent 分辨率保持一致。
- `patchSize` 会影响 token 数：`time = (width / patchSize) * (height / patchSize)`。
- `maxContextLen`、`textEmbedDim` 需要和文本编码器输出匹配。
- `learnSigma`、loss 类型、训练入口方法需要和当前训练目标保持一致。
