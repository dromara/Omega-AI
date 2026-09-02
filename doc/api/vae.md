# VAE 变分自编码器

本页只记录 Omega-AI 当前源码中真实存在的实现入口和阅读路径。API 示例必须以源码为准，不使用伪类名或非项目实现的示例代码。

[[toc]]

---

## 一、源码入口

| 类型 | 位置 |
| --- | --- |
| 主要实现 | `com.omega.engine.nn.network.vae.*` |
| 源码路径 | `src/main/java/com/omega/engine/nn/network/vae` |
| 示例入口 | `src/main/java/com/omega/example/vae/test/VAETest.java` |
| 示例代码（Demo） | `VAETest.tiny_vae()` |

## 二、入口代码片段

节选自 `src/main/java/com/omega/engine/nn/network/vae/VQVAE.java`，仅保留入口签名和关键初始化，完整实现以源码为准：

```java
public class VQVAE extends Network {
    public float beta = 0.2f;
    public int headNum = 4;
    public int latendDim = 4;

    public VQVAE(LossType lossType, UpdaterType updater, int latendDim,
                 int imageSize, int numLayers, int headNum, int num_vq_embeddings,
                 int[] downChannels, boolean[] downSample, int[] midChannels) {
        this.lossFunction = LossFactory.create(lossType, this);
        this.downChannels = downChannels;
        this.downSample = downSample;
    }
}
```

节选自 `src/main/java/com/omega/engine/nn/network/vae/Flux_VAE.java`，仅保留入口签名和关键初始化，完整实现以源码为准：

```java
public class Flux_VAE extends Network {
    public float beta = 0.25f;
    public float decay = 0.999f;

    public Flux_VAE(LossType lossType, UpdaterType updater, int latendDim,
                    int imageSize, int[] ch_mult, int ch, int num_res_blocks) {
        this.lossFunction = LossFactory.create(lossType, this);
        this.latendDim = latendDim;
        this.imageSize = imageSize;
    }
}
```

## 三、示例代码（Demo）

节选自 `src/main/java/com/omega/example/vae/test/VAETest.java`，保留 TinyVAE 训练入口的关键初始化和训练调用：

```java
public class VAETest {
    public static void tiny_vae() {
        int batchSize = 8;
        int imageSize = 256;
        int z_dims = 64;
        int latendDim = 4;
        float[] mean = new float[]{0.5f, 0.5f, 0.5f};
        float[] std = new float[]{0.5f, 0.5f, 0.5f};
        String imgDirPath = "H:\\vae_dataset\\pokemon-blip\\dataset\\";

        DiffusionImageDataLoader dataLoader = new DiffusionImageDataLoader(imgDirPath, imageSize, imageSize, batchSize, false, mean, std);
        TinyVAE network = new TinyVAE(LossType.MSE_SUM, UpdaterType.adamw, z_dims, latendDim, imageSize);
        network.CUDNN = true;
        network.learnRate = 0.001f;

        MBSGDOptimizer optimizer = new MBSGDOptimizer(network, 500, 0.00001f, batchSize, LearnRateUpdate.SMART_HALF, false);
        optimizer.lr_step = new int[]{50, 100, 150, 200, 250, 300, 350, 400, 450};
        optimizer.trainTinyVAE(dataLoader);
    }
}
```

## 四、相关组件

`VAE、VQVAE、TinyVAE、SD_VAE、Flux_VAE、编码器、解码器`

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
