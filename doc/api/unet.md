# Unet U形网络

本页只记录 Omega-AI 当前源码中真实存在的实现入口和阅读路径。API 示例必须以源码为准，不使用伪类名或非项目实现的示例代码。

[[toc]]

---

## 一、源码入口

| 类型 | 位置 |
| --- | --- |
| 主要实现 | `com.omega.engine.nn.network.UNet` |
| 源码路径 | `src/main/java/com/omega/engine/nn/network/UNet.java` |
| 示例入口 | `src/main/java/com/omega/example/diffusion/test/DiffusionModelTest.java` |
| 示例代码（Demo） | `DiffusionModelTest.duffsion_anime()` |

## 二、入口代码片段

节选自 `src/main/java/com/omega/engine/nn/network/UNet.java`，仅保留入口签名和关键初始化，完整实现以源码为准：

```java
public class UNet extends Network {
    public int inChannel;
    public int outChannel;
    private boolean bias = true;

    public UNet(LossType lossType, UpdaterType updater, int inChannel, int outChannel,
                int width, int height, boolean bilinear, boolean bias) {
        this.lossFunction = LossFactory.create(lossType, this);
        this.bilinear = bilinear;
        this.bias = bias;
    }
}
```

## 三、示例代码（Demo）

节选自 `src/main/java/com/omega/example/diffusion/test/DiffusionModelTest.java`，保留 UNet 扩散训练入口的关键初始化和训练调用：

```java
public class DiffusionModelTest {
    public static void duffsion_anime() {
        boolean bias = false;
        int batchSize = 4;
        int imw = 96;
        int imh = 96;
        int mChannel = 64;
        int resBlockNum = 2;
        int T = 1000;
        int[] channelMult = new int[]{1, 2};
        String imgDirPath = "H:\\voc\\gan_anime\\ml2021spring-hw6\\faces\\";

        DiffusionImageDataLoader dataLoader = new DiffusionImageDataLoader(imgDirPath, imw, imh, batchSize, false);
        DiffusionUNet network = new DiffusionUNet(LossType.MSE, UpdaterType.adamw, T, 3, mChannel, channelMult, resBlockNum, imw, imh, bias);
        network.CUDNN = true;
        network.learnRate = 0.0005f;

        MBSGDOptimizer optimizer = new MBSGDOptimizer(network, 50, 0.00001f, batchSize, LearnRateUpdate.GD_GECAY, false);
        optimizer.trainGaussianDiffusion(dataLoader);
    }
}
```

## 四、相关组件

`UNet、UNetDownBlock、UNetMidBlock、UNetUpBlock、UNetCrossAttentionLayer`

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
