# GAN 对抗神经网络

本页只记录 Omega-AI 当前源码中真实存在的实现入口和阅读路径。API 示例必须以源码为准，不使用伪类名或非项目实现的示例代码。

[[toc]]

---

## 一、源码入口

| 类型 | 位置 |
| --- | --- |
| 主要实现 | `com.omega.example.gan.test.GAN` |
| 源码路径 | `src/main/java/com/omega/example/gan/test/GAN.java` |
| 示例入口 | `src/main/java/com/omega/example/gan/test/MinistGAN.java` |
| 示例代码（Demo） | `MinistGAN.gan_anime()` |

## 二、入口代码片段

节选自 `src/main/java/com/omega/example/gan/test/MinistGAN.java`，仅保留入口签名和关键初始化，完整实现以源码为准：

```java
public class MinistGAN {
    public static BPNetwork NetG(int imgSize, int latentSize) {
        BPNetwork netWork = new BPNetwork(LossType.MSE, UpdaterType.adamw);
        netWork.CUDNN = true;
        netWork.learnRate = 0.0001f;
        InputLayer inputLayer = new InputLayer(1, 1, latentSize);
        FullyLayer full1 = new FullyLayer(latentSize, 256, true);
        ReluLayer active1 = new ReluLayer();
        FullyLayer full2 = new FullyLayer(256, 256, true);
        ReluLayer active2 = new ReluLayer();
        FullyLayer full3 = new FullyLayer(256, imgSize, true);
        TanhLayer active4 = new TanhLayer();
        netWork.addLayer(inputLayer);
        netWork.addLayer(full1);
        netWork.addLayer(active1);
        netWork.addLayer(full2);
        netWork.addLayer(active2);
        netWork.addLayer(full3);
        netWork.addLayer(active4);
        return netWork;
    }
}
```

## 三、示例代码（Demo）

节选自 `src/main/java/com/omega/example/gan/test/MinistGAN.java`，保留 GAN 训练入口的关键初始化和训练调用：

```java
public class MinistGAN {
    public void gan_anime() {
        int imgSize = 784;
        int ngf = 784;
        int nz = 100;
        int batchSize = 2048;
        int d_every = 1;
        int g_every = 1;
        float[] mean = new float[]{0.5f};
        float[] std = new float[]{0.5f};

        String mnist_train_data = "/dataset/mnist/train-images.idx3-ubyte";
        String mnist_train_label = "/dataset/mnist/train-labels.idx1-ubyte";
        String[] labelSet = new String[]{"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"};
        File trainDataRes = new File(this.getClass().getClassLoader().getResource(mnist_train_data).toURI());
        File trainLabelRes = new File(this.getClass().getClassLoader().getResource(mnist_train_label).toURI());
        DataSet trainData = DataLoader.loadDataByUByte(trainDataRes, trainLabelRes, labelSet, 1, 1, 784, true, mean, std);

        BPNetwork netG = NetG(ngf, nz);
        BPNetwork netD = NetD(imgSize);
        GANOptimizer optimizer = new GANOptimizer(netG, netD, batchSize, 3500, d_every, g_every, 0.001f, LearnRateUpdate.CONSTANT, false);
        optimizer.train(trainData);
    }
}
```

## 四、相关组件

`生成器、判别器、FullyLayer、BCELoss、MBSGDOptimizer`

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
