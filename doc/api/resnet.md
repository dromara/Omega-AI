# Resnet 残差网络

本页只记录 Omega-AI 当前源码中真实存在的实现入口和阅读路径。API 示例必须以源码为准，不使用伪类名或非项目实现的示例代码。

[[toc]]

---

## 一、源码入口

| 类型 | 位置 |
| --- | --- |
| 主要实现 | `com.omega.example.resnet.test.Resnet18` |
| 源码路径 | `src/main/java/com/omega/example/resnet/test/Resnet18.java` |
| 示例入口 | `src/main/java/com/omega/example/resnet/test/ResnetTest.java` |
| 示例代码（Demo） | `ResnetTest.resnet18_cifar10()` |

## 二、入口代码片段

节选自 `src/main/java/com/omega/example/resnet/test/Resnet18.java`，仅保留入口签名和关键初始化，完整实现以源码为准：

```java
public class Resnet18 {
    public static CNN instance(int channel, int height, int width, int output) {
        CNN netWork = new CNN(LossType.softmax_with_cross_entropy, UpdaterType.adamw);
        netWork.CUDNN = true;
        netWork.learnRate = 0.1f;
        InputLayer inputLayer = new InputLayer(channel, height, width);
        ConvolutionLayer conv1 = new ConvolutionLayer(channel, 64, width, height, 3, 3, 1, 1, false);
        BNLayer bn1 = new BNLayer();
        ReluLayer active1 = new ReluLayer();
        PoolingLayer pool1 = new PoolingLayer(conv1.oChannel, conv1.oWidth, conv1.oHeight, 2, 2, 2, PoolingType.MAX_POOLING);
        BasicBlockLayer bl1 = new BasicBlockLayer(pool1.oChannel, 64, pool1.oHeight, pool1.oWidth, 1, netWork);
        return netWork;
    }
}
```

## 三、示例代码（Demo）

节选自 `src/main/java/com/omega/example/resnet/test/ResnetTest.java`，保留 ResNet18 CIFAR-10 训练入口的关键初始化和训练调用：

```java
public class ResnetTest {
    public void resnet18_cifar10() {
        String[] labelSet = new String[]{"airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"};
        String[] train_data_filenames = new String[]{"H:/dataset/cifar-10/data_batch_1.bin", "H:/dataset/cifar-10/data_batch_2.bin", "H:/dataset/cifar-10/data_batch_3.bin", "H:/dataset/cifar-10/data_batch_4.bin", "H:/dataset/cifar-10/data_batch_5.bin"};
        String test_data_filename = "H:/dataset/cifar-10/test_batch.bin";
        float[] mean = new float[]{0.4914f, 0.4822f, 0.4465f};
        float[] std = new float[]{0.2023f, 0.1994f, 0.2010f};
        DataSet trainData = DataLoader.getImagesToDataSetByBin(train_data_filenames, 10000, 3, 32, 32, 10, labelSet, true);
        DataSet testData = DataLoader.getImagesToDataSetByBin(test_data_filename, 10000, 3, 32, 32, 10, labelSet, true, mean, std);

        int batchSize = 128;
        CNN netWork = new CNN(LossType.softmax_with_cross_entropy, UpdaterType.adamw);
        netWork.CUDNN = true;
        netWork.learnRate = 0.01f;
        netWork.addLayer(new InputLayer(3, 32, 32));

        ConvolutionLayer conv1 = new ConvolutionLayer(3, 64, 32, 32, 3, 3, 1, 1, false);
        conv1.paramsInit = ParamsInit.relu;
        netWork.addLayer(conv1);
        netWork.addLayer(new BNLayer());
        netWork.addLayer(new ReluLayer());

        BasicBlockLayer bl1 = new BasicBlockLayer(conv1.oChannel, 64, conv1.oHeight, conv1.oWidth, 1, netWork);
        BasicBlockLayer bl2 = new BasicBlockLayer(bl1.oChannel, 64, bl1.oHeight, bl1.oWidth, 1, netWork);
        BasicBlockLayer bl3 = new BasicBlockLayer(bl2.oChannel, 128, bl2.oHeight, bl2.oWidth, 2, netWork);
        BasicBlockLayer bl4 = new BasicBlockLayer(bl3.oChannel, 128, bl3.oHeight, bl3.oWidth, 1, netWork);
        BasicBlockLayer bl5 = new BasicBlockLayer(bl4.oChannel, 256, bl4.oHeight, bl4.oWidth, 2, netWork);
        BasicBlockLayer bl6 = new BasicBlockLayer(bl5.oChannel, 256, bl5.oHeight, bl5.oWidth, 1, netWork);
        BasicBlockLayer bl7 = new BasicBlockLayer(bl6.oChannel, 512, bl6.oHeight, bl6.oWidth, 2, netWork);
        BasicBlockLayer bl8 = new BasicBlockLayer(bl7.oChannel, 512, bl7.oHeight, bl7.oWidth, 1, netWork);

        netWork.addLayer(bl1);
        netWork.addLayer(new ReluLayer());
        netWork.addLayer(bl2);
        netWork.addLayer(new ReluLayer());
        netWork.addLayer(bl3);
        netWork.addLayer(new ReluLayer());
        netWork.addLayer(bl4);
        netWork.addLayer(new ReluLayer());
        netWork.addLayer(bl5);
        netWork.addLayer(new ReluLayer());
        netWork.addLayer(bl6);
        netWork.addLayer(new ReluLayer());
        netWork.addLayer(bl7);
        netWork.addLayer(new ReluLayer());
        netWork.addLayer(bl8);
        netWork.addLayer(new ReluLayer());

        AVGPoolingLayer pool2 = new AVGPoolingLayer(bl8.oChannel, bl8.oWidth, bl8.oHeight);
        netWork.addLayer(pool2);
        netWork.addLayer(new FullyLayer(pool2.oChannel * pool2.oWidth * pool2.oHeight, 10));
        netWork.addLayer(new SoftmaxWithCrossEntropyLayer(10));

        MBSGDOptimizer optimizer = new MBSGDOptimizer(netWork, 500, 0.0001f, batchSize, LearnRateUpdate.GD_GECAY, false);
        optimizer.train(trainData, testData, mean, std);
        optimizer.test(testData);
    }
}
```

## 四、相关组件

`ConvolutionLayer、BNLayer、ReluLayer、BasicBlockLayer、FullyLayer`

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
