# CNN 卷积神经网络

本页只记录 Omega-AI 当前源码中真实存在的实现入口和阅读路径。API 示例必须以源码为准，不使用伪类名或非项目实现的示例代码。

[[toc]]

---

## 一、源码入口

| 类型 | 位置 |
| --- | --- |
| 主要实现 | `com.omega.engine.nn.network.CNN` |
| 源码路径 | `src/main/java/com/omega/engine/nn/network/CNN.java` |
| 示例入口 | `src/main/java/com/omega/example/cnn/test/CNNTest.java` |
| 示例代码（Demo） | `CNNTest.cnnNetwork_cifar10()` |

## 二、入口代码片段

节选自 `src/main/java/com/omega/engine/nn/network/CNN.java`，仅保留入口签名和关键初始化，完整实现以源码为准：

```java
public class CNN extends Network {
    public CNN(LossFunction lossFunction) {
        this.lossFunction = lossFunction;
    }

    public CNN(LossFunction lossFunction, UpdaterType updater) {
        this.lossFunction = lossFunction;
        this.updater = updater;
    }

    public CNN(LossType lossType, UpdaterType updater) {
        this.lossFunction = LossFactory.create(lossType, this);
        this.updater = updater;
    }
}
```

示例入口节选自 `src/main/java/com/omega/example/cnn/test/CNNTest.java`，完整实现以源码为准：

```java
public class CNNTest {
    public static void main(String[] args) {
        CNNTest cnn = new CNNTest();
        cnn.cifar10();
    }
}
```

## 三、示例代码（Demo）

节选自 `src/main/java/com/omega/example/cnn/test/CNNTest.java`，保留 CIFAR-10 训练入口的关键初始化和训练调用：

```java
public class CNNTest {
    public void cnnNetwork_cifar10() {
        String[] labelSet = new String[]{"airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"};
        String[] train_data_filenames = new String[]{"H:/dataset/cifar-10/data_batch_1.bin", "H:/dataset/cifar-10/data_batch_2.bin", "H:/dataset/cifar-10/data_batch_3.bin", "H:/dataset/cifar-10/data_batch_4.bin", "H:/dataset/cifar-10/data_batch_5.bin"};
        String test_data_filename = "H:/dataset/cifar-10/test_batch.bin";
        DataSet trainData = DataLoader.getImagesToDataSetByBin(train_data_filenames, 10000, 3, 32, 32, 10, labelSet, true);
        DataSet testData = DataLoader.getImagesToDataSetByBin(test_data_filename, 10000, 3, 32, 32, 10, labelSet, true);

        int channel = 3;
        int height = 32;
        int width = 32;
        CNN netWork = new CNN(LossType.softmax_with_cross_entropy, UpdaterType.adam);
        netWork.learnRate = 0.01f;
        netWork.addLayer(new InputLayer(channel, height, width));

        ConvolutionLayer conv1 = new ConvolutionLayer(channel, 16, width, height, 3, 3, 1, 1, false);
        netWork.addLayer(conv1);
        netWork.addLayer(new BNLayer());
        netWork.addLayer(new ReluLayer());
        PoolingLayer pool1 = new PoolingLayer(conv1.oChannel, conv1.oWidth, conv1.oHeight, 2, 2, 2, PoolingType.MAX_POOLING);
        netWork.addLayer(pool1);

        ConvolutionLayer conv3 = new ConvolutionLayer(pool1.oChannel, 32, pool1.oWidth, pool1.oHeight, 3, 3, 1, 1, false);
        netWork.addLayer(conv3);
        netWork.addLayer(new BNLayer());
        netWork.addLayer(new ReluLayer());
        PoolingLayer pool2 = new PoolingLayer(conv3.oChannel, conv3.oWidth, conv3.oHeight, 2, 2, 2, PoolingType.MAX_POOLING);
        netWork.addLayer(pool2);

        ConvolutionLayer conv4 = new ConvolutionLayer(pool2.oChannel, 64, pool2.oWidth, pool2.oHeight, 3, 3, 1, 1, false);
        netWork.addLayer(conv4);
        netWork.addLayer(new BNLayer());
        netWork.addLayer(new ReluLayer());
        PoolingLayer pool3 = new PoolingLayer(conv4.oChannel, conv4.oWidth, conv4.oHeight, 2, 2, 2, PoolingType.MAX_POOLING);
        netWork.addLayer(pool3);

        int fInputCount = pool3.oChannel * pool3.oWidth * pool3.oHeight;
        FullyLayer full1 = new FullyLayer(fInputCount, 256, true);
        netWork.addLayer(full1);
        netWork.addLayer(new ReluLayer());
        netWork.addLayer(new DropoutLayer(0.5f));
        netWork.addLayer(new FullyLayer(full1.oWidth, 10, true));
        netWork.addLayer(new SoftmaxWithCrossEntropyLayer(10));

        MBSGDOptimizer optimizer = new MBSGDOptimizer(netWork, 20, 0.001f, 128, LearnRateUpdate.CONSTANT, false);
        optimizer.train(trainData);
        optimizer.test(testData);
    }
}
```

## 四、相关组件

`CNN、InputLayer、ConvolutionLayer、PoolingLayer、FullyLayer、MBSGDOptimizer`

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
