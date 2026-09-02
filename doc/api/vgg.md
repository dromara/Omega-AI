# VGG vgg神经网络

本页只记录 Omega-AI 当前源码中真实存在的实现入口和阅读路径。API 示例必须以源码为准，不使用伪类名或非项目实现的示例代码。

[[toc]]

---

## 一、源码入口

| 类型 | 位置 |
| --- | --- |
| 主要实现 | `com.omega.example.vggnet.test.vggnetTest` |
| 源码路径 | `src/main/java/com/omega/example/vggnet/test/vggnetTest.java` |
| 示例入口 | `src/main/java/com/omega/example/vggnet/test/vggnetTest.java` |
| 示例代码（Demo） | `vggnetTest.vgg16_cifar10()` |

## 二、入口代码片段

节选自 `src/main/java/com/omega/example/vggnet/test/vggnetTest.java`，仅保留入口签名和关键初始化，完整实现以源码为准：

```java
public class vggnetTest {
    public void vgg16_cifar10() {
        CNN netWork = new CNN(LossType.softmax_with_cross_entropy, UpdaterType.adam);
        netWork.CUDNN = true;
        netWork.learnRate = 0.001f;
        InputLayer inputLayer = new InputLayer(channel, height, width);
        ConvolutionLayer conv1 = new ConvolutionLayer(channel, 64, width, height, 3, 3, 1, 1, false);
        ReluLayer active1 = new ReluLayer();
        ConvolutionLayer conv2 = new ConvolutionLayer(conv1.oChannel, 64, conv1.oWidth, conv1.oHeight, 3, 3, 1, 1, false);
        ReluLayer active2 = new ReluLayer();
        PoolingLayer pool1 = new PoolingLayer(conv2.oChannel, conv2.oWidth, conv2.oHeight, 2, 2, 2, PoolingType.MAX_POOLING);
    }
}
```

## 三、示例代码（Demo）

节选自 `src/main/java/com/omega/example/vggnet/test/vggnetTest.java`，保留 VGG16 CIFAR-10 训练入口的关键初始化和训练调用：

```java
public class vggnetTest {
    public void vgg16_cifar10() {
        String[] labelSet = new String[]{"airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"};
        String[] train_data_filenames = new String[]{"H:/dataset/cifar-10/data_batch_1.bin", "H:/dataset/cifar-10/data_batch_2.bin", "H:/dataset/cifar-10/data_batch_3.bin", "H:/dataset/cifar-10/data_batch_4.bin", "H:/dataset/cifar-10/data_batch_5.bin"};
        String test_data_filename = "H:/dataset/cifar-10/test_batch.bin";
        float[] mean = new float[]{0.485f, 0.456f, 0.406f};
        float[] std = new float[]{0.229f, 0.224f, 0.225f};
        DataSet trainData = DataLoader.getImagesToDataSetByBin(train_data_filenames, 10000, 3, 32, 32, 10, labelSet, true);
        DataSet testData = DataLoader.getImagesToDataSetByBin(test_data_filename, 10000, 3, 32, 32, 10, labelSet, true, mean, std);

        CNN netWork = new CNN(LossType.softmax_with_cross_entropy, UpdaterType.adam);
        netWork.CUDNN = true;
        netWork.learnRate = 0.001f;
        netWork.addLayer(new InputLayer(3, 32, 32));

        ConvolutionLayer conv1 = new ConvolutionLayer(3, 64, 32, 32, 3, 3, 1, 1, false);
        ConvolutionLayer conv2 = new ConvolutionLayer(conv1.oChannel, 64, conv1.oWidth, conv1.oHeight, 3, 3, 1, 1, false);
        PoolingLayer pool1 = new PoolingLayer(conv2.oChannel, conv2.oWidth, conv2.oHeight, 2, 2, 2, PoolingType.MAX_POOLING);
        ConvolutionLayer conv3 = new ConvolutionLayer(pool1.oChannel, 128, pool1.oWidth, pool1.oHeight, 3, 3, 1, 1, false);
        ConvolutionLayer conv4 = new ConvolutionLayer(conv3.oChannel, 128, conv3.oWidth, conv3.oHeight, 3, 3, 1, 1, false);
        PoolingLayer pool2 = new PoolingLayer(conv4.oChannel, conv4.oWidth, conv4.oHeight, 2, 2, 2, PoolingType.MAX_POOLING);
        ConvolutionLayer conv5 = new ConvolutionLayer(pool2.oChannel, 256, pool2.oWidth, pool2.oHeight, 3, 3, 1, 1, false);
        ConvolutionLayer conv6 = new ConvolutionLayer(conv5.oChannel, 256, conv5.oWidth, conv5.oHeight, 3, 3, 1, 1, false);
        ConvolutionLayer conv7 = new ConvolutionLayer(conv6.oChannel, 256, conv6.oWidth, conv6.oHeight, 3, 3, 1, 1, false);
        PoolingLayer pool3 = new PoolingLayer(conv7.oChannel, conv7.oWidth, conv7.oHeight, 2, 2, 2, PoolingType.MAX_POOLING);
        ConvolutionLayer conv8 = new ConvolutionLayer(pool3.oChannel, 512, pool3.oWidth, pool3.oHeight, 3, 3, 1, 1, false);
        ConvolutionLayer conv9 = new ConvolutionLayer(conv8.oChannel, 512, conv8.oWidth, conv8.oHeight, 3, 3, 1, 1, false);
        ConvolutionLayer conv10 = new ConvolutionLayer(conv9.oChannel, 512, conv9.oWidth, conv9.oHeight, 3, 3, 1, 1, false);
        ConvolutionLayer conv20 = new ConvolutionLayer(conv10.oChannel, 512, conv10.oWidth, conv10.oHeight, 3, 3, 1, 1, false);
        PoolingLayer pool4 = new PoolingLayer(conv20.oChannel, conv20.oWidth, conv20.oHeight, 2, 2, 2, PoolingType.MAX_POOLING);
        ConvolutionLayer conv11 = new ConvolutionLayer(pool4.oChannel, 512, pool4.oWidth, pool4.oHeight, 3, 3, 1, 1, false);
        ConvolutionLayer conv12 = new ConvolutionLayer(conv11.oChannel, 512, conv11.oWidth, conv11.oHeight, 3, 3, 1, 1, false);
        ConvolutionLayer conv13 = new ConvolutionLayer(conv12.oChannel, 512, conv12.oWidth, conv12.oHeight, 3, 3, 1, 1, false);
        ConvolutionLayer conv21 = new ConvolutionLayer(conv13.oChannel, 512, conv13.oWidth, conv13.oHeight, 3, 3, 1, 1, false);
        PoolingLayer pool5 = new PoolingLayer(conv21.oChannel, conv21.oWidth, conv21.oHeight, 2, 2, 2, PoolingType.MAX_POOLING);

        netWork.addLayer(conv1);
        netWork.addLayer(new ReluLayer());
        netWork.addLayer(conv2);
        netWork.addLayer(new ReluLayer());
        netWork.addLayer(pool1);
        netWork.addLayer(conv3);
        netWork.addLayer(new ReluLayer());
        netWork.addLayer(conv4);
        netWork.addLayer(new ReluLayer());
        netWork.addLayer(pool2);
        netWork.addLayer(conv5);
        netWork.addLayer(new ReluLayer());
        netWork.addLayer(conv6);
        netWork.addLayer(new ReluLayer());
        netWork.addLayer(conv7);
        netWork.addLayer(new ReluLayer());
        netWork.addLayer(pool3);
        netWork.addLayer(conv8);
        netWork.addLayer(new ReluLayer());
        netWork.addLayer(conv9);
        netWork.addLayer(new ReluLayer());
        netWork.addLayer(conv10);
        netWork.addLayer(new ReluLayer());
        netWork.addLayer(conv20);
        netWork.addLayer(new ReluLayer());
        netWork.addLayer(pool4);
        netWork.addLayer(conv11);
        netWork.addLayer(new ReluLayer());
        netWork.addLayer(conv12);
        netWork.addLayer(new ReluLayer());
        netWork.addLayer(conv13);
        netWork.addLayer(new ReluLayer());
        netWork.addLayer(conv21);
        netWork.addLayer(new ReluLayer());
        netWork.addLayer(pool5);

        int fInputCount = pool5.oChannel * pool5.oWidth * pool5.oHeight;
        netWork.addLayer(new FullyLayer(fInputCount, 4096, false));
        netWork.addLayer(new ReluLayer());
        netWork.addLayer(new FullyLayer(4096, 4096, false));
        netWork.addLayer(new ReluLayer());
        netWork.addLayer(new FullyLayer(4096, 10));
        netWork.addLayer(new SoftmaxWithCrossEntropyLayer(10));

        MBSGDOptimizer optimizer = new MBSGDOptimizer(netWork, 20, 0.001f, 128, LearnRateUpdate.CONSTANT, false);
        optimizer.train(trainData);
        optimizer.test(testData);
    }
}
```

## 四、相关组件

`ConvolutionLayer、PoolingLayer、FullyLayer、MBSGDOptimizer`

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
