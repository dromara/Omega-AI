# MNIST 入门训练

MNIST 是理解 Omega-AI 训练流程的推荐入口。它数据量小、模型简单、日志反馈快，适合验证环境和学习基础 API。

[[toc]]

---

## 一、示例入口

项目中可以从以下示例开始阅读：

```text
src/main/java/com/omega/example/bp/test/BPTest.java
src/main/java/com/omega/example/cnn/test/CNNTest.java
```

`BPTest` 更适合理解全连接网络，`CNNTest` 更适合理解卷积网络。

## 二、准备数据

MNIST 常见文件包括：

```text
train-images.idx3-ubyte
train-labels.idx1-ubyte
t10k-images.idx3-ubyte
t10k-labels.idx1-ubyte
```

如果示例中写死了本地路径，请改成本机数据目录。Windows 路径注意使用 `\\` 或 `/`。

## 三、训练主线

以下代码来自 `src/main/java/com/omega/example/bp/test/BPTest.java` 中的 `bpNetwork_mnist()`，文档示例应以源码实现为准：

```java
String[] labelSet = new String[]{"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"};
DataSet trainData = DataLoader.loadDataByUByte(mnist_train_data, mnist_train_label, labelSet, 1, 1, 784, true);
DataSet testData = DataLoader.loadDataByUByte(mnist_test_data, mnist_test_label, labelSet, 1, 1, 784, true);
BPNetwork netWork = new BPNetwork(new SoftmaxWithCrossEntropyLoss(), UpdaterType.adamw);
netWork.CUDNN = true;
netWork.learnRate = 0.001f;
int inputCount = (int) (Math.sqrt(794) + 10);
InputLayer inputLayer = new InputLayer(1, 1, 784);
FullyLayer hidden1 = new FullyLayer(784, inputCount, false);
BNLayer bn1 = new BNLayer();
ReluLayer active1 = new ReluLayer();
FullyLayer hidden2 = new FullyLayer(inputCount, inputCount, false);
BNLayer bn2 = new BNLayer();
ReluLayer active2 = new ReluLayer();
FullyLayer hidden3 = new FullyLayer(inputCount, inputCount, false);
BNLayer bn3 = new BNLayer();
ReluLayer active3 = new ReluLayer();
FullyLayer hidden4 = new FullyLayer(inputCount, 10);
DropoutLayer dropout = new DropoutLayer(0.2f);
SoftmaxWithCrossEntropyLayer softmax = new SoftmaxWithCrossEntropyLayer(10);
netWork.addLayer(inputLayer);
netWork.addLayer(hidden1);
netWork.addLayer(bn1);
netWork.addLayer(active1);
netWork.addLayer(hidden2);
netWork.addLayer(bn2);
netWork.addLayer(active2);
netWork.addLayer(hidden3);
netWork.addLayer(bn3);
netWork.addLayer(active3);
netWork.addLayer(hidden4);
netWork.addLayer(dropout);
netWork.addLayer(softmax);

MBSGDOptimizer optimizer = new MBSGDOptimizer(netWork, 10, 0.001f, 128, LearnRateUpdate.NONE, false);
optimizer.train(trainData);
optimizer.test(testData);
```

运行前只需要把 `mnist_train_data`、`mnist_train_label`、`mnist_test_data`、`mnist_test_label` 改成本机数据集路径。阅读示例时重点看网络层顺序、loss 类型、updater 类型和 batch size。

## 四、如何判断跑通

跑通后应该看到：

- 程序能够加载数据。
- 训练 step 正常推进。
- loss 有输出。
- 没有 CUDA 或数据路径异常。

如果 loss 不下降，先检查输入是否归一化、label 是否 one-hot、输出层和 loss 是否匹配。

## 五、常见调整

| 调整项 | 建议 |
| --- | --- |
| batch size | CPU 可先用 32 或 64，GPU 可用 128 起步 |
| learning rate | 先用示例默认值，再小范围调整 |
| hidden size | 入门可用 128 或 256 |
| epoch | 先跑 1 到 3 个 epoch 验证流程 |

MNIST 不是为了追求最高精度，而是为了确认环境、训练循环和模型搭建方式正确。
