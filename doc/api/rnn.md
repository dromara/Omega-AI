# RNN 循环神经网络

本页只记录 Omega-AI 当前源码中真实存在的实现入口和阅读路径。API 示例必须以源码为准，不使用伪类名或非项目实现的示例代码。

[[toc]]

---

## 一、源码入口

| 类型 | 位置 |
| --- | --- |
| 主要实现 | `com.omega.engine.nn.network.RNN` |
| 源码路径 | `src/main/java/com/omega/engine/nn/network/RNN.java` |
| 示例入口 | `src/main/java/com/omega/example/rnn/test/CharRNN.java` |
| 示例代码（Demo） | `CharRNN.charRNN()` |

## 二、入口代码片段

节选自 `src/main/java/com/omega/engine/nn/network/RNN.java`，仅保留入口签名和关键初始化，完整实现以源码为准：

```java
public class RNN extends Network {
    public int time = 1;

    public RNN(LossFunction lossFunction) {
        this.lossFunction = lossFunction;
    }

    public RNN(LossFunction lossFunction, UpdaterType updater) {
        this.lossFunction = lossFunction;
        this.updater = updater;
    }

    public RNN(LossType lossType, UpdaterType updater) {
        this.lossFunction = LossFactory.create(lossType, this);
        this.updater = updater;
    }

    public RNN(LossType lossType, UpdaterType updater, int time) {
        this.lossFunction = LossFactory.create(lossType, this);
        this.updater = updater;
        this.time = time;
    }
}
```

## 三、示例代码（Demo）

节选自 `src/main/java/com/omega/example/rnn/test/CharRNN.java`，保留 charRNN 训练入口的关键初始化和训练调用：

```java
public class CharRNN {
    public void charRNN() {
        int time = 256;
        int batchSize = 64;
        int embedding_dim = 256;
        int hiddenSize = 512;
        String trainPath = "H:\\rnn_dataset\\dpcc.txt";
        OneHotDataLoader trainData = new OneHotDataLoader(trainPath, time, batchSize);

        RNN netWork = new RNN(LossType.softmax_with_cross_entropy, UpdaterType.adamw, time);
        netWork.addLayer(new InputLayer(1, 1, trainData.characters));
        netWork.addLayer(new EmbeddingLayer(trainData.characters, embedding_dim));
        netWork.addLayer(new RNNLayer(embedding_dim, hiddenSize, time, ActiveType.tanh, false, netWork));
        netWork.addLayer(new RNNLayer(hiddenSize, hiddenSize, time, ActiveType.tanh, false, netWork));
        netWork.addLayer(new RNNLayer(hiddenSize, hiddenSize, time, ActiveType.tanh, false, netWork));
        netWork.addLayer(new FullyLayer(hiddenSize, hiddenSize, false));
        netWork.addLayer(new BNLayer());
        netWork.addLayer(new LeakyReluLayer());
        netWork.addLayer(new FullyLayer(hiddenSize, trainData.characters, true));
        netWork.CUDNN = true;
        netWork.learnRate = 0.01f;

        MBSGDOptimizer optimizer = new MBSGDOptimizer(netWork, 2, 0.001f, batchSize, LearnRateUpdate.POLY, false);
        optimizer.trainRNN(trainData);
    }
}
```

## 四、相关组件

`RNN、RNNLayer、RNNDataLoader、序列输入`

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
