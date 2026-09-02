# Seq2Seq 序列到序列模型

本页只记录 Omega-AI 当前源码中真实存在的实现入口和阅读路径。API 示例必须以源码为准，不使用伪类名或非项目实现的示例代码。

[[toc]]

---

## 一、源码入口

| 类型 | 位置 |
| --- | --- |
| 主要实现 | `com.omega.engine.nn.network.Seq2Seq / Seq2SeqRNN` |
| 源码路径 | `src/main/java/com/omega/engine/nn/network/Seq2Seq.java` |
| 示例入口 | `src/main/java/com/omega/example/rnn/seq2seq/Seq2seq.java` |
| 示例代码（Demo） | `Seq2seq.seq2seq()` |

## 二、入口代码片段

节选自 `src/main/java/com/omega/engine/nn/network/Seq2Seq.java`，仅保留入口签名和关键初始化，完整实现以源码为准：

```java
public class Seq2Seq extends Network {
    public int en_time = 1;
    public int de_time = 1;

    public Seq2Seq(RNNCellType cellType, LossType lossType, UpdaterType updater,
                   int en_time, int de_time, int en_em, int en_hidden, int en_len,
                   int de_em, int de_hidden, int de_len) {
        this.cellType = cellType;
        this.lossFunction = LossFactory.create(lossType, this);
        this.updater = updater;
    }
}
```

## 三、示例代码（Demo）

节选自 `src/main/java/com/omega/example/rnn/seq2seq/Seq2seq.java`，保留 Seq2Seq 训练入口的关键初始化和训练调用：

```java
public class Seq2seq {
    public void seq2seq() {
        int batchSize = 128;
        int en_em = 64;
        int de_em = 128;
        int en_hidden = 512;
        int de_hidden = 512;
        String trainPath = "H:\\rnn_dataset\\translate.csv";
        IndexDataLoader trainData = new IndexDataLoader(trainPath, batchSize);

        Seq2Seq network = new Seq2Seq(RNNCellType.LSTM, LossType.softmax_with_cross_entropy, UpdaterType.adamw,
                trainData.max_en, trainData.max_ch - 1, en_em, en_hidden, trainData.en_characters,
                de_em, de_hidden, trainData.ch_characters);
        network.CUDNN = true;
        network.learnRate = 0.01f;

        EDOptimizer optimizer = new EDOptimizer(network, batchSize, 200, 0.001f, LearnRateUpdate.SMART_HALF, false);
        optimizer.lr_step = new int[]{100};
        optimizer.trainSeq2Seq(trainData);
    }
}
```

## 四、相关组件

`Seq2Seq、Seq2SeqRNN、RNNLayer、LSTMLayer、EmbeddingLayer`

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
