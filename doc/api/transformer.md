# Transformer transformer模型

本页只记录 Omega-AI 当前源码中真实存在的实现入口和阅读路径。API 示例必须以源码为准，不使用伪类名或非项目实现的示例代码。

[[toc]]

---

## 一、源码入口

| 类型 | 位置 |
| --- | --- |
| 主要实现 | `com.omega.engine.nn.network.Transformer` |
| 源码路径 | `src/main/java/com/omega/engine/nn/network/Transformer.java` |
| 示例入口 | `src/main/java/com/omega/example/transformer/test/GPTTest.java` |
| 示例代码（Demo） | `GPTTest.gpt()` |

## 二、入口代码片段

节选自 `src/main/java/com/omega/engine/nn/network/Transformer.java`，仅保留入口签名和关键字段，完整实现以源码为准：

```java
public class Transformer extends Network {
    public int en_time = 1;
    public int de_time = 1;
    public int en_len;
    public int de_len;

    public Transformer() {
    }
}
```

## 三、示例代码（Demo）

节选自 `src/main/java/com/omega/example/transformer/test/GPTTest.java`，保留 Transformer/GPT 训练入口的关键初始化和训练调用：

```java
public class GPTTest {
    public static void gpt() {
        boolean bias = false;
        boolean dropout = false;
        int batchSize = 32;
        int max_len = 128;
        int embedDim = 768;
        int head_num = 12;
        int decoderNum = 12;
        String trainPath = "H:\\transformer_dataset\\gpt\\wikitext-2-v1\\wikitext-2\\wiki.train.tokens";

        ENTokenizer trainData = new ENTokenizer(trainPath, max_len, batchSize);
        NanoGPT network = new NanoGPT(LossType.softmax_with_cross_entropy, UpdaterType.adamw, head_num,
                decoderNum, trainData.vocab_size, max_len, embedDim, bias, dropout, false);
        network.learnRate = 0.0001f;

        EDOptimizer optimizer = new EDOptimizer(network, batchSize, 100, 0.001f, LearnRateUpdate.GD_GECAY, false);
        optimizer.trainGPT(trainData);
    }
}
```

## 四、相关组件

`Transformer、TransformerBlock、EmbeddingLayer、Attention、MBSGDOptimizer`

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
