# LLama 生成式预训练模型

本页只记录 Omega-AI 当前源码中真实存在的实现入口和阅读路径。API 示例必须以源码为准，不使用伪类名或非项目实现的示例代码。

[[toc]]

---

## 一、源码入口

| 类型 | 位置 |
| --- | --- |
| 主要实现 | `com.omega.engine.nn.network.Llama2 / Llama3` |
| 源码路径 | `src/main/java/com/omega/engine/nn/network/Llama3.java` |
| 示例入口 | `src/main/java/com/omega/example/transformer/test/Llama3Test.java` |
| 示例代码（Demo） | `Llama3Test.llama3_monkey_sft()` |

## 二、入口代码片段

节选自 `src/main/java/com/omega/engine/nn/network/Llama3.java`，仅保留入口签名和关键初始化，完整实现以源码为准：

```java
public class Llama3 extends Network {
    public int vocabSize;
    public int embedDim;
    public int headNum = 8;

    public Llama3(LossType lossType, UpdaterType updater, int headNum, int nKVHeadNum,
                  int decoderNum, int vocabSize, int time, int embedDim,
                  boolean bias, boolean dropout) {
        this.lossFunction = LossFactory.create(lossType, this);
        this.bias = bias;
        this.dropout = dropout;
    }

    public Llama3(LossType lossType, UpdaterType updater, int headNum, int nKVHeadNum,
                  int decoderNum, int vocabSize, int time, int embedDim,
                  boolean bias, boolean dropout, boolean flashAttention) {
        this.flashAttention = flashAttention;
        this.lossFunction = LossFactory.create(lossType, this);
    }
}
```

## 三、示例代码（Demo）

节选自 `src/main/java/com/omega/example/transformer/test/Llama3Test.java`，保留 Llama3 SFT 训练入口的关键初始化和训练调用：

```java
public class Llama3Test {
    public static void llama3_monkey_sft() {
        boolean bias = false;
        boolean dropout = false;
        boolean flashAttention = false;
        int batchSize = 3;
        int max_len = 512;
        int embedDim = 512;
        int head_num = 16;
        int nKVHeadNum = 8;
        int decoderNum = 8;
        String trainPath = "H:\\transformer_dataset\\6400\\sft_data_single.csv";
        String vocabPath = "H:\\transformer_dataset\\6400\\vocab.json";
        String mergesPath = "H:\\transformer_dataset\\6400\\merges.txt";

        BPETokenizer3 tokenizer = new BPETokenizer3(vocabPath, mergesPath);
        SFTDataset trainData = new SFTDataset(trainPath, max_len, batchSize, tokenizer);
        Llama3 network = new Llama3(LossType.softmax_with_cross_entropy_idx, UpdaterType.adamw, head_num,
                nKVHeadNum, decoderNum, trainData.vocab_size, max_len, embedDim, bias, dropout, flashAttention);
        network.learnRate = 1e-4f;

        String model_path = "H:\\model\\llama3-26m-chinese.model";
        ModelUtils.loadModel(network, model_path);
        EDOptimizer optimizer = new EDOptimizer(network, batchSize, 2, 0.0001f, LearnRateUpdate.CONSTANT, false);
        optimizer.trainLlama3_chinese_sft(trainData, 8, true);

        String save_model_path = "H:\\model\\llama3-26m-chinese.model";
        ModelUtils.saveModel(network, save_model_path);
    }
}
```

## 四、相关组件

`Llama2、Llama3、RMSLayer、RoPE、LlamaCausalSelfAttentionLayer、Tokenizer`

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
