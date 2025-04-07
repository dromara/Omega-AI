# Seq2Seq 序列到序列模型

## 概述
序列到序列模型（Sequence to Sequence）通过编码器-解码器架构实现序列转换，广泛应用于机器翻译、文本摘要、对话系统等场景。

## 核心结构
```java
/**
 * Seq2Seq基础实现
 * 包含编码器、解码器、注意力机制三个核心组件
 */
public class Seq2Seq {
    private Encoder encoder;          // 编码器（LSTM/GRU）
    private Decoder decoder;          // 解码器（LSTM/GRU）
    private AttentionLayer attention; // 注意力层
    private int vocabSize;            // 词表大小
    private int embeddingDim;         // 词向量维度
    
    // 特殊标记
    private final int SOS_TOKEN = 0;  // 句子起始符
    private final int EOS_TOKEN = 1;  // 句子结束符
}
```

## 完整实现

### 1. 编码器实现
```java
public class Encoder {
    private LSTM[] layers;          // 多层LSTM
    private Embedding embedding;     // 词嵌入层
    
    public float[][][] encode(String[] input) {
        // 词嵌入
        float[][] embeddings = embedding.lookup(input);
        
        // 多层LSTM处理
        float[][][] states = new float[layers.length][][];
        for (int i=0; i<layers.length; i++) {
            states[i] = layers[i].processSequence(embeddings);
        }
        return states; // [layer][time_step][hidden_dim]
    }
}
```

### 2. 解码器实现
```java
public class Decoder {
    private LSTM[] layers;
    private Dense outputLayer;
    
    public String decode(float[][][] encoderStates, int maxLength) {
        List<Integer> output = new ArrayList<>();
        int currentToken = SOS_TOKEN;
        float[] hiddenState = initState();
        
        for (int t=0; t<maxLength; t++) {
            // 注意力计算
            float[] context = attention.calculate(encoderStates, hiddenState);
            
            // LSTM前向计算
            float[] stepInput = concatenate(
                embedding.lookup(currentToken), 
                context
            );
            
            for (int i=0; i<layers.length; i++) {
                hiddenState = layers[i].stepForward(stepInput, hiddenState);
                stepInput = hiddenState;
            }
            
            // 生成预测结果
            float[] logits = outputLayer.forward(hiddenState);
            currentToken = argmax(logits);
            
            if (currentToken == EOS_TOKEN) break;
            output.add(currentToken);
        }
        return convertToText(output);
    }
}
```

## 使用示例（机器翻译）
```java
public class Translator {
    public static void main(String[] args) {
        // 创建模型 (英译中任务)
        Seq2Seq model = new Seq2Seq(
            10000,   // 英文词表大小
            5000,    // 中文词表大小
            512,     // 词向量维度
            3        // LSTM层数
        );
        
        // 配置训练参数
        model.setTeacherForcingRatio(0.7f); // 教师强制比例
        model.setBeamWidth(5);             // Beam Search宽度
        
        // 加载数据集
        TranslationDataset dataset = new TranslationDataset(
            "data/eng2chi/train.en",
            "data/eng2chi/train.zh"
        );
        
        // 训练循环
        for (int epoch=0; epoch<30; epoch++) {
            float totalLoss = 0;
            for (Pair<String[], String[]> batch : dataset.getBatches(64)) {
                // 编码器前向
                float[][][] encStates = model.encode(batch.getKey());
                
                // 解码器训练
                float loss = model.trainDecoder(
                    encStates, 
                    batch.getValue()
                );
                
                totalLoss += loss;
                
                // 反向传播
                model.backward();
                model.update();
            }
            System.out.printf("Epoch %02d Loss: %.3f\n", 
                epoch+1, totalLoss/dataset.size());
        }
        
        // 翻译示例
        String english = "Hello world";
        String chinese = model.translate(english.split(" "));
        System.out.println(english + " => " + chinese);
    }
}
```

## 性能优化
1. **批处理加速**：并行处理多个序列
```java
public List<String> batchTranslate(List<String[]> inputs) {
    return inputs.parallelStream()
                 .map(this::translate)
                 .collect(Collectors.toList());
}
```

2. **内存优化**：状态缓存机制
```java
public class CachedAttention extends AttentionLayer {
    private float[][] cacheKeys;
    private float[][] cacheValues;
    
    public float[] calculate(float[][][] encoderStates, float[] query) {
        if (cacheKeys == null) {
            // 首次运行缓存计算结果
            cacheKeys = extractKeys(encoderStates);
            cacheValues = extractValues(encoderStates);
        }
        return calculateWithCache(query);
    }
}
```

## 常见问题
### Q1：如何提升长序列翻译质量？
- 解决方案：组合使用注意力机制
```java
public void setupMultiHeadAttention(int heads) {
    this.attention = new MultiHeadAttention(
        heads, 
        512,  // 模型维度
        64    // 每个头的维度
    );
}
```

### Q2：如何避免重复生成？
- 覆盖惩罚实现：
```java
public float[] applyCoveragePenalty(float[] logits, List<Integer> generated) {
    for (int token : generated) {
        logits[token] -= 1.0f; // 惩罚重复token
    }
    return logits;
}
```

## 扩展阅读
- [Transformer架构解析](/doc/api/transformer)
- [神经机器翻译进阶](/doc/extension/nmt-advanced)
- [对话系统实战指南](/doc/extension/dialogue-system)