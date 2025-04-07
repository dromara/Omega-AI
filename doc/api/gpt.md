# GPT 生成式预训练模型

## 概述
生成式预训练Transformer（Generative Pre-trained Transformer）通过自回归方式预训练大规模语言模型，核心特点包括：
- 单向注意力机制
- 大规模文本预训练
- 零样本/小样本学习能力

## 核心结构
```java
/**
 * GPT-2基础实现
 * 包含多层Transformer解码器堆叠
 */
public class GPT {
    private List<TransformerBlock> layers;  // Transformer块堆叠
    private Embedding tokenEmbedding;      // 词嵌入
    private PositionEmbedding posEmbedding; // 位置嵌入
    private int contextSize;               // 上下文窗口
    
    public GPT(int vocabSize, int layers, int heads, int dModel) {
        this.layers = IntStream.range(0, layers)
            .mapToObj(i -> new TransformerBlock(heads, dModel))
            .collect(Collectors.toList());
    }
}
```

## 完整实现

### 1. 自注意力层
```java
public class CausalSelfAttention {
    private Linear query;
    private Linear key;
    private Linear value;
    
    public float[][] forward(float[][] x) {
        float[][] Q = query.forward(x);
        float[][] K = key.forward(x);
        float[][] V = value.forward(x);
        
        // 因果掩码防止信息泄露
        float[][] mask = createCausalMask(x.length);
        float[][] attn = scaledDotProduct(Q, K, V, mask);
        return attn;
    }
    
    private float[][] createCausalMask(int seqLen) {
        float[][] mask = new float[seqLen][seqLen];
        for (int i=0; i<seqLen; i++) {
            Arrays.fill(mask[i], 0, i+1, 1.0f); // 下三角矩阵
        }
        return mask;
    }
}
```

### 2. 文本生成
```java
public class TextGenerator {
    public String generate(String prompt, int maxLength) {
        List<Integer> tokens = tokenizer.encode(prompt);
        for (int i=0; i<maxLength; i++) {
            float[][] logits = model.forward(tokens);
            int next = sample(logits[logits.length-1]); // 基于最后位置预测
            tokens.add(next);
            if (next == tokenizer.eosToken()) break;
        }
        return tokenizer.decode(tokens);
    }
    
    private int sample(float[] logits) {
        // 温度采样
        float temperature = 0.7f;
        float[] probs = softmax(logits / temperature);
        return multinomialSample(probs);
    }
}
```

## 使用示例（代码补全）
```java
public class CodeAssistant {
    public static void main(String[] args) {
        // 加载预训练GPT-2
        GPT model = GPT.loadPretrained("gpt2_java.bin");
        
        // 代码补全示例
        String prompt = "public class HelloWorld { public static void main(String[] args) {";
        String completion = model.generate(prompt, 128, 0.9f);
        
        System.out.println("补全结果：");
        System.out.println(completion);
    }
}

// 输出示例：
// public class HelloWorld { 
//   public static void main(String[] args) {
//     System.out.println("Hello, World!"); 
//   }
// }
```

## 性能优化
1. **KV缓存**：加速自回归生成
```java
public class GenerationCache {
    private Map<Integer, float[][]> keyCache = new HashMap<>();
    private Map<Integer, float[][]> valueCache = new HashMap<>();
    
    public float[][] getKey(int layer) { return keyCache.get(layer); }
    public void cacheKey(int layer, float[][] keys) { keyCache.put(layer, keys); }
}
```

2. **模型并行**：拆分参数到多GPU
```java
public void parallelizeAcrossGPUs(List<Integer> gpuIds) {
    int layersPerGPU = layers.size() / gpuIds.size();
    for (int i=0; i<layers.size(); i++) {
        layers.get(i).toDevice(gpuIds.get(i/layersPerGPU));
    }
}
```

## 常见问题
### Q1：生成文本重复怎么办？
- 解决方案：使用Top-p采样
```java
private int nucleusSample(float[] logits, float p=0.9f) {
    float[] probs = softmax(logits);
    Arrays.sort(probs);
    
    float cumulative = 0;
    int cutoff = 0;
    for (int i=probs.length-1; i>=0; i--) {
        cumulative += probs[i];
        if (cumulative > p) {
            cutoff = i;
            break;
        }
    }
    return randomSelect(probs, cutoff);
}
```

### Q2：如何避免有害内容？
- 实现安全层：
```java
public class SafetyFilter {
    public boolean isSafe(String text) {
        return !containsHarmfulKeywords(text) 
            && sentimentAnalyze(text) > 0.5f;
    }
}
```

## 扩展阅读
- [BERT双向预训练模型](/doc/extension/bert)
- [模型微调指南](/doc/extension/fine-tuning)
- [大模型分布式训练](/doc/extension/distributed-training)