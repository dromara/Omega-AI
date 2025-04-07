# LLaMA 生成式预训练模型

## 概述
LLaMA（Large Language Model Meta AI）是Meta开源的预训练语言模型系列，核心创新包括：
- 旋转位置编码（RoPE）
- 改进的Transformer架构
- 高效的大模型训练策略

## 核心结构
```java
/**
 * LLaMA基础实现
 * 包含预归一化、旋转位置编码和门控激活
 */
public class LLaMA {
    private List<TransformerBlock> layers;  // Transformer块堆叠
    private RMSNorm preNorm;               // 预归一化
    private RoPEEncoding ropeEncoding;    // 旋转位置编码
    private SwiGLU gateActivation;        // 门控激活
    
    // 模型参数
    private int dim;          // 隐藏层维度
    private int numHeads;     // 注意力头数
    private int numKvHeads;   // Key/Value头数（分组查询）
}
```

## 完整实现

### 1. 旋转位置编码
```java
public class RoPEEncoding {
    public float[][] apply(float[][] x, int pos) {
        float[][] output = new float[x.length][x[0].length];
        for (int i=0; i<x.length; i++) {
            for (int j=0; j<x[i].length; j+=2) {
                float theta = pos / Math.pow(10000, 2*(j/2)/dim);
                float cos = (float) Math.cos(theta);
                float sin = (float) Math.sin(theta);
                output[i][j]   = x[i][j] * cos - x[i][j+1] * sin;
                output[i][j+1] = x[i][j] * sin + x[i][j+1] * cos;
            }
        }
        return output;
    }
}
```

### 2. 门控注意力块
```java
public class GateAttention {
    private Linear Wq;     // 查询投影
    private Linear Wk;     // 键投影
    private Linear Wv;     // 值投影
    private Linear Wo;     // 输出投影
    
    public float[][] forward(float[][] x) {
        // 预归一化
        x = preNorm.forward(x);
        
        // 旋转位置编码
        float[][] q = applyRoPE(Wq.forward(x));
        float[][] k = applyRoPE(Wk.forward(x));
        
        // 分组查询注意力
        float[][] attn = groupQueryAttention(q, k, Wv.forward(x));
        return Wo.forward(attn);
    }
}
```

## 使用示例（对话生成）
```java
public class ChatBot {
    public static void main(String[] args) {
        // 加载LLaMA-2 7B
        LLaMA model = LLaMA.load("llama2-7b.bin");
        
        // 配置生成参数
        model.setTemperature(0.7f)
             .setTopP(0.9f)
             .setMaxLength(512);
        
        // 对话循环
        while (true) {
            String input = getInput("用户：");
            String response = model.generate(
                "<human>: " + input + "\n<bot>:"
            );
            System.out.println("助手：" + response);
        }
    }
}
```

## 性能优化
1. **KV缓存优化**： 
```java
public class KVCache {
    private Map<Integer, float[][]> keyCache = new ConcurrentHashMap<>();
    private Map<Integer, float[][]> valueCache = new ConcurrentHashMap<>();
    
    public void update(int layer, float[][] newKey, float[][] newValue) {
        keyCache.compute(layer, (k, v) -> concat(v, newKey));
        valueCache.compute(layer, (k, v) -> concat(v, newValue));
    }
}
```

2. **量化部署**：
```java
public class Quantizer {
    public static byte[] quantize(float[] weights) {
        byte[] quantized = new byte[weights.length];
        float scale = 127 / maxAbs(weights);
        for (int i=0; i<weights.length; i++) {
            quantized[i] = (byte) Math.round(weights[i] * scale);
        }
        return quantized;
    }
}
```

## 常见问题
### Q1：如何处理长文本记忆？
- 扩展上下文窗口：
```java
public void extendContext(int newSize) {
    // 线性插值旋转角度
    float scale = (float) newSize / originalSize;
    for (RoPEEncoding layer : ropeLayers) {
        layer.scaleTheta(scale);
    }
}
```

### Q2：如何支持中文？
- 分词器适配：
```java
public class ChineseTokenizer {
    public List<String> tokenize(String text) {
        // 使用SentencePiece中文分词
        return sentencePiece.encode(text);
    }
}
```

## 扩展阅读
- [混合专家模型实现](/doc/extension/moe)
- [RLHF对齐训练](/doc/extension/rlhf)
- [LLaMA边缘部署](/doc/extension/edge-llama)