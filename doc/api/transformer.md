# Transformer 自注意力模型

## 概述
Transformer模型基于自注意力机制，完全摒弃了循环和卷积结构，在机器翻译、文本生成等任务中取得突破性进展。核心组件包括：

- 多头自注意力机制
- 位置前馈网络
- 位置编码
- 残差连接与层归一化

## 核心结构
```java
/**
 * Transformer基础实现
 * 包含编码器堆叠、解码器堆叠和嵌入层
 */
public class Transformer {
    private List<EncoderBlock> encoders;  // 编码器堆叠
    private List<DecoderBlock> decoders; // 解码器堆叠
    private EmbeddingLayer tokenEmbedding; // 词嵌入
    private PositionalEncoding posEncoding; // 位置编码
    
    // 模型参数
    private int dModel;         // 模型维度
    private int numHeads;       // 注意力头数
    private int vocabSize;      // 词表大小
}
```

## 完整实现

### 1. 多头自注意力
```java
public class MultiHeadAttention {
    private Linear[] Wq;  // 查询矩阵
    private Linear[] Wk;  // 键矩阵
    private Linear[] Wv;  // 值矩阵
    private Linear Wo;    // 输出矩阵
    
    public float[][] forward(float[][] Q, float[][] K, float[][] V, 
                            float[][] mask) {
        int batchSize = Q.length;
        List<float[][]> heads = new ArrayList<>();
        
        // 分头处理
        for (int h=0; h<numHeads; h++) {
            float[][] q = matMul(Q, Wq[h].weight);
            float[][] k = matMul(K, Wk[h].weight);
            float[][] v = matMul(V, Wv[h].weight);
            
            // 缩放点积注意力
            float[][] attn = scaledDotProductAttention(q, k, v, mask);
            heads.add(attn);
        }
        
        // 合并多头输出
        float[][] output = concatenate(heads);
        return matMul(output, Wo.weight);
    }
}
```

### 2. 位置编码
```java
public class PositionalEncoding {
    public float[][] encode(int seqLength, int dModel) {
        float[][] pe = new float[seqLength][dModel];
        for (int pos=0; pos<seqLength; pos++) {
            for (int i=0; i<dModel/2; i++) {
                float angle = pos / (float)Math.pow(10000, 2*i/(float)dModel);
                pe[pos][2*i] = (float) Math.sin(angle);
                pe[pos][2*i+1] = (float) Math.cos(angle);
            }
        }
        return pe;
    }
}
```

## 使用示例（英法翻译）
```java
public class Translator {
    public static void main(String[] args) {
        // 创建Transformer模型
        Transformer model = new Transformer()
            .setVocabSize(30000)  // 共享词表
            .setModelDim(512)
            .setHeadNum(8)
            .setEncoderLayers(6)
            .setDecoderLayers(6);
        
        // 加载预训练词嵌入
        model.loadEmbeddings("embeddings.bin");
        
        // 训练配置
        model.setOptimizer(new Adam(0.0001f, new StepLR(10000, 0.5)))
            .setLabelSmoothing(0.1f)
            .setMaxSeqLength(256);
        
        // 加载数据集
        BilingualDataset dataset = new BilingualDataset(
            "data/english.txt", 
            "data/french.txt"
        );
        
        // 训练循环
        for (int epoch=0; epoch<100; epoch++) {
            float loss = model.trainEpoch(dataset);
            float bleu = evaluateOnValidSet();
            System.out.printf("Epoch %02d Loss:%.3f BLEU:%.2f\n", 
                epoch+1, loss, bleu);
        }
        
        // 翻译示例
        String english = "Hello world";
        String french = model.translate(english);
        System.out.println(english + " => " + french);
    }
}
```

## 性能优化
1. **矩阵分块计算**：加速大矩阵运算
```java
public float[][] optimizedMatMul(float[][] a, float[][] b) {
    int blockSize = 64; // 分块尺寸
    float[][] result = new float[a.length][b[0].length];
    
    for (int iBlock=0; iBlock<a.length; iBlock+=blockSize) {
        for (int jBlock=0; jBlock<b[0].length; jBlock+=blockSize) {
            for (int kBlock=0; kBlock<a[0].length; kBlock+=blockSize) {
                // 分块矩阵乘法
                multiplyBlock(a, b, result, iBlock, jBlock, kBlock, blockSize);
            }
        }
    }
    return result;
}
```

2. **融合操作**：减少内存访问
```java
public float[][] fusedDropoutLayerNorm(float[][] x) {
    // 合并Dropout和LayerNorm计算
    float[][] dropped = dropout(x, 0.1f);
    return layerNorm(dropped);
}
```

## 常见问题
### Q1：如何应对长序列内存溢出？
- 分块注意力实现：
```java
public class BlockwiseAttention {
    public float[][] forward(float[][] Q, float[][] K, float[][] V) {
        int blockSize = 256;
        List<float[][]> blocks = new ArrayList<>();
        
        for (int i=0; i<Q.length; i+=blockSize) {
            float[][] qBlock = getBlock(Q, i, blockSize);
            // 分块计算注意力...
            blocks.add(attnBlock);
        }
        return concatenate(blocks);
    }
}
```

### Q2：训练不稳定如何处理？
- 学习率预热策略：
```java
public class WarmupOptimizer extends Adam {
    private int step = 0;
    
    public void update() {
        float lr = baseLR * Math.min(step++ / 10000.0f, 1.0f);
        setLearningRate(lr);
        super.update();
    }
}
```

## 扩展阅读
- [BERT预训练模型](/doc/extension/bert)
- [ViT视觉Transformer](/doc/extension/vit)
- [Transformer量化部署](/doc/extension/quantization)