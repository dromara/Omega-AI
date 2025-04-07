# LSTM 长短期记忆网络

## 概述
长短期记忆网络（Long Short-Term Memory）是RNN的改进型，通过门控机制解决长期依赖问题，擅长处理需要长期记忆的时序数据。

## 核心结构
```java
/**
 * LSTM基础实现
 * 包含输入门、遗忘门、输出门三种门控机制
 */
public class LSTM {
    // 门控权重矩阵
    private float[][] Wf;    // 遗忘门权重
    private float[][] Wi;    // 输入门权重
    private float[][] Wo;    // 输出门权重
    private float[][] Wc;    // 候选状态权重
    
    // 偏置向量
    private float[] bf;      // 遗忘门偏置
    private float[] bi;      // 输入门偏置
    private float[] bo;      // 输出门偏置
    private float[] bc;      // 候选状态偏置
    
    // 状态存储
    private float[] c;       // 细胞状态
    private float[] h;       // 隐藏状态
}
```

## 完整实现

### 1. 网络初始化
```java
public LSTM(int inputSize, int hiddenSize) {
    // 初始化门控权重
    Wf = new float[hiddenSize][inputSize + hiddenSize];
    Wi = new float[hiddenSize][inputSize + hiddenSize];
    Wo = new float[hiddenSize][inputSize + hiddenSize];
    Wc = new float[hiddenSize][inputSize + hiddenSize];
    
    // Xavier初始化
    initializeWeights(Wf, (inputSize + hiddenSize) * 2);
    initializeWeights(Wi, (inputSize + hiddenSize) * 2);
    initializeWeights(Wo, (inputSize + hiddenSize) * 2);
    initializeWeights(Wc, (inputSize + hiddenSize) * 2);
    
    // 初始化偏置
    bf = new float[hiddenSize];
    bi = new float[hiddenSize];
    bo = new float[hiddenSize];
    bc = new float[hiddenSize];
}

private void initializeWeights(float[][] matrix, int fanAvg) {
    float scale = (float) Math.sqrt(2.0 / fanAvg); // He初始化
    for (int i=0; i<matrix.length; i++) {
        for (int j=0; j<matrix[i].length; j++) {
            matrix[i][j] = (float) (-scale + Math.random() * 2 * scale);
        }
    }
}
```

### 2. 时间步前向传播
```java
public float[] stepForward(float[] x, float[] h_prev, float[] c_prev) {
    // 拼接输入和前一隐藏状态
    float[] concat = concatenate(x, h_prev);
    
    // 计算各个门控信号
    float[] ft = sigmoid(matmul(Wf, concat) + bf); // 遗忘门
    float[] it = sigmoid(matmul(Wi, concat) + bi); // 输入门
    float[] ot = sigmoid(matmul(Wo, concat) + bo); // 输出门
    float[] ct_hat = tanh(matmul(Wc, concat) + bc); // 候选状态
    
    // 更新细胞状态
    c = elementWiseMultiply(ft, c_prev) + elementWiseMultiply(it, ct_hat);
    
    // 计算新隐藏状态
    h = elementWiseMultiply(ot, tanh(c));
    
    return h;
}
```

### 3. 序列处理
```java
public float[][] processSequence(float[][] inputs) {
    float[][] outputs = new float[inputs.length][h.length];
    float[] c_prev = new float[h.length]; // 初始细胞状态
    
    for (int t=0; t<inputs.length; t++) {
        h = stepForward(inputs[t], h, c_prev);
        c_prev = Arrays.copyOf(c, c.length);
        outputs[t] = Arrays.copyOf(h, h.length);
    }
    
    return outputs;
}
```

## 使用示例（文本生成）
```java
public class TextGenerator {
    public static void main(String[] args) {
        // 创建LSTM (256维词向量输入，512隐藏单元)
        LSTM model = new LSTM(256, 512);
        
        // 配置训练参数
        model.setLearningRate(0.001f);
        model.setGradientClip(5.0f); // 梯度裁剪阈值
        
        // 加载文本数据集
        TextDataset dataset = new TextDataset("shakespeare.txt");
        
        // 训练循环
        for (int epoch=0; epoch<50; epoch++) {
            float totalLoss = 0;
            for (List<String> batch : dataset.getBatches(64)) {
                // 转换输入序列
                float[][][] sequences = dataset.vectorize(batch);
                
                // 前向传播
                float[][][] outputs = model.forward(sequences);
                
                // 计算损失并反向传播
                totalLoss += model.calculateLoss(outputs, batch);
                model.backward();
                
                // 参数更新
                model.updateParameters();
            }
            System.out.printf("Epoch %02d Loss: %.3f\n", epoch+1, totalLoss/dataset.size());
        }
        
        // 文本生成示例
        String generated = model.generateText("The ", 100); 
        System.out.println("Generated text:\n" + generated);
    }
}
```

## 性能优化
1. **门控并行计算**：使用矩阵拼接加速
```java
private float[][] computeGates(float[][] concatInput) {
    float[][] allGates = new float[4][hiddenSize];
    Arrays.parallelSetAll(allGates, i -> {
        return matMulBlock(gateWeights[i], concatInput);
    });
    return allGates; // [forget, input, output, candidate]
}
```

2. **内存优化**：状态缓存重用
```java
public void enableMemoryOptimization() {
    this.cache = new float[][][] { // 预分配内存
        new float[hiddenSize][], 
        new float[hiddenSize][],
        new float[hiddenSize][]
    };
}
```

## 常见问题
### Q1：如何防止梯度爆炸？
- 解决方案：梯度裁剪 + 权重正则化
```java
public void clipGradients(float threshold) {
    float norm = calculateGradientNorm();
    if (norm > threshold) {
        float scale = threshold / (norm + 1e-6f);
        scaleGradients(scale);
    }
}
```

### Q2：输出文本重复怎么办？
- 采样策略改进：
```java
public String sampleCharacter(float[] probs) {
    // 温度参数控制多样性
    float temperature = 0.7f;
    float[] scaled = scaleProbs(probs, temperature);
    return selectByDistribution(scaled);
}
```

## 扩展阅读
- [双向LSTM实现](/doc/extension/bilstm)
- [注意力LSTM模型](/doc/extension/attention-lstm)
- [LSTM硬件加速](/doc/extension/hardware-accel)