# RNN 循环神经网络

## 概述
循环神经网络（Recurrent Neural Network）是一种用于处理序列数据的神经网络，通过循环隐藏状态保留时序信息，广泛应用于自然语言处理、时间序列预测等领域。

## 核心结构
```java
/**
 * RNN基础实现
 * 结构：输入层 → 循环层 → 输出层
 */
public class RNN {
    private float[][] Wxh;    // 输入到隐藏层权重
    private float[][] Whh;    // 隐藏层到隐藏层权重
    private float[][] Why;    // 隐藏层到输出层权重
    private float[] bh;       // 隐藏层偏置
    private float[] by;       // 输出层偏置
    private float[] h;        // 隐藏状态
    
    // 网络参数
    private int inputSize;    // 输入维度
    private int hiddenSize;   // 隐藏层维度
    private int outputSize;   // 输出维度
}
```

## 完整实现

### 1. 网络初始化
```java
public RNN(int inputSize, int hiddenSize, int outputSize) {
    this.inputSize = inputSize;
    this.hiddenSize = hiddenSize;
    this.outputSize = outputSize;
    
    // 初始化权重矩阵
    Wxh = new float[hiddenSize][inputSize];
    Whh = new float[hiddenSize][hiddenSize];
    Why = new float[outputSize][hiddenSize];
    
    // Xavier初始化
    initializeWeights(Wxh, inputSize + hiddenSize);
    initializeWeights(Whh, 2 * hiddenSize);
    initializeWeights(Why, hiddenSize + outputSize);
    
    // 初始化偏置
    bh = new float[hiddenSize];
    by = new float[outputSize];
}

private void initializeWeights(float[][] matrix, int fanAvg) {
    float scale = (float) Math.sqrt(6.0 / fanAvg);
    for (int i=0; i<matrix.length; i++) {
        for (int j=0; j<matrix[i].length; j++) {
            matrix[i][j] = (float) (-scale + Math.random() * 2 * scale);
        }
    }
}
```

### 2. 时间步前向传播
```java
/**
 * 处理单个时间步的前向计算
 * @param x 当前时间步输入 (inputSize维)
 * @param h_prev 前一时间步隐藏状态
 * @return 当前输出和隐藏状态
 */
public float[] stepForward(float[] x, float[] h_prev) {
    // 计算新隐藏状态
    float[] h = new float[hiddenSize];
    for (int i=0; i<hiddenSize; i++) {
        float sum = bh[i];
        for (int j=0; j<inputSize; j++) sum += Wxh[i][j] * x[j];
        for (int j=0; j<hiddenSize; j++) sum += Whh[i][j] * h_prev[j];
        h[i] = (float) Math.tanh(sum);
    }
    
    // 计算输出
    float[] y = new float[outputSize];
    for (int i=0; i<outputSize; i++) {
        float sum = by[i];
        for (int j=0; j<hiddenSize; j++) sum += Why[i][j] * h[j];
        y[i] = sum;  // 线性输出，可接Softmax
    }
    
    return y;
}
```

### 3. 序列处理
```java
/**
 * 处理完整输入序列
 * @param inputs 时序数据 [time_steps][inputSize]
 * @return 每个时间步的输出
 */
public float[][] forwardSequences(float[][] inputs) {
    float[][] outputs = new float[inputs.length][outputSize];
    float[] h_prev = new float[hiddenSize];  // 初始隐藏状态
    
    for (int t=0; t<inputs.length; t++) {
        float[] x = inputs[t];
        float[] y = stepForward(x, h_prev);
        h_prev = Arrays.copyOf(h_prev, h_prev.length); // 保存当前隐藏状态
        outputs[t] = y;
    }
    
    return outputs;
}
```

## 使用示例（股票预测）
```java
public class StockPredictor {
    public static void main(String[] args) {
        // 创建RNN (5个特征输入，128隐藏单元，1个输出值)
        RNN model = new RNN(5, 128, 1);
        
        // 配置训练参数
        model.setLearningRate(0.001f);
        model.setLossFunction(new MSELoss());
        
        // 加载股票数据 [时间步][开盘价,收盘价,最高,最低,成交量]
        float[][][] trainData = loadCSV("stock_data.csv");
        
        // 训练循环
        for (int epoch=0; epoch<100; epoch++) {
            float totalLoss = 0;
            for (float[][] sequence : trainData) {
                // 前向传播
                float[][] outputs = model.forwardSequences(sequence);
                
                // 计算损失（预测最后一天收盘价）
                float[] pred = outputs[outputs.length-1];
                float[] target = {sequence[sequence.length-1][1]};
                float loss = model.lossFunction.calculate(pred, target);
                
                // 反向传播
                model.backward(sequence, target);
                
                totalLoss += loss;
            }
            System.out.printf("Epoch %03d Loss: %.4f\n", epoch+1, totalLoss/trainData.length);
        }
    }
}
```

## 性能优化
1. **时序并行处理**：使用批处理加速
```java
public float[][][] batchForward(float[][][] batchInputs) {
    float[][][] batchOutputs = new float[batchInputs.length][][];
    Arrays.parallelSetAll(batchOutputs, i -> forwardSequences(batchInputs[i]));
    return batchOutputs;
}
```

2. **梯度裁剪**：防止梯度爆炸
```java
public void clipGradients(float maxNorm) {
    float norm = 0;
    for (float[] row : dWxh) norm += squaredSum(row);
    for (float[] row : dWhh) norm += squaredSum(row);
    norm = (float) Math.sqrt(norm);
    
    if (norm > maxNorm) {
        float scale = maxNorm / (norm + 1e-6f);
        scaleMatrix(dWxh, scale);
        scaleMatrix(dWhh, scale);
        // 其他参数同理...
    }
}
```

## 常见问题
### Q1：长期依赖问题如何解决？
- 解决方案：使用LSTM或GRU结构
```java
public class LSTM extends RNN {
    // 新增门控机制参数
    private float[][] Wf; // 遗忘门权重
    private float[][] Wi; // 输入门权重
    private float[][] Wo; // 输出门权重
    
    // 实现门控计算逻辑...
}
```

### Q2：如何处理变长序列？
- 动态padding实现：
```java
public float[][] processVariableLength(float[][] seq) {
    int maxLen = 100; // 最大序列长度
    float[][] padded = new float[maxLen][inputSize];
    for (int i=0; i<Math.min(seq.length, maxLen); i++) {
        padded[i] = Arrays.copyOf(seq[i], inputSize);
    }
    return padded;
}
```

## 扩展阅读
- [文本生成实战](/doc/extension/text-generation)
- [注意力机制详解](/doc/extension/attention-mechanism)
- [序列建模最佳实践](/doc/extension/sequence-modeling)