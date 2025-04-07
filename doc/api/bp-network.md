# BPNetwork 反向传播神经网络

## 概述
BP（Back Propagation）神经网络是一种通过误差反向传播算法进行训练的多层前馈神经网络，包含输入层、隐藏层和输出层，适用于各种监督学习任务。

## 网络结构
```java
/**
 * BP神经网络核心实现
 * 结构：输入层 → 隐藏层（可多层） → 输出层
 */
public class BPNetwork {
    private List<Layer> layers;          // 网络层集合
    private LossFunction lossFunction;   // 损失函数
    private Optimizer optimizer;         // 优化器
    
    // 网络参数
    private int inputSize;               // 输入层维度
    private int[] hiddenSizes;           // 隐藏层结构
    private int outputSize;              // 输出层维度
    private float learningRate;          // 学习率
    
    public BPNetwork(int inputSize, int[] hiddenSizes, int outputSize) {
        this.inputSize = inputSize;
        this.hiddenSizes = hiddenSizes;
        this.outputSize = outputSize;
        initializeLayers();
    }
}
```

## 完整实现

### 1. 网络初始化
```java
private void initializeLayers() {
    layers = new ArrayList<>();
    int prevSize = inputSize;
    
    // 构建隐藏层
    for (int size : hiddenSizes) {
        layers.add(new FullyConnectedLayer(prevSize, size));
        layers.add(new SigmoidActivation()); // 默认使用Sigmoid激活
        prevSize = size;
    }
    
    // 输出层
    layers.add(new FullyConnectedLayer(prevSize, outputSize));
    layers.add(new SoftmaxActivation());     // 分类任务使用Softmax
}

/**
 * 添加自定义隐藏层
 * @param layer 全连接层
 * @param activation 激活函数
 */
public void addHiddenLayer(FullyConnectedLayer layer, Activation activation) {
    layers.add(layers.size()-2, layer);  // 在输出层前插入
    layers.add(layers.size()-2, activation);
}
```

### 2. 前向传播
```java
public float[] forward(float[] input) {
    if (input.length != inputSize) {
        throw new IllegalArgumentException("输入维度错误，应为 " + inputSize);
    }
    
    float[] current = input;
    for (Layer layer : layers) {
        current = layer.forward(current);
    }
    return current;
}
```

### 3. 反向传播
```java
public void backward(float[] input, float[] target) {
    // 前向计算
    float[] output = forward(input);
    
    // 计算输出层梯度
    float[] delta = lossFunction.gradient(output, target);
    
    // 反向传播
    for (int i = layers.size()-1; i >= 0; i--) {
        delta = layers.get(i).backward(delta, learningRate);
    }
}

// 交叉熵损失实现
private static class CrossEntropyLoss implements LossFunction {
    public float loss(float[] output, float[] target) {
        float sum = 0;
        for (int i=0; i<output.length; i++) {
            sum += target[i] * Math.log(output[i] + 1e-10f);
        }
        return -sum;
    }
    
    public float[] gradient(float[] output, float[] target) {
        float[] grad = new float[output.length];
        for (int i=0; i<output.length; i++) {
            grad[i] = output[i] - target[i];
        }
        return grad;
    }
}
```

## 使用示例（MNIST分类）
```java
public class DigitClassifier {
    public static void main(String[] args) {
        // 创建网络 (784输入, 两个256单元的隐藏层, 10分类输出)
        BPNetwork model = new BPNetwork(784, new int[]{256, 256}, 10);
        
        // 配置训练参数
        model.setLearningRate(0.01f);
        model.setLossFunction(new CrossEntropyLoss());
        model.setOptimizer(new AdamOptimizer());
        
        // 加载数据集
        MNISTDataset dataset = new MNISTDataset("train-images.idx3-ubyte", 
                                              "train-labels.idx1-ubyte");
        
        // 训练循环
        for (int epoch=0; epoch<10; epoch++) {
            float totalLoss = 0;
            for (Example example : dataset) {
                model.forward(example.pixels);
                model.backward(example.pixels, example.label);
                totalLoss += model.getLastLoss();
            }
            System.out.printf("Epoch %d Loss: %.4f\n", epoch+1, totalLoss/dataset.size());
        }
    }
}
```

## 性能优化技巧
1. **矩阵加速**：使用矩阵运算代替循环
```java
// 全连接层前向传播优化
public float[] forward(float[] input) {
    float[] output = new float[weights.length];
    Arrays.parallelSetAll(output, i -> {
        float sum = 0;
        for (int j=0; j<weights[i].length; j++) {
            sum += weights[i][j] * input[j];
        }
        return sum + bias[i];
    });
    return output;
}
```

2. **权重初始化**：Xavier初始化方法
```java
public void initializeWeights() {
    float scale = (float) Math.sqrt(6.0 / (inputSize + outputSize));
    for (int i=0; i<weights.length; i++) {
        for (int j=0; j<weights[i].length; j++) {
            weights[i][j] = (float) (-scale + Math.random() * 2 * scale);
        }
    }
}
```

## 常见问题
### Q1：如何防止过拟合？
- 解决方案：添加L2正则化、使用Dropout层、早停法
```java
// 添加Dropout层
model.addHiddenLayer(new FullyConnectedLayer(256, 128), new Dropout(0.5f));
```

### Q2：梯度爆炸如何处理？
- 梯度裁剪实现：
```java
public float[] clipGradients(float[] gradients, float maxNorm) {
    float norm = 0;
    for (float g : gradients) norm += g*g;
    norm = (float) Math.sqrt(norm);
    
    if (norm > maxNorm) {
        float scale = maxNorm / (norm + 1e-6f);
        for (int i=0; i<gradients.length; i++) {
            gradients[i] *= scale;
        }
    }
    return gradients;
}
```

## 扩展阅读
- [手写数字识别实战](/doc/extension/mnist-tutorial)
- [自动微分原理](/doc/extension/autograd)
- [神经网络可视化工具](/doc/extension/visualization)