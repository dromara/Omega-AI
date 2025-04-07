# CNN 卷积神经网络

## 概述
卷积神经网络（Convolutional Neural Network）是一种专用于处理网格结构数据（如图像）的前馈神经网络，通过卷积层、池化层等结构自动提取空间特征。

## 核心结构
```java
/**
 * 卷积神经网络基础实现
 * 结构：输入层 → [[卷积层 → ReLU] × N → 池化层] × M → 全连接层 → 输出层
 */
public class CNN {
    private List<ConvLayer> convLayers = new ArrayList<>();
    private List<PoolingLayer> poolingLayers = new ArrayList<>();
    private FullyConnectedLayer fcLayer;
    private SoftmaxLayer outputLayer;
    
    // 网络参数
    private int inputWidth;
    private int inputHeight;
    private int channels;
    
    public CNN(int width, int height, int channels) {
        this.inputWidth = width;
        this.inputHeight = height;
        this.channels = channels;
    }
}
```

## 完整实现

### 1. 网络初始化
```java
/**
 * 添加卷积层
 * @param kernelSize 卷积核尺寸（正方形）
 * @param stride 步长
 * @param filters 滤波器数量
 */
public void addConvLayer(int kernelSize, int stride, int filters) {
    ConvLayer layer = new ConvLayer(kernelSize, stride, filters);
    convLayers.add(layer);
}

/**
 * 添加最大池化层
 * @param poolSize 池化窗口尺寸
 * @param stride 滑动步长
 */
public void addMaxPoolingLayer(int poolSize, int stride) {
    PoolingLayer layer = new PoolingLayer(poolSize, stride, PoolingType.MAX);
    poolingLayers.add(layer);
}
```

### 2. 前向传播
```java
public float[] forward(float[][][] input) {
    // 输入数据格式校验
    if (input.length != channels || input[0].length != inputHeight 
        || input[0][0].length != inputWidth) {
        throw new IllegalArgumentException("输入数据维度不匹配");
    }

    // 逐层处理
    float[][][] output = input;
    for (int i=0; i<convLayers.size(); i++) {
        output = convLayers.get(i).forward(output);
        output = ReLU(output);
        output = poolingLayers.get(i).forward(output);
    }

    // 全连接层处理
    float[] flattened = flatten(output);
    return fcLayer.forward(flattened);
}

// ReLU激活函数实现
private float[][][] ReLU(float[][][] input) {
    float[][][] output = new float[input.length][input[0].length][input[0][0].length];
    for (int c=0; c<input.length; c++) {
        for (int h=0; h<input[0].length; h++) {
            for (int w=0; w<input[0][0].length; w++) {
                output[c][h][w] = Math.max(0, input[c][h][w]);
            }
        }
    }
    return output;
}
```

### 3. 训练配置
```java
/**
 * 网络训练配置
 * @param learningRate 初始学习率 (默认0.001)
 * @param batchSize 批处理大小 (默认32)
 * @param lossFunction 损失函数 (支持CrossEntropy/MSE)
 */
public void setupTraining(float learningRate, int batchSize, String lossFunction) {
    this.optimizer = new AdamOptimizer(learningRate);
    this.lossFunction = LossFunctionFactory.create(lossFunction);
    this.batchSize = batchSize;
    
    // 参数初始化
    for (ConvLayer layer : convLayers) {
        layer.initializeHe();
    }
    fcLayer.initializeXavier();
}
```

## 使用示例（MNIST识别）
```java
public class MNISTClassifier {
    public static void main(String[] args) {
        // 创建CNN (输入28x28灰度图)
        CNN model = new CNN(28, 28, 1);
        
        // 添加网络层
        model.addConvLayer(5, 1, 32);  // 32个5x5卷积核
        model.addMaxPoolingLayer(2, 2); // 2x2最大池化
        model.addConvLayer(3, 1, 64);
        model.addMaxPoolingLayer(2, 2);
        
        // 全连接层配置
        model.setFullyConnected(256, 10); // 256隐藏单元 → 10分类
        
        // 训练配置
        model.setupTraining(0.001f, 64, "CrossEntropy");
        
        // 加载数据
        MNISTDataset dataset = new MNISTDataset("train-images.idx3-ubyte", 
                                              "train-labels.idx1-ubyte");
                                              
        // 开始训练
        model.train(dataset, 10); // 训练10个epoch
    }
}
```

## 性能优化技巧
1. **并行计算**：使用Java并行流加速卷积运算
```java
// 在ConvLayer中启用并行计算
Arrays.parallelSetAll(output, i -> {
    // 卷积核计算逻辑...
});
```

2. **内存优化**：重用中间结果缓冲区
```java
private float[][][] convBuffer; // 预分配内存

public void forward(float[][][] input) {
    if (convBuffer == null) {
        convBuffer = new float[outputChannels]
                          [outputHeight][outputWidth];
    }
    // 使用buffer进行卷积计算...
}
```

## 常见问题
### Q1：出现梯度消失怎么办？
- 解决方案：使用He初始化、添加BatchNorm层、改用LeakyReLU

### Q2：输入输出维度如何计算？
输入尺寸公式：
$$输出宽度 = \lfloor \frac{输入宽度 - 卷积核尺寸 + 2×填充}{步长} \rfloor + 1$$

### Q3：如何选择卷积核数量？
- 推荐配置：
```
输入层 → 32-64 filters
中间层 → 64-256 filters
末尾层 → 256-512 filters
```

## 扩展阅读
- [ImageNet冠军模型解析](/doc/extension/alexnet)
- [目标检测实战教程](/doc/extension/object-detection)
- [多GPU训练指南](/doc/extension/multi-gpu)