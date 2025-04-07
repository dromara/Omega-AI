# ResNet 残差网络

## 概述
残差网络（Residual Network）通过引入跳跃连接解决深层网络梯度消失问题，支持训练超过1000层的深度模型。核心创新是残差学习框架：

$$\mathbf{y} = \mathcal{F}(\mathbf{x}, \{W_i\}) + \mathbf{x}$$

## 核心结构
```java
/**
 * 基础残差块实现
 * 结构：卷积 → 批归一化 → ReLU → 卷积 → 批归一化 → 跳跃连接 → ReLU
 */
public class ResidualBlock {
    private ConvLayer conv1;
    private BNLayer bn1;
    private ConvLayer conv2;
    private BNLayer bn2;
    private ConvLayer shortcut; // 维度匹配时的跳跃连接
    
    public float[] forward(float[] x) {
        // 残差路径
        float[] residual = x;
        residual = conv1.forward(residual);
        residual = bn1.forward(residual);
        residual = relu(residual);
        
        residual = conv2.forward(residual);
        residual = bn2.forward(residual);

        // 跳跃连接
        float[] identity = (shortcut != null) ? shortcut.forward(x) : x;
        
        return relu(add(residual, identity));
    }
}
```

## 完整实现

### 1. 网络初始化
```java
public class ResNet {
    private List<ResidualBlock> stages;
    private DenseLayer fcLayer;
    
    // 构建不同深度的ResNet
    public static ResNet createResNet(int depth) {
        int[] blockNums;
        if (depth == 18) blockNums = new int[]{2, 2, 2, 2};
        else if (depth == 34) blockNums = new int[]{3, 4, 6, 3};
        else if (depth == 50) blockNums = new int[]{3, 4, 6, 3}; // 使用Bottleneck
        else throw new IllegalArgumentException("Unsupported depth");
        
        return new ResNet(blockNums, depth >= 50);
    }
    
    // Bottleneck块实现（用于50层以上）
    private class BottleneckBlock {
        private ConvLayer conv1x1_a;
        private ConvLayer conv3x3;
        private ConvLayer conv1x1_b;
        // ... 其他实现类似基础残差块
    }
}
```

### 2. 前向传播
```java
public float[] forward(float[] x) {
    // 输入预处理
    x = stem(x); // 初始卷积层
    
    // 残差阶段
    for (ResidualBlock block : stages) {
        x = block.forward(x);
    }
    
    // 分类输出
    x = globalAvgPool(x);
    return fcLayer.forward(x);
}

// 初始卷积层组
private float[] stem(float[] x) {
    x = conv7x7(x);    // 7x7卷积
    x = bn.forward(x);
    x = relu(x);
    x = maxPool3x3(x); // 3x3最大池化
    return x;
}
```

## 使用示例（CIFAR-10分类）
```java
public class CifarClassifier {
    public static void main(String[] args) {
        // 创建ResNet-34
        ResNet model = ResNet.createResNet(34);
        
        // 替换最后的全连接层
        model.setFCLayer(512, 10); // CIFAR-10的10分类
        
        // 配置混合精度训练
        model.enableMixedPrecision();
        
        // 加载数据集
        CIFARDataset dataset = new CIFARDataset("data/cifar-10-batches-bin");
        
        // 训练配置
        model.setOptimizer(new AdamW(0.001f));
        model.setLoss(new CrossEntropyLoss());
        
        // 训练循环
        for (int epoch=0; epoch<100; epoch++) {
            float acc = model.trainEpoch(dataset);
            System.out.printf("Epoch %02d Acc: %.2f%%\n", epoch+1, acc*100);
        }
    }
}
```

## 性能优化
1. **通道分割策略**：优化显存使用
```java
public class MemoryOptimizedBlock extends ResidualBlock {
    // 在前向传播中分割计算图
    public float[] forward(float[] x) {
        // 分割计算图减少峰值显存
        return checkpoint(super.forward(x));
    }
}
```

2. **分布式训练**：数据并行实现
```java
public void enableDataParallel(int gpuCount) {
    this.replicas = new ResNet[gpuCount];
    Arrays.parallelSetAll(replicas, i -> cloneModel());
    this.parallel = true;
}
```

## 常见问题
### Q1：残差连接失效怎么办？
- 解决方案：使用预激活结构
```java
public class PreActBlock extends ResidualBlock {
    // 调整顺序为：BN → ReLU → Conv
    public float[] forward(float[] x) {
        float[] identity = x;
        
        x = bn1.forward(x);
        x = relu(x);
        x = conv1.forward(x);
        
        x = bn2.forward(x);
        x = relu(x);
        x = conv2.forward(x);
        
        return add(x, identity);
    }
}
```

### Q2：如何迁移学习？
- 特征提取实现：
```java
public void freezeBackbone() {
    for (ResidualBlock block : stages) {
        block.setTrainable(false); // 冻结特征提取层
    }
    fcLayer.setTrainable(true); // 仅训练分类层
}
```

## 扩展阅读
- [ResNeXt改进架构](/doc/extension/resnext)
- [DenseNet密集连接](/doc/extension/densenet)
- [目标检测应用](/doc/extension/object-detection)