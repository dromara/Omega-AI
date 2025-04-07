# VGG 深度卷积神经网络

## 概述
VGG网络由牛津大学Visual Geometry Group提出，通过重复使用3x3卷积核和2x2最大池化层构建深度网络，在ImageNet竞赛中取得优异表现。主要特点是小尺寸卷积核和网络深度。

## 核心结构
```java
/**
 * VGG基础实现
 * 结构：多个卷积块（包含连续卷积层） + 池化层 + 全连接层
 */
public class VGG {
    private List<ConvBlock> convBlocks;
    private List<DenseLayer> fcLayers;
    
    // 网络配置
    private static final int[] VGG16_CONFIG = {
        2,  // 卷积块1包含2个卷积层
        2,  // 卷积块2包含2个卷积层
        3,  // 卷积块3包含3个卷积层
        3,  // 卷积块4包含3个卷积层
        3   // 卷积块5包含3个卷积层
    };
}
```

## 完整实现

### 1. 卷积块实现
```java
public class ConvBlock {
    private List<ConvLayer> convLayers;
    private PoolingLayer poolLayer;
    
    public ConvBlock(int convNum, int inChannels, int outChannels) {
        convLayers = new ArrayList<>();
        for (int i=0; i<convNum; i++) {
            convLayers.add(new ConvLayer(3, 1, 1, // 3x3卷积核
                inChannels, outChannels));
            inChannels = outChannels; // 后续层通道数一致
        }
        poolLayer = new MaxPoolingLayer(2, 2); // 2x2池化
    }
    
    public float[][][] forward(float[][][] x) {
        for (ConvLayer conv : convLayers) {
            x = conv.forward(x);
            x = ReLU(x);
        }
        return poolLayer.forward(x);
    }
}
```

### 2. 网络构建
```java
public VGG(int numClasses, boolean batchNorm) {
    // 输入配置：224x224 RGB图像
    int channels = 3;
    convBlocks = new ArrayList<>();
    
    // 构建5个卷积块
    int[] config = VGG16_CONFIG;
    int[] outChannels = {64, 128, 256, 512, 512}; // 各块输出通道数
    
    for (int i=0; i<config.length; i++) {
        convBlocks.add(new ConvBlock(
            config[i], 
            channels, 
            outChannels[i],
            batchNorm
        ));
        channels = outChannels[i];
    }
    
    // 全连接层
    fcLayers = Arrays.asList(
        new DenseLayer(512*7*7, 4096), // 假设经过5次池化后特征图尺寸7x7
        new DenseLayer(4096, 4096),
        new DenseLayer(4096, numClasses)
    );
}
```

## 使用示例（ImageNet分类）
```java
public class ImageNetClassifier {
    public static void main(String[] args) {
        // 创建VGG16 (1000分类)
        VGG model = new VGG(1000, true); // 使用BN层
        
        // 加载预训练权重
        model.loadWeights("vgg16_weights.bin");
        
        // 配置推理参数
        model.setInputSize(224, 224)  // 输入尺寸
             .setMeanRGB(new float[]{0.485f, 0.456f, 0.406f})  // 均值
             .setStdRGB(new float[]{0.229f, 0.224f, 0.225f});  // 标准差
        
        // 预处理并推理
        float[] probs = model.predict("elephant.jpg");
        
        // 输出Top-5结果
        printTopK(probs, 5);
    }
}
```

## 性能优化
1. **内存优化**：特征图复用
```java
public class MemoryOptimizedBlock extends ConvBlock {
    private float[][][] cachedOutput; // 缓存特征图
    
    public float[][][] forward(float[][][] x) {
        if (cachedOutput == null) {
            cachedOutput = super.forward(x);
        }
        return cachedOutput;
    }
}
```

2. **加速技巧**：Winograd卷积优化
```java
public void enableWinogradConv() {
    for (ConvBlock block : convBlocks) {
        for (ConvLayer conv : block.getConvLayers()) {
            conv.setAlgorithm(ConvAlgorithm.WINOGRAD);
        }
    }
}
```

## 常见问题
### Q1：如何减少模型参数？
- 解决方案：使用1x1卷积降维
```java
public void addBottleneck() {
    // 在原始3x3卷积前添加1x1卷积
    convLayers.add(0, new ConvLayer(1, 1, 0, inChannels, reducedChannels));
}
```

### Q2：输入尺寸不是224x224怎么办？
- 动态调整实现：
```java
public void adaptInputSize(int newWidth, int newHeight) {
    // 计算最终特征图尺寸
    int finalSize = newWidth / 32; // 经过5次2x池化
    adjustFCLayer(finalSize * finalSize);
}
```

## 扩展阅读
- [VGG19深度扩展](/doc/extension/vgg19)
- [VGG在风格迁移中的应用](/doc/extension/style-transfer)
- [轻量化VGG实现](/doc/extension/vgg-lite)