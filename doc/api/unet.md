# UNet U形网络

## 概述
UNet是一种用于图像分割的对称编码器-解码器架构，通过跳跃连接融合浅层细节与深层语义信息，在医学图像分割等领域表现优异。

## 核心结构
```java
/**
 * UNet基础实现
 * 结构：编码器 → 桥接层 → 解码器 + 跳跃连接
 */
public class UNet {
    private List<EncoderBlock> encoder;  // 编码路径
    private BridgeBlock bridge;        // 中间桥接层
    private List<DecoderBlock> decoder; // 解码路径
    
    // 网络参数
    private int inputChannels;
    private int[] filters = {64, 128, 256, 512}; // 各阶段卷积核数量
}
```

## 完整实现

### 1. 编码器模块
```java
public class EncoderBlock {
    private ConvLayer conv1;
    private ConvLayer conv2;
    private MaxPoolingLayer pool;
    
    public float[][][] forward(float[][][] x) {
        // 两次卷积 + ReLU
        x = conv1.forward(x);
        x = ReLU(x);
        x = conv2.forward(x);
        x = ReLU(x);
        
        // 保存跳跃连接值
        float[][][] skip = x.clone();
        
        // 下采样
        return pool.forward(x);
    }
}
```

### 2. 解码器模块
```java
public class DecoderBlock {
    private ConvTransposeLayer upconv;
    private ConvLayer conv1;
    private ConvLayer conv2;
    
    public float[][][] forward(float[][][] x, float[][][] skip) {
        // 上采样
        x = upconv.forward(x);
        
        // 拼接跳跃连接
        x = concatenate(x, skip);
        
        // 两次卷积
        x = conv1.forward(x);
        x = ReLU(x);
        x = conv2.forward(x);
        return ReLU(x);
    }
}
```

## 使用示例（医学图像分割）
```java
public class MedicalSegmenter {
    public static void main(String[] args) {
        // 创建UNet (1通道输入，2分类输出)
        UNet model = new UNet()
            .setInputChannels(1)
            .setOutputChannels(2);
        
        // 配置深度监督
        model.enableDeepSupervision();
        
        // 加载医学数据集
        MedicalDataset dataset = new MedicalDataset("chest_xrays/");
        
        // 训练配置
        model.setLoss(new DiceLoss())  // 使用Dice损失
             .setOptimizer(new Adam(0.0001f));
        
        // 训练循环
        for (int epoch=0; epoch<100; epoch++) {
            float totalLoss = 0;
            for (ImageMaskPair pair : dataset) {
                float[][][] pred = model.forward(pair.image);
                float loss = model.loss(pred, pair.mask);
                model.backward();
                totalLoss += loss;
            }
            System.out.printf("Epoch %02d Loss: %.3f\n", epoch+1, totalLoss/dataset.size());
        }
        
        // 分割预测
        float[][][] segmentation = model.predict("patient_001.png");
        saveMask(segmentation, "result_mask.png");
    }
}
```

## 性能优化
1. **内存优化**：特征图缓存
```java
public class CachedEncoder extends EncoderBlock {
    private float[][][] cachedFeature;
    
    public float[][][] forward(float[][][] x) {
        if (cachedFeature == null) {
            cachedFeature = super.forward(x);
        }
        return cachedFeature;
    }
}
```

2. **轻量化改进**：深度可分离卷积
```java
public void replaceWithDepthwiseConv() {
    this.conv1 = new DepthwiseConvLayer(3, 1, 1);
    this.conv2 = new DepthwiseConvLayer(3, 1, 1);
}
```

## 常见问题
### Q1：边缘信息丢失严重？
- 解决方案：镜像填充 + 数据增强
```java
public void preprocess(Image img) {
    // 使用镜像填充保持尺寸
    img.padMirror(16); 
    
    // 添加随机弹性变形
    applyElasticDeformation(img);
}
```

### Q2：显存不足如何解决？
- 梯度检查点技术：
```java
public void enableGradientCheckpoint() {
    this.checkpoint = true; // 在前向时存储中间结果
    this.recompute = true;  // 反向时重新计算部分结果
}
```

## 扩展阅读
- [ResUNet改进架构](/doc/extension/resunet)
- [3D UNet体数据分割](/doc/extension/3d-unet)
- [UNet++嵌套结构](/doc/extension/unet-plus)