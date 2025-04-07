# VAE 变分自编码器

## 概述
变分自编码器（Variational Autoencoder）通过概率编码实现数据生成，核心特点包括：
- 概率编码器学习潜在空间分布
- 重参数化技巧实现可导采样
- 证据下界（ELBO）优化目标

## 核心结构
```java
/**
 * VAE基础实现
 * 包含编码器、解码器和KL散度计算
 */
public class VAE {
    private Encoder encoder;     // 编码器网络
    private Decoder decoder;     // 解码器网络
    private int latentDim;       // 潜在空间维度
    
    // 重参数化参数
    private float epsilon = 1e-6;
}
```

## 完整实现

### 1. 编码器实现
```java
public class Encoder {
    private Linear muLayer;     // 均值层
    private Linear logvarLayer; // 对数方差层
    
    public GaussianSample encode(float[][] x) {
        float[][] mu = muLayer.forward(x);
        float[][] logvar = logvarLayer.forward(x);
        return new GaussianSample(mu, logvar);
    }
}

// 重参数化采样
public float[][] reparameterize(float[][] mu, float[][] logvar) {
    float[][] std = exp(0.5f * logvar);
    float[][] eps = randnLike(std);
    return add(mu, multiply(std, eps));
}
```

### 2. 解码器实现
```java
public class Decoder {
    private List<DenseLayer> layers;
    
    public float[][] decode(float[][] z) {
        float[][] x = z;
        for (DenseLayer layer : layers) {
            x = layer.forward(x);
            x = relu(x);
        }
        return sigmoid(x); // 输出概率
    }
}
```

## 使用示例（图像生成）
```java
public class ImageGenerator {
    public static void main(String[] args) {
        // 创建VAE模型（输入784维，潜在空间32维）
        VAE model = new VAE(784, 32);
        
        // 配置训练参数
        model.setBeta(0.5f); // KL散度权重
        model.setLearningRate(0.001f);
        
        // 加载MNIST数据集
        MNISTDataset dataset = new MNISTDataset("train-images.idx3-ubyte");
        
        // 训练循环
        for (int epoch=0; epoch<100; epoch++) {
            float totalLoss = 0;
            for (float[][] batch : dataset.getBatches(128)) {
                // 前向传播
                GaussianSample q = model.encode(batch);
                float[][] z = model.reparameterize(q.mu, q.logvar);
                float[][] recon = model.decode(z);
                
                // 计算损失
                float reconLoss = binaryCrossEntropy(batch, recon);
                float klLoss = klDivergence(q.mu, q.logvar);
                float loss = reconLoss + model.beta * klLoss;
                
                // 反向传播
                model.backward(loss);
                model.update();
                
                totalLoss += loss;
            }
            System.out.printf("Epoch %02d Loss: %.3f\n", epoch+1, totalLoss/dataset.size());
        }
        
        // 生成新样本
        float[][] z = randn(16, 32); // 16个潜在向量
        float[][] generated = model.decode(z);
        saveImages(generated, "samples.png");
    }
}
```

## 性能优化
1. **正则化技术**：
```java
public void applyRegularization() {
    this.encoder.addRegularizer(new L2Regularizer(0.001f));
    this.decoder.addRegularizer(new L2Regularizer(0.001f));
}
```

2. **并行采样**：
```java
public float[][] batchGenerate(int numSamples) {
    return IntStream.range(0, numSamples)
        .parallel()
        .mapToObj(i -> decoder(randn(latentDim)))
        .toArray(float[][][]::new);
}
```

## 常见问题
### Q1：生成图像模糊如何改善？
- 解决方案：使用更深层的解码器
```java
public void buildDeepDecoder() {
    this.decoder.addLayer(512, Activation.LEAKY_RELU)
                .addLayer(256, Activation.LEAKY_RELU)
                .addLayer(128, Activation.LEAKY_RELU);
}
```

### Q2：KL散度趋近于0怎么办？
- 退火策略实现：
```java
public void klAnnealing(int epoch, int total) {
    this.beta = Math.min(1.0f, epoch / (float)total);
}
```

## 扩展阅读
- [条件VAE实现](/doc/extension/cvae)
- [变分扩散模型原理](/doc/extension/vdm)
- [半监督VAE应用](/doc/extension/semi-supervised)