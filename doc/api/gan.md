# GAN 对抗神经网络

## 概述
生成对抗网络（Generative Adversarial Network）通过生成器与判别器的对抗训练，学习数据分布并生成逼真样本。包含以下核心组件：

- **生成器(Generator)**：将随机噪声转换为数据样本
- **判别器(Discriminator)**：区分真实样本与生成样本
- **对抗损失函数**：驱动两者博弈式训练

## 核心结构
```java
/**
 * GAN基础实现
 */
public class GAN {
    // 网络组件
    private Generator generator;
    private Discriminator discriminator;
    
    // 优化参数
    private Optimizer gOptimizer;
    private Optimizer dOptimizer;
    
    // 噪声维度
    private int noiseDim;
    
    public GAN(int noiseDim, int dataDim) {
        this.noiseDim = noiseDim;
        initComponents(dataDim);
    }
}
```

## 完整实现

### 1. 生成器实现
```java
public class Generator {
    private List<DenseLayer> layers;
    
    public float[] generate(float[] noise) {
        float[] output = noise;
        for (DenseLayer layer : layers) {
            output = layer.forward(output);
            output = applyActivation(output); // 使用LeakyReLU激活
        }
        return output; // 生成与真实数据同维度的样本
    }
    
    // 上采样块（用于图像生成）
    private float[] upsamplingBlock(float[] input) {
        // 包含转置卷积、批归一化、激活函数
    }
}
```

### 2. 判别器实现
```java
public class Discriminator {
    private List<DenseLayer> layers;
    
    public float discriminate(float[] input) {
        float[] output = input;
        for (DenseLayer layer : layers) {
            output = layer.forward(output);
            output = applyActivation(output); // 使用LeakyReLU激活
        }
        return sigmoid(output[0]); // 返回真实概率
    }
    
    // 下采样块（用于图像判别）
    private float[] downsamplingBlock(float[] input) {
        // 包含卷积层、批归一化、激活函数
    }
}
```

## 对抗训练
```java
public class AdversarialTrainer {
    public void train(int epochs) {
        for (int epoch=0; epoch<epochs; epoch++) {
            // 1. 训练判别器
            discriminatorTrainStep(realData);
            
            // 2. 训练生成器
            generatorTrainStep();
            
            // 3. 输出训练状态
            if (epoch % 100 == 0) {
                printTrainingStatus(epoch);
                generateSamples(epoch); // 生成示例样本
            }
        }
    }
    
    private void discriminatorTrainStep(float[][] realSamples) {
        // 生成假样本
        float[][] fakeSamples = generateFakeSamples(realSamples.length);
        
        // 计算判别损失
        float dLossReal = discriminator.loss(realSamples, 1.0f);
        float dLossFake = discriminator.loss(fakeSamples, 0.0f);
        float dLoss = (dLossReal + dLossFake) / 2;
        
        // 反向传播更新判别器
        discriminator.backward();
        dOptimizer.update();
    }
    
    private void generatorTrainStep() {
        // 计算生成损失
        float[] noise = generateNoise(batchSize);
        float gLoss = generator.loss(noise);
        
        // 冻结判别器后反向传播
        freezeDiscriminator();
        generator.backward();
        gOptimizer.update();
        unfreezeDiscriminator();
    }
}
```

## 使用示例（手写数字生成）
```java
public class MNISTGAN {
    public static void main(String[] args) {
        // 创建GAN (100维噪声输入，784维图像输出)
        GAN gan = new GAN(100, 28*28);
        
        // 配置网络结构
        gan.getGenerator()
           .addLayer(256, Activation.LEAKY_RELU)
           .addLayer(512, Activation.LEAKY_RELU)
           .addLayer(784, Activation.TANH); // 输出归一化到[-1,1]
        
        gan.getDiscriminator()
           .addLayer(512, Activation.LEAKY_RELU)
           .addLayer(256, Activation.LEAKY_RELU)
           .addLayer(1, Activation.SIGMOID);
           
        // 配置优化器
        gan.setGOptimizer(new Adam(0.0002f));
        gan.setDOptimizer(new Adam(0.0002f));
        
        // 加载MNIST数据集
        MNISTDataset dataset = new MNISTDataset("train-images.idx3-ubyte");
        
        // 开始训练
        gan.train(dataset, 10000, 64);
        
        // 生成示例图像
        gan.generateImages("samples/epoch_final.png");
    }
}
```

## 数学原理
### 损失函数
生成器损失：
$$\mathcal{L}_G = -\mathbb{E}[\log D(G(z))]$$

判别器损失：
$$\mathcal{L}_D = -\mathbb{E}[\log D(x)] - \mathbb{E}[\log (1 - D(G(z)))]$$

## 常见问题
### Q1：出现模式崩溃怎么办？
- 解决方案：
```java
// 使用Wasserstein GAN
public class WGAN extends GAN {
    // 移除判别器最后的Sigmoid
    // 添加梯度惩罚项
    public float gradientPenalty() {
        // 实现梯度惩罚逻辑
    }
}
```

### Q2：生成样本模糊如何改善？
- 改进方案：
1. 添加感知损失
2. 使用更深的网络结构
3. 采用渐进式训练

## 扩展阅读
- [DCGAN深度卷积GAN](/doc/extension/dcgan)
- [CycleGAN跨域转换](/doc/extension/cyclegan)
- [StyleGAN风格控制](/doc/extension/stylegan)