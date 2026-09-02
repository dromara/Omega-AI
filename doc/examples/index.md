# 示例总览

Omega-AI 的示例集中在 `src/main/java/com/omega/example`。建议按难度从基础分类任务开始，再进入 Transformer、VAE、Diffusion、DiT 等复杂模型。

[[toc]]

---

## 一、推荐学习路径

| 阶段 | 示例 | 学习目标 |
| --- | --- | --- |
| 入门 | BP / MNIST | Tensor、FullyLayer、Loss、Optimizer |
| 图像分类 | CNN / AlexNet / VGG / ResNet | 卷积、池化、批归一化 |
| 目标检测 | YOLO | 检测标签、输出解码、检测 loss |
| 序列模型 | RNN / LSTM / Seq2Seq | 序列输入和循环结构 |
| 语言模型 | Transformer / GPT / Llama | attention、embedding、decoder block |
| 生成模型 | VAE / GAN / Diffusion | 图像生成和重建 |
| 文生图 | DiT / Stable Diffusion | 条件生成、时间步、VAE latent |

## 二、运行示例前要改什么

多数示例需要按本机环境修改：

- 数据集目录。
- 权重保存目录。
- 图片输出目录。
- batch size。
- 是否启用 GPU。
- CUDA/cuDNN 版本。

## 三、示例阅读建议

先看 `main` 方法中做了什么，再追踪：

1. 数据集如何加载。
2. 网络如何创建。
3. loss 如何选择。
4. optimizer 如何设置。
5. 训练循环如何进入。
6. 测试或采样如何调用。

## 四、下一步

可以先跑 [MNIST 入门训练](/doc/examples/mnist.md)，再尝试 [自定义 Layer](/doc/examples/custom-layer.md)、[自定义 CUDA 算子](/doc/examples/custom-cuda-kernel.md)，或继续阅读 [DiT Diffusion Transformer](/doc/api/dit.md)。
