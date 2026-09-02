# Tensor 数据结构

`Tensor` 是 Omega-AI 中最核心的数据容器。网络输入、层输出、权重、梯度、损失中间值都会以 Tensor 的形式流转。

[[toc]]

---

## 一、基本形状

Omega-AI 中常见 Tensor 形状由四个维度组成：

```text
number, channel, height, width
```

常见含义：

| 维度 | 说明 |
| --- | --- |
| number | batch 大小或样本数量 |
| channel | 通道数、特征组数或 token 分组 |
| height | 图像高度、序列长度或中间维度 |
| width | 图像宽度、隐藏维度或特征维度 |

对于全连接层，常见输入会被看作 `[batch, 1, 1, feature]`。对于图像模型，常见输入是 `[batch, channel, height, width]`。

## 二、CPU 与 GPU 数据

Tensor 通常同时涉及 CPU 数组和 GPU 显存指针：

- CPU 数据适合调试、数据加载、日志统计。
- GPU 数据适合训练和推理计算。
- 频繁从 GPU 同步到 CPU 会打断异步执行，影响训练速度。

因此训练循环里不要高频调用 `syncHost()` 做日志统计，建议降低打印频率，或使用 GPU reduction 后只同步少量统计值。

## 三、创建 Tensor

以下构造方式来自 `com.omega.engine.tensor.Tensor` 源码：

```java
Tensor x = new Tensor(number, channel, height, width);
Tensor input = new Tensor(batchSize, 1, 1, featureSize, true);
Tensor label = new Tensor(batchSize, 1, 1, classSize, labelData, true);
```

其中带 `boolean hasGPU` 的构造函数会按源码逻辑决定是否准备 GPU 数据。带 `float[] data` 的构造函数会使用传入数组作为 Tensor 数据来源。

## 四、resize 的注意事项

训练中如果频繁改变 Tensor 形状，可能触发显存重新分配。对于固定 shape 的训练任务，建议：

- 在初始化阶段分配好常用 Tensor。
- 尽量复用中间 Tensor。
- 避免在每个 step 中创建大量临时 Tensor。
- 需要变 batch 时，优先确认是否会触发 GPU memory free/alloc。

## 五、调试建议

调试 Tensor 时优先看：

```text
number/channel/height/width
dataLength
hasGPU
```

如果训练结果异常，再检查：

- 输入 shape 是否和网络第一层匹配。
- label shape 是否和 loss 匹配。
- forward 输出是否被后续层覆盖。
- backward 梯度 Tensor 是否复用了错误的显存区域。
