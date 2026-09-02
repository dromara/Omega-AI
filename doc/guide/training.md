# 训练流程

Omega-AI 的训练流程主要由 `MBSGDOptimizer`、`Network`、`LossFunction` 和 `Updater` 协同完成。本篇解释一次训练从数据进入到参数更新的大致过程。

[[toc]]

---

## 一、训练主线

```text
读取 batch
  -> 拷贝到 Tensor
  -> forward
  -> loss
  -> backward
  -> update
  -> 日志与评估
```

在 GPU 训练中，forward、backward、update 通常会提交 CUDA kernel 或 cuBLAS/cuDNN 调用。CPU 侧只负责调度和必要的数据准备。

## 二、MBSGDOptimizer

`MBSGDOptimizer` 负责训练循环，包括：

- epoch 和 iteration 控制。
- batch 数据读取。
- 调用网络 forward/backward。
- 调用优化器更新参数。
- 按配置调整学习率。
- 打印 loss、耗时和验证信息。

常见构造参数包括网络、epoch、学习率、batch size、学习率更新策略等。不同版本构造函数可能略有不同，请以源码为准。

## 三、学习率

学习率策略会影响收敛速度和稳定性。常见策略：

| 策略 | 适合场景 |
| --- | --- |
| `NONE` | 固定学习率或由外部逻辑控制 |
| `CONSTANT` | 简单稳定的固定学习率训练 |
| `GD_DECAY` | 训练后期逐渐降低学习率 |
| `POLY` | 分阶段平滑衰减 |
| `EXP` | 指数衰减 |

大模型训练时建议保存每次实验的学习率、batch size、数据量、loss 曲线和采样结果，避免只看单个 loss 数值。

## 四、日志统计

训练中常见日志包括：

- 当前 epoch / step。
- 当前 learning rate。
- loss 均值。
- forward/backward/update 耗时。
- 数据加载耗时。
- 显存占用。

GPU 训练不要在每个 step 中同步大量 Tensor 到 CPU。需要统计 loss 时，推荐只同步已经 reduce 后的标量。

## 五、稳定性建议

- 先用小 batch 验证 loss 是否下降。
- 再逐步增大 batch 和模型规模。
- 出现 loss 突增时，检查学习率、梯度裁剪、混合精度、数据异常和 attention backward。
- 每次改 CUDA kernel 后，先用小 shape 做 forward/backward 数值对比。
- 长训练建议开启定期保存权重，避免中途异常丢失进度。
