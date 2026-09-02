# Layer 与 Network

Omega-AI 的模型由 `Network` 管理整体训练流程，由多个 `Layer` 串联或组合完成前向计算和反向传播。

[[toc]]

---

## 一、Layer 的职责

一个 Layer 通常负责：

- 保存输入 Tensor。
- 计算输出 Tensor。
- 保存可训练参数，例如 weight、bias、gamma、beta。
- 根据上游梯度计算本层梯度。
- 将参数梯度交给更新器处理。

常见方法含义：

| 方法 | 作用 |
| --- | --- |
| `output()` | 前向计算 |
| `diff()` | 反向传播 |
| `forward()` | 包装输入设置和前向过程 |
| `back()` | 包装梯度设置和反向过程 |
| `update()` | 参数更新 |

不同 Layer 的方法细节会有差异，请结合具体源码阅读。

## 二、Network 的职责

`Network` 负责把多个 Layer 组织成完整模型：

- 保存网络层列表。
- 控制 forward 顺序。
- 控制 backward 顺序。
- 保存 loss、updater、学习率等训练状态。
- 提供参数更新、清理、保存加载等能力。

## 三、典型执行流程

一次训练 step 通常是：

```text
input -> network.forward -> loss -> loss.diff -> network.back -> network.update
```

如果是复杂模型，例如 Transformer、VAE、DiT，网络内部可能包含多条分支、残差连接、条件输入、时间嵌入或注意力模块，但主线仍然是 forward、loss、backward、update。

## 四、如何新增 Layer

新增 Layer 通常需要：

1. 定义输入输出 shape。
2. 在 `init()` 中分配参数和中间 Tensor。
3. 在 `output()` 中实现前向计算。
4. 在 `diff()` 中实现反向传播。
5. 如果有参数，准备梯度 Tensor。
6. 接入 updater 或复用已有参数更新逻辑。

如果 Layer 使用 CUDA kernel，还需要准备 Java wrapper、CUDA 源码、PTX 编译或运行时加载逻辑。

## 五、常见错误

- 前向输出 shape 与下一层输入不匹配。
- 反向梯度 shape 与本层输入不匹配。
- 残差连接复用了同一个 Tensor，导致值被覆盖。
- 参数梯度没有清零或被错误累加。
- 归一化层的 gamma/beta 没有被优化器收集。
