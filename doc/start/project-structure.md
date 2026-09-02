# 项目结构

Omega-AI 的代码组织围绕“张量、网络层、网络、损失函数、优化器、GPU 算子、示例”展开。理解这些目录后，再阅读示例代码会轻松很多。

[[toc]]

---

## 一、核心目录

| 目录 | 作用 |
| --- | --- |
| `src/main/java/com/omega/engine/tensor` | Tensor 数据结构与张量辅助类 |
| `src/main/java/com/omega/engine/nn/layer` | 全连接、卷积、归一化、注意力等网络层 |
| `src/main/java/com/omega/engine/nn/network` | BP、CNN、Transformer、GPT、Llama、DiT、VAE 等网络结构 |
| `src/main/java/com/omega/engine/loss` | MSE、交叉熵等损失函数 |
| `src/main/java/com/omega/engine/optimizer` | 训练器与优化过程 |
| `src/main/java/com/omega/engine/updater` | SGD、Adam、AdamW 等参数更新器 |
| `src/main/java/com/omega/engine/gpu` | JCuda、cuDNN、CUDA kernel 管理 |
| `src/main/resources/cu` | CUDA kernel 源码与 PTX 资源 |
| `src/main/java/com/omega/example` | 各类模型训练、推理、数据处理示例 |

## 二、一次训练会经过哪些模块

1. 示例代码准备数据集和网络。
2. `Network` 管理各个 `Layer` 的 forward 和 backward。
3. `LossFunction` 计算损失和输出梯度。
4. `MBSGDOptimizer` 控制 epoch、batch、学习率和验证逻辑。
5. `Updater` 根据梯度更新参数。
6. 如果启用 GPU，矩阵乘、卷积、归一化、注意力等会进入 JCuda/cuDNN/自定义 CUDA kernel。

## 三、示例目录怎么读

`com.omega.example` 按模型类型分组，建议先看：

| 示例目录 | 适合学习 |
| --- | --- |
| `bp` | 最小网络、全连接层、基础训练流程 |
| `cnn` | 卷积、池化、图像分类 |
| `yolo` | 目标检测数据格式和网络输出 |
| `transformer` | 序列模型、注意力机制 |
| `vae` | 图像编码和生成 |
| `dit` | Diffusion Transformer 与文生图训练 |

## 四、阅读源码建议

如果你是第一次看深度学习框架源码，可以按这个顺序：

1. `Tensor`
2. `Layer`
3. `FullyLayer`
4. `Network`
5. `MBSGDOptimizer`
6. `LossFunction`
7. `Updater`
8. GPU kernel 相关类

这样能先建立主线，再进入具体模型。
