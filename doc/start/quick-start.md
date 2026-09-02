# 快速开始

本篇带你用最短路径跑通 Omega-AI。第一次使用时建议先运行基础示例，确认 JDK、Maven、CUDA、cuDNN 与显卡驱动没有版本冲突，再尝试大模型训练。

[[toc]]

---

## 一、准备环境

| 环境 | 建议版本 | 说明 |
| --- | --- | --- |
| JDK | 8 或以上 | 项目主体为 Java 实现 |
| Maven | 3.6 或以上 | 用于编译和打包 |
| CUDA Toolkit | 11.7 | 当前文档以 11.7 为主线 |
| cuDNN | 与 CUDA 匹配 | GPU 卷积、归一化、注意力等算子会用到 |
| NVIDIA Driver | 支持目标 CUDA 版本 | 以 `nvidia-smi` 显示为准 |

如果只想先阅读代码或运行不依赖 GPU 的示例，可以先跳过 CUDA/cuDNN；如果需要 GPU 训练，请先完成 [CUDA安装](/doc/start/cuda.md)。

## 二、获取源码

```bash
git clone https://gitee.com/dromara/omega-ai.git
cd omega-ai
```

也可以从 GitHub 获取：

```bash
git clone https://github.com/dromara/Omega-AI.git
cd Omega-AI
```

## 三、编译项目

```bash
mvn -DskipTests package
```

编译完成后，通常会在 `target` 目录下生成 jar 包。不同分支的 jar 名称可能略有不同，请以实际输出为准。

## 四、运行一个基础示例

Omega-AI 的示例主要放在 `src/main/java/com/omega/example` 目录下。第一次建议从 BP 或 CNN 示例开始，例如：

```bash
java -cp target/omega-engine-v4-gpu-win-cu11.7-v1.0-beta-jar-with-dependencies.jar com.omega.example.bp.test.BPTest
```

如果你在 IDE 中运行，可以直接打开对应的 `main` 方法执行。数据集路径、权重路径、输出路径需要按本机目录调整。

## 五、运行 GPU 示例前检查

GPU 示例运行前建议先确认：

```bash
nvidia-smi
nvcc --version
```

Windows 还需要确认 CUDA 的 `bin` 目录在 `PATH` 中；Linux 需要确认 `LD_LIBRARY_PATH` 包含 CUDA 与 cuDNN 的动态库目录。

## 六、推荐阅读顺序

1. [项目结构](/doc/start/project-structure.md)
2. [Tensor 数据结构](/doc/guide/tensor.md)
3. [Layer 与 Network](/doc/guide/layer-network.md)
4. [训练流程](/doc/guide/training.md)
5. [GPU运行时与动态库](/doc/guide/cuda-runtime.md)
6. [示例总览](/doc/examples/index.md)

## 七、常见卡点

如果遇到 `UnsatisfiedLinkError`、`no kernel image is available`、`CUDNN_STATUS`、显存不足、找不到数据集等问题，请先看 [常见问题排查](/doc/more/common-questions.md)。
