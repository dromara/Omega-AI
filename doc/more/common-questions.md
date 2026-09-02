# 常见问题排查

本篇整理 Omega-AI 使用过程中常见的环境、训练、GPU 和数据问题。遇到报错时建议先看完整堆栈，再按本页顺序排查。

[[toc]]

---

## 一、找不到动态库

### Q：出现 `UnsatisfiedLinkError` 怎么办？

通常是 JVM 找不到 CUDA、cuDNN、JCuda 或项目自定义动态库。

优先检查：

```bash
nvidia-smi
nvcc --version
```

Windows 检查：

```text
CUDA bin 目录是否加入 PATH
项目自定义 dll 是否在 java.library.path 中
JCuda native 依赖版本是否匹配
```

Linux 检查：

```bash
echo $LD_LIBRARY_PATH
ldconfig -p | grep cudnn
```

也可以启动时显式指定：

```bash
java -Djava.library.path=/path/to/native/libs -cp target/xxx.jar com.omega.example.xxx.TrainDemo
```

## 二、CUDA 版本不匹配

### Q：`no kernel image is available for execution on the device` 是什么原因？

常见原因是 PTX、cubin 或编译架构不支持当前显卡。解决方向：

- 重新编译 `.cu` 文件。
- 编译时加入当前显卡支持的 `sm_xx`。
- 确认运行时没有加载旧 PTX。
- 确认 jar 中资源已经更新。

## 三、cuDNN 报错

### Q：`CUDNN_STATUS_BAD_PARAM` 怎么查？

优先检查：

- Tensor shape 是否正确。
- NCHW/NHWC 是否一致。
- batch、channel、height、width 是否写反。
- dtype 是否和 descriptor 匹配。
- workspace 是否分配成功。

### Q：`No execution plans support the graph` 是什么原因？

这通常表示当前 cuDNN 版本、shape、dtype、mask、dropout、deterministic 等组合没有可用执行计划。可以尝试：

- 关闭 deterministic。
- 换一个 cuDNN 版本。
- 调整 headDim、seqLen、batch、mask 配置。
- 先用小 demo 对比 forward/backward 是否支持。

## 四、显存不足

### Q：训练出现 out of memory 怎么办？

优先降低：

- batch size。
- 输入分辨率。
- 序列长度。
- 模型深度或隐藏维度。
- 保存的中间状态数量。

同时检查是否存在：

- 每步创建新 Tensor。
- resize 触发重新分配。
- backward scratch 没有复用。
- 日志统计同步了完整 Tensor。

## 五、loss 不下降或突然变大

常见原因：

- 学习率过大。
- label 或 target 构造错误。
- 输入没有归一化。
- loss 和输出层不匹配。
- backward 梯度 shape 错位。
- 混合精度溢出或精度不足。
- 自定义 CUDA kernel backward 有误差。

排查建议：

1. 固定随机种子。
2. 用小 batch 过拟合少量数据。
3. 关闭混合精度或 flash attention 做对照。
4. 打印关键 Tensor shape。
5. 对新 kernel 做 forward/backward 数值对比。

## 六、训练速度不理想

先用 nsys 或日志拆分：

- 数据加载。
- forward。
- backward。
- optimizer update。
- CPU/GPU 同步。
- cudaMalloc/cudaFree。

常见优化方向：

- 减少 `syncHost()` 调用。
- 复用 Tensor 和 workspace。
- 融合多个小的逐元素 kernel。
- 使用 cuBLAS/cuDNN 处理矩阵乘和卷积。
- 增大 batch 让 GPU 吃满，但不要超过显存。

## 七、数据集路径错误

示例中经常包含作者本机路径，例如 `D:\test\...`。运行前需要改成自己的数据目录。

Windows 推荐：

```java
String dataPath = "D:/dataset/mnist/";
```

Linux 推荐：

```java
String dataPath = "/data/dataset/mnist/";
```

## 八、如何提 issue

提交 issue 时建议包含：

- 操作系统。
- JDK、Maven、CUDA、cuDNN、显卡型号。
- 使用的分支和 commit。
- 完整启动命令。
- 完整报错堆栈。
- 复现用的最小代码或示例入口。

交流入口：[加入讨论群](/doc/more/join-group.md)
