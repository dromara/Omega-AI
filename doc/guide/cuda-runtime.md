# GPU运行时与动态库

Omega-AI 的 GPU 加速依赖 NVIDIA Driver、CUDA Toolkit、cuDNN、JCuda 以及项目中的自定义 CUDA kernel。环境问题通常不是代码 bug，而是版本或动态库加载路径不一致。

[[toc]]

---

## 一、运行时组成

| 组件 | 作用 |
| --- | --- |
| NVIDIA Driver | 驱动 GPU，决定可支持的最高 CUDA 运行能力 |
| CUDA Toolkit | 提供 nvcc、cuBLAS、运行时库 |
| cuDNN | 提供深度学习高性能算子 |
| JCuda | Java 调用 CUDA/cuBLAS/cuDNN 的桥接层 |
| 自定义 CUDA kernel | Omega-AI 中部分算子的 CUDA 实现 |

## 二、动态库查找位置

Windows 常见位置：

```text
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.7\bin
```

Linux 常见位置：

```text
/usr/local/cuda-11.7/lib64
```

如果项目生成了自定义 `.dll` 或 `.so`，需要确保 JVM 能找到它。常见方式：

```bash
java -Djava.library.path=/path/to/native/libs -cp target/xxx.jar com.omega.example.xxx.TrainDemo
```

Linux 也可以通过：

```bash
export LD_LIBRARY_PATH=/path/to/native/libs:/usr/local/cuda-11.7/lib64:$LD_LIBRARY_PATH
```

## 三、PTX 与 cu 文件

项目中的 CUDA kernel 通常放在 `src/main/resources/cu` 或构建后的 `target/classes/cu` 中。运行时会加载对应的 PTX 或编译产物。

如果修改了 `.cu` 文件，需要确认：

- PTX 是否重新生成。
- jar 中是否包含最新资源。
- 运行目录是否加载了旧版本 target/classes。
- GPU 架构是否匹配当前显卡。

## 四、常见错误

| 错误 | 常见原因 |
| --- | --- |
| `UnsatisfiedLinkError` | JVM 找不到 `.dll`、`.so` 或 JCuda native lib |
| `no kernel image is available` | PTX 或 cubin 不支持当前 GPU 架构 |
| `CUDNN_STATUS_BAD_PARAM` | Tensor shape、stride、dtype 或 descriptor 不匹配 |
| `CUDNN_STATUS_NOT_SUPPORTED` | 当前 cuDNN 版本不支持该算子配置 |
| `out of memory` | batch、模型、中间状态或 workspace 超过显存 |

## 五、性能分析建议

训练变慢时不要只看总耗时，建议拆分：

- 数据加载时间。
- forward kernel 时间。
- backward kernel 时间。
- optimizer update 时间。
- CPU/GPU 同步点。
- cudaMalloc/cudaFree 次数。

Linux 可使用 Nsight Systems：

```bash
nsys profile -t cuda,cublas,cudnn,nvtx,osrt -o report java -cp target/xxx.jar com.omega.example.xxx.TrainDemo
```

Windows 需要先安装 Nsight Systems，并使用 `nsys.exe` 的完整路径或把安装目录加入 `PATH`。
