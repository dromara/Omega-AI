# 自定义 CUDA 算子

Omega-AI 支持通过自定义 CUDA kernel 加速热点算子。适合优化逐元素运算、通道拆分合并、归一化、简单 reduction 或多个小算子的融合。

[[toc]]

---

## 一、什么时候需要写 CUDA kernel

建议满足以下条件再写：

- 该算子在 nsys 报告中占比较高。
- Java 侧由多个小 TensorOP 组成，kernel launch 很多。
- cuBLAS/cuDNN 没有直接对应算子。
- 算子逻辑稳定，不会频繁改接口。

矩阵乘、卷积、标准 attention 等成熟算子，优先使用 cuBLAS、cuDNN 或已有实现。

## 二、开发流程

1. 写 CPU 或简单 GPU 版本验证公式。
2. 在 `.cu` 中实现 forward kernel。
3. 如果参与训练，实现 backward kernel。
4. 在 Java wrapper 中加载 kernel。
5. 写 demo 对比输出误差。
6. 用 nsys 检查耗时和 launch 次数。

## 三、CUDA kernel 示例

```cuda
extern "C"
__global__ void add_scalar_kernel(float *x, float v, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        x[idx] += v;
    }
}
```

启动参数通常按元素数计算：

```text
threads = 256
blocks = (n + threads - 1) / threads
```

## 四、Java 侧注意事项

- kernel 名称要和 `.cu` 中导出的名称一致。
- 参数顺序必须和 CUDA kernel 完全一致。
- Tensor 的 GPU 指针不能传错。
- int/float/long 参数类型要对应。
- 修改 `.cu` 后确认 PTX 或资源文件已经更新。

## 五、常见性能坑

- 每个 step 创建和释放 GPU 内存。
- 多个小 kernel 可以融合却没有融合。
- block/thread 配置过小。
- 读写不连续导致访存效率低。
- backward 保存了过多中间状态，显存压力过大。

## 六、推荐验证方式

每个新 CUDA 算子至少准备两个测试：

1. 小 shape 正确性测试，方便人工检查。
2. 接近真实训练 shape 的性能测试，用于评估收益。

如果 backward 参与训练，还要对比 CPU/原始 GPU 实现的梯度误差。
