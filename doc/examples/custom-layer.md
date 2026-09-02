# 自定义 Layer

当已有 Layer 不能满足需求时，可以新增自己的 Layer。建议先用 Java 或已有 TensorOP 组合实现正确性，再考虑 CUDA kernel 优化。

[[toc]]

---

## 一、适合自定义 Layer 的场景

- 新增激活函数。
- 新增归一化方式。
- 新增注意力、门控、残差结构。
- 融合多个简单算子，减少 kernel launch。
- 实现论文中的特殊模块。

## 二、实现步骤

1. 明确输入输出 shape。
2. 定义需要保存的中间 Tensor。
3. 实现 `output()` 前向逻辑。
4. 实现 `diff()` 反向逻辑。
5. 如果有参数，准备参数 Tensor 和梯度 Tensor。
6. 写一个小 shape 对比测试，验证 forward/backward。

## 三、必须实现的方法

`Layer` 是抽象类，新增 Layer 时必须严格实现源码中定义的抽象方法。当前 `com.omega.engine.nn.layer.Layer` 包含以下抽象方法：

```java
public abstract void init();
public abstract void initBack();
public abstract void initParam();
public abstract void output();
public abstract Tensor getOutput();
public abstract void diff();
public abstract void forward();
public abstract void back();
public abstract void backTemp();
public abstract void forward(Tensor input);
public abstract void back(Tensor delta);
public abstract void update();
public abstract void accGrad(float scale);
public abstract void showDiff();
public abstract LayerType getLayerType();
public abstract float[][][][] output(float[][][][] input);
public abstract void initCache();
```

具体实现不要从空类开始猜，建议直接参考已有源码：

- `com.omega.engine.nn.layer.active.ReluLayer`
- `com.omega.engine.nn.layer.FullyLayer`
- `com.omega.engine.nn.layer.normalization.BNLayer`

新增 Layer 时应保持字段含义、forward/backward 调用方式、Tensor 分配方式和 updater 接入方式与现有实现一致。

## 四、验证建议

自定义 Layer 不建议直接接入大模型测试，推荐：

- 用小 Tensor 手工构造输入。
- 打印 forward 输出。
- 对比 CPU 版本和 GPU 版本。
- 对 backward 做数值梯度检查。
- 确认 batch size 变化时不会 shape 错乱。

## 五、性能优化顺序

先保证正确性，再优化：

1. 减少临时 Tensor 创建。
2. 复用中间显存。
3. 合并简单逐元素算子。
4. 避免每步 CPU 同步。
5. 对热点算子写 CUDA kernel。
