# BPNetwork bp神经网络

`BPNetwork` 是 Omega-AI 中用于搭建全连接前馈神经网络的基础网络实现。它继承自 `Network`，通过顺序添加 `Layer` 的方式组织输入层、全连接层、归一化层、激活层、Dropout 和损失层，训练流程由 `MBSGDOptimizer` 驱动。

本页代码均来自 Omega-AI 当前源码节选，类名、构造参数和调用方式以源码为准。

[[toc]]

---

## 一、源码入口

| 类型 | 位置 |
| --- | --- |
| 主要实现 | `com.omega.engine.nn.network.BPNetwork` |
| 源码路径 | `src/main/java/com/omega/engine/nn/network/BPNetwork.java` |
| 示例入口 | `src/main/java/com/omega/example/bp/test/BPTest.java` |
| 示例代码（Demo） | `BPTest.bpNetwork_mnist()`、`BPTest.bpNetwork_iris()` |

## 二、网络入口

节选自 `src/main/java/com/omega/engine/nn/network/BPNetwork.java`，`BPNetwork` 主要负责保存损失函数和更新器类型，具体网络结构由外部调用 `addLayer(...)` 逐层装配。

```java
public class BPNetwork extends Network {

    public BPNetwork(LossFunction lossFunction) {
        this.lossFunction = lossFunction;
    }

    public BPNetwork(LossFunction lossFunction, UpdaterType updater) {
        this.lossFunction = lossFunction;
        this.updater = updater;
    }

    public BPNetwork(LossType lossType, UpdaterType updater) {
        this.lossFunction = LossFactory.create(lossType, this);
        this.updater = updater;
    }
}
```

## 三、Demo 执行流程

`BPTest.bpNetwork_mnist()` 的训练流程可以拆成 5 步：

1. 准备 MNIST 图像和标签路径，并定义 `labelSet`。
2. 使用 `DataLoader.loadDataByUByte(...)` 读取训练集和测试集。
3. 创建 `BPNetwork`，设置 `CUDNN` 和 `learnRate`。
4. 按顺序添加 `InputLayer -> FullyLayer -> BNLayer -> ReluLayer -> DropoutLayer -> SoftmaxWithCrossEntropyLayer`。
5. 创建 `MBSGDOptimizer`，调用 `optimizer.train(trainData)` 和 `optimizer.test(testData)`。

## 四、示例代码（Demo）

节选自 `src/main/java/com/omega/example/bp/test/BPTest.java`，保留 MNIST 训练入口的完整关键代码：

```java
public static void bpNetwork_mnist() {
    // TODO Auto-generated method stub
    /**
     * 读取训练数据集

     */
    String mnist_train_data = "H:\\omega\\20240716\\omega-ai\\src\\main\\resources\\dataset\\mnist\\train-images.idx3-ubyte";
    String mnist_train_label = "H:\\omega\\20240716\\omega-ai\\src\\main\\resources\\dataset\\mnist\\train-labels.idx1-ubyte";
    String mnist_test_data = "H:\\omega\\20240716\\omega-ai\\src\\main\\resources\\dataset\\mnist\\t10k-images.idx3-ubyte";
    String mnist_test_label = "H:\\omega\\20240716\\omega-ai\\src\\main\\resources\\dataset\\mnist\\t10k-labels.idx1-ubyte";
    String[] labelSet = new String[]{"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"};
    DataSet trainData = DataLoader.loadDataByUByte(mnist_train_data, mnist_train_label, labelSet, 1, 1, 784, true);
    DataSet testData = DataLoader.loadDataByUByte(mnist_test_data, mnist_test_label, labelSet, 1, 1, 784, true);
    BPNetwork netWork = new BPNetwork(new SoftmaxWithCrossEntropyLoss(), UpdaterType.adamw);
    netWork.CUDNN = true;
    netWork.learnRate = 0.001f;
    int inputCount = (int) (Math.sqrt(794) + 10);
    InputLayer inputLayer = new InputLayer(1, 1, 784);
    FullyLayer hidden1 = new FullyLayer(784, inputCount, false);
    BNLayer bn1 = new BNLayer();
    ReluLayer active1 = new ReluLayer();
    FullyLayer hidden2 = new FullyLayer(inputCount, inputCount, false);
    BNLayer bn2 = new BNLayer();
    ReluLayer active2 = new ReluLayer();
    FullyLayer hidden3 = new FullyLayer(inputCount, inputCount, false);
    BNLayer bn3 = new BNLayer();
    ReluLayer active3 = new ReluLayer();
    FullyLayer hidden4 = new FullyLayer(inputCount, 10);
    DropoutLayer dropout = new DropoutLayer(0.2f);
    SoftmaxWithCrossEntropyLayer softmax = new SoftmaxWithCrossEntropyLayer(10);
    netWork.addLayer(inputLayer);
    netWork.addLayer(hidden1);
    netWork.addLayer(bn1);
    netWork.addLayer(active1);
    netWork.addLayer(hidden2);
    netWork.addLayer(bn2);
    netWork.addLayer(active2);
    netWork.addLayer(hidden3);
    netWork.addLayer(bn3);
    netWork.addLayer(active3);
    netWork.addLayer(hidden4);
    netWork.addLayer(dropout);
    netWork.addLayer(softmax);
    try {
        MBSGDOptimizer optimizer = new MBSGDOptimizer(netWork, 10, 0.001f, 128, LearnRateUpdate.NONE, false);
        //			netWork.GRADIENT_CHECK = true;
        long start = System.nanoTime();
        long trainTime = System.nanoTime();
        optimizer.train(trainData);
        System.out.println("trainTime:" + ((System.nanoTime() - trainTime) / 1e9) + "s.");
        long testTime = System.nanoTime();
        optimizer.test(testData);
        System.out.println("testTime:" + ((System.nanoTime() - testTime) / 1e9) + "s.");
        System.out.println(((System.nanoTime() - start) / 1e9) + "s.");
    } catch (Exception e) {
        // TODO Auto-generated catch block
        e.printStackTrace();
    } finally {
        try {
            CUDAMemoryManager.freeAll();
        } catch (Exception e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
    }
}
```

## 五、关键组件

| 组件 | 作用 |
| --- | --- |
| `DataLoader.loadDataByUByte(...)` | 读取 MNIST 的 ubyte 图像和标签文件，生成 `DataSet`。 |
| `BPNetwork` | 顺序容器，保存 loss、updater、学习率和 layer 列表。 |
| `InputLayer` | 声明输入数据 shape，本例为 `1 x 1 x 784`。 |
| `FullyLayer` | 全连接层，负责主要参数计算。 |
| `BNLayer` | Batch Normalization，用于稳定中间层分布。 |
| `ReluLayer` | 激活函数层。 |
| `DropoutLayer` | 随机失活，本例概率为 `0.2f`。 |
| `SoftmaxWithCrossEntropyLayer` | 分类输出层，本例输出 10 类。 |
| `MBSGDOptimizer` | 训练调度器，负责 mini-batch 训练、测试和学习率策略。 |

## 六、结构说明

本例的网络结构如下：

```text
Input(784)
  -> Fully(784, inputCount)
  -> BN
  -> ReLU
  -> Fully(inputCount, inputCount)
  -> BN
  -> ReLU
  -> Fully(inputCount, inputCount)
  -> BN
  -> ReLU
  -> Fully(inputCount, 10)
  -> Dropout(0.2)
  -> SoftmaxWithCrossEntropy(10)
```

其中：

- `inputCount = (int) (Math.sqrt(794) + 10)` 来自示例源码，用于设置隐藏层宽度。
- `new FullyLayer(..., false)` 表示该层按源码构造参数关闭 bias。
- `UpdaterType.adamw` 表示网络参数更新器使用 AdamW。
- `LearnRateUpdate.NONE` 表示 optimizer 不启用额外学习率衰减策略。

## 七、常见注意事项

- 示例中的 MNIST 路径是作者本机路径，运行前需要改成本机真实路径。
- `DataLoader.loadDataByUByte(..., true)` 中最后一个参数会影响数据归一化方式，修改时要同步检查训练效果。
- `BPNetwork` 本身不自动构建层，必须通过 `addLayer(...)` 按 forward 顺序添加。
- 最后一层使用 `SoftmaxWithCrossEntropyLayer(10)`，因此标签维度需要和 10 类输出匹配。
- GPU 训练结束后示例会调用 `CUDAMemoryManager.freeAll()` 和 `CUDAMemoryManager.free()` 释放显存资源。
