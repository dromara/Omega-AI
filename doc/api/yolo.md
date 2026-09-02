# Yolo 实时目标检测模型

本页只记录 Omega-AI 当前源码中真实存在的实现入口和阅读路径。API 示例必须以源码为准，不使用伪类名或非项目实现的示例代码。

[[toc]]

---

## 一、源码入口

| 类型 | 位置 |
| --- | --- |
| 主要实现 | `com.omega.engine.nn.network.Yolo` |
| 源码路径 | `src/main/java/com/omega/engine/nn/network/Yolo.java` |
| 示例入口 | `src/main/java/com/omega/example/yolo/test/YoloV3Test.java` |
| 示例代码（Demo） | `YoloV3Test.yolov3_tiny()` |

## 二、入口代码片段

节选自 `src/main/java/com/omega/engine/nn/network/Yolo.java`，仅保留入口签名和关键初始化，完整实现以源码为准：

```java
public class Yolo extends OutputsNetwork {
    private LossFunction[] losses;
    private LossType lossType;
    private Tensor[] loss;
    private Tensor[] lossDiff;
    private int class_num = 1;

    public Yolo(LossFunction lossFunction) {
        this.lossFunction = lossFunction;
    }

    public Yolo(LossType lossType, UpdaterType updater) {
        this.lossType = lossType;
        this.updater = updater;
    }
}
```

## 三、示例代码（Demo）

节选自 `src/main/java/com/omega/example/yolo/test/YoloV3Test.java`，保留 YOLOv3-tiny 训练入口的关键初始化和训练调用：

```java
public class YoloV3Test {
    public void yolov3_tiny() {
        int im_w = 256;
        int im_h = 256;
        int batchSize = 64;
        int class_num = 1;
        String cfg_path = "H:\\voc\\banana-detection\\yolov3-tiny-banana.cfg";
        String trainPath = "H:\\voc\\banana-detection\\bananas_train\\images";
        String trainLabelPath = "H:\\voc\\banana-detection\\bananas_train\\label.csv";
        String testPath = "H:\\voc\\banana-detection\\bananas_val\\images";
        String testLabelPath = "H:\\voc\\banana-detection\\bananas_val\\label.csv";

        YoloDataTransform2 dt = new YoloDataTransform2(class_num, DataType.yolov3, 90);
        DetectionDataLoader trainData = new DetectionDataLoader(trainPath, trainLabelPath, LabelFileType.csv, im_w, im_h, class_num, batchSize, DataType.yolov3, dt);
        DetectionDataLoader vailData = new DetectionDataLoader(testPath, testLabelPath, LabelFileType.csv, im_w, im_h, class_num, batchSize, DataType.yolov3);

        Yolo netWork = new Yolo(LossType.yolov3, UpdaterType.adamw);
        netWork.CUDNN = true;
        netWork.learnRate = 0.001f;
        ModelLoader.loadConfigToModel(netWork, cfg_path);

        MBSGDOptimizer optimizer = new MBSGDOptimizer(netWork, 2000, 0.001f, batchSize, LearnRateUpdate.SMART_HALF, false);
        optimizer.lr_step = new int[]{200, 500, 1000, 1200, 2000};
        optimizer.trainObjectRecognitionOutputs(trainData, vailData);

        List<YoloBox> draw_bbox = optimizer.showObjectRecognitionYoloV3(vailData, batchSize);
        String outputPath = "H:\\voc\\banana-detection\\test_yolov3\\";
        showImg(outputPath, vailData, class_num, draw_bbox, batchSize, false, im_w, im_h, null);
    }
}
```

## 四、相关组件

`Yolo、YoloLayer、YoloLoss、YoloDataLoader、YoloDecode`

## 五、阅读建议

1. 先打开示例入口，查看 `main` 方法或训练方法中如何准备数据、创建网络和启动训练。
2. 再进入主要实现类，查看构造函数、`init`、`forward`、`back`、`loss`、`update` 等方法。
3. 如果涉及 GPU 加速，继续追踪对应 layer、kernel wrapper 和 `src/main/resources/cu` 下的 CUDA 实现。
4. 文档中的类名、构造参数和调用方式必须与当前源码保持一致；如果源码变更，以源码为准同步更新本文档。

## 六、常见注意事项

- 示例中的数据集路径通常是作者本机路径，运行前需要改成本机目录。
- 训练 batch size、学习率、是否启用 CUDNN/CUDA 需要结合显存和任务规模调整。
- 不同模型的 loss、label 格式和输出 shape 不同，不能直接混用其它模型示例。
- 若需要补充代码片段，应直接从对应示例类中摘取真实代码，并标注来源方法。
