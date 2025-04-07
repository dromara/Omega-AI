# YOLO 实时目标检测模型

## 概述
YOLO（You Only Look Once）是一种单阶段目标检测算法，通过将目标检测转化为回归问题实现实时检测。最新版本在保持高精度的同时达到100+ FPS。

## 核心结构
```java
/**
 * YOLOv4基础实现
 * 包含CSPDarknet骨干网络 + PANet特征金字塔 + YOLO检测头
 */
public class YOLO {
    private CSPDarknet backbone;   // 骨干网络
    private PANet neck;           // 特征金字塔
    private YOLOHead head;        // 检测头
    
    // 检测参数
    private float confThreshold = 0.5f;  // 置信度阈值
    private float nmsThreshold = 0.4f;   // NMS阈值
    private int[] inputSize = {608, 608};// 输入尺寸
}
```

## 完整实现

### 1. YOLO检测层
```java
public class YOLOLayer {
    private int gridSize;       // 特征图尺寸
    private int numBBox;        // 每个网格预测框数量
    private int numClasses;     // 类别数量
    
    public Detection[] decode(float[] features) {
        List<Detection> detections = new ArrayList<>();
        // 将特征图转换为[grid,grid,bbox*(5+numClasses)]
        for (int i=0; i<gridSize; i++) {
            for (int j=0; j<gridSize; j++) {
                int base = (i * gridSize + j) * (5 + numClasses);
                // 解码预测框
                float x = (sigmoid(features[base]) + j) / gridSize;
                float y = (sigmoid(features[base+1]) + i) / gridSize;
                float w = exp(features[base+2]) * anchors[0];
                float h = exp(features[base+3]) * anchors[1];
                float conf = sigmoid(features[base+4]);
                
                // 筛选高置信度预测
                if (conf > confThreshold) {
                    Detection det = new Detection(x, y, w, h, conf);
                    det.classProbs = softmax(Arrays.copyOfRange(
                        features, base+5, base+5+numClasses));
                    detections.add(det);
                }
            }
        }
        return nms(detections);
    }
}
```

### 2. 损失函数
```java
public class YOLOLoss {
    // 三部分损失：坐标损失 + 置信度损失 + 分类损失
    public float calculate(float[][][] predictions, float[][][] targets) {
        float totalLoss = 0;
        for (int s=0; s<scales.length; s++) { // 多尺度预测
            float[][] pred = predictions[s];
            float[][] target = targets[s];
            
            // 计算坐标损失（带权重）
            float xyLoss = calculateXYLoss(pred, target);
            float whLoss = calculateWHLoss(pred, target);
            float objLoss = calculateConfidenceLoss(pred, target);
            float clsLoss = calculateClassLoss(pred, target);
            
            totalLoss += (5 * xyLoss + 5 * whLoss + objLoss + clsLoss);
        }
        return totalLoss;
    }
}
```

## 使用示例（COCO数据集训练）
```java
public class ObjectDetector {
    public static void main(String[] args) {
        // 创建YOLOv4模型（80类COCO数据集）
        YOLO model = new YOLO(80)
            .setInputSize(608)
            .setPretrainedBackbone("cspdarknet53.cfg");
        
        // 配置混合精度训练
        model.enableAMP();
        
        // 加载COCO数据集
        COCODataset dataset = new COCODataset("annotations/instances_train2017.json");
        
        // 训练配置
        model.setOptimizer(new SGD(0.001f, 0.9f))
             .setLoss(new YOLOLoss())
             .setAugmentation(new MosaicAugmentation());
        
        // 训练循环
        for (int epoch=0; epoch<300; epoch++) {
            float map = model.trainEpoch(dataset);
            System.out.printf("Epoch %03d mAP@0.5: %.2f%%\n", epoch+1, map*100);
            
            // 保存检查点
            if (epoch % 10 == 0) {
                model.saveWeights("checkpoints/yolov4_epoch"+epoch+".weights");
            }
        }
        
        // 检测示例
        BufferedImage img = ImageIO.read(new File("street.jpg"));
        Detection[] results = model.detect(img);
        drawBoxes(img, results).save("detection_result.jpg");
    }
}
```

## 性能优化
1. **TensorRT加速**：
```java
public void convertToTensorRT() {
    this.engine = new TensorRTEngine()
        .setFP16Mode(true)
        .buildFromONNX("yolov4.onnx");
}
```

2. **多线程预处理**：
```java
public void enableParallelProcessing() {
    this.executor = Executors.newFixedThreadPool(
        Runtime.getRuntime().availableProcessors());
}
```

## 常见问题
### Q1：如何处理重叠框？
- NMS非极大值抑制实现：
```java
private Detection[] nms(Detection[] detections) {
    Arrays.sort(detections, (a,b) -> Float.compare(b.confidence, a.confidence));
    List<Detection> result = new ArrayList<>();
    while (!detections.isEmpty()) {
        Detection keep = detections[0];
        result.add(keep);
        detections = Arrays.stream(detections)
            .filter(d -> iou(keep, d) < nmsThreshold)
            .toArray(Detection[]::new);
    }
    return result.toArray(new Detection[0]);
}
```

### Q2：小目标检测效果差？
- 改进方案：添加高分辨率检测层
```java
public void addHighResolutionHead() {
    this.heads.add(new YOLOHead(1024, 512, new float[]{ 
        new Size(12,16), new Size(19,36), new Size(40,28) 
    }));
}
```

## 扩展阅读
- [YOLOv5改进架构](/doc/extension/yolov5)
- [YOLOX锚框改进](/doc/extension/yolox)
- [边缘设备部署指南](/doc/extension/edge-deployment)