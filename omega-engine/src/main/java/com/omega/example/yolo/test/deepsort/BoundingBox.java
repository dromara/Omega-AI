package com.omega.example.yolo.test.deepsort;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * 
 */
public class BoundingBox implements Comparable<BoundingBox> {
    /**
     * 置信度。
     */
    public float confidence;
    /**
     * 所属类别。
     */
    public int predictedClass;
    
//    置信度阈值
    private static float CONFIDENCE_THRESHOLD = 0.05f;
    
    // 交并比阈值
    private static float IOU_THRESHOLD = 0.7f;
    private float[] box;
    
    
    /**
     * 构造函数。
     */
    public BoundingBox(float[] box, float confidence, int predictedClass) {
        this.box = box;
        this.confidence = confidence;
        this.predictedClass = predictedClass;
    }
    /**
     * 依据置信度比较边界框。
     */
    @Override
    public int compareTo(BoundingBox other) {
        return Float.compare(this.confidence, other.confidence);
    }
    /**
     * 计算并返回两个边界框的交并比。
     */
    public static float getIOU(float[] box1, float[] box2) {  	
        // box1[0]=centerX, box1[1]=centerY, box1[2]=width, box1[3]=height
        float box1_x1 = box1[0] - box1[2] / 2.0f;
        float box1_y1 = box1[1] - box1[3] / 2.0f;
        float box1_x2 = box1[0] + box1[2] / 2.0f;
        float box1_y2 = box1[1] + box1[3] / 2.0f;
        
        float box2_x1 = box2[0] - box2[2] / 2.0f;
        float box2_y1 = box2[1] - box2[3] / 2.0f;
        float box2_x2 = box2[0] + box2[2] / 2.0f;
        float box2_y2 = box2[1] + box2[3] / 2.0f;
        
        // 计算交集区域的坐标[2,4]
        float inter_x1 = Math.max(box1_x1, box2_x1);
        float inter_y1 = Math.max(box1_y1, box2_y1);
        float inter_x2 = Math.min(box1_x2, box2_x2);
        float inter_y2 = Math.min(box1_y2, box2_y2);
        
        // 计算交集面积
        float inter_width = Math.max(0, inter_x2 - inter_x1);
        float inter_height = Math.max(0, inter_y2 - inter_y1);
        float intersection = inter_width * inter_height;
        
        // 计算各自面积
        float area1 = box1[2] * box1[3];
        float area2 = box2[2] * box2[3];
        
        // 计算并集面积
        float union = area1 + area2 - intersection;
        
        // 避免除零错误
        if (union <= 0) {
            return 0.0f;
        }
        
        return intersection / union;
    }
    
    /**
     * 非极大值抑制方法。
     */
    public static List<BoundingBox> nms(List<BoundingBox> boxes) {
        List<BoundingBox> result = new ArrayList<>();
        boxes.sort(Collections.reverseOrder());
        for (int i = 0; i < boxes.size(); ++i) {
            if (boxes.get(i).confidence < CONFIDENCE_THRESHOLD) continue;
            result.add(boxes.get(i));
            for (int j = i + 1; j < boxes.size(); ++j) {
            	float iou = getIOU(boxes.get(i).box, boxes.get(j).box);
                if (iou > IOU_THRESHOLD) {
                    boxes.remove(j);
                    --j;
                }
            }
        }
        return result;
    }
    
    public float getX(float width) {
    	return (box[0] - box[2] / 2.0f) * width;
	}
    public float getY(float height) {
    	return (box[1] - box[3] / 2.0f) * height;
    }
    public float getWidth(float width) {
    	return ((box[0] + box[2] / 2.0f) - (box[0] - box[2] / 2.0f)) * width;
    }
    public float getHeight(float width) {
    	return ((box[1] + box[3] / 2.0f) - (box[1] - box[3] / 2.0f)) * width;
    }
}