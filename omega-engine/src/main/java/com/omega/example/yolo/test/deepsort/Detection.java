package com.omega.example.yolo.test.deepsort;

/**
 * 检测结果类
 * rect需要传入非标准化的数据
 */
public class Detection {
    public Rect bbox;
    public float confidence;
    public String label;
    public int classId;
    
    public Detection(Rect bbox, float confidence, String label, int classId) {
        this.bbox = bbox;
        this.confidence = confidence;
        this.label = label;
        this.classId = classId;
    }
}