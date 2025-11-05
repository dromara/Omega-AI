package com.omega.example.yolo.test.deepsort;


/**
 * 跟踪结果类
 */
public class Track {
    public int trackId;
    public Rect bbox;
    public String label;
    public int age; // 跟踪存在的时间
    public int totalVisibleCount; // 总共被看到的次数
    public int consecutiveInvisibleCount; // 连续未被看到的次数
    public KalmanFilter kf;
    
    public Track(int trackId, Rect bbox, String label, KalmanFilter kf) {
        this.trackId = trackId;
        this.bbox = bbox;
        this.label = label;
        this.kf = kf;
        this.age = 1;
        this.totalVisibleCount = 1;
        this.consecutiveInvisibleCount = 0;
    }
}