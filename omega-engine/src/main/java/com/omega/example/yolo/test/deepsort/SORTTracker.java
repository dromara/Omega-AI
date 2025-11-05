package com.omega.example.yolo.test.deepsort;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

/**
 * 追踪器
 */
public class SORTTracker {
    private List<Track> tracks;
    private int nextId;
    private int maxAge; // 最大丢失帧数
    private int minHits; // 最小匹配次数才确认轨迹
    private double iouThreshold; // IOU匹配阈值
    
    public SORTTracker() {
        this.tracks = new ArrayList<>();
        this.nextId = 1;
        this.maxAge = 5; // 目标丢失30帧后删除
        this.minHits = 3; // 匹配3次后才确认显示
        this.iouThreshold = 0.3; // IOU阈值
    }
    
    public List<Track> update(List<Detection> detections) {
        // 步骤1: 对现有轨迹进行预测
        List<Rect> predictedBoxes = new ArrayList<>();
        for (Track track : tracks) {
            Rect predicted = track.kf.predict();
            predictedBoxes.add(predicted);
        }
        
        // 步骤2: 使用匈牙利算法进行匹配
        int[][] matches = hungarianMatch(detections, predictedBoxes);
        
        List<Integer> matchedDetections = new ArrayList<>();
        List<Integer> matchedTracks = new ArrayList<>();
        List<Integer> unmatchedDetections = new ArrayList<>();
        List<Integer> unmatchedTracks = new ArrayList<>();
        
        // 处理匹配结果
        processMatches(matches, detections, predictedBoxes, 
                      matchedDetections, matchedTracks, 
                      unmatchedDetections, unmatchedTracks);
        
        // 步骤3: 更新匹配的轨迹
        for (int i = 0; i < matchedDetections.size(); i++) {
            int detIdx = matchedDetections.get(i);
            int trackIdx = matchedTracks.get(i);
            
            Track track = tracks.get(trackIdx);
            Detection det = detections.get(detIdx);
            
            // 更新卡尔曼滤波器
            track.kf.update(det.bbox);
            track.bbox = det.bbox; // 使用检测结果
            track.totalVisibleCount++;
            track.consecutiveInvisibleCount = 0;
        }
        
        // 步骤4: 为未匹配的检测创建新轨迹
        for (int detIdx : unmatchedDetections) {
            Detection det = detections.get(detIdx);
            KalmanFilter kf = new KalmanFilter(det.bbox);
            Track newTrack = new Track(nextId++, det.bbox, det.label, kf);
            tracks.add(newTrack);
        }
        
        // 步骤5: 处理未匹配的轨迹（目标丢失）
        List<Track> tracksToRemove = new ArrayList<>();
        for (int trackIdx : unmatchedTracks) {
            Track track = tracks.get(trackIdx);
            track.consecutiveInvisibleCount++;
            
            if (track.consecutiveInvisibleCount > maxAge) {
                tracksToRemove.add(track);
            }
        }
        tracks.removeAll(tracksToRemove);
        
        // 步骤6: 返回活跃的轨迹（满足最小匹配次数的）
        return tracks.stream()
                .filter(track -> track.totalVisibleCount >= minHits)
                .collect(Collectors.toList());
    }
    
    private int[][] hungarianMatch(List<Detection> detections, List<Rect> predictedBoxes) {
        if (detections.isEmpty() || predictedBoxes.isEmpty()) {
            return new int[0][2];
        }
        
        // 计算IOU矩阵
        double[][] iouMatrix = new double[detections.size()][predictedBoxes.size()];
        for (int i = 0; i < detections.size(); i++) {
            for (int j = 0; j < predictedBoxes.size(); j++) {
                iouMatrix[i][j] = calculateIOU(detections.get(i).bbox, predictedBoxes.get(j));
            }
        }
        
        // 简化版匈牙利匹配（实际应用中可使用更高效的实现）
        return greedyMatching(iouMatrix);
    }
    
    private int[][] greedyMatching(double[][] iouMatrix) {
        List<int[]> matches = new ArrayList<>();
        boolean[] usedDetections = new boolean[iouMatrix.length];
        boolean[] usedTracks = new boolean[iouMatrix[0].length];
        
        // 按IOU从大到小排序可能的匹配
        List<MatchCandidate> candidates = new ArrayList<>();
        for (int i = 0; i < iouMatrix.length; i++) {
            for (int j = 0; j < iouMatrix[0].length; j++) {
                if (iouMatrix[i][j] > iouThreshold) {
                    candidates.add(new MatchCandidate(i, j, iouMatrix[i][j]));
                }
            }
        }
        
        // 按IOU降序排序
        candidates.sort((a, b) -> Double.compare(b.iou, a.iou));
        
        // 贪心匹配
        for (MatchCandidate candidate : candidates) {
            if (!usedDetections[candidate.detIdx] && !usedTracks[candidate.trackIdx]) {
                matches.add(new int[]{candidate.detIdx, candidate.trackIdx});
                usedDetections[candidate.detIdx] = true;
                usedTracks[candidate.trackIdx] = true;
            }
        }
        
        return matches.toArray(new int[0][2]);
    }
    
    private void processMatches(int[][] matches, List<Detection> detections, 
                               List<Rect> predictedBoxes,
                               List<Integer> matchedDetections,
                               List<Integer> matchedTracks,
                               List<Integer> unmatchedDetections,
                               List<Integer> unmatchedTracks) {
        // 初始化所有检测和轨迹为未匹配
        for (int i = 0; i < detections.size(); i++) {
            unmatchedDetections.add(i);
        }
        for (int i = 0; i < predictedBoxes.size(); i++) {
            unmatchedTracks.add(i);
        }
        
        // 处理匹配
        for (int[] match : matches) {
            int detIdx = match[0];
            int trackIdx = match[1];
            
            matchedDetections.add(detIdx);
            matchedTracks.add(trackIdx);
            unmatchedDetections.remove((Integer) detIdx);
            unmatchedTracks.remove((Integer) trackIdx);
        }
    }
    
    private double calculateIOU(Rect rect1, Rect rect2) {
        int x1 = Math.max(rect1.x, rect2.x);
        int y1 = Math.max(rect1.y, rect2.y);
        int x2 = Math.min(rect1.x + rect1.width, rect2.x + rect2.width);
        int y2 = Math.min(rect1.y + rect1.height, rect2.y + rect2.height);
        
        int intersectionArea = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
        double unionArea = rect1.area() + rect2.area() - intersectionArea;
        
        if (unionArea == 0) return 0;
        return (double) intersectionArea / unionArea;
    }
    
    private static class MatchCandidate {
        int detIdx;
        int trackIdx;
        double iou;
        
        MatchCandidate(int detIdx, int trackIdx, double iou) {
            this.detIdx = detIdx;
            this.trackIdx = trackIdx;
            this.iou = iou;
        }
    }
}