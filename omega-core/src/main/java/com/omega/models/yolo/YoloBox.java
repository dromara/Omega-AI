package com.omega.models.yolo;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class YoloBox implements Serializable {
    /**
     *
     */
    private static final long serialVersionUID = -3853067091486009638L;
    private List<YoloDetection> dets;

    public YoloBox(YoloDetection[] array) {
        this.dets = new ArrayList<>(Arrays.asList(array));
    }

    public List<YoloDetection> getDets() {
        return dets;
    }

    public void setDets(List<YoloDetection> dets) {
        this.dets = dets;
    }
}

