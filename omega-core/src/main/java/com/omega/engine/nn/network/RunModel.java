package com.omega.engine.nn.network;

public enum RunModel {
    TRAIN("train"), EVAL("eval"), TEST("test");
    private String key;

    RunModel(String key) {
        this.key = key;
    }

    public String getKey() {
        return key;
    }
}

