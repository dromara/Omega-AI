package com.omega.engine.updater;

/**
 * UpdaterType
 *
 * @author Administrator
 */
public enum UpdaterType {
    none("none"), momentum("momentum"), sgd("sgd"), adam("adam"), RMSProp("rmsprop"), adamw("adamw");
    private String key;

    UpdaterType(String key) {
        this.key = key;
    }

    public String getKey() {
        return key;
    }
}

