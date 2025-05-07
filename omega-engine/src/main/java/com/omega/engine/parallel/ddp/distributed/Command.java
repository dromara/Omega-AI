package com.omega.engine.parallel.ddp.distributed;

import java.io.Serializable;

public class Command implements Serializable {
    /**
     *
     */
    private static final long serialVersionUID = 1L;
    private String code;
    private Object data;

    public Command() {
    }

    public Command(String code, Object data) {
        this.code = code;
        this.data = data;
    }

    public String getCode() {
        return code;
    }

    public void setCode(String code) {
        this.code = code;
    }

    public Object getData() {
        return data;
    }

    public void setData(Object data) {
        this.data = data;
    }
}

