package com.omega.engine.gpu;

import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;

import java.util.HashMap;
import java.util.Map;

/**
 * @author Administrator
 */
public class MyCUDAModule extends CUmodule {
    private Map<String, CUfunction> functions = new HashMap<String, CUfunction>();

    public Map<String, CUfunction> getFunctions() {
        return functions;
    }

    public void setFunctions(Map<String, CUfunction> functions) {
        this.functions = functions;
    }
}

