package com.omega.models.transformer;

import java.lang.instrument.Instrumentation;

public class ObjectSizeCalculator {
    private static Instrumentation instrumentation;

    public static void premain(String args, Instrumentation inst) {
        instrumentation = inst;
    }

    public static long getObjectSize(Object object) {
        return instrumentation.getObjectSize(object);
    }
}

