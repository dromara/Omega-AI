package com.omega.engine.optimizer.lr;

public class HalfDecay {
    public static float decay_rate = 0.5f;

    public static float decayedLR(float olr) {
        olr = olr * decay_rate;
        return olr;
    }
}

