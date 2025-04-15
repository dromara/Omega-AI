package com.omega.engine.optimizer.lr;

/**
 * LRDecay
 *
 * @author Administrator
 * @remak decayed_learning_rate = learning_rate * decay_rate ^ (global_steps/decay_steps)
 */
public class LRDecay {
    public static float decay_rate = 0.99f;

    public static float decayedLR(float max_lr, float olr, int index, int decay_steps) {
        if (index % decay_steps == 0) {
            olr = (float) (max_lr * Math.pow(decay_rate, (index / decay_steps)));
        }
        return olr;
    }
}

