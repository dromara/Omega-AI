package com.omega.engine.loss;

/**
 * LossType
 *
 * @author Administrator
 */
public enum LossType {
    cross_entropy, softmax_with_cross_entropy, softmax_with_cross_entropy_idx, multiLabel_soft_margin, detection, yolo, yolov2, yolov3, yolov7, MSE, MSE_SUM, BCE, BCEWithLogits, Hinge
}

