package com.omega.engine.nn.model;

import com.omega.engine.nn.layer.Convolution3DTransposeLayer;
import com.omega.engine.nn.layer.ConvolutionLayer;
import com.omega.engine.nn.layer.ConvolutionTransposeLayer;

/**
 * ConvLayerInit
 *
 * @author Administrator
 */
public class ConvLayerInit extends LayerInit {
    /**
     *
     */
    private static final long serialVersionUID = -8722541684397318328L;
    public int kernelNum = 0;
    public int kWidth = 0;
    public int kHeight = 0;
    public int stride = 1;
    public int padding = 0;

    public ConvLayerInit(ConvolutionLayer layer) {
        // TODO Auto-generated constructor stub
        super(layer);
        this.kernelNum = layer.kernelNum;
        this.kWidth = layer.kWidth;
        this.kHeight = layer.kHeight;
        this.stride = layer.stride;
        this.padding = layer.padding;
    }

    public ConvLayerInit(ConvolutionTransposeLayer layer) {
        // TODO Auto-generated constructor stub
        super(layer);
        this.kernelNum = layer.kernelNum;
        this.kWidth = layer.kWidth;
        this.kHeight = layer.kHeight;
        this.stride = layer.stride;
        this.padding = layer.padding;
    }
    
    public ConvLayerInit(Convolution3DTransposeLayer layer) {
        // TODO Auto-generated constructor stub
        super(layer);
        this.kernelNum = layer.kernelNum;
        this.kWidth = layer.kWidth;
        this.kHeight = layer.kHeight;
        this.stride = layer.stride;
        this.padding = layer.padding;
    }

    public int getKernelNum() {
        return kernelNum;
    }

    public void setKernelNum(int kernelNum) {
        this.kernelNum = kernelNum;
    }

    public int getkWidth() {
        return kWidth;
    }

    public void setkWidth(int kWidth) {
        this.kWidth = kWidth;
    }

    public int getkHeight() {
        return kHeight;
    }

    public void setkHeight(int kHeight) {
        this.kHeight = kHeight;
    }

    public int getStride() {
        return stride;
    }

    public void setStride(int stride) {
        this.stride = stride;
    }

    public int getPadding() {
        return padding;
    }

    public void setPadding(int padding) {
        this.padding = padding;
    }
}

