package com.omega.example.transformer.dataset.parallel.params;

import com.omega.engine.tensor.Tensor;

public class SFTBinParamters extends DataLoaderParamters {
    /**
     *
     */
    private static final long serialVersionUID = -2932915859259390507L;
    private Tensor label;
    private float[] tmpInput;
    private float[] tmpLabel;
    private int[] padCount;
    private int it;
    private Tensor cos;
    private Tensor sin;

    public SFTBinParamters(Tensor input, Tensor label, float[] tmpInput, float[] tmpLabel, int[] padCount, int it, Tensor cos, Tensor sin) {
        setInput(input);
        this.label = label;
        this.tmpInput = tmpInput;
        this.tmpLabel = tmpLabel;
        this.padCount = padCount;
        this.it = it;
        this.setCos(cos);
        this.setSin(sin);
    }

    public Tensor getLabel() {
        return label;
    }

    public void setLabel(Tensor label) {
        this.label = label;
    }

    public float[] getTmpInput() {
        return tmpInput;
    }

    public void setTmpInput(float[] tmpInput) {
        this.tmpInput = tmpInput;
    }

    public float[] getTmpLabel() {
        return tmpLabel;
    }

    public void setTmpLabel(float[] tmpLabel) {
        this.tmpLabel = tmpLabel;
    }

    public int[] getPadCount() {
        return padCount;
    }

    public void setPadCount(int[] padCount) {
        this.padCount = padCount;
    }

    public int getIt() {
        return it;
    }

    public void setIt(int it) {
        this.it = it;
    }

    public Tensor getCos() {
        return cos;
    }

    public void setCos(Tensor cos) {
        this.cos = cos;
    }

    public Tensor getSin() {
        return sin;
    }

    public void setSin(Tensor sin) {
        this.sin = sin;
    }
}

