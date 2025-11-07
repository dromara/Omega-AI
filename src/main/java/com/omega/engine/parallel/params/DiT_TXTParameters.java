package com.omega.engine.parallel.params;

import com.omega.engine.loss.LossType;
import com.omega.engine.updater.UpdaterType;

public class DiT_TXTParameters extends Parameters {
    /**
     *
     */
    private static final long serialVersionUID = -7757960768584358683L;
    private int ditHeadNum = 12;
    private int latendSize = 16;
    private int depth = 12;
    private int timeSteps = 1000;
    private int mlpRatio = 4;
    private int patchSize = 2;
    private int hiddenSize = 768;
    private int latendDim = 32;
    private int textEmbedDim = 2048;
    private int maxContext = 77;
    private boolean learnSigma = false;
    
    private float y_prob = 0.3f;
    
    private float lr;

    public DiT_TXTParameters(LossType lossType, UpdaterType updater, int latendDim, int latendSize, int patchSize, int hiddenSize, int headNum, int depth, int timeSteps, int textEmbedDim, int maxContextLen, int mlpRatio, boolean learnSigma, float y_drop_prob, float lr) {
        this.ditHeadNum = headNum;
        this.lossType = lossType;
        this.updater = updater;
        this.latendDim = latendDim;
        this.latendSize = latendSize;
        this.patchSize = patchSize;
        this.hiddenSize = hiddenSize;
        this.depth = depth;
        this.timeSteps = timeSteps;
        this.textEmbedDim = textEmbedDim;
        this.maxContext = maxContextLen;
        this.mlpRatio = mlpRatio;
        this.learnSigma = learnSigma;
        this.y_prob = y_drop_prob;
        this.lr = lr;
    }

	public int getDitHeadNum() {
		return ditHeadNum;
	}

	public void setDitHeadNum(int ditHeadNum) {
		this.ditHeadNum = ditHeadNum;
	}

	public int getLatendSize() {
		return latendSize;
	}

	public void setLatendSize(int latendSize) {
		this.latendSize = latendSize;
	}

	public int getDepth() {
		return depth;
	}

	public void setDepth(int depth) {
		this.depth = depth;
	}

	public int getTimeSteps() {
		return timeSteps;
	}

	public void setTimeSteps(int timeSteps) {
		this.timeSteps = timeSteps;
	}

	public int getMlpRatio() {
		return mlpRatio;
	}

	public void setMlpRatio(int mlpRatio) {
		this.mlpRatio = mlpRatio;
	}

	public int getPatchSize() {
		return patchSize;
	}

	public void setPatchSize(int patchSize) {
		this.patchSize = patchSize;
	}

	public int getHiddenSize() {
		return hiddenSize;
	}

	public void setHiddenSize(int hiddenSize) {
		this.hiddenSize = hiddenSize;
	}

	public int getLatendDim() {
		return latendDim;
	}

	public void setLatendDim(int latendDim) {
		this.latendDim = latendDim;
	}

	public int getTextEmbedDim() {
		return textEmbedDim;
	}

	public void setTextEmbedDim(int textEmbedDim) {
		this.textEmbedDim = textEmbedDim;
	}

	public int getMaxContext() {
		return maxContext;
	}

	public void setMaxContext(int maxContext) {
		this.maxContext = maxContext;
	}

	public boolean isLearnSigma() {
		return learnSigma;
	}

	public void setLearnSigma(boolean learnSigma) {
		this.learnSigma = learnSigma;
	}

	public float getY_prob() {
		return y_prob;
	}

	public void setY_prob(float y_prob) {
		this.y_prob = y_prob;
	}

	public float getLr() {
		return lr;
	}

	public void setLr(float lr) {
		this.lr = lr;
	}

}

