package com.omega.engine.parallel.params;

import com.omega.engine.loss.LossType;
import com.omega.engine.updater.UpdaterType;

import java.util.Map;

public class Llama3Parameters extends Parameters {
    /**
     *
     */
    private static final long serialVersionUID = -7757960768584358683L;
    private int vocabSize;
    private int time;
    private int embedDim;
    private int headNum = 8;
    private int nKVHeadNum = 8;
    private int decoderNum = 1;
    private int multiple_of = 64;
    private boolean bias = true;
    private boolean flashAttention = false;
    private boolean dropout;

    public Llama3Parameters(LossType lossType, UpdaterType updater, int headNum, int nKVHeadNum, int decoderNum, int vocabSize, int time, int embedDim, boolean bias, boolean dropout, boolean flashAttention, float learnRate) {
        this.flashAttention = flashAttention;
        this.lossType = lossType;
        this.bias = bias;
        this.dropout = dropout;
        this.decoderNum = decoderNum;
        this.updater = updater;
        this.headNum = headNum;
        this.nKVHeadNum = nKVHeadNum;
        this.time = time;
        this.vocabSize = vocabSize;
        this.embedDim = embedDim;
        this.learnRate = learnRate;
    }

    public static Llama3Parameters createByMap(Map<String, Object> params) {
        LossType lossType = Enum.valueOf(LossType.class, params.get("lossType").toString());
        UpdaterType updater = Enum.valueOf(UpdaterType.class, params.get("updater").toString());
        int headNum = (int) Double.parseDouble(params.get("headNum").toString());
        int nKVHeadNum = (int) Double.parseDouble(params.get("nKVHeadNum").toString());
        int decoderNum = (int) Double.parseDouble(params.get("decoderNum").toString());
        int vocabSize = (int) Double.parseDouble(params.get("vocabSize").toString());
        int time = (int) Double.parseDouble(params.get("time").toString());
        int embedDim = (int) Double.parseDouble(params.get("embedDim").toString());
        boolean bias = (boolean) params.get("bias");
        boolean dropout = (boolean) params.get("dropout");
        boolean flashAttention = (boolean) params.get("flashAttention");
        float learnRate = (float) params.get("learnRate");
        return new Llama3Parameters(lossType, updater, headNum, nKVHeadNum, decoderNum, vocabSize, time, embedDim, bias, dropout, flashAttention, learnRate);
    }

    public int getVocabSize() {
        return vocabSize;
    }

    public void setVocabSize(int vocabSize) {
        this.vocabSize = vocabSize;
    }

    public int getEmbedDim() {
        return embedDim;
    }

    public void setEmbedDim(int embedDim) {
        this.embedDim = embedDim;
    }

    public int getHeadNum() {
        return headNum;
    }

    public void setHeadNum(int headNum) {
        this.headNum = headNum;
    }

    public int getnKVHeadNum() {
        return nKVHeadNum;
    }

    public void setnKVHeadNum(int nKVHeadNum) {
        this.nKVHeadNum = nKVHeadNum;
    }

    public int getDecoderNum() {
        return decoderNum;
    }

    public void setDecoderNum(int decoderNum) {
        this.decoderNum = decoderNum;
    }

    public int getMultiple_of() {
        return multiple_of;
    }

    public void setMultiple_of(int multiple_of) {
        this.multiple_of = multiple_of;
    }

    public boolean isBias() {
        return bias;
    }

    public void setBias(boolean bias) {
        this.bias = bias;
    }

    public boolean isFlashAttention() {
        return flashAttention;
    }

    public void setFlashAttention(boolean flashAttention) {
        this.flashAttention = flashAttention;
    }

    public boolean isDropout() {
        return dropout;
    }

    public void setDropout(boolean dropout) {
        this.dropout = dropout;
    }

    public int getTime() {
        return time;
    }

    public void setTime(int time) {
        this.time = time;
    }
}

