package com.omega.example.asr.dataset;

import com.omega.common.utils.MathUtils;
import com.omega.common.utils.MatrixUtils;
import com.omega.engine.tensor.Tensor;
import com.omega.example.asr.utils.FBank;
import com.omega.example.asr.utils.WAVFBankLoader;
import com.omega.example.transformer.utils.LagJsonReader;
import com.omega.example.transformer.utils.tokenizers.Tokenizer;
import com.omega.example.yolo.data.BaseDataLoader;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class AudioDataset extends BaseDataLoader {
    public Tokenizer tokenizer;
    public int vocab_size;
    public String[] vocab;
    public int count;
    public int count_it;
    private String labelPath;
    private String wavDirPath;
    private int maxWavLength = 10;
    private int numBins = 80;
    private int maxContextLen;
    private List<Map<String, Object>> datas;
    private String[] idxSet;

    public AudioDataset(Tokenizer tokenizer, String labelPath, String wavDirPath, int numBins, int maxWavLength, int maxContextLen, int batchSize) {
        this.tokenizer = tokenizer;
        this.labelPath = labelPath;
        this.wavDirPath = wavDirPath;
        this.numBins = numBins;
        this.setMaxWavLength(maxWavLength);
        this.setMaxContextLen(maxContextLen);
        this.batchSize = batchSize;
        init();
    }

    public void init() {
        loadFileCount();
    }

    public void loadFileCount() {
        try {
            File file = new File(wavDirPath);
            if (file.exists()) {
                datas = LagJsonReader.readJsonDataSamll(labelPath);
                List<String> idxList = new ArrayList<String>();
                List<Map<String, Object>> rmList = new ArrayList<Map<String, Object>>();
                for (int i = 0; i < datas.size(); i++) {
                    String path = datas.get(i).get("path").toString();
                    if (FBank.checkMaxSeqLen(wavDirPath + path, i, getMaxWavLength())) {
                        idxList.add(path);
                    } else {
                        rmList.add(datas.get(i));
                    }
                }
                for (Map<String, Object> idx : rmList) {
                    datas.remove(idx);
                }
                idxSet = new String[datas.size()];
                idxSet = idxList.toArray(idxSet);
            }
            this.number = datas.size();
            count = datas.size();
            count_it = datas.size() / batchSize;
            System.err.println("data count[" + count + "].");
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    @Override
    public int[][] shuffle() {
        // TODO Auto-generated method stub
        return MathUtils.randomInts(this.number, this.batchSize);
    }

    public int[][] order() {
        // TODO Auto-generated method stub
        return MathUtils.orderInts(this.number, this.batchSize);
    }

    @Override
    public void loadData(int[] indexs, Tensor input) {
        // TODO Auto-generated method stub
        //		WAVFBankLoader.load(wavDirPath, idxSet, indexs, batchSize, input, numBins, maxWavLength);
        //		input.hostToDevice();
    }

    @Override
    public void loadData(int[] indexs, Tensor input, Tensor label) {
        // TODO Auto-generated method stub
        //
        //		WAVFBankLoader.load(wavDirPath, idxSet, indexs, batchSize, input, numBins, maxWavLength);
        //
        //		loadLabels(indexs, label);
        //		/**
        //		 * copy data to gpu.
        //		 *
        //		 */
        //		input.hostToDevice();
        //		label.hostToDevice();
    }

    public void loadData(int[] indexs, Tensor input, Tensor inputLen, Tensor labelInput, Tensor labelLen, Tensor label) {
        // TODO Auto-generated method stub
        WAVFBankLoader.load(wavDirPath, idxSet, indexs, batchSize, input, inputLen, getNumBins(), getMaxWavLength());
        loadLabels(indexs, labelInput, labelLen);
        /**
         * copy data to gpu.
         *
         */
        input.hostToDevice();
        labelInput.hostToDevice();
        inputLen.hostToDevice();
        labelLen.hostToDevice();
        labelInput.copyGPU(label);
    }

    public void loadData(int[] indexs, Tensor input, Tensor inputLen, Tensor labelInput, Tensor labelLen, Tensor label, int[] count) {
        // TODO Auto-generated method stub
        WAVFBankLoader.load(wavDirPath, idxSet, indexs, batchSize, input, inputLen, getNumBins(), getMaxWavLength());
        loadLabels(indexs, labelInput, labelLen, label);
        count[0] = (int) MatrixUtils.sum(labelLen.data);
        /**
         * copy data to gpu.
         *
         */
        input.hostToDevice();
        labelInput.hostToDevice();
        inputLen.hostToDevice();
        labelLen.hostToDevice();
        label.hostToDevice();
    }

    @Override
    public void loadData(int pageIndex, int batchSize, Tensor input, Tensor label) {
        // TODO Auto-generated method stub
    }

    @Override
    public float[] loadData(int index) {
        // TODO Auto-generated method stub
        return null;
    }

    @Override
    public Tensor initLabelTensor() {
        // TODO Auto-generated method stub
        return null;
    }

    public void loadLabels(int[] indexs, Tensor label) {
        for (int i = 0; i < indexs.length; i++) {
            int idx = indexs[i];
            String text = tokenizer.sos_str() + datas.get(idx).get("text").toString() + tokenizer.eos_str();
            int[] ids = tokenizer.encodeInt(text, getMaxContextLen());
            for (int j = 0; j < getMaxContextLen(); j++) {
                label.data[i * getMaxContextLen() + j] = ids[j];
            }
        }
    }
    
    public void loadLabels(int[] indexs, Tensor label, Tensor labelLen) {
        for (int i = 0; i < indexs.length; i++) {
            int idx = indexs[i];
            String text = tokenizer.sos_str() + datas.get(idx).get("text").toString() + tokenizer.eos_str();
            int[] ids = tokenizer.encodeInt(text, getMaxContextLen());
            for (int j = 0; j < getMaxContextLen(); j++) {
                float v = ids[j];
                label.data[i * getMaxContextLen() + j] = v;
                if (v == tokenizer.eos()) {
                    labelLen.data[i] = j + 1;
                }
            }
        }
    }
    
	public void loadLabels(int[] indexs, Tensor labelInput, Tensor labelLen, Tensor label) {
		for (int i = 0; i < indexs.length; i++) {
			int idx = indexs[i];
			String text = tokenizer.sos_str() + datas.get(idx).get("text").toString() + tokenizer.eos_str();
			int[] ids = tokenizer.encodeInt(text, getMaxContextLen() + 1);
			for (int j = 0; j < getMaxContextLen(); j++) {
				float current = ids[j];
				float next = ids[j + 1];
				labelInput.data[i * getMaxContextLen() + j] = current;
				label.data[i * getMaxContextLen() + j] = next;
				if (current == tokenizer.eos()) {
					labelLen.data[i] = j + 1;
				}
			}
		}
	}

    public void loadLabels(int[] indexs, Tensor label, String[] labels) {
        for (int i = 0; i < indexs.length; i++) {
            int idx = indexs[i];
            String text = datas.get(idx).get("text").toString();
            labels[i] = text;
            int[] ids = tokenizer.encodeInt(text, getMaxContextLen());
            for (int j = 0; j < getMaxContextLen(); j++) {
                if (j < ids.length) {
                    label.data[i * getMaxContextLen() + j] = ids[j];
                } else {
                    label.data[i * getMaxContextLen() + j] = 0;
                }
            }
        }
    }

    public int getMaxContextLen() {
        return maxContextLen;
    }

    public void setMaxContextLen(int maxContextLen) {
        this.maxContextLen = maxContextLen;
    }

    public int getMaxWavLength() {
        return maxWavLength;
    }

    public void setMaxWavLength(int maxWavLength) {
        this.maxWavLength = maxWavLength;
    }

    public int getNumBins() {
        return numBins;
    }
}
