package com.omega.example.resnet.data;

import com.omega.common.utils.MathUtils;
import com.omega.engine.tensor.Tensor;
import com.omega.example.yolo.data.BaseDataLoader;
import com.omega.example.yolo.utils.YoloImageUtils;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * VailCodeDataLoader
 *
 * @author Administrator
 */
public class VailCodeDataLoader extends BaseDataLoader {
    private String imgDirPath;
    private String labelPath;
    private int classNum;
    private int labelClassLength;
    private Tensor labelData;
    private Map<String, String> labelOrgData = new LinkedHashMap<String, String>();
    private List<String> keyList = new ArrayList<String>();

    public VailCodeDataLoader(String imgDirPath, String labelPath, int batchSize, String[] labelSet, int classNum, int labelClassLength) {
        this.imgDirPath = imgDirPath;
        this.labelPath = labelPath;
        this.labelSet = labelSet;
        this.classNum = classNum;
        this.labelClassLength = labelClassLength;
        this.batchSize = batchSize;
        this.init();
    }

    public Tensor initLabelTensor() {
        return new Tensor(batchSize, 1, 1, labelClassLength * classNum, true);
    }

    public void init() {
        this.loadLabelDataForTXT();
        loadLabel();
    }

    public void loadLabel() {
        this.labelData = new Tensor(this.number, 1, 1, labelClassLength * classNum);
        for (int i = 0; i < keyList.size(); i++) {
            String filename = keyList.get(i);
            String labelStr = labelOrgData.get(filename);
            char[] labels = labelStr.toCharArray();
            for (int c = 0; c < labelClassLength; c++) {
                char once = labels[c];
                int index = getLabelIndex(once);
                //				System.out.println(once+":"+index);
                int labelIndex = i * this.labelData.getOnceSize() + c * classNum + index;
                this.labelData.getData()[labelIndex] = 1.0f;
            }
        }
    }

    public void loadLabelDataForTXT() {
        try (FileInputStream fin = new FileInputStream(this.labelPath); InputStreamReader reader = new InputStreamReader(fin); BufferedReader buffReader = new BufferedReader(reader);) {
            String strTmp = "";
            int idx = 0;
            while ((strTmp = buffReader.readLine()) != null) {
                String[] list = strTmp.split(" ");
                labelOrgData.put(list[0], list[1]);
                keyList.add(list[0]);
                idx++;
            }
            this.number = idx;
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void loadLabelToData(int[] indexs, Tensor label) {
        for (int b = 0; b < indexs.length; b++) {
            int index = indexs[b];
            System.arraycopy(this.labelData.getData(), index * label.getShape()[3], label.getData(), b * label.getShape()[3], label.getShape()[3]);
        }
    }

    public void loadInputToData(int[] indexs, Tensor input) {
        VailCodeFileDataLoader.load(imgDirPath, keyList, indexs, batchSize, input, false);
    }

    public int getLabelIndex(char once) {
        for (int i = 0; i < labelSet.length; i++) {
            String label = labelSet[i];
            if (label.equals(once + "")) {
                return i;
            }
        }
        return 0;
    }

    @Override
    public int[][] shuffle() {
        // TODO Auto-generated method stub
        return MathUtils.randomInts(this.number, this.batchSize);
    }

    @Override
    public void loadData(int[] indexs, Tensor input, Tensor label) {
        // TODO Auto-generated method stub
        /**
         * 加载label数据

         */
        loadLabelToData(indexs, label);
        /**
         * 加载input数据

         */
        loadInputToData(indexs, input);
        input.hostToDevice();
        label.hostToDevice();
    }

    @Override
    public void loadData(int pageIndex, int batchSize, Tensor input, Tensor label) {
        // TODO Auto-generated method stub
        for (int i = 0; i < batchSize; i++) {
            int index = pageIndex * batchSize + i;
            if ((pageIndex + 1) * batchSize > this.labelData.getShape()[0]) {
                int offset = batchSize - (this.labelData.getShape()[0] % batchSize);
                index = pageIndex * batchSize - offset + i;
            }
            String filePath = imgDirPath + "/" + keyList.get(index);
            YoloImageUtils.loadImgDataToGrayTensor(filePath, input, i);
            System.arraycopy(this.labelData.getData(), index * label.getShape()[3], label.getData(), i * label.getShape()[3], label.getShape()[3]);
        }
    }

    @Override
    public float[] loadData(int index) {
        // TODO Auto-generated method stub
        String filePath = imgDirPath + "/" + keyList.get(index);
        return YoloImageUtils.loadImgDataToArray(filePath);
    }

    @Override
    public void loadData(int[] indexs, Tensor input) {
        // TODO Auto-generated method stub
    }
}

