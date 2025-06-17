package com.omega.example.yolo.utils;

import com.omega.common.utils.MathUtils;
import com.omega.engine.nn.data.DataSet;
import com.omega.engine.tensor.Tensor;
import com.omega.example.yolo.data.BaseDataLoader;

import java.io.File;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

/**
 * DataLoader
 *
 * @author Administrator
 */
public class YoloDataLoader extends BaseDataLoader {
    private String imgDirPath;
    private String[] idxSet;
    private Tensor imgSet;
    private Tensor labelSet;

    public YoloDataLoader(String imgDirPath, String labelPath, int batchSize, int channel, int height, int width, int labelSize, LabelType mode, boolean normalization) {
        this.batchSize = batchSize;
        this.imgDirPath = imgDirPath;
        this.labelSize = labelSize;
        switch (mode) {
            case text:
                loadDataIdx(imgDirPath, labelPath);
                break;
            case text_v3:
                loadDataIdxV3(imgDirPath, labelPath, 90);
                break;
            case csv:
                loadDataIdxFromCSV(imgDirPath, labelPath, channel, width, height, labelSize, normalization);
                break;
            case csv_v3:
                loadDataIdxFromCSV(imgDirPath, labelPath, channel, width, height, labelSize, 90, normalization);
                break;
            default:
                break;
        }
    }

    public DataSet getDataSet() {
        if (imgSet == null) {
            return new DataSet(number, 0, 0, 0, labelSet.getShape()[1], labelSize, null, labelSet.getData());
        }
        return new DataSet(number, imgSet.getShape()[1], imgSet.getShape()[2], imgSet.getShape()[3], labelSet.getShape()[1], labelSize, imgSet.getData(), labelSet.getData());
    }

    public void loadDataIdx(String imgDirPath, String labelPath) {
        try {
            File file = new File(imgDirPath);
            if (file.exists() && file.isDirectory()) {
                File[] files = file.listFiles();
                number = files.length;
                idxSet = new String[files.length];
                for (int i = 0; i < files.length; i++) {
                    File img = files[i];
                    String filename = img.getName();
                    idxSet[i] = filename;
                }
                orderByName();
                setLabelSet(new Tensor(files.length, 1, 1, labelSize));
                LabelUtils.loadLabel(labelPath, getLabelSet());
            }
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void loadDataIdxV3(String imgDirPath, String labelPath, int maxBox) {
        try {
            File file = new File(imgDirPath);
            if (file.exists() && file.isDirectory()) {
                File[] files = file.listFiles();
                number = files.length;
                idxSet = new String[files.length];
                for (int i = 0; i < files.length; i++) {
                    File img = files[i];
                    String filename = img.getName();
                    idxSet[i] = filename;
                }
                orderByName();
                this.labelSize = this.labelSize * maxBox;
                setLabelSet(new Tensor(files.length, 1, 1, labelSize));
                LabelUtils.loadLabel(labelPath, getLabelSet());
            }
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void loadDataIdxFromCSV(String imgDirPath, String label_csv_Path, int channel, int width, int height, int labelSize, boolean normalization) {
        try {
            File file = new File(imgDirPath);
            if (file.exists() && file.isDirectory()) {
                File[] files = file.listFiles();
                number = files.length;
                idxSet = new String[files.length];
                setLabelSet(new Tensor(files.length, 1, 1, labelSize));
                LabelUtils.loadLabelCSV(label_csv_Path, getLabelSet(), idxSet);
                setImgSet(new Tensor(files.length, channel, height, width));
                int index = 0;
                for (String filename : idxSet) {
                    String filePath = imgDirPath + "/" + filename;
                    YoloImageUtils.loadImgDataToTensor(filePath, getImgSet(), index, normalization);
                    index++;
                }
            }
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void loadDataIdxFromCSV(String imgDirPath, String label_csv_Path, int channel, int width, int height, int labelSize, int maxBox, boolean normalization) {
        try {
            File file = new File(imgDirPath);
            if (file.exists() && file.isDirectory()) {
                File[] files = file.listFiles();
                number = files.length;
                idxSet = new String[files.length];
                this.labelSize = this.labelSize * maxBox;
                setLabelSet(new Tensor(files.length, 1, 1, labelSize * maxBox));
                LabelUtils.loadLabelCSV(label_csv_Path, getLabelSet(), idxSet);
                setImgSet(new Tensor(files.length, channel, height, width));
                int index = 0;
                for (String filename : idxSet) {
                    String filePath = imgDirPath + "/" + filename;
                    YoloImageUtils.loadImgDataToTensor(filePath, getImgSet(), index, normalization);
                    index++;
                }
            }
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public void orderByName() {
        List<String> list = Arrays.asList(idxSet);
        Collections.sort(list, new Comparator<String>() {
            @Override
            public int compare(String o1, String o2) {
                return o1.compareTo(o2);
            }
        });
        idxSet = (String[]) list.toArray();
    }

    public int[][] shuffle() {
        return MathUtils.randomInts(this.number, this.batchSize);
    }

    @Override
    public void loadData(int[] indexs, Tensor input, Tensor label) {
        for (int i = 0; i < indexs.length; i++) {
            int index = indexs[i];
            String filePath = imgDirPath + "/" + idxSet[index];
            YoloImageUtils.loadImgDataToTensor(filePath, input, i);
            System.arraycopy(this.getLabelSet().getData(), index * label.getShape()[3], label.getData(), i * label.getShape()[3], label.getShape()[3]);
        }
    }

    public Tensor getImgSet() {
        return imgSet;
    }

    public void setImgSet(Tensor imgSet) {
        this.imgSet = imgSet;
    }

    public Tensor getLabelSet() {
        return labelSet;
    }

    public void setLabelSet(Tensor labelSet) {
        this.labelSet = labelSet;
    }

    @Override
    public void loadData(int pageIndex, int batchSize, Tensor input, Tensor label) {
        // TODO Auto-generated method stub
        for (int i = 0; i < batchSize; i++) {
            int index = pageIndex * batchSize + i;
            if ((pageIndex + 1) * batchSize > this.getLabelSet().getShape()[0]) {
                int offset = batchSize - (this.getLabelSet().getShape()[0] % batchSize);
                index = pageIndex * batchSize - offset + i;
            }
            String filePath = imgDirPath + "/" + idxSet[index];
            YoloImageUtils.loadImgDataToTensor(filePath, input, i);
            System.arraycopy(this.getLabelSet().getData(), index * label.getShape()[3], label.getData(), i * label.getShape()[3], label.getShape()[3]);
        }
    }

    public void loadData(int pageIndex, int batchSize, Tensor input) {
        // TODO Auto-generated method stub
        for (int i = 0; i < batchSize; i++) {
            int index = pageIndex * batchSize + i;
            if ((pageIndex + 1) * batchSize > this.getLabelSet().getShape()[0]) {
                int offset = batchSize - (this.getLabelSet().getShape()[0] % batchSize);
                index = pageIndex * batchSize - offset + i;
            }
            String filePath = imgDirPath + "/" + idxSet[index];
            YoloImageUtils.loadImgDataToTensor(filePath, input, i);
        }
    }

    @Override
    public float[] loadData(int index) {
        // TODO Auto-generated method stub
        String filePath = imgDirPath + "/" + idxSet[index];
        return YoloImageUtils.loadImgDataToArray(filePath);
    }

    @Override
    public Tensor initLabelTensor() {
        // TODO Auto-generated method stub
        return null;
    }

    @Override
    public void loadData(int[] indexs, Tensor input) {
        // TODO Auto-generated method stub
    }
}

