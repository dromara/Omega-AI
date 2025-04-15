package com.omega.example.transformer.dataset;

import com.omega.common.config.Tensor;
import com.omega.common.utils.MatrixOperation;
import com.omega.example.transformer.utils.LagJsonReader;
import com.omega.example.transformer.utils.tokenizers.Tokenizer;
import com.omega.example.yolo.data.ImageLoader;
import jcuda.runtime.JCuda;

import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.stream.Collectors;

public class LVMPreTrainDataset {
    public final static String image_special_token = "<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>";
    public final static float[] mean = new float[]{0.48145466f, 0.4578275f, 0.40821073f};
    public final static float[] std = new float[]{0.26862954f, 0.26130258f, 0.27577711f};
    public boolean shuffle = false;
    public int number = 0;
    public int channel = 3;
    public int imageSize = 224;
    public int onceImgSize = 0;
    public int count_it = 0;
    public Tokenizer tokenizer;
    public int max_len = 256;
    public int vocab_size;
    public String[] vocab;
    public Tensor testInput;
    public int[] image_special_idx;
    private int batchSize = 1;
    private String dataPath;
    private String imagePath;
    private String imagePrefix = "";
    private List<Map<String, Object>> dataList;
    private CompletableFuture<Boolean> cf;
    private int current = 0;
    private float[] tmpInput;
    private float[] tmpLabel;
    private float[] tmpImageInput;
    private float[] tmpIndice;

    public LVMPreTrainDataset(String dataPath, String imagePath, String imagePrefix, int imageSize, int max_len, int batchSize, boolean shuffle, Tokenizer tokenizer) {
        this.shuffle = shuffle;
        this.dataPath = dataPath;
        this.imagePath = imagePath;
        this.imagePrefix = imagePrefix;
        this.imageSize = imageSize;
        this.onceImgSize = 3 * this.imageSize * this.imageSize;
        this.max_len = max_len;
        this.batchSize = batchSize;
        this.tokenizer = tokenizer;
        this.vocab_size = tokenizer.voc_size();
        this.number = loadCount();
        this.count_it = this.number / batchSize;
        image_special_idx = tokenizer.encodeInt(image_special_token);
        System.out.println("dataCount:" + this.number);
        System.out.println("vocab_size:" + this.vocab_size);
        System.out.println("count_it:" + this.count_it);
    }

    public static int findImageSpecialIdx(int[] tokens, int[] specials) {
        String tokenStr = Arrays.stream(tokens).mapToObj(String::valueOf).collect(Collectors.joining(""));
        String specialStr = Arrays.stream(specials).mapToObj(String::valueOf).collect(Collectors.joining(""));
        return tokenStr.indexOf(specialStr);
    }

    public static Tensor getPositions(int b, int time) {
        float[] data = new float[b * time];
        for (int n = 0; n < b; n++) {
            for (int t = 0; t < time; t++) {
                data[n * time + t] = t;
            }
        }
        Tensor positions = new Tensor(b * time, 1, 1, 1, data, true);
        return positions;
    }

    public static Tensor getPositions(int b, int c, int time) {
        float[] data = new float[b * c * time];
        for (int n = 0; n < b * c; n++) {
            int pt = n % c;
            for (int t = 0; t < time; t++) {
                if (pt == t) {
                    data[n * time + t] = 1;
                }
            }
        }
        Tensor positions = new Tensor(b * c, 1, 1, time, data, true);
        return positions;
    }

    public static void getPositions(int b, int c, int time, Tensor positions) {
        positions = Tensor.createTensor(positions, b * time, 1, 1, time, true);
        for (int n = 0; n < b * c; n++) {
            int pt = n % b;
            for (int t = 0; t < time; t++) {
                if (pt == t) {
                    positions.data[n * time + t] = 1;
                }
            }
        }
        positions.hostToDevice();
    }

    public static void getPositions(int b, int time, Tensor positions) {
        positions = Tensor.createTensor(positions, b * time, 1, 1, 1, true);
        for (int n = 0; n < b; n++) {
            for (int t = 0; t < time; t++) {
                positions.data[n * time + t] = t;
            }
        }
        positions.hostToDevice();
    }

    public static Tensor triu(int b, int h, int size1, int size2, float val) {
        float[] data = new float[b * h * size1 * size2];
        for (int n = 0; n < b; n++) {
            for (int hn = 0; hn < h; hn++) {
                for (int i = 0; i < size1; i++) {
                    for (int j = 0; j < size2; j++) {
                        if (i < j) {
                            data[n * h * size1 * size2 + hn * size1 * size2 + i * size1 + j] = val;
                        }
                    }
                }
            }
        }
        Tensor mask = new Tensor(b, h, size1, size2, data, true);
        return mask;
    }

    public static void triu(int b, int h, int size1, int size2, float val, Tensor mask) {
        mask = Tensor.createTensor(mask, b, h, size1, size2, true);
        for (int n = 0; n < b; n++) {
            for (int hn = 0; hn < h; hn++) {
                for (int i = 0; i < size1; i++) {
                    for (int j = 0; j < size2; j++) {
                        if (i < j) {
                            mask.data[n * h * size1 * size2 + hn * size1 * size2 + i * size1 + j] = val;
                        }
                    }
                }
            }
        }
        mask.hostToDevice();
    }

    public static void triu(float val, int[] targetLens, Tensor mask) {
        for (int n = 0; n < mask.number; n++) {
            for (int hn = 0; hn < mask.channel; hn++) {
                for (int i = 0; i < mask.height; i++) {
                    for (int j = 0; j < mask.width; j++) {
                        //						System.out.println(i+":"+targetLens[n]);
                        if (i < targetLens[n]) {
                            if (i < j) {
                                //								System.out.println(i+":"+j);
                                mask.data[n * mask.channel * mask.height * mask.width + hn * mask.height * mask.width + i * mask.height + j] = val;
                            }
                        } else {
                            mask.data[n * mask.channel * mask.height * mask.width + hn * mask.height * mask.width + i * mask.height + j] = val;
                        }
                    }
                }
            }
        }
        mask.hostToDevice();
    }

    public int loadCount() {
        try {
            this.dataList = LagJsonReader.readJsonDataSamll(dataPath);
            number = dataList.size();
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
        return number;
    }

    public void initReader() {
        current = 1;
    }

    public void loadData(Tensor input, Tensor label, Tensor imageInput, Tensor indice) {
        try {
            //			System.out.println(it);
            if (tmpInput == null) {
                tmpInput = new float[input.dataLength];
                tmpLabel = new float[label.dataLength];
                tmpImageInput = new float[imageInput.dataLength];
                tmpIndice = new float[indice.dataLength];
            }
            if (cf != null) {
                boolean success = cf.get();
                if (success) {
                    //					System.err.println(it+"/"+count_it+":"+success);
                    //					System.out.println(JsonUtils.toJson(input.data));
                    input.hostToDevice(tmpInput);
                    label.hostToDevice(tmpLabel);
                    imageInput.hostToDevice(tmpImageInput);
                    indice.hostToDevice(tmpIndice);
                    System.arraycopy(tmpLabel, 0, label.data, 0, tmpLabel.length);
                    JCuda.cudaDeviceSynchronize();
                    //					System.out.println(JsonUtils.toJson(tmpLabel));
                }
                cf = loadAsyncData(tmpInput, tmpLabel, tmpImageInput, tmpIndice);
            } else {
                cf = loadAsyncData(tmpInput, tmpLabel, tmpImageInput, tmpIndice);
            }
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public String[] readQA(int b) throws IOException {
        Map<String, Object> item = dataList.get(current);
        if (current >= number - 1) {
            if (shuffle) {
                Collections.shuffle(dataList);
            }
            current = 0;
        }
        List<Map<String, String>> conversations = (List<Map<String, String>>) item.get("conversations");
        if (conversations.size() < 2) {
            return readQA(b);
        }
        String q = conversations.get(0).get("value").replaceAll("<image>", image_special_token);
        String a = conversations.get(1).get("value");
        String[] qa = new String[]{q, a};
        String imgPath = imagePath + imagePrefix + item.get("image").toString();
        //		long start = System.nanoTime();
        //		System.err.println(imgPath);
        imageProcessor(imgPath, b);
        //		System.out.println((System.nanoTime() - start)/1e6);
        current++;
        return qa;
    }

    public void imageProcessor(String imgPath, int b) {
        ImageLoader.loadImage(tmpImageInput, onceImgSize, b, imgPath, imageSize, imageSize, mean, std, true);
    }

    public CompletableFuture<Boolean> loadAsyncData(float[] input, float[] label, float[] imageData, float[] indice) {
        CompletableFuture<Boolean> cf = CompletableFuture.supplyAsync(() -> {
            try {
                for (int b = 0; b < batchSize; b++) {
                    String[] qa = readQA(b);
                    String q = qa[0];
                    String a = qa[1];
                    String qStr = tokenizer.sos_str() + "user\n" + q + tokenizer.eos_str() + "\n" + tokenizer.sos_str() + "assistant\n";
                    String qaStr = tokenizer.sos_str() + "user\n" + q + tokenizer.eos_str() + "\n" + tokenizer.sos_str() + "assistant\n" + a + tokenizer.eos_str();
                    int qLen = tokenizer.encodeInt(qStr).length;
                    int[] onceToken = tokenizer.encodeInt(qaStr);
                    indice[b] = findImageSpecialIdx(onceToken, image_special_idx);
                    for (int t = 0; t < max_len; t++) {
                        formatToIdxPad(b, t, onceToken, qLen, input, label);
                    }
                }
            } catch (Exception e) {
                // TODO: handle exception
                e.printStackTrace();
            }
            return true;
        });
        return cf;
    }

    public String decode(Tensor output) {
        int[] tokens = new int[output.number];
        for (int t = 0; t < output.number; t++) {
            int predictIndex = MatrixOperation.maxIndex(output.getByNumber(t));
            tokens[t] = predictIndex;
        }
        return tokenizer.decode(tokens);
    }

    public Tensor loadByTxtToIdx(String txt) {
        int[] idx = tokenizer.encodeInt(txt);
        testInput = Tensor.createTensor(testInput, txt.length(), 1, 1, 1, true);
        for (int t = 0; t < txt.length(); t++) {
            testInput.data[t] = idx[t];
        }
        testInput.hostToDevice();
        return testInput;
    }

    public Tensor loadByTxtToIdx(int[] idxs) {
        //		System.out.println(idxs.length);
        testInput = Tensor.createTensor(testInput, idxs.length, 1, 1, 1, true);
        for (int t = 0; t < idxs.length; t++) {
            testInput.data[t] = idxs[t];
        }
        testInput.hostToDevice();
        return testInput;
    }

    public Tensor loadByTxtToIdx(String txt, int maxLen) {
        int[] idx = tokenizer.encodeInt(txt);
        testInput = Tensor.createTensor(testInput, maxLen, 1, 1, 1, true);
        for (int t = 0; t < idx.length; t++) {
            testInput.data[t] = idx[t];
        }
        testInput.hostToDevice();
        return testInput;
    }

    public Tensor loadByTxtToIdx(int[] idxs, int maxLen) {
        if (testInput != null) {
            testInput.clear();
            testInput.clearGPU();
        }
        testInput = Tensor.createTensor(testInput, maxLen, 1, 1, 1, true);
        for (int t = 0; t < idxs.length; t++) {
            testInput.data[t] = idxs[t];
        }
        testInput.hostToDevice();
        return testInput;
    }

    public void formatToIdxPad(int b, int t, int[] onceToken, float[] input, float[] label) {
        if (t < onceToken.length - 2) {
            int curr = onceToken[t];
            int next = onceToken[t + 1];
            input[b * max_len + t] = curr;
            label[b * max_len + t] = next;
        } else if (t == onceToken.length - 2) {
            int curr = onceToken[t];
            int next = onceToken[t + 1];
            input[b * max_len + t] = curr;
            label[b * max_len + t] = next;
        } else {
            input[b * max_len + t] = tokenizer.pad();
            label[b * max_len + t] = tokenizer.pad();
        }
    }

    public void formatToIdxPad(int b, int t, int[] onceToken, int qLen, float[] input, float[] label) {
        if (t < onceToken.length - 2) {
            int curr = onceToken[t];
            int next = onceToken[t + 1];
            if (t < qLen - 1) {
                next = tokenizer.pad();
            }
            input[b * max_len + t] = curr;
            label[b * max_len + t] = next;
        } else if (t == onceToken.length - 2) {
            int curr = onceToken[t];
            int next = onceToken[t + 1];
            if (t < qLen - 1) {
                next = tokenizer.pad();
            }
            input[b * max_len + t] = curr;
            label[b * max_len + t] = next;
        } else {
            input[b * max_len + t] = tokenizer.pad();
            label[b * max_len + t] = tokenizer.pad();
        }
    }
}

