package com.omega.common.data.utils;

import com.omega.common.data.Tensor;
import com.omega.common.utils.MatrixUtils;
import com.omega.common.utils.PrintUtils;
import com.omega.common.utils.RandomUtils;

import java.util.Arrays;

/**
 * 数据预处理工具
 *
 * @author Administrator
 */
public class DataTransforms {
    private int class_num = 1;
    private int bbox_size = 4;

    public DataTransforms(int class_num, int bbox_size) {
        this.class_num = class_num;
        this.bbox_size = bbox_size;
    }

    public static void randomCrop(Tensor input, Tensor output, int ch, int cw, int padding) {
        int maxHeight = input.height + padding * 2 - ch;
        int maxWidth = input.width + padding * 2 - cw;
        int channel = input.channel;
        int height = input.height;
        int width = input.width;
        for (int n = 0; n < input.number; n++) {
            int rh = (int) (Math.random() * (maxHeight + 1));
            int rw = (int) (Math.random() * (maxWidth + 1));
            for (int c = 0; c < channel; c++) {
                for (int h = 0; h < ch; h++) {
                    for (int w = 0; w < cw; w++) {
                        int oh = h + rh;
                        int ow = w + rw;
                        int toh = oh - padding;
                        int tow = ow - padding;
                        if (oh < padding || ow < padding || toh >= height || tow >= width) {
                            output.data[n * channel * ch * cw + c * ch * cw + h * cw + w] = 0.0f;
                        } else {
                            output.data[n * channel * ch * cw + c * ch * cw + h * cw + w] = input.data[n * channel * height * width + c * height * width + toh * width + tow];
                        }
                    }
                }
            }
        }
    }

    public static void randomCropWithLabel(int b, Tensor input, float[] img, float[] orglabel, float[] label, int labelSize, int ch, int cw, float jitter) {
        int ph = (int) (input.height * jitter);
        int pw = (int) (input.width * jitter);
        int maxHeight = input.height + ph * 2 - ch;
        int maxWidth = input.width + pw * 2 - cw;
        int channel = input.channel;
        int height = input.height;
        int width = input.width;
        int rh = (int) (Math.random() * (maxHeight + 1));
        int rw = (int) (Math.random() * (maxWidth + 1));
        input.getByNumber(b, img);
        if (jitter > 0) {
            for (int c = 0; c < channel; c++) {
                for (int h = 0; h < ch; h++) {
                    for (int w = 0; w < cw; w++) {
                        int oh = h + rh;
                        int ow = w + rw;
                        int toh = oh - ph;
                        int tow = ow - pw;
                        if (oh < ph || ow < pw || toh >= height || tow >= width) {
                            input.data[b * channel * ch * cw + c * ch * cw + h * cw + w] = 0.0f;
                        } else {
                            input.data[b * channel * ch * cw + c * ch * cw + h * cw + w] = img[c * height * width + toh * width + tow];
                        }
                    }
                }
            }
            for (int c = 0; c < labelSize; c++) {
                float x1 = orglabel[c * 5 + 1];
                float y1 = orglabel[c * 5 + 2];
                float x2 = orglabel[c * 5 + 3];
                float y2 = orglabel[c * 5 + 4];
                label[c * 5 + 0] = orglabel[c * 5 + 0];
                label[c * 5 + 1] = x1 + ph - rh;
                label[c * 5 + 2] = y1 + pw - rw;
                label[c * 5 + 3] = x2 + ph - rh;
                label[c * 5 + 4] = y2 + pw - rw;
                label[c * 5 + 1] = constrain(label[c * 5 + 1], 0, input.width - 1);
                label[c * 5 + 2] = constrain(label[c * 5 + 2], 0, input.height - 1);
                label[c * 5 + 3] = constrain(label[c * 5 + 3], 0, input.width - 1);
                label[c * 5 + 4] = constrain(label[c * 5 + 4], 0, input.height - 1);
            }
        } else {
            for (int c = 0; c < labelSize; c++) {
                float x1 = orglabel[c * 5 + 1];
                float y1 = orglabel[c * 5 + 2];
                float x2 = orglabel[c * 5 + 3];
                float y2 = orglabel[c * 5 + 4];
                label[c * 5 + 0] = orglabel[c * 5 + 0];
                label[c * 5 + 1] = x1;
                label[c * 5 + 2] = y1;
                label[c * 5 + 3] = x2;
                label[c * 5 + 4] = y2;
            }
        }
    }

    public static void randomHorizontalFilp(Tensor input, Tensor output) {
        int channel = input.channel;
        int height = input.height;
        int width = input.width;
        float[] org = new float[channel * height * width];
        for (int n = 0; n < input.number; n++) {
            if (Math.random() >= 0.5d) {
                input.getByNumber(n, org);
                for (int c = 0; c < input.channel; c++) {
                    for (int h = 0; h < input.height; h++) {
                        for (int w = 0; w < input.width / 2; w++) {
                            int ow = input.width - 1 - w;
                            output.data[n * channel * height * width + c * height * width + h * width + w] = org[c * height * width + h * width + ow];
                            output.data[n * channel * height * width + c * height * width + h * width + ow] = org[c * height * width + h * width + w];
                        }
                        if (input.width % 2 == 1) {
                            int w = width / 2;
                            output.data[n * channel * height * width + c * height * width + h * width + w] = org[c * height * width + h * width + w];
                        }
                    }
                }
            } else {
                for (int c = 0; c < input.channel; c++) {
                    for (int h = 0; h < input.height; h++) {
                        for (int w = 0; w < input.width; w++) {
                            output.data[n * channel * height * width + c * height * width + h * width + w] = input.data[n * channel * height * width + c * height * width + h * width + w];
                        }
                    }
                }
            }
        }
    }

    public static float[] randomHorizontalFilp(float[] input, float[] output, int channel, int height, int width) {
        float[] org = Arrays.copyOf(input, input.length);
        if (Math.random() >= 0.5d) {
            for (int c = 0; c < channel; c++) {
                for (int h = 0; h < height; h++) {
                    for (int w = 0; w < width / 2; w++) {
                        int ow = width - 1 - w;
                        output[c * height * width + h * width + w] = org[c * height * width + h * width + ow];
                        output[c * height * width + h * width + ow] = org[c * height * width + h * width + w];
                    }
                    if (width % 2 == 1) {
                        int w = width / 2;
                        output[c * height * width + h * width + w] = org[c * height * width + h * width + w];
                    }
                }
            }
        } else {
            output = org;
        }
        return output;
    }

    public static void randomHorizontalFilpWithLabel(Tensor input, float[] img, int b, float[] orgLabel, float[] label, int labelSize) {
        int channel = input.channel;
        int height = input.height;
        int width = input.width;
        if (Math.random() >= 0.5d) {
            input.getByNumber(b, img);
            for (int c = 0; c < input.channel; c++) {
                for (int h = 0; h < input.height; h++) {
                    for (int w = 0; w < input.width / 2; w++) {
                        int ow = input.width - 1 - w;
                        input.data[b * channel * height * width + c * height * width + h * width + w] = img[c * height * width + h * width + ow];
                        input.data[b * channel * height * width + c * height * width + h * width + ow] = img[c * height * width + h * width + w];
                    }
                    if (input.width % 2 == 1) {
                        int w = width / 2;
                        input.data[b * channel * height * width + c * height * width + h * width + w] = img[c * height * width + h * width + w];
                    }
                }
            }
            for (int c = 0; c < labelSize; c++) {
                float y1 = orgLabel[c * 5 + 2];
                float y2 = orgLabel[c * 5 + 4];
                float hl = y2 - y1;
                y1 = (height - y1) - hl;
                y2 = (height - y2) + hl;
                label[c * 5 + 2] = y1;
                label[c * 5 + 4] = y2;
                label[c * 5 + 2] = constrain(label[c * 5 + 2], 0, input.height - 1);
                label[c * 5 + 4] = constrain(label[c * 5 + 4], 0, input.height - 1);
            }
        }
    }

    public static void cutout(Tensor input, Tensor output, int len, int count) {
        int channel = input.channel;
        int height = input.height;
        int width = input.width;
        for (int n = 0; n < input.number; n++) {
            for (int num = 0; num < count; num++) {
                //				if(Math.random() >= 0.5d) {
                int y = RandomUtils.getInstance().nextInt(height);
                int x = RandomUtils.getInstance().nextInt(width);
                int y1 = (int) (y - Math.floor(len / 2));
                int y2 = (int) (y + Math.floor(len / 2));
                int x1 = (int) (x - Math.floor(len / 2));
                int x2 = (int) (x + Math.floor(len / 2));
                for (int c = 0; c < input.channel; c++) {
                    for (int h = 0; h < input.height; h++) {
                        for (int w = 0; w < input.width; w++) {
                            if (h >= y1 && h <= y2 && w >= x1 && w <= x2) {
                                output.data[n * channel * height * width + c * height * width + h * width + w] = 0.0f;
                            } else {
                                output.data[n * channel * height * width + c * height * width + h * width + w] = input.data[n * channel * height * width + c * height * width + h * width + w];
                            }
                        }
                    }
                }
                //				}
            }
        }
    }

    public static void normalize(Tensor input, Tensor output, float[] mean, float[] std) {
        int channel = input.channel;
        int height = input.height;
        int width = input.width;
        for (int i = 0; i < input.dataLength; i++) {
            int c = (i / height / width) % channel;
            output.data[i] = (input.data[i] - mean[c]) / std[c];
        }
    }

    public static void normalize(Tensor input, Tensor output, int color, float[] mean, float[] std) {
        int channel = input.channel;
        int height = input.height;
        int width = input.width;
        for (int i = 0; i < input.dataLength; i++) {
            int c = (i / height / width) % channel;
            output.data[i] = (input.data[i] / color - mean[c]) / std[c];
        }
    }

    public static void normalize(Tensor input, Tensor output, int color) {
        int channel = input.channel;
        int height = input.height;
        int width = input.width;
        for (int i = 0; i < input.dataLength; i++) {
            int c = (i / height / width) % channel;
            output.data[i] = input.data[i] / color;
        }
    }

    public static void randomCrop(Tensor input, int ch, int cw, int padding) {
        int maxHeight = input.height + padding * 2 - ch;
        int maxWidth = input.width + padding * 2 - cw;
        float[] out = new float[input.number * input.channel * ch * cw];
        int channel = input.channel;
        int height = input.height;
        int width = input.width;
        for (int n = 0; n < input.number; n++) {
            int rh = (int) (Math.random() * (maxHeight + 1));
            int rw = (int) (Math.random() * (maxWidth + 1));
            for (int c = 0; c < channel; c++) {
                for (int h = 0; h < ch; h++) {
                    for (int w = 0; w < cw; w++) {
                        int oh = h + rh;
                        int ow = w + rw;
                        int toh = oh - padding;
                        int tow = ow - padding;
                        if (oh < padding || ow < padding || toh >= height || tow >= width) {
                            //							out[n * channel * ch * cw + c * ch * cw + h * cw + w] = 0.0f;
                        } else {
                            out[n * channel * ch * cw + c * ch * cw + h * cw + w] = input.data[n * channel * height * width + c * height * width + toh * width + tow];
                        }
                    }
                }
            }
        }
        input.data = out;
    }

    public static void randomHorizontalFilp(Tensor input) {
        float[] out = new float[input.dataLength];
        int channel = input.channel;
        int height = input.height;
        int width = input.width;
        for (int n = 0; n < input.number; n++) {
            if (Math.random() >= 0.5d) {
                for (int c = 0; c < input.channel; c++) {
                    for (int h = 0; h < input.height; h++) {
                        for (int w = 0; w < input.width / 2; w++) {
                            int ow = input.width - 1 - w;
                            out[n * channel * height * width + c * height * width + h * width + w] = input.data[n * channel * height * width + c * height * width + h * width + ow];
                            out[n * channel * height * width + c * height * width + h * width + ow] = input.data[n * channel * height * width + c * height * width + h * width + w];
                        }
                        if (input.width % 2 == 1) {
                            int w = width / 2;
                            out[n * channel * height * width + c * height * width + h * width + w] = input.data[n * channel * height * width + c * height * width + h * width + w];
                        }
                    }
                }
            } else {
                for (int c = 0; c < input.channel; c++) {
                    for (int h = 0; h < input.height; h++) {
                        for (int w = 0; w < input.width; w++) {
                            out[n * channel * height * width + c * height * width + h * width + w] = input.data[n * channel * height * width + c * height * width + h * width + w];
                        }
                    }
                }
            }
        }
        input.data = out;
    }

    public static void cutout(Tensor input, int len) {
        float[] out = new float[input.dataLength];
        int channel = input.channel;
        int height = input.height;
        int width = input.width;
        for (int n = 0; n < input.number; n++) {
            int y = RandomUtils.getInstance().nextInt(height);
            int x = RandomUtils.getInstance().nextInt(width);
            int y1 = (int) (y - Math.floor(len / 2));
            int y2 = (int) (y + Math.floor(len / 2));
            int x1 = (int) (x - Math.floor(len / 2));
            int x2 = (int) (x + Math.floor(len / 2));
            //			System.out.println(y1+","+y2+","+x1+","+x2);
            for (int c = 0; c < input.channel; c++) {
                for (int h = 0; h < input.height; h++) {
                    for (int w = 0; w < input.width; w++) {
                        if (h >= y1 && h <= y2 && w >= x1 && w <= x2) {
                            out[n * channel * height * width + c * height * width + h * width + w] = 0.0f;
                        } else {
                            out[n * channel * height * width + c * height * width + h * width + w] = input.data[n * channel * height * width + c * height * width + h * width + w];
                        }
                    }
                }
            }
        }
        input.data = out;
    }

    public static void normalize(Tensor input, float[] mean, float[] std) {
        int channel = input.channel;
        int height = input.height;
        int width = input.width;
        for (int i = 0; i < input.dataLength; i++) {
            int c = (i / height / width) % channel;
            input.data[i] = (input.data[i] - mean[c]) / std[c];
        }
    }

    public static float constrain(float a, float min, float max) {
        if (a < min)
            return min;
        if (a > max)
            return max;
        return a;
    }

    public static void main(String[] args) {
        float[][] x = new float[][]{{0, 0, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 1, 1, 1, 1, 0, 0, 0}, {0, 0, 0, 0, 0, 1, 0, 0, 0}, {0, 0, 0, 0, 1, 0, 0, 0, 0}, {0, 0, 0, 1, 0, 0, 0, 0, 0}, {0, 0, 0, 1, 0, 0, 0, 0, 0}, {0, 0, 1, 0, 0, 0, 0, 0, 0}, {0, 1, 1, 1, 1, 1, 1, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0, 0}};
        float[][] x2 = new float[][]{{0, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 1, 1, 1, 1, 0, 0}, {0, 0, 0, 0, 0, 1, 0, 0}, {0, 0, 0, 0, 1, 0, 0, 0}, {0, 0, 0, 1, 0, 0, 0, 0}, {0, 0, 0, 1, 0, 0, 0, 0}, {0, 0, 0, 1, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0}};
        float[] x1 = MatrixUtils.transform(x);
        Tensor input = new Tensor(1, 1, 9, 9, x1);
        Tensor output = new Tensor(1, 1, 9, 9);
        //		DataTransforms.randomHorizontalFilp(input);
        //
        //		float[][] out1 = MatrixUtils.transform(input.data, 9, 9);
        //
        //		PrintUtils.printImage(out1);
        DataTransforms.randomCrop(input, 9, 9, 2);
        float[][] out2 = MatrixUtils.transform(input.data, 9, 9);
        PrintUtils.printImage(out2);
        //		DataTransforms.cutout(input, output, 4);
        //		float[][] out3 = MatrixUtils.transform(output.data, 9, 9);
        //		PrintUtils.printImage(out3);
    }

    public void randomCropWithLabel(Tensor input, Tensor label, int ch, int cw, int padding) {
        int maxHeight = input.height + padding * 2 - ch;
        int maxWidth = input.width + padding * 2 - cw;
        int channel = input.channel;
        int height = input.height;
        int width = input.width;
        int labelOnceLength = class_num + bbox_size;
        float[] org = new float[channel * height * width];
        for (int n = 0; n < input.number; n++) {
            int rh = (int) (Math.random() * (maxHeight + 1));
            int rw = (int) (Math.random() * (maxWidth + 1));
            input.getByNumber(n, org);
            for (int c = 0; c < channel; c++) {
                for (int h = 0; h < ch; h++) {
                    for (int w = 0; w < cw; w++) {
                        int oh = h + rh;
                        int ow = w + rw;
                        int toh = oh - padding;
                        int tow = ow - padding;
                        if (oh < padding || ow < padding || toh >= height || tow >= width) {
                            input.data[n * channel * ch * cw + c * ch * cw + h * cw + w] = 0.0f;
                        } else {
                            input.data[n * channel * ch * cw + c * ch * cw + h * cw + w] = org[c * height * width + toh * width + tow];
                        }
                    }
                }
            }
            for (int c = 0; c < label.channel; c++) {
                float x1 = label.data[n * label.channel * labelOnceLength + c * labelOnceLength + class_num + 0];
                float y1 = label.data[n * label.channel * labelOnceLength + c * labelOnceLength + class_num + 1];
                float x2 = label.data[n * label.channel * labelOnceLength + c * labelOnceLength + class_num + 2];
                float y2 = label.data[n * label.channel * labelOnceLength + c * labelOnceLength + class_num + 3];
                label.data[n * label.channel * labelOnceLength + c * labelOnceLength + class_num + 0] = x1 + padding - rh;
                label.data[n * label.channel * labelOnceLength + c * labelOnceLength + class_num + 1] = y1 + padding - rw;
                label.data[n * label.channel * labelOnceLength + c * labelOnceLength + class_num + 2] = x2 + padding - rh;
                label.data[n * label.channel * labelOnceLength + c * labelOnceLength + class_num + 3] = y2 + padding - rw;
            }
        }
    }

    public void randomCropWithLabel(Tensor input, Tensor label, int ch, int cw, float jitter) {
        int ph = (int) (input.height * jitter);
        int pw = (int) (input.width * jitter);
        int maxHeight = input.height + ph * 2 - ch;
        int maxWidth = input.width + pw * 2 - cw;
        int channel = input.channel;
        int height = input.height;
        int width = input.width;
        int labelOnceLength = class_num + bbox_size;
        float[] org = new float[channel * height * width];
        for (int n = 0; n < input.number; n++) {
            int rh = (int) (Math.random() * (maxHeight + 1));
            int rw = (int) (Math.random() * (maxWidth + 1));
            input.getByNumber(n, org);
            for (int c = 0; c < channel; c++) {
                for (int h = 0; h < ch; h++) {
                    for (int w = 0; w < cw; w++) {
                        int oh = h + rh;
                        int ow = w + rw;
                        int toh = oh - ph;
                        int tow = ow - pw;
                        if (oh < ph || ow < pw || toh >= height || tow >= width) {
                            input.data[n * channel * ch * cw + c * ch * cw + h * cw + w] = 0.0f;
                        } else {
                            input.data[n * channel * ch * cw + c * ch * cw + h * cw + w] = org[c * height * width + toh * width + tow];
                        }
                    }
                }
            }
            for (int c = 0; c < label.channel; c++) {
                float x1 = label.data[n * label.channel * labelOnceLength + c * labelOnceLength + class_num + 0];
                float y1 = label.data[n * label.channel * labelOnceLength + c * labelOnceLength + class_num + 1];
                float x2 = label.data[n * label.channel * labelOnceLength + c * labelOnceLength + class_num + 2];
                float y2 = label.data[n * label.channel * labelOnceLength + c * labelOnceLength + class_num + 3];
                label.data[n * label.channel * labelOnceLength + c * labelOnceLength + class_num + 0] = x1 + ph - rh;
                label.data[n * label.channel * labelOnceLength + c * labelOnceLength + class_num + 1] = y1 + pw - rw;
                label.data[n * label.channel * labelOnceLength + c * labelOnceLength + class_num + 2] = x2 + ph - rh;
                label.data[n * label.channel * labelOnceLength + c * labelOnceLength + class_num + 3] = y2 + pw - rw;
            }
        }
    }

    public void randomHorizontalFilpWithLabel(Tensor input, Tensor output, Tensor label) {
        int channel = input.channel;
        int height = input.height;
        int width = input.width;
        int labelOnceLength = class_num + bbox_size;
        float[] org = new float[channel * height * width];
        for (int n = 0; n < input.number; n++) {
            if (Math.random() >= 0.5d) {
                input.getByNumber(n, org);
                for (int c = 0; c < input.channel; c++) {
                    for (int h = 0; h < input.height; h++) {
                        for (int w = 0; w < input.width / 2; w++) {
                            int ow = input.width - 1 - w;
                            output.data[n * channel * height * width + c * height * width + h * width + w] = org[c * height * width + h * width + ow];
                            output.data[n * channel * height * width + c * height * width + h * width + ow] = org[c * height * width + h * width + w];
                        }
                        if (input.width % 2 == 1) {
                            int w = width / 2;
                            output.data[n * channel * height * width + c * height * width + h * width + w] = org[c * height * width + h * width + w];
                        }
                    }
                }
                for (int c = 0; c < label.channel; c++) {
                    float y1 = label.data[n * label.channel * labelOnceLength + c * labelOnceLength + class_num + 1];
                    float y2 = label.data[n * label.channel * labelOnceLength + c * labelOnceLength + class_num + 3];
                    float hl = y2 - y1;
                    y1 = (height - y1) - hl;
                    y2 = (height - y2) + hl;
                    label.data[n * label.channel * labelOnceLength + c * labelOnceLength + class_num + 1] = y1;
                    label.data[n * label.channel * labelOnceLength + c * labelOnceLength + class_num + 3] = y2;
                }
            } else {
                for (int c = 0; c < input.channel; c++) {
                    for (int h = 0; h < input.height; h++) {
                        for (int w = 0; w < input.width; w++) {
                            output.data[n * channel * height * width + c * height * width + h * width + w] = input.data[n * channel * height * width + c * height * width + h * width + w];
                        }
                    }
                }
            }
        }
    }
}

