package com.omega.engine.nn.data;

import com.omega.common.utils.RandomUtils;
import com.omega.example.yolo.utils.YoloDataLoader;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class TrainDataLoader {
    private List<Integer> trainDataIndex;
    private int orgH;
    private int orgW;

    public TrainDataLoader(int dataSize, int orgH, int orgW) {
        trainDataIndex = new ArrayList<Integer>(dataSize);
        for (int i = 0; i < dataSize; i++) {
            trainDataIndex.add(i);
        }
    }

    public static void setPixel(float[] x, int ow, int oh, int w, int h, int c, float val) {
        x[c * oh * ow + h * ow + w] = val;
    }

    public static float getPixel(float[] x, int ow, int oh, int w, int h, int c) {
        return x[c * oh * ow + h * ow + w];
    }

    public void loadDataRegion(YoloDataLoader dataLoader, int n, int size, int classes, float jitter, float hue, float saturation, float exposure) {
        Collections.shuffle(trainDataIndex);
        int dw = (int) (orgW * jitter);
        int dh = (int) (orgH * jitter);
        int channel = dataLoader.getImgSet().getShape()[1];
        for (int b = 0; b < n; b++) {
            int index = trainDataIndex.get(b);
            float[] orig = dataLoader.getImgSet().getByNumber(index);
            int pleft = RandomUtils.randomInt(-dw, dw);
            int pright = RandomUtils.randomInt(-dw, dw);
            int ptop = RandomUtils.randomInt(-dh, dh);
            int pbot = RandomUtils.randomInt(-dh, dh);
            int swidth = orgW - pleft - pright;
            int sheight = orgH - ptop - pbot;
            float sx = (float) swidth / orgW;
            float sy = (float) sheight / orgH;
            /**
             * 抖动图片

             */
            float[] cropped = cropImage(orig, pleft, ptop, swidth, sheight, channel);
            float dx = ((float) pleft / orgW) / sx;
            float dy = ((float) ptop / orgH) / sy;
            /**
             * 调整图片大小

             */
            float[] resized = resizeImage(cropped, swidth, sheight, orgW, orgH, channel);
            /**
             * 翻转图片

             */
            boolean flip = false;
            if (Math.random() >= 0.5d) {
                flip = true;
                flipImage(resized, orgW, orgH, channel);
            }
            randomDistortImage(resized, orgW, orgH, channel, hue, saturation, exposure);
        }
    }

    public void fillTruthSwag() {
    }

    public void randomDistortImage(float[] img, int w, int h, int c, float hue, float saturation, float exposure) {
        float dhue = RandomUtils.randomFloat(-hue, hue);
        float dsat = RandomUtils.randomScale(saturation);
        float dexp = RandomUtils.randomScale(exposure);
        distortImage(img, w, h, c, dhue, dsat, dexp);
    }

    public void distortImage(float[] image, int w, int h, int c, float hue, float sat, float val) {
        if (c >= 3) {
            HsvUtils.rgb2hsv(image, c, w, h);
            scaleImageChannel(image, 1, w, h, sat);
            scaleImageChannel(image, 2, w, h, val);
            for (int i = 0; i < w * h; i++) {
                image[i] += hue;
                if (image[i] > 1)
                    image[i] -= 1;
                if (image[i] < 0)
                    image[i] += 1;
            }
            HsvUtils.hsv2rgb(image, c, w, h);
        } else {
            scaleImageChannel(image, 0, w, h, val);
        }
        constrainImage(image);
    }

    public void scaleImageChannel(float[] im, int c, int w, int h, float v) {
        int i, j;
        for (j = 0; j < h; ++j) {
            for (i = 0; i < w; ++i) {
                float pix = getPixel(im, w, h, i, j, c);
                pix = pix * v;
                setPixel(im, w, h, i, j, c, pix);
            }
        }
    }

    public void constrainImage(float[] img) {
        for (int i = 0; i < img.length; i++) {
            if (img[i] < 0) {
                img[i] = 0;
            } else if (img[i] > 1) {
                img[i] = 1;
            }
        }
    }

    public void flipImage(float[] img, int w, int h, int c) {
        int i, j, k;
        for (k = 0; k < c; ++k) {
            for (i = 0; i < h; ++i) {
                for (j = 0; j < w / 2; ++j) {
                    int index = j + w * (i + h * (k));
                    int flip = (w - j - 1) + w * (i + h * (k));
                    float swap = img[flip];
                    img[flip] = img[index];
                    img[index] = swap;
                }
            }
        }
    }

    public float[] resizeImage(float[] img, int imgW, int imgH, int w, int h, int c) {
        if (imgW == w && imgH == h) {
            return img;
        }
        float[] resized = new float[c * h * w];
        float[] part = new float[c * imgH * w];
        int row, col, k;
        float w_scale = (float) (imgW - 1) / (w - 1);
        float h_scale = (float) (imgH - 1) / (h - 1);
        for (k = 0; k < c; ++k) {
            for (row = 0; row < imgH; ++row) {
                for (col = 0; col < w; ++col) {
                    float val = 0;
                    if (col == w - 1 || imgW == 1) {
                        val = getPixel(img, imgH, imgW, imgW - 1, row, k);
                    } else {
                        float sx = col * w_scale;
                        int ix = (int) sx;
                        float dx = sx - ix;
                        val = (1 - dx) * getPixel(img, imgH, imgW, ix, row, k) + dx * getPixel(img, imgH, imgW, ix + 1, row, k);
                    }
                    part[k * imgH * w + row * w + col] = val;
                }
            }
        }
        for (k = 0; k < c; ++k) {
            for (row = 0; row < h; ++row) {
                float sy = row * h_scale;
                int iy = (int) sy;
                float dy = sy - iy;
                for (col = 0; col < w; ++col) {
                    float val = (1 - dy) * getPixel(part, imgH, w, col, iy, k);
                    setPixel(resized, w, h, col, row, k, val);
                }
                if (row == h - 1 || imgH == 1)
                    continue;
                for (col = 0; col < w; ++col) {
                    float val = dy * getPixel(part, imgH, w, col, iy + 1, k);
                    assert (col < w && row < h && k < c);
                    resized[k * h * w + row * w + col] += val;
                }
            }
        }
        return resized;
    }

    public float[] cropImage(float[] orig, int dx, int dy, int w, int h, int channel) {
        float[] cropped = new float[channel * h * w];
        for (int c = 0; c < channel; c++) {
            for (int j = 0; j < h; j++) {
                for (int i = 0; i < w; i++) {
                    int row = j + dy;
                    int col = i + dx;
                    float val = 0;
                    row = limit(row, 0, orgH - 1);
                    col = limit(col, 0, orgW - 1);
                    if (row >= 0 && row < orgH && col >= 0 && col < orgW) {
                        val = orig[c * orgH * orgW + row * orgW + col];
                    }
                    cropped[c * h * w + j * h + i] = val;
                }
            }
        }
        return cropped;
    }

    public int limit(int val, int min, int max) {
        if (val < min) {
            return 0;
        } else if (val > max) {
            return max;
        }
        return val;
    }
}

