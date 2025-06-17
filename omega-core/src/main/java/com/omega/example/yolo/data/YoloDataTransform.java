package com.omega.example.yolo.data;

import com.omega.common.utils.ImageUtils;
import com.omega.common.utils.JsonUtils;
import com.omega.common.utils.MatrixOperation;
import com.omega.common.utils.RandomUtils;
import com.omega.engine.tensor.Tensor;
import com.omega.example.yolo.utils.OMImage;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * @author Administrator
 */
public class YoloDataTransform extends DataTransform {
    private float jitter = 0.2f;
    private float hue = 0.1f;
    private float saturation = 0.75f;
    private float exposure = 0.75f;
    private int classnum = 1;
    private int numBoxes = 7;
    private DataType dataType;

    public YoloDataTransform(int classnum, DataType dataType) {
        this.classnum = classnum;
        this.dataType = dataType;
    }

    public static List<float[]> fillTruthRegion(int index, float[] orgLabelData, int classnum, int numBoxes, int flip, float dx, float dy, float sx, float sy, int imgw, int imgh) {
        /**
         * 根据偏移量调整box数据

         */
        return correctBoxes(orgLabelData, dx, dy, sx, sy, flip, imgw, imgh);
    }

    public static void loadLabelToYolov1(float[] box, int b, Tensor label, int imgw, int imgh, int classnum, int stride) {
        int once = (5 + classnum);
        int oneSize = stride * stride * once;
        for (int n = 0; n < box.length / 5; n++) {
            int clazz = (int) box[n * 5 + 0] + 1;
            float x1 = box[n * 5 + 1] / imgw;
            float y1 = box[n * 5 + 2] / imgh;
            float x2 = box[n * 5 + 3] / imgw;
            float y2 = box[n * 5 + 4] / imgh;
            float cx = (x1 + x2) / 2;
            float cy = (y1 + y2) / 2;
            float w = (x2 - x1);
            float h = (y2 - y1);
            int gridx = (int) (cx * stride);
            int gridy = (int) (cy * stride);
            float px = cx * stride - gridx;
            float py = cy * stride - gridy;
            /**
             * c1

             */
            label.getData()[b * oneSize + gridy * stride * once + gridx * once + 0] = 1.0f;
            /**
             * class

             */
            label.getData()[b * oneSize + gridy * stride * once + gridx * once + clazz] = 1.0f;
            /**
             * px,py,w,h

             */
            label.getData()[b * oneSize + gridy * stride * once + gridx * once + classnum + 1] = px;
            label.getData()[b * oneSize + gridy * stride * once + gridx * once + classnum + 2] = py;
            label.getData()[b * oneSize + gridy * stride * once + gridx * once + classnum + 3] = w;
            label.getData()[b * oneSize + gridy * stride * once + gridx * once + classnum + 4] = h;
        }
    }

    public static void loadLabelToYolov1(List<float[]> list, int b, Tensor label, int img_w, int img_h, int classnum, int numBoxes) {
        int once = (5 + classnum);
        int oneSize = numBoxes * numBoxes * once;
        for (int n = 0; n < list.size(); n++) {
            float[] box = list.get(n);
            int clazz = (int) box[0] + 1;
            float cx = box[1];
            float cy = box[2];
            float w = box[3];
            float h = box[4];
            int gridx = (int) (cx * numBoxes);
            int gridy = (int) (cy * numBoxes);
            float px = cx * numBoxes - gridx;
            float py = cy * numBoxes - gridy;
            //			System.out.println("cx:"+cx+",cy:"+cy+"["+gridx+":"+gridy+"]");
            /**
             * c1

             */
            label.getData()[b * oneSize + gridy * numBoxes * once + gridx * once + 0] = 1.0f;
            /**
             * class

             */
            label.getData()[b * oneSize + gridy * numBoxes * once + gridx * once + clazz] = 1.0f;
            /**
             * x1,y1,w1,h1

             */
            label.getData()[b * oneSize + gridy * numBoxes * once + gridx * once + classnum + 1] = px;
            label.getData()[b * oneSize + gridy * numBoxes * once + gridx * once + classnum + 2] = py;
            label.getData()[b * oneSize + gridy * numBoxes * once + gridx * once + classnum + 3] = w;
            label.getData()[b * oneSize + gridy * numBoxes * once + gridx * once + classnum + 4] = h;
        }
    }

    public static void loadLabelToYolov3(List<float[]> list, int i, Tensor label, int img_w, int img_h, int bbox_num) {
        for (int c = 0; c < list.size(); c++) {
            float[] box = list.get(c);
            float clazz = box[0];
            float cx = box[1];
            float cy = box[2];
            float w = box[3];
            float h = box[4];
            label.getData()[i * bbox_num * 5 + c * 5 + 0] = cx;
            label.getData()[i * bbox_num * 5 + c * 5 + 1] = cy;
            label.getData()[i * bbox_num * 5 + c * 5 + 2] = w;
            label.getData()[i * bbox_num * 5 + c * 5 + 3] = h;
            label.getData()[i * bbox_num * 5 + c * 5 + 4] = clazz;
        }
    }

    public static void loadLabelToYolov3(float[] box, int i, Tensor label, int imgw, int imgh, int bbox_num) {
        int ignore = 0;
        for (int c = 0; c < box.length / 5; c++) {
            int clazz = (int) box[c * 5 + 0];
            //			System.out.println(JsonUtils.toJson(box));
            float x1 = box[c * 5 + 1] / imgw;
            float y1 = box[c * 5 + 2] / imgh;
            float x2 = box[c * 5 + 3] / imgw;
            float y2 = box[c * 5 + 4] / imgh;
            float cx = (x1 + x2) / 2;
            float cy = (y1 + y2) / 2;
            float w = (x2 - x1);
            float h = (y2 - y1);
            if (w == 0 || h == 0) {
                //				System.out.println(x2+"-"+x1+"|"+y2+"-"+y1);
                ignore++;
                continue;
            }
            int currentC = c - ignore;
            label.getData()[i * bbox_num * 5 + currentC * 5 + 0] = cx;
            label.getData()[i * bbox_num * 5 + currentC * 5 + 1] = cy;
            label.getData()[i * bbox_num * 5 + currentC * 5 + 2] = w;
            label.getData()[i * bbox_num * 5 + currentC * 5 + 3] = h;
            label.getData()[i * bbox_num * 5 + currentC * 5 + 4] = clazz;
        }
    }

    public static void loadLabelToYolov7(float[] box, int i, Tensor label, int imgw, int imgh, int bbox_num, int index) {
        int ignore = 0;
        float lowest_w = 1.0f / imgw;
        float lowest_h = 1.0f / imgh;
        for (int c = 0; c < box.length / 5; c++) {
            int clazz = (int) box[c * 5 + 0];
            //			System.out.println(JsonUtils.toJson(box));
            float x1 = box[c * 5 + 1] / imgw;
            float y1 = box[c * 5 + 2] / imgh;
            float x2 = box[c * 5 + 3] / imgw;
            float y2 = box[c * 5 + 4] / imgh;
            float cx = (x1 + x2) / 2;
            float cy = (y1 + y2) / 2;
            float w = (x2 - x1);
            float h = (y2 - y1);
            if (w == 0 || h == 0) {
                //				System.out.println(x2+"-"+x1+"|"+y2+"-"+y1);
                ignore++;
                continue;
            }
            int currentC = c - ignore;
            label.getData()[i * bbox_num * 6 + currentC * 6 + 0] = cx;
            label.getData()[i * bbox_num * 6 + currentC * 6 + 1] = cy;
            label.getData()[i * bbox_num * 6 + currentC * 6 + 2] = w;
            label.getData()[i * bbox_num * 6 + currentC * 6 + 3] = h;
            label.getData()[i * bbox_num * 6 + currentC * 6 + 4] = clazz;
            label.getData()[i * bbox_num * 6 + currentC * 6 + 5] = index;
        }
    }

    public static List<float[]> correctBoxes(float[] orgLabelData, float dx, float dy, float sx, float sy, int flip, int img_w, int img_h) {
        List<float[]> rList = new ArrayList<float[]>();
        int labelSize = orgLabelData.length / 5;
        for (int i = 0; i < labelSize; ++i) {
            float[] r = new float[5];
            float x1 = orgLabelData[i * 5 + 1] / img_w;
            float y1 = orgLabelData[i * 5 + 2] / img_h;
            float x2 = orgLabelData[i * 5 + 3] / img_w;
            float y2 = orgLabelData[i * 5 + 4] / img_h;
            float x = (x1 + x2) / (2);
            float y = (y1 + y2) / (2);
            float w = (x2 - x1);
            float h = (y2 - y1);
            //			System.out.println("===>"+x+":"+y+":"+w+":"+h);
            //			System.out.println(sx+":"+sy);
            //			System.out.println(dx+":="+dy);
            float left = ((x - w / 2.0f) * sx - dx);
            float right = ((x + w / 2.0f) * sx - dx);
            float top = ((y - h / 2.0f) * sy - dy);
            float bottom = ((y + h / 2.0f) * sy - dy);
            if (flip == 1) {
                float swap = top;
                top = 1.0f - bottom;
                bottom = 1.0f - swap;
            }
            //	        System.out.println(left+":"+right);
            //	        System.out.println(left+":"+right+":"+top+":"+bottom);
            left = constrain(left, 0, 1);
            right = constrain(right, 0, 1);
            top = constrain(top, 0, 1);
            bottom = constrain(bottom, 0, 1);
            r[1] = (left + right) / 2.0f;
            r[2] = (top + bottom) / 2.0f;
            r[3] = right - left;
            r[4] = bottom - top;
            r[3] = constrain(r[3], 0, 1);
            r[4] = constrain(r[4], 0, 1);
            rList.add(r);
        }
        return rList;
    }

    public static void randomDistortImage(float[] img, int imgw, int imgh, int imgc, float hue, float saturation, float exposure) {
        if (imgc >= 3) {
            float dhue = RandomUtils.randomFloat(-hue, hue);
            float dsat = RandomUtils.randomScale(saturation);
            float dexp = RandomUtils.randomScale(exposure);
            rgb2hsv(img, imgw, imgh, imgc);
            scaleImageChannel(img, imgw, imgh, imgc, 1, dsat);
            scaleImageChannel(img, imgw, imgh, imgc, 2, dexp);
            for (int i = 0; i < imgw * imgh; ++i) {
                img[i] = img[i] + dhue;
                if (img[i] > 1)
                    img[i] -= 1;
                if (img[i] < 0)
                    img[i] += 1;
            }
            hsv2rgb(img, imgw, imgh, imgc);
            constrainImage(img, imgw, imgh, imgc);
        }
    }

    public static void distortImage(OMImage img, float hue, float saturation, float exposure) {
        if (img.getChannel() >= 3) {
            rgb2hsv(img.getData(), img.getWidth(), img.getHeight(), img.getChannel());
            scaleImageChannel(img.getData(), img.getWidth(), img.getHeight(), img.getChannel(), 1, saturation);
            scaleImageChannel(img.getData(), img.getWidth(), img.getHeight(), img.getChannel(), 2, exposure);
            for (int i = 0; i < img.getWidth() * img.getHeight(); ++i) {
                img.getData()[i] = img.getData()[i] + hue;
                if (img.getData()[i] > 1)
                    img.getData()[i] -= 1;
                if (img.getData()[i] < 0)
                    img.getData()[i] += 1;
            }
            hsv2rgb(img.getData(), img.getWidth(), img.getHeight(), img.getChannel());
            constrainImage(img.getData(), img.getWidth(), img.getHeight(), img.getChannel());
        }
    }

    public static void constrainImage(float[] img, int imgw, int imgh, int imgc) {
        for (int i = 0; i < imgw * imgh * imgc; ++i) {
            if (img[i] < 0)
                img[i] = 0;
            if (img[i] > 1)
                img[i] = 1;
        }
    }

    public static void hsv2rgb(float[] img, int imgw, int imgh, int imgc) {
        assert (imgc == 3);
        float r = 0;
        float g = 0;
        float b = 0;
        float h = 0;
        float s = 0;
        float v = 0;
        float f = 0;
        float p = 0;
        float q = 0;
        float t = 0;
        for (int j = 0; j < imgh; ++j) {
            for (int i = 0; i < imgw; ++i) {
                h = 6 * getPixel(img, i, j, 0, imgw, imgh, imgc);
                s = getPixel(img, i, j, 1, imgw, imgh, imgc);
                v = getPixel(img, i, j, 2, imgw, imgh, imgc);
                if (s == 0) {
                    r = g = b = v;
                } else {
                    int index = (int) Math.floor(h);
                    f = h - index;
                    p = v * (1 - s);
                    q = v * (1 - s * f);
                    t = v * (1 - s * (1 - f));
                    if (index == 0) {
                        r = v;
                        g = t;
                        b = p;
                    } else if (index == 1) {
                        r = q;
                        g = v;
                        b = p;
                    } else if (index == 2) {
                        r = p;
                        g = v;
                        b = t;
                    } else if (index == 3) {
                        r = p;
                        g = q;
                        b = v;
                    } else if (index == 4) {
                        r = t;
                        g = p;
                        b = v;
                    } else {
                        r = v;
                        g = p;
                        b = q;
                    }
                }
                setPixel(img, i, j, 0, imgw, imgh, imgc, r);
                setPixel(img, i, j, 1, imgw, imgh, imgc, g);
                setPixel(img, i, j, 2, imgw, imgh, imgc, b);
            }
        }
    }

    public static void scaleImageChannel(float[] img, int imgw, int imgh, int imgc, int c, float v) {
        int i, j;
        for (j = 0; j < imgh; ++j) {
            for (i = 0; i < imgw; ++i) {
                float pix = getPixel(img, i, j, c, imgh, imgw, imgc);
                pix = pix * v;
                setPixel(img, i, j, c, imgh, imgw, imgc, pix);
            }
        }
    }

    public static void rgb2hsv(float[] img, int imgw, int imgh, int imgc) {
        assert (imgc == 3);
        float r = 0;
        float g = 0;
        float b = 0;
        float h = 0;
        float s = 0;
        float v = 0;
        for (int j = 0; j < imgh; ++j) {
            for (int i = 0; i < imgw; ++i) {
                r = getPixel(img, i, j, 0, imgh, imgw, imgc);
                g = getPixel(img, i, j, 1, imgh, imgw, imgc);
                b = getPixel(img, i, j, 2, imgh, imgw, imgc);
                float max = threeWayMax(r, g, b);
                float min = threeWayMin(r, g, b);
                //	            System.out.println(max+":"+min);
                float delta = max - min;
                v = max;
                if (max == 0) {
                    s = 0;
                    h = 0;
                } else {
                    s = delta / max;
                    if (r == max) {
                        h = (g - b) / delta;
                    } else if (g == max) {
                        h = 2 + (b - r) / delta;
                    } else {
                        h = 4 + (r - g) / delta;
                    }
                    if (h < 0)
                        h += 6;
                    h = h / 6.0f;
                }
                setPixel(img, i, j, 0, imgh, imgw, imgc, h);
                setPixel(img, i, j, 1, imgh, imgw, imgc, s);
                setPixel(img, i, j, 2, imgh, imgw, imgc, v);
            }
        }
    }

    public static float threeWayMax(float a, float b, float c) {
        return (a > b) ? ((a > c) ? a : c) : ((b > c) ? b : c);
    }

    public static float threeWayMin(float a, float b, float c) {
        return (a < b) ? ((a < c) ? a : c) : ((b < c) ? b : c);
    }

    public static void flipImage(float[] image, int imgc, int imgw, int imgh) {
        for (int k = 0; k < imgc; ++k) {
            for (int i = 0; i < imgh; ++i) {
                for (int j = 0; j < imgw / 2; ++j) {
                    int index = j + imgw * (i + imgh * (k));
                    int flip = (imgw - j - 1) + imgw * (i + imgh * (k));
                    float swap = image[flip];
                    image[flip] = image[index];
                    image[index] = swap;
                }
            }
        }
    }

    public static float[] resizeImage(float[] im, int imgw, int imgh, int imgc, int w, int h) {
        float[] resized = new float[h * w * imgc];
        float[] part = new float[imgh * w * imgc];
        float w_scale = (imgw - 1.0f) / (w - 1.0f);
        float h_scale = (imgh - 1.0f) / (h - 1.0f);
        //	    System.out.println(w_scale+":"+h_scale);
        for (int k = 0; k < imgc; ++k) {
            for (int r = 0; r < imgh; ++r) {
                for (int c = 0; c < w; ++c) {
                    float val = 0;
                    if (c == w - 1 || imgw == 1) {
                        val = getPixel(im, imgw - 1, r, k, imgh, imgw, imgc);
                    } else {
                        float sx = c * w_scale;
                        int ix = (int) sx;
                        float dx = sx - ix;
                        val = (1.0f - dx) * getPixel(im, ix, r, k, imgh, imgw, imgc) + dx * getPixel(im, ix + 1, r, k, imgh, imgw, imgc);
                    }
                    setPixel(part, c, r, k, imgh, w, imgc, val);
                }
            }
        }
        for (int k = 0; k < imgc; ++k) {
            for (int r = 0; r < h; ++r) {
                float sy = r * h_scale;
                int iy = (int) sy;
                float dy = sy - iy;
                for (int c = 0; c < w; ++c) {
                    float val = (1.0f - dy) * getPixel(part, c, iy, k, imgh, w, imgc);
                    setPixel(resized, c, r, k, h, w, imgc, val);
                }
                if (r == h - 1 || imgh == 1)
                    continue;
                for (int c = 0; c < w; ++c) {
                    float val = dy * getPixel(part, c, iy + 1, k, imgh, w, imgc);
                    add_pixel(resized, c, r, k, h, w, imgc, val);
                }
            }
        }
        return resized;
    }

    public static float[] cropImage(float[] orig, int imgc, int imgw, int imgh, int dx, int dy, int w, int h) {
        float[] cropped = new float[imgc * h * w];
        //		System.out.println(dx+":"+dy);
        for (int k = 0; k < imgc; ++k) {
            for (int j = 0; j < h; ++j) {
                for (int i = 0; i < w; ++i) {
                    int row = j + dy;
                    int col = i + dx;
                    float val = 0;
                    row = constrainInt(row, 0, imgh - 1);
                    col = constrainInt(col, 0, imgw - 1);
                    val = getPixel(orig, col, row, k, imgh, imgw, imgc);
                    //	                if(row == 0 || row == imgh-1 || col == 0 || col == imgw-1) {
                    //	                	val = 0;
                    //	                }
                    //
                    setPixel(cropped, i, j, k, h, w, imgc, val);
                }
            }
        }
        return cropped;
    }

    public static void main(String[] args) {
        ImageUtils utils = new ImageUtils();
        String testPath = "H:\\voc\\banana-detection\\show\\test0.png";
        int width = 256;
        int height = 256;
        int dw = (int) (width * 0.2f);
        int dh = (int) (height * 0.2f);
        int pleft = RandomUtils.uniformInt(-dw, dw);
        int pright = RandomUtils.uniformInt(-dw, dw);
        int ptop = RandomUtils.uniformInt(-dh, dh);
        int pbot = RandomUtils.uniformInt(-dh, dh);
        int swidth = width - pleft - pright;
        int sheight = height - ptop - pbot;
        //			System.out.println(swidth+":"+sheight);
        try {
            File file = new File(testPath);
            if (file.exists()) {
                float[] orig = utils.getImageData(file, false, false);
                float[] cropped = cropImage(orig, 3, width, height, pleft, ptop, swidth, sheight);
                float[] sized = resizeImage(cropped, swidth, sheight, 3, width, height);
                //					System.out.println(JsonUtils.toJson(sized));
                utils.createRGBImage("H:\\voc\\banana-detection\\show\\test11111.png", "png", ImageUtils.color2rgb2(cropped, sheight, swidth), swidth, sheight, null);
                utils.createRGBImage("H:\\voc\\banana-detection\\show\\test22222.png", "png", ImageUtils.color2rgb2(sized, height, width), width, height, null);
            }
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public static float getPixel(float[] img, int x, int y, int c, int imgh, int imgw, int imgc) {
        assert (x < imgw && y < imgh && c < imgc);
        return img[c * imgh * imgw + y * imgw + x];
    }

    public static void setPixel(float[] img, int x, int y, int c, int imgh, int imgw, int imgc, float val) {
        if (x < 0 || y < 0 || c < 0 || x >= imgw || y >= imgh || c >= imgc)
            return;
        assert (x < imgw && y < imgh && c < imgc);
        img[c * imgh * imgw + y * imgw + x] = val;
    }

    public static void add_pixel(float[] img, int x, int y, int c, int imgh, int imgw, int imgc, float val) {
        assert (x < imgw && y < imgh && c < imgc);
        img[c * imgh * imgw + y * imgw + x] += val;
    }

    public static int constrainInt(int a, int min, int max) {
        if (a < min)
            return min;
        if (a > max)
            return max;
        return a;
    }

    public static float constrain(float a, float min, float max) {
        if (a < min)
            return min;
        if (a > max)
            return max;
        return a;
    }

    @Override
    public void transform(Tensor input, Tensor label, String[] idxSet, int[] indexs, Map<String, float[]> orgLabelData) {
        // TODO Auto-generated method stub
        label.setData(new float[label.getDataLength()]);
        for (int i = 0; i < input.getShape()[0]; i++) {
            float[] orig = input.getByNumber(i);
            int oh = input.getShape()[2];
            int ow = input.getShape()[3];
            int dw = (int) (ow * jitter);
            int dh = (int) (oh * jitter);
            //			int pleft = RandomUtils.uniformInt(-dw, dw);
            //			int pright = RandomUtils.uniformInt(-dw, dw);
            //			int ptop = RandomUtils.uniformInt(-dh, dh);
            //			int pbot = RandomUtils.uniformInt(-dh, dh);
            int pleft = 0;
            int pright = 0;
            int ptop = 0;
            int pbot = 0;
            int swidth = ow - pleft - pright;
            int sheight = oh - ptop - pbot;
            float sx = swidth * 1.0f / ow;
            float sy = sheight * 1.0f / oh;
            int flip = RandomUtils.rand() % 2;
            float[] cropped = cropImage(orig, input.getShape()[1], ow, oh, pleft, ptop, swidth, sheight);
            float dx = pleft * 1.0f / ow / sx;
            float dy = ptop * 1.0f / oh / sy;
            float[] sized = resizeImage(cropped, swidth, sheight, input.getShape()[1], ow, oh);
            if (flip == 1) {
                flipImage(sized, input.getShape()[1], ow, oh);
            }
            randomDistortImage(sized, ow, oh, input.getShape()[1], hue, saturation, exposure);
            input.setByNumber(i, sized);
            /**
             * 处理变换后label

             */
            float[] orgLabel = orgLabelData.get(idxSet[indexs[i]]);
            List<float[]> r = fillTruthRegion(i, orgLabel, classnum, numBoxes, flip, dx, dy, 1.0f / sx, 1.0f / sy, ow, oh);
            /**
             * 转换对应版本yolo格式label数据

             */
            switch (dataType) {
                case yolov1:
                    loadLabelToYolov1(r, i, label, ow, oh, classnum, numBoxes);
                    break;
                case yolov3:
                    loadLabelToYolov3(r, i, label, ow, oh, 90);
                    break;
            }
            //			System.out.println(JsonUtils.toJson(label.getByNumber(i)));
        }
    }

    public void transform(Tensor input, Tensor label, Tensor oLabel, String[] idxSet, int[] indexs, Map<String, float[]> orgLabelData) {
        // TODO Auto-generated method stub
        label.setData(new float[label.getDataLength()]);
        for (int i = 0; i < input.getShape()[0]; i++) {
            float[] orig = input.getByNumber(i);
            int oh = input.getShape()[2];
            int ow = input.getShape()[3];
            int dw = (int) (ow * jitter);
            int dh = (int) (oh * jitter);
            int pleft = 0;
            int pright = 0;
            int ptop = 0;
            int pbot = 0;
            int swidth = ow - pleft - pright;
            int sheight = oh - ptop - pbot;
            float sx = swidth * 1.0f / ow;
            float sy = sheight * 1.0f / oh;
            int flip = RandomUtils.rand() % 2;
            float[] cropped = cropImage(orig, input.getShape()[1], ow, oh, pleft, ptop, swidth, sheight);
            float dx = pleft * 1.0f / ow / sx;
            float dy = ptop * 1.0f / oh / sy;
            float[] sized = resizeImage(cropped, swidth, sheight, input.getShape()[1], ow, oh);
            if (flip == 1) {
                flipImage(sized, input.getShape()[1], ow, oh);
            }
            randomDistortImage(sized, ow, oh, input.getShape()[1], hue, saturation, exposure);
            input.setByNumber(i, sized);
            /**
             * 处理变换后label

             */
            float[] orgLabel = orgLabelData.get(idxSet[indexs[i]]);
            List<float[]> r = fillTruthRegion(i, orgLabel, classnum, numBoxes, flip, dx, dy, 1.0f / sx, 1.0f / sy, ow, oh);
            /**
             * 转换对应版本yolo格式label数据

             */
            for (int c = 0; c < r.size(); c++) {
                oLabel.setByNumberAndChannel(i, c, r.get(c));
            }
            //			System.out.println(JsonUtils.toJson(label.getByNumber(i)));
        }
    }

    public void showTransform(String outputPath, Tensor input, Tensor label, String[] idxSet, int[] indexs, Map<String, float[]> orgLabelData) {
        Tensor oLabel = new Tensor(input.getShape()[0], 1, 1, 5);
        this.transform(input, label, oLabel, idxSet, indexs, orgLabelData);
        ImageUtils utils = new ImageUtils();
        input.setData(MatrixOperation.multiplication(input.getData(), 255.0f));
        oLabel.setData(MatrixOperation.multiplication(oLabel.getData(), input.getShape()[3]));
        for (int b = 0; b < input.getShape()[0]; b++) {
            float[] once = input.getByNumber(b);
            float[] labelArray = oLabel.getByNumber(b);
            int[][] bbox = new int[][]{{0, (int) (labelArray[1] - labelArray[3] / 2), (int) (labelArray[2] - labelArray[4] / 2), (int) (labelArray[1] + labelArray[3] / 2), (int) (labelArray[2] + labelArray[3] / 2)}};
            System.out.println(JsonUtils.toJson(bbox));
            utils.createRGBImage(outputPath + b + ".png", "png", ImageUtils.color2rgb2(once, input.getShape()[2], input.getShape()[3]), input.getShape()[3], input.getShape()[2], bbox);
        }
    }
}

