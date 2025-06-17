package com.omega.engine.nn.data;

import com.omega.common.utils.RandomUtils;
import com.omega.engine.tensor.Tensor;

/**
 * hsv utils
 *
 * @author Administrator
 */
public class HsvUtils {
    public static void hsv(Tensor input, float hue, float sat, float exp) {
        rgb2hsv(input);
        scaleImageChannel(input, 1, sat);
        scaleImageChannel(input, 2, exp);
        addImageChannel(input, 0, hue);
        hsv2rgb(input);
    }

    public static void rgb2hsv(Tensor input) {
        int c = input.getShape()[1];
        int oh = input.getShape()[2];
        int ow = input.getShape()[3];
        assert (c == 3);
        float[] img = input.getData();
        int i, j;
        float r, g, b;
        float h, s, v;
        for (int bs = 0; bs < input.getShape()[0]; bs++) {
            for (j = 0; j < oh; ++j) {
                for (i = 0; i < ow; ++i) {
                    r = getPixel(img, c, ow, oh, bs, 0, i, j);
                    g = getPixel(img, c, ow, oh, bs, 1, i, j);
                    b = getPixel(img, c, ow, oh, bs, 2, i, j);
                    float max = threeWayMax(r, g, b);
                    float min = threeWayMin(r, g, b);
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
                    setPixel(img, c, ow, oh, bs, 0, i, j, h);
                    setPixel(img, c, ow, oh, bs, 1, i, j, s);
                    setPixel(img, c, ow, oh, bs, 2, i, j, v);
                }
            }
        }
        input.setData(img);
    }

    public static void hsv2rgb(Tensor input) {
        int c = input.getShape()[1];
        int oh = input.getShape()[2];
        int ow = input.getShape()[3];
        assert (c == 3);
        int i, j;
        float r, g, b;
        float h, s, v;
        float f, p, q, t;
        float[] img = input.getData();
        for (int bs = 0; bs < input.getShape()[0]; bs++) {
            for (j = 0; j < oh; ++j) {
                for (i = 0; i < ow; ++i) {
                    h = 6 * getPixel(img, c, ow, oh, bs, 0, i, j);
                    s = getPixel(img, c, ow, oh, bs, 1, i, j);
                    v = getPixel(img, c, ow, oh, bs, 2, i, j);
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
                    setPixel(img, c, ow, oh, bs, 0, i, j, r);
                    setPixel(img, c, ow, oh, bs, 1, i, j, g);
                    setPixel(img, c, ow, oh, bs, 2, i, j, b);
                }
            }
        }
        input.setData(img);
    }

    public static void rgb2hsv(float[] img, int c, int ow, int oh) {
        assert (c == 3);
        int i, j;
        float r, g, b;
        float h, s, v;
        for (j = 0; j < oh; ++j) {
            for (i = 0; i < ow; ++i) {
                r = getPixel(img, ow, oh, i, j, 0);
                g = getPixel(img, ow, oh, i, j, 1);
                b = getPixel(img, ow, oh, i, j, 2);
                float max = threeWayMax(r, g, b);
                float min = threeWayMin(r, g, b);
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
                setPixel(img, ow, oh, i, j, 0, h);
                setPixel(img, ow, oh, i, j, 1, s);
                setPixel(img, ow, oh, i, j, 2, v);
            }
        }
    }

    public static void hsv2rgb(float[] img, int c, int ow, int oh) {
        assert (c == 3);
        int i, j;
        float r, g, b;
        float h, s, v;
        float f, p, q, t;
        for (j = 0; j < oh; ++j) {
            for (i = 0; i < ow; ++i) {
                h = 6 * getPixel(img, ow, oh, i, j, 0);
                s = getPixel(img, ow, oh, i, j, 1);
                v = getPixel(img, ow, oh, i, j, 2);
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
                setPixel(img, ow, oh, i, j, 0, r);
                setPixel(img, ow, oh, i, j, 1, g);
                setPixel(img, ow, oh, i, j, 2, b);
            }
        }
    }

    public static float threeWayMax(float a, float b, float c) {
        return (a > b) ? ((a > c) ? a : c) : ((b > c) ? b : c);
    }

    public static float threeWayMin(float a, float b, float c) {
        return (a < b) ? ((a < c) ? a : c) : ((b < c) ? b : c);
    }

    public static void setPixel(float[] x, int oc, int ow, int oh, int n, int c, int w, int h, float val) {
        x[n * oc * oh * ow + c * oh * ow + h * ow + w] = val;
    }

    public static float getPixel(float[] x, int oc, int ow, int oh, int n, int c, int w, int h) {
        return x[n * oc * oh * ow + c * oh * ow + h * ow + w];
    }

    public static void setPixel(float[] x, int ow, int oh, int w, int h, int c, float val) {
        x[c * oh * ow + h * ow + w] = val;
    }

    public static float getPixel(float[] x, int ow, int oh, int w, int h, int c) {
        return x[c * oh * ow + h * ow + w];
    }

    public static void scaleImageChannel(Tensor input, int ci, float v) {
        float[] im = input.getData();
        int i, j;
        for (int n = 0; n < input.getShape()[0]; n++) {
            float val = RandomUtils.randomFloat(-v, v);
            for (j = 0; j < input.getShape()[2]; ++j) {
                for (i = 0; i < input.getShape()[3]; ++i) {
                    float pix = getPixel(im, input.getShape()[1], input.getShape()[3], input.getShape()[2], n, ci, i, j);
                    pix = pix * val;
                    setPixel(im, input.getShape()[1], input.getShape()[3], input.getShape()[2], n, ci, i, j, pix);
                }
            }
        }
        input.setData(im);
    }

    public static void addImageChannel(Tensor input, int ci, float v) {
        float[] im = input.getData();
        int i, j;
        for (int n = 0; n < input.getShape()[0]; n++) {
            float val = RandomUtils.randomFloat(-v, v);
            for (j = 0; j < input.getShape()[2]; ++j) {
                for (i = 0; i < input.getShape()[3]; ++i) {
                    float pix = getPixel(im, input.getShape()[1], input.getShape()[3], input.getShape()[2], n, ci, i, j);
                    pix = pix + val;
                    if (pix > 1) {
                        pix -= 1;
                    } else if (pix < 0) {
                        pix += 1;
                    }
                    setPixel(im, input.getShape()[1], input.getShape()[3], input.getShape()[2], n, ci, i, j, pix);
                }
            }
        }
        input.setData(im);
    }
}

