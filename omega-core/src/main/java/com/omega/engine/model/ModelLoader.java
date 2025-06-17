package com.omega.engine.model;

import com.omega.common.utils.JsonUtils;
import com.omega.engine.nn.layer.*;
import com.omega.engine.nn.layer.active.*;
import com.omega.engine.nn.layer.normalization.BNLayer;
import com.omega.engine.nn.network.Network;
import com.omega.engine.pooling.PoolingType;

import java.io.File;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.io.Reader;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * ModelLoader
 *
 * @author Administrator
 */
public class ModelLoader {
    public static void loadConfigToModel(Network nn, String filepath) {
        try {
            List<Map<String, Object>> layerCfgs = loadData(filepath);
            if (layerCfgs == null) {
                throw new RuntimeException("load the config file error.");
            }
            addLayer(layerCfgs, nn);
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    @SuppressWarnings("unused")
    public static void addLayer(List<Map<String, Object>> layerCfgs, Network nn) {
        for (int i = 0; i < layerCfgs.size(); i++) {
            Map<String, Object> cfg = layerCfgs.get(i);
            String layerType = cfg.get("layerType").toString();
            Layer layer = null;
            int[] shape = null;
            switch (layerType) {
                case "fully":
                    shape = addFullyLayers(cfg, nn);
                    break;
                case "convolutional":
                    shape = addConvLayers(cfg, nn);
                    break;
                case "cbl":
                    shape = addCBLs(cfg, nn);
                    break;
                case "maxpool":
                    shape = addMaxPoolingLayer(cfg, nn);
                    break;
                case "meanpool":
                    shape = addMeanPoolingLayer(cfg, nn);
                    break;
                case "avgpool":
                    shape = addAvgPoolingLayer(cfg, nn);
                    break;
                case "input":
                    shape = addInputLayer(cfg, nn);
                    break;
                case "route":
                    shape = addRouteLayer(cfg, nn, layerCfgs, i);
                    break;
                case "upsample":
                    shape = addUpsampleLayer(cfg, nn);
                    break;
                case "yolo":
                    shape = addYoloLayer(cfg, nn);
                    break;
                default:
                    break;
            }
            System.out.println(layerType + "(" + i + ")" + ":" + JsonUtils.toJson(shape));
        }
    }

    public static int getInt(String val) {
        return new Double(val).intValue();
    }

    public static float getFloat(String val) {
        return new Float(val);
    }

    public static int[] addInputLayer(Map<String, Object> cfg, Network nn) {
        int channel = getInt(cfg.get("channel").toString());
        int width = getInt(cfg.get("width").toString());
        int height = getInt(cfg.get("height").toString());
        InputLayer inputLayer = new InputLayer(channel, height, width);
        nn.addLayer(inputLayer);
        cfg.put("lastIndex", inputLayer.index);
        return inputLayer.outputShape();
    }

    public static int[] addMaxPoolingLayer(Map<String, Object> cfg, Network nn) {
        Layer pre = nn.getLastLayer();
        if (pre == null) {
            throw new RuntimeException("the pooling layer cant be the fisrt layer.");
        }
        int size = getInt(cfg.get("size").toString());
        int stride = getInt(cfg.get("stride").toString());
        int padding = 0;
        if (cfg.get("padding") != null) {
            padding = getInt(cfg.get("padding").toString());
        } else {
            padding = size - 1;
        }
        PoolingLayer pool = new PoolingLayer(pre.oChannel, pre.oWidth, pre.oHeight, size, size, stride, padding, PoolingType.MAX_POOLING);
        nn.addLayer(pool);
        cfg.put("lastIndex", pool.index);
        return pool.outputShape();
    }

    public static int[] addMeanPoolingLayer(Map<String, Object> cfg, Network nn) {
        Layer pre = nn.getLastLayer();
        if (pre == null) {
            throw new RuntimeException("the pooling layer cant be the fisrt layer.");
        }
        int size = getInt(cfg.get("size").toString());
        int stride = getInt(cfg.get("stride").toString());
        PoolingLayer pool1 = new PoolingLayer(pre.oChannel, pre.oWidth, pre.oHeight, size, size, stride, PoolingType.MEAN_POOLING);
        nn.addLayer(pool1);
        cfg.put("lastIndex", pool1.index);
        return pool1.outputShape();
    }

    public static int[] addAvgPoolingLayer(Map<String, Object> cfg, Network nn) {
        Layer pre = nn.getLastLayer();
        if (pre == null) {
            throw new RuntimeException("the pooling layer cant be the fisrt layer.");
        }
        AVGPoolingLayer pool1 = new AVGPoolingLayer(pre.oChannel, pre.oWidth, pre.oHeight);
        nn.addLayer(pool1);
        cfg.put("lastIndex", pool1.index);
        return pool1.outputShape();
    }

    public static int[] addConvLayers(Map<String, Object> cfg, Network nn) {
        Layer pre = nn.getLastLayer();
        if (pre == null) {
            throw new RuntimeException("the convolution layer cant be the fisrt layer.");
        }
        int kernel = getInt(cfg.get("kernel").toString());
        int size = getInt(cfg.get("size").toString());
        int stride = getInt(cfg.get("stride").toString());
        int pad = getInt(cfg.get("pad").toString());
        int freeze = 0;
        if (cfg.get("freeze") != null) {
            freeze = getInt(cfg.get("freeze").toString());
        }
        int bn = 0;
        boolean hasBias = true;
        if (cfg.get("batch_normalize") != null) {
            bn = getInt(cfg.get("batch_normalize").toString());
            if (bn > 0) {
                hasBias = false;
            }
        }
        String activation = null;
        if (cfg.get("activation") != null) {
            activation = cfg.get("activation").toString();
        }
        ConvolutionLayer conv = new ConvolutionLayer(pre.oChannel, kernel, pre.oWidth, pre.oHeight, size, size, pad, stride, hasBias);
        if (freeze == 1) {
            conv.freeze = true;
        }
        nn.addLayer(conv);
        cfg.put("lastIndex", conv.index);
        if (bn == 1) {
            BNLayer bn1 = new BNLayer(conv);
            bn1.preLayer = conv;
            if (freeze == 1) {
                bn1.freeze = true;
            }
            nn.addLayer(bn1);
            cfg.put("lastIndex", bn1.index);
        }
        Layer activeLayer = makeActivation(activation, conv);
        if (activeLayer != null) {
            nn.addLayer(activeLayer);
            cfg.put("lastIndex", activeLayer.index);
        }
        return conv.outputShape();
    }

    public static int[] addCBLs(Map<String, Object> cfg, Network nn) {
        Layer pre = nn.getLastLayer();
        if (pre == null) {
            throw new RuntimeException("the convolution layer cant be the fisrt layer.");
        }
        int kernel = getInt(cfg.get("kernel").toString());
        int size = getInt(cfg.get("size").toString());
        int stride = getInt(cfg.get("stride").toString());
        int pad = getInt(cfg.get("pad").toString());
        String activation = null;
        if (cfg.get("activation") != null) {
            activation = cfg.get("activation").toString();
            if (activation.equals("leaky")) {
                activation = "leaky_relu";
            }
        }
        CBLLayer cbl = new CBLLayer(pre.oChannel, kernel, pre.oHeight, pre.oWidth, size, size, stride, pad, true, activation, nn);
        nn.addLayer(cbl);
        cfg.put("lastIndex", cbl.index);
        return cbl.outputShape();
    }

    public static int[] addFullyLayers(Map<String, Object> cfg, Network nn) {
        Layer pre = nn.getLastLayer();
        if (pre == null) {
            throw new RuntimeException("the fully layer cant be the fisrt layer.");
        }
        int inputSize = pre.oChannel * pre.oHeight * pre.oWidth;
        int outputSize = getInt(cfg.get("output").toString());
        int freeze = 0;
        if (cfg.get("freeze") != null) {
            freeze = getInt(cfg.get("freeze").toString());
        }
        int bn = 0;
        boolean hasBias = true;
        if (cfg.get("batch_normalize") != null) {
            bn = getInt(cfg.get("batch_normalize").toString());
            if (bn > 0) {
                hasBias = false;
            }
        }
        String activation = null;
        if (cfg.get("activation") != null) {
            activation = cfg.get("activation").toString();
        }
        FullyLayer fully = new FullyLayer(inputSize, outputSize, hasBias);
        if (freeze == 1) {
            fully.freeze = true;
        }
        nn.addLayer(fully);
        cfg.put("lastIndex", fully.index);
        if (bn == 1) {
            BNLayer bn1 = new BNLayer();
            bn1.preLayer = fully;
            if (freeze == 1) {
                bn1.freeze = true;
            }
            nn.addLayer(bn1);
            cfg.put("lastIndex", bn1.index);
        }
        Layer activeLayer = makeActivation(activation, fully);
        if (activeLayer != null) {
            nn.addLayer(activeLayer);
            cfg.put("lastIndex", activeLayer.index);
        }
        return fully.outputShape();
    }

    public static int[] addRouteLayer(Map<String, Object> cfg, Network nn, List<Map<String, Object>> layerCfgs, int current) {
        int group = 1;
        if (cfg.get("group") != null) {
            group = getInt(cfg.get("group").toString());
        }
        int groupId = 0;
        if (cfg.get("group_id") != null) {
            groupId = getInt(cfg.get("group_id").toString());
        }
        List<Double> layerIndexList = (List<Double>) cfg.get("layers");
        int[] layerIndexs = new int[layerIndexList.size()];
        for (int i = 0; i < layerIndexList.size(); i++) {
            layerIndexs[i] = layerIndexList.get(i).intValue();
        }
        Layer[] layers = new Layer[layerIndexs.length];
        for (int i = 0; i < layerIndexs.length; i++) {
            int ridx = layerIndexs[i];
            int index = 0;
            if (ridx < 0) {
                index = (int) layerCfgs.get(current + ridx).get("lastIndex");
            } else {
                index = (int) layerCfgs.get(ridx).get("lastIndex");
            }
            layers[i] = nn.layerList.get(index);
        }
        RouteLayer routeLayer = new RouteLayer(layers, group, groupId);
        nn.addLayer(routeLayer);
        cfg.put("lastIndex", routeLayer.index);
        return routeLayer.outputShape();
    }

    public static int[] addUpsampleLayer(Map<String, Object> cfg, Network nn) {
        Layer pre = nn.getLastLayer();
        if (pre == null) {
            throw new RuntimeException("the upsample layer cant be the fisrt layer.");
        }
        int stride = getInt(cfg.get("stride").toString());
        UPSampleLayer upsampleLayer = new UPSampleLayer(pre.oChannel, pre.oHeight, pre.oWidth, stride);
        nn.addLayer(upsampleLayer);
        cfg.put("lastIndex", upsampleLayer.index);
        return upsampleLayer.outputShape();
    }

    public static int[] addYoloLayer(Map<String, Object> cfg, Network nn) {
        int class_number = getInt(cfg.get("classes").toString());
        int total = getInt(cfg.get("num").toString());
        int maxBox = getInt(cfg.get("maxBox").toString());
        float ignoreThresh = getFloat(cfg.get("ignore_thresh").toString());
        float truthThresh = getFloat(cfg.get("truth_thresh").toString());
        float scale_x_y = 1;
        if (cfg.get("scale_x_y") != null) {
            scale_x_y = getInt(cfg.get("scale_x_y").toString());
        }
        int active = 1;
        if (cfg.get("active") != null) {
            active = getInt(cfg.get("active").toString());
        }
        List<Double> anchorsList = (List<Double>) cfg.get("anchors");
        float[] anchors = new float[anchorsList.size()];
        for (int i = 0; i < anchorsList.size(); i++) {
            anchors[i] = anchorsList.get(i).floatValue();
        }
        List<Double> maskList = (List<Double>) cfg.get("mask");
        int[] mask = null;
        if (maskList != null) {
            mask = new int[maskList.size()];
            for (int i = 0; i < maskList.size(); i++) {
                mask[i] = maskList.get(i).intValue();
            }
        } else {
            mask = new int[total];
        }
        YoloLayer yoloLayer = new YoloLayer(class_number, mask.length, mask, anchors, maxBox, total, ignoreThresh, truthThresh, active, scale_x_y);
        nn.addLayer(yoloLayer);
        cfg.put("lastIndex", yoloLayer.index);
        return yoloLayer.outputShape();
    }

    public static Layer makeActivation(String activation, Layer preLayer) {
        Layer layer = null;
        switch (activation) {
            case "relu":
                layer = new ReluLayer(preLayer);
                preLayer.paramsInit = ParamsInit.relu;
                break;
            case "sigmoid":
                layer = new SigmodLayer(preLayer);
                preLayer.paramsInit = ParamsInit.sigmoid;
                break;
            case "leaky":
                layer = new LeakyReluLayer(preLayer);
                preLayer.paramsInit = ParamsInit.leaky_relu;
                break;
            case "tanh":
                layer = new TanhLayer(preLayer);
                preLayer.paramsInit = ParamsInit.tanh;
                break;
            case "silu":
                layer = new SiLULayer(preLayer);
                preLayer.paramsInit = ParamsInit.silu;
                break;
            case "none":
                break;
            default:
                throw new RuntimeException("not support this active function.");
        }
        return layer;
    }

    @SuppressWarnings("unchecked")
    public static List<Map<String, Object>> loadData(String filepath) {
        try {
            File file = new File(filepath);
            if (file.exists()) {
                try (FileInputStream fos = new FileInputStream(file); Reader reader = new InputStreamReader(fos, "utf-8");) {
                    int ch = 0;
                    StringBuffer sb = new StringBuffer();
                    while ((ch = reader.read()) != -1) {
                        sb.append((char) ch);
                    }
                    String json = sb.toString();
                    List<Map<String, Object>> list = new ArrayList<Map<String, Object>>();
                    list = JsonUtils.gson.fromJson(json, list.getClass());
                    return list;
                } catch (Exception e) {
                    // TODO: handle exception
                    e.printStackTrace();
                }
            } else {
                throw new RuntimeException("the config file is not exists.");
            }
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
        return null;
    }
}

