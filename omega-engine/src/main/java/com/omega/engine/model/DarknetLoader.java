package com.omega.engine.model;

import com.omega.common.data.Tensor;
import com.omega.engine.nn.layer.CBLLayer;
import com.omega.engine.nn.layer.ConvolutionLayer;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.normalization.BNLayer;
import com.omega.engine.nn.network.Network;
import jcuda.Sizeof;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.List;

/**
 * darknet weight file loader
 *
 * @author Administrator
 */
public class DarknetLoader {
    public static void loadWeight(Network net, String path, int layerCount, boolean freeze) {
        System.out.println("start load weight.");
        try (RandomAccessFile file = new RandomAccessFile(path, "r")) {
            int major = readInt(file);
            int minor = readInt(file);
            int revision = readInt(file);
            System.out.println("major:" + major);
            System.out.println("minor:" + minor);
            System.out.println("revision:" + revision);
            if ((major * 10 + minor) >= 2) {
                long seen = readBigInt(file);
                System.out.println("seen:" + seen);
            } else {
                int seen = readInt(file);
                System.out.println("seen:" + seen);
            }
            int index = 0;
            //		    System.out.println(net.layerList.size());
            /**
             * load weight by layers

             */
            for (int l = 0; l < net.layerList.size(); l++) {
                Layer layer = net.layerList.get(l);
                if (index < layerCount) {
                    //		    		System.out.println(index+":"+layerCount);
                    switch (layer.getLayerType()) {
                        case conv:
                            loadConvWeights(file, l, layer, net.layerList, freeze);
                            index++;
                            break;
                        case cbl:
                            loadCBLWeights(file, l, layer, net.layerList, freeze);
                            index++;
                            break;
                        case full:
                            loadFullyWeights(file, l, layer, net.layerList, freeze);
                            index++;
                            break;
                        case pooling:
                            index++;
                            break;
                        case route:
                            index++;
                            break;
                        default:
                            break;
                    }
                }
            }
            System.out.println("load weight finish.");
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static void loadWeight(Network net, String path) {
        System.out.println("start load weight.");
        try (RandomAccessFile file = new RandomAccessFile(path, "r")) {
            int major = readInt(file);
            int minor = readInt(file);
            int revision = readInt(file);
            System.out.println("major:" + major);
            System.out.println("minor:" + minor);
            System.out.println("revision:" + revision);
            if ((major * 10 + minor) >= 2) {
                long seen = readBigInt(file);
                System.out.println("seen:" + seen);
            } else {
                int seen = readInt(file);
                System.out.println("seen:" + seen);
            }
            /**
             * load weight by layers

             */
            for (int l = 0; l < net.layerList.size(); l++) {
                Layer layer = net.layerList.get(l);
                switch (layer.getLayerType()) {
                    case conv:
                        loadConvWeights(file, l, layer, net.layerList, false);
                        break;
                    case full:
                        loadFullyWeights(file, l, layer, net.layerList, false);
                        break;
                    default:
                        break;
                }
            }
            System.out.println("load weight finish.");
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static void loadConvWeights(RandomAccessFile inputStream, int index, Layer layer, List<Layer> layerList, boolean freeze) throws IOException {
        /**
         * load biases

         */
        readFloat(inputStream, layer.bias);
        /**
         * load bn params

         */
        if (!layer.hasBias && index < layerList.size() - 1 && layerList.get(index + 1) instanceof BNLayer) {
            BNLayer bnl = (BNLayer) layerList.get(index + 1);
            bnl.init();
            readFloat(inputStream, bnl.gamma);
            bnl.beta = layer.bias.copyGPU();
            bnl.beta.syncHost();
            readFloat(inputStream, bnl.runingMean);
            readFloat(inputStream, bnl.runingVar);
            bnl.freeze = freeze;
        }
        /**
         * load conv weight

         */
        readFloat(inputStream, layer.weight);
        layer.freeze = freeze;
    }

    public static void loadCBLWeights(RandomAccessFile inputStream, int index, Layer layer, List<Layer> layerList, boolean freeze) throws IOException {
        CBLLayer cbl = (CBLLayer) layer;
        ConvolutionLayer conv = cbl.getConvLayer();
        /**
         * load biases

         */
        readFloat(inputStream, conv.bias);
        /**
         * load bn params

         */
        BNLayer bnl = cbl.getBnLayer();
        bnl.init();
        readFloat(inputStream, bnl.gamma);
        bnl.beta = conv.bias.copyGPU();
        readFloat(inputStream, bnl.runingMean);
        readFloat(inputStream, bnl.runingVar);
        bnl.freeze = freeze;
        /**
         * load conv weight

         */
        readFloat(inputStream, conv.weight);
        layer.freeze = freeze;
    }

    public static void loadFullyWeights(RandomAccessFile inputStream, int index, Layer layer, List<Layer> layerList, boolean freeze) throws IOException {
        /**
         * load biases

         */
        readFloat(inputStream, layer.bias);
        /**
         * load conv weight

         */
        readFloat(inputStream, layer.weight);
        /**
         * load bn params

         */
        if (!layer.hasBias && index < layerList.size() - 1) {
            BNLayer bnl = (BNLayer) layerList.get(index + 1);
            bnl.init();
            readFloat(inputStream, bnl.gamma);
            readFloat(inputStream, bnl.runingMean);
            readFloat(inputStream, bnl.runingVar);
            bnl.freeze = freeze;
        }
        layer.freeze = freeze;
    }

    public static long readBigInt(RandomAccessFile inputStream) throws IOException {
        long retVal;
        byte[] buffer = new byte[Sizeof.LONG];
        inputStream.readFully(buffer);
        ByteBuffer wrapped = ByteBuffer.wrap(buffer);
        wrapped.order(ByteOrder.LITTLE_ENDIAN);
        retVal = wrapped.getLong();
        return retVal;
    }

    public static int readInt(RandomAccessFile inputStream) throws IOException {
        int retVal;
        byte[] buffer = new byte[Sizeof.INT];
        inputStream.readFully(buffer);
        ByteBuffer wrapped = ByteBuffer.wrap(buffer);
        wrapped.order(ByteOrder.LITTLE_ENDIAN);
        retVal = wrapped.getInt();
        return retVal;
    }

    public static float readFloat(RandomAccessFile inputStream) throws IOException {
        float retVal;
        byte[] buffer = new byte[Sizeof.FLOAT];
        inputStream.readFully(buffer);
        ByteBuffer wrapped = ByteBuffer.wrap(buffer);
        wrapped.order(ByteOrder.LITTLE_ENDIAN);
        retVal = wrapped.getFloat();
        return retVal;
    }

    public static void skipFloat(RandomAccessFile inputStream, int length) throws IOException {
        byte[] buffer = new byte[length * Sizeof.FLOAT];
        inputStream.readFully(buffer);
        ByteBuffer wrapped = ByteBuffer.wrap(buffer);
        wrapped.order(ByteOrder.LITTLE_ENDIAN);
        wrapped.getFloat();
    }

    public static void readFloat(RandomAccessFile inputStream, Tensor data) throws IOException {
        for (int i = 0; i < data.data.length; i++) {
            data.data[i] = readFloat(inputStream);
        }
        if (data.isHasGPU()) {
            data.hostToDevice();
        }
    }

    public static void readFloat(RandomAccessFile inputStream, float[] data) throws IOException {
        for (int i = 0; i < data.length; i++) {
            data[i] = readFloat(inputStream);
        }
    }
}

