package com.omega.common.data.utils;//package com.omega.common.data.utils;
//
//import java.io.File;
//import java.io.FileInputStream;
//import java.io.FileOutputStream;
//import java.io.IOException;
//import java.io.InputStreamReader;
//import java.io.OutputStreamWriter;
//import java.io.Reader;
//import java.io.Writer;
//import java.nio.charset.StandardCharsets;
//
//import com.alibaba.fastjson.JSON;
//import com.alibaba.fastjson.serializer.SerializerFeature;
//import com.omega.common.data.LayerConfig;
//import com.omega.common.data.NetworkConfig;
//import com.omega.common.utils.JsonUtils;
//import com.omega.engine.nn.layer.ConvolutionLayer;
//import com.omega.engine.nn.layer.FullyLayer;
//import com.omega.engine.nn.layer.InputLayer;
//import com.omega.engine.nn.layer.Layer;
//import com.omega.engine.nn.layer.PoolingLayer;
//import com.omega.engine.nn.layer.SoftmaxWithCrossEntropyLayer;
//import com.omega.engine.nn.layer.active.LeakyReluLayer;
//import com.omega.engine.nn.layer.active.ReluLayer;
//import com.omega.engine.nn.layer.active.SigmodLayer;
//import com.omega.engine.nn.layer.normalization.BNLayer;
//import com.omega.engine.nn.network.BPNetwork;
//import com.omega.engine.nn.network.CNN;
//import com.omega.engine.nn.network.Network;
//import com.omega.engine.pooling.PoolingType;
//
//public class NetworkUtils {
//
//
//	public static void save(Network nn,String path,String name) {
//
//		if(nn == null) {
//			throw new RuntimeException("the network must be not null.");
//		}
//
//		if(path == null || path.equals("")) {
//			throw new RuntimeException("the path must be not null.");
//		}
//
//		NetworkConfig nc = new NetworkConfig();
//		nc.setName(name);
//		nc.setNetworkType(nn.getNetworkType().getKey());
//
//		for(Layer layer:nn.layerList) {
//			LayerConfig lc = new LayerConfig();
//			lc.setIndex(layer.index);
//			System.out.println(layer.index+":"+layer.getLayerType());
//			lc.setLayerType(layer.getLayerType().getKey());
//			lc.setNumber(layer.number);
//			lc.setChannel(layer.channel);
//			lc.setHeight(layer.height);
//			lc.setWidth(layer.width);
//			lc.setoChannel(layer.oChannel);
//			lc.setoHeight(layer.oHeight);
//			lc.setoWidth(layer.oWidth);
//			lc.setHasBias(layer.hasBias);
//
//			switch (layer.getLayerType()) {
//			case input:
//
//				break;
//			case full:
//				lc.setWeight(layer.weight);
//				lc.setBias(layer.bias);
//				break;
//			case conv:
//				ConvolutionLayer cl = (ConvolutionLayer) layer;
//				lc.setWeight(cl.weight);
//				lc.setBias(cl.bias);
//				lc.setkNumber(cl.kernelNum);
//				lc.setStride(cl.stride);
//				lc.setPadding(cl.padding);
//				lc.setpHeight(cl.kHeight);
//				lc.setpWidth(cl.kWidth);
//				break;
//			case pooling:
//				PoolingLayer pl = (PoolingLayer) layer;
//				lc.setpHeight(pl.pHeight);
//				lc.setpWidth(pl.pWidth);
//				lc.setStride(pl.stride);
//				lc.setPoolingType(pl.poolingType.toString());
//				break;
//			case bn:
//				BNLayer bnl = (BNLayer) layer;
//				lc.setGama(bnl.gama);
//				lc.setBeta(bnl.beta);
//				lc.setRuningMean(bnl.getKernel().getRuningMean());
//				lc.setRuningVar(bnl.getKernel().getRuningVar());
//				break;
//			default:
//				break;
//			}
//
//			nc.getLayers().add(lc);
//
//		}
//
//		NetworkUtils.writeJsonFile(nc, path);
//
//	}
//
//	public static void writeJsonFile(Object obj,String path){
//	    String content = JSON.toJSONString(obj, SerializerFeature.PrettyFormat, SerializerFeature.WriteMapNullValue,
//	            SerializerFeature.WriteDateUseDateFormat);
//	    try {
//	        File file = new File(path);
//	        if (file.exists()) {
//	            file.delete();
//	        }
//	        file.createNewFile();
//	        // 写入文件
//	        Writer write = new OutputStreamWriter(new FileOutputStream(file), StandardCharsets.UTF_8);
//	        write.write(content);
//	        write.flush();
//	        write.close();
//	    } catch (Exception e) {
//	        e.printStackTrace();
//	    }
//	}
//
//	public static String readJsonFile(String path) {
//	    String jsonStr = "";
//	    FileInputStream in = null;
//	    try {
//	    	File file = new File(path);
//	    	in = new FileInputStream(file);
//	        Reader reader = new InputStreamReader(in,"utf-8");
//	        int ch = 0;
//	        StringBuffer sb = new StringBuffer();
//	        while ((ch = reader.read()) != -1) {
//	            sb.append((char) ch);
//	        }
//
//	        reader.close();
//	        jsonStr = sb.toString();
//	        return jsonStr;
//	    } catch (IOException e) {
//	        e.printStackTrace();
//	        return null;
//	    }finally {
//	    	if(in!=null) {
//	    		try {
//					in.close();
//				} catch (IOException e) {
//					// TODO Auto-generated catch block
//					e.printStackTrace();
//				}
//	    	}
//	    }
//	}
//
//	public static Network loadNetworkConfig(String path) {
//
//		if(path == null || path.equals("")) {
//			throw new RuntimeException("the path must be not null.");
//		}
//
//		String json = NetworkUtils.readJsonFile(path);
//
//		NetworkConfig nc = new NetworkConfig();
//
//		nc = JsonUtils.gson.fromJson(json, nc.getClass());
//
//		Network nn = null;
//
//		switch (nc.getNetworkType()) {
//		case "BP":
//			nn = new BPNetwork(null);
//			break;
//		case "CNN":
//			nn = new CNN(null);
//			break;
//		case "RNN":
//
//			break;
//		case "ANN":
//
//			break;
//		default:
//			break;
//		}
//
//		if(nn == null) {
//			throw new RuntimeException("not support this network type.");
//		}
//
//		LayerConfig preLayer = null;
//
//		for(int i = 0;i<nc.getLayers().size();i++) {
//			LayerConfig lc = nc.getLayers().get(i);
//
//			Layer l = null;
//			switch (lc.getLayerType()) {
//			case "input":
//				l = new InputLayer(lc.getChannel(), lc.getHeight(), lc.getWidth());
//				break;
//			case "full":
//				l = new FullyLayer(lc.getWidth(), lc.getoWidth(), lc.isHasBias());
//				l.weight = lc.getWeight();
//				l.bias = lc.getBias();
//				break;
//			case "conv":
//				l = new ConvolutionLayer(lc.getChannel(), lc.getkNumber(), preLayer.getoWidth(), preLayer.getoHeight(), lc.getWeight().width, lc.getWeight().height, lc.getPadding(), lc.getStride(), lc.isHasBias());
//				l.weight = lc.getWeight();
//				l.bias = lc.getBias();
//
//				break;
//			case "pooling":
//
//				switch (lc.getPoolingType()) {
//				case "MAX_POOLING":
//					l = new PoolingLayer(lc.getChannel(),preLayer.getoWidth(), preLayer.getoHeight(), lc.getpWidth(), lc.getpHeight(), lc.getStride(), PoolingType.MAX_POOLING);
//					break;
//				default:
//					l = new PoolingLayer(lc.getChannel(),preLayer.getoWidth(), preLayer.getoHeight(), lc.getpWidth(), lc.getpHeight(), lc.getStride(), PoolingType.MEAN_POOLING);
//					break;
//				}
//
//				break;
//			case "bn":
//				l = new BNLayer();
//				BNLayer bnl = (BNLayer) l;
//				bnl.gama = lc.getGama();
//				bnl.beta = lc.getBeta();
//				bnl.hasRuning = false;
//				bnl.runingMean = lc.getRuningMean();
//				bnl.runingVar = lc.getRuningVar();
//				break;
//			case "softmax_cross_entropy":
//				l = new SoftmaxWithCrossEntropyLayer(preLayer.getoWidth());
//				break;
//			case "leakRelu":
//				l = new LeakyReluLayer();
//				break;
//			case "relu":
//				l = new ReluLayer();
//				break;
//			case "sigmod":
//				l = new SigmodLayer();
//				break;
//			default:
//				break;
//			}
//
//			nn.addLayer(l);
//
//			preLayer = lc;
//
//		}
//
//		return nn;
//	}
//
//}
