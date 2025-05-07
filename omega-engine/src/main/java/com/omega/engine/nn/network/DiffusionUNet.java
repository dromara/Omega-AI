package com.omega.engine.nn.network;

import com.omega.common.data.Tensor;
import com.omega.engine.gpu.CUDAMemoryManager;
import com.omega.engine.loss.LossFactory;
import com.omega.engine.loss.LossType;
import com.omega.engine.nn.layer.*;
import com.omega.engine.nn.layer.active.SiLULayer;
import com.omega.engine.nn.layer.diffusion.ResidualBlockLayer;
import com.omega.engine.nn.layer.diffusion.TimeEmbeddingLayer;
import com.omega.engine.nn.layer.diffusion.UpSampleLayer;
import com.omega.engine.nn.layer.normalization.BNType;
import com.omega.engine.nn.layer.normalization.GNLayer;
import com.omega.engine.updater.UpdaterType;
import jcuda.Sizeof;
import jcuda.runtime.JCuda;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.util.ArrayList;
import java.util.List;
import java.util.Stack;

/**
 * Duffsion-UNet
 *
 * @author Administrator
 */
public class DiffusionUNet extends Network {
    public int width;
    public int height;
    private int T;
    private int inChannel;
    private int mChannel;
    private int[] channelMult;
    private int resBlockNum;
    private int tdim;
    private boolean bias = true;
    private Stack<Integer> chs = new Stack<Integer>();
    private Stack<Layer> hs = new Stack<Layer>();
    private InputLayer inputLayer;
    private TimeEmbeddingLayer temb;
    private ConvolutionLayer head;
    private List<Layer> downBlocks = new ArrayList<Layer>();
    private ResidualBlockLayer midResBlock1;
    private ResidualBlockLayer midResBlock2;
    //	private BNLayer gn;
    private List<Layer> upBlocks = new ArrayList<Layer>();
    private GNLayer gn;
    private SiLULayer act;
    private ConvolutionLayer conv;
    private Tensor d_temb;

    public DiffusionUNet(LossType lossType, UpdaterType updater, int T, int inChannel, int mChannel, int[] channelMult, int resBlockNum, int width, int height, boolean bias) {
        this.lossFunction = LossFactory.create(lossType, this);
        this.bias = bias;
        this.updater = updater;
        this.T = T;
        this.width = width;
        this.height = height;
        this.inChannel = inChannel;
        this.mChannel = mChannel;
        this.channelMult = channelMult;
        this.resBlockNum = resBlockNum;
        this.tdim = mChannel * 4;
        initLayers();
    }

    public void initLayers() {
        this.inputLayer = new InputLayer(inChannel, height, width);
        temb = new TimeEmbeddingLayer(T, mChannel, tdim, bias, this);
        head = new ConvolutionLayer(inChannel, mChannel, width, height, 3, 3, 1, 1, true, this);
        chs.push(mChannel);
        hs.push(getHead());
        /**
         * downBlocks

         */
        int now_ch = mChannel;
        int oHeight = height;
        int oWidth = width;
        for (int i = 0; i < channelMult.length; i++) {
            int mult = channelMult[i];
            int out_ch = mChannel * mult;
            for (int r = 0; r < resBlockNum; r++) {
                boolean attn = false;
                if (i == 2) {
                    attn = true;
                }
                ResidualBlockLayer rbl = new ResidualBlockLayer(now_ch, out_ch, oHeight, oWidth, tdim, attn, bias, this);
                getDownBlocks().add(rbl);
                now_ch = out_ch;
                chs.push(now_ch);
                hs.push(rbl);
                oHeight = rbl.oHeight;
                oWidth = rbl.oWidth;
            }
            now_ch = out_ch;
            if (i != channelMult.length - 1) {
                ConvolutionLayer down = new ConvolutionLayer(now_ch, now_ch, oWidth, oHeight, 3, 3, 1, 2, bias, this);
                getDownBlocks().add(down);
                chs.push(now_ch);
                hs.push(down);
                oHeight = down.oHeight;
                oWidth = down.oWidth;
            }
        }
        //		System.out.println(JsonUtils.toJson(chs));
        /**
         * middleBlocks

         */
        midResBlock1 = new ResidualBlockLayer(now_ch, now_ch, oHeight, oWidth, tdim, true, bias, this);
        midResBlock2 = new ResidualBlockLayer(now_ch, now_ch, oHeight, oWidth, tdim, false, bias, this);
        /**
         * upBlocks

         */
        int index = 0;
        for (int i = channelMult.length - 1; i >= 0; i--) {
            int mult = channelMult[i];
            int out_ch = mChannel * mult;
            for (int r = 0; r < resBlockNum + 1; r++) {
                int in_ch = chs.pop();
                boolean attn = false;
                if (i == 2) {
                    attn = true;
                }
                RouteLayer rl = null;
                if (index == 0) {
                    rl = new RouteLayer(new Layer[]{getMidResBlock2(), hs.pop()});
                } else {
                    //					System.err.println(upBlocks.get(index - 1));
                    rl = new RouteLayer(new Layer[]{getUpBlocks().get(index - 1), hs.pop()});
                }
                getUpBlocks().add(rl);
                ResidualBlockLayer rbl = new ResidualBlockLayer(in_ch + now_ch, out_ch, oHeight, oWidth, tdim, attn, bias, this);
                getUpBlocks().add(rbl);
                now_ch = out_ch;
                index = index + 2;
                oHeight = rbl.oHeight;
                oWidth = rbl.oWidth;
            }
            if (i != 0) {
                UpSampleLayer up = new UpSampleLayer(now_ch, now_ch, oHeight, oWidth, this);
                getUpBlocks().add(up);
                index++;
                oHeight = up.oHeight;
                oWidth = up.oWidth;
            }
        }
        //		for(Layer layer:upBlocks) {
        //			System.out.println(layer);
        //		}
        //		System.err.println("------------");
        /**
         * tail

         */
        gn = new GNLayer(32, this, BNType.conv_bn);
        //		gn = new BNLayer(this, BNType.conv_bn);
        act = new SiLULayer(getGn());
        conv = new ConvolutionLayer(now_ch, inChannel, getUpBlocks().get(getUpBlocks().size() - 1).oWidth, getUpBlocks().get(getUpBlocks().size() - 1).oHeight, 3, 3, 1, 1, bias, this);
        this.addLayer(inputLayer);
        this.addLayer(getHead());
        this.addLayer(getConv());
    }

    @Override
    public void init() throws Exception {
        // TODO Auto-generated method stub
        if (layerList.size() <= 1) {
            throw new Exception("layer size must greater than 2.");
        }
        this.layerCount = layerList.size();
        this.setChannel(layerList.get(0).channel);
        this.setHeight(layerList.get(0).height);
        this.setWidth(layerList.get(0).width);
        this.oChannel = this.getLastLayer().oChannel;
        this.oHeight = this.getLastLayer().oHeight;
        this.oWidth = this.getLastLayer().oWidth;
        if (layerList.get(0).getLayerType() != LayerType.input) {
            throw new Exception("first layer must be input layer.");
        }
        if ((layerList.get(layerList.size() - 1).getLayerType() == LayerType.softmax || layerList.get(layerList.size() - 1).getLayerType() == LayerType.softmax_cross_entropy) && this.lossFunction.getLossType() != LossType.cross_entropy) {
            throw new Exception("The softmax function support only cross entropy loss function now.");
        }
        System.out.println("the network is ready.");
    }

    @Override
    public NetworkType getNetworkType() {
        // TODO Auto-generated method stub
        return NetworkType.DUFFSION_UNET;
    }

    @Override
    public Tensor predict(Tensor input) {
        // TODO Auto-generated method stub
        this.RUN_MODEL = RunModel.TEST;
        this.forward(input);
        return this.getOutput();
    }

    @Override
    public Tensor forward(Tensor input) {
        return null;
    }

    public Tensor forward(Tensor input, Tensor t) {
        //		System.out.println("en_time:"+en_time+",de_time:"+de_time);
        //		input.showDMByOffset(0, 500);
        //		System.err.println("input:");
        //		input.showDMByOffset(0, 96);
        /**
         * 设置输入数据

         */
        this.setInputData(input);
        /**
         * timestep embedding

         */
        temb.forward(t);
        /**
         * head

         */
        head.forward(input);
        //		System.err.println("ho:");
        //		head.weight.showDM();
        //		head.getOutput().showDMByOffset(0, 96);
        //		System.out.println("head:"+MatrixOperation.isNaN(tmp.syncHost()));
        /**
         * downsampling

         */
        for (int i = 0; i < getDownBlocks().size(); i++) {
            Layer layer = getDownBlocks().get(i);
            if (layer instanceof ResidualBlockLayer) {
                ResidualBlockLayer rbl = (ResidualBlockLayer) layer;
                if (i == 0) {
                    rbl.forward(head.getOutput(), getTemb().getOutput());
                } else {
                    Layer preLayer = getDownBlocks().get(i - 1);
                    rbl.forward(preLayer.getOutput(), getTemb().getOutput());
                }
            } else {
                Layer preLayer = getDownBlocks().get(i - 1);
                layer.forward(preLayer.getOutput());
            }
            //			System.out.println("downsampling tmp:"+MatrixOperation.isNaN(tmp.syncHost()));
        }
        //		System.out.println(MatrixOperation.isNaN(tmp.syncHost()));
        //		System.out.println("downsampling:"+MatrixOperation.isNaN(tmp.syncHost()));
        /**
         * middle

         */
        midResBlock1.forward(getDownBlocks().get(getDownBlocks().size() - 1).getOutput(), getTemb().getOutput());
        midResBlock2.forward(getMidResBlock1().getOutput(), getTemb().getOutput());
        //		System.out.println(MatrixOperation.isNaN(tmp.syncHost()));
        //		tmp.showDMByOffset(0, 100);
        //		temb.getOutput().showDMByOffset(0, 100);
        /**
         * upsampling

         */
        //		int i = 0;
        for (int i = 0; i < getUpBlocks().size(); i++) {
            Layer layer = getUpBlocks().get(i);
            //			System.out.println(layer.getClass());
            if (layer instanceof ResidualBlockLayer) {
                ResidualBlockLayer rbl = (ResidualBlockLayer) layer;
                if (i == 0) {
                    rbl.forward(getMidResBlock2().getOutput(), getTemb().getOutput());
                } else {
                    Layer preLayer = getUpBlocks().get(i - 1);
                    rbl.forward(preLayer.getOutput(), getTemb().getOutput());
                }
                //				System.err.println(i+"[tmp]:"+MatrixOperation.isNaN(tmp.syncHost()));
            } else if (layer instanceof RouteLayer) {
                RouteLayer rl = (RouteLayer) layer;
                rl.forward();
            } else {
                Layer preLayer = getUpBlocks().get(i - 1);
                layer.forward(preLayer.getOutput());
            }
            //			System.err.println(i+":"+MatrixOperation.isNaN(tmp.syncHost()));
            //			i++;
        }
        //		tmp.showShape();
        //		System.out.println(MatrixOperation.isNaN(tmp.syncHost()));
        /**
         * tail

         */
        gn.forward(getUpBlocks().get(getUpBlocks().size() - 1).getOutput());
        act.forward(getGn().getOutput());
        conv.forward(act.getOutput());
        //		System.err.println("------");
        //		System.err.println("output:");
        //		this.conv.getOutput().showDMByOffset(0, 96);
        //		System.err.println("------");
        return this.getConv().getOutput();
    }

    public void initBack() {
        if (d_temb == null || d_temb.number != this.number) {
            d_temb = Tensor.createGPUTensor(d_temb, this.number, getTemb().getOutput().channel, getTemb().getOutput().height, getTemb().getOutput().width, true);
        }
    }

    @Override
    public void back(Tensor lossDiff) {
        // TODO Auto-generated method stub
        //		lossDiff.showDMByNumber(0);
        initBack();
        /**
         * 设置误差
         * 将误差值输入到最后一层

         */
        this.setLossDiff(lossDiff);
        //		System.err.println("lossDiff:");
        //		lossDiff.showDMByOffset(0, 96*96);
        /**
         * tail backward

         */
        getConv().back(lossDiff);
        act.back(getConv().diff);
        //		act.diff.showDMByOffset(0, 500);
        getGn().back(act.diff);
        //		gn.diff.showDMByOffset(0, 500);
        /**
         * upsampling backward

         */
        //		System.err.println("back:");
        //		System.out.println(MatrixOperation.isNaN(tmp.syncHost()));
        for (int i = getUpBlocks().size() - 1; i >= 0; i--) {
            Layer layer = getUpBlocks().get(i);
            if (layer instanceof ResidualBlockLayer) {
                ResidualBlockLayer rbl = (ResidualBlockLayer) layer;
                if (i == getUpBlocks().size() - 1) {
                    rbl.back(getGn().diff, d_temb);
                } else {
                    rbl.back(getUpBlocks().get(i + 1).diff, d_temb);
                }
            } else {
                layer.back(getUpBlocks().get(i + 1).diff);
            }
        }
        //		System.out.println(MatrixOperation.isNaN(tmp.syncHost()));
        //		System.err.println("====upsampling tmp===");
        //		tmp.showDMByOffset(0, 500);
        /**
         * middle backward

         */
        getMidResBlock2().back(getUpBlocks().get(0).diff, d_temb);
        getMidResBlock1().back(getMidResBlock2().diff, d_temb);
        //		System.out.println(MatrixOperation.isNaN(tmp.syncHost()));
        //		System.err.println("====middle tmp===");
        //		tmp.showDMByOffset(0, 500);
        /**
         * downsampling backward

         */
        for (int i = getDownBlocks().size() - 1; i >= 0; i--) {
            Layer layer = getDownBlocks().get(i);
            if (layer instanceof ResidualBlockLayer) {
                ResidualBlockLayer rbl = (ResidualBlockLayer) layer;
                if (i == getDownBlocks().size() - 1) {
                    rbl.back(getMidResBlock1().diff, d_temb);
                } else {
                    rbl.back(getDownBlocks().get(i + 1).diff, d_temb);
                }
            } else {
                layer.back(getDownBlocks().get(i + 1).diff);
            }
        }
        //		System.err.println("====down tmp===");
        //		tmp.showDMByOffset(0, 500);
        //		System.out.println(MatrixOperation.isNaN(tmp.syncHost()));
        /**
         * head backward

         */
        head.back(getDownBlocks().get(0).diff);
        //		head.delta.showDMByOffset(0, 500);
        /**
         * timestep embedding backward

         */
        getTemb().back(d_temb);
        //		d_temb.showDMByOffset(0, 500);
        d_temb.clearGPU();
    }

    @Override
    public Tensor loss(Tensor output, Tensor label) {
        // TODO Auto-generated method stub
        switch (this.getLastLayer().getLayerType()) {
            case softmax:
                //			SoftmaxLayer softmaxLayer = (SoftmaxLayer)this.getLastLayer();
                //			softmaxLayer.setCurrentLabel(label);
                break;
            case softmax_cross_entropy:
                SoftmaxWithCrossEntropyLayer softmaxWithCrossEntropyLayer = (SoftmaxWithCrossEntropyLayer) this.getLastLayer();
                softmaxWithCrossEntropyLayer.setCurrentLabel(label);
                break;
            default:
                break;
        }
        return this.lossFunction.loss(output, label);
    }

    @Override
    public Tensor lossDiff(Tensor output, Tensor label) {
        // TODO Auto-generated method stub
        this.clearGrad();
        Tensor t = this.lossFunction.diff(output, label);
        //		PrintUtils.printImage(t.data);
        return t;
    }

    public void update() {
        this.train_time += 1;
        getTemb().update();
        getHead().update();
        for (Layer layer : getDownBlocks()) {
            layer.update();
        }
        getMidResBlock1().update();
        getMidResBlock2().update();
        for (Layer layer : getUpBlocks()) {
            layer.update();
        }
        getGn().update();
        getConv().update();
    }

    @Override
    public void clearGrad() {
        // TODO Auto-generated method stub
        /**
         * forward

         */
        JCuda.cudaMemset(CUDAMemoryManager.workspace.getPointer(), 0, CUDAMemoryManager.workspace.getSize() * Sizeof.FLOAT);
        for (int i = 0; i < layerCount; i++) {
            Layer layer = layerList.get(i);
            if (layer.cache_delta != null) {
                layer.cache_delta.clearGPU();
            }
        }
        for (int i = 0; i < getDownBlocks().size(); i++) {
            if (getDownBlocks().get(i).cache_delta != null) {
                getDownBlocks().get(i).cache_delta.clearGPU();
            }
        }
        for (int i = 0; i < getUpBlocks().size(); i++) {
            if (getUpBlocks().get(i).cache_delta != null) {
                getUpBlocks().get(i).cache_delta.clearGPU();
            }
        }
        if (getMidResBlock1().cache_delta != null) {
            getMidResBlock1().cache_delta.clearGPU();
        }
        if (getMidResBlock2().cache_delta != null) {
            getMidResBlock2().cache_delta.clearGPU();
        }
        JCuda.cudaDeviceSynchronize();
    }

    @Override
    public Tensor loss(Tensor output, Tensor label, Tensor loss) {
        // TODO Auto-generated method stub
        return this.lossFunction.loss(output, label, loss);
    }

    @Override
    public Tensor lossDiff(Tensor output, Tensor label, Tensor diff) {
        // TODO Auto-generated method stub
        this.clearGrad();
        return this.lossFunction.diff(output, label, diff);
    }

    public Tensor loss(Tensor output, Tensor label, int igonre) {
        // TODO Auto-generated method stub
        return this.lossFunction.loss(output, label, igonre);
    }

    public Tensor lossDiff(Tensor output, Tensor label, int igonre) {
        // TODO Auto-generated method stub
        return this.lossFunction.diff(output, label, igonre);
    }

    public TimeEmbeddingLayer getTemb() {
        return temb;
    }

    public List<Layer> getDownBlocks() {
        return downBlocks;
    }

    public ResidualBlockLayer getMidResBlock1() {
        return midResBlock1;
    }

    public ResidualBlockLayer getMidResBlock2() {
        return midResBlock2;
    }

    public List<Layer> getUpBlocks() {
        return upBlocks;
    }

    public ConvolutionLayer getHead() {
        return head;
    }

    public GNLayer getGn() {
        return gn;
    }

    public ConvolutionLayer getConv() {
        return conv;
    }

    public void saveModel(RandomAccessFile outputStream) throws IOException {
        temb.saveModel(outputStream);
        head.saveModel(outputStream);
        System.out.println("head save success...");
        for (int i = 0; i < downBlocks.size(); i++) {
            Layer layer = downBlocks.get(i);
            if (layer instanceof ResidualBlockLayer) {
                ResidualBlockLayer rbl = (ResidualBlockLayer) layer;
                rbl.saveModel(outputStream);
            } else if (layer instanceof ConvolutionLayer) {
                ConvolutionLayer conv = (ConvolutionLayer) layer;
                conv.saveModel(outputStream);
            }
        }
        System.out.println("downBlocks save success...");
        midResBlock1.saveModel(outputStream);
        midResBlock2.saveModel(outputStream);
        System.out.println("midResBlocks save success...");
        for (int i = 0; i < upBlocks.size(); i++) {
            Layer layer = downBlocks.get(i);
            if (layer instanceof ResidualBlockLayer) {
                ResidualBlockLayer rbl = (ResidualBlockLayer) layer;
                rbl.saveModel(outputStream);
            } else if (layer instanceof ConvolutionLayer) {
                ConvolutionLayer conv = (ConvolutionLayer) layer;
                conv.saveModel(outputStream);
            } else if (layer instanceof UpSampleLayer) {
                UpSampleLayer up = (UpSampleLayer) layer;
                up.saveModel(outputStream);
            }
        }
        System.out.println("upBlocks save success...");
        gn.saveModel(outputStream);
        conv.saveModel(outputStream);
        System.out.println("tail save success...");
    }

    public void loadModel(RandomAccessFile inputStream) throws IOException {
        temb.loadModel(inputStream);
        head.loadModel(inputStream);
        System.out.println("head load success...");
        for (int i = 0; i < downBlocks.size(); i++) {
            Layer layer = downBlocks.get(i);
            if (layer instanceof ResidualBlockLayer) {
                ResidualBlockLayer rbl = (ResidualBlockLayer) layer;
                rbl.loadModel(inputStream);
            } else if (layer instanceof ConvolutionLayer) {
                ConvolutionLayer conv = (ConvolutionLayer) layer;
                conv.loadModel(inputStream);
            }
        }
        System.out.println("downBlocks load success...");
        midResBlock1.loadModel(inputStream);
        midResBlock2.loadModel(inputStream);
        System.out.println("midResBlocks load success...");
        for (int i = 0; i < upBlocks.size(); i++) {
            Layer layer = downBlocks.get(i);
            if (layer instanceof ResidualBlockLayer) {
                ResidualBlockLayer rbl = (ResidualBlockLayer) layer;
                rbl.loadModel(inputStream);
            } else if (layer instanceof ConvolutionLayer) {
                ConvolutionLayer conv = (ConvolutionLayer) layer;
                conv.loadModel(inputStream);
            } else if (layer instanceof UpSampleLayer) {
                UpSampleLayer up = (UpSampleLayer) layer;
                up.loadModel(inputStream);
            }
        }
        System.out.println("upBlocks load success...");
        gn.loadModel(inputStream);
        conv.loadModel(inputStream);
        System.out.println("tail load success...");
    }

    @Override
    public void putParamters() {
        // TODO Auto-generated method stub
    }

    @Override
    public void putParamterGrads() {
        // TODO Auto-generated method stub
    }
}

