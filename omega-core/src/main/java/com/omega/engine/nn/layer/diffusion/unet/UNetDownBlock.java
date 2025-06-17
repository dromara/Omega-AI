package com.omega.engine.nn.layer.diffusion.unet;

import com.omega.common.utils.RandomUtils;
import com.omega.engine.nn.layer.ConvolutionLayer;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.network.Network;
import com.omega.engine.nn.network.RunModel;
import com.omega.engine.nn.network.Transformer;
import com.omega.engine.tensor.Tensor;
import com.omega.engine.updater.UpdaterType;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.util.Map;
import java.util.Stack;

/**
 * UNetDownBlock
 *
 * @author Administrator
 */
public class UNetDownBlock extends Layer {
    public UNetResidualBlock[] res;
    public UNetAttentionBlock[] attns;
    public ConvolutionLayer down;
    private int numLayer = 1;
    private int headNum = 4;
    private int contextTime = 0;
    private int contextDim = 0;
    private int nTime = 0;
    private int groupNum = 32;

    public UNetDownBlock(int channel, int oChannel, int height, int width, int nTime, int headNum, int contextTime, int contextDim, int groupNum, int numLayer, Network network) {
        this.network = network;
        this.numLayer = numLayer;
        this.nTime = nTime;
        this.groupNum = groupNum;
        this.headNum = headNum;
        this.contextTime = contextTime;
        this.contextDim = contextDim;
        this.channel = channel;
        this.oChannel = oChannel;
        this.height = height;
        this.width = width;
        initLayers();
    }

    public UNetDownBlock(String name, int channel, int oChannel, int height, int width, int nTime, int headNum, int contextTime, int contextDim, int groupNum, int numLayer, Network network) {
        this.setName(name);
        this.network = network;
        this.numLayer = numLayer;
        this.nTime = nTime;
        this.groupNum = groupNum;
        this.headNum = headNum;
        this.contextTime = contextTime;
        this.contextDim = contextDim;
        this.channel = channel;
        this.oChannel = oChannel;
        this.height = height;
        this.width = width;
        initLayers();
    }

    public static void loadWeight(Map<String, Object> weightMap, UNetDownBlock network, boolean showLayers) {
        if (showLayers) {
            for (String key : weightMap.keySet()) {
                System.out.println(key);
            }
        }
        //
        //		network.gn_feature.gamma = ClipModelUtils.loadData(network.gn_feature.gamma, weightMap, 1, "groupnorm_feature.weight");
        //		network.gn_feature.beta = ClipModelUtils.loadData(network.gn_feature.beta, weightMap, 1, "groupnorm_feature.bias");
        //
        //		ClipModelUtils.loadData(network.conv_feature.weight, weightMap, "conv_feature.weight");
        //		ClipModelUtils.loadData(network.conv_feature.bias, weightMap, "conv_feature.bias");
        //
        //		ClipModelUtils.loadData(network.temb.linear.weight, weightMap, "linear_time.weight");
        //		ClipModelUtils.loadData(network.temb.linear.bias, weightMap, "linear_time.bias");
        //
        //		network.gn_merged.gamma = ClipModelUtils.loadData(network.gn_merged.gamma, weightMap, 1, "groupnorm_merged.weight");
        //		network.gn_merged.beta = ClipModelUtils.loadData(network.gn_merged.beta, weightMap, 1, "groupnorm_merged.bias");
        //
        //		ClipModelUtils.loadData(network.conv_merged.weight, weightMap, "conv_merged.weight");
        //		ClipModelUtils.loadData(network.conv_merged.bias, weightMap, "conv_merged.bias");
        //
        //		network.residual_layer.weight = ClipModelUtils.loadData(network.residual_layer.weight, weightMap, 4, "residual_layer.weight");
        //		ClipModelUtils.loadData(network.residual_layer.bias, weightMap, "residual_layer.bias");
    }

    public static void main(String[] args) {
        int batchSize = 2;
        int channel = 64;
        int height = 32;
        int width = 32;
        int oChannel = 128;
        int timeDim = 64;
        Transformer tf = new Transformer();
        tf.updater = UpdaterType.adamw;
        tf.CUDNN = true;
        tf.learnRate = 0.001f;
        tf.RUN_MODEL = RunModel.TRAIN;
        tf.number = batchSize;
        float[] data = RandomUtils.order(batchSize * channel * height * width, 0.1f, 0.1f);
        Tensor input = new Tensor(batchSize, channel, height, width, data, true);
        float[] tdata = RandomUtils.order(batchSize * timeDim, 0.1f, 0.1f);
        Tensor time = new Tensor(batchSize, 1, 1, timeDim, tdata, true);
        float[] delta_data = RandomUtils.order(batchSize * oChannel * height * width, 0.01f, 0.01f);
        Tensor delta = new Tensor(batchSize, oChannel, height, width, delta_data, true);
        Tensor timeDiff = new Tensor(batchSize, 1, 1, timeDim, true);
        //		UNetDownBlock block = new UNetDownBlock(channel, oChannel, height, width, timeDim, 32, tf);
        //
        //		String weight = "H:\\model\\resnet_block.json";
        //		loadWeight(LagJsonReader.readJsonFileSmallWeight(weight), block, true);
        //
        //		for(int i = 0;i<10;i++) {
        ////			input.showDM();
        //			tf.train_time++;
        //			block.forward(input, time);
        //
        //			block.getOutput().showShape();
        //
        //			block.getOutput().showDM();
        //
        //			block.back(delta, timeDiff);
        ////			delta.showDM();
        //			block.diff.showDM();
        //			timeDiff.showDM();
        ////			block.gn.diffGamma.showDM("dgamma");
        ////			block.gn.diffBeta.showDM("dbeta");
        //			block.update();
        ////			block.gn.gamma.showDM("gamma");
        ////			block.gn.beta.showDM("beta");
        ////			delta.copyData(tmp);
        //		}
    }

    public void initLayers() {
        res = new UNetResidualBlock[numLayer];
        attns = new UNetAttentionBlock[numLayer];
        int ic = channel;
        int oc = oChannel;
        for (int i = 0; i < numLayer; i++) {
            UNetResidualBlock rb = new UNetResidualBlock(ic, oc, height, width, nTime, groupNum, network);
            UNetAttentionBlock attn = new UNetAttentionBlock(oc, height, width, headNum, contextTime, contextDim, groupNum, network);
            attn.setName("[" + name + "]donw-attn-" + i);
            res[i] = rb;
            attns[i] = attn;
            ic = oc;
        }
        down = new ConvolutionLayer(oChannel, oChannel, width, height, 3, 3, 1, 2, true, network);
        this.oHeight = down.oHeight;
        this.oWidth = down.oWidth;
    }

    public void pushStack(Stack<Layer> downLayers) {
        for (UNetAttentionBlock attn : attns) {
            downLayers.push(attn);
        }
        //		downLayers.push(down);
    }

    @Override
    public void init() {
        // TODO Auto-generated method stub
        this.number = this.network.number;
    }

    public void init(Tensor input) {
        // TODO Auto-generated method stub
        this.number = input.getShape()[0];
    }

    @Override
    public void initBack() {
        // TODO Auto-generated method stub
    }

    @Override
    public void initParam() {
        // TODO Auto-generated method stub
    }

    @Override
    public void output() {
        // TODO Auto-generated method stub
    }

    public void output(Tensor time) {
        // TODO Auto-generated method stub
        Tensor x = input;
        for (int i = 0; i < numLayer; i++) {
            //			x.showDM("in====>");
            res[i].forward(x, time);
            //			res[i].getOutput().showDM("res-out");
            attns[i].forward(res[i].getOutput());
            //			attns[i].gn.getOutput().showDM("gn");
            x = attns[i].getOutput();
            //			x.showDM("attn-out");
        }
        down.forward(x);
        this.output = down.getOutput();
    }

    public void output(Tensor time, Tensor context) {
        // TODO Auto-generated method stub
        Tensor x = input;
        for (int i = 0; i < numLayer; i++) {
            //			x.showDM("in====>");
            res[i].forward(x, time);
            //			res[i].getOutput().showDM("res-out");
            attns[i].forward(res[i].getOutput(), context);
            //			attns[i].gn.getOutput().showDM("gn");
            x = attns[i].getOutput();
            //			x.showDM("attn-out");
        }
        down.forward(x);
        this.output = down.getOutput();
    }

    @Override
    public Tensor getOutput() {
        // TODO Auto-generated method stub
        return this.output;
    }

    @Override
    public void diff() {
        // TODO Auto-generated method stub
    }

    public void diff(Tensor timeDiff) {
        // TODO Auto-generated method stub
        down.back(delta);
        Tensor d = down.diff;
        for (int i = numLayer - 1; i >= 0; i--) {
            attns[i].back(d);
            res[i].back(attns[i].diff, timeDiff);
            d = res[i].diff;
        }
        this.diff = d;
    }

    @Override
    public void forward() {
        // TODO Auto-generated method stub
        /**
         * 参数初始化

         */
        this.init();
        /**
         * 设置输入

         */
        this.setInput();
        /**
         * 计算输出

         */
        this.output();
    }

    @Override
    public void back() {
        // TODO Auto-generated method stub
        initBack();
        /**
         * 设置梯度

         */
        this.setDelta();
        /**
         * 计算梯度

         */
        this.diff();
    }

    @Override
    public void backTemp() {
        // TODO Auto-generated method stub
    }

    @Override
    public void forward(Tensor input) {
        // TODO Auto-generated method stub
        /**
         * 参数初始化

         */
        this.init(input);
        /**
         * 设置输入

         */
        this.setInput(input);
        /**
         * 计算输出

         */
        this.output();
    }

    public void forward(Tensor input, Tensor time) {
        // TODO Auto-generated method stub
        /**
         * 参数初始化

         */
        this.init(input);
        /**
         * 设置输入

         */
        this.setInput(input);
        /**
         * 计算输出

         */
        this.output(time);
    }

    public void forward(Tensor input, Tensor time, Tensor context) {
        // TODO Auto-generated method stub
        /**
         * 参数初始化

         */
        this.init(input);
        /**
         * 设置输入

         */
        this.setInput(input);
        /**
         * 计算输出

         */
        this.output(time, context);
    }

    @Override
    public void back(Tensor delta) {
        // TODO Auto-generated method stub
        initBack();
        /**
         * 设置梯度

         */
        this.setDelta(delta);
        /**
         * 计算梯度

         */
        this.diff();
    }

    public void back(Tensor delta, Tensor timeDiff) {
        // TODO Auto-generated method stub
        initBack();
        /**
         * 设置梯度

         */
        this.setDelta(delta);
        /**
         * 计算梯度

         */
        this.diff(timeDiff);
    }

    @Override
    public void update() {
        // TODO Auto-generated method stub
        for (int i = 0; i < numLayer; i++) {
            res[i].update();
            attns[i].update();
        }
        down.update();
    }

    @Override
    public void showDiff() {
        // TODO Auto-generated method stub
    }

    @Override
    public LayerType getLayerType() {
        // TODO Auto-generated method stub
        return LayerType.unet_down;
    }

    @Override
    public float[][][][] output(float[][][][] input) {
        // TODO Auto-generated method stub
        return null;
    }

    @Override
    public void initCache() {
        // TODO Auto-generated method stub
    }

    @Override
    public void accGrad(float scale) {
        // TODO Auto-generated method stub
        for (int i = 0; i < numLayer; i++) {
            res[i].accGrad(scale);
            attns[i].accGrad(scale);
        }
        down.accGrad(scale);
    }

    public void saveModel(RandomAccessFile outputStream) throws IOException {
        for (int i = 0; i < numLayer; i++) {
            res[i].saveModel(outputStream);
            attns[i].saveModel(outputStream);
        }
        down.saveModel(outputStream);
    }

    public void loadModel(RandomAccessFile inputStream) throws IOException {
        for (int i = 0; i < numLayer; i++) {
            res[i].loadModel(inputStream);
            attns[i].loadModel(inputStream);
        }
        down.loadModel(inputStream);
    }
}

