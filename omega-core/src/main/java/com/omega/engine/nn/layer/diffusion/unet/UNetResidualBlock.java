package com.omega.engine.nn.layer.diffusion.unet;

import com.omega.common.tensor.Tensor;
import com.omega.utils.RandomUtils;
import com.omega.engine.nn.layer.ConvolutionLayer;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.active.SiLULayer;
import com.omega.engine.nn.layer.normalization.BNType;
import com.omega.engine.nn.layer.normalization.GNLayer;
import com.omega.engine.nn.layer.unet.UNetTEmbLayer;
import com.omega.engine.nn.network.Network;
import com.omega.engine.nn.network.RunModel;
import com.omega.engine.nn.network.Transformer;
import com.omega.engine.updater.UpdaterType;
import com.omega.utils.clip.ClipModelUtils;
import com.omega.models.transformer.LagJsonReader;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.util.Map;

/**
 * UNetResidualBlock
 *
 * @author Administrator
 */
public class UNetResidualBlock extends Layer {
    public GNLayer gn_feature;
    public ConvolutionLayer conv_feature;
    public UNetTEmbLayer temb;
    public GNLayer gn_merged;
    public ConvolutionLayer conv_merged;
    public ConvolutionLayer residual_layer;
    private int nTime;
    private int groupNum = 32;
    private SiLULayer act_feature;
    private SiLULayer act_merged;
    private Tensor tout;
    private Tensor dt;

    public UNetResidualBlock(int channel, int oChannel, int height, int width, int nTime, int groupNum, Network network) {
        this.network = network;
        this.nTime = nTime;
        this.groupNum = groupNum;
        this.channel = channel;
        this.oChannel = oChannel;
        this.height = height;
        this.width = width;
        this.oHeight = height;
        this.oWidth = width;
        initLayers();
    }

    public static void loadWeight(Map<String, Object> weightMap, UNetResidualBlock network, boolean showLayers) {
        if (showLayers) {
            for (String key : weightMap.keySet()) {
                System.out.println(key);
            }
        }
        network.gn_feature.gamma = ClipModelUtils.loadData(network.gn_feature.gamma, weightMap, 1, "groupnorm_feature.weight");
        network.gn_feature.beta = ClipModelUtils.loadData(network.gn_feature.beta, weightMap, 1, "groupnorm_feature.bias");
        ClipModelUtils.loadData(network.conv_feature.weight, weightMap, "conv_feature.weight");
        ClipModelUtils.loadData(network.conv_feature.bias, weightMap, "conv_feature.bias");
        ClipModelUtils.loadData(network.temb.linear.weight, weightMap, "linear_time.weight");
        ClipModelUtils.loadData(network.temb.linear.bias, weightMap, "linear_time.bias");
        network.gn_merged.gamma = ClipModelUtils.loadData(network.gn_merged.gamma, weightMap, 1, "groupnorm_merged.weight");
        network.gn_merged.beta = ClipModelUtils.loadData(network.gn_merged.beta, weightMap, 1, "groupnorm_merged.bias");
        ClipModelUtils.loadData(network.conv_merged.weight, weightMap, "conv_merged.weight");
        ClipModelUtils.loadData(network.conv_merged.bias, weightMap, "conv_merged.bias");
        network.residual_layer.weight = ClipModelUtils.loadData(network.residual_layer.weight, weightMap, 4, "residual_layer.weight");
        ClipModelUtils.loadData(network.residual_layer.bias, weightMap, "residual_layer.bias");
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
        UNetResidualBlock block = new UNetResidualBlock(channel, oChannel, height, width, timeDim, 32, tf);
        String weight = "H:\\model\\resnet_block.json";
        loadWeight(LagJsonReader.readJsonFileSmallWeight(weight), block, true);
        for (int i = 0; i < 10; i++) {
            //			input.showDM();
            tf.train_time++;
            block.forward(input, time);
            block.getOutput().showShape();
            block.getOutput().showDM();
            block.back(delta, timeDiff);
            //			delta.showDM();
            block.diff.showDM();
            timeDiff.showDM();
            //			block.gn.diffGamma.showDM("dgamma");
            //			block.gn.diffBeta.showDM("dbeta");
            block.update();
            //			block.gn.gamma.showDM("gamma");
            //			block.gn.beta.showDM("beta");
            //			delta.copyData(tmp);
        }
    }

    public void initLayers() {
        gn_feature = new GNLayer(groupNum, channel, height, width, BNType.conv_bn, this);
        act_feature = new SiLULayer(gn_feature);
        conv_feature = new ConvolutionLayer(channel, oChannel, width, height, 3, 3, 1, 1, true, network);
        temb = new UNetTEmbLayer(nTime, oChannel, network);
        gn_merged = new GNLayer(groupNum, oChannel, height, width, BNType.conv_bn, this);
        act_merged = new SiLULayer(gn_merged);
        conv_merged = new ConvolutionLayer(oChannel, oChannel, width, height, 3, 3, 1, 1, true, network);
        if (channel != oChannel) {
            residual_layer = new ConvolutionLayer(channel, oChannel, width, height, 1, 1, 0, 1, true, this.network);
        }
    }

    @Override
    public void init() {
        // TODO Auto-generated method stub
        this.number = this.network.number;
    }

    public void init(Tensor input) {
        // TODO Auto-generated method stub
        this.number = input.number;
        if (tout == null || tout.number != this.number) {
            tout = Tensor.createGPUTensor(tout, number, oChannel, height, width, true);
        }
        //		if(this.output == null || this.output.number != this.number) {
        //			this.output = Tensor.createGPUTensor(this.output, number, oChannel, oHeight, oWidth, true);
        //		}
    }

    @Override
    public void initBack() {
        // TODO Auto-generated method stub
        if (dt == null || dt.number != this.number) {
            dt = Tensor.createGPUTensor(dt, this.number, 1, 1, oChannel, true);
        }
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
        gn_feature.forward(input);
        act_feature.forward(gn_feature.getOutput());
        conv_feature.forward(act_feature.getOutput());
        temb.forward(time);
        Tensor_OP().add(conv_feature.getOutput(), temb.getOutput(), tout, height * width);
        gn_merged.forward(tout);
        act_merged.forward(gn_merged.getOutput());
        conv_merged.forward(act_merged.getOutput());
        Tensor x = input;
        if (residual_layer != null) {
            residual_layer.forward(input);
            x = residual_layer.getOutput();
        }
        Tensor_OP().add(x, this.conv_merged.getOutput(), this.conv_merged.getOutput());
        this.output = this.conv_merged.getOutput();
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
        conv_merged.back(delta);
        act_merged.back(conv_merged.diff);
        gn_merged.back(act_merged.diff);
        dt.clearGPU();
        Tensor_OP().sum(gn_merged.diff, dt, 2);
        temb.back(dt);
        Tensor_OP().add(timeDiff, temb.diff, timeDiff);
        conv_feature.back(gn_merged.diff);
        act_feature.back(conv_feature.diff);
        gn_feature.back(act_feature.diff);
        if (channel != oChannel) {
            residual_layer.back(delta);
            Tensor_OP().add(gn_feature.diff, residual_layer.diff, gn_feature.diff);
        } else {
            Tensor_OP().add(gn_feature.diff, delta, gn_feature.diff);
        }
        this.diff = gn_feature.diff;
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
        gn_feature.update();
        conv_feature.update();
        temb.update();
        //		temb.linear.weight.showDM("temb_w");
        gn_merged.update();
        conv_merged.update();
        if (residual_layer != null) {
            residual_layer.update();
        }
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
        gn_feature.accGrad(scale);
        act_feature.accGrad(scale);
        conv_feature.accGrad(scale);
        temb.accGrad(scale);
        gn_merged.accGrad(scale);
        act_merged.accGrad(scale);
        conv_merged.accGrad(scale);
        if (residual_layer != null) {
            residual_layer.accGrad(scale);
        }
    }

    public void saveModel(RandomAccessFile outputStream) throws IOException {
        gn_feature.saveModel(outputStream);
        conv_feature.saveModel(outputStream);
        temb.saveModel(outputStream);
        gn_merged.saveModel(outputStream);
        conv_merged.saveModel(outputStream);
        if (residual_layer != null) {
            residual_layer.saveModel(outputStream);
        }
    }

    public void loadModel(RandomAccessFile inputStream) throws IOException {
        gn_feature.loadModel(inputStream);
        conv_feature.loadModel(inputStream);
        temb.loadModel(inputStream);
        gn_merged.loadModel(inputStream);
        conv_merged.loadModel(inputStream);
        if (residual_layer != null) {
            residual_layer.loadModel(inputStream);
        }
    }
}

