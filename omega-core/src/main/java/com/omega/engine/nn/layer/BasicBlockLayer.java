package com.omega.engine.nn.layer;

import com.omega.common.tensor.Tensor;
import com.omega.engine.nn.layer.active.ActiveFunctionLayer;
import com.omega.engine.nn.layer.active.ReluLayer;
import com.omega.engine.nn.layer.gpu.BasicBlockKernel;
import com.omega.engine.nn.layer.normalization.BNLayer;
import com.omega.engine.nn.network.Network;
import com.omega.engine.updater.UpdaterFactory;

/**
 * resnet block layer
 *
 * @author Administrator
 */
public class BasicBlockLayer extends Layer {
    private BasicBlockKernel kernel;
    private ConvolutionLayer identityConv;
    private BNLayer identityBN;
    private ConvolutionLayer conv1;
    private BNLayer bn1;
    private ActiveFunctionLayer a1;
    private ConvolutionLayer conv2;
    private BNLayer bn2;
    private int kHeight = 3;
    private int kWidth = 3;
    private int padding = 1;
    private int fisrtLayerStride = 2;
    private boolean downsample = false;
    private Tensor cache_delta;

    public BasicBlockLayer(int channel, int oChannel, int height, int width, int fisrtLayerStride, Network network) {
        this.network = network;
        this.channel = channel;
        this.oChannel = oChannel;
        this.height = height;
        this.width = width;
        this.fisrtLayerStride = fisrtLayerStride;
        if (this.fisrtLayerStride == 1) {
            this.oHeight = height;
            this.oWidth = width;
        } else {
            this.oHeight = (height + padding * 2 - kHeight) / fisrtLayerStride + 1;
            this.oWidth = (width + padding * 2 - kWidth) / fisrtLayerStride + 1;
        }
        if (channel != oChannel) {
            downsample = true;
        }
        kernel = new BasicBlockKernel(cuda());
        initLayers();
    }

    public void initLayers() {
        conv1 = new ConvolutionLayer(channel, oChannel, width, height, 3, 3, 1, fisrtLayerStride, false, this.network);
        conv1.setUpdater(UpdaterFactory.create(this.network));
        conv1.paramsInit = ParamsInit.relu;
        bn1 = new BNLayer(conv1);
        a1 = new ReluLayer(bn1);
        conv2 = new ConvolutionLayer(conv1.oChannel, oChannel, conv1.oWidth, conv1.oHeight, 3, 3, 1, 1, false, this.network);
        conv2.setUpdater(UpdaterFactory.create(this.network));
        conv2.paramsInit = ParamsInit.relu;
        bn2 = new BNLayer(conv2);
        if (downsample) {
            identityConv = new ConvolutionLayer(channel, oChannel, width, height, 1, 1, 0, fisrtLayerStride, false, this.network);
            identityConv.setUpdater(UpdaterFactory.create(this.network));
            identityConv.paramsInit = ParamsInit.relu;
            identityBN = new BNLayer(identityConv);
        }
        //		shortcut = new ShortcutLayer(bn2.oChannel, bn2.oHeight, bn2.oWidth, a1.oChannel, a1.oHeight, a1.oWidth, this.network);
    }

    @Override
    public void init() {
        this.number = this.network.number;
        if (this.output == null || this.output.number != this.network.number) {
            this.output = Tensor.createTensor(this.output, number, oChannel, oHeight, oWidth, true);
            //			this.output = new Tensor(number, oChannel, oHeight, oWidth, true);
        }
    }

    @Override
    public void initBack() {
        if (this.diff == null || conv1.number != conv1.diff.number) {
            conv1.initBack();
            this.diff = conv1.diff;
            this.cache_delta = new Tensor(number, oChannel, oHeight, oWidth, true);
        }
    }

    @Override
    public void initParam() {
        // TODO Auto-generated method stub
    }

    @Override
    public void output() {
        // TODO Auto-generated method stub
        conv1.forward(this.input);
        bn1.forward(conv1.output);
        a1.forward(bn1.output);
        conv2.forward(a1.output);
        bn2.forward(conv2.output);
        if (downsample) {
            identityConv.forward(this.input);
            identityBN.forward(identityConv.output);
            kernel.add(identityBN.output, bn2.output, output);
        } else {
            kernel.add(input, bn2.output, output);
        }
    }

    @Override
    public Tensor getOutput() {
        // TODO Auto-generated method stub
        return this.output;
    }

    @Override
    public void diff() {
        // TODO Auto-generated method stub
        //		System.out.println(index);
        /**
         * deltax = deltao * (f'(x) + 1)

         */
        baseKernel().copy_gpu(delta, this.cache_delta, delta.getDataLength(), 1, 1);
        bn2.back(delta);
        conv2.back(bn2.diff);
        a1.back(conv2.diff);
        bn1.back(a1.diff);
        conv1.back(bn1.diff);
        //		System.out.println("basicBlock2:"+conv1.diff.checkDM());
        if (downsample) {
            identityBN.back(this.cache_delta);
            identityConv.back(identityBN.diff);
            kernel.add(conv1.diff, identityConv.diff, this.diff);
        } else {
            kernel.add(conv1.diff, this.cache_delta, this.diff);
        }
        //		shortcut.back(delta, conv1.diff);
        //		this.diff = conv1.diff;
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
    public void update() {
        // TODO Auto-generated method stub
        conv1.update();
        bn1.update();
        conv2.update();
        bn2.update();
        if (downsample) {
            identityBN.update();
            identityConv.update();
        }
    }

    @Override
    public void showDiff() {
        // TODO Auto-generated method stub
    }

    @Override
    public LayerType getLayerType() {
        // TODO Auto-generated method stub
        return LayerType.block;
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
    public void forward(Tensor inpnut) {
        // TODO Auto-generated method stub
        /**
         * 参数初始化

         */
        this.init();
        /**
         * 设置输入

         */
        this.setInput(inpnut);
        /**
         * 计算输出

         */
        this.output();
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

    @Override
    public void backTemp() {
        // TODO Auto-generated method stub
    }

    @Override
    public void accGrad(float scale) {
        // TODO Auto-generated method stub
    }
}

