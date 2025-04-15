package com.omega.engine.nn.layer.tae;

import com.omega.common.tensor.Tensor;
import com.omega.engine.nn.layer.ConvolutionLayer;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.ParamsInit;
import com.omega.engine.nn.layer.active.ActiveFunctionLayer;
import com.omega.engine.nn.layer.active.ReluLayer;
import com.omega.engine.nn.layer.gpu.BasicBlockKernel;
import com.omega.engine.nn.network.Network;
import com.omega.engine.updater.UpdaterFactory;

/**
 * resnet block layer
 *
 * @author Administrator
 */
public class TAEBlock extends Layer {
    private BasicBlockKernel kernel;
    private ActiveFunctionLayer a0;
    private ConvolutionLayer conv1;
    private ActiveFunctionLayer a1;
    private ConvolutionLayer conv2;
    private ActiveFunctionLayer a2;
    private ConvolutionLayer conv3;
    private ActiveFunctionLayer fuse;
    private ConvolutionLayer conv_shortcut;
    private boolean shortcut = false;
    private Tensor cache_delta;
    private Tensor tmp;

    public TAEBlock(int channel, int oChannel, int height, int width, Network network) {
        this.network = network;
        this.channel = channel;
        this.oChannel = oChannel;
        this.height = height;
        this.width = width;
        if (channel != oChannel) {
            shortcut = true;
        }
        kernel = new BasicBlockKernel(cuda());
        initLayers();
        this.oHeight = conv3.oHeight;
        this.oWidth = conv3.oWidth;
    }

    public void initLayers() {
        a0 = new ReluLayer(this);
        conv1 = new ConvolutionLayer(channel, oChannel, width, height, 3, 3, 1, 1, true, this.network);
        conv1.setUpdater(UpdaterFactory.create(this.network));
        conv1.paramsInit = ParamsInit.relu;
        a1 = new ReluLayer(conv1);
        conv2 = new ConvolutionLayer(oChannel, oChannel, conv1.oWidth, conv1.oHeight, 3, 3, 1, 1, true, this.network);
        conv2.setUpdater(UpdaterFactory.create(this.network));
        conv2.paramsInit = ParamsInit.relu;
        a2 = new ReluLayer(conv2);
        conv3 = new ConvolutionLayer(oChannel, oChannel, conv2.oWidth, conv2.oHeight, 3, 3, 1, 1, true, this.network);
        conv3.setUpdater(UpdaterFactory.create(this.network));
        conv3.paramsInit = ParamsInit.relu;
        if (shortcut) {
            conv_shortcut = new ConvolutionLayer(channel, oChannel, width, height, 1, 1, 0, 1, false, this.network);
            conv_shortcut.setUpdater(UpdaterFactory.create(this.network));
            conv_shortcut.paramsInit = ParamsInit.silu;
        }
        fuse = new ReluLayer(conv3);
    }

    @Override
    public void init() {
        this.number = this.network.number;
        if (this.tmp == null || this.output.number != this.network.number) {
            this.tmp = Tensor.createGPUTensor(this.tmp, number, oChannel, oHeight, oWidth, true);
        }
    }

    @Override
    public void initBack() {
        if (this.diff == null || conv1.number != conv1.diff.number) {
            conv1.initBack();
            this.diff = conv1.diff;
            this.cache_delta = Tensor.createGPUTensor(this.cache_delta, number, oChannel, oHeight, oWidth, true);
        }
    }

    @Override
    public void initParam() {
        // TODO Auto-generated method stub
    }

    @Override
    public void output() {
        // TODO Auto-generated method stub
        a0.forward(this.input);
        conv1.forward(a0.getOutput());
        a1.forward(conv1.getOutput());
        conv2.forward(a1.getOutput());
        a2.forward(conv2.getOutput());
        conv3.forward(a2.getOutput());
        if (shortcut) {
            conv_shortcut.forward(this.input);
            kernel.add(conv_shortcut.getOutput(), conv3.getOutput(), tmp);
        } else {
            kernel.add(input, conv3.getOutput(), tmp);
        }
        fuse.forward(tmp);
        this.output = fuse.getOutput();
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
        fuse.back(delta);
        conv3.back(fuse.diff);
        a2.back(conv3.diff);
        conv2.back(a2.diff);
        a1.back(conv2.diff);
        conv1.back(a1.diff);
        a1.back(conv1.diff);
        if (shortcut) {
            conv_shortcut.back(this.cache_delta);
            kernel.add(a1.diff, conv_shortcut.diff, this.diff);
        } else {
            kernel.add(a1.diff, this.cache_delta, this.diff);
        }
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
        conv2.update();
        conv3.update();
        if (shortcut) {
            conv_shortcut.update();
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
    public void forward(Tensor input) {
        // TODO Auto-generated method stub
        /**
         * 参数初始化

         */
        this.init();
        /**
         * 设置输入

         */
        this.setInput(input);
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

