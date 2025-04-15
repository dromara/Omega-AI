package com.omega.engine.nn.layer;

import com.omega.common.config.Tensor;
import com.omega.common.utils.MatrixUtils;
import com.omega.common.utils.RandomUtils;
import com.omega.engine.gpu.BaseKernel;
import com.omega.engine.gpu.CUDAManager;

/**
 * 路由层
 *
 * @author Administrator
 */
public class RouteLayer extends Layer {
    private Layer[] layers;
    private int groups = 1;
    private int groupId = 0;

    public RouteLayer(Layer[] layers) {
        this.layers = layers;
        Layer first = layers[0];
        this.oHeight = first.oHeight;
        this.oWidth = first.oWidth;
        String names = "[";
        for (Layer layer : layers) {
            if (layer.oHeight != this.oHeight || layer.oWidth != this.oWidth) {
                throw new RuntimeException("input size must be all same in the route layer.[" + layer.oHeight + ":" + this.oHeight + "]");
            }
            this.network = layer.network;
            this.oChannel += layer.oChannel;
            names += layer.name + ",";
        }
        names += "]";
        System.out.println(names);
        this.network.addRouteLayer(this);
    }

    public RouteLayer(Layer[] layers, int groups, int groupId) {
        this.groups = groups;
        this.groupId = groupId;
        this.layers = layers;
        Layer first = layers[0];
        this.oHeight = first.oHeight;
        this.oWidth = first.oWidth;
        for (Layer layer : layers) {
            if (layer.oHeight != this.oHeight || layer.oWidth != this.oWidth) {
                throw new RuntimeException("input size must be all same in the route layer.");
            }
            this.network = layer.network;
            this.oChannel += layer.oChannel;
        }
    }

    public static void main(String[] args) {
        int N = 2;
        int C = 3;
        int C2 = 2;
        int H = 4;
        int W = 4;
        int oHeight = H;
        int oWidth = W;
        int oChannel = C + C2;
        float[] x = MatrixUtils.order(N * C * H * W, 1, 1);
        float[] x2 = MatrixUtils.order(N * C2 * H * W, 1, 1);
        float[] d = RandomUtils.order(N * oChannel * oHeight * oWidth, 1, 1);
        Tensor input = new Tensor(N, C, H, W, x, true);
        Tensor input2 = new Tensor(N, C2, H, W, x2, true);
        Tensor[] inputs = new Tensor[]{input, input2};
        Tensor output = new Tensor(N, oChannel, oHeight, oWidth, true);
        Tensor delta = new Tensor(N, oChannel, oHeight, oWidth, d, true);
        Tensor diff1 = new Tensor(N, C, H, W, true);
        Tensor diff2 = new Tensor(N, C2, H, W, true);
        Tensor[] diffs = new Tensor[]{diff1, diff2};
        CUDAManager cudaManager = new CUDAManager(0);
        BaseKernel kernel = new BaseKernel(cudaManager);
        testForward(inputs, output, kernel);
        output.showDM();
        testBackward(diffs, delta, kernel);
        delta.showDM();
        for (Tensor diff : diffs) {
            diff.showDM();
        }
    }

    public static void testForward(Tensor[] x, Tensor output, BaseKernel kernel) {
        int offset = 0;
        for (int l = 0; l < x.length; l++) {
            Tensor input = x[l];
            for (int n = 0; n < output.number; n++) {
                kernel.copy_gpu(input, output, input.getOnceSize(), n * input.getOnceSize(), 1, offset + n * output.getOnceSize(), 1);
            }
            offset += input.getOnceSize();
        }
    }

    public static void testBackward(Tensor[] diffs, Tensor delta, BaseKernel kernel) {
        int offset = 0;
        for (int l = 0; l < diffs.length; l++) {
            Tensor diff = diffs[l];
            for (int n = 0; n < delta.number; n++) {
                //				kernel.axpy_gpu(delta, diff, diff.getOnceSize(), 1, offset + n * delta.getOnceSize(), 1, n * diff.getOnceSize(), 1);
                kernel.copy_gpu(delta, diff, diff.getOnceSize(), offset + n * delta.getOnceSize(), 1, n * diff.getOnceSize(), 1);
            }
            offset += diff.getOnceSize();
        }
    }

    @Override
    public void init() {
        // TODO Auto-generated method stub
        this.number = this.network.number;
        if (this.output == null || this.output.number != this.number) {
            this.output = Tensor.createTensor(this.output, number, oChannel, oHeight, oWidth, true);
            //			this.output = new Tensor(number, oChannel, oHeight, oWidth, true);
        }
    }

    @Override
    public void initBack() {
        // TODO Auto-generated method stub
        //		if(layers[0].cache_delta == null || layers[0].cache_delta.number != this.number) {
        for (Layer layer : layers) {
            if (layer.cache_delta == null || layer.cache_delta.number != this.number) {
                layer.cache_delta = new Tensor(number, layer.oChannel, oHeight, oWidth, true);
            }
        }
        //		}
    }

    @Override
    public void initParam() {
        // TODO Auto-generated method stub
    }

    @Override
    public void output() {
        // TODO Auto-generated method stub
        int offset = 0;
        for (int l = 0; l < layers.length; l++) {
            Tensor input = layers[l].output;
            //			input.showDM("l"+l);
            int part_input_size = input.getOnceSize() / groups;
            for (int n = 0; n < this.number; n++) {
                baseKernel().copy_gpu(input, this.output, part_input_size, n * input.getOnceSize() + part_input_size * groupId, 1, offset + n * output.getOnceSize(), 1);
            }
            offset += part_input_size;
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
        int offset = 0;
        for (int l = 0; l < layers.length; l++) {
            Tensor delta = layers[l].cache_delta;
            //			System.out.println(layers[l].index+":"+delta);
            int part_input_size = delta.getOnceSize() / groups;
            for (int n = 0; n < this.number; n++) {
                baseKernel().axpy_gpu(this.delta, delta, part_input_size, 1, offset + n * this.delta.getOnceSize(), 1, n * delta.getOnceSize() + part_input_size * groupId, 1);
                //				baseKernel().copy_gpu(this.delta, delta, delta.getOnceSize(), offset + n * this.delta.getOnceSize(), 1, n * delta.getOnceSize(), 1);
            }
            offset += part_input_size;
        }
    }

    public void diff(Layer skip) {
        // TODO Auto-generated method stub
        int offset = 0;
        for (int l = 0; l < layers.length; l++) {
            if (skip != layers[l]) {
                Tensor delta = layers[l].cache_delta;
                //				System.out.println(layers[l].index+":"+delta);
                int part_input_size = delta.getOnceSize() / groups;
                for (int n = 0; n < this.number; n++) {
                    baseKernel().axpy_gpu(this.delta, delta, part_input_size, 1, offset + n * this.delta.getOnceSize(), 1, n * delta.getOnceSize() + part_input_size * groupId, 1);
                    //					baseKernel().copy_gpu(this.delta, delta, delta.getOnceSize(), offset + n * this.delta.getOnceSize(), 1, n * delta.getOnceSize(), 1);
                }
                offset += part_input_size;
            }
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
         * 计算输出

         */
        this.output();
    }

    @Override
    public void back() {
        // TODO Auto-generated method stub
        this.initBack();
        /**
         * 设置梯度

         */
        this.setDelta();
        /**
         * 计算梯度

         */
        this.diff();
        if (this.network.GRADIENT_CHECK) {
            this.gradientCheck();
        }
    }

    @Override
    public void forward(Tensor input) {
        // TODO Auto-generated method stub
        /**
         * 参数初始化

         */
        this.init();
        /**
         * 计算输出

         */
        this.output();
    }

    @Override
    public void back(Tensor delta) {
        // TODO Auto-generated method stub
        this.initBack();
        /**
         * 设置梯度

         */
        this.setDelta(delta);
        /**
         * 计算梯度

         */
        this.diff();
        if (this.network.GRADIENT_CHECK) {
            this.gradientCheck();
        }
    }

    public void back(Tensor delta, Layer skip) {
        // TODO Auto-generated method stub
        this.initBack();
        /**
         * 设置梯度

         */
        this.setDelta(delta);
        /**
         * 计算梯度

         */
        this.diff(skip);
        if (this.network.GRADIENT_CHECK) {
            this.gradientCheck();
        }
    }

    @Override
    public void update() {
        // TODO Auto-generated method stub
    }

    @Override
    public void showDiff() {
        // TODO Auto-generated method stub
    }

    @Override
    public LayerType getLayerType() {
        // TODO Auto-generated method stub
        return LayerType.route;
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
    public void backTemp() {
        // TODO Auto-generated method stub
    }

    @Override
    public void accGrad(float scale) {
        // TODO Auto-generated method stub
    }

    public void clearCacheDelta() {
        for (Layer layer : layers) {
            if (layer.cache_delta != null) {
                layer.cache_delta.clearGPU();
            }
        }
    }
}

