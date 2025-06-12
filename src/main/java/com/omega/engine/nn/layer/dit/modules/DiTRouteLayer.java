package com.omega.engine.nn.layer.dit.modules;

import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.network.DiT;
import com.omega.engine.tensor.Tensor;

/**
 * 路由层
 *
 * @author Administrator
 */
public class DiTRouteLayer extends Layer {
    private Layer[] layers;
//    private int groups = 1;
//    private int groupId = 0;

    public DiTRouteLayer(Layer[] layers,int T) {
        this.layers = layers;
        Layer first = layers[0];
        this.oHeight = first.oHeight;
        this.oChannel = T;
        String names = "[";
        for (Layer layer : layers) {
            if (layer.oHeight != this.oHeight) {
                throw new RuntimeException("input size must be all same in the route layer.[" + layer.oHeight + ":" + this.oHeight + "]");
            }
            this.network = layer.network;
            this.oWidth += layer.oWidth;
            names += layer.name + ",";
        }
        names += "]";
        System.out.println(names);
        DiT dit = (DiT) this.network;
        dit.addSkip(this);
    }

    @Override
    public void init() {
        // TODO Auto-generated method stub
        this.number = this.network.number;
        if (this.output == null || this.output.number != this.number * oChannel) {
            this.output = Tensor.createGPUTensor(this.output, number * oChannel, 1, oHeight, oWidth, true);
        }
    }

    @Override
    public void initBack() {
        // TODO Auto-generated method stub
        for (Layer layer : getLayers()) {
            if (layer.cache_delta == null || layer.cache_delta.checkShape(layer.getOutput())) {
                layer.cache_delta = Tensor.createGPUTensor(layer.cache_delta, layer.getOutput().shape(), true);
            }
        }
    }

    @Override
    public void initParam() {
        // TODO Auto-generated method stub
    }

    @Override
    public void output() {
        // TODO Auto-generated method stub
    	
    	Tensor_OP().cat_width(getLayers()[0].getOutput(), getLayers()[1].getOutput(), output);
    	
//        int offset = 0;
//        for (int l = 0; l < getLayers().length; l++) {
//            Tensor input = getLayers()[l].output;
//            int part_input_size = input.getOnceSize() / groups;
//            for (int n = 0; n < this.number * oChannel; n++) {
//                baseKernel().copy_gpu(input, this.output, part_input_size, n * input.getOnceSize() + part_input_size * groupId, 1, offset + n * output.getOnceSize(), 1);
//            }
//            offset += part_input_size;
//            input.viewOrg();
//        }
    }

    @Override
    public Tensor getOutput() {
        // TODO Auto-generated method stub
        return this.output;
    }

    @Override
    public void diff() {
        // TODO Auto-generated method stub
    	Tensor_OP().cat_width_back(delta, getLayers()[0].cache_delta, getLayers()[1].cache_delta);
//        int offset = 0;
//        for (int l = 0; l < getLayers().length; l++) {
//            Tensor delta = getLayers()[l].cache_delta;
//            int part_input_size = delta.getOnceSize() / groups;
//            for (int n = 0; n < this.number * oChannel; n++) {
//                baseKernel().axpy_gpu(this.delta, delta, part_input_size, 1, offset + n * this.delta.getOnceSize(), 1, n * delta.getOnceSize() + part_input_size * groupId, 1);
//            }
//            offset += part_input_size;
//        }
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
        for (Layer layer : getLayers()) {
            if (layer.cache_delta != null) {
                layer.cache_delta.clearGPU();
            }
        }
    }

	public Layer[] getLayers() {
		return layers;
	}
}

