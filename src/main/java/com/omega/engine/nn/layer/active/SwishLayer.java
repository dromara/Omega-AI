package com.omega.engine.nn.layer.active;

import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.active.gpu.SwishKernel;
import com.omega.engine.nn.network.Network;
import com.omega.engine.tensor.Tensor;

/**
 * x * Sigmod active function Layer
 *
 * @author Administrator
 */
public class SwishLayer extends ActiveFunctionLayer {
	
    private SwishKernel kernel;

    public SwishLayer() {
    }

    public SwishLayer(Layer preLayer) {
        this.setPreLayer(preLayer);
    }

    public SwishLayer(Network network) {
        this.network = network;
    }

    @Override
    public void initParam() {
        // TODO Auto-generated method stub
    }

    public void init() {
        super.init();
        if (kernel == null) {
            kernel = new SwishKernel(network.cudaManager);
        }
    }

    public void init(Tensor input) {
        super.init(input);
        if (kernel == null) {
            kernel = new SwishKernel(network.cudaManager);
        }
    }

    @Override
    public void output() {
        // TODO Auto-generated method stub
        //		input.showDM();
        kernel.forward(input, output);
    }

    @Override
    public Tensor getOutput() {
        // TODO Auto-generated method stub
        return this.output;
    }

    @Override
    public void diff() {
        // TODO Auto-generated method stub
        kernel.backward(input, delta, diff);
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
    public void showDiff() {
        // TODO Auto-generated method stub
    }

    @Override
    public LayerType getLayerType() {
        // TODO Auto-generated method stub
        return LayerType.swish;
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

    public void initBack(Tensor diff) {
        this.diff = diff;
    }

    @Override
    public void forward(Tensor inpnut) {
        // TODO Auto-generated method stub
        /**
         * 参数初始化

         */
        this.init(input);
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
        this.initBack(delta);
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
    public void forward(Tensor input, int batch, int step) {
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
        this.output(batch, step);
    }

    @Override
    public void back(Tensor delta, int batch, int step) {
        // TODO Auto-generated method stub
        this.initBack(delta);
        /**
         * 设置梯度

         */
        this.setDelta(delta);
        /**
         * 计算梯度

         */
        this.diff(batch, step);
        if (this.network.GRADIENT_CHECK) {
            this.gradientCheck();
        }
    }

    @Override
    public void output(int batch, int step) {
        // TODO Auto-generated method stub
        
    }

    @Override
    public void diff(int batch, int step) {
        // TODO Auto-generated method stub

    }

    @Override
    public void accGrad(float scale) {
        // TODO Auto-generated method stub
    }

	@Override
	public void backTemp() {
		// TODO Auto-generated method stub
		
	}
}

