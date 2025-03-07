package com.omega.engine.nn.layer.diffusion.unet;

import com.omega.common.data.Tensor;
import com.omega.engine.gpu.BaseKernel;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;

/**
 * 路由层
 * @author Administrator
 *
 */
public class UNetRouteLayer extends Layer{
	
	private Layer layer_start;
	
	private Layer layer_end;
	
	public UNetRouteLayer(Layer layer_start, Layer layer_end) {
		this.layer_start = layer_start;
		this.layer_end = layer_end;
		Layer first = layer_start;
		this.oHeight = first.oHeight;
		this.oWidth = first.oWidth;
		if(layer_start.oHeight != layer_end.oHeight || layer_start.oWidth != layer_end.oWidth) {
			throw new RuntimeException("input size must be all same in the route layer.["+layer_start.oHeight+":"+layer_end.oHeight+"]");
		}
		this.network = layer_start.network;
		this.oChannel = layer_start.oChannel + layer_end.channel;
	}
	
	@Override
	public void init() {
		// TODO Auto-generated method stub
		
		this.number = this.network.number;
		
		if(this.output == null || this.output.number != this.number) {
			this.output = Tensor.createTensor(this.output, number, oChannel, oHeight, oWidth, true);
		}

	}

	@Override
	public void initBack() {
		// TODO Auto-generated method stub
		if(layer_start.cache_delta == null || layer_start.cache_delta.number != this.number) {
			layer_start.cache_delta = new Tensor(number, layer_start.oChannel, oHeight, oWidth, true);
		}
	}

	@Override
	public void initParam() {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void output() {
		// TODO Auto-generated method stub
		Tensor_OP().cat(layer_start.getOutput(), layer_end.input, this.output);
	}

	@Override
	public Tensor getOutput() {
		// TODO Auto-generated method stub
		return this.output;
	}

	@Override
	public void diff() {
		// TODO Auto-generated method stub
		Tensor_OP().cat_back(this.delta, layer_start.cache_delta, layer_end.diff);
	}
	
	public static void testForward(Tensor[] x,Tensor output,BaseKernel kernel) {

    	int offset = 0;
		for(int l = 0;l<x.length;l++) {
			Tensor input = x[l];
			for(int n = 0;n<output.number;n++) {
				kernel.copy_gpu(input, output, input.getOnceSize(), n * input.getOnceSize(), 1, offset + n * output.getOnceSize(), 1);
			}
			offset += input.getOnceSize();
		}
    	
	}
	
	public static void testBackward(Tensor[] diffs,Tensor delta,BaseKernel kernel) {

		int offset = 0;
		for(int l = 0;l<diffs.length;l++) {
			Tensor diff = diffs[l];
			for(int n = 0;n<delta.number;n++) {
//				kernel.axpy_gpu(delta, diff, diff.getOnceSize(), 1, offset + n * delta.getOnceSize(), 1, n * diff.getOnceSize(), 1);
				kernel.copy_gpu(delta, diff, diff.getOnceSize(), offset + n * delta.getOnceSize(), 1, n * diff.getOnceSize(), 1);
			}
			offset += diff.getOnceSize();
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
		
		if(this.network.GRADIENT_CHECK) {
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
		
		if(this.network.GRADIENT_CHECK) {
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

}
