package com.omega.engine.nn.layer;

import com.omega.common.utils.MatrixOperation;
import com.omega.common.utils.MatrixUtils;
import com.omega.engine.nn.data.Blob;
import com.omega.engine.nn.data.Blobs;
import com.omega.engine.nn.network.RunModel;

/**
 * Dropout Layer
 * @author Administrator
 *
 */
public class DropoutLayer extends Layer {
	
	private float probability = 0.5f;
	
	private float[][][][] mask;
	
	public Layer preLayer;
	
	public DropoutLayer(float probability) {
		this.probability = probability;
	}
	
	@Override
	public void init() {
		// TODO Auto-generated method stub
		if(preLayer == null) {
			preLayer = this.network.getPreLayer(this.index);
			this.channel = preLayer.oChannel;
			this.height = preLayer.oHeight;
			this.width = preLayer.oWidth;
			this.oChannel = this.channel;
			this.oHeight = this.height;
			this.oWidth = this.width;
		}
		this.number = this.network.number;
		this.output = Blobs.zero(number, oChannel, oHeight, oWidth, this.output);
		initParam();
	}

	@Override
	public void initParam() {
		// TODO Auto-generated method stub

		/**
		 * 训练
		 */
		if(this.network.RUN_MODEL == RunModel.TRAIN) {
			this.mask = MatrixUtils.val(this.number, this.oChannel, this.oHeight, this.oWidth, probability, 1.0f - probability);
		}
		
	}

	@Override
	public void initBack() {
		// TODO Auto-generated method stub
		this.diff = Blobs.zero(number, channel, height, width, this.diff);
	}

	@Override
	public void output() {
		// TODO Auto-generated method stub
		if(this.network.RUN_MODEL == RunModel.TRAIN) {
			this.output.maxtir = MatrixOperation.multiplication(this.input.maxtir, this.mask);
		}else {
			this.output.maxtir = MatrixOperation.multiplication(this.input.maxtir, 1.0f - probability);
		}
	}

	@Override
	public Blob getOutput() {
		// TODO Auto-generated method stub
		return this.output;
	}

	@Override
	public void diff() {
		// TODO Auto-generated method stub
		this.diff.maxtir = MatrixOperation.multiplication(this.delta.maxtir, this.mask);
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
		return LayerType.dropout;
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

}
