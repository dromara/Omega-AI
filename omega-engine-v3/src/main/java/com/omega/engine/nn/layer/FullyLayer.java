package com.omega.engine.nn.layer;

import com.omega.common.data.Tensor;
import com.omega.common.utils.MathUtils;
import com.omega.common.utils.MatrixOperation;
import com.omega.common.utils.MatrixUtils;
import com.omega.common.utils.RandomUtils;
import com.omega.common.utils.Transpose;
import com.omega.engine.gpu.GPUOP;
import com.omega.engine.nn.data.Blob;
import com.omega.engine.nn.data.Blobs;

/**
 * 
 * FullyLayer
 * 
 * @author Administrator
 *
 */
public class FullyLayer extends Layer{
	
	private float[] input1d;
	
	private float[] weight1d;
	
	public FullyLayer(int inputNum,int outputNum) {
		this.channel = 1;
		this.height = 1;
		this.width = inputNum;
		this.oChannel = channel;
		this.oHeight = height;
		this.oWidth = outputNum;
		this.initParam();
	}

	public FullyLayer(int inputNum,int outputNum,boolean hasBias) {
		this.channel = 1;
		this.height = 1;
		this.width = inputNum;
		this.oChannel = channel;
		this.oHeight = height;
		this.oWidth = outputNum;
		this.hasBias = hasBias;
		this.initParam();
	}
	
	@Override
	public void initBack() {
		// TODO Auto-generated method stub
		if(this.diff == null || this.number != this.diff.number){
			this.diff = Blobs.zero(number, channel, height, width, this.diff);
		}
		MatrixUtils.zero(this.deltaB);
		MatrixUtils.zero(this.deltaW);
	}

	@Override
	public void init() {
		// TODO Auto-generated method stub
		this.number = this.network.number;
		int inputSize = this.number * this.width;
		if(this.output == null || this.number != this.output.number){
			this.output = Blobs.zero(number, oChannel, oHeight, oWidth, this.output);
		}
		if(this.input1d == null || inputSize != this.input1d.length) {
			this.input1d = new Tensor(1, 1, 1, inputSize).data;
		}
	}
	
	@Override
	public void initParam() {
		// TODO Auto-generated method stub
//		this.weight = MatrixOperation.gaussianRandom(this.width, this.oWidth, 0.01);
		this.weight = RandomUtils.xavierRandom(this.width, this.oWidth, this.width, this.oWidth);
//		this.weight = RandomUtils.heRandom(this.width, this.oWidth, this.width * this.oWidth);
		this.bias = MatrixUtils.zero(this.oWidth);
		this.deltaB = new float[this.oWidth];
		this.deltaW = new float[width][oWidth];
		this.weight1d = new float[width * oWidth];
//		this.bias = MatrixOperation.gaussianRandom(this.outputNum, 0.1);
	}
	
	@Override
	public void output() {
		
		// TODO Auto-generated method stub
		
		if(this.input != null) {
			
//			long start = System.nanoTime();
			
			MatrixUtils.transform(input.maxtir, input1d);
			MatrixUtils.transform(weight, weight1d);
			
			float[] r = new float[this.number * this.oWidth];
			
			GPUOP.getInstance().multiplyFloat(this.number, this.width, this.oWidth, input1d, weight1d, r);
			
			MatrixUtils.transform(r, output.maxtir, this.number, 1, 1, this.oWidth);
			
			if(hasBias) {
			
				for(int n = 0;n<this.number;n++) {
					for(int w = 0;w<this.oWidth;w++) {
						output.maxtir[n][0][0][w] += bias[w];
					}
				}
			
			}
			
		}
		
	}
	
	
	@Override
	public void diff() {
		// TODO Auto-generated method stub
		
		/**
		 * deltaW = inputT * delta
		 * diff = delta * weightT
		 */
		float[] dw = new float[width * oWidth];
		float[] delta1d = new float[this.number * this.oWidth];
		MatrixUtils.transform(this.delta.maxtir, delta1d);
		float[] inputT = Transpose.transpose(this.input1d, this.number, this.width);
		GPUOP.getInstance().multiplyFloat(this.width, this.number, this.oWidth, inputT, delta1d, dw);
		dw = MatrixOperation.multiplication(dw, (1.0f / this.number));
		MatrixUtils.transform(dw, deltaW, width, oWidth);
		
		float[] diff1d = new float[this.number * this.width];
		float[] weightT = Transpose.transpose(this.weight1d, this.width, this.oWidth);
		GPUOP.getInstance().multiplyFloat(this.number, this.oWidth, this.width, delta1d, weightT, diff1d);
		MatrixUtils.transform(diff1d, diff.maxtir, this.number, 1, 1, width);
		
		if(hasBias) {
			
			for(int n = 0;n<this.number;n++) {
				for(int ow = 0;ow<this.oWidth;ow++) {
					deltaB[ow] += delta.maxtir[index][0][0][ow] / number;
				}
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
		
		if(this.network.GRADIENT_CHECK) {
			this.gradientCheck();
		}

	}

	/**
	 * w(t) = w(t-1) + θ * deltaW
	 * b(t) = b(t-1) + θ * deltaB
	 * θ : learningRate
	 */
	@Override
	public void update() {
		// TODO Auto-generated method stub
		
		if(this.updater != null){
			this.updater.update(this);
		}else{
			for(int i = 0;i<this.weight.length;i++) {
				for(int j = 0;j<this.weight[i].length;j++) {
					this.weight[i][j] -= this.learnRate * this.deltaW[i][j];
				}
			}
			for(int i = 0;i<this.oWidth;i++) {
				this.bias[i] -= this.learnRate * this.deltaB[i];
			}
		}
		
	}

	@Override
	public Blob getOutput() {
		// TODO Auto-generated method stub
		return this.output;
	}

	@Override
	public void showDiff() {
		// TODO Auto-generated method stub

		float[] x = MatrixUtils.transform(this.diff.maxtir);
		
		System.out.println("fully layer["+this.index+"]diff-max:"+MathUtils.max(x)+" min:"+MathUtils.min(x));
		
	}

	@Override
	public LayerType getLayerType() {
		// TODO Auto-generated method stub
		return LayerType.full;
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
