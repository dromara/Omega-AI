package com.omega.engine.nn.layer;

import com.omega.common.utils.MatrixOperation;
import com.omega.engine.active.ActiveFunction;
import com.omega.engine.updater.Updater;

/**
 * 
 * ConvolutionLayer
 * 
 * @author Administrator
 *
 */
public class ConvolutionLayer extends Layer {

	public int channel = 0;

	public int kernelNum = 0;
	
	public int width = 0;
	
	public int height = 0;
	
	public int kWidth = 0;
	
	public int kHeight = 0;
	
	public int stride = 1;
	
	public int padding = 0;
	
	public int oChannel = 0;
	
	public int oWidth = 0;
	
	public int oHeight = 0;

	public int diffPadding = 0;
	
	public double[][][] input;  //c * h * w
	
	public double[][][] pInput;  //c * h * w
	
	public double[][][] output;  //oc * oh * ow
	
	public double[][][][] kernel;  //c * kn * kh * kw
	
//	public double[][][] bias;
	
	public double[] bias;
	
	public double[][][] active;
	
	public double[][][] diff;
	
	public double[][][] delta;
	
	public double[][][][] deltaW;
	
	public double[] deltaB;
	
	public double[][][] nextDiff;
	
	/**
	 * ConvolutionLayer
	 * @param channel
	 * @param kernelNum
	 * @param width
	 * @param height
	 * @param kWidth
	 * @param kHeight
	 * @param padding
	 * @param stride
	 * @param activeFunction
	 * @param updater
	 */
	public ConvolutionLayer(int channel,int kernelNum,int width,int height,int kWidth,int kHeight,int padding,
			int stride,ActiveFunction activeFunction,Updater updater) {
		this.channel = channel;
		this.kernelNum = kernelNum;
		this.width = width;
		this.height = height;
		this.kWidth = kWidth;
		this.kHeight = kHeight;
		this.padding = padding;
		this.stride = stride;
		this.activeFunction = activeFunction;
		this.updater = updater;
		this.layerType = LayerType.conv;
		this.initParam();
	}
	
	@Override
	public void init() {
		// TODO Auto-generated method stub
		this.output = MatrixOperation.zero(this.oChannel, this.oHeight, this.oWidth);
		this.diff = MatrixOperation.zero(this.channel, this.height, this.width);
	}

	@Override
	public void initParam() {
		// TODO Auto-generated method stub
		this.oChannel = this.kernelNum;
		this.oWidth = (this.width + this.padding * 2 - kWidth) / this.stride + 1;
		this.oHeight = (this.height + this.padding * 2 - kHeight) / this.stride + 1;
		this.kernel = MatrixOperation.gaussianRandom(this.channel, this.kernelNum, this.kHeight, this.kWidth, 0.1d);
		this.bias = MatrixOperation.zero(this.kernelNum);
		this.diffPadding = ((this.height - 1) * this.stride + this.kHeight - this.oHeight) / 2;
	}
	
	public void nextDiff(double[][][] data) {
		// TODO Auto-generated method stub
		this.nextDiff = MatrixOperation.clone(data);
	}
	
	public void input(double[][][] data) {
		// TODO Auto-generated method stub
		this.input = MatrixOperation.clone(data);
		this.pInput = MatrixOperation.zeroPadding(this.input, this.padding);
	}
	
	@Override
	public void output() {
		// TODO Auto-generated method stub
		
//		MatrixOperation.printImage(this.pInput[0]);
		
		this.output = MatrixOperation.convnVail(this.pInput, this.kernel, this.stride);
		
//		MatrixOperation.printImage(this.output[0]);
//		System.out.println("======================================>");
//		MatrixOperation.printImage(this.bias[0]);
		
		this.output = MatrixOperation.add(this.output, this.bias);
		
	}

	@Override
	public void active() {
		// TODO Auto-generated method stub
		if(this.activeFunction != null) {
//			System.out.println("realOuput:"+JsonUtils.toJson(this.output));
//			System.out.println("realActive:"+JsonUtils.toJson(this.activeFunction.active(this.output)));
			this.active = this.activeFunction.active(this.output);
		}
	}


	/**
	 * delta = diff(i + 1) * f'(xi)
	 * dx = padding(delta) conv r180(kernel)
	 * dw = delta * px
	 * remark: px is zeropadding x
	 */
	@Override
	public void diff() {
		// TODO Auto-generated method stub
		if(this.activeFunction!=null) {
			this.activeFunction.diff2d();
//			System.out.println("nextDiff:"+JsonUtils.toJson(this.nextDiff));
//			System.out.println("activeFunction.diff:"+JsonUtils.toJson(this.activeFunction.diff));
			/**
			 * 计算当前层delta
			 * delta(i) = diff(i + 1) * f'(xi)
			 */
			this.delta = MatrixOperation.multiplication(this.nextDiff, this.activeFunction.diff2d);

//			
//			System.out.println("<========================>");
//			
		}else {

			/**
			 * 计算当前层delta
			 * delta(i) = diff(i + 1)
			 */
			this.delta = this.nextDiff;
			
		}

		/**
		 * 计算deltaW
		 */
		this.deltaW = MatrixOperation.convnVail(this.pInput, this.delta, 1);

		this.deltaB = MatrixOperation.sumBias(this.delta);
		
//		MatrixOperation.printImage(this.deltaW);
		
		/**
		 * 梯度添加zeroPadding使得size与卷积输入一致
		 */
		double[][][] deltaP = MatrixOperation.zeroPadding(this.delta, this.diffPadding);
		
		/**
		 * 旋转kernel180
		 */
		double[][][][] kernel180 = MatrixOperation.rotate180(this.kernel);
		
//		System.out.println("----------------"+this.index+"-------------------------");
//		
//		MatrixOperation.printImage(this.nextDiff);
//		
//		System.out.println("==============="+this.index+"===================");
//		
//		MatrixOperation.printImage(deltaP);
		
		/**
		 * 计算当前层梯度
		 */
		double[][][] diffP = MatrixOperation.convnVail(kernel180, deltaP, 1);

		/**
		 * 去除输入层zeroPadding
		 */
		//this.diff = MatrixOperation.dislodgeZeroPadding(diffP, this.padding);
		this.diff = diffP;
		
	}

	@Override
	public void forward() {
		// TODO Auto-generated method stub
		this.init();
		this.output();
		this.active();
	}

	@Override
	public void back() {
		// TODO Auto-generated method stub
		this.diff();
		if(this.GRADIENT_CHECK) {
			this.gradientCheck();
		}
	}

	@Override
	public void update() {
		// TODO Auto-generated method stub
		if(this.updater != null){
			this.updater.updateForMatrix(this);
		}else{
			
			for(int c = 0;c<this.channel;c++) {
				for(int k = 0;k<this.kernelNum;k++) {
					for(int kh = 0;kh<this.kHeight;kh++) {
						for(int kw = 0;kw<this.kWidth;kw++) {
							this.kernel[c][k][kh][kw] -= this.learnRate * this.deltaW[c][k][kh][kw];
						}
					}
				}
			}
			
			for(int k = 0;k<this.kernelNum;k++) {
				for(int oh = 0;oh<this.oHeight;oh++) {
					for(int ow = 0;ow<this.oWidth;ow++) {
						this.bias[k] -= this.learnRate * this.deltaB[k];
					}
				}
			}
			
		}
	}

	@Override
	public LayerType getLayerType() {
		// TODO Auto-generated method stub
		return LayerType.conv;
	}

	@Override
	public double[] activeTemp(double[] output) {
		// TODO Auto-generated method stub
		return this.activeFunction.activeTemp(output);
	}

	@Override
	public double[] diffTemp() {
		// TODO Auto-generated method stub
		return MatrixOperation.transform(this.active);
	}

	@Override
	public double[] getOutput() {
		// TODO Auto-generated method stub
		return MatrixOperation.transform(this.output);
	}

	@Override
	public void showDiff() {
		// TODO Auto-generated method stub
//		System.out.println("conv layer["+this.index+"]diff start:");
//		
//		
//		System.out.println("conv layer["+this.index+"]diff end.");
	}

}
