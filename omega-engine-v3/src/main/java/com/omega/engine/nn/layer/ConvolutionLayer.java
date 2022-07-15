package com.omega.engine.nn.layer;

import com.omega.common.data.Tensor;
import com.omega.common.utils.Dilation;
import com.omega.common.utils.Im2col4d2T;
import com.omega.common.utils.Im2colForWeight;
import com.omega.common.utils.Im2colToVector;
import com.omega.common.utils.Im2colUtils;
import com.omega.common.utils.MathUtils;
import com.omega.common.utils.MatrixOperation;
import com.omega.common.utils.MatrixUtils;
import com.omega.common.utils.RandomUtils;
import com.omega.engine.gpu.GPUOP;
import com.omega.engine.gpu.data.CacheDataSet;
import com.omega.engine.nn.data.Blob;
import com.omega.engine.nn.data.Blobs;
import com.omega.engine.nn.model.ConvLayerInit;
import com.omega.engine.nn.model.LayerInit;

/**
 * 
 * ConvolutionLayer
 * 
 * @author Administrator
 *
 */
public class ConvolutionLayer extends Layer {

	public int kernelNum = 0;
	
	public int kWidth = 0;
	
	public int kHeight = 0;
	
	public int stride = 1;
	
	public int padding = 0;

	public int diffPadding = 0;
	
	public Blob pInput;  //n * c * h * w
	
	public Tensor pInput1D;  //n * c * h * w im2col
	
	public Tensor dInput1D;
	
	public float[][][][] kernel;  //kn * c * kh * kw

	public float[] bias;
	
	public float[][][][] deltaW;
	
	public float[] deltaB;
	
	public float[][][][] dwd;
	
//	public float[][][][] pdwd;
	
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
			int stride) {
		this.kernelNum = kernelNum;
		this.channel = channel;
		this.width = width;
		this.height = height;
		this.kWidth = kWidth;
		this.kHeight = kHeight;
		this.padding = padding;
		this.stride = stride;
		this.initParam();
	}
	
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
			int stride,boolean hasBias) {
		this.kernelNum = kernelNum;
		this.channel = channel;
		this.width = width;
		this.height = height;
		this.kWidth = kWidth;
		this.kHeight = kHeight;
		this.padding = padding;
		this.stride = stride;
		this.hasBias = hasBias;
		this.initParam();
	}
	
	@Override
	public void initParam() {
		// TODO Auto-generated method stub
		this.oChannel = this.kernelNum;
		this.oWidth = (this.width + this.padding * 2 - kWidth) / this.stride + 1;
		this.oHeight = (this.height + this.padding * 2 - kHeight) / this.stride + 1;
//		this.kernel = MatrixOperation.gaussianRandom(this.kernelNum, this.channel, this.kHeight, this.kWidth, 0.01d);
//		this.kernel = RandomUtils.heRandom(this.kernelNum, this.channel, this.kHeight, this.kWidth, this.width * this.height);
		this.kernel = RandomUtils.xavierRandom(this.kernelNum, this.channel, this.kHeight, this.kWidth, this.channel * this.height * this.width, this.oChannel * this.oHeight * this.oWidth);
//		this.kernel = RandomUtils.heRandom(this.kernelNum, this.channel, this.kHeight, this.kWidth, this.channel * this.oChannel * this.height * this.width);
		this.bias = MatrixUtils.zero(this.kernelNum);
		this.diffPadding = ((this.height - 1) * this.stride + this.kHeight - this.oHeight) / 2;
		this.deltaB = MatrixUtils.zero(this.kernelNum);
		this.deltaW = MatrixUtils.zero(this.kernelNum,this.channel,this.kHeight,this.kWidth);
	}

	@Override
	public void init() {
		// TODO Auto-generated method stub
		this.number = this.network.number;
		if(this.pInput == null || this.number != this.pInput.number){
			this.output = Blobs.zero(number, oChannel, oHeight, oWidth, this.output);
			this.pInput = Blobs.zero(number, channel, height + padding * 2, width + padding * 2, this.pInput);
			int pLength = this.channel * kHeight * kWidth * this.number * oHeight * oWidth;
			this.pInput1D = new Tensor(1, 1, 1, pLength);
		}
	}

	@Override
	public void initBack() {
		// TODO Auto-generated method stub
		if(this.diff == null || this.number != this.diff.number){
			this.diff = Blobs.zero(number, channel, height, width, this.diff);
			int kh = this.oHeight;
			int kw = this.oWidth;
			if(this.stride > 1) {
				kh = this.oHeight + (this.oHeight - 1) * (this.stride - 1);
				kw = this.oWidth + (this.oWidth - 1) * (this.stride - 1);
				this.dwd = new float[this.number][this.oChannel][kh][kw];
			}
			int oHeight = ((this.pInput.height - kh ) / 1) + 1;
			int oWidth = ((this.pInput.width - kw) / 1) + 1;
			int xm = this.channel * oHeight * oWidth;
			int xn = this.number * kh * kw;
			int pLength = xm * xn;
			this.dInput1D = new Tensor(1, 1, 1, pLength);
		}
		MatrixUtils.zero(this.deltaB);
		MatrixUtils.zero(this.deltaW);
	}

	@Override
	public void output() {
		// TODO Auto-generated method stub

		MatrixOperation.zeroPadding(this.input.maxtir, this.pInput.maxtir, this.padding);

//		this.output.maxtir = MatrixOperation.convnVail(this.pInput.maxtir, this.kernel, this.stride);
		
//		this.output.maxtir = MatrixOperation.convnVailByIm2Col(this.pInput.maxtir, this.kernel, this.stride, false);
//		
//		System.out.println("");
//		System.out.println("================start=====================");
//		
//		PrintUtils.printImage(this.output.maxtir[0][0][0]);
		
		Im2colToVector.im2col(this.pInput.maxtir, this.pInput1D.data, this.kHeight, this.kWidth, this.stride);
//		System.out.println("this.pInput1D.data:"+this.pInput1D.data.length);
//		Im2colToVector.im2col(this.pInput.maxtir, this.pInput1D.data, this.kHeight, this.kWidth, this.stride);
		
		MatrixOperation.convnVailByIm2ColGPUV2(this.pInput1D.data, this.kernel, this.number, this.channel, this.height + this.padding * 2, this.width + this.padding * 2, this.stride, this.output.maxtir);
		
//		System.out.println("");
//		System.out.println("=================end====================");
//		
		
		if(this.hasBias) {
//			long start3 = System.nanoTime();
			this.output.maxtir = MatrixOperation.add(this.output.maxtir, this.bias);
//			System.out.println((System.nanoTime() - start3) / 1e6+"ms->add bias");
		}
//		System.out.println((System.nanoTime() - start) / 1e6+"ms.");
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
		
		/**
		 * 计算deltaW
		 */
		this.computeDeltaW();
		
		if(this.hasBias) {
			this.deltaB = MatrixOperation.division(MatrixOperation.sumBias(this.delta.maxtir),this.number);
		}

		/**
		 * 梯度添加zeroPadding使得size与卷积输入一致
		 */
		float[][][][] deltaP = null;
		
		if(this.stride > 1) {
			deltaP = MatrixOperation.zeroPadding(this.dwd, this.stride);
		}else {
			deltaP = MatrixOperation.zeroPadding(this.delta.maxtir, this.diffPadding);
		}
		
		/**
		 * 旋转kernel180
		 */
		float[][][][] kernel180 = MatrixOperation.rotate180V2(this.kernel);

		/**
		 * 计算当前层梯度
		 */
		MatrixOperation.convnDeltaByIm2ColGPUV2(deltaP, kernel180, this.diff.maxtir, 1);

//		System.out.println((System.nanoTime() - start) / 1e6+"ms->all back========>");
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
//		long start = System.nanoTime();
//		System.out.println("back start.");
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
//		System.out.println((System.nanoTime() - start) / 1e6+"ms->all back");
	}

	@Override
	public void update() {
		// TODO Auto-generated method stub
//		long start = System.nanoTime();
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
//		System.out.println((System.nanoTime() - start) / 1e6+"ms->all update========>");
	}

	@Override
	public LayerType getLayerType() {
		// TODO Auto-generated method stub
		return LayerType.conv;
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
		
		System.out.println("conv layer["+this.index+"]diff-max:"+MathUtils.max(x)+" min:"+MathUtils.min(x));
		
	}

	@Override
	public LayerInit save() {
		// TODO Auto-generated method stub
		return new ConvLayerInit(this);
	}

	@Override
	public float[][][][] output(float[][][][] input) {
		// TODO Auto-generated method stub
		
		float[][][][] output = new float[this.number][this.oChannel][this.oHeight][this.oWidth];
		
		float[][][][] pInput = MatrixOperation.zeroPadding(input, this.padding);
		
		output = MatrixOperation.convnVailByIm2Col(pInput, this.kernel, this.stride, false);

		output = MatrixOperation.add(output, this.bias);
		
		return output;
	}

	@Override
	public void initCache() {
		// TODO Auto-generated method stub
		
		CacheDataSet cache = new CacheDataSet(this.number);
		
		this.setTampDataSet(cache);
		
		/**
		 * 创建输出层矩阵乘法缓存
		 */
		float[] r = new float[this.number * this.kernelNum * oHeight * oWidth];
		cache.getDim1dSet().add(r);
		
		/**
		 * 创建输入层im2col缓存
		 */
		float[][] col = new float[this.number * oHeight * oWidth][this.kHeight * this.kWidth * this.channel];
		cache.getDim2dSet().add(col);
		
		float[] col2 =  new float[this.number * oHeight * oWidth * this.kHeight * this.kWidth * this.channel];
		cache.getDim1dSet().add(col2);
	}
	
	/**
	 * 计算deltaW
	 */
	public void computeDeltaW (){
		int ko = this.kernelNum;
		int kh = this.delta.height;
		int kw = this.delta.width;
		
		if(this.stride > 1) {
			kh = this.dwd[0][0].length;
			kw = this.dwd[0][0][0].length;
		}
		
		int oHeight = ((this.pInput.height - kh ) / 1) + 1;
		int oWidth = ((this.pInput.width - kw) / 1) + 1;
		
		int xm = this.channel * oHeight * oWidth;
		int xn = this.number * kh * kw;
		
//		System.out.println(kh);
		
		/**
		 * input im2col
		 */
		Im2colForWeight.im2col(this.pInput.maxtir, this.dInput1D.data, kh, kw, 1);
		
		float[] delta1d = null;
		
		if(this.stride == 1) {
			delta1d = Im2colUtils.kernalToVector(this.delta.maxtir, true);
		}else {
			Dilation.dilation(this.delta.maxtir, this.dwd, this.stride);
			delta1d = Im2colUtils.kernalToVector(this.dwd, true);
		}

		float[] c = MatrixUtils.zero(xm * ko);
		
		GPUOP.getInstance().multiplyFloat(xm, xn, ko, this.dInput1D.data, delta1d, c);
		
		Im2col4d2T.to4d(c, this.deltaW, this.kernelNum, this.channel, oHeight, oWidth, this.number);
	}
	
}
