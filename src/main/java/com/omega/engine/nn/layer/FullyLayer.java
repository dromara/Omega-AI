package com.omega.engine.nn.layer;

import java.io.IOException;
import java.io.RandomAccessFile;

import com.omega.common.data.Tensor;
import com.omega.common.utils.MatrixUtils;
import com.omega.common.utils.RandomUtils;
import com.omega.engine.nn.layer.gpu.FullyKernel;
import com.omega.engine.nn.network.Network;
import com.omega.engine.nn.network.utils.ModelUtils;
import com.omega.engine.updater.UpdaterFactory;

import jcuda.Sizeof;
import jcuda.jcublas.cublasOperation;

/**
 * 
 * FullyLayer
 * 
 * @author Administrator
 *
 */
public class FullyLayer extends Layer{
	
	private FullyKernel kernel;
	
//	private FullyCudnnKernel cudnnKernel;
	
	public FullyLayer(int inputNum,int outputNum) {
		this.channel = 1;
		this.height = 1;
		this.width = inputNum;
		this.oChannel = channel;
		this.oHeight = height;
		this.oWidth = outputNum;
		this.hasParams = true;
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
		this.hasParams = true;
		this.initParam();
	}
	
	public FullyLayer(int inputNum,int outputNum,boolean hasBias,Network network) {
		this.network = network;
		this.channel = 1;
		this.height = 1;
		this.width = inputNum;
		this.oChannel = channel;
		this.oHeight = height;
		this.oWidth = outputNum;
		this.hasBias = hasBias;
		this.hasParams = true;
		this.initParam();
		network.paramLayers.add(this);
		this.setUpdater(UpdaterFactory.create(network.updater, network.updaterParams));
	}
	
	public FullyLayer(int inputNum,int outputNum,int time,boolean hasBias,Network network) {
		this.network = network;
		this.channel = 1;
		this.height = 1;
		this.width = inputNum;
		this.oChannel = channel;
		this.oHeight = height;
		this.oWidth = outputNum;
		this.hasBias = hasBias;
		this.hasParams = true;
		network.paramLayers.add(this);
		this.initParamRNNCell();
		this.setUpdater(UpdaterFactory.create(network.updater, network.updaterParams));
	}
	
	public static FullyLayer createRNNCell(int inputNum,int outputNum,int time,boolean hasBias,Network network) {
		return new FullyLayer(inputNum, outputNum, time, hasBias, network);
	}
	
	public void initKernel() {
//		if(this.network.CUDNN && cudnnKernel == null) {
//			cudnnKernel = new FullyCudnnKernel(network, width, oWidth);
//		}
		if(kernel == null) {
			kernel = new FullyKernel(cuda());
		}
	}
	
	@Override
	public void initBack() {
		// TODO Auto-generated method stub
		if(this.diff == null || this.number != this.diff.number){
			this.diff = new Tensor(number, channel, height, width, true, true);
			if(this.diffW == null) {
				this.diffW = new Tensor(1, 1, oWidth, width, true, true);
				if(hasBias) {
					this.diffB = new Tensor(1, 1, 1, oWidth, true);
				}
			}
		}
	}
	
	public void initPatamDiff() {
		if(this.diffW == null) {
			this.diffW = new Tensor(1, 1, oWidth, width, true, true);
			if(hasBias) {
				this.diffB = new Tensor(1, 1, 1, oWidth, true);
			}
		}
	}

	@Override
	public void init() {
		initKernel();
		// TODO Auto-generated method stub
		this.number = this.network.number;
		if(this.output == null || this.number != this.output.number){
			this.output = Tensor.createGPUTensor(this.output, number, oChannel, oHeight, oWidth, true);
		}
	}
	
	public void init(Tensor input) {
		// TODO Auto-generated method stub
		initKernel();
		this.number = input.number;
		if(this.output == null || this.number != this.output.number){
			this.output = Tensor.createGPUTensor(this.output, this.number, oChannel, oHeight, oWidth, true);
		}
	}
	
	@Override
	public void initParam() {
		// TODO Auto-generated method stub
//		this.weight = new Tensor(1, 1, oWidth, width, RandomUtils.xavierUniform(this.width * this.oWidth, width, oWidth, 1.0f), true);
		this.weight = new Tensor(1, 1, oWidth, width, RandomUtils.kaiming_uniform(this.width * this.oWidth, this.width, this.paramsInit), true);
//		this.weight = new Tensor(1, 1, width, oWidth, RandomUtils.kaiming_normal(this.width * this.oWidth, this.width, this.paramsInit), true);
//		this.weight = new Tensor(1, 1, width, oWidth,RandomUtils.xavierReluRandom(this.width * this.oWidth, this.width, this.oWidth), true);
//		this.weight = new Tensor(1, 1, width, oWidth, RandomUtils.kaimingNormalRandom(this.width * this.oWidth, 0, this.oWidth), true);
//		this.weight = new Tensor(1, 1, width, oWidth, RandomUtils.kaimingUniformRandom(this.width * this.oWidth, 0, this.oWidth), true);
//		this.weight = new Tensor(1, 1, width, oWidth,RandomUtils.xavierRandom(this.width * this.oWidth, this.width, this.oWidth));
//		this.weight = new Tensor(1, 1, width, oWidth,RandomUtils.order(this.width * this.oWidth, 0.001f, 0.001f), true);
//		this.weight = new Tensor(1, 1, oWidth, width,RandomUtils.order(this.width * this.oWidth, 0.001f, 0.001f), true);
//		this.weight = new Tensor(1, 1, width, oWidth,RandomUtils.val(this.width * this.oWidth, 0.1f), true);
//		this.weight = new Tensor(1, 1, width, oWidth, RandomUtils.heRandom(this.width * this.oWidth, this.width * this.oWidth));
//		if(this.network!=null){
//			this.diffW = this.network.createParamterGrad(1, 1, width, oWidth, true);
//		}else {
//			this.diffW = new Tensor(1, 1, width, oWidth, true, true);
//		}
		
//		this.diffW = new Tensor(1, 1, oWidth, width, true, true);
		if(hasBias){
			this.bias = new Tensor(1, 1, 1, oWidth, MatrixUtils.one(oWidth), true);
//			if(this.network != null){
//				this.diffB = this.network.createParamterGrad(1, 1, 1, oWidth, true);
//			}else {
//				this.diffB = new Tensor(1, 1, 1, oWidth, true);
//			}
			
		}
	}
	
	public void initParamRNNCell() {
		// TODO Auto-generated method stub
		this.weight = new Tensor(1, 1, width, oWidth, RandomUtils.uniformFloat(this.width * this.oWidth, this.oWidth), true);
		this.diffW = this.network.createParamterGrad(1, 1, width, oWidth, true);
		if(hasBias) {
			this.bias = new Tensor(1, 1, 1, oWidth, RandomUtils.uniformFloat(oWidth, this.oWidth), true);
			this.diffB = this.network.createParamterGrad(1, 1, 1, oWidth, true);
		}
	}
	
	@Override
	public void output() {
		
		// TODO Auto-generated method stub
		
//		if(this.network.CUDNN) {
//			cudnnKernel.conv(input, weight, output);
//		}else {
//			if(this.input != null) {
////				input.showDMByNumber(0);
////				GPU_OP().multiplyFloatEX(cublasOperation.CUBLAS_OP_N, cublasOperation.CUBLAS_OP_T, this.number, this.oWidth, this.width, 1, input.getGpuData(),
////						this.width, weight.getGpuData(), this.width, 0, output.getGpuData(), this.oWidth);
//				GPU_OP().multiplyFloat(number, oWidth, width, input.getGpuData(), weight.getGpuData(), output.getGpuData(),
//						cublasOperation.CUBLAS_OP_N, cublasOperation.CUBLAS_OP_N, 1.0f, 0.0f);
////				output.showDMByNumber(0);
//			}
//		}
		
		if(this.input != null) {
//			input.showDMByNumber(0);
			GPU_OP().multiplyFloatEX(cublasOperation.CUBLAS_OP_N, cublasOperation.CUBLAS_OP_T, this.number, this.oWidth, this.width, 1, input.getGpuData(),
					this.width, weight.getGpuData(), this.width, 0, output.getGpuData(), this.oWidth);
//			GPU_OP().multiplyFloat(number, oWidth, width, input.getGpuData(), weight.getGpuData(), output.getGpuData(),
//					cublasOperation.CUBLAS_OP_N, cublasOperation.CUBLAS_OP_N, 1.0f, 0.0f);
//			output.showDMByNumber(0);
		}
		
		if(hasBias) {
			kernel.addBias(output, bias);
		}
		
	}
	
	public void output(int batch,int step) {
		
		// TODO Auto-generated method stub
		
		if(this.input != null) {

			GPU_OP().multiplyFloat(batch, oWidth, width, input.getGpuData().withByteOffset(step * batch * input.getOnceSize() * Sizeof.FLOAT),
					weight.getGpuData(), output.getGpuData().withByteOffset(step * batch * output.getOnceSize() * Sizeof.FLOAT),
					cublasOperation.CUBLAS_OP_N, cublasOperation.CUBLAS_OP_N, 1.0f, 0.0f);

			if(hasBias) {
				kernel.addBias(output, bias, batch, step);
			}
			
		}
		
	}
	
	public void output(int batch,int inputStep,int step) {
		
		// TODO Auto-generated method stub
		
		if(this.input != null) {
			
			if(inputStep >= 0) {

				GPU_OP().multiplyFloat(batch, oWidth, width, input.getGpuData().withByteOffset(inputStep * batch * input.getOnceSize() * Sizeof.FLOAT),
						weight.getGpuData(), output.getGpuData().withByteOffset(step * batch * output.getOnceSize() * Sizeof.FLOAT),
						cublasOperation.CUBLAS_OP_N, cublasOperation.CUBLAS_OP_N, 1.0f, 0.0f);

			}
			
			if(hasBias) {
				kernel.addBias(output, bias, batch, step);
			}
			
		}
		
	}
	
	public void output(Tensor input,int batch,int inputStep,int step) {
		
		// TODO Auto-generated method stub
		
		if(this.input != null) {
			
			if(inputStep >= 0) {

				GPU_OP().multiplyFloat(batch, oWidth, width, input.getGpuData().withByteOffset(inputStep * batch * input.getOnceSize() * Sizeof.FLOAT),
						weight.getGpuData(), output.getGpuData().withByteOffset(step * batch * output.getOnceSize() * Sizeof.FLOAT),
						cublasOperation.CUBLAS_OP_N, cublasOperation.CUBLAS_OP_N, 1.0f, 0.0f);

			}
			
			if(hasBias) {
				kernel.addBias(output, bias, batch, step);
			}
			
		}
		
	}
	
	@Override
	public void diff() {
		// TODO Auto-generated method stub
		
//		if(this.network.CUDNN){
//			cudnnKernel.dw(input, delta, diffW);
//			cudnnKernel.dx(delta, weight, diff);
//		}else {

			/**
			 * deltaW = inputT * delta
			 * int m,int n,int k, float A[],float B[], float C[],int CUBLAS_OP_A,int CUBLAS_OP_B,float alpha,float beta
			 * number * w
			 * number * ow
			 * m = w,k = number,n = ow
			 */
			GPU_OP().multiplyFloatEX(cublasOperation.CUBLAS_OP_T, cublasOperation.CUBLAS_OP_N, this.oWidth, this.width, this.number, 1,
					delta.getGpuData(), this.oWidth, input.getGpuData(), this.width, 0, diffW.getGpuData(), this.width);
//			GPU_OP().multiplyFloat(this.width, this.oWidth, this.number, input.getGpuData(), delta.getGpuData(), diffW.getGpuData(),
//					cublasOperation.CUBLAS_OP_T, cublasOperation.CUBLAS_OP_N, 1.0f, 0.0f);

			/**
			 * diff = delta * weightT
			 * number * ow
			 * w * ow
			 * m = number,k = ow,n = w
			 */
			GPU_OP().multiplyFloatEX(cublasOperation.CUBLAS_OP_N, cublasOperation.CUBLAS_OP_N, this.number, this.width, this.oWidth, 1, delta.getGpuData(), this.oWidth,
					weight.getGpuData(), this.width, 0, diff.getGpuData(), this.width);
//			GPU_OP().multiplyFloat(this.number, this.width, this.oWidth, delta.getGpuData(), weight.getGpuData(), diff.getGpuData(),
//					cublasOperation.CUBLAS_OP_N, cublasOperation.CUBLAS_OP_T, 1.0f, 0.0f);
//		}

		if(hasBias) {
			kernel.backwardBias(diffB, delta);
		}
//		System.out.println("------------------");
//		System.out.println(delta.isZero());
//		System.out.println(diff.isZero());
//		delta.showDMByNumber(0);
//		diff.showDMByNumber(0);
	}
	
	public void diff(Tensor diff) {
		// TODO Auto-generated method stub
		
		/**
		 * deltaW = inputT * delta
		 * int m,int n,int k, float A[],float B[], float C[],int CUBLAS_OP_A,int CUBLAS_OP_B,float alpha,float beta
		 * number * w
		 * number * ow
		 * m = w,k = number,n = ow
		 */
		GPU_OP().multiplyFloatEX(cublasOperation.CUBLAS_OP_T, cublasOperation.CUBLAS_OP_N, this.oWidth, this.width, this.number, 1,
				delta.getGpuData(), this.oWidth, input.getGpuData(), this.width, 0, diffW.getGpuData(), this.width);
//		GPU_OP().multiplyFloat(this.width, this.oWidth, this.number, input.getGpuData(), delta.getGpuData(), diffW.getGpuData(),
//				cublasOperation.CUBLAS_OP_T, cublasOperation.CUBLAS_OP_N, 1.0f, 0.0f);

		/**
		 * diff = delta * weightT
		 * number * ow
		 * w * ow
		 * m = number,k = ow,n = w
		 */
		GPU_OP().multiplyFloatEX(cublasOperation.CUBLAS_OP_N, cublasOperation.CUBLAS_OP_N, this.number, this.width, this.oWidth, 1, delta.getGpuData(), this.oWidth,
				weight.getGpuData(), this.width, 0, diff.getGpuData(), this.width);
//		GPU_OP().multiplyFloat(this.number, this.width, this.oWidth, delta.getGpuData(), weight.getGpuData(), diff.getGpuData(),
//				cublasOperation.CUBLAS_OP_N, cublasOperation.CUBLAS_OP_T, 1.0f, 0.0f);
		
		if(hasBias) {
			kernel.backwardBias(diffB, delta);
		}

	}
	
	public void diff(int batch,int step) {
		// TODO Auto-generated method stub

		/**
		 * deltaW = inputT * delta
		 * int m,int n,int k, float A[],float B[], float C[],int CUBLAS_OP_A,int CUBLAS_OP_B,float alpha,float beta
		 * number * w
		 * number * ow
		 * m = w,k = number,n = ow
		 */
		GPU_OP().multiplyFloat(this.width, this.oWidth, batch, input.getGpuData().withByteOffset(step * batch * input.getOnceSize() * Sizeof.FLOAT),
				delta.getGpuData().withByteOffset(step * batch * delta.getOnceSize() * Sizeof.FLOAT), diffW.getGpuData(),
				cublasOperation.CUBLAS_OP_T, cublasOperation.CUBLAS_OP_N, 1.0f, 1.0f);

		if(hasBias) {
			kernel.backwardBias(diffB, delta, batch, step);
		}

		/**
		 * diff = delta * weightT
		 * number * ow
		 * w * ow
		 * m = number,k = ow,n = w
		 */
		GPU_OP().multiplyFloat(batch, this.width, this.oWidth, delta.getGpuData().withByteOffset(step * batch * delta.getOnceSize() * Sizeof.FLOAT),
				weight.getGpuData(), diff.getGpuData().withByteOffset(step * batch * diff.getOnceSize() * Sizeof.FLOAT),
				cublasOperation.CUBLAS_OP_N, cublasOperation.CUBLAS_OP_T, 1.0f, 0.0f);

	}
	
	public void diff(int batch,int inputStep,int step) {
		// TODO Auto-generated method stub

		/**
		 * deltaW = inputT * delta
		 * int m,int n,int k, float A[],float B[], float C[],int CUBLAS_OP_A,int CUBLAS_OP_B,float alpha,float beta
		 * number * w
		 * number * ow
		 * m = w,k = number,n = ow
		 */
		GPU_OP().multiplyFloat(this.width, this.oWidth, batch, input.getGpuData().withByteOffset(inputStep * batch * input.getOnceSize() * Sizeof.FLOAT),
				delta.getGpuData().withByteOffset(step * batch * delta.getOnceSize() * Sizeof.FLOAT), diffW.getGpuData(),
				cublasOperation.CUBLAS_OP_T, cublasOperation.CUBLAS_OP_N, 1.0f, 1.0f);

		
		if(hasBias) {
			kernel.backwardBias(diffB, delta, batch, step);
		}

		/**
		 * diff = delta * weightT
		 * number * ow
		 * w * ow
		 * m = number,k = ow,n = w
		 */
		GPU_OP().multiplyFloat(batch, this.width, this.oWidth, delta.getGpuData().withByteOffset(step * batch * delta.getOnceSize() * Sizeof.FLOAT),
				weight.getGpuData(), diff.getGpuData().withByteOffset(step * batch * diff.getOnceSize() * Sizeof.FLOAT),
				cublasOperation.CUBLAS_OP_N, cublasOperation.CUBLAS_OP_T, 1.0f, 0.0f);

	}
	
	public void diff(int batch,int inputStep,int deltaStep,int step) {
		// TODO Auto-generated method stub
		
		if(step >= 0) {
			/**
			 * deltaW = inputT * delta
			 * int m,int n,int k, float A[],float B[], float C[],int CUBLAS_OP_A,int CUBLAS_OP_B,float alpha,float beta
			 * number * w
			 * number * ow
			 * m = w,k = number,n = ow
			 */
			GPU_OP().multiplyFloat(this.width, this.oWidth, batch, input.getGpuData().withByteOffset(step * batch * input.getOnceSize() * Sizeof.FLOAT),
					delta.getGpuData().withByteOffset(inputStep * batch * delta.getOnceSize() * Sizeof.FLOAT), diffW.getGpuData(),
					cublasOperation.CUBLAS_OP_T, cublasOperation.CUBLAS_OP_N, 1.0f, 1.0f);
		

			if(hasBias) {
				kernel.backwardBias(diffB, delta, batch, inputStep);
			}

			/**
			 * diff = delta * weightT
			 * number * ow
			 * w * ow
			 * m = number,k = ow,n = w
			 */
			GPU_OP().multiplyFloat(batch, this.width, this.oWidth, delta.getGpuData().withByteOffset(inputStep * batch * delta.getOnceSize() * Sizeof.FLOAT),
					weight.getGpuData(), diff.getGpuData().withByteOffset(step * batch * diff.getOnceSize() * Sizeof.FLOAT),
					cublasOperation.CUBLAS_OP_N, cublasOperation.CUBLAS_OP_T, 1.0f, 0.0f);
		}

	}
	
	public void diff(Tensor input,Tensor diff,int batch,int deltaStep) {
		// TODO Auto-generated method stub
		
		/**
		 * deltaW = inputT * delta
		 * int m,int n,int k, float A[],float B[], float C[],int CUBLAS_OP_A,int CUBLAS_OP_B,float alpha,float beta
		 * number * w
		 * number * ow
		 * m = w,k = number,n = ow
		 */
		GPU_OP().multiplyFloat(this.width, this.oWidth, batch, input.getGpuData(),
				delta.getGpuData().withByteOffset(deltaStep * batch * delta.getOnceSize() * Sizeof.FLOAT), diffW.getGpuData(),
				cublasOperation.CUBLAS_OP_T, cublasOperation.CUBLAS_OP_N, 1.0f, 1.0f);

		if(hasBias) {
			kernel.backwardBias(diffB, delta, batch, deltaStep);
		}

		/**
		 * diff = delta * weightT
		 * number * ow
		 * w * ow
		 * m = number,k = ow,n = w
		 */
		GPU_OP().multiplyFloat(batch, this.width, this.oWidth, delta.getGpuData().withByteOffset(deltaStep * batch * delta.getOnceSize() * Sizeof.FLOAT),
				weight.getGpuData(), diff.getGpuData(),
				cublasOperation.CUBLAS_OP_N, cublasOperation.CUBLAS_OP_T, 1.0f, 0.0f);

	}
	
	public void diffTemp() {
		// TODO Auto-generated method stub

		/**
		 * deltaW = inputT * delta
		 * int m,int n,int k, float A[],float B[], float C[],int CUBLAS_OP_A,int CUBLAS_OP_B,float alpha,float beta
		 * number * w
		 * number * ow
		 * m = w,k = number,n = ow
		 */
		GPU_OP().multiplyFloat(this.width, this.oWidth, this.number, input.getGpuData(), delta.getGpuData(), diffW.getGpuData(),
				cublasOperation.CUBLAS_OP_T, cublasOperation.CUBLAS_OP_N, 0.5f, 1.0f);

		
		if(hasBias) {
			kernel.backwardBias(diffB, delta);
		}

		/**
		 * diff = delta * weightT
		 * number * ow
		 * w * ow
		 * m = number,k = ow,n = w
		 */
		GPU_OP().multiplyFloat(this.number, this.width, this.oWidth, delta.getGpuData(), weight.getGpuData(), diff.getGpuData(),
				cublasOperation.CUBLAS_OP_N, cublasOperation.CUBLAS_OP_T, 1.0f, 1.0f);
		
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
	
	public void backTemp() {
		// TODO Auto-generated method stub

		this.initBack();
		/**
		 * 设置梯度
		 */
		this.setDelta();
		/**
		 * 计算梯度
		 */
		this.diffTemp();
		
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
		if(!this.freeze) {
			if(accDW != null) {
				this.accDW.copy(diffW);
				if(hasBias) {
					this.accDB.copy(diffB);
				}
			}
			if(this.updater != null){
				this.updater.update(this);
			}else{
				for(int i = 0;i<this.weight.getDataLength();i++) {
					this.weight.data[i] -= this.learnRate * this.diffW.data[i];
				}
				for(int i = 0;i<this.bias.getDataLength();i++) {
					this.bias.data[i] -= this.learnRate * this.diffB.data[i];
				}
			}
			this.clearAccGrad();
		}

	}
	
	@Override
	public void accGrad(float scale) {
		// TODO Auto-generated method stub
		if(accDW == null) {
			accDW = diffW.copyGPU();
		}else {
			kernel.axpy_gpu(diffW, accDW, accDW.dataLength, scale, 1, 1);
		}
		if(hasBias) {
			if(accDB == null) {
				accDB = diffB.copyGPU();
			}else {
				kernel.axpy_gpu(diffB, accDB, accDB.dataLength, scale, 1, 1);
			}
		}
	}
	
	public void update(int batchSize) {
		// TODO Auto-generated method stub
		if(!this.freeze) {
			if(this.updater != null){
				this.updater.update(this, batchSize);
			}else{
				for(int i = 0;i<this.weight.getDataLength();i++) {
					this.weight.data[i] -= this.learnRate * this.diffW.data[i];
				}
				if(hasBias) {
					for(int i = 0;i<this.bias.getDataLength();i++) {
						this.bias.data[i] -= this.learnRate * this.diffB.data[i];
					}
				}
			}
		}
	}

	@Override
	public Tensor getOutput() {
		// TODO Auto-generated method stub
		return this.output;
	}

	@Override
	public void showDiff() {
		// TODO Auto-generated method stub

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

	@Override
	public void forward(Tensor input) {
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
		this.output();
	}
	
	public void forward(Tensor input,Tensor output) {
		// TODO Auto-generated method stub
		this.output = output;
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
		this.output();
	}
	
	public void forward(Tensor input,int batch,int step) {
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
	
	public void forward(Tensor input,int batch,int inputStep,int step) {
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
		this.output(batch, inputStep, step);
	}
	
	public void forward(Tensor input,int batch) {
		// TODO Auto-generated method stub
		/**
		 * 参数初始化
		 */
		this.init();

		/**
		 * 计算输出
		 */
		this.output(input, batch, 0, 0);
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
	
	public void back(Tensor delta,Tensor diff) {
		// TODO Auto-generated method stub

		initPatamDiff();
		/**
		 * 设置梯度
		 */
		this.setDelta(delta);
		/**
		 * 计算梯度
		 */
		this.diff(diff);
		
		if(this.network.GRADIENT_CHECK) {
			this.gradientCheck();
		}

	}
	
	public void back(Tensor delta,int batch,int step) {
		// TODO Auto-generated method stub

		this.initBack();
		/**
		 * 设置梯度
		 */
		this.setDelta(delta);
		/**
		 * 计算梯度
		 */
		this.diff(batch, step);
		
		if(this.network.GRADIENT_CHECK) {
			this.gradientCheck();
		}

	}
	
	public void back(Tensor delta,int batch,int inputStep,int step) {
		// TODO Auto-generated method stub

		this.initBack();
		/**
		 * 设置梯度
		 */
		this.setDelta(delta);
		/**
		 * 计算梯度
		 */
		this.diff(batch, inputStep, step);
		
		if(this.network.GRADIENT_CHECK) {
			this.gradientCheck();
		}

	}
	
	public void back(Tensor delta,int batch,int inputStep,int outputStep,int step) {
		// TODO Auto-generated method stub

		this.initBack();
		/**
		 * 设置梯度
		 */
		this.setDelta(delta);
		/**
		 * 计算梯度
		 */
		this.diff(batch, inputStep, outputStep, step);
		
		if(this.network.GRADIENT_CHECK) {
			this.gradientCheck();
		}

	}
	
	public void back(Tensor delta,Tensor input,Tensor diff,int batch,int deltaStep) {
		// TODO Auto-generated method stub

		this.initBack();
		/**
		 * 设置梯度
		 */
		this.setDelta(delta);
		/**
		 * 计算梯度
		 */
		this.diff(input, diff, batch, deltaStep);
		
		if(this.network.GRADIENT_CHECK) {
			this.gradientCheck();
		}

	}
	
	public void clear() {
		if(hasBias) {
			this.diffB.clearGPU();
		}
		this.diffW.clearGPU();
	}
	
	public void saveModel(RandomAccessFile outputStream) throws IOException {
		
		ModelUtils.saveParams(outputStream, weight);
		
		if(hasBias) {
			ModelUtils.saveParams(outputStream, bias);
		}
		
	}
	
	public void loadModel(RandomAccessFile inputStream) throws IOException {
		
		ModelUtils.loadParams(inputStream, weight);
		
		if(hasBias) {
			ModelUtils.loadParams(inputStream, bias);
		}
		
	}

	public void putParamters() {
		this.network.addPamamter(weight);
		if(hasBias) {
			this.network.addPamamter(bias);
		}
	}

	public void putParamterGrads() {
		this.network.addDeltaParamters(diffW);
		if(hasBias) {
			this.network.addDeltaParamters(diffB);
		}
	}
	
}
