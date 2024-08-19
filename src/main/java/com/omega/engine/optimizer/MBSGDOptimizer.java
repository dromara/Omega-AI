package com.omega.engine.optimizer;

import java.util.Arrays;

import com.omega.common.data.Tensor;
import com.omega.common.data.utils.DataTransforms;
import com.omega.common.utils.ImageUtils;
import com.omega.common.utils.JsonUtils;
import com.omega.common.utils.MathUtils;
import com.omega.common.utils.MatrixOperation;
import com.omega.common.utils.MatrixUtils;
import com.omega.common.utils.RandomUtils;
import com.omega.engine.check.BaseCheck;
import com.omega.engine.gpu.CUDAModules;
import com.omega.engine.nn.data.BaseData;
import com.omega.engine.nn.grad.GradClipping;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.network.DuffsionUNet;
import com.omega.engine.nn.network.Network;
import com.omega.engine.nn.network.OutputsNetwork;
import com.omega.engine.nn.network.RunModel;
import com.omega.engine.nn.network.Yolo;
import com.omega.engine.optimizer.lr.LearnRateUpdate;
import com.omega.example.duffsion.utils.DuffsionImageDataLoader;
import com.omega.example.rnn.data.OneHotDataLoader;
import com.omega.example.rnn.data.RNNDataLoader;
import com.omega.example.yolo.data.BaseDataLoader;
import com.omega.example.yolo.data.DetectionDataLoader;
import com.omega.example.yolo.utils.YoloLabelUtils;

import jcuda.driver.JCudaDriver;

/**
 * 
 * Mini Batch Stochastic Gradient Descent
 * 
 * @author Administrator
 *
 */
public class MBSGDOptimizer extends Optimizer {
	
	private YoloLabelUtils u;

	public YoloLabelUtils dataEnhanceInstance() {
		if(u == null) {
			u = new YoloLabelUtils(1, 4);
		}
		return u;
	}
	
	public MBSGDOptimizer(Network network, int trainTime, float error,int batchSize,boolean warmUp) throws Exception {
		super(network, batchSize, trainTime, error, warmUp);
		// TODO Auto-generated constructor stub
		this.batchSize = batchSize;
		this.loss = new Tensor(batchSize, this.network.oChannel, this.network.oHeight, this.network.oWidth);
		this.lossDiff = new Tensor(batchSize, this.network.oChannel, this.network.oHeight, this.network.oWidth);
	}
	
	public MBSGDOptimizer(String sid,Network network, int trainTime, float error,int batchSize,boolean warmUp) throws Exception {
		super(network, batchSize, trainTime, error, warmUp);
		// TODO Auto-generated constructor stub
		this.setSid(sid);
		this.batchSize = batchSize;
		this.loss = new Tensor(batchSize, this.network.oChannel, this.network.oHeight, this.network.oWidth);
		this.lossDiff = new Tensor(batchSize, this.network.oChannel, this.network.oHeight, this.network.oWidth);
	}

	public MBSGDOptimizer(Network network, int trainTime, float error,int batchSize,LearnRateUpdate learnRateUpdate,boolean warmUp) throws Exception {
		super(network, batchSize, trainTime, error, warmUp);
		// TODO Auto-generated constructor stub
		this.batchSize = batchSize;
		this.loss = new Tensor(batchSize, this.network.oChannel, this.network.oHeight, this.network.oWidth);
		this.lossDiff = new Tensor(batchSize, this.network.oChannel, this.network.oHeight, this.network.oWidth);
		this.learnRateUpdate = learnRateUpdate;
	}
	
	public MBSGDOptimizer(Network network, int trainTime, float error,int batchSize,LearnRateUpdate learnRateUpdate,boolean warmUp,BaseCheck check) throws Exception {
		super(network, batchSize, trainTime, error, warmUp);
		// TODO Auto-generated constructor stub
		this.batchSize = batchSize;
		this.loss = new Tensor(batchSize, this.network.oChannel, this.network.oHeight, this.network.oWidth);
		this.lossDiff = new Tensor(batchSize, this.network.oChannel, this.network.oHeight, this.network.oWidth);
		this.learnRateUpdate = learnRateUpdate;
		this.check = check;
	}
	
	public MBSGDOptimizer(String sid,Network network, int trainTime, float error,int batchSize,LearnRateUpdate learnRateUpdate,boolean warmUp) throws Exception {
		super(network, batchSize, trainTime, error, warmUp);
		// TODO Auto-generated constructor stub
		this.setSid(sid);
		this.batchSize = batchSize;
		this.loss = new Tensor(batchSize, this.network.oChannel, this.network.oHeight, this.network.oWidth);
		this.lossDiff = new Tensor(batchSize, this.network.oChannel, this.network.oHeight, this.network.oWidth);
		this.learnRateUpdate = learnRateUpdate;
	}
	
	@Override
	public void train(BaseData trainingData) {
		// TODO Auto-generated method stub

		try {
			
			CUDAModules.initCUDAFunctions();

			this.dataSize = trainingData.number;

			if(isWarmUp()) {
				this.network.learnRate = (float) (this.lr * Math.pow(batchIndex * 1.0f/burnIn * 1.0f, power));
			}
			
			Tensor input = new Tensor(batchSize, this.network.getChannel(), this.network.getHeight(), this.network.getWidth(), true);
			
			Tensor label = new Tensor(batchSize, 1, 1, trainingData.labelSize, true);
			
			for(int i = 0;i<this.trainTime;i++) {
				
				if(this.trainIndex >= this.minTrainTime) {
					break;
				}
				
				this.trainIndex = i + 1;
				
				int[][] indexs = MathUtils.randomInts(trainingData.number,this.batchSize);
				
				/**
				 * 遍历整个训练集
				 */
				for(int it = 0;it<indexs.length;it++) {
//				for(int it = 0;it<1;it++) {
					
					if(Math.abs(this.currentError) <= this.error) {
						break;
					}
					
					long start = System.nanoTime();

					this.loss.clear();
					
					this.lossDiff.clear();
					
					trainingData.getRandomData(indexs[it], input, label); 

					input.hostToDevice();
					
					label.hostToDevice();
					
//					input.showDM();
					
//					long output_start = System.nanoTime();
					
					/**
					 * forward
					 */
					Tensor output = this.network.forward(input);
					
//					System.out.println(JsonUtils.toJson(output.data));
//					System.out.println("output1:"+(System.nanoTime() - output_start) / 1e6 + "ms.");
					
//					output.syncHost();
					
//					System.out.println(JsonUtils.toJson(output.data));
					
//					System.out.println("output2:"+(System.nanoTime() - output_start) / 1e6 + "ms.");
					
					/**
					 * loss
					 */
					this.loss = this.network.loss(output, label);
					
					/**
					 * loss diff
					 */
					this.lossDiff = this.network.lossDiff(output, label);
					
//					System.out.println("=========>:"+JsonUtils.toJson(lossDiff.data));

					/**
					 * current time error
					 */
					if(this.loss.isHasGPU()){
						this.currentError = MatrixOperation.sum(this.loss.syncHost()) / this.batchSize;
					}else {
						this.currentError = MatrixOperation.sum(this.loss.data) / this.batchSize;
					}

//					long back_start = System.nanoTime();
					
					lossDiff.hostToDevice();
					
					/**
					 * back
					 */
					this.network.back(this.lossDiff);
					
					/**
					 * update
					 */
					this.network.update();
					
					output.syncHost();

//					System.out.println("back:"+(System.nanoTime() - back_start) / 1e6 + "ms.");
					
					float error = this.accuracy(output, label, trainingData.labelSet);
					
					String msg = "training["+this.trainIndex+"]{"+it+"} (lr:"+this.network.learnRate+") accuracy:{"+error+"%} currentError:"+this.currentError + " [costTime:"+(System.nanoTime() - start)/1e6+"ms.]";
					
					System.out.println(msg);

//					/**
//					 * update learning rate
//					 */
//					this.updateLR();
					
					this.batchIndex++;
				}
				
				/**
				 * update learning rate
				 */
				this.updateLR(this.lr_step);

			}
			
			/**
			 * 停止训练
			 */
			System.out.println("training finish. ["+this.trainIndex+"] finalError:"+this.currentError);
//			System.out.println(JsonUtils.toJson(this.network.layerList));
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}

	}

	@Override
	public void train(BaseData trainingData, BaseData testData) {
		// TODO Auto-generated method stub

		try {
			
			CUDAModules.initCUDAFunctions();

			this.dataSize = trainingData.number;

			if(isWarmUp()) {
				this.network.learnRate = (float) (this.lr * Math.pow(batchIndex * 1.0f/burnIn * 1.0f, power));
			}
			
			Tensor input = new Tensor(batchSize, this.network.getChannel(), this.network.getHeight(), this.network.getWidth(), true);
			
			Tensor label = new Tensor(batchSize, 1, 1, trainingData.labelSize, true);
			
			for(int i = 0;i<this.trainTime;i++) {
				
				if(this.trainIndex >= this.minTrainTime) {
					break;
				}
				
				this.trainIndex = i + 1;
				
				int[][] indexs = MathUtils.randomInts(trainingData.number,this.batchSize);
				
				this.network.RUN_MODEL = RunModel.TRAIN;
				
				/**
				 * 遍历整个训练集
				 */
				for(int it = 0;it<indexs.length;it++) {
					
					if(Math.abs(this.currentError) <= this.error) {
						break;
					}
					
					long start = System.nanoTime();

//					this.loss.clear();
//					
//					this.lossDiff.clear();
					
					trainingData.getRandomData(indexs[it], input, label); 

					input.hostToDevice();
					
					label.hostToDevice();
					
//					input.showDM();
					
//					long output_start = System.nanoTime();
					
					/**
					 * forward
					 */
					Tensor output = this.network.forward(input);
					
//					System.out.println(JsonUtils.toJson(output.data));
//					System.out.println("output1:"+(System.nanoTime() - output_start) / 1e6 + "ms.");
					
//					System.out.println(JsonUtils.toJson(output.data));
					
//					System.out.println("output2:"+(System.nanoTime() - output_start) / 1e6 + "ms.");
					
					/**
					 * loss
					 */
					this.loss = this.network.loss(output, label);
					
					/**
					 * loss diff
					 */
					this.lossDiff = this.network.lossDiff(output, label);
					
//					System.out.println(JsonUtils.toJson(label.syncHost()));
//					
//					System.out.println(JsonUtils.toJson(output.syncHost()));
//					
//					System.out.println(JsonUtils.toJson(this.lossDiff.syncHost()));
					
//					System.out.println("=========>:"+JsonUtils.toJson(lossDiff.data));

//					long back_start = System.nanoTime();
					
//					loss.hostToDevice();
					
//					lossDiff.hostToDevice();
					
					/**
					 * back
					 */
					this.network.back(this.lossDiff);
					
					/**
					 * update
					 */
					this.network.update();
					
					JCudaDriver.cuCtxSynchronize();
					
//					System.out.println("back:"+(System.nanoTime() - back_start) / 1e6 + "ms.");

					output.syncHost();
					
					float error = this.accuracy(output, label, trainingData.labelSet);

					/**
					 * current time error
					 */
					this.currentError = MatrixOperation.sum(this.loss.syncHost()) / this.batchSize;
					
					String msg = "training["+this.trainIndex+"]{"+it+"} (lr:"+this.network.learnRate+") accuracy:{"+error+"%} currentError:"+this.currentError + " [costTime:"+(System.nanoTime() - start)/1e6+"ms.]";
					
					System.out.println(msg);
					
//					/**
//					 * update learning rate
//					 */
//					this.updateLR();
					
					this.batchIndex++;
				}
				
				/**
				 * update learning rate
				 */
				this.updateLR(this.lr_step);
				
				/**
				 * vail data test
				 */
				this.test(testData, this.batchSize);
			}
			
			/**
			 * 停止训练
			 */
			System.out.println("training finish. ["+this.trainIndex+"] finalError:"+this.currentError);
//			System.out.println(JsonUtils.toJson(this.network.layerList));
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}

	}

	@Override
	public void train(BaseData trainingData, BaseData validata, BaseData testData) {
		// TODO Auto-generated method stub
		
	}
	
	public void train(BaseData trainingData, BaseData validata, float[] mean, float[] std) {
		// TODO Auto-generated method stub

		try {
			
			CUDAModules.initCUDAFunctions();

			this.dataSize = trainingData.number;
			
			if(isWarmUp()) {
				this.network.learnRate = (float) (this.lr * Math.pow(batchIndex * 1.0f/burnIn * 1.0f, power));
			}
			
			Tensor input = new Tensor(batchSize, this.network.getChannel(), this.network.getHeight(), this.network.getWidth(), true);
			
			Tensor label = new Tensor(batchSize, 1, 1, trainingData.labelSize, true);
			
			Tensor transData = new Tensor(trainingData.number, trainingData.channel, trainingData.height, trainingData.width);
			
			Tensor vail_input = new Tensor(batchSize, validata.channel, validata.height, validata.width, true);
			
			Tensor vail_label = new Tensor(batchSize, 1, 1, validata.labelSize, true);

			for(int i = 0;i<this.trainTime;i++) {
				
				if(this.trainIndex >= this.minTrainTime) {
					break;
				}

				transforms(trainingData.input, transData, mean, std);
				
				this.trainIndex = i + 1;
				
				int[][] indexs = MathUtils.randomInts(trainingData.number,this.batchSize);

				this.network.RUN_MODEL = RunModel.TRAIN;
				
				float train_loss = 0.0f;
				
				/**
				 * 遍历整个训练集
				 */
				for(int it = 0;it<indexs.length;it++) {

					long start = System.nanoTime();

					if(Math.abs(this.currentError) <= this.error) {
						break;
					}

					trainingData.randomData(indexs[it], transData.data, input, label);

					input.hostToDevice();
					
					label.hostToDevice();

					/**
					 * forward
					 */
					Tensor output = this.network.forward(input);

					/**
					 * loss
					 */
					this.loss = this.network.loss(output, label);
					
					/**
					 * loss diff
					 */
					this.lossDiff = this.network.lossDiff(output, label);
					
					/**
					 * back
					 */
					this.network.back(this.lossDiff);
					
					/**
					 * update
					 */
					this.network.update();

					output.syncHost();

					float error = this.accuracy(output, label, trainingData.labelSet);

					/**
					 * current time error
					 */
					if(this.loss.isHasGPU()) {
						this.currentError = MatrixOperation.sum(this.loss.syncHost()) / this.batchSize;
					}else {
						this.currentError = MatrixOperation.sum(this.loss.data) / this.batchSize;
					}

					train_loss += this.currentError;
					
					String msg = "training["+this.trainIndex+"]{"+it+"} (lr:"+this.network.learnRate+") accuracy:{"+error+"%} train_loss:" + this.currentError + " [costTime:"+(System.nanoTime() - start)/1e6+"ms.]";
					
					System.out.println(msg);
					
					this.batchIndex++;
				}
				
				System.out.println("training["+this.trainIndex+"] train loss:{"+train_loss/indexs.length+"} ");
				
				/**
				 * vail data test
				 */
				float vail_loss = this.testAndLoss(validata, vail_input, vail_label, this.batchSize);

				/**
				 * update learning rate
				 */
				this.updateLR(vail_loss);
				
			}
			
			/**
			 * 停止训练
			 */
			System.out.println("training finish. ["+this.trainIndex+"] finalError:"+this.currentError);
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}

	}
	
	public void train(BaseDataLoader trainingData) {
		// TODO Auto-generated method stub

		try {
			
			CUDAModules.initCUDAFunctions();

			this.dataSize = trainingData.number;
			
//			/**
//			 * normalize vailSet
//			 */
//			DataTransforms.normalize(validata.input, mean, std);

			if(isWarmUp()) {
				this.network.learnRate = (float) (this.lr * Math.pow(batchIndex * 1.0f/burnIn * 1.0f, power));
			}
			
			Tensor input = new Tensor(batchSize, this.network.getChannel(), this.network.getHeight(), this.network.getWidth(), true);
			
			Tensor label = new Tensor(batchSize, 1, 1, trainingData.labelSize, true);
			
			for(int i = 0;i<this.trainTime;i++) {
				
				if(this.trainIndex >= this.minTrainTime) {
					break;
				}

				this.trainIndex = i + 1;
				
				int[][] indexs = trainingData.shuffle();

				this.network.RUN_MODEL = RunModel.TRAIN;
				
				float train_loss = 0.0f;
				
				/**
				 * 遍历整个训练集
				 */
				for(int it = 0;it<indexs.length;it++) {

					long start = System.nanoTime();

					if(Math.abs(this.currentError) <= this.error) {
						break;
					}
					
					trainingData.loadData(indexs[it], input, label);
					
//					System.out.println(JsonUtils.toJson(label.data));
					
//					input.hostToDevice();
//					
//					label.hostToDevice();

					/**
					 * forward
					 */
					Tensor output = this.network.forward(input);
					
//					System.out.println(JsonUtils.toJson(output.syncHost()));
					
					/**
					 * loss
					 */
					this.loss = this.network.loss(output, label);
					
					/**
					 * loss diff
					 */
					this.lossDiff = this.network.lossDiff(output, label);
					
					/**
					 * back
					 */
					this.network.back(this.lossDiff);
					
					/**
					 * update
					 */
					this.network.update();

//					output.syncHost();

//					float error = this.accuracy(output, label, trainingData.labelSet);
					
					JCudaDriver.cuCtxSynchronize();
					
					/**
					 * current time error
					 */
					if(this.loss.isHasGPU()) {
						this.currentError = MatrixOperation.sum(this.loss.syncHost()) / this.batchSize;
					}else {
						this.currentError = MatrixOperation.sum(this.loss.data) / this.batchSize;
					}

					train_loss += this.currentError;
					
					String msg = "training["+this.trainIndex+"]{"+it+"} (lr:"+this.network.learnRate+") accuracy:{"+error+"%} train_loss:" + this.currentError + " [costTime:"+(System.nanoTime() - start)/1e6+"ms.]";
					
					System.out.println(msg);
					
					this.batchIndex++;
				}
				
				System.out.println("training["+this.trainIndex+"] train loss:{"+train_loss/indexs.length+"} ");
				
//				/**
//				 * vail data test
//				 */
//				float vail_loss = this.testAndLoss(validata, vail_input, vail_label, this.batchSize);

				/**
				 * update learning rate
				 */
				this.updateLR(this.lr_step);
				
			}
			
			/**
			 * 停止训练
			 */
			System.out.println("training finish. ["+this.trainIndex+"] finalError:"+this.currentError);
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}

	}
	
	public void train(BaseDataLoader trainingData,BaseDataLoader valiData,BaseCheck check) {
		// TODO Auto-generated method stub
		try {
			
			CUDAModules.initCUDAFunctions();
			
			this.dataSize = trainingData.number;
			
			if(isWarmUp()) {
				this.network.learnRate = (float) (this.lr * Math.pow(batchIndex * 1.0f/burnIn * 1.0f, power));
			}

			Tensor input = new Tensor(batchSize, this.network.getChannel(), this.network.getHeight(), this.network.getWidth(), true);
			
			Tensor label = trainingData.initLabelTensor();
			
			for(int i = 0;i<this.trainTime;i++) {
				
				if(this.trainIndex >= this.minTrainTime) {
					break;
				}

				this.network.RUN_MODEL = RunModel.TRAIN;
				
				this.trainIndex = i + 1;
				
				int[][] indexs = trainingData.shuffle();
				
				/**
				 * 遍历整个训练集
				 */
				for(int it = 0;it<indexs.length;it++) {
					
					long start = System.nanoTime();

					this.loss.clear();

					this.lossDiff.clear();
					
					/**
					 * 读取训练数据
					 */
					trainingData.loadData(indexs[it], input, label);
					
					/**
					 * forward
					 */
					Tensor output = network.forward(input);
					
					/**
					 * loss
					 */
					Tensor loss = this.network.loss(output, label);
					
					/**
					 * loss diff
					 */
					this.lossDiff = network.lossDiff(output, label);
					
					/**
					 * back
					 */
					network.back(lossDiff);
					
					/**
					 * update
					 */
					this.network.update();
					
					if(loss.isHasGPU()) {
						loss.syncHost();
					}
					
					float accuracy = check.check(output, label, trainingData.labelSet, false);
					
					String msg = "training["+this.trainIndex+"]{"+it+"} (lr:"+this.network.learnRate+") (loss:"+loss.getByIndex(0, 0, 0, 0)+") (accuracy:"+accuracy/batchSize*100+"%) [costTime:"+(System.nanoTime() - start)/1e6+"ms.]";
					
					System.out.println(msg);

					this.batchIndex++;
				}
				
				/**
				 * update learning rate
				 */
				this.updateLR(this.lr_step);
				
				if(this.trainIndex % 10 == 0) {
					
					System.out.println("----------------testing start----------------");
					
					this.testAndLoss(valiData, input, label, this.batchSize, check);
					
					System.out.println("----------------testing finish---------------");
					
				}
				
			}
			
			/**
			 * 停止训练
			 */
			System.out.println("training finish. ["+this.trainIndex+"] finalError:"+this.currentError);
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}

	}
	
	public void trainObjectRecognition(BaseData trainingData) {
		// TODO Auto-generated method stub

		try {
			
			CUDAModules.initCUDAFunctions();

			this.dataSize = trainingData.number;

			if(isWarmUp()) {
				this.network.learnRate = (float) (this.lr * Math.pow(batchIndex * 1.0f/burnIn * 1.0f, power));
			}
			
			Tensor input = new Tensor(batchSize, this.network.getChannel(), this.network.getHeight(), this.network.getWidth(), true);
			
			Tensor label = new Tensor(batchSize, 1, 1, trainingData.labelSize, true);
			
			for(int i = 0;i<this.trainTime;i++) {
				
				if(this.trainIndex >= this.minTrainTime) {
					break;
				}
				
				this.trainIndex = i + 1;
				
				int[][] indexs = MathUtils.randomInts(trainingData.number,this.batchSize);

				/**
				 * 遍历整个训练集
				 */
				for(int it = 0;it<indexs.length;it++) {

					long start = System.nanoTime();

					this.loss.clear();
					
					this.lossDiff.clear();
					
					trainingData.getRandomData(indexs[it], input, label); 

					input.hostToDevice();
					
					label.hostToDevice();
					
//					input.showDM();
					
//					long output_start = System.nanoTime();
					
					/**
					 * forward
					 */
					Tensor output = this.network.forward(input);
					
//					System.out.println(JsonUtils.toJson(output.data));
//					System.out.println("output1:"+(System.nanoTime() - output_start) / 1e6 + "ms.");
					
//					output.syncHost();
					
//					System.out.println(JsonUtils.toJson(output.data));
					
//					System.out.println("output2:"+(System.nanoTime() - output_start) / 1e6 + "ms.");
					
					/**
					 * loss
					 */
					this.loss = this.network.loss(output, label);
					
					/**
					 * loss diff
					 */
					this.lossDiff = this.network.lossDiff(output, label);
					
//					System.out.println("=========>:"+JsonUtils.toJson(lossDiff.data));

//					long back_start = System.nanoTime();
					
					/**
					 * back
					 */
					this.network.back(this.lossDiff);
					
					/**
					 * update
					 */
					this.network.update();
					
//					JCudaDriver.cuCtxSynchronize();
					
					/**
					 * current time error
					 */
					if(this.loss.isHasGPU()) {
						this.currentError = MatrixOperation.sum(this.loss.syncHost()) / this.batchSize;
					}else {
						this.currentError = MatrixOperation.sum(this.loss.data) / this.batchSize;
					}
					
//					System.out.println("back:"+(System.nanoTime() - back_start) / 1e6 + "ms.");
					
//					float error = 0.0f;
//					
					String msg = "training["+this.trainIndex+"]{"+it+"} (lr:"+this.network.learnRate+") train_loss:"+this.currentError+" [costTime:"+(System.nanoTime() - start)/1e6+"ms.]";
					
					System.out.println(msg);

					this.batchIndex++;
				}
				
				/**
				 * update learning rate
				 */
				this.updateLR(this.lr_step);
				
			}
			
			/**
			 * 停止训练
			 */
			System.out.println("training finish. ["+this.trainIndex+"] finalError:"+this.currentError);
//			System.out.println(JsonUtils.toJson(this.network.layerList));
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}

	}
	
	public void trainObjectRecognition(BaseData trainingData,BaseData validata) {
		// TODO Auto-generated method stub

		try {
			
			CUDAModules.initCUDAFunctions();

			this.dataSize = trainingData.number;

			if(isWarmUp()) {
				this.network.learnRate = (float) (this.lr * Math.pow(batchIndex * 1.0f/burnIn * 1.0f, power));
			}
			
			Tensor input = new Tensor(batchSize, this.network.getChannel(), this.network.getHeight(), this.network.getWidth(), true);
			
			Tensor label = new Tensor(batchSize, 1, 1, trainingData.labelSize, true);
			
			Tensor vail_input = new Tensor(batchSize, validata.channel, validata.height, validata.width, true);
			
			Tensor vail_label = new Tensor(batchSize, 1, 1, validata.labelSize, true);
			
			for(int i = 0;i<this.trainTime;i++) {
				
				if(this.trainIndex >= this.minTrainTime) {
					break;
				}

				this.network.RUN_MODEL = RunModel.TRAIN;
				
				this.trainIndex = i + 1;
				
				int[][] indexs = MathUtils.randomInts(trainingData.number,this.batchSize);

				/**
				 * 遍历整个训练集
				 */
				for(int it = 0;it<indexs.length;it++) {
					
					long start = System.nanoTime();

					this.loss.clear();
					
					this.lossDiff.clear();
					
					trainingData.getRandomData(indexs[it], input, label);

					input.hostToDevice();
					
					label.hostToDevice();
					
					/**
					 * forward
					 */
					Tensor output = this.network.forward(input);

					/**
					 * loss
					 */
					this.loss = this.network.loss(output, label);
					
					/**
					 * loss diff
					 */
					this.lossDiff = this.network.lossDiff(output, label);
					
					/**
					 * back
					 */
					this.network.back(this.lossDiff);
					
					/**
					 * update
					 */
					this.network.update();
					
					/**
					 * current time error
					 */
					if(this.loss.isHasGPU()) {
						this.currentError = MatrixOperation.sum(this.loss.syncHost()) / this.batchSize;
					}else {
						this.currentError = MatrixOperation.sum(this.loss.data) / this.batchSize;
					}
//					
					String msg = "training["+this.trainIndex+"]{"+it+"} (lr:"+this.network.learnRate+") train_loss:"+this.currentError+" [costTime:"+(System.nanoTime() - start)/1e6+"ms.]";
					
					System.out.println(msg);

					this.batchIndex++;
				}
				
				/**
				 * update learning rate
				 */
				this.updateLR(this.lr_step);
				
				if(this.trainIndex % 100 == 0) {
					
					System.out.println("----------------testing start----------------");
					
					this.testObjectRecognition(validata, vail_input, vail_label, this.batchSize);
					
					System.out.println("----------------testing finish---------------");
					
				}
				
			}
			
			/**
			 * 停止训练
			 */
			System.out.println("training finish. ["+this.trainIndex+"] finalError:"+this.currentError);
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}

	}
	
	public void trainObjectRecognition(BaseData trainingData,BaseData validata,boolean dataEnhance) {
		// TODO Auto-generated method stub

		try {
			
			CUDAModules.initCUDAFunctions();

			this.dataSize = trainingData.number;

			if(isWarmUp()) {
				this.network.learnRate = (float) (this.lr * Math.pow(batchIndex * 1.0f/burnIn * 1.0f, power));
			}
			
			Tensor input = new Tensor(batchSize, this.network.getChannel(), this.network.getHeight(), this.network.getWidth(), true);
			
			Tensor label = new Tensor(batchSize, 1, 1, trainingData.labelSize, true);
			
			Tensor vail_input = new Tensor(batchSize, validata.channel, validata.height, validata.width, true);
			
			for(int i = 0;i<this.trainTime;i++) {
				
				if(this.trainIndex >= this.minTrainTime) {
					break;
				}

				this.network.RUN_MODEL = RunModel.TRAIN;
				
				this.trainIndex = i + 1;
				
				int[][] indexs = MathUtils.randomInts(trainingData.number,this.batchSize);

				/**
				 * 遍历整个训练集
				 */
				for(int it = 0;it<indexs.length;it++) {
					
					long start = System.nanoTime();

					this.loss.clear();
					
					this.lossDiff.clear();
					
					trainingData.getRandomData(indexs[it], input, label); 
					
					/**
					 * 数据增强
					 */
					if(dataEnhance) {
						dataEnhanceInstance().transforms(input, label);
						YoloLabelUtils.formatToYolo(label, input.height, input.width);
					}

					input.hostToDevice();
					
					label.hostToDevice();
					
					/**
					 * forward
					 */
					Tensor output = this.network.forward(input);

					/**
					 * loss
					 */
					this.loss = this.network.loss(output, label);
					
					/**
					 * loss diff
					 */
					this.lossDiff = this.network.lossDiff(output, label);
					
					/**
					 * back
					 */
					this.network.back(this.lossDiff);
					
					/**
					 * update
					 */
					this.network.update();
					
					/**
					 * current time error
					 */
					if(this.loss.isHasGPU()) {
						this.currentError = MatrixOperation.sum(this.loss.syncHost()) / this.batchSize;
					}else {
						this.currentError = MatrixOperation.sum(this.loss.data) / this.batchSize;
					}
//					
					String msg = "training["+this.trainIndex+"]{"+it+"} (lr:"+this.network.learnRate+") train_loss:"+this.currentError+" [costTime:"+(System.nanoTime() - start)/1e6+"ms.]";
					
					System.out.println(msg);

					this.batchIndex++;
				}
				
				/**
				 * update learning rate
				 */
				this.updateLR(this.lr_step);
				
				if(this.trainIndex % 100 == 0) {
					
					System.out.println("----------------testing start----------------");
					
					this.testObjectRecognition(validata, vail_input, label, this.batchSize);
					
					System.out.println("----------------testing finish---------------");
					
				}
				
			}
			
			/**
			 * 停止训练
			 */
			System.out.println("training finish. ["+this.trainIndex+"] finalError:"+this.currentError);
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}

	}
	
	public void trainObjectRecognition(DetectionDataLoader trainingData,DetectionDataLoader valiData) {
		// TODO Auto-generated method stub
		try {
			
			CUDAModules.initCUDAFunctions();
			
			this.dataSize = trainingData.number;
			
			Yolo network = (Yolo) this.network;

			if(isWarmUp()) {
				this.network.learnRate = (float) (this.lr * Math.pow(batchIndex * 1.0f/burnIn * 1.0f, power));
			}
			
			Tensor input = new Tensor(batchSize, this.network.getChannel(), this.network.getHeight(), this.network.getWidth(), true);
			
			Tensor label = trainingData.initLabelTensor();
			
			for(int i = 0;i<this.trainTime;i++) {
				
				if(this.trainIndex >= this.minTrainTime) {
					break;
				}

				this.network.RUN_MODEL = RunModel.TRAIN;
				
				this.trainIndex = i + 1;
				
				int[][] indexs = trainingData.shuffle();
				
				/**
				 * 遍历整个训练集
				 */
				for(int it = 0;it<indexs.length;it++) {
					
					long start = System.nanoTime();

					this.loss.clear();

					this.lossDiff.clear();
					
					/**
					 * 读取训练数据
					 */
					trainingData.loadData(indexs[it], input, label);
					
					/**
					 * forward
					 */
					Tensor output = network.forward(input);
					
					/**
					 * loss
					 */
					this.network.loss(output, label);
					
					/**
					 * loss diff
					 */
					this.lossDiff = network.lossDiff(output, label);
					
					/**
					 * back
					 */
					network.back(lossDiff);
					
					/**
					 * update
					 */
					this.network.update();
					
					String msg = "training["+this.trainIndex+"]{"+it+"} (lr:"+this.network.learnRate+") [costTime:"+(System.nanoTime() - start)/1e6+"ms.]";
					
					System.out.println(msg);

					this.batchIndex++;
				}
				
				/**
				 * update learning rate
				 */
				this.updateLR(this.lr_step);
				
				if(this.trainIndex % 100 == 0) {
					
					System.out.println("----------------testing start----------------");
					
					this.testObjectRecognition(valiData, input, label, this.batchSize);
					
					System.out.println("----------------testing finish---------------");
					
				}
				
			}
			
			/**
			 * 停止训练
			 */
			System.out.println("training finish. ["+this.trainIndex+"] finalError:"+this.currentError);
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}

	}
	
	public void trainObjectRecognitionOutputs(BaseData trainingData,BaseData valiData,boolean dataEnhance) {
		// TODO Auto-generated method stub

		try {
			
			CUDAModules.initCUDAFunctions();
			
			OutputsNetwork network = (OutputsNetwork) this.network;
			
			this.dataSize = trainingData.number;

			if(isWarmUp()) {
				this.network.learnRate = (float) (this.lr * Math.pow(batchIndex * 1.0f/burnIn * 1.0f, power));
			}
			
			Tensor input = new Tensor(batchSize, this.network.getChannel(), this.network.getHeight(), this.network.getWidth(), true);
			
			Tensor label = new Tensor(batchSize, 1, 1, trainingData.labelSize);
			
			Tensor vail_input = new Tensor(batchSize, this.network.getChannel(), this.network.getHeight(), this.network.getWidth(), true);
			
			for(int i = 0;i<this.trainTime;i++) {
				
				if(this.trainIndex >= this.minTrainTime) {
					break;
				}

				this.network.RUN_MODEL = RunModel.TRAIN;
				
				this.trainIndex = i + 1;
				
				int[][] indexs = MathUtils.randomInts(trainingData.number,this.batchSize);

				/**
				 * 遍历整个训练集
				 */
				for(int it = 0;it<indexs.length;it++) {
					
					long start = System.nanoTime();

					this.loss.clear();
					
					this.lossDiff.clear();
					
					trainingData.getRandomData(indexs[it], input, label); 
					
					/**
					 * 数据增强
					 */
					if(dataEnhance) {
						dataEnhanceInstance().transforms(input, label);
						YoloLabelUtils.formatToYoloV3(label, input.height, input.width);
					}

					input.hostToDevice();
					
					label.hostToDevice();
					
					/**
					 * forward
					 */
					network.forward(input);
					
					/**
					 * loss
					 */
					network.loss(label);
					
					/**
					 * loss diff
					 */
					Tensor[] lossDiffs = network.lossDiff(label);

					/**
					 * back
					 */
					network.back(lossDiffs);
					
					/**
					 * update
					 */
					this.network.update();
					
					String msg = "training["+this.trainIndex+"]{"+it+"} (lr:"+this.network.learnRate+") [costTime:"+(System.nanoTime() - start)/1e6+"ms.]";
					
					System.out.println(msg);

					this.batchIndex++;
				}
				
				/**
				 * update learning rate
				 */
				this.updateLR(this.lr_step);
				
				if(this.trainIndex % 100 == 0) {
					
					System.out.println("----------------testing start----------------");
					
					this.testObjectRecognitionOutputs(valiData, vail_input, label, this.batchSize);
					
					System.out.println("----------------testing finish---------------");
					
				}
				
			}
			
			/**
			 * 停止训练
			 */
			System.out.println("training finish. ["+this.trainIndex+"] finalError:"+this.currentError);
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}

	}
	
	public void trainObjectRecognitionOutputs(BaseDataLoader trainingData,BaseDataLoader valiData,boolean dataEnhance) {
		// TODO Auto-generated method stub

		try {
			
			CUDAModules.initCUDAFunctions();
			
			OutputsNetwork network = (OutputsNetwork) this.network;
			
			this.dataSize = trainingData.number;

			if(isWarmUp()) {
				this.network.learnRate = (float) (this.lr * Math.pow(batchIndex * 1.0f/burnIn * 1.0f, power));
			}
			
			Tensor input = new Tensor(batchSize, this.network.getChannel(), this.network.getHeight(), this.network.getWidth(), true);
			
			Tensor label = new Tensor(batchSize, 1, 1, trainingData.labelSize, true);
			
			Tensor vail_input = new Tensor(batchSize, this.network.getChannel(), this.network.getHeight(), this.network.getWidth(), true);
			
			Tensor vail_label = new Tensor(batchSize, 1, 1, valiData.labelSize, true);
			
			for(int i = 0;i<this.trainTime;i++) {
				
				if(this.trainIndex >= this.minTrainTime) {
					break;
				}

				this.network.RUN_MODEL = RunModel.TRAIN;
				
				this.trainIndex = i + 1;
				
				int[][] indexs = trainingData.shuffle();
				
				/**
				 * 遍历整个训练集
				 */
				for(int it = 0;it<indexs.length;it++) {
					
					long start = System.nanoTime();

					this.loss.clear();
					
					this.lossDiff.clear();
					
					trainingData.loadData(indexs[it], input, label);
					
					/**
					 * 数据增强
					 */
					if(dataEnhance) {
						dataEnhanceInstance().transforms(input, label);
						YoloLabelUtils.formatToYolo(label, input.height, input.width);
					}

					input.hostToDevice();
					
					label.hostToDevice();
					
					/**
					 * forward
					 */
					network.forward(input);
					
					/**
					 * loss
					 */
					network.loss(label);
					System.out.println("in--------------->");
					/**
					 * loss diff
					 */
					Tensor[] lossDiffs = network.lossDiff(label);
					
					/**
					 * back
					 */
					network.back(lossDiffs);
					
					/**
					 * update
					 */
					this.network.update();
					
					String msg = "training["+this.trainIndex+"]{"+it+"} (lr:"+this.network.learnRate+") [costTime:"+(System.nanoTime() - start)/1e6+"ms.]";
					
					System.out.println(msg);

					this.batchIndex++;
				}
				
				/**
				 * update learning rate
				 */
				this.updateLR(this.lr_step);
				
				if(this.trainIndex % 100 == 0) {
					
					System.out.println("----------------testing start----------------");
					
					this.testObjectRecognitionOutputs(valiData, vail_input, vail_label, this.batchSize);
					
					System.out.println("----------------testing finish---------------");
					
				}
				
			}
			
			/**
			 * 停止训练
			 */
			System.out.println("training finish. ["+this.trainIndex+"] finalError:"+this.currentError);
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}

	}
	
	public void trainObjectRecognitionOutputs(DetectionDataLoader trainingData,DetectionDataLoader valiData) {
		// TODO Auto-generated method stub

		try {
			
			CUDAModules.initCUDAFunctions();
			
			OutputsNetwork network = (OutputsNetwork) this.network;
			
			this.dataSize = trainingData.number;

			if(isWarmUp()) {
				this.network.learnRate = (float) (this.lr * Math.pow(batchIndex * 1.0f/burnIn * 1.0f, power));
			}
			
			Tensor input = new Tensor(batchSize, this.network.getChannel(), this.network.getHeight(), this.network.getWidth(), true);
			
			Tensor label = trainingData.initLabelTensor();
			
			Tensor vail_input = new Tensor(batchSize, this.network.getChannel(), this.network.getHeight(), this.network.getWidth(), true);
			
			for(int i = 0;i<this.trainTime;i++) {
				
				if(this.trainIndex >= this.minTrainTime) {
					break;
				}
				
				if(this.trainIndex == 2) {
					this.network.unfreeze();
				}

				this.network.RUN_MODEL = RunModel.TRAIN;
				
				this.trainIndex = i + 1;
				
				int[][] indexs = trainingData.shuffle();
				
				/**
				 * 遍历整个训练集
				 */
				for(int it = 0;it<indexs.length;it++) {
					
					long start = System.nanoTime();

					this.loss.clear();
					
					this.lossDiff.clear();
					
					trainingData.loadData(indexs[it], input, label);
					
					/**
					 * forward
					 */
					network.forward(input);
					
					/**
					 * loss
					 */
					network.loss(label);
					
					/**
					 * loss diff
					 */
					Tensor[] lossDiffs = network.lossDiff(label);

					/**
					 * back
					 */
					network.back(lossDiffs);
					
					/**
					 * update
					 */
					this.network.update();

					String msg = "training["+this.trainIndex+"]{"+it+"} (lr:"+this.network.learnRate+") [costTime:"+(System.nanoTime() - start)/1e6+"ms.]";
					
					System.out.println(msg);

					this.batchIndex++;
					
				}
				
				/**
				 * update learning rate
				 */
				this.updateLR(this.lr_step);
				
				if(this.trainIndex % 100 == 0) {
					
					System.out.println("----------------testing start----------------");
					
					this.testObjectRecognitionOutputs(valiData, vail_input, label, this.batchSize);
					
					System.out.println("----------------testing finish---------------");
					
				}
				
			}
			
			/**
			 * 停止训练
			 */
			System.out.println("training finish. ["+this.trainIndex+"] finalError:"+this.currentError);
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}

	}
	
	public void testRNN(Tensor input) {
		try {
			
			CUDAModules.initCUDAFunctions();
			
			/**
			 * forward
			 */
			Tensor output = this.network.forward(input);
			
			output.showDM();
			
			/**
			 * loss diff
			 */
			float[] ld = MatrixUtils.one(output.dataLength);
			this.lossDiff = new Tensor(output.number, output.channel, output.height, output.width, ld, true);

			/**
			 * back
			 */
			this.network.back(this.lossDiff);
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
	}
	
	public void trainRNN(RNNDataLoader trainingData) {
		// TODO Auto-generated method stub
		try {
			
			CUDAModules.initCUDAFunctions();
			
			this.dataSize = trainingData.number;
			
			if(isWarmUp()) {
				this.network.learnRate = (float) (this.lr * Math.pow(batchIndex * 1.0f/burnIn * 1.0f, power));
			}
			
			Tensor input = new Tensor(trainingData.time * batchSize, this.network.getChannel(), this.network.getHeight(), this.network.getWidth(), true);
			
			Tensor label = trainingData.initLabelTensor();
			
			for(int i = 0;i<this.trainTime;i++) {
				
				if(this.trainIndex >= this.minTrainTime) {
					break;
				}

				this.network.RUN_MODEL = RunModel.TRAIN;
				
				this.trainIndex = i + 1;
				
				int[][] indexs = trainingData.shuffle();
				
				/**
				 * 遍历整个训练集
				 */
				for(int it = 0;it<indexs.length;it++) {
					
					long start = System.nanoTime();

					this.loss.clear();

					this.lossDiff.clear();
					
					/**
					 * 读取训练数据
					 */
					trainingData.loadData(indexs[it], input, label);

//					System.out.println(output2TXT(input, trainingData));
					
					/**
					 * forward
					 */
					Tensor output = this.network.forward(input);
					
					/**
					 * loss
					 */
					this.loss = this.network.loss(output, label);
					
					/**
					 * loss diff
					 */
					this.lossDiff = network.lossDiff(output, label);
					
//					System.out.println(JsonUtils.toJson(output.syncHost()));
					
//					GradClipping.gradClipping(this.lossDiff, 1e-7f);

					/**
					 * back
					 */
					this.network.back(this.lossDiff);
					
					/**
					 * grad clipping
					 */
//					this.gradClipping(this.network);
					
					/**
					 * update
					 */
					this.network.update();
					
					JCudaDriver.cuCtxSynchronize();
					
					/**
					 * current time error
					 */
					if(this.loss.isHasGPU()) {
						this.currentError = MatrixOperation.sum(this.loss.syncHost()) / input.number;
					}else {
						this.currentError = MatrixOperation.sum(this.loss.data) / input.number;
					}

//					train_loss += this.currentError;
					
					output.syncHost();
					
					float error = this.accuracy(output, label);
					
//					if(error > 99) {
//						break;
//					}
					
					String msg = "training["+this.trainIndex+"]{"+it+"} (lr:"+this.network.learnRate+") accuracy:{"+error+"%} train_loss:" + this.currentError + " [costTime:"+(System.nanoTime() - start)/1e6+"ms.]";
					
					System.out.println(msg);

					this.batchIndex++;
				}
				
				/**
				 * update learning rate
				 */
				this.updateLR(this.lr_step);
				
			}
			
			/**
			 * 停止训练
			 */
			System.out.println("training finish. ["+this.trainIndex+"] finalError:"+this.currentError);
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}

	}
	
	public void trainSeg(BaseDataLoader trainingData) {
		// TODO Auto-generated method stub

		try {
			
			CUDAModules.initCUDAFunctions();

			this.dataSize = trainingData.number;

			if(isWarmUp()) {
				this.network.learnRate = (float) (this.lr * Math.pow(batchIndex * 1.0f/burnIn * 1.0f, power));
			}
			
			Tensor input = new Tensor(batchSize, this.network.getChannel(), this.network.getHeight(), this.network.getWidth(), true);
			
			Tensor label = new Tensor(batchSize, 1, this.network.getHeight(), this.network.getWidth(), true);
			
			for(int i = 0;i<this.trainTime;i++) {
				
				if(this.trainIndex >= this.minTrainTime) {
					break;
				}

				this.trainIndex = i + 1;
				
				int[][] indexs = trainingData.shuffle();

				this.network.RUN_MODEL = RunModel.TRAIN;
				
				float train_loss = 0.0f;
				
				/**
				 * 遍历整个训练集
				 */
				for(int it = 0;it<indexs.length;it++) {

					long start = System.nanoTime();

					if(Math.abs(this.currentError) <= this.error) {
						break;
					}
					
					trainingData.loadData(indexs[it], input, label);
					
					/**
					 * forward
					 */
					Tensor output = this.network.forward(input);
					
					/**
					 * loss
					 */
					this.loss = this.network.loss(output, label);
					
					/**
					 * loss diff
					 */
					this.lossDiff = this.network.lossDiff(output, label);
					
					/**
					 * back
					 */
					this.network.back(this.lossDiff);
					
					/**
					 * update
					 */
					this.network.update();

					JCudaDriver.cuCtxSynchronize();
					
					/**
					 * current time error
					 */
					if(this.loss.isHasGPU()) {
						this.currentError = MatrixOperation.sum(this.loss.syncHost()) / this.batchSize;
					}else {
						this.currentError = MatrixOperation.sum(this.loss.data) / this.batchSize;
					}

					train_loss += this.currentError;
					
					String msg = "training["+this.trainIndex+"]{"+it+"} (lr:"+this.network.learnRate+") train_loss:" + this.currentError + " [costTime:"+(System.nanoTime() - start)/1e6+"ms.]";
					
					System.out.println(msg);
					
					this.batchIndex++;
				}
				
				System.out.println("training["+this.trainIndex+"] train loss:{"+train_loss/indexs.length+"} ");
				
				/**
				 * update learning rate
				 */
				this.updateLR(this.lr_step);
				
			}
			
			/**
			 * 停止训练
			 */
			System.out.println("training finish. ["+this.trainIndex+"] finalError:"+this.currentError);
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}

	}
	
	public void testGaussianDiffusion(int ddim_timesteps) {
		
		try {
			
			DuffsionUNet network = (DuffsionUNet) this.network;
			
			float beta_1 = 1e-4f;
			float beta_T = 0.02f;
			int T = 1000;
			
			Tensor noiseInput = new Tensor(batchSize, this.network.getChannel(), this.network.getHeight(), this.network.getWidth(), true);
			
			Tensor t = new Tensor(batchSize, 1, 1, 1, true);
			
			RandomUtils.gaussianRandom(noiseInput);
			
			float[] betas = MatrixUtils.linspace(beta_1, beta_T, T);
			float[] alphas = MatrixOperation.subtraction(1, betas);
			float[] alphas_bar = MatrixUtils.cumprod(alphas);
			
			int step = T / ddim_timesteps;
			
			float[] ddim_timestep_seq = MatrixUtils.range(0, T, step, 1);
			
			float[] ddim_timestep_prev_seq = new float[ddim_timestep_seq.length];
			
			for(int i = 1;i<ddim_timestep_seq.length;i++) {
				ddim_timestep_prev_seq[i] = ddim_timestep_seq[i - 1];
			}
			int[] t_data = new int[batchSize];
			int[] prev_t_data = new int[batchSize];
			for(int timestep = ddim_timesteps - 1;timestep>=0;timestep--) {
				for(int i = 0;i<batchSize;i++) {
					t_data[i] = (int) ddim_timestep_seq[timestep];
					prev_t_data[i] = (int) ddim_timestep_prev_seq[timestep];
				}
				t.setData(t_data);
				
				Tensor eps = network.forward(noiseInput, t);
				
				float[] exsa1 = MatrixUtils.gather(alphas_bar, t_data);
				
				float[] exsa2 = MatrixUtils.gather(alphas_bar, prev_t_data);
				
				prev_mean_from_eps(noiseInput, eps, exsa1, exsa2, 1, timestep);
				
				noiseInput.hostToDevice();
			}
			
			/**
			 * print image
			 */
			showImgs("H:\\voc\\gan_anime\\duffsion_test\\", noiseInput);
			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public static void showImgs(String outputPath,Tensor input) {

		ImageUtils utils = new ImageUtils();
		
		for(int b = 0;b<input.number;b++) {
			float[] once = input.getByNumber(b);
			once = MatrixOperation.multiplication(MatrixOperation.add(once, 1.0f), 255.0f/2);
			utils.createRGBImage(outputPath + b + ".png", "png", ImageUtils.color2rgb2(once, input.channel, input.height, input.width, true), input.height, input.width, null, null);
		}
		
	}
	
	public void prev_mean_from_eps(Tensor xt,Tensor eps,float[] alphas_bar,float[] alphas_bar_prev,float eta,int t) {
		eps.syncHost();
		
		for(int b = 0;b<xt.number;b++) {
			float sigma_t = (float) (eta * Math.sqrt(1 - alphas_bar_prev[b]) / (1 - alphas_bar[b]) * (1 - alphas_bar[b] / alphas_bar_prev[b]));
			for(int l = 0;l<xt.getOnceSize();l++) {
				int i = b * xt.getOnceSize() + l;
				float pred_x0 = (float) ((xt.data[i] - Math.sqrt(1 - alphas_bar[b]) * eps.data[i]) / Math.sqrt(alphas_bar[b]));
				float pred_dir_xt = (float) (Math.sqrt(1 - alphas_bar_prev[b] - sigma_t * sigma_t) * eps.data[i]);
				if(t > 0) {
					xt.data[i] = (float) (Math.sqrt(alphas_bar_prev[b]) * pred_x0 + pred_dir_xt + sigma_t * RandomUtils.randomFloat());
				}else {
					xt.data[i] = (float) (Math.sqrt(alphas_bar_prev[b]) * pred_x0 + pred_dir_xt + sigma_t);
				}
			}
		}
		
	}
	
	public void trainGaussianDiffusion(DuffsionImageDataLoader trainingData) {
		// TODO Auto-generated method stub

		try {
			
			CUDAModules.initCUDAFunctions();

			DuffsionUNet network = (DuffsionUNet) this.network;
			
			this.dataSize = trainingData.number;

			if(isWarmUp()) {
				this.network.learnRate = (float) (this.lr * Math.pow(batchIndex * 1.0f/burnIn * 1.0f, power));
			}

			float beta_1 = 1e-4f;
			float beta_T = 0.02f;
			int T = 1000;
			
			Tensor input = new Tensor(batchSize, this.network.getChannel(), this.network.getHeight(), this.network.getWidth(), true);
			
			Tensor t = new Tensor(batchSize, 1, 1, 1, true);
			
			Tensor noise = new Tensor(batchSize, this.network.getChannel(), this.network.getHeight(), this.network.getWidth(), true);
			
			float[] betas = MatrixUtils.linspace(beta_1, beta_T, T);
			float[] alphas = MatrixOperation.subtraction(1, betas);
			float[] alphas_bar = MatrixUtils.cumprod(alphas);
			float[] sqrt_alphas_bar = MatrixOperation.sqrt(alphas_bar);
			float[] sqrt_one_minus_alphas_bar = MatrixOperation.sqrt(MatrixOperation.subtraction(1, alphas_bar));
			
			for(int i = 0;i<this.trainTime;i++) {
				
				if(this.trainIndex >= this.minTrainTime) {
					break;
				}

				this.trainIndex = i + 1;
				
				int[][] indexs = trainingData.shuffle();

				this.network.RUN_MODEL = RunModel.TRAIN;
				
				float train_loss = 0.0f;
				
				/**
				 * 遍历整个训练集
				 */
				for(int it = 0;it<indexs.length;it++) {

					long start = System.nanoTime();

					if(Math.abs(this.currentError) <= this.error) {
						break;
					}
					
					int[] t_data = RandomUtils.randomInt(0, T, batchSize);
//					System.out.println(JsonUtils.toJson(t_data));
					t.setData(t_data);

					float[] exsa1 = MatrixUtils.gather(sqrt_alphas_bar, t_data);
					
					float[] exsa2 = MatrixUtils.gather(sqrt_one_minus_alphas_bar, t_data);
					
					trainingData.loadData(indexs[it], exsa1, exsa2, input, noise);
					
					/**
					 * forward
					 */
					Tensor output = network.forward(input, t);
					
					/**
					 * loss
					 */
					this.loss = network.loss(output, noise);

					/**
					 * loss diff
					 */
					this.lossDiff = network.lossDiff(output, noise);

					/**
					 * back
					 */
					network.back(this.lossDiff);
//					System.out.println(JsonUtils.toJson(this.loss.syncHost()));
					/**
					 * update
					 */
					network.update();

					JCudaDriver.cuCtxSynchronize();
					
					/**
					 * current time error
					 */
					if(this.loss.isHasGPU()) {
						this.currentError = MatrixOperation.sum(this.loss.syncHost()) / this.batchSize;
//						System.out.println(JsonUtils.toJson(this.loss.syncHost()));
					}else {
						this.currentError = MatrixOperation.sum(this.loss.data) / this.batchSize;
					}

					train_loss += this.currentError;
					
					String msg = "training["+this.trainIndex+"]{"+it+"} (lr:"+this.network.learnRate+") train_loss:" + this.currentError + " [costTime:"+(System.nanoTime() - start)/1e6+"ms.]";
					
					System.out.println(msg);
					
					this.batchIndex++;
					
					if(it > 0 && it % 100 == 0) {
						testGaussianDiffusion(200);
					}
					
				}
				
				System.out.println("training["+this.trainIndex+"] train loss:{"+train_loss/indexs.length+"} ");
				
				/**
				 * update learning rate
				 */
				this.updateLR(this.lr_step);
				
			}
			
			/**
			 * 停止训练
			 */
			System.out.println("training finish. ["+this.trainIndex+"] finalError:"+this.currentError);
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}

	}
	
	public static String output2TXT(Tensor output,RNNDataLoader trainData) {
		String txt = "";
//		output.showDMByNumber(0);
		OneHotDataLoader tr = (OneHotDataLoader) trainData;
		for(int i = 0;i<output.number;i++) {
			int charIndex = pickTopN(output.getByNumber(i), 1);
			char c = tr.dictionaryData[charIndex];
			txt += c;
		}
		return txt;
	}
	
	public static int pickTopN(float[] x,int n) {

		float[] sort = Arrays.copyOf(x, x.length);
		
		Arrays.sort(sort);

		float[] topN = Arrays.copyOfRange(sort, sort.length - n, sort.length);

		float v = topN[RandomUtils.getRandomNumber(topN)];
		
		for(int i = 0;i<x.length;i++) {
			if(v == x[i]) {
				return i;
			}
		}
		
		return 0;
	}
	
	public void gradClipping(Network network) {
		
		for(Layer layer:network.layerList) {
			if(layer.diffW != null) {
//				System.out.println(layer.getLayerType()+"-diffW");
				GradClipping.gradClipping(layer.diffW, 1e-7f);
			}
			if(layer.diffB != null) {
//				System.out.println("diffB");
				GradClipping.gradClipping(layer.diffB, 1e-7f);
			}
		}
		
	}
	
	public void transforms(Tensor trainData,Tensor transData, float[] mean,float[] std){
		
		/**
		 * 随机裁剪
		 */
		DataTransforms.randomCrop(trainData, transData, 32, 32, 4);
		
		/**
		 * 随机翻转
		 */
		DataTransforms.randomHorizontalFilp(transData, transData);
		
		/**
		 * normalize
		 */
		DataTransforms.normalize(transData, transData, mean, std);

		/**
		 * cutcout
		 */
		DataTransforms.cutout(transData, transData, 16, 1);
		
		System.out.println("data transform finish.");
		
	}
	
	public void transforms2(Tensor trainData,Tensor transData, float[] mean,float[] std){
		
		/**
		 * normalize
		 */
		DataTransforms.normalize(trainData, transData, mean, std);

		System.out.println("data transform finish.");
		
	}

}
