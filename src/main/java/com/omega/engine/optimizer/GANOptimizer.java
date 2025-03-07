package com.omega.engine.optimizer;

import com.omega.common.data.Tensor;
import com.omega.common.utils.ImageUtils;
import com.omega.common.utils.MathUtils;
import com.omega.common.utils.MatrixOperation;
import com.omega.common.utils.MatrixUtils;
import com.omega.engine.ad.Graph;
import com.omega.engine.gpu.CUDAModules;
import com.omega.engine.loss.BCELoss;
import com.omega.engine.nn.data.BaseData;
import com.omega.engine.nn.network.Network;
import com.omega.engine.nn.network.RunModel;
import com.omega.engine.optimizer.lr.LearnRateUpdate;
import com.omega.example.gan.utils.ImageDataLoader;

public class GANOptimizer extends Optimizer {
	
	private Network netG;
	
	private Network netD;
	
	private Tensor g_loss_diff;
	
	private int d_every = 1;
	
	private int g_every = 5;
	
	private int gif_index = 0;
	
	private float clamp_val = -100;
	
	public GANOptimizer(Network netG, Network netD, int batchSize, int trainTime, int d_every, int g_every, float error,LearnRateUpdate learnRateUpdate, boolean warmUp) throws Exception {
		super(netD, batchSize, trainTime, error, warmUp);
		this.netG = netG;
		this.netD = netD;
		this.d_every = d_every;
		this.g_every = g_every;
		this.batchSize = batchSize;
		this.trainTime = trainTime;
		this.learnRateUpdate = learnRateUpdate;
		netG.init();
	}
	
	public GANOptimizer(Network netG, Network netD, int batchSize, int trainTime, float error,LearnRateUpdate learnRateUpdate, boolean warmUp) throws Exception {
		super(netD, batchSize, trainTime, error, warmUp);
		this.netG = netG;
		this.netD = netD;
		this.batchSize = batchSize;
		this.trainTime = trainTime;
		this.learnRateUpdate = learnRateUpdate;
		netG.init();
	}

	public GANOptimizer(Network network, int batchSize, int trainTime, float error, boolean warmUp) throws Exception {
		super(network, batchSize, trainTime, error, warmUp);
		// TODO Auto-generated constructor stub
		netG.init();
	}
	
	public void trainByAutoGrad(ImageDataLoader trainingData) {
		try {
			
			CUDAModules.initCUDAFunctions();

			this.dataSize = trainingData.number;

			if(isWarmUp()) {
				this.network.learnRate = (float) (this.lr * Math.pow(batchIndex * 1.0f/burnIn * 1.0f, power));
			}
			
			Graph gd1 = new Graph(this.netD.tensorOP);
			Graph gd2 = new Graph(this.netD.tensorOP);
			Graph gg = new Graph(this.netD.tensorOP);
			
			float dlr = this.netD.learnRate;
			float glr = this.netG.learnRate;
			
			Tensor inputD = new Tensor(batchSize, this.netD.getChannel(), this.netD.getHeight(), this.netD.getWidth(), true);
			Tensor inputG = new Tensor(batchSize, this.netG.getChannel(), this.netG.getHeight(), this.netG.getWidth(), true);
			
			float[] trueData = MatrixUtils.one(batchSize * 1);
			
			Tensor true_label = new Tensor(batchSize, 1, 1, 1, trueData, true);
			Tensor fake_label = new Tensor(batchSize, 1, 1, 1, true);
			
			
			Tensor d_true_output = new Tensor(batchSize, 1, 1, 1, true, gd1);
			Tensor d_fake_output = new Tensor(batchSize, 1, 1, 1, true, gd2);
			
			true_label.hostToDevice();
			fake_label.hostToDevice();
			
			float d_loss = 99f;
			
			for(int i = 0;i<this.trainTime;i++) {
				
				if(this.trainIndex >= this.minTrainTime) {
					break;
				}
				
				this.trainIndex = i + 1;
				
				int[][] indexs = MathUtils.randomInts(trainingData.number,this.batchSize);
				
				this.netG.RUN_MODEL = RunModel.TRAIN;
				this.netD.RUN_MODEL = RunModel.TRAIN;
				
				/**
				 * 遍历整个训练集
				 */
				for(int it = 0;it<indexs.length;it++) {
					
					if(Math.abs(this.currentError) <= this.error) {
						break;
					}
					
					long start = System.nanoTime();
					
					d_loss = trainDG(trainingData, i, indexs[it], it, true_label, fake_label, inputG, inputD, gd1, gd2, gg, d_fake_output, d_true_output, d_loss);

					String msg = "training["+this.trainIndex+"]{"+it+"} (glr:"+this.netG.learnRate+" dlr:"+this.netD.learnRate+") [costTime:"+(System.nanoTime() - start)/1e6+"ms.]";
					
					System.out.println(msg);
					
					this.batchIndex++;
				}
				
				/**
				 * update learning rate
				 */
				this.netG.learnRate = this.updateLR(this.lr_step,this.netG.learnRate,glr);
				this.netD.learnRate = this.updateLR(this.lr_step,this.netD.learnRate,dlr);
				
				if(this.trainIndex % 1 == 0) {
					this.netG.RUN_MODEL = RunModel.TEST;
					inputG.random();
					Tensor output = this.netG.forward(inputG);
					output.syncHost();
//					showImgs("H:\\voc\\gan_anime\\test\\", output);
					showBigImg("H:\\voc\\gan_anime\\test\\", output, this.trainIndex);
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
	
	public void train(ImageDataLoader trainingData) {
		try {
			
			CUDAModules.initCUDAFunctions();

			this.dataSize = trainingData.number;

			if(isWarmUp()) {
				this.network.learnRate = (float) (this.lr * Math.pow(batchIndex * 1.0f/burnIn * 1.0f, power));
			}
			
			float dlr = this.netD.learnRate;
			float glr = this.netG.learnRate;
			
			Tensor inputD = new Tensor(batchSize, this.netD.getChannel(), this.netD.getHeight(), this.netD.getWidth(), true);
			Tensor inputG = new Tensor(batchSize, this.netG.getChannel(), this.netG.getHeight(), this.netG.getWidth(), true);
			
			float[] trueData = MatrixUtils.one(batchSize * 1);
			
			Tensor true_label = new Tensor(batchSize, 1, 1, 1, trueData, true);
			Tensor fake_label = new Tensor(batchSize, 1, 1, 1, true);
			
			
			Tensor d_true_output = new Tensor(batchSize, 1, 1, 1, true);
			Tensor d_fake_output = new Tensor(batchSize, 1, 1, 1, true);
			
			true_label.hostToDevice();
			fake_label.hostToDevice();
			

			BCELoss lossD = new BCELoss(network);
			BCELoss lossG = new BCELoss(network);

			float d_loss = 99f;
			
			for(int i = 0;i<this.trainTime;i++) {
				
				if(this.trainIndex >= this.minTrainTime) {
					break;
				}
				
				this.trainIndex = i + 1;
				
				int[][] indexs = MathUtils.randomInts(trainingData.number,this.batchSize);
				
				this.netG.RUN_MODEL = RunModel.TRAIN;
				this.netD.RUN_MODEL = RunModel.TRAIN;
				
				/**
				 * 遍历整个训练集
				 */
				for(int it = 0;it<indexs.length;it++) {
					
					if(Math.abs(this.currentError) <= this.error) {
						break;
					}
					
					long start = System.nanoTime();
					
					d_loss = trainDG(trainingData, i, indexs[it], it, true_label, fake_label, inputG, inputD, d_fake_output, d_true_output, lossD, lossG, d_loss);

					String msg = "training["+this.trainIndex+"]{"+it+"} (glr:"+this.netG.learnRate+" dlr:"+this.netD.learnRate+") [costTime:"+(System.nanoTime() - start)/1e6+"ms.]";
					
					System.out.println(msg);
					
					this.batchIndex++;
				}
				
				/**
				 * update learning rate
				 */
				this.netG.learnRate = this.updateLR(this.lr_step,this.netG.learnRate,glr);
				this.netD.learnRate = this.updateLR(this.lr_step,this.netD.learnRate,dlr);
				
				if(this.trainIndex % 1 == 0) {
					this.netG.RUN_MODEL = RunModel.TEST;
					inputG.random();
					Tensor output = this.netG.forward(inputG);
					output.syncHost();
//					showImgs("H:\\voc\\gan_anime\\test\\", output);
					showBigImg("H:\\voc\\gan_anime\\test\\", output, this.trainIndex);
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
	
	@Override
	public void train(BaseData trainingData) {
		// TODO Auto-generated method stub
		try {
			
			CUDAModules.initCUDAFunctions();

			this.dataSize = trainingData.number;

			if(isWarmUp()) {
				this.network.learnRate = (float) (this.lr * Math.pow(batchIndex * 1.0f/burnIn * 1.0f, power));
			}

			Graph gd1 = new Graph(this.netD.tensorOP);
			Graph gd2 = new Graph(this.netD.tensorOP);
			Graph gg = new Graph(this.netD.tensorOP);
			
			Tensor inputD = new Tensor(batchSize, this.netD.getChannel(), this.netD.getHeight(), this.netD.getWidth(), true);

			Tensor inputG = new Tensor(batchSize, this.netG.getChannel(), this.netG.getHeight(), this.netG.getWidth(), true);
			
			float[] trueData = MatrixUtils.one(batchSize * 1);
			
			Tensor true_label = new Tensor(batchSize, 1, 1, 1, trueData, true);
			
			Tensor fake_label = new Tensor(batchSize, 1, 1, 1, true);
			
			Tensor d_true_output = new Tensor(batchSize, 1, 1, 1, true, gd1);
			Tensor d_fake_output = new Tensor(batchSize, 1, 1, 1, true, gd2);
			
			true_label.hostToDevice();
			fake_label.hostToDevice();
			
//			int[][][] rgbs = new int[300][64*28][64*28];
			
			for(int i = 0;i<this.trainTime;i++) {
				
				if(this.trainIndex >= this.minTrainTime) {
					break;
				}
				
				this.trainIndex = i + 1;
				
				int[][] indexs = MathUtils.randomInts(trainingData.number,this.batchSize);
				
				this.netD.RUN_MODEL = RunModel.TRAIN;
				this.netG.RUN_MODEL = RunModel.TRAIN;
				
				/**
				 * 遍历整个训练集
				 */
				for(int it = 0;it<indexs.length;it++) {
					
					if(Math.abs(this.currentError) <= this.error) {
						break;
					}
					
					long start = System.nanoTime();
					
					trainDG(trainingData, i, indexs, it, true_label, fake_label, inputG, inputD, gd1, gd2, gg, d_fake_output, d_true_output);
					
					String msg = "training["+this.trainIndex+"]{"+it+"} (lr:"+this.network.learnRate+") [costTime:"+(System.nanoTime() - start)/1e6+"ms.]";
					
					System.out.println(msg);
					
					this.batchIndex++;
				}
				
				/**
				 * update learning rate
				 */
				this.updateLR(this.lr_step);
				this.netG.learnRate = this.network.learnRate;
				
				if(this.trainIndex % 1 == 0) {
					this.netG.RUN_MODEL = RunModel.TEST;
					inputG.random();
					Tensor output = this.netG.forward(inputG);
					output.syncHost();
//					showImg("H:\\voc\\gan_anime\\test\\", output);
//					showGif("H:\\voc\\gan_anime\\test\\", output, this.trainIndex);
					showBigImg("H:\\voc\\gan_anime\\test\\", output, this.trainIndex);
//					showGif("H:\\voc\\gan_anime\\test\\", output, rgbs, trainIndex, rgbs.length);
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
	
	/**
	 * - (y * log(x) + (1 - y) * log(1 - x))
	 * @param x
	 * @param y
	 * @param g
	 * @return
	 */
	public Tensor BCELossFunction(Tensor x,Tensor y,Graph g) {
		x.setRequiresGrad(true);
		x.setG(g);
		y.setG(g);
//		Tensor a = sigmoid(x);
//		a.showDM();
		Tensor i = x.clamp(1e-7f, 1f - 1e-7f);
		Tensor loss1 = y.mul(i.log());
		Tensor loss2 = y.scalarSub(1.0f).mul(i.scalarSub(1.0f).log());
		Tensor loss = loss1.add(loss2);
		return loss.sum(0).div(-x.number * x.width);
	}
	
	public Tensor MSELoss(Tensor x,Tensor y,Graph g) {
		x.setRequiresGrad(true);
		x.setG(g);
		y.setG(g);
		Tensor loss = x.sub(y).pow(2);
		return loss.sum(0).div(x.number * x.width);
	}
	
	public static Tensor sigmoid(Tensor x) {
		return x.mul(-1).exp().add(1).scalarDiv(1);
	}
	
	public void trainDG2(BaseData trainingData,int index,int[][] indexs,int it,Tensor true_labels,
			Tensor fake_labels,Tensor inputG,Tensor inputD,  Graph gd1, Graph gd2, Graph gg,Tensor d_fake_output,Tensor d_true_output) {
		
		trainingData.getRandomData(indexs[it], inputD); 
		inputD.hostToDevice();

		Tensor g_fake_output = null;
		
		if(it % d_every == 0) {
			

			/**
			 * 判别器判断假图片
			 */
			/**
			 * 生成器生成假图片
			 */
			inputG.random();
			g_fake_output = this.netG.forward(inputG);
			this.netD.forward(g_fake_output).copy(d_fake_output);
			d_fake_output.showDM();
			Tensor d_fake_loss = this.netD.loss(d_fake_output, fake_labels);

			Tensor d_fake_diff = this.netD.lossDiff(d_fake_output, fake_labels);

			this.netD.back(d_fake_diff);
			this.netD.update();
			
			float dfl = d_fake_loss.syncHost()[0];
			
			/**
			 * 判别器判断真图片
			 */
			this.netD.forward(inputD).copy(d_true_output);
			d_true_output.showDM();
			Tensor d_true_loss = this.netD.loss(d_true_output, true_labels);
			
			Tensor d_true_diff = this.netD.lossDiff(d_true_output, true_labels);
			
			this.netD.back(d_true_diff);
			this.netD.update();

			float dtl = d_true_loss.syncHost()[0];
			
//			this.netD.back(d_fake_diff);
//			this.netD.update();
			
			float d_loss = (dtl + dfl) / 2;
			
			System.out.println("(d_true_loss:"+dtl+")(d_f_loss:"+dfl+")(d_fd_loss:"+d_loss+")");

		}
		
		if(it % g_every == 0) {
			/**
			 * 训练生成器
			 */

			inputG.random();
			g_fake_output = this.netG.forward(inputG);
			
			/**
			 * 判别器判断假图片
			 */
			this.netD.forward(g_fake_output).copy(d_fake_output);
			
			Tensor g_fake_loss = this.netD.loss(d_fake_output, true_labels);
			
			Tensor g_fake_diff = this.netD.lossDiff(d_fake_output, true_labels);
			
			System.out.println("g_loss:"+g_fake_loss.syncHost()[0]);
			
			this.netD.back(g_fake_diff);
			this.g_loss_diff = this.netD.getDiff();
			this.netG.back(this.g_loss_diff);
			this.netG.update();
		}	

	}
	
	public void trainDG(BaseData trainingData,int index,int[][] indexs,int it,Tensor true_labels,
			Tensor fake_labels,Tensor inputG,Tensor inputD,  Graph gd1, Graph gd2, Graph gg,Tensor d_fake_output,Tensor d_true_output) {
		
		Tensor g_fake_output = null;
		
		if(it % d_every == 0) {

			trainingData.getRandomData(indexs[it], inputD); 
			inputD.hostToDevice();

			gd1.start();
			gd2.start();	
			
			/**
			 * 判别器判断真图片
			 */
			d_true_output = this.netD.forward(inputD);
			
			Tensor real_loss = BCELossFunction(d_true_output, true_labels, gd1);
			
			gd1.clearGrad();
			gd1.backward();
			
//			BPNetwork bp = (BPNetwork) this.netD;
			
			this.netD.back(d_true_output.getGrad());
//			bp.back(d_true_output.getGrad());
			this.netD.update();
			
			/**
			 * 判别器判断假图片
			 */
			/**
			 * 生成器生成假图片
			 */
			inputG.random();
			g_fake_output = this.netG.forward(inputG);
			d_fake_output = this.netD.forward(g_fake_output);
			Tensor fake_loss = BCELossFunction(d_fake_output, fake_labels, gd2);

//			Tensor d_loss = real_loss.add(fake_loss).div(2);
			
			gd2.clearGrad();
			gd2.backward();

//			bp.backTemp(d_fake_output.getGrad());
			this.netD.back(d_fake_output.getGrad());
			this.netD.update();

			float d_loss = (real_loss.syncHost()[0] + fake_loss.syncHost()[0]) / 2;
			
			System.out.println("(d_true_loss:"+real_loss.syncHost()[0]+")(d_f_loss:"+fake_loss.syncHost()[0]+")(d_fd_loss:"+d_loss+")");

		}
		
		if(it % g_every == 0) {
			/**
			 * 训练生成器
			 */
			gg.start();
			
			inputG.random();
			g_fake_output = this.netG.forward(inputG);
			
			/**
			 * 判别器判断假图片
			 */
			d_fake_output = this.netD.forward(g_fake_output);

			Tensor g_loss = BCELossFunction(d_fake_output, true_labels, gg);

			System.out.println("g_loss:"+g_loss.syncHost()[0]);
			
			gg.clearGrad();
			gg.backward();

			this.netD.back(d_fake_output.getGrad());
			this.g_loss_diff = this.netD.getDiff();
			this.netG.back(this.g_loss_diff);
			this.netG.update();
		}	
		
	}
	
	public float trainDG(ImageDataLoader trainingData,int index,int[] indexs,int it,Tensor true_labels,
			Tensor fake_labels,Tensor inputG,Tensor inputD,  Graph gd1, Graph gd2, Graph gg,Tensor d_fake_output,Tensor d_true_output,float d_loss_current) {

		Tensor g_fake_output = null;
		
		gd1.start();
		gd2.start();
		
		if(it % d_every == 0) {
			
			trainingData.loadData(indexs, inputD, null);

			/**
			 * 生成器生成假图片
			 */
			inputG.random();
			g_fake_output = this.netG.forward(inputG);

			/**
			 * 判别器判断真图片
			 */
			d_true_output = this.netD.forward(inputD);
			Tensor real_loss = BCELossFunction(d_true_output, true_labels, gd1);
			
			gd1.clearGrad();
			gd1.backward();
			this.netD.back(d_true_output.getGrad());
			this.netD.update();
			
			/**
			 * 判别器判断假图片
			 */
			d_fake_output = this.netD.forward(g_fake_output);

			Tensor fake_loss = BCELossFunction(d_fake_output, fake_labels, gd2);
			
			gd2.clearGrad();
			gd2.backward();
			this.netD.back(d_fake_output.getGrad());
			this.netD.update();

			float d_loss = (real_loss.syncHost()[0] + fake_loss.syncHost()[0]) / 2;
			
			d_loss_current = d_loss;
			
			System.out.println("(d_true_loss:"+real_loss.syncHost()[0]+")(d_f_loss:"+fake_loss.syncHost()[0]+")(d_fd_loss:"+d_loss+")");

		}	
		
		if(it % g_every == 0) {

				/**
				 * 训练生成器
				 */
				gg.start();
				
				inputG.random();
				g_fake_output = this.netG.forward(inputG);
				
				/**
				 * 判别器判断假图片
				 */
				d_fake_output = this.netD.forward(g_fake_output);

				Tensor g_loss = BCELossFunction(d_fake_output, true_labels, gg);
				
				System.out.println("g_loss:"+g_loss.syncHost()[0]);
				
				gg.clearGrad();
				gg.backward();

				this.netD.back(d_fake_output.getGrad());
				this.g_loss_diff = this.netD.getDiff();

				this.netG.back(this.g_loss_diff);
				this.netG.update();

		}
		
		return d_loss_current;
	}
	
	public float trainDG(ImageDataLoader trainingData,int index,int[] indexs,int it,Tensor true_labels,
			Tensor fake_labels,Tensor inputG,Tensor inputD,Tensor d_fake_output,Tensor d_true_output,BCELoss lossD,BCELoss lossG,float d_loss_current) {

		/**
		 * 生成器生成假图片
		 */
		inputG.random();
		Tensor g_fake_output = this.netG.forward(inputG);

		
		if(it % d_every == 0) {
			
			trainingData.loadData(indexs, inputD, null);
			
			/**
			 * 判别器判断真图片
			 */
			d_true_output = this.netD.forward(inputD);
			Tensor real_loss = lossD.loss(d_true_output, true_labels);
			Tensor real_diff = lossD.diff(d_true_output, true_labels);
			this.netD.back(real_diff);
			this.netD.update();
			
			float dt_loss = real_loss.syncHost()[0];
			
			/**
			 * 判别器判断假图片
			 */
			d_fake_output = this.netD.forward(g_fake_output);
			Tensor fake_loss = lossD.loss(d_fake_output, fake_labels);
			Tensor fake_diff = lossD.diff(d_fake_output, fake_labels);
			this.netD.back(fake_diff);
			this.netD.update();
			
			float df_loss = fake_loss.syncHost()[0];
			
			float d_loss = (dt_loss + df_loss) / 2;
			
			d_loss_current = d_loss;
			
			System.out.println("(d_true_loss:"+dt_loss+")(d_f_loss:"+df_loss+")(d_fd_loss:"+d_loss+")");

		}	
		
		/**
		 * 训练生成器
		 */
		if(it % g_every == 0) {

			/**
			 * 判别器判断假图片
			 */
			d_fake_output = this.netD.forward(g_fake_output);

			Tensor g_loss = lossG.loss(d_fake_output, true_labels);
			Tensor g_diff = lossG.diff(d_fake_output, true_labels);
			
			System.out.println("g_loss:"+g_loss.syncHost()[0]);
			
//			g_diff.showDMByNumber(0);
			
			this.netD.back(g_diff);
			this.g_loss_diff = this.netD.getDiff();

			this.netG.back(this.g_loss_diff);
			this.netG.update();

		}
		
		return d_loss_current;
	}
	
	/**
	 * 训练判别器
	 */
	public void trainNetD(BaseData trainingData,int[][] indexs,int it,Tensor true_labels,Tensor fake_labels,Tensor inputG,Tensor inputD, Graph g,Tensor output1,Tensor output2) {

		trainingData.getRandomData(indexs[it], inputD); 
		inputD.hostToDevice();
		
		g.start();
		
		inputG.random();
		Tensor fake_output = this.netG.forward(inputG);
		this.netD.forward(fake_output).copy(output2);
		
		Tensor loss_fake = this.BCELossFunction(output2, fake_labels, g);

		/**
		 * 训练判别器学习判别真图片
		 */
		this.netD.forward(inputD).copy(output1);
		
		Tensor loss_true = this.BCELossFunction(output1, true_labels, g);
		
		Tensor loss = loss_true.add(loss_fake).div(2);
		
		g.clearGrad();
		g.backward();
		
		this.netD.back(output1.getGrad());
		this.netD.update();
//		this.netD.back(output2.getGrad());
//		this.netD.update();
		
//		/**
//		 * 训练判别器学习判别真图片
//		 */
//		this.netD.forward(inputD).copy(output1);
//		g.start();
//		output1.setG(g);
//		true_labels.setG(g);
//		
//		Tensor loss_true = this.netD.loss(output1, true_labels);
//		
//		/**
//		 * 训练判别器学习判别假图片
//		 */
//		inputG.random();
//		Tensor fake_output = this.netG.forward(inputG);
//		this.netD.forward(fake_output).copy(output2);
//		output2.setG(g);
//		fake_labels.setG(g);
//		Tensor loss_fake = this.netD.loss(output2, fake_labels);
//		Tensor loss = loss_true.add(loss_fake).div(2);
//		loss.getG().clearGrad();
//		loss.getG().backward();
////		output1.getGrad().showDM();
////		output2.getGrad().showDM();
////		Tensor diff = new Tensor(batchSize, 1, 1, 1, true);
////		TensorOP.add(output1.getGrad(), output2.getGrad(), diff);
////		diff.showDM();
////		this.netD.back(diff);
//		
//		this.netD.back(output1.getGrad());
////		this.netD.update();
//		this.netD.back(output2.getGrad());
//		this.netD.update();

		System.out.println("(d_loss:"+loss_true.syncHost()[0]+")(d_f_loss:"+loss_fake.syncHost()[0]+")(d_fd_loss:"+loss.syncHost()[0]+")");
		
	}
	
	/**
	 * 训练判别器
	 */
	public void trainNetD(ImageDataLoader trainingData,int[][] indexs,int it,Tensor true_labels,Tensor fake_labels,Tensor inputG,Tensor inputD,
			Tensor d_real_loss,Tensor d_fake_loss,Tensor d_real_diff,Tensor d_fake_diff,Tensor diff) {

		trainingData.loadData(indexs[it], inputD, null);
		inputD.hostToDevice();
		/**
		 * 训练判别器学习判别真图片
		 */
		Tensor output = this.netD.forward(inputD);
//		System.out.println("outputD:");
//		output.showDM();
		this.netD.loss(output, true_labels, d_real_loss);
//		System.out.println("d_real_loss:");
//		d_real_loss.showDM();
		this.netD.lossDiff(output, true_labels, d_real_diff);
//		this.netD.back(d_real_diff);


		/**
		 * 训练判别器学习判别假图片
		 */
		inputG.random();
		Tensor fake_output = this.netG.forward(inputG);
		Tensor outputDFake = this.netD.forward(fake_output);
//		System.out.println("outputDFake:");
//		outputDFake.showDM();
		this.netD.loss(outputDFake, fake_labels, d_fake_loss);
//		System.out.println("d_fake_loss:");
//		d_fake_loss.showDM();
		this.netD.lossDiff(outputDFake, fake_labels, d_fake_diff);
//		System.out.println("d_fake_diff:");
//		lossDiff2.showDM();
//		d_real_diff.showDM();
//		d_fake_diff.showDM();
		netD.tensorOP.add(d_real_diff, d_fake_diff, diff);
		netD.tensorOP.div(diff, 2.0f, diff);
//		diff.showDM();
//		System.out.println("d_fake_diff:"+d_fake_diff.isZero());
//		diff.showDM();
		this.netD.back(diff);
		this.netD.update();
		
		float d_loss = MatrixUtils.sum(d_real_loss.syncHost()) / batchSize;
		
		float d_f_loss = MatrixUtils.sum(d_fake_loss.syncHost()) / batchSize;

		float d_fd_loss = (d_loss +  d_f_loss) / 2;

		System.out.println("(d_loss:"+d_loss+")(d_f_loss:"+d_f_loss+")(d_fd_loss:"+d_fd_loss+")");
		
	}
	
	/**
	 * 训练生成器
	 */
	public void trainNetG(int it,Tensor true_labels,Tensor inputG,Tensor inputD, Graph g,Tensor output) {

		inputG.random();
		Tensor fake_output = this.netG.forward(inputG);
		this.netD.forward(fake_output).copy(output);
		g.start();
		
		Tensor loss = BCELossFunction(output, true_labels, g);
		
		g.clearGrad();
		g.backward();
		
		this.netD.back(output.getGrad());
		this.g_loss_diff = this.netD.getDiff();
		System.out.println("g_loss_diff-max-g:"+MatrixOperation.max(g_loss_diff.syncHost()));
		this.netG.back(g_loss_diff);
		this.netG.update();
		
		System.out.println("g_loss:"+loss.syncHost()[0]);

	}

	@Override
	public void train(BaseData trainingData, BaseData testData) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void train(BaseData trainingData, BaseData validata, BaseData testData) {
		// TODO Auto-generated method stub
		
	}
	
	public static void showBigImg(String outputPath,Tensor input,int it) {

		ImageUtils utils = new ImageUtils();
		
		int gifw = 8;
		int gifh = 8;
		int imgw = input.width;
		int imgh = input.height;
		
		if(imgh == 1 && imgw > 1) {
			imgh = (int) Math.sqrt(imgw);
			imgw = (int) Math.sqrt(imgw);
		}
		
		int gifWidth = gifw * imgw;
		int gifHeight = gifh * imgh;
		
		float[] gif = new float[imgw * imgh * 64 * input.channel];
		
		for(int c = 0;c<input.channel;c++) {

			for(int b = 0;b<gifw * gifh;b++) {
				
				int gh = b / gifw;
				int gw = b % gifh;
				
				float[] once = input.getByNumber(b);
				
				once = MatrixOperation.multiplication(MatrixOperation.add(once, 1.0f), 255.0f/2);
				
				for(int i = 0;i<imgh;i++) {
					int startH = gh * imgh + i;
					for(int j = 0;j<imgw;j++) {
						int startW = gw * imgw + j;
						gif[c * imgh * gifh * imgw * gifw + startH * imgw * gifw + startW] = once[c * imgh * imgw + i * imgw + j];
					}
				}
		
			}

		}
		
		utils.createRGBImage(outputPath + it + ".png", "png", ImageUtils.color2rgb2(gif, input.channel, gifHeight, gifWidth, false), gifWidth, gifHeight, null, null);
		
	}
	
	public static void showImgs(String outputPath,Tensor input) {

		ImageUtils utils = new ImageUtils();
		
		for(int b = 0;b<64;b++) {
			float[] once = input.getByNumber(b);
			once = MatrixOperation.multiplication(MatrixOperation.add(once, 1.0f), 255.0f/2);
			utils.createRGBImage(outputPath + b + ".png", "png", ImageUtils.color2rgb2(once, input.channel, 28, 28, false), 28, 28, null, null);
		}
		
	}
	
	public void showGif(String outputPath,Tensor input,int[][][] rgbs,int it,int count) {

		ImageUtils utils = new ImageUtils();
		
		int gifw = 8;
		int gifh = 8;
		int imgw = 28;
		int imgh = 28;
		
		int gifWidth = gifw * imgw;
		int gifHeight = gifh * imgh;
		
		float[] gif = new float[imgw * imgh * 64];
		
		for(int b = 0;b<gifw * gifh;b++) {
			
			int gh = b / gifw;
			int gw = b % gifh;
			
			float[] once = input.getByNumber(b);
			
			once = MatrixOperation.multiplication(MatrixOperation.add(once, 1.0f), 255.0f/2);
			
			for(int i = 0;i<imgh;i++) {
				int startH = gh * imgh + i;
				for(int j = 0;j<imgw;j++) {
					int startW = gw * imgw + j;
					gif[startH * imgw * gifw + startW] = once[i * imgw + j];
				}
			}
	
		}
		
		int[][] rgb = ImageUtils.color2rgb2(gif, input.channel, gifHeight, gifWidth, false);
		rgbs[gif_index] = rgb;
		if(gif_index == count - 1) {
			utils.createRGBGIF(outputPath + it + ".gif", "gif", rgbs, gifWidth, gifHeight);
			gif_index = 0;
		}else {
			gif_index++;
		}

	}


}
