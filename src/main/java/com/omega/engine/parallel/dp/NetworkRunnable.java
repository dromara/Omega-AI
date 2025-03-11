package com.omega.engine.parallel.dp;

import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.BrokenBarrierException;
import java.util.concurrent.CyclicBarrier;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import com.omega.common.data.Tensor;
import com.omega.common.utils.MatrixOperation;
import com.omega.engine.gpu.CUDAModules;
import com.omega.engine.nn.network.Llama3;
import com.omega.engine.nn.network.Network;
import com.omega.engine.nn.network.NetworkType;
import com.omega.engine.parallel.params.Llama3Parameters;
import com.omega.engine.parallel.params.Parameters;
import com.omega.example.transformer.dataset.parallel.params.DataLoaderParamters;
import com.omega.example.transformer.dataset.parallel.params.SFTBinParamters;
import com.omega.example.transformer.utils.tokenizers.Tokenizer;

import jcuda.Sizeof;
import jcuda.driver.CUstream;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaMemcpyKind;
import jcuda.runtime.cudaStream_t;

public class NetworkRunnable implements Runnable {
	
	private DP dp;
	
	private CyclicBarrier barriar;
	
	private boolean master = false;
	
	private int rankId;
	
	private NetworkType networkType;
	
	private Parameters parameters;
	
	private Network network;
	
	public Map<String,Tensor> caches = new HashMap<String, Tensor>();
	
	private DataLoaderParamters dlp;
	
	private Tensor loss;
	
	private Tensor lossDiff;
	
	private float currentError = 0.0f;
	
	private int trainIndex = 1;
	
	private float lr;
	
	private List<cudaStream_t> cudaStreams = new ArrayList<cudaStream_t>();
	
	private List<Tensor> cacheBoxs = new ArrayList<Tensor>();
	
	public NetworkRunnable(int rankId,CyclicBarrier barriar,NetworkType networkType,Parameters parameters,boolean master,DP dp) {
		this.dp = dp;
		this.barriar = barriar;
		this.master = master;
		this.rankId = rankId;
		this.networkType = networkType;
		this.parameters = parameters;
	}
	
	public Network createModel() {
		switch (networkType) {
		case LLAMA3:
			Llama3Parameters params = (Llama3Parameters) parameters;
			network = new Llama3(params.lossType, params.updater, params.getHeadNum(), params.getnKVHeadNum(), params.getDecoderNum(), params.getVocabSize(), params.getTime(), params.getEmbedDim(), params.isBias(), params.isDropout(), params.isFlashAttention(), rankId);
			getNetwork().learnRate = params.learnRate;
			this.lr = getNetwork().learnRate;
			break;
		default:
			break;
		}
		System.out.println("CUDA["+rankId+"] "+networkType.toString()+" model instance create success.");
		return getNetwork();
	}
	

	public Tensor getCache(String key,int N, int C, int H, int W) {
		Tensor c = null;
		if(caches.containsKey(key)) {
			c = caches.get(key);
			if(c.gpuLength < N * C * H * W) {
				c = Tensor.createGPUTensor(c, N, C, H, W, true);
			}else {
				c = c.viewOrg(N, C, H, W);
			}
		}else {
			c = Tensor.createGPUTensor(c, N, C, H, W, true);
			caches.put(key, c);
		}
		return c;
	}
	
	public void initStream() {
		
		for(int i = 0;i<getNetwork().paramters.size();i++) {
			cudaStream_t stream = new cudaStream_t();
			CUDAModules.checkCUDA(JCuda.cudaStreamCreate(stream));
			getCudaStreams().add(stream);
			Tensor src = getNetwork().paramters.get(i);
			Tensor rec_box = getCache("["+rankId+"]reduce_cache_"+i, src.number, src.channel, src.height, src.width);
			getCacheBoxs().add(rec_box);
		}
		
	}
	
	public cudaStream_t getStream(int paramterIndex) {
		return getCudaStreams().get(paramterIndex);
	}
	
	public Tensor getCacheBox(int idx) {
		return getCacheBoxs().get(idx);
	}
	
	@Override
	public void run() {
		// TODO Auto-generated method stub
		
		try {

			/**
			 * create model
			 */
			this.createModel();
			
			this.getNetwork().init();
			
			/**
			 * init paramters
			 */
			getNetwork().putParamters();
			initStream();
			System.out.println("CUDA["+rankId+"] init paramters finish.");
			/**
			 * init dataloader
			 */
			dlp = dp.getPd().getDataloaders().get(rankId).createParamters(getNetwork());
			dp.getPd().getDataloaders().get(rankId).loadData(dlp);
			System.out.println("CUDA["+rankId+"] init dataloader finish.");

			/**
			 * init loss
			 */
			this.loss = new Tensor(dp.getPd().getDataloaders().get(rankId).getBatchSize(), this.getNetwork().oChannel, this.getNetwork().oHeight, this.getNetwork().oWidth);
			this.lossDiff = new Tensor(dp.getPd().getDataloaders().get(rankId).getBatchSize(), this.getNetwork().oChannel, this.getNetwork().oHeight, this.getNetwork().oWidth);
			
			/**
			 * check device connect
			 */
			if(master) {
				for(int key:dp.getThreads().keySet()) {
					if(key != rankId) {
						int[] device = new int[1];
						CUDAModules.checkCUDA(JCuda.cudaGetDevice(device));
						System.out.println("current cuda["+device[0]+"]");
						int[] peer_access_available = new int[1];
						int c = JCuda.cudaDeviceCanAccessPeer(peer_access_available, rankId, key);
						CUDAModules.checkCUDA(c);
						System.out.println("cuda["+rankId+"]->cuda["+key+"]:"+peer_access_available[0]);
						if(peer_access_available[0] == 1) {
							int code = JCuda.cudaDeviceEnablePeerAccess(key, 0);
							System.out.println(rankId+"->"+key+":"+code);
							CUDAModules.checkCUDA(code);
						}
					}
				}
				System.out.println("check device connect finish.");
			}
			
			await();
			
			/**
			 * broadcast master pararmters
			 */
			if(master) {
				allReduce_broadcast();
				System.out.println("broadcast master pararmters finish.");
			}
			
			await();
			
			/**
			 * start train model
			 */
			for(int i = 0;i<dp.getTrainTime();i++) {

				/**
				 * load data->forward->loss->lossdiff->backward
				 */
				step();
				
			}
			
			await();
			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public void step() throws InterruptedException, BrokenBarrierException {
		
		switch (networkType) {
		case LLAMA3:
			llama3_step((Llama3)getNetwork(),(SFTBinParamters)dlp);
			break;
		default:
			break;
		}

		trainIndex++;
		
		await();
	}
	
	public void llama3_step(Llama3 network,SFTBinParamters paramters) throws InterruptedException, BrokenBarrierException {

		Tensor output = null;
		
		int pad = dp.getPd().getDataloaders().get(rankId).tokenizer.pad();
		
//		int pad = -1;
		
		Tensor input = paramters.getInput();
		Tensor label = paramters.getLabel();
		Tensor cos = paramters.getCos();
		Tensor sin = paramters.getSin();

		/**
		 * 遍历整个训练集
		 */
		for(int it = 0;it<dp.getPd().getCount_it();it++) {
			
			long start = System.nanoTime();
			
			this.loss.clear();

			this.lossDiff.clear();
			
			/**
			 * load data
			 */
			dp.getPd().getDataloaders().get(rankId).loadData(paramters);
			
			/**
			 * forward
			 */
			output = network.forward(cos, sin, input);

			/**
			 * loss
			 */
			this.loss = network.loss(output, label, pad);

			/**
			 * loss diff
			 */
			if(pad >= 0) {
				this.lossDiff = network.lossDiff(output, label, pad, paramters.getPadCount()[0]);
			}else {
				this.lossDiff = network.lossDiff(output, label, pad);
			}

			/**
			 * backward
			 */
			network.back(cos, sin, this.lossDiff);
			
			if(trainIndex == 1 && it == 0) {
				JCuda.cudaDeviceSynchronize();
				network.putParamterGrads();
			}
			
			/**
			 * sum loss
			 */
			int count = input.number;
			if(pad >= 0) {
				count = paramters.getPadCount()[0];
			}
			this.currentError = MatrixOperation.sum(this.loss.syncHost()) / count;
//			System.out.println(rankId+":"+currentError);
			await();
			
			/**
			 * update params
			 */
			if(master) {
				long start1 = System.nanoTime();
				/**
				 * collect diff
				 */
				allReduce_sum();
//				JCuda.cudaDeviceSynchronize();
				/**
				 * master update paramters
				 */
				this.getNetwork().update();
				JCuda.cudaDeviceSynchronize();
				
				/**
				 * broadcast master pararmters
				 */
				allReduce_broadcast();
//				JCuda.cudaDeviceSynchronize();
				System.out.println("update params cost:"+(System.nanoTime() - start1)/1e6+"ms");
			}
			
//			/**
//			 * ring all-reduce update pararmters
//			 */
//			long start1 = System.nanoTime();
//			
//			ringAllReduce();
////			await();
//			/**
//			 * update paramters
//			 */
//			this.network.update();
//			
//			System.out.println("update params cost:"+(System.nanoTime() - start1)/1e6+"ms");

			/**
			 * collect and compute loss
			 */
			if(master) {
				
				if(it % 100 == 0) {
					Tokenizer tokenizer = dp.getPd().getDataloaders().get(rankId).tokenizer;
					int batchSize = dp.getPd().getDataloaders().get(rankId).getBatchSize();
					int time = output.number / dp.getPd().getDataloaders().get(rankId).getBatchSize();
					this.accuracyBatchFisrt(input, output, label, time, batchSize, tokenizer, pad);
				}
				
				allReduce_sum_loss(start, it);
			}
			
			/**
			 * dynamic update learnRate
			 */
			updateLRDynamic((trainIndex - 1) * dp.getPd().getCount_it() + it, dp.getTrainTime() * dp.getPd().getCount_it());
			
			await();
			
		}
		
	}
	
	public void await() throws InterruptedException, BrokenBarrierException {
		JCuda.cudaDeviceSynchronize();
		barriar.await();
	}
	
	public void updateLRDynamic(int it,int count) {
		int warmup_iters = 0;
		int lr_decay_iters = count;
//		System.out.println(this.lr);
//		System.out.println(lr_decay_iters);
	    double min_lr = this.lr / 10.0d;
		
	    if (it < warmup_iters){
	    	getNetwork().learnRate = this.lr * it / warmup_iters;
	        return;
	    }
	    if(it > lr_decay_iters) {
	    	getNetwork().learnRate = (float) min_lr;
	    	return;
	    }
	    BigDecimal decay_ratio = new BigDecimal(0);
	    
	    if(it > 0) {
	    	decay_ratio = new BigDecimal(it - warmup_iters).divide(new BigDecimal(lr_decay_iters - warmup_iters), 24, BigDecimal.ROUND_HALF_DOWN);
	    }
//	    System.out.println(decay_ratio.doubleValue());
	    
	    BigDecimal coeff = new BigDecimal(0.5d).multiply(new BigDecimal(1).add(new BigDecimal(Math.cos(new BigDecimal(Math.PI).multiply(decay_ratio).doubleValue()))));
	    
	    BigDecimal tlr = new BigDecimal(min_lr).add(coeff.multiply(new BigDecimal((this.lr - min_lr))));
	    tlr = tlr.setScale(24, BigDecimal.ROUND_HALF_DOWN);

	    getNetwork().learnRate = (float)tlr.doubleValue();
	}
	
	public void allReduce_sum_loss(long start,int it) {
		float finalLoss = 0.0f;
		
		for(int key:dp.getThreads().keySet()) {
			NetworkRunnable rank = dp.getThreads().get(key);
			finalLoss+=rank.getCurrentError();
		}
		
		finalLoss/=dp.getThreads().size();
		String msg = "training["+trainIndex+"]{"+it+"/"+dp.getPd().getCount_it()+"} (lr:"+this.getNetwork().learnRate+") train_loss:" + finalLoss + " [costTime:"+(System.nanoTime() - start)/1e6+"ms.]";
		System.out.println(msg);
	}
	
	public void allReduce_sum() {
		JCuda.cudaDeviceSynchronize();
		for(int i = 0;i<getNetwork().deltaParamters.size();i++) {
			Tensor src = getNetwork().deltaParamters.get(i);
//			System.out.println(i+":"+src);
			Tensor cache = getCache("reduce_cache", src.number, src.channel, src.height, src.width);
			for(int key:dp.getThreads().keySet()) {
				if(key != rankId) {
					NetworkRunnable rank = dp.getThreads().get(key);
					Tensor tag = rank.getNetwork().deltaParamters.get(i);
//					CUDAModules.checkCUDA(JCuda.cudaMemcpy(cache.getGpuData(), tag.getGpuData(), tag.dataLength * (long)Sizeof.FLOAT, cudaMemcpyKind.cudaMemcpyDeviceToDevice));
					CUDAModules.checkCUDA(JCuda.cudaMemcpyPeer(cache.getGpuData(), rankId, tag.getGpuData(), key, tag.dataLength * (long)Sizeof.FLOAT));
					JCuda.cudaDeviceSynchronize();
					getNetwork().tensorOP.add(src, cache, src);
					JCuda.cudaDeviceSynchronize();
				}
			}
			getNetwork().tensorOP.div(src, dp.getThreads().size(), src);
//			src.showDMByOffsetRed(9, 1, i+"");
		}
		JCuda.cudaDeviceSynchronize();
	}
	
	public void allReduce_sum_asyn() {

		ExecutorService executorService = Executors.newFixedThreadPool(getNetwork().deltaParamters.size());
		
		for(int i = 0;i<getNetwork().deltaParamters.size();i++) {
			List<cudaStream_t> cudaStreams = this.cudaStreams;
			int idx = i;
			executorService.execute(new Runnable() {
				@Override
				public void run() {
					// TODO Auto-generated method stub
					cudaStream_t stream = cudaStreams.get(idx);
					Tensor src = getNetwork().deltaParamters.get(idx);
					Tensor cache = cacheBoxs.get(idx);
					for(int key:dp.getThreads().keySet()) {
						if(key != rankId) {
							NetworkRunnable rank = dp.getThreads().get(key);
							Tensor tag = rank.getNetwork().deltaParamters.get(idx);
							JCuda.cudaMemcpyPeerAsync(cache.getGpuData(), rankId, tag.getGpuData(), key, tag.dataLength * (long)Sizeof.FLOAT, stream);
							getNetwork().tensorOP.add(src, cache, src, new CUstream(stream));
						}
					}
					getNetwork().tensorOP.div(src, dp.getThreads().size(), src, new CUstream(stream));
					JCuda.cudaStreamSynchronize(stream);
				}
			});
		}
	
		executorService.shutdown();
	}
	
	public void ringAllReduce() {
		
		ExecutorService executorService = Executors.newFixedThreadPool(getNetwork().deltaParamters.size());
		
		for(int i = 0;i<getNetwork().deltaParamters.size();i++) {
			List<Tensor> cacheBoxs = this.cacheBoxs;
			List<cudaStream_t> cudaStreams = this.cudaStreams;
			int idx = i;
			executorService.execute(new Runnable() {
				
				@Override
				public void run() {
					// TODO Auto-generated method stub
					cudaStream_t stream = cudaStreams.get(idx);
					for(int r = 0;r<dp.getThreads().size() - 1;r++) {
						
						NetworkRunnable nextRand = dp.getRight(rankId);
						Tensor next = null;
						Tensor src = null;
						if(idx % 2 == 0) {
							next = nextRand.getCacheBox(idx);
							src = getNetwork().deltaParamters.get(idx);
						}else {
							next = nextRand.getNetwork().deltaParamters.get(idx);
							src = cacheBoxs.get(idx);
						}
						cudaStream_t nextStream = nextRand.getStream(idx);
						/**
						 * unblocking send src
						 */
						JCuda.cudaMemcpyPeerAsync(next.getGpuData(), nextRand.rankId, src.getGpuData(), rankId, src.dataLength * (long)Sizeof.FLOAT, stream);
						
						/**
						 * blocking receive rec_box
						 */
						JCuda.cudaStreamSynchronize(nextStream);
						if(idx % 2 == 0) {
							getNetwork().tensorOP.add(cacheBoxs.get(idx), src, cacheBoxs.get(idx), new CUstream(stream));
						}else {
							Tensor selfSrc = getNetwork().deltaParamters.get(idx);
							getNetwork().tensorOP.add(selfSrc, cacheBoxs.get(idx), selfSrc, new CUstream(stream));
						}
						
					}
//					JCuda.cudaStreamSynchronize(stream);
					if(dp.getThreads().size() % 2 == 0) {
						JCuda.cudaMemcpyAsync(getNetwork().deltaParamters.get(idx).getGpuData(), cacheBoxs.get(idx).getGpuData(), cacheBoxs.get(idx).dataLength * (long)Sizeof.FLOAT, cudaMemcpyKind.cudaMemcpyDeviceToDevice, stream);
//						network.tensorOP.copyGPU(cacheBoxs.get(idx), network.deltaParamters.get(idx));
					}
					JCuda.cudaStreamSynchronize(stream);
				}
				
			});
			
		}
		
		executorService.shutdown();
		
	}
	
	public void allReduce_broadcast() {
		JCuda.cudaDeviceSynchronize();
		for(int i = 0;i<getNetwork().paramters.size();i++) {
			Tensor src = getNetwork().paramters.get(i);
			for(int key:dp.getThreads().keySet()) {
				if(key != rankId) {
					NetworkRunnable rank = dp.getThreads().get(key);
					Tensor tag = rank.getNetwork().paramters.get(i);
//					CUDAModules.checkCUDA(JCuda.cudaMemcpy(tag.getGpuData(), src.getGpuData(), src.dataLength * (long)Sizeof.FLOAT, cudaMemcpyKind.cudaMemcpyDeviceToDevice));
					CUDAModules.checkCUDA(JCuda.cudaMemcpyPeer(tag.getGpuData(), key, src.getGpuData(), this.rankId, src.dataLength * (long)Sizeof.FLOAT));
				}
			}
			
		}
		JCuda.cudaDeviceSynchronize();
		
	}
	
	public void allReduce_broadcast_asyn() {
		ExecutorService executorService = Executors.newFixedThreadPool(getNetwork().deltaParamters.size());
		for(int i = 0;i<getNetwork().paramters.size();i++) {
			List<cudaStream_t> cudaStreams = this.cudaStreams;
			int idx = i;
			Tensor src = getNetwork().paramters.get(i);
			cudaStream_t stream = cudaStreams.get(idx);
			executorService.execute(new Runnable() {
				@Override
				public void run() {
					// TODO Auto-generated method stub
					for(int key:dp.getThreads().keySet()) {
						if(key != rankId) {
							NetworkRunnable rank = dp.getThreads().get(key);
							Tensor tag = rank.getNetwork().paramters.get(idx);
							CUDAModules.checkCUDA(JCuda.cudaMemcpyPeerAsync(tag.getGpuData(), key, src.getGpuData(), rankId, src.dataLength * (long)Sizeof.FLOAT, stream));
						}
					}
					JCuda.cudaStreamSynchronize(stream);
				}
			});
		}
		executorService.shutdown();
	}
	
	public void allReduce_broadcast_diff() {
		JCuda.cudaDeviceSynchronize();
		for(int i = 0;i<getNetwork().deltaParamters.size();i++) {
			Tensor src = getNetwork().deltaParamters.get(i);
			for(int key:dp.getThreads().keySet()) {
				if(key != rankId) {
					NetworkRunnable rank = dp.getThreads().get(key);
					Tensor tag = rank.getNetwork().deltaParamters.get(i);
//					CUDAModules.checkCUDA(JCuda.cudaMemcpy(tag.getGpuData(), src.getGpuData(), src.dataLength * (long)Sizeof.FLOAT, cudaMemcpyKind.cudaMemcpyDeviceToDevice));
					CUDAModules.checkCUDA(JCuda.cudaMemcpyPeer(tag.getGpuData(), key, src.getGpuData(), this.rankId, src.dataLength * (long)Sizeof.FLOAT));
				}
			}
			
		}
		JCuda.cudaDeviceSynchronize();
		
	}

	public float accuracyBatchFisrt(Tensor input,Tensor output,Tensor label,int time,int batchSize,Tokenizer tokenizer,int igonre) {
		float error = 0.0f;
		float trueCount = 0;
		int max_score = -9999;
		int max_index = 0;
		int[] itxt = new int[time];
		int[] ptxt = new int[time];
		int[] ltxt = new int[time];
		label.syncHost();
		input.syncHost();
		output.syncHost();
		for(int n = 0;n<batchSize;n++) {
			
			boolean allRight = true;
			int score = time;
			for(int t = 0;t<time;t++) {
				int predictIndex = MatrixOperation.maxIndex(output.getByNumber(n * time + t));
//					int labelIndex = MatrixOperation.maxIndex(labelData.getByNumber(n * time + t));
				int labelIndex = (int) label.data[n * time + t];
				if(labelIndex != igonre && labelIndex != predictIndex) {
					allRight = false;
					score--;
				}
			}
			
			if(max_score <= score) {
				max_score = score;
				max_index = n;
			}

			if(allRight) {
				trueCount++;
			}

		}
		
		for(int t = 0;t<time;t++) {
			int predictIndex = MatrixOperation.maxIndex(output.getByNumber(max_index * time + t));
//			int labelIndex = MatrixOperation.maxIndex(labelData.getByNumber(n * time + t));
			int labelIndex = (int) label.data[max_index * time + t];
			int inputIndex = (int) input.data[max_index * time + t];
			itxt[t] = inputIndex;
			ptxt[t] = predictIndex;
			ltxt[t] = labelIndex;
		}
		System.out.println("max_score:"+max_score);
		System.out.println("itxt:"+tokenizer.decode(itxt));
		System.out.println("ptxt:"+tokenizer.decode(ptxt));
		System.out.println("ltxt:"+tokenizer.decode(ltxt));

		error = trueCount / batchSize * 100;

		return error;
	}
	
	public float getCurrentError() {
		return currentError;
	}

	public List<cudaStream_t> getCudaStreams() {
		return cudaStreams;
	}

	public void setCudaStreams(List<cudaStream_t> cudaStreams) {
		this.cudaStreams = cudaStreams;
	}

	public List<Tensor> getCacheBoxs() {
		return cacheBoxs;
	}

	public void setCacheBoxs(List<Tensor> cacheBoxs) {
		this.cacheBoxs = cacheBoxs;
	}

	public Network getNetwork() {
		return network;
	}

}
