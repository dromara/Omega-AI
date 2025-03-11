package com.omega.engine.nn.layer.vqvae.tiny;

import static jcuda.jcublas.cublasOperation.CUBLAS_OP_N;
import static jcuda.jcublas.cublasOperation.CUBLAS_OP_T;

import java.io.IOException;
import java.io.RandomAccessFile;

import com.omega.common.data.Tensor;
import com.omega.common.utils.MatrixUtils;
import com.omega.common.utils.RandomUtils;
import com.omega.engine.gpu.CUDAMemoryManager;
import com.omega.engine.gpu.cudnn.SoftmaxCudnnKernel;
import com.omega.engine.nn.layer.FullyLayer;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.gpu.AttentionKernel;
import com.omega.engine.nn.network.Network;
import com.omega.engine.nn.network.RunModel;
import com.omega.engine.nn.network.Transformer;
import com.omega.engine.updater.UpdaterFactory;

/**
 * UNetSelfAttentionLayer
 * @author Administrator
 *
 */
public class TinySelfAttentionLayer extends Layer{
	
	private int time;
	
	private int headNum = 1;
	
	private int embedDim = 0;
	
	private int dk = 0;
	
	private boolean bias = false;

	public FullyLayer qLinerLayer;
	public FullyLayer kLinerLayer;
	public FullyLayer vLinerLayer;
	
	public FullyLayer oLinerLayer;
	
	private AttentionKernel attentionKernel;
	
	private SoftmaxCudnnKernel softmaxKernel;
	
	private Tensor qt;
	private Tensor kt;
	private Tensor vt;
	
	private Tensor dqkvt;
	
	private Tensor temp;
	
	private Tensor attn;
	
	private Tensor oi;

	private Tensor dattn;

	private int batchSize = 1;
	
	public TinySelfAttentionLayer(int embedDim,int headNum,int time,boolean bias) {
		this.bias = bias;
		this.time = time;
		this.embedDim = embedDim;
		this.headNum = headNum;
		if(embedDim % headNum != 0){
			throw new RuntimeException("embedDim % headNum must be zero.");
		}
		this.dk = embedDim / headNum;
		this.bias = bias;
		this.oChannel = 1;
		this.oHeight = 1;
		this.oWidth = embedDim;
		this.initLayers();
	}
	
	public TinySelfAttentionLayer(int embedDim,int headNum,int time,boolean bias,Network network) {
		this.bias = bias;
		this.network = network;
		if(this.updater == null) {
			this.setUpdater(UpdaterFactory.create(network));
		}
		this.time = time;
		this.embedDim = embedDim;
		this.headNum = headNum;
		if(embedDim % headNum != 0){
			throw new RuntimeException("embedDim % headNum must be zero.");
		}
		this.dk = embedDim / headNum;
		this.bias = bias;
		this.oChannel = 1;
		this.oHeight = 1;
		this.oWidth = embedDim;
		this.initLayers();
	}
	
	public void initLayers() {
		
		this.qLinerLayer = new FullyLayer(embedDim, embedDim, bias, this.network);
		this.kLinerLayer = new FullyLayer(embedDim, embedDim, bias, this.network);
		this.vLinerLayer = new FullyLayer(embedDim, embedDim, bias, this.network);
//		this.qkvLinerLayer.weight = new Tensor(1, 1, embedDim, 3 * embedDim, RandomUtils.order(this.embedDim * 3 * this.embedDim, 0.1f, 0.1f), true);
		
		this.oLinerLayer = new FullyLayer(embedDim, embedDim, true, this.network);
//		this.oLinerLayer.weight = new Tensor(1, 1, embedDim, embedDim, RandomUtils.order(this.embedDim * this.embedDim, 0.1f, 0.1f), true);

		if(attentionKernel == null) {
			attentionKernel = new AttentionKernel(cuda());
		}
		
		if(softmaxKernel == null) {
			softmaxKernel = new SoftmaxCudnnKernel(time, 1, 1, cuda());
		}

	}
	
	@Override
	public void init() {
		// TODO Auto-generated method stub

	}
	
	public void init(Tensor input) {
		// TODO Auto-generated method stub
		this.number = input.number;
		this.batchSize = this.number / time;
		
		if(this.qt != null) {
			this.qt.viewOrg();
			this.kt.viewOrg();
			this.vt.viewOrg();
			this.oi.viewOrg();
		}
		
		if(network.RUN_MODEL == RunModel.EVAL) {
			// [batch_size，time，head_num，d_k]
			this.qt = CUDAMemoryManager.getCache("attn-qt", batchSize, headNum, time, dk);
			this.kt = CUDAMemoryManager.getCache("attn-kt", batchSize, headNum, time, dk);
			this.vt = CUDAMemoryManager.getCache("attn-vt", batchSize, headNum, time, dk);
			// [batch_size，n_heads，len_q，len_k]
			if(time < dk) {
				this.temp = CUDAMemoryManager.getCache("attn-temp1", batchSize, headNum, time, dk);
			}else {
				this.temp = CUDAMemoryManager.getCache("attn-temp2", batchSize, headNum, time, time);
			}
			// [batch_size，n_heads，len_q，len_k]
			this.attn = CUDAMemoryManager.getCache("attn-attn", batchSize, headNum, time, time);
			// [batch_size, len_q, n_heads * dim_v]
			this.oi = CUDAMemoryManager.getCache("attn-oi", batchSize * time, 1, 1, embedDim);
		}else {
			if(this.qt == null || this.qt.number != this.batchSize) {
				// [batch_size，time，head_num，d_k]
				this.qt = Tensor.createGPUTensor(this.qt, batchSize, headNum, time, dk, true);
				this.kt = Tensor.createGPUTensor(this.kt, batchSize, headNum, time, dk, true);
				this.vt = Tensor.createGPUTensor(this.vt, batchSize, headNum, time, dk, true);
				// [batch_size，n_heads，len_q，len_k]
				if(time < dk) {
					this.temp = Tensor.createGPUTensor(this.temp, batchSize, headNum, time, dk, true);
				}else {
					this.temp = Tensor.createGPUTensor(this.temp, batchSize, headNum, time, time, true);
				}
				// [batch_size，n_heads，len_q，len_k]
				this.attn = Tensor.createGPUTensor(this.attn, batchSize, headNum, time, time, true);
				// [batch_size, len_q, n_heads * dim_v]
				this.oi = Tensor.createGPUTensor(this.oi, batchSize * time, 1, 1, embedDim, true);
			}
		}
	}
	
	@Override
	public void initBack() {
		// TODO Auto-generated method stub
		if(this.dattn == null){
			this.dqkvt = Tensor.createGPUTensor(this.dqkvt, batchSize, headNum, time, dk, true);
			this.dattn = Tensor.createGPUTensor(this.dattn, batchSize, headNum, time, time, true);
		}
	}

	@Override
	public void initParam() {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void output() {
		// TODO Auto-generated method stub
		if(network.RUN_MODEL == RunModel.EVAL) {
			eval();
		}else{
			train();
		}
	}
	
	public void eval() {
		
		Tensor qfo = CUDAMemoryManager.getCache("VQVAEAttn_qfo_cache", batchSize * time, 1, 1, embedDim);
		Tensor kfo = CUDAMemoryManager.getCache("VQVAEAttn_kfo_cache", batchSize * time, 1, 1, embedDim);
		Tensor vfo = CUDAMemoryManager.getCache("VQVAEAttn_vfo_cache", batchSize * time, 1, 1, embedDim);
		this.qLinerLayer.forward(this.input, qfo);
		this.kLinerLayer.forward(this.input, kfo);
		this.vLinerLayer.forward(this.input, vfo);
		
		Tensor q = this.qLinerLayer.getOutput().view(batchSize, time, headNum, dk);
		Tensor k = this.kLinerLayer.getOutput().view(batchSize, time, headNum, dk);
		Tensor v = this.vLinerLayer.getOutput().view(batchSize, time, headNum, dk);
		
		Tensor_OP().permute(q, qt, new int[] {0, 2, 1, 3});
		Tensor_OP().permute(k, kt, new int[] {0, 2, 1, 3});
		Tensor_OP().permute(v, vt, new int[] {0, 2, 1, 3});
		
		scaledDotProductAttention(qt, kt, vt);

		Tensor vaccum = temp;
		attentionKernel.unpermute(vaccum, oi, batchSize, time, headNum, dk);
		
		this.getoLinerLayer().forward(oi);
		
		this.output = this.getoLinerLayer().getOutput();

	}
	
	public void train() {
		
		this.qLinerLayer.forward(this.input);
		this.kLinerLayer.forward(this.input);
		this.vLinerLayer.forward(this.input);
		
		Tensor q = this.qLinerLayer.getOutput().view(batchSize, time, headNum, dk);
		Tensor k = this.kLinerLayer.getOutput().view(batchSize, time, headNum, dk);
		Tensor v = this.vLinerLayer.getOutput().view(batchSize, time, headNum, dk);
		
		Tensor_OP().permute(q, qt, new int[] {0, 2, 1, 3});
		Tensor_OP().permute(k, kt, new int[] {0, 2, 1, 3});
		Tensor_OP().permute(v, vt, new int[] {0, 2, 1, 3});
		
		scaledDotProductAttention(qt, kt, vt);

		Tensor vaccum = temp;
		attentionKernel.unpermute(vaccum, oi, batchSize, time, headNum, dk);
		
		this.getoLinerLayer().forward(oi);
		
		this.output = this.getoLinerLayer().getOutput();
	}
	
	public void scaledDotProductAttention(Tensor query,Tensor key,Tensor value) {

		float d_k = (float) (1.0f / Math.sqrt(dk));
		
		Tensor preatt = temp;
		
		GPU_OP().bmmEX(CUBLAS_OP_T, CUBLAS_OP_N, time, time, dk, 1.0f, key.getGpuData(), dk, time * dk, query.getGpuData(), dk, time * dk, 0.0f, preatt.getGpuData(), time, time * time, batchSize * headNum);
		
		Tensor_OP().mul(preatt, d_k, preatt);
		
		softmaxKernel.softmax(preatt, attn, batchSize * headNum * time);

		Tensor tmp = attn;
		
		Tensor vaccum = temp;
		GPU_OP().bmmEX(CUBLAS_OP_N, CUBLAS_OP_N, dk, time, time, 1.0f, value.getGpuData(), dk, time * dk, tmp.getGpuData(), time, time * time, 0.0f, vaccum.getGpuData(), dk, time * dk, batchSize * headNum);
	}

//	public void scaledDotProductAttentionBackward() {
//		
//		Tensor dvaccum = temp;
//	    // backward into datt
//		GPU_OP().bmmEX(CUBLAS_OP_T, CUBLAS_OP_N, time, time, dk, 1.0f, vt.getGpuData(), dk, time * dk, dvaccum.getGpuData(), dk, time * dk, 0.0f, dattn.getGpuData(), time, time * time, batchSize * headNum);
//		
//		// backward into dv
//		GPU_OP().bmmEX(CUBLAS_OP_N, CUBLAS_OP_T, dk, time, time, 1.0f, dvaccum.getGpuData(), dk, time * dk, attn.getGpuData(), time, time * time, 0.0f, dqkvt.getGpuData(), dk, time * dk, batchSize * headNum);
//
//		// backward into preatt
//		softmaxKernel.softmax_backward(attn, dattn, dattn);
//		float d_k = (float) (1.0f / Math.sqrt(dk));
//		Tensor_OP().mul(dattn, d_k, dattn);
//		Tensor dpreatt = dattn;
//
//		// backward into q
//		GPU_OP().bmmEX(CUBLAS_OP_N, CUBLAS_OP_N, dk, time, time, 1.0f, kt.getGpuData(), dk, time * dk, dpreatt.getGpuData(), time, time * time, 0.0f, dqt.getGpuData(), dk, time * dk, batchSize * headNum);
//		
//		// backward into k
//		GPU_OP().bmmEX(CUBLAS_OP_N, CUBLAS_OP_T, dk, time, time, 1.0f, qt.getGpuData(), dk, time * dk, dpreatt.getGpuData(), time, time * time, 0.0f, dkt.getGpuData(), dk, time * dk, batchSize * headNum);
//	}

	@Override
	public Tensor getOutput() {
		// TODO Auto-generated method stub
		return output;
	}
	
	@Override
	public void diff() {
		// TODO Auto-generated method stub
		
		this.getoLinerLayer().back(delta, oi);
		
		attentionKernel.unpermute_backward(temp, oi, batchSize, time, headNum, dk);
		
		Tensor dvaccum = temp;
	    // backward into datt
		GPU_OP().bmmEX(CUBLAS_OP_T, CUBLAS_OP_N, time, time, dk, 1.0f, vt.getGpuData(), dk, time * dk, dvaccum.getGpuData(), dk, time * dk, 0.0f, dattn.getGpuData(), time, time * time, batchSize * headNum);
		
		// backward into preatt
		softmaxKernel.softmax_backward(attn, dattn, dattn);
		float d_k = (float) (1.0f / Math.sqrt(dk));
		Tensor_OP().mul(dattn, d_k, dattn);
		Tensor dpreatt = dattn;
		
		// backward into dv
		GPU_OP().bmmEX(CUBLAS_OP_N, CUBLAS_OP_T, dk, time, time, 1.0f, dvaccum.getGpuData(), dk, time * dk, attn.getGpuData(), time, time * time, 0.0f, dqkvt.getGpuData(), dk, time * dk, batchSize * headNum);
		vt.view(batchSize, time, headNum, dk);
		Tensor_OP().permute(dqkvt, vt, new int[] {0, 2, 1, 3});
		Tensor vDelta = vt.view(batchSize * time, 1, 1, headNum * dk);
		this.vLinerLayer.back(vDelta);
		
		// backward into q
		GPU_OP().bmmEX(CUBLAS_OP_N, CUBLAS_OP_N, dk, time, time, 1.0f, kt.getGpuData(), dk, time * dk, dpreatt.getGpuData(), time, time * time, 0.0f, dqkvt.getGpuData(), dk, time * dk, batchSize * headNum);
		qt.view(batchSize, time, headNum, dk);
		Tensor_OP().permute(dqkvt, qt, new int[] {0, 2, 1, 3});
		Tensor qDelta = qt.view(batchSize * time, 1, 1, headNum * dk);
		this.qLinerLayer.back(qDelta);
		
		// backward into k
		GPU_OP().bmmEX(CUBLAS_OP_N, CUBLAS_OP_T, dk, time, time, 1.0f, qt.getGpuData(), dk, time * dk, dpreatt.getGpuData(), time, time * time, 0.0f, dqkvt.getGpuData(), dk, time * dk, batchSize * headNum);
		kt.view(batchSize, time, headNum, dk);
		Tensor_OP().permute(dqkvt, kt, new int[] {0, 2, 1, 3});
		Tensor kDelta = kt.view(batchSize * time, 1, 1, headNum * dk);
		this.kLinerLayer.back(kDelta);
		
		Tensor_OP().add(this.qLinerLayer.diff, this.kLinerLayer.diff, this.qLinerLayer.diff);
		Tensor_OP().add(this.qLinerLayer.diff, this.vLinerLayer.diff, this.qLinerLayer.diff);
		
		this.diff = this.qLinerLayer.diff;
		
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
		this.setInput(input);
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
		qLinerLayer.update();
		kLinerLayer.update();
		vLinerLayer.update();
		getoLinerLayer().update();
	}

	@Override
	public void showDiff() {
		// TODO Auto-generated method stub
		
	}

	@Override
	public LayerType getLayerType() {
		// TODO Auto-generated method stub
		return LayerType.mutli_head_attention;
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

	public static void main(String[] args) {
		
		int embedDim = 64;
		int headNum = 8;
		int batchSize = 2;
		int time = 512;
		
		Transformer tf = new Transformer();
		tf.number = batchSize * time;
		tf.time = time;
		
		float[] data = RandomUtils.order(batchSize * time * embedDim, 0.1f, 0.1f);

		Tensor input = new Tensor(batchSize * time, 1, 1, embedDim, data, true);
		
		float[] delta_data = MatrixUtils.val(batchSize * time * embedDim, 1.0f);
		
		Tensor delta = new Tensor(batchSize * time, 1, 1, embedDim, delta_data, true);
		
		TinySelfAttentionLayer mal = new TinySelfAttentionLayer(embedDim, headNum, time, false, tf);
		
		for(int i = 0;i<10;i++) {
//			input.showDM();
			mal.forward(input);
			
			mal.getOutput().showShape();
			
			mal.getOutput().showDM();
			
			mal.back(delta);
//			delta.showDM();
			mal.diff.showDM();
//			delta.copyData(tmp);
		}
		
	}
	
	public static boolean same(Tensor a,Tensor b) {
		float[] ad = a.syncHost();
		float[] bd = b.syncHost();
		for(int i=0;i<ad.length;i++) {
			if(ad[i] != bd[i]) {
				System.out.println(ad[i]+":"+bd[i] + "["+i+"]");
				return false;
			}
		}
		return true;
	}
	
	public void saveModel(RandomAccessFile outputStream) throws IOException {
		qLinerLayer.saveModel(outputStream);
		kLinerLayer.saveModel(outputStream);
		vLinerLayer.saveModel(outputStream);
		getoLinerLayer().saveModel(outputStream);
	}
	
	public void loadModel(RandomAccessFile inputStream) throws IOException {
		qLinerLayer.loadModel(inputStream);
		kLinerLayer.loadModel(inputStream);
		vLinerLayer.loadModel(inputStream);
		getoLinerLayer().loadModel(inputStream);
	}

	public FullyLayer getoLinerLayer() {
		return oLinerLayer;
	}

	public void setoLinerLayer(FullyLayer oLinerLayer) {
		this.oLinerLayer = oLinerLayer;
	}

	@Override
	public void accGrad(float scale) {
		// TODO Auto-generated method stub
		qLinerLayer.accGrad(scale);
		kLinerLayer.accGrad(scale);
		vLinerLayer.accGrad(scale);
		oLinerLayer.accGrad(scale);
	}
	
}
