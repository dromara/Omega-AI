package com.omega.engine.nn.layer.transformer;

import static jcuda.jcublas.cublasOperation.CUBLAS_OP_N;
import static jcuda.jcublas.cublasOperation.CUBLAS_OP_T;

import com.omega.common.data.Tensor;
import com.omega.common.utils.MatrixUtils;
import com.omega.common.utils.RandomUtils;
import com.omega.engine.gpu.GPUOP;
import com.omega.engine.gpu.SoftmaxKernel;
import com.omega.engine.nn.layer.DropoutLayer;
import com.omega.engine.nn.layer.FullyLayer;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.network.Network;
import com.omega.engine.nn.network.Transformer;
import com.omega.engine.updater.UpdaterFactory;
import com.omega.example.transformer.utils.ENTokenizer;

/**
 * CausalSelfAttentionLayer
 * @author Administrator
 *
 */
public class CausalSelfAttentionLayer extends Layer{
	
	private boolean dropout = false;
	
	private float dropput_probability = 0.5f;
	
	private int time;
	
	private int headNum = 1;
	
	private int embedDim = 0;
	
	private int dk = 0;
	
	private boolean bias = false;
	
	private FullyLayer qLinerLayer;
	private FullyLayer kLinerLayer;
	private FullyLayer vLinerLayer;
	
	private FullyLayer oLinerLayer;
	
	private DropoutLayer dropoutLayer;
	
	private Tensor qt;
	private Tensor kt;
	private Tensor vt;
	
	private Tensor scores;
	
	private Tensor scores_masked;
	
	private Tensor weights;
	
	private Tensor attn_outputs;
	
	private Tensor ot;
	
	private Tensor oi;
	
	private SoftmaxKernel softmax;
	
	private int batchSize = 1;
	
	public CausalSelfAttentionLayer(int embedDim,int headNum,int time,boolean bias,boolean dropout) {
		this.bias = bias;
		this.dropout = dropout;
		this.time = time;
		this.embedDim = embedDim;
		this.headNum = headNum;
		this.bias = bias;
		this.oChannel = 1;
		this.oHeight = 1;
		this.oWidth = embedDim;
		this.initLayers();
	}
	
	public CausalSelfAttentionLayer(int embedDim,int headNum,int time,boolean bias,boolean dropout,Network network) {
		this.bias = bias;
		this.dropout = dropout;
		this.network = network;
		if(this.updater == null) {
			this.setUpdater(UpdaterFactory.create(network));
		}
		this.time = time;
		this.embedDim = embedDim;
		this.headNum = headNum;
		this.bias = bias;
		this.oChannel = 1;
		this.oHeight = 1;
		this.oWidth = embedDim;
		this.initLayers();
	}
	
	public void initLayers() {
		
//		float stdv = (float) (1.0f / Math.sqrt(embedDim));
//		float stdv = (float) (2.0f / Math.sqrt(embedDim + embedDim));
		
		this.qLinerLayer = new FullyLayer(embedDim, embedDim, bias, this.network);
//		this.qLinerLayer.weight = new Tensor(1, 1, embedDim, embedDim, RandomUtils.uniform(this.embedDim * this.embedDim, 0.0f, 0.02f), true);
		this.qLinerLayer.weight = new Tensor(1, 1, embedDim, embedDim, RandomUtils.order(this.embedDim * this.embedDim, 0.1f, 0.1f), true);
		this.qLinerLayer.weight.showDM();
//		Tensor qwt = new Tensor(1, 1, embedDim, embedDim, true);
//		TensorOP.permute(this.qLinerLayer.weight, qwt, new int[] {0, 1, 3, 2});
//		this.qLinerLayer.weight = qwt;
//		this.qLinerLayer.bias = new Tensor(1, 1, 1, embedDim, RandomUtils.uniform(this.embedDim * this.embedDim, 0, stdv), true);
//		this.qLinerLayer.weight = new Tensor(1, 1, embedDim, embedDim, RandomUtils.uniform(this.embedDim * this.embedDim, 0, stdv), true);
//		this.qLinerLayer.weight = new Tensor(1, 1, embedDim, embedDim, RandomUtils.val(this.embedDim * this.embedDim, 0.1f), true);
		
		this.kLinerLayer = new FullyLayer(embedDim, embedDim, bias, this.network);
//		this.kLinerLayer.weight = new Tensor(1, 1, embedDim, embedDim, RandomUtils.uniform(this.embedDim * this.embedDim, 0.0f, 0.02f), true);
		this.kLinerLayer.weight = new Tensor(1, 1, embedDim, embedDim, RandomUtils.order(this.embedDim * this.embedDim, 0.1f, this.embedDim * this.embedDim * 0.1f + 0.1f), true);
		this.kLinerLayer.weight.showDM();
//		Tensor kwt = new Tensor(1, 1, embedDim, embedDim, true);
//		TensorOP.permute(this.kLinerLayer.weight, kwt, new int[] {0, 1, 3, 2});
//		this.kLinerLayer.weight = kwt;
//		this.kLinerLayer.weight = new Tensor(1, 1, hiddenSize, hiddenSize, RandomUtils.uniform(this.hiddenSize * this.hiddenSize, 0, stdv), true);
//		this.selfLayer.bias = new Tensor(1, 1, 1, hiddenSize, RandomUtils.uniform(this.hiddenSize, 0, stdv), true);
//		this.kLinerLayer.weight = new Tensor(1, 1, embedDim, embedDim, RandomUtils.val(this.embedDim * this.embedDim, 0.2f), true);

		this.vLinerLayer = new FullyLayer(embedDim, embedDim, bias, this.network);
//		this.vLinerLayer.weight = new Tensor(1, 1, embedDim, embedDim, RandomUtils.uniform(this.embedDim * this.embedDim, 0.0f, 0.02f), true);
		this.vLinerLayer.weight = new Tensor(1, 1, embedDim, embedDim, RandomUtils.order(this.embedDim * this.embedDim, 0.1f, this.embedDim * this.embedDim * 0.2f + 0.1f), true);
		this.vLinerLayer.weight.showDM();
//		Tensor vwt = new Tensor(1, 1, embedDim, embedDim, true);
//		TensorOP.permute(this.vLinerLayer.weight, vwt, new int[] {0, 1, 3, 2});
//		this.vLinerLayer.weight = vwt;
//		this.vLinerLayer.weight = new Tensor(1, 1, embedDim, embedDim, RandomUtils.val(this.embedDim * this.embedDim, 0.01f), true);
		
		this.oLinerLayer = new FullyLayer(embedDim, embedDim, bias, this.network);
//		this.oLinerLayer.weight = new Tensor(1, 1, embedDim, embedDim, RandomUtils.uniform(this.embedDim * this.embedDim, 0.0f, 0.02f), true);
		this.oLinerLayer.weight = new Tensor(1, 1, embedDim, embedDim, RandomUtils.order(this.embedDim * this.embedDim, 0.01f, 0.01f), true);
//		Tensor owt = new Tensor(1, 1, embedDim, embedDim, true);
//		TensorOP.permute(this.oLinerLayer.weight, owt, new int[] {0, 1, 3, 2});
//		this.oLinerLayer.weight = owt;
//		this.oLinerLayer.weight = new Tensor(1, 1, embedDim, embedDim, RandomUtils.val(this.embedDim * this.embedDim, 0.02f), true);
		
		if(dropout) {
			this.dropoutLayer = new DropoutLayer(dropput_probability, this.oLinerLayer);
		}
		
		if(softmax == null) {
			softmax = new SoftmaxKernel(cuda());
		}
		
//		System.out.println(JsonUtils.toJson(this.inputLayer.weight.syncHost()));
//		System.out.println(JsonUtils.toJson(this.inputLayer.bias.syncHost()));
//		System.out.println(JsonUtils.toJson(this.selfLayer.weight.syncHost()));
//		System.out.println(JsonUtils.toJson(this.selfLayer.bias.syncHost()));
		
	}
	
	@Override
	public void init() {
		// TODO Auto-generated method stub
		this.number = this.network.number;
		this.time = this.network.time;
		this.dk = embedDim / headNum;
		this.batchSize = number / time;

		if(this.scores == null || this.scores.number != this.batchSize || this.scores.width != this.time) {
			// [batch_size，time，head_num，d_k]
			this.qt = Tensor.createTensor(this.qt, batchSize, headNum, time, dk, true);
			this.kt = Tensor.createTensor(this.kt, batchSize, headNum, time, dk, true);
			this.vt = Tensor.createTensor(this.vt, batchSize, headNum, time, dk, true);
			// [batch_size，n_heads，len_q，len_k]
			this.scores = Tensor.createTensor(this.scores, batchSize, headNum, time, time, true);
			this.scores_masked = Tensor.createTensor(this.scores_masked, batchSize, headNum, time, time, true);
			// [batch_size，n_heads，len_q，len_k]
			this.weights = Tensor.createTensor(this.weights, batchSize, headNum, time, time, true);
			// [batch_size, n_heads, len_q, dim_v]
			this.attn_outputs = Tensor.createTensor(this.attn_outputs, batchSize, headNum, time, dk, true);
			// [batch_size, len_q, n_heads * dim_v]
			this.ot = Tensor.createTensor(this.ot, batchSize, time, headNum, dk, true);
			this.oi = Tensor.createTensor(this.oi, batchSize, time, headNum, dk, true);
		}else {
			resize();
		}

	}
	
	public void init(Tensor input) {
		// TODO Auto-generated method stub
		this.number = input.number;
		this.time = this.network.time;
		this.dk = embedDim / headNum;
		this.batchSize = number / time;
		
		if(this.scores == null || this.scores.number != this.batchSize || this.scores.width != this.time) {
			// [batch_size，time，head_num，d_k]
			this.qt = Tensor.createTensor(this.qt, batchSize, headNum, time, dk, true);
			this.kt = Tensor.createTensor(this.kt, batchSize, headNum, time, dk, true);
			this.vt = Tensor.createTensor(this.vt, batchSize, headNum, time, dk, true);
			// [batch_size，n_heads，len_q，len_k]
			this.scores = Tensor.createTensor(this.scores, batchSize, headNum, time, time, true);
//			this.scores_masked = Tensor.createTensor(this.scores_masked, batchSize, headNum, time, time, true);
			// [batch_size，n_heads，len_q，len_k]
			this.weights = Tensor.createTensor(this.weights, batchSize, headNum, time, time, true);
			// [batch_size, n_heads, len_q, dim_v]
			this.attn_outputs = Tensor.createTensor(this.attn_outputs, batchSize, headNum, time, dk, true);
			// [batch_size, len_q, n_heads * dim_v]
			this.ot = Tensor.createTensor(this.ot, batchSize, time, headNum, dk, true);
			this.oi = Tensor.createTensor(this.oi, batchSize, time, headNum, dk, true);
		}else {
			resize();
		}

	}
	
	public void resize() {
		this.qt.viewOrg();
		this.kt.viewOrg();
		this.vt.viewOrg();
		this.scores.viewOrg();
		this.weights.viewOrg();
//		this.scores_masked.viewOrg();
		this.attn_outputs.viewOrg();
		this.ot.viewOrg();
		this.oi.viewOrg();
		if(this.qLinerLayer.getOutput() != null) {
			this.qLinerLayer.getOutput().viewOrg();
			this.kLinerLayer.getOutput().viewOrg();
			this.vLinerLayer.getOutput().viewOrg();
		}
	}
	
	@Override
	public void initBack() {
		// TODO Auto-generated method stub
//		if(this.cache_delta == null || output.number != cache_delta.number){
//			this.cache_delta = new Tensor(number, output.channel, output.height, output.width, true);
//		}
	}

	@Override
	public void initParam() {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void output() {
		// TODO Auto-generated method stub
//		System.out.println("in");
		this.qLinerLayer.forward(this.input);
		this.kLinerLayer.forward(this.input);
		this.vLinerLayer.forward(this.input);

		Tensor query = this.qLinerLayer.getOutput().view(batchSize, time, headNum, dk);
		Tensor key = this.kLinerLayer.getOutput().view(batchSize, time, headNum, dk);
		Tensor value = this.vLinerLayer.getOutput().view(batchSize, time, headNum, dk);
//		query.showShape();
//		qt.showShape();
		Tensor_OP().permute(query, qt, new int[] {0, 2, 1, 3});
		Tensor_OP().permute(key, kt, new int[] {0, 2, 1, 3});
		Tensor_OP().permute(value, vt, new int[] {0, 2, 1, 3});

		scaledDotProductAttention(qt, kt, vt, null);

		Tensor_OP().permute(attn_outputs, ot, new int[] {0, 2, 1, 3});

		ot.view(batchSize * time, 1, 1, headNum * dk);

		this.oLinerLayer.forward(ot);
		
		if(dropout) {
			this.dropoutLayer.forward(this.oLinerLayer.getOutput());
			this.output = this.oLinerLayer.getOutput();
		}else {
			this.output = this.oLinerLayer.getOutput();
		}

	}
	
	public void output(Tensor mask) {
		// TODO Auto-generated method stub

		this.qLinerLayer.forward(this.input);
		this.kLinerLayer.forward(this.input);
		this.vLinerLayer.forward(this.input);
		
		Tensor query = this.qLinerLayer.getOutput().view(batchSize, time, headNum, dk);
		Tensor key = this.kLinerLayer.getOutput().view(batchSize, time, headNum, dk);
		Tensor value = this.vLinerLayer.getOutput().view(batchSize, time, headNum, dk);
		
		Tensor_OP().permute(query, qt, new int[] {0, 2, 1, 3});
		Tensor_OP().permute(key, kt, new int[] {0, 2, 1, 3});
		Tensor_OP().permute(value, vt, new int[] {0, 2, 1, 3});
		
		scaledDotProductAttention(qt, kt, vt, mask);

		Tensor_OP().permute(attn_outputs, ot, new int[] {0, 2, 1, 3});

		ot.view(batchSize * time, 1, 1, headNum * dk);
		
		this.oLinerLayer.forward(ot);
		
		if(dropout) {
			this.dropoutLayer.forward(this.oLinerLayer.getOutput());
			this.output = this.oLinerLayer.getOutput();
		}else {
			this.output = this.oLinerLayer.getOutput();
		}
		
	}
	
	public void scaledDotProductAttention(Tensor query,Tensor key,Tensor value,Tensor mask) {

		float d_k = (float) (1.0f / Math.sqrt(dk));

		GPUOP.getInstance().bmm(query.getGpuData(), key.getGpuData(), scores.getGpuData(), query.number * query.channel, query.height, key.height, query.width,
				CUBLAS_OP_N, CUBLAS_OP_T, d_k, 0.0f);
		
		if(mask != null) {
//			mask.showDMByNumber(0);
			
//			scores_masked.showDMByNumber(0);
//			TensorOP.bool(scores, mask, scores_masked, -1e9f);
//			softmax.softmax(scores_masked, weights);
			softmax.softmaxMask(scores, mask, weights, -1e9f);

//			scores.showDMByNumber(0);
//			weights.showDMByNumber(0);
		}else {
			softmax.softmax(scores, weights);
		}

		GPUOP.getInstance().bmm(weights.getGpuData(), value.getGpuData(), attn_outputs.getGpuData(), weights.number * weights.channel, weights.height, value.width, weights.width,
				CUBLAS_OP_N, CUBLAS_OP_N, 1.0f, 0.0f);

	}
	
	public void scaledDotProductAttentionBackward(Tensor query,Tensor key,Tensor value,Tensor delta,Tensor diffQ,Tensor diffK,Tensor diffV) {
		
		// vt_diff = weightsT * delta
		diffV.view(value.shape());
		GPUOP.getInstance().bmm(weights.getGpuData(), delta.getGpuData(), diffV.getGpuData(), weights.number * weights.channel, weights.width, delta.width, weights.height,
				CUBLAS_OP_T, CUBLAS_OP_N, 1.0f, 0.0f);

		// weights_diff = delta * vt
		GPUOP.getInstance().bmm(delta.getGpuData(), value.getGpuData(), scores.getGpuData(), delta.number * delta.channel, delta.height, value.height, value.width,
				CUBLAS_OP_N, CUBLAS_OP_T, 1.0f, 0.0f);

		// scores_diff = softmax_backward
		softmax.backward_noloss(weights, scores, scores);
		
		float d_k = (float) (1.0f / Math.sqrt(dk));
		
//		TensorOP.mul(scores, d_k, scores);

		// kt_diff = deltaT / sqrt(dk) * qt
		diffK.view(key.shape());
		GPUOP.getInstance().bmm(scores.getGpuData(), query.getGpuData(), diffK.getGpuData(), scores.number * scores.channel, scores.width, query.width, scores.height,
				CUBLAS_OP_T, CUBLAS_OP_N, d_k, 0.0f);
		
		// qt_diff = delta / sqrt(dk) * kt
		diffQ.view(query.shape());

		GPUOP.getInstance().bmm(scores.getGpuData(), key.getGpuData(), diffQ.getGpuData(), scores.number * scores.channel, scores.height, key.width, scores.width,
				CUBLAS_OP_N, CUBLAS_OP_N, d_k, 0.0f);
//		
//		System.out.println("");
//		diffV.showDM();
//		diffK.showDM();
//		diffQ.showDM();
	}
	
	public void scaledDotProductAttentionBackward(Tensor query,Tensor key,Tensor value,Tensor delta) {
		
		// vt_diff = weightsT * delta
		GPUOP.getInstance().bmm(weights.getGpuData(), delta.getGpuData(), value.getGrad().getGpuData(), weights.number * weights.channel, weights.width, delta.width, weights.height,
				CUBLAS_OP_T, CUBLAS_OP_N, 1.0f, 0.0f);

		// weights_diff = delta * vt
		GPUOP.getInstance().bmm(delta.getGpuData(), value.getGpuData(), scores.getGpuData(), delta.number * delta.channel, delta.height, weights.width, delta.width,
				CUBLAS_OP_N, CUBLAS_OP_T, 1.0f, 0.0f);
		
		// scores_diff = softmax_backward
		softmax.backward_noloss(weights, scores, scores);
		
		float d_k = (float) (1.0f / Math.sqrt(dk));
		
//		TensorOP.mul(scores, d_k, scores);

		// kt_diff = deltaT / sqrt(dk) * qt
		GPUOP.getInstance().bmm(scores.getGpuData(), query.getGpuData(), key.getGrad().getGpuData(), scores.number * scores.channel, scores.width, query.width, scores.height,
				CUBLAS_OP_T, CUBLAS_OP_N, d_k, 0.0f);
		
		// qt_diff = delta / sqrt(dk) * kt
		GPUOP.getInstance().bmm(scores.getGpuData(), key.getGpuData(), query.getGrad().getGpuData(), scores.number * scores.channel, scores.height, key.width, scores.width,
				CUBLAS_OP_N, CUBLAS_OP_T, d_k, 0.0f);
		
	}

	@Override
	public Tensor getOutput() {
		// TODO Auto-generated method stub
		return output;
	}

	@Override
	public void diff() {
		// TODO Auto-generated method stub
		
		if(dropout) {
//			delta.showDMByNumber(0);
			this.dropoutLayer.back(delta);
//			dropoutLayer.diff.showDMByNumber(0);
//			System.out.println(same(delta, dropoutLayer.diff));
			this.oLinerLayer.back(dropoutLayer.diff, oi);
		}else {
			this.oLinerLayer.back(delta, oi);
		}
		
		oi.view(batchSize, time, headNum, dk);
		
		Tensor_OP().permute(oi, attn_outputs, new int[] {0, 2, 1, 3});
		
		int[] qo_shape = this.qLinerLayer.getOutput().shape();
		int[] ko_shape = this.kLinerLayer.getOutput().shape();
		int[] vo_shape = this.vLinerLayer.getOutput().shape();	
		scaledDotProductAttentionBackward(qt, kt, vt, attn_outputs, this.qLinerLayer.getOutput(), this.kLinerLayer.getOutput(), this.vLinerLayer.getOutput());
		qt.view(qo_shape);
		kt.view(ko_shape);
		vt.view(vo_shape);
		Tensor_OP().permute(this.qLinerLayer.getOutput(), qt, new int[] {0, 2, 1, 3});
		Tensor_OP().permute(this.kLinerLayer.getOutput(), kt, new int[] {0, 2, 1, 3});
		Tensor_OP().permute(this.vLinerLayer.getOutput(), vt, new int[] {0, 2, 1, 3});
		
		Tensor queryDelta = qt.view(batchSize * time, 1, 1, headNum * dk);
		Tensor keyDelta = kt.view(batchSize * time, 1, 1, headNum * dk);
		Tensor valueDelta = vt.view(batchSize * time, 1, 1, headNum * dk);
		
		this.qLinerLayer.back(queryDelta);
		this.kLinerLayer.back(keyDelta);
		this.vLinerLayer.back(valueDelta);
		
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
	
	public void forward(Tensor input,Tensor mask) {
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
		this.output(mask);
		
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
		oLinerLayer.update();
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
	
	public Tensor getWeights() {
		return weights;
	}

	public static void main(String[] args) {
		
		int embedDim = 4;
		int headNum = 2;
		int batchSize = 3;
		int time = 3;
		
		Transformer tf = new Transformer();
		tf.number = batchSize * time;
		tf.time = time;
		
		float[] data = RandomUtils.order(batchSize * time * embedDim, 0.1f, 0.1f);
		
//		int[] rts = new int[] {2, 3, 3};
//		
//		for(int b = 0;b<batchSize;b++) {
//			int rt = rts[b];
//			for(int t = 0;t<time;t++) {
//				if(t > rt) {
//					for(int n = 0;n<embedDim;n++) {
//						data[b * time * embedDim + t * embedDim + n] = 0;
//					}
//				}
//			}
//		}
		
//		float[] maskData = new float[] {1,1,1,0,0,1,1,1,1,0,1,1,1,1,0};
//		
//		Tensor mask = new Tensor(batchSize, 1, time, time, maskData, true);
		
		Tensor mask = ENTokenizer.triu(batchSize, headNum, time, time, 1);
		
		Tensor input = new Tensor(batchSize * time, 1, 1, embedDim, data, true);
		
//		input.showDM();
		
		float[] delta_data = MatrixUtils.val(batchSize * time * embedDim, 1.0f);
		
		float[] tmp = MatrixUtils.val(batchSize * time * embedDim, 1.0f);
		
		Tensor delta = new Tensor(batchSize * time, 1, 1, embedDim, delta_data, true);
		
		CausalSelfAttentionLayer mal = new CausalSelfAttentionLayer(embedDim, headNum, time, false, false, tf);
		
//		mal.forward(input);
		
		for(int i = 0;i<10;i++) {

			mal.forward(input, mask);
			
//			input.showDM();
			
//			mal.getWeights().showDM();
			
			mal.getOutput().showShape();
			
			mal.getOutput().showDM();
			
			mal.back(delta);
//			delta.showDM();
			mal.diff.showDM();
			delta.copyData(tmp);
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

	@Override
	public void accGrad(float scale) {
		// TODO Auto-generated method stub
		qLinerLayer.accGrad(scale);
		kLinerLayer.accGrad(scale);
		vLinerLayer.accGrad(scale);
		oLinerLayer.accGrad(scale);
	}

}
