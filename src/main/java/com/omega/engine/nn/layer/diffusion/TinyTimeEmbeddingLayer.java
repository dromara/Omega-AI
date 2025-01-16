package com.omega.engine.nn.layer.diffusion;

import java.io.IOException;
import java.io.RandomAccessFile;

import com.omega.common.data.Tensor;
import com.omega.common.utils.MatrixUtils;
import com.omega.engine.gpu.CUDAMemoryManager;
import com.omega.engine.gpu.CUDAModules;
import com.omega.engine.nn.layer.EmbeddingIDLayer;
import com.omega.engine.nn.layer.FullyLayer;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.active.SiLULayer;
import com.omega.engine.nn.network.Network;
import com.omega.engine.nn.network.Transformer;

/**
 * diffsion model TimeEmbeddingLayer
 * @author Administrator
 *
 */
public class TinyTimeEmbeddingLayer extends Layer{
	
	private boolean bias = true;
	
	private int T;
	
	private int tdim;
	
	public EmbeddingIDLayer emb;
	public FullyLayer linear1;
	public SiLULayer act;
	public FullyLayer linear2;
	
	public TinyTimeEmbeddingLayer(int T,int tdim,boolean bias, Network network) {
		this.network = network;
		this.bias = bias;
		this.T = T;
		this.tdim = tdim;
		this.height = 1;
		this.width = T;
		this.oHeight = 1;
		this.oWidth = tdim;
		initLayers();
	}

	public void initLayers() {

		emb = new EmbeddingIDLayer(T, tdim, true, network);
//		emb.weight = emb.createTimeEMB(T, tdim);
//		emb.weight.showDM();
		emb.weight = emb.getTimeEMB(T, tdim);
		emb.initFactor(T, tdim);

		
		linear1 = new FullyLayer(tdim, tdim * 4, bias, network);
//		linear1.weight = new Tensor(1, 1, dim, d_model, MatrixUtils.order(dim * d_model, 0.01f, 0.01f), true);
		
		act = new SiLULayer(linear1);
		
		linear2 = new FullyLayer(tdim * 4, tdim, bias, network);
//		linear2.weight = new Tensor(1, 1, dim, dim, MatrixUtils.order(dim * dim, 0.01f, 0.01f), true);
		
	}

	@Override
	public void init() {
		// TODO Auto-generated method stub
		this.number = this.network.number;
	}
	
	@Override
	public void initBack() {
		// TODO Auto-generated method stub

	}

	@Override
	public void initParam() {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void output() {
		// TODO Auto-generated method stub
//		input.showDM();
//		emb.forward(input);
		emb.getTimeEmbedding(input);

		linear1.forward(emb.getOutput());
//		linear1.getOutput().showDM();
		act.forward(linear1.getOutput());
//		act.getOutput().showDM();
		linear2.forward(act.getOutput());
//		linear2.getOutput().showDM();
		this.output = linear2.getOutput();
	}

	@Override
	public Tensor getOutput() {
		// TODO Auto-generated method stub
		return this.output;
	}

	@Override
	public void diff() {
		// TODO Auto-generated method stub
//		System.out.println("index:["+index+"]("+oChannel+")"+this.delta);
		linear2.back(delta);
		act.back(linear2.diff);
		linear1.back(act.diff);
//		linear1.diff.showDM();
//		this.diff = linear1.diff;
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
	}

	@Override
	public void backTemp() {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void forward(Tensor input) {
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
	public void back(Tensor delta) {
		// TODO Auto-generated method stub

		initBack();
		/**
		 * 设置梯度
		 */
		this.setDelta(delta);
		/**
		 * 计算梯度
		 */
		this.diff();

	}
	
	@Override
	public void update() {
		// TODO Auto-generated method stub
		linear1.update();
		linear2.update();
	}

	@Override
	public void showDiff() {
		// TODO Auto-generated method stub
		
	}

	@Override
	public LayerType getLayerType() {
		// TODO Auto-generated method stub
		return LayerType.time_embedding;
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
	
	public void saveModel(RandomAccessFile outputStream) throws IOException {
		linear1.saveModel(outputStream);
		linear2.saveModel(outputStream);
	}
	
	public void loadModel(RandomAccessFile inputStream) throws IOException {
		linear1.loadModel(inputStream);
		linear2.loadModel(inputStream);
	}
	
	public static void main(String[] args) {
    	
	   	  try {


	  		CUDAModules.initContext();
	  		int N = 2;
	  		int T = 1000;
	  		int dim = 4;
	  		
	  		float[] data = new float[] {100, 200};
	  		
	  		Tensor input = new Tensor(N, 1, 1, 1, data, true);
	  		
	  		float[] data2 = MatrixUtils.order(N * dim, 0.01f, 0.01f);
	  		
	  		Tensor delta = new Tensor(N, 1, 1, dim, data2, true);
	  		
	  		Transformer tf = new Transformer();
	  		
	  		tf.CUDNN = true;
	  		tf.number = 2;
	  		
	  		TinyTimeEmbeddingLayer mal = new TinyTimeEmbeddingLayer(T, dim, false, tf);
	  		
	  		mal.forward(input);
	  		
	  		mal.getOutput().showShape();
	  		mal.getOutput().showDM();
	  		
	  		mal.back(delta);
//	  		
//	  		mal.diff.showDM();
			} catch (Exception e) {
				// TODO: handle exception
				e.printStackTrace();
			} finally {
				// TODO: handle finally clause
				CUDAMemoryManager.free();
			}

	   }

	@Override
	public void accGrad(float scale) {
		// TODO Auto-generated method stub
		linear1.accGrad(scale);
		linear2.accGrad(scale);
	}
	
}
