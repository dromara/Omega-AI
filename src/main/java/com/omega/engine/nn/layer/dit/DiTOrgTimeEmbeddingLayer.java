package com.omega.engine.nn.layer.dit;

import java.io.IOException;
import java.io.RandomAccessFile;

import com.omega.common.utils.MatrixOperation;
import com.omega.common.utils.MatrixUtils;
import com.omega.common.utils.RandomUtils;
import com.omega.engine.gpu.CUDAMemoryManager;
import com.omega.engine.gpu.CUDAModules;
import com.omega.engine.nn.layer.FullyLayer;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.active.SiLULayer;
import com.omega.engine.nn.layer.dit.kernel.TokenDropKernel;
import com.omega.engine.nn.network.Network;
import com.omega.engine.nn.network.Transformer;
import com.omega.engine.tensor.Tensor;

/**
 * diffsion model TimeEmbeddingLayer
 *
 * @author Administrator
 */
public class DiTOrgTimeEmbeddingLayer extends Layer {

    public FullyLayer linear1;
    public SiLULayer act;
    public FullyLayer linear2;
    private boolean bias = true;
    private int T;
    private int d_model;
    private int dim;

    private Tensor freqs;
    private Tensor emb;
    
    private TokenDropKernel tokenDropKernel;
    
    public DiTOrgTimeEmbeddingLayer(int T, int d_model, int dim, boolean bias, Network network) {
        this.network = network;
        this.bias = bias;
        this.T = T;
        this.d_model = d_model;
        this.dim = dim;
        this.height = 1;
        this.width = T;
        this.oHeight = 1;
        this.oWidth = dim;
        initLayers();
    }

    public static void main(String[] args) {
        try {
            CUDAModules.initContext();
            int N = 2;
            int T = 1000;
            int d_model = 128;
            int dim = d_model * 4;
            float[] data = new float[]{0.04f, 0.9f};
            Tensor input = new Tensor(N, 1, 1, 1, data, true);
            float[] data2 = MatrixUtils.order(N * dim, 0.01f, 0.01f);
            Tensor delta = new Tensor(N, 1, 1, dim, data2, true);
            Transformer tf = new Transformer();
            tf.CUDNN = true;
            tf.number = 2;
            DiTOrgTimeEmbeddingLayer mal = new DiTOrgTimeEmbeddingLayer(T, d_model, dim, false, tf);
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

    public void initLayers() {
    	
    	float[] emb = MatrixUtils.order(d_model / 2, 0, (float) (-2.0f / d_model * Math.log(10000)));
        emb = MatrixOperation.exp(emb);
    	freqs = new Tensor(1, 1, 1, d_model / 2, emb, true);

        linear1 = new FullyLayer(d_model, dim, bias, network);
        linear1.weight.setData(RandomUtils.normal_(d_model * dim, 0.0f, 0.02f));
        if(linear1.bias != null) {
        	linear1.bias.clearGPU();
        }
        act = new SiLULayer(linear1);
        linear2 = new FullyLayer(dim, dim, bias, network);
        linear2.weight.setData(RandomUtils.normal_(dim * dim, 0.0f, 0.02f));
        if(linear2.bias != null) {
        	linear2.bias.clearGPU();
        }
        if(tokenDropKernel == null) {
        	tokenDropKernel = new TokenDropKernel(cuda());
        }
//        linear2.weight = new Tensor(1, 1, dim, dim, MatrixUtils.order(dim * dim, 0.01f, 0.01f), true);
    }

    @Override
    public void init() {
        // TODO Auto-generated method stub
        this.number = this.network.number;
        if(emb == null || emb.number != number) {
        	emb = Tensor.createGPUTensor(emb, number, 1, 1, d_model, true);
        }
    }
    
    public void init(Tensor input) {
        // TODO Auto-generated method stub
        this.number = input.number;
        if(emb == null || emb.number != number) {
        	emb = Tensor.createGPUTensor(emb, number, 1, 1, d_model, true);
        }
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
    	tokenDropKernel.timestep_embedding(input, freqs, emb, d_model);
//    	emb.showDM();
        linear1.forward(emb);
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

    @Override
    public void accGrad(float scale) {
        // TODO Auto-generated method stub
        linear1.accGrad(scale);
        linear2.accGrad(scale);
    }
}

