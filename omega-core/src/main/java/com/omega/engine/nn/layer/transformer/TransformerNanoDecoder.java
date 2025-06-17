package com.omega.engine.nn.layer.transformer;

import com.omega.common.utils.RandomUtils;
import com.omega.engine.nn.layer.DropoutLayer;
import com.omega.engine.nn.layer.EmbeddingIDLayer;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.normalization.LNLayer;
import com.omega.engine.nn.network.Network;
import com.omega.engine.tensor.Tensor;
import com.omega.engine.updater.UpdaterFactory;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.util.ArrayList;
import java.util.List;

/**
 * Transformer Decoder Layer
 *
 * @author Administrator
 */
public class TransformerNanoDecoder extends Layer {
    private int time;
    private int vocab_size;
    private int embedDim = 0;
    private boolean bias = false;
    private boolean dropout = false;
    private int headNum = 8;
    private int n_layers = 6;
    private EmbeddingIDLayer src_emb;
    private EmbeddingIDLayer pos_emb;
    private List<TransformerBlock> decoderLayers;
    private LNLayer ln;
    private DropoutLayer dropoutLayer;
    private Tensor positions;

    public TransformerNanoDecoder(int vocab_size, int n_layers, int headNum, int time, int embedDim, boolean bias, boolean dropout) {
        this.headNum = headNum;
        this.n_layers = n_layers;
        this.vocab_size = vocab_size;
        this.time = time;
        this.embedDim = embedDim;
        this.bias = bias;
        this.dropout = dropout;
        this.channel = 1;
        this.height = 1;
        this.width = embedDim;
        this.oChannel = 1;
        this.oHeight = 1;
        this.oWidth = embedDim;
        this.initLayers();
    }

    public TransformerNanoDecoder(int vocab_size, int n_layers, int headNum, int time, int embedDim, boolean bias, boolean dropout, Network network) {
        this.headNum = headNum;
        this.n_layers = n_layers;
        this.network = network;
        if (this.updater == null) {
            this.setUpdater(UpdaterFactory.create(network));
        }
        this.vocab_size = vocab_size;
        this.time = time;
        this.embedDim = embedDim;
        this.bias = bias;
        this.dropout = dropout;
        this.channel = 1;
        this.height = 1;
        this.width = embedDim;
        this.oChannel = 1;
        this.oHeight = 1;
        this.oWidth = embedDim;
        this.initLayers();
    }

    public void initLayers() {
        this.src_emb = new EmbeddingIDLayer(vocab_size, embedDim, network);
        this.src_emb.weight = new Tensor(1, 1, src_emb.width, src_emb.oWidth, RandomUtils.uniform(this.src_emb.width * this.src_emb.oWidth, 0.0f, 0.02f), true);
        this.pos_emb = new EmbeddingIDLayer(time, embedDim, network);
        this.pos_emb.weight = new Tensor(1, 1, pos_emb.width, pos_emb.oWidth, RandomUtils.uniform(this.pos_emb.width * this.pos_emb.oWidth, 0.0f, 0.02f), true);
        decoderLayers = new ArrayList<TransformerBlock>();
        for (int i = 0; i < n_layers; i++) {
            TransformerBlock decoderLayer = new TransformerBlock(headNum, time, embedDim, bias, dropout, network);
            decoderLayers.add(decoderLayer);
        }
        this.ln = new LNLayer(decoderLayers.get(n_layers - 1), bias);
        if (dropout) {
            dropoutLayer = new DropoutLayer(0.1f, src_emb);
        }
    }

    @Override
    public void init() {
        // TODO Auto-generated method stub
        this.number = this.input.getShape()[0];
        this.time = this.network.time;
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
        src_emb.forward(input);
        pos_emb.forward(positions);
        Tensor_OP().add(src_emb.getOutput(), pos_emb.getOutput(), src_emb.getOutput());
        Tensor out1 = src_emb.getOutput();
        if (dropout) {
            this.dropoutLayer.forward(out1);
            out1 = dropoutLayer.getOutput();
        }
        for (int i = 0; i < n_layers; i++) {
            decoderLayers.get(i).forward(out1);
            out1 = decoderLayers.get(i).getOutput();
        }
        this.ln.forward(out1);
        this.output = this.ln.getOutput();
        //		this.output = decoderOutput;
    }

    public void output(Tensor positions) {
        // TODO Auto-generated method stub
        src_emb.forward(input);
        pos_emb.forward(positions);
        Tensor_OP().add(src_emb.getOutput(), pos_emb.getOutput(), src_emb.getOutput());
        Tensor out1 = src_emb.getOutput();
        if (dropout) {
            this.dropoutLayer.forward(out1);
            out1 = dropoutLayer.getOutput();
        }
        for (int i = 0; i < n_layers; i++) {
            decoderLayers.get(i).forward(out1);
            out1 = decoderLayers.get(i).getOutput();
        }
        this.ln.forward(out1);
        this.output = this.ln.getOutput();
        //		output.showDMByNumber(output.number - 1);
        //		this.output = out1;
    }

    //	public void output(Tensor mask,Tensor positions) {
    //		// TODO Auto-generated method stub
    //
    //		src_emb.forward(input);
    //
    //		pos_emb.forward(positions);
    //
    //		TensorOP.add(src_emb.getOutput(), pos_emb.getOutput(), src_emb.getOutput());
    //
    //		Tensor out1 = src_emb.getOutput();
    //
    //		if(dropout) {
    //			this.dropoutLayer.forward(out1);
    //			out1 = dropoutLayer.getOutput();
    //		}
    //
    //		for(int i = 0;i<n_layers;i++) {
    //			decoderLayers.get(i).forward(out1, mask);
    //			out1 = decoderLayers.get(i).getOutput();
    //		}
    //
    //		this.ln.forward(out1);
    //		this.output = this.ln.getOutput();
    ////		this.output = decoderOutput;
    //	}
    @Override
    public Tensor getOutput() {
        // TODO Auto-generated method stub
        return output;
    }

    @Override
    public void diff() {
        // TODO Auto-generated method stub
        this.ln.back(delta);
        Tensor decoderDiff = this.ln.diff;
        //		Tensor decoderDiff = delta;
        for (int i = n_layers - 1; i >= 0; i--) {
            decoderLayers.get(i).back(decoderDiff);
            decoderDiff = decoderLayers.get(i).diff;
        }
        if (dropout) {
            this.dropoutLayer.back(decoderDiff);
            decoderDiff = dropoutLayer.diff;
        }
        src_emb.back(decoderDiff);
        pos_emb.back(decoderDiff);
        this.diff = this.src_emb.diff;
    }

    @Override
    public void forward() {
        // TODO Auto-generated method stub
        /**
         * 设置输入

         */
        this.setInput();
        /**
         * 参数初始化

         */
        this.init();
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
        if (this.network.GRADIENT_CHECK) {
            this.gradientCheck();
        }
    }

    @Override
    public void forward(Tensor input) {
        // TODO Auto-generated method stub
        /**
         * 设置输入

         */
        this.setInput(input);
        /**
         * 参数初始化

         */
        this.init();
        /**
         * 计算输出

         */
        this.output();
    }

    public void forward(Tensor input, Tensor positions) {
        // TODO Auto-generated method stub
        /**
         * 设置输入

         */
        this.setInput(input);
        /**
         * 参数初始化

         */
        this.init();
        /**
         * 计算输出

         */
        this.output(positions);
    }

    //	public void forward(Tensor input,Tensor mask,Tensor positions) {
    //		// TODO Auto-generated method stub
    //		/**
    //		 * 设置输入
    //		 */
    //		this.setInput(input);
    //		/**
    //		 * 参数初始化
    //		 */
    //		this.init();
    //		/**
    //		 * 计算输出
    //		 */
    //		this.output(mask, positions);
    //
    //	}
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
        if (this.network.GRADIENT_CHECK) {
            this.gradientCheck();
        }
    }

    @Override
    public void update() {
        // TODO Auto-generated method stub
        src_emb.update();
        pos_emb.update();
        ln.update();
        for (int i = 0; i < n_layers; i++) {
            decoderLayers.get(i).update();
        }
    }

    @Override
    public void showDiff() {
        // TODO Auto-generated method stub
    }

    @Override
    public LayerType getLayerType() {
        // TODO Auto-generated method stub
        return LayerType.transformer_decoder;
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

    public void saveModel(RandomAccessFile outputStream) throws IOException {
        src_emb.saveModel(outputStream);
        System.out.println("src_emb save success...");
        pos_emb.saveModel(outputStream);
        System.out.println("pos_emb save success...");
        for (int i = 0; i < n_layers; i++) {
            decoderLayers.get(i).saveModel(outputStream);
        }
        ln.saveModel(outputStream);
        System.out.println("ln save success...");
    }

    public void loadModel(RandomAccessFile inputStream) throws IOException {
        src_emb.loadModel(inputStream);
        pos_emb.loadModel(inputStream);
        for (int i = 0; i < n_layers; i++) {
            decoderLayers.get(i).loadModel(inputStream);
        }
        ln.loadModel(inputStream);
    }

    @Override
    public void accGrad(float scale) {
        // TODO Auto-generated method stub
        src_emb.accGrad(scale);
        pos_emb.accGrad(scale);
        for (int i = 0; i < n_layers; i++) {
            decoderLayers.get(i).accGrad(scale);
        }
        ln.accGrad(scale);
    }
}

