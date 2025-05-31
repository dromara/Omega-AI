package com.omega.engine.nn.layer.asr;

import com.omega.engine.gpu.MaskKernel;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.network.Network;
import com.omega.engine.tensor.Tensor;
import com.omega.engine.updater.UpdaterFactory;
import com.omega.example.transformer.utils.ENTokenizer;

import java.io.IOException;
import java.io.RandomAccessFile;

/**
 * ASRTransformer
 *
 * @author Administrator
 */
public class ASRTransformer extends Layer {
    public ASREncoder encoder;
    public ASRDecoder decoder;
    private int wavDim;
    private int wavTime;
    private int vocSize;
    private int vocTime;
    private int embedDim = 0;
    private int nChannel = 1;
    private boolean bias = false;
    private int headNum = 8;
    private int n_layers = 4;
    private Tensor wavMask;
    private Tensor vocMask;
    private Tensor en_de_mask;
    private Tensor encoder_positions;
    private Tensor decoder_positions;
    private MaskKernel maskKernel;

    public ASRTransformer(int wavDim, int wavTime, int vocSize, int vocTime, int embedDim, int headNum, int nChannel, int n_layers, boolean bias) {
        this.wavDim = wavDim;
        this.wavTime = wavTime;
        this.vocSize = vocSize;
        this.vocTime = vocTime;
        this.embedDim = embedDim;
        this.headNum = headNum;
        this.nChannel = nChannel;
        this.n_layers = n_layers;
        this.bias = bias;
        this.initLayers();
    }

    public ASRTransformer(int wavDim, int wavTime, int vocSize, int vocTime, int embedDim, int headNum, int nChannel, int n_layers, boolean bias, Network network) {
        this.network = network;
        if (this.updater == null) {
            this.setUpdater(UpdaterFactory.create(network));
        }
        this.wavDim = wavDim;
        this.wavTime = wavTime;
        this.vocSize = vocSize;
        this.vocTime = vocTime;
        this.embedDim = embedDim;
        this.headNum = headNum;
        this.nChannel = nChannel;
        this.n_layers = n_layers;
        this.bias = bias;
        this.initLayers();
    }

    public void initLayers() {
        this.encoder = new ASREncoder(wavDim, wavTime, embedDim, headNum, nChannel, n_layers, bias, network);
        this.decoder = new ASRDecoder(vocSize, vocTime, wavTime, embedDim, headNum, nChannel, n_layers, bias, network);
        if (maskKernel == null) {
            maskKernel = new MaskKernel(network.cudaManager);
        }
    }

    @Override
    public void init() {
        // TODO Auto-generated method stub
        this.number = this.input.number / wavTime;
        if (wavMask == null || wavMask.number != number) {
            wavMask = maskKernel.createOutput(number, wavTime, headNum);
            encoder_positions = createPositions(number, wavTime);
            decoder_positions = createPositions(number, vocTime);
            vocMask = ENTokenizer.triu(number, headNum, vocTime, vocTime, 1.0f);
            en_de_mask = maskKernel.createUnMaskOutput(number, vocTime, wavTime, headNum);
        }
    }

    public Tensor createPositions(int number, int time) {
        float[] data = new float[number * time];
        for (int n = 0; n < number; n++) {
            for (int t = 0; t < time; t++) {
                data[n * time + t] = t;
            }
        }
        return new Tensor(number * time, 1, 1, 1, data, true);
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
    }

    public void output(Tensor wavInput, Tensor wavLen, Tensor labelInput, Tensor labelLen) {
        // TODO Auto-generated method stub
        maskKernel.createHeadMask(wavLen, wavMask, number, wavTime, headNum);
        maskKernel.createHeadUnMask(wavLen, en_de_mask, number, vocTime, wavTime, headNum);
        encoder.forward(wavInput, wavMask, encoder_positions);
        decoder.forward(labelInput, encoder.getOutput(), vocMask, en_de_mask, decoder_positions);
        this.output = decoder.getOutput();
    }

    @Override
    public Tensor getOutput() {
        // TODO Auto-generated method stub
        return output;
    }

    @Override
    public void diff() {
        // TODO Auto-generated method stub
        decoder.back(delta);
        encoder.back(decoder.diff);
    }

    @Override
    public void forward() {
        // TODO Auto-generated method stub
        /**
         * 设置输入
         *
         */
        this.setInput();
        /**
         * 参数初始化
         *
         */
        this.init();
        /**
         * 计算输出
         *
         */
        this.output();
    }

    @Override
    public void back() {
        // TODO Auto-generated method stub
        this.initBack();
        /**
         * 设置梯度
         *
         */
        this.setDelta();
        /**
         * 计算梯度
         *
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
         *
         */
        this.setInput(input);
        /**
         * 参数初始化
         *
         */
        this.init();
        /**
         * 计算输出
         *
         */
        this.output();
    }

    public void forward(Tensor wavInput, Tensor wavLen, Tensor labelInput, Tensor labelLen) {
        // TODO Auto-generated method stub
        /**
         * 设置输入
         *
         */
        this.setInput(wavInput);
        /**
         * 参数初始化
         *
         */
        this.init();
        /**
         * 计算输出
         *
         */
        this.output(wavInput, wavLen, labelInput, labelLen);
    }

    @Override
    public void back(Tensor delta) {
        // TODO Auto-generated method stub
        this.initBack();
        /**
         * 设置梯度
         *
         */
        this.setDelta(delta);
        /**
         * 计算梯度
         *
         */
        this.diff();
        if (this.network.GRADIENT_CHECK) {
            this.gradientCheck();
        }
    }

    @Override
    public void update() {
        // TODO Auto-generated method stub
        decoder.update();
        encoder.update();
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

    @Override
    public void accGrad(float scale) {
        // TODO Auto-generated method stub
        decoder.accGrad(scale);
        encoder.accGrad(scale);
    }

    public void saveModel(RandomAccessFile outputStream) throws IOException {
        encoder.saveModel(outputStream);
        decoder.saveModel(outputStream);
    }

    public void loadModel(RandomAccessFile inputStream) throws IOException {
        encoder.loadModel(inputStream);
        decoder.loadModel(inputStream);
    }
}
