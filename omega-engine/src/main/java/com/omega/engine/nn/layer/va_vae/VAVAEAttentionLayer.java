package com.omega.engine.nn.layer.va_vae;

import java.io.IOException;
import java.io.RandomAccessFile;

import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.normalization.BNType;
import com.omega.engine.nn.layer.normalization.GNLayer;
import com.omega.engine.nn.network.Network;
import com.omega.engine.nn.network.RunModel;
import com.omega.engine.tensor.Tensor;
import com.omega.engine.updater.UpdaterFactory;

/**
 * VAVAEAttentionLayer
 *
 * @author Administrator
 */
public class VAVAEAttentionLayer extends Layer {
    public GNLayer gn;
    public VAVAESelfAttentionLayer attn;
    private int groups = 0;
    private int channel;
    private int height;
    private int width;
    private boolean bias = false;
    private int batchSize = 1;

    public VAVAEAttentionLayer(int channel, int height, int width, int groups, boolean bias, Network network) {
        this.bias = bias;
        this.groups = groups;
        this.network = network;
        if (this.updater == null) {
            this.setUpdater(UpdaterFactory.create(network));
        }

        this.bias = bias;
        this.channel = channel;
        this.height = height;
        this.width = width;
        this.oChannel = channel;
        this.oHeight = height;
        this.oWidth = width;
        this.initLayers();
    }

    public static void main(String[] args) {
        try {
//            int N = 2;
//            int headNum = 4;
//            int groups = 2;
//            int channel = 4;
//            int height = 8;
//            int width = 8;
//            int dataSize = N * channel * height * width;
//            Transformer network = new Transformer();
//            network.CUDNN = true;
//            VAVAEAttentionLayer attn = new VAVAEAttentionLayer(channel, headNum, height, width, groups, false, network);
//            Tensor x = new Tensor(N, channel, height, width, MatrixUtils.order(dataSize, 0.1f, 0.1f), true);
//            Tensor delta = new Tensor(N, channel, height, width, MatrixUtils.order(dataSize, 0.1f, 0.1f), true);
//            attn.forward(x);
//            attn.getOutput().showDM();
//            attn.back(delta);
//            attn.diff.showDM();
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public static boolean same(Tensor a, Tensor b) {
        float[] ad = a.syncHost();
        float[] bd = b.syncHost();
        for (int i = 0; i < ad.length; i++) {
            if (ad[i] != bd[i]) {
                System.out.println(ad[i] + ":" + bd[i] + "[" + i + "]");
                return false;
            }
        }
        return true;
    }

    public void initLayers() {
        if (groups > 0) {
            gn = new GNLayer(groups, channel, height, width, BNType.conv_bn, this);
        }
        attn = new VAVAESelfAttentionLayer(channel, height, width, bias, network);
    }

    @Override
    public void init() {
        // TODO Auto-generated method stub
    }

    public void init(Tensor input) {
        // TODO Auto-generated method stub
        this.number = input.number;
        this.batchSize = this.number;

        if (network.RUN_MODEL == RunModel.EVAL) {
            // [batch_size，time，head_num，d_k]
        	if (this.output == null || output.number != batchSize) {
                this.output = Tensor.createGPUTensor(this.output, batchSize, channel, height, width, true);
            }
        } else {
        	if (this.output == null || output.number != batchSize) {
                this.output = Tensor.createGPUTensor(this.output, batchSize, channel, height, width, true);
            }
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
        Tensor x = this.input;
//        x.showDMByOffsetRed(0, 100, "attn-x");
        if (gn != null) {
            gn.forward(x);
            x = gn.getOutput();
        }
        attn.forward(x);
        attn.getOutput().showDMByOffsetRed(0, 100, "h_");
        Tensor_OP().add(this.input, attn.getOutput(), output);
    }

    @Override
    public Tensor getOutput() {
        // TODO Auto-generated method stub
        return output;
    }

    @Override
    public void diff() {
        // TODO Auto-generated method stub
        // B,C,H,W ==> B,HW,C
        attn.back(delta);
        if (gn != null) {
            gn.back(attn.diff);
            this.diff = gn.diff;
        } else {
            this.diff = attn.diff;
        }
        Tensor_OP().add(this.diff, this.delta, this.diff);
    }

    @Override
    public void forward() {
        // TODO Auto-generated method stub
    }

    @Override
    public void back() {
        // TODO Auto-generated method stub
    }

    @Override
    public void forward(Tensor input) {
        // TODO Auto-generated method stub
        //		input.showDMByOffset(0, 100, "123");
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
        if (this.network.GRADIENT_CHECK) {
            this.gradientCheck();
        }
    }

    @Override
    public void update() {
        // TODO Auto-generated method stub
        if (gn != null) {
            gn.update();
        }
        attn.update();
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

    //	public Tensor getWeights() {
    //		return weights;
    //	}
    @Override
    public void initCache() {
        // TODO Auto-generated method stub
    }

    @Override
    public void backTemp() {
        // TODO Auto-generated method stub
    }

    public void saveModel(RandomAccessFile outputStream) throws IOException {
        if (groups > 0) {
            gn.saveModel(outputStream);
        }
        attn.saveModel(outputStream);
    }

    public void loadModel(RandomAccessFile inputStream) throws IOException {
        if (groups > 0) {
            gn.loadModel(inputStream);
        }
        attn.loadModel(inputStream);
    }

    @Override
    public void accGrad(float scale) {
        // TODO Auto-generated method stub
        if (groups > 0) {
            gn.accGrad(scale);
        }
        attn.accGrad(scale);
    }
}

