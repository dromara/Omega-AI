package com.omega.engine.nn.layer.diffusion.unet;

import com.omega.common.config.Tensor;
import com.omega.common.utils.RandomUtils;
import com.omega.engine.nn.layer.ConvolutionLayer;
import com.omega.engine.nn.layer.FullyLayer;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.active.GeluLayer;
import com.omega.engine.nn.layer.normalization.BNType;
import com.omega.engine.nn.layer.normalization.GNLayer;
import com.omega.engine.nn.layer.normalization.LNLayer;
import com.omega.engine.nn.network.Network;
import com.omega.engine.nn.network.RunModel;
import com.omega.engine.nn.network.Transformer;
import com.omega.engine.updater.UpdaterType;
import com.omega.example.clip.utils.ClipModelUtils;
import com.omega.example.transformer.utils.LagJsonReader;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.util.Map;

/**
 * UNetFFNBlockLayer
 *
 * @author Administrator
 */
public class UNetAttentionBlock extends Layer {
    public GNLayer gn;
    public ConvolutionLayer conv_in;
    public LNLayer ln1;
    public UNetSelfAttentionLayer attn;
    public LNLayer ln2;
    public UNetCrossAttentionLayer cross_attn;
    public LNLayer ln3;
    public FullyLayer geglu1;
    public GeluLayer gelu;
    public FullyLayer geglu2;
    public ConvolutionLayer conv_out;
    private int embDim;
    private int time;
    private int contextDim;
    private int contextTime;
    private int headNum;
    private int groupNum = 32;
    private Tensor xt;
    private Tensor x1;
    private Tensor x2;
    private Tensor x3;
    private Tensor tmp;

    public UNetAttentionBlock(int channel, int height, int width, int headNum, int contextTime, int contextDim, int groupNum, Network network) {
        this.network = network;
        this.groupNum = groupNum;
        this.headNum = headNum;
        this.contextTime = contextTime;
        this.contextDim = contextDim;
        this.time = height * width;
        this.embDim = channel;
        this.channel = channel;
        this.oChannel = channel;
        this.height = height;
        this.width = width;
        this.oHeight = height;
        this.oWidth = width;
        initLayers();
    }

    public static void loadWeight(Map<String, Object> weightMap, UNetAttentionBlock network, boolean showLayers) {
        if (showLayers) {
            for (String key : weightMap.keySet()) {
                System.out.println(key);
            }
        }
        network.gn.gamma = ClipModelUtils.loadData(network.gn.gamma, weightMap, 1, "groupnorm.weight");
        network.gn.beta = ClipModelUtils.loadData(network.gn.beta, weightMap, 1, "groupnorm.bias");
        network.conv_in.weight = ClipModelUtils.loadData(network.conv_in.weight, weightMap, 4, "conv_input.weight");
        ClipModelUtils.loadData(network.conv_in.bias, weightMap, "conv_input.bias");
        network.ln1.gamma = ClipModelUtils.loadData(network.ln1.gamma, weightMap, 1, "layernorm_1.weight");
        network.ln1.beta = ClipModelUtils.loadData(network.ln1.beta, weightMap, 1, "layernorm_1.bias");
        network.ln2.gamma = ClipModelUtils.loadData(network.ln2.gamma, weightMap, 1, "layernorm_2.weight");
        network.ln2.beta = ClipModelUtils.loadData(network.ln2.beta, weightMap, 1, "layernorm_2.bias");
        network.ln3.gamma = ClipModelUtils.loadData(network.ln3.gamma, weightMap, 1, "layernorm_3.weight");
        network.ln3.beta = ClipModelUtils.loadData(network.ln3.beta, weightMap, 1, "layernorm_3.bias");
        ClipModelUtils.loadData(network.attn.qkvLinerLayer.weight, weightMap, "attention_1.in_proj.weight");
        ClipModelUtils.loadData(network.attn.oLinerLayer.weight, weightMap, "attention_1.out_proj.weight");
        ClipModelUtils.loadData(network.attn.oLinerLayer.bias, weightMap, "attention_1.out_proj.bias");
        ClipModelUtils.loadData(network.cross_attn.qLinerLayer.weight, weightMap, "attention_2.q_proj.weight");
        ClipModelUtils.loadData(network.cross_attn.kLinerLayer.weight, weightMap, "attention_2.k_proj.weight");
        ClipModelUtils.loadData(network.cross_attn.vLinerLayer.weight, weightMap, "attention_2.v_proj.weight");
        ClipModelUtils.loadData(network.cross_attn.oLinerLayer.weight, weightMap, "attention_2.out_proj.weight");
        ClipModelUtils.loadData(network.cross_attn.oLinerLayer.bias, weightMap, "attention_2.out_proj.bias");
        ClipModelUtils.loadData(network.geglu1.weight, weightMap, "linear_geglu_1.weight");
        ClipModelUtils.loadData(network.geglu1.bias, weightMap, "linear_geglu_1.bias");
        ClipModelUtils.loadData(network.geglu2.weight, weightMap, "linear_geglu_2.weight");
        ClipModelUtils.loadData(network.geglu2.bias, weightMap, "linear_geglu_2.bias");
        network.conv_out.weight = ClipModelUtils.loadData(network.conv_out.weight, weightMap, 4, "conv_output.weight");
        ClipModelUtils.loadData(network.conv_out.bias, weightMap, "conv_output.bias");
    }

    public static void main(String[] args) {
        int batchSize = 2;
        int channel = 64;
        int height = 32;
        int width = 32;
        int headNum = 8;
        int context_time = 64;
        int context_dim = 128;
        Transformer tf = new Transformer();
        tf.updater = UpdaterType.adamw;
        tf.CUDNN = true;
        tf.learnRate = 0.001f;
        tf.RUN_MODEL = RunModel.TRAIN;
        tf.number = batchSize;
        float[] data = RandomUtils.order(batchSize * channel * height * width, 0.1f, 0.1f);
        Tensor input = new Tensor(batchSize, channel, height, width, data, true);
        float[] cdata = RandomUtils.order(batchSize * context_time * context_dim, 0.1f, 0.1f);
        Tensor context = new Tensor(batchSize * context_time, 1, 1, context_dim, cdata, true);
        //		float[] delta_data = MatrixUtils.val(batchSize * channel * height * width, 1.0f);
        float[] delta_data = RandomUtils.order(batchSize * channel * height * width, 0.01f, 0.01f);
        Tensor delta = new Tensor(batchSize, channel, height, width, delta_data, true);
        UNetAttentionBlock block = new UNetAttentionBlock(channel, height, width, headNum, context_time, context_dim, 32, tf);
        String weight = "H:\\model\\attn_block.json";
        loadWeight(LagJsonReader.readJsonFileSmallWeight(weight), block, true);
        for (int i = 0; i < 10; i++) {
            //			input.showDM();
            tf.train_time++;
            block.forward(input, context);
            block.getOutput().showShape();
            block.getOutput().showDM();
            block.back(delta);
            //			delta.showDM();
            block.diff.showDM();
            block.gn.diffGamma.showDM("dgamma");
            block.gn.diffBeta.showDM("dbeta");
            block.update();
            block.gn.gamma.showDM("gamma");
            block.gn.beta.showDM("beta");
            //			delta.copyData(tmp);
        }
    }

    public void initLayers() {
        gn = new GNLayer(groupNum, channel, height, width, BNType.conv_bn, this);
        conv_in = new ConvolutionLayer(channel, channel, width, height, 1, 1, 0, 1, true, network);
        ln1 = new LNLayer(conv_in, BNType.fully_bn, 1, 1, channel);
        attn = new UNetSelfAttentionLayer(embDim, headNum, time, false, false, network);
        if (contextDim > 0) {
            ln2 = new LNLayer(attn, BNType.fully_bn, 1, 1, channel);
            cross_attn = new UNetCrossAttentionLayer(embDim, contextDim, headNum, time, contextTime, false, false, network);
            ln3 = new LNLayer(cross_attn, BNType.fully_bn, 1, 1, channel);
        } else {
            ln3 = new LNLayer(attn, BNType.fully_bn, 1, 1, channel);
        }
        geglu1 = new FullyLayer(channel, 4 * channel, true, network);
        gelu = new GeluLayer(geglu1);
        geglu2 = new FullyLayer(4 * channel, channel, true, network);
        conv_out = new ConvolutionLayer(channel, channel, width, height, 1, 1, 0, 1, true, network);
    }

    @Override
    public void init() {
        // TODO Auto-generated method stub
        this.number = this.network.number;
    }

    public void init(Tensor input) {
        // TODO Auto-generated method stub
        this.number = input.number;
        if (xt != null) {
            xt.viewOrg();
            x1.viewOrg();
            x2.viewOrg();
            x3.viewOrg();
            tmp.viewOrg();
        }
        if (this.xt == null || this.xt.number != this.number) {
            this.xt = Tensor.createGPUTensor(this.xt, number, height, width, channel, true);
        }
        if (this.x1 == null || this.x1.number != this.number * time) {
            this.x1 = Tensor.createGPUTensor(this.x1, number * time, 1, 1, channel, true);
            this.x2 = Tensor.createGPUTensor(this.x2, number * time, 1, 1, channel, true);
            this.x3 = Tensor.createGPUTensor(this.x3, number * time, 1, 1, channel, true);
        }
        if (this.tmp == null || this.tmp.number != this.number) {
            this.tmp = Tensor.createGPUTensor(this.tmp, number, height, width, channel, true);
        }
        if (this.output == null || this.output.number != this.number) {
            this.output = Tensor.createGPUTensor(this.output, number, oChannel, oHeight, oWidth, true);
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
        gn.forward(input);
        conv_in.forward(gn.getOutput());
        //[b, c, h, w] --> [b, h*w, c]
        Tensor_OP().permute(conv_in.getOutput(), xt, new int[]{0, 2, 3, 1});
        xt.view(number * time, 1, 1, channel);
        ln1.forward(xt);
        attn.forward(ln1.getOutput());
        Tensor_OP().add(attn.getOutput(), xt, x1);
        ln3.forward(x1);
        geglu1.forward(ln3.getOutput());
        gelu.forward(geglu1.getOutput());
        geglu2.forward(gelu.getOutput());
        Tensor_OP().add(geglu2.getOutput(), x2, x3);
        //[b, h*w, c] --> [b, c, h, w]
        x3.view(number, time, 1, channel);
        tmp.view(number, channel, 1, time);
        Tensor_OP().permute(x3, tmp, new int[]{0, 3, 2, 1});
        tmp.view(number, channel, height, width);
        conv_out.forward(tmp);
        this.output = conv_out.getOutput();
        //		output.showDMByOffsetRed(0, 100, "down-output");
        Tensor_OP().add(this.output, this.input, this.output);
    }

    public void output(Tensor context) {
        // TODO Auto-generated method stub
        gn.forward(input);
        conv_in.forward(gn.getOutput());
        //[b, c, h, w] --> [b, h*w, c]
        Tensor_OP().permute(conv_in.getOutput(), xt, new int[]{0, 2, 3, 1});
        xt.view(number * time, 1, 1, channel);
        //		xt.showDM("xt");
        //		ln1.gamma.showDM("ln1.gamma");
        //		ln1.beta.showDM("ln1.beta");
        ln1.forward(xt);
        //		ln1.getOutput().showDM("ln1");
        //		System.out.println("attn---->in");
        attn.forward(ln1.getOutput());
        //		attn.getOutput().showDM("attn");
        Tensor_OP().add(attn.getOutput(), xt, x1);
        //		x1.showDM("x1");
        ln2.forward(x1);
        //		System.out.println("cross_attn---->in");
        cross_attn.forward(ln2.getOutput(), context);
        //		cross_attn.getOutput().showDMByOffsetRed(10 * x2.width * x2.height, x2.width * x2.height, "cross_attn.getOutput()");
        Tensor_OP().add(cross_attn.getOutput(), x1, x2);
        ln3.forward(x2);
        //		System.out.println("geglu1---->in");
        geglu1.forward(ln3.getOutput());
        gelu.forward(geglu1.getOutput());
        //		System.out.println("geglu2---->in");
        geglu2.forward(gelu.getOutput());
        Tensor_OP().add(geglu2.getOutput(), x2, x3);
        //[b, h*w, c] --> [b, c, h, w]
        x3.view(number, time, 1, channel);
        tmp.view(number, channel, 1, time);
        Tensor_OP().permute(x3, tmp, new int[]{0, 3, 2, 1});
        tmp.view(number, channel, height, width);
        conv_out.forward(tmp);
        this.output = conv_out.getOutput();
        //		output.showDMByOffsetRed(0, 100, "down-output");
        Tensor_OP().add(this.output, this.input, this.output);
        //		output.showDMByOffsetRed(0, 100, "down-output-add");
        //		output.showDMByOffsetRed(0, output.width, "down-output");
    }

    @Override
    public Tensor getOutput() {
        // TODO Auto-generated method stub
        return this.output;
    }

    @Override
    public void diff() {
        // TODO Auto-generated method stub
        conv_out.back(delta);
        //delta[b, c, h, w] --> [b, h*w, c]
        tmp.view(number, height, width, channel);
        Tensor_OP().permute(conv_out.diff, tmp, new int[]{0, 2, 3, 1});
        Tensor gdiff = tmp.view(number * time, 1, 1, channel);
        geglu2.back(gdiff);
        gelu.back(geglu2.diff);
        geglu1.back(gelu.diff);
        ln3.back(geglu1.diff);
        Tensor_OP().add(ln3.diff, gdiff, ln3.diff);
        Tensor d = ln3.diff;
        if (cross_attn != null) {
            //			ln3.diff.showDM("cross_attn_in");
            cross_attn.back(d);
            //			cross_attn.diff.showDM("cross_attn");
            ln2.back(cross_attn.diff);
            Tensor_OP().add(ln2.diff, ln3.diff, ln2.diff);
            d = ln2.diff;
        }
        attn.back(d);
        ln1.back(attn.diff);
        Tensor_OP().add(ln1.diff, d, ln1.diff);
        //[b, h*w, c] --> [b, c, h, w]
        ln1.diff.view(number, time, 1, channel);
        tmp.view(number, channel, 1, time);
        Tensor_OP().permute(ln1.diff, tmp, new int[]{0, 3, 2, 1});
        tmp.view(number, channel, height, width);
        //		xt.showDM("xt");
        conv_in.back(tmp);
        //		conv_in.diff.showDM("conv_in");
        gn.back(conv_in.diff);
        Tensor_OP().add(gn.diff, delta, gn.diff);
        this.diff = gn.diff;
        ln1.diff.viewOrg();
        //		xt.viewOrg();
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

    public void forward(Tensor input, Tensor context) {
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
        this.output(context);
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
        gn.update();
        conv_in.update();
        ln1.update();
        attn.update();
        if (cross_attn != null) {
            ln2.update();
            cross_attn.update();
        }
        ln3.update();
        geglu1.update();
        geglu2.update();
        conv_out.update();
    }

    @Override
    public void showDiff() {
        // TODO Auto-generated method stub
    }

    @Override
    public LayerType getLayerType() {
        // TODO Auto-generated method stub
        return LayerType.unet_down;
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
    public void accGrad(float scale) {
        // TODO Auto-generated method stub
        gn.accGrad(scale);
        conv_in.accGrad(scale);
        ln1.accGrad(scale);
        attn.accGrad(scale);
        ln2.accGrad(scale);
        cross_attn.accGrad(scale);
        ln3.accGrad(scale);
        geglu1.accGrad(scale);
        geglu2.accGrad(scale);
        conv_out.accGrad(scale);
    }

    public void saveModel(RandomAccessFile outputStream) throws IOException {
        gn.saveModel(outputStream);
        conv_in.saveModel(outputStream);
        ln1.saveModel(outputStream);
        attn.saveModel(outputStream);
        if (cross_attn != null) {
            ln2.saveModel(outputStream);
            cross_attn.saveModel(outputStream);
        }
        ln3.saveModel(outputStream);
        geglu1.saveModel(outputStream);
        geglu2.saveModel(outputStream);
        conv_out.saveModel(outputStream);
    }

    public void loadModel(RandomAccessFile inputStream) throws IOException {
        gn.loadModel(inputStream);
        conv_in.loadModel(inputStream);
        ln1.loadModel(inputStream);
        attn.loadModel(inputStream);
        if (cross_attn != null) {
            ln2.loadModel(inputStream);
            cross_attn.loadModel(inputStream);
        }
        ln3.loadModel(inputStream);
        geglu1.loadModel(inputStream);
        geglu2.loadModel(inputStream);
        conv_out.loadModel(inputStream);
    }
}

