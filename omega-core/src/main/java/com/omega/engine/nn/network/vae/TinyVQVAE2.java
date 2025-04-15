package com.omega.engine.nn.network.vae;

import com.omega.common.tensor.Tensor;
import com.omega.utils.MatrixOperation;
import com.omega.utils.RandomUtils;
import com.omega.engine.gpu.GPUOP;
import com.omega.engine.loss.LossFactory;
import com.omega.engine.loss.LossType;
import com.omega.engine.nn.layer.*;
import com.omega.engine.nn.layer.vqvae.tiny.TinyVQVAEDecoder;
import com.omega.engine.nn.layer.vqvae.tiny.TinyVQVAEEncoder;
import com.omega.engine.nn.network.Network;
import com.omega.engine.nn.network.NetworkType;
import com.omega.engine.nn.network.RunModel;
import com.omega.engine.updater.UpdaterFactory;
import com.omega.engine.updater.UpdaterType;
import jcuda.jcublas.cublasOperation;

import java.io.IOException;
import java.io.RandomAccessFile;

/**
 * TinyVAE
 *
 * @author Administrator
 */
public class TinyVQVAE2 extends Network {
    public float beta = 0.25f;
    public float decay = 0.999f;
    public int num_res_blocks;
    public int num_vq_embeddings;
    public int z_dims;
    public int latendDim = 4;
    public int imageSize;
    public TinyVQVAEEncoder encoder;
    public TinyVQVAEDecoder decoder;
    public ConvolutionLayer pre_quant_conv;
    public EmbeddingIDLayer embedding;
    public ConvolutionLayer post_quant_conv;
    public Tensor vqLoss;
    private int groups = 32;
    private int headNum = 4;
    private int[] channels;
    private boolean[] attn_resolutions;
    private InputLayer inputLayer;
    private Tensor zq;
    private Tensor ze;
    private Tensor z_flattened;
    private Tensor idx;
    private Tensor dzqT;
    private Tensor dze;
    private Tensor dzeT;
    private VAEKernel vaeKernel;
    private Tensor zc;
    private Tensor ec;
    private Tensor ie;
    private Tensor sum_encodings;
    private Tensor ema_count;
    private Tensor onehot;
    private Tensor ema_weight;
    private Tensor dw;
    private Tensor ema_count_n;
    private Tensor avg_probs;
    private Tensor avg_probs_log;

    public TinyVQVAE2(LossType lossType, UpdaterType updater, int z_dims, int latendDim, int num_vq_embeddings, int imageSize, int[] channels, boolean[] attn_resolutions, int num_res_blocks) {
        this.lossFunction = LossFactory.create(lossType, this);
        this.z_dims = z_dims;
        this.latendDim = latendDim;
        this.num_vq_embeddings = num_vq_embeddings;
        this.imageSize = imageSize;
        this.channels = channels;
        this.num_res_blocks = num_res_blocks;
        this.attn_resolutions = attn_resolutions;
        this.updater = updater;
        initLayers();
    }

    public void initLayers() {
        this.inputLayer = new InputLayer(3, imageSize, imageSize);
        this.encoder = new TinyVQVAEEncoder(3, z_dims, imageSize, imageSize, num_res_blocks, groups, headNum, channels, attn_resolutions, this);
        pre_quant_conv = new ConvolutionLayer(z_dims, latendDim, encoder.oWidth, encoder.oHeight, 1, 1, 0, 1, true, this);
        pre_quant_conv.setUpdater(UpdaterFactory.create(this));
        pre_quant_conv.paramsInit = ParamsInit.silu;
        embedding = new EmbeddingIDLayer(num_vq_embeddings, latendDim, true, this);
        float initrange = 1.0f / num_vq_embeddings;
        embedding.weight = new Tensor(1, 1, num_vq_embeddings, latendDim, RandomUtils.uniform(num_vq_embeddings * latendDim, -initrange, initrange), true);
        post_quant_conv = new ConvolutionLayer(latendDim, z_dims, encoder.oWidth, encoder.oHeight, 1, 1, 0, 1, true, this);
        post_quant_conv.setUpdater(UpdaterFactory.create(this));
        post_quant_conv.paramsInit = ParamsInit.silu;
        this.decoder = new TinyVQVAEDecoder(z_dims, 3, encoder.oHeight, encoder.oWidth, num_res_blocks, groups, headNum, channels, attn_resolutions, this);
        this.addLayer(inputLayer);
        this.addLayer(encoder);
        this.addLayer(pre_quant_conv);
        this.addLayer(embedding);
        this.addLayer(post_quant_conv);
        this.addLayer(decoder);
        vaeKernel = new VAEKernel(cudaManager);
    }

    @Override
    public void init() throws Exception {
        // TODO Auto-generated method stub
        if (layerList.size() <= 0) {
            throw new Exception("layer size must greater than 2.");
        }
        this.layerCount = layerList.size();
        this.setChannel(layerList.get(0).channel);
        this.setHeight(layerList.get(0).height);
        this.setWidth(layerList.get(0).width);
        this.oChannel = this.getLastLayer().oChannel;
        this.oHeight = this.getLastLayer().oHeight;
        this.oWidth = this.getLastLayer().oWidth;
        if (layerList.get(0).getLayerType() != LayerType.input) {
            throw new Exception("first layer must be input layer.");
        }
        if ((layerList.get(layerList.size() - 1).getLayerType() == LayerType.softmax || layerList.get(layerList.size() - 1).getLayerType() == LayerType.softmax_cross_entropy) && this.lossFunction.getLossType() != LossType.cross_entropy) {
            throw new Exception("The softmax function support only cross entropy loss function now.");
        }
        System.out.println("the network is ready.");
    }

    @Override
    public NetworkType getNetworkType() {
        // TODO Auto-generated method stub
        return NetworkType.VQVAE;
    }

    @Override
    public Tensor predict(Tensor input) {
        // TODO Auto-generated method stub
        this.RUN_MODEL = RunModel.TEST;
        this.forward(input);
        return this.getOutput();
    }

    @Override
    public Tensor forward(Tensor input) {
        /**
         * 设置输入数据

         */
        this.setInputData(input);
        //		input.showDMByOffset(50 * 256, 256);
        inputLayer.forward();
        encoder.forward(input);
        //		encoder.getOutput().showDMByOffset(0, 10, "encoder");
        pre_quant_conv.forward(encoder.getOutput());
        this.ze = pre_quant_conv.getOutput();
        //		this.ze.showDMByOffset(0, 10, "ze");
        quantizer(pre_quant_conv.getOutput());
        post_quant_conv.forward(this.zq);
        decoder.forward(post_quant_conv.getOutput());
        //		System.err.println("in==========>out:");
        //		this.decoder.getOutput().showDMByOffset(0, 256);
        return this.getOutput();
    }

    public Tensor encode(Tensor input) {
        /**
         * 设置输入数据

         */
        this.setInputData(input);
        inputLayer.forward();
        encoder.forward(input);
        pre_quant_conv.forward(encoder.getOutput());
        quantizer(pre_quant_conv.getOutput());
        return zq;
    }

    public Tensor decode(Tensor latent) {
        if (this.input == null) {
            this.setInputData(latent);
        }
        post_quant_conv.forward(latent);
        decoder.forward(post_quant_conv.getOutput());
        return decoder.getOutput();
    }

    public Tensor decodeCode(Tensor codeB) {
        quantizer(codeB);
        return decode(this.zq);
    }

    public void quantizer(Tensor ze) {
        if (this.z_flattened == null || this.z_flattened.number != ze.number) {
            this.z_flattened = Tensor.createGPUTensor(this.z_flattened, ze.number, ze.height, ze.width, this.latendDim, true);
            this.idx = Tensor.createGPUTensor(this.idx, ze.number * ze.height * ze.width, 1, 1, 1, true);
            this.zq = Tensor.createGPUTensor(this.zq, ze.number, ze.channel, ze.height, ze.width, true);
            if (this.RUN_MODEL == RunModel.TRAIN) {
                this.avg_probs = Tensor.createGPUTensor(this.avg_probs, 1, 1, 1, num_vq_embeddings, true);
                this.avg_probs_log = Tensor.createGPUTensor(this.avg_probs_log, 1, 1, 1, num_vq_embeddings, true);
            }
        } else {
            z_flattened.viewOrg();
            if (this.RUN_MODEL == RunModel.TRAIN) {
                avg_probs.clear();
            }
        }
        tensorOP.permute(ze, z_flattened, new int[]{0, 2, 3, 1});  //B,C,H,W ==> B,H,W,C
        z_flattened = z_flattened.view(ze.number * ze.height * ze.width, 1, 1, this.latendDim);
        cdist();
        //		System.err.println("ie:");
        //		ie.showDMByOffset(0, num_vq_embeddings);
        vaeKernel.argmin(ie, idx);
        if (embedding.getOutput() != null) {
            embedding.getOutput().viewOrg();
        }
        //		embedding.weight.showDM();
        embedding.forward(idx);
        Tensor emo = embedding.getOutput().view(new int[]{ze.number, ze.height, ze.width, ze.channel});
        tensorOP.permute(emo, zq, new int[]{0, 3, 1, 2}); //B*H*W*C ==> B*C*H*W
        //		if(this.RUN_MODEL == RunModel.TRAIN) {
        //			vaeKernel.mean(idx, avg_probs);
        //			TensorOP.add(avg_probs, 1e-6f, avg_probs_log);
        //			TensorOP.log(avg_probs_log, avg_probs_log);
        //			TensorOP.mul(avg_probs, avg_probs_log, avg_probs);
        //			float perplexity =  (float) Math.exp(-MatrixUtils.sum(avg_probs.syncHost()));
        //			System.out.println("perplexity:"+perplexity);
        //		}
    }

    /**
     * sqrt(sum(pow(x - y))) ==> sum(x^2) + sum(y^2) - 2 * x * y
     *
     * @return
     */
    public void cdist() {
        if (zc == null || z_flattened.number != zc.number) {
            zc = Tensor.createGPUTensor(this.zc, z_flattened.number, 1, 1, 1, true);
            ec = Tensor.createGPUTensor(this.ec, num_vq_embeddings, 1, 1, 1, true);
            ie = Tensor.createGPUTensor(this.ie, z_flattened.number, 1, 1, num_vq_embeddings, true);
        } else {
            ie.clearGPU();
            zc.clearGPU();
            ec.clearGPU();
        }
        //		long start1 = System.nanoTime();
        tensorOP.sum_pow(z_flattened, zc, 2, 1);
        tensorOP.sum_pow(embedding.weight.view(num_vq_embeddings, 1, 1, latendDim), ec, 2, 1);
        tensorOP.broadcast(zc, ie, 1);
        tensorOP.broadcast_row(ec, ie);
        GPUOP.getInstance().multiplyFloat(z_flattened.number, embedding.weight.number, embedding.weight.width, z_flattened.getGpuData(), embedding.weight.getGpuData(), ie.getGpuData(), cublasOperation.CUBLAS_OP_N, cublasOperation.CUBLAS_OP_T, -2.0f, 1.0f);
        embedding.weight.viewOrg();
    }

    public Tensor quantizer_back(Tensor delta) {
        dzqT.viewOrg();
        //		TensorOP.permute(delta, dzqT, new int[] {0, 2, 3, 1}); // B,C,H,W ==> B,H,W,C
        // dzq , dze
        vaeKernel.MSE_BACK(embedding.getOutput(), z_flattened, dzqT, dzeT, beta);
        //		vaeKernel.MSE_SUM_C_BACK(embedding.getOutput(), z_flattened, dzeT, beta);
        //		dzqT.view(dzqT.number * dzqT.height * dzqT.width, 1, 1, latendDim);
        //		dzqT.showDM();
        //		embedding.weight.showDM();
        //		embedding.back(dzqT);
        //		System.err.println("emb-DW:");
        //		embedding.diffW.showDM();
        ema();
        tensorOP.permute(dzeT, dze, new int[]{0, 3, 1, 2});  //B,H,W,C ==》 B,C,H,W
        tensorOP.add(dze, delta, dze);
        //		System.err.println("dze:");
        //		dze.showDMByOffset(0, 32);
        return dze;
    }

    public void ema() {
        if (ema_count == null) {
            sum_encodings = Tensor.createGPUTensor(sum_encodings, 1, 1, 1, num_vq_embeddings, true);
            ema_count = Tensor.createGPUTensor(ema_count, 1, 1, 1, num_vq_embeddings, true);
            ema_count_n = Tensor.createGPUTensor(ema_count_n, 1, 1, 1, 1, true);
            onehot = Tensor.createGPUTensor(onehot, z_flattened.number, 1, 1, num_vq_embeddings, true);
            ema_weight = Tensor.createGPUTensor(ema_weight, 1, 1, num_vq_embeddings, latendDim, true);
            vaeKernel.copy_gpu(embedding.weight, this.ema_weight, ema_weight.getDataLength(), 1, 1);
            dw = Tensor.createGPUTensor(dw, 1, 1, num_vq_embeddings, latendDim, true);
        } else {
            sum_encodings.clearGPU();
            onehot.clearGPU();
            ema_count_n.clearGPU();
        }
        tensorOP.onehot(idx, onehot);
        vaeKernel.ema_count(idx, sum_encodings);
        vaeKernel.move_ema_count(sum_encodings, ema_count, decay);
        tensorOP.sum(ema_count, ema_count_n, 0);
        vaeKernel.move_ema_count2(ema_count, ema_count_n, 1e-5f, num_vq_embeddings);
        GPUOP.getInstance().multiplyFloat(onehot.width, z_flattened.width, onehot.number, onehot.getGpuData(), z_flattened.getGpuData(), dw.getGpuData(), cublasOperation.CUBLAS_OP_T, cublasOperation.CUBLAS_OP_N, 1.0f, 0.0f);
        vaeKernel.update_emb_weight(dw, embedding.weight, ema_weight, ema_count, decay);
    }

    //	public static void main(String args[]) {
    //
    //		int N = 3;
    //		int D = 4;
    //		int num_embeddings = 3;
    //
    //		float decay = 0.999f;
    //		float epsilon = 1e-5f;
    //
    //		Tensor sum_encodings = new Tensor(1, 1, 1, num_embeddings, true);
    //		Tensor ema_count = new Tensor(1, 1, 1, num_embeddings, true);
    //		Tensor ema_count_n = new Tensor(1, 1, 1, 1, true);
    //		Tensor onehot = new Tensor(N, 1, 1, num_embeddings, true);
    //
    //		Tensor ema_weight = new Tensor(1, 1, num_embeddings, D, MatrixUtils.order(N * D, 0.1f, 0.1f), true);
    //
    //		Tensor weight = new Tensor(1, 1, num_embeddings, D, true);
    //
    //		Tensor dw = new Tensor(1, 1, num_embeddings, D, true);
    //
    //		Tensor z_flattened = new Tensor(N, 1, 1, D, MatrixUtils.order(N * D, 0.1f, 0.1f), true);
    //
    //		float[] t = new float[] {1, 2, 0};
    //
    //		Tensor idx = new Tensor(N, 1, 1, 1, t, true);
    //
    //		VAEKernel vaeKernel = new VAEKernel();
    //
    //		TensorOP.onehot(idx, onehot);
    //
    //		vaeKernel.ema_count(idx, sum_encodings);
    //
    //		sum_encodings.showDM();
    //
    //		vaeKernel.move_ema_count(sum_encodings, ema_count, decay);
    //
    //		TensorOP.sum(ema_count, ema_count_n, 0);
    //
    //		ema_count_n.showDM();
    //
    //		vaeKernel.move_ema_count2(ema_count, ema_count_n, epsilon, num_embeddings);
    //
    //		ema_count.showDM();
    //
    //		onehot.showDM();
    //		z_flattened.showDM();
    //
    //		GPUOP.getInstance().multiplyFloat(onehot.width, z_flattened.width, onehot.number, onehot.getGpuData(), z_flattened.getGpuData(), dw.getGpuData(),
    //				cublasOperation.CUBLAS_OP_T, cublasOperation.CUBLAS_OP_N, 1.0f, 0.0f);
    //
    //		dw.showDM();
    //
    //		vaeKernel.update_emb_weight(dw, weight, ema_weight, ema_count, decay);
    //
    //		weight.showDM();
    //
    //	}
    public void initBack() {
        if (this.dzqT == null || this.dzqT.number != zq.number) {
            this.dzqT = Tensor.createGPUTensor(this.dzqT, zq.number, zq.height, zq.width, zq.channel, true);
            this.dze = Tensor.createGPUTensor(this.dze, ze.number, ze.channel, ze.height, ze.width, true);
            this.dzeT = Tensor.createGPUTensor(this.dzeT, zq.number, zq.height, zq.width, zq.channel, true);
        }
    }

    @Override
    public void back(Tensor lossDiff) {
        // TODO Auto-generated method stub
        //		lossDiff.showDMByNumber(0);
        /**
         * 设置误差
         * 将误差值输入到最后一层

         */
        this.setLossDiff(lossDiff);  //only decoder delta
        initBack();
        //		lossDiff.showDMByOffset(0, 256);
        // dz
        this.decoder.back(lossDiff);
        //		decoder.diff.showDMByOffset(0, 32);
        post_quant_conv.back(decoder.diff);
        Tensor encoderDelta = quantizer_back(post_quant_conv.diff);
        pre_quant_conv.back(encoderDelta);
        //		System.err.println("pre_quant_conv-diff:");
        //		pre_quant_conv.diff.showDMByOffset(0, 32);;
        this.encoder.back(pre_quant_conv.diff);
    }

    @Override
    public Tensor loss(Tensor output, Tensor label) {
        // TODO Auto-generated method stub
        return this.lossFunction.loss(output, label);
    }

    public float totalLoss(Tensor output, Tensor label) {
        if (vqLoss == null) {
            this.vqLoss = Tensor.createTensor(this.vqLoss, 1, 1, 1, 1, true);
        }
        //		output.showDMByOffset(0, 10, "out");
        Tensor decoerLoss = this.lossFunction.loss(output, label);
        System.out.println("decoderLoss:" + MatrixOperation.sum(decoerLoss.syncHost()) / output.number);
        embedding.getOutput().viewOrg();
        //		embedding.getOutput().showDMByOffset(0, 10, "embedding");
        //		z_flattened.showDMByOffset(0, 10, "z_flattened");
        vaeKernel.MSE_C(embedding.getOutput(), z_flattened, vqLoss, beta);
        //		vaeKernel.MSE_C_SUM(embedding.getOutput(), z_flattened, vqLoss, beta);
        vqLoss.showDM(0, "vqLoss");
        return (MatrixOperation.sum(decoerLoss.syncHost()) / output.number + MatrixOperation.sum(vqLoss.syncHost()));
    }

    @Override
    public Tensor lossDiff(Tensor output, Tensor label) {
        // TODO Auto-generated method stub
        Tensor t = this.lossFunction.diff(output, label);
        return t;
    }

    @Override
    public void clearGrad() {
        // TODO Auto-generated method stub
    }

    @Override
    public Tensor loss(Tensor output, Tensor label, Tensor loss) {
        // TODO Auto-generated method stub
        return this.lossFunction.loss(output, label, loss);
    }

    @Override
    public Tensor lossDiff(Tensor output, Tensor label, Tensor diff) {
        // TODO Auto-generated method stub
        return this.lossFunction.diff(output, label, diff);
    }

    public Tensor loss(Tensor output, Tensor label, int igonre) {
        // TODO Auto-generated method stub
        return this.lossFunction.loss(output, label, igonre);
    }

    public Tensor lossDiff(Tensor output, Tensor label, int igonre) {
        // TODO Auto-generated method stub
        return this.lossFunction.diff(output, label, igonre);
    }

    public void saveModel(RandomAccessFile outputStream) throws IOException {
        encoder.saveModel(outputStream);
        pre_quant_conv.saveModel(outputStream);
        embedding.saveModel(outputStream);
        post_quant_conv.saveModel(outputStream);
        decoder.saveModel(outputStream);
    }

    public void loadModel(RandomAccessFile inputStream) throws IOException {
        encoder.loadModel(inputStream);
        pre_quant_conv.loadModel(inputStream);
        embedding.loadModel(inputStream);
        post_quant_conv.loadModel(inputStream);
        decoder.loadModel(inputStream);
    }

    @Override
    public void putParamters() {
        // TODO Auto-generated method stub
    }

    @Override
    public void putParamterGrads() {
        // TODO Auto-generated method stub
    }
}

