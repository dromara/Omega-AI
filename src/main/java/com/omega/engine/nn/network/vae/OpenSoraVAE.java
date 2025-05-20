package com.omega.engine.nn.network.vae;

import java.io.IOException;
import java.io.RandomAccessFile;

import com.omega.common.utils.MatrixOperation;
import com.omega.common.utils.MatrixUtils;
import com.omega.engine.loss.LossFactory;
import com.omega.engine.loss.LossType;
import com.omega.engine.nn.layer.InputLayer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.opensora.vae.VideoDecoder;
import com.omega.engine.nn.layer.opensora.vae.VideoEncoder;
import com.omega.engine.nn.network.Network;
import com.omega.engine.nn.network.NetworkType;
import com.omega.engine.nn.network.RunModel;
import com.omega.engine.nn.network.vqgan.Opensora_LPIPS;
import com.omega.engine.tensor.Tensor;
import com.omega.engine.updater.UpdaterType;

/**
 * OpenSoraVAE
 *
 * @author Administrator
 */
public class OpenSoraVAE extends Network {
    public float beta = 0.25f;
    public float decay = 0.999f;
    public int num_res_blocks;
    public int latendDim = 4;
    public int depth;
    public int imageSize;
    
    public int latendDepth;
    public int latendHeight;
    public int latendWidth;
    
    public Tensor vqLoss;

    private int[] ch_mult;
    private int ch;
    
    private int[] down_sampling_layer = new int[] {1, 2};
    private int[] temporal_up_layer = new int[] {2, 3};
    private int temporal_downsample = 4;
    
    
    private InputLayer inputLayer;
    public VideoEncoder encoder;
    public VideoDecoder decoder;

    private Tensor z;
    private Tensor eps;
    private Tensor mu;
    private Tensor logvar;
    private Tensor dmu;
    private Tensor dlogvar;
    
    private Tensor video;
    private Tensor rec_video;
    
    private Tensor recLoss;
    private Tensor lpipLoss;
    
    private Tensor lpipsLossDiff;
    public Tensor encoderDelta;
    
    public float kl_weight = 1e-6f;
    public Tensor klLoss;
    private VAEKernel vaeKernel;
    
    private Opensora_LPIPS lpips;

    public OpenSoraVAE(LossType lossType, UpdaterType updater, int latendDim,int depth, int imageSize, int[] ch_mult, int ch, int num_res_blocks) {
        this.lossFunction = LossFactory.create(lossType, this);
        this.latendDim = latendDim;
        this.depth = depth;
        this.imageSize = imageSize;
        this.ch_mult = ch_mult;
        this.num_res_blocks = num_res_blocks;
        this.ch = ch;
        this.updater = updater;
        initLayers();
    }

    public void initLayers() {
        this.inputLayer = new InputLayer(3 * depth, imageSize, imageSize);
        this.encoder = new VideoEncoder(3, latendDim, depth, imageSize, imageSize, ch, num_res_blocks, ch_mult, down_sampling_layer, true, this);
        this.decoder = new VideoDecoder(latendDim, 3, encoder.oDepth, encoder.oHeight, encoder.oWidth, ch, num_res_blocks, ch_mult, temporal_up_layer, temporal_downsample, this);
        this.addLayer(inputLayer);
        this.addLayer(encoder);
        this.addLayer(decoder);
        vaeKernel = new VAEKernel(cudaManager);
        latendDepth = encoder.oDepth;
        latendHeight = encoder.oHeight;
        latendWidth = encoder.oWidth;
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
        return NetworkType.ORVAE;
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
        inputLayer.forward(input);
        
        encoder.forward(input);

        reparameterize(encoder.getOutput());

        decoder.forward(z);

        return this.getOutput();
    }
    
    public Tensor forward(Tensor input,Tensor z) {
        /**
         * 设置输入数据
         */
        this.setInputData(input);

        //		input.showDMByOffset(50 * 256, 256);
        inputLayer.forward(input);
        
        encoder.forward(input);

        reparameterize(encoder.getOutput());

        decoder.forward(z);

        return this.getOutput();
    }

    public Tensor encode(Tensor input) {
        /**
         * 设置输入数据
         */
        this.setInputData(input);
        inputLayer.forward();
        encoder.forward(input);
        reparameterize(encoder.getOutput());
        return z;
    }

    public Tensor decode(Tensor latent) {
        this.setInputData(latent);
        decoder.forward(latent);
        return decoder.getOutput();
    }

    public void reparameterize(Tensor encode) {
        if (this.z == null || this.z.number != encode.number) {
            this.z = Tensor.createGPUTensor(this.z, encode.number, this.latendDim * latendDepth, encode.height, encode.width, true);
            this.eps = Tensor.createGPUTensor(this.eps, encode.number, this.latendDim * latendDepth, encode.height, encode.width, true);
            this.mu = Tensor.createGPUTensor(this.mu, encode.number, this.latendDim * latendDepth, encode.height, encode.width, true);
            this.logvar = Tensor.createGPUTensor(this.logvar, encode.number, this.latendDim * latendDepth, encode.height, encode.width, true);
        }
//        GPUOP.getInstance().cudaRandn(this.eps);
        this.eps.fill(1, this.tensorOP.op);
        vaeKernel.concat_channel_backward(encode, mu, logvar, encode.number, this.latendDim, this.latendDim, latendDepth, encode.height * encode.width);
        vaeKernel.forward(mu, logvar, eps, z);
        z.showDM("z-fffff");
    }

    public void reparameterize_back(Tensor delta) {
        vaeKernel.backward(delta, eps, logvar, dmu, dlogvar);
        vaeKernel.concat_channel_forward(dmu, dlogvar, encoderDelta, dmu.number, this.latendDim, this.latendDim, latendDepth * dmu.height, dmu.width);
    }

    public void initBack() {
        if (this.dlogvar == null || this.dlogvar.number != logvar.number) {
        	lpipsLossDiff = new Tensor(recLoss.number, 1, 1, 1, MatrixUtils.val(recLoss.number, 1.0f / recLoss.number * recLoss.channel * recLoss.height * recLoss.width), true);
            this.dlogvar = Tensor.createGPUTensor(this.dlogvar, logvar.number, this.latendDim * latendDepth, logvar.height, logvar.width, true);
            this.dmu = Tensor.createGPUTensor(this.dmu, mu.number, this.latendDim * latendDepth, mu.height, mu.width, true);
            this.encoderDelta = Tensor.createGPUTensor(this.encoderDelta, mu.number, this.latendDim * 2 * latendDepth, mu.height, mu.width, true);
        } else {
            dmu.clearGPU();
            dlogvar.clearGPU();
        }
    }

    @Override
    public void back(Tensor lossDiff) {
        // TODO Auto-generated method stub
    	/**
         * 设置误差
         * 将误差值输入到最后一层
         */
        this.setLossDiff(lossDiff);  //only decoder delta
        initBack();
//        lpipsLossDiff.showDM("lpipsLossDelta");
        lpips.back(lpipsLossDiff);
        int last = 17 * lpips.lpips.diff.getOnceSize() + 2 * 32 * 32 + 31 * 32;
        lpips.lpips.diff.showDMByOffsetRed(last, 32, "lpipsLossDiff:");
        lpips.lpips.diff.showDM("lpipsLossDiff");
        
        tensorOP.sub(rec_video, video, recLoss);
        rec_video.fill(1.0f/video.number, tensorOP.op);
        tensorOP.abs_backward(recLoss, rec_video, rec_video);
        
        tensorOP.add(rec_video, lpips.lpips.diff, rec_video);
        rec_video.showDM("recon_video-diff");
        //output.number, decoder.oDepth, decoder.oChannel, output.height * output.width
        recLoss.view(decoder.number, decoder.oChannel, decoder.oDepth, decoder.oHeight * decoder.oWidth);
        rec_video.viewOrg();
        tensorOP.permute(rec_video, recLoss, new int[] {0, 2, 1, 3});
        recLoss.view(decoder.number, decoder.oChannel * decoder.oDepth, decoder.oHeight, decoder.oWidth);
        // dmu , dlogvar
        vaeKernel.kl_back(mu, logvar, kl_weight, dmu, dlogvar);
        
        this.decoder.back(recLoss);
        reparameterize_back(decoder.diff);
        this.encoder.back(encoderDelta);
    }
    
    @Override
    public Tensor loss(Tensor output, Tensor label) {
        // TODO Auto-generated method stub
        return this.lossFunction.loss(output, label);
    }

    public float totalLoss(Tensor output, Tensor label, Opensora_LPIPS lpips) {
        if (klLoss == null || klLoss.number != mu.number) {
        	this.lpips = lpips;
        	this.rec_video = Tensor.createTensor(this.rec_video, output.number, decoder.oDepth, decoder.oChannel, output.height * output.width, true);
        	this.video = Tensor.createTensor(this.video, output.number, decoder.oDepth, decoder.oChannel, output.height * output.width, true);
            this.klLoss = Tensor.createTensor(this.klLoss, mu.number, mu.channel, mu.height, mu.width, true);
            this.lpipLoss = Tensor.createTensor(lpipLoss, 1, 1, 1, 1, true);
            this.recLoss = Tensor.createTensor(this.recLoss, video.number * video.channel, video.height, output.height, output.width, true);//(b t) c h w
        }
        
        label.view(output.number, decoder.oChannel, decoder.oDepth, output.height * output.width);
        output.view(output.number, decoder.oChannel, decoder.oDepth, output.height * output.width);
        tensorOP.permute(label, video, new int[] {0, 2, 1, 3});
        tensorOP.permute(output, rec_video, new int[] {0, 2, 1, 3});
        label.viewOrg();
        output.viewOrg();
        video.view(video.number * video.channel, video.height, output.height, output.width); //(b t) c h w
        rec_video.view(rec_video.number * rec_video.channel, rec_video.height, output.height, output.width); //(b t) c h w
        rec_video.showDM("rec_video");
        /**
         * reconstruction loss
         */
        tensorOP.sub(rec_video, video, recLoss);
        tensorOP.abs(recLoss, recLoss);
        /**
         * perceptual_loss
         */
        Tensor lpipsOutput = lpips.forward(rec_video, video);

        /**
         * current time error
         */
        tensorOP.mean(lpipsOutput, 0, lpipLoss);
        
        float nll_loss = MatrixOperation.sum(recLoss.syncHost())/video.number + lpipLoss.syncHost()[0];
        System.out.println("nll_loss:"+nll_loss);
        
        vaeKernel.kl(mu, logvar, kl_weight, klLoss);
        float klLossf = MatrixOperation.sum(klLoss.syncHost()) / mu.number;
        System.out.println("klLossf:"+klLossf);
        return nll_loss + klLossf;
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
//        encoder.saveModel(outputStream);
//        decoder.saveModel(outputStream);
    }

    public void loadModel(RandomAccessFile inputStream) throws IOException {
//        encoder.loadModel(inputStream);
//        decoder.loadModel(inputStream);
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

