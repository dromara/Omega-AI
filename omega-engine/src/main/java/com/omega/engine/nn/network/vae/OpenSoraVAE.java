//package com.omega.engine.nn.network.vae;
//
//import com.omega.common.data.Tensor;
//import com.omega.common.utils.MatrixOperation;
//import com.omega.common.utils.RandomUtils;
//import com.omega.engine.gpu.GPUOP;
//import com.omega.engine.loss.LossFactory;
//import com.omega.engine.loss.LossType;
//import com.omega.engine.nn.layer.ConvolutionLayer;
//import com.omega.engine.nn.layer.EmbeddingIDLayer;
//import com.omega.engine.nn.layer.InputLayer;
//import com.omega.engine.nn.layer.LayerType;
//import com.omega.engine.nn.layer.opensora.vae.VideoDecoder;
//import com.omega.engine.nn.layer.opensora.vae.VideoEncoder;
//import com.omega.engine.nn.layer.vqvae.tiny.TinyVQVAEDecoder2;
//import com.omega.engine.nn.layer.vqvae.tiny.TinyVQVAEEncoder2;
//import com.omega.engine.nn.network.Network;
//import com.omega.engine.nn.network.NetworkType;
//import com.omega.engine.nn.network.RunModel;
//import com.omega.engine.updater.UpdaterType;
//import jcuda.jcublas.cublasOperation;
//
//import java.io.IOException;
//import java.io.RandomAccessFile;
//
///**
// * OpenSoraVAE
// *
// * @author Administrator
// */
//public class OpenSoraVAE extends Network {
//    public float beta = 0.25f;
//    public float decay = 0.999f;
//    public int num_res_blocks;
//    public int latendDim = 4;
//    public int depth;
//    public int imageSize;
//    
//    public int latendDepth;
//    public int latendHeight;
//    public int latendWidth;
//    
//    public Tensor vqLoss;
//
//    private int[] ch_mult;
//    private int ch;
//    
//    private int[] down_sampling_layer = new int[] {1, 2};
//    private int[] temporal_up_layer = new int[] {2, 3};
//    private int temporal_downsample = 4;
//    
//    
//    private InputLayer inputLayer;
//    public VideoEncoder encoder;
//    public VideoDecoder decoder;
//
//    private Tensor z;
//    private Tensor eps;
//    private Tensor mu;
//    private Tensor logvar;
//    private Tensor dmu;
//    private Tensor dlogvar;
//    
//    public Tensor encoderDelta;
//    
//    private VAEKernel vaeKernel;
//
//    public OpenSoraVAE(LossType lossType, UpdaterType updater, int latendDim,int depth, int imageSize, int[] ch_mult, int ch, int num_res_blocks) {
//        this.lossFunction = LossFactory.create(lossType, this);
//        this.latendDim = latendDim;
//        this.depth = depth;
//        this.imageSize = imageSize;
//        this.ch_mult = ch_mult;
//        this.num_res_blocks = num_res_blocks;
//        this.ch = ch;
//        this.updater = updater;
//        initLayers();
//    }
//
//    public void initLayers() {
//        this.inputLayer = new InputLayer(3 * depth, imageSize, imageSize);
//        this.encoder = new VideoEncoder(3, latendDim, depth, imageSize, imageSize, ch, num_res_blocks, ch_mult, down_sampling_layer, true, this);
//        this.decoder = new VideoDecoder(latendDim, 3, encoder.oDepth, encoder.oHeight, encoder.oWidth, ch, num_res_blocks, ch_mult, temporal_up_layer, temporal_downsample, this);
//        this.addLayer(inputLayer);
//        this.addLayer(encoder);
//        this.addLayer(decoder);
//        vaeKernel = new VAEKernel(cudaManager);
//        latendDepth = encoder.oDepth;
//        latendHeight = encoder.oHeight;
//        latendWidth = encoder.oWidth;
//    }
//
//    @Override
//    public void init() throws Exception {
//        // TODO Auto-generated method stub
//        if (layerList.size() <= 0) {
//            throw new Exception("layer size must greater than 2.");
//        }
//        this.layerCount = layerList.size();
//        this.setChannel(layerList.get(0).channel);
//        this.setHeight(layerList.get(0).height);
//        this.setWidth(layerList.get(0).width);
//        this.oChannel = this.getLastLayer().oChannel;
//        this.oHeight = this.getLastLayer().oHeight;
//        this.oWidth = this.getLastLayer().oWidth;
//        if (layerList.get(0).getLayerType() != LayerType.input) {
//            throw new Exception("first layer must be input layer.");
//        }
//        if ((layerList.get(layerList.size() - 1).getLayerType() == LayerType.softmax || layerList.get(layerList.size() - 1).getLayerType() == LayerType.softmax_cross_entropy) && this.lossFunction.getLossType() != LossType.cross_entropy) {
//            throw new Exception("The softmax function support only cross entropy loss function now.");
//        }
//        System.out.println("the network is ready.");
//    }
//
//    @Override
//    public NetworkType getNetworkType() {
//        // TODO Auto-generated method stub
//        return NetworkType.ORVAE;
//    }
//
//    @Override
//    public Tensor predict(Tensor input) {
//        // TODO Auto-generated method stub
//        this.RUN_MODEL = RunModel.TEST;
//        this.forward(input);
//        return this.getOutput();
//    }
//
//    @Override
//    public Tensor forward(Tensor input) {
//        /**
//         * 设置输入数据
//         */
//        this.setInputData(input);
//        //		input.showDMByOffset(50 * 256, 256);
//        inputLayer.forward(input);
//        
//        encoder.forward(input);
//
//        reparameterize(encoder.getOutput());
//
//        decoder.forward(z);
//
//        return this.getOutput();
//    }
//
//    public Tensor encode(Tensor input) {
//        /**
//         * 设置输入数据
//
//         */
//        this.setInputData(input);
//        inputLayer.forward();
//        encoder.forward(input);
//        reparameterize(encoder.getOutput());
//        return z;
//    }
//
//    public Tensor decode(Tensor latent) {
//        this.setInputData(latent);
//        decoder.forward(latent);
//        return decoder.getOutput();
//    }
//
//    public void reparameterize(Tensor encode) {
//        if (this.z == null || this.z.number != encode.number) {
//            this.z = Tensor.createGPUTensor(this.z, encode.number, this.latendDim * latendDepth, encode.height, encode.width, true);
//            this.eps = Tensor.createGPUTensor(this.eps, encode.number, this.latendDim * latendDepth, encode.height, encode.width, true);
//            this.mu = Tensor.createGPUTensor(this.mu, encode.number, this.latendDim * latendDepth, encode.height, encode.width, true);
//            this.logvar = Tensor.createGPUTensor(this.logvar, encode.number, this.latendDim * latendDepth, encode.height, encode.width, true);
//        }
//        GPUOP.getInstance().cudaRandn(this.eps);
//        vaeKernel.concat_channel_backward(encode, mu, logvar, encode.number, this.latendDim, this.latendDim, latendDepth * encode.height, encode.width);
//        vaeKernel.forward(mu, logvar, eps, z);
//    }
//
//    public void reparameterize_back(Tensor delta) {
//        vaeKernel.backward(delta, eps, logvar, dmu, dlogvar);
//        vaeKernel.concat_channel_forward(dmu, dlogvar, encoderDelta, dmu.number, this.latendDim, this.latendDim, latendDepth * dmu.height, dmu.width);
//    }
//
//    public void initBack() {
//        if (this.dlogvar == null || this.dlogvar.number != logvar.number) {
//            this.dlogvar = Tensor.createGPUTensor(this.dlogvar, logvar.number, this.latendDim * latendDepth, logvar.height, logvar.width, true);
//            this.dmu = Tensor.createGPUTensor(this.dmu, mu.number, this.latendDim * latendDepth, mu.height, mu.width, true);
//            this.encoderDelta = Tensor.createGPUTensor(this.encoderDelta, mu.number, this.latendDim * 2 * latendDepth, mu.height, mu.width, true);
//        } else {
//            dmu.clearGPU();
//            dlogvar.clearGPU();
//        }
//    }
//
//    @Override
//    public void back(Tensor lossDiff) {
//        // TODO Auto-generated method stub
//    	/**
//         * 设置误差
//         * 将误差值输入到最后一层
//         */
//        this.setLossDiff(lossDiff);  //only decoder delta
//        initBack();
//        // dmu , dlogvar
//        vaeKernel.kl_back(mu, logvar, kl_weight, dmu, dlogvar);
//        this.decoder.back(lossDiff);
//        reparameterize_back(decoder.diff);
//        this.encoder.back(encoderDelta);
//    }
//
//    @Override
//    public Tensor loss(Tensor output, Tensor label) {
//        // TODO Auto-generated method stub
//        return this.lossFunction.loss(output, label);
//    }
//
//    public float totalLoss(Tensor output, Tensor label) {
//        if (vqLoss == null) {
//            this.vqLoss = Tensor.createTensor(this.vqLoss, 1, 1, 1, 1, true);
//        }
//        //		output.showDMByOffset(0, 10, "out");
//        Tensor decoerLoss = this.lossFunction.loss(output, label);
//        System.out.println("decoderLoss:" + MatrixOperation.sum(decoerLoss.syncHost()) / output.number);
//        embedding.getOutput().viewOrg();
//        //		embedding.getOutput().showDMByOffset(0, 10, "embedding");
//        //		z_flattened.showDMByOffset(0, 10, "z_flattened");
//        vaeKernel.MSE_C(embedding.getOutput(), z_flattened, vqLoss, beta);
//        //		vaeKernel.MSE_C_SUM(embedding.getOutput(), z_flattened, vqLoss, beta);
//        vqLoss.showDM(0, "vqLoss");
//        return (MatrixOperation.sum(decoerLoss.syncHost()) / output.number + MatrixOperation.sum(vqLoss.syncHost()));
//    }
//
//    @Override
//    public Tensor lossDiff(Tensor output, Tensor label) {
//        // TODO Auto-generated method stub
//        Tensor t = this.lossFunction.diff(output, label);
//        return t;
//    }
//
//    @Override
//    public void clearGrad() {
//        // TODO Auto-generated method stub
//    }
//
//    @Override
//    public Tensor loss(Tensor output, Tensor label, Tensor loss) {
//        // TODO Auto-generated method stub
//        return this.lossFunction.loss(output, label, loss);
//    }
//
//    @Override
//    public Tensor lossDiff(Tensor output, Tensor label, Tensor diff) {
//        // TODO Auto-generated method stub
//        return this.lossFunction.diff(output, label, diff);
//    }
//
//    public Tensor loss(Tensor output, Tensor label, int igonre) {
//        // TODO Auto-generated method stub
//        return this.lossFunction.loss(output, label, igonre);
//    }
//
//    public Tensor lossDiff(Tensor output, Tensor label, int igonre) {
//        // TODO Auto-generated method stub
//        return this.lossFunction.diff(output, label, igonre);
//    }
//
//    public void saveModel(RandomAccessFile outputStream) throws IOException {
//        encoder.saveModel(outputStream);
//        pre_quant_conv.saveModel(outputStream);
//        embedding.saveModel(outputStream);
//        post_quant_conv.saveModel(outputStream);
//        decoder.saveModel(outputStream);
//    }
//
//    public void loadModel(RandomAccessFile inputStream) throws IOException {
//        encoder.loadModel(inputStream);
//        pre_quant_conv.loadModel(inputStream);
//        embedding.loadModel(inputStream);
//        post_quant_conv.loadModel(inputStream);
//        decoder.loadModel(inputStream);
//    }
//
//    @Override
//    public void putParamters() {
//        // TODO Auto-generated method stub
//    }
//
//    @Override
//    public void putParamterGrads() {
//        // TODO Auto-generated method stub
//    }
//}
//
