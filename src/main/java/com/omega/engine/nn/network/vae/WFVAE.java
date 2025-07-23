package com.omega.engine.nn.network.vae;

import java.io.IOException;
import java.io.RandomAccessFile;

import com.omega.common.utils.RandomUtils;
import com.omega.engine.loss.LossFactory;
import com.omega.engine.loss.LossType;
import com.omega.engine.nn.layer.InputLayer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.opensora.wfvae.decoder.WFDecoder;
import com.omega.engine.nn.layer.opensora.wfvae.encoder.WFEncoder;
import com.omega.engine.nn.network.Network;
import com.omega.engine.nn.network.NetworkType;
import com.omega.engine.nn.network.RunModel;
import com.omega.engine.tensor.Tensor;
import com.omega.engine.updater.UpdaterType;

/**
 * WFVAE
 *
 * @author Administrator
 */
public class WFVAE extends Network {
    public float beta = 0.25f;
    public float decay = 0.999f;
    public int num_res_blocks;
    public int energy_flow_hidden_size;
    public int latendDim = 4;
    public int depth;;
    public int imageSize;
    public WFEncoder encoder;
    public WFDecoder decoder;

    public Tensor vqLoss;
    private int groups = 32;
    private int headNum = 4;
    private int base_channels;
    private InputLayer inputLayer;
    
    private Tensor r_z;
    private Tensor z;
    private Tensor mean;
    private Tensor logvar;

    
    private VAEKernel vaeKernel;

    public WFVAE(LossType lossType, UpdaterType updater, int depth, int latendDim, int imageSize, int base_channels, int energy_flow_hidden_size, int num_res_blocks) {
        this.lossFunction = LossFactory.create(lossType, this);
        this.latendDim = latendDim;
        this.energy_flow_hidden_size = energy_flow_hidden_size;
        this.imageSize = imageSize;
        this.num_res_blocks = num_res_blocks;
        this.base_channels = base_channels;
        this.depth = depth;
        this.updater = updater;
        initLayers();
    }

    public void initLayers() {
        this.inputLayer = new InputLayer(3, imageSize, imageSize);
        this.encoder = new WFEncoder(3, depth, imageSize, imageSize, num_res_blocks, base_channels, energy_flow_hidden_size, latendDim, this);
        this.decoder = new WFDecoder(3, encoder.oDepth, encoder.oHeight, encoder.oWidth, num_res_blocks, base_channels, energy_flow_hidden_size, latendDim, this);
        this.addLayer(inputLayer);
        this.addLayer(encoder);
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
        inputLayer.forward();
        encoder.forward(input);
        posterior(encoder.getOutput());
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
        posterior(encoder.getOutput());
        return z;
    }

    public Tensor decode(Tensor latent) {
        this.setInputData(latent);
        decoder.forward(latent);
        return decoder.getOutput();
    }

    public void posterior(Tensor encoder_out) {
    	sample(encoder_out);
    }
    
    public void sample(Tensor en_out) {
    	
    	if(z == null || z.number != en_out.number) {
    		mean = Tensor.createGPUTensor(mean, en_out.number, encoder.oChannel * latendDim, en_out.height, en_out.width, true);
    		logvar = Tensor.createGPUTensor(logvar, mean.shape(), true);
    		r_z = Tensor.createGPUTensor(r_z, mean.shape(), true);
    		z = Tensor.createGPUTensor(z, mean.shape(), true);
    	}
    	
    	RandomUtils.gaussianRandom(r_z);
    	
    	vaeKernel.forward(mean, logvar, r_z, z);
    	
    }
   
//    public void initBack() {
//        if (this.dzqT == null || this.dzqT.number != zq.number) {
//            this.dzqT = Tensor.createGPUTensor(this.dzqT, zq.number, zq.height, zq.width, zq.channel, true);
//            this.dze = Tensor.createGPUTensor(this.dze, ze.number, ze.channel, ze.height, ze.width, true);
//            this.dzeT = Tensor.createGPUTensor(this.dzeT, zq.number, zq.height, zq.width, zq.channel, true);
//        }
//    }

    @Override
    public void back(Tensor lossDiff) {
        // TODO Auto-generated method stub
        //		lossDiff.showDMByNumber(0);
//        /**
//         * 设置误差
//         * 将误差值输入到最后一层
//         */
//        this.setLossDiff(lossDiff);  //only decoder delta
////        initBack();
//        //		lossDiff.showDMByOffset(0, 256);
//        // dz
//        this.decoder.back(lossDiff);
//        Tensor encoderDelta = quantizer_back(decoder.diff);
//        //		System.err.println("pre_quant_conv-diff:");
//        //		pre_quant_conv.diff.showDMByOffset(0, 32);;
//        this.encoder.back(encoderDelta);
    }

    @Override
    public Tensor loss(Tensor output, Tensor label) {
        // TODO Auto-generated method stub
        return this.lossFunction.loss(output, label);
    }

    public float totalLoss(Tensor output, Tensor label) {
//        if (vqLoss == null) {
//            this.vqLoss = Tensor.createTensor(this.vqLoss, 1, 1, 1, 1, true);
//        }
//        //		output.showDMByOffset(0, 10, "out");
//        Tensor decoerLoss = this.lossFunction.loss(output, label);
//        float decoderLossV = MatrixOperation.sum(decoerLoss.syncHost()) / output.number;
//        System.out.println("decoderLoss:" + decoderLossV);
//        embedding.getOutput().viewOrg();
//        //		embedding.getOutput().showDMByOffset(0, 10, "embedding");
////        		z_flattened.showDMByOffset(0, 10, "z_flattened");
//        vaeKernel.MSE_C(embedding.getOutput(), z_flattened, vqLoss, beta);
//        //		vaeKernel.MSE_C_SUM(embedding.getOutput(), z_flattened, vqLoss, beta);
//        vqLoss.showDM(0, "vqLoss");
//        return (decoderLossV + MatrixOperation.sum(vqLoss.syncHost()));
    	return 0.0f;
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
        decoder.saveModel(outputStream);
    }

    public void loadModel(RandomAccessFile inputStream) throws IOException {
        encoder.loadModel(inputStream);
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

