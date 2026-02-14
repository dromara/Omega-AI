package com.omega.engine.nn.network.dit;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.util.Map;

import com.omega.common.utils.MatrixOperation;
import com.omega.engine.ad.op.gpu.NormalizeKernel;
import com.omega.engine.gpu.CUDAMemoryManager;
import com.omega.engine.loss.LossFactory;
import com.omega.engine.loss.LossType;
import com.omega.engine.loss.gpu.MSELossKernel;
import com.omega.engine.nn.layer.InputLayer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.SoftmaxWithCrossEntropyLayer;
import com.omega.engine.nn.layer.dinovision.DinoVisionTransformer;
import com.omega.engine.nn.network.CNN;
import com.omega.engine.nn.network.Network;
import com.omega.engine.nn.network.NetworkType;
import com.omega.engine.nn.network.RunModel;
import com.omega.engine.tensor.Tensor;
import com.omega.engine.updater.UpdaterType;
import com.omega.example.common.ModeLoaderlUtils;
import com.omega.example.transformer.utils.LagJsonReader;

import jcuda.Sizeof;
import jcuda.runtime.JCuda;

/**
 * Duffsion Transformer
 *
 * @author Administrator
 */
public class Dinov2 extends Network {
	
    public int inChannel;
    public int width;
    public int height;
    public int patchSize;
    public int hiddenSize;
    private int depth;
    public int headNum;
    private int mlpRatio = 4;

    private InputLayer inputLayer;
    public DinoVisionTransformer main;
    
    private Tensor clstokens;
    private Tensor patchtokens;
    
    private NormalizeKernel norm_kernel;
    
    private Tensor n_z;
    private Tensor r_z;
    
    private Tensor loss;
    
    private MSELossKernel mse_kernel;
    
    private Tensor cls_loss;
    private Tensor cls_delta;
    
    private Tensor cfm_loss;
    private Tensor cfm_delta;
    
    private Tensor cfm_cls_loss;
    private Tensor cfm_cls_delta;
    
    public Dinov2(LossType lossType, UpdaterType updater, int inChannel, int width, int height, int patchSize, int hiddenSize, int headNum, int depth, int mlpRatio) {
        this.lossFunction = LossFactory.create(lossType, this);
        this.updater = updater;
        this.inChannel = inChannel;
        this.width = width;
        this.height = height;
        this.patchSize = patchSize;
        this.headNum = headNum;
        this.hiddenSize = hiddenSize;
        this.depth = depth;
        this.mlpRatio = mlpRatio;
        this.time = (width / patchSize) * (height / patchSize);
        initLayers();
    }

    public void initLayers() {
    	
        this.inputLayer = new InputLayer(inChannel, height, width);

        main = new DinoVisionTransformer(inChannel, width, height, patchSize, hiddenSize, headNum, depth, mlpRatio, this);
        
        this.addLayer(inputLayer);
        this.addLayer(main);
        
    }

    @Override
    public void init() throws Exception {
        // TODO Auto-generated method stub
        if (layerList.size() <= 1) {
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
        return NetworkType.DiT;
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
        this.main.forward(input);
        return this.main.getOutput();
    }
    
    public Tensor forward_features(Tensor input) {
    	 /**
         * 设置输入数据
         */
        this.setInputData(input);
        this.main.forward(input);
        if(patchtokens == null) {
        	patchtokens = Tensor.createGPUTensor(patchtokens, input.number, this.main.hw, 1, hiddenSize, true);
        	clstokens = Tensor.createGPUTensor(clstokens, input.number, 1, 1, hiddenSize, true);
        }
        tensorOP.getByChannel(this.main.getOutput(), clstokens, new int[] {input.number, this.main.hw + this.main.num_tokens, 1, hiddenSize}, 0, this.main.num_tokens);
        tensorOP.getByChannel(this.main.getOutput(), patchtokens, new int[] {input.number, this.main.hw + this.main.num_tokens, 1, hiddenSize}, this.main.num_tokens, this.main.hw);
        return patchtokens;
    }
    
    public Tensor forward_features_all(Tensor input) {
   	 /**
        * 设置输入数据
        */
       this.setInputData(input);
       this.main.forward(input);
       if(clstokens == null) {
    	   clstokens = Tensor.createGPUTensor(clstokens, input.number, 1, 1, hiddenSize, true);
       }
       tensorOP.getByChannel(this.main.getOutput(), clstokens, new int[] {input.number, this.main.hw + this.main.num_tokens, 1, hiddenSize}, 0, this.main.num_tokens);
       return this.main.getOutput();
   }
    
    public void initBack() {
    	
    }

    @Override
    public void back(Tensor lossDiff) {
        // TODO Auto-generated method stub
        //		lossDiff.showDMByNumber(0);
        initBack();
        /**
         * 设置误差
         * 将误差值输入到最后一层
         */
        //		lossDiff.showDMByOffset(0, 100, "lossDiff");
        this.setLossDiff(lossDiff);
        //		lossDiff.showDM("lossDiff");
        this.main.back(lossDiff);
        //		this.unet.diff.showDMByOffset(0, 100, "unet.diff");
    }
    
    public void back(Tensor lossDiff,Tensor cos, Tensor sin) {
        // TODO Auto-generated method stub
        //		lossDiff.showDMByNumber(0);
        initBack();
        /**
         * 设置误差
         * 将误差值输入到最后一层
         */
        //		lossDiff.showDMByOffset(0, 100, "lossDiff");
        this.setLossDiff(lossDiff);
        //		lossDiff.showDM("lossDiff");
        this.main.back(lossDiff, cos, sin);
        //		this.unet.diff.showDMByOffset(0, 100, "unet.diff");
    }

    @Override
    public Tensor loss(Tensor output, Tensor label) {
        // TODO Auto-generated method stub
        switch (this.getLastLayer().getLayerType()) {
            case softmax:
                //			SoftmaxLayer softmaxLayer = (SoftmaxLayer)this.getLastLayer();
                //			softmaxLayer.setCurrentLabel(label);
                break;
            case softmax_cross_entropy:
                SoftmaxWithCrossEntropyLayer softmaxWithCrossEntropyLayer = (SoftmaxWithCrossEntropyLayer) this.getLastLayer();
                softmaxWithCrossEntropyLayer.setCurrentLabel(label);
                break;
            default:
                break;
        }
        return this.lossFunction.loss(output, label);
    }

    @Override
    public Tensor lossDiff(Tensor output, Tensor label) {
        // TODO Auto-generated method stub
        this.clearGrad();
        Tensor t = this.lossFunction.diff(output, label);
        //		PrintUtils.printImage(t.data);
        return t;
    }
    
    public void accGrad(int steps) {
        float scale = 1.0f / steps;
        main.accGrad(scale);
    }
    
    public void update() {
        this.train_time += 1;
        this.main.update();
    }

    @Override
    public void clearGrad() {
        // TODO Auto-generated method stub
        /**
         * forward
         */
        JCuda.cudaMemset(CUDAMemoryManager.workspace.getPointer(), 0, CUDAMemoryManager.workspace.getSize() * Sizeof.FLOAT);
        JCuda.cudaDeviceSynchronize();
    }

    @Override
    public Tensor loss(Tensor output, Tensor label, Tensor loss) {
        // TODO Auto-generated method stub
        return this.lossFunction.loss(output, label, loss);
    }

    @Override
    public Tensor lossDiff(Tensor output, Tensor label, Tensor diff) {
        // TODO Auto-generated method stub
        this.clearGrad();
        return this.lossFunction.diff(output, label, diff);
    }

    public Tensor loss(Tensor output, Tensor label, int igonre) {
        // TODO Auto-generated method stub
        return this.lossFunction.loss(output, label, igonre);
    }
    
    public Tensor projection_loss(Tensor dit_z, Tensor img) {
    	if(norm_kernel == null) {
    		norm_kernel = new NormalizeKernel(cudaManager);
    	}
    	
    	if(n_z == null || n_z.number != dit_z.number) {
    		n_z = Tensor.createGPUTensor(n_z, dit_z.shape(), true);
    		r_z = Tensor.createGPUTensor(r_z, dit_z.shape(), true);
    		loss = Tensor.createGPUTensor(loss, dit_z.number, dit_z.channel, dit_z.height, 1, true);
    	}
    	
    	Tensor rz = this.forward_features(img);
//    	rz.showShape("rz");
//    	dit_z.showShape("dit_z");
    	norm_kernel.l2norm3Dim(dit_z, n_z);
    	norm_kernel.l2norm3Dim(rz, r_z);
    	
    	norm_kernel.projection_loss(n_z, r_z, loss);
    	return loss;
    }
    
    public Tensor projection_z_loss(Tensor dit_z, Tensor z) {
    	if(norm_kernel == null) {
    		norm_kernel = new NormalizeKernel(cudaManager);
    	}
    	
    	if(n_z == null || n_z.number != dit_z.number) {
    		n_z = Tensor.createGPUTensor(n_z, dit_z.shape(), true);
    		r_z = Tensor.createGPUTensor(r_z, dit_z.shape(), true);
    		loss = Tensor.createGPUTensor(loss, dit_z.number, dit_z.channel, dit_z.height, 1, true);
    	}
    	
    	norm_kernel.l2norm3Dim(dit_z, n_z);
    	norm_kernel.l2norm3Dim(z, r_z);
    	
    	norm_kernel.projection_loss(n_z, r_z, loss);
    	return loss;
    }
    
    public Tensor projection_loss_back(Tensor dit_z) {
    	
    	norm_kernel.projection_loss_back(r_z, n_z);
    	
    	norm_kernel.l2norm3Dim_back4(dit_z, n_z, r_z);
    	
    	return r_z;
    }
    
    public Tensor lossDiff(Tensor output, Tensor label, int igonre) {
        // TODO Auto-generated method stub
        return this.lossFunction.diff(output, label, igonre);
    }

    public void saveModel(RandomAccessFile outputStream) throws IOException {
    	main.saveModel(outputStream);
        System.out.println("tail save success...");
    }

    public void loadModel(RandomAccessFile inputStream) throws IOException {
    	main.loadModel(inputStream);
        System.out.println("tail load success...");
    }

    @Override
    public void putParamters() {
        // TODO Auto-generated method stub
    }

    @Override
    public void putParamterGrads() {
        // TODO Auto-generated method stub
    }

	public Tensor getClstokens() {
		return this.clstokens;
	}
    
	public Tensor cls_loss(Tensor p, Tensor t) {
		if(mse_kernel == null) {
			mse_kernel = new MSELossKernel(cudaManager);
		}
		if(cls_loss == null || cls_loss.number != p.number) {
			cls_loss = Tensor.createGPUTensor(cls_loss, p.shape(), true);
		}
		mse_kernel.forward(p, t, cls_loss);
		return cls_loss;
	}
	
	public Tensor cls_loss_back(Tensor p, Tensor t) {
		if(mse_kernel == null) {
			mse_kernel = new MSELossKernel(cudaManager);
		}
		if(cls_delta == null || cls_delta.number != p.number) {
			cls_delta = Tensor.createGPUTensor(cls_delta, p.shape(), true);
		}
		mse_kernel.backward(p, t, cls_delta);
		return cls_delta;
	}
	
	public Tensor cfm_loss(Tensor p, Tensor t) {
		if(mse_kernel == null) {
			mse_kernel = new MSELossKernel(cudaManager);
		}
		if(cfm_loss == null || cfm_loss.number != p.number) {
			cfm_loss = Tensor.createGPUTensor(cfm_loss, p.shape(), true);
		}
		mse_kernel.forward(p, t, cfm_loss);
		return cfm_loss;
	}
	
	public Tensor cfm_loss_back(Tensor p, Tensor t) {
		if(mse_kernel == null) {
			mse_kernel = new MSELossKernel(cudaManager);
		}
		if(cfm_delta == null || cfm_delta.number != p.number) {
			cfm_delta = Tensor.createGPUTensor(cfm_delta, p.shape(), true);
		}
		mse_kernel.backward(p, t, cfm_delta);
		return cfm_delta;
	}
	
	public Tensor cfm_cls_loss(Tensor p, Tensor t) {
		if(mse_kernel == null) {
			mse_kernel = new MSELossKernel(cudaManager);
		}
		if(cfm_cls_loss == null || cfm_cls_loss.number != p.number) {
			cfm_cls_loss = Tensor.createGPUTensor(cfm_cls_loss, p.shape(), true);
		}
		mse_kernel.forward(p, t, cfm_cls_loss);
		return cfm_cls_loss;
	}
	
	public Tensor cfm_cls_loss_back(Tensor p, Tensor t) {
		if(mse_kernel == null) {
			mse_kernel = new MSELossKernel(cudaManager);
		}
		if(cfm_cls_delta == null || cfm_cls_delta.number != p.number) {
			cfm_cls_delta = Tensor.createGPUTensor(cfm_cls_delta, p.shape(), true);
		}
		mse_kernel.backward(p, t, cfm_cls_delta);
		return cfm_cls_delta;
	}
	
	public static void main(String[] args) {
		
		int N = 2;
		int C = 32;
		int H = 16;
		int W = 16;
		
		String inputPath = "D:\\models\\roll_x.json";
	    Map<String, Object> datas = LagJsonReader.readJsonFileSmallWeight(inputPath);
	    Tensor input = new Tensor(N, C, H, W, true);
	    ModeLoaderlUtils.loadData(input, datas, "x");
    	
	    String cyPath = "D:\\models\\roll_target.json";
	    Map<String, Object> cydatas = LagJsonReader.readJsonFileSmallWeight(cyPath);
	    Tensor target = new Tensor(N, C, H, W, true);
	    ModeLoaderlUtils.loadData(target, cydatas, "target");
	    
	    Tensor cfm_ut = new Tensor(N, C, H, W, true);
	    
	    CNN nn = new CNN(null);
        nn.CUDNN = true;
        nn.number = N;
        
        nn.tensorOP.roll(target, cfm_ut, 1, 0);
        
        MSELossKernel mse_kernel = new MSELossKernel(nn.cudaManager);
        Tensor cfm_loss = new Tensor(N, C, H, W, true);
		mse_kernel.forward(input, cfm_ut, cfm_loss);
		
		cfm_loss.showDM("cfm_cls_loss");
		
		float cfm_loss_mean = MatrixOperation.sum(cfm_loss.syncHost()) / cfm_loss.number * -1;
		System.err.println(cfm_loss_mean);
	}
	
}

