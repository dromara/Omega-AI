package com.omega.engine.nn.network;

import com.omega.engine.ad.op.TensorOP;
import com.omega.engine.gpu.BaseKernel;
import com.omega.engine.gpu.CUDAManager;
import com.omega.engine.gpu.GlobalNormKernel;
import com.omega.engine.loss.LossFunction;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.RouteLayer;
import com.omega.engine.nn.layer.normalization.NormalizationLayer;
import com.omega.engine.nn.model.NetworkInit;
import com.omega.engine.parallel.cuda.CUDACommonManager;
import com.omega.engine.parallel.cuda.CUDAPool;
import com.omega.engine.tensor.Tensor;
import com.omega.engine.updater.UpdaterFactory;
import com.omega.engine.updater.UpdaterType;
import jcuda.Pointer;
import jcuda.runtime.JCuda;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * base network
 *
 * @author Administrator
 */
public abstract class Network {
    public boolean CUDNN = false;
    public boolean gradCacheMode = false;
    public CUDAManager cudaManager;
    public TensorOP tensorOP;
    public BaseKernel baseKernel;
    public int time;
    public NetworkType networkType;
    public List<Tensor> paramters = new ArrayList<Tensor>();
    public UpdaterType updater = UpdaterType.none;
    public RunModel RUN_MODEL = RunModel.TRAIN;
    public Map<String, Float> updaterParams;
    public boolean GRADIENT_CHECK = false;
    public int rankId = 0;
    public int layerCount = 0;
    public int trainingTime = 100;
    public int currentTrainingTime = 0;
    public List<Layer> layerList = new ArrayList<Layer>();
    public List<Layer> paramLayers = new ArrayList<Layer>();
    public LossFunction lossFunction;
    public Tensor input;
    public Tensor lossDiff;
    public float accuracy = 0.01f;
    public float learnRate = 0.01f;
    public float errorRate = 0.001f;
    public float currentError = 0.0f;
    public float weight_decay = 1e-4f;
    /**
     * batch number
     */
    public int number = 0;
    public int oChannel = 0;
    public int oHeight = 0;
    public int oWidth = 0;
    public int train_time = 0;
    public long workspaceSize = 0;
    public Pointer workspace;
    public boolean PROPAGATE_DOWN = false;
    public boolean CLIP_GRAD_NORM = false;
    public float clip_coef;
    public GlobalNormKernel normKernel;
    public List<RouteLayer> routeLayers;
    public List<Tensor> deltaParamters;
    private int threadNum = 8;
    private Map<Layer, Tensor[]> grads = new HashMap<Layer, Tensor[]>();
    private int channel = 0;
    private int height = 0;
    private int width = 0;

    public Network() {
        initCUDA();
    }

    public Network(int rankId) {
        this.rankId = rankId;
        initCUDA();
    }

    public abstract void init() throws Exception;

    public abstract Tensor predict(Tensor input);

    public abstract Tensor forward(Tensor input);

    public abstract void back(Tensor lossDiff);

    public abstract Tensor loss(Tensor output, Tensor label);

    public abstract Tensor lossDiff(Tensor output, Tensor label);

    public abstract Tensor loss(Tensor output, Tensor label, Tensor loss);

    public abstract Tensor lossDiff(Tensor output, Tensor label, Tensor diff);

    public abstract NetworkType getNetworkType();

    public abstract void clearGrad();

    public abstract void putParamters();

    public abstract void putParamterGrads();

    private void initCUDA() {
        CUDACommonManager cuda = CUDAPool.cudaCommonManager(rankId);
        cudaManager = cuda.getCudaManager();
        tensorOP = cuda.getTensorOP();
        baseKernel = cuda.getBaseKernel();
    }

    public Tensor createParamterGrad(int number, int channel, int height, int width, boolean hasGPU) {
        Tensor pGrad = new Tensor(number, channel, height, width, hasGPU);
        this.addPamamter(pGrad);
        return pGrad;
    }

    public void addPamamter(Tensor pGrad) {
        if (this.paramters == null) {
            this.paramters = new ArrayList<Tensor>();
        }
        this.paramters.add(pGrad);
    }

    public void addDeltaParamters(Tensor pGrad) {
        if (this.deltaParamters == null) {
            this.deltaParamters = new ArrayList<Tensor>();
        }
        this.deltaParamters.add(pGrad);
    }

    public Tensor getDiff() {
        //		System.out.println(this.getNextLayer(0).getLayerType().toString());
        return this.getNextLayer(0).diff;
    }

    public void setNumber(int number) {
        this.number = number;
    }

    public void setInputData(Tensor input) {
        this.number = input.getShape()[0];
        this.input = input;
    }

    public void setLossDiff(Tensor lossDiff) {
        this.lossDiff = lossDiff;
        this.getLastLayer().setDelta(this.lossDiff);
    }

    public Tensor getOutput() {
        // TODO Auto-generated method stub
        return this.getLastLayer().getOutput();
    }

    public int getOutputNum() {
        int num = this.getLastLayer().getOutput().getShape()[1] * this.getLastLayer().getOutput().getShape()[2] * this.getLastLayer().getOutput().getShape()[3];
        return num;
    }

    public Layer getLastLayer() {
        return this.layerList.get(this.layerList.size() - 1);
    }

    public Layer getPreLayer(int index) {
        if (index > 0 && index < this.layerCount) {
            return this.layerList.get(index - 1);
        }
        return null;
    }

    public Layer getNextLayer(int index) {
        //		if(index > 0 && index < this.layerCount - 1) {
        if (index < this.layerCount - 1) {
            return this.layerList.get(index + 1);
        }
        return null;
    }

    public Layer getLayer(int index) {
        if (index < this.layerCount) {
            return this.layerList.get(index);
        }
        return null;
    }

    public Tensor getDelta(int index) {
        if (index > 0 && index < this.layerCount - 1) {
            return this.layerList.get(index + 1).diff;
        } else {
            return this.lossDiff;
        }
    }

    public void addLayer(Layer layer) {
        layer.setNetwork(this);
        layer.setIndex(this.layerList.size());
        if (layer.updater == null) {
            layer.setUpdater(UpdaterFactory.create(this));
        }
        if (layer.index <= 1) {
            layer.PROPAGATE_DOWN = false;
        }
        this.layerList.add(layer);
    }

    public NetworkInit save() {
        NetworkInit init = new NetworkInit(this);
        for (Layer layer : this.layerList) {
            init.getLayers().add(layer.save());
        }
        return init;
    }

    public void update() {
        this.train_time += 1;
        for (int i = layerCount - 1; i >= 0; i--) {
            Layer layer = layerList.get(i);
            layer.learnRate = this.learnRate;
            layer.update();
        }
    }

    public void update(int count) {
        this.train_time += 1;
        for (int i = layerCount - 1; i >= 0; i--) {
            Layer layer = layerList.get(i);
            layer.learnRate = this.learnRate;
            layer.update();
        }
    }

    /**
     * accumulation gradient
     */
    public void accGrad(int steps) {
        float scale = 1.0f / steps;
        for (Layer layer : paramLayers) {
            layer.accGrad(scale);
        }
    }

    public void unfreeze() {
        for (int i = 0; i < layerCount; i++) {
            layerList.get(i).freeze = false;
        }
    }

    public void saveToJson(String path) {
    }

    public int getThreadNum() {
        return threadNum;
    }

    public void setThreadNum(int threadNum) {
        this.threadNum = threadNum;
    }

    //	public void clipGradNorm(float clip) {
    //		float total_norm = 0.0f;
    //		float[] norms = new float[paramters.size()];
    //		System.out.println(paramters.size());
    //		for(int i = 0;i<paramters.size();i++) {
    //			Tensor pGrad = paramters.get(i);
    //			norms[i] = pGrad.norm();
    ////			total_norm += Math.pow(pGrad.norm(), 2.0d);
    //		}
    //		total_norm = MatrixOperation.norm(norms);
    //		total_norm = (float) Math.pow(total_norm, (1.0d / 2.0d));
    //		float clip_coef = clip / (total_norm + 1e-6f);
    //		System.out.println("clip_coef:"+clip_coef);
    //		if(clip_coef < 1) {
    //			for(Tensor pGrad:paramters) {
    //				TensorOP.mul(pGrad, clip_coef, pGrad);
    ////				pGrad.showDM();
    //			}
    //		}
    //	}
    public int getChannel() {
        return channel;
    }

    public void setChannel(int channel) {
        this.channel = channel;
    }

    public int getHeight() {
        return height;
    }

    public void setHeight(int height) {
        this.height = height;
    }

    public int getWidth() {
        return width;
    }

    public void setWidth(int width) {
        this.width = width;
    }

    public Map<Layer, Tensor[]> getGrads() {
        return grads;
    }

    public void setGrads(Map<Layer, Tensor[]> grads) {
        this.grads = grads;
    }

    public void getGradNorm(Layer layer) {
        if (normKernel == null) {
            normKernel = new GlobalNormKernel(cudaManager);
        }
        Tensor[] gradList = null;
        if (grads.containsKey(layer)) {
            gradList = grads.get(layer);
            gradList[0].clearGPU();
            gradList[1].clearGPU();
        } else {
            gradList = new Tensor[2];
            gradList[0] = new Tensor(1, 1, 1, 1, true);
            gradList[1] = new Tensor(1, 1, 1, 1, true);
            grads.put(layer, gradList);
        }
        if (layer instanceof NormalizationLayer) {
            NormalizationLayer nl = (NormalizationLayer) layer;
            normKernel.globalNorm(gradList[0], nl.diffGamma);
            if (nl.diffBeta != null) {
                normKernel.globalNorm(gradList[1], nl.diffBeta);
            }
        } else {
            if (layer.diffW != null) {
                normKernel.globalNorm(gradList[0], layer.diffW);
            }
            if (layer.diffB != null) {
                normKernel.globalNorm(gradList[1], layer.diffB);
            }
        }
    }

    public void getAccGradNorm(Layer layer) {
        if (normKernel == null) {
            normKernel = new GlobalNormKernel(cudaManager);
        }
        Tensor[] gradList = null;
        if (grads.containsKey(layer)) {
            gradList = grads.get(layer);
            gradList[0].clearGPU();
            gradList[1].clearGPU();
        } else {
            gradList = new Tensor[2];
            gradList[0] = new Tensor(1, 1, 1, 1, true);
            gradList[1] = new Tensor(1, 1, 1, 1, true);
            grads.put(layer, gradList);
        }
        if (layer.accDW != null) {
            normKernel.globalNorm(gradList[0], layer.accDW);
        }
        if (layer.accDB != null) {
            normKernel.globalNorm(gradList[1], layer.accDB);
        }
    }

    public void clipGradNorm(float gradClip) {
        for (Layer layer : paramLayers) {
            if (layer.accDW != null) {
                getAccGradNorm(layer);
            } else {
                getGradNorm(layer);
            }
        }
        float clip_coef = 0.0f;
        for (Layer layer : grads.keySet()) {
            for (Tensor c : grads.get(layer)) {
                //				clip_coef += Math.pow(Math.sqrt(c.syncHost()[0]), 2);
                clip_coef += c.syncHost()[0];
            }
        }
        //		System.err.println(clip_coef);
        clip_coef = (float) (gradClip / (Math.sqrt(clip_coef) + 1e-6f));
        //		System.out.println(clip_coef);
        if (clip_coef > 1) {
            clip_coef = 1;
        }
        System.err.println(clip_coef);
        for (Layer layer : grads.keySet()) {
            if (layer instanceof NormalizationLayer) {
                NormalizationLayer nl = (NormalizationLayer) layer;
                if (nl.accDW != null) {
                    tensorOP.mul(nl.accDW, clip_coef, nl.accDW);
                } else {
                    tensorOP.mul(nl.diffGamma, clip_coef, nl.diffGamma);
                }
                if (nl.diffBeta != null) {
                    if (nl.accDB != null) {
                        tensorOP.mul(nl.accDB, clip_coef, nl.accDB);
                    } else {
                        tensorOP.mul(nl.diffBeta, clip_coef, nl.diffBeta);
                    }
                }
            } else {
                if (layer.diffW != null) {
                    if (layer.accDW != null) {
                        tensorOP.mul(layer.accDW, clip_coef, layer.accDW);
                    } else {
                        tensorOP.mul(layer.diffW, clip_coef, layer.diffW);
                    }
                }
                if (layer.diffB != null) {
                    if (layer.accDB != null) {
                        tensorOP.mul(layer.accDB, clip_coef, layer.accDB);
                    } else {
                        tensorOP.mul(layer.diffB, clip_coef, layer.diffB);
                    }
                }
            }
        }
    }

    public void addRouteLayer(RouteLayer layer) {
        if (routeLayers == null) {
            routeLayers = new ArrayList<RouteLayer>();
        }
        routeLayers.add(layer);
    }
    
    public void clearCacheDelta() {
    	System.err.println(routeLayers.size());
        for (RouteLayer rl : routeLayers) {
            rl.clearCacheDelta();
        }
        JCuda.cudaDeviceSynchronize();
    }
    
    public void copyParams(Network model) {
    	for(int i = 0;i<paramLayers.size();i++) {
    		Layer l1 = paramLayers.get(i);
    		Layer l2 = model.paramLayers.get(i);
    		if (l1 instanceof NormalizationLayer) {
                NormalizationLayer ema_nl = (NormalizationLayer) l1;
                NormalizationLayer model_nl = (NormalizationLayer) l2;
                if(ema_nl.gamma == null) {
                	ema_nl.gamma = model_nl.gamma.createLike();
                }
                tensorOP.copyTensorGPU(model_nl.gamma, ema_nl.gamma);
                if(model_nl.beta != null) {
                	if(ema_nl.beta == null) {
                    	ema_nl.beta = model_nl.beta.createLike();
                    }
                	tensorOP.copyTensorGPU(model_nl.beta, ema_nl.beta);
                }
    		}else {
    			if(l1.weight == null) {
    				l1.weight = l2.weight.createLike();
                }
    			tensorOP.copyTensorGPU(l2.weight, l1.weight);
    			if(l2.bias != null) {
    				if(l1.bias == null) {
        				l1.bias = l2.bias.createLike();
                    }
       			 	tensorOP.copyTensorGPU(l2.bias, l1.bias);
    			}
    		}
    	}
    }
    
    public void update_ema(Network model, float decay) {
    	for(int i = 0;i<paramLayers.size();i++) {
    		Layer l1 = paramLayers.get(i);
    		Layer l2 = model.paramLayers.get(i);
    		if (l1 instanceof NormalizationLayer) {
                NormalizationLayer ema_nl = (NormalizationLayer) l1;
                NormalizationLayer model_nl = (NormalizationLayer) l2;
                tensorOP.update_ema(ema_nl.gamma, model_nl.gamma, decay);
                if(ema_nl.beta != null) {
                	tensorOP.update_ema(ema_nl.beta, model_nl.beta, decay);
                }
    		}else {
    			tensorOP.update_ema(l1.weight, l2.weight, decay);
    			if(l1.bias != null) {
    				tensorOP.update_ema(l1.bias, l2.bias, decay);
    			}
    		}
    	}
    }
}
