package com.omega.engine.nn.layer;

import com.omega.common.config.Tensor;
import com.omega.engine.ad.op.TensorOP;
import com.omega.engine.gpu.BaseKernel;
import com.omega.engine.gpu.CUDAManager;
import com.omega.engine.gpu.GPUOP;
import com.omega.engine.gpu.data.CacheDataSet;
import com.omega.engine.nn.layer.utils.LayerHook;
import com.omega.engine.nn.model.LayerInit;
import com.omega.engine.nn.network.Network;
import com.omega.engine.updater.Updater;

import java.util.ArrayList;
import java.util.List;

/**
 * Base Layer
 *
 * @author Administrator
 */
public abstract class Layer {
    public String name;
    public Network network;
    public boolean PROPAGATE_DOWN = true;
    ;
    public int index = 0;
    /**
     * batch number
     */
    public int number = 0;
    public int channel = 0;
    public int height = 0;
    public int width = 0;
    public int oChannel = 0;
    public int oHeight = 0;
    public int oWidth = 0;
    public Tensor input;
    public Tensor output;
    public Tensor diff;
    public Tensor delta;
    public Tensor weight;
    public Tensor bias;
    public Tensor diffW;
    public Tensor accDW;
    public Tensor diffB;
    public Tensor accDB;
    public Tensor cache_delta;
    public Tensor org_delta;
    public ParamsInit paramsInit = ParamsInit.linear;
    public boolean hasBias = true;
    public float lambda = 0.01f;
    public float learnRate = 0.001f;
    public float eta = 0.00001f;
    public LayerType layerType;
    public Updater updater;
    public boolean freeze = false;
    public boolean hasParams = false;
    public boolean isOutput = false;
    /**
     * cache data
     */
    private CacheDataSet tampDataSet;
    private List<LayerHook> hooks;

    public abstract void init();

    public abstract void initBack();

    public abstract void initParam();

    public abstract void output();

    //	/**
    //	 * use for gradient check
    //	 * @param eta
    //	 * @return
    //	 */
    //	public abstract Blob output(float eta);
    public abstract Tensor getOutput();

    public abstract void diff();

    public abstract void forward();

    public abstract void back();

    public abstract void backTemp();

    public abstract void forward(Tensor input);

    public abstract void back(Tensor delta);

    public abstract void update();

    public abstract void accGrad(float scale);

    public abstract void showDiff();

    public abstract LayerType getLayerType();

    public abstract float[][][][] output(float[][][][] input);

    public abstract void initCache();

    public void setUpdater(Updater updater) {
        this.updater = updater;
    }

    public void setNetwork(Network network) {
        this.network = network;
    }

    public void setIndex(int index) {
        this.index = index;
    }

    public void setName(String name) {
        this.name = name;
    }

    public LayerInit save() {
        // TODO Auto-generated method stub
        return new LayerInit(this);
    }

    public void setInput(Tensor input) {
        this.input = input;
    }

    /**
     * 转换并设置输入数据
     */
    public void setInput() {
        /**
         * 获取上一层的输出作为当前层的输入

         */
        this.input = this.network.getPreLayer(this.index).output;
    }

    /**
     * 转换并设置输入数据
     */
    public void setDelta() {
        if (this.delta == null) {
            /**
             * 获取上一层的输出作为当前层的输入

             */
            if (this.index < this.network.layerList.size() - 1) {
                if (this.network.getNextLayer(this.index).getLayerType() != LayerType.route) {
                    this.delta = this.network.getNextLayer(this.index).diff;
                }
            }
        }
        /**
         * 合并路由层误差

         */
        if (this.cache_delta != null) {
            //			System.out.println("in===>:"+this.getLayerType());
            if (this.delta == null || this.delta.number != this.cache_delta.number) {
                this.delta = this.cache_delta;
            } else if (this.cache_delta != this.delta) {
                //				System.out.println("in===>add:"+this.getLayerType()+"["+index+"]");
                network.baseKernel.axpy_gpu(this.cache_delta, this.delta, this.delta.getDataLength(), 1, 1, 1);
                this.cache_delta.clearGPU();
            }
            //			this.cache_delta.clearGPU();
        }
    }

    /**
     * 转换并设置输入数据
     */
    public void setDelta(Tensor delta) {
        /**
         * 获取上一层的输出作为当前层的输入

         */
        this.delta = delta;
        /**
         * 合并路由层误差

         */
        if (this.cache_delta != null) {
            //			System.out.println("in===>:"+this.getLayerType());
            if (this.delta == null || this.delta.number != this.cache_delta.number) {
                this.delta = this.cache_delta;
            } else if (this.cache_delta != this.delta) {
                //				System.out.println("in===>add:"+this.getLayerType()+"["+index+"]");
                network.baseKernel.axpy_gpu(this.cache_delta, this.delta, this.delta.getDataLength(), 1, 1, 1);
                this.cache_delta.clearGPU();
            }
            //			this.cache_delta.clearGPU();
        }
    }

    /**
     * @param x
     * @return
     * @Title: gradientCheck
     * @Description: TODO(这里用一句话描述这个方法的作用)
     * <p>
     * gradientCheck:
     * <p>
     * (f(x + eta) - f(x - eta)) / (2 * eta) ≈ f'(x)
     */
    public float gradientCheck() {
        return 0.0f;
    }

    public CacheDataSet getTampDataSet() {
        return tampDataSet;
    }

    public void setTampDataSet(CacheDataSet tampDataSet) {
        this.tampDataSet = tampDataSet;
    }

    public int[] outputShape() {
        return new int[]{number, oChannel, oHeight, oWidth};
    }

    public void clearAccGrad() {
        if (accDW != null) {
            accDW.clearGPU();
            if (accDB != null) {
                accDB.clearGPU();
            }
        }
    }

    public void registerHook(LayerHook hook) {
        if (hooks == null) {
            hooks = new ArrayList<LayerHook>();
        }
        hooks.add(hook);
    }

    public void runHooks() {
        if (hooks != null) {
            for (LayerHook hook : hooks) {
                hook.runHookFn(this);
            }
        }
    }

    public void getGradNorm() {
        if (network.CLIP_GRAD_NORM && diffW != null) {
            network.getGradNorm(this);
        }
    }

    public void freeze() {
        this.freeze = true;
    }

    public GPUOP GPU_OP() {
        return this.network.cudaManager.getOp();
    }

    public TensorOP Tensor_OP() {
        return this.network.tensorOP;
    }

    public CUDAManager cuda() {
        return network.cudaManager;
    }

    public BaseKernel baseKernel() {
        return network.baseKernel;
    }
}

