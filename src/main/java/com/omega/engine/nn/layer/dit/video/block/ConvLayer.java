package com.omega.engine.nn.layer.dit.video.block;

import java.io.IOException;
import java.io.RandomAccessFile;

import com.omega.common.utils.RandomUtils;
import com.omega.engine.gpu.cudnn.ConvCudnnKernel;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.active.SiLULayer;
import com.omega.engine.nn.layer.gpu.BiasKernel;
import com.omega.engine.nn.network.Network;
import com.omega.engine.nn.network.utils.ModelUtils;
import com.omega.engine.tensor.Tensor;
import com.omega.engine.updater.UpdaterFactory;

/**
 * ConvLayer
 *
 * @author Administrator
 */
public class ConvLayer extends Layer {
	
	public int kernel_size = 3;
	public int stride = 1;
    public int dilation = 1;
    public int groups = 1;
    public int padding;
    
    private boolean hasAct = true;

    private ConvCudnnKernel kernel;
    private BiasKernel biasKernel;
    
    public SiLULayer act;
    
    private Tensor convOutput;

    public ConvLayer(int channel, int oChannel, int H, int W, int kernel_size, int stride, int dilation, int groups, int padding, boolean hasAct, boolean bias, Network network) {
        this.network = network;
        if (this.updater == null) {
            this.setUpdater(UpdaterFactory.create(network));
        }
        this.channel = channel;
        this.oChannel = oChannel;
        this.height = H;
        this.width = W;
        this.kernel_size = kernel_size;
        this.stride = stride;
        this.dilation = dilation;
        this.groups = groups;
        this.padding = padding;
        this.hasAct = hasAct;
        this.hasBias = bias;
        initLayers();
    }

    public static void main(String[] args) {

    }

    @Override
    public void init() {
        // TODO Auto-generated method stub
        this.number = network.number;
        if (convOutput == null || convOutput.number != this.number) {
        	convOutput = Tensor.createGPUTensor(convOutput, this.number, oChannel, oHeight, oWidth, true);
        }
    }

    public void init(Tensor input) {
        // TODO Auto-generated method stub
        this.number = input.number;
        if (convOutput == null || convOutput.number != this.number) {
        	convOutput = Tensor.createGPUTensor(convOutput, this.number, oChannel, oHeight, oWidth, true);
        }
    }

    public void initLayers() {
    	int dataLength = oChannel * (channel/groups) * kernel_size * kernel_size;
    	kernel = new ConvCudnnKernel(network, channel, height, width, oChannel, kernel_size, kernel_size, stride, padding, groups, cuda());
    	int[] outshape = kernel.outputShape();
    	this.oHeight = outshape[2];
    	this.oWidth = outshape[3];
    	this.weight = new Tensor(oChannel, channel/groups, kernel_size, kernel_size, RandomUtils.kaiming_uniform(dataLength, channel/groups * kernel_size * kernel_size, this.paramsInit), true);
		if (hasBias) {
			biasKernel = new BiasKernel(cuda());
		    this.bias = new Tensor(1, 1, 1, oChannel, true);
		}
		if(hasAct) {
			act = new SiLULayer(this);
		}
    }

    @Override
    public void initBack() {
        // TODO Auto-generated method stub
    	if (this.diff == null || this.number != this.diff.number) {
            this.diff = new Tensor(number, channel, height, width, true);
        }
        if (this.diffW == null && !freeze) {
        	if (this.hasBias) {
                this.diffB = new Tensor(1, 1, 1, oChannel, true);
            }
            this.diffW = new Tensor(this.oChannel, this.channel/groups, this.kernel_size, this.kernel_size, true);
        }
    }

    @Override
    public void initParam() {
        // TODO Auto-generated method stub
    }

    @Override
    public void output() {
        // TODO Auto-generated method stub
    	 kernel.conv(input, weight, convOutput);
         if (this.hasBias) {
             biasKernel.addConvBiasFast(convOutput, bias);
         }
         if(hasAct) {
        	 act.forward(convOutput);
        	 this.output = act.getOutput();
         }else {
        	 this.output = convOutput;
         }
    }

    @Override
    public Tensor getOutput() {
        // TODO Auto-generated method stub
        return output;
    }

    @Override
    public void diff() {
        // TODO Auto-generated method stub
    	Tensor d = delta;
    	if(hasAct) {
    		act.back(delta);
       		d = act.diff;
        }
        kernel.dw(input, d, diffW);
        if (this.hasBias) {
            biasKernel.backwardConvBias(diffB, d);
        }
        kernel.dx(d, weight, diff);
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
    
    public void back(Tensor delta, Tensor diff) {
        // TODO Auto-generated method stub
    	this.diff = diff;
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
    	if (this.updater != null) {
            this.updater.update(this);

        } else {
            for (int i = 0; i < this.weight.getDataLength(); i++) {
                this.weight.data[i] -= this.learnRate * this.diffW.data[i];
            }
            for (int i = 0; i < this.bias.getDataLength(); i++) {
                this.bias.data[i] -= this.learnRate * this.diffB.data[i];
            }
        }
    }

    @Override
    public void showDiff() {
        // TODO Auto-generated method stub
    }

    @Override
    public LayerType getLayerType() {
        // TODO Auto-generated method stub
        return LayerType.clip_vision_embedding;
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

    public void saveModel(RandomAccessFile outputStream) throws IOException {
        ModelUtils.saveParams(outputStream, weight);
        if (hasBias) {
            ModelUtils.saveParams(outputStream, bias);
        }
    }

    public void loadModel(RandomAccessFile inputStream) throws IOException {
        ModelUtils.loadParams(inputStream, weight);
        if (hasBias) {
            ModelUtils.loadParams(inputStream, bias);
        }
    }

    @Override
    public void accGrad(float scale) {
        // TODO Auto-generated method stub

    }

}

