package com.omega.engine.nn.layer.lpips;

import com.omega.engine.ad.op.gpu.NormalizeKernel;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.lpips.NetLinLayer;
import com.omega.engine.nn.layer.lpips.VGG;
import com.omega.engine.nn.layer.lpips.gpu.LPIPSKernel;
import com.omega.engine.nn.network.Network;
import com.omega.engine.tensor.Tensor;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.util.ArrayList;
import java.util.List;

public class LPIPSBlock extends Layer {
    public VGG vgg;
    //[64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"]
    public List<NetLinLayer> lins;
    private boolean dropout = false;
    private String[] cfg = new String[]{"64", "64", "M", "128", "128", "M", "256", "256", "256", "M", "512", "512", "512", "M", "512", "512", "512", "M"};
    private int numClass;
    private int[] featuresIndex;
    private NormalizeKernel normKernel;
    private LPIPSKernel lpipsKernel;
    private boolean scaling = false;
    private Tensor shift;
    private Tensor scale;
    private Tensor[] outputs;
    private Tensor[] feats0;
    private Tensor[] feats1;
    private Tensor[] diffs;
    
    private Tensor scalingInput;
    private Tensor scalingTaget;
    
    public LPIPSBlock(int channel, int height, int width, boolean dropout, String[] cfg, int numClass, int[] featuresIndex,boolean scaling, Network network) {
        this.network = network;
        this.dropout = dropout;
        this.channel = channel;
        this.height = height;
        this.width = width;
        this.oChannel = 1;
        this.oHeight = 1;
        this.oWidth = 1;
        this.cfg = cfg;
        this.numClass = numClass;
        this.featuresIndex = featuresIndex;
        this.scaling = scaling;
        initLayers();
    }

    public void initLayers() {
        shift = new Tensor(1, 1, 1, 3, new float[]{-0.030f, -0.088f, -0.188f}, true);
        scale = new Tensor(1, 1, 1, 3, new float[]{0.458f, 0.448f, 0.450f}, true);
        vgg = new VGG(channel, numClass, height, width, false, cfg, true, true, network);
        lins = new ArrayList<NetLinLayer>();
        for (int i = 0; i < featuresIndex.length; i++) {
            Layer ol = vgg.features.get(featuresIndex[i]);
            int ic = ol.oChannel;
            int ih = ol.oHeight;
            int iw = ol.oWidth;
            NetLinLayer n = new NetLinLayer(ic, 1, ih, iw, dropout, network);
            lins.add(n);
        }
        normKernel = new NormalizeKernel(cuda());
        lpipsKernel = new LPIPSKernel(cuda());
    }

    @Override
    public void init() {
        this.number = this.network.number;
        if (outputs == null) {
            outputs = new Tensor[featuresIndex.length];
            feats0 = new Tensor[featuresIndex.length];
            feats1 = new Tensor[featuresIndex.length];
            diffs = new Tensor[featuresIndex.length];
        }
        if (output == null || output.number != this.input.number) {
            output = Tensor.createGPUTensor(output, this.input.number, 1, 1, 1, true);
        } else {
            this.output.clearGPU();
        }
    }

    public void init(Tensor input) {
        this.number = input.number;
        if (outputs == null) {
            outputs = new Tensor[featuresIndex.length];
            feats0 = new Tensor[featuresIndex.length];
            feats1 = new Tensor[featuresIndex.length];
            diffs = new Tensor[featuresIndex.length];
            for (int i = 0; i < outputs.length; i++) {
                int index = featuresIndex[i];
                Layer layer = vgg.features.get(index);
                outputs[i] = Tensor.createGPUTensor(outputs[i], this.number, layer.oChannel, layer.oHeight, layer.oWidth, true);
                feats0[i] = outputs[i].createLike();
                feats1[i] = outputs[i].createLike();
                diffs[i] = outputs[i].createLike();
            }
            if(scaling) {
            	scalingInput = Tensor.createGPUTensor(scalingInput, input.number, input.channel, input.height, input.width, true);
            	scalingTaget = Tensor.createGPUTensor(scalingTaget, input.number, input.channel, input.height, input.width, true);
            }
        }
        if (output == null || output.number != input.number) {
            output = Tensor.createGPUTensor(output, input.number, 1, 1, 1, true);
        } else {
            this.output.clearGPU();
        }
    }

    @Override
    public void initBack() {
    }

    @Override
    public void initParam() {
        // TODO Auto-generated method stub
    }

    @Override
    public void output() {
        // TODO Auto-generated method stub
    }

    public void output(Tensor input, Tensor label) {
        // TODO Auto-generated method stub
        //		lpipsKernel.scaling(input, shift, scale, input);
        //		lpipsKernel.scaling(label, shift, scale, label);
    	
    	if(scaling) {
            lpipsKernel.scaling(label, shift, scale, scalingTaget);
    		lpipsKernel.scaling(input, shift, scale, scalingInput);
//    		int last1 = 16 * input.getOnceSize();
//    		input.showDMByOffsetRed(last1, input.getOnceSize(), "input");
            vgg.featuresCopy(scalingTaget, featuresIndex, feats1);
            vgg.features(scalingInput, featuresIndex, outputs);
    	}else {
    		vgg.featuresCopy(label, featuresIndex, feats1);
            vgg.features(input, featuresIndex, outputs);
    	}
        
        for (int i = 0; i < feats1.length; i++) {
            normKernel.l2norm1Dim(outputs[i], feats0[i]);
//            int last1 = 17 * feats1[i].getOnceSize();
//            feats0[i].showDMByOffsetRed(last1, feats0[i].getOnceSize(), "1["+i+"]");
            normKernel.l2norm1Dim(feats1[i], feats1[i]);
            lpipsKernel.lpip_l2(feats0[i], feats1[i], diffs[i]);  //y = (x1 - x2)^2
            lins.get(i).forward(diffs[i]);
            Tensor_OP().mean2Dim(lins.get(i).getOutput(), this.output);
        }
    }

    @Override
    public Tensor getOutput() {
        // TODO Auto-generated method stub
        return this.output;
    }

    @Override
    public void diff() {
        // TODO Auto-generated method stub
        //		System.out.println(index);
        for (int i = feats1.length - 1; i >= 0; i--) {
            Tensor_OP().mean2DimBack(delta, lins.get(i).getOutput());
           
            lins.get(i).back(lins.get(i).getOutput());

            lpipsKernel.lpip_l2_backward(lins.get(i).diff, feats0[i], feats1[i], feats1[i]);
            
            normKernel.l2norm1Dim_back3(outputs[i], feats1[i], feats0[i]);
//            feats0[i].showDM("feats0:"+i);
        }
        Tensor diff = vgg.featuresBackward(feats0, featuresIndex);
//        diff.showDM("dsix");
        if(scaling) {
        	lpipsKernel.scaling_backwad(diff, scale, diff);
        }
        //		lpipsKernel.scaling_backwad(diff, scale, diff);
        this.diff = diff;
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
    public void update() {
        // TODO Auto-generated method stub
        //		for(int i = 0;i<lins.size();i++) {
        //			lins.get(i).update();
        //		}
    }

    @Override
    public void showDiff() {
        // TODO Auto-generated method stub
    }

    @Override
    public LayerType getLayerType() {
        // TODO Auto-generated method stub
        return LayerType.lpips;
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
    public void forward(Tensor input) {
        // TODO Auto-generated method stub
        /**
         * 参数初始化

         */
        this.init();
        /**
         * 设置输入

         */
        this.setInput(input);
        /**
         * 计算输出

         */
        this.output();
    }

    public void forward(Tensor input, Tensor label) {
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
        this.output(input, label);
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
    public void backTemp() {
        // TODO Auto-generated method stub
    }

    @Override
    public void accGrad(float scale) {
        // TODO Auto-generated method stub
    }

    public void saveModel(RandomAccessFile outputStream) throws IOException {
    }

    public void loadModel(RandomAccessFile inputStream) throws IOException {
    }
}