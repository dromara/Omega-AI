package com.omega.engine.nn.layer.opensora.wfvae.modules;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.util.Map;

import com.omega.common.utils.MatrixUtils;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.ParamsInit;
import com.omega.engine.nn.layer.gpu.UpSample3DKernel;
import com.omega.engine.nn.network.CNN;
import com.omega.engine.nn.network.Network;
import com.omega.engine.tensor.Tensor;
import com.omega.engine.updater.UpdaterFactory;
import com.omega.example.clip.utils.ClipModelUtils;
import com.omega.example.transformer.utils.LagJsonReader;

import jcuda.runtime.JCuda;

/**
 * Spatial2xTime2x3DDownsample
 *
 * @author Administrator
 */
public class Spatial2xTime2x3DUpsample extends Layer {

	public UpSample3DKernel kernel;

    public WFCausalConv3D conv;

    private Tensor upOutput;
    
    private int pd;
    private int pw;
    private int ph;
    
    public int depth;
    public int oDepth;
    
    public Spatial2xTime2x3DUpsample(int channel, int oChannel, int depth, int height, int width, Network network) {
        this.network = network;
        this.channel = channel;
        this.depth = depth;
        this.oChannel = oChannel;
        this.height = height;
        this.width = width;
        initLayers();
        this.oHeight = conv.oHeight;
        this.oWidth = conv.oWidth;
        this.oChannel = conv.oChannel;
        this.oDepth = conv.oDepth;
    }

    public void initLayers() {
    	
        kernel = new UpSample3DKernel(cuda());
        
        pd = depth * 2 - 1;
        pw = width * 2;
        ph = height * 2;

        conv = new WFCausalConv3D(channel, oChannel, pd, pw, ph, 3, 1, 1, true, network);
        conv.setUpdater(UpdaterFactory.create(this.network));
        conv.paramsInit = ParamsInit.silu;
       
    }

    @Override
    public void init() {
        this.number = this.network.number;
        if(upOutput == null || upOutput.number != this.number) {
        	upOutput = Tensor.createGPUTensor(upOutput, number, channel * pd, ph, pw, true);
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
    	kernel.upsample3d_trilinear_offset(input, upOutput, number, channel, depth, 1, height, width, pd, 1, 2, 2, false, 0);
    	kernel.upsample3d_trilinear_offset(input, upOutput, number, channel, depth, depth - 1, height, width, pd, 2, 2, 2, false, 1);
//    	upOutput.showDMByOffsetRed(1 * channel * pd * ph * pw + 1 * pd * ph * pw, pd * ph * pw, "3d:");
//    	upOutput.showDM("upOutput");
    	conv.forward(upOutput);
    	this.output = conv.getOutput();
    }

    @Override
    public Tensor getOutput() {
        // TODO Auto-generated method stub
        return this.output;
    }

    @Override
    public void diff() {
        // TODO Auto-generated method stub
        conv.back(delta, upOutput);
//        upOutput.showDM("upOutput-diff");
        kernel.upsample3d_trilinear_delta_offset(upOutput, input, number, channel, depth, depth - 1, height, width, pd, 2, 2, 2, false, 1);
        kernel.upsample3d_trilinear_delta_offset(upOutput, input, number, channel, depth, 1, height, width, pd, 1, 2, 2, false, 0);
        this.diff = input;
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
        conv.update();
    }

    @Override
    public void showDiff() {
        // TODO Auto-generated method stub
    }

    @Override
    public LayerType getLayerType() {
        // TODO Auto-generated method stub
        return LayerType.block;
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
        conv.saveModel(outputStream);
    }

    public void loadModel(RandomAccessFile inputStream) throws IOException {
        conv.loadModel(inputStream);
    }
    
    public static void loadWeight(Map<String, Object> weightMap, Spatial2xTime2x3DUpsample network, boolean showLayers) {
        if (showLayers) {
            for (String key : weightMap.keySet()) {
                System.out.println(key);
            }
        }

        ClipModelUtils.loadData(network.conv.weight, weightMap, "conv.conv.weight", 5);
        ClipModelUtils.loadData(network.conv.bias, weightMap, "conv.conv.bias");
        
    }
    
    public static void main(String[] args) {

      int N = 2;
      int C = 3;
      int F = 17;
      int H = 32;
      int W = 32;
      
      int OC = 4;

      String inputPath = "D:\\models\\input_wf.json";
      Map<String, Object> datas = LagJsonReader.readJsonFileSmallWeight(inputPath);
      Tensor input = new Tensor(N, C * F, H, W, true);
      ClipModelUtils.loadData(input, datas, "x", 5);

      CNN nn = new CNN(null);
      nn.CUDNN = true;
      nn.number = N;
      //int channel,int kernelNum,int depth,int width,int height,int kDepth,int kWidth,int kHeight,int padding,int stride
      Spatial2xTime2x3DUpsample st = new Spatial2xTime2x3DUpsample(C, OC, F, H, W, nn);
      
      String weight = "D:\\models\\Spatial2xTime2x3DUpsample.json";
      loadWeight(LagJsonReader.readJsonFileSmallWeight(weight), st, true);
      
      Tensor delta = new Tensor(N, OC * (F * 2 - 1), H * 2, W * 2, true);
      delta.setData(MatrixUtils.order(delta.dataLength, 0.01f, 0.01f));
      
      for(int i = 0;i<1;i++) {
      		long start = System.nanoTime();
      		st.forward(input);
      		st.back(delta);
            JCuda.cudaDeviceSynchronize();
            System.err.println((System.nanoTime() - start)/1e6+"ms.");
      }

      st.getOutput().showShape();
      st.getOutput().showDM();
      
      st.diff.showDM("diff");
  }
    
}

