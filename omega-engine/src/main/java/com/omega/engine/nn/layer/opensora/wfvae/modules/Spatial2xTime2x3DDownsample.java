package com.omega.engine.nn.layer.opensora.wfvae.modules;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.util.Map;

import com.omega.common.utils.MatrixUtils;
import com.omega.engine.gpu.PaddingKernel;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.ParamsInit;
import com.omega.engine.nn.network.CNN;
import com.omega.engine.nn.network.Network;
import com.omega.engine.tensor.Tensor;
import com.omega.engine.updater.UpdaterFactory;
import com.omega.example.common.ModeLoaderlUtils;
import com.omega.example.transformer.utils.LagJsonReader;

import jcuda.runtime.JCuda;

/**
 * Spatial2xTime2x3DDownsample
 *
 * @author Administrator
 */
public class Spatial2xTime2x3DDownsample extends Layer {

	private PaddingKernel paddingKernel;

    public WFCausalConv3D conv;

    public int depth;
    public int oDepth;
    
    private int[] padding3d = new int[] {0, 1, 0, 1, 0, 0};
    
    private int pDepth;
    private int pHeight;
    private int pWidth;
    
    private Tensor pOutput;
    
    public Spatial2xTime2x3DDownsample(int channel, int oChannel, int depth, int height, int width, Network network) {
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
    	
    	paddingKernel = new PaddingKernel(cuda());
        
    	this.pDepth = this.depth + padding3d[4] + padding3d[5];
        this.pHeight = this.height + padding3d[2] + padding3d[3];
        this.pWidth = this.width + padding3d[0] + padding3d[1];

        conv = new WFCausalConv3D(channel, oChannel, pDepth, pWidth, pHeight, 3, 2, 0, true, network);
        conv.setUpdater(UpdaterFactory.create(this.network));
        conv.paramsInit = ParamsInit.silu;
       
    }

    @Override
    public void init() {
        this.number = this.network.number;
        if(this.pOutput == null || this.number != this.pOutput.number) {
        	this.pOutput = Tensor.createGPUTensor(pOutput, number, channel * pDepth, pHeight, pWidth, true);
        }
    }

    @Override
    public void initBack() {
        if(this.diff == null || this.number != this.diff.number) {
        	this.diff = new Tensor(number, channel * depth, height, width, true);
        }
    }

    @Override
    public void initParam() {
        // TODO Auto-generated method stub
    }

    @Override
    public void output() {
        // TODO Auto-generated method stub
    	paddingKernel.padding3d(input, pOutput, depth, padding3d, 0);
    	conv.forward(pOutput);
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
        conv.back(delta, pOutput);
        paddingKernel.padding3dGrad(pOutput, diff, depth, padding3d);
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
    
    public static void loadWeight(Map<String, Object> weightMap, Spatial2xTime2x3DDownsample network, boolean showLayers) {
        if (showLayers) {
            for (String key : weightMap.keySet()) {
                System.out.println(key);
            }
        }

        ModeLoaderlUtils.loadData(network.conv.weight, weightMap, "conv.conv.weight", 5);
        ModeLoaderlUtils.loadData(network.conv.bias, weightMap, "conv.conv.bias");
        
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
      ModeLoaderlUtils.loadData(input, datas, "x", 5);

      CNN nn = new CNN(null);
      nn.CUDNN = true;
      nn.number = N;
      //int channel,int kernelNum,int depth,int width,int height,int kDepth,int kWidth,int kHeight,int padding,int stride
      Spatial2xTime2x3DDownsample st = new Spatial2xTime2x3DDownsample(C, OC, F, H, W, nn);
      
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

