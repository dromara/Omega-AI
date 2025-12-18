package com.omega.engine.nn.layer.dit.deco;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.util.ArrayList;
import java.util.List;
import com.omega.common.utils.RandomUtils;
import com.omega.engine.nn.layer.FullyLayer;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.network.Network;
import com.omega.engine.tensor.Tensor;
import com.omega.engine.updater.UpdaterFactory;

/**
 * SimpleMLPAdaLN
 * @author Administrator
 */
public class SimpleMLPAdaLN extends Layer {
    
    private int in_channels;
    private int model_channels;
    private int out_channels;
    private int z_channels;
    private int num_res_blocks;
    private int patch_size;
    
    public FullyLayer cond_embed;
    public FullyLayer input_proj;
    
    public List<ResBlock> res_blocks;
    
    public FinalLayer final_layer;
    
    private Tensor dy;
    
    public SimpleMLPAdaLN(int in_channels, int model_channels, int out_channels, int z_channels, int num_res_blocks, int patch_size, boolean bias, Network network) {
        this.network = network;
        if (this.updater == null) {
            this.setUpdater(UpdaterFactory.create(network));
        }
        this.in_channels = in_channels;
        this.model_channels = model_channels;
        this.out_channels = out_channels;
        this.z_channels = z_channels;
        this.num_res_blocks = num_res_blocks;
        this.patch_size = patch_size;
        this.hasBias = bias;
        initLayers();
        // 保持 channel 和 height 不变，只改变 width (in_chans -> embed_dim)
        this.oChannel = 1;
        this.oHeight = 1;
        this.oWidth = out_channels;
    }
    
    /**
     * 初始化子层
     */
    private void initLayers() {
        // 线性投影层: in_chans -> embed_dim
        // FullyLayer 的输入维度是 width，输出维度是 oWidth
        this.cond_embed = new FullyLayer(z_channels, patch_size * patch_size * model_channels, hasBias, network);
        RandomUtils.xavier_uniform(this.cond_embed.weight, 1, z_channels, patch_size * patch_size * model_channels);
        if(this.cond_embed.bias != null) {
        	this.cond_embed.bias.clearGPU();
        }
        this.input_proj = new FullyLayer(in_channels, model_channels, hasBias, network);
        RandomUtils.xavier_uniform(this.input_proj.weight, 1, in_channels,  model_channels);
        if(this.input_proj.bias != null) {
        	this.input_proj.bias.clearGPU();
        }
        
        res_blocks = new ArrayList<ResBlock>();
        for(int i = 0;i<num_res_blocks;i++) {
        	ResBlock res = new ResBlock(model_channels, hasBias, network);
        	res_blocks.add(res);
        }
        
        final_layer = new FinalLayer(model_channels, out_channels, hasBias, network);

    }

    @Override
    public void init() {
        this.number = network.number;
    }
    
    /**
     * 根据输入初始化
     *
     * @param input 输入张量
     */
    public void init(Tensor input) {
        this.number = input.number;
    }

    @Override
    public void initBack() {
    	if(dy == null || dy.number != this.number) {
    		dy = Tensor.createGPUTensor(dy, this.cond_embed.getOutput().shape(), true);
    	}else {
    		dy.clearGPU();
    	}
    }

    @Override
    public void initParam() {

    }

    @Override
    public void output() {

    }

    public void output(Tensor c) {
    	
    	input_proj.forward(input);
    	
    	cond_embed.forward(c);
    	
    	Tensor x = input_proj.getOutput();
    	Tensor y = cond_embed.getOutput();
    	
    	for(int i = 0;i<num_res_blocks;i++) {
    		ResBlock res = res_blocks.get(i);
    		res.forward(x, y);
    		x = res.getOutput();
    	}
    	
    	final_layer.forward(x);
    	
    	this.output = final_layer.getOutput();
    }
    
    @Override
    public Tensor getOutput() {
        return output;
    }

    @Override
    public void diff() {

    	final_layer.back(delta);
        
    	Tensor dx = final_layer.diff;
    	
    	for(int i = num_res_blocks - 1;i>=0;i--) {
    		ResBlock res = res_blocks.get(i);
    		res.back(dx, dy);
    		dx = res.diff;
    	}
    	
    	cond_embed.back(dy);
    	
    	input_proj.back(dx);
    	
        this.diff = input_proj.diff;
    }
    
    public Tensor getCDiff() {
    	return cond_embed.diff;
    }
    
    @Override
    public void forward() {
        this.init();
        this.setInput();
        this.output();
    }

    @Override
    public void back() {
        this.initBack();
        this.setDelta();
        this.diff();
        if (this.network.GRADIENT_CHECK) {
            this.gradientCheck();
        }
    }

    @Override
    public void forward(Tensor input) {
        this.init(input);
        this.setInput(input);
        this.output();
    }
    
    public void forward(Tensor input,Tensor c) {
        this.init(input);
        this.setInput(input);
        this.output(c);
    }

    @Override
    public void back(Tensor delta) {
        this.initBack();
        this.setDelta(delta);
        this.diff();
        if (this.network.GRADIENT_CHECK) {
            this.gradientCheck();
        }
    }

    @Override
    public void update() {
    	input_proj.update();
    	cond_embed.update();
    	for(int i = 0;i<num_res_blocks;i++) {
    		res_blocks.get(i).update();
    	}
    	final_layer.update();
    }

    @Override
    public void accGrad(float scale) {
    	input_proj.accGrad(scale);
    	cond_embed.accGrad(scale);
    	for(int i = 0;i<num_res_blocks;i++) {
    		res_blocks.get(i).accGrad(scale);
    	}
    	final_layer.accGrad(scale);
    }

    @Override
    public void showDiff() {
        // TODO Auto-generated method stub
    }

    @Override
    public LayerType getLayerType() {
        return LayerType.mlp;
    }

    @Override
    public float[][][][] output(float[][][][] input) {
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
    
    /**
     * 保存模型参数
     * 
     * @param outputStream 输出流
     * @throws IOException IO异常
     */
    public void saveModel(RandomAccessFile outputStream) throws IOException {
        input_proj.saveModel(outputStream);
    	cond_embed.saveModel(outputStream);
    	for(int i = 0;i<num_res_blocks;i++) {
    		res_blocks.get(i).saveModel(outputStream);
    	}
    	final_layer.saveModel(outputStream);
    }
    
    /**
     * 加载模型参数
     * 
     * @param inputStream 输入流
     * @throws IOException IO异常
     */
    public void loadModel(RandomAccessFile inputStream) throws IOException {
        input_proj.loadModel(inputStream);
    	cond_embed.loadModel(inputStream);
    	for(int i = 0;i<num_res_blocks;i++) {
    		res_blocks.get(i).loadModel(inputStream);
    	}
    	final_layer.loadModel(inputStream);
    }
    
}