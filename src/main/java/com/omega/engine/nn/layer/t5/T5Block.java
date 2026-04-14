package com.omega.engine.nn.layer.t5;

import java.io.IOException;
import java.io.RandomAccessFile;

import com.omega.common.utils.JsonUtils;
import com.omega.engine.nn.layer.EmbeddingIDLayer;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.normalization.BNType;
import com.omega.engine.nn.layer.normalization.LNLayer;
import com.omega.engine.nn.layer.t5.kernel.T5Kernel;
import com.omega.engine.nn.network.Network;
import com.omega.engine.tensor.Tensor;
import com.omega.engine.updater.UpdaterFactory;

/**
 * T5Block
 * @author Administrator
 */
public class T5Block extends Layer {
	
	private boolean has_relative_attention_bias = false;
	
	private int batchSize;
	private int headNum = 12;
	private int time = 0;
    private int embed_size = 0;
    private int d_ff = 1;
    private boolean bias = false;

    private int relative_attention_num_buckets = 32;
    
    public EmbeddingIDLayer relative_attention_bias;
    public LNLayer norm;
    public T5AttentionLayer attn;
    public T5LayerFF ffn;
    
    public T5Kernel kernel;
    
    private Tensor attention_bias;
    private Tensor position_bias_masked;
    private Tensor relativeBuckets;

    public T5Block(int headNum, int time,int embed_size, int d_ff, boolean bias, boolean has_relative_attention_bias) {
        this.embed_size = embed_size;
        this.d_ff = d_ff;
        this.headNum = headNum;
        this.time = time;
        this.bias = bias;
        this.oChannel = 1;
        this.oHeight = 1;
        this.oWidth = embed_size;
        this.has_relative_attention_bias = has_relative_attention_bias;
        this.initLayers();
    }

    public T5Block(int headNum, int time, int embed_size, int d_ff, boolean bias, boolean has_relative_attention_bias, Network network) {
        this.network = network;
        if (this.updater == null) {
            this.setUpdater(UpdaterFactory.create(network));
        }
        this.embed_size = embed_size;
        this.d_ff = d_ff;
        this.headNum = headNum;
        this.time = time;
        this.bias = bias;
        this.oChannel = 1;
        this.oHeight = 1;
        this.oWidth = embed_size;
        this.has_relative_attention_bias = has_relative_attention_bias;
        this.initLayers();
    }

    public static void main(String[] args) {
    	
    	int time = 120;
    	
    	float[] relative_buckets = new float[time * time];
    	int num_buckets = 32;
    	int max_distance = 128;
    	for(int i = 0;i<time;i++) {
    		for(int j = 0;j<time;j++) {
    			int tmp = j - i;
    			float relative_position = Math.abs(tmp);
    			int _num_buckets = num_buckets / 2;
    			if(tmp > 0) {
    				relative_buckets[i * time + j] = _num_buckets;
    			}else {
    				relative_buckets[i * time + j] = 0;
    			}
    			int max_exact = _num_buckets / 2;
    			float relative_position_if_large = (int) (max_exact + (Math.log(relative_position / max_exact) / Math.log(max_distance / max_exact) * (_num_buckets - max_exact)));
    			if(relative_position_if_large >= _num_buckets - 1) {
    				relative_position_if_large = _num_buckets - 1;
    			}
    			boolean is_small = relative_position < max_exact;
    			if(is_small) {
    				relative_buckets[i * time + j] += relative_position;
    			}else {
    				relative_buckets[i * time + j] += relative_position_if_large;
    			}
    		}
    	}
    	
    	System.err.println(JsonUtils.toJson(relative_buckets));
    }

    public void initLayers() {
    	if(has_relative_attention_bias) {
    		this.relative_attention_bias = new EmbeddingIDLayer(relative_attention_num_buckets, headNum, network);
    	}
        this.norm = new LNLayer(1, 1, embed_size, true, false, BNType.fully_bn, network);
        this.attn = new T5AttentionLayer(embed_size, headNum, time, bias, network);
        this.ffn = new T5LayerFF(embed_size, d_ff, bias, network);
        
        kernel = new T5Kernel(this.cuda());
    }
    
    public void createRelativeBuckets() {
    	float[] relative_buckets = new float[time * time];
    	int num_buckets = 32;
    	int max_distance = 128;
    	for(int i = 0;i<time;i++) {
    		for(int j = 0;j<time;j++) {
    			int tmp = j - i;
    			float relative_position = Math.abs(tmp);
    			int _num_buckets = num_buckets / 2;
    			if(tmp > 0) {
    				relative_buckets[i * time + j] = _num_buckets;
    			}else {
    				relative_buckets[i * time + j] = 0;
    			}
    			int max_exact = _num_buckets / 2;
    			float relative_position_if_large = (int) (max_exact + (Math.log(relative_position / max_exact) / Math.log(max_distance / max_exact) * (_num_buckets - max_exact)));
    			if(relative_position_if_large >= _num_buckets - 1) {
    				relative_position_if_large = _num_buckets - 1;
    			}
    			boolean is_small = relative_position < max_exact;
    			if(is_small) {
    				relative_buckets[i * time + j] += relative_position;
    			}else {
    				relative_buckets[i * time + j] += relative_position_if_large;
    			}
    		}
    	}
    	
    	relativeBuckets = new Tensor(time * time, 1, 1, 1, relative_buckets, true);
    	attention_bias = new Tensor(1, num_buckets, time, time, true);
//    	relativeBuckets.showDM("relativeBuckets");
    	relative_attention_bias.forward(relativeBuckets);
//    	relative_attention_bias.weight.showDM("weight");
//    	relative_attention_bias.getOutput().showDM("relative_attention_bias");
    	Tensor_OP().permute(relative_attention_bias.getOutput(), attention_bias, new int[] {1, time, time, num_buckets}, new int[] {1, num_buckets, time, time}, new int[] {0, 3, 1, 2});
    }
    
    @Override
    public void init() {
        // TODO Auto-generated method stub
        this.number = this.input.number;
//        System.err.println(number);
        this.batchSize = number / time;
        if(has_relative_attention_bias && attention_bias == null) {
        	createRelativeBuckets();
        }
        if(getPosition_bias_masked() == null || getPosition_bias_masked().number != batchSize) {
        	setPosition_bias_masked(Tensor.createGPUTensor(getPosition_bias_masked(), batchSize, headNum, time, time, true));
        }
    }

    @Override
    public void initBack() {
        // TODO Auto-generated method stub
    }

    @Override
    public void initParam() {
        // TODO Auto-generated method stub
    }

    @Override
    public void output() {
        // TODO Auto-generated method stub
    	
    }
    
    public void output(Tensor mask) {
    	if(has_relative_attention_bias) {
    		kernel.compute_bias(attention_bias, mask, position_bias_masked, headNum * time, time);
//        	position_bias_masked.showDM("position_bias_masked");
    	}
    	norm.forward_t5(input);
    	attn.forward(norm.getOutput(), position_bias_masked);
    	Tensor_OP().add(attn.getOutput(), input, attn.getOutput());
    	ffn.forward(attn.getOutput());
        this.output = ffn.getOutput();
    }
    
    @Override
    public Tensor getOutput() {
        // TODO Auto-generated method stub
        return output;
    }

    @Override
    public void diff() {
        // TODO Auto-generated method stub
    	
    }

    @Override
    public void forward() {
        // TODO Auto-generated method stub
        /**
         * 设置输入
         */
        this.setInput();
        /**
         * 参数初始化
         */
        this.init();
        /**
         * 计算输出
         */
        this.output();
    }

    @Override
    public void back() {
        // TODO Auto-generated method stub
        this.initBack();
        /**
         * 设置梯度
         */
        this.setDelta();
        /**
         * 计算梯度
         */
        this.diff();
        if (this.network.GRADIENT_CHECK) {
            this.gradientCheck();
        }
    }

    @Override
    public void forward(Tensor input) {
        // TODO Auto-generated method stub
        /**
         * 设置输入
         */
        this.setInput(input);
        /**
         * 参数初始化
         */
        this.init();
        /**
         * 计算输出
         */
        this.output();
    }
    
    public void forward(Tensor input, Tensor mask) {
        // TODO Auto-generated method stub
        /**
         * 设置输入
         */
        this.setInput(input);
        /**
         * 参数初始化
         */
        this.init();
        /**
         * 计算输出
         */
        this.output(mask);
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

    @Override
    public void update() {
        // TODO Auto-generated method stub

    }

    @Override
    public void showDiff() {
        // TODO Auto-generated method stub
    }

    @Override
    public LayerType getLayerType() {
        // TODO Auto-generated method stub
        return LayerType.mlp;
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
    	if(has_relative_attention_bias) {
    		relative_attention_bias.saveModel(outputStream);
    	}
        norm.saveModel(outputStream);
        attn.saveModel(outputStream);
        ffn.saveModel(outputStream);
    }

    public void loadModel(RandomAccessFile inputStream) throws IOException {
    	if(has_relative_attention_bias) {
    		relative_attention_bias.loadModel(inputStream);
    	}
    	norm.loadModel(inputStream);
    	attn.loadModel(inputStream);
    	ffn.loadModel(inputStream);
    }

    @Override
    public void accGrad(float scale) {
        // TODO Auto-generated method stub

    }

	public Tensor getPosition_bias_masked() {
		return position_bias_masked;
	}

	public void setPosition_bias_masked(Tensor position_bias_masked) {
		this.position_bias_masked = position_bias_masked;
	}
    
}

