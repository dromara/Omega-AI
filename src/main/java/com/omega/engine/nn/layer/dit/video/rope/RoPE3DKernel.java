package com.omega.engine.nn.layer.dit.video.rope;

import static jcuda.driver.JCudaDriver.cuLaunchKernel;

import java.util.Map;

import com.omega.engine.gpu.BaseKernel;
import com.omega.engine.gpu.CUDAManager;
import com.omega.engine.nn.network.CNN;
import com.omega.engine.tensor.Tensor;
import com.omega.example.common.ModeLoaderlUtils;
import com.omega.example.transformer.utils.LagJsonReader;

import jcuda.Pointer;
import jcuda.driver.CUfunction;

public class RoPE3DKernel extends BaseKernel {
	
    /**
     * 向前方法
     */
    private CUfunction forward_function;
    /**
     * 反向传播方法
     */
    private CUfunction backward_function;
    
    private int CAFFE_CUDA_NUM_THREADS = 1024;
    /**
     * 前向方法参数
     */
    private Pointer forwardParameters;
    private Pointer backwardParameters;

    public RoPE3DKernel(CUDAManager cudaManager) {
        super(cudaManager);
        init();
    }
    
    public void initFunction() {
        try {
            if (forward_function == null) {
                forward_function = getCudaManager().getLocalFunctionByModule("RoPE3DKernel.cu", "rope_3d_norm_igone");
            }
            if (backward_function == null) {
                backward_function = getCudaManager().getLocalFunctionByModule("RoPE3DKernel.cu", "rope_3d_back_igone");
            }
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
    
    public void init() {
        /**
         * 初始化cuda函数

         */
        initFunction();
    }
	
	public static int[][] PositionGetter3D(int T, int H, int W){
		
		int[][] poses = new int[3][T * H * W];
		
		int idx = 0;
		for(int t = 0;t<T;t++) {
			for(int h = 0;h<H;h++) {
				for(int w = 0;w<W;w++) {
					poses[0][idx] = t;
					poses[1][idx] = h;
					poses[2][idx] = w;
					idx++;
				}
			}
		}
		
//		System.err.println(JsonUtils.toJson(poses[1][109]));
		return poses;
	}
	
	public static float[][][] get_cos_sin(int D, int seq_len, float interpolation_scale){
		
		float[][][] cs = new float[2][seq_len][D];
		
		double[] inv_freq = new double[D/2];
		for(int i = 0;i<D/2;i++) {
			inv_freq[i] = 1.0 / Math.pow(10000, i * 2.0f / D);
		}
		
		double[] t = new double[seq_len];
		for(int i = 0;i<seq_len;i++) {
			t[i] = i / interpolation_scale;
		}
		
		double[][] freqs = new double[seq_len][D];
		for(int ti = 0;ti<seq_len;ti++) {
			for(int i = 0;i<D/2;i++) {
				double v = t[ti] * inv_freq[i];
				freqs[ti][i] = v;
				freqs[ti][i+D/2] = v;
			}
		}
		for(int ti = 0;ti<seq_len;ti++) {
			for(int i = 0;i<D;i++) {
				cs[0][ti][i] = (float) Math.cos(freqs[ti][i]);
				cs[1][ti][i] = (float) Math.sin(freqs[ti][i]);
			}
		}
//		System.err.println(JsonUtils.toJson(inv_freq));
		return cs;
	}
	
	public static float[][] embbd_rope(float[][][] cs, int[] poses) {
		int seq_len = poses.length;
		int D = cs[0][0].length;
//		System.err.println(D+":"+seq_len);
		float[][] cs_emb = new float[2][seq_len * D];
		for(int i = 0;i<seq_len * D;i++) {
			int pi = i / D;
			int si = i % D;
//			System.out.println(pi+":"+si+":"+poses[pi]);
			cs_emb[0][i] = cs[0][poses[pi]][si];
			cs_emb[1][i] = cs[1][poses[pi]][si];
//			if(pi == 187) {
//				System.err.println(poses[pi]);
//				System.err.println(JsonUtils.toJson(cs[0][poses[pi]]));
//				System.err.println(JsonUtils.toJson(cs[1][poses[pi]]));
//			}
		}
		return cs_emb;
	}
	
	public static Tensor[][] init3DRoPE(int T, int H, int W, int hiddenSize, int headNum, float interpolation_scale_t, float interpolation_scale_h, float interpolation_scale_w){
		int D = hiddenSize / headNum / 3;
		Tensor[][] cs = new Tensor[2][3];
		int[][] pos_thw = PositionGetter3D(T, H, W);
		int seq_len = pos_thw[0].length;
		float[][][] t_cs = get_cos_sin(D, T, interpolation_scale_t);
		float[][][] h_cs = get_cos_sin(D, H, interpolation_scale_h);
		float[][][] w_cs = get_cos_sin(D, W, interpolation_scale_w);

		float[][] t_cse = embbd_rope(t_cs, pos_thw[0]);
		float[][] h_cse = embbd_rope(h_cs, pos_thw[1]);
		float[][] w_cse = embbd_rope(w_cs, pos_thw[2]);
		Tensor t_cos = new Tensor(seq_len, 1, 1, D, t_cse[0], true);
		Tensor t_sin = new Tensor(seq_len, 1, 1, D, t_cse[1], true);
		Tensor h_cos = new Tensor(seq_len, 1, 1, D, h_cse[0], true);
		Tensor h_sin = new Tensor(seq_len, 1, 1, D, h_cse[1], true);
		Tensor w_cos = new Tensor(seq_len, 1, 1, D, w_cse[0], true);
		Tensor w_sin = new Tensor(seq_len, 1, 1, D, w_cse[1], true);
		cs[0][0] = t_cos;
		cs[0][1] = h_cos;
		cs[0][2] = w_cos;
		cs[1][0] = t_sin;
		cs[1][1] = h_sin;
		cs[1][2] = w_sin;
		return cs;
	}
	
    public void forward3d(Tensor t_cos, Tensor t_sin,Tensor h_cos, Tensor h_sin,Tensor w_cos, Tensor w_sin, Tensor input, Tensor output, int T, int HN, int HS, int igone) {
        try {
            /**
             * float* x, float* out, float* t_cos, float* t_sin, float* h_cos, float* h_sin, float* w_cos, float* w_sin, int N, int T, int headNum, int headSize, int igoneIdx
             */
            forwardParameters = Pointer.to(Pointer.to(input.getGpuData()), Pointer.to(output.getGpuData()), Pointer.to(t_cos.getGpuData()), Pointer.to(t_sin.getGpuData()), Pointer.to(h_cos.getGpuData()), Pointer.to(h_sin.getGpuData()), Pointer.to(w_cos.getGpuData()), Pointer.to(w_sin.getGpuData()), Pointer.to(new int[]{input.dataLength/2/3}), Pointer.to(new int[]{T}), Pointer.to(new int[]{HN}), Pointer.to(new int[]{HS}), Pointer.to(new int[]{igone}));

            checkCUDA(cuLaunchKernel(forward_function, this.CAFFE_GET_BLOCKS(input.dataLength/2/3), 1, 1,      // Grid dimension
            		CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    forwardParameters, null // Kernel- and extra parameters
            ));
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
    
    public void backward3d(Tensor t_cos, Tensor t_sin,Tensor h_cos, Tensor h_sin,Tensor w_cos, Tensor w_sin, Tensor delta, Tensor diff,int T,int HN,int HS, int igone) {
        try {
           
            /**
             * float* delta, float* diff, float* t_cos, float* t_sin, float* h_cos, float* h_sin, float* w_cos, float* w_sin, int N, int T, int headNum,int headSize, int igoneIdx
             */
        	backwardParameters = Pointer.to(Pointer.to(delta.getGpuData()), Pointer.to(diff.getGpuData()), Pointer.to(t_cos.getGpuData()), Pointer.to(t_sin.getGpuData()), Pointer.to(h_cos.getGpuData()), Pointer.to(h_sin.getGpuData()), Pointer.to(w_cos.getGpuData()), Pointer.to(w_sin.getGpuData()), Pointer.to(new int[]{delta.dataLength/2/3}), Pointer.to(new int[]{T}), Pointer.to(new int[]{HN}), Pointer.to(new int[]{HS}), Pointer.to(new int[]{igone}));

            checkCUDA(cuLaunchKernel(backward_function, this.CAFFE_GET_BLOCKS(delta.dataLength/2/3), 1, 1,      // Grid dimension
            		CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    backwardParameters, null // Kernel- and extra parameters
            ));
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
	
	public static void main(String[] args) {
//		int[][] pos_thw = PositionGetter3D(3, 11, 20);
//		System.err.println(JsonUtils.toJson(pos_thw[0]));
//		System.err.println(JsonUtils.toJson(pos_thw[1]));
//		System.err.println(JsonUtils.toJson(pos_thw[2]));
//		
//		float[][][] cs = get_cos_sin(24, 3, 1);
//		System.err.println(JsonUtils.toJson(cs[0]));
//		System.err.println(JsonUtils.toJson(cs[1]));
//		
//		float[][] t_cs = embbd_rope(cs, pos_thw[0]);
//		System.err.println(JsonUtils.toJson(t_cs[0]));
//		System.err.println(JsonUtils.toJson(t_cs[1]));
		
		
	    int N = 2;
	    int C = 77 + 660;
	    int HD = 72;
	    int F = 3;
	    int H = 11;
	    int W = 20;
	    int hiddenSize = 1152;
	    int headNum = 16;
	    Tensor[][] cs = RoPE3DKernel.init3DRoPE(F, H, W, hiddenSize, headNum, 1.0f, 1.4375f, 2.5f);
	    Tensor[] cos = cs[0];
	    Tensor[] sin = cs[1];

		String inputPath = "D:\\models\\ltx_vae\\rope_x.json";
	    Map<String, Object> datas = LagJsonReader.readJsonFileSmallWeight(inputPath);
	    Tensor input = new Tensor(N, headNum, C, HD, true);
	    ModeLoaderlUtils.loadData(input, datas, "rope_x");
	    
//	    input.showDM();
	    
	    Tensor qr = new Tensor(N, headNum, C, HD, true);
	    
	    CNN nn = new CNN(null);
        nn.CUDNN = true;
        nn.number = N;
	    
	    RoPE3DKernel kernel = new RoPE3DKernel(nn.cudaManager);
	    
//	    System.err.println(JsonUtils.toJson(cos[0].data));
//	    System.err.println(JsonUtils.toJson(sin[0].data));
//	    System.err.println(JsonUtils.toJson(cos[1].data));
//	    System.err.println(JsonUtils.toJson(sin[1].data));
//	    System.err.println(JsonUtils.toJson(cos[2].data));
//	    System.err.println(JsonUtils.toJson(sin[2].data));
	    
	    System.err.println(C+":"+headNum+":"+HD);
	    
	    kernel.forward3d(cos[0], sin[0], cos[1], sin[1], cos[2], sin[2], input, qr, C, headNum, HD/3, 77);
	    
	    cos[1].showShape("cos_h");
	    cos[0].showDMByOffsetRed(337 * 24, 24, "cos_t");
	    cos[1].showDMByOffsetRed(337 * 24, 24, "cos_h");
	    
	    sin[0].showDMByOffsetRed(337 * 24, 24, "sin_t");
	    sin[1].showDMByOffsetRed(337 * 24, 24, "sin_h");
	    
	    input.showDMByOffsetRed(1 * HD, HD, "input");
	    qr.showDMByOffsetRed(1 * HD, HD, "qr");
	    qr.showDM();
//	    Tensor qr2 = new Tensor(N, headNum, C, HD, true);
//	    
//	    kernel.forward3d2(cos[0], sin[0], cos[1], sin[1], cos[2], sin[2], input, qr2, C, headNum, HD/3, 77);
//	    
//	    qr2.showDMByOffsetRed(337 * HD, HD, "qr2");
	    
	}
	
}
