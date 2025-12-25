package com.omega.example.dit.models;

import java.util.Map;

import com.omega.common.utils.MatrixUtils;
import com.omega.common.utils.RandomUtils;
import com.omega.engine.ad.op.TensorOP;
import com.omega.engine.nn.network.CNN;
import com.omega.engine.nn.network.Transformer;
import com.omega.engine.nn.network.dit.DiT_ORG;
import com.omega.engine.nn.network.dit.DiT_TXT;
import com.omega.engine.nn.network.dit.FluxDiT;
import com.omega.engine.nn.network.dit.FluxDiT2;
import com.omega.engine.nn.network.dit.FluxDiT3;
import com.omega.engine.nn.network.dit.FluxDiT_REPA;
import com.omega.engine.nn.network.dit.FluxDiT_SPRINT;
import com.omega.engine.nn.network.dit.JiT;
import com.omega.engine.nn.network.dit.JiT_REPA;
import com.omega.engine.nn.network.dit.MMDiT_RoPE;
import com.omega.engine.nn.network.dit.SanaDiT;
import com.omega.engine.tensor.Tensor;
import com.omega.example.common.ModeLoaderlUtils;
import com.omega.example.transformer.utils.LagJsonReader;

public class ICPlan {
	
//	private float sigma = 0.0f;
	
	private TensorOP op;
	
	private ICPlanKernel kernel;
	
	private Tensor norm1;
	private Tensor norm2;
	
	private float[] T;
	
	private float timestep_shift = 0.3f;
	
//	private float atol = 1e-6f;
//	private float rtol = 1e-3f;
	
	private int count = 250;
	
	private Tensor tmp;
	
	private Tensor t_input;
	private Tensor t_next_input;
	private Tensor z_next_euler;
	private Tensor z_next;
	
	public ICPlan(TensorOP op) {
		this.op = op;
		init();
	}
	
	public ICPlan(TensorOP op, int sample_count, float timestep_shift) {
		this.op = op;
		this.count = sample_count;
		this.setTimestep_shift(timestep_shift);
		init();
	}
	
	public void init() {
		if(kernel == null) {
			kernel = new ICPlanKernel(op.cudaManager);
		}
	}
	
	public void ininT(int start,int end,int count) {
		if(T == null) {
			T = MatrixUtils.linspace(start, end, count);
//			System.err.println(timestep_shift);
			if(timestep_shift > 0) {
				for(int i = 0;i<T.length;i++) {
					float numerator = timestep_shift * T[i];
					float denominator = 1 + (timestep_shift - 1) * T[i];
					T[i] = numerator / denominator;
				}
				
			}
		}
	}
	
	/**
	 * sample
	 * @param y0 is noise
	 */
	public Tensor sample(MMDiT_RoPE dit,Tensor y0, Tensor t, Tensor context, Tensor cos, Tensor sin, Tensor y1) {
		ininT(0, 1, count);//250
		int j = 1;
		Tensor out = null;
		Tensor f0 = null;
		for(int i = 0;i<count - 1;i++) {
			float t0 = T[i];
			float t1 = T[i + 1];
			float dt = t1 - t0;
			MatrixUtils.val(t.data, t0);
			t.hostToDevice();
			f0 = dit.forward(y0, t, context, cos, sin);
//			f0 = dit.tensorOP.copyTensorGPU(y0, f0);
			
			dit.tensorOP.mul(f0, dt, f0);
			
			dit.tensorOP.add(y0, f0, y1);
			float tj = T[j];
			if(j < T.length && t1 >= tj) {
				out = linear_interp(dit.tensorOP, t0, t1, y0, y1, tj);
				j++;
			}
			dit.tensorOP.copyGPU(y1, y0);
//			y0.showDM();
		}
		return out;
	}
	
	/**
	 * sample
	 * @param y0 is noise
	 */
	public Tensor sample(FluxDiT3 dit,Tensor y0, Tensor t, Tensor context, Tensor cos, Tensor sin, Tensor y1) {
		ininT(0, 1, count);
		int j = 1;
		Tensor out = null;
		Tensor f0 = null;
		for(int i = 0;i<count - 1;i++) {
			float t0 = T[i];
			float t1 = T[i + 1];
			float dt = t1 - t0;
			MatrixUtils.val(t.data, t0);
			t.hostToDevice();
			f0 = dit.forward(y0, t, context, cos, sin);
//			f0 = dit.tensorOP.copyTensorGPU(y0, f0);
			
			dit.tensorOP.mul(f0, dt, f0);
			
			dit.tensorOP.add(y0, f0, y1);
			float tj = T[j];
			if(j < T.length && t1 >= tj) {
				out = linear_interp(dit.tensorOP, t0, t1, y0, y1, tj);
				j++;
			}
			dit.tensorOP.copyGPU(y1, y0);
//			y0.showDM();
		}
		return out;
	}
	
	/**
	 * sample
	 * @param y0 is noise
	 */
	public Tensor sample(DiT_TXT dit,Tensor y0, Tensor t, Tensor context, Tensor cos, Tensor sin, Tensor y1) {
		ininT(0, 1, count);
		int j = 1;
		Tensor out = null;
		Tensor f0 = null;
		for(int i = 0;i<count - 1;i++) {
			float t0 = T[i];
			float t1 = T[i + 1];
			float dt = t1 - t0;
			MatrixUtils.val(t.data, t0);
			t.hostToDevice();
			f0 = dit.forward(y0, t, context, cos, sin);
//			f0 = dit.tensorOP.copyTensorGPU(y0, f0);
			
			dit.tensorOP.mul(f0, dt, f0);
			
			dit.tensorOP.add(y0, f0, y1);
			float tj = T[j];
			if(j < T.length && t1 >= tj) {
				out = linear_interp(dit.tensorOP, t0, t1, y0, y1, tj);
				j++;
			}
			dit.tensorOP.copyGPU(y1, y0);
//			y0.showDM();
		}
		return out;
	}
	
	public Tensor forward_with_cfg(MMDiT_RoPE dit, Tensor y0, Tensor t, Tensor context, Tensor cos, Tensor sin, Tensor y1, Tensor eps, float cfg_scale) {
		ininT(0, 1, count);
		int j = 1;
		Tensor out = null;
		Tensor f0 = null;
		for(int i = 0;i<count - 1;i++) {
			float t0 = T[i];
			float t1 = T[i + 1];
			float dt = t1 - t0;
			MatrixUtils.val(t.data, t0);
			t.hostToDevice();
			f0 = dit.forward_with_cfg(y0, t, context, cos, sin, eps, cfg_scale, 3);
//			f0.showDM("f0");
			dit.tensorOP.mul(f0, dt, f0);
			
			dit.tensorOP.add(y0, f0, y1);
			float tj = T[j];
			if(j < T.length && t1 >= tj) {
				out = linear_interp(dit.tensorOP, t0, t1, y0, y1, tj);
				j++;
			}
			dit.tensorOP.copyGPU(y1, y0);

		}
		return out;
	}
	
	public Tensor forward_with_cfg(DiT_TXT dit, Tensor y0, Tensor t, Tensor context, Tensor cos, Tensor sin, Tensor y1, Tensor eps, float cfg_scale) {
		ininT(0, 1, count);
		int j = 1;
		Tensor out = null;
		Tensor f0 = null;
		for(int i = 0;i<count - 1;i++) {
			float t0 = T[i];
			float t1 = T[i + 1];
			float dt = t1 - t0;
			MatrixUtils.val(t.data, t0);
			t.hostToDevice();
			f0 = dit.forward_with_cfg(y0, t, context, cos, sin, eps, cfg_scale, 3);
//			f0.showDM("f0");
			dit.tensorOP.mul(f0, dt, f0);
			
			dit.tensorOP.add(y0, f0, y1);
			float tj = T[j];
			if(j < T.length && t1 >= tj) {
				out = linear_interp(dit.tensorOP, t0, t1, y0, y1, tj);
				j++;
			}
			dit.tensorOP.copyGPU(y1, y0);

		}
		return out;
	}
	
	public Tensor forward_with_cfg(FluxDiT dit, Tensor y0, Tensor t, Tensor context, Tensor cos, Tensor sin, Tensor y1, Tensor eps, float cfg_scale) {
		ininT(0, 1, count);
		int j = 1;
		Tensor out = null;
		Tensor f0 = null;
		for(int i = 0;i<count - 1;i++) {
			float t0 = T[i];
			float t1 = T[i + 1];
			float dt = t1 - t0;
			MatrixUtils.val(t.data, t0);
			t.hostToDevice();
			f0 = dit.forward_with_cfg(y0, t, context, cos, sin, eps, cfg_scale);
//			f0 = dit.forward_with_cfg(y0, t, context, cos, sin, eps, cfg_scale, 3);
//			f0.showDM("f0");
			dit.tensorOP.mul(f0, dt, f0);
			
			dit.tensorOP.add(y0, f0, y1);
			float tj = T[j];
			if(j < T.length && t1 >= tj) {
				out = linear_interp(dit.tensorOP, t0, t1, y0, y1, tj);
				j++;
			}
			dit.tensorOP.copyGPU(y1, y0);

		}
		return out;
	}
	
	public Tensor forward_with_cfg(FluxDiT_REPA dit, Tensor y0, Tensor t, Tensor context, Tensor cos, Tensor sin, Tensor y1, Tensor eps, float cfg_scale) {
		ininT(0, 1, count);
		int j = 1;
		Tensor out = null;
		Tensor f0 = null;
		for(int i = 0;i<count - 1;i++) {
			float t0 = T[i];
			float t1 = T[i + 1];
			float dt = t1 - t0;
			MatrixUtils.val(t.data, t0);
			t.hostToDevice();
			f0 = dit.forward_with_cfg(y0, t, context, cos, sin, eps, cfg_scale);
//			f0.showDM("f0");
			dit.tensorOP.mul(f0, dt, f0);
			
			dit.tensorOP.add(y0, f0, y1);
			float tj = T[j];
			if(j < T.length && t1 >= tj) {
				out = linear_interp(dit.tensorOP, t0, t1, y0, y1, tj);
				j++;
			}
			dit.tensorOP.copyGPU(y1, y0);

		}
		return out;
	}
	
	public Tensor forward_with_cfg(FluxDiT_SPRINT dit, Tensor y0, Tensor t, Tensor context, Tensor cos, Tensor sin, Tensor y1, Tensor eps, float cfg_scale) {
		ininT(0, 1, count);
		int j = 1;
		Tensor out = null;
		Tensor f0 = null;
		for(int i = 0;i<count - 1;i++) {
			float t0 = T[i];
			float t1 = T[i + 1];
			float dt = t1 - t0;
			MatrixUtils.val(t.data, t0);
			t.hostToDevice();
			f0 = dit.forward_with_cfg(y0, t, context, cos, sin, eps, cfg_scale);
//			f0.showDM("f0");
			dit.tensorOP.mul(f0, dt, f0);
			
			dit.tensorOP.add(y0, f0, y1);
			float tj = T[j];
			if(j < T.length && t1 >= tj) {
				out = linear_interp(dit.tensorOP, t0, t1, y0, y1, tj);
				j++;
			}
			dit.tensorOP.copyGPU(y1, y0);

		}
		return out;
	}
	
	public Tensor forward_with_cfg(FluxDiT2 dit, Tensor y0, Tensor t, Tensor context, Tensor cos, Tensor sin, Tensor y1, Tensor eps, float cfg_scale) {
		ininT(0, 1, count);
		int j = 1;
		Tensor out = null;
		Tensor f0 = null;
		for(int i = 0;i<count - 1;i++) {
			float t0 = T[i];
			float t1 = T[i + 1];
			float dt = t1 - t0;
			MatrixUtils.val(t.data, t0);
			t.hostToDevice();
			f0 = dit.forward_with_cfg(y0, t, context, cos, sin, eps, cfg_scale, 3);
//			f0.showDM("f0");
			dit.tensorOP.mul(f0, dt, f0);
			
			dit.tensorOP.add(y0, f0, y1);
			float tj = T[j];
			if(j < T.length && t1 >= tj) {
				out = linear_interp(dit.tensorOP, t0, t1, y0, y1, tj);
				j++;
			}
			dit.tensorOP.copyGPU(y1, y0);

		}
		return out;
	}
	
	public Tensor forward_with_cfg(FluxDiT3 dit, Tensor y0, Tensor t, Tensor context, Tensor cos, Tensor sin, Tensor y1, Tensor eps, float cfg_scale) {
		ininT(0, 1, count);
		int j = 1;
		Tensor out = null;
		Tensor f0 = null;
		for(int i = 0;i<count - 1;i++) {
			float t0 = T[i];
			float t1 = T[i + 1];
			float dt = t1 - t0;
			MatrixUtils.val(t.data, t0);
			t.hostToDevice();
			f0 = dit.forward_with_cfg(y0, t, context, cos, sin, eps, cfg_scale, 3);
//			f0.showDM("f0");
			dit.tensorOP.mul(f0, dt, f0);
			
			dit.tensorOP.add(y0, f0, y1);
			float tj = T[j];
			if(j < T.length && t1 >= tj) {
				out = linear_interp(dit.tensorOP, t0, t1, y0, y1, tj);
				j++;
			}
			dit.tensorOP.copyGPU(y1, y0);

		}
		return out;
	}
	
	public Tensor forward_with_cfg(SanaDiT dit, Tensor y0, Tensor t, Tensor context, Tensor y1, Tensor eps, float cfg_scale) {
		ininT(0, 1, count);
		int j = 1;
		Tensor out = null;
		Tensor f0 = null;
		for(int i = 0;i<count - 1;i++) {
			float t0 = T[i];
			float t1 = T[i + 1];
			float dt = t1 - t0;
			MatrixUtils.val(t.data, t0);
			t.hostToDevice();
			f0 = dit.forward_with_cfg(y0, t, context, eps, cfg_scale);
//			f0.showDM("f0");
			dit.tensorOP.mul(f0, dt, f0);
			
			dit.tensorOP.add(y0, f0, y1);
			float tj = T[j];
			if(j < T.length && t1 >= tj) {
				out = linear_interp(dit.tensorOP, t0, t1, y0, y1, tj);
				j++;
			}
			dit.tensorOP.copyGPU(y1, y0);

		}
		return out;
	}
	
	public Tensor forward_with_cfg(JiT jit, Tensor noise, Tensor context, Tensor cos, Tensor sin, float cfg_scale) {
		
		if(t_input == null || t_input.number != noise.number) {
			t_input = new Tensor(noise.number * 2, 1, 1, 1, true);
			t_next_input = new Tensor(noise.number * 2, 1, 1, 1, true);
			z_next_euler = Tensor.createGPUTensor(z_next_euler, noise.number, noise.channel, noise.height, noise.width, true);
			z_next = Tensor.createGPUTensor(z_next, noise.number, noise.channel, noise.height, noise.width, true);
		}
		
		ininT(0, 1, count + 1);
		Tensor x_next = null;
		for(int i = 0;i<count - 1;i++) {
			float t_ = T[i];
			float t_next_ = T[i + 1];
			MatrixUtils.val(t_input.data, t_);
			MatrixUtils.val(t_next_input.data, t_next_);
			t_input.hostToDevice();
			t_next_input.hostToDevice();
			x_next = heun_step(jit, noise, t_input, t_next_input, t_, t_next_, context, cos, sin, z_next_euler, z_next, cfg_scale);
			jit.tensorOP.copyGPU(x_next, noise);
		}
		
		float t_ = T[count - 2];
		float t_next_ = T[count - 1];
		MatrixUtils.val(t_input.data, t_);
		t_input.hostToDevice();
		x_next = euler_step(jit, noise, t_input, t_, t_next_, context, cos, sin, z_next, cfg_scale);
		
		return x_next;
	}
	
	public Tensor forward_with_cfg(JiT_REPA jit, Tensor noise, Tensor context, Tensor cos, Tensor sin, float cfg_scale) {
		
		if(t_input == null || t_input.number != noise.number) {
			t_input = new Tensor(noise.number * 2, 1, 1, 1, true);
			t_next_input = new Tensor(noise.number * 2, 1, 1, 1, true);
			z_next_euler = Tensor.createGPUTensor(z_next_euler, noise.number, noise.channel, noise.height, noise.width, true);
			z_next = Tensor.createGPUTensor(z_next, noise.number, noise.channel, noise.height, noise.width, true);
		}
		
		ininT(0, 1, count + 1);
		Tensor x_next = null;
		for(int i = 0;i<count - 1;i++) {
			float t_ = T[i];
			float t_next_ = T[i + 1];
			MatrixUtils.val(t_input.data, t_);
			MatrixUtils.val(t_next_input.data, t_next_);
			t_input.hostToDevice();
			t_next_input.hostToDevice();
			x_next = heun_step(jit, noise, t_input, t_next_input, t_, t_next_, context, cos, sin, z_next_euler, z_next, cfg_scale);
			jit.tensorOP.copyGPU(x_next, noise);
		}
		
		float t_ = T[count - 2];
		float t_next_ = T[count - 1];
		MatrixUtils.val(t_input.data, t_);
		t_input.hostToDevice();
		x_next = euler_step(jit, noise, t_input, t_, t_next_, context, cos, sin, z_next, cfg_scale);
		
		return x_next;
	}
	
	public Tensor forward_with_cfg(Tensor noise, float cfg_scale) {
		
		if(t_input == null || t_input.number != noise.number) {
			t_input = new Tensor(noise.number * 2, 1, 1, 1, true);
			t_next_input = new Tensor(noise.number * 2, 1, 1, 1, true);
			z_next_euler = Tensor.createGPUTensor(z_next_euler, noise.number, noise.channel, noise.height, noise.width, true);
			z_next = Tensor.createGPUTensor(z_next, noise.number, noise.channel, noise.height, noise.width, true);
		}
		
		ininT(0, 1, count + 1);
		Tensor x_next = null;
		for(int i = 0;i<count - 1;i++) {
			float t_ = T[i];
			float t_next_ = T[i + 1];
			MatrixUtils.val(t_input.data, t_);
			MatrixUtils.val(t_next_input.data, t_next_);
			t_input.hostToDevice();
			t_next_input.hostToDevice();
			x_next = heun_step(noise, t_input, t_next_input, t_, t_next_, z_next_euler, z_next, cfg_scale);

			op.copyGPU(x_next, noise);
		}
		
		float t_ = T[count - 2];
		float t_next_ = T[count - 1];
		MatrixUtils.val(t_input.data, t_);
		t_input.hostToDevice();
		x_next = euler_step(noise, t_input, t_, t_next_, z_next, cfg_scale);
		
		return x_next;
	}
	
	
	/**
	 * sample
	 * @param y0 is noise
	 */
	public Tensor sample(DiT_ORG dit,Tensor y0, Tensor t, Tensor context, Tensor cos, Tensor sin, Tensor y1) {
		ininT(0, 1, count);
		int j = 1;
		Tensor out = null;
		Tensor f0 = null;
		for(int i = 0;i<count - 1;i++) {
			float t0 = T[i];
			float t1 = T[i + 1];
			float dt = t1 - t0;
			MatrixUtils.val(t.data, t0);
			t.hostToDevice();
			f0 = dit.forward(y0, t, context, cos, sin);

			dit.tensorOP.mul(f0, dt, f0);
			
			dit.tensorOP.add(y0, f0, y1);
			
			float tj = T[j];
			if(j < T.length && t1 >= tj) {
				out = linear_interp(dit.tensorOP, t0, t1, y0, y1, tj);
				j++;
			}
			dit.tensorOP.copyGPU(y1, y0);
		}
		return out;
	}
	
	public Tensor linear_interp(TensorOP tensorOP,float t0, float t1, Tensor y0, Tensor y1, float t) {
		if(t == t0) {
			return y0;
		}
		if(t == t1) {
			return y1;
		}
		float slope = (t - t0) / (t1 - t0);
		if(tmp == null) {
			tmp = Tensor.createGPUTensor(tmp, y0.shape(), true);
		}
		tensorOP.sub(y1, y0, tmp);
		tensorOP.mul(tmp, slope, tmp);
		tensorOP.add(y0, tmp, tmp);
		return tmp;
	}
	
	public static void cos_f() {
		
		int N = 2;
		int C = 4;
		int H = 32;
		int W = 32;
		
		String inputPath = "D:\\models\\x.json";
	    Map<String, Object> datas = LagJsonReader.readJsonFileSmallWeight(inputPath);
	    Tensor input = new Tensor(N, C, H, W, true);
	    ModeLoaderlUtils.loadData(input, datas, "x");
		
	    String input2Path = "D:\\models\\x2.json";
	    Map<String, Object> datas2 = LagJsonReader.readJsonFileSmallWeight(input2Path);
	    Tensor input2 = new Tensor(N, C, H, W, true);
	    ModeLoaderlUtils.loadData(input2, datas2, "x2");
	    
	    Tensor loss = new Tensor(N, C, H, W, true);
	    
	    CNN nn = new CNN(null);
        nn.CUDNN = true;
        nn.number = N;
	    
	    ICPlan k = new ICPlan(nn.tensorOP);
	    
	    k.cosine_similarity_loss(input, input2, loss);
	    
	    loss.showDM();
	    
	    k.cosine_similarity_loss_back(input, input2, loss);
	    
	    loss.showDM();
	}
	
	public static void plan() {
		
		int N = 2;
		int C = 4;
		int H = 32;
		int W = 32;
		
		String inputPath = "D:\\models\\x.json";
	    Map<String, Object> datas = LagJsonReader.readJsonFileSmallWeight(inputPath);
	    Tensor input = new Tensor(N, C, H, W, true);
	    ModeLoaderlUtils.loadData(input, datas, "x");
		
	    String input2Path = "D:\\models\\noise.json";
	    Map<String, Object> datas2 = LagJsonReader.readJsonFileSmallWeight(input2Path);
	    Tensor noise = new Tensor(N, C, H, W, true);
	    ModeLoaderlUtils.loadData(noise, datas2, "noise");
	    
	    String tPath = "D:\\models\\t.json";
	    Map<String, Object> tdata = LagJsonReader.readJsonFileSmallWeight(tPath);
	    Tensor t = new Tensor(N, 1, 1, 1, true);
	    ModeLoaderlUtils.loadData(t, tdata, "t");
	    
	    CNN nn = new CNN(null);
        nn.CUDNN = true;
        nn.number = N;
	    
	    ICPlan k = new ICPlan(nn.tensorOP);
	    
	    Tensor xt = new Tensor(N, C, H, W, true);
	    Tensor ut = new Tensor(N, C, H, W, true);
	    
	    k.compute_xt(t, noise, input, xt);
	    
	    k.compute_ut(t, noise, input, ut);
	    t.showDM();
	    xt.showDM();
	    ut.showDM();
	}
	
	public void latend_norm(Tensor x,Tensor mean,Tensor std) {
		kernel.latend_norm(x, mean, std);
	}
	 
	public void latend_un_norm(Tensor x,Tensor mean,Tensor std) {
		kernel.latend_un_norm(x, mean, std);
	}
	
	public static void main(String[] args) {
		
//		cos_f();
		
//		plan();
		
		int N = 2;
		int C = 3;
		int H = 16;
		int W = 16;
		
		Transformer tf = new Transformer();
		
		ICPlan icplan = new ICPlan(tf.tensorOP, 50, 0);
		
		String xPath = "D:\\models\\he_x.json";
	    Map<String, Object> xDatas = LagJsonReader.readJsonFileSmallWeight(xPath);
	    Tensor x = new Tensor(N, C, H, W, true);
	    ModeLoaderlUtils.loadData(x, xDatas, "x");
		
	    x.showDM("x");
	    
	    Tensor img = icplan.forward_with_cfg(x, C);

	    img.showDM("img");
	    
//		String xPath = "D:\\models\\cos_x.json";
//	    Map<String, Object> xDatas = LagJsonReader.readJsonFileSmallWeight(xPath);
//	    Tensor x = new Tensor(N, C, H, W, true);
//	    ModeLoaderlUtils.loadData(x, xDatas, "x");
//		
//		String utPath = "D:\\models\\cos_ut.json";
//	    Map<String, Object> utDatas = LagJsonReader.readJsonFileSmallWeight(utPath);
//	    Tensor ut = new Tensor(N, C, H, W, true);
//	    ModeLoaderlUtils.loadData(ut, utDatas, "ut");
//	    
//	    Tensor loss = new Tensor(N, 1, H, W, true);
//		
//	    Tensor dx = new Tensor(N, C, H, W, true);
//	    for(int i = 0;i<10;i++) {
//	    	icplan.cosine_similarity_loss_dim1(x, ut, loss);
//			loss.showDM("loss");
//			icplan.cosine_similarity_loss_back(x, ut, dx);
//			dx.showDM("dx");
//	    }
//		
//	    Tensor w12 = new Tensor(2, 1, 1, 8, true);
//	    Tensor w1 = new Tensor(2, 1, 1, 4, true);
//	    Tensor w2 = new Tensor(2, 1, 1, 4, true);
//	    
//	    RandomUtils.xavier_uniform(w12, 1, 4, 2 * 2);
//	    
//	    int[] shape = new int[] {2, 2, 1, 4};
//	    
//	    tf.tensorOP.getByChannel(w12, w1, shape, 0);
//	    tf.tensorOP.getByChannel(w12, w2, shape, 1);
////	    tf.tensorOP.cat_width_back(w12, w1, w2);
//	    
//	    w12.showDM("w12");
//	    w1.showDM("w1");
//	    w2.showDM("w2");
	}
	
	public void cosine_similarity_loss(Tensor x1,Tensor x2,Tensor loss) {
		if(norm1 == null || norm1.number != x1.number) {
			norm1 = Tensor.createGPUTensor(norm1, x1.number, 1, x1.height, x1.width, true);
			norm2 = Tensor.createGPUTensor(norm2, x1.number, 1, x1.height, x1.width, true);
		}
		op.normalizeKernel.l2norm1Dim2(x1, norm1);
//		norm1.showDM();
		op.normalizeKernel.l2norm1Dim2(x2, norm2);
//		x2.showDM();
		kernel.cosine_similarity_loss(x1, norm1, x2, norm2, loss);
//		loss.showDM();
	}
	
	public void cosine_similarity_loss_dim1(Tensor x1,Tensor x2,Tensor loss) {
		if(norm1 == null || norm1.number != x1.number) {
			norm1 = Tensor.createGPUTensor(norm1, x1.number, 1, x1.height, x1.width, true);
			norm2 = Tensor.createGPUTensor(norm2, x1.number, 1, x1.height, x1.width, true);
		}
		op.normalizeKernel.l2norm1Dim2(x1, norm1);
//		norm1.showDM();
		op.normalizeKernel.l2norm1Dim2(x2, norm2);
//		norm2.showDM();
		kernel.cosine_similarity_loss_dim1(x1, norm1, x2, norm2, loss);
//		loss.showDM();
	}
	
	public void cosine_similarity_loss_back(Tensor x1,Tensor x2,Tensor dx1) {
//		x1.showDM("x1");
		float delta = 1.0f / x1.number / x1.height / x1.width;
		kernel.cosine_similarity_loss_back1(delta, x1, norm1, x2, norm2, dx1);
//		dx1.showDM("dx1");
		kernel.cosine_similarity_loss_back2(delta, x1, norm1, x2, norm2, norm2);
//		norm2.showDM("dnorm1");
		op.normalizeKernel.l2norm1Dim2_back_plus(x1, norm2, dx1);
//		x2.showDM("dnorm1");
//		op.add(x2, dx1, dx1);
	}
	
	public void t(Tensor t) {
		RandomUtils.gaussianRandomLogitNormal(t);
	}
	
	/**
	 * 
	 * @param t
	 * @param x0 is noise
	 * @param x1 is org latend
	 * alpha_t = t
	 * sigma_t = 1 - t
	 * xt = alpha_t * x1 + sigma_t * x0
	 */
	public void compute_xt(Tensor t,Tensor x0, Tensor x1, Tensor xt) {
		kernel.compute_xt(x1, x0, t, xt);
	}
	
	/**
	 * 
	 * @param t
	 * @param x0 is noise
	 * @param x1 is org latend
	 * @param ut is velocity
	 * d_alpha_t = 1
	 * d_sigma_t = -1
	 * ut = d_alpha_t * x1 + d_sigma_t * x0
	 */
	public void compute_ut(Tensor t,Tensor x0, Tensor x1,Tensor ut) {
		kernel.compute_ut(x1, x0, t, ut);
	}
	
	public void sample_t(Tensor t, float mean, float std) {
		RandomUtils.gaussianRandomLogitNormal(t, mean, std);
	}
	
	public void compute_z(Tensor x, Tensor t, Tensor noise,Tensor z) {
		kernel.compute_xt(x, noise, t, z);
	}
	
	public void compute_v(Tensor x, Tensor t,Tensor z, Tensor v, float t_eps) {
		kernel.compute_v(x, z, t, v, t_eps);
	}
	
	public void compute_dv(Tensor delta,Tensor t,Tensor dx, float t_eps) {
		kernel.compute_dv(delta, t, dx, t_eps);
	}
	
	public Tensor euler_step(JiT network, Tensor z, Tensor t, float t_, float t_next, Tensor context, Tensor cos, Tensor sin, Tensor z_next, float cfg_scale) {
		if(tmp == null || tmp.number != z.number) {
			tmp = Tensor.createGPUTensor(tmp, z.shape(), true);
		}
		Tensor v_pred = network.forward_with_cfg(this, z, t, context, cos, sin, tmp, cfg_scale);
		kernel.compute_z_next(v_pred, z, t_, t_next, z_next);
		return z_next;
	}
	
	public Tensor heun_step(JiT network, Tensor z, Tensor t, Tensor t_next, float t_, float t_next_, Tensor context, Tensor cos, Tensor sin, Tensor z_next_euler, Tensor z_next, float cfg_scale) {
		
		if(tmp == null || tmp.number != z.number) {
			tmp = Tensor.createGPUTensor(tmp, z.shape(), true);
		}
		
		Tensor v_pred_t = network.forward_with_cfg(this, z, t, context, cos, sin, tmp, cfg_scale);
		//z_next_euler = z + (t_next - t) * v_pred_t
		kernel.compute_z_next(v_pred_t, z, t_, t_next_, z_next_euler);

		Tensor v_pred_next = network.forward_with_cfg(this, z_next_euler, t_next, context, cos, sin, tmp, cfg_scale);
		
		op.add(v_pred_t, v_pred_next, v_pred_t);
		op.mul(v_pred_t, 0.5f, v_pred_t);
		//z_next = z + (t_next - t) * v_pred
		kernel.compute_z_next(v_pred_t, z, t_, t_next_, z_next);
		
		return z_next;
	}

	public Tensor euler_step(JiT_REPA network, Tensor z, Tensor t, float t_, float t_next, Tensor context, Tensor cos, Tensor sin, Tensor z_next, float cfg_scale) {
		if(tmp == null || tmp.number != z.number) {
			tmp = Tensor.createGPUTensor(tmp, z.shape(), true);
		}
		Tensor v_pred = network.forward_with_cfg(this, z, t, context, cos, sin, tmp, cfg_scale);
		kernel.compute_z_next(v_pred, z, t_, t_next, z_next);
		return z_next;
	}
	
	public Tensor heun_step(JiT_REPA network, Tensor z, Tensor t, Tensor t_next, float t_, float t_next_, Tensor context, Tensor cos, Tensor sin, Tensor z_next_euler, Tensor z_next, float cfg_scale) {
		
		if(tmp == null || tmp.number != z.number) {
			tmp = Tensor.createGPUTensor(tmp, z.shape(), true);
		}
		
		Tensor v_pred_t = network.forward_with_cfg(this, z, t, context, cos, sin, tmp, cfg_scale);
		//z_next_euler = z + (t_next - t) * v_pred_t
		kernel.compute_z_next(v_pred_t, z, t_, t_next_, z_next_euler);

		Tensor v_pred_next = network.forward_with_cfg(this, z_next_euler, t_next, context, cos, sin, tmp, cfg_scale);
		
		op.add(v_pred_t, v_pred_next, v_pred_t);
		op.mul(v_pred_t, 0.5f, v_pred_t);
		//z_next = z + (t_next - t) * v_pred
		kernel.compute_z_next(v_pred_t, z, t_, t_next_, z_next);
		
		return z_next;
	}
	
	public Tensor euler_step(Tensor z, Tensor t, float t_, float t_next, Tensor z_next, float cfg_scale) {
		if(tmp == null || tmp.number != z.number) {
			tmp = Tensor.createGPUTensor(tmp, z.shape(), true);
		}
		op.copyTensorGPU(z, tmp);
		Tensor v_pred = tmp;
		kernel.compute_z_next(v_pred, z, t_, t_next, z_next);
		return z_next;
	}
	
	public Tensor heun_step(Tensor z, Tensor t, Tensor t_next, float t_, float t_next_, Tensor z_next_euler, Tensor z_next, float cfg_scale) {
		
		if(tmp == null || tmp.number != z.number) {
			tmp = Tensor.createGPUTensor(tmp, z.shape(), true);
		}
		
		op.copyTensorGPU(z, tmp);
		
		Tensor v_pred_t = tmp;
		//z_next_euler = z + (t_next - t) * v_pred_t
		kernel.compute_z_next(v_pred_t, z, t_, t_next_, z_next_euler);

		Tensor v_pred_next = z_next_euler;
		
		op.add(v_pred_t, v_pred_next, v_pred_t);
		op.mul(v_pred_t, 0.5f, v_pred_t);

		//z_next = z + (t_next - t) * v_pred
		kernel.compute_z_next(v_pred_t, z, t_, t_next_, z_next);

		return z_next;
	}

	public void setTimestep_shift(float timestep_shift) {
		if(timestep_shift > 0) {
			this.timestep_shift = timestep_shift;
			T = null;
		}
	}
	
}
