package com.omega.example.dit.models;

import java.util.Map;

import com.omega.common.utils.RandomUtils;
import com.omega.engine.ad.op.TensorOP;
import com.omega.engine.nn.network.CNN;
import com.omega.engine.tensor.Tensor;
import com.omega.example.clip.utils.ClipModelUtils;
import com.omega.example.transformer.utils.LagJsonReader;

public class ICPlan {
	
//	private float sigma = 0.0f;
	
	private TensorOP op;
	
	private ICPlanKernel kernel;
	
	private Tensor norm1;
	private Tensor norm2;
	
	public ICPlan(TensorOP op) {
		this.op = op;
		init();
	}
	
	public void init() {
		if(kernel == null) {
			kernel = new ICPlanKernel(op.cudaManager);
		}
	}
	
	public static void cos_f() {
		
		int N = 2;
		int C = 4;
		int H = 32;
		int W = 32;
		
		String inputPath = "D:\\models\\x.json";
	    Map<String, Object> datas = LagJsonReader.readJsonFileSmallWeight(inputPath);
	    Tensor input = new Tensor(N, C, H, W, true);
	    ClipModelUtils.loadData(input, datas, "x");
		
	    String input2Path = "D:\\models\\x2.json";
	    Map<String, Object> datas2 = LagJsonReader.readJsonFileSmallWeight(input2Path);
	    Tensor input2 = new Tensor(N, C, H, W, true);
	    ClipModelUtils.loadData(input2, datas2, "x2");
	    
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
	    ClipModelUtils.loadData(input, datas, "x");
		
	    String input2Path = "D:\\models\\noise.json";
	    Map<String, Object> datas2 = LagJsonReader.readJsonFileSmallWeight(input2Path);
	    Tensor noise = new Tensor(N, C, H, W, true);
	    ClipModelUtils.loadData(noise, datas2, "noise");
	    
	    String tPath = "D:\\models\\t.json";
	    Map<String, Object> tdata = LagJsonReader.readJsonFileSmallWeight(tPath);
	    Tensor t = new Tensor(N, 1, 1, 1, true);
	    ClipModelUtils.loadData(t, tdata, "t");
	    
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
		
		plan();
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
	
	public void cosine_similarity_loss_back(Tensor x1,Tensor x2,Tensor dx1) {
		float delta = 1.0f / x1.number / x1.height / x1.width;
		kernel.cosine_similarity_loss_back1(delta, x1, norm1, x2, norm2, dx1);
//		dx1.showDM("dx1");
		kernel.cosine_similarity_loss_back2(delta, x1, norm1, x2, norm2, norm2);
//		norm2.showDM("dnorm1");
		op.normalizeKernel.l2norm1Dim2_back(x1, norm2, x2);
//		x2.showDM("dnorm1");
		op.add(x2, dx1, dx1);
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
	
}
