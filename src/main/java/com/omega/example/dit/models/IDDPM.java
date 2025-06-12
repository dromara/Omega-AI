package com.omega.example.dit.models;

import com.omega.common.utils.MatrixOperation;
import com.omega.common.utils.MatrixUtils;
import com.omega.common.utils.RandomUtils;
import com.omega.engine.gpu.BaseKernel;
import com.omega.engine.gpu.CUDAManager;
import com.omega.engine.nn.network.DiT;
import com.omega.engine.nn.network.DiT_ORG;
import com.omega.engine.nn.network.DiT_SRA;
import com.omega.engine.nn.network.DiffusionUNetCond2;
import com.omega.engine.tensor.Tensor;

public class IDDPM {
		
//	private int batchSize;
	
	private BetaType betaType = BetaType.linear; 
	
	private float[] betas;
	
//	private float[] newBetas;
	
	private float[] sqrt_alphas_cumprod;
	private Tensor sqrt_alphas_cumprod_t;
	
	private float[] sqrt_one_minus_alphas_cumprod;
	private Tensor sqrt_one_minus_alphas_cumprod_t;
	private Tensor log_one_minus_alphas_cumprod_t;
	
	private Tensor posterior_mean_coef1_t;
	private Tensor posterior_mean_coef2_t;
	
	private Tensor posterior_variance_t;
	private Tensor posterior_log_variance_clipped_t;
	
	private Tensor log_betas_t;
	
	private Tensor sqrt_recip_alphas_cumprod_t;
	private Tensor sqrt_recipm1_alphas_cumprod_t;
	
	private Tensor minLog;
	private Tensor maxLog;
	
	private Tensor kl;
	private Tensor kl_mean;
	private Tensor vb;
	
	private Tensor tureMean;
	private Tensor trueLogVar;
	private Tensor pred_xstart;
	private Tensor mean;
	private Tensor logvar;
	private Tensor decoder_nll;
	private Tensor decoder_nll_mean;
	
	private Tensor d_logvar1;
	private Tensor d_logvarKL;
	
	private int diffusion_steps = 1000;
	
	private CUDAManager cudaManager;
	
    private BaseKernel kernel;
    
    public IDDPMKernel iddpmKernel;
	
	public IDDPM(int diffusion_steps,BetaType betaType,CUDAManager cudaManager) {
		this.cudaManager = cudaManager;
		this.betaType = betaType;
		this.diffusion_steps = diffusion_steps;
		init();
		initBetaSchedule();
	}
	
	public void init() {
		if(kernel == null) {
			kernel = new BaseKernel(cudaManager);
		}
		if(iddpmKernel == null) {
			iddpmKernel = new IDDPMKernel(cudaManager);
		}
	}
	
	public double alphaBar(float t) {
		return Math.pow(Math.cos((t + 0.008) / 1.008 * Math.PI / 2), 2);
	}
	
	public float[] betas_for_alpha_bar() {
		float[] betas = new float[diffusion_steps];
		float max_beta = 0.999f;
		for(int i = 0;i<diffusion_steps;i++) {
			float t1 = i * 1.0f / diffusion_steps;
			float t2 = (i + 1) * 1.0f / diffusion_steps;
			betas[i] = (float) Math.min((1 - alphaBar(t2) / alphaBar(t1)), max_beta);
		}
		return betas;
	}
	
	public void initBetas() {
		
		if(betaType == BetaType.squaredcos) {
			this.betas = betas_for_alpha_bar();
		}else if(betaType == BetaType.scaled_linear){
			float scale = 1000 / diffusion_steps;
			float beta_start = (float) (scale * Math.pow(0.00085, 0.5));
			float beta_end = (float) (scale * Math.pow(0.012, 0.5));
			this.betas = MatrixUtils.linspace(beta_start, beta_end, diffusion_steps);
		}else {
			float scale = 1000 / diffusion_steps;
			float beta_start = scale * 0.0001f;
			float beta_end = scale * 0.02f;
			this.betas = MatrixUtils.linspace(beta_start, beta_end, diffusion_steps);
		}
//		System.err.println(JsonUtils.toJson(this.betas));
	}
	
	public void initBetaSchedule() {
		initBetas();
		log_betas_t = new Tensor(betas.length, 1, 1, 1, MatrixOperation.log(betas), true);
		float[] alphas = MatrixOperation.subtraction(1, betas);
        float[] alphas_cumprod = MatrixUtils.cumprod(alphas);
        sqrt_alphas_cumprod = MatrixOperation.sqrt(alphas_cumprod);
        sqrt_alphas_cumprod_t = new Tensor(sqrt_alphas_cumprod.length, 1, 1, 1, sqrt_alphas_cumprod, true);
        float[] one_sub = MatrixOperation.subtraction(1, alphas_cumprod);
        sqrt_one_minus_alphas_cumprod = MatrixOperation.sqrt(one_sub);
        sqrt_one_minus_alphas_cumprod_t = new Tensor(sqrt_one_minus_alphas_cumprod.length, 1, 1, 1, sqrt_one_minus_alphas_cumprod, true);
        float[] log_one_minus_alphas_cumprod = MatrixOperation.log(one_sub);
        log_one_minus_alphas_cumprod_t = new Tensor(log_one_minus_alphas_cumprod.length, 1, 1, 1, log_one_minus_alphas_cumprod, true);
        float[] one_div = MatrixOperation.division(1, alphas_cumprod);
        float[] sqrt_recip_alphas_cumprod =  MatrixOperation.sqrt(one_div);
        sqrt_recip_alphas_cumprod_t = new Tensor(sqrt_recip_alphas_cumprod.length, 1, 1, 1, sqrt_recip_alphas_cumprod, true);
        float[] sqrt_recipm1_alphas_cumprod =  MatrixOperation.sqrt(MatrixOperation.subtraction(one_div, 1));
        sqrt_recipm1_alphas_cumprod_t = new Tensor(sqrt_recipm1_alphas_cumprod.length, 1, 1, 1, sqrt_recipm1_alphas_cumprod, true);
        float[] alphas_cumprod_prev = new float[alphas_cumprod.length];
        alphas_cumprod_prev[0] = 1;
        System.arraycopy(alphas_cumprod, 0, alphas_cumprod_prev, 1, alphas_cumprod.length - 1);
        float[] alphas_cumprod_next = new float[alphas_cumprod.length];
        System.arraycopy(alphas_cumprod, 1, alphas_cumprod_next, 0, alphas_cumprod.length - 1);
        //calculations for posterior q(x_{t-1} | x_t, x_0)
        float[] posterior_variance = new float[alphas_cumprod.length];
        for(int i = 0;i<alphas_cumprod.length;i++) {
        	posterior_variance[i] = betas[i] * (1.0f - alphas_cumprod_prev[i]) / (1.0f - alphas_cumprod[i]);
        }
        posterior_variance_t = new Tensor(posterior_variance.length, 1, 1, 1, posterior_variance, true);

        float[] posterior_log_variance_clipped = new float[alphas_cumprod.length];
        System.arraycopy(posterior_variance, 1, posterior_log_variance_clipped, 1, alphas_cumprod.length - 1);
        posterior_log_variance_clipped[0] = posterior_variance[1];
        posterior_log_variance_clipped = MatrixOperation.log(posterior_log_variance_clipped);
        posterior_log_variance_clipped_t = new Tensor(posterior_log_variance_clipped.length, 1, 1, 1, posterior_log_variance_clipped, true);

        float[] posterior_mean_coef1 = new float[alphas_cumprod.length];
        float[] posterior_mean_coef2 = new float[alphas_cumprod.length];
        for(int i = 0;i<alphas_cumprod.length;i++) {
        	posterior_mean_coef1[i] = (float) (betas[i] * Math.sqrt(alphas_cumprod_prev[i]) / (1.0f - alphas_cumprod[i]));
        	posterior_mean_coef2[i] = (float) ((1.0f - alphas_cumprod_prev[i]) * Math.sqrt(alphas[i]) / (1.0f - alphas_cumprod[i]));
        }
        posterior_mean_coef1_t = new Tensor(posterior_mean_coef1.length, 1, 1, 1, posterior_mean_coef1, true);
        posterior_mean_coef2_t = new Tensor(posterior_mean_coef2.length, 1, 1, 1, posterior_mean_coef2, true);
	}
	
	public void getAB(Tensor a, Tensor b, int[] t_data) {
		 float[] exsa1 = MatrixUtils.gather(sqrt_alphas_cumprod, t_data);
         float[] exsa2 = MatrixUtils.gather(sqrt_one_minus_alphas_cumprod, t_data);
         a.setData(exsa1);
         b.setData(exsa2);
	}
	
	public Tensor p_sample(DiT network, Tensor cos, Tensor sin, Tensor noiseInput, Tensor noise, Tensor condInput, Tensor t, Tensor predMean, Tensor predVar) {
		
		RandomUtils.gaussianRandom(noiseInput, 0, 1);
		
		Tensor xt = noiseInput;
		
		for(int time = diffusion_steps - 1;time>=0;time--) {
	        for (int i = 0; i < noiseInput.number; i++) {
	            t.data[i] = time;
	        }
	        t.hostToDevice();
	        Tensor pred = network.forward(xt, t, condInput, cos, sin);
	        network.tensorOP.getByChannel(pred, predMean, 0, 4);
            network.tensorOP.getByChannel(pred, predVar, 4, 4);
            
            p_mean_variance(predMean, predVar, t, xt, pred_xstart, predMean, predVar);
            
            network.tensorOP.mul(predVar, 0.5f, noiseInput);
        	network.tensorOP.exp(noiseInput, noiseInput);
            
            if(time > 0) {
            	RandomUtils.gaussianRandom(noise, 0, 1);
            	network.tensorOP.mul(noiseInput, noise, noiseInput);
            	network.tensorOP.add(predMean, noiseInput, noiseInput);
            }else {
            	network.tensorOP.add(predMean, noiseInput, noiseInput);
            }
            
            xt = noiseInput;
            System.err.println("p_sample:" + time);
		}
		
		return noiseInput;
	}
	
	public Tensor p_sample(DiT_ORG network, Tensor cos, Tensor sin, Tensor noiseInput, Tensor noise, Tensor condInput, Tensor t, Tensor predMean, Tensor predVar) {
		
		RandomUtils.gaussianRandom(noiseInput, 0, 1);
		
		Tensor xt = noiseInput;
		
		for(int time = diffusion_steps - 1;time>=0;time--) {
	        for (int i = 0; i < noiseInput.number; i++) {
	            t.data[i] = time;
	        }
	        t.hostToDevice();
	        Tensor pred = network.forward(xt, t, condInput, cos, sin);
	        network.tensorOP.getByChannel(pred, predMean, 0, 4);
            network.tensorOP.getByChannel(pred, predVar, 4, 4);
            
            p_mean_variance(predMean, predVar, t, xt, pred_xstart, predMean, predVar);
            
            network.tensorOP.mul(predVar, 0.5f, noiseInput);
        	network.tensorOP.exp(noiseInput, noiseInput);
            
            if(time > 0) {
            	RandomUtils.gaussianRandom(noise, 0, 1);
            	network.tensorOP.mul(noiseInput, noise, noiseInput);
            	network.tensorOP.add(predMean, noiseInput, noiseInput);
            }else {
            	network.tensorOP.add(predMean, noiseInput, noiseInput);
            }
            
            xt = noiseInput;
            System.err.println("p_sample:" + time);
		}
		
		return noiseInput;
	}
	
	public Tensor p_sample(DiT_SRA network, Tensor cos, Tensor sin, Tensor noiseInput, Tensor noise, Tensor condInput, Tensor t, Tensor predMean, Tensor predVar) {
		
		RandomUtils.gaussianRandom(noiseInput, 0, 1);
		
		Tensor xt = noiseInput;
		
		for(int time = diffusion_steps - 1;time>=0;time--) {
	        for (int i = 0; i < noiseInput.number; i++) {
	            t.data[i] = time;
	        }
	        t.hostToDevice();
	        Tensor pred = network.forward(xt, t, condInput, cos, sin);
	        network.tensorOP.getByChannel(pred, predMean, 0, 4);
            network.tensorOP.getByChannel(pred, predVar, 4, 4);
            
            p_mean_variance(predMean, predVar, t, xt, pred_xstart, predMean, predVar);
            
            network.tensorOP.mul(predVar, 0.5f, noiseInput);
        	network.tensorOP.exp(noiseInput, noiseInput);
            
            if(time > 0) {
            	RandomUtils.gaussianRandom(noise, 0, 1);
            	network.tensorOP.mul(noiseInput, noise, noiseInput);
            	network.tensorOP.add(predMean, noiseInput, noiseInput);
            }else {
            	network.tensorOP.add(predMean, noiseInput, noiseInput);
            }
            
            xt = noiseInput;
            System.err.println("p_sample:" + time);
		}
		
		return noiseInput;
	}
	
	public Tensor p_sample(DiffusionUNetCond2 network, Tensor noiseInput, Tensor noise, Tensor condInput, Tensor t, Tensor predMean, Tensor predVar) {
		
		RandomUtils.gaussianRandom(noiseInput, 0, 1);
		
		Tensor xt = noiseInput;
		
		for(int time = diffusion_steps - 1;time>=0;time--) {
	        for (int i = 0; i < noiseInput.number; i++) {
	            t.data[i] = time;
	        }
	        t.hostToDevice();
	        Tensor pred = network.forward(xt, t, condInput);
	        network.tensorOP.getByChannel(pred, predMean, 0, 4);
            network.tensorOP.getByChannel(pred, predVar, 4, 4);
            
            p_mean_variance(predMean, predVar, t, xt, pred_xstart, predMean, predVar);
            
            network.tensorOP.mul(predVar, 0.5f, noiseInput);
        	network.tensorOP.exp(noiseInput, noiseInput);
            
            if(time > 0) {
            	RandomUtils.gaussianRandom(noise, 0, 1);
            	network.tensorOP.mul(noiseInput, noise, noiseInput);
            	network.tensorOP.add(predMean, noiseInput, noiseInput);
            }else {
            	network.tensorOP.add(predMean, noiseInput, noiseInput);
            }
            
            xt = noiseInput;
            System.err.println("p_sample:" + time);
		}
		
		return noiseInput;
	}
	
	public void q_sample(Tensor latend,Tensor noise,Tensor output,Tensor t) {
//		sqrt_alphas_cumprod_t.showDM("sqrt_alphas_cumprod_t");
//		sqrt_one_minus_alphas_cumprod_t.showDM("sqrt_one_minus_alphas_cumprod_t");
		iddpmKernel.add_mul(sqrt_alphas_cumprod_t, sqrt_one_minus_alphas_cumprod_t, latend, noise, t, output);
	}
	
	public void q_posterior_mean_variance(Tensor xStart,Tensor xt,Tensor t,Tensor posterior_mean,Tensor posterior_log_variance_clipped) {
		iddpmKernel.add_mul(posterior_mean_coef1_t, posterior_mean_coef2_t, xStart, xt, t, posterior_mean);
		iddpmKernel.extract_into(posterior_log_variance_clipped_t, t, posterior_log_variance_clipped);
	}
	
	public void q_posterior_mean_variance(Tensor xStart,Tensor xt,Tensor t,Tensor posterior_mean,Tensor posterior_variance,Tensor posterior_log_variance_clipped) {
		iddpmKernel.add_mul(posterior_mean_coef1_t, posterior_mean_coef2_t, xStart, xt, t, posterior_mean);
		iddpmKernel.extract_into(posterior_variance_t, t, posterior_variance);
		iddpmKernel.extract_into(posterior_log_variance_clipped_t, t, posterior_log_variance_clipped);
	}
	
	public Tensor min_log(Tensor t,Tensor x) {
		if(minLog == null || minLog.checkShape(x)){
			minLog = Tensor.createGPUTensor(minLog, x.shape(), true);
		}
		iddpmKernel.extract_into(posterior_log_variance_clipped_t, t, minLog);
		return minLog;
	}
	
	public Tensor max_log(Tensor t,Tensor x) {
		if(maxLog == null || maxLog.checkShape(x)){
			maxLog = Tensor.createGPUTensor(maxLog, x.shape(), true);
		}
		iddpmKernel.extract_into(log_betas_t, t, maxLog);
		return maxLog;
	}
	
	public void model_log_variance(Tensor var, Tensor max_log, Tensor min_log,Tensor output) {
		iddpmKernel.model_log_variance(var, max_log, min_log, output);
	}

	public void predict_xstart_from_eps(Tensor xt,Tensor t,Tensor eps,Tensor output) {
		iddpmKernel.sub_mul(sqrt_recip_alphas_cumprod_t, sqrt_recipm1_alphas_cumprod_t, xt, eps, t, output);
	}
	
	public void p_mean_variance(Tensor out_m,Tensor out_v,Tensor t,Tensor xt,Tensor pred_xstart,Tensor model_mean,Tensor model_log_variance) {
		Tensor minLog = min_log(t, out_v);
		Tensor maxLog = max_log(t, out_v);
		iddpmKernel.model_log_variance(out_v, maxLog, minLog, model_log_variance);
		predict_xstart_from_eps(xt, t, out_m, pred_xstart);
		iddpmKernel.add_mul(posterior_mean_coef1_t, posterior_mean_coef2_t, pred_xstart, xt, t, model_mean);
	}
	
	public Tensor normal_kl(Tensor tureMean,Tensor trueLogVar,Tensor mean,Tensor logvar) {
		if(kl == null || kl.checkShape(mean)) {
			kl = Tensor.createGPUTensor(kl, mean.shape(), true);
		}
		iddpmKernel.normal_kl(tureMean, trueLogVar, mean, logvar, kl);
		return kl;
	}
	
	public Tensor normal_kl_back(Tensor tureMean,Tensor trueLogVar,Tensor mean,Tensor logvar) {
		if(d_logvarKL == null || d_logvarKL.checkShape(logvar)) {
			d_logvarKL = Tensor.createGPUTensor(d_logvarKL, logvar.shape(), true);
		}
		iddpmKernel.normal_kl_back(tureMean, trueLogVar, mean, logvar, d_logvarKL);
		return d_logvarKL;
	}
	
	public Tensor discretized_gaussian_log_likelihood(Tensor xStart, Tensor mean, Tensor logvar) {
		if(decoder_nll == null || decoder_nll.checkShape(xStart)) {
			decoder_nll = Tensor.createGPUTensor(decoder_nll, xStart.shape(), true);
		}
//		xStart.showDM("xStart");
		iddpmKernel.discretized_gaussian_log_likelihood(xStart, mean, logvar, decoder_nll);
//		decoder_nll.showDM("decoder_nll");
		return decoder_nll;
	}
	
	public Tensor discretized_gaussian_log_likelihood_back(Tensor xStart, Tensor mean, Tensor logvar) {
		if(d_logvar1 == null || d_logvar1.checkShape(xStart)) {
			d_logvar1 = Tensor.createGPUTensor(d_logvar1, xStart.shape(), true);
		}
		iddpmKernel.discretized_gaussian_log_likelihood_back(xStart, mean, logvar, d_logvar1);
		return d_logvar1;
	}
	
	public Tensor vb_terms_bpd(Tensor out_m,Tensor out_v,Tensor t,Tensor xStart,Tensor xt) {
		if(vb == null || vb.checkShape(out_m)) {
			vb = Tensor.createGPUTensor(vb, out_m.number, 1, 1, 1, true);
		}
		if(tureMean == null || tureMean.checkShape(out_m)) {
			tureMean = Tensor.createGPUTensor(tureMean, out_m.shape(), true);
			trueLogVar = Tensor.createGPUTensor(trueLogVar, out_m.shape(), true);
			pred_xstart = Tensor.createGPUTensor(pred_xstart, out_m.shape(), true);
			mean = Tensor.createGPUTensor(mean, out_m.shape(), true);
			logvar = Tensor.createGPUTensor(logvar, out_m.shape(), true);
			kl_mean = Tensor.createGPUTensor(kl_mean, out_m.number, 1, 1, 1, true);
			decoder_nll_mean = Tensor.createGPUTensor(decoder_nll_mean, out_m.number, 1, 1, 1, true);
		}
		
		q_posterior_mean_variance(xStart, xt, t, tureMean, trueLogVar);

		p_mean_variance(out_m, out_v, t, xt, pred_xstart, mean, logvar);

		Tensor kl = normal_kl(tureMean, trueLogVar, mean, logvar);

		iddpmKernel.mean(kl, kl_mean);
//		kl_mean.showDM("kl_mean");

		Tensor decoder_nll = discretized_gaussian_log_likelihood(xStart, mean, logvar) ;
		
//		decoder_nll.showDM("decoder_nll");
		
		iddpmKernel.mean(decoder_nll, decoder_nll_mean);
//		decoder_nll_mean.showDM("decoder_nll_mean");
		
		iddpmKernel.where(decoder_nll_mean, kl_mean, t, vb);
		
		return vb;
	}
	
	public Tensor vb_terms_bpd_back(Tensor xStart,Tensor t) {
		Tensor d_logvar1 = discretized_gaussian_log_likelihood_back(xStart, mean, logvar);
		Tensor d_logvarKL = normal_kl_back(tureMean, trueLogVar, mean, logvar);
//		d_logvarKL.showDM("d_logvarKL");
		iddpmKernel.where(d_logvar1, d_logvarKL, t, d_logvarKL);
//		d_logvarKL.showDM("d_logvarKL2");
		iddpmKernel.dvar_back(maxLog, minLog, d_logvarKL, d_logvarKL);
		
		return d_logvarKL;
	}
	
}
