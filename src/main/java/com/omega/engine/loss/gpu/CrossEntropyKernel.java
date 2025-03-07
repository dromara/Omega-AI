package com.omega.engine.loss.gpu;

import static jcuda.driver.JCudaDriver.cuLaunchKernel;

import com.omega.common.data.Tensor;
import com.omega.common.utils.JsonUtils;
import com.omega.common.utils.MatrixOperation;
import com.omega.common.utils.MatrixUtils;
import com.omega.common.utils.RandomUtils;
import com.omega.engine.gpu.BaseKernel;
import com.omega.engine.gpu.CUDAManager;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUfunction;
import jcuda.runtime.cudaError;

/**
 * CrossEntropyLoss = softmax + NLLLoss
 * 
 * softmax(x) = exp(x - m) / sum(exp(x - m))
 * m = ax(X)
 * 
 * NLLLoss(x) = -log(x)
 * 
 * detail:
 * 	loss = NLLLoss(softmax(x))
 * 	S = softmax(X)
 *  L = -log(S)
 *  loss = sum(L) / batch
 *  
 *  label is a one hot data
 *  label = [[0,0,1],[1,0,0],[0,1,0]]
 *  
 *  NLLLoss(S)' = - 1 / S
 *  softmax(x)' = softmax(x) * (label - softmax(x))
 *  x' = (softmax(x) - label) / batch

 * @author Administrator
 *
 */
public class CrossEntropyKernel extends BaseKernel {
	
	private CUfunction loss_function;
	
	private CUfunction nl_loss_function;
	
	private CUfunction log_softmax_nl_loss_function;
	
	private CUfunction log_softmax_nl_loss_igonre_function;
	
	private CUfunction log_softmax_nl_loss_igonre_idx_function;
	
	private CUfunction check_function;
	
	private CUfunction loss_backward_function;
	
	private CUfunction loss_igonre_backward_function;
	
	private CUfunction loss_igonre_backward_idx_function;
	
	private CUfunction softmax_function;
	
	private CUfunction cross_softmax_function;
	
	private CUfunction cross_softmax_back_function;
	
	private CUfunction cross_function;
	
	private CUfunction cross_igone_function;
	
	private CUfunction cross_backward_function;
	
	private CUfunction cross_igone_backward_function;
	
	private int CAFFE_CUDA_NUM_THREADS = 1024;
	
	private Pointer log_softmax_nl_loss_kernelParameters;
	
	private Pointer checkParameters;
	
	private Pointer backKernelParameters;
	
	private Pointer softmaxKernelParameters;
	
	private Pointer crossKernelParameters;
	
	private Pointer crossBackwardParameters;
	
	private Pointer crossSoftmaxForwardParameters;
	
	private Pointer crossSoftmaxBackwardParameters;
	
	public CrossEntropyKernel(CUDAManager cudaManager) {
		super(cudaManager);
		init();
	}
	
	public void initFunction() {
		
		try {

			if(loss_function == null) {
				
				loss_function = getCudaManager().getLocalFunctionByModule("CrossEntropyKernel.cu", "loss");
        
			}
			
			if(nl_loss_function == null) {
				
				nl_loss_function = getCudaManager().getLocalFunctionByModule("CrossEntropyKernel.cu", "nl_loss");
        
			}
			
			if(log_softmax_nl_loss_function == null) {
				
				log_softmax_nl_loss_function = getCudaManager().getLocalFunctionByModule("CrossEntropyKernel.cu", "log_softmax_nl_loss");
        
			}
			
			if(log_softmax_nl_loss_igonre_function == null) {
				
				log_softmax_nl_loss_igonre_function = getCudaManager().getLocalFunctionByModule("CrossEntropyKernel.cu", "log_softmax_nl_loss_igone");
        
			}
			
			if(log_softmax_nl_loss_igonre_idx_function == null) {
				
				log_softmax_nl_loss_igonre_idx_function = getCudaManager().getLocalFunctionByModule("CrossEntropyKernel.cu", "log_softmax_nl_loss_igone_idx");
        
			}
			
			if(check_function == null) {
				
				check_function = getCudaManager().getLocalFunctionByModule("CrossEntropyKernel.cu", "check");
        
			}
			
			if(loss_backward_function == null) {
				
				loss_backward_function = getCudaManager().getLocalFunctionByModule("CrossEntropyKernel.cu", "loss_back2");
        
			}
			
			if(loss_igonre_backward_function == null) {
				
				loss_igonre_backward_function = getCudaManager().getLocalFunctionByModule("CrossEntropyKernel.cu", "loss_back_igonre2");
        
			}
			
			if(loss_igonre_backward_idx_function == null) {
				loss_igonre_backward_idx_function = getCudaManager().getLocalFunctionByModule("CrossEntropyKernel.cu", "loss_back_igonre2_idx");
			}
			
			if(softmax_function == null) {
				softmax_function = getCudaManager().getLocalFunctionByModule("CrossEntropyKernel.cu", "softmax_forward_kernel7");
			}
			
			if(cross_function == null) {
				cross_function = getCudaManager().getLocalFunctionByModule("CrossEntropyKernel.cu", "crossentropy_forward_kernel");
			}
			
			if(cross_igone_function == null) {
				cross_igone_function = getCudaManager().getLocalFunctionByModule("CrossEntropyKernel.cu", "crossentropy_forward_kernel_igone");
			}
			
			if(cross_backward_function == null) {
				cross_backward_function = getCudaManager().getLocalFunctionByModule("CrossEntropyKernel.cu", "crossentropy_softmax_backward_kernel");
			}
			
			if(cross_igone_backward_function == null) {
				cross_igone_backward_function = getCudaManager().getLocalFunctionByModule("CrossEntropyKernel.cu", "crossentropy_softmax_igone_backward_kernel");
			}
			
			if(cross_softmax_function == null) {
				cross_softmax_function = getCudaManager().getLocalFunctionByModule("CrossEntropyKernel.cu", "cross_softmax_forward_kernel");
			}
			
			if(cross_softmax_back_function == null) {
				cross_softmax_back_function = getCudaManager().getLocalFunctionByModule("CrossEntropyKernel.cu", "cross_softmax_backward_kernel");
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
	
	public int CAFFE_GET_BLOCKS(int N) {
	    return (N + CAFFE_CUDA_NUM_THREADS - 1) / CAFFE_CUDA_NUM_THREADS;
	}
	
	public void softmax(Tensor input,Tensor output) {

			
			/**
			 * float* out, const float* inp, int N, int C
			 */
		softmaxKernelParameters = Pointer.to(
	                Pointer.to(output.getGpuData()),
	                Pointer.to(input.getGpuData()),
	                Pointer.to(new int[] {input.number}),
	                Pointer.to(new int[] {input.width})
	            );
			
		this.N = output.number;
		
		int grid_size = N;
		
		int shared_mem_size = 2 * CAFFE_CUDA_NUM_THREADS / 32 * Sizeof.FLOAT;
		
		cuLaunchKernel(softmax_function,
				grid_size,  1, 1,      // Grid dimension
				CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
	            shared_mem_size, null,               // Shared memory size and stream
	            softmaxKernelParameters, null // Kernel- and extra parameters
	        );
		
	}
	
	public void crossentropy(Tensor input,Tensor currentLabel,Tensor output) {

		/**
		 * float* losses,const float* probs, const int* targets,int B, int T, int V,const int igone
		 */
		crossKernelParameters = Pointer.to(
	                Pointer.to(output.getGpuData()),
	                Pointer.to(input.getGpuData()),
	                Pointer.to(currentLabel.getGpuData()),
	                Pointer.to(new int[] {currentLabel.number}),  //batch
	                Pointer.to(new int[] {currentLabel.width}), //time
	                Pointer.to(new int[] {input.width})  //dim_size
	            );
			
		this.N = currentLabel.number;
		
		int grid_size = (int) Math.ceil(currentLabel.number * currentLabel.width / CAFFE_CUDA_NUM_THREADS);
		if(grid_size <= 0) {
			grid_size = 1;
		}
		cuLaunchKernel(cross_function,
				grid_size,  1, 1,      // Grid dimension
				CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
	            0, null,               // Shared memory size and stream
	            crossKernelParameters, null // Kernel- and extra parameters
	        );
		
	}
	
	public void crossentropy_igone(Tensor input,Tensor currentLabel,Tensor output,int igonre) {
		
		/**
		 * float* losses,const float* probs, const int* targets,int B, int T, int V,const int igone
		 */
		crossKernelParameters = Pointer.to(
	                Pointer.to(output.getGpuData()),
	                Pointer.to(input.getGpuData()),
	                Pointer.to(currentLabel.getGpuData()),
	                Pointer.to(new int[] {currentLabel.number}),  //batch
	                Pointer.to(new int[] {currentLabel.width}), //time
	                Pointer.to(new int[] {input.width}),  //dim_size
	                Pointer.to(new int[] {igonre})
	            );
			
		this.N = input.number;
		
		int grid_size = (int) Math.ceil(input.number / CAFFE_CUDA_NUM_THREADS);
		
		cuLaunchKernel(cross_igone_function,
				grid_size,  1, 1,      // Grid dimension
				CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
	            0, null,               // Shared memory size and stream
	            crossKernelParameters, null // Kernel- and extra parameters
	        );

	}
	
	public void crossentropy_backward(Tensor input,Tensor currentLabel,Tensor diff) {

		/**
		 * float* dlogits, const float* probs, const int* targets,int B, int T, int V
		 */
		crossBackwardParameters = Pointer.to(
	                Pointer.to(diff.getGpuData()),
	                Pointer.to(input.getGpuData()),
	                Pointer.to(currentLabel.getGpuData()),
	                Pointer.to(new int[] {input.number}),  //batch
	                Pointer.to(new int[] {currentLabel.width}), //time
	                Pointer.to(new int[] {input.width})  //dim_size
	            );
			
		this.N = input.number;
		
		int grid_size = (int) Math.ceil(input.number * input.width / CAFFE_CUDA_NUM_THREADS);
		
		cuLaunchKernel(cross_backward_function,
				grid_size,  1, 1,      // Grid dimension
				CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
	            0, null,               // Shared memory size and stream
	            crossBackwardParameters, null // Kernel- and extra parameters
	        );
		
	}
	
	public void crossentropy_backward_igone(Tensor input,Tensor currentLabel,Tensor diff,int igonre) {
		input.showDMByNumber(0);
		currentLabel.showDMByNumber(0);
		input.showShape();
		currentLabel.showShape();
		diff.showShape();
		/**
		 * float* dlogits, const float* probs, const int* targets,int B, int T, int V, int igone
		 */
		crossBackwardParameters = Pointer.to(
	                Pointer.to(diff.getGpuData()),
	                Pointer.to(input.getGpuData()),
	                Pointer.to(currentLabel.getGpuData()),
	                Pointer.to(new int[] {currentLabel.number}),  //batch
	                Pointer.to(new int[] {currentLabel.width}), //time
	                Pointer.to(new int[] {input.width}),  //dim_size
	                Pointer.to(new int[] {igonre})
	            );
			
//		this.N = input.number;
		
		int grid_size = (int) Math.ceil(input.number * input.width / CAFFE_CUDA_NUM_THREADS);
		
		cuLaunchKernel(cross_igone_backward_function,
				grid_size,  1, 1,      // Grid dimension
				CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
	            0, null,               // Shared memory size and stream
	            crossBackwardParameters, null // Kernel- and extra parameters
	        );
		
	}
	
	public void forward(Tensor input,Tensor currentLabel,Tensor output) {
		
//		if(log_softmax_nl_loss_kernelParameters == null || this.N != output.number || model != this.model) {
			
//			this.model = model;
			
			/**
			 * float *input, float *label, float *output, int batch, int n
			 */
			log_softmax_nl_loss_kernelParameters = Pointer.to(
	                Pointer.to(input.getGpuData()),
	                Pointer.to(currentLabel.getGpuData()),
	                Pointer.to(output.getGpuData()),
	                Pointer.to(new int[] {input.number}),
	                Pointer.to(new int[] {input.width})
	            );
			
			this.N = output.number;
			
//		}
		
		cuLaunchKernel(log_softmax_nl_loss_function,
				input.number,  1, 1,      // Grid dimension
	            CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
	            0, null,               // Shared memory size and stream
	            log_softmax_nl_loss_kernelParameters, null // Kernel- and extra parameters
	        );

//		output.showDM();
//		JCudaDriver.cuCtxSynchronize();
		
	}
	
	public void forward(Tensor input,Tensor currentLabel,Tensor output,int igonre) {
		
//		if(log_softmax_nl_loss_kernelParameters == null || this.N != output.number || model != this.model) {
			
//			this.model = model;
			
			/**
			 * float *input, float *label, float *output, int batch, int n
			 */
			log_softmax_nl_loss_kernelParameters = Pointer.to(
	                Pointer.to(input.getGpuData()),
	                Pointer.to(currentLabel.getGpuData()),
	                Pointer.to(output.getGpuData()),
	                Pointer.to(new int[] {input.number}),
	                Pointer.to(new int[] {input.width}),
	                Pointer.to(new int[] {igonre})
	            );
			
			this.N = output.number;
			
//		}
		
		cuLaunchKernel(log_softmax_nl_loss_igonre_function,
				input.number,  1, 1,      // Grid dimension
	            CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
	            0, null,               // Shared memory size and stream
	            log_softmax_nl_loss_kernelParameters, null // Kernel- and extra parameters
	        );

//		output.showDM();
//		JCudaDriver.cuCtxSynchronize();
		
	}
	
	public void forwardIDX(Tensor input,Tensor currentLabel,Tensor output,int igonre) {

		/**
		 * float *input, float *label, float *output, int batch, int n
		 */
		log_softmax_nl_loss_kernelParameters = Pointer.to(
                Pointer.to(input.getGpuData()),
                Pointer.to(currentLabel.getGpuData()),
                Pointer.to(output.getGpuData()),
                Pointer.to(new int[] {input.number}),
                Pointer.to(new int[] {input.width}),
                Pointer.to(new int[] {igonre})
            );
		
		this.N = output.number;

		cuLaunchKernel(log_softmax_nl_loss_igonre_idx_function,
				input.number,  1, 1,      // Grid dimension
	            CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
	            0, null,               // Shared memory size and stream
	            log_softmax_nl_loss_kernelParameters, null // Kernel- and extra parameters
	        );

	}
	
	public void forwardIDX2(Tensor input,Tensor label,Tensor prods,Tensor output,int igonre) {

		/**
		 * float* out, const float* inp, const float* label, int igone, int N, int C
		 */
		crossSoftmaxForwardParameters = Pointer.to(
				Pointer.to(output.getGpuData()),
                Pointer.to(prods.getGpuData()),
                Pointer.to(input.getGpuData()),
                Pointer.to(label.getGpuData()),
                Pointer.to(new int[] {igonre}),
                Pointer.to(new int[] {input.number}),
                Pointer.to(new int[] {input.width})
            );
		
		this.N = input.number;
		
		int grid_size = N;
		
		int shared_mem_size = 2 * CAFFE_CUDA_NUM_THREADS / 32 * Sizeof.FLOAT;
		
		cuLaunchKernel(cross_softmax_function,
				grid_size,  1, 1,      // Grid dimension
	            CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
	            shared_mem_size, null,               // Shared memory size and stream
	            crossSoftmaxForwardParameters, null // Kernel- and extra parameters
	        );

	}
	
	public void backwardIDX2(Tensor prods,Tensor label,Tensor output,int igonre) {
		
		int N = prods.number;
		
		if(igonre > -1){
			N = N - MatrixUtils.countOccurrences(label.data, igonre);
		}
		
		/**
		 * float* out, const float* inp, const float* label, int igone, int N, int C
		 */
		crossSoftmaxBackwardParameters = Pointer.to(
				Pointer.to(output.getGpuData()),
                Pointer.to(prods.getGpuData()),
                Pointer.to(label.getGpuData()),
                Pointer.to(new int[] {igonre}),
                Pointer.to(new int[] {N}),
                Pointer.to(new int[] {prods.width})
            );
		
		this.N = prods.number;

		int grid_size = this.N;
		
		cuLaunchKernel(cross_softmax_back_function,
				grid_size,  1, 1,      // Grid dimension
	            CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
	            0, null,               // Shared memory size and stream
	            crossSoftmaxBackwardParameters, null // Kernel- and extra parameters
	        );
		
//		backwardCPU(prods, label, igonre);
		
	}
	
	public void backwardIDX2(Tensor prods,Tensor label,Tensor output,int igonre,int count) {
		
		int N = prods.number;
		
		if(igonre > -1){
			N = N - MatrixUtils.countOccurrences(label.data, igonre);
		}
		
		/**
		 * float* out, const float* inp, const float* label, int igone, int N, int C
		 */
		crossSoftmaxBackwardParameters = Pointer.to(
				Pointer.to(output.getGpuData()),
                Pointer.to(prods.getGpuData()),
                Pointer.to(label.getGpuData()),
                Pointer.to(new int[] {igonre}),
                Pointer.to(new int[] {count}),
                Pointer.to(new int[] {prods.width})
            );
		
		this.N = prods.number;

		int grid_size = this.N;
		
		cuLaunchKernel(cross_softmax_back_function,
				grid_size,  1, 1,      // Grid dimension
	            CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
	            0, null,               // Shared memory size and stream
	            crossSoftmaxBackwardParameters, null // Kernel- and extra parameters
	        );
		
//		backwardCPU(prods, label, igonre);
		
	}
	
	public void backwardCPU(Tensor prods,Tensor label,int igonre) {
		int C = prods.width;
		float[] p = prods.syncHost();
		float[] l = label.syncHost();
		float[] o = new float[prods.getDataLength()];
		for(int b = 0;b<label.number;b++) {
			float idx = l[b];
			System.out.println(idx);
			for(int c = 0;c<C;c++) {
				System.out.println(p[b * C + c]);
				float indicator = c == idx ? 1.0f : 0.0f;
				o[b * C + c] = (p[b * C + c] - indicator) / label.number;
			}
		}
		System.out.println(JsonUtils.toJson(o));
	}
	
	public void forwardCheck(Tensor input,Tensor currentLabel,Tensor output) {
		
//		if(checkParameters == null || this.N != output.number) {
			/**
			 * float *input, float *output, int batch, int n, float temp
			 */
			checkParameters = Pointer.to(
	                Pointer.to(input.getGpuData()),
	                Pointer.to(currentLabel.getGpuData()),
	                Pointer.to(output.getGpuData()),
	                Pointer.to(new int[] {input.number}),
	                Pointer.to(new int[] {input.width})
	            );
			
			this.N = output.number;
			
//		}
		
		cuLaunchKernel(check_function,
				input.number,  1, 1,      // Grid dimension
	            CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
	            0, null,               // Shared memory size and stream
	            checkParameters, null // Kernel- and extra parameters
	        );
		
//		JCudaDriver.cuCtxSynchronize();
		
	}
	
	public void backward(Tensor input,Tensor currentLabel,Tensor diff) {

		if(backKernelParameters == null) {

			/**
			 * float *input, float *currentLabel, float *diff, int n, int batch
			 */
			backKernelParameters = Pointer.to(
	                Pointer.to(input.getGpuData()),
	                Pointer.to(currentLabel.getGpuData()),
	                Pointer.to(diff.getGpuData()),
	                Pointer.to(new int[] {input.number}),
	                Pointer.to(new int[] {input.width})
	            );

		}
		
		cuLaunchKernel(loss_backward_function,
				input.number,  1, 1,      // Grid dimension
	            CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
	            0, null,               // Shared memory size and stream
	            backKernelParameters, null // Kernel- and extra parameters
	        );
		

		if(MatrixOperation.isNaN(diff.syncHost())){
			input.showDMByNumber(0);
		}

		
//		JCudaDriver.cuCtxSynchronize();
		
	}
	
	public void backward(Tensor input,Tensor currentLabel,Tensor diff,int igonre) {

		if(backKernelParameters == null || this.BN != input.number) {
			
			/**
			 * float *input, float *currentLabel, float *diff, int n, int batch
			 */
			backKernelParameters = Pointer.to(
	                Pointer.to(input.getGpuData()),
	                Pointer.to(currentLabel.getGpuData()),
	                Pointer.to(diff.getGpuData()),
	                Pointer.to(new int[] {input.number}),
	                Pointer.to(new int[] {input.width}),
	                Pointer.to(new int[] {igonre})
	            );
			
			 this.BN = input.number;
			
		}
		
		cuLaunchKernel(loss_igonre_backward_function,
				input.number,  1, 1,      // Grid dimension
	            CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
	            0, null,               // Shared memory size and stream
	            backKernelParameters, null // Kernel- and extra parameters
	        );
		

//		if(MatrixOperation.isNaN(diff.syncHost())){
//			input.showDMByNumber(0);
//		}

		
//		JCudaDriver.cuCtxSynchronize();
		
	}
	
	public void backwardIDX(Tensor input,Tensor currentLabel,Tensor diff,int igonre) {
		
		int N = input.number;
		
		if(igonre > -1){
			N = N - MatrixUtils.countOccurrences(currentLabel.data, igonre);
		}
		System.out.println(N);
		/**
		 * float *input, float *currentLabel, float *diff, int n, int batch
		 */
		backKernelParameters = Pointer.to(
                Pointer.to(input.getGpuData()),
                Pointer.to(currentLabel.getGpuData()),
                Pointer.to(diff.getGpuData()),
                Pointer.to(new int[] {N}),
                Pointer.to(new int[] {input.width}),
                Pointer.to(new int[] {igonre})
            );
		
		 this.BN = input.number;
		
		cuLaunchKernel(loss_igonre_backward_idx_function,
				input.number,  1, 1,      // Grid dimension
	            CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
	            0, null,               // Shared memory size and stream
	            backKernelParameters, null // Kernel- and extra parameters
	        );

	}
	
	
	public void checkCUDA(int code) {
		if(code != cudaError.cudaSuccess) {
			System.err.println("Error code "+code+":"+cudaError.stringFor(code));
		}
	}
	
	public static void check() {
		
		int N = 2;
		int C = 1;
		int H = 1;
		int W = 10;
		
		int size = N * C * H * W;
		
		float[] x = RandomUtils.x2Random(size);
		
		float[] lx = new float[] {0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0};
		
		Tensor input = new Tensor(N, C, H, W, x, true);
		Tensor label = new Tensor(N, C, H, W, lx, true);
		Tensor output = new Tensor(N, 1, 1, 1, 1, true);
		
		loss_gpu(input, label, output);
		
		System.out.println("gpu:"+JsonUtils.toJson(output.syncHost()));
		
		for(int i = 0;i<N;i++) {
			System.out.println(loss_cpu(input.getByNumber(i),label.getByNumber(i)));
		}
		
	}
	
	public static void loss_gpu(Tensor input,Tensor label,Tensor output) {
		
		CUDAManager cudaManager = new CUDAManager(0);
		
		CrossEntropyKernel kernel = new CrossEntropyKernel(cudaManager);
		
		kernel.forward(input, label, output);
		
	}
	
	public static float loss_cpu(float[] input,float[] label) {
		
//		System.out.println(JsonUtils.toJson(label));
		
		float sum = 0.0f;
		
		float loss = 0.0f;
		
		/**
		 * max
		 */
		float max = MatrixOperation.max(input);
		
		/**
		 * sum
		 */
		for(int i = 0;i<input.length;i++) {
			sum += Math.exp(input[i] - max);
		}
		
		/**
		 * softmax + log + nlloss
		 */
		for(int i = 0;i<input.length;i++) {
			loss += (float) (-((input[i] - max) - Math.log(sum)) * label[i]);
		}
		return loss;
	}
	
	public static void main(String args[]) {
		check();
	}
	
}
